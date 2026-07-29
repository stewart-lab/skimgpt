import argparse
import csv
import datetime
import json
import logging
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from main_wrapper import copy_project_src, flatten_and_cleanup
from skimgpt.utils import sanitize_term_for_filename, setup_wrapper_logger, strip_pipe

logger = logging.getLogger(__name__)


def read_bracket_terms(b_terms_file):
    with open(b_terms_file) as f:
        raw_terms = [line.strip() for line in f if line.strip()]

    seen = {}
    deduped = []
    for term in raw_terms:
        key = strip_pipe(term)
        if key in seen:
            logger.warning(f"Duplicate B term collapsed: '{term}' matches earlier term '{seen[key]}' (canonical '{key}')")
            continue
        seen[key] = term
        deduped.append(term)

    if len(deduped) < 2:
        raise ValueError(
            f"Tournament mode requires >=2 distinct B terms after de-duplication; found {len(deduped)} in "
            f"{b_terms_file}. Use a normal is_dch (or plain km_with_gpt) run instead."
        )
    return deduped


def build_round_pairs(term_list, rng, avoid_bye=None):
    """Shuffle and pair up term_list; the odd one out (if any) gets a bye.

    avoid_bye: a term that had a bye last round. If set and still present, it is
    excluded from bye-eligibility so it's guaranteed to be paired this round
    instead of sitting out two rounds in a row untested.
    """
    shuffled = term_list[:]
    rng.shuffle(shuffled)
    bye = None
    if len(shuffled) % 2 == 1:
        eligible = shuffled
        if avoid_bye is not None and avoid_bye in shuffled and len(shuffled) > 1:
            eligible = [t for t in shuffled if t != avoid_bye]
        bye = eligible[-1]
        shuffled.remove(bye)
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
    return pairs, bye


def make_pair_id(idx, term1, term2):
    return f"pair{idx:02d}_{sanitize_term_for_filename(term1)}_vs_{sanitize_term_for_filename(term2)}"


def build_pair_config(master_cfg, term1, term2, pair_id, work_dir):
    os.makedirs(work_dir, exist_ok=True)

    # line 0 = term1 = hypothesis1: score->100 favors row 0 per the DCH scoring convention.
    b_terms_path = os.path.join(work_dir, "b_terms_pair.txt")
    with open(b_terms_path, "w") as f:
        f.write(f"{term1}\n{term2}\n")

    cfg = json.loads(json.dumps(master_cfg))
    km_settings = cfg["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]
    km_settings["is_dch"] = True
    km_settings["position"] = False
    km_settings["B_TERMS_FILE"] = os.path.abspath(b_terms_path)
    cfg.setdefault("GLOBAL_SETTINGS", {})["OUTDIR_SUFFIX"] = pair_id

    with open(os.path.join(work_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)


def run_pair_with_retries(term1, term2, pair_id, round_dir, project_dir, master_cfg, main_py_path, pair_retries):
    work_dir = os.path.join(round_dir, "output", pair_id)
    for attempt in range(pair_retries + 1):
        if attempt > 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.warning(f"{pair_id}: retrying (attempt {attempt + 1})")

        build_pair_config(master_cfg, term1, term2, pair_id, work_dir)
        copy_project_src(project_dir, work_dir)

        start = time.time()
        res = subprocess.run([sys.executable, main_py_path], cwd=work_dir, env=os.environ.copy())
        elapsed = time.time() - start

        if res.returncode == 0:
            logger.info(f"{pair_id}: completed in {elapsed:.1f}s (attempt {attempt + 1})")
            return pair_id, True
        logger.error(f"{pair_id}: attempt {attempt + 1} failed (rc={res.returncode})")

    return pair_id, False


def parse_pair_results(results_tsv_path):
    if not os.path.isfile(results_tsv_path):
        return []
    with open(results_tsv_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _sum_int_column(rows, column):
    total = 0
    for row in rows:
        try:
            total += int(row.get(column, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def resolve_pair_winner(work_dir, term1, term2, tie_threshold, subprocess_ok):
    rows = parse_pair_results(os.path.join(work_dir, "results.tsv"))
    scores = []
    for row in rows:
        try:
            scores.append(float(row.get("Score", "")))
        except (TypeError, ValueError):
            continue

    if not subprocess_ok or not scores:
        return {"term1": term1, "term2": term2, "status": "error", "mean_score": None,
                "winner": None, "iteration_scores": scores}

    mean_score = statistics.mean(scores)
    term1_has_support = _sum_int_column(rows, "support_H1") > 0
    term2_has_support = _sum_int_column(rows, "support_H2") > 0

    if abs(mean_score - 50) <= tie_threshold:
        # A tie from zero literature evidence on one or both sides isn't a real
        # tie -- an unsupported term shouldn't advance just because it also
        # failed to lose outright. Only tie/advance both when each side has
        # at least some supporting evidence of its own.
        if term1_has_support and term2_has_support:
            winner = "tie"
        elif term1_has_support:
            winner = term1
        elif term2_has_support:
            winner = term2
        else:
            winner = "eliminated_no_support"
    elif mean_score > 50:
        winner = term1
    else:
        winner = term2

    return {"term1": term1, "term2": term2, "status": "ok", "mean_score": mean_score,
            "winner": winner, "iteration_scores": scores,
            "term1_has_support": term1_has_support, "term2_has_support": term2_has_support}


def run_round(round_num, pairs, bye, round_root, project_dir, master_cfg, main_py_path,
              tie_threshold, pair_retries, max_parallel_pairs):
    round_dir = os.path.join(round_root, f"round_{round_num}")
    os.makedirs(os.path.join(round_dir, "output"), exist_ok=True)

    pair_specs = [(make_pair_id(idx, t1, t2), t1, t2) for idx, (t1, t2) in enumerate(pairs)]

    subprocess_ok = {}
    if pair_specs:
        workers = max_parallel_pairs or len(pair_specs)
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {
                exe.submit(run_pair_with_retries, t1, t2, pid, round_dir, project_dir,
                           master_cfg, main_py_path, pair_retries): pid
                for pid, t1, t2 in pair_specs
            }
            for f in as_completed(futures):
                pid, ok = f.result()
                subprocess_ok[pid] = ok

    flatten_and_cleanup(round_dir)

    pair_results = []
    survivors = [bye] if bye else []
    for pid, t1, t2 in pair_specs:
        work_dir = os.path.join(round_dir, "output", pid)
        outcome = resolve_pair_winner(work_dir, t1, t2, tie_threshold, subprocess_ok.get(pid, False))
        pair_results.append({"pair_id": pid, **outcome})

        if outcome["status"] == "error" or outcome["winner"] == "tie":
            survivors.extend([t1, t2])
        elif outcome["winner"] == "eliminated_no_support":
            pass  # neither term had any literature support; both are dropped
        else:
            survivors.append(outcome["winner"])

    return {"round": round_num, "pairs": pair_results, "bye": bye, "survivors": survivors}


def run_tournament(initial_terms, round_root, project_dir, main_py_path, master_cfg, tconf):
    rng = random.Random(tconf.get("seed"))
    tie_threshold = tconf.get("tie_threshold", 5)
    on_pair_error = tconf.get("on_pair_error", "advance_both")
    pair_retries = tconf.get("pair_retries", 1)
    max_parallel_pairs = tconf.get("max_parallel_pairs")
    max_consecutive_stalls = tconf.get("max_consecutive_stalls", 2)
    max_rounds = tconf.get("max_rounds") or max(10, math.ceil(math.log2(len(initial_terms))) + 5)

    current = initial_terms[:]
    history = []
    stalls = 0
    round_num = 1
    previous_bye = None

    while len(current) > 1 and round_num <= max_rounds:
        pairs, bye = build_round_pairs(current, rng, avoid_bye=previous_bye)
        logger.info(f"Round {round_num}: {len(pairs)} pair(s){' + 1 bye' if bye else ''} from {len(current)} term(s)")

        result = run_round(round_num, pairs, bye, round_root, project_dir, master_cfg, main_py_path,
                            tie_threshold, pair_retries, max_parallel_pairs)
        history.append(result)

        if on_pair_error == "abort_round" and any(p["status"] == "error" for p in result["pairs"]):
            logger.error(f"Round {round_num} had a pair failure and on_pair_error=abort_round; stopping tournament.")
            return [], history, True

        next_terms = result["survivors"]
        stalls = stalls + 1 if len(next_terms) >= len(current) else 0
        current = next_terms
        previous_bye = bye
        round_num += 1

        if stalls >= max_consecutive_stalls:
            logger.warning(f"Stalled for {stalls} consecutive round(s) with no reduction; declaring co-winners.")
            break

    return current, history, False


def write_bracket_outputs(parent_dir, initial_terms, history, final_terms, tconf, started_at, finished_at, aborted):
    with open(os.path.join(parent_dir, "bracket_history.json"), "w") as f:
        json.dump({
            "tournament_config": tconf,
            "initial_terms": initial_terms,
            "rounds": history,
            "final_terms": final_terms,
            "aborted": aborted,
            "started_at": started_at,
            "finished_at": finished_at,
        }, f, indent=2)

    with open(os.path.join(parent_dir, "bracket_summary.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Round", "Pair_ID", "Term1", "Term2", "Mean_Score", "Winner", "Status"])
        for round_result in history:
            for pair in round_result["pairs"]:
                mean_score = pair["mean_score"]
                writer.writerow([
                    round_result["round"], pair["pair_id"], pair["term1"], pair["term2"],
                    f"{mean_score:.2f}" if mean_score is not None else "",
                    pair["winner"] or "", pair["status"],
                ])
            if round_result["bye"]:
                writer.writerow([round_result["round"], "-", round_result["bye"], "", "", "bye", "ok"])

    with open(os.path.join(parent_dir, "TOURNAMENT_WINNER.txt"), "w") as f:
        if aborted:
            f.write("Tournament aborted: a pair failed and on_pair_error=abort_round. See bracket_history.json / logs for details.\n")
        elif len(final_terms) == 1:
            f.write(f"Winner: {final_terms[0]}\n")
        elif not final_terms:
            f.write(
                "No term survived: every remaining B term was eliminated for lacking literature support. "
                "See bracket_history.json for details.\n"
            )
        else:
            f.write(f"Co-winners (unresolved after {len(history)} round(s)): {', '.join(final_terms)}\n")


def validate_tournament_config(cfg):
    if cfg.get("JOB_TYPE", "").strip() != "km_with_gpt":
        sys.exit("tournament_wrapper.py requires JOB_TYPE == 'km_with_gpt'")

    km_settings = cfg.get("JOB_SPECIFIC_SETTINGS", {}).get("km_with_gpt", {})
    tconf = km_settings.get("tournament", {})
    if not tconf.get("enabled", False):
        sys.exit("JOB_SPECIFIC_SETTINGS.km_with_gpt.tournament.enabled must be true to run tournament_wrapper.py")
    if km_settings.get("is_dch", False):
        sys.exit(
            "Master config must have is_dch=false when tournament.enabled=true "
            "(tournament mode sets is_dch per-pair internally)."
        )

    if km_settings.get("A_TERM_LIST", False):
        a_terms_file = km_settings.get("A_TERMS_FILE", "")
        if not os.path.isabs(a_terms_file):
            a_terms_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), a_terms_file)
        with open(a_terms_file) as f:
            a_terms = [line.strip() for line in f if line.strip()]
        if len(a_terms) != 1:
            sys.exit(
                f"tournament_wrapper.py requires exactly one A term; found {len(a_terms)} in {a_terms_file}. "
                "Multiple A terms per tournament run are not supported (v1 limitation)."
            )

    return tconf, km_settings


def main():
    parser = argparse.ArgumentParser(
        prog="tournament_wrapper.py",
        description="Run a tournament-bracket style N-way DCH comparison over a B_TERMS_FILE with more than 2 terms.",
    )
    parser.add_argument("-config", default="config.json", help="Path to master config.json")
    parser.add_argument("-seed", type=int, default=None, help="Override tournament.seed for reproducible pairing")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    master_cfg_path = os.path.abspath(args.config)
    master_cfg = json.load(open(master_cfg_path))

    tconf, km_settings = validate_tournament_config(master_cfg)
    if args.seed is not None:
        tconf["seed"] = args.seed

    b_terms_file = km_settings["B_TERMS_FILE"]
    if not os.path.isabs(b_terms_file):
        b_terms_file = os.path.join(project_dir, b_terms_file)
    initial_terms = read_bracket_terms(b_terms_file)

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = master_cfg.get("GLOBAL_SETTINGS", {}).get("OUTDIR_SUFFIX", "")
    parent_name = f"output_{ts}_tournament" + (f"_{suffix}" if suffix else "")
    parent_dir = os.path.join(os.path.abspath("output"), parent_name)
    os.makedirs(parent_dir, exist_ok=True)

    shutil.copy2(master_cfg_path, os.path.join(parent_dir, "config.json"))
    copy_project_src(project_dir, parent_dir)
    setup_wrapper_logger(parent_dir, "km_with_gpt_tournament")

    logger.info(f"Tournament parent dir: {parent_dir}")
    logger.info(f"{len(initial_terms)} distinct B term(s) loaded from {b_terms_file}")

    main_py_path = os.path.join(project_dir, "skimgpt", "main.py")
    started_at = ts

    final_terms, history, aborted = run_tournament(
        initial_terms, parent_dir, project_dir, main_py_path, master_cfg, tconf
    )

    finished_at = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    write_bracket_outputs(parent_dir, initial_terms, history, final_terms, tconf, started_at, finished_at, aborted)

    if aborted:
        logger.error("Tournament aborted due to a pair failure (on_pair_error=abort_round).")
        sys.exit(1)

    logger.info(f"Tournament complete. Final term(s): {final_terms}")


if __name__ == "__main__":
    main()
