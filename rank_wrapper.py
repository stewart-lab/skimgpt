import argparse
import csv
import datetime
import json
import logging
import os
import random
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from main_wrapper import copy_project_src, flatten_and_cleanup
from skimgpt.utils import sanitize_term_for_filename, setup_wrapper_logger
from tournament_wrapper import read_bracket_terms, run_pair_with_retries, resolve_pair_winner

logger = logging.getLogger(__name__)


class ComparisonCache:
    """Cache DCH pairwise outcomes by unordered term pair.

    Odd-even transposition sort can re-check the same adjacent pair across
    consecutive passes (e.g. right up to the two-zero-swap convergence check),
    and small lists can end up re-comparing the same two terms in a later pass
    after other swaps. Each comparison is a real GPT+PubMed run, so the same
    pair is only ever actually run once regardless of which one lands in the
    term1/term2 slot that time.
    """

    def __init__(self):
        self._store = {}

    def get(self, term_i, term_j):
        outcome = self._store.get(frozenset((term_i, term_j)))
        if outcome is None:
            return None
        return outcome if outcome["term1"] == term_i else _reoriented(outcome, term_i, term_j)

    def set(self, term_i, term_j, outcome):
        self._store[frozenset((term_i, term_j))] = outcome


def _reoriented(outcome, term_i, term_j):
    mean_score = outcome["mean_score"]
    return {
        "term1": term_i,
        "term2": term_j,
        "status": outcome["status"],
        "mean_score": (100 - mean_score) if mean_score is not None else None,
        "winner": outcome["winner"],
        "iteration_scores": [100 - s for s in (outcome.get("iteration_scores") or [])],
        "term1_has_support": outcome.get("term2_has_support", False),
        "term2_has_support": outcome.get("term1_has_support", False),
    }


def get_phase_pairs(order, even_phase):
    start = 0 if even_phase else 1
    return [(i, i + 1) for i in range(start, len(order) - 1, 2)]


def find_frozen_chains(order, cache):
    """Find maximal runs of adjacent positions that are all cached, genuine ties.

    Returns every such run regardless of position -- callers decide which
    directions (upward escape, downward validation) are eligible for a given
    chain. Uses only the existing cache (no new state), recomputed fresh from
    the current order each time it's called.
    """
    chains = []
    n = len(order)
    i = 0
    while i < n - 1:
        outcome = cache.get(order[i], order[i + 1])
        if outcome and outcome["status"] == "ok" and outcome["winner"] == "tie":
            j = i
            while j < n - 1:
                nxt = cache.get(order[j], order[j + 1])
                if nxt and nxt["status"] == "ok" and nxt["winner"] == "tie":
                    j += 1
                else:
                    break
            chains.append((i, j))
            i = j + 1
        else:
            i += 1
    return chains


def pick_escape_candidate(order, chain_start, chain_end, cache):
    """Deepest-first: the first chain member not yet compared against the
    chain's upward neighbor (order[chain_start - 1]).

    A tied chain moves as a block, but only its outward-facing member ever
    gets compared to a new neighbor -- the others just inherit that member's
    position without ever proving it themselves. This tests whether a buried
    member is actually BETTER than what's above the whole chain.

    Returns a normalized (position_i, position_j, term_i, term_j) tuple with
    position_i < position_j, or None if this chain isn't upward-eligible (it's
    already at the very top) or every member has already had its attempt.
    """
    if chain_start == 0:
        return None
    target_pos, target_term = chain_start - 1, order[chain_start - 1]
    for pos in range(chain_end, chain_start - 1, -1):
        member_term = order[pos]
        if cache.get(member_term, target_term) is None:
            return target_pos, pos, target_term, member_term
    return None


def pick_downward_candidate(order, chain_start, chain_end, cache):
    """Shallowest-first: the first chain member not yet compared against the
    chain's downward neighbor (order[chain_end + 1]).

    Symmetric to pick_escape_candidate: tests whether a chain member that's
    just inheriting its position from the block is actually WORSE than what's
    below the whole chain, rather than only ever checking upward.

    Returns a normalized (position_i, position_j, term_i, term_j) tuple with
    position_i < position_j, or None if this chain isn't downward-eligible
    (it's already at the very bottom) or every member has already been tried.
    """
    if chain_end == len(order) - 1:
        return None
    target_pos, target_term = chain_end + 1, order[chain_end + 1]
    for pos in range(chain_start, chain_end + 1):
        member_term = order[pos]
        if cache.get(member_term, target_term) is None:
            return pos, target_pos, member_term, target_term
    return None


def run_pass(pass_num, phase, order, idx_pairs, round_root, project_dir, master_cfg, main_py_path,
             tie_threshold, pair_retries, max_parallel_pairs, cache):
    pass_dir = os.path.join(round_root, f"pass_{pass_num:02d}_{phase}")

    pair_specs = []
    for i, j in idx_pairs:
        t1, t2 = order[i], order[j]
        pid = f"pos{i:02d}_{sanitize_term_for_filename(t1)}_vs_{sanitize_term_for_filename(t2)}"
        pair_specs.append((pid, i, j, t1, t2))

    to_run = [spec for spec in pair_specs if cache.get(spec[3], spec[4]) is None]

    subprocess_ok = {}
    if to_run:
        os.makedirs(os.path.join(pass_dir, "output"), exist_ok=True)
        workers = max_parallel_pairs or len(to_run)
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {
                exe.submit(run_pair_with_retries, t1, t2, pid, pass_dir, project_dir,
                           master_cfg, main_py_path, pair_retries): pid
                for pid, _i, _j, t1, t2 in to_run
            }
            for f in as_completed(futures):
                pid, ok = f.result()
                subprocess_ok[pid] = ok
        flatten_and_cleanup(pass_dir)

    comparisons = []
    new_order = order[:]
    swaps = 0
    for pid, i, j, t1, t2 in pair_specs:
        cached = cache.get(t1, t2)
        if cached is not None:
            outcome, from_cache = cached, True
        else:
            work_dir = os.path.join(pass_dir, "output", pid)
            outcome = resolve_pair_winner(work_dir, t1, t2, tie_threshold, subprocess_ok.get(pid, False))
            # Don't cache a failed comparison -- an "error" reflects a subprocess/
            # infrastructure failure for that one attempt, not a reproducible fact
            # about the two terms. Caching it would permanently deny the pair any
            # further retry if it becomes adjacent again in a later pass.
            if outcome["status"] == "ok":
                cache.set(t1, t2, outcome)
            from_cache = False

        if outcome["status"] == "error":
            tag, swap = "error", False
        elif outcome["winner"] == "tie":
            tag, swap = "tie", False
        elif outcome["winner"] == "eliminated_no_support":
            tag, swap = "no_evidence", False
        elif outcome["winner"] == t1:
            tag, swap = "term_i", False
        else:
            tag, swap = "term_j", True

        if swap:
            new_order[i], new_order[j] = new_order[j], new_order[i]
            swaps += 1

        comparisons.append({
            "pair_id": pid, "position_i": i, "position_j": j, "term_i": t1, "term_j": t2,
            "status": outcome["status"], "mean_score": outcome["mean_score"],
            "outcome": tag, "swapped": swap, "from_cache": from_cache,
            "is_escape": False, "boundary_kind": None,
            "term_i_has_support": outcome.get("term1_has_support"),
            "term_j_has_support": outcome.get("term2_has_support"),
        })

    return new_order, comparisons, swaps


def run_boundary_attempts(specs, kind, order, pass_num, round_root, project_dir, master_cfg, main_py_path,
                           tie_threshold, pair_retries, max_parallel_pairs, cache):
    """Run one real comparison per already-normalized (position_i, position_j,
    term_i, term_j) spec, position_i < position_j guaranteed by the caller.

    Uses the exact same position_i < position_j / term1 = term_i / swap-on-
    term_j-wins convention as run_pass, so the results plug into
    compute_term_stats and the quarantine logic completely unchanged. This is
    correct for BOTH boundary directions: for an upward escape, term_i is the
    chain's upward neighbor and term_j is the deep member (a term_j win moves
    the member up, out of the chain); for a downward validation, term_i is the
    chain member and term_j is the downward neighbor (a term_j win moves the
    neighbor up into the chain and demotes the member below it).

    kind: "escape" or "validate" -- only used for directory/record labeling.
    """
    if not specs:
        return [], 0, order

    pass_dir = os.path.join(round_root, f"pass_{pass_num:02d}_{kind}")
    pair_specs = []
    for idx, (i, j, t1, t2) in enumerate(specs):
        pid = f"{kind}{idx:02d}_{sanitize_term_for_filename(t1)}_vs_{sanitize_term_for_filename(t2)}"
        pair_specs.append((pid, i, j, t1, t2))

    os.makedirs(os.path.join(pass_dir, "output"), exist_ok=True)
    workers = max_parallel_pairs or len(pair_specs)
    subprocess_ok = {}
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(run_pair_with_retries, t1, t2, pid, pass_dir, project_dir,
                       master_cfg, main_py_path, pair_retries): pid
            for pid, _i, _j, t1, t2 in pair_specs
        }
        for f in as_completed(futures):
            pid, ok = f.result()
            subprocess_ok[pid] = ok
    flatten_and_cleanup(pass_dir)

    new_order = order[:]
    comparisons = []
    swaps = 0
    for pid, i, j, t1, t2 in pair_specs:
        work_dir = os.path.join(pass_dir, "output", pid)
        outcome = resolve_pair_winner(work_dir, t1, t2, tie_threshold, subprocess_ok.get(pid, False))
        if outcome["status"] == "ok":
            cache.set(t1, t2, outcome)

        if outcome["status"] == "error":
            tag, swap = "error", False
        elif outcome["winner"] == "tie":
            tag, swap = "tie", False
        elif outcome["winner"] == "eliminated_no_support":
            tag, swap = "no_evidence", False
        elif outcome["winner"] == t1:
            tag, swap = "term_i", False
        else:
            tag, swap = "term_j", True

        if swap:
            new_order[i], new_order[j] = new_order[j], new_order[i]
            swaps += 1

        comparisons.append({
            "pair_id": pid, "position_i": i, "position_j": j, "term_i": t1, "term_j": t2,
            "status": outcome["status"], "mean_score": outcome["mean_score"],
            "outcome": tag, "swapped": swap, "from_cache": False,
            "is_escape": kind == "escape", "boundary_kind": kind,
            "term_i_has_support": outcome.get("term1_has_support"),
            "term_j_has_support": outcome.get("term2_has_support"),
        })

    return comparisons, swaps, new_order


def compute_term_stats(all_terms, passes):
    """Aggregate win/loss/tie/etc. counts per term.

    Cache-hit comparisons are skipped -- they're the exact same underlying GPT
    run being re-examined at a new position, not a new independent data point,
    so counting them again would double (or triple, ...) the real tallies.
    """
    stats = {t: {"comparisons": 0, "wins": 0, "ties": 0, "losses": 0, "no_evidence": 0, "errors": 0} for t in all_terms}
    for p in passes:
        for c in p["comparisons"]:
            if c["from_cache"]:
                continue
            ti, tj, tag = c["term_i"], c["term_j"], c["outcome"]
            stats[ti]["comparisons"] += 1
            stats[tj]["comparisons"] += 1
            if tag == "term_i":
                stats[ti]["wins"] += 1
                stats[tj]["losses"] += 1
            elif tag == "term_j":
                stats[tj]["wins"] += 1
                stats[ti]["losses"] += 1
            elif tag == "tie":
                stats[ti]["ties"] += 1
                stats[tj]["ties"] += 1
            elif tag == "no_evidence":
                stats[ti]["no_evidence"] += 1
                stats[tj]["no_evidence"] += 1
            elif tag == "error":
                stats[ti]["errors"] += 1
                stats[tj]["errors"] += 1
    return stats


def compute_term_scores(all_terms, passes):
    """Per-term list of scores, each corrected to that term's own perspective.

    A comparison's score is always given from term_i/H1's perspective (100
    favors term_i, 0 favors term_j). To average scores from a term's own point
    of view, term_j's score for that comparison is 100 minus the recorded
    score. Skips cache hits (the same underlying observation re-examined, not
    a new data point -- same reasoning as compute_term_stats) and errored
    comparisons (no valid score at all).

    Returns (scores, averages): scores maps term -> list of corrected scores
    in the order encountered; averages maps term -> mean of that list, or
    None if the term was never in a real, successful comparison.
    """
    scores = {t: [] for t in all_terms}
    for p in passes:
        for c in p["comparisons"]:
            if c["from_cache"] or c["status"] != "ok" or c["mean_score"] is None:
                continue
            scores[c["term_i"]].append(c["mean_score"])
            scores[c["term_j"]].append(100 - c["mean_score"])
    averages = {t: (statistics.mean(v) if v else None) for t, v in scores.items()}
    return scores, averages


def reseed_by_record(order, term_stats):
    """Re-sort the current active order by (wins - losses), most first.

    Adjacent-only comparisons can leave a term's rank reflecting an accident of
    position (e.g. its neighbors got quarantined, or it's riding along tied to
    a stronger term) rather than its actual record. A stable sort by record
    corrects that directly using information already gathered -- no new
    comparisons, so no extra cost. Ties in record keep their current relative
    order (Python's sort is stable), so this doesn't invent a preference where
    there isn't evidence for one.
    """
    return sorted(order, key=lambda t: -(term_stats[t]["wins"] - term_stats[t]["losses"]))


def repair_order_with_cache(order, cache):
    """Fix any adjacent pair a re-seed placed in an order that contradicts an
    already-cached DIRECT result between them.

    Aggregate win/loss record doesn't know about specific head-to-head
    outcomes -- two terms can have equal (or record-favoring-the-wrong-one)
    tallies from facing different opponents, even though they've already been
    directly compared with a clear answer. Without this, a re-seed can put
    such a pair in the "wrong" order, the next normal pass corrects it via the
    real adjacent comparison, and the following re-seed undoes that fix again
    -- oscillating forever. This uses only the existing cache (no new
    comparisons, so no extra cost) and is bounded like a standard bubble pass
    so it terminates even if cached results are cyclic (non-transitive).
    """
    order = order[:]
    for _ in range(len(order)):
        changed = False
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            outcome = cache.get(a, b)
            if outcome and outcome["status"] == "ok" and outcome["winner"] == b:
                order[i], order[i + 1] = b, a
                changed = True
        if not changed:
            break
    return order


def build_final_ranking(order, term_stats, term_scores, quarantine_reason=None):
    quarantine_reason = quarantine_reason or {}
    ranking = []
    for i, term in enumerate(order, start=1):
        s = term_stats[term]
        scores = term_scores[term]
        reason = quarantine_reason.get(term)
        ranking.append({
            "rank": i, "term": term, "wins": s["wins"], "ties": s["ties"], "losses": s["losses"],
            "no_evidence": s["no_evidence"], "errors": s["errors"],
            "insufficient_evidence": reason is not None,
            "quarantine_reason": reason,
            "avg_score": statistics.mean(scores) if scores else None,
            "scores": scores,
        })
    return ranking


def run_ranking(initial_terms, round_root, project_dir, main_py_path, master_cfg, rconf):
    rng = random.Random(rconf.get("seed"))
    tie_threshold = rconf.get("tie_threshold", 5)
    pair_retries = rconf.get("pair_retries", 1)
    max_parallel_pairs = rconf.get("max_parallel_pairs")
    # The classic odd-even transposition sort proof bounds correctness at n passes,
    # but that assumes pure swaps; quarantine events reset the no-change streak too
    # (correctly, since removing a term changes adjacencies), which can push the
    # two-consecutive-no-change confirmation slightly past n. A small margin avoids
    # ending on an unconfirmed (if already-correct) order for no real extra cost --
    # a pass with nothing left to do makes no new API calls.
    max_passes = rconf.get("max_passes") or (len(initial_terms) + 2)
    # GPT/abstract-sampling noise means a term with sparse (but real) evidence could
    # occasionally land a joint-zero-support verdict against one particular opponent
    # without truly having zero support overall. Default is 1 (quarantine on the
    # first hit) since a term with genuinely zero co-occurring abstracts will show
    # this consistently regardless of partner; raise it if quarantining looks
    # over-aggressive in practice.
    no_evidence_threshold = rconf.get("min_no_evidence_before_quarantine", 1)
    # An "error" is a subprocess/infrastructure failure for that one attempt, not a
    # reproducible fact about either term (unlike zero literature support, which is
    # stable). Give it a couple of fresh retries across passes before giving up,
    # rather than quarantining as eagerly as a genuine no-evidence verdict.
    error_threshold = rconf.get("min_errors_before_quarantine", 2)
    # A chain of genuine adjacent ties freezes solid (nothing inside it can ever
    # swap past the chain boundary, since that requires a decisive result there).
    # Deep members never get compared against anything above the chain, so their
    # rank reflects where they got stuck rather than their real strength. Give
    # the deepest untried member of each frozen chain one shot per pass against
    # the chain's upward neighbor.
    enable_escape = rconf.get("enable_escape_comparisons", True)
    # Adjacent-only sorting can leave a term's position reflecting an accident
    # (its blockers got quarantined, or it's riding a tie with a stronger
    # neighbor) rather than its actual record. Periodically re-sort the active
    # list by (wins - losses) using information already gathered, so a term's
    # aggregate record can correct its position even without a fresh chain
    # escape. 0/null disables this.
    reseed_every_n_passes = rconf.get("reseed_every_n_passes", 2)
    cache = ComparisonCache()

    order = initial_terms[:]
    rng.shuffle(order)
    initial_order = order[:]

    passes = []
    recent_changed = []
    converged = False
    quarantined = []  # terms pulled out after crossing a threshold, in encounter order
    quarantined_set = set()
    quarantine_reason = {}
    no_evidence_counts = {}
    error_counts = {}

    for pass_num in range(1, max_passes + 1):
        if len(order) <= 1:
            break

        even_phase = (pass_num % 2 == 1)
        phase = "even" if even_phase else "odd"
        idx_pairs = get_phase_pairs(order, even_phase)

        if idx_pairs:
            order, comparisons, swaps = run_pass(
                pass_num, phase, order, idx_pairs, round_root, project_dir, master_cfg, main_py_path,
                tie_threshold, pair_retries, max_parallel_pairs, cache,
            )
        else:
            comparisons, swaps = [], 0

        if enable_escape:
            # Only ONE boundary attempt per pass, even if multiple chains are stuck
            # or a chain is eligible in both directions. Detecting every candidate
            # against one snapshot and then applying their swaps sequentially is
            # unsafe: an earlier swap can move the exact term a later one's target
            # position was counting on, corrupting that later swap. Taking one at
            # a time means each pass's detection always reflects the truth, and
            # anything else stuck simply gets its turn on a later pass instead.
            #
            # Upward escape asks "is this buried member secretly BETTER than what's
            # above the chain" (only the chain's top ever gets tested against it
            # normally). Downward validation asks the mirror question: a tied
            # block moves together, but only its bottom-facing member ever gets
            # compared to what's below -- other members just inherit that result
            # without ever proving it. This checks whether they actually belong
            # there too.
            spec, kind = None, None
            for chain_start, chain_end in find_frozen_chains(order, cache):
                spec = pick_escape_candidate(order, chain_start, chain_end, cache)
                if spec is not None:
                    kind = "escape"
                    break
                spec = pick_downward_candidate(order, chain_start, chain_end, cache)
                if spec is not None:
                    kind = "validate"
                    break
            if spec is not None:
                boundary_comparisons, boundary_swaps, order = run_boundary_attempts(
                    [spec], kind, order, pass_num, round_root, project_dir, master_cfg, main_py_path,
                    tie_threshold, pair_retries, max_parallel_pairs, cache,
                )
                comparisons = comparisons + boundary_comparisons
                swaps += boundary_swaps

        # Zero literature support for a term is a stable, reproducible property (it
        # comes from that term's own co-occurring-abstract pool, independent of the
        # opponent) -- not just something that shows up in symmetric ties. A term
        # left in place after showing this never swaps regardless of its neighbor,
        # so it becomes an immovable wall blocking correct ordering on either side
        # of it. Quarantine on either a per-term zero-support verdict or a run of
        # unresolved errors; skip cache hits everywhere since they're the exact same
        # underlying observation being re-examined, not a new data point.
        newly_quarantined = []
        for c in comparisons:
            if c["from_cache"]:
                continue
            if c["status"] == "ok":
                for t, has_support in ((c["term_i"], c["term_i_has_support"]),
                                        (c["term_j"], c["term_j_has_support"])):
                    if has_support is not False or t in quarantined_set:
                        continue
                    no_evidence_counts[t] = no_evidence_counts.get(t, 0) + 1
                    if no_evidence_counts[t] >= no_evidence_threshold:
                        quarantined_set.add(t)
                        quarantined.append(t)
                        quarantine_reason[t] = "no_evidence"
                        newly_quarantined.append(t)
            elif c["status"] == "error":
                for t in (c["term_i"], c["term_j"]):
                    if t in quarantined_set:
                        continue
                    error_counts[t] = error_counts.get(t, 0) + 1
                    if error_counts[t] >= error_threshold:
                        quarantined_set.add(t)
                        quarantined.append(t)
                        quarantine_reason[t] = "repeated_error"
                        newly_quarantined.append(t)
        if newly_quarantined:
            order = [t for t in order if t not in quarantined_set]

        pass_record = {"pass": pass_num, "phase": phase, "comparisons": comparisons,
                       "order_after_pass": order[:], "swaps": swaps,
                       "quarantined_this_pass": newly_quarantined, "reseeded": False}
        passes.append(pass_record)
        changed = swaps > 0 or bool(newly_quarantined)

        if reseed_every_n_passes and pass_num % reseed_every_n_passes == 0:
            reseeded_order = reseed_by_record(order, compute_term_stats(initial_terms, passes))
            reseeded_order = repair_order_with_cache(reseeded_order, cache)
            if reseeded_order != order:
                logger.info(f"Pass {pass_num}: re-seeding by record -> {reseeded_order}")
                order = reseeded_order
                pass_record["order_after_pass"] = order[:]
                pass_record["reseeded"] = True
                changed = True

        recent_changed.append(changed)
        logger.info(f"Pass {pass_num} ({phase}): {swaps} swap(s), {len(newly_quarantined)} newly quarantined"
                     f"{', reseeded' if pass_record['reseeded'] else ''}")

        # A full even+odd cycle with no swaps and no new quarantines means no further
        # pass can change anything either, so the active list is sorted -- stop instead
        # of burning the remaining passes (and their API cost) for no reason.
        if len(recent_changed) >= 2 and not recent_changed[-1] and not recent_changed[-2]:
            converged = True
            logger.info(f"Converged after {pass_num} pass(es) (two consecutive passes with no change).")
            break

    if not converged and len(order) > 1:
        logger.warning(f"Reached max_passes={max_passes} without a confirmed two-pass convergence.")

    final_order = order + quarantined
    term_stats = compute_term_stats(initial_terms, passes)
    return final_order, passes, term_stats, converged, initial_order, quarantined, quarantine_reason


def write_ranking_outputs(parent_dir, initial_terms, initial_order, passes, final_ranking,
                           converged, rconf, started_at, finished_at):
    with open(os.path.join(parent_dir, "ranking_history.json"), "w") as f:
        json.dump({
            "ranking_config": rconf,
            "initial_terms": initial_terms,
            "initial_order": initial_order,
            "passes": passes,
            "final_ranking": final_ranking,
            "converged": converged,
            "num_passes": len(passes),
            "started_at": started_at,
            "finished_at": finished_at,
        }, f, indent=2)

    with open(os.path.join(parent_dir, "ranking_summary.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Pass", "Phase", "Position_i", "Position_j", "Term_i", "Term_j",
                          "Mean_Score", "Outcome", "Swapped", "From_Cache", "Boundary_Kind"])
        for p in passes:
            for c in p["comparisons"]:
                mean_score = c["mean_score"]
                writer.writerow([
                    p["pass"], p["phase"], c["position_i"], c["position_j"], c["term_i"], c["term_j"],
                    f"{mean_score:.2f}" if mean_score is not None else "",
                    c["outcome"], c["swapped"], c["from_cache"], c["boundary_kind"] or "",
                ])
            if p["reseeded"]:
                writer.writerow([p["pass"], p["phase"], "", "", "", "", "", "reseed", "", "", "reseed"])

    with open(os.path.join(parent_dir, "FINAL_RANKING.txt"), "w") as f:
        status = "converged" if converged else "stopped at max_passes -- may not be fully converged"
        f.write(f"Final ranking ({status} after {len(passes)} pass(es)):\n\n")
        for entry in final_ranking:
            flag = ""
            if entry["quarantine_reason"] == "no_evidence":
                flag = "  [insufficient evidence -- rank not meaningful]"
            elif entry["quarantine_reason"] == "repeated_error":
                flag = "  [repeated pipeline error -- comparison never resolved, rank not meaningful]"
            avg = f"{entry['avg_score']:.2f}" if entry["avg_score"] is not None else "n/a"
            f.write(f"{entry['rank']}. {entry['term']} (W{entry['wins']}-T{entry['ties']}-L{entry['losses']}, "
                    f"avg_score={avg}){flag}\n")


def validate_ranking_config(cfg):
    if cfg.get("JOB_TYPE", "").strip() != "km_with_gpt":
        sys.exit("rank_wrapper.py requires JOB_TYPE == 'km_with_gpt'")

    km_settings = cfg.get("JOB_SPECIFIC_SETTINGS", {}).get("km_with_gpt", {})
    rconf = km_settings.get("ranking", {})
    if not rconf.get("enabled", False):
        sys.exit("JOB_SPECIFIC_SETTINGS.km_with_gpt.ranking.enabled must be true to run rank_wrapper.py")
    if km_settings.get("is_dch", False):
        sys.exit(
            "Master config must have is_dch=false when ranking.enabled=true "
            "(rank_wrapper.py sets is_dch per-pair internally)."
        )

    if km_settings.get("A_TERM_LIST", False):
        a_terms_file = km_settings.get("A_TERMS_FILE", "")
        if not os.path.isabs(a_terms_file):
            a_terms_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), a_terms_file)
        with open(a_terms_file) as f:
            a_terms = [line.strip() for line in f if line.strip()]
        if len(a_terms) != 1:
            sys.exit(
                f"rank_wrapper.py requires exactly one A term; found {len(a_terms)} in {a_terms_file}. "
                "Multiple A terms per ranking run are not supported (v1 limitation)."
            )

    return rconf, km_settings


def main():
    parser = argparse.ArgumentParser(
        prog="rank_wrapper.py",
        description="Rank all B terms in B_TERMS_FILE via a parallel bubble sort (odd-even transposition) of pairwise DCH comparisons.",
    )
    parser.add_argument("-config", default="config.json", help="Path to master config.json")
    parser.add_argument("-seed", type=int, default=None, help="Override ranking.seed for reproducible initial shuffling")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    master_cfg_path = os.path.abspath(args.config)
    master_cfg = json.load(open(master_cfg_path))

    rconf, km_settings = validate_ranking_config(master_cfg)
    if args.seed is not None:
        rconf["seed"] = args.seed

    b_terms_file = km_settings["B_TERMS_FILE"]
    if not os.path.isabs(b_terms_file):
        b_terms_file = os.path.join(project_dir, b_terms_file)
    initial_terms = read_bracket_terms(b_terms_file)

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = master_cfg.get("GLOBAL_SETTINGS", {}).get("OUTDIR_SUFFIX", "")
    parent_name = f"output_{ts}_rank" + (f"_{suffix}" if suffix else "")
    parent_dir = os.path.join(os.path.abspath("output"), parent_name)
    os.makedirs(parent_dir, exist_ok=True)

    shutil.copy2(master_cfg_path, os.path.join(parent_dir, "config.json"))
    copy_project_src(project_dir, parent_dir)
    setup_wrapper_logger(parent_dir, "km_with_gpt_rank")

    logger.info(f"Ranking parent dir: {parent_dir}")
    logger.info(f"{len(initial_terms)} distinct B term(s) loaded from {b_terms_file}")

    main_py_path = os.path.join(project_dir, "skimgpt", "main.py")
    started_at = ts

    final_order, passes, term_stats, converged, initial_order, quarantined, quarantine_reason = run_ranking(
        initial_terms, parent_dir, project_dir, main_py_path, master_cfg, rconf
    )
    term_scores, _term_score_averages = compute_term_scores(initial_terms, passes)
    final_ranking = build_final_ranking(final_order, term_stats, term_scores, quarantine_reason=quarantine_reason)

    finished_at = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    write_ranking_outputs(parent_dir, initial_terms, initial_order, passes, final_ranking,
                           converged, rconf, started_at, finished_at)

    logger.info(f"Ranking complete ({'converged' if converged else 'stopped at max_passes'}). Final order: {final_order}")


if __name__ == "__main__":
    main()
