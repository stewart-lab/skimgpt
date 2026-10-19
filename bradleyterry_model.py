import datetime
import json
import logging
import math
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# R plotting scripts and the cmdlogtime command-line definition live next to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_SCRIPTS = ["plot_bradley_terry_timeline.R", "plot_bradley_terry_bump.R"]
COMMAND_LINE_DEF_FILE = os.path.join(SCRIPT_DIR, "bradleyterry_model_commandline.txt")

def build_win_table(comparisons, penalty=1.0):
    """
    Build win/loss table from pairwise scores.

    w[i][j] = number of times i beat j. With partial wins, an LLM score
    of 80 for (i, j) adds 0.8 to w[i][j] and 0.2 to w[j][i].

    penalty: each gene plays this many imaginary tied games against
    an extra "average" gene. This keeps strengths from going to zero
    or infinity for genes that always lose or always win.
    """
    genes = sorted({gene for a, b, _ in comparisons for gene in (a, b)})

    if penalty > 0:
        genes.append("_AVERAGE_")

    # Start with an all-zero table
    w = {i: {j: 0.0 for j in genes} for i in genes}

    # Fill in partial wins from the real comparisons
    for a, b, score in comparisons:
        w[a][b] += score / 100.0
        w[b][a] += 1.0 - score / 100.0

    # Imaginary tied games against the average gene (regularization)
    if penalty > 0:
        for gene in genes:
            if gene != "_AVERAGE_":
                w[gene]["_AVERAGE_"] += penalty / 2
                w["_AVERAGE_"][gene] += penalty / 2

    return w


def bradley_terry(w, iterations=10):
    """
    Iterative algorithm for the Bradley-Terry model (from Wikipedia).

    w[i][j] = number of times i beat j.
    Returns p, where p[i] is item i's strength and the chance
    that i beats j is p[i] / (p[i] + p[j]).
    """
    items = list(w)

    # Start with every strength equal to 1
    p = {i: 1.0 for i in items}

    for _ in range(iterations):

        # Update rule
        new_p = {}
        for i in items:
            numerator = 0.0
            denominator = 0.0
            for j in items:
                if j == i:
                    continue
                numerator += w[i][j] * p[j] / (p[i] + p[j])
                denominator += w[j][i] / (p[i] + p[j])
            new_p[i] = numerator / denominator

        # Normalize: divide by the geometric mean
        log_sum = sum(math.log(new_p[i]) for i in items)
        geometric_mean = math.exp(log_sum / len(items))
        p = {i: new_p[i] / geometric_mean for i in items}

    return p


def load_comparisons(pass_dirs):
    """
    Read (gene1, gene2, score) tuples from rank_wrapper pass folders.

    Each comparison folder is named like "pos00_PRDM13_vs_NR4A2" and holds a
    results.tsv with one row per iteration. Score is the support for
    Hypothesis1 (gene1), so it is averaged across iterations and used as-is.
    """
    comparisons = []
    for pass_dir in pass_dirs:
        output_dir = os.path.join(pass_dir, "output")
        if not os.path.isdir(output_dir):
            continue
        for folder in sorted(os.listdir(output_dir)):
            results_file = os.path.join(output_dir, folder, "results.tsv")
            if not os.path.isfile(results_file):
                continue
            # Drop the "pos00_" / "validate00_" / "tie000_" prefix
            h1, h2 = folder.split("_", 1)[1].split("_vs_")
            scores = []
            with open(results_file) as f:
                header = f.readline().rstrip("\n").split("\t")
                score_col = header.index("Score")
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        scores.append(float(line.rstrip("\n").split("\t")[score_col]))
                    except (ValueError, IndexError):
                        logger.warning(f"Skipping non-numeric score in {results_file}")
            if scores:
                comparisons.append((h1, h2, sum(scores) / len(scores)))
    return comparisons


def rank(comparisons, label, out_file, penalty=1.0, iterations=100):
    w = build_win_table(comparisons, penalty=penalty)
    p = bradley_terry(w, iterations=iterations)

    p.pop("_AVERAGE_", None)  # drop the imaginary gene from the results

    ranking = sorted(p.items(), key=lambda item: item[1], reverse=True)
    lines = [f"=== Bradley-Terry: {label} ==="]
    lines += [f"{rank}. {gene:8s} {strength:.3f}"
              for rank, (gene, strength) in enumerate(ranking, start=1)]
    logger.info("\n".join(lines))

    with open(out_file, "w") as f:
        # Label is repeated on every row so the R plots can title each result
        f.write("Rank\tGene\tStrength\tLabel\n")
        for rank, (gene, strength) in enumerate(ranking, start=1):
            f.write(f"{rank}\t{gene}\t{strength:.6f}\t{label}\n")
    logger.info(f"Saved to {out_file}")
    return ranking


def make_plots(run_dir, out_dir, celltype_file=None, initial_passes=1):
    """
    Run the R plotting scripts on the TSVs in out_dir, saving PDFs there.
    Skipped if Rscript isn't available.

    run_dir: rank_wrapper output directory (for FINAL_RANKING.txt).
    celltype_file: optional TSV (columns: gene, celltype with values amacrine /
    ganglion / both) used to colour genes in the shared-axis timeline and
    the bump/slope plots.
    initial_passes: which early-pass result to compare against all passes
    (selects the TSV and output file suffix in the R scripts).
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        logger.warning("Rscript not found on PATH; skipping Bradley-Terry plots")
        return
    for script in PLOT_SCRIPTS:
        cmd = [rscript, os.path.join(SCRIPT_DIR, script), run_dir,
               "--out-dir", out_dir, "--initial-passes", str(initial_passes)]
        if celltype_file:
            cmd += ["--celltypes", celltype_file]
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        if result.returncode != 0:
            logger.warning(f"{script} failed:\n{result.stderr.strip()}")
        else:
            for line in result.stderr.splitlines():
                if line.startswith("Saved to"):
                    logger.info(line)


def pass_number(folder):
    """Pass number from a folder name like "pass_03_escape"."""
    return int(folder.split("_")[1])


def count_passes(run_dir, pass_folders):
    """
    Number of sort passes in the run. Prefer ranking_history.json, since a pass
    made entirely of cache hits doesn't create a folder; otherwise use the
    highest pass number among the folders.
    """
    history_file = os.path.join(run_dir, "ranking_history.json")
    if os.path.isfile(history_file):
        with open(history_file) as f:
            passes = json.load(f).get("passes", [])
        return sum(1 for p in passes if isinstance(p.get("pass"), int))
    return max((pass_number(d) for d in pass_folders), default=0)


def initial_tsv_name(initial_passes):
    """bradley_terry_initial.tsv for pass 1 only, else bradley_terry_first<N>passes.tsv"""
    if initial_passes == 1:
        return "bradley_terry_initial.tsv"
    return f"bradley_terry_first{initial_passes}passes.tsv"


def make_out_dir(run_dir):
    """Create and return a new run_dir/output_BT_<timestamp>/ folder."""
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(run_dir, f"output_BT_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_bradley_terry(run_dir, penalty=1.0, iterations=100, plots=True, celltype_file=None,
                      initial_passes=1, out_dir=None):
    """
    Fit Bradley-Terry on a rank_wrapper output directory and write the results
    and plots into out_dir (default: a new run_dir/output_BT_<timestamp>/ folder):
      - for each N in `initial_passes` (an int or a list of ints), the first N
        passes (bradley_terry_initial.tsv for 1, bradley_terry_first<N>passes.tsv
        otherwise; includes any escape/validate comparisons in those passes)
      - every pass plus the tie-break round robin (bradley_terry_all.tsv)
    Returns the output folder.
    """
    if isinstance(initial_passes, int):
        initial_passes = [initial_passes]
    initial_passes = sorted(set(initial_passes))
    if not initial_passes or initial_passes[0] < 1:
        raise ValueError("initial_passes must be at least 1")

    if out_dir is None:
        out_dir = make_out_dir(run_dir)
    logger.info(f"Bradley-Terry output dir: {out_dir}")

    subdirs = sorted(os.listdir(run_dir))
    pass_folders = [d for d in subdirs if d.startswith("pass_")]
    has_tie_break = "tie_break_round_robin" in subdirs
    n_passes = count_passes(run_dir, pass_folders)

    # Result 1: the first N passes, for each N
    for n in initial_passes:
        if n >= n_passes:
            logger.warning(f"initial_passes={n} covers all {n_passes} pass(es) in this run")
        early = [d for d in pass_folders if pass_number(d) <= n]
        if not early:
            logger.warning(f"No pass folders <= {n} found in {run_dir}; skipping that result")
            continue
        comparisons = load_comparisons([os.path.join(run_dir, d) for d in early])
        if n == 1:
            label = f"Initial comparisons (pass 1, {len(comparisons)} comparisons)"
        else:
            label = f"First {n} passes ({len(comparisons)} comparisons)"
        rank(comparisons, label, os.path.join(out_dir, initial_tsv_name(n)), penalty, iterations)

    # Result 2: every pass (plus the tie-break round robin)
    all_dirs = pass_folders + (["tie_break_round_robin"] if has_tie_break else [])
    comparisons = load_comparisons([os.path.join(run_dir, d) for d in all_dirs])
    passes_desc = f"{n_passes} passes" + (" + tie-break" if has_tie_break else "")
    rank(comparisons, f"All passes ({passes_desc}, {len(comparisons)} comparisons)",
         os.path.join(out_dir, "bradley_terry_all.tsv"), penalty, iterations)

    if plots:
        for n in initial_passes:
            if os.path.isfile(os.path.join(out_dir, initial_tsv_name(n))):
                make_plots(run_dir, out_dir, celltype_file, n)
    return out_dir


def main():
    # Imported here so rank_wrapper.py can use run_bradley_terry() without cmdlogtime
    import cmdlogtime

    # cmdlogtime parses the arguments in bradleyterry_model_commandline.txt and
    # records the args, this script, installed packages and timing for the run
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    run_dir = my_args["run_dir"]
    out_dir = make_out_dir(run_dir)

    # cmdlogtime.begin() created its own bookkeeping directory (addl/parms/
    # script/pkgs logs + err.txt) directly under run_dir; move it into the
    # output folder so everything from this run lives together. The
    # addl_logfile handle stays valid (same inode) after the move, so
    # cmdlogtime.end() below still writes to the right place.
    shutil.move(my_args["out_dir"], os.path.join(out_dir, "cmdlogtime"))

    # cmdlogtime sends stderr to err.txt, so log to stdout and to its addl_log
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger().addHandler(logging.StreamHandler(addl_logfile))

    initial_passes = [int(n) for n in str(my_args["initial_passes"]).split(",") if n.strip()]
    celltype_file = my_args["celltypes"] or None
    if celltype_file and not os.path.isfile(celltype_file):
        sys.exit(f"Cell-type file not found: {celltype_file}")

    run_bradley_terry(run_dir, my_args["penalty"], my_args["iterations"], not my_args["no_plots"],
                      celltype_file, initial_passes, out_dir=out_dir)

    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()
