# Compares many DIFFERENT hypothesis-pairs (each its own H1 vs H2, e.g. one
# per disease/topic) at a single timepoint, using the same closed-form Beta
# posterior model as bayes_ci_updated.py (n_effective shrinkage + exact HDI -
# see that file for the full model rationale). Where bayes_ci_updated.py walks
# one hypothesis-pair across many censor years to show a timecourse, this
# script walks many hypothesis-pair directories at one censor-year window and
# shows them side by side as a violin plot, ordered by posterior mode.
#
# Expected input: projpath contains one subdirectory per DCH run (e.g.
# "output_<timestamp>_<topic>_kmgptdch_<years>_<model>/", the standard
# SKiM-GPT DCH run-output naming), each holding results/iteration_N/
# *_km_with_gpt_direct_comp.json files. A subdirectory may itself contain
# several distinct hypothesis-pairs when it comes from an A-term-list run
# (one *_direct_comp.json basename per A term, e.g. one per gene, all
# comparing that gene against the same fixed B-term pair) - each such
# basename group is pooled across its own iterations into its own posterior
# and gets its own violin (no year-splitting within a group - this is a
# snapshot, not a timecourse). H1/H2 short labels are derived automatically
# from the JSON's "hypothesis1"/"hypothesis2" text by diffing out the shared
# wording they're templated from (see short_hypothesis_labels()); when a
# subdirectory holds multiple hypothesis-pairs, the varying term across
# their "hypothesis1" texts (e.g. the A term) is extracted the same way (see
# extract_varying_term()) and appended to the subdirectory-derived topic
# label so each pair gets its own axis entry instead of being pooled
# together.
#
# Python port of bayes_ci_multihyp_violin.R.

import os
import re
import sys
import json
import shutil
from pathlib import Path

import cmdlogtime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import gaussian_kde

# reuse the exact Beta-posterior model (n_effective, posterior_params, hdi_beta,
# A0/RHO/SLOPE) from bayes_ci_updated.py so both scripts always agree.
sys.path.insert(0, str(Path(__file__).parent))
from bayes_ci_updated import posterior_params, posterior_beta, hdi_beta  # noqa: E402

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "bayes_ci_violinplot_commandLine.txt")

DPI = 96

KEEP_LABELS = {"supports_H1", "supports_H2", "both"}


# ---------------------------------------------------------------------------
# Per-directory data loading: a directory can hold more than one distinct
# hypothesis-pair (one *_direct_comp.json basename per pair - e.g. one per A
# term, under an A-term-list DCH run). Group files by basename and pool each
# group's iterations into its own list of "calls", pulling hypothesis1/
# hypothesis2 straight from the JSON (same for every iteration within a
# group, so the first file in the group is enough). Returns one entry per
# distinct hypothesis-pair found.
#
# calls: [(llm_score, [(pmid, label), ...]), ...]  (same as bayes_ci_updated.py)
# ---------------------------------------------------------------------------

def load_hypothesis_groups(topic_dir):
    files = []
    for dirpath, _, filenames in os.walk(topic_dir):
        for f in filenames:
            path = os.path.join(dirpath, f)
            if f.endswith("gpt_direct_comp.json") and ".backup" not in path:
                files.append(path)
    if not files:
        return None

    by_basename = {}
    for path in sorted(files):
        by_basename.setdefault(os.path.basename(path), []).append(path)

    groups = []
    for basename in sorted(by_basename):
        calls = []
        hyp1_text = None
        hyp2_text = None

        for path in by_basename[basename]:
            with open(path, "r") as f:
                content = json.load(f)
            hc = content[0]["Hypothesis_Comparison"]
            if hyp1_text is None:
                hyp1_text = hc.get("hypothesis1")
                hyp2_text = hc.get("hypothesis2")

            if not hc.get("Result"):
                print(f"no result for {path}")
                continue

            result = hc["Result"][0]
            pmids = [(str(a["pmid"]), a["label"]) for a in result["per_abstract"]
                     if a["label"] in KEEP_LABELS]
            calls.append((result["score"], pmids))

        if calls:
            groups.append({"calls": calls, "hypothesis1": hyp1_text, "hypothesis2": hyp2_text})

    return groups or None


# ---------------------------------------------------------------------------
# Derive short H1/H2 labels by diffing out the wording hypothesis1/hypothesis2
# share (they're both instantiations of the same template, differing only in
# the term that was substituted in) - e.g.
#   "The main cause of Schizophrenia is due to dopamine signaling."
#   "The main cause of Schizophrenia is due to glutamate signaling."
# -> "dopamine" / "glutamate"
# Falls back to the full text if no common prefix/suffix is found.
# ---------------------------------------------------------------------------

def _clean(words):
    return re.sub(r"[.,;:]+$", "", " ".join(words)).strip()


def _middle_words(word_lists):
    """Words left in each list after removing the prefix/suffix common to all."""
    max_common = min((len(w) for w in word_lists), default=0)

    n_pre = 0
    while n_pre < max_common and len({w[n_pre] for w in word_lists}) == 1:
        n_pre += 1

    n_suf = 0
    max_suf = max_common - n_pre
    while n_suf < max_suf and len({w[len(w) - 1 - n_suf] for w in word_lists}) == 1:
        n_suf += 1

    return [w[n_pre:len(w) - n_suf] for w in word_lists]


def short_hypothesis_labels(h1, h2):
    w1 = h1.strip().split()
    w2 = h2.strip().split()
    mid1, mid2 = _middle_words([w1, w2])

    term1 = _clean(mid1) if mid1 else _clean(w1)
    term2 = _clean(mid2) if mid2 else _clean(w2)

    return term1, term2


# ---------------------------------------------------------------------------
# N-way generalization of short_hypothesis_labels()'s diff: given several
# strings templated from the same wording but with one term substituted
# (e.g. one hypothesis1 per A term, all sharing the same B-term comparison),
# diff out the prefix/suffix common to ALL of them and return each string's
# differing middle segment (the substituted term). Falls back to the full
# string for any input where no common prefix/suffix could be established.
# ---------------------------------------------------------------------------

def extract_varying_term(strings):
    word_lists = [s.strip().split() for s in strings]
    mids = _middle_words(word_lists)
    return [_clean(mid) if mid else _clean(w) for w, mid in zip(word_lists, mids)]


def topic_from_dirname(dir_name):
    """
    Short topic label from a run directory name: drops the "output_<timestamp>_"
    prefix, then everything from the job type ("_kmgpt"/"_kmgptdch") or the
    censor-year range ("_2020-2026") onward (which also drops the model suffix,
    e.g. "_terra"/"_o3") - e.g.
      output_20260914111316_Schizophrenia_2020-2026_terra -> Schizophrenia
      output_20260911133046_REM_sleep_kmgptdch_2020-2026_terra -> REM_sleep
    """
    topic = re.sub(r"^output_[0-9]+_", "", dir_name)
    topic = re.sub(r"(_kmgpt(dch)?(_|$)|_[0-9]{4}-[0-9]{4}(_|$)).*$", "", topic)
    return topic or dir_name


# ---------------------------------------------------------------------------
# Per-topic posterior summary (exact Beta(a,b), same math as bayes_ci_updated.py)
# ---------------------------------------------------------------------------

def summarize_topic(topic, hyp1_label, hyp2_label, hypothesis1, hypothesis2,
                    calls, level=0.95, n_samples=2000, rng=None):
    rng = rng or np.random.default_rng()

    a, b = posterior_params(calls)

    hdi_lo, hdi_hi = [100 * v for v in hdi_beta(a, b, level)]
    grid, p, _, _ = posterior_beta(calls, n_theta=300)
    posterior_mode = grid[p.argmax()] * 100
    # posterior_mean = (a / (a + b)) * 100
    scores = [s for s, _ in calls]

    all_pmids = {pmid for _, pmids in calls for pmid, _ in pmids}

    # Mean per-iteration count of abstracts labeled as supporting H1/H2 -
    # the raw evidence tally that accompanies the posterior estimate in the
    # side-by-side bar plot. Averaged (not summed) across calls/iterations so
    # topics are comparable regardless of how many iterations each ran.
    def count_label(lbl):
        return [sum(1 for _, label in pmids if label == lbl) for _, pmids in calls]

    summary = {
        "topic": topic, "hyp1_label": hyp1_label, "hyp2_label": hyp2_label,
        "hypothesis1": hypothesis1, "hypothesis2": hypothesis2,
        "n_calls": len(calls), "n_unique_pmids": len(all_pmids),
        "mean_llm_score": float(np.mean(scores)), "posterior_mode": posterior_mode,
        "hdi_level": level, "hdi_lo": hdi_lo, "hdi_hi": hdi_hi,
        "shape1": a, "shape2": b,
        "mean_support_h1": float(np.mean(count_label("supports_H1"))),
        "mean_support_h2": float(np.mean(count_label("supports_H2"))),
    }

    # samples for the violin shape: drawn directly from the exact posterior
    # Beta(a,b) - not a resampling/refitting step, just visualizing that
    # closed-form distribution.
    samples = pd.DataFrame({
        "topic": topic,
        "posterior": beta_dist.rvs(a, b, size=n_samples, random_state=rng) * 100,
    })

    return summary, samples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _violin_density(values, n_grid=512):
    """
    Gaussian KDE matching ggplot2::geom_violin(trim = FALSE): Silverman's
    rule-of-thumb bandwidth (R's bw.nrd0) and a grid extended 3 bandwidths
    past the data range.
    """
    values = np.asarray(values, dtype=float)
    sd = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = 0.9 * spread * len(values) ** (-0.2)
    if not np.isfinite(bw) or bw <= 0:
        bw = 1e-3

    grid = np.linspace(values.min() - 3 * bw, values.max() + 3 * bw, n_grid)
    kde = gaussian_kde(values, bw_method=bw / sd if sd > 0 else 1.0)
    return grid, kde(grid)


def plot_violin(summary_df, samples_df, level=0.95):
    """
    Left: horizontal violin of each topic's posterior (ordered by posterior
    mode, highest at top) with mode + HDI pointrange and the H2/H1 short
    labels just outside the 0/100 edges. Right: mean supporting-abstract
    counts per H1/H2, sharing the violin's topic ordering.
    """
    topics = list(summary_df["topic"])
    n = len(topics)
    positions = np.arange(n)

    plot_height = max(6, 0.45 * n + 2)
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(11, plot_height), dpi=DPI, sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    )

    # discrete viridis "C" (plasma) fill, one colour per topic in axis order
    colors = plt.colormaps["plasma"](np.linspace(0, 1, n)) if n > 1 else [plt.colormaps["plasma"](0.0)]

    # ggplot2's default scale = "area": every violin has the same area, so
    # scale all densities by the single largest density across topics
    densities = [_violin_density(samples_df.loc[samples_df["topic"] == t, "posterior"]) for t in topics]
    max_dens = max(d.max() for _, d in densities)
    half_width = 0.9 / 2

    for pos, (grid, dens), color in zip(positions, densities, colors):
        w = dens / max_dens * half_width
        ax.fill_between(grid, pos - w, pos + w, facecolor=color, edgecolor="black",
                        linewidth=0.5, zorder=2)

    # mode + HDI pointrange
    ax.hlines(positions, summary_df["hdi_lo"], summary_df["hdi_hi"],
              color="black", linewidth=1.2, zorder=3)
    ax.scatter(summary_df["posterior_mode"], positions, color="black", s=12, zorder=4)

    ax.axvline(50, linestyle="--", color="darkgrey", linewidth=1, zorder=1)

    # H1/H2 labels sit just outside the 0/100 edges (not deep in the margin -
    # just enough to clear a violin whose mean sits close to 0 or 100) -
    # topic name is left as the normal axis label, so it stays on the left
    # where it's always been, and isn't competing with these for space.
    for pos, h1, h2 in zip(positions, summary_df["hyp1_label"], summary_df["hyp2_label"]):
        ax.text(-6, pos, h2, ha="right", va="center", fontsize=7)
        ax.text(106, pos, h1, ha="left", va="center", fontsize=7)

    ax.set_xlim(-55, 155)
    ax.set_xticks(range(0, 101, 25))
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel(f"posterior score (mode, {level:.0%} HDI)")
    ax.grid(True, which="major", color="#ebebeb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # side-by-side bars of mean supporting-abstract counts per H1/H2
    # (position_dodge(0.8), width 0.7 -> bars 0.35 tall at +/-0.2)
    ax_bar.barh(positions - 0.2, summary_df["mean_support_h1"], height=0.35,
                color="#31688e", label="H1", zorder=2)
    ax_bar.barh(positions + 0.2, summary_df["mean_support_h2"], height=0.35,
                color="#8fd744", label="H2", zorder=2)
    ax_bar.set_xlabel("mean supporting abstracts")
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.grid(True, which="major", color="#ebebeb", linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  frameon=False, fontsize=8)

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    proj_path = os.path.abspath(my_args["projpath"])
    level = my_args["level"]
    n_samples = my_args["n_samples"]

    output_dir = os.path.join(proj_path, f"output_model_{pretty_start_time}")
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    # cmdlogtime.begin() already created its own bookkeeping directory
    # directly under proj_path; move it into our output folder so everything
    # from this run lives together (and it isn't mistaken for a run directory
    # below). The addl_logfile handle stays valid after the move.
    shutil.move(my_args["out_dir"], os.path.join(output_dir, "cmdlogtime"))

    # projpath can point either at a parent folder holding many DCH run
    # directories (the normal multi-topic case) or directly at a single run's
    # own output directory (which itself contains a top-level "results/"
    # folder). Without this check, the listing below would enumerate that
    # single run's own results/debug/src folders as if each were a separate
    # hypothesis-pair run, and "results" - which doesn't match the
    # output_<ts>_<topic>_kmgptdch naming topic_from_dirname() expects - would
    # leak into every topic label verbatim (e.g. "results: CHAT").
    if os.path.isdir(os.path.join(proj_path, "results")):
        dir_labels = [os.path.basename(proj_path)]
        dir_paths = [proj_path]
    else:
        dir_labels = sorted(
            d for d in os.listdir(proj_path)
            if os.path.isdir(os.path.join(proj_path, d))
            and not re.match(r"^output_(model|visualization)_", d)  # skip our own (and older) output folders
        )
        dir_paths = [os.path.join(proj_path, d) for d in dir_labels]

    rng = np.random.default_rng()
    summaries = []
    samples_list = []

    for d, full_dir in zip(dir_labels, dir_paths):
        groups = load_hypothesis_groups(full_dir)
        if groups is None:
            print(f"{d}: no hypothesis-comparison JSON found, skipping")
            continue

        dir_topic = topic_from_dirname(d)

        # A directory with more than one distinct hypothesis-pair (an A-term-list
        # run) needs a per-pair topic label, not one shared by the whole
        # directory - otherwise every pair's violin would be plotted as if it
        # were the same comparison. Derive that label from what varies across
        # the pairs' hypothesis1 texts (e.g. the A term).
        if len(groups) > 1:
            varying_terms = extract_varying_term([g["hypothesis1"] for g in groups])
        else:
            varying_terms = [None]

        for g, varying in zip(groups, varying_terms):
            hyp1_label, hyp2_label = short_hypothesis_labels(g["hypothesis1"], g["hypothesis2"])
            topic = f"{dir_topic}: {varying}" if len(groups) > 1 else dir_topic

            summary, samples = summarize_topic(
                topic=topic, hyp1_label=hyp1_label, hyp2_label=hyp2_label,
                hypothesis1=g["hypothesis1"], hypothesis2=g["hypothesis2"],
                calls=g["calls"], level=level, n_samples=n_samples, rng=rng,
            )

            print(f"{d} [{topic}]: ({hyp1_label} vs {hyp2_label}) - {len(g['calls'])} calls")

            summaries.append(summary)
            samples_list.append(samples)

    if not summaries:
        raise RuntimeError(f"No hypothesis-pair directories with usable data were found under {proj_path}")

    summary_df = pd.DataFrame(summaries)
    samples_df = pd.concat(samples_list, ignore_index=True)

    summary_df.to_csv(os.path.join(output_dir, "summary_stats.txt"), sep="\t", index=False)
    samples_df.to_csv(os.path.join(output_dir, "posterior_samples.txt"), sep="\t", index=False)

    # order topics by posterior mode, low to high (lowest at the bottom of the
    # plot, highest at the top)
    summary_df = summary_df.sort_values("posterior_mode", kind="stable").reset_index(drop=True)

    fig = plot_violin(summary_df, samples_df, level=level)
    fig.savefig(os.path.join(output_dir, "posterior_violin_plot.pdf"), bbox_inches="tight")

    print(f"Plots and data saved to {output_dir}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()
