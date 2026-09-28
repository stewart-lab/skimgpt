"""Bayesian timecourse of literature support for one hypothesis pair.

Walks one DCH hypothesis pair across censor years and plots the posterior
score with its HDI ribbon, above a stacked bar of unique abstracts per year
by label. The model itself - and its rationale - lives in
:mod:`skimgpt.stats.bayes_posterior`, shared with SKiM_web; this module only
loads results and draws. ``bayes_ci_updated.R`` is the R port.
"""

import os
import json
import shutil
from pathlib import Path

import cmdlogtime
import matplotlib.pyplot as plt
import numpy as np

from skimgpt.stats.bayes_posterior import (
    A0,
    RHO,
    SLOPE,
    call_from_result,
    label_counts,
    posterior_summary,
)

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "bayes_ci_updated_commandline.txt")

DPI = 96

TIE_BREAK_ORDER = ["supports_H1", "supports_H2", "both"]


def timecourse_data(data, level=0.95, hyp1_label="hyp1", hyp2_label="hyp2"):
    """
    Per-year summary statistics underlying the plots: the mean raw LLM score,
    the shrunk posterior score (mode), and the HDI bounds at `level`, all on
    the 0-100 scale; plus unique-abstract counts, per-iteration label
    averages, and the per-iteration H1 support proportion (0-1 scale, "both"
    split evenly between H1 and H2).
    """
    rows = []

    for year in sorted(data):
        calls = data[year]
        fit = posterior_summary(calls, level=level)

        counts = np.array(label_counts(calls), dtype=float)
        n_h1, n_h2, n_both = counts[:, 0], counts[:, 1], counts[:, 2]

        adj_h1 = n_h1 + n_both / 2
        denom = n_h1 + n_h2 + n_both
        with np.errstate(invalid="ignore", divide="ignore"):
            proportions = np.where(denom == 0, np.nan, adj_h1 / denom)
        valid_props = proportions[~np.isnan(proportions)]
        avg_proportion = float(np.mean(valid_props)) if valid_props.size else np.nan

        rows.append({
            "year": year,
            "hyp1": hyp1_label,
            "hyp2": hyp2_label,
            "mean_llm_score": fit.mean_llm_score,
            "posterior_score": fit.posterior_mode,
            "hdi_level": level,
            "hdi_lo": fit.hdi_lo,
            "hdi_hi": fit.hdi_hi,
            "total_unique_abstracts": fit.n_unique_pmids,
            "avg_abstracts_per_iteration": float(np.mean(denom)),
            "avg_supports_H1": float(np.mean(n_h1)),
            "avg_supports_H2": float(np.mean(n_h2)),
            "avg_both": float(np.mean(n_both)),
            "avg_proportion": avg_proportion,
        })

    return rows


def write_timecourse_csv(data, path, level=0.95, hyp1_label="hyp1", hyp2_label="hyp2"):
    import csv

    rows = timecourse_data(data, level=level, hyp1_label=hyp1_label, hyp2_label=hyp2_label)
    fieldnames = ["year", "hyp1", "hyp2", "mean_llm_score", "posterior_score",
                  "hdi_level", "hdi_lo", "hdi_hi", "total_unique_abstracts",
                  "avg_abstracts_per_iteration", "avg_supports_H1", "avg_supports_H2",
                  "avg_both", "avg_proportion"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_milestones(ax, proposed_date=None, decision_date=None, decision_label=None,
                      reconsidered_date=None, show_labels=True):
    """
    Vertical dashed reference lines shared across the timecourse and
    label-counts plots: when the hypothesis was proposed (dark red), when
    the literature accepted/rejected it (black - decision_label must be
    "accepted", "rejected" or "unknown"), and optionally when it was reconsidered
    (dark grey).
    """
    if decision_date is not None and decision_label not in ("accepted", "rejected","unknown"):
        raise ValueError('decision_label must be "accepted" or "rejected" or "unknown"')

    milestones = []
    if proposed_date is not None:
        milestones.append((proposed_date, "darkred", "proposed"))
    if decision_date is not None:
        milestones.append((decision_date, "black", decision_label))
    if reconsidered_date is not None:
        milestones.append((reconsidered_date, "dimgray", "reconsidered"))

    for x, color, label in milestones:
        ax.axvline(x=x, color=color, linestyle="--", linewidth=1.2, zorder=4)
        if show_labels:
            ax.text(x, 0.02, label, transform=ax.get_xaxis_transform(),
                    rotation=90, ha="right", va="bottom", fontsize=8, color=color)


def _draw_timecourse(ax, data, level=0.95, title=None,
                      hyp1_label="hyp1", hyp2_label="hyp2", dot_size=22,
                      proposed_date=None, decision_date=None, decision_label=None,
                      reconsidered_date=None, show_milestone_labels=True):
    rows = timecourse_data(data, level=level, hyp1_label=hyp1_label, hyp2_label=hyp2_label)

    years = [r["year"] for r in rows]
    lo_y = [r["hdi_lo"] for r in rows]
    hi_y = [r["hdi_hi"] for r in rows]
    mode_y = [r["posterior_score"] for r in rows]
    dot_y = [r["mean_llm_score"] for r in rows]

    # 50 = hypotheses equally likely
    ax.axhline(50, color="black", linewidth=1, linestyle="--")

    # HDI ribbon
    ax.fill_between(years, lo_y, hi_y, color="gray", alpha=0.35,
                     linewidth=0, label=f"{level:.0%} HDI")

    ax.plot(years, mode_y, color="darkblue", linewidth=2, label="posterior")

    ax.scatter(years, dot_y, color="black", s=dot_size, edgecolor="white",
               linewidth=1, label="LLM score", zorder=5)

    _draw_milestones(ax, proposed_date=proposed_date, decision_date=decision_date,
                      decision_label=decision_label, reconsidered_date=reconsidered_date,
                      show_labels=show_milestone_labels)

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("score")
    ax.set_title(title or "Literature support over time")

    # label the poles so the axis is readable
    ax.text(0.01, 97, f"favors {hyp1_label}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="gray")
    ax.text(0.01, 3, f"favors {hyp2_label}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="gray")

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)


def plot_timecourse(data, level=0.95, title=None,
                     hyp1_label="hyp1", hyp2_label="hyp2", dot_size=22,
                     proposed_date=None, decision_date=None, decision_label=None,
                     reconsidered_date=None):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)

    _draw_timecourse(ax, data, level=level, title=title,
                      hyp1_label=hyp1_label, hyp2_label=hyp2_label,
                      dot_size=dot_size,
                      proposed_date=proposed_date, decision_date=decision_date,
                      decision_label=decision_label, reconsidered_date=reconsidered_date)

    ax.set_xlim(1975, 2025)
    ax.set_xticks(range(1975, 2026, 1))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_xlabel("year")
    fig.tight_layout()

    return fig


def _draw_label_counts(ax, data, hyp1_label="hyp1", hyp2_label="hyp2",
                        title=None, normalize=False, show_title=True,
                        proposed_date=None, decision_date=None, decision_label=None,
                        reconsidered_date=None, show_milestone_labels=True):
    """
    Stacked bar of unique abstracts per year, split by label.

    data: {year: [(llm_score, [(pmid, label), ...]), ...]}

    A PMID seen in multiple calls is counted once; if calls disagree on its
    label, the most common label wins (ties broken by tie_break_order).
    Bars are stacked bottom-to-top as: both, hyp2, hyp1 - so hyp1 support
    ends up on top of the bar.
    """
    from collections import Counter

    tie_break_order = TIE_BREAK_ORDER
    stack_order = ["both", "supports_H2", "supports_H1"]
    label_colors = {
        "supports_H1": "darkorange",
        "supports_H2": "purple",
        "both": "green",
    }
    label_display = {
        "supports_H1": hyp1_label,
        "supports_H2": hyp2_label,
        "both": "both",
    }

    years = sorted(data)
    counts = {}

    for year in years:
        # pmid -> Counter of labels assigned across calls
        labels_by_pmid = {}
        for _, pmids in data[year]:
            for pmid, label in pmids:
                if label not in label_colors:
                    continue
                labels_by_pmid.setdefault(pmid, Counter())[label] += 1

        year_counts = Counter()
        for label_counts in labels_by_pmid.values():
            best = max(label_counts.items(),
                       key=lambda kv: (kv[1], -tie_break_order.index(kv[0])))[0]
            year_counts[best] += 1

        counts[year] = year_counts

    totals = {y: sum(counts[y].values()) for y in years}

    bottom = np.zeros(len(years))
    for label in stack_order:
        raw = [counts[y].get(label, 0) for y in years]
        if normalize:
            y_vals = np.array([100 * n / totals[y] if totals[y] else 0
                                for n, y in zip(raw, years)])
        else:
            y_vals = np.array(raw, dtype=float)

        ax.bar(years, y_vals, width=0.6, bottom=bottom, color=label_colors[label],
               linewidth=0, label=label_display[label])
        bottom += y_vals

    _draw_milestones(ax, proposed_date=proposed_date, decision_date=decision_date,
                      decision_label=decision_label, reconsidered_date=reconsidered_date,
                      show_labels=show_milestone_labels)

    if normalize:
        ax.set_ylim(0, 100)
    ax.set_ylabel("% of abstracts" if normalize else "unique abstracts")
    if show_title:
        ax.set_title(title or ("Abstract labels by year"
                                + (" (proportion)" if normalize else "")))

    ax.spines[["top", "right"]].set_visible(False)
    # stack_order is bottom-to-top; reverse so the legend reads top-to-bottom
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="label",
              loc="upper left", fontsize=8)

    return years


def plot_label_counts(data, title=None, normalize=False,
                       hyp1_label="hyp1", hyp2_label="hyp2",
                       proposed_date=None, decision_date=None, decision_label=None,
                       reconsidered_date=None):
    fig, ax = plt.subplots(figsize=(8, 3), dpi=DPI)

    years = _draw_label_counts(ax, data, hyp1_label=hyp1_label,
                                hyp2_label=hyp2_label, title=title,
                                normalize=normalize,
                                proposed_date=proposed_date, decision_date=decision_date,
                                decision_label=decision_label,
                                reconsidered_date=reconsidered_date)

    ax.set_xticks(range(min(years), max(years) + 1, 1))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_xlabel("year")
    fig.tight_layout()

    return fig


def plot_combined(data, hyp1_label="hyp1", hyp2_label="hyp2", level=0.95,
                   title=None, normalize=False, dot_size=22,
                   proposed_date=None, decision_date=None, decision_label=None,
                   reconsidered_date=None):
    """Timecourse plot stacked above the label-counts plot, sharing a year axis."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), dpi=DPI, sharex=True,
        gridspec_kw={"height_ratios": [5, 3], "hspace": 0.08},
    )

    _draw_timecourse(ax1, data, level=level, title=title,
                      hyp1_label=hyp1_label, hyp2_label=hyp2_label,
                      dot_size=dot_size,
                      proposed_date=proposed_date, decision_date=decision_date,
                      decision_label=decision_label, reconsidered_date=reconsidered_date,
                      show_milestone_labels=True)
    _draw_label_counts(ax2, data, hyp1_label=hyp1_label, hyp2_label=hyp2_label,
                        normalize=normalize, show_title=False,
                        proposed_date=proposed_date, decision_date=decision_date,
                        decision_label=decision_label, reconsidered_date=reconsidered_date,
                        show_milestone_labels=False)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_xlim(1975, 2025)
    ax2.set_xticks(range(1975, 2026, 1))
    ax2.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax2.set_xlabel("year")

    fig.tight_layout()

    return fig


def plot_llm_vs_proportion(data, level=0.95, title=None,
                            hyp1_label="hyp1", hyp2_label="hyp2",
                            proposed_date=None, decision_date=None, decision_label=None,
                            reconsidered_date=None, show_milestone_labels=True):
    """
    Mean LLM score (rescaled to 0-1) vs. the average per-iteration proportion
    of abstracts supporting H1 ("both" split evenly), per year.
    """
    rows = timecourse_data(data, level=level, hyp1_label=hyp1_label, hyp2_label=hyp2_label)

    years = [r["year"] for r in rows]
    llm_y = [r["mean_llm_score"] / 100 for r in rows]
    prop_y = [r["avg_proportion"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)

    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")

    ax.plot(years, llm_y, color="darkblue", linewidth=2, marker="o", markersize=5,
            label="LLM score")
    ax.plot(years, prop_y, color="darkorange", linewidth=2, marker="o", markersize=5,
            label="abstract proportion")

    _draw_milestones(ax, proposed_date=proposed_date, decision_date=decision_date,
                      decision_label=decision_label, reconsidered_date=reconsidered_date,
                      show_labels=show_milestone_labels)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("value (0-1)")
    ax.set_title(title or "LLM score vs. abstract support proportion")

    ax.set_xlim(1975, 2025)
    ax.set_xticks(range(1975, 2026, 1))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_xlabel("year")

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    return fig


def get_data(data_dir: str):
    data = dict()

    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory does not exist: {data_dir}")

    # get iteration result .json
    json_files = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for f in filenames:
            path = os.path.join(dirpath, f)
            if f.endswith("gpt_direct_comp.json") and ".backup" not in path:
                json_files.append(os.path.join(dirpath, f))

    for iteration_json in json_files:
        # get config.json for this iteration (nearest ancestor directory)
        config_json = None
        config_dir = os.path.dirname(iteration_json)
        while True:
            candidate = os.path.join(config_dir, "config.json")
            if os.path.isfile(candidate):
                config_json = candidate
                break
            parent = os.path.dirname(config_dir)
            if parent == config_dir:  # reached filesystem root without finding one
                break
            config_dir = parent
        if config_json is None:
            raise FileNotFoundError(f"No config.json found for {iteration_json}")

        # read config.json to get the censor year
        with open(config_json, 'r') as f:
            config_json_content = json.load(f)
        
        km = config_json_content["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]
        year = km.get("censor_year_upper")
        if year is None:
            year = km["km_with_gpt"]["censor_year_upper"]

        # read the iteration result .json and add the data to the dictionary
        with open(iteration_json, 'r') as f:
            iter_json_content = json.load(f)

        hyp_eval = iter_json_content[0]["Hypothesis_Comparison"]["Result"]

        if not hyp_eval:
            print(f"no result for {iteration_json}")
            continue

        data.setdefault(year, []).append(call_from_result(hyp_eval[0]))

    return data


def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    data_dir = my_args["data_dir"]
    hyp1_label = my_args["hyp1_label"]
    hyp2_label = my_args["hyp2_label"]
    level = my_args["level"]
    dot_size = my_args["dot_size"]
    normalize = my_args["normalize"]
    title = my_args.get("title") or None
    proposed_date = my_args.get("proposed_date")
    decision_date = my_args.get("decision_date")
    decision_label = my_args.get("decision_label") or None
    reconsidered_date = my_args.get("reconsidered_date")

    output_dir = os.path.join(data_dir, f"output_model_{pretty_start_time}")
    os.makedirs(output_dir, exist_ok=True)

    # cmdlogtime.begin() already created its own bookkeeping directory
    # (addl/parms/script/pkgs logs + err.txt) directly under data_dir; move
    # it into our output folder so everything from this run lives together.
    # The addl_logfile handle stays valid (same inode) after the move, so
    # cmdlogtime.end() below still writes to the right place.
    shutil.move(my_args["out_dir"], os.path.join(output_dir, "cmdlogtime"))

    data = get_data(data_dir)

    fig = plot_combined(data, hyp1_label=hyp1_label, hyp2_label=hyp2_label,
                         level=level, title=title, normalize=normalize, dot_size=dot_size,
                         proposed_date=proposed_date, decision_date=decision_date,
                         decision_label=decision_label, reconsidered_date=reconsidered_date)
    fig.savefig(os.path.join(output_dir, "combined.pdf"))

    write_timecourse_csv(data, os.path.join(output_dir, "timecourse_data.csv"),
                          level=level, hyp1_label=hyp1_label, hyp2_label=hyp2_label)

    fig = plot_llm_vs_proportion(data, level=level, title=title,
                                  hyp1_label=hyp1_label, hyp2_label=hyp2_label,
                                  proposed_date=proposed_date, decision_date=decision_date,
                                  decision_label=decision_label, reconsidered_date=reconsidered_date)
    fig.savefig(os.path.join(output_dir, "llm_vs_proportion.pdf"))

    # reproducibility: record the exact args and model constants for this run
    import csv
    parameters = {
        "data_dir": data_dir, "hyp1_label": hyp1_label, "hyp2_label": hyp2_label,
        "level": level, "dot_size": dot_size, "title": title or "",
        "normalize": normalize, "proposed_date": proposed_date,
        "decision_date": decision_date, "decision_label": decision_label,
        "reconsidered_date": reconsidered_date, "A0": A0, "rho": RHO, "slope": SLOPE,
    }
    with open(os.path.join(output_dir, "parameters.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerows(parameters.items())

    print(f"Plots and data saved to {output_dir}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()