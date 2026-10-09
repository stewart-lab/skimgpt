"""Visualise separate KM-GPT runs over time (averaged over iterations).

Reads ``km_with_gpt_wrapper_results.tsv`` (one row per censor_year x
Hypothesis x iter_number), averages the score across iterations for each
(censor_year, Hypothesis), and produces three PDF variants (line, point,
point-line) of the mean score by censor year, coloured by hypothesis.
Point and point-line plots show +/- SE error bars across iterations.
Mirrors the R script ``visualize_kmgpt_averaged.R``.
"""

import os
import platform
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import cmdlogtime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "plot_separate_runs_commandline.txt")


def _find_score_col(df, score_col):
    """Return the score column: the one given, else the single ``*_score`` column."""
    if score_col:
        if score_col not in df.columns:
            raise ValueError(f"Score column '{score_col}' not found in {list(df.columns)}")
        return score_col
    candidates = [c for c in df.columns if c.endswith("_score")]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not auto-detect score column (found {candidates}); pass --score_col"
        )
    return candidates[0]


def _average_iterations(df, score_col):
    """Collapse iterations to one row per (censor_year, Hypothesis): mean/sd/n/se."""
    grouped = df.groupby(["censor_year", "Hypothesis"], sort=False)[score_col]
    avg = grouped.agg(
        mean_score="mean",
        sd_score="std",  # sample sd (ddof=1), same as R's sd()
        n_iter="count",  # non-NaN count
    ).reset_index()
    avg["se_score"] = avg["sd_score"] / np.sqrt(avg["n_iter"])
    return avg.sort_values(["censor_year", "Hypothesis"]).reset_index(drop=True)


def _ggplot_align(hjust, vjust):
    """Map ggplot hjust/vjust (in the text's own frame) to matplotlib ha/va.

    Used with ``rotation_mode="anchor"`` so alignment is applied before
    rotation, like ggplot's annotate(angle=90).
    """
    ha = "left" if hjust <= 0.25 else ("right" if hjust >= 0.75 else "center")
    va = "bottom" if vjust <= 0.25 else ("top" if vjust >= 0.75 else "center")
    return ha, va


def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    proj_path = my_args["projpath"]
    datatype = my_args.get("datatype", "km")
    d_date = my_args.get("discover") or None
    a_date = my_args.get("accept") or None
    x_date = my_args.get("x_date") or None
    title = my_args.get("title") or None
    x_interval = int(my_args.get("xinterval", 1))
    score_col = my_args.get("score_col") or None

    label_str = my_args.get("labels") or None
    labelx = [s.strip() for s in label_str.split(",")] if label_str else None
    labels2 = [s.strip() for s in (my_args.get("labels2") or "discover,acceptance").split(",")]
    movex = [float(v) for v in (my_args.get("move") or "-0.1,1,-0.05,1").split(",")]
    movex = (movex + [-0.1, 1, -0.05, 1][len(movex):])[:4]  # pad to 4 elements

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(proj_path, f"output_visualization_{timestamp}")
    os.makedirs(output, exist_ok=True)
    print(output)

    # Read data
    if datatype == "km":
        filename = "km_with_gpt_wrapper_results.tsv"
    else:
        raise ValueError(f"Unsupported datatype '{datatype}'. Currently only 'km' is supported.")
    df = pd.read_csv(os.path.join(proj_path, filename), sep="\t")

    # Convert score column to numeric ("N/A" strings -> NaN)
    score_col = _find_score_col(df, score_col)
    df[score_col] = pd.to_numeric(df[score_col].replace("N/A", np.nan), errors="coerce")

    # Average score across iterations for each year/hypothesis
    km_data = _average_iterations(df, score_col)

    hyps = sorted(km_data["Hypothesis"].unique())  # alphabetical, like R factor levels
    colors = ["gold", "#433E85"]
    hyp_colors = {h: colors[i % len(colors)] for i, h in enumerate(hyps)}
    # Override legend labels if --labels provided
    if labelx and len(labelx) >= len(hyps):
        hyp_labels = {h: labelx[i] for i, h in enumerate(hyps)}
    else:
        hyp_labels = {h: h for h in hyps}

    all_years = sorted(km_data["censor_year"].unique())
    date_breaks = all_years[::x_interval]

    yc2 = 1.5  # y-position for annotations

    def _add_event_lines(ax):
        """Add discovery/acceptance/extra vertical lines with rotated labels.

        Label positions honour ``movex`` (hjust_d, vjust_d, hjust_a, vjust_a),
        matching the R version; the extra x_date line reuses the acceptance
        offsets, as in R.
        """
        events = [
            (d_date, "brown", "brown", labels2[0] if labels2 else None, movex[0], movex[1]),
            (a_date, "black", "black", labels2[1] if len(labels2) > 1 else None, movex[2], movex[3]),
            (x_date, "grey", "black", labels2[2] if len(labels2) > 2 else None, movex[2], movex[3]),
        ]
        for date, line_color, text_color, label, hjust, vjust in events:
            if date is None:
                continue
            ax.axvline(x=float(date), linestyle="--", color=line_color, linewidth=1)
            if label:
                ha, va = _ggplot_align(hjust, vjust)
                ax.text(float(date), yc2, label, rotation=90, rotation_mode="anchor",
                        ha=ha, va=va, color=text_color)
        # ggplot expands the y-range to include annotation positions
        lo, hi = ax.get_ylim()
        if hi < yc2:
            ax.set_ylim(lo, yc2 + 0.05 * (yc2 - lo))

    def _style_ax(ax):
        ax.set_ylabel("Mean score")
        ax.set_xlabel("Year")
        ax.set_xticks(date_breaks)
        ax.set_xticklabels([str(y) for y in date_breaks], rotation=90)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(title="Term", loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=7)

    def _draw_errorbars(ax, sub, color):
        ax.errorbar(sub["censor_year"], sub["mean_score"], yerr=sub["se_score"],
                    fmt="none", ecolor=color, alpha=0.6, capsize=2, linewidth=0.8)

    wrapped_title = textwrap.fill(title, 80) if title else ""
    full_path = os.path.abspath(output)
    wrapped_caption = "\n".join(textwrap.wrap(full_path, 130, break_on_hyphens=False))
    caption_height = 0.03 * (wrapped_caption.count("\n") + 1)

    def _save(fig, ax, name):
        _add_event_lines(ax)
        _style_ax(ax)
        if wrapped_title:
            fig.suptitle(wrapped_title, fontsize=10, x=0.02, ha="left")
        fig.text(0.99, 0.01, wrapped_caption, ha="right", va="bottom", fontsize=6, color="grey")
        fig.tight_layout(rect=[0, caption_height, 1, 0.97])
        fig.savefig(os.path.join(output, name))
        plt.close(fig)

    # --- Plot 1: line plot ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for h in hyps:
        sub = km_data[km_data["Hypothesis"] == h]
        ax1.plot(sub["censor_year"], sub["mean_score"], color=hyp_colors[h],
                 label=hyp_labels[h], linewidth=1, solid_capstyle="round")
    _save(fig1, ax1, "KM_GPT_scores_line.pdf")

    # --- Plot 2: point plot with SE error bars ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for h in hyps:
        sub = km_data[km_data["Hypothesis"] == h]
        _draw_errorbars(ax2, sub, hyp_colors[h])
        ax2.scatter(sub["censor_year"], sub["mean_score"], color=hyp_colors[h],
                    label=hyp_labels[h], s=12, zorder=3)
    _save(fig2, ax2, "KM_GPT_scores_points.pdf")

    # --- Plot 3: point-line plot with SE error bars ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for h in hyps:
        sub = km_data[km_data["Hypothesis"] == h]
        _draw_errorbars(ax3, sub, hyp_colors[h])
        ax3.plot(sub["censor_year"], sub["mean_score"], color=hyp_colors[h],
                 label=hyp_labels[h], linewidth=1, marker="o", markersize=3)
    _save(fig3, ax3, "KM_GPT_scores_point-lines.pdf")

    # Parameter list
    b1_term = hyps[0] if len(hyps) > 0 else ""
    b2_term = hyps[1] if len(hyps) > 1 else ""
    parameter_df = pd.DataFrame({
        "Parameter": ["filename", "ProjectPath", "title", "discover_date",
                      "acceptance_date", "term_B1", "term_B2", "score_column"],
        "Value": [filename, full_path, title, d_date, a_date, b1_term, b2_term, score_col],
    })
    print(parameter_df.to_string(index=False))
    parameter_df.to_csv(os.path.join(output, "parameters.csv"), index=False)

    # Averaged data, for records/checking
    km_data.to_csv(os.path.join(output, "averaged_scores_by_year_hypothesis.csv"), index=False)

    # Package versions (analogue of R sessionInfo())
    with open(os.path.join(output, "sessionInfo.txt"), "w") as fh:
        fh.write(f"python {sys.version}\nplatform {platform.platform()}\n")
        fh.write(f"pandas {pd.__version__}\nnumpy {np.__version__}\n")
        fh.write(f"matplotlib {matplotlib.__version__}\n")

    print(f"Plots saved to {output}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()
