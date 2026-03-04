"""Visualise separate KM-GPT runs over time.

Produces three PDF variants (line, point, point-line) of O3 scores by
censor year, coloured by B-term.  Mirrors the R script
``visualize_separate_kmgpt_runs.R``.
"""

import os
import textwrap
from datetime import datetime
from pathlib import Path

import cmdlogtime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "plot_separate_runs_commandline.txt")


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

    label_str = my_args.get("labels") or None
    labelx = [s.strip() for s in label_str.split(",")] if label_str else None
    labels2 = [s.strip() for s in my_args.get("labels2", "discover,acceptance").split(",")]
    movex = [float(v) for v in my_args.get("move", "-0.1,1,-0.05,1").split(",")]
    movex = (movex + [-0.1, 1, -0.05, 1])[:4]  # pad to 4 elements

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(proj_path, f"output_visualization_{timestamp}")
    os.makedirs(output, exist_ok=True)

    # Read data
    if datatype == "km":
        in_file = os.path.join(proj_path, "km_with_gpt_wrapper_results.tsv")
    else:
        raise ValueError(f"Unsupported datatype '{datatype}'. Currently only 'km' is supported.")
    df = pd.read_csv(in_file, sep="\t")

    # Replace N/A strings with NaN and convert score to float
    df["o3_score"] = pd.to_numeric(df["o3_score"].replace("N/A", np.nan))

    b_terms = df["B_term"].unique()
    colors = ["#FFD700", "#433E85"]
    term_colors = {t: colors[i % len(colors)] for i, t in enumerate(b_terms)}
    # Override legend labels if --labels provided
    if labelx and len(labelx) >= len(b_terms):
        term_labels = {t: labelx[i] for i, t in enumerate(b_terms)}
    else:
        term_labels = {t: t for t in b_terms}

    all_years = sorted(df["censor_year"].unique())
    date_breaks = all_years[::x_interval]

    yc2 = 1.5  # y-position for annotations

    def _add_event_lines(ax):
        """Add discovery/acceptance/extra vertical lines.

        Annotation positioning honours the ``movex`` parameter
        (hjust_d, vjust_d, hjust_a, vjust_a) matching the R version.
        """
        if d_date is not None:
            ax.axvline(x=float(d_date), linestyle="--", color="brown", linewidth=1)
            ax.annotate(
                labels2[0], xy=(float(d_date), yc2), rotation=90,
                ha="left" if movex[0] < 0 else "right",
                va="bottom" if movex[1] > 0 else "top",
                color="brown", fontsize=8,
            )
        if a_date is not None:
            ax.axvline(x=float(a_date), linestyle="--", color="black", linewidth=1)
            ax.annotate(
                labels2[1] if len(labels2) > 1 else "acceptance",
                xy=(float(a_date), yc2), rotation=90,
                ha="left" if movex[2] < 0 else "right",
                va="bottom" if movex[3] > 0 else "top",
                color="black", fontsize=8,
            )
        if x_date is not None:
            ax.axvline(x=float(x_date), linestyle="--", color="grey", linewidth=1)
            if len(labels2) > 2:
                ax.annotate(
                    labels2[2], xy=(float(x_date), yc2), rotation=90,
                    ha="left" if movex[2] < 0 else "right",
                    va="bottom" if movex[3] > 0 else "top",
                    color="black", fontsize=8,
                )

    def _style_ax(ax):
        ax.set_ylabel("O3 score")
        ax.set_xlabel("Year")
        ax.set_xticks([y for y in all_years if y in date_breaks])
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.legend(title="Term", loc="best", fontsize=7)

    wrapped_title = textwrap.fill(title, 80) if title else ""
    full_path = output

    # --- Plot 1: line plot ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for term in b_terms:
        sub = df[df["B_term"] == term].sort_values("censor_year")
        ax1.plot(sub["censor_year"], sub["o3_score"], color=term_colors[term],
                 label=term_labels[term], linewidth=1)
    _add_event_lines(ax1)
    _style_ax(ax1)
    if wrapped_title:
        ax1.set_title(wrapped_title, fontsize=10)
    fig1.text(0.5, 0.01, full_path, ha="center", fontsize=6, color="grey")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1.savefig(os.path.join(output, "KM_GPT_scores_line.pdf"))
    plt.close(fig1)

    # --- Plot 2: point plot ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for term in b_terms:
        sub = df[df["B_term"] == term]
        ax2.scatter(sub["censor_year"], sub["o3_score"], color=term_colors[term],
                    label=term_labels[term], s=20, zorder=3)
    _add_event_lines(ax2)
    _style_ax(ax2)
    if wrapped_title:
        ax2.set_title(wrapped_title, fontsize=10)
    fig2.text(0.5, 0.01, full_path, ha="center", fontsize=6, color="grey")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig2.savefig(os.path.join(output, "KM_GPT_scores_points.pdf"))
    plt.close(fig2)

    # --- Plot 3: point-line plot ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for term in b_terms:
        sub = df[df["B_term"] == term].sort_values("censor_year")
        ax3.plot(sub["censor_year"], sub["o3_score"], color=term_colors[term],
                 label=term_labels[term], linewidth=1, marker="o", markersize=3)
    _add_event_lines(ax3)
    _style_ax(ax3)
    if wrapped_title:
        ax3.set_title(wrapped_title, fontsize=10)
    fig3.text(0.5, 0.01, full_path, ha="center", fontsize=6, color="grey")
    fig3.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig3.savefig(os.path.join(output, "KM_GPT_scores_point-lines.pdf"))
    plt.close(fig3)

    print(f"Plots saved to {output}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()
