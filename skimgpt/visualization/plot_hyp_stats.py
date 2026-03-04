"""Visualise hypothesis comparison statistics over time.

Produces a three-panel PDF (p-value scatter, statistic scatter, stacked
bar of supporting abstracts).  Mirrors the R script
``visualizeStatsHyp1vsHyp2.R``.
"""

import os
import textwrap
from datetime import datetime
from pathlib import Path

import cmdlogtime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "plot_hyp_stats_commandline.txt")

# Default colour palette (viridis-inspired + orange)
COLOR_3 = ["#433E85", "#51C56A", "#E69F00"]
COLOR_2 = ["#433E85", "#51C56A"]
BAR_COLORS = ["#E69F00", "#440154"]


def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    proj_path = my_args["projpath"]
    datatype = my_args.get("datatype", "km")
    stattype = my_args.get("stattype", "ratio_of_ratios_zprop")
    d_date = my_args.get("discover") or None
    a_date = my_args.get("accept") or None
    x_date = my_args.get("x_date") or None
    title_str = my_args.get("title") or None
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
        stats_file = os.path.join(proj_path, "km_hyp_stats.txt")
        kept_file = os.path.join(proj_path, "km_kept.txt")
    else:
        stats_file = os.path.join(proj_path, "skim_hyp_stats.txt")
        kept_file = os.path.join(proj_path, "skim_kept.txt")

    km_data = pd.read_csv(stats_file, sep="\t")
    km_aterm_data = pd.read_csv(kept_file, sep="\t")

    # A-term — use column name (handles both "A_Term" and "A.Term")
    a_term_col = [c for c in km_aterm_data.columns if c.replace(".", "_").lower() == "a_term"]
    a_term_col = a_term_col[0] if a_term_col else km_aterm_data.columns[1]
    aterm = km_aterm_data[a_term_col].unique()
    aterm_str = aterm[0] if len(aterm) > 0 else "unknown"

    # Split out ONLY1 and NOLINES rows
    km_data_only1 = km_data[km_data["StatType"] == "ONLY1"].copy()
    km_data_nolines = km_data[km_data["StatType"] == "NOLINES"].copy()

    # Filter for requested stat type
    km_stat = km_data[km_data["StatType"] == stattype].copy()
    km_stat["P-Value"] = pd.to_numeric(km_stat["P-Value"], errors="coerce")
    km_stat["Statistic"] = pd.to_numeric(km_stat["Statistic"], errors="coerce")

    maxstat = km_stat["Statistic"].max()
    minstat = km_stat["Statistic"].min()

    # Compute -log10(p-value), cap Inf at 300
    km_stat["neglog10Pval"] = -np.log10(km_stat["P-Value"].replace(0, np.nan))
    km_stat["neglog10Pval"] = km_stat["neglog10Pval"].replace(np.inf, 300).fillna(300)

    maxP = km_stat["neglog10Pval"].max()

    # Parse terms from first row
    if km_stat.empty:
        raise ValueError(f"No rows match StatType '{stattype}' in {stats_file}")
    raw_terms = km_stat["Terms"].iloc[0].split(",")
    term1_full = raw_terms[0].strip()
    term2_full = raw_terms[1].strip() if len(raw_terms) > 1 else ""
    term1_short = term1_full.split("|")[0].split()[0] if term1_full else "T1"
    term2_short = term2_full.split("|")[0].split()[0] if term2_full else "T2"

    # Simplify Terms column to "term1vs.term2"
    vs_label = f"{term1_short}vs.{term2_short}"
    km_stat["Terms"] = vs_label

    # Handle ONLY1 rows
    has_only1 = len(km_data_only1) > 0
    if has_only1:
        km_data_only1 = km_data_only1.copy()
        km_data_only1["OrigTerms"] = km_data_only1["Statistic"].astype(str)
        km_data_only1["Statistic"] = km_data_only1["OrigTerms"].apply(
            lambda t: maxstat if t.strip() == term1_full else minstat
        )
        km_data_only1["neglog10Pval"] = maxP
        km_data_only1["Terms"] = km_data_only1["OrigTerms"].apply(
            lambda t: f"{term1_short}_only" if t.strip() == term1_full else f"{term2_short}_only"
        )
        km_data_only1["P-Value"] = np.nan
        km_data_only1 = km_data_only1.drop(columns=["OrigTerms"], errors="ignore")

        km_stat = pd.concat([km_stat, km_data_only1], ignore_index=True)

        if len(km_data_nolines) > 0:
            km_data_nolines = km_data_nolines.copy()
            km_data_nolines["Terms"] = "NOLINES"
            km_data_nolines["Statistic"] = np.nan
            km_data_nolines["neglog10Pval"] = np.nan
            km_data_nolines["P-Value"] = np.nan
            km_stat = pd.concat([km_stat, km_data_nolines], ignore_index=True)

    km_stat = km_stat.sort_values("Year").reset_index(drop=True)

    # Wrap long term text (matches R str_wrap(Terms, width=20))
    km_stat["Terms"] = km_stat["Terms"].apply(
        lambda t: textwrap.fill(t, width=20) if isinstance(t, str) else t
    )

    # Determine colour vector
    unique_terms = km_stat["Terms"].dropna().unique()
    n_terms = len(unique_terms)
    if n_terms >= 3:
        color_vector = COLOR_3[:n_terms]
    elif n_terms == 2:
        color_vector = COLOR_2
    else:
        color_vector = [COLOR_3[0]]
    term_color_map = {t: color_vector[i % len(color_vector)] for i, t in enumerate(unique_terms)}

    # Label mapping for legend
    if labelx and has_only1:
        legend_map = {}
        for t in unique_terms:
            if t == vs_label:
                legend_map[t] = " vs ".join(labelx) if len(labelx) >= 2 else t
            elif "_only" in t:
                for lbl in labelx:
                    if t.startswith(lbl.split()[0]) or t.replace("_only", "") in [l.split("|")[0].split()[0] for l in labelx]:
                        legend_map[t] = t
                        break
                else:
                    legend_map[t] = t
            else:
                legend_map[t] = t
    elif labelx:
        legend_map = {t: " vs ".join(labelx) if t == vs_label else t for t in unique_terms}
    else:
        legend_map = {t: t for t in unique_terms}

    # x-axis reference year for annotations
    years = km_stat["Year"].dropna().unique()
    xc = years[-3] if len(years) >= 3 else years[-1] if len(years) > 0 else 2000

    # Approximate y positions for event-line annotations
    yc_pval = maxP - maxP / 3
    yc_stat = maxstat - maxstat / 3 if not np.isnan(maxstat) else 0

    # ---- Create the three-panel figure ----
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 3, 1]},
        sharex=False,
    )

    # --- Panel 1: -log10(p-value) ---
    for term in unique_terms:
        sub = km_stat[km_stat["Terms"] == term]
        ax1.scatter(
            sub["Year"], sub["neglog10Pval"],
            color=term_color_map[term], label=legend_map.get(term, term),
            s=20, zorder=3, edgecolors="none",
        )
    ax1.axhline(y=1.3, linestyle="--", color="blue", linewidth=0.8)
    ax1.annotate("p=0.05", xy=(xc, 1.3), fontsize=7, va="bottom")
    ax1.set_ylabel("Neg log10 Pvalue")
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.legend(title="Term", fontsize=7, loc="upper left")
    ax1.grid(False)

    # --- Panel 2: Statistic ---
    for term in unique_terms:
        sub = km_stat[km_stat["Terms"] == term]
        ax2.scatter(
            sub["Year"], sub["Statistic"],
            color=term_color_map[term], s=20, zorder=3, edgecolors="none",
        )
    ax2.axhline(y=0, linestyle="--", color="darkgrey", linewidth=0.8)
    ax2.annotate("zscore=0", xy=(xc, 1.3), fontsize=7, va="bottom")
    ax2.set_ylabel(stattype)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.grid(False)

    # Event lines on panels 1 & 2 (movex controls annotation positioning)
    for ax, yc in [(ax1, yc_pval), (ax2, yc_stat)]:
        if d_date is not None:
            ax.axvline(x=float(d_date), linestyle="--", color="brown", linewidth=0.8)
            ax.annotate(labels2[0], xy=(float(d_date), yc), rotation=90,
                        fontsize=7, color="brown",
                        ha="left" if movex[0] < 0 else "right",
                        va="bottom" if movex[1] > 0 else "top")
        if a_date is not None:
            ax.axvline(x=float(a_date), linestyle="--", color="black", linewidth=0.8)
            lbl2 = labels2[1] if len(labels2) > 1 else "acceptance"
            ax.annotate(lbl2, xy=(float(a_date), yc), rotation=90,
                        fontsize=7, color="black",
                        ha="left" if movex[2] < 0 else "right",
                        va="bottom" if movex[3] > 0 else "top")
        if x_date is not None:
            ax.axvline(x=float(x_date), linestyle="--", color="grey", linewidth=0.8)
            if len(labels2) > 2:
                ax.annotate(labels2[2], xy=(float(x_date), yc), rotation=90,
                            fontsize=7, color="black",
                            ha="left" if movex[2] < 0 else "right",
                            va="bottom" if movex[3] > 0 else "top")

    # --- Panel 3: Stacked bar of abstract counts ---
    # B-terms from kept file — resolve column names robustly
    def _find_col(df, candidates):
        """Find first matching column (case-insensitive, dot/underscore agnostic)."""
        norm = {c.replace(".", "_").lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.replace(".", "_").lower()
            if key in norm:
                return norm[key]
        return None

    b_col = _find_col(km_aterm_data, ["B_Term", "B.Term"]) or km_aterm_data.columns[3]
    ab_col = _find_col(km_aterm_data, ["AB_Count", "AB.Count"]) or km_aterm_data.columns[5]
    date_col = _find_col(km_aterm_data, ["Date"]) or km_aterm_data.columns[0]

    km_aterm_data[date_col] = pd.to_numeric(km_aterm_data[date_col], errors="coerce")
    km_aterm_data[ab_col] = pd.to_numeric(km_aterm_data[ab_col], errors="coerce")

    b_terms_unique = km_aterm_data[b_col].unique()
    bterm1_short = b_terms_unique[0].split("|")[0].split()[0] if len(b_terms_unique) > 0 else "B1"
    bterm2_short = b_terms_unique[1].split("|")[0].split()[0] if len(b_terms_unique) > 1 else "B2"

    all_dates = sorted(km_aterm_data[date_col].dropna().unique())
    date_breaks_bar = all_dates[::x_interval]

    # Pivot for stacking
    bar_data = km_aterm_data.pivot_table(
        index=date_col, columns=b_col, values=ab_col, aggfunc="sum"
    ).fillna(0)

    bar_width = 0.6
    bottoms = np.zeros(len(bar_data))
    for i, col in enumerate(bar_data.columns):
        short_name = col.split("|")[0].split()[0]
        ax3.bar(
            bar_data.index, bar_data[col], bottom=bottoms,
            width=bar_width, alpha=0.8,
            color=BAR_COLORS[i % len(BAR_COLORS)], label=short_name,
        )
        bottoms += bar_data[col].values

    ax3.set_ylabel("Number of abstracts")
    ax3.set_xlabel("Year")
    ax3.set_xticks([d for d in all_dates if d in date_breaks_bar])
    ax3.tick_params(axis="x", rotation=90)
    ax3.legend(title="Support type", fontsize=7, loc="upper left")
    ax3.grid(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    if d_date is not None:
        ax3.axvline(x=float(d_date), linestyle="--", color="brown", linewidth=0.8)
    if a_date is not None:
        ax3.axvline(x=float(a_date), linestyle="--", color="black", linewidth=0.8)

    # Title
    if title_str:
        wrapped_title = textwrap.fill(title_str, 80)
    else:
        first_year = km_stat["Year"].min()
        last_year = km_stat["Year"].max()
        default_title = (
            f"A term: {aterm_str}  Co-occurrence terms: {term1_short} vs. {term2_short}"
            f"  Years: {first_year} - {last_year}  {datatype} data"
        )
        wrapped_title = textwrap.fill(default_title, 80)

    fig.suptitle(wrapped_title, fontsize=10, y=0.99)
    full_path = output
    fig.text(0.5, 0.01, full_path, ha="center", fontsize=6, color="grey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save PDF
    first_year = km_stat["Year"].min()
    last_year = km_stat["Year"].max()
    pdf_name = f"{aterm_str}_{term1_short}vs.{term2_short}_years_{first_year}-{last_year}_{stattype}.pdf"
    fig.savefig(os.path.join(output, pdf_name))
    plt.close(fig)

    # Write parameters CSV
    params = pd.DataFrame({
        "Parameter": ["datatype", "ProjectPath", "StatType", "discover_date",
                       "acceptance_date", "term_A", "term_1", "term_2"],
        "Value": [datatype, proj_path, stattype, d_date, a_date,
                  aterm_str, term1_full, term2_full],
    })
    params.to_csv(os.path.join(output, "parameters.csv"), index=False)

    print(f"Plot saved to {output}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()
