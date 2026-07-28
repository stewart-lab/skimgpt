"""
KM-GPT Bayesian Posterior Visualization
Python conversion of the original R script.

Dependencies:
    pip install scipy numpy pandas matplotlib seaborn bayesian-credible-interval
    (or just: pip install scipy numpy pandas matplotlib seaborn) cmdlogtime

Usage:
    python kmgpt_bayesian_viz.py /path/to/your/data/
"""
import logging
import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cmdlogtime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from scipy.stats import beta as beta_dist
from scipy.special import gammaln
import seaborn as sns

logger = logging.getLogger(__name__)

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "bayes_ci_violinplot_commandLine.txt")


# ── Argument parsing ──────────────────────────────────────────────────────────

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Bayesian posterior visualization for KM-GPT results."
#     )
#     parser.add_argument(
#         "-p", "--projpath",
#         type=str,
#         required=True,
#         help="Path where input subdirectories (each containing results.tsv) are located."
#     )
#     return parser.parse_args()


# ── Beta parameter estimation ─────────────────────────────────────────────────

def ebeta_mle(data):
    """
    Estimate Beta distribution parameters via MLE using scipy.
    Equivalent to EnvStats::ebeta(score, method='mle') in R.
    """
    # Clip data away from 0/1 boundaries to avoid numerical issues
    data_clipped = np.clip(data, 1e-9, 1 - 1e-9)
    alpha_hat, beta_hat, _, _ = beta_dist.fit(data_clipped, floc=0, fscale=1)
    return alpha_hat, beta_hat


def ebeta_mom(mu, var):
    """
    Estimate Beta distribution parameters via method of moments (mean + variance).
    Equivalent to the custom estBetaParams() function in R.
    """
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta  = alpha * (1 / mu - 1)
    return alpha, beta


# ── Equal-tailed credible interval ───────────────────────────────────────────

def eti(samples, ci=0.95):
    """
    Equal-Tailed Interval (ETI) — equivalent to bayestestR::ci(method='ETI').
    Returns (CI_low, CI_high).
    """
    lower = (1 - ci) / 2
    upper = 1 - lower
    return np.quantile(samples, lower), np.quantile(samples, upper)


# ── Prior construction ────────────────────────────────────────────────────────

def build_prior_params(df, rel_abstracts):
    """
    Derive Beta prior parameters from H1/H2/Both abstract counts,
    mirroring the if/else logic in the R script.
    """
    nhyp1 = df["H1"].mean()
    nhyp2 = df["H2"].mean()

    if df["Both"].sum() != 0:
        both   = df["Both"].mean() / 2
        nhyp1 += both
        nhyp2 += both

    if rel_abstracts <= 50:
        alpha2 = nhyp1
        beta2  = nhyp2
    else:
        ntot   = nhyp1 + nhyp2
        prop1  = nhyp1 / ntot if ntot > 0 else 0.5
        prop2  = nhyp2 / ntot if ntot > 0 else 0.5
        alpha2 = prop1 * rel_abstracts
        beta2  = prop2 * rel_abstracts

    alpha2 = max(alpha2, 0.01)
    beta2  = max(beta2,  0.01)
    return alpha2, beta2


# ── Distribution sampling helper ─────────────────────────────────────────────

def sample_beta(a, b, n=1000, seed=None):
    """Sample n values from Beta(a, b). Clamps degenerate parameters."""
    a = max(a, 1e-22)
    b = max(b, 1e-22)
    rng = np.random.default_rng(seed)
    return rng.beta(a, b, n)


# ── Four-panel diagnostic plot ────────────────────────────────────────────────

def plot_beta_panels(likelihood_mle, likelihood_mom, prior_samples,
                     posterior2, ci_low, ci_high, output_path, a_term):
    """
    Reproduce the 2×2 grid of R plots:
      top-left:  likelihood MLE (orchid)
      top-right: likelihood MOM (gold)
      bot-left:  prior (red)
      bot-right: posterior with ETI lines (orange + violet)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{a_term} — Beta distributions", fontsize=13)

    panels = [
        (likelihood_mle, "orchid",   "Likelihood MLE",        axes[0, 0]),
        (likelihood_mom, "gold",     "Likelihood (MOM)",      axes[0, 1]),
        (prior_samples,  "red",      "Prior",                  axes[1, 0]),
        (posterior2,     "orange",   "Posterior",              axes[1, 1]),
    ]

    for samples, color, xlabel, ax in panels:
        xs = np.linspace(0, 1, 500)
        # Use KDE for a smooth density curve
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(samples, bw_method="scott")
            ys  = kde(xs)
            ax.fill_between(xs, ys, color=color, alpha=0.7)
            ax.plot(xs, ys, color=color, linewidth=1.2)
        except Exception:
            ax.hist(samples, bins=40, color=color, alpha=0.7, density=True)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if xlabel == "Posterior":
            ax.axvline(ci_low,  color="violet", linewidth=1.5)
            ax.axvline(ci_high, color="violet", linewidth=1.5)

    plt.tight_layout()
    fname = os.path.join(output_path, f"{a_term}_post_beta_distributions_withCI.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# ── Violin summary plot ───────────────────────────────────────────────────────

def plot_violin(df1, output_path):
    """
    Reproduce the R violin plot (p5):
      - Violin per A_term, filled by mean posterior (viridis C palette)
      - ETI mean ± CI error bars
      - Dashed line at 50
      - B1_term label at top, B2_term label at bottom of each violin
      - Horizontal orientation (coord_flip equivalent)
    """
    a_terms = df1["A_term"].unique()

    # Compute mean posterior per group for colour mapping
    group_means = df1.groupby("A_term")["posterior"].mean()

    # Build label lookup (B1 / B2 per A_term)
    label_df = (
        df1.groupby("A_term")
           .first()
           .reset_index()[["A_term", "B1_term", "B2_term"]]
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(a_terms) * 0.9)))

    cmap   = plt.colormaps["plasma"] #plt.cm.get_cmap("plasma")
    vmin   = group_means.min()
    vmax   = group_means.max()
    norm   = plt.Normalize(vmin=vmin, vmax=vmax)

    y_positions = np.arange(len(a_terms))

    for idx, term in enumerate(a_terms):
        data  = df1.loc[df1["A_term"] == term, "posterior"].values
        color = cmap(norm(group_means[term]))

        # Violin
        parts = ax.violinplot(
            data,
            positions=[idx],
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("none")
            pc.set_alpha(0.85)

        # ETI error bar + mean dot
        mean_val          = np.mean(data)
        ci_low, ci_high   = eti(data, ci=0.95)
        ax.plot([ci_low, ci_high], [idx, idx], color="black", linewidth=1.2, zorder=5)
        ax.scatter(mean_val, idx, color="black", s=20, zorder=6)

        # B-term labels
        row = label_df.loc[label_df["A_term"] == term].iloc[0]
        ax.text(
            99, idx + 0.28,
            row["B1_term"],
            ha="right", va="bottom", fontsize=7, color="#444"
        )
        ax.text(
            1, idx - 0.28,
            row["B2_term"],
            ha="left", va="top", fontsize=7, color="#444"
        )

    # Dashed line at 50
    ax.axvline(50, linestyle="--", color="darkgrey", linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(a_terms, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Posterior (0–100 scale)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(output_path, "KM-GPT_posterior_vln_plot.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Violin plot saved → {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    #args     = parse_args()
    #projPath = Path(args.projpath).resolve()
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    projPath = Path(my_args["projpath"])
    os.chdir(projPath)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output    = os.path.join(projPath, f"output_visualization_{timestamp}")
    os.makedirs(output, mode=0o777, exist_ok=True)
    print(f"Output directory: {output}")

    # Collect subdirectories (skip any pre-existing output_visualization_ dirs)
    dirs = [
        d for d in projPath.iterdir()
        if d.is_dir() and not d.name.startswith("output_visualization_")
    ]

    all_posts = []

    for d in dirs:
        tsv_path = d / "results.tsv"
        if not tsv_path.exists():
            print(f"  {d.name} — no results.tsv, skipping")
            continue

        print(f"  {d.name} — running")

        df = pd.read_csv(tsv_path, sep="\t")

        hyp1    = df["Hypothesis1"].iloc[0]
        hyp2    = df["Hypothesis2"].iloc[0]
        a_term  = df["A_term"].iloc[0]
        b1_term = df["B1_term"].iloc[0]
        b2_term = df["B2_term"].iloc[0]

        # Raw scores → 0-1
        score = (df["Score"] / 100).values
        score = np.where(np.isnan(score), 0.5, score)

        abstracts_num = (df["H1"] + df["H2"] + df["Both"]).values
        rel_abstracts = abstracts_num.mean()
        iter_number   = df["Iteration"].nunique()

        print(f"    iterations={iter_number}  rel_abstracts={rel_abstracts:.1f}")
        print(f"    scores: {score}")

        # Handle degenerate case where all scores are identical
        if len(np.unique(score)) == 1:
            score = score.copy()
            score[0] += 0.01
            score[1]  = max(score[1] - 0.01, 1e-22)

        # ── Likelihood 1: MLE beta ────────────────────────────────────────────
        alpha_mle, beta_mle = ebeta_mle(score)
        alpha_mle = max(alpha_mle, 1e-22)
        beta_mle  = max(beta_mle,  1e-22)
        likelihood_mle = sample_beta(alpha_mle, beta_mle)

        # ── Likelihood 2: Method-of-moments beta ──────────────────────────────
        mu      = score.mean()
        var_s   = score.var(ddof=1)  # R uses var() which is unbiased (ddof=1)
        alpha_mom, beta_mom = ebeta_mom(mu, var_s)
        likelihood_mom = sample_beta(alpha_mom, beta_mom)

        # ── Prior from abstract counts ─────────────────────────────────────────
        alpha2, beta2 = build_prior_params(df, rel_abstracts)
        prior_samples = sample_beta(alpha2, beta2)

        # ── Posteriors (conjugate Beta update) ────────────────────────────────
        # Posterior 1: MLE likelihood + prior
        a1_post = alpha_mle + alpha2
        b1_post = beta_mle  + beta2

        # Posterior 2: MOM likelihood + prior  (used for violin / summary)
        a2_post = alpha_mom + alpha2
        b2_post = beta_mom  + beta2

        posterior1 = sample_beta(a1_post, b1_post)
        posterior2 = sample_beta(a2_post, b2_post)

        # ── ETI on posterior 2 ────────────────────────────────────────────────
        ci_low, ci_high = eti(posterior2)

        # ── Four-panel diagnostic PDF ─────────────────────────────────────────
        plot_beta_panels(
            likelihood_mle, likelihood_mom, prior_samples,
            posterior2, ci_low, ci_high,
            str(output), a_term
        )

        # ── Accumulate results ────────────────────────────────────────────────
        for val in posterior2:
            all_posts.append({
                "posterior": val,
                "A_term":    a_term,
                "B1_term":   b1_term,
                "B2_term":   b2_term,
            })

    if not all_posts:
        print("No results collected — check that subdirectories contain results.tsv files.")
        sys.exit(1)

    df1 = pd.DataFrame(all_posts)

    # Convert posterior to 0–100 scale (mirrors R's df1$posterior <- ... * 100)
    df1["posterior"] *= 100

    # ── Write raw posterior data ───────────────────────────────────────────────
    posterior_out = os.path.join(output, "posterior_data.txt")
    df1.to_csv(posterior_out, sep="\t", index=False, quoting=False)
    print(f"Posterior data saved → {posterior_out}")

    # ── Summary stats table ────────────────────────────────────────────────────
    def summarise(group):
        low, high = eti(group["posterior"].values)
        return pd.Series({
            "mean":    group["posterior"].mean(),
            "CI_low":  low,
            "CI_high": high,
        })

    summary = (
        df1.groupby(["A_term", "B1_term", "B2_term"])
           .apply(summarise)
           .reset_index()
    )
    summary_out = os.path.join(output, "summary_stats.txt")
    summary.to_csv(summary_out, sep="\t", index=False, quoting=False)
    print(f"Summary stats saved → {summary_out}")

    # ── Violin plot ────────────────────────────────────────────────────────────
    plot_violin(df1, output)

    # parameters
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()