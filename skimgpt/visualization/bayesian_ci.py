"""Bayesian credible interval analysis for hypothesis comparison.

Fits Beta distributions (likelihood, prior, posterior) per censor year,
computes HDI and ETI credible intervals, and produces ribbon + bar plots.
Mirrors the R script ``bayes_citest.R``.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import cmdlogtime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import beta as beta_dist

logger = logging.getLogger(__name__)

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "bayesian_ci_commandline.txt")


# ---------------------------------------------------------------------------
# Beta-distribution helpers
# ---------------------------------------------------------------------------

def _estimate_beta_mom(mu, var):
    """Method-of-moments estimator for Beta(alpha, beta)."""
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    if alpha <= 0 or beta <= 0:
        logger.warning("MoM produced non-positive params (mu=%.4f, var=%.4f) — clamping", mu, var)
    return max(alpha, 1e-22), max(beta, 1e-22)


def _estimate_beta_mle(data):
    """MLE estimator using scipy.stats.beta.fit (location=0, scale=1)."""
    # Clamp data to open interval (0, 1)
    data = np.clip(data, 1e-10, 1 - 1e-10)
    a, b, loc, scale = beta_dist.fit(data, floc=0, fscale=1)
    return max(a, 1e-22), max(b, 1e-22)


def _hdi(a, b, credible_mass=0.95):
    """Highest Density Interval for Beta(a, b).

    Finds the narrowest interval containing *credible_mass* of the
    probability.  Uses numerical optimisation on the inverse-CDF.
    """
    # For unimodal Beta (a>1, b>1) the HDI is the shortest credible interval.
    # For other shapes, fall back to the ETI.
    def _interval_width(low_tail):
        low = beta_dist.ppf(low_tail, a, b)
        high = beta_dist.ppf(low_tail + credible_mass, a, b)
        return high - low

    try:
        result = optimize.minimize_scalar(
            _interval_width,
            bounds=(0, 1 - credible_mass),
            method="bounded",
        )
        low_tail = result.x
    except Exception:
        low_tail = (1 - credible_mass) / 2  # fall back to ETI

    low = beta_dist.ppf(low_tail, a, b)
    high = beta_dist.ppf(low_tail + credible_mass, a, b)
    return low, high


def _eti(a, b, credible_mass=0.95):
    """Equal-Tailed Interval for Beta(a, b)."""
    tail = (1 - credible_mass) / 2
    return beta_dist.ppf(tail, a, b), beta_dist.ppf(1 - tail, a, b)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_beta_density(ax, a, b, color, xlabel):
    """Plot a Beta(a,b) density on *ax*."""
    x = np.linspace(0, 1, 500)
    y = beta_dist.pdf(x, a, b)
    ax.fill_between(x, y, color=color, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)


def _add_event_lines(ax, d_date, a_date, x_date, labels2, movex):
    """Add event vertical lines with annotation positioning from movex."""
    if d_date is not None:
        ax.axvline(x=d_date, linestyle="--", color="brown", linewidth=0.8)
        ax.annotate(
            labels2[0], xy=(d_date, 25), rotation=90,
            fontsize=7, color="brown",
            ha="left" if movex[0] < 0 else "right",
            va="bottom" if movex[1] > 0 else "top",
        )
    if a_date is not None:
        ax.axvline(x=a_date, linestyle="--", color="black", linewidth=0.8)
        lbl = labels2[1] if len(labels2) > 1 else "acceptance"
        ax.annotate(
            lbl, xy=(a_date, 25), rotation=90,
            fontsize=7, color="black",
            ha="left" if movex[2] < 0 else "right",
            va="bottom" if movex[3] > 0 else "top",
        )
    if x_date is not None:
        ax.axvline(x=float(x_date), linestyle="--", color="grey", linewidth=0.8)
        if len(labels2) > 2:
            ax.annotate(
                labels2[2], xy=(float(x_date), 25), rotation=90,
                fontsize=7, color="black",
                ha="left" if movex[2] < 0 else "right",
                va="bottom" if movex[3] > 0 else "top",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(
        COMMAND_LINE_DEF_FILE
    )

    proj_path = my_args["projpath"]
    filename = my_args["filename"]
    d_date = int(my_args["discover"]) if my_args.get("discover") else None
    a_date = int(my_args["accept"]) if my_args.get("accept") else None
    x_date = my_args.get("x_date") or None
    title_str = my_args.get("title") or None
    x_interval = int(my_args.get("xinterval", 1))

    labels2 = [s.strip() for s in my_args.get("labels", "discover,acceptance").split(",")]
    movex = [float(v) for v in my_args.get("move", "-0.1,1,-0.05,1").split(",")]
    movex = (movex + [-0.1, 1, -0.05, 1])[:4]  # pad to 4 elements

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(proj_path, f"output_visualization_{timestamp}")
    os.makedirs(output, exist_ok=True)

    # Read input
    df = pd.read_csv(os.path.join(proj_path, filename))
    A_term = df["A_term"].unique()[0]
    B1_term = df["B1_term"].unique()[0]
    B2_term = df["B2_term"].unique()[0]

    years = sorted(df["censor_year"].unique())
    ci_rows = []

    for year in years:
        print(year)
        dfsub = df[df["censor_year"] == year].copy()
        score = (dfsub["score"] / 100).values
        score = np.nan_to_num(score, nan=0.5)

        abstracts_num = (dfsub["num_abstracts"] - dfsub["neither_or_inconclusive"]).mean()
        rel_abstracts = dfsub["num_abstracts"].mean()

        # Handle constant scores
        if len(np.unique(score)) == 1:
            score = score.copy()
            score[0] += 0.01
            if len(score) > 1:
                score[1] -= 0.01
                if score[1] < 0:
                    score[1] = 1e-22

        # Estimate likelihood parameters (MLE)
        a_mle, b_mle = _estimate_beta_mle(score)

        # Estimate likelihood parameters (method of moments)
        meanscore = np.mean(score)
        varscore = np.var(score, ddof=1) if len(score) > 1 else 1e-6
        if varscore == 0 or np.isnan(varscore):
            varscore = 1e-6
        alpha1, beta1 = _estimate_beta_mom(meanscore, varscore)

        # Estimate prior from supporting abstract counts
        if rel_abstracts <= 50:
            alpha2 = dfsub["support_H1"].mean()
            beta2 = dfsub["support_H2"].mean()
            if dfsub["both"].sum() != 0:
                both = dfsub["both"].mean() / 2
                alpha2 += both
                beta2 += both
        else:
            nhyp1 = dfsub["support_H1"].mean()
            nhyp2 = dfsub["support_H2"].mean()
            if dfsub["both"].sum() != 0:
                both = dfsub["both"].mean() / 2
                nhyp1 += both
                nhyp2 += both
            ntot = nhyp1 + nhyp2
            if ntot == 0:
                ntot = 1
            prop1 = nhyp1 / ntot
            prop2 = nhyp2 / ntot
            alpha2 = prop1 * rel_abstracts
            beta2 = prop2 * rel_abstracts

        alpha2 = max(alpha2, 0.01)
        beta2 = max(beta2, 0.01)

        # Posterior parameters (MLE likelihood + prior)
        a_post = a_mle + alpha2
        b_post = b_mle + beta2
        # Posterior parameters (MoM likelihood + prior)
        a2_post = alpha1 + alpha2
        b2_post = beta1 + beta2

        # Compute HDI and ETI on posterior2 (MoM-based)
        hdi_low, hdi_high = _hdi(a2_post, b2_post, 0.95)
        eti_low, eti_high = _eti(a2_post, b2_post, 0.95)

        posterior2_mean = a2_post / (a2_post + b2_post)

        ci_row = {
            "CI_hdi": 0.95,
            "CI_low_hdi": hdi_low,
            "CI_high_hdi": hdi_high,
            "CI_eti": 0.95,
            "CI_low_eti": eti_low,
            "CI_high_eti": eti_high,
            "Year": year,
            "Mean.score": meanscore,
            "Shape1": a_post,
            "Shape2": b_post,
            "Mean.posterior": posterior2_mean,
            "num_abstracts": abstracts_num,
            "support_H1": dfsub["support_H1"].mean(),
            "support_H2": dfsub["support_H2"].mean(),
            "both": dfsub["both"].mean(),
        }
        ci_rows.append(ci_row)

        # Per-year beta distribution diagnostic plot
        fig_diag, axes = plt.subplots(2, 2, figsize=(8, 6))
        _plot_beta_density(axes[0, 0], a_mle, b_mle, "orchid", "likelihood MLE")
        _plot_beta_density(axes[0, 1], alpha1, beta1, "gold", "likelihood MoM")
        _plot_beta_density(axes[1, 0], alpha2, beta2, "red", "prior")
        _plot_beta_density(axes[1, 1], a2_post, b2_post, "orange", "posterior")
        # Add ETI lines to posterior
        axes[1, 1].axvline(x=eti_low, color="violet", linewidth=1.5)
        axes[1, 1].axvline(x=eti_high, color="violet", linewidth=1.5)
        fig_diag.suptitle(f"Year {year}", fontsize=10)
        fig_diag.tight_layout(rect=[0, 0, 1, 0.95])
        fig_diag.savefig(
            os.path.join(output, f"{year}_post_beta_distributions_withCI_betaparams-newman.pdf")
        )
        plt.close(fig_diag)

    # Build results dataframe
    df1 = pd.DataFrame(ci_rows)

    # Write CI data
    df1.to_csv(
        os.path.join(output, f"{A_term}_{B1_term}vs.{B2_term}_CIdata_newman.txt"),
        sep="\t", index=False,
    )

    # Prepare ribbon-plot data
    df2 = df1[["Year", "CI_low_eti", "CI_high_eti", "Mean.score",
               "Mean.posterior", "num_abstracts", "support_H1",
               "support_H2", "both"]].copy()
    df2["CI_low_eti"] *= 100
    df2["CI_high_eti"] *= 100
    df2["Mean"] = df2["Mean.score"] * 100
    df2["post.Mean"] = df2["Mean.posterior"] * 100

    # x-axis tick breaks (matching R breakfunc)
    all_years = sorted(df2["Year"].unique())
    year_breaks = all_years[::x_interval]

    # ---- Ribbon plot 1: Mean score ----
    fig1, (ax_r1, ax_b1) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    ax_r1.fill_between(df2["Year"], df2["CI_low_eti"], df2["CI_high_eti"],
                        alpha=0.3, color="orchid")
    ax_r1.plot(df2["Year"], df2["Mean"], color="skyblue", linewidth=1)
    ax_r1.set_ylabel("Mean score")
    ax_r1.axhline(y=50, linestyle="--", color="darkgrey", linewidth=0.8)
    ax_r1.tick_params(axis="x", labelbottom=False)
    ax_r1.grid(False)
    ax_r1.spines["top"].set_visible(False)
    ax_r1.spines["right"].set_visible(False)
    if title_str:
        ax_r1.set_title(title_str, fontsize=10)
    _add_event_lines(ax_r1, d_date, a_date, x_date, labels2, movex)

    # Bar plot for support counts
    bar_width = 0.6
    ax_b1.bar(df2["Year"], df2["support_H1"], width=bar_width, alpha=0.8,
              color="#E69F00", label=B1_term)
    ax_b1.bar(df2["Year"], df2["support_H2"], width=bar_width, alpha=0.8,
              bottom=df2["support_H1"], color="#440154", label=B2_term)
    ax_b1.bar(df2["Year"], df2["both"], width=bar_width, alpha=0.8,
              bottom=df2["support_H1"] + df2["support_H2"], color="#009E73", label="both")
    ax_b1.set_ylabel("Number of abstracts")
    ax_b1.set_xlabel("Year")
    ax_b1.set_xticks(year_breaks)
    ax_b1.legend(title="Support type", fontsize=7, loc="upper left")
    ax_b1.grid(False)
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)
    if d_date is not None:
        ax_b1.axvline(x=d_date, linestyle="--", color="brown", linewidth=0.8)
    if a_date is not None:
        ax_b1.axvline(x=a_date, linestyle="--", color="black", linewidth=0.8)

    fig1.tight_layout()
    fig1.savefig(os.path.join(output, "KM-GPT_OT_ribbon_plot_score_newman.pdf"))
    plt.close(fig1)

    # ---- Ribbon plot 2: Posterior mean ----
    fig2, (ax_r2, ax_b2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    ax_r2.fill_between(df2["Year"], df2["CI_low_eti"], df2["CI_high_eti"],
                        alpha=0.3, color="orchid")
    ax_r2.plot(df2["Year"], df2["post.Mean"], color="blue", linewidth=1)
    ax_r2.set_ylabel("Mean posterior score")
    ax_r2.axhline(y=50, linestyle="--", color="darkgrey", linewidth=0.8)
    ax_r2.tick_params(axis="x", labelbottom=False)
    ax_r2.grid(False)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)
    if title_str:
        ax_r2.set_title(title_str, fontsize=10)
    _add_event_lines(ax_r2, d_date, a_date, x_date, labels2, movex)

    ax_b2.bar(df2["Year"], df2["support_H1"], width=bar_width, alpha=0.8,
              color="#E69F00", label=B1_term)
    ax_b2.bar(df2["Year"], df2["support_H2"], width=bar_width, alpha=0.8,
              bottom=df2["support_H1"], color="#440154", label=B2_term)
    ax_b2.bar(df2["Year"], df2["both"], width=bar_width, alpha=0.8,
              bottom=df2["support_H1"] + df2["support_H2"], color="#009E73", label="both")
    ax_b2.set_ylabel("Number of abstracts")
    ax_b2.set_xlabel("Year")
    ax_b2.set_xticks(year_breaks)
    ax_b2.legend(title="Support type", fontsize=7, loc="upper left")
    ax_b2.grid(False)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
    if d_date is not None:
        ax_b2.axvline(x=d_date, linestyle="--", color="brown", linewidth=0.8)
    if a_date is not None:
        ax_b2.axvline(x=a_date, linestyle="--", color="black", linewidth=0.8)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output, "KM-GPT_OT_ribbon_plot_postmean_newman.pdf"))
    plt.close(fig2)

    # Write parameters
    params = pd.DataFrame({
        "Parameter": ["filename", "ProjectPath", "title", "discover_date",
                       "acceptance_date", "term_A", "term_B1", "term_B2"],
        "Value": [filename, proj_path, title_str, d_date, a_date,
                  A_term, B1_term, B2_term],
    })
    params.to_csv(os.path.join(output, "parameters.csv"), index=False)

    print(f"Plots saved to {output}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()
