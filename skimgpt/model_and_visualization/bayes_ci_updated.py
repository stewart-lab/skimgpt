# an LLM is asked to score two mutually exclusive hypotheses to explain some 
# phenomenon, using literature evidence retrieved from pubmed. the LLM 
# produces a score between 0 and 100, with 0 favoring one hypothesis and 100 
# favoring the other hypothesis. a score of 50 means that the LLM thinks the 
# two hypotheses are equally likely based on the provided texts.

# the goal of this system is not to say if hyp1 or hyp2 are true. the goal
# is to estimate when the literature came to a consensus regarding hyp1
# or hyp2.

# the LLM score is susceptible to drawing strong conclusions from few 
# abstracts. so, we introduce a Bayesian prior here to shrink the posterior
# score towards 50. if there is little evidence (i.e., few abstracts), then
# the prior dominates. if there is a lot of evidence, then the prior doesn't
# influence the score much.

# each LLM call gets up to 50 abstracts. the pool to draw abstracts from can 
# be very large (thousands of abstracts). if the literature is sparse, the 
# same abstracts will be sent to the LLM repeatedly, so only unique PMIDs are 
# counted as evidence.

# implementation: 
# - n_effective is the effective number of abstracts observed
# - we view the LLM score as a noisy approximation of the 'true LLM score'.
#   we model sources of noise (variance between LLM calls and 
#   literature sampling) that we add together.
# - we weight an LLM observation by the n_effective number of abstracts.
#   this is the 'learning rate'.
# - abstracts are not treated as independent evidence. papers in a field
#   cite each other and share priors.

# notes:
# - the HDI (error bar) represents the pipeline's (i.e., our) ability to 
#   estimate how confident the literature was. it does not represent how 
#   confident the literature was. 
# - a high error bar is generally caused by low numbers of abstracts. we don't
#   have enough information to get a solid read on what the scientific opinion
#   was at that point in time.
# - how confident the literature was at a given point in time is represented 
#   by the LLM score.
# - sampling a small portion of the available literature is not penalized
#   (i.e., sampling 100 out of 10000 abstracts is treated the same as sampling
#   100 out of 100 abstracts).
# - "how confident the literature is" is a vague quantity without a 
#   statistical definition. but, imagine we have a field of 100 experts, each
#   publishing once per year, and they support hyp1 and hyp2 in equal measure
#   (i.e., 50 abstracts each). we would say the field is probably evenly 
#   divided (minimum "literature confidence"). because we have 100 abstracts
#   to sample from, we can say that our ability to measure confidence is 
#   pretty high - we are confident in our ability to measure literature
#   confidence.

import os
import json
import shutil
from pathlib import Path

import cmdlogtime
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize_scalar

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "bayes_ci_updated_commandline.txt")

DPI = 96

# assumptions. need to calibrate w/ benchmark data.
A0 = 1.0               # strength of the prior. pseudo-abstracts per side
RHO = 0             # intra-field correlation
SIGMA2_CALL = 0.0007823878    # variance of a single LLM call, 0-1 scale, calculated by rerunning the same abstracts through 50 iterations

def n_effective(m, n_calls, rho=RHO, sigma2_call=SIGMA2_CALL, theta=0.5):
    if m == 0 or n_calls == 0:
        return 0.0
    n_lit = m / (1.0 + (m - 1) * rho)
    n_call = (theta * (1 - theta) * n_calls) / sigma2_call

    return 1.0 / (1.0 / n_lit + 1.0 / n_call)

def posterior_params(calls, a0=A0):
    scores = np.array([s / 100.0 for s, _ in calls], dtype=float)
    weights = np.array([len(p) for _, p in calls], dtype=float)

    if weights.sum() == 0:
        s_bar = 0.5
    else:
        s_bar = float(np.average(scores, weights=weights))
    s_bar = float(np.clip(s_bar, 1e-6, 1 - 1e-6))

    pmids = set()
    for call in calls:
        call_pmids = [t[0] for t in call[1]]
        pmids.update(call_pmids)

    m = len(pmids)
    n_eff = n_effective(m, len(calls), theta=s_bar)

    a = a0 + n_eff * s_bar
    b = a0 + n_eff * (1.0 - s_bar)

    return a, b

def hdi_beta(a, b, level=0.95):
    if abs(a - b) < 1e-9:
        tail = (1 - level) / 2
        return beta_dist.ppf(tail, a, b), beta_dist.ppf(1 - tail, a, b)

    def width(lo_p):
        return beta_dist.ppf(lo_p + level, a, b) - beta_dist.ppf(lo_p, a, b)

    res = minimize_scalar(width, bounds=(1e-9, 1 - level - 1e-9), method="bounded")
    return beta_dist.ppf(res.x, a, b), beta_dist.ppf(res.x + level, a, b)

def posterior_beta(calls, n_theta=300):
    a, b, = posterior_params(calls)
    grid = np.linspace(0.005, 0.995, n_theta)
    p = beta_dist.pdf(grid, a, b)
    return grid, p / p.sum(), a, b

def timecourse_data(data, level=0.95, hyp1_label="hyp1", hyp2_label="hyp2"):
    """
    Per-year summary statistics underlying the timecourse plot: the mean raw
    LLM score, the shrunk posterior score (mode), and the HDI bounds at
    `level`, all on the 0-100 scale.
    """
    rows = []

    for year in sorted(data):
        calls = data[year]
        grid, p, a, b = posterior_beta(calls)

        lo, hi = [100 * v for v in hdi_beta(a, b, level)]
        posterior_mode = grid[p.argmax()] * 100

        scores = [s for s, _ in calls]
        llm = float(np.mean(scores))

        rows.append({
            "year": year,
            "hyp1": hyp1_label,
            "hyp2": hyp2_label,
            "mean_llm_score": llm,
            "posterior_score": posterior_mode,
            "hdi_level": level,
            "hdi_lo": lo,
            "hdi_hi": hi,
        })

    return rows


def write_timecourse_csv(data, path, level=0.95, hyp1_label="hyp1", hyp2_label="hyp2"):
    import csv

    rows = timecourse_data(data, level=level, hyp1_label=hyp1_label, hyp2_label=hyp2_label)
    fieldnames = ["year", "hyp1", "hyp2", "mean_llm_score", "posterior_score",
                  "hdi_level", "hdi_lo", "hdi_hi"]

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
    "accepted" or "rejected"), and optionally when it was reconsidered
    (dark grey).
    """
    if decision_date is not None and decision_label not in ("accepted", "rejected","unknown"):
        raise ValueError('decision_label must be "accepted" or "rejected"')

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

    tie_break_order = ["supports_H1", "supports_H2", "both"]
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

        ax.bar(years, y_vals, bottom=bottom, color=label_colors[label],
               edgecolor="white", linewidth=0.5, label=label_display[label])
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
    
def get_data(data_dir: str):
    data = dict()

    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory does not exist: {data_dir}")

    # get iteration result .json
    json_files = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith("gpt_direct_comp.json") and ".backup" not in dirpath:
                json_files.append(os.path.join(dirpath, f))

    for iteration_json in json_files:
        # get config.json for this iteration
        config_found = False
        config_dir = os.path.dirname(iteration_json)
        while not config_found:
            configs = [x for x in os.listdir(config_dir) if x == "config.json"]
            if configs:
                config_found = True
                config_json = os.path.join(config_dir, configs[0])
                break
            config_dir = os.path.dirname(config_dir)

        # read config.json to get the censor year
        with open(config_json, 'r') as f:
            config_json_content = json.load(f)
        
        try:
            year = config_json_content["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["censor_year_upper"]
        except:
            year = config_json_content["JOB_SPECIFIC_SETTINGS"]["km_with_gpt"]["km_with_gpt"]["censor_year_upper"]

        # read the iteration result .json and add the data to the dictionary
        if year not in data:
            data[year] = []

        with open(iteration_json, 'r') as f:
            iter_json_content = json.load(f)

        hyp_eval = iter_json_content[0]["Hypothesis_Comparison"]["Result"]

        if not hyp_eval:
            print(f"no result for {iteration_json}")
            continue

        hyp_eval = hyp_eval[0]
        pmids = [(a["pmid"], a["label"]) for a in hyp_eval["per_abstract"] if a["label"] in {"supports_H1", "supports_H2", "both"}]
        llm_score = hyp_eval["score"]

        data[year].append((llm_score, pmids))

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

    print(f"Plots and data saved to {output_dir}")
    cmdlogtime.end(addl_logfile, start_time_secs)


if __name__ == "__main__":
    main()