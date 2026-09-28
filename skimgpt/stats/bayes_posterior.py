"""Closed-form Beta posterior for DCH hypothesis scores (Beth's updated model).

The single source of truth for the model introduced on the ``update_model``
branch (``bayes_ci_updated.py`` / ``bayes_ci_updated.R``). It supersedes the
MoM/MLE Beta-Binomial fit in :mod:`skimgpt.stats.bayes_ci`.

An LLM scores two mutually exclusive hypotheses from 0 (favours H2) to 100
(favours H1), using abstracts retrieved from PubMed. The goal is not to say
which hypothesis is true but to estimate *when* the literature reached a
consensus. The LLM will draw strong conclusions from few abstracts, so a
Beta prior shrinks the score towards 50: with little evidence the prior
dominates, with a lot it barely matters.

Model:

* Each LLM call sees up to 50 abstracts, drawn from a pool that may be far
  larger. When the literature is sparse the same abstracts recur across
  calls, so only **unique PMIDs** count as evidence (``m``).
* The LLM score is a noisy estimate of the "true" LLM score. Literature
  sampling noise and between-call noise add, giving an effective abstract
  count :func:`n_effective` that acts as the learning rate.
* Abstracts are not independent - papers cite each other and share priors -
  which ``rho`` (intra-field correlation) discounts.
* The posterior is ``Beta(a0 + n_eff * s_bar, a0 + n_eff * (1 - s_bar))``
  where ``s_bar`` is the call scores averaged with weights equal to each
  call's labelled-abstract count.

Reading the output: the HDI is *our* ability to estimate how confident the
literature was, not how confident the literature was - that is the score. A
wide HDI mostly means few abstracts. Sampling a small fraction of a large
pool is not penalised.

Callers - the ``skimgpt.visualization`` CLIs and SKiM_web's result pages -
reduce their own data to a list of calls and call :func:`posterior_summary`.

A *call* is ``(llm_score, [(pmid, label), ...])``: the score on the **0-100
scale**, and the evidence abstracts (labels in :data:`EVIDENCE_LABELS`; use
:func:`call_from_result` to build one from a ``Hypothesis_Comparison``
result dict). Summaries are reported on the 0-100 scale.

numpy + scipy only - see the package docstring.
"""

from typing import Iterable, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist

# Assumptions - to be calibrated against benchmark data.

# Strength of the prior, in pseudo-abstracts per side.
A0 = 1.2

# Intra-field correlation between abstracts.
RHO = 0

# Variance of repeated LLM calls on the same 50 abstracts, divided by their
# proportional variance theta * (1 - theta) on the 0-1 scale (theta = mean
# call score). Fitted as a line across multiple runs; effectively
# sigma2_call / sigma2_theta_hat.
SLOPE = 0.00761

# Labels that count as evidence. "neither" and "inconclusive" abstracts
# contribute neither weight nor unique PMIDs.
EVIDENCE_LABELS = ("supports_H1", "supports_H2", "both")

# Grid the posterior mode is read from. Kept (rather than the analytic mode)
# so published figures and CSVs reproduce exactly.
MODE_GRID = np.linspace(0.005, 0.995, 300)

Call = Tuple[float, Sequence[Tuple[str, str]]]


class PosteriorSummary(NamedTuple):
    """One window/year/hypothesis-pair. Score-like fields are on 0-100."""

    a: float
    b: float
    posterior_mean: float
    posterior_mode: float
    hdi_level: float
    hdi_lo: float
    hdi_hi: float
    mean_llm_score: float
    n_calls: int
    n_unique_pmids: int
    n_eff: float


def call_from_result(result: Mapping) -> Call:
    """Build a call from one ``Hypothesis_Comparison`` result dict.

    Accepts anything with ``score`` and ``per_abstract`` (each entry having
    ``pmid`` and ``label``) - the ``gpt_direct_comp.json`` ``Result[0]`` block
    and SKiM_web's stored ``ab_result`` alike. Non-evidence labels are dropped.
    """
    pmids = [
        (str(a["pmid"]), a["label"])
        for a in result.get("per_abstract") or []
        if a.get("label") in EVIDENCE_LABELS
    ]
    return result["score"], pmids


def unique_pmids(calls: Iterable[Call]) -> set:
    return {pmid for _, pmids in calls for pmid, _ in pmids}


def n_effective(m, n_calls, rho=RHO, slope=SLOPE):
    """Effective abstract count from ``m`` unique PMIDs over ``n_calls`` calls."""
    if m == 0 or n_calls == 0:
        return 0.0
    n_lit = m / (1.0 + (m - 1) * rho)
    sigma_n_call = slope / n_calls
    return 1.0 / (1.0 / n_lit + sigma_n_call)


def posterior_params(calls: Sequence[Call], a0=A0, rho=RHO, slope=SLOPE):
    """Posterior Beta shapes ``(a, b)``: prior ``a0`` plus the likelihood."""
    scores = np.array([s / 100.0 for s, _ in calls], dtype=float)
    weights = np.array([len(p) for _, p in calls], dtype=float)

    if weights.sum() == 0:
        s_bar = 0.5
    else:
        s_bar = float(np.average(scores, weights=weights))
    s_bar = float(np.clip(s_bar, 1e-6, 1 - 1e-6))

    n_eff = n_effective(len(unique_pmids(calls)), len(calls), rho=rho, slope=slope)

    a = a0 + n_eff * s_bar
    b = a0 + n_eff * (1.0 - s_bar)
    return a, b


def hdi_beta(a, b, level=0.95):
    """Exact Highest Density Interval of Beta(a, b), on the 0-1 scale."""
    if abs(a - b) < 1e-9:
        tail = (1 - level) / 2
        return beta_dist.ppf(tail, a, b), beta_dist.ppf(1 - tail, a, b)

    def width(lo_p):
        return beta_dist.ppf(lo_p + level, a, b) - beta_dist.ppf(lo_p, a, b)

    res = minimize_scalar(width, bounds=(1e-9, 1 - level - 1e-9), method="bounded")
    return beta_dist.ppf(res.x, a, b), beta_dist.ppf(res.x + level, a, b)


def posterior_mode(a, b):
    """Posterior mode on the 0-1 scale, read off :data:`MODE_GRID`."""
    p = beta_dist.pdf(MODE_GRID, a, b)
    return float(MODE_GRID[p.argmax()])


def posterior_summary(calls: Sequence[Call], level=0.95, a0=A0, rho=RHO,
                      slope=SLOPE) -> PosteriorSummary:
    """Fit the posterior for one set of calls and interval it.

    Raises ``ValueError`` on an empty ``calls`` - there is nothing to average.
    """
    calls = list(calls)
    if not calls:
        raise ValueError("posterior_summary needs at least one call")

    a, b = posterior_params(calls, a0=a0, rho=rho, slope=slope)
    lo, hi = hdi_beta(a, b, level)
    m = len(unique_pmids(calls))

    return PosteriorSummary(
        a=float(a),
        b=float(b),
        posterior_mean=100 * a / (a + b),
        posterior_mode=100 * posterior_mode(a, b),
        hdi_level=level,
        hdi_lo=100 * float(lo),
        hdi_hi=100 * float(hi),
        mean_llm_score=float(np.mean([s for s, _ in calls])),
        n_calls=len(calls),
        n_unique_pmids=m,
        n_eff=float(n_effective(m, len(calls), rho=rho, slope=slope)),
    )


def label_counts(calls: Sequence[Call]) -> List[Tuple[int, int, int]]:
    """Per-call ``(supports_H1, supports_H2, both)`` counts."""
    out = []
    for _, pmids in calls:
        labels = [label for _, label in pmids]
        out.append(tuple(labels.count(lbl) for lbl in EVIDENCE_LABELS))
    return out
