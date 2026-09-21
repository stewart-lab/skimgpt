"""Beta-Binomial credible interval for DCH hypothesis scores.

The single source of truth for the Beta-Binomial conjugate model (Beth's
method, mirroring ``bayes_citest.R``): a Beta likelihood fitted to the
per-iteration scores, a prior built from the support-label counts, and
credible intervals on the resulting posterior.

Two callers share this module:

* ``skimgpt.visualization.bayesian_ci`` - the ribbon-plot CLI, one row per
  iteration in a pandas DataFrame, needs the MLE shapes for its CI table.
* SKiM_web's ``application/bayes_ci.py`` - the web result pages, which hold
  summed label tallies and need only the MoM posterior and its ETI.

Both reduce their own data shape to per-iteration averages and call
:func:`posterior_ci`. Aggregation stays with the caller; the Beta maths lives
here.

Scores are on the **0-1 scale** throughout. Callers working on the 0-100
scale divide on the way in and multiply on the way out.

numpy + scipy only - see the package docstring.
"""

import logging
from typing import NamedTuple, Optional, Sequence

import numpy as np
from scipy import optimize
from scipy.stats import beta as beta_dist

logger = logging.getLogger(__name__)

# Smallest Beta shape parameter we will hand to scipy. Shapes are clamped to
# this rather than allowed to go non-positive, which scipy rejects outright.
MIN_SHAPE = 1e-22

# Floor for the prior shapes. Larger than MIN_SHAPE so that a window with no
# supporting abstracts at all still yields a (very weak) proper prior.
MIN_PRIOR_SHAPE = 0.01

# Below this many relevant abstracts the support counts are used as Beta
# pseudo-counts directly; above it they are renormalised to proportions so a
# large corpus does not swamp the likelihood.
PRIOR_COUNT_THRESHOLD = 50

# Variance substituted when the observed scores have none (a single iteration,
# or several identical ones that survived perturbation).
FALLBACK_VARIANCE = 1e-6

# Upper bound for a perturbed score. It cannot mirror MIN_SHAPE: 1 - 1e-22 is
# not representable in float64 and collapses back to exactly 1.0, where the
# Beta density is zero. Near zero the same magnitude is representable, so the
# lower bound stays at MIN_SHAPE and reproduces the historical numbers.
MAX_SCORE = 1.0 - 1e-10


class PosteriorCI(NamedTuple):
    """Everything both callers need from one window/year.

    All score-like fields are on the 0-1 scale.

    ``alpha_lik_mle``/``beta_lik_mle``/``alpha_post_mle``/``beta_post_mle`` are
    populated only when ``include_mle=True``; they are ``None`` otherwise, and
    also when the MLE solver failed to converge (``mle_ok`` records which).
    """

    mean_score: float
    var_score: float
    alpha_lik: float
    beta_lik: float
    alpha_prior: float
    beta_prior: float
    alpha_post: float
    beta_post: float
    post_mean: float
    eti_low: float
    eti_high: float
    hdi_low: float
    hdi_high: float
    alpha_lik_mle: Optional[float] = None
    beta_lik_mle: Optional[float] = None
    alpha_post_mle: Optional[float] = None
    beta_post_mle: Optional[float] = None
    mle_ok: Optional[bool] = None


def estimate_beta_mom(mu, var):
    """Method-of-moments estimator for Beta(alpha, beta)."""
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    if alpha <= 0 or beta <= 0:
        logger.warning(
            "MoM produced non-positive params (mu=%.4f, var=%.4f) - clamping", mu, var
        )
    return max(alpha, MIN_SHAPE), max(beta, MIN_SHAPE)


def estimate_beta_mle(data):
    """MLE estimator using ``scipy.stats.beta.fit`` (location=0, scale=1).

    Returns ``(alpha, beta, ok)``. When the solver fails to converge - which it
    does for degenerate input such as a single iteration, or scores that are
    all identical - ``ok`` is ``False`` and the shapes are ``(None, None)``.
    Callers decide what to do with that; previously the exception escaped and
    killed the whole run.
    """
    data = np.clip(np.asarray(data, dtype=float), 1e-10, 1 - 1e-10)
    try:
        a, b, _loc, _scale = beta_dist.fit(data, floc=0, fscale=1)
    except Exception as exc:  # scipy raises FitSolverError, among others
        logger.warning("Beta MLE did not converge (%s); falling back to MoM", exc)
        return None, None, False
    return max(a, MIN_SHAPE), max(b, MIN_SHAPE), True


def hdi(a, b, credible_mass=0.95):
    """Highest Density Interval for Beta(a, b).

    Finds the narrowest interval containing *credible_mass* of the
    probability. Uses numerical optimisation on the inverse-CDF.
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


def eti(a, b, credible_mass=0.95):
    """Equal-Tailed Interval for Beta(a, b)."""
    tail = (1 - credible_mass) / 2
    return beta_dist.ppf(tail, a, b), beta_dist.ppf(1 - tail, a, b)


def prepare_scores(scores):
    """Coerce scores to a clean 0-1 array with a fittable spread.

    NaNs become 0.5 (no evidence either way). A set of wholly identical scores
    has no variance for the likelihood to fit, so the first two are nudged
    apart - staying strictly inside (0, 1), since a score of exactly 1.0
    otherwise perturbs to 1.01, which made ``scipy.stats.beta.fit`` fail to
    converge and raise, killing the whole run.
    """
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.5)

    if len(np.unique(s)) == 1:
        s = s.copy()
        s[0] = min(MAX_SCORE, s[0] + 0.01)
        if len(s) > 1:
            s[1] = max(MIN_SHAPE, s[1] - 0.01)
    return s


def prior_from_support(support_h1, support_h2, both, rel_abstracts):
    """Build the prior Beta shapes from per-iteration support-label averages.

    Abstracts supporting *both* hypotheses are split evenly between the two
    shapes. Below ``PRIOR_COUNT_THRESHOLD`` relevant abstracts the averages act
    as pseudo-counts directly; above it they are renormalised to proportions
    and rescaled, so the prior's strength tracks the corpus size rather than
    exploding with it.
    """
    alpha = float(support_h1)
    beta = float(support_h2)
    both = float(both)

    if rel_abstracts <= PRIOR_COUNT_THRESHOLD:
        if both != 0:
            alpha += both / 2
            beta += both / 2
    else:
        if both != 0:
            alpha += both / 2
            beta += both / 2
        total = alpha + beta
        if total == 0:
            total = 1
        alpha, beta = (alpha / total) * rel_abstracts, (beta / total) * rel_abstracts

    return max(alpha, MIN_PRIOR_SHAPE), max(beta, MIN_PRIOR_SHAPE)


def posterior_ci(
    scores: Sequence[float],
    support_h1: float,
    support_h2: float,
    both: float,
    rel_abstracts: float,
    credible_mass: float = 0.95,
    include_mle: bool = False,
) -> PosteriorCI:
    """Fit the Beta-Binomial posterior for one window and interval it.

    Args:
        scores: Per-iteration hypothesis scores on the 0-1 scale.
        support_h1: Mean per-iteration count of abstracts supporting H1.
        support_h2: Mean per-iteration count of abstracts supporting H2.
        both: Mean per-iteration count of abstracts supporting both.
        rel_abstracts: Mean per-iteration relevant-abstract count. Selects the
            prior branch, so callers must agree on what it counts.
        credible_mass: Interval mass. Defaults to the usual 0.95.
        include_mle: Also fit the likelihood by MLE and report the MLE-based
            posterior shapes. The CLI's CI table wants these; the web does not.

    Returns:
        A :class:`PosteriorCI`. The interval and ``post_mean`` always come from
        the **MoM** likelihood - the MLE shapes are reported, never intervalled.
    """
    s = prepare_scores(scores)

    mean_score = float(np.mean(s))
    var_score = float(np.var(s, ddof=1)) if len(s) > 1 else FALLBACK_VARIANCE
    if var_score == 0 or np.isnan(var_score):
        var_score = FALLBACK_VARIANCE

    alpha_lik, beta_lik = estimate_beta_mom(mean_score, var_score)
    alpha_prior, beta_prior = prior_from_support(
        support_h1, support_h2, both, rel_abstracts
    )

    alpha_post = alpha_lik + alpha_prior
    beta_post = beta_lik + beta_prior

    eti_low, eti_high = eti(alpha_post, beta_post, credible_mass)
    hdi_low, hdi_high = hdi(alpha_post, beta_post, credible_mass)
    post_mean = alpha_post / (alpha_post + beta_post)

    alpha_mle = beta_mle = alpha_post_mle = beta_post_mle = None
    mle_ok = None
    if include_mle:
        alpha_mle, beta_mle, mle_ok = estimate_beta_mle(s)
        if mle_ok:
            alpha_post_mle = alpha_mle + alpha_prior
            beta_post_mle = beta_mle + beta_prior

    return PosteriorCI(
        mean_score=mean_score,
        var_score=var_score,
        alpha_lik=alpha_lik,
        beta_lik=beta_lik,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        alpha_post=alpha_post,
        beta_post=beta_post,
        post_mean=float(post_mean),
        eti_low=float(eti_low),
        eti_high=float(eti_high),
        hdi_low=float(hdi_low),
        hdi_high=float(hdi_high),
        alpha_lik_mle=alpha_mle,
        beta_lik_mle=beta_mle,
        alpha_post_mle=alpha_post_mle,
        beta_post_mle=beta_post_mle,
        mle_ok=mle_ok,
    )
