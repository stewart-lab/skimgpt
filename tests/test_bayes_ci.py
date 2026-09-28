"""Golden vectors for the legacy Beta-Binomial model in ``skimgpt.stats``.

``skimgpt/stats/bayes_ci.py`` is superseded by ``bayes_posterior.py`` (see
``test_bayes_posterior.py``) but stays pinned while SKiM_web builds still
import it.

The expectations in ``bayes_ci_gold.json`` were captured from the pre-refactor
implementation (the inline block in ``bayesian_ci.py::main()``), so a passing
run means the extraction changed no published number.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from skimgpt.stats.bayes_ci import posterior_ci, prepare_scores, prior_from_support

GOLD = json.loads((Path(__file__).parent / "bayes_ci_gold.json").read_text())

# gold key -> PosteriorCI field
FIELDS = {
    "Mean.score": "mean_score",
    "Mean.posterior": "post_mean",
    "CI_low_eti": "eti_low",
    "CI_high_eti": "eti_high",
    "CI_low_hdi": "hdi_low",
    "CI_high_hdi": "hdi_high",
    "alpha_mom": "alpha_lik",
    "beta_mom": "beta_lik",
    "alpha_prior": "alpha_prior",
    "beta_prior": "beta_prior",
    "Shape1": "alpha_post_mle",
    "Shape2": "beta_post_mle",
}


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def _fit(rows):
    """Aggregate per-iteration rows the way the CLI does, then fit."""
    return posterior_ci(
        [r["score"] / 100 for r in rows],
        support_h1=_mean(rows, "support_H1"),
        support_h2=_mean(rows, "support_H2"),
        both=_mean(rows, "both"),
        rel_abstracts=_mean(rows, "num_abstracts"),
        include_mle=True,
    )


@pytest.mark.parametrize(
    "name", sorted(k for k, v in GOLD.items() if "expected" in v)
)
def test_matches_pre_refactor_output(name):
    """Every field the CLI writes to its CI table is unchanged."""
    case = GOLD[name]
    fit = _fit(case["rows"])
    for gold_key, field in FIELDS.items():
        assert getattr(fit, field) == pytest.approx(
            case["expected"][gold_key], rel=1e-12, abs=1e-12
        ), f"{name}.{gold_key}"


@pytest.mark.parametrize(
    "name", sorted(k for k, v in GOLD.items() if "legacy_crash" in v)
)
def test_degenerate_windows_no_longer_crash(name):
    """Windows that used to raise FitSolverError now return a usable interval.

    A single iteration, or a whole window scored identically, made
    ``scipy.stats.beta.fit`` fail to converge and took the entire run with it.
    The MoM path was always fine, so the interval is computed and only the
    MLE-derived shapes are dropped.
    """
    fit = _fit(GOLD[name]["rows"])
    assert fit.mle_ok is False
    assert fit.alpha_post_mle is None and fit.beta_post_mle is None
    assert 0.0 <= fit.eti_low <= fit.eti_high <= 1.0
    assert 0.0 <= fit.post_mean <= 1.0


def test_perturbation_never_leaves_the_unit_interval():
    """An all-100 window must not perturb a score to 1.01 (it used to).

    Only the first two entries are nudged - that is the historical behaviour,
    and exact 0.0/1.0 elsewhere in the array are handled by the MLE's own
    clipping - so the invariant is that nothing lands *outside* [0, 1].
    """
    for scores in ([1.0, 1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0, 0.0]):
        out = prepare_scores(scores)
        assert out.min() >= 0.0 and out.max() <= 1.0, scores

    # The specific regression: 1.0 + 0.01 must be clamped, not passed through.
    assert prepare_scores([1.0, 1.0])[0] < 1.0


def test_prior_splits_both_evenly_and_floors_at_zero_support():
    alpha, beta = prior_from_support(10, 4, 2, rel_abstracts=20)
    assert (alpha, beta) == (11.0, 5.0)
    # No supporting abstracts at all -> a near-flat prior, not a confident one.
    assert prior_from_support(0, 0, 0, rel_abstracts=200) == (0.01, 0.01)


def test_core_imports_without_pandas_or_matplotlib():
    """SKiM_web installs skimgpt with --no-deps into a Flask image.

    If this fails, someone added a heavy import to ``skimgpt.stats`` and the
    web containers will fail to boot.
    """
    code = (
        "import sys; import skimgpt.stats.bayes_ci as m; "
        "heavy = [n for n in ('pandas','matplotlib','torch','openai','cmdlogtime') "
        "if n in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"heavy imports pulled in: {out.stdout.strip()}"
