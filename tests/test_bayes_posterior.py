"""Golden vectors for Beth's updated Beta posterior in ``skimgpt.stats``.

``skimgpt/stats/bayes_posterior.py`` is the single source of truth for the
DCH posterior: the ``skimgpt.visualization`` CLIs and SKiM_web's result pages
both call it, and ``bayes_ci_updated.R`` ports it.

The expectations in ``bayes_posterior_gold.json`` were captured by running
the original ``bayes_ci_updated.py`` from the ``update_model`` branch, so a
passing run means moving the maths into ``skimgpt.stats`` changed no number.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from skimgpt.stats.bayes_posterior import (
    call_from_result,
    n_effective,
    posterior_params,
    posterior_summary,
)

GOLD = json.loads((Path(__file__).parent / "bayes_posterior_gold.json").read_text())


def _calls(case):
    return [(score, [tuple(p) for p in pmids]) for score, pmids in case["calls"]]


@pytest.mark.parametrize("name", sorted(GOLD))
def test_matches_original_script(name):
    case = GOLD[name]
    calls = _calls(case)
    a, b = posterior_params(calls)
    assert (a, b) == pytest.approx((case["a"], case["b"]), rel=1e-12)

    fit = posterior_summary(calls)
    row = case["row"]
    assert fit.posterior_mode == pytest.approx(row["posterior_score"], rel=1e-12)
    assert fit.posterior_mean == pytest.approx(100 * case["posterior_mean"], rel=1e-12)
    assert fit.hdi_lo == pytest.approx(row["hdi_lo"], rel=1e-9, abs=1e-12)
    assert fit.hdi_hi == pytest.approx(row["hdi_hi"], rel=1e-9, abs=1e-12)
    assert fit.mean_llm_score == pytest.approx(row["mean_llm_score"], rel=1e-12)
    assert fit.n_unique_pmids == row["total_unique_abstracts"]

    lo90, hi90 = case["hdi90"]
    fit90 = posterior_summary(calls, level=0.90)
    assert (fit90.hdi_lo, fit90.hdi_hi) == pytest.approx(
        (100 * lo90, 100 * hi90), rel=1e-9, abs=1e-12
    )


@pytest.mark.parametrize("name", sorted(GOLD))
def test_timecourse_rows_match_original_script(name):
    """The CLI's CSV row is unchanged (skipped without matplotlib/cmdlogtime)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cmdlogtime")
    from skimgpt.visualization.bayes_ci_updated import timecourse_data

    row = timecourse_data({2000: _calls(GOLD[name])})[0]
    for key, expected in GOLD[name]["row"].items():
        if isinstance(expected, float) and expected != expected:  # NaN
            assert row[key] != row[key], key
        else:
            assert row[key] == pytest.approx(expected, rel=1e-9, abs=1e-12), key


def test_no_evidence_is_the_prior():
    fit = posterior_summary([(90, []), (95, [])])
    assert (fit.a, fit.b) == (1.2, 1.2)
    assert fit.n_eff == 0.0


def test_repeated_pmids_count_once():
    one = [(70, [("1", "supports_H1"), ("2", "supports_H1")])]
    assert posterior_params(one * 5)[0] < posterior_params(
        [(70, [(str(i), "supports_H1"), (str(i + 100), "supports_H1")]) for i in range(5)]
    )[0]


def test_n_effective_is_capped_by_call_noise():
    # More unique abstracts helps, but never beyond n_calls / SLOPE.
    assert n_effective(10_000_000, 10) < 10 / 0.00761
    assert n_effective(10, 10) < 10


def test_call_from_result_drops_non_evidence():
    result = {
        "score": 60,
        "per_abstract": [
            {"pmid": 1, "label": "supports_H1", "evidence": ["x"]},
            {"pmid": "2", "label": "neither", "evidence": ["x"]},
            {"pmid": "3", "label": "inconclusive", "evidence": ["x"]},
            {"pmid": "4", "label": "both", "evidence": ["x"]},
        ],
    }
    assert call_from_result(result) == (60, [("1", "supports_H1"), ("4", "both")])
    assert call_from_result({"score": 50}) == (50, [])


def test_empty_calls_rejected():
    with pytest.raises(ValueError):
        posterior_summary([])


def test_core_imports_without_pandas_or_matplotlib():
    """SKiM_web installs skimgpt with --no-deps into a Flask image."""
    code = (
        "import sys; import skimgpt.stats.bayes_posterior; import skimgpt.stats; "
        "heavy = [n for n in ('pandas','matplotlib','torch','openai','cmdlogtime') "
        "if n in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"heavy imports pulled in: {out.stdout.strip()}"
