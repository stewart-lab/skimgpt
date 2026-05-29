"""SKiM-ABC payload validation — locks in the holistic chain schema.

SKiM-ABC no longer does per-abstract relation extraction or per-leg tallies.
The model reads both pools and returns a single holistic verdict:
score_rationale + score + decision. The validator only enforces those three
fields and applies none of the per_abstract length / tally-consistency checks
used by KM / SKiM-AC / DCH.
"""
from types import SimpleNamespace

import pytest

from skimgpt.classifier import _validate_payload


def _payload(*, score=1.0, decision="supports", rationale=None):
    return {
        "score_rationale": rationale
        or ["BC abstracts establish C↓B (PMID 3); AB abstracts establish B↑A (PMID 1); chain supported."],
        "score": score,
        "decision": decision,
    }


def _abc_config():
    return SimpleNamespace(
        is_dch=False, is_km_with_gpt=False, is_skim_with_gpt=True,
        job_type="skim_with_gpt",
    )


def _abc_call(payload, expected_count=None):
    """Run _validate_payload in the SKiM-ABC mode (relationship_type='A_B_C')."""
    return _validate_payload(payload, _abc_config(), expected_count, relationship_type="A_B_C")


def test_accepts_holistic_payload():
    """A flat payload with just score_rationale + score + decision passes."""
    _abc_call(_payload())


def test_expected_count_is_ignored():
    """ABC is holistic — there is no per_abstract array, so a provided
    expected_per_abstract_count must NOT trigger a length check."""
    _abc_call(_payload(), expected_count=12)


def test_extra_legacy_fields_are_tolerated():
    """If a model still emits per_abstract / tallies / expected_chain, the
    validator ignores them — it neither requires nor consistency-checks them."""
    payload = _payload()
    payload["per_abstract"] = [{"pmid": "1", "pool": "AB"}]
    payload["tallies"] = {"ab_establishes_leg": 99}  # deliberately inconsistent
    payload["expected_chain"] = {"ab_expectation": "x", "bc_expectation": "y"}
    _abc_call(payload, expected_count=5)


@pytest.mark.parametrize("missing", ["score_rationale", "score", "decision"])
def test_rejects_missing_required_field(missing):
    payload = _payload()
    del payload[missing]
    with pytest.raises(ValueError, match=missing):
        _abc_call(payload)
