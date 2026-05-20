"""SKiM-ABC payload validation — locks in the chain-extraction schema's
consistency checks. The ABC validator branch reconstructs per-leg tallies
from per_abstract (pool, direction, supports_chain) and must catch drift
the same way the DCH branch does for label/tally drift.
"""
from types import SimpleNamespace

import pytest

from skimgpt.classifier import _validate_payload


def _payload(per_abstract, tallies, *, score=1.0, decision="supports",
             ab_exp="B is elevated in A", bc_exp="C decreases B"):
    return {
        "expected_chain": {"ab_expectation": ab_exp, "bc_expectation": bc_exp},
        "per_abstract": per_abstract,
        "score_rationale": ["BC abstracts establish C↓B; AB abstracts establish B↑A; chain supported."],
        "tallies": tallies,
        "score": score,
        "decision": decision,
    }


def _abc_config():
    return SimpleNamespace(
        is_dch=False, is_km_with_gpt=False, is_skim_with_gpt=True,
        job_type="skim_with_gpt",
    )


def _abc_call(payload, expected_count):
    """Run _validate_payload in the SKiM-ABC mode (relationship_type='A_B_C')."""
    return _validate_payload(payload, _abc_config(), expected_count, relationship_type="A_B_C")


def _ab(pmid, direction, supports):
    return {"pmid": pmid, "pool": "AB", "evidence": ["q"],
            "subject": "B", "object": "A", "direction": direction,
            "supports_chain": supports}


def _bc(pmid, direction, supports):
    return {"pmid": pmid, "pool": "BC", "evidence": ["q"],
            "subject": "C", "object": "B", "direction": direction,
            "supports_chain": supports}


def test_accepts_consistent_payload():
    """A well-formed payload with tallies that match the reconstructed
    per-leg counts must pass."""
    per = [
        _ab("1", "increases", True),    # AB leg established
        _ab("2", "decreases", False),   # AB leg opposed
        _bc("3", "decreases", True),    # BC leg established
        _bc("4", "decreases", True),    # BC leg established
        _ab("5", "absent",    False),   # irrelevant
    ]
    tallies = {
        "ab_establishes_leg": 1, "ab_opposes_leg": 1,
        "bc_establishes_leg": 2, "bc_opposes_leg": 0,
        "irrelevant": 1,
    }
    _abc_call(_payload(per, tallies), 5)


def test_rejects_drifted_tallies():
    """Tallies that don't match the per_abstract reconstruction must fail
    so the retry loop catches it."""
    per = [_ab("1", "increases", True)]   # one AB establishes-leg entry
    tallies = {
        "ab_establishes_leg": 2,          # claims 2; only 1 present
        "ab_opposes_leg": 0,
        "bc_establishes_leg": 0, "bc_opposes_leg": 0,
        "irrelevant": 0,
    }
    with pytest.raises(ValueError, match="ab_establishes_leg"):
        _abc_call(_payload(per, tallies), 1)


def test_rejects_missing_expected_chain():
    """`expected_chain` is the schema-ordered CoT anchor that forces the
    model to commit to leg directions before judging abstracts. A payload
    missing it must fail loudly — otherwise the model can fall back to
    name-the-bucket-first behaviour."""
    per = [_ab("1", "increases", True)]
    tallies = {
        "ab_establishes_leg": 1, "ab_opposes_leg": 0,
        "bc_establishes_leg": 0, "bc_opposes_leg": 0,
        "irrelevant": 0,
    }
    bad = _payload(per, tallies)
    del bad["expected_chain"]
    with pytest.raises(ValueError, match="expected_chain"):
        _abc_call(bad, 1)


def test_pool_mismatch_routes_to_irrelevant():
    """An entry with an unknown pool value (e.g. truncated or model
    hallucination) is bucketed as irrelevant; tallies that count it as a
    leg-establishing entry must therefore mismatch."""
    weird = {"pmid": "1", "pool": "XYZ", "evidence": ["q"],
             "subject": "?", "object": "?", "direction": "increases",
             "supports_chain": True}
    # If we (incorrectly) tally it as an AB leg, validation must reject.
    tallies = {
        "ab_establishes_leg": 1, "ab_opposes_leg": 0,
        "bc_establishes_leg": 0, "bc_opposes_leg": 0,
        "irrelevant": 0,
    }
    with pytest.raises(ValueError, match="ab_establishes_leg|irrelevant"):
        _abc_call(_payload([weird], tallies), 1)


def test_rejects_per_abstract_length_mismatch():
    per = [_ab("1", "increases", True)]
    tallies = {
        "ab_establishes_leg": 1, "ab_opposes_leg": 0,
        "bc_establishes_leg": 0, "bc_opposes_leg": 0,
        "irrelevant": 0,
    }
    with pytest.raises(ValueError, match="per_abstract length"):
        _abc_call(_payload(per, tallies), 5)


def test_absent_direction_counts_as_irrelevant_not_per_pool():
    """An abstract with direction=absent is irrelevant regardless of pool —
    used for topical co-mention, belief/survey reports, and reviews that
    take no directional stance. Tally must put it in `irrelevant`, not in
    `ab_opposes_leg` or `bc_opposes_leg`."""
    per = [
        _ab("1", "absent", False),   # absent → irrelevant, NOT ab_opposes_leg
        _bc("2", "absent", False),   # absent → irrelevant, NOT bc_opposes_leg
    ]
    tallies = {
        "ab_establishes_leg": 0, "ab_opposes_leg": 0,
        "bc_establishes_leg": 0, "bc_opposes_leg": 0,
        "irrelevant": 2,
    }
    _abc_call(_payload(per, tallies), 2)


def test_rejects_absent_misrouted_as_opposes():
    """Inverse of the above: if the model claims ab_opposes_leg=1 but the
    AB entry is direction=absent, validation must fail."""
    per = [_ab("1", "absent", False)]
    tallies = {
        "ab_establishes_leg": 0, "ab_opposes_leg": 1,   # wrong: absent goes to irrelevant
        "bc_establishes_leg": 0, "bc_opposes_leg": 0,
        "irrelevant": 0,
    }
    with pytest.raises(ValueError, match="ab_opposes_leg|irrelevant"):
        _abc_call(_payload(per, tallies), 1)
