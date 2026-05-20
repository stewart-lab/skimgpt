"""DCH payload validation — locks in the consistency check that catches
LLM tally/per_abstract drift. Without these tests it is easy to silently
weaken `_validate_payload` and reintroduce the ±1-2 bucket drift that
shipped to users in skimgpt 2.0.1 (see SKiM_web km_query 26: ~44% of
DCH rows had bucket disagreements between `tallies` and `per_abstract`).
"""
from types import SimpleNamespace

import pytest

from skimgpt.classifier import _validate_payload


def _payload(per_abstract, tallies, *, score=50, decision="tie"):
    return {
        "per_abstract": per_abstract,
        "score_rationale": ["rationale"],
        "tallies": tallies,
        "score": score,
        "decision": decision,
    }


def _dch_config():
    return SimpleNamespace(is_dch=True)


def test_accepts_consistent_payload():
    per = [
        {"pmid": "1", "label": "supports_H1"},
        {"pmid": "2", "label": "supports_H2"},
        {"pmid": "3", "label": "both"},
        {"pmid": "4", "label": "neither"},
        {"pmid": "5", "label": "inconclusive"},
    ]
    tallies = {"support_H1": 1, "support_H2": 1, "both": 1, "neither_or_inconclusive": 2}
    _validate_payload(_payload(per, tallies), _dch_config(), 5)


def test_rejects_drifted_tallies():
    per = [{"pmid": "1", "label": "supports_H1"}]
    tallies = {"support_H1": 2, "support_H2": 0, "both": 0, "neither_or_inconclusive": 0}
    with pytest.raises(ValueError, match="support_H1"):
        _validate_payload(_payload(per, tallies), _dch_config(), 1)


def test_rejects_phantom_neither_count():
    """The exact failure mode observed in km_query 26 row 318: tally claims
    1 neither_or_inconclusive, but no per_abstract entry carries that label."""
    per = [
        {"pmid": "1", "label": "supports_H1"},
        {"pmid": "2", "label": "supports_H2"},
    ]
    # Only the neither_or_inconclusive bucket is off — H1/H2/both all match.
    tallies = {"support_H1": 1, "support_H2": 1, "both": 0, "neither_or_inconclusive": 1}
    with pytest.raises(ValueError, match="neither_or_inconclusive"):
        _validate_payload(_payload(per, tallies), _dch_config(), 2)


def test_rejects_legacy_six_key_schema():
    """Trunk's dual-label experiment emitted 6-key tallies. The deployed
    schema is 4-key; a payload missing 'both' or 'neither_or_inconclusive'
    must fail loudly so we cannot silently regress to the dual-label shape."""
    per = [{"pmid": "1", "label": "supports_H1"}]
    tallies = {
        "support_H1": 1, "refute_H1": 0, "inconclusive_H1": 0,
        "support_H2": 0, "refute_H2": 0, "inconclusive_H2": 0,
    }
    with pytest.raises(ValueError, match="both"):
        _validate_payload(_payload(per, tallies), _dch_config(), 1)


def test_neither_and_inconclusive_merge_into_one_bucket():
    """Per_abstract labels split neither vs inconclusive; the tally collapses
    them. The validator must accept payloads where the merged count matches."""
    per = [
        {"pmid": "1", "label": "neither"},
        {"pmid": "2", "label": "neither"},
        {"pmid": "3", "label": "inconclusive"},
    ]
    tallies = {"support_H1": 0, "support_H2": 0, "both": 0, "neither_or_inconclusive": 3}
    _validate_payload(_payload(per, tallies), _dch_config(), 3)


def test_rejects_missing_required_top_level_field():
    bad = _payload([{"pmid": "1", "label": "supports_H1"}],
                   {"support_H1": 1, "support_H2": 0, "both": 0, "neither_or_inconclusive": 0})
    del bad["score"]
    with pytest.raises(ValueError, match="score"):
        _validate_payload(bad, _dch_config(), 1)


def test_rejects_per_abstract_length_mismatch():
    per = [{"pmid": "1", "label": "supports_H1"}]
    tallies = {"support_H1": 1, "support_H2": 0, "both": 0, "neither_or_inconclusive": 0}
    with pytest.raises(ValueError, match="per_abstract length"):
        _validate_payload(_payload(per, tallies), _dch_config(), 5)
