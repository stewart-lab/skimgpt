"""DCH sampling reproducibility — locks in the DCH_SAMPLE_SEED contract.

Seeded sampling fails invisibly: without these tests, an edit that drops the
`rng` argument anywhere along the chain leaves sampling working perfectly, just
no longer reproducible, and nothing downstream complains. The per-iteration
derivation is equally load-bearing — collapsing it to a single seed would make
every iteration draw the same sample, quietly turning an N-iteration variance
estimate into N copies of one draw.
"""
from types import SimpleNamespace

import pytest

from skimgpt.relevance_helper import (
    _dch_rng,
    _pool_fingerprint,
    _sample_entries,
    sample_consolidated_abstracts,
)
from skimgpt.utils import extract_pmid

DELIMITER = "===END OF ABSTRACT==="


def _pool(prefix: str, count: int) -> list[str]:
    """Build a pool of distinguishable abstract entries with parseable PMIDs."""
    return [
        f"Title {prefix}{i}. Abstract body {prefix}{i}. PMID: {prefix}{i:04d}{DELIMITER}"
        for i in range(count)
    ]


def _config(seed=None, sample_size=10, min_fraction=0.06):
    return SimpleNamespace(
        global_settings={
            "DCH_SAMPLE_SEED": seed,
            "DCH_SAMPLE_SIZE": sample_size,
            "DCH_MIN_SAMPLING_FRACTION": min_fraction,
        }
    )


def _sampled_pmids(v1, v2, config, rng):
    consolidated, _, _ = sample_consolidated_abstracts(v1, v2, config, rng)
    return [extract_pmid(entry) for entry in consolidated.split(DELIMITER) if entry.strip()]


# --- seeded reproducibility ---------------------------------------------------


def test_same_seed_same_iteration_reproduces_sample():
    v1, v2 = _pool("1", 200), _pool("2", 80)
    config = _config(seed=1234)

    first = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))
    second = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))

    assert first == second
    assert len(first) == 10


def test_iterations_draw_different_samples():
    """The whole point of iterations — a shared seed must not collapse them."""
    v1, v2 = _pool("1", 200), _pool("2", 80)
    config = _config(seed=1234)

    iter1 = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))
    iter2 = _sampled_pmids(v1, v2, config, _dch_rng(config, 2))

    assert iter1 != iter2


def test_different_seeds_draw_different_samples():
    v1, v2 = _pool("1", 200), _pool("2", 80)

    seed_a = _config(seed=1)
    seed_b = _config(seed=2)

    assert _sampled_pmids(v1, v2, seed_a, _dch_rng(seed_a, 1)) != _sampled_pmids(
        v1, v2, seed_b, _dch_rng(seed_b, 1)
    )


def test_cross_seed_iteration_pairs_do_not_collide():
    """(seed=1, iter=2) must differ from (seed=2, iter=1).

    Integer addition would make these identical, silently sharing draws between
    runs that are meant to be independent.
    """
    v1, v2 = _pool("1", 200), _pool("2", 80)

    seed_a = _config(seed=1)
    seed_b = _config(seed=2)

    assert _sampled_pmids(v1, v2, seed_a, _dch_rng(seed_a, 2)) != _sampled_pmids(
        v1, v2, seed_b, _dch_rng(seed_b, 1)
    )


def test_seeded_sampling_is_independent_of_global_random_state():
    """Iterations run concurrently, so the draw must not depend on shared state."""
    import random

    v1, v2 = _pool("1", 200), _pool("2", 80)
    config = _config(seed=99)

    random.seed(0)
    first = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))
    random.seed(12345)
    [random.random() for _ in range(50)]
    second = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))

    assert first == second


# --- unseeded default --------------------------------------------------------


def test_no_seed_means_no_rng():
    assert _dch_rng(_config(seed=None), 1) is None


def test_sample_entries_falls_back_to_module_random():
    pool = _pool("1", 100)
    drawn = _sample_entries(pool, 5, None)

    assert len(drawn) == 5
    assert all(entry in pool for entry in drawn)


def test_unseeded_sampling_still_returns_the_target_size():
    v1, v2 = _pool("1", 200), _pool("2", 80)
    config = _config(seed=None)

    consolidated, expected_count, total = sample_consolidated_abstracts(
        v1, v2, config, _dch_rng(config, 1)
    )

    assert expected_count == 10
    assert total == 280
    assert consolidated


# --- edges -------------------------------------------------------------------


def test_zero_count_returns_empty_without_touching_rng():
    """count == 0 must not consume RNG draws, or iteration streams would shift."""
    assert _sample_entries(_pool("1", 10), 0, _dch_rng(_config(seed=7), 1)) == []


def test_non_iterated_run_is_seeded_too():
    """iteration_number == 0 (no iterations configured) still gets a stable draw."""
    v1, v2 = _pool("1", 200), _pool("2", 80)
    config = _config(seed=1234)

    first = _sampled_pmids(v1, v2, config, _dch_rng(config, 0))
    second = _sampled_pmids(v1, v2, config, _dch_rng(config, 0))

    assert first == second


def test_seed_zero_is_honored_not_treated_as_unset():
    config = _config(seed=0)
    assert _dch_rng(config, 1) is not None


def test_pool_smaller_than_sample_size_is_exhausted_deterministically():
    v1, v2 = _pool("1", 3), _pool("2", 2)
    config = _config(seed=1234, sample_size=50)

    first = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))
    second = _sampled_pmids(v1, v2, config, _dch_rng(config, 1))

    assert first == second
    assert sorted(first) == sorted(
        [extract_pmid(entry) for entry in v1 + v2]
    )


# --- pool fingerprint --------------------------------------------------------


def test_fingerprint_is_stable_and_order_sensitive():
    pmids = ["111", "222", "333"]

    assert _pool_fingerprint(pmids) == _pool_fingerprint(list(pmids))
    assert _pool_fingerprint(pmids) != _pool_fingerprint(list(reversed(pmids)))
    assert _pool_fingerprint(pmids) != _pool_fingerprint(pmids + ["444"])
    assert len(_pool_fingerprint(pmids)) == 8


def test_fingerprint_handles_empty_pool():
    assert len(_pool_fingerprint([])) == 8


# --- config validation -------------------------------------------------------


@pytest.mark.parametrize("bad_seed", ["abc", 1.5, [1], {"a": 1}, True])
def test_config_rejects_non_integer_seed(bad_seed):
    from skimgpt.utils import Config

    config = Config.__new__(Config)
    config.global_settings = {"DCH_SAMPLE_SEED": bad_seed}

    with pytest.raises(ValueError, match="DCH_SAMPLE_SEED"):
        config._validate_dch_sample_seed()


@pytest.mark.parametrize("good_seed", [None, 0, 1, 1234, -5])
def test_config_accepts_integer_or_null_seed(good_seed):
    from skimgpt.utils import Config

    config = Config.__new__(Config)
    config.global_settings = {"DCH_SAMPLE_SEED": good_seed}

    config._validate_dch_sample_seed()


def test_config_accepts_missing_seed_key():
    from skimgpt.utils import Config

    config = Config.__new__(Config)
    config.global_settings = {}

    config._validate_dch_sample_seed()
