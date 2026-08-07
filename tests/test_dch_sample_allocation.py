"""DCH sample allocation — characterization tests for `_normalize_fractions`.

This function decides how many of the `DCH_SAMPLE_SIZE` abstracts come from
hypothesis 1's pool versus hypothesis 2's. It therefore sets the evidence
balance the DCH verdict rests on: skew the allocation and you skew the
H1-vs-H2 comparison, with nothing in the output revealing it happened.

Most of these are characterization tests — they pin the behavior that shipped,
not a specification derived independently. The structural invariants (target
total, pool caps, minimum floor) are the real contract; the golden-value cases
exist so that a refactor of the allocation arithmetic has to prove it preserves
the exact splits.

The one place the tests assert *changed* behavior is the minimum floor: it used
to under-deliver at high floor/target settings, and those cases are now
regression cover rather than pinned defects.
"""
import pytest

from skimgpt.relevance_helper import _normalize_fractions

# Shipped defaults, from config_template.json.
DEFAULT_FLOOR = 0.06
DEFAULT_TARGET = 50


# --- structural invariants ---------------------------------------------------


@pytest.mark.parametrize("target", [10, 50])
@pytest.mark.parametrize("total1", [0, 1, 2, 3, 7, 25, 49, 50, 51, 200, 1337])
@pytest.mark.parametrize("total2", [0, 1, 2, 3, 7, 25, 49, 50, 51, 200, 504])
def test_allocation_hits_the_target_or_exhausts_the_pools(total1, total2, target):
    """n1 + n2 must equal the target, or everything available when short."""
    n1, n2 = _normalize_fractions(total1, total2, DEFAULT_FLOOR, target)

    assert n1 + n2 == min(target, total1 + total2)


@pytest.mark.parametrize("target", [10, 50])
@pytest.mark.parametrize("total1", [0, 1, 3, 25, 51, 1337])
@pytest.mark.parametrize("total2", [0, 1, 3, 25, 51, 504])
def test_allocation_never_exceeds_either_pool(total1, total2, target):
    """Over-allocating would make random.sample raise ValueError at draw time."""
    n1, n2 = _normalize_fractions(total1, total2, DEFAULT_FLOOR, target)

    assert 0 <= n1 <= total1
    assert 0 <= n2 <= total2


@pytest.mark.parametrize("total1", [1, 2, 3, 5, 12, 30, 100, 1337])
@pytest.mark.parametrize("total2", [1, 2, 3, 5, 12, 504, 1500, 9000])
def test_min_floor_is_honored_at_default_settings(total1, total2):
    """At the shipped floor/target, the weaker pool always gets its minimum.

    The floor exists so a small pool still contributes evidence rather than
    being rounded out of the comparison entirely. Verified exhaustively at the
    default settings; see the under-delivery test below for where it breaks.
    """
    n1, n2 = _normalize_fractions(total1, total2, DEFAULT_FLOOR, DEFAULT_TARGET)
    floor_count = round(DEFAULT_FLOOR * DEFAULT_TARGET)

    assert n1 >= min(total1, floor_count)
    assert n2 >= min(total2, floor_count)


# --- golden allocations ------------------------------------------------------


@pytest.mark.parametrize(
    "total1,total2,expected",
    [
        # Empty pools.
        (0, 0, (0, 0)),
        (0, 10, (0, 10)),
        (10, 0, (10, 0)),
        (0, 1, (0, 1)),
        (1, 0, (1, 0)),
        # Both pools smaller than the target — take everything.
        (1, 1, (1, 1)),
        (2, 2, (2, 2)),
        (3, 3, (3, 3)),
        # Exactly the target.
        (25, 25, (25, 25)),
        # Balanced and larger than the target — an even split.
        (100, 100, (25, 25)),
        # Heavily skewed pools — the floor lifts the weaker side.
        (5, 1500, (3, 47)),
        (30, 1800, (3, 47)),
        # Weaker pool below even the floor — capped at what exists.
        (1, 999, (1, 49)),
        # A real observed run (tools/integration_test_run.log).
        (1337, 504, (36, 14)),
    ],
)
def test_golden_allocations_at_default_settings(total1, total2, expected):
    assert _normalize_fractions(total1, total2, DEFAULT_FLOOR, DEFAULT_TARGET) == expected


def test_allocation_is_symmetric_under_pool_swap():
    """Swapping the pools must mirror the split — neither hypothesis is favored."""
    for total1, total2 in [(5, 1500), (30, 1800), (1337, 504), (1, 999), (7, 12)]:
        n1, n2 = _normalize_fractions(total1, total2, DEFAULT_FLOOR, DEFAULT_TARGET)
        swapped = _normalize_fractions(total2, total1, DEFAULT_FLOOR, DEFAULT_TARGET)

        assert swapped == (n2, n1), f"asymmetry at ({total1}, {total2})"


# --- minimum floor at non-default settings -----------------------------------
#
# The floor used to be applied to the proportional shares and then renormalized
# by their sum, which divided the floored share back below the floor. These
# cases are the ones that exposed it; they are kept as regression cover.


@pytest.mark.parametrize("floor", [0.02, 0.06, 0.1, 0.2, 0.33, 0.5])
@pytest.mark.parametrize("target", [5, 10, 20, 50, 100, 200])
@pytest.mark.parametrize("total1,total2", [(12, 1504), (10, 1000), (1, 999), (30, 1800), (5, 9000)])
def test_min_floor_is_honored_at_every_setting(floor, target, total1, total2):
    """The floor holds regardless of how high min_floor or the target is set."""
    n1, n2 = _normalize_fractions(total1, total2, floor, target)
    floor_count = round(floor * target)

    assert n1 >= min(total1, floor_count)
    assert n2 >= min(total2, floor_count)
    assert n1 + n2 == min(target, total1 + total2)


def test_high_floor_lifts_the_weaker_pool_to_its_full_entitlement():
    """Regression: this returned (11, 189) — one short of the 12 the floor asks for."""
    assert _normalize_fractions(12, 1504, 0.06, 200) == (12, 188)


def test_worst_previous_shortfall_now_delivers_the_floor():
    """Regression: floor=0.33/target=200 used to come up 16 abstracts short."""
    n1, n2 = _normalize_fractions(80, 9000, 0.33, 200)

    assert n1 >= round(0.33 * 200)
    assert n1 + n2 == 200


def test_unsatisfiable_floors_split_evenly(caplog):
    """min_floor > 0.5 cannot be honored for two pools — split rather than skew."""
    with caplog.at_level("WARNING"):
        n1, n2 = _normalize_fractions(1000, 1000, 0.6, 50)

    assert (n1, n2) == (25, 25)
    assert "cannot be honored" in caplog.text


def test_unsatisfiable_floors_respect_pool_caps():
    """An even split must still never over-allocate a small pool."""
    n1, n2 = _normalize_fractions(4, 1000, 0.6, 50)

    assert n1 <= 4
    assert n2 <= 1000
    assert n1 + n2 == 50


# --- degenerate inputs -------------------------------------------------------


def test_both_pools_empty_allocates_nothing():
    assert _normalize_fractions(0, 0, DEFAULT_FLOOR, DEFAULT_TARGET) == (0, 0)


def test_zero_target_allocates_nothing():
    assert _normalize_fractions(100, 100, DEFAULT_FLOOR, 0) == (0, 0)


def test_odd_slot_tie_break_is_fixed_by_convention():
    """Equal pools with an odd slot count: the tie-break is arbitrary but pinned.

    `int(round(...))` is banker's rounding, so an exact .5 share resolves to the
    even count — which hands the odd slot to candidate 2 here. The direction is
    not meaningful; it is pinned so a change of rounding mode has to be a
    deliberate decision rather than a silent side effect. Unreachable at the
    shipped DCH_SAMPLE_SIZE of 50, where equal pools split exactly 25/25.
    """
    assert _normalize_fractions(1, 1, DEFAULT_FLOOR, 1) == (0, 1)
    assert _normalize_fractions(10, 10, DEFAULT_FLOOR, 50) == (10, 10)


def test_zero_floor_falls_back_to_proportional_split():
    """Without a floor, the split tracks pool sizes proportionally."""
    n1, n2 = _normalize_fractions(1000, 1000, 0.0, 50)
    assert (n1, n2) == (25, 25)

    n1, n2 = _normalize_fractions(1800, 200, 0.0, 50)
    assert n1 + n2 == 50
    assert n1 > n2
