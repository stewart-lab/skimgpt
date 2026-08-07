"""DCH sample allocation — characterization tests for `_normalize_fractions`.

This function decides how many of the `DCH_SAMPLE_SIZE` abstracts come from
hypothesis 1's pool versus hypothesis 2's. It therefore sets the evidence
balance the DCH verdict rests on: skew the allocation and you skew the
H1-vs-H2 comparison, with nothing in the output revealing it happened.

These tests are deliberately characterization tests — they pin the behavior
that shipped, not a specification derived independently. The three structural
invariants (target total, pool caps, minimum floor at default settings) are the
real contract; the golden-value cases exist so that a refactor of the
allocation arithmetic has to prove it preserves the exact splits.

`test_min_floor_can_under_deliver_at_high_floor_settings` pins a known defect
rather than a desired behavior. See its docstring.
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


# --- known defect ------------------------------------------------------------


def test_min_floor_can_under_deliver_at_high_floor_settings():
    """KNOWN DEFECT, pinned deliberately — this is not desired behavior.

    The floor is applied as ``max(share, min_floor)``, but both shares are then
    renormalized by their sum, which divides the floored share back down below
    the floor. So DCH_MIN_SAMPLING_FRACTION is not actually guaranteed.

    At the shipped defaults (0.06 / 50) the rounding hides it completely — a
    sweep of ~1M combinations found zero violations, which is why nothing has
    noticed. It surfaces as the floor or target rises, and the shortfall grows
    with the floor: short by 1 at floor=0.06/target=200, by 3 at
    floor=0.2/target=100, by 16 at floor=0.33/target=200. Whoever first raises
    DCH_MIN_SAMPLING_FRACTION to get a more balanced comparison silently gets
    less balance than they asked for.

    If this is fixed, delete this test — do not adjust its numbers to match.
    """
    floor, target = 0.06, 200
    n1, n2 = _normalize_fractions(12, 1504, floor, target)

    assert (n1, n2) == (11, 189)
    assert n1 < round(floor * target)  # asked for 12, got 11


# --- degenerate inputs -------------------------------------------------------


def test_both_pools_empty_allocates_nothing():
    assert _normalize_fractions(0, 0, DEFAULT_FLOOR, DEFAULT_TARGET) == (0, 0)


def test_zero_target_allocates_nothing():
    assert _normalize_fractions(100, 100, DEFAULT_FLOOR, 0) == (0, 0)


def test_zero_floor_falls_back_to_proportional_split():
    """Without a floor, the split tracks pool sizes proportionally."""
    n1, n2 = _normalize_fractions(1000, 1000, 0.0, 50)
    assert (n1, n2) == (25, 25)

    n1, n2 = _normalize_fractions(1800, 200, 0.0, 50)
    assert n1 + n2 == 50
    assert n1 > n2
