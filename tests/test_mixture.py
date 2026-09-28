"""The heavy combinations of a product distribution (``dynamics/mixture.py``, PLAN.md Sections 5.2 to 5.4): the pruned
depth-first walk returns what the walk over the full product returns, combination for combination, in its order and bit
for bit, and refuses weights its bound does not hold for."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qutip_trap.dynamics.mixture import Combination, heavy_combinations

PROBABILITIES = st.one_of(
    st.sampled_from([0.0, 1e-3, 0.1, 0.25, 0.5, 1.0]),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0).map(lambda p: p**6),
)
"""An option's probability: exact zeros, ties and round values beside arbitrary ones, and small ones near the cuts."""
CUTS = st.one_of(
    st.sampled_from([0.0, 5e-324, 1e-310, 1e-6, 1e-4, 1e-3, 1e-2, 0.2, 1.0]), st.floats(0.0, 1.0)
)
"""A cut: zero, subnormal (which prunes nothing before the leaf test), the run's scales and arbitrary."""


def _full_walk(
    factors: Sequence[Sequence[float]], weight_min: float, scale: float | None
) -> list[tuple[tuple[int, ...], float]]:
    """Every leaf of ``itertools.product``, weighed and cut as the enumerations weighed and cut it before the pruned walk:
    ``float(np.prod(ps)) if ps else 1.0`` at the top level, ``scale * float(np.prod(ps)) if ps else scale`` under a choice of
    weight ``scale`` (``run.job.enumerate_branches``' modes under its internal levels)."""
    out: list[tuple[tuple[int, ...], float]] = []
    for combo in itertools.product(*[list(enumerate(f)) for f in factors]):
        ps = [p for _k, p in combo]
        if scale is None:
            w = float(np.prod(ps)) if combo else 1.0
        else:
            w = scale * float(np.prod(ps)) if combo else scale
        if w < weight_min:
            continue
        out.append((tuple(k for k, _p in combo), w))
    return out


@given(
    factors=st.lists(st.lists(PROBABILITIES, min_size=0, max_size=6), min_size=0, max_size=4),
    weight_min=CUTS,
    scale=st.one_of(st.none(), st.sampled_from([1.0, 0.5, 1e-5]), st.floats(0.0, 1.0)),
)
def test_the_pruned_walk_keeps_what_the_full_walk_keeps_in_its_order_and_bit_for_bit(
    factors: list[list[float]], weight_min: float, scale: float | None
) -> None:
    kept = (
        heavy_combinations(factors, weight_min)
        if scale is None
        else heavy_combinations(factors, weight_min, scale=scale)
    )
    expected = _full_walk(factors, weight_min, scale)
    assert [c.choice for c in kept] == [choice for choice, _w in expected]
    assert [c.weight.hex() for c in kept] == [w.hex() for _choice, w in expected]


def test_no_factor_is_one_empty_combination_and_a_factor_without_options_empties_the_product() -> None:
    assert heavy_combinations([], 1e-6) == [Combination((), 1.0)]
    assert heavy_combinations([], 1e-6, scale=0.25) == [Combination((), 0.25)]
    assert heavy_combinations([], 0.5, scale=0.25) == []
    assert heavy_combinations([[0.5, 0.5], []], 0.0) == []
    assert heavy_combinations([[1.0]], 1e-6) == [Combination((0,), 1.0)]


def test_equal_probabilities_keep_the_product_order_and_the_cut_keeps_its_boundary() -> None:
    """Ties are kept in the order of the product, and a weight exactly at the cut is kept."""
    kept = heavy_combinations([[0.5, 0.5], [0.25, 0.5, 0.25]], 0.125)
    assert [c.choice for c in kept] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert [c.weight for c in kept] == [0.125, 0.25, 0.125, 0.125, 0.25, 0.125]
    assert [c.choice for c in heavy_combinations([[0.5, 0.5], [0.25, 0.5, 0.25]], 0.2)] == [(0, 1), (1, 1)]


@pytest.mark.parametrize(
    ("factors", "scale"),
    [
        ([[0.5, -0.1]], 1.0),
        ([[math.nan]], 1.0),
        ([[0.5], [math.inf]], 1.0),
        ([[0.5]], -1.0),
        ([[0.5]], math.nan),
    ],
)
def test_a_weight_the_bound_does_not_hold_for_is_refused(factors: list[list[float]], scale: float) -> None:
    """The pruning bound (a prefix times the largest remaining probabilities) holds for finite non-negative weights only."""
    with pytest.raises(ValueError, match="finite"):
        heavy_combinations(factors, 1e-6, scale=scale)
