"""The collision channel's physics, not just its labels (PLAN.md Section 6.7; M7 audit E-7).

The audit found the collision outcomes were "labels, not physics": no energy distribution for the heating kick, no
permutation distribution, and ``run/job.py`` only discarded the shot. ``Collisions`` now carries both distributions and
``noise/collisions.py`` samples them at Section 6.7's stated scale, the neutral's thermal energy times the mass ratio.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.noise.collisions import (
    GAS_MASS_U,
    collision_rate_per_ion,
    mean_kick_energy_j,
    mean_kick_quanta,
    sample_kick_quanta,
    sample_reorder,
)
from qutip_trap.noise.spectra import Collisions
from qutip_trap.units import ATOMIC_MASS_KG, K_B_J_PER_K

TORR_PA = 133.322368421
YB_KG = 171.0 * ATOMIC_MASS_KG
OUTCOMES: dict[str, float] = {"heating_kick": 0.4, "reorder": 0.3, "loss": 0.15, "dark_ion": 0.15}


def _col(**kw: object) -> Collisions:
    base: dict[str, object] = {
        "pressure_pa": 1e-11 * TORR_PA,
        "gas": {"H2": 1.0},
        "outcome_probabilities": dict(OUTCOMES),
    }
    base.update(kw)
    return Collisions(**base)  # type: ignore[arg-type]


def test_the_kick_energy_scale_is_the_neutrals_thermal_energy_times_the_mass_ratio() -> None:
    """Section 6.7: "a heating kick drawn from a configured energy distribution whose scale is the neutral's thermal
    energy times the mass ratio". <E> = k_B T (m_gas/m_ion), exactly, with the partial-pressure-weighted gas mass."""
    col = _col()
    expect = K_B_J_PER_K * 300.0 * (GAS_MASS_U["H2"] * ATOMIC_MASS_KG / YB_KG)
    assert mean_kick_energy_j(col, YB_KG) == pytest.approx(expect, rel=1e-12)
    # a heavier gas transfers proportionally more
    assert mean_kick_energy_j(_col(gas={"Ar": 1.0}), YB_KG) / mean_kick_energy_j(col, YB_KG) == pytest.approx(
        GAS_MASS_U["Ar"] / GAS_MASS_U["H2"], rel=1e-12
    )
    # a mixture's mass is the partial-pressure-weighted mean
    mixed = _col(gas={"H2": 0.5, "Ar": 0.5})
    assert mean_kick_energy_j(mixed, YB_KG) == pytest.approx(
        0.5 * (mean_kick_energy_j(col, YB_KG) + mean_kick_energy_j(_col(gas={"Ar": 1.0}), YB_KG)), rel=1e-12
    )
    # a cryogenic chamber kicks proportionally less
    assert mean_kick_energy_j(_col(temperature_k=4.0), YB_KG) / expect == pytest.approx(
        4.0 / 300.0, rel=1e-12
    )
    # the multiplier is the device's handle on the O(1) coefficient the plan does not state
    assert mean_kick_energy_j(_col(kick_scale_multiplier=2.5), YB_KG) == pytest.approx(
        2.5 * expect, rel=1e-12
    )
    with pytest.raises(ValueError):
        _col(kick_scale_multiplier=0.0)
    with pytest.raises(ValueError):
        mean_kick_energy_j(col, 0.0)


def test_the_kick_in_quanta_and_where_it_sits_against_the_plans_parenthetical() -> None:
    """<delta nbar> = <E>/(hbar omega_m). H2 at 300 K on 171Yb+ gives 7.4e4 quanta of a 1 MHz mode and 2.5e4 of a 3 MHz
    one, ABOVE Section 6.7's parenthetical "tens to thousands"; the plan states the SCALE, and the formula it states is
    what is implemented (``conv.collision_outcomes``). A cryogenic 4 K chamber lands inside the parenthetical (988 quanta
    of a 1 MHz mode), which is one way the two readings reconcile."""
    col = _col()
    q_1mhz = mean_kick_quanta(col, YB_KG, 2.0 * math.pi * 1e6)
    q_3mhz = mean_kick_quanta(col, YB_KG, 2.0 * math.pi * 3e6)
    assert q_1mhz == pytest.approx(7.36e4, rel=2e-2), q_1mhz
    assert q_3mhz == pytest.approx(q_1mhz / 3.0, rel=1e-12)
    assert q_1mhz > 1e4, "the plan's 'tens to thousands' is not reached at 300 K on a heavy ion"
    cryo = mean_kick_quanta(_col(temperature_k=4.0), YB_KG, 2.0 * math.pi * 1e6)
    assert 100.0 < cryo < 1e4, cryo
    with pytest.raises(ValueError):
        mean_kick_quanta(col, YB_KG, 0.0)
    assert mean_kick_quanta(_col(gas={}), YB_KG, 2.0 * math.pi * 1e6) == 0.0


@pytest.mark.parametrize("shape", ["exponential", "thermal_maxwell"])
def test_the_two_kick_distributions_have_the_stated_mean_and_the_right_spread(shape: str) -> None:
    """Both shapes carry the Section 6.7 mean; they differ in their tail, which is the whole reason the shape is a device
    input. An exponential has Var = <E>^2 (relative spread 1); the Maxwell chi^2_3 has Var = (2/3)<E>^2 (0.816), so the
    exponential produces the rarer very large kicks a laboratory's outlier tail shows."""
    col = _col(kick_distribution=shape)
    w = 2.0 * math.pi * 1e6
    mean = mean_kick_quanta(col, YB_KG, w)
    rng = np.random.default_rng(4)
    draws = np.array([sample_kick_quanta(rng, col, YB_KG, w) for _ in range(20000)])
    assert float(np.mean(draws)) == pytest.approx(mean, rel=0.03)
    assert np.all(draws >= 0.0)
    rel_sd = float(np.std(draws)) / mean
    assert rel_sd == pytest.approx(1.0 if shape == "exponential" else math.sqrt(2.0 / 3.0), rel=0.05)
    # a zero-mean configuration draws exactly zero, never a small invented number
    assert sample_kick_quanta(rng, _col(gas={}), YB_KG, w) == 0.0
    # reproducible from the seed (Section 9.17)
    a = np.array([sample_kick_quanta(np.random.default_rng(9), col, YB_KG, w) for _ in range(3)])
    b = np.array([sample_kick_quanta(np.random.default_rng(9), col, YB_KG, w) for _ in range(3)])
    assert np.array_equal(a, b)


def test_the_reorder_draws_from_the_configured_permutation_distribution() -> None:
    """Section 6.7: "a reorder, sampled from a configured permutation distribution when the kick exceeds the ordering
    barrier, which permutes the ion labels". With permutations configured they are drawn uniformly and applied as
    new[k] = order[perm[k]]; without one the adjacent transposition at the struck ion is used, the single swap a marginal
    kick makes."""
    rng = np.random.default_rng(1)
    three = _col(reorder_permutations=((1, 0, 2), (0, 2, 1), (2, 1, 0)))
    seen = {sample_reorder(rng, three, (0, 1, 2), 0) for _ in range(400)}
    assert seen == {(1, 0, 2), (0, 2, 1), (2, 1, 0)}, seen
    # applied to an already-permuted order, the permutation composes rather than resetting
    assert sample_reorder(np.random.default_rng(0), _col(reorder_permutations=((1, 0),)), (1, 0), 0) == (0, 1)
    # no configured distribution: the adjacent transposition at the struck ion
    plain = _col()
    assert sample_reorder(rng, plain, (0, 1, 2, 3), 1) == (0, 2, 1, 3)
    assert sample_reorder(rng, plain, (0, 1, 2, 3), 3) == (0, 1, 3, 2), "the last ion swaps downwards"
    assert sample_reorder(rng, plain, (5,), 5) == (5,), "a single ion cannot reorder"
    # a permutation of the wrong length is ignored rather than mis-applied
    assert sample_reorder(rng, _col(reorder_permutations=((1, 0),)), (0, 1, 2), 0) == (1, 0, 2)
    with pytest.raises(ValueError):
        _col(reorder_permutations=((0, 0),))
    with pytest.raises(ValueError):
        _col(reorder_permutations=((1, 2),))


def test_the_langevin_rate_is_unchanged_by_the_new_fields() -> None:
    """A regression guard on the new ``Collisions`` fields: the Section 6.7 rate is still k_L = 1.48e-9 cm^3/s and
    Gamma_L = 4.78e-4 /s for H2 at 1e-11 torr and 300 K on 171Yb+, one event per 35 minutes per ion."""
    rate = collision_rate_per_ion(_col(kick_distribution="thermal_maxwell", kick_scale_multiplier=3.0), YB_KG)
    assert rate == pytest.approx(4.78e-4, rel=5e-3)
    assert 1.0 / rate / 60.0 == pytest.approx(34.9, rel=5e-3), "one event per 35 minutes per ion"
