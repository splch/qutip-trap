"""Background-gas collisions as a discrete event process: the Langevin rate, the event sampling, the kick energy and the
reorder distribution (PLAN.md Section 6.7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.noise.collisions import (
    GAS_MASS_U,
    POLARIZABILITY_VOLUME_M3,
    collision_rate_per_ion,
    langevin_rate_coefficient_m3_s,
    mean_kick_energy_j,
    mean_kick_quanta,
    number_density_per_m3,
    sample_collisions,
    sample_kick_quanta,
    sample_reorder,
)
from qutip_trap.noise.spectra import Collisions
from qutip_trap.units import ATOMIC_MASS_KG, K_B_J_PER_K

TORR_PA = 133.32236842105263
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


def test_langevin_rate_at_1e_11_torr() -> None:
    """H2 at 1e-11 torr and 300 K: k_L = 1.63e-9 / 1.51e-9 / 1.48e-9 cm^3/s and Gamma_L = 5.25e-4 / 4.87e-4 / 4.78e-4 s^-1
    per ion for 9, 40 and 171 u, one event per 32 to 35 minutes per ion."""
    col = Collisions(1e-11 * TORR_PA, {"H2": 1.0}, {"heating_kick": 1.0})
    assert number_density_per_m3(col.pressure_pa, 300.0) * 1e-6 == pytest.approx(3.22e5, rel=2e-3)
    for mass_u, k_l, gamma in (
        (9.012, 1.63e-9, 5.25e-4),
        (40.08, 1.51e-9, 4.87e-4),
        (171.0, 1.48e-9, 4.78e-4),
    ):
        m = mass_u * ATOMIC_MASS_KG
        assert langevin_rate_coefficient_m3_s(
            POLARIZABILITY_VOLUME_M3["H2"], m, GAS_MASS_U["H2"] * ATOMIC_MASS_KG
        ) * 1e6 == pytest.approx(k_l, rel=5e-3)
        assert collision_rate_per_ion(col, m) == pytest.approx(gamma, rel=5e-3)
    assert 1.0 / collision_rate_per_ion(col, YB_KG) / 60.0 == pytest.approx(35.0, abs=1.0)
    mixed = Collisions(1e-11 * TORR_PA, {"H2": 0.5, "N2": 0.5}, {"heating_kick": 0.5, "loss": 0.5})
    assert collision_rate_per_ion(mixed, YB_KG) < collision_rate_per_ion(col, YB_KG), (
        "N2 is heavier and only twice as polarizable"
    )
    with pytest.raises(ValueError):
        Collisions(1.0, {"H2": 0.7}, {"heating_kick": 1.0})
    with pytest.raises(KeyError):
        collision_rate_per_ion(Collisions(1.0, {"Xe": 1.0}, {"loss": 1.0}), YB_KG)


def test_event_sampling_is_poisson_with_uniform_times_and_the_configured_outcomes() -> None:
    col = Collisions(
        1e-11 * TORR_PA, {"H2": 1.0}, {"heating_kick": 0.6, "reorder": 0.1, "loss": 0.2, "dark_ion": 0.1}
    )
    rng = np.random.default_rng(0)
    rates = {0: 2.0, 1: 2.0}
    n_events, outcomes, times = 0, {"heating_kick": 0, "reorder": 0, "loss": 0, "dark_ion": 0}, []
    for _ in range(2000):
        events = sample_collisions(rng, col, rates, 1.0)
        n_events += len(events)
        for e in events:
            outcomes[e.outcome] += 1
            times.append(e.time_s)
            assert e.ion in rates
        assert all(a.time_s <= b.time_s for a, b in zip(events, events[1:]))
    assert n_events / 2000 == pytest.approx(4.0, rel=0.05)
    assert outcomes["heating_kick"] / n_events == pytest.approx(0.6, abs=0.03)
    assert np.mean(times) == pytest.approx(0.5, abs=0.02)
    assert sample_collisions(rng, col, {0: 0.0}, 1.0) == ()
    assert math.isfinite(collision_rate_per_ion(col, YB_KG))


def test_the_kick_energy_scale_is_the_neutrals_thermal_energy_times_the_mass_ratio() -> None:
    """<E> = k_B T (m_gas/m_ion) x multiplier, with the partial-pressure-weighted gas mass."""
    col = _col()
    expect = K_B_J_PER_K * 300.0 * (GAS_MASS_U["H2"] * ATOMIC_MASS_KG / YB_KG)
    assert mean_kick_energy_j(col, YB_KG) == pytest.approx(expect, rel=1e-12)
    assert mean_kick_energy_j(_col(gas={"Ar": 1.0}), YB_KG) / mean_kick_energy_j(col, YB_KG) == pytest.approx(
        GAS_MASS_U["Ar"] / GAS_MASS_U["H2"], rel=1e-12
    )
    mixed = _col(gas={"H2": 0.5, "Ar": 0.5})
    assert mean_kick_energy_j(mixed, YB_KG) == pytest.approx(
        0.5 * (mean_kick_energy_j(col, YB_KG) + mean_kick_energy_j(_col(gas={"Ar": 1.0}), YB_KG)), rel=1e-12
    )
    assert mean_kick_energy_j(_col(temperature_k=4.0), YB_KG) / expect == pytest.approx(
        4.0 / 300.0, rel=1e-12
    )
    assert mean_kick_energy_j(_col(kick_scale_multiplier=2.5), YB_KG) == pytest.approx(
        2.5 * expect, rel=1e-12
    )
    with pytest.raises(ValueError):
        _col(kick_scale_multiplier=0.0)
    with pytest.raises(ValueError):
        mean_kick_energy_j(col, 0.0)


def test_the_kick_in_quanta() -> None:
    """<delta nbar> = <E>/(hbar omega_m): H2 at 300 K on 171Yb+ gives 7.4e4 quanta of a 1 MHz mode, above Section 6.7's
    parenthetical "tens to thousands"; a 4 K chamber lands inside it."""
    col = _col()
    q_1mhz = mean_kick_quanta(col, YB_KG, 2.0 * math.pi * 1e6)
    assert q_1mhz == pytest.approx(7.36e4, rel=2e-2), q_1mhz
    assert mean_kick_quanta(col, YB_KG, 2.0 * math.pi * 3e6) == pytest.approx(q_1mhz / 3.0, rel=1e-12)
    cryo = mean_kick_quanta(_col(temperature_k=4.0), YB_KG, 2.0 * math.pi * 1e6)
    assert 100.0 < cryo < 1e4, cryo
    with pytest.raises(ValueError):
        mean_kick_quanta(col, YB_KG, 0.0)
    assert mean_kick_quanta(_col(gas={}), YB_KG, 2.0 * math.pi * 1e6) == 0.0


@pytest.mark.parametrize("shape", ["exponential", "thermal_maxwell"])
def test_the_two_kick_distributions_have_the_stated_mean_and_the_right_spread(shape: str) -> None:
    """Both shapes carry the same mean and differ in their tail: an exponential has relative spread 1, the Maxwell chi^2_3
    sqrt(2/3)."""
    col = _col(kick_distribution=shape)
    w = 2.0 * math.pi * 1e6
    mean = mean_kick_quanta(col, YB_KG, w)
    rng = np.random.default_rng(4)
    draws = np.array([sample_kick_quanta(rng, col, YB_KG, w) for _ in range(20000)])
    assert float(np.mean(draws)) == pytest.approx(mean, rel=0.03)
    assert np.all(draws >= 0.0)
    rel_sd = float(np.std(draws)) / mean
    assert rel_sd == pytest.approx(1.0 if shape == "exponential" else math.sqrt(2.0 / 3.0), rel=0.05)
    assert sample_kick_quanta(rng, _col(gas={}), YB_KG, w) == 0.0
    a = np.array([sample_kick_quanta(np.random.default_rng(9), col, YB_KG, w) for _ in range(3)])
    b = np.array([sample_kick_quanta(np.random.default_rng(9), col, YB_KG, w) for _ in range(3)])
    assert np.array_equal(a, b)


def test_the_reorder_draws_from_the_configured_permutation_distribution() -> None:
    """Configured permutations are drawn uniformly and applied as new[k] = order[perm[k]]; without one the adjacent
    transposition at the struck ion is used."""
    rng = np.random.default_rng(1)
    three = _col(reorder_permutations=((1, 0, 2), (0, 2, 1), (2, 1, 0)))
    seen = {sample_reorder(rng, three, (0, 1, 2), 0) for _ in range(400)}
    assert seen == {(1, 0, 2), (0, 2, 1), (2, 1, 0)}, seen
    # applied to an already-permuted order, the permutation composes rather than resetting
    assert sample_reorder(np.random.default_rng(0), _col(reorder_permutations=((1, 0),)), (1, 0), 0) == (0, 1)
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
