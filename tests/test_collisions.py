"""Background-gas collisions as a discrete event process (PLAN.md Section 6.7, Section 6.1 route (e); ``check_collisions.py``; M7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.noise.collisions import (
    GAS_MASS_U,
    POLARIZABILITY_VOLUME_M3,
    collision_rate_per_ion,
    langevin_rate_coefficient_m3_s,
    number_density_per_m3,
    sample_collisions,
)
from qutip_trap.noise.spectra import Collisions
from qutip_trap.units import ATOMIC_MASS_KG

TORR_PA = 133.32236842105263


def test_langevin_rate_reproduces_the_check_script_numbers() -> None:
    """H2 at 1e-11 torr and 300 K: k_L = 1.63e-9 / 1.51e-9 / 1.48e-9 cm^3/s and Gamma_L = 5.25e-4 / 4.87e-4 / 4.78e-4 s^-1 per ion for
    9, 40 and 171 u (check_collisions.py), one event per 32 to 35 minutes per ion."""
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
    assert 1.0 / collision_rate_per_ion(col, 171.0 * ATOMIC_MASS_KG) / 60.0 == pytest.approx(35.0, abs=1.0)
    mixed = Collisions(1e-11 * TORR_PA, {"H2": 0.5, "N2": 0.5}, {"heating_kick": 0.5, "loss": 0.5})
    r_mixed = collision_rate_per_ion(mixed, 171.0 * ATOMIC_MASS_KG)
    assert r_mixed < collision_rate_per_ion(col, 171.0 * ATOMIC_MASS_KG), (
        "N2 is heavier and only twice as polarizable"
    )
    with pytest.raises(ValueError):
        Collisions(1.0, {"H2": 0.7}, {"heating_kick": 1.0})
    with pytest.raises(KeyError):
        collision_rate_per_ion(Collisions(1.0, {"Xe": 1.0}, {"loss": 1.0}), 171.0 * ATOMIC_MASS_KG)


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
    assert math.isfinite(collision_rate_per_ion(col, 171.0 * ATOMIC_MASS_KG))
