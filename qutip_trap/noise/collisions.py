"""Background-gas collisions as a discrete event process (PLAN.md Section 6.7, Section 6.1 route (e); M7).

Langevin capture (the ion-induced-dipole collision that dominates for a neutral of polarizability alpha_p): the rate
coefficient k_L = (e/(2 eps_0)) sqrt(alpha_SI/mu_r) with alpha_SI = 4 pi eps_0 alpha_vol (alpha_vol the polarizability
volume) and mu_r the reduced mass, independent of the collision energy; the rate per ion is Gamma_L = sum_s n_s k_L,s with
n_s = f_s p/(k_B T) (``check_collisions.py``: H2 at 1e-11 torr and 300 K on 171Yb+ gives k_L = 1.48e-9 cm^3/s,
n = 3.22e5 cm^-3, Gamma_L = 4.78e-4 s^-1 per ion, one event per 35 minutes per ion). The polarizability volumes are
textbook values (CRC Handbook) **[background]**; the outcome distribution conditional on a collision (heating kick,
reorder, loss, dark ion) is a device input, because no source quantifies it (Section 6.7).

Per shot the event count is Poisson with mean N Gamma_L T_shot; the event time is uniform in the shot; the run
(``run/job.py``) turns the outcome into a herald, a discarded shot, a permuted ion order or a dark/lost flag that
persists for every later shot (Appendix E ``RunState``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qutip_trap.noise.spectra import Collisions
from qutip_trap.units import ATOMIC_MASS_KG, E_C, EPSILON_0_F_PER_M, K_B_J_PER_K

Outcome = Literal["heating_kick", "reorder", "loss", "dark_ion"]

POLARIZABILITY_VOLUME_M3: dict[str, float] = {
    "H2": 0.80e-30,
    "He": 0.205e-30,
    "N2": 1.74e-30,
    "O2": 1.58e-30,
    "CO": 1.95e-30,
    "CO2": 2.91e-30,
    "H2O": 1.45e-30,
    "Ar": 1.64e-30,
    "CH4": 2.59e-30,
}
"""Static polarizability volumes (m^3 = 1e-30 x A^3), CRC Handbook of Chemistry and Physics [background]."""

GAS_MASS_U: dict[str, float] = {
    "H2": 2.016,
    "He": 4.0026,
    "N2": 28.014,
    "O2": 31.998,
    "CO": 28.010,
    "CO2": 44.010,
    "H2O": 18.015,
    "Ar": 39.948,
    "CH4": 16.043,
}


def langevin_rate_coefficient_m3_s(alpha_volume_m3: float, ion_mass_kg: float, gas_mass_kg: float) -> float:
    """k_L = (e/(2 eps_0)) sqrt(4 pi eps_0 alpha_vol/mu_r), m^3/s (Section 6.7)."""
    if alpha_volume_m3 <= 0.0 or ion_mass_kg <= 0.0 or gas_mass_kg <= 0.0:
        raise ValueError("polarizability volume and masses are positive")
    mu = ion_mass_kg * gas_mass_kg / (ion_mass_kg + gas_mass_kg)
    alpha_si = 4.0 * math.pi * EPSILON_0_F_PER_M * alpha_volume_m3
    return E_C / (2.0 * EPSILON_0_F_PER_M) * math.sqrt(alpha_si / mu)


def number_density_per_m3(pressure_pa: float, temperature_k: float) -> float:
    return pressure_pa / (K_B_J_PER_K * temperature_k)


def collision_rate_per_ion(collisions: Collisions, ion_mass_kg: float) -> float:
    """Gamma_L = sum_s f_s n k_L,s per ion in s^-1 for the configured gas mixture and pressure."""
    n = number_density_per_m3(collisions.pressure_pa, collisions.temperature_k)
    total = 0.0
    for gas, fraction in collisions.gas.items():
        if gas not in POLARIZABILITY_VOLUME_M3:
            raise KeyError(
                f"no polarizability tabulated for {gas!r}; known gases: {sorted(POLARIZABILITY_VOLUME_M3)}"
            )
        k_l = langevin_rate_coefficient_m3_s(
            POLARIZABILITY_VOLUME_M3[gas], ion_mass_kg, GAS_MASS_U[gas] * ATOMIC_MASS_KG
        )
        total += fraction * n * k_l
    return total


@dataclass(frozen=True)
class CollisionEvent:
    time_s: float
    """Time within the shot (0 = the start of the preparation)."""
    ion: int
    outcome: Outcome


def sample_collisions(
    rng: np.random.Generator, collisions: Collisions, rates_per_ion: dict[int, float], duration_s: float
) -> tuple[CollisionEvent, ...]:
    """The collision events of one shot of length ``duration_s``: Poisson per ion at its rate, uniform times, outcomes from
    the configured conditional distribution; sorted in time."""
    outcomes = list(collisions.outcome_probabilities)
    probs = np.array([collisions.outcome_probabilities[o] for o in outcomes], dtype=float)
    events: list[CollisionEvent] = []
    for ion, rate in rates_per_ion.items():
        n = int(rng.poisson(rate * duration_s)) if rate > 0.0 else 0
        for _ in range(n):
            t = float(rng.uniform(0.0, duration_s))
            out = (
                outcomes[int(rng.choice(len(outcomes), p=probs / probs.sum()))]
                if outcomes
                else "heating_kick"
            )
            events.append(CollisionEvent(t, int(ion), out))
    events.sort(key=lambda e: e.time_s)
    return tuple(events)


__all__ = [
    "GAS_MASS_U",
    "POLARIZABILITY_VOLUME_M3",
    "CollisionEvent",
    "Outcome",
    "collision_rate_per_ion",
    "langevin_rate_coefficient_m3_s",
    "number_density_per_m3",
    "sample_collisions",
]
