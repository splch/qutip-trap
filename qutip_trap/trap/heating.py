"""Electric-field noise spectra to heating rates (PLAN.md Section 4.1.5; Section 13; milestone M1).

Convention (Section 13, rows "Electric-field noise density" and "Heating-rate meaning"): S_E is
SINGLE-sided and Gamma_h = e^2 S_E(omega)/(4 m hbar omega) (Brownnutt Eqs. 11-12); a quoted heating rate is
d<n>/dt at n = 0, that is Gamma N_bar, and the master equation carries Gamma(N_bar + 1) on a and Gamma N_bar
on a^dagger. The filter-function stack of Section 6.9 uses TWO-sided spectra, so the conversion between the
two conventions is explicit at the boundary (Section 13, Run 5 amendment).
"""

from __future__ import annotations

from qutip_trap.units import E_C, HBAR_J_S

M1 = "milestone M1 (trap/heating.py, PLAN.md Section 4.1.5)"


def heating_rate_quanta_per_s(s_e_single_sided_v2_m2_hz: float, mass_kg: float, omega_rad_s: float) -> float:
    """n_dot = e^2 S_E(omega)/(4 m hbar omega) for a single ion (Brownnutt Eqs. 11-12; Section 9.15 round trip).

    ``omega_rad_s`` is ANGULAR. Section 9.15: S_E = 2.2250e-13 (V/m)^2/Hz for 9Be+ at omega_z/2pi = 3.6 MHz
    gives 40 quanta/s, the round-trip test.
    """
    if omega_rad_s <= 0.0 or mass_kg <= 0.0:
        raise ValueError("mass and angular frequency must be positive")
    return E_C * E_C * s_e_single_sided_v2_m2_hz / (4.0 * mass_kg * HBAR_J_S * omega_rad_s)


def heating_rates_per_mode(*args: object, **kwargs: object) -> object:
    """Multi-ion generalization with the mass-weighted participation and the parity rule (Sections 4.1.5, 9.13)."""
    raise NotImplementedError(f"heating_rates_per_mode is {M1}")


__all__ = ["heating_rate_quanta_per_s", "heating_rates_per_mode"]
