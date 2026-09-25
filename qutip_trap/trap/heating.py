"""Electric-field noise to motional heating rates (PLAN.md Section 4.1.5).

S_E is SINGLE-sided, S_E(omega) = 2 int dtau <dE(tau) dE(0)> e^{-i omega tau} in (V/m)^2/Hz, and a single ion heats at
Gamma_h = e^2 S_E(omega)/(4 m hbar omega) (Brownnutt 2015 Eqs. 11-12); ``NoiseSpectrum`` is two-sided, S^(1) = 2 S^(2). A
quoted heating rate is d<n>/dt at n = 0, and the master equation carries Gamma(N + 1) on a and Gamma N on a^dag. Mode k of
a crystal heats at e^2/(4 hbar omega_k) sum_ij c_i c_j g_ij S_E(omega_k)/sqrt(m_i m_j) with the mass-weighted eigenvector
c and the spatial correlation g_ij = exp(-r_ij/l_c) (Brownnutt Eqs. 19-23; Kielpinski Eq. 20).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.trap.crystal import Crystal
from qutip_trap.units import E_C, HBAR_J_S

if TYPE_CHECKING:
    from qutip_trap.noise.spectra import NoiseSpectrum

SpectralDensity = Callable[[float], float]
"""A single-sided S_E in (V/m)^2/Hz as a function of ANGULAR frequency (rad/s)."""


def heating_rate_quanta_per_s(s_e_single_sided_v2_m2_hz: float, mass_kg: float, omega_rad_s: float) -> float:
    """n_dot = e^2 S_E(omega)/(4 m hbar omega) for a single ion; ``omega_rad_s`` is angular."""
    if omega_rad_s <= 0.0 or mass_kg <= 0.0:
        raise ValueError("mass and angular frequency must be positive")
    if s_e_single_sided_v2_m2_hz < 0.0:
        raise ValueError("a spectral density is non-negative")
    return E_C * E_C * s_e_single_sided_v2_m2_hz / (4.0 * mass_kg * HBAR_J_S * omega_rad_s)


def s_e_from_heating_rate(n_dot_quanta_per_s: float, mass_kg: float, omega_rad_s: float) -> float:
    """The inverse: the single-sided S_E = 4 m hbar omega n_dot/e^2 a measured rate implies."""
    return 4.0 * mass_kg * HBAR_J_S * omega_rad_s * n_dot_quanta_per_s / (E_C * E_C)


def single_sided_from_spectrum(spectrum: NoiseSpectrum) -> SpectralDensity:
    """S_E(omega) = 2 S^(2)(|omega|) through the spectrum's own ``value``: the tabulated band folded onto |omega| and zero
    above it, plus the white level everywhere (so a caller must not add the white level again)."""

    def s_e(omega: float) -> float:
        return 2.0 * float(spectrum.value(abs(omega)))

    return s_e


def thermal_collapse_rates(
    heating_rate_quanta_per_s: float, n_bar_bath: float | None = None
) -> tuple[float, float]:
    """(rate on a, rate on a^dag) = (Gamma (N + 1), Gamma N) with Gamma N the quoted heating rate.

    ``None`` is the N -> infinity limit of electric-field noise, both rates Gamma_h.
    """
    if heating_rate_quanta_per_s < 0.0:
        raise ValueError("heating rate is non-negative")
    if n_bar_bath is None:
        return heating_rate_quanta_per_s, heating_rate_quanta_per_s
    if n_bar_bath <= 0.0:
        raise ValueError("the bath occupation must be positive (or None for the high-temperature limit)")
    gamma = heating_rate_quanta_per_s / n_bar_bath
    return gamma * (n_bar_bath + 1.0), gamma * n_bar_bath


def correlation_matrix(positions_m: np.ndarray, correlation_length_m: float) -> np.ndarray:
    """g_ij = exp(-|r_i - r_j|/l_c): all ones for l_c = inf (uniform), the identity for l_c = 0 (uncorrelated)."""
    pos = np.asarray(positions_m, dtype=float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    if correlation_length_m < 0.0:
        raise ValueError("the correlation length is non-negative (0 uncorrelated, inf uniform)")
    if math.isinf(correlation_length_m):
        return np.ones_like(d)
    if correlation_length_m == 0.0:
        return np.eye(len(pos))
    return np.asarray(np.exp(-d / correlation_length_m))


def heating_rates_per_mode(
    crystal: Crystal, s_e_single_sided: SpectralDensity | float, correlation_length_m: float
) -> np.ndarray:
    """n_dot_k in quanta/s per mode for an isotropic single-sided S_E (a constant or a function of angular frequency).

    ``correlation_length_m`` is required: inf heats only the centre-of-mass modes of an equal-mass chain, at N times the
    single-ion rate; 0 heats every mode at the single-ion rate.
    """
    masses = crystal.masses_kg
    g = correlation_matrix(crystal.positions_m, correlation_length_m)
    out = []
    for mode in crystal.modes:
        pattern = mode.displacement_pattern() / np.sqrt(masses)[:, None]  # c_{ia}/sqrt(m_i), shape (N, 3)
        w = mode.omega_rad_s
        s_e = s_e_single_sided(w) if callable(s_e_single_sided) else float(s_e_single_sided)
        coupling = float(np.einsum("ia,ij,ja->", pattern, g, pattern))
        out.append(E_C * E_C * s_e * coupling / (4.0 * HBAR_J_S * w))
    return np.asarray(out)
