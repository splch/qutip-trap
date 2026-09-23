"""Excess micromotion: displacement, modulation index, carrier and sideband weights (Berkeland et al. 1998).

Amplitudes are peak and signed: under the Mathieu origin a - 2q cos 2xi the in-phase micromotion at the rf phase origin is
a contraction, x_mu(t) = -(q/2) x_sec(t) cos(omega_rf t). The laser phase is modulated as beta cos(Omega t + delta), so a
drive splits into a comb weighted by J_n(beta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import jv

from qutip_trap.units import C_M_PER_S, E_C


@dataclass(frozen=True)
class MicromotionIndex:
    """Residual modulation index beta = delta_k . u_1 for the full wavevector of a drive: ``in_phase`` the signed
    stray-field part, nullable by shims; ``out_of_phase`` Berkeland's rf-quadrature phi_ac term, not nullable."""

    in_phase: float
    out_of_phase: float
    convention: Literal["peak", "rms"]

    def __post_init__(self) -> None:
        if self.convention not in ("peak", "rms"):
            raise ValueError("convention is 'peak' or 'rms'")

    @property
    def total(self) -> float:
        """sqrt(in_phase^2 + out_of_phase^2): the two terms are in quadrature, so the total index adds in quadrature."""
        return float(np.hypot(self.in_phase, self.out_of_phase))

    def as_peak(self) -> MicromotionIndex:
        if self.convention == "peak":
            return self
        return MicromotionIndex(self.in_phase * np.sqrt(2.0), self.out_of_phase * np.sqrt(2.0), "peak")

    def as_modulation(self) -> tuple[float, float]:
        """(beta, phase offset) such that beta cos(Omega t + delta + offset) = in_phase cos(Omega t + delta) + out_of_phase sin(Omega t + delta).

        Adding the offset to the rf phase keeps both quadratures and the sign of ``in_phase`` in the modulated coefficient.
        """
        return self.total, -float(np.arctan2(self.out_of_phase, self.in_phase))


def displacement_m(
    field_v_per_m: np.ndarray | float, mass_kg: float, omega_rad_s: np.ndarray | float, *, charge: int = 1
) -> np.ndarray:
    """u_0 = Q E/(m omega^2) per principal axis (Berkeland Eq. 16)."""
    e = np.atleast_1d(np.asarray(field_v_per_m, dtype=float))
    w = np.atleast_1d(np.asarray(omega_rad_s, dtype=float))
    if np.any(w <= 0.0):
        raise ValueError("secular frequencies must be positive")
    return np.asarray(charge * E_C * e / (mass_kg * w * w))


def excess_amplitude_m(displacement_m_: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    """Signed peak excess-micromotion amplitude u_1 = -(1/2) q u_0 per axis (Berkeland Eq. 17 with the module's sign); ``q``
    keeps its sign, so the x/y antiphase of a linear trap survives."""
    return np.asarray(
        -0.5
        * np.atleast_1d(np.asarray(displacement_m_, dtype=float))
        * np.atleast_1d(np.asarray(q, dtype=float))
    )


def out_of_phase_amplitude_m(q_x: float, r_m: float, alpha: float, phi_ac_rad: float) -> float:
    """(1/4) q_x R alpha phi_ac: the irreducible quadrature micromotion from an rf phase difference (Berkeland Eq. 18)."""
    return 0.25 * abs(q_x) * r_m * alpha * phi_ac_rad


def modulation_index(delta_k_rad_per_m: np.ndarray, amplitude_m: np.ndarray) -> float:
    """Signed beta = delta_k . u_1 for the full wavevector (single-photon k or Raman Delta k). The sign does not reach the
    carrier (J_0 is even) but does reach the first micromotion sideband."""
    return float(np.dot(np.asarray(delta_k_rad_per_m, dtype=float), np.asarray(amplitude_m, dtype=float)))


def carrier_factor(beta: float) -> float:
    """J_0(beta)^2: the carrier Rabi-frequency suppression (squared amplitude) under phase modulation of index beta."""
    return float(jv(0, beta)) ** 2


def carrier_amplitude(beta: float) -> float:
    """J_0(beta), signed: the carrier amplitude inverts past the first zero at beta = 2.4048."""
    return float(jv(0, beta))


def sideband_ratio(beta: float) -> float:
    """J_1(beta)^2/J_0(beta)^2 -> (beta/2)^2 for beta << 1: the first micromotion sideband over the carrier."""
    j0 = float(jv(0, beta))
    if j0 == 0.0:
        raise ZeroDivisionError("the carrier vanishes at this modulation index")
    return float(jv(1, beta)) ** 2 / j0**2


def sideband_weights(beta: float, n_max: int = 5) -> dict[int, float]:
    """J_n(beta)^2 for |n| <= n_max: the comb of micromotion sidebands on a drive."""
    return {n: float(jv(n, beta)) ** 2 for n in range(-n_max, n_max + 1)}


def second_order_doppler_fraction(amplitude_m: np.ndarray | float, omega_rf_rad_s: float) -> float:
    """<Delta nu/nu> = -<v^2>/(2 c^2) with <v^2> = (|u| Omega)^2/2 for micromotion of peak amplitude u (Berkeland Eq. 39)."""
    u = np.atleast_1d(np.asarray(amplitude_m, dtype=float))
    v2 = float(np.dot(u, u)) * omega_rf_rad_s**2 / 2.0
    return -v2 / (2.0 * C_M_PER_S**2)
