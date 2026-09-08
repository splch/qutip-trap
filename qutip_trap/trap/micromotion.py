"""Excess micromotion: displacement, modulation index, carrier and sideband weights (PLAN.md Section 4.1.1; Appendix E; M1).

Berkeland et al. 1998 [verified]: a uniform stray field E_dc displaces the ion to u_0i = Q E_dc . u_i/(m omega_i^2) and
produces excess micromotion of amplitude (1/2) u_0i q_i along each principal axis; an rf phase difference phi_ac between
opposite electrodes adds the irreducible term (1/4) q_x R alpha phi_ac along x that no dc shim can null (in quadrature with
the in-phase term). In the ion frame the laser phase is modulated, k . u'(t) = beta cos(Omega t + delta) (Eq. 21's printed
stray factor Omega is a misprint), so the drive splits into a comb weighted by J_n(beta): the carrier is suppressed by
J_0(beta)^2 and the first micromotion sideband has strength ratio J_1^2/J_0^2 ~ (beta/2)^2 for beta << 1. Section 9.12:
171Yb+ at 1 MHz, q = 0.28, E_dc = 1, 10, 100 V/m give u_0 = 14.3, 143, 1429 nm, beta = 0.034, 0.340, 3.40 at 369.5 nm and
J_0 = 0.9997, 0.971, -0.365 (the carrier sign inverts past the first zero at beta = 2.4048). Under the adopted Mathieu sign
the in-phase micromotion at the rf phase origin is a contraction, x_mu(t) = -(q/2) x_sec(t) cos(omega_rf t) (Section 13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import jv

from qutip_trap.units import C_M_PER_S, E_C


@dataclass(frozen=True)
class MicromotionIndex:
    """Residual modulation index beta = delta_k . u_1 for the FULL wavevector of a drive.

    ``in_phase`` is the stray-field part (nullable by shims) and ``out_of_phase`` the rf-quadrature part
    (Berkeland's phi_ac term, not nullable); they never collapse into one number (Section 4.1.1). The
    peak/rms tag is declared, never inferred (Section 13). ``in_phase`` is SIGNED: under the adopted Mathieu
    origin it is -delta_k . (1/2) Q u_0, and its sign flips as a shim crosses the compensated value (Section 9.17).
    """

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

        beta = hypot(in_phase, out_of_phase) >= 0 and offset = -atan2(out_of_phase, in_phase); the drive builder adds the
        offset to the rf phase so that e^{i beta cos(Omega_rf t + delta)} carries BOTH quadratures, and the pi step of a
        negative in-phase index at compensation (Section 9.17) reaches the modulated coefficient.
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
    """SIGNED peak excess-micromotion amplitude u_1 = -(1/2) q u_0 per axis (Berkeland Eq. 17 under the adopted sign).

    The minus sign is the adopted Mathieu origin ``a - 2q cos 2xi`` (Section 13, row "Floquet function and rf phase
    origin"): the in-phase micromotion at the rf phase origin is a CONTRACTION, x_mu(t) = -(q_x/2) x_sec(t) cos(omega_rf t),
    and every micromotion-sideband phase follows from that one statement (Section 4.1.1). ``q``'s own sign is kept, so
    the x/y antiphase of a linear trap (q_y = -q_x) survives; the magnitude is Berkeland's (1/2)|q| u_0.
    """
    return np.asarray(
        -0.5
        * np.atleast_1d(np.asarray(displacement_m_, dtype=float))
        * np.atleast_1d(np.asarray(q, dtype=float))
    )


def out_of_phase_amplitude_m(q_x: float, r_m: float, alpha: float, phi_ac_rad: float) -> float:
    """(1/4) q_x R alpha phi_ac: the irreducible quadrature micromotion from an rf phase difference (Berkeland Eq. 18)."""
    return 0.25 * abs(q_x) * r_m * alpha * phi_ac_rad


def modulation_index(delta_k_rad_per_m: np.ndarray, amplitude_m: np.ndarray) -> float:
    """SIGNED beta = delta_k . u_1 for the FULL wavevector (single-photon k or Raman Delta k): scales with |delta_k|, not k_hat alone.

    Appendix E declares beta = delta_k . u_1, not its modulus: the sign is what the 9.17 row "Mathieu sign of the Floquet
    function" requires to survive, a phase step of pi in the rf-photon correlation signal as a shim crosses the
    compensated value. J_0 is even, so the carrier suppression does not see it; the first micromotion sideband does.
    """
    return float(np.dot(np.asarray(delta_k_rad_per_m, dtype=float), np.asarray(amplitude_m, dtype=float)))


def carrier_factor(beta: float) -> float:
    """J_0(beta)^2: the carrier Rabi-frequency suppression (squared amplitude) under phase modulation of index beta."""
    return float(jv(0, beta)) ** 2


def carrier_amplitude(beta: float) -> float:
    """J_0(beta), signed: -0.365 at beta = 3.40 (inverted past the first zero 2.4048)."""
    return float(jv(0, beta))


def sideband_ratio(beta: float) -> float:
    """J_1(beta)^2/J_0(beta)^2 -> (beta/2)^2 for beta << 1: the first micromotion sideband over the carrier."""
    j0 = float(jv(0, beta))
    if j0 == 0.0:
        raise ZeroDivisionError("the carrier vanishes at this modulation index")
    return float(jv(1, beta)) ** 2 / j0**2


def sideband_weights(beta: float, n_max: int = 5) -> dict[int, float]:
    """J_n(beta)^2 for |n| <= n_max: the comb of micromotion sidebands on every drive (gates, cooling, detection; Section 13)."""
    return {n: float(jv(n, beta)) ** 2 for n in range(-n_max, n_max + 1)}


def second_order_doppler_fraction(amplitude_m: np.ndarray | float, omega_rf_rad_s: float) -> float:
    """<Delta nu/nu> = -<v^2>/(2 c^2) with <v^2> = (|u| Omega)^2/2 for micromotion of peak amplitude u (Berkeland Eq. 39)."""
    u = np.atleast_1d(np.asarray(amplitude_m, dtype=float))
    v2 = float(np.dot(u, u)) * omega_rf_rad_s**2 / 2.0
    return -v2 / (2.0 * C_M_PER_S**2)


__all__ = [
    "MicromotionIndex",
    "carrier_amplitude",
    "carrier_factor",
    "displacement_m",
    "excess_amplitude_m",
    "modulation_index",
    "out_of_phase_amplitude_m",
    "second_order_doppler_fraction",
    "sideband_ratio",
    "sideband_weights",
]
