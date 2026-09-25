"""Excess micromotion (Berkeland et al. 1998; PLAN.md Section 4.1.1).

A residual field E displaces the ion to u_0 = Q E/(m omega^2) and drives in-phase micromotion u_1 = -(1/2) Q u_0 under the
adopted Mathieu origin a - 2q cos 2xi (a contraction at the rf phase origin); an rf phase difference phi_ac between the
electrodes adds the quadrature term (1/4) q_x R alpha phi_ac that no dc shim nulls. A drive of wavevector delta_k sees the
phase modulation beta cos(Omega t + delta) with beta = delta_k . u_1, so its carrier carries J_0(beta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap.units import C_M_PER_S


@dataclass(frozen=True)
class MicromotionIndex:
    """The PEAK modulation index of one drive: the SIGNED in-phase part (the stray field, nullable by shims; it flips
    sign as a shim crosses compensation) and the out-of-phase part (Berkeland's phi_ac term, not nullable)."""

    in_phase: float
    out_of_phase: float

    @property
    def total(self) -> float:
        """sqrt(in_phase^2 + out_of_phase^2): the two terms are in quadrature."""
        return float(np.hypot(self.in_phase, self.out_of_phase))

    def as_modulation(self) -> tuple[float, float]:
        """(beta, offset) with beta cos(theta + offset) = in_phase cos(theta) + out_of_phase sin(theta) for every theta.

        The builder adds the offset to the rf phase, so e^{i beta cos(Omega_rf t + delta)} carries both quadratures and the
        pi step of a negative in-phase index at compensation.
        """
        return self.total, -float(np.arctan2(self.out_of_phase, self.in_phase))


def out_of_phase_amplitude_m(q_x: float, r_m: float, alpha: float, phi_ac_rad: float) -> float:
    """(1/4) q_x R alpha phi_ac: the irreducible quadrature micromotion from an rf phase difference (Berkeland Eq. 18)."""
    return 0.25 * abs(q_x) * r_m * alpha * phi_ac_rad


def modulation_index(delta_k_rad_per_m: np.ndarray, amplitude_m: np.ndarray) -> float:
    """SIGNED beta = delta_k . u_1 for the full wavevector (single photon k or Raman Delta k)."""
    return float(np.dot(np.asarray(delta_k_rad_per_m, dtype=float), np.asarray(amplitude_m, dtype=float)))


def second_order_doppler_fraction(amplitude_m: np.ndarray | float, omega_rf_rad_s: float) -> float:
    """<Delta nu/nu> = -<v^2>/(2 c^2) with <v^2> = (|u| Omega)^2/2 for micromotion of peak amplitude u (Berkeland Eq. 39)."""
    u = np.atleast_1d(np.asarray(amplitude_m, dtype=float))
    v2 = float(np.dot(u, u)) * omega_rf_rad_s**2 / 2.0
    return -v2 / (2.0 * C_M_PER_S**2)
