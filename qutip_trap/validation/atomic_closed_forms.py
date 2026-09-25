"""Closed forms of Ozeri et al. 2007 and Wineland et al. 2003 that the atomic layer's explicit sums must reproduce
(PLAN.md Section 4.5).

``gamma`` is the ANGULAR P-level decay rate (one for both fine-structure levels in LS coupling), ``omega_f`` the angular
fine-structure splitting, ``delta`` the detuning omega_L - omega(P1/2) from the lower ground state (positive blue), and
Ozeri's ``g`` the HALF-convention stretched-state coupling E |<P3/2 stretched|d . sigma+|S1/2 stretched>|/(2 hbar). Ozeri's
Omega_R is a half-convention two-photon Rabi frequency (tau_pi = pi/(2 Omega_R)), twice which is the plan's Omega_{g1 g2}.
"""

from __future__ import annotations

import math


def ozeri_raman_rabi_half(
    g_b: float,
    g_r: float,
    b_minus: float,
    b_plus: float,
    r_minus: float,
    r_plus: float,
    delta: float,
    omega_f: float,
) -> float:
    """Omega_R = (g_b g_r/3)(b_- r_- - b_+ r_+) omega_f/[Delta(Delta - omega_f)] (half convention)."""
    return (g_b * g_r / 3.0) * (b_minus * r_minus - b_plus * r_plus) * omega_f / (delta * (delta - omega_f))


def ozeri_gamma_total(
    gamma: float, g_b: float, g_r: float, b_sq: float, r_sq: float, delta: float, omega_f: float
) -> float:
    """Gamma_total = (gamma/3)[g_b^2 (b_-^2 + b_+^2) + g_r^2 (r_-^2 + r_+^2)][1/Delta^2 + 2/(Delta - omega_f)^2] (Eq. 14)."""
    return (gamma / 3.0) * (g_b**2 * b_sq + g_r**2 * r_sq) * (1.0 / delta**2 + 2.0 / (delta - omega_f) ** 2)


def ozeri_gamma_raman(
    gamma: float, g_b: float, g_r: float, b_sq: float, r_sq: float, delta: float, omega_f: float
) -> float:
    """Gamma_Raman = (2 gamma/9)[same bracket][omega_f/(Delta(Delta - omega_f))]^2 (Eq. 15)."""
    return (
        (2.0 * gamma / 9.0) * (g_b**2 * b_sq + g_r**2 * r_sq) * (omega_f / (delta * (delta - omega_f))) ** 2
    )


def ozeri_p_total(gamma: float, omega_f: float, delta: float) -> float:
    """P_total = (pi gamma/omega_f)(2 Delta^2 + (Delta - omega_f)^2)/|Delta(Delta - omega_f)|, minimum 2 sqrt2 pi gamma/omega_f."""
    return (
        (math.pi * gamma / omega_f)
        * (2.0 * delta**2 + (delta - omega_f) ** 2)
        / abs(delta * (delta - omega_f))
    )


def wineland_p_se_clock(gamma: float, omega_f: float) -> float:
    """P_SE = 2 sqrt2 pi gamma/omega_F for the m_F = 0 clock line at the optimum detuning (8.885766 gamma/omega_F)."""
    return 2.0 * math.sqrt(2.0) * math.pi * gamma / omega_f


def wineland_clock_light_shift(g_b: float, g_r: float, omega_0: float, delta: float, omega_f: float) -> float:
    """delta_{0<->0} = -(g_b^2 + g_r^2)(omega_0/3)[1/Delta^2 + 2/(Delta - omega_F)^2] (Eq. 2.17), the differential shift of
    the clock transition (upper minus lower clock state), first order in omega_0/Delta."""
    return -(g_b**2 + g_r**2) * (omega_0 / 3.0) * (1.0 / delta**2 + 2.0 / (delta - omega_f) ** 2)
