"""Closed forms of Ozeri et al. 2007, Wineland et al. 2003 and Uys et al. 2010 that the atomic layer's sums must reproduce.
``gamma`` is the ANGULAR P-level decay rate, ``omega_f`` the angular fine-structure splitting and ``delta`` the detuning
omega_L - omega(P1/2) from the lower ground state (positive blue). Ozeri's half-convention stretched-state coupling
g = E |<P3/2|d . sigma+|S1/2>|/(2 hbar) is half the single-photon Rabi frequency, and his Omega_R (tau_pi = pi/(2 Omega_R))
half this package's two-photon Omega_{g1 g2}."""

from __future__ import annotations

import math

import numpy as np

from qutip_trap.units import C_M_PER_S, HBAR_J_S

SQRT2 = math.sqrt(2.0)


# ---- Ozeri et al. 2007 (Eqs. 4-6, 12-18) ----------------------------------------------------------------


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


def ozeri_p_raman(gamma: float, omega_f: float, delta: float) -> float:
    """P_Raman = (2 pi gamma/3) omega_f/|Delta(Delta - omega_f)|, interior minimum 8 pi gamma/(3 omega_f) at Delta = omega_f/2."""
    return (2.0 * math.pi * gamma / 3.0) * omega_f / abs(delta * (delta - omega_f))


def ozeri_p_rayleigh(gamma: float, omega_f: float, delta: float) -> float:
    """P_Rayleigh = (pi gamma/omega_f)(3 Delta^2 - 2 Delta omega_f + omega_f^2/3)/|Delta(Delta - omega_f)| (Eq. 16)."""
    return (
        (math.pi * gamma / omega_f)
        * (3.0 * delta**2 - 2.0 * delta * omega_f + omega_f**2 / 3.0)
        / abs(delta * (delta - omega_f))
    )


def ozeri_p_total_optimum_delta(omega_f: float) -> tuple[float, float]:
    """The two equally deep minima of P_total: Delta = (sqrt2 - 1) omega_f and -(sqrt2 + 1) omega_f."""
    return (SQRT2 - 1.0) * omega_f, -(SQRT2 + 1.0) * omega_f


def ozeri_epsilon_s_from_power(omega_r_half: float, omega_32: float, w0_m: float, power_w: float) -> float:
    """Ozeri 2007 Eq. 17: eps_S = 2 pi |Omega_R| hbar omega_{3/2}^3 w_0^2/(3 c^2 P), angular frequencies in rad/s."""
    return (
        2.0 * math.pi * abs(omega_r_half) * HBAR_J_S * omega_32**3 * w0_m**2 / (3.0 * C_M_PER_S**2 * power_w)
    )


def ozeri_gamma_over_g_squared(omega: float, e0_v_per_m: float) -> float:
    """gamma/g^2 = 4 hbar omega^3/(3 pi eps0 c^3 E^2), the atomic-constant ratio that removes the dipole element."""
    from qutip_trap.units import EPSILON_0_F_PER_M

    return 4.0 * HBAR_J_S * omega**3 / (3.0 * math.pi * EPSILON_0_F_PER_M * C_M_PER_S**3 * e0_v_per_m**2)


# ---- Wineland et al. 2003 (Eqs. 2.13-2.18, Table 1) --------------------------------------------------------


def wineland_ratio_function(x: float) -> float:
    """|x(x - 1)|[1/x^2 + 2/(x - 1)^2] with x = Delta/omega_F: the R_SE/|Omega| ratio to minimize; minimum 2 sqrt2 at sqrt2 - 1."""
    return abs(x * (x - 1.0)) * (1.0 / x**2 + 2.0 / (x - 1.0) ** 2)


def wineland_bracket(x: float) -> float:
    """1/x^2 + 2/(x - 1)^2 alone: minimizing THIS returns the wrong x = 1/(1 + 2^(1/3)) (negative control)."""
    return 1.0 / x**2 + 2.0 / (x - 1.0) ** 2


def wineland_p_se_clock(gamma: float, omega_f: float) -> float:
    """P_SE = 2 sqrt2 pi gamma/omega_F for the m_F = 0 clock line at the optimum (8.885766 gamma/omega_F)."""
    return 2.0 * SQRT2 * math.pi * gamma / omega_f


def wineland_p_se_zeeman_22_11(gamma: float, omega_f: float) -> float:
    """P_SE = (8 pi/sqrt6) gamma/omega_F for the 9Be+ |2,2> <-> |1,1> line (10.260399 gamma/omega_F)."""
    return 8.0 * math.pi / math.sqrt(6.0) * gamma / omega_f


def wineland_delta_over_omega_clock(omega_0: float, omega_f: float) -> float:
    """|delta_{0<->0}/Omega_{0<->0}| = 4 sqrt2 omega_0/omega_F at the optimum."""
    return 4.0 * SQRT2 * omega_0 / omega_f


def wineland_clock_light_shift(g_b: float, g_r: float, omega_0: float, delta: float, omega_f: float) -> float:
    """delta_{0<->0} = -(g_b^2 + g_r^2)(omega_0/3)[1/Delta^2 + 2/(Delta - omega_F)^2] (Eq. 2.17), the differential
    shift of the clock transition (upper minus lower clock state), first order in omega_0/Delta."""
    return -(g_b**2 + g_r**2) * (omega_0 / 3.0) * (1.0 / delta**2 + 2.0 / (delta - omega_f) ** 2)


def wineland_r_se_clock(gamma: float, g_b: float, g_r: float, delta: float, omega_f: float) -> float:
    """R_SE = gamma (g_b^2 + g_r^2)/3 [1/Delta^2 + 2/(Delta - omega_F)^2] (Eq. 2.18), so delta_{0<->0} = -(omega_0/gamma) R_SE."""
    return gamma * (g_b**2 + g_r**2) / 3.0 * (1.0 / delta**2 + 2.0 / (delta - omega_f) ** 2)


OZERI_PHOTONS_PER_RADIAN_COEFFICIENT = 0.9579
"""The saturation coefficient of Gamma_total/Delta_St for a far-detuned clock-qubit Raman drive (Ozeri 2005), transcribed
rather than derived; Wineland's Eqs. 2.17-2.18 give exactly 1 (``wineland_photons_per_stark_radian``)."""


def ozeri_photons_per_stark_radian(
    gamma: float, delta_hf: float, coefficient: float = OZERI_PHOTONS_PER_RADIAN_COEFFICIENT
) -> float:
    """Gamma_total/Delta_St -> C gamma/Delta_hf (Ozeri 2005): scattered photons per radian of Stark phase, saturated in
    the detuning; ``gamma`` and ``delta_hf`` in the same units."""
    if delta_hf == 0.0:
        raise ZeroDivisionError("Delta_hf is the ground-state hyperfine splitting and is nonzero")
    return coefficient * gamma / delta_hf


def wineland_photons_per_stark_radian(gamma: float, omega_0: float) -> float:
    """R_SE/|delta_{0<->0}| = gamma/omega_0 exactly, for any Delta and polarization (Wineland 2003 Eqs. 2.17-2.18 share
    prefactor and bracket), so the clock-qubit shift cannot be nulled by polarization."""
    return gamma / omega_0


# ---- Uys et al. 2010 (Eqs. 4-8) ------------------------------------------------------------------------------


def uys_gamma_el(
    amplitudes_dd: np.ndarray, amplitudes_uu: np.ndarray, omega_r_half: float, gamma: float
) -> float:
    """Gamma_el = Omega_R^2 gamma sum_lambda (sum_J A^{dd}_{J,lambda} - sum_J A^{uu}_{J,lambda})^2 with signed real amplitudes
    summed over the intermediate levels J (axis 1) inside the square and over the incident components lambda (axis 0) outside."""
    dd = np.asarray(amplitudes_dd, dtype=float).sum(axis=1)
    uu = np.asarray(amplitudes_uu, dtype=float).sum(axis=1)
    return float(omega_r_half**2 * gamma * np.sum((dd - uu) ** 2))


def uys_gamma_ij(amplitudes: np.ndarray, omega_r_half: float, gamma: float) -> float:
    """Gamma_ij = Omega_R^2 gamma sum_lambda (sum_J A^{i->j}_{J,lambda})^2."""
    return float(omega_r_half**2 * gamma * np.sum(np.asarray(amplitudes, dtype=float).sum(axis=1) ** 2))


def uys_bounds(gamma_dd: float, gamma_uu: float) -> tuple[float, float]:
    """0 <= Gamma_el <= 2(Gamma_dd + Gamma_uu): zero for equal amplitudes, the maximum for equal and opposite ones."""
    return 0.0, 2.0 * (gamma_dd + gamma_uu)


__all__ = [
    "OZERI_PHOTONS_PER_RADIAN_COEFFICIENT",
    "ozeri_epsilon_s_from_power",
    "ozeri_gamma_over_g_squared",
    "ozeri_gamma_raman",
    "ozeri_gamma_total",
    "ozeri_p_raman",
    "ozeri_p_rayleigh",
    "ozeri_p_total",
    "ozeri_p_total_optimum_delta",
    "ozeri_photons_per_stark_radian",
    "ozeri_raman_rabi_half",
    "uys_bounds",
    "uys_gamma_el",
    "uys_gamma_ij",
    "wineland_bracket",
    "wineland_clock_light_shift",
    "wineland_delta_over_omega_clock",
    "wineland_p_se_clock",
    "wineland_p_se_zeeman_22_11",
    "wineland_photons_per_stark_radian",
    "wineland_r_se_clock",
    "wineland_ratio_function",
]
