"""Spin-motion closed forms used as oracles for the Hamiltonian builder.

All frequencies angular (rad/s), in the (hbar Omega/2) convention: a resonant carrier flops as sin^2(Omega t/2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.special import eval_genlaguerre

from qutip_trap.hilbert.operators import displacement_element_analytic


def generalized_rabi_rad_s(omega_rad_s: float, detuning_rad_s: float) -> float:
    """sqrt(Omega^2 + Delta^2), Wineland's (Delta^2 + 4 Omega_W^2)^{1/2} with Omega = 2 Omega_W."""
    return math.sqrt(omega_rad_s**2 + detuning_rad_s**2)


def two_level_population(omega_rad_s: float, detuning_rad_s: float, t_s: float) -> float:
    """P_up(t) = [Omega^2/(Omega^2 + Delta^2)] sin^2((t/2) sqrt(Omega^2 + Delta^2)) from |down>."""
    g = generalized_rabi_rad_s(omega_rad_s, detuning_rad_s)
    if g == 0.0:
        return 0.0
    return (omega_rad_s**2 / g**2) * math.sin(0.5 * g * t_s) ** 2


def sideband_rabi_rad_s(omega_rad_s: float, eta: float, n_from: int, n_to: int) -> float:
    """Omega_{n', n} = Omega |<n'|D(i eta)|n>| (Wineland 1998 Eq. 18)."""
    return omega_rad_s * abs(displacement_element_analytic(n_to, n_from, 1j * eta))


def lamb_dicke_sideband_rabi_rad_s(omega_rad_s: float, eta: float, n_from: int, n_to: int) -> float:
    """The Lamb-Dicke limits: Omega (carrier), eta Omega sqrt(n) (first red), eta Omega sqrt(n+1) (first blue), else 0."""
    k = n_to - n_from
    if k == 0:
        return omega_rad_s
    if k == -1:
        return eta * omega_rad_s * math.sqrt(n_from)
    if k == 1:
        return eta * omega_rad_s * math.sqrt(n_from + 1)
    return 0.0


def resonant_transition_amplitude(
    omega_rad_s: float, eta: float, phi_rad: float, n_from: int, n_to: int, t_s: float
) -> complex:
    """<up, n'| U(t) |down, n> on the resonant sideband: -i e^{i phi} (M/|M|) sin(|M| Omega t/2), M = <n'|D(i eta)|n>, with
    arg M = (pi/2)|n' - n| (+ pi where the Laguerre polynomial is negative) (Wineland Eq. 21 / RMP Eq. 84)."""
    m = displacement_element_analytic(n_to, n_from, 1j * eta)
    if m == 0.0:
        return 0.0j
    return complex(-1j * np.exp(1j * phi_rad) * (m / abs(m)) * math.sin(0.5 * abs(m) * omega_rad_s * t_s))


def carrier_debye_waller(n: int, eta: float) -> float:
    return float(math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2))


def thermal_debye_waller(eta: float, nbar: float) -> float:
    """The thermal average of e^{i eta (a + a^dag)}: exp[-eta^2 (nbar + 1/2)], exact for a thermal state."""
    return math.exp(-(eta**2) * (nbar + 0.5))


def cetina_theta(b_im: float, xi_m: float, kappa_per_m2: float, nbar: float) -> float:
    """theta_im = -b_im^2 xi_m^2 (Omega''/Omega) nbar (Cetina 2022)."""
    return -(b_im**2) * xi_m**2 * kappa_per_m2 * nbar


def cetina_contrast(thetas: Sequence[float], omega_rad_s: float, t_s: float) -> float:
    """C = prod_m (1 + theta_m^2 Omega^2 t^2)^{-1/2}."""
    return float(np.prod([(1.0 + th**2 * omega_rad_s**2 * t_s**2) ** -0.5 for th in thetas]))


def cetina_phase_lag(thetas: Sequence[float], omega_rad_s: float, t_s: float) -> float:
    """phi = sum_m arctan(theta_m Omega t), a LAG (the source's Eq. 2 prints the wrong sign)."""
    return float(sum(math.atan(th * omega_rad_s * t_s) for th in thetas))


def cetina_population(thetas: Sequence[float], omega_rad_s: float, t_s: float) -> float:
    """p_1(t) = [1 - C cos(Omega t - phi)]/2 for the carrier drive of a thermal ion through the curved beam."""
    c = cetina_contrast(thetas, omega_rad_s, t_s)
    phi = cetina_phase_lag(thetas, omega_rad_s, t_s)
    return 0.5 * (1.0 - c * math.cos(omega_rad_s * t_s - phi))


def gaussian_curvature_per_m2(waist_m: float, offset_m: float = 0.0) -> float:
    """Omega''/Omega of a Gaussian FIELD profile: -(2/w^2)(1 - 2 x^2/w^2); the intensity form is exactly twice."""
    return -(2.0 / waist_m**2) * (1.0 - 2.0 * offset_m**2 / waist_m**2)


def frozen_thermal_population(
    omega_rad_s: float, eta: float, nbar: float, t_s: float, *, n_max: int | None = None
) -> float:
    """sum_n P_n sin^2(Omega_n t/2) with Omega_n = Omega e^{-eta^2/2} L_n(eta^2): the carrier under a frozen thermal spectator."""
    from qutip_trap.hilbert.operators import thermal_populations

    if n_max is None:
        n_max = int(60 + 40 * nbar)
    p = thermal_populations(nbar, n_max + 1)
    return float(
        sum(
            p[n] * math.sin(0.5 * omega_rad_s * carrier_debye_waller(n, eta) * t_s) ** 2
            for n in range(n_max + 1)
        )
    )


__all__ = [
    "carrier_debye_waller",
    "cetina_contrast",
    "cetina_phase_lag",
    "cetina_population",
    "cetina_theta",
    "frozen_thermal_population",
    "gaussian_curvature_per_m2",
    "generalized_rabi_rad_s",
    "lamb_dicke_sideband_rabi_rad_s",
    "resonant_transition_amplitude",
    "sideband_rabi_rad_s",
    "thermal_debye_waller",
    "two_level_population",
]
