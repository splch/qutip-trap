"""Closed forms for the noise channels: Hughes et al. 2025's two-term mode-frequency-fluctuation infidelity, the N-ion
intensity-noise error, Baldwin's laser-phase-noise filter form, published laboratory anchors and the D-level scattering
branch eps_D = f P_total (Ozeri 2007).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

# ---- Hughes et al. 2025 (arXiv:2510.17286, App. B): mode-frequency fluctuations ------------------------------------------


def hughes_spin_moments(n_ions: int) -> tuple[float, float]:
    """(Var S_alpha, Var S_alpha^2) = (N, 2 N (N - 1)) for |0...0> in the S_alpha basis, by enumeration."""
    if n_ions < 1:
        raise ValueError("n_ions must be positive")
    s = np.array([float(sum(v)) for v in _pm_tuples(n_ions)])
    var_s = float(np.mean(s**2) - np.mean(s) ** 2)
    s2 = s**2
    var_s2 = float(np.mean(s2**2) - np.mean(s2) ** 2)
    return var_s, var_s2


def _pm_tuples(n: int) -> list[tuple[int, ...]]:
    from itertools import product

    return list(product((1, -1), repeat=n))


def hughes_mode_frequency_infidelity(
    alpha_t_abs: float, gate_angle_error_rad: float, nbar: float, n_ions: int
) -> float:
    """Hughes 2025 Eq. 33 for a mode-frequency error H_e = hbar eps(t) a^dag a, to second order (I << 1):
    I = (2 nbar + 1)|alpha_t|^2 Var(S_alpha) + (delta_theta^2/4) Var(S_alpha^2) with gamma(t) the gate's displacement,
    ``alpha_t_abs`` = |int_0^{t_g} eps gamma dt| and ``gate_angle_error_rad`` = delta_theta = 2 int_0^{t_g} eps |gamma|^2 dt."""
    if nbar < 0.0:
        raise ValueError("nbar is non-negative")
    var_s, var_s2 = hughes_spin_moments(n_ions)
    thermal = (2.0 * nbar + 1.0) * alpha_t_abs**2 * var_s
    angle = gate_angle_error_rad**2 / 4.0 * var_s2
    return thermal + angle


def hughes_static_offset_alpha_t(loops: int, relative_offset: float) -> float:
    """|alpha_t| = (pi sqrt(K)/2)(eps/delta) for a constant mode-frequency offset eps on a symmetric K-loop gate."""
    if loops < 1:
        raise ValueError("loops must be a positive integer")
    return 0.5 * math.pi * math.sqrt(loops) * abs(relative_offset)


def hughes_static_offset_gate_angle_error(relative_offset: float) -> float:
    """delta_theta = (pi/2)(eps/delta), independent of K, for a constant offset on a symmetric K-loop gate."""
    return 0.5 * math.pi * abs(relative_offset)


def hughes_static_offset_infidelity(loops: int, relative_offset: float, nbar: float, n_ions: int) -> float:
    """The two-term infidelity for a quasi-static fractional mode-frequency offset eps/delta."""
    return hughes_mode_frequency_infidelity(
        hughes_static_offset_alpha_t(loops, relative_offset),
        hughes_static_offset_gate_angle_error(relative_offset),
        nbar,
        n_ions,
    )


def hughes_alpha_t_quadrature(
    epsilon_of_t: Callable[[float], float],
    gamma_of_t: Callable[[float], complex],
    t_g_s: float,
    n: int = 20001,
) -> float:
    """|int_0^{t_g} eps(t) gamma(t) dt| by the trapezoidal rule on ``n`` points, for a time-dependent eps(t)."""
    t = np.linspace(0.0, t_g_s, int(n))
    vals = np.array([epsilon_of_t(float(x)) * gamma_of_t(float(x)) for x in t], dtype=complex)
    return float(abs(np.trapezoid(vals, t)))


def hughes_gate_angle_error_quadrature(
    epsilon_of_t: Callable[[float], float],
    gamma_of_t: Callable[[float], complex],
    t_g_s: float,
    n: int = 20001,
) -> float:
    """delta_theta = 2 int_0^{t_g} eps(t)|gamma(t)|^2 dt by the trapezoidal rule on ``n`` points."""
    t = np.linspace(0.0, t_g_s, int(n))
    vals = np.array([epsilon_of_t(float(x)) * abs(gamma_of_t(float(x))) ** 2 for x in t], dtype=float)
    return float(2.0 * np.trapezoid(vals, t))


def intensity_noise_error_n_ions(
    gamma_i_per_s: float, t_g_s: float, eta: float, nbar: float, n_ions: int, loops: int
) -> float:
    """A(2 nbar + 1) + B with A = N Gamma_I t_g eta^2/2 and B = A x 3(N - 1)/(4K): N times
    ``two_qubit_closed_forms.intensity_noise_error_derived``, for one intensity-noise source shared by the ions."""
    if n_ions < 1 or loops < 1:
        raise ValueError("n_ions and loops must be positive integers")
    a = float(n_ions) * gamma_i_per_s * t_g_s * eta**2 / 2.0
    b = a * 3.0 * (n_ions - 1) / (4.0 * loops)
    return a * (2.0 * nbar + 1.0) + b


def intensity_noise_amplitude_n_ions(gamma_i_per_s: float, t_g_s: float, eta: float, n_ions: int) -> float:
    """A(N) = N Gamma_I t_g eta^2/2, the coefficient of (2 nbar + 1) in ``intensity_noise_error_n_ions``."""
    if n_ions < 1:
        raise ValueError("n_ions must be a positive integer")
    return float(n_ions) * gamma_i_per_s * t_g_s * eta**2 / 2.0


def intensity_noise_intercept_n_ions(
    gamma_i_per_s: float, t_g_s: float, eta: float, n_ions: int, loops: int
) -> float:
    """B(N, K) = N Gamma_I t_g eta^2 3(N - 1)/(8K) = A(N) x 3(N - 1)/(4K) for global (correlated) intensity noise;
    independent per-beam noise halves the ratio."""
    return (
        intensity_noise_amplitude_n_ions(gamma_i_per_s, t_g_s, eta, n_ions)
        * 3.0
        * (n_ions - 1)
        / (4.0 * loops)
    )


def baldwin_phase_noise_infidelity(
    s_phi: Callable[[np.ndarray], np.ndarray],
    filter_function: Callable[[np.ndarray], np.ndarray],
    omega_min_rad_s: float,
    omega_max_rad_s: float,
    *,
    coupling_ratio: float = 1.0,
    n: int = 4001,
) -> float:
    """Baldwin's filter form eps = (1/pi) int S_phi(omega) F(omega) d omega x (4 g^2/Delta^2)^2, a one-sided angular
    integral (rad/s); ``coupling_ratio`` is 4 g^2/Delta^2, the two-photon Rabi frequency over the single-photon detuning."""
    if omega_min_rad_s <= 0.0 or omega_max_rad_s <= omega_min_rad_s:
        raise ValueError("0 < omega_min < omega_max")
    w = np.geomspace(omega_min_rad_s, omega_max_rad_s, int(n))
    integrand = np.asarray(s_phi(w), dtype=float) * np.asarray(filter_function(w), dtype=float)
    return float(np.trapezoid(integrand, w) / math.pi) * coupling_ratio**2


def kirchmair_carrier_excitation_error() -> float:
    """2 x 10^-3 per gate from incoherent carrier excitation, a laboratory anchor (Kirchmair 2009, after Benhelm 2008)."""
    return 2.0e-3


def kirchmair_rabi_drift_fraction() -> float:
    """delta Omega/Omega = 1.4 x 10^-2, Kirchmair 2009's slow Rabi-frequency variation."""
    return 1.4e-2


def harty_rabi_drift_fraction() -> float:
    """delta Omega/Omega <= 5 x 10^-4, Harty 2014's amplitude stability."""
    return 5.0e-4


def epsilon_d_from_p_total(branching_fraction: float, p_total: float) -> float:
    """eps_D = f P_total, the D-level leakage error per pi pulse with f the excited manifold's branching fraction into
    the D levels (Ozeri 2007, whose P_Rayleigh includes this channel for Ca+, Sr+, Ba+ and Yb+)."""
    if not 0.0 <= branching_fraction <= 1.0:
        raise ValueError("a branching fraction lies in [0, 1]")
    if p_total < 0.0:
        raise ValueError("P_total is non-negative")
    return branching_fraction * p_total


def egan_xy_n_t2_s() -> tuple[float, float]:
    """(T_2, its uncertainty) = 2.84(16) s under microwave (XY)^N decoupling (Egan 2021)."""
    return 2.84, 0.16


def harty_t2_star_s() -> tuple[float, float]:
    """(T_2*, its uncertainty) = 50(10) s for an unshielded 43Ca+ clock qubit (Harty 2014)."""
    return 50.0, 10.0


def gaussian_decay_t2_from_chi(
    chi_of_tau: Callable[[float], float], target: float = 1.0, bracket: Sequence[float] = (1e-4, 1e4)
) -> float:
    """The tau at which chi(tau) = ``target`` (W = e^{-chi} = 1/e by default): T_2 from the decoherence function rather
    than an exponential fit, by bisection in log tau on a monotone chi over ``bracket``."""
    lo, hi = float(bracket[0]), float(bracket[1])
    f_lo, f_hi = chi_of_tau(lo) - target, chi_of_tau(hi) - target
    if f_lo * f_hi > 0.0:
        raise ValueError(f"chi - {target} does not change sign over {bracket}: {f_lo}, {f_hi}")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if (chi_of_tau(mid) - target) * f_lo > 0.0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)
