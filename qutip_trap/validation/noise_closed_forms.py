"""Closed forms for the noise channels of Section 6 that the M7 audit found missing (PLAN.md Sections 6.2 to 6.5;
Section 9.7 rows "Mode-frequency fluctuation", "Intensity noise", "Scattering"; Section 9.16 row 4.4-8).

The forms in ``two_qubit_closed_forms.py`` transcribe the plan's own gate-error expressions; the ones here are the
four the plan names but does not transcribe, plus one correction to a transcribed one:

- **Hughes's two-term mode-frequency-fluctuation infidelity** (Section 6.2, Section 9.7 row 3). Hughes et al.,
  arXiv:2510.17286 (2025), Appendix B "Infidelity Due to Mode Frequency Fluctuations", Eq. 33. The plan names the two
  terms ("thermal spin-motion (2 nbar + 1)|alpha_t|^2 lambda^2 plus a temperature-insensitive gate-angle term") without
  transcribing the second, so the second is read from the source and re-derived here.
- **The N-ion intensity-noise amplitude** (Section 6.4, Section 9.16 row 4.4-8). The plan's eps_I first term carries no
  factor N; the simulator's own Lindblad channel gives A = N Gamma_I t_g eta^2/2 (``anchor.m7.intensity_noise_n_ions``).
- **Baldwin's laser-phase-noise filter form** and **Kirchmair's carrier-excitation anchor** (Section 6.3).
- **eps_D = f P_total**, the D-level branch of the scattering error reported beside eps_S (Section 6.5, Section 9.7
  row 8; Ozeri 2007).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

# ---- Hughes: mode-frequency fluctuations (Section 6.2, Section 9.7 row "Mode-frequency fluctuation") ------------------
#
# Hughes et al. arXiv:2510.17286 App. B writes the gate in the frame hbar(omega_m - delta) a^dag a as
# H_g = hbar delta(t) a^dag a + (hbar Omega_g(t)/2) S_alpha (a^dag + a) (their Eq. 1 / Section 4.4.6 of the plan), whose
# ideal propagator is U_g(t) = exp{(gamma[t] a^dag - gamma*[t] a) S_alpha} exp{-i (theta[t]/2) S_alpha^2} (Eq. 27). A
# mode-frequency error enters as H_e(t) = hbar eps(t) a^dag a (Eq. 29). Expanding to second order in eps (Eq. 33):
#
#     I = (2 nbar + 1) |alpha_t|^2 lambda^2_{S_alpha} + (delta_theta^2/4) lambda^2_{S_alpha^2},
#     alpha_t = int_0^{t_g} eps(t) gamma(t) dt,      delta_theta = 2 int_0^{t_g} eps(t) |gamma(t)|^2 dt.
#
# The lambda^2 are the VARIANCES of the two spin operators over the input state (the source's own gloss; for the usual
# |0...0> input <S_alpha> = 0 so lambda^2_{S_alpha} = <S_alpha^2>, which is why the plan's sentence prints it that way).
# Verified against exact two- and three-ion propagation of the plan's own force model: the ratio of the exact spin
# infidelity to this form extrapolates to 1.0000 as eps -> 0 with a linear-in-eps remainder
# (``tests/test_m7_hughes.py``, ``anchor.m7.hughes_two_term``).


def hughes_spin_moments(n_ions: int) -> tuple[float, float]:
    """(lambda^2_{S_alpha}, lambda^2_{S_alpha^2}) = (Var S_alpha, Var S_alpha^2) for |0...0> in the S_alpha basis.

    S_alpha = sum_i sigma_alpha^{(i)} has iid +-1 eigenvalues s_i on that input, so <S> = 0 and Var S = <S^2> = N,
    while Var S^2 = <S^4> - <S^2>^2 = (3 N^2 - 2 N) - N^2 = 2 N (N - 1); enumerated rather than taken from that
    polynomial so the moments cannot drift from the operator (N = 2 gives (2, 4), N = 3 gives (3, 12)).
    """
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
    """Hughes Eq. 33: (2 nbar + 1)|alpha_t|^2 Var(S_alpha) + (delta_theta^2/4) Var(S_alpha^2).

    ``alpha_t_abs`` is |int eps gamma dt| and ``gate_angle_error_rad`` is delta_theta = 2 int eps |gamma|^2 dt; the two
    static-offset closed forms below produce them for a symmetric K-loop gate. Valid for I << 1 (the source expands
    U_e keeping quadratic terms only).
    """
    if nbar < 0.0:
        raise ValueError("nbar is non-negative")
    var_s, var_s2 = hughes_spin_moments(n_ions)
    thermal = (2.0 * nbar + 1.0) * alpha_t_abs**2 * var_s
    angle = gate_angle_error_rad**2 / 4.0 * var_s2
    return thermal + angle


def hughes_static_offset_alpha_t(loops: int, relative_offset: float) -> float:
    """|alpha_t| = (pi sqrt(K)/2)(eps/delta) for a constant mode-frequency offset eps on a symmetric K-loop gate.

    gamma(t) = -(f/delta)(e^{i delta t} - 1) with f = delta/(4 sqrt K) the closure force and t_g = 2 pi K/delta, so
    int_0^{t_g} gamma dt = f t_g/delta and |alpha_t| = eps f t_g/delta = pi sqrt(K) (eps/delta)/2.
    """
    if loops < 1:
        raise ValueError("loops must be a positive integer")
    return 0.5 * math.pi * math.sqrt(loops) * abs(relative_offset)


def hughes_static_offset_gate_angle_error(relative_offset: float) -> float:
    """delta_theta = (pi/2)(eps/delta), K-INDEPENDENT, for a constant offset on a symmetric K-loop gate.

    delta_theta = 2 eps int_0^{t_g}|gamma|^2 dt = 4 eps f^2 t_g/delta^2, and f^2 t_g = delta/(8 pi K) x 2 pi K = ...
    cancels K exactly: the extra loops shrink the force as 1/sqrt(K) while lengthening the gate as K.
    """
    return 0.5 * math.pi * abs(relative_offset)


def hughes_static_offset_infidelity(loops: int, relative_offset: float, nbar: float, n_ions: int) -> float:
    """The two-term infidelity for a quasi-static fractional mode-frequency offset eps/delta (Section 6.2's route (c))."""
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
    """|int_0^{t_g} eps(t) gamma(t) dt| by Simpson quadrature, for a time-dependent eps(t) (the general Eq. 33 input)."""
    t = np.linspace(0.0, t_g_s, int(n))
    vals = np.array([epsilon_of_t(float(x)) * gamma_of_t(float(x)) for x in t], dtype=complex)
    return float(abs(np.trapezoid(vals, t)))


def hughes_gate_angle_error_quadrature(
    epsilon_of_t: Callable[[float], float],
    gamma_of_t: Callable[[float], complex],
    t_g_s: float,
    n: int = 20001,
) -> float:
    """delta_theta = 2 int_0^{t_g} eps(t)|gamma(t)|^2 dt by Simpson quadrature (the general Eq. 33 input)."""
    t = np.linspace(0.0, t_g_s, int(n))
    vals = np.array([epsilon_of_t(float(x)) * abs(gamma_of_t(float(x))) ** 2 for x in t], dtype=float)
    return float(2.0 * np.trapezoid(vals, t))


# ---- intensity noise with its factor N (Section 6.4, Section 9.16 row 4.4-8) ------------------------------------------


def intensity_noise_error_n_ions(
    gamma_i_per_s: float, t_g_s: float, eta: float, nbar: float, n_ions: int, loops: int
) -> float:
    """A(2 nbar + 1) + B with A = N Gamma_I t_g eta^2/2 and B = A x 3(N - 1)/(4K): the plan's eps_I times N.

    Derivation (``anchor.m7.intensity_noise_n_ions``). With c_op = sqrt(2k) H_int on the K-loop force model and
    Gamma_I = k Omega^2 the carrier-contrast rate, block s of the sigma_alpha basis closes with a residual displacement
    s x delta_alpha, <|delta_alpha|^2> = f^2 (2 Gamma_I/Omega^2) t_g with f = eta Omega/2; the pair (s, s') decoheres by
    exp(-|s - s'|^2 <|delta_alpha|^2>(2 nbar + 1)/2), so

        1 - F = (1/2) <(s - s')^2> (2 nbar + 1) f^2 D t_g = Var(sum_i s_i) (2 nbar + 1) (eta^2/2) Gamma_I t_g,

    and Var(sum_i s_i) = N for |0...0>. That variance is the SAME lambda^2_{S_alpha} that carries Hughes's first term,
    which is the cross-check that fixes the factor. The plan's ``intensity_noise_error_derived`` is this at N = 1.
    """
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
    """B(N, K) = N Gamma_I t_g eta^2 3(N - 1)/(8K), i.e. A(N) x 3(N - 1)/(4K).

    Only a GLOBAL (correlated) intensity-noise operator reproduces this ratio: N independent per-beam operators give
    A(N) unchanged but halve the ratio to 3(N - 1)/(8K) (``anchor.m7.intensity_noise_n_ions``), so the plan's B term is
    itself evidence that its eps_I describes one laser shared by the register.
    """
    return (
        intensity_noise_amplitude_n_ions(gamma_i_per_s, t_g_s, eta, n_ions)
        * 3.0
        * (n_ions - 1)
        / (4.0 * loops)
    )


# ---- laser phase noise (Section 6.3) ----------------------------------------------------------------------------------


def baldwin_phase_noise_infidelity(
    s_phi: Callable[[np.ndarray], np.ndarray],
    filter_function: Callable[[np.ndarray], np.ndarray],
    omega_min_rad_s: float,
    omega_max_rad_s: float,
    *,
    coupling_ratio: float = 1.0,
    n: int = 4001,
) -> float:
    """eps = [int S_phi(omega) F(omega) d omega] x (4 g^2/Delta^2)^2, Section 6.3's Baldwin form **[extracted]**.

    ``coupling_ratio`` is 4 g^2/Delta^2 (the two-photon Rabi frequency over the single-photon detuning, the factor by
    which a Raman pair converts optical phase noise into beat-note phase noise); the plan prints the scaling as its
    SQUARE, so the returned value carries ``coupling_ratio**2``. The integral is the one-sided angular integral of the
    module's own convention (Section 6.9: exactly one 1/(2 pi), so the one-sided fold carries 1/pi).
    """
    if omega_min_rad_s <= 0.0 or omega_max_rad_s <= omega_min_rad_s:
        raise ValueError("0 < omega_min < omega_max")
    w = np.geomspace(omega_min_rad_s, omega_max_rad_s, int(n))
    integrand = np.asarray(s_phi(w), dtype=float) * np.asarray(filter_function(w), dtype=float)
    return float(np.trapezoid(integrand, w) / math.pi) * coupling_ratio**2


def kirchmair_carrier_excitation_error() -> float:
    """2 x 10^-3 per gate from incoherent carrier excitation (Kirchmair 2009, attributed there to Benhelm 2008).

    A published laboratory anchor, not a closed form: Section 9's preamble makes it a consistency anchor that the suite
    reports rather than fails on (Section 6.3's "checked against the Kirchmair anchor" **[corrected minor]**).
    """
    return 2.0e-3


def kirchmair_rabi_drift_fraction() -> float:
    """delta Omega/Omega = 1.4 x 10^-2, Kirchmair 2009's slow Rabi-frequency variation (Section 6.4) **[verified]**."""
    return 1.4e-2


def harty_rabi_drift_fraction() -> float:
    """delta Omega/Omega <= 5 x 10^-4, Harty 2014's amplitude stability (Sections 6.4 and 7.10) **[verified]**."""
    return 5.0e-4


# ---- the D-level scattering branch (Section 6.5, Section 9.7 row "Scattering") ----------------------------------------


def epsilon_d_from_p_total(branching_fraction: float, p_total: float) -> float:
    """eps_D = f P_total, the D-level leakage error per pi pulse (Ozeri 2007; Section 9.7 row "Scattering").

    ``f`` is the excited manifold's branching fraction into the D levels. Section 4.5.5 records that Ozeri's own
    P_Rayleigh silently includes this channel for Ca+, Sr+, Ba+ and Yb+, "overstating the elastic rate by f P_total",
    so eps_D is reported BESIDE eps_S = P_Raman and subtracted from the elastic rate rather than added to it.
    """
    if not 0.0 <= branching_fraction <= 1.0:
        raise ValueError("a branching fraction lies in [0, 1]")
    if p_total < 0.0:
        raise ValueError("P_total is non-negative")
    return branching_fraction * p_total


def egan_xy_n_t2_s() -> tuple[float, float]:
    """T_2 = 2.84(16) s under microwave (XY)^N decoupling, Egan 2021 (Sections 6.3 and 6.9) **[verified]**."""
    return 2.84, 0.16


def harty_t2_star_s() -> tuple[float, float]:
    """T_2* = 50(10) s for an unshielded 43Ca+ clock qubit, Harty 2014 (Sections 6.3 and 6.9) **[verified]**."""
    return 50.0, 10.0


def gaussian_decay_t2_from_chi(
    chi_of_tau: Callable[[float], float], target: float = 1.0, bracket: Sequence[float] = (1e-4, 1e4)
) -> float:
    """The tau at which chi(tau) = ``target``, i.e. W = e^{-chi} = 1/e: the T_2 of Section 6.9's "evaluate W = e^{-chi}
    rather than fitting an exponential" rule. Bisection on a monotone chi over ``bracket``."""
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


__all__ = [
    "baldwin_phase_noise_infidelity",
    "egan_xy_n_t2_s",
    "epsilon_d_from_p_total",
    "gaussian_decay_t2_from_chi",
    "harty_rabi_drift_fraction",
    "harty_t2_star_s",
    "hughes_alpha_t_quadrature",
    "hughes_gate_angle_error_quadrature",
    "hughes_mode_frequency_infidelity",
    "hughes_spin_moments",
    "hughes_static_offset_alpha_t",
    "hughes_static_offset_gate_angle_error",
    "hughes_static_offset_infidelity",
    "intensity_noise_amplitude_n_ions",
    "intensity_noise_error_n_ions",
    "intensity_noise_intercept_n_ions",
    "kirchmair_carrier_excitation_error",
    "kirchmair_rabi_drift_fraction",
]
