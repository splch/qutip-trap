"""Closed forms of Section 4.4 used as test oracles for the two-qubit gates (PLAN.md Sections 4.4.1-4.4.7, 6.2, 9.4, 9.16).

All frequencies angular (rad/s), the plan's per-tone (hbar Omega/2) convention and S_alpha = sum_i sigma_alpha^i (Section 13,
"Spin operator in MS formulas"): the bichromatic force is -(hbar eta Omega/2) S_phi (a^dag e^{i eps t} + h.c.), the sign
of Section 13's own row (the builder's is the opposite, from the i of i eta(a + a^dag), and only |alpha| is consumed
downstream), the loop closes at eps t = 2 pi K, the two-body angle on sigma sigma is pi K (eta Omega/eps)^2 and eta Omega/eps = 1/(2 sqrt K) is
maximally entangling (chi = pi/4; ``check_ms_closure.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import qutip as qt
from scipy.integrate import quad
from scipy.special import jn_zeros, jv

from qutip_trap.hilbert.operators import displacement_element_analytic, thermal_populations

# ---- Section 4.4.1: the exact Molmer-Sorensen propagator and its observables ---------------------------------------------------------


def ms_closure_ratio(loops: int = 1) -> float:
    """eta Omega/eps = 1/(2 sqrt K) in every spin normalization (Sorensen-Molmer 2000 with J_y ingested unchanged; Section 13)."""
    return 1.0 / (2.0 * math.sqrt(loops))


def ms_closure_time_s(epsilon_rad_s: float, loops: int = 1) -> float:
    """tau = 2 pi K/eps = pi sqrt K/(eta Omega) at the maximally entangling closure."""
    return 2.0 * math.pi * loops / abs(epsilon_rad_s)


def ms_alpha(eta: float, omega_rad_s: float, epsilon_rad_s: float, t_s: float) -> complex:
    """alpha(t) = (eta Omega/(2 eps))(e^{i eps t} - 1): each S_y eigenstate m moves on a circle of radius |m| eta Omega/(2 eps) (Kirchmair Eq. 4)."""
    return complex(eta * omega_rad_s / (2.0 * epsilon_rad_s) * (np.exp(1j * epsilon_rad_s * t_s) - 1.0))


def ms_lambda_rad_s(eta: float, omega_rad_s: float, epsilon_rad_s: float) -> float:
    """lambda = eta^2 Omega^2/(4 eps): the secular rate of the S_y^2 phase."""
    return eta**2 * omega_rad_s**2 / (4.0 * epsilon_rad_s)


def ms_chi(eta: float, omega_rad_s: float, epsilon_rad_s: float) -> float:
    """chi = eta^2 Omega^2/(4 eps^2): the oscillating part of the S_y^2 phase (Kirchmair's chi, not the entangling angle)."""
    return eta**2 * omega_rad_s**2 / (4.0 * epsilon_rad_s**2)


def ms_gamma(eta: float, omega_rad_s: float, epsilon_rad_s: float, t_s: float) -> float:
    """gamma(t) = lambda t - chi sin(eps t): the coefficient of S_y^2 in the exact propagator D(alpha S_y) exp[i gamma S_y^2]."""
    return ms_lambda_rad_s(eta, omega_rad_s, epsilon_rad_s) * t_s - ms_chi(
        eta, omega_rad_s, epsilon_rad_s
    ) * math.sin(epsilon_rad_s * t_s)


def ms_two_body_angle(
    eta_a: float, eta_b: float, omega_rad_s: float, epsilon_rad_s: float, loops: int = 1
) -> float:
    """The coefficient of sigma_a sigma_b after K closed loops: pi K eta_a eta_b (Omega/eps)^2 with the sign of eps (S^2 = 2 + 2 sigma sigma)."""
    return (
        math.pi
        * loops
        * eta_a
        * eta_b
        * (omega_rad_s / epsilon_rad_s) ** 2
        * math.copysign(1.0, epsilon_rad_s)
    )


def spin_projectors(
    phi_rad: float = math.pi / 2.0, phi2_rad: float | None = None
) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """(S, P_0, P_+2, P_-2) for S = sigma_phi^1 + sigma_phi2^2 (phi2 defaults to phi): P_0 = 1 - S^2/4, P_{+-2} = (S^2 +- 2S)/8
    (Section 4.4.1); the scheduler's MS(0, 0) plays phi = 0 on the first ion and pi on the second when the kernel sign is positive."""
    phi2 = phi_rad if phi2_rad is None else phi2_rad
    s1 = math.cos(phi_rad) * qt.sigmax() + math.sin(phi_rad) * qt.sigmay()
    s2 = math.cos(phi2) * qt.sigmax() + math.sin(phi2) * qt.sigmay()
    s = qt.tensor(s1, qt.qeye(2)) + qt.tensor(qt.qeye(2), s2)
    one = qt.tensor(qt.qeye(2), qt.qeye(2))
    p0 = one - s * s / 4.0
    pp = (s * s + 2.0 * s) / 8.0
    pm = (s * s - 2.0 * s) / 8.0
    return s, p0, pp, pm


def ms_propagator(
    alpha: complex, gamma: float, d: int, phi_rad: float = math.pi / 2.0, phi2_rad: float | None = None
) -> qt.Qobj:
    """D(alpha S) exp[i gamma S^2] on two qubits x one mode of d levels through the P_0 + P_2 D(2 alpha) + P_-2 D(-2 alpha) decomposition."""
    s, p0, pp, pm = spin_projectors(phi_rad, phi2_rad)
    ident = qt.qeye(d)
    disp = (
        qt.tensor(p0, ident)
        + qt.tensor(pp, qt.displace(d, 2.0 * alpha))
        + qt.tensor(pm, qt.displace(d, -2.0 * alpha))
    )
    phase = qt.tensor((1j * gamma * s * s).expm(), ident)
    return disp * phase


def kirchmair_populations(alpha_abs: float, gamma: float, nbar: float) -> tuple[float, float, float]:
    """(p_0, p_1, p_2) populations with zero, one and two ions BRIGHT from |dd> with a thermal mode (Kirchmair 2009 Eq. 14):
    p_2 = (1/8)(3 + e^{-16|a|^2(n+1/2)} + 4 cos(4 gamma) e^{-4|a|^2(n+1/2)}), p_1 = (1/4)(1 - e^{-16|a|^2(n+1/2)}).

    Bright is the fluorescing S1/2 state, the LOWER qubit level |d> of 40Ca+: p_2 = P(dd) = P_00 in the computational ordering,
    p_0 = P(uu) = P_11; at t = 0 the formula gives p_2 = 1."""
    x = alpha_abs**2 * (nbar + 0.5)
    p2 = (3.0 + math.exp(-16.0 * x) + 4.0 * math.cos(4.0 * gamma) * math.exp(-4.0 * x)) / 8.0
    p1 = (1.0 - math.exp(-16.0 * x)) / 4.0
    return 1.0 - p1 - p2, p1, p2


def haljan_cat_signal(
    alpha0: float, alpha_tau: complex, nbar: float, ndot_per_s: float, tau_s: float
) -> float:
    """P_dn = (1/2)[1 - exp(-(1/2) ndot tau |4 alpha_0|^2 - (nbar + 1/2)|2 alpha(tau)|^2)] with the heating term (Haljan 2005)."""
    return 0.5 * (
        1.0
        - math.exp(
            -0.5 * ndot_per_s * tau_s * abs(4.0 * alpha0) ** 2 - (nbar + 0.5) * abs(2.0 * alpha_tau) ** 2
        )
    )


# ---- effective couplings (Sorensen-Molmer 1999, Molmer-Sorensen 1999) ----------------------------------------------------------------


def effective_bichromatic_coupling_rad_s(
    omega_rad_s: float, eta: float, nu_rad_s: float, delta_rad_s: float
) -> float:
    """Omega~ = -(Omega eta)^2/(nu - delta) for |ggn> <-> |een>, n-independent (twice the monochromatic value)."""
    return -((omega_rad_s * eta) ** 2) / (nu_rad_s - delta_rad_s)


def ghz_coupling_rad_s(eta: float, omega_rad_s: float, nu_rad_s: float, delta_rad_s: float) -> float:
    """chi of H = 4 chi J_x^2 (J = (1/2) sum sigma): chi = eta^2 Omega^2 nu/(2(nu^2 - delta^2)) (Molmer-Sorensen 1999)."""
    return eta**2 * omega_rad_s**2 * nu_rad_s / (2.0 * (nu_rad_s**2 - delta_rad_s**2))


def ghz_time_s(chi_rad_s: float) -> float:
    """The GHZ state appears at t = pi/(8 chi) for any even N (Molmer-Sorensen 1999)."""
    return math.pi / (8.0 * chi_rad_s)


# ---- corrections beyond the leading order (Roos 2008) -----------------------------------------------------------------------------


def roos_force_saturation(omega_rad_s: float, delta_rad_s: float) -> float:
    """J_0(x) + J_2(x) at x = 4 Omega/delta: the carrier's saturation of the spin-dependent force (Roos Eq. 17)."""
    x = 4.0 * omega_rad_s / delta_rad_s
    return float(jv(0, x) + jv(2, x))


def roos_spin_axis_rotation_rad(omega_rad_s: float, delta_rad_s: float, zeta_rad: float) -> float:
    """psi = (4 Omega/delta) sin zeta: the rotation of the spin axis by the carrier at the pulse's start phase."""
    return 4.0 * omega_rad_s / delta_rad_s * math.sin(zeta_rad)


def roos_counter_rotating_coupling_rad_s(eta: float, omega_rad_s: float, delta_rad_s: float) -> float:
    """-(eta^2 Omega^2/(2 delta)) J_0^2(4 Omega/delta): the S_y^2 coupling from the counter-rotating terms (a second-order Magnus term)."""
    x = 4.0 * omega_rad_s / delta_rad_s
    return -(eta**2 * omega_rad_s**2 / (2.0 * delta_rad_s)) * float(jv(0, x)) ** 2


def roos_sz2_coupling_rad_s(eta: float, omega_rad_s: float, delta_rad_s: float) -> float:
    """(2 eta^2 Omega^2/(3 delta)) J_1^2(4 Omega/delta): the unwanted S_z^2 coupling."""
    x = 4.0 * omega_rad_s / delta_rad_s
    return (2.0 * eta**2 * omega_rad_s**2 / (3.0 * delta_rad_s)) * float(jv(1, x)) ** 2


# ---- thermal (Debye-Waller) infidelities: one formula, three references (Section 4.4.7 (1)) -----------------------------------------


def chi_of_n(chi0: float, eta: float, n: int) -> float:
    """chi(n) = chi_0 [1 - eta^2 (2n + 1)]: the Debye-Waller law behind every thermal gate infidelity (derivation audit)."""
    return chi0 * (1.0 - eta**2 * (2 * n + 1))


def sideband_coupling_squared_difference(eta: float, n: int) -> float:
    """|M_n|^2 - |M_{n-1}|^2 with M_n = <n+1|D(i eta)|n>: eta^2 [1 - eta^2 (2n + 1)] + O(eta^6) (Section 9.16 row 4.4-7)."""
    m_n = abs(displacement_element_analytic(n + 1, n, 1j * eta)) ** 2
    m_prev = abs(displacement_element_analytic(n, n - 1, 1j * eta)) ** 2 if n >= 1 else 0.0
    return float(m_n - m_prev)


ThermalReference = Literal["mean", "n0", "minus_half"]


def thermal_debye_waller_infidelity(eta: float, nbar: float, reference: ThermalReference) -> float:
    """(pi^2/4) eta^4 <(n - n_ref)^2> over the thermal distribution: n_ref = nbar (Sorensen-Molmer, re-optimized duration:
    nbar^2 + nbar), 0 (Ballance, calibrated at n = 0: 2 nbar^2 + nbar), -1/2 (Zhu, referenced to eta^2 (2n + 1) = 0:
    2 nbar^2 + 2 nbar + 1/4), in units of (pi^2/4) eta^4 (Section 4.4.7 (1))."""
    pref = (math.pi**2 / 4.0) * eta**4
    var = nbar * (nbar + 1.0)
    if reference == "mean":
        return pref * var
    if reference == "n0":
        return pref * (var + nbar**2)
    if reference == "minus_half":
        return pref * (var + (nbar + 0.5) ** 2)
    raise ValueError("reference is 'mean', 'n0' or 'minus_half'")


def sorensen_molmer_thermal_fidelity(n_ions: int, eta: float, var_n: float) -> float:
    """F = 1 - pi^2 N(N - 1) eta^4 Var(n)/8 (Sorensen-Molmer 2000), the bracket being Var(n) and not 1.2 nbar^2 + 1.4 nbar."""
    return 1.0 - math.pi**2 * n_ions * (n_ions - 1) * eta**4 * var_n / 8.0


def ballance_thermal_error(eta: float, nbar: float) -> float:
    """eps_nbar = (1/4) pi^2 eta^4 nbar (2 nbar + 1) = (pi^2/4) eta^4 <n^2>, calibrated at n = 0 (Ballance 2016 supplement)."""
    return 0.25 * math.pi**2 * eta**4 * nbar * (2.0 * nbar + 1.0)


def recalibrated_gate_time_s(t_g0_s: float, eta: float, nbar: float) -> float:
    """t_g(nbar) = t_g(0)(1 + eta^2 (2 nbar + 1)) after heating: the recalibration rule without Bermudez's 1/N (Section 4.4.7)."""
    return t_g0_s * (1.0 + eta**2 * (2.0 * nbar + 1.0))


# ---- residual displacement: one quantity in three fidelity measures (Section 4.4.7 (8)) ----------------------------------------------


def entanglement_infidelity_from_displacements(alphas: Sequence[complex], nbars: Sequence[float]) -> float:
    """eps_ent = sum_{j,m} |alpha_jm|^2 (2 nbar_m + 1) for the (ion, mode) list of residual displacements."""
    return float(sum(abs(a) ** 2 * (2.0 * nb + 1.0) for a, nb in zip(alphas, nbars)))


def landsman_average_gate_infidelity(eps_ent: float) -> float:
    """(4/5) eps_ent: the two-qubit average gate infidelity d/(d + 1) (Landsman 2019)."""
    return 0.8 * eps_ent


def leung_zero_temperature_error(alphas: Sequence[complex]) -> float:
    """sum_k |alpha_k|^2: eps_ent at nbar = 0 (Leung 2018)."""
    return float(sum(abs(a) ** 2 for a in alphas))


def zhu_state_infidelity_as_printed(
    alphas_j: Sequence[complex], alphas_n: Sequence[complex], nbars: Sequence[float]
) -> float:
    """sum_k beta_k (|alpha_j^k|^2 + |alpha_n^k|^2)/4 with beta_k = 2 nbar_k + 1 (Zhu 2006 Eq. 8 as printed)."""
    return float(
        sum(
            (2.0 * nb + 1.0) * (abs(aj) ** 2 + abs(an) ** 2) / 4.0
            for aj, an, nb in zip(alphas_j, alphas_n, nbars)
        )
    )


def zhu_beta_bar(mu_k: float, nbar_1: float) -> float:
    """coth[sqrt(mu_k/4) ln(1 + 1/nbar_1)] = 2 nbar_k + 1 for the thermal mode k at nbar_k = 1/((1 + 1/nbar_1)^{sqrt(mu_k)} - 1)."""
    return 1.0 / math.tanh(math.sqrt(mu_k / 4.0) * math.log1p(1.0 / nbar_1))


def overlap_fidelity(alpha: complex) -> float:
    """|<ideal|actual>|^2 = e^{-|alpha|^2} for a single residual spin-dependent displacement +-alpha of a superposition (Leung)."""
    return math.exp(-(abs(alpha) ** 2))


def traced_out_fidelity(alpha: complex) -> float:
    """(1/2)(1 + e^{-2|alpha|^2}) after tracing the motion out (Section 4.4.3)."""
    return 0.5 * (1.0 + math.exp(-2.0 * abs(alpha) ** 2))


def state_infidelity_uniform_input(alphas_per_ion: Sequence[complex], nbar: float) -> float:
    """Exact state infidelity of an equal-superposition input (|dd> for a sigma_x-type force) under the residual displacements of one
    mode: F = (1/16) sum_{s,s'} exp(-|beta_s - beta_s'|^2 (nbar + 1/2)), beta_s = sum_j alpha_j s_j, s_j = +-1."""
    n = len(alphas_per_ion)
    signs = list(np.ndindex(*([2] * n)))
    total = 0.0
    for s in signs:
        beta_s = sum(a * (1.0 if b else -1.0) for a, b in zip(alphas_per_ion, s))
        for sp in signs:
            beta_sp = sum(a * (1.0 if b else -1.0) for a, b in zip(alphas_per_ion, sp))
            total += math.exp(-(abs(beta_s - beta_sp) ** 2) * (nbar + 0.5))
    return 1.0 - total / len(signs) ** 2


# ---- light-shift gate closed forms (Section 4.4.4) --------------------------------------------------------------------------------


def ballance_heating_error(ndot_per_s: float, t_g_s: float, loops: int) -> float:
    """eps_h = ndot t_g/(2K), stated for eps_h << 0.1 only (Ballance 2016 supplement; Section 6.2)."""
    if loops < 1:
        raise ValueError("at least one loop")
    val = ndot_per_s * t_g_s / (2.0 * loops)
    if val >= 0.1:
        raise ValueError("eps_h = ndot t_g/(2K) is stated for eps_h << 0.1 only (Section 6.2)")
    return val


def ballance_dephasing_coefficient(loops: int) -> float:
    """alpha_K = 1/(2K) + 3/(16 K^2) = (8K + 3)/(16 K^2): 11/16, 19/64, 35/256 for K = 1, 2, 4 (derivation audit; Ballance prints 0.686)."""
    return (8.0 * loops + 3.0) / (16.0 * loops**2)


def ballance_dephasing_error(t_g_s: float, tau_s: float, loops: int) -> float:
    """eps_d = alpha_K t_g/tau for the Lindblad operator L = a^dag a sqrt(2/tau) (Section 6.2)."""
    return ballance_dephasing_coefficient(loops) * t_g_s / tau_s


def baldwin_loop_phase(eta: float, omega_rad_s: float, delta_rad_s: float) -> float:
    """Phi_loop = 2 pi (eta Omega/delta)^2 on S^2 = (sigma_z^2 - sigma_z^1)^2 per closed loop of H = eta Omega S (a e^{i delta t} + h.c.);
    S^2 = 2 - 2 sigma_z sigma_z, so 4 pi (eta Omega/delta)^2 on sigma_z sigma_z per loop and 8 pi (...)^2 over the two echo loops (Eq. 1)."""
    return 2.0 * math.pi * (eta * omega_rad_s / delta_rad_s) ** 2


def baldwin_echo_unitary(
    eta: float, omega_rad_s: float, delta_rad_s: float, *, d: int = 24, nsteps: int = 10**7
) -> np.ndarray:
    """R_x(pi) U(loop) R_{-x}(pi) U(loop) for the printed H = eta Omega sum_j (-1)^j (1 + sigma_z^j)(a e^{i delta t} + h.c.) integrated
    numerically from |0> of the mode: the 4 x 4 spin unitary (the motion returns to |0> at each closed loop)."""
    a = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(d))
    sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(d))
    sz2 = qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(d))
    one = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(d))
    s_op = -(one + sz1) + (one + sz2)  # (-1)^j (1 + sigma_z^j), j = 1, 2
    force = eta * omega_rad_s * s_op * a
    h = qt.QobjEvo(
        [
            [force, lambda t, args: np.exp(1j * delta_rad_s * t)],
            [force.dag(), lambda t, args: np.exp(-1j * delta_rad_s * t)],
        ]
    )
    tau = 2.0 * math.pi / abs(delta_rad_s)
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": nsteps, "method": "dop853"}
    cols = []
    rx = qt.tensor(
        (-1j * math.pi / 2.0 * qt.sigmax()).expm(), (-1j * math.pi / 2.0 * qt.sigmax()).expm(), qt.qeye(d)
    )
    for basis in range(4):
        spin = qt.basis(4, basis)
        spin.dims = [[2, 2], [1, 1]]
        psi0 = qt.tensor(spin, qt.basis(d, 0))
        psi = qt.sesolve(h, psi0, [0.0, tau], options=opts).states[-1]
        psi = rx * psi
        psi = qt.sesolve(h, psi, [tau, 2.0 * tau], options=opts).states[-1]
        psi = rx.dag() * psi
        arr = np.asarray(psi.full()).reshape(4, d)
        leak = float(1.0 - np.sum(np.abs(arr[:, 0]) ** 2))
        if leak > 1e-6:
            raise RuntimeError(f"the loop did not close: {leak:.2e} of the population left |0>")
        cols.append(arr[:, 0])
    return np.array(cols).T


def zhu_lab_frame_force_rad_s(omega_ls_rad_s: float, eta: float) -> float:
    """The Lamb-Dicke force amplitude eta Omega_j of H = hbar Omega_j cos(Delta k q_j + mu t) sigma_j^z (Zhu-Monroe-Duan Eq. 2)."""
    return eta * omega_ls_rad_s


# ---- microwave-gradient gate (Srinivas 2021) ----------------------------------------------------------------------------------------


def srinivas_gradient_rabi_rad_s(
    r0_m: float, gradient_t_per_m: float, field_sensitivity_rad_s_per_t: float
) -> float:
    """Omega_g = (r_0/4) [grad(B_g . r_q) . r] (d omega_0/dB) (Srinivas 2021 Eq. 1; r_0 carries the TOTAL two-ion mass there, Section 13)."""
    return 0.25 * r0_m * gradient_t_per_m * field_sensitivity_rad_s_per_t


def srinivas_effective_coupling_rad_s(
    omega_g_rad_s: float, omega_mu_rad_s: float, delta_rad_s: float
) -> float:
    """Omega_g J_2(4 Omega_mu/delta): the microwave-dressed spin-dependent force on (sigma_z1 - sigma_z2)."""
    return omega_g_rad_s * float(jv(2, 4.0 * omega_mu_rad_s / delta_rad_s))


def intrinsic_dynamical_decoupling_ratio() -> float:
    """Omega_mu/delta at J_0(4 Omega_mu/delta) = 0: the first zero 2.4048/4 = 0.6012 (Srinivas 2021)."""
    return float(jn_zeros(0, 1)[0]) / 4.0


# ---- smooth gates (Hughes 2025) ------------------------------------------------------------------------------------------------------


def hughes_gate_angle_printed(
    omega_g_rad_s: Callable[[float], float],
    delta_rad_s: Callable[[float], float],
    alpha_dot: Callable[[float], float],
    tau_s: float,
) -> float:
    """theta_g ~ int (Omega_g^2 + alpha_dot^2)/delta dt as Hughes 2025 print it (Section 4.4.6)."""
    val, _err = quad(
        lambda t: (omega_g_rad_s(t) ** 2 + alpha_dot(t) ** 2) / delta_rad_s(t), 0.0, tau_s, limit=400
    )
    return float(val)


def hughes_gate_angle_derived(
    omega_g_rad_s: Callable[[float], float], delta_rad_s: Callable[[float], float], tau_s: float
) -> float:
    """theta_g = int Omega_g^2/(2 delta) dt for H_g = hbar delta a^dag a + (hbar Omega_g/2) S (a^dag + a): the displaced-oscillator energy
    shift -s^2 (Omega_g/2)^2/delta gives (Omega_g^2/(4 delta)) t on S^2, i.e. Omega_g^2 t/(2 delta) on S^2/2, whose maximal value is
    pi/4 (S^2 has eigenvalues 0 and 4); the printed form is twice this for the displayed Hamiltonian (recorded in the ledger)."""
    val, _err = quad(lambda t: omega_g_rad_s(t) ** 2 / (2.0 * delta_rad_s(t)), 0.0, tau_s, limit=400)
    return float(val)


HUGHES_MAXIMAL_ANGLE_RAD = math.pi / 4.0
"""theta_g = pi/4 is maximally entangling in Hughes Eq. 4 (S_alpha^2 has eigenvalues 0 and 4), not pi/2 [corrected]."""


# ---- Bermudez 2017 budgets: the derived forms and the printed ones (Section 4.4.7) -----------------------------------------------


def spectator_loop_error(
    n_ions: int,
    loops: int,
    gate_omega_rad_s: float,
    spectators: Sequence[tuple[float, float, float]],
    t_g_s: float,
) -> float:
    """eps_loop = pi^2 N K sum_{m != g} (2 n_m + 1)(omega_g/omega_m) sin^2(delta_m t_g/2)/(delta_m t_g)^2 for a COM-mode gate with equal
    per-ion participation; ``spectators`` are (omega_m, delta_m = mu - omega_m, n_m); the thermal weight is 2 n_m + 1 (derived form)."""
    total = 0.0
    for omega_m, delta_m, n_m in spectators:
        x = delta_m * t_g_s
        total += (2.0 * n_m + 1.0) * (gate_omega_rad_s / omega_m) * math.sin(0.5 * x) ** 2 / x**2
    return math.pi**2 * n_ions * loops * total


def bermudez_printed_motional_error(
    n_ions: int, delta_minus_omega_z: float, omega_z: float, t_g_s: float, nbar: float, eta: float
) -> float:
    """eps_m ~ [pi N (delta - omega_z)/(2 omega_z^2 t_g)] 0.8 (nbar + 1) + [pi^2 N(N-1) eta^4/(8 N^2)](1.2 nbar^2 + 1.4 nbar), as printed."""
    loops = math.pi * n_ions * delta_minus_omega_z / (2.0 * omega_z**2 * t_g_s) * 0.8 * (nbar + 1.0)
    dw = math.pi**2 * n_ions * (n_ions - 1) * eta**4 / (8.0 * n_ions**2) * (1.2 * nbar**2 + 1.4 * nbar)
    return loops + dw


def debye_waller_error_derived(n_ions: int, eta: float, nbar: float) -> float:
    """(pi^2/8) N(N - 1)(eta^4/N^2)(nbar^2 + nbar): the bracket is Var(n), 28 to 34% below Bermudez's 1.2 nbar^2 + 1.4 nbar."""
    return math.pi**2 / 8.0 * n_ions * (n_ions - 1) * eta**4 / n_ions**2 * (nbar**2 + nbar)


def dephasing_error_derived(n_ions: int, t_g_s: float, t2_s: float, correlated: bool) -> float:
    """N t_g/(2 T_2) for uncorrelated and N^2 t_g/(4 T_2) for globally correlated field noise (T_2 the single-ion 1/e time)."""
    return n_ions**2 * t_g_s / (4.0 * t2_s) if correlated else n_ions * t_g_s / (2.0 * t2_s)


def dephasing_error_printed(n_ions: int, t_g_s: float, t2_s: float, correlated: bool) -> float:
    """2 t_g N^2/T_2 (global) or 2 t_g N/T_2 (uncorrelated), as Bermudez print them (8 and 4 times the derived forms)."""
    return 2.0 * t_g_s * (n_ions**2 if correlated else n_ions) / t2_s


def intensity_noise_error_derived(
    gamma_i_per_s: float, t_g_s: float, eta: float, nbar: float, n_ions: int, loops: int
) -> float:
    """Gamma_I t_g eta^2 (nbar + 1/2) + Gamma_I t_g eta^2 3(N - 1)/(8K): the spin term from int (1 - cos delta t)^2 dt = (3/2) t_g."""
    return gamma_i_per_s * t_g_s * eta**2 * (nbar + 0.5) + gamma_i_per_s * t_g_s * eta**2 * 3.0 * (
        n_ions - 1
    ) / (8.0 * loops)


def intensity_noise_error_printed(
    gamma_i_per_s: float, t_g_s: float, eta: float, nbar: float, n_ions: int
) -> float:
    """Gamma_I t_g eta^2 (nbar + 1/2) + Gamma_I t_g eta^2 (N - 1)/4 as printed."""
    return gamma_i_per_s * t_g_s * eta**2 * (nbar + 0.5) + gamma_i_per_s * t_g_s * eta**2 * (n_ions - 1) / 4.0


def kirchmair_heating_error(gamma_h_per_s: float, t_g_s: float) -> float:
    """Delta F = Gamma_h t_g/2 for the single-loop MS gate (Kirchmair 2009 after Sorensen-Molmer 2000)."""
    return 0.5 * gamma_h_per_s * t_g_s


# ---- segment and constraint counting (Section 4.4.7 (2)) ------------------------------------------------------------------------------


def choi_segment_count(n_ions: int, transverse_families: int = 1) -> int:
    """2N + 1 segments for one transverse family, 4N + 1 with both (35 -> 69 at N = 17)."""
    return 2 * n_ions * transverse_families + 1


def blumel_constraint_rows(n_ions: int, stabilization_order: int, transverse_families: int = 1) -> int:
    """N(K + 1) closure-plus-derivative rows per family, 2N(K + 1) with both families (51 -> 102 at N = 17, K = 2)."""
    return n_ions * transverse_families * (stabilization_order + 1)


PUBLISHED_RECORDS: dict[str, tuple[float, str]] = {
    "fastest_gate_2018": (1.6e-6, "1.6 us at 99.8% fidelity (Schafer et al. 2018)"),
    "lowest_error_2025": (8.4e-5, "8.4(7) x 10^-5 two-qubit error (2025)"),
    "helios_2025": (7.9e-4, "7.9(2) x 10^-4 two-qubit error, Quantinuum Helios"),
}
"""Consistency anchors of Section 9.4 (tracked, never reproduced from first principles)."""


def thermal_average(values_by_n: Sequence[float], nbar: float) -> float:
    """sum_n P_n f(n) over the thermal distribution truncated at the given length (renormalized)."""
    p = thermal_populations(nbar, len(values_by_n))
    p = p / p.sum()
    return float(np.dot(p, np.asarray(values_by_n, dtype=float)))


__all__ = [
    "fang_bell_fidelity",
    "fang_crosstalk_unitary",
    "fang_printed_bell_fidelity",
    "fang_printed_spectator_excitation",
    "fang_spectator_excitation",
    "inter_pair_phase_scaling",
    "landsman_parallel_gate_bound",
    "ou_field_heating_finite_time",
    "ou_field_heating_slope",
    "sigma_phi",
    "HUGHES_MAXIMAL_ANGLE_RAD",
    "PUBLISHED_RECORDS",
    "ThermalReference",
    "baldwin_echo_unitary",
    "baldwin_loop_phase",
    "ballance_dephasing_coefficient",
    "ballance_dephasing_error",
    "ballance_heating_error",
    "ballance_thermal_error",
    "bermudez_printed_motional_error",
    "blumel_constraint_rows",
    "chi_of_n",
    "choi_segment_count",
    "debye_waller_error_derived",
    "dephasing_error_derived",
    "dephasing_error_printed",
    "effective_bichromatic_coupling_rad_s",
    "entanglement_infidelity_from_displacements",
    "ghz_coupling_rad_s",
    "ghz_time_s",
    "haljan_cat_signal",
    "hughes_gate_angle_derived",
    "hughes_gate_angle_printed",
    "intensity_noise_error_derived",
    "intensity_noise_error_printed",
    "intrinsic_dynamical_decoupling_ratio",
    "kirchmair_heating_error",
    "kirchmair_populations",
    "landsman_average_gate_infidelity",
    "leung_zero_temperature_error",
    "ms_alpha",
    "ms_chi",
    "ms_closure_ratio",
    "ms_closure_time_s",
    "ms_gamma",
    "ms_lambda_rad_s",
    "ms_propagator",
    "ms_two_body_angle",
    "overlap_fidelity",
    "recalibrated_gate_time_s",
    "roos_counter_rotating_coupling_rad_s",
    "roos_force_saturation",
    "roos_spin_axis_rotation_rad",
    "roos_sz2_coupling_rad_s",
    "sideband_coupling_squared_difference",
    "sorensen_molmer_thermal_fidelity",
    "spectator_loop_error",
    "spin_projectors",
    "srinivas_effective_coupling_rad_s",
    "srinivas_gradient_rabi_rad_s",
    "state_infidelity_uniform_input",
    "thermal_average",
    "thermal_debye_waller_infidelity",
    "traced_out_fidelity",
    "zhu_beta_bar",
    "zhu_lab_frame_force_rad_s",
    "zhu_state_infidelity_as_printed",
]

# ---- crosstalk, parallel gates and field-noise heating (Sections 6.2, 6.6; Section 9.7 rows 'Crosstalk unitary', 'Parallel gates';
# Section 9.16 rows 4.1-3, 6-4; M7) ------------------------------------------------------------------------------------------------


def _pauli(name: str) -> np.ndarray:
    return {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }[name]


def _kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def sigma_phi(phi: float) -> np.ndarray:
    return math.cos(phi) * _pauli("X") + math.sin(phi) * _pauli("Y")


def fang_crosstalk_unitary(theta: float, theta_13: float, theta_23: float, phi_beam: float) -> np.ndarray:
    """U_xtalk = XX(theta) exp[-i(theta_13 X^(1) sigma_phi^(3) + theta_23 X^(2) sigma_phi^(3))] on ions (1, 2, 3), the leading-order
    crosstalk model of Fang et al. 2022 (Section 6.6): XX(theta) = exp(-i theta X1 X2) with no 1/2 (Section 13)."""
    from scipy.linalg import expm

    x1x2 = _kron3(_pauli("X"), _pauli("X"), _pauli("I"))
    gate = expm(-1j * theta * x1x2)
    leak = theta_13 * _kron3(_pauli("X"), _pauli("I"), sigma_phi(phi_beam)) + theta_23 * _kron3(
        _pauli("I"), _pauli("X"), sigma_phi(phi_beam)
    )
    return np.asarray(gate @ expm(-1j * leak))


def fang_bell_fidelity(theta_13: float, theta_23: float) -> float:
    """The exact Bell-state fidelity of the pair under the crosstalk unitary: cos^2 theta_13 cos^2 theta_23 (0.900790 at
    (0.1644, -0.2763)); Fang's printed form carries half the angles (0.974422) [corrected]."""
    return math.cos(theta_13) ** 2 * math.cos(theta_23) ** 2


def fang_spectator_excitation(theta_13: float, theta_23: float) -> float:
    """P_ion3 = [1 - cos(2 theta_13) cos(2 theta_23)]/2 exactly (0.097217 at the quoted angles; 0 at theta = pi/2 where the
    printed halved form gives 0.5) [corrected]."""
    return 0.5 * (1.0 - math.cos(2.0 * theta_13) * math.cos(2.0 * theta_23))


def fang_printed_bell_fidelity(theta_13: float, theta_23: float) -> float:
    """The printed closed form, a factor 2 inside the cosines: cos^2(theta_13/2) cos^2(theta_23/2)."""
    return math.cos(theta_13 / 2.0) ** 2 * math.cos(theta_23 / 2.0) ** 2


def fang_printed_spectator_excitation(theta_13: float, theta_23: float) -> float:
    return 0.5 * (1.0 - math.cos(theta_13) * math.cos(theta_23))


def landsman_parallel_gate_bound(inter_pair_phases_rad: Sequence[float]) -> tuple[float, bool]:
    """(1/2)||E||_diamond <= sum_rs |Theta_rs| over the four inter-pair phases of two parallel gates (Landsman 2019; the 1/2 is
    load-bearing [corrected]); returns the bound on (1/2)||E||_diamond and whether it is vacuous (above 1, since
    ||E||_diamond <= 2 always), in which case the simulator reports the exact simulated channel instead."""
    bound = float(sum(abs(t) for t in inter_pair_phases_rad))
    return bound, bound > 1.0


def inter_pair_phase_scaling(n_sites: float, theta_adjacent: float) -> float:
    """Theta proportional to 1/n^3 with the ion separation in sites (Landsman 2019; Section 9.7 row 'Parallel gates')."""
    return theta_adjacent / float(n_sites) ** 3


def ou_field_heating_slope(sigma2: float, tau_c: float, omega: float) -> float:
    """d<n>/dt = e^2 S_E^(1)(omega)/(4 m hbar omega) for Ornstein-Uhlenbeck field noise of variance sigma^2 and correlation time
    tau_c in units e = m = hbar = 1: S_E^(1) = 4 sigma^2 tau_c/(1 + omega^2 tau_c^2), so the slope is sigma^2 tau_c/(1 + omega^2
    tau_c^2) (Section 9.16 row 4.1-3: 9.97506e-4 at sigma^2 = 0.02, tau_c = 0.05, omega = 1)."""
    return sigma2 * tau_c / (1.0 + (omega * tau_c) ** 2)


def ou_field_heating_finite_time(sigma2: float, tau_c: float, omega: float, t: float) -> float:
    """<n>(t) = (1/(2 hbar m omega)) int_0^t int_0^t C(t' - t'') cos(omega (t' - t'')) dt' dt'' for the classical oscillator driven by
    the OU field (units e = m = hbar = 1): the exact double integral whose slope approaches ``ou_field_heating_slope``."""
    from scipy.integrate import quad

    def inner(u: float) -> float:
        return float((t - u) * sigma2 * math.exp(-u / tau_c) * math.cos(omega * u))

    val = quad(inner, 0.0, t, limit=400)[0]
    return float(2.0 * val / (2.0 * omega))
