"""Closed forms from the literature, and the level-C reference models, that the tests compare the package against."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigs
from scipy.special import eval_genlaguerre, jn_zeros, jv

from qutip_trap.dynamics.multilevel import SINK, ModeSpec, MultiLevelBuild, MultiLevelOptions
from qutip_trap.dynamics.operators import displacement_element_analytic, thermal_populations
from qutip_trap.dynamics.steady import steady_state_direct
from qutip_trap.light.beams import PolGradientBeams
from qutip_trap.light.bloch import BlochModel, CoolingError
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.wigner import Half, as_half_integer
from qutip_trap.units import TWO_PI


def ms_closure_ratio(loops: int = 1) -> float:
    """eta Omega/eps = 1/(2 sqrt K), the maximally entangling closure in every spin normalization (Sorensen-Molmer 2000)."""
    return 1.0 / (2.0 * math.sqrt(loops))


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
    """(S, P_0, P_+2, P_-2) for S = sigma_phi^1 + sigma_phi2^2 (phi2 defaults to phi): P_0 = 1 - S^2/4, P_{+-2} = (S^2 +- 2S)/8;
    the scheduler's MS(0, 0) plays phi = 0 on the first ion and pi on the second when the kernel sign is positive."""
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


def entanglement_infidelity_from_displacements(alphas: Sequence[complex], nbars: Sequence[float]) -> float:
    """eps_ent = sum_{j,m} |alpha_jm|^2 (2 nbar_m + 1) for the (ion, mode) list of residual displacements."""
    return float(sum(abs(a) ** 2 * (2.0 * nb + 1.0) for a, nb in zip(alphas, nbars)))


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


def ballance_heating_error(ndot_per_s: float, t_g_s: float, loops: int) -> float:
    """eps_h = ndot t_g/(2K), stated for eps_h << 0.1 only (Ballance 2016 supplement; Section 6.2)."""
    if loops < 1:
        raise ValueError("at least one loop")
    val = ndot_per_s * t_g_s / (2.0 * loops)
    if val >= 0.1:
        raise ValueError("eps_h = ndot t_g/(2K) is stated for eps_h << 0.1 only (Section 6.2)")
    return val


def ballance_dephasing_coefficient(loops: int) -> float:
    """alpha_K = 1/(2K) + 3/(16 K^2) = (8K + 3)/(16 K^2): 11/16, 19/64, 35/256 for K = 1, 2, 4 (Ballance prints 0.686)."""
    return (8.0 * loops + 3.0) / (16.0 * loops**2)


def ballance_dephasing_error(t_g_s: float, tau_s: float, loops: int) -> float:
    """eps_d = alpha_K t_g/tau for the Lindblad operator L = a^dag a sqrt(2/tau) (Section 6.2)."""
    return ballance_dephasing_coefficient(loops) * t_g_s / tau_s


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


def choi_segment_count(n_ions: int, transverse_families: int = 1) -> int:
    """2N + 1 segments for one transverse family, 4N + 1 with both (35 -> 69 at N = 17)."""
    return 2 * n_ions * transverse_families + 1


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


def yb171_detection_rate(s_o: float, gamma_rad_s: float, detuning_rad_s: float = 0.0) -> float:
    """R_o = (Gamma/18) s_o/[1 + (2/9) s_o + (2 Delta/Gamma)^2] for the 171Yb+ F = 1 -> F' = 0 cycle, saturating at Gamma/4
    (Noek 2013; Crain 2019)."""
    return (gamma_rad_s / 18.0) * s_o / (1.0 + (2.0 / 9.0) * s_o + (2.0 * detuning_rad_s / gamma_rad_s) ** 2)


def generalized_rabi_rad_s(omega_rad_s: float, detuning_rad_s: float) -> float:
    """sqrt(Omega^2 + Delta^2) in the plan's convention (Wineland's (Delta^2 + 4 Omega_W^2)^{1/2} with Omega = 2 Omega_W)."""
    return math.sqrt(omega_rad_s**2 + detuning_rad_s**2)


def two_level_population(omega_rad_s: float, detuning_rad_s: float, t_s: float) -> float:
    """P_up(t) = [Omega^2/(Omega^2 + Delta^2)] sin^2((t/2) sqrt(Omega^2 + Delta^2)) from |down> (Section 13 lineshape row)."""
    g = generalized_rabi_rad_s(omega_rad_s, detuning_rad_s)
    if g == 0.0:
        return 0.0
    return (omega_rad_s**2 / g**2) * math.sin(0.5 * g * t_s) ** 2


def sideband_rabi_rad_s(omega_rad_s: float, eta: float, n_from: int, n_to: int) -> float:
    """Omega_{n', n} = Omega |<n'|D(i eta)|n>| (Wineland 1998 Eq. 18)."""
    return omega_rad_s * abs(displacement_element_analytic(n_to, n_from, 1j * eta))


def resonant_transition_amplitude(
    omega_rad_s: float, eta: float, phi_rad: float, n_from: int, n_to: int, t_s: float
) -> complex:
    """<up, n'| U(t) |down, n> on the resonant sideband: -i e^{i phi} (M/|M|) sin(|M| Omega t/2) with M = <n'|D(i eta)|n>.

    The drive phase enters as phi + arg M = phi + (pi/2)|n' - n| (+ pi where the Laguerre polynomial is negative),
    Wineland Eq. 21 / RMP Eq. 84 in the plan's convention (Section 4.3.1).
    """
    m = displacement_element_analytic(n_to, n_from, 1j * eta)
    if m == 0.0:
        return 0.0j
    return complex(-1j * np.exp(1j * phi_rad) * (m / abs(m)) * math.sin(0.5 * abs(m) * omega_rad_s * t_s))


def carrier_debye_waller(n: int, eta: float) -> float:
    """e^{-eta^2/2} L_n(eta^2): the carrier matrix element <n|D(i eta)|n> of Fock state n."""
    return float(math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2))


def cetina_theta(b_im: float, xi_m: float, kappa_per_m2: float, nbar: float) -> float:
    """theta_im = -b_im^2 xi_m^2 (Omega''/Omega) nbar (Cetina 2022; Section 6.2)."""
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
    """Omega''/Omega of a Gaussian FIELD profile: -(2/w^2)(1 - 2 x^2/w^2); the intensity form is exactly twice (Section 6.2)."""
    return -(2.0 / waist_m**2) * (1.0 - 2.0 * offset_m**2 / waist_m**2)


def frozen_thermal_population(
    omega_rad_s: float, eta: float, nbar: float, t_s: float, *, n_max: int | None = None
) -> float:
    """sum_n P_n sin^2(Omega_n t/2) with Omega_n = Omega e^{-eta^2/2} L_n(eta^2): the carrier under a frozen thermal spectator."""
    if n_max is None:
        n_max = int(60 + 40 * nbar)
    p = thermal_populations(nbar, n_max + 1)
    return float(
        sum(
            p[n] * math.sin(0.5 * omega_rad_s * carrier_debye_waller(n, eta) * t_s) ** 2
            for n in range(n_max + 1)
        )
    )


def beta_lowest_order(a: float, q: float) -> float:
    """beta ~ sqrt(a + q^2/2), the lowest-order exponent (a check on the monodromy result)."""
    radicand = a + q * q / 2.0
    if radicand <= 0.0:
        raise ValueError(f"a + q^2/2 = {radicand} <= 0: outside the lowest-order stable region")
    return math.sqrt(radicand)


def c0_series(q: float) -> float:
    """C0 = 1 + 3q^2/16, the O(q^2) micromotion factor on eta."""
    return 1.0 + 3.0 * q * q / 16.0


def five_wire_null_height_m(a_m: float, b_m: float) -> float:
    """h = sqrt(a(a + 2b))/2 for FULL widths a (centre) and b (rails) (House 2008)."""
    if a_m <= 0.0 or b_m <= 0.0:
        raise ValueError("widths must be positive")
    return math.sqrt(a_m * (a_m + 2.0 * b_m)) / 2.0


def stretched_element_factor(J_lower: Half, J_upper: Half) -> float:
    """|<stretched upper|d_{+1}|stretched lower>| / |<J||d||J'>| = sqrt((2J+1)/(2J'+1)) (1/sqrt 2 on a 1/2 -> 3/2 line)."""
    return math.sqrt(
        (2.0 * float(as_half_integer(J_lower)) + 1.0) / (2.0 * float(as_half_integer(J_upper)) + 1.0)
    )


def decay_sum_rule_residual(build: MultiLevelBuild) -> float:
    """max over decaying sublevels of |(sum_k C_k^dagger C_k)_{ee} - Gamma_e| / Gamma_e (Section 4.2.8: the kicks are unitary).

    Zero to round-off under ``sink`` and ``renormalize``; under ``include`` it reports the dropped (untabulated) share.
    """
    decay = [
        c for k, c in enumerate(build.c_ops) if not build.dephasing_slice[0] <= k < build.dephasing_slice[1]
    ]
    if not decay:
        return 0.0
    total = decay[0].dag() * decay[0]
    for c in decay[1:]:
        total = total + c.dag() * c
    if build.space is not None:
        tot = np.asarray(total.full()).reshape(build.space.dims + build.space.dims)
        internal = np.trace(tot, axis1=1, axis2=3) / build.space.dims[1]
    else:
        internal = np.asarray(total.full())
    worst = 0.0
    for lab in build.labels:
        if lab == SINK:
            continue
        gamma = build.level_rates_rad_s.get(build.level_of(lab))
        if gamma is None:
            continue
        got = float(np.real(internal[build.index(lab), build.index(lab)]))
        worst = max(worst, abs(got - gamma) / gamma)
    return worst


def cooling_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi (Joshi Eqs. 7-10), proportional to s^2 through xi."""
    return 16.0 / 9.0 * eta**2 * gamma_rad_s * s * xi * math.cos(2.0 * phi_rad) ** 2


def heating_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi)."""
    c = math.cos(2.0 * phi_rad)
    return 2.0 / 9.0 * eta**2 * gamma_rad_s * s * (8.0 * xi**2 * c**4 + 2.0 + math.sin(2.0 * phi_rad) ** 2)


def xi_from_d1_sigma_rabi(
    omega_d1_sigma_rad_s: float, delta_rad_s: float, gamma_rad_s: float, omega_mode_rad_s: float
) -> float:
    """xi from the D1 sigma Rabi frequency one beam's full amplitude drives: the ac Stark amplitude
    Omega_1^2 Delta/(4 (Delta^2 + Gamma^2/4)) equals (1/3) Delta s with Omega_1 = sqrt(2/3) Omega_stretched."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning")
    shift = omega_d1_sigma_rad_s**2 * delta_rad_s / (4.0 * (delta_rad_s**2 + gamma_rad_s**2 / 4.0))
    return shift / omega_mode_rad_s


@dataclass(frozen=True)
class PolarizationGradientLevelC:
    """The Lindblad model as a Bloch build, with the parameters the analytic model takes."""

    model: BlochModel
    omega_d1_sigma_rad_s: float
    """The D1 sigma Rabi frequency one beam's full amplitude drives (sqrt 2 times the builder's coupling of one linear
    beam)."""
    delta_rad_s: float
    gamma_rad_s: float
    eta: float
    """k x0 of one beam on the mode."""
    xi: float
    phi_rad: float


def polarization_gradient_model(
    structure: AtomicStructure,
    pair: PolGradientBeams,
    mode: ModeSpec,
    *,
    lower: str,
    upper: str,
    levels: Sequence[str] = ("S1/2", "P1/2"),
    leak: str = "renormalize",
) -> PolarizationGradientLevelC:
    """The lin-perp-lin pair along B as a level-C model: beam phases (2 phi, 0) put the gradient phase phi at the origin;
    the S1/2 and P1/2 manifolds only (the D branch renormalized) and the three-class recoil kernel per decay channel."""
    k_hat = np.asarray(pair.beam_a.k_hat, dtype=float)
    if not np.allclose(np.abs(np.dot(k_hat, structure.b_hat)), 1.0, atol=1e-9):
        raise ValueError(
            "the analytic and Lindblad models both need the beam pair along the quantization axis"
        )
    if not np.allclose(np.abs(np.dot(k_hat, np.asarray(mode.axis))), 1.0, atol=1e-9):
        raise ValueError("the mode must lie along the beam pair (a 1D model)")
    options = MultiLevelOptions(leak=leak, recoil="minimal", beam_phases_rad=(2.0 * pair.phase_rad, 0.0))
    model = BlochModel(structure, [pair.beam_a, pair.beam_b], levels=levels, mode=mode, options=options)
    b = model.build
    coupling = next(
        (c for c in b.couplings if c.beam == 0 and c.lower == lower and c.upper == upper),
        None,
    )
    if coupling is None:
        raise ValueError(f"beam a does not couple {lower} -> {upper}; check the polarizations against B")
    gamma = b.level_rates_rad_s[b.level_of(upper)]
    omega_1 = math.sqrt(2.0) * abs(coupling.omega_rad_s)
    delta = TWO_PI * pair.detuning_hz
    return PolarizationGradientLevelC(
        model=model,
        omega_d1_sigma_rad_s=omega_1,
        delta_rad_s=delta,
        gamma_rad_s=gamma,
        eta=abs(mode.eta(pair.beam_a.k_vector())),
        xi=xi_from_d1_sigma_rabi(omega_1, delta, gamma, mode.omega_rad_s),
        phi_rad=pair.phase_rad,
    )


@dataclass(frozen=True)
class LevelCSteadyState:
    nbar: float
    rho: qt.Qobj
    populations: dict[str, float]
    fock_populations: np.ndarray
    boundary_population: float
    """Population of the top two Fock levels: the truncation check."""


def _static_mode_build(build: MultiLevelBuild, what: str) -> None:
    if build.space is None:
        raise ValueError("the level-C solve needs a build with a mode")
    if not build.static:
        raise NotImplementedError(f"the {what} is built for a consistent frame")


def level_c_steady_state(build: MultiLevelBuild) -> LevelCSteadyState:
    """The joint steady state (direct sparse solve) and its mean phonon number."""
    _static_mode_build(build, "level-C steady state")
    assert build.space is not None and isinstance(build.H, qt.Qobj)
    rho = steady_state_direct(build.H, build.c_ops)
    pn = build.space.fock_populations(rho, 0)
    return LevelCSteadyState(
        nbar=float(np.dot(np.arange(pn.size), pn)),
        rho=rho,
        populations=build.populations(rho),
        fock_populations=pn,
        boundary_population=float(pn[-2:].sum()),
    )


def level_c_relaxation_rate(build: MultiLevelBuild, *, k: int = 6) -> float:
    """W = -Re of the slowest nonzero Liouvillian eigenvalue (s^-1), by shift-invert ARPACK about zero; valid when the
    motional relaxation is the slowest mode (W << every internal rate)."""
    _static_mode_build(build, "relaxation rate")
    L = build.liouvillian().to("CSR").data.as_scipy().tocsc()
    # the shift sits just below zero: a shift AT the stationary eigenvalue makes the factorization singular
    sigma = -1e-7 * max(build.level_rates_rad_s.values())
    vals = eigs(L, k=k, sigma=sigma, return_eigenvectors=False)
    rates = np.sort(-np.real(vals))
    keep = [r for r in rates if r > 10.0 * abs(sigma)]
    if not keep:
        raise RuntimeError("no nonzero Liouvillian eigenvalue found near zero; increase k")
    return float(keep[0])
