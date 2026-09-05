"""Polarization-gradient cooling (PLAN.md Section 4.2.4; M3): the analytic j = 1/2 <-> 1/2 model of Joshi et al. 2020 (Section 9.15 rows
"PGC cooling limits", "PGC recoil-heating angular factor", "PGC operating-point round trips", "Three-axis Lamb-Dicke triple",
"Saturation-intensity convention", "Recoil-kernel normalization"), the Appendix E ``PolGradientBeams`` methods, the static-gradient
raise (Section 9.17), and the Lindblad layer through the M3a builder checked against the analytic limit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import PolGradientBeams
from qutip_trap.dynamics.multilevel import ModeSpec, decay_sum_rule_residual
from qutip_trap.light.bloch import CoolingError
from qutip_trap.light.recoil import minimal_quadrature
from qutip_trap.prep.level_c import level_c_relaxation_rate, level_c_steady_state
from qutip_trap.prep.polarization_gradient import (
    UncooledPhaseError,
    cooling_rate_per_s,
    detailed_balance_populations,
    fixed_phase_minimum,
    fixed_phase_nbar,
    heating_rate_per_s,
    lin_perp_lin_pair,
    moving_gradient_window,
    multi_ion_fit_offset,
    phase_averaged_minimum,
    phase_averaged_nbar,
    polarization_gradient_model,
    potentials_rad_s,
    pumping_rates_per_s,
    recoil_heating_terms,
    saturation_bridge,
    saturation_for_xi,
    static_gradient_nbar,
    three_axis_lamb_dicke,
    xi_depth,
    xi_from_d1_sigma_rabi,
)
from qutip_trap.species import species
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, TWO_PI


def test_fixed_phase_and_phase_averaged_limits() -> None:
    """<n_0> = xi + 1/(4 xi) - 1/2 with minimum exactly 1/2 at xi = 1/2; phase-averaged (3/4) xi + 5/(8 xi) - 1/2 with minimum sqrt(15/8) - 1/2 =
    0.8693063938 at xi = sqrt(5/6) = 0.9128709292; the multi-ion fit offset n_0 = sqrt(15/8) = 1.369306 (Section 9.15)."""
    assert fixed_phase_minimum() == (0.5, 0.5)
    assert fixed_phase_nbar(0.5) == pytest.approx(0.5, abs=1e-14)
    for xi in (0.2, 0.7, 1.3):
        assert fixed_phase_nbar(xi) == pytest.approx(xi + 1.0 / (4.0 * xi) - 0.5, rel=1e-12)
        assert fixed_phase_nbar(xi) >= 0.5
    xi_min, n_min = phase_averaged_minimum()
    assert xi_min == pytest.approx(0.9128709292, abs=1e-10) and n_min == pytest.approx(
        0.8693063938, abs=1e-10
    )
    assert phase_averaged_nbar(xi_min) == pytest.approx(n_min, abs=1e-13)
    grid = np.linspace(0.3, 3.0, 2701)
    assert grid[int(np.argmin([phase_averaged_nbar(x) for x in grid]))] == pytest.approx(xi_min, abs=1e-3)
    # the phase average of H/W by quadrature equals the closed form (<cos^2 2phi> = 1/2, <cos^4> = 3/8, <sin^2> = 1/2)
    phis = np.linspace(0.0, math.pi, 20000, endpoint=False)
    h = np.mean([heating_rate_per_s(1.0, 1.0, 1.0, 0.8, p) for p in phis])
    w = np.mean([cooling_rate_per_s(1.0, 1.0, 1.0, 0.8, p) for p in phis])
    assert h / w - 0.5 == pytest.approx(phase_averaged_nbar(0.8), rel=1e-6)
    assert multi_ion_fit_offset() == pytest.approx(math.sqrt(15.0 / 8.0))
    # the fixed-phase value minimizes the occupation only for xi <~ 0.612 and always maximizes the rate
    assert fixed_phase_nbar(0.5, 0.0) < fixed_phase_nbar(0.5, 0.3) and fixed_phase_nbar(
        1.0, 0.0
    ) > fixed_phase_nbar(1.0, 0.3)
    assert cooling_rate_per_s(1.0, 1.0, 1.0, 1.0, 0.0) > cooling_rate_per_s(1.0, 1.0, 1.0, 1.0, 0.3)


def test_isotropic_alpha_is_the_only_consistent_recoil_factor() -> None:
    """H_carr + H_sb with alpha = 1/3 equals the non-xi^2 part (2/9) eta^2 Gamma s (2 + sin^2 2phi) exactly; alpha = 2/5 fails by 5 % (Section 9.15)."""
    for phi in (0.0, 0.3, 1.1):
        hc, hs = recoil_heating_terms(0.1, 2.0, 0.05, phi)
        assert hc + hs == pytest.approx(
            2.0 / 9.0 * 0.01 * 2.0 * 0.05 * (2.0 + math.sin(2.0 * phi) ** 2), rel=1e-14
        )
    hc, hs = recoil_heating_terms(1.0, 1.0, 1.0, 0.0, alpha=0.4)
    assert (hc + hs) / (4.0 / 9.0) == pytest.approx(1.05, abs=1e-12)


def test_potentials_pumping_rates_and_detailed_balance_carry_the_same_sign_of_the_sine() -> None:
    """U_+- = (1/3) Delta s (1 -+ sin), Gamma_{+- -> -+} = (1/9) Gamma s (1 -+ sin): fastest pumping out of a state at its own maximum; the
    z-substituted rates reproduce p_+- = (1/2)(1 +- sin) by detailed balance (Section 4.2.4)."""
    k, phi = 2.0, 0.2
    for z in (0.0, 0.11, 0.37):
        u_plus, u_minus = potentials_rad_s(z, k, phi, 5.0, 0.3)
        g_pm, g_mp = pumping_rates_per_s(z, k, phi, 7.0, 0.3)
        p_plus, p_minus = detailed_balance_populations(z, k, phi)
        s = math.sin(2.0 * k * z + 2.0 * phi)
        assert u_plus - u_minus == pytest.approx(-2.0 / 3.0 * 5.0 * 0.3 * s)
        assert g_pm / g_mp == pytest.approx((1.0 - s) / (1.0 + s))
        assert p_plus * g_pm == pytest.approx(p_minus * g_mp, rel=1e-12)  # detailed balance
        assert (u_plus > u_minus) == (g_pm > g_mp) or s == 0.0


def test_operating_point_round_trips_window_and_bridge() -> None:
    """xi = Delta s/(3 omega) at Delta = 2 pi x 210 MHz, omega = 2 pi x 1088 kHz, xi = 1.35 returns s = 0.02098; the moving-gradient window
    W < delta = 2 pi x 60 kHz < omega holds at that point; Ejtemaee's s0 = 11-15 at 310 MHz maps to s_Joshi = 0.016-0.023 < 0.07."""
    delta, omega = TWO_PI * 210e6, TWO_PI * 1088e3
    s = saturation_for_xi(1.35, delta, omega)
    assert s == pytest.approx(0.02098, abs=2e-5)
    assert xi_depth(delta, s, omega) == pytest.approx(1.35, rel=1e-12)
    eta = (TWO_PI / 397e-9) * math.sqrt(1.0546e-34 / (2.0 * 39.9626 * ATOMIC_MASS_KG * omega))
    w_avg = 0.5 * cooling_rate_per_s(eta, TWO_PI * 21.57e6, s, 1.35, 0.0)
    assert 5e4 < w_avg < 1.5e5  # the plan quotes ~6.6e4 s^-1 from the source; this eta gives 9.9e4
    assert moving_gradient_window(w_avg, TWO_PI * 60e3, omega)
    assert not moving_gradient_window(w_avg, TWO_PI * 5e3, omega) and not moving_gradient_window(
        w_avg, TWO_PI * 2e6, omega
    )
    for s0 in (11.0, 15.0):
        assert 0.016 < saturation_bridge(s0, TWO_PI * 19.6e6, TWO_PI * 310e6) < 0.023
    with pytest.raises(CoolingError):
        xi_depth(-delta, s, omega)


def test_three_axis_lamb_dicke_triple_of_ejtemaee_and_haljan() -> None:
    """{1/2, 1/2, 1/sqrt 2} x 2 pi/369.5 nm at (0.790, 0.766, 0.525) MHz with x0 = sqrt(hbar/(2 m omega)): eta = (0.05201, 0.05282, 0.09023) at 170.9363 u
    (the source's 0.052, 0.053, 0.090); hbar/(4 m omega) would give 0.0368 (negative test)."""
    etas = three_axis_lamb_dicke(TWO_PI / 369.5e-9, 170.9363 * ATOMIC_MASS_KG, (0.790e6, 0.766e6, 0.525e6))
    assert etas == (
        pytest.approx(0.05201, abs=1e-5),
        pytest.approx(0.05282, abs=1e-5),
        pytest.approx(0.09023, abs=1e-5),
    )
    assert etas[0] / math.sqrt(2.0) == pytest.approx(0.0368, abs=1e-4)
    assert 0.09023 * math.sqrt(21.0) == pytest.approx(0.413, abs=2e-3)  # the Lamb-Dicke guard at nbar = 20
    with pytest.raises(ValueError):
        three_axis_lamb_dicke(1.0, 1.0, (1.0, 1.0, 1.0), (0.5, 0.5, 0.5))


def test_static_gradient_raises_at_a_node_and_averages_per_ion_otherwise() -> None:
    """Section 9.17: an ion at phi = pi/4 raises the uncooled-mode error; the moving-gradient 0.8693 is returned only inside W < delta < omega."""
    with pytest.raises(UncooledPhaseError):
        static_gradient_nbar(0.5, [0.0, math.pi / 4.0])
    values = static_gradient_nbar(0.5, [0.0, 0.3])
    assert values[0] == pytest.approx(0.5) and values[1] > values[0]


def test_pol_gradient_beams_methods_delegate_to_the_analytic_model() -> None:
    pair = lin_perp_lin_pair(397e-9, 1e-3, 20e-6, 210e6, phase_rad=0.0)
    assert pair.xi(1088e3, 0.02098) == pytest.approx(1.35, abs=1e-3)
    fixed, averaged = pair.limits(1088e3, 0.02098)
    xi = pair.xi(1088e3, 0.02098)
    assert fixed == pytest.approx(fixed_phase_nbar(xi)) and averaged == pytest.approx(phase_averaged_nbar(xi))
    assert pair.moving_gradient_ok(10e3, 1088e3) is False  # beat_hz = 0
    moving = lin_perp_lin_pair(397e-9, 1e-3, 20e-6, 210e6, beat_hz=60e3)
    assert moving.moving_gradient_ok(10e3, 1088e3) and not moving.moving_gradient_ok(100e3, 1088e3)
    yb_scheme = PolGradientBeams(pair.beam_a, pair.beam_b, 310e6, 0.0, 0.0, "F1_to_F0")
    with pytest.raises(ValueError):
        yb_scheme.limits(0.79e6, 0.02)
    with pytest.raises(ValueError):
        lin_perp_lin_pair(397e-9, 1e-3, 20e-6, -210e6)
    node = lin_perp_lin_pair(397e-9, 1e-3, 20e-6, 210e6, phase_rad=math.pi / 4.0)
    with pytest.raises(UncooledPhaseError):
        node.limits(1088e3, 0.02098)


def test_minimal_kernel_weights_have_unit_norm_and_the_printed_coherent_sum_fails() -> None:
    """sum_q p_mq^2 = 1 and sum_q p_mq^2 (k_q/k)^2 = 2/5 (sigma), 1/5 (pi); the coherent q-sum (sum_q p_mq)^2 = 2.7856 (sigma) and 2.3314 (pi)
    (Section 13 "Recoil kernel discretization")."""
    for alpha, coherent in ((0.4, 2.7856406), (0.2, 2.3313708)):
        quad = minimal_quadrature(alpha)
        p = np.sqrt(quad.weights)
        assert float(np.sum(p**2)) == pytest.approx(1.0) and float(
            np.sum(p**2 * quad.nodes**2)
        ) == pytest.approx(alpha)
        assert float(np.sum(p)) ** 2 == pytest.approx(coherent, abs=1e-6)


def _ca_model(f_mode_hz: float, delta_hz: float, xi_target: float, phase: float = 0.0):  # type: ignore[no-untyped-def]
    ca = species("40Ca+")
    st = AtomicStructure(ca, 1e-6, (0.0, 0.0, 1.0))
    line = ca.transition("S1/2-P1/2")
    mode = ModeSpec(TWO_PI * f_mode_hz, ca.mass_u * ATOMIC_MASS_KG, (0.0, 0.0, 1.0), d=20, expected_n_max=6)
    lam = TWO_PI * C_M_PER_S / (TWO_PI * (line.frequency_hz + delta_hz))
    probe = polarization_gradient_model(
        st, lin_perp_lin_pair(lam, 1e-3, 20e-6, delta_hz), mode, lower="S1/2 mJ=-1/2", upper="P1/2 mJ=1/2"
    )
    gamma = probe.gamma_rad_s
    om_target = math.sqrt(
        xi_target * 4.0 * mode.omega_rad_s * (probe.delta_rad_s**2 + gamma**2 / 4.0) / probe.delta_rad_s
    )
    power = 1e-3 * (om_target / probe.omega_d1_sigma_rad_s) ** 2
    return polarization_gradient_model(
        st,
        lin_perp_lin_pair(lam, power, 20e-6, delta_hz, phase_rad=phase),
        mode,
        lower="S1/2 mJ=-1/2",
        upper="P1/2 mJ=1/2",
    )


@pytest.mark.slow
def test_lindblad_layer_has_nine_recoil_resolved_operators_and_approaches_the_analytic_limit() -> None:
    """The M3a builder gives 3 polarization channels x 3 recoil classes = 9 operators for the j = 1/2 -> 1/2 line (Joshi's twelve count the two pi
    decays separately; the Wigner-Eckart form sums them into one operator with the same second moments), the sum rule holds to 1e-12,
    and at xi = 1, Delta = 500 MHz, omega = 2 pi x 4 MHz (eta = 0.089, s ~ 0.02) the level-C steady state 0.756 matches the analytic
    xi + 1/(4 xi) - 1/2 = 0.75 to 1 %, the relaxation rate to 15 %, and the gradient node (phi = pi/4) leaves the ion hot."""
    lc = _ca_model(4.0e6, 500e6, 1.0)
    b = lc.model.build
    assert lc.n_collapse_operators == 9 and decay_sum_rule_residual(b) < 1e-12
    alphas = {q: a for ch in b.channels for q, a in ch.alpha.items()}
    assert alphas == {-1: pytest.approx(0.4), 0: pytest.approx(0.2), 1: pytest.approx(0.4)}
    assert lc.xi == pytest.approx(1.0, rel=1e-9) and lc.eta == pytest.approx(0.089, abs=1e-3)
    ss = level_c_steady_state(b)
    assert ss.nbar == pytest.approx(fixed_phase_nbar(1.0), rel=0.02) and ss.boundary_population < 1e-6
    s_joshi = 1.5 * lc.omega_d1_sigma_rad_s**2 / 2.0 / (lc.delta_rad_s**2 + lc.gamma_rad_s**2 / 4.0)
    assert level_c_relaxation_rate(b) == pytest.approx(
        cooling_rate_per_s(lc.eta, lc.gamma_rad_s, s_joshi, lc.xi, 0.0), rel=0.2
    )
    node = _ca_model(4.0e6, 500e6, 1.0, phase=math.pi / 4.0)
    assert level_c_steady_state(node.model.build).nbar > 5.0 * ss.nbar
    assert xi_from_d1_sigma_rabi(
        lc.omega_d1_sigma_rad_s, lc.delta_rad_s, lc.gamma_rad_s, TWO_PI * 4.0e6
    ) == pytest.approx(1.0, rel=1e-9)
