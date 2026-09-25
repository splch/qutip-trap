"""Polarization-gradient cooling: Joshi's analytic j = 1/2 <-> 1/2 model (the cooling limits and the operating point), the
``PolGradientBeams`` methods, the recoil kernel normalization, and the Lindblad layer checked against the analytic limit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.dynamics.multilevel import ModeSpec
from qutip_trap.light.beams import PolGradientBeams
from qutip_trap.light.bloch import CoolingError
from qutip_trap.light.recoil import minimal_quadrature
from qutip_trap.prep.polarization_gradient import (
    UncooledPhaseError,
    fixed_phase_nbar,
    moving_gradient_window,
    phase_averaged_nbar,
    xi_depth,
)
from qutip_trap.species import species
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, TWO_PI
from tests.fixtures import lin_perp_lin_pair
from tests.oracles import (
    cooling_rate_per_s,
    decay_sum_rule_residual,
    heating_rate_per_s,
    level_c_relaxation_rate,
    level_c_steady_state,
    polarization_gradient_model,
    xi_from_d1_sigma_rabi,
)


def test_fixed_phase_and_phase_averaged_limits() -> None:
    """Joshi's fixed-phase <n_0> = xi + 1/(4 xi) - 1/2 (1e-12) has its minimum 1/2 at xi = 1/2, and the phase average
    its minimum sqrt(15/8) - 1/2 = 0.8693063938 (1e-10) at xi = sqrt(5/6), which the H/W quadrature reproduces
    (1e-6)."""
    assert fixed_phase_nbar(0.5) == pytest.approx(0.5, abs=1e-14)
    for xi in (0.2, 0.7, 1.3):
        assert fixed_phase_nbar(xi) == pytest.approx(xi + 1.0 / (4.0 * xi) - 0.5, rel=1e-12)
        assert fixed_phase_nbar(xi) >= 0.5
    xi_min = math.sqrt(5.0 / 6.0)
    assert phase_averaged_nbar(xi_min) == pytest.approx(0.8693063938, abs=1e-10)
    grid = np.linspace(0.3, 3.0, 2701)
    assert grid[int(np.argmin([phase_averaged_nbar(x) for x in grid]))] == pytest.approx(xi_min, abs=1e-3)
    # the phase average of H/W by quadrature equals the closed form (<cos^2 2phi> = 1/2, <cos^4> = 3/8, <sin^2> = 1/2)
    phis = np.linspace(0.0, math.pi, 20000, endpoint=False)
    h = np.mean([heating_rate_per_s(1.0, 1.0, 1.0, 0.8, p) for p in phis])
    w = np.mean([cooling_rate_per_s(1.0, 1.0, 1.0, 0.8, p) for p in phis])
    assert h / w - 0.5 == pytest.approx(phase_averaged_nbar(0.8), rel=1e-6)
    # the fixed-phase value minimizes the occupation only for xi <~ 0.612 and always maximizes the rate
    assert fixed_phase_nbar(0.5, 0.0) < fixed_phase_nbar(0.5, 0.3) and fixed_phase_nbar(
        1.0, 0.0
    ) > fixed_phase_nbar(1.0, 0.3)
    assert cooling_rate_per_s(1.0, 1.0, 1.0, 1.0, 0.0) > cooling_rate_per_s(1.0, 1.0, 1.0, 1.0, 0.3)


def test_operating_point_round_trips_window_and_bridge() -> None:
    """xi = Delta s/(3 omega) = 1.35 at Delta = 2 pi x 210 MHz, omega = 2 pi x 1088 kHz and s = 0.02098 (2e-5), and the
    moving-gradient window W < delta < omega holds at delta = 2 pi x 60 kHz."""
    delta, omega = TWO_PI * 210e6, TWO_PI * 1088e3
    s = 3.0 * omega * 1.35 / delta
    assert s == pytest.approx(0.02098, abs=2e-5)
    assert xi_depth(delta, s, omega) == pytest.approx(1.35, rel=1e-12)
    eta = (TWO_PI / 397e-9) * math.sqrt(1.0546e-34 / (2.0 * 39.9626 * ATOMIC_MASS_KG * omega))
    w_avg = 0.5 * cooling_rate_per_s(eta, TWO_PI * 21.57e6, s, 1.35, 0.0)
    assert 5e4 < w_avg < 1.5e5  # the source quotes about 6.6e4 s^-1; this eta gives 9.9e4
    assert moving_gradient_window(w_avg, TWO_PI * 60e3, omega)
    assert not moving_gradient_window(w_avg, TWO_PI * 5e3, omega) and not moving_gradient_window(
        w_avg, TWO_PI * 2e6, omega
    )
    with pytest.raises(CoolingError):
        xi_depth(-delta, s, omega)


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
    """The minimal kernel's weights have sum_q p^2 = 1 and second moment 2/5 (sigma) or 1/5 (pi), while the coherent sum
    (sum_q p)^2 is 2.7856 and 2.3314 (1e-6)."""
    for alpha, coherent in ((0.4, 2.7856406), (0.2, 2.3313708)):
        quad = minimal_quadrature(alpha)
        p = np.sqrt(quad.weights)
        assert float(np.sum(p**2)) == pytest.approx(1.0) and float(
            np.sum(p**2 * quad.nodes**2)
        ) == pytest.approx(alpha)
        assert float(np.sum(p)) ** 2 == pytest.approx(coherent, abs=1e-6)


def _ca_model(f_mode_hz: float, delta_hz: float, xi_target: float, phase: float = 0.0):
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
    """The j = 1/2 -> 1/2 builder has 9 recoil-resolved operators (sum rule 1e-12), and at xi = 1, Delta = 500 MHz and a
    4 MHz mode level C reaches Joshi's 0.75 to 2 % and his relaxation rate to 20 %, the gradient node staying 5x
    hotter."""
    lc = _ca_model(4.0e6, 500e6, 1.0)
    b = lc.model.build
    assert len(b.c_ops) == 9 and decay_sum_rule_residual(b) < 1e-12
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
