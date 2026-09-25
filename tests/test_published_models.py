"""The published models the app and the run pipeline use (PLAN.md Section 4): Harty et al.'s
randomized benchmarking with the paper's own error model, and the Molmer-Sorensen, Ballance and Roos closed forms."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import jv

from qutip_trap.light.microwave import effective_detuning_hz
from qutip_trap.units import TWO_PI
from qutip_trap.validation.harty_rb import (
    HartyParameters,
    delay_propagator,
    pulse_propagator,
    random_sequences,
    simulate_epg_sets,
    simulate_sequences,
)
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_thermal_error,
    ms_alpha,
    ms_closure_ratio,
    roos_force_saturation,
)


def test_ac_zeeman_sign_and_pulse_parameters() -> None:
    """delta_eff = delta - delta_ac = +4.5 - (-1.0) = +5.5 Hz; Omega/2pi = 20.66 kHz for a 12.1 us pi/2."""
    assert effective_detuning_hz(4.5, -1.0) == pytest.approx(5.5)
    p = HartyParameters()
    assert p.rabi_rad_s / (2 * math.pi) == pytest.approx(20661.157, rel=1e-6)
    ideal = replace(p, detuning_hz=0.0, rabi_error=0.0)
    u = pulse_propagator(0.0, ideal)
    assert np.allclose(u @ u, np.array([[0, -1j], [-1j, 0]])), "two ideal pi/2 pulses are a pi pulse"
    assert np.allclose(delay_propagator(1.0, ideal), np.eye(2))


def test_zero_error_sequences_have_zero_error() -> None:
    paulis, cliffords, targets = random_sequences(np.random.default_rng(5), 40, 300)
    errs = simulate_sequences(paulis, cliffords, targets, HartyParameters(detuning_hz=0.0, rabi_error=0.0))
    assert np.max(np.abs(errs)) < 1e-12


def test_harty_error_per_gate_budget() -> None:
    """Harty prints 0.81(14)e-6 for 12.1 us pulses, 14 us dead times, +4.5 Hz and a 5e-4 Rabi error; pulse area alone 0.3e-6;
    detuning alone (+5.5 Hz effective) 0.7e-6. With one delay per replaced pi/2 pulse the model gives 0.75(12)e-6, 0.31e-6,
    and 0.59e-6 (the 5.5 Hz during pulses only) to 0.80e-6 (everywhere); a single delay gives 0.49e-6."""
    p = HartyParameters()
    both = simulate_epg_sets(16, params=p, seed=3)
    assert both.mean() == pytest.approx(0.81e-6, abs=0.14e-6)
    assert 0.05e-6 < both.std(ddof=1) < 0.25e-6
    assert simulate_epg_sets(4, params=p, detuning_hz=0.0, seed=2).mean() == pytest.approx(
        0.3e-6, abs=0.05e-6
    )
    det_pulses = simulate_epg_sets(
        6, params=HartyParameters(ac_zeeman_hz=-1.0), rabi_error=0.0, seed=1
    ).mean()
    det_everywhere = simulate_epg_sets(6, params=p, detuning_hz=5.5, rabi_error=0.0, seed=1).mean()
    assert det_pulses < 0.7e-6 < det_everywhere
    assert det_pulses == pytest.approx(0.7e-6, abs=0.15e-6)
    assert det_everywhere == pytest.approx(0.7e-6, abs=0.15e-6)
    assert simulate_epg_sets(4, params=HartyParameters(identity_slots=1), seed=3).mean() < both.mean()


def test_ms_displacement_carries_the_half_of_the_sideband_coupling() -> None:
    """The sideband coupling is eta Omega/2, so |alpha(eps t = pi)| = eta Omega/eps at the maximally entangling closure
    (not the 2x and 1/2x of the withdrawn readings eta Omega/eps and eta Omega/(4 eps) of the prefactor), and alpha returns
    to zero at eps t = 2 pi."""
    eta, eps = 0.05, TWO_PI * 10e3
    omega = ms_closure_ratio(1) * eps / eta
    assert abs(ms_alpha(eta, omega, eps, math.pi / eps)) == pytest.approx(eta * omega / eps, rel=1e-12)
    assert abs(ms_alpha(eta, omega, eps, 2.0 * math.pi / eps)) < 1e-15


def test_ballance_thermal_error_reproduces_the_printed_cooling_floor() -> None:
    """eps_nbar at eta = 0.123 is 1.1e-5 / 2.8e-5 / 3.9e-5 / 0.238 at nbar = 0.02 / 0.05 / 0.06 / 15 as the plan prints
    them, to the 20% of a consistency anchor (the printed digits carry Ballance's rounding of eta)."""
    for nbar, value in {0.02: 1.1e-5, 0.05: 2.8e-5, 0.06: 3.9e-5, 15.0: 0.238}.items():
        assert ballance_thermal_error(0.123, nbar) == pytest.approx(value, rel=0.2), nbar


def test_roos_bessel_saturation_of_the_force() -> None:
    """Roos Eq. 17: J_0(x) + J_2(x) = 2 J_1(x)/x at x = 4 Omega/delta; at Omega/mu ~ 0.03 the force is reduced by 1.80e-3."""
    for ratio, value in ((0.03, 0.998201080), (0.1, 0.980132890), (0.2, 0.922105115)):
        assert roos_force_saturation(ratio, 1.0) == pytest.approx(value, rel=1e-9)
        assert roos_force_saturation(ratio, 1.0) == pytest.approx(
            2.0 * float(jv(1, 4.0 * ratio)) / (4.0 * ratio), rel=1e-12
        )
    assert 1.0 - roos_force_saturation(0.03, 1.0) == pytest.approx(1.80e-3, rel=0.01)
    assert roos_force_saturation(0.0, 1.0) == pytest.approx(1.0, rel=1e-12), "no saturation at zero drive"
