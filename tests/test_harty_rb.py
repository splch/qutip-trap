"""Harty et al. 2014 microwave randomized benchmarking (PLAN.md Sections 4.3.3, 9.2, 9.12) with the paper's own error model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.light.microwave import effective_detuning_hz
from qutip_trap.validation.harty_rb import (
    HartyParameters,
    delay_propagator,
    pulse_propagator,
    random_sequences,
    simulate_epg_sets,
    simulate_sequences,
)


def test_ac_zeeman_sign_and_pulse_parameters() -> None:
    """delta_eff = delta - delta_ac = +4.5 - (-1.0) = +5.5 Hz (Section 9.2 'ac Zeeman sign'); Omega/2pi = 20.66 kHz for a 12.1 us pi/2."""
    assert effective_detuning_hz(4.5, -1.0) == pytest.approx(5.5)
    p = HartyParameters()
    assert p.rabi_rad_s / (2 * math.pi) == pytest.approx(20661.157, rel=1e-6)
    u = pulse_propagator(0.0, p, detuning_hz=0.0, rabi_error=0.0)
    assert np.allclose(u @ u, np.array([[0, -1j], [-1j, 0]]))  # two ideal pi/2 pulses = a pi pulse
    assert np.allclose(delay_propagator(1.0, p, detuning_hz=0.0), np.eye(2))


def test_zero_error_sequences_have_zero_error() -> None:
    rng = np.random.default_rng(5)
    paulis, cliffords, targets = random_sequences(rng, 40, 300)
    errs = simulate_sequences(paulis, cliffords, targets, HartyParameters(), detuning_hz=0.0, rabi_error=0.0)
    assert np.max(np.abs(errs)) < 1e-12


def test_harty_error_per_gate_budget() -> None:
    """Section 9.12: 0.81(14)e-6 over 500 sets of 32 sequences of 2000 gates with 12.1 us pulses, 14 us dead times,
    +4.5 Hz and a constant 5e-4 Rabi error; pulse area alone 0.3e-6; detuning alone (+5.5 Hz effective) 0.7e-6.

    The paper does not state how many delays stand in for an identity or z Pauli; with one delay per replaced pi/2 pulse
    (two slots, each followed by the phase-switching dead time) the model gives 0.77(11)e-6, 0.31e-6 and 0.59 to 0.80e-6
    (the 5.5 Hz applied during pulses only, or everywhere); with a single delay it gives 0.53e-6 (recorded in the ledger).
    """
    p = HartyParameters()
    both = simulate_epg_sets(16, params=p, seed=3)
    assert both.mean() == pytest.approx(0.81e-6, abs=0.14e-6)
    assert 0.05e-6 < both.std(ddof=1) < 0.25e-6
    area = simulate_epg_sets(4, params=p, detuning_hz=0.0, seed=2).mean()
    assert area == pytest.approx(0.3e-6, abs=0.05e-6)
    det_pulses = simulate_epg_sets(
        6, params=HartyParameters(ac_zeeman_hz=-1.0), rabi_error=0.0, seed=1
    ).mean()
    det_everywhere = simulate_epg_sets(6, params=p, detuning_hz=5.5, rabi_error=0.0, seed=1).mean()
    assert det_pulses < 0.7e-6 < det_everywhere
    assert det_pulses == pytest.approx(0.7e-6, abs=0.15e-6) or det_everywhere == pytest.approx(
        0.7e-6, abs=0.15e-6
    )
    one_slot = simulate_epg_sets(4, params=HartyParameters(identity_slots=1), seed=3).mean()
    assert one_slot < both.mean()
