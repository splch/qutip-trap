"""Frequency-comb Raman drives (PLAN.md Sections 4.3.7, 9.15) against the numbers of check_comb.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import CombSpec
from qutip_trap.light.comb import (
    FIELD_FWHM_OVER_TAU,
    INTENSITY_FWHM_OVER_TAU,
    ZETA3_COEFFICIENT,
    amplitude_noise_error,
    harmonic_gain,
    lock_residual_factor,
    pulse_count_map,
    rosen_zener_ceiling,
    tau_field_from,
)

LEE = CombSpec(120e6, 14e-12, "field_sech", 105, 0.0)
APB = CombSpec(80e6, 10e-12, "field_sech", 158, 0.0)


def test_pair_weight_uses_the_half_argument() -> None:
    """sech(pi l nu_rep tau) = 0.863911 at l = 105 against the wrong sech(2 pi l ...) = 0.595332, a 27% error; the closed
    form is itself about 5% high against the exact convolution."""
    assert LEE.pair_weight(105) == pytest.approx(0.863911, abs=1e-6)
    assert 1.0 / math.cosh(2 * math.pi * 105 * 120e6 * 14e-12) == pytest.approx(0.595332, abs=1e-6)
    exact = LEE.pair_sum_exact(105, n_teeth=200_000)
    assert 0.03 < LEE.pair_weight(105) / exact - 1.0 < 0.07


def test_sum_rule_is_asymptotic_and_renormalized() -> None:
    assert LEE.sum_rule_raw(200_000) == pytest.approx(1.0, abs=1e-9)
    hot = CombSpec(1e9, 1e-9, "field_sech", 1, 0.0)  # nu_rep tau = 1: the prefactor sum is 3.14, not 1
    assert hot.sum_rule_raw(4000) == pytest.approx(3.14, abs=0.01)
    assert np.sum(hot.tooth_amplitudes(4000) ** 2) == pytest.approx(1.0, abs=1e-12)


def test_comb_factor_convergence_sweep() -> None:
    """C_00,10 partial sums 0.746343, 0.453162, 0.450920, 0.669087, 0.940671, 0.943780, 0.943789 for |k| <= 0, 5, 10, 100,
    500, 1000, 5000: the |k| ~ 5 to 10 plateau converges falsely a factor 2.093 low."""
    expected = {
        0: 0.746343,
        5: 0.453162,
        10: 0.450920,
        100: 0.669087,
        500: 0.940671,
        1000: 0.943780,
        5000: 0.943789,
    }
    for lmax, val in expected.items():
        assert LEE.comb_factor(12.642821e9, l_max=lmax) == pytest.approx(val, abs=2e-6), lmax
    assert LEE.comb_factor(12.642821e9, l_max=5000) / LEE.comb_factor(12.642821e9, l_max=10) == pytest.approx(
        2.093, abs=2e-3
    )
    # j = 0 is the operative case for the Zeeman partners: C_10,11 = C_10,1-1 = 0.988841 exactly
    assert LEE.comb_factor(7e6, l_max=20000) == pytest.approx(0.988841, abs=2e-6)
    assert LEE.comb_factor(-7e6, l_max=20000) == pytest.approx(LEE.comb_factor(7e6, l_max=20000), rel=1e-12)


def test_fourth_order_single_sum_guards_and_converges() -> None:
    levels = [0.0, 12.642821e9, 12.642821e9 + 7e6, 12.642821e9 - 7e6]
    shift = LEE.stark4_hz(levels, couplings_hz=[0.0, 1e6, 0.5e6, 0.5e6], n_index=0)
    assert shift.shape == (1,) and np.isfinite(shift[0])
    on_tooth = CombSpec(1e6, 14e-12, "field_sech", 1, 0.0)
    with pytest.raises(ZeroDivisionError):
        on_tooth.stark4_hz(
            [0.0, 5e6], couplings_hz=[0.0, 1e3], n_index=0
        )  # 5 MHz sits exactly on a beat note
    with pytest.raises(ValueError):
        CombSpec(120e6, 14e-12, "field_sech", 105, 0.0, pair_order_max=100)


def test_resonance_lock_and_beat_note_arithmetic() -> None:
    """Hayes q = 156.5090, 313.0181, 469.5213 (integer only at the middle rate); Islam's lock n = 157 with |Delta nu_M| = 11.381 MHz."""
    for nu, q in ((80.78e6, 156.5090), (40.39e6, 313.0181), (26.927e6, 469.5213)):
        assert CombSpec(nu, 10e-12, "field_sech", 1, 0.0).resonance_index(12.6428e9) == pytest.approx(
            q, abs=5e-4
        )
    islam = CombSpec(80.6e6, 10e-12, "field_sech", 157, 0.0)
    j, off = islam.solve_offset(12.642819e9)
    assert j == 157 and off == pytest.approx(-11.381e6, abs=2e3)
    assert 157 * 80.6e6 - 12.438e9 == pytest.approx(216.200e6, abs=1e3)
    assert 12.642819e9 - 12.438e9 == pytest.approx(204.819e6, abs=1e3)
    assert islam.beat_note_hz(157) == pytest.approx(157 * 80.6e6)


def test_rosen_zener_widths_pulse_count_and_noise_budget() -> None:
    nu_hf = 12.642812118466e9
    assert rosen_zener_ceiling(nu_hf, 14.8e-12) == pytest.approx(0.720860, abs=1e-6)
    assert rosen_zener_ceiling(nu_hf, 7.6e-12) == pytest.approx(0.914142, abs=1e-6)
    assert rosen_zener_ceiling(2 * math.pi * nu_hf, 14.8e-12) == pytest.approx(0.002474, abs=1e-6), (
        "the angular misreading"
    )
    assert FIELD_FWHM_OVER_TAU == pytest.approx(1.6768, abs=1e-4)
    assert INTENSITY_FWHM_OVER_TAU == pytest.approx(1.1222, abs=1e-4)
    assert tau_field_from(1.6768e-12, "field_fwhm") == pytest.approx(1e-12, rel=1e-3)
    n, om0 = pulse_count_map(80e6, 108e-6, 1e-3)
    assert n == pytest.approx(8640.0) and om0 == pytest.approx(8e4)
    assert harmonic_gain(1.0 / 60.0, 157) == pytest.approx(2.617, abs=1e-3)
    assert lock_residual_factor(167, 157, "lower") == 10 and lock_residual_factor(167, 157, "upper") == 324
    assert amplitude_noise_error(10 ** (-115 / 10), 600e3, 1e-3) == pytest.approx(5.6179e-3, rel=1e-4)
    assert amplitude_noise_error(10 ** (-115 / 10), 2 * math.pi * 600e3, 1e-3) / 5.6179e-3 == pytest.approx(
        (2 * math.pi) ** 2, rel=1e-3
    )
    assert ZETA3_COEFFICIENT == pytest.approx(7 * 1.2020569031595942 / math.pi**2, rel=1e-10)


def test_tone_set_is_cut_at_the_gate_window() -> None:
    tones = APB.tones(12.642812e9, None, omega0_hz=1e6, gate_time_s=100e-6)
    assert len(tones) == 1
    all_tones = APB.tones(12.642812e9, None, omega0_hz=1e6, gate_time_s=None)
    assert len(all_tones) == 2 * APB.pair_order_max  # the self-beat l = 0 pair is excluded
    resonant = min(all_tones, key=lambda t: abs(float(t.detuning_hz)))  # type: ignore[arg-type]
    assert float(resonant.envelope_hz) == pytest.approx(1e6 * APB.pair_weight(158), rel=1e-12)  # type: ignore[arg-type]
    guards = APB.guards(12.6428e9, 0.1, 0.1, 108e-6, 1.64e6)
    assert all(guards.values())
    assert not APB.guards(12.6428e9, 0.2233, 40.0, 108e-6, 0.5e6)["lamb_dicke"]
