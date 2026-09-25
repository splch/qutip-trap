"""Frequency-comb Raman drives (PLAN.md Sections 4.3.7, 9.15) against the numbers of check_comb.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.light.comb import (
    FIELD_FWHM_OVER_TAU,
    INTENSITY_FWHM_OVER_TAU,
    SUM_DEPTH_HALF_WIDTHS,
    ZETA3_COEFFICIENT,
    CombSpec,
    amplitude_noise_error,
    harmonic_gain,
    lock_residual_factor,
    pulse_count_map,
    rosen_zener_ceiling,
    tau_field_from,
)
from qutip_trap.units import TWO_PI

LEE = CombSpec(120e6, 14e-12, "field_sech", 105, 0.0)
APB = CombSpec(80e6, 10e-12, "field_sech", 158, 0.0)
NU_Q = 12.642812118466e9
"""The 171Yb+ zero-field clock splitting (Section 13): the APB operating point's qubit frequency."""


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
    assert len(all_tones) == 2 * APB.sum_depth  # the self-beat l = 0 pair is excluded
    resonant = min(all_tones, key=lambda t: abs(float(t.detuning_hz)))  # type: ignore[arg-type]
    assert float(resonant.envelope_hz) == pytest.approx(1e6 * APB.pair_weight(158), rel=1e-12)  # type: ignore[arg-type]
    guards = APB.guards(12.6428e9, 0.1, 0.1, 108e-6, 1.64e6)
    assert all(guards.values())
    assert not APB.guards(12.6428e9, 0.2233, 40.0, 108e-6, 0.5e6)["lamb_dicke"]
    # the two clauses that were literal True are now absent unless their inputs are supplied, and named as missing
    assert "adiabatic_elimination" not in guards and "small_pulse_area" not in guards
    assert len(APB.unevaluated_guards()) == 2
    with_inputs = APB.guards(
        12.6428e9,
        0.1,
        0.1,
        108e-6,
        1.64e6,
        detuning_hz=22.11e12,
        fine_structure_hz=67e12,
        theta_per_pulse_rad=1e-3,
    )
    assert with_inputs["adiabatic_elimination"] and with_inputs["small_pulse_area"]
    assert APB.unevaluated_guards(detuning_hz=22.11e12, theta_per_pulse_rad=1e-3) == ()
    # 1/tau = 100 GHz against a 1 THz detuning fails the "1/tau << |Delta|" clause
    assert not APB.guards(12.6428e9, 0.1, 0.1, 108e-6, 1.64e6, detuning_hz=1e12, theta_per_pulse_rad=1e-3)[
        "adiabatic_elimination"
    ]
    assert not APB.guards(12.6428e9, 0.1, 0.1, 108e-6, 1.64e6, detuning_hz=22.11e12, theta_per_pulse_rad=0.5)[
        "small_pulse_area"
    ]


def test_sum_depth_comes_from_the_envelope_and_covers_both_operating_points() -> None:
    """l_max = ceil(8/(pi nu_rep tau)): 1516 at the Lee point and 3184 at the APB one, so the plan's ">~ 1000" is a
    consequence of the envelope and not a literal. The literal 1200 of Appendix E refused the APB point outright."""
    assert LEE.sum_half_width == pytest.approx(189.470, abs=1e-3)
    assert APB.sum_half_width == pytest.approx(397.887, abs=1e-3)
    assert LEE.sum_depth == math.ceil(SUM_DEPTH_HALF_WIDTHS * LEE.sum_half_width) == 1516
    assert APB.sum_depth == math.ceil(SUM_DEPTH_HALF_WIDTHS * APB.sum_half_width) == 3184
    assert LEE.sum_depth > 1000 and APB.sum_depth > 1000
    # both operating points now converge under the doubling check; APB used to raise at pair_order_max = 1200
    assert LEE.stark4_hz([0.0, 12.642821e9], couplings_hz=[0.0, 1e6])[0] == pytest.approx(5510.08, rel=1e-5)
    assert APB.stark4_hz([0.0, NU_Q], couplings_hz=[0.0, 1e6])[0] == pytest.approx(79680.65, rel=1e-5)
    with pytest.raises(ValueError, match="plateaus falsely"):
        CombSpec(80e6, 10e-12, "field_sech", 158, 0.0, pair_order_max=1200)
    # comb_factor now carries the same doubling check, which it had none of before
    assert LEE.comb_factor(12.642821e9, validate=True) == pytest.approx(0.943789, abs=2e-6)
    with pytest.raises(ValueError, match="comb_factor has not converged"):
        LEE.comb_factor(12.642821e9, l_max=10, validate=True)


def test_the_resonant_comb_is_refused_rather_than_returning_a_huge_shift() -> None:
    """A comb whose rep rate nearly divides the splitting (nu_rep = nu_q/158, resonant to 1.9 microhertz) returned
    1.12e17 Hz silently: the guard was exact float equality, not a threshold (Section 4.3.7's delta_(n,a) != 0)."""
    resonant = CombSpec(NU_Q / 158.0, 10e-12, "field_sech", 158, 0.0)
    assert abs(resonant.resonance_index(NU_Q) - 158.0) < 1e-12
    with pytest.raises(ZeroDivisionError, match="sits on beat note"):
        resonant.stark4_hz([0.0, NU_Q], couplings_hz=[0.0, 1e6])
    with pytest.raises(ZeroDivisionError, match="sits on beat note"):
        resonant.comb_factor(NU_Q)
    # with the explicit/folded partition the locked comb is legal: the resonant beat note is a QobjEvo coefficient,
    # not a term of the static sum, and the far-detuned teeth still shift the level
    locked = resonant.stark4_hz([0.0, NU_Q], couplings_hz=[0.0, 1e6], gate_time_s=100e-6, resonance_hz=NU_Q)[
        0
    ]
    assert math.isfinite(locked) and abs(locked) < 1e6
    # j = 0 stays legal: the guard is on delta, never on j (the Zeeman partners at 7 MHz < nu_rep/2)
    assert LEE.comb_factor(7e6, l_max=20000) == pytest.approx(0.988841, abs=2e-6)
    # the "marginal at full power" band is warned, not refused (Omega_0 exceeds delta at the source's own point)
    with pytest.warns(RuntimeWarning, match="marginal at full power"):
        LEE.stark4_hz([0.0, 12.642821e9], couplings_hz=[0.0, 45.7e6])


def test_near_resonant_explicit_far_detuned_folded_never_both() -> None:
    """PLAN.md:515. The static l-sum must exclude every beat note ``tones()`` keeps explicit; keeping the resonant one in
    both inflated the shift by 4.78 at the Lee point (5510 Hz against 1153 Hz)."""
    w = 12.642821e9
    both = LEE.stark4_hz([0.0, w], couplings_hz=[0.0, 1e6])[0]
    partitioned = LEE.stark4_hz([0.0, w], couplings_hz=[0.0, 1e6], gate_time_s=100e-6, resonance_hz=w)[0]
    assert both / partitioned == pytest.approx(4.78, abs=0.01)
    assert LEE.explicit_orders(w, 100e-6) == frozenset({105})
    # 9.17: moving the cut 10/t_g -> 20/t_g changes the gate phase by < 1e-4 rad. At 80 MHz the window (100 and
    # 200 kHz) never reaches the next beat note, so the folded set is identical and the change is exactly zero.
    t_g = 100e-6
    assert APB.explicit_orders(NU_Q, t_g) == APB.explicit_orders(NU_Q, t_g / 2.0) == frozenset({158})
    kw = {"couplings_hz": [0.0, 1e6], "resonance_hz": NU_Q}
    phase_10 = TWO_PI * APB.stark4_hz([0.0, NU_Q], gate_time_s=t_g, **kw)[0] * t_g  # type: ignore[arg-type]
    phase_20 = TWO_PI * APB.stark4_hz([0.0, NU_Q], gate_time_s=t_g / 2.0, **kw)[0] * t_g  # type: ignore[arg-type]
    assert abs(phase_20 - phase_10) < 1e-4

    # with a window deliberately wide enough to move the l = 158 +- 1 neighbours across the cut, the TOTAL phase
    # (folded shift plus the retained off-resonant tones' own second-order shift) is what stays put
    def total_phase(gate_time: float) -> float:
        explicit = APB.explicit_orders(NU_Q, gate_time)
        folded = APB.stark4_hz(
            [0.0, NU_Q], couplings_hz=[0.0, 1e6], n_index=0, gate_time_s=gate_time, resonance_hz=NU_Q
        )[0]
        extra = 0.0
        for j in explicit:
            if j == 158:
                continue  # the resonant tone is the drive, not a shift
            mu = (j * APB.rep_rate_hz + APB.aom_offset_hz) - NU_Q
            extra += (1e6 * APB.pair_weight(j)) ** 2 / 4.0 / (-mu)
        return TWO_PI * (folded + extra) * t_g

    wide = 10.0 / (1.5 * APB.rep_rate_hz)
    assert APB.explicit_orders(NU_Q, wide) == frozenset({157, 158, 159})
    assert APB.explicit_orders(NU_Q, wide / 2.0) == frozenset(range(155, 162))
    assert abs(total_phase(wide) - total_phase(wide / 2.0)) < 1e-4
    # and the folded part alone does move, which is what makes the invariance a test and not a tautology
    folded_only = [
        APB.stark4_hz([0.0, NU_Q], couplings_hz=[0.0, 1e6], gate_time_s=g, resonance_hz=NU_Q)[0]
        for g in (wide, wide / 2.0)
    ]
    assert abs(folded_only[1] - folded_only[0]) > 1.0


def test_aom_offset_is_on_the_same_grid_in_both_paths() -> None:
    """``tones()`` places beat notes at j nu_rep + Delta_nu_M; the fourth-order sum used l nu_rep, so a two-comb drive
    had its explicit tones and its static shift on different grids (M2 audit E7)."""
    offset = 11.381e6
    two_comb = CombSpec(80.6e6, 10e-12, "field_sech", 157, -offset)
    target = 12.642819e9
    j, off = two_comb.solve_offset(target)
    assert j == 157
    tones = two_comb.tones(target, None, omega0_hz=1e6, gate_time_s=100e-6)
    assert len(tones) == 1
    # the retained tone sits on the beat note the l-sum excludes: one grid, one exclusion
    assert float(tones[0].detuning_hz) == pytest.approx(  # type: ignore[arg-type]
        (157 * two_comb.rep_rate_hz + two_comb.aom_offset_hz) - target, abs=1e-6
    )
    assert two_comb.explicit_orders(target, 100e-6) == frozenset({157})
    kw = {"couplings_hz": [0.0, 1e6], "gate_time_s": 100e-6, "resonance_hz": target}
    with_offset = two_comb.stark4_hz([0.0, target], **kw)[0]  # type: ignore[arg-type]
    no_offset = CombSpec(80.6e6, 10e-12, "field_sech", 157, 0.0).stark4_hz([0.0, target], **kw)[0]  # type: ignore[arg-type]
    assert with_offset != pytest.approx(no_offset, rel=1e-6), "the AOM offset must move the l-sum grid"
    assert two_comb.beat_notes_hz(np.array([157]))[0] == pytest.approx(157 * two_comb.rep_rate_hz - offset)
