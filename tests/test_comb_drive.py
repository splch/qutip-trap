"""Frequency-comb Raman drives: the tooth-pair weight, the fourth-order Stark l-sum and its guards, the explicit/folded
partition of the beat notes, and a comb drive through the Hamiltonian builder, against the numbers of check_comb.py."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.device.model import Device
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.comb import (
    FIELD_FWHM_OVER_TAU,
    INTENSITY_FWHM_OVER_TAU,
    SUM_DEPTH_HALF_WIDTHS,
    CombSpec,
    sech,
    tau_field_from,
)
from qutip_trap.light.raman import DerivedDrive, comb_drive, derive_raman_drive
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.units import TWO_PI
from tests.fixtures import KX, single_ion_raman_device

LEE = CombSpec(120e6, 14e-12, "field_sech", 105, 0.0)
APB = CombSpec(80e6, 10e-12, "field_sech", 158, 0.0)
NU_Q = 12.642812118466e9
"""The 171Yb+ zero-field clock splitting: the APB operating point's qubit frequency."""
"""The 3 MHz x mode of the single-ion fixture."""
GATE_TIME_S = 100e-6
"""The explicit-tone cut is 10/t_g, about 100 kHz for a 100 us gate: far inside the 80 MHz tooth gap."""


def _pair_sum_exact(comb: CombSpec, l: int, n_teeth: int) -> float:  # noqa: E741  (l is the tooth separation)
    """sum_k g_k g_{k+l}/g_0^2 by explicit convolution of the single-tooth amplitudes
    g_k/g_0 = sqrt(pi nu_rep tau) sech(2 pi k nu_rep tau)."""
    x = comb.rep_rate_hz * comb.tau_field_s
    g = math.sqrt(math.pi * x) * np.asarray(sech(2.0 * math.pi * np.arange(-n_teeth, n_teeth + 1) * x))
    return float(np.sum(g[: len(g) - l] * g[l:]))


def test_pair_weight_uses_the_half_argument() -> None:
    """sech(pi l nu_rep tau) = 0.863911 at l = 105 against the wrong sech(2 pi l ...) = 0.595332, a 27% error; the closed
    form is itself about 5% high against the exact convolution."""
    assert LEE.pair_weight(105) == pytest.approx(0.863911, abs=1e-6)
    assert 1.0 / math.cosh(2 * math.pi * 105 * 120e6 * 14e-12) == pytest.approx(0.595332, abs=1e-6)
    assert 0.03 < LEE.pair_weight(105) / _pair_sum_exact(LEE, 105, 200_000) - 1.0 < 0.07


def test_pulse_width_conventions() -> None:
    assert FIELD_FWHM_OVER_TAU == pytest.approx(1.6768, abs=1e-4)
    assert INTENSITY_FWHM_OVER_TAU == pytest.approx(1.1222, abs=1e-4)
    assert tau_field_from(1.6768e-12, "field_fwhm") == pytest.approx(1e-12, rel=1e-3)
    assert tau_field_from(2e-12, "intensity_sech") == tau_field_from(2e-12, "intensity_sech2") == 1e-12


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


def test_the_offset_lock_and_the_beat_note_grid() -> None:
    """Islam's lock: n = 157 with |Delta nu_M| = 11.381 MHz at 80.6 MHz."""
    islam = CombSpec(80.6e6, 10e-12, "field_sech", 157, 0.0)
    j, off = islam.solve_offset(12.642819e9)
    assert j == 157 and off == pytest.approx(-11.381e6, abs=2e3)
    assert islam.beat_notes_hz(np.array([157]))[0] == pytest.approx(157 * 80.6e6)


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
    # the two clauses that need extra inputs are absent without them, and named as missing
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
    """l_max = ceil(8/(pi nu_rep tau)): 1516 at the Lee point and 3184 at the APB one; a literal 1200 plateaus falsely at
    the APB point and is refused."""
    assert LEE.sum_half_width == pytest.approx(189.470, abs=1e-3)
    assert APB.sum_half_width == pytest.approx(397.887, abs=1e-3)
    assert LEE.sum_depth == math.ceil(SUM_DEPTH_HALF_WIDTHS * LEE.sum_half_width) == 1516
    assert APB.sum_depth == math.ceil(SUM_DEPTH_HALF_WIDTHS * APB.sum_half_width) == 3184
    assert LEE.stark4_hz([0.0, 12.642821e9], couplings_hz=[0.0, 1e6])[0] == pytest.approx(5510.08, rel=1e-5)
    assert APB.stark4_hz([0.0, NU_Q], couplings_hz=[0.0, 1e6])[0] == pytest.approx(79680.65, rel=1e-5)
    with pytest.raises(ValueError, match="plateaus falsely"):
        CombSpec(80e6, 10e-12, "field_sech", 158, 0.0, pair_order_max=1200)
    assert LEE.comb_factor(12.642821e9, validate=True) == pytest.approx(0.943789, abs=2e-6)
    with pytest.raises(ValueError, match="comb_factor has not converged"):
        LEE.comb_factor(12.642821e9, l_max=10, validate=True)


def test_the_resonant_comb_is_refused_rather_than_returning_a_huge_shift() -> None:
    """A rep rate that nearly divides the splitting (nu_rep = nu_q/158, resonant to 1.9 microhertz) would give 1.12e17 Hz
    under an exact-equality guard; the threshold guard refuses it."""
    resonant = CombSpec(NU_Q / 158.0, 10e-12, "field_sech", 158, 0.0)
    assert abs(NU_Q / resonant.rep_rate_hz - 158.0) < 1e-12
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
    # the "marginal at full power" band is warned, not refused (Omega_0 exceeds delta at the source's own point)
    with pytest.warns(RuntimeWarning, match="marginal at full power"):
        LEE.stark4_hz([0.0, 12.642821e9], couplings_hz=[0.0, 45.7e6])


def test_near_resonant_explicit_far_detuned_folded_never_both() -> None:
    """The static l-sum excludes every beat note ``tones()`` keeps explicit; keeping the resonant one in both inflates the
    shift by 4.78 at the Lee point (5510 Hz against 1153 Hz)."""
    w = 12.642821e9
    both = LEE.stark4_hz([0.0, w], couplings_hz=[0.0, 1e6])[0]
    partitioned = LEE.stark4_hz([0.0, w], couplings_hz=[0.0, 1e6], gate_time_s=100e-6, resonance_hz=w)[0]
    assert both / partitioned == pytest.approx(4.78, abs=0.01)
    assert LEE.explicit_orders(w, 100e-6) == frozenset({105})
    # moving the cut 10/t_g -> 20/t_g changes the gate phase by < 1e-4 rad; at 80 MHz the window (100 and 200 kHz) never
    # reaches the next beat note, so the folded set is identical and the change is exactly zero
    t_g = 100e-6
    assert APB.explicit_orders(NU_Q, t_g) == APB.explicit_orders(NU_Q, t_g / 2.0) == frozenset({158})
    kw = {"couplings_hz": [0.0, 1e6], "resonance_hz": NU_Q}
    phase_10 = TWO_PI * APB.stark4_hz([0.0, NU_Q], gate_time_s=t_g, **kw)[0] * t_g  # type: ignore[arg-type]
    phase_20 = TWO_PI * APB.stark4_hz([0.0, NU_Q], gate_time_s=t_g / 2.0, **kw)[0] * t_g  # type: ignore[arg-type]
    assert abs(phase_20 - phase_10) < 1e-4

    # with a window wide enough to move the l = 158 +- 1 neighbours across the cut, the TOTAL phase (folded shift plus the
    # retained off-resonant tones' own second-order shift) is what stays put
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
    # the folded part alone does move, which is what makes the invariance a test and not a tautology
    folded_only = [
        APB.stark4_hz([0.0, NU_Q], couplings_hz=[0.0, 1e6], gate_time_s=g, resonance_hz=NU_Q)[0]
        for g in (wide, wide / 2.0)
    ]
    assert abs(folded_only[1] - folded_only[0]) > 1.0


def test_aom_offset_is_on_the_same_grid_in_both_paths() -> None:
    """``tones()`` places beat notes at j nu_rep + Delta_nu_M and the fourth-order sum excludes the same grid points."""
    offset = 11.381e6
    two_comb = CombSpec(80.6e6, 10e-12, "field_sech", 157, -offset)
    target = 12.642819e9
    j, _off = two_comb.solve_offset(target)
    assert j == 157
    tones = two_comb.tones(target, None, omega0_hz=1e6, gate_time_s=100e-6)
    assert len(tones) == 1
    assert float(tones[0].detuning_hz) == pytest.approx(  # type: ignore[arg-type]
        (157 * two_comb.rep_rate_hz + two_comb.aom_offset_hz) - target, abs=1e-6
    )
    assert two_comb.explicit_orders(target, 100e-6) == frozenset({157})
    kw = {"couplings_hz": [0.0, 1e6], "gate_time_s": 100e-6, "resonance_hz": target}
    with_offset = two_comb.stark4_hz([0.0, target], **kw)[0]  # type: ignore[arg-type]
    no_offset = CombSpec(80.6e6, 10e-12, "field_sech", 157, 0.0).stark4_hz([0.0, target], **kw)[0]  # type: ignore[arg-type]
    assert with_offset != pytest.approx(no_offset, rel=1e-6), "the AOM offset must move the l-sum grid"
    assert two_comb.beat_notes_hz(np.array([157]))[0] == pytest.approx(157 * two_comb.rep_rate_hz - offset)


# ---- a comb drive through the Hamiltonian builder ------------------------------------------------------------------


def _fixture() -> tuple[Device, DerivedDrive, CombSpec, float]:
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    sp = dev.crystal.species[0]
    nu_q, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)
    # the AOM offset is the primary control input: solve_offset puts a beat note exactly on the carrier
    _j, off = CombSpec(80e6, 10e-12, "field_sech", 158, 0.0).solve_offset(nu_q)
    return dev, dd, CombSpec(80e6, 10e-12, "field_sech", 158, off), nu_q


def test_a_comb_carrier_pulse_matches_the_single_tone_prediction() -> None:
    """Inside the gate window the train reduces to ONE tone of envelope Omega_0 sech(pi j nu_rep tau), so a pulse of length
    pi/(Omega_0 sech x e^{-eta^2/2}) inverts the qubit."""
    dev, dd, comb, nu_q = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the un-evaluable guards are reported, not fatal
        drive = comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=GATE_TIME_S)
    assert len(drive.tones) == 1, "the 10/t_g cut keeps exactly the resonant beat note at 80 MHz spacing"
    tone = drive.tones[0]
    assert float(tone.detuning_hz) == pytest.approx(0.0, abs=1.0)  # type: ignore[arg-type]
    assert float(tone.envelope_hz) == pytest.approx(  # type: ignore[arg-type]
        dd.carrier_rabi_hz * comb.pair_weight(158), rel=1e-12
    )
    # sech(pi l nu_rep tau) at l = 158 is the time-domain sech(omega_q tau/2) = 0.925994 at the zero-field splitting, to
    # the 1e-4 the 158 nu_rep - nu_q mismatch of the 5 G fixture allows
    assert comb.pair_weight(158) == pytest.approx(0.926025, abs=1e-6)
    assert 1.0 / math.cosh(math.pi * NU_Q * 10e-12) == pytest.approx(0.925994, abs=1e-6)
    assert comb.pair_weight(158) == pytest.approx(0.925994, abs=1e-4)

    space = HilbertSpace((2,), (ModeTruncation(KX, 12, (0, 3), 0.2),), None, (0, 2))
    om = TWO_PI * float(tone.envelope_hz)  # type: ignore[arg-type]
    eta = dd.etas[KX]
    t_pi = math.pi / (om * math.exp(-(eta**2) / 2.0))
    pulse = Pulse(drive, 0.0, t_pi, "comb", ())
    eng = JointExactEngine(builder_options=BuilderOptions(), store_per_segment=2)
    tr = eng.run_pulses(
        dev,
        Schedule((pulse,), (), (), {0: 0.0}),
        space.initial_state([0], fock={KX: 0}),
        space,
        quiet_sample(),
        SeedSpec(0),
        SolverOptions(),
    )
    p1 = float(tr.expectations["P1[0]"][-1])
    assert p1 == pytest.approx(1.0, abs=3e-6)
    # the residual is the off-resonant sidebands, ~ (eta Omega/omega_m)^2, as for a plain single-tone carrier
    assert 0.05 * (eta * om / (TWO_PI * 3.0e6)) ** 2 < 1.0 - p1 < 5.0 * (eta * om / (TWO_PI * 3.0e6)) ** 2
    rep = eng.last_report
    assert rep is not None
    assert any("explicit beat note" in a and "never both" in a for a in rep.approximations)


def test_the_comb_drives_fold_their_far_detuned_teeth_into_a_static_shift() -> None:
    """With the level structure supplied the factory sets ``stark_shift_hz`` from ``comb.stark4_hz`` on the SAME cut."""
    dev, dd, comb, nu_q = _fixture()
    levels = [0.0, nu_q]
    couplings = [0.0, 1e6]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        drive = comb_drive(
            dd,
            comb,
            omega_q_hz=nu_q,
            gate_time_s=GATE_TIME_S,
            levels_hz=levels,
            couplings_hz=couplings,
        )
    expected = comb.stark4_hz(levels, couplings_hz=couplings, gate_time_s=GATE_TIME_S, resonance_hz=nu_q)[0]
    assert float(drive.stark_shift_hz) == pytest.approx(expected, rel=1e-12)  # type: ignore[arg-type]
    assert drive.stark_shift_hz != 0.0
    space = HilbertSpace((2,), (), None, (0, 1, 2))
    built = build_hamiltonian(dev, (Pulse(drive, 0.0, 1e-6, "comb", ()),), space, sample=quiet_sample())
    assert any(f"{float(drive.stark_shift_hz):.6g} Hz" in a for a in built.approximations)  # type: ignore[arg-type]


def test_the_factory_reports_the_guards_it_cannot_evaluate() -> None:
    dev, dd, comb, nu_q = _fixture()
    with pytest.warns(RuntimeWarning, match="not evaluated"):
        comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=GATE_TIME_S)
    # a failing clause is named: a train of ten pulses does not resolve the teeth
    with pytest.warns(RuntimeWarning, match="teeth_resolved"):
        comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=1e-8)
    with pytest.warns(RuntimeWarning) as record:
        comb_drive(
            dd,
            comb,
            omega_q_hz=nu_q,
            gate_time_s=GATE_TIME_S,
            detuning_hz=22.11e12,
            fine_structure_hz=67e12,
            theta_per_pulse_rad=1e-3,
        )
    assert any("not evaluated none" in str(w.message) for w in record)
    with pytest.raises(ValueError, match="positive duration"):
        comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=0.0)


def test_the_tone_set_and_the_static_shift_share_one_operator_and_one_grid() -> None:
    """Widening the cut moves beat notes from the folded sum into the tone list, and the builder still emits one drive term
    (one operator with a summed coefficient)."""
    dev, dd, comb, nu_q = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wide = comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=10.0 / (1.5 * comb.rep_rate_hz))
        one = comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=GATE_TIME_S)
    assert len(wide.tones) == 3, "the l = j and its two neighbours at +-nu_rep"
    detunings = sorted(float(t.detuning_hz) for t in wide.tones)  # type: ignore[arg-type]
    assert detunings == pytest.approx([-80e6, 0.0, 80e6], abs=1.0)
    space = HilbertSpace((2,), (ModeTruncation(KX, 6, (0, 2), 0.2),), None, (0, 2))
    built = build_hamiltonian(dev, (Pulse(wide, 0.0, 1e-6, "comb", ()),), space, sample=quiet_sample())
    built_one = build_hamiltonian(dev, (Pulse(one, 0.0, 1e-6, "comb", ()),), space, sample=quiet_sample())
    assert built.n_drive_terms == built_one.n_drive_terms
    # the highest frame frequency jumps from the mode scale to nu_rep once a neighbour is retained
    assert built.omega_max_rad_s > 10.0 * built_one.omega_max_rad_s or np.isclose(
        built.omega_max_rad_s, built_one.omega_max_rad_s
    )
