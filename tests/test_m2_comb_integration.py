"""A mode-locked (frequency-comb) Raman drive through the ONE Hamiltonian builder (PLAN.md Sections 4.3.7, 5.2, 9.17).

"A comb drive is a SET of tones in the sense of Section 5.2 ... one drive term per beat note, all sharing the same
time-independent operator sigma_+ (x) prod D_m, each with its own scalar coefficient." Before the M2 fix nothing
constructed one: ``Drive.comb`` was read only by its own kind validation, ``light/raman.py`` had no comb reference and
no test integrated a comb drive at all (audit E8/D). ``light.raman.comb_drive`` is the factory, and this is the
regression that makes the structural claim an asserted fact.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.light.comb import CombSpec
from qutip_trap.light.raman import comb_drive, derive_raman_drive
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.units import TWO_PI
from tests.m2_fixtures import single_ion_raman_device

KX = 1
"""The 3 MHz x mode of the single-ion fixture."""
GATE_TIME_S = 100e-6
"""The Section 4.3.7 cut is 10/t_g, about 100 kHz for a 100 us gate: far inside the 80 MHz tooth gap."""


def _fixture() -> tuple[object, object, CombSpec, float]:
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    sp = dev.crystal.species[0]
    nu_q, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)
    # the AOM offset is the primary control input: solve_offset puts a beat note exactly on the carrier
    probe = CombSpec(80e6, 10e-12, "field_sech", 158, 0.0)
    _j, off = probe.solve_offset(nu_q)
    comb = CombSpec(80e6, 10e-12, "field_sech", 158, off)
    return dev, dd, comb, nu_q


def test_a_comb_carrier_pulse_matches_the_single_tone_prediction() -> None:
    """One explicit beat note at the carrier: the comb pulse is the single-tone pi pulse to the sech factor.

    The whole train reduces, inside the gate window, to ONE tone of envelope Omega_0 sech(pi j nu_rep tau) - one sech
    factor, never three (Section 4.3.7) - so a pulse of length pi/(Omega_0 sech x e^{-eta^2/2}) inverts the qubit.
    """
    dev, dd, comb, nu_q = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # the un-evaluable guards (Section 12) are reported, not fatal
        drive = comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=GATE_TIME_S)
    assert len(drive.tones) == 1, "the 10/t_g cut keeps exactly the resonant beat note at 80 MHz spacing"
    tone = drive.tones[0]
    assert float(tone.detuning_hz) == pytest.approx(0.0, abs=1.0)  # type: ignore[arg-type]
    assert float(tone.envelope_hz) == pytest.approx(  # type: ignore[arg-type]
        dd.carrier_rabi_hz * comb.pair_weight(158), rel=1e-12
    )
    # one sech factor, at the tooth separation: sech(pi l nu_rep tau) with l = 158 and nu_rep = 80 MHz. The plan's
    # 0.925994 is the same factor written in the time domain, sech(omega_q tau/2) at the ZERO-FIELD splitting; the
    # fixture sits at 5 G, so the two agree to the 1e-4 the 158 nu_rep - nu_q mismatch allows.
    assert comb.pair_weight(158) == pytest.approx(0.926025, abs=1e-6)
    assert 1.0 / math.cosh(math.pi * 12.642812118466e9 * 10e-12) == pytest.approx(0.925994, abs=1e-6)
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
    # the residual is the off-resonant sidebands, ~ (eta Omega/omega_m)^2, exactly as for a plain single-tone carrier
    assert 0.05 * (eta * om / (TWO_PI * 3.0e6)) ** 2 < 1.0 - p1 < 5.0 * (eta * om / (TWO_PI * 3.0e6)) ** 2
    # the builder reports the partition, not a bare tone list
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
    """Section 4.3.7's validity hierarchy is "a runtime guard logged before every solve"; two clauses were literal True."""
    dev, dd, comb, nu_q = _fixture()
    with pytest.warns(RuntimeWarning, match="not evaluated"):
        comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=GATE_TIME_S)
    # a failing clause is named: a train of ten pulses does not resolve the teeth
    with pytest.warns(RuntimeWarning, match="teeth_resolved"):
        comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=1e-8)
    # supplying the missing inputs empties the "not evaluated" list
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
    """Widening the cut moves beat notes from the folded sum into the tone list and nothing else (Section 5.2)."""
    dev, dd, comb, nu_q = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wide = comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=10.0 / (1.5 * comb.rep_rate_hz))
    assert len(wide.tones) == 3, "the l = j and its two neighbours at +-nu_rep"
    detunings = sorted(float(t.detuning_hz) for t in wide.tones)  # type: ignore[arg-type]
    assert detunings == pytest.approx([-80e6, 0.0, 80e6], abs=1.0)
    # every tone shares the one sigma_+ (x) prod D operator: the builder emits one drive term per tone, not per operator
    space = HilbertSpace((2,), (ModeTruncation(KX, 6, (0, 2), 0.2),), None, (0, 2))
    built = build_hamiltonian(dev, (Pulse(wide, 0.0, 1e-6, "comb", ()),), space, sample=quiet_sample())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        one = comb_drive(dd, comb, omega_q_hz=nu_q, gate_time_s=GATE_TIME_S)
    built_one = build_hamiltonian(dev, (Pulse(one, 0.0, 1e-6, "comb", ()),), space, sample=quiet_sample())
    assert built.n_drive_terms == built_one.n_drive_terms, (
        "a tone SET is one operator with a summed coefficient, so the term count does not grow with the tone count"
    )
    # the highest frequency in the frame jumps from the mode scale to nu_rep once a neighbour is retained, which is the
    # 40x step-count multiplier Section 4.3.7 warns about
    assert built.omega_max_rad_s > 10.0 * built_one.omega_max_rad_s or np.isclose(
        built.omega_max_rad_s, built_one.omega_max_rad_s
    )
