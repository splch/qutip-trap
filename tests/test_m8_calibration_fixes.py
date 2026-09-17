"""Regressions for the M8 audit's calibration defects (PLAN.md Sections 7.3, 7.5, 7.10; milestone M8).

Each test would have passed the wrong number, or passed a fabricated one, before the fix it names:

- B1 ``stark_scan``'s fringe branch resolved with two probe signs (a shift above the probe aliased onto one below it and
  was written ``calibrated``).
- B2 a refused experiment marks its OWN entries ``uncalibrated``, so what the calibration could not establish refuses to
  schedule instead of falling back to the derived seed (Section 7.3).
- B3 ``run()`` refuses an ``uncalibrated`` qubit frequency instead of taking the frame at the true transition, which made
  the entry error-free by construction.
- B4 the calibrated micromotion shims are programmed onto the device the run evolves, so a compensation fitted against a
  stray field that has since drifted leaves the residual beta a laboratory would have (the closed loop of Section 7.5).
- B5 the entangling scans run in the frame the table programs and at the mode frequencies it believes; both keywords used
  to land in ``**kw`` and be discarded.
- B8 no ``calibrated`` 0 +- 0 entry under the name of an experiment that never ran.
- C every name ``calibrate`` accepts runs something.
- P2-8 the run-level over-rotation a miscalibrated Rabi entry causes.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.api import CalEntry, Circuit, Operation, RfDrive, SolverOptions, run, schedule, white_spectrum
from qutip_trap.calibration import EXPERIMENTS
from qutip_trap.calibration.experiments import (
    ALIASES,
    ORDER,
    PRODUCES,
    UPSTREAM,
    CalibrationScans,
    full_calibration,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.schedule import ScheduleError
from qutip_trap.experiments.light import stark_scan
from qutip_trap.experiments.result import ExperimentResult
from qutip_trap.options import Numerics
from qutip_trap.run.job import RunError, last_record
from qutip_trap.units import TWO_PI
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
GPI = Circuit(2, (Operation("gpi", (0,), (0.0,)),), (0,))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=1000, detection_windows_s=WINDOWS)
    return fx, sur


# ---- B1: the Stark scan's fringe branch --------------------------------------------------------------------------------------


def _exact_fringe(shift_hz: float):  # type: ignore[no-untyped-def]
    """A ``ramsey`` replacement returning the EXACT fringe frequency |detuning - shift| of a qubit light-shifted by
    ``shift_hz``, which is what the fit sees (its sign is unobservable in a cosine). The audit's own probe."""

    def fake(device, ion, delays_s, **kw):  # type: ignore[no-untyped-def]
        detuning = float(kw["detuning_hz"])
        delays = np.array(sorted(float(x) for x in delays_s))
        f = abs(detuning - shift_hz)
        data = np.column_stack([delays, 0.5 * np.cos(TWO_PI * f * delays) + 0.5])
        return ExperimentResult(
            data=data,
            fitted={
                "contrast": (0.5, 0.0),
                "delta_hz": (f, 0.0),
                "phi0_rad": (0.0, 0.0),
                "offset": (0.5, 0.0),
            },
            model="exact_fringe",
            provenance_id="conv.detuning_symbols",
            converged=True,
        )

    return fake


@pytest.mark.parametrize(
    ("shift_hz", "ok"),
    [(-38.4, True), (300.0, True), (-900.0, True), (1500.0, False), (2500.0, False)],
)
def test_stark_scan_resolves_the_fringe_branch_with_two_probe_signs(
    two_ion, monkeypatch, shift_hz, ok
) -> None:  # type: ignore[no-untyped-def]
    """The Ramsey fit returns |probe - delta|, so ONE probe sign loses the sign of (delta - probe): a light shift of
    1.5 x probe was reported as 0.5 x probe with ``converged=True`` (kilohertz shifts are ordinary for Raman gates - the
    plan's own 7.10 text quotes 1.27 kHz at Delta = 4 MHz). Two probe signs give delta = (f_minus - f_plus)/2, exact while
    |delta| < probe, and the sum f_plus + f_minus = 2 probe detects the out-of-range case."""
    fx, _sur = two_ion
    monkeypatch.setattr("qutip_trap.experiments.light.ramsey", _exact_fringe(shift_hz))
    probe = 1e3
    res = stark_scan(fx.device, 0, np.linspace(0.0, 2e-3, 9), probe_hz=probe, shots=None)
    beams = fx.gate_drives[0].beams
    assert res.converged is ok, res.notes
    if ok:
        for b in beams:
            assert res.fitted[f"stark_shift_hz[{b}]"][0] == pytest.approx(shift_hz, abs=1e-9)
            assert res.fitted[f"fringe_plus_hz[{b}]"][0] == pytest.approx(abs(probe - shift_hz), abs=1e-9)
            assert res.fitted[f"fringe_minus_hz[{b}]"][0] == pytest.approx(abs(probe + shift_hz), abs=1e-9)
        assert res.fitted["stark_shift_hz"][0] == pytest.approx(len(beams) * shift_hz, abs=1e-9)
    else:
        # the estimator saturates at the probe and the two fringes no longer sum to 2 probe: BOTH guards fire
        for b in beams:
            assert abs(res.fitted[f"stark_shift_hz[{b}]"][0]) == pytest.approx(probe, abs=1e-9)
        assert any("outside the probe" in n for n in res.notes), res.notes


def test_stark_scan_refuses_a_fringe_above_the_delay_grids_nyquist_frequency(two_ion, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The ledger records the aliasing caveat for ``mode='beat_note'``; the same delay grid aliases the per-beam fringe at
    probe + |delta| when the step is too coarse. Five delays over 2 ms resolve 1 kHz at best, so a 1 kHz probe with any
    non-zero shift is refused, and nine delays over the same span accept it."""
    fx, _sur = two_ion
    monkeypatch.setattr("qutip_trap.experiments.light.ramsey", _exact_fringe(-38.4))
    kw = dict(probe_hz=1e3, shots=None)
    coarse = stark_scan(fx.device, 0, np.linspace(0.0, 2e-3, 5), **kw)  # type: ignore[arg-type]
    fine = stark_scan(fx.device, 0, np.linspace(0.0, 2e-3, 9), **kw)  # type: ignore[arg-type]
    assert not coarse.converged and any("Nyquist" in n for n in coarse.notes), coarse.notes
    assert fine.converged, fine.notes
    # the VALUE is the same either way: the guard is about what the grid can resolve, not about the estimator
    for b in fx.gate_drives[0].beams:
        assert coarse.fitted[f"stark_shift_hz[{b}]"][0] == pytest.approx(-38.4, abs=1e-9)


def test_stark_scan_does_not_leak_its_mode_switch_into_the_ramsey_setup(two_ion, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``_setup`` reads a ``mode`` keyword as a driven-mode INDEX, and ``stark_scan`` reads its own ``mode`` switch from the
    same keyword dictionary and forwarded it: ``mode='beat_note'`` raised int('beat_note') before the switch was stripped."""
    fx, _sur = two_ion
    monkeypatch.setattr("qutip_trap.experiments.light.ramsey", _exact_fringe(-38.4))
    res = stark_scan(fx.device, 0, np.linspace(0.0, 2e-3, 9), probe_hz=1e3, shots=None, mode="beat_note")
    assert "stark_shift_hz" in res.fitted and "coupling_shift_hz" in res.fitted
    with pytest.raises(ValueError, match="per_beam"):
        stark_scan(fx.device, 0, np.linspace(0.0, 2e-3, 9), mode="nonsense")


# ---- B2: a refused experiment's own entries ----------------------------------------------------------------------------------


def test_a_refused_experiment_marks_its_own_entries_uncalibrated_and_the_schedule_then_refuses(
    two_ion,
) -> None:  # type: ignore[no-untyped-def]
    """Section 7.3: 'an entry the calibration could not establish is uncalibrated and refuses to schedule rather than
    falling back to them [the derived values]'. An uncalibrated field refuses the micromotion, mode, Rabi, Stark and
    Ramsey fits in turn; each marks the entry groups it would have written, and the uncalibrated Rabi entry then makes
    ``schedule()`` raise. The first M8 build recorded the refusals in the report and left every entry a schedulable seed
    (a Bell circuit scheduled 15 pulses from it)."""
    fx, sur = two_ion
    bad_field = dataclasses.replace(
        sur.table,
        field=dataclasses.replace(sur.table.field, status="uncalibrated"),
    )
    report = full_calibration(
        fx.device,
        experiments=(
            "micromotion_scan",
            "sideband_spectroscopy",
            "rabi_scan",
            "stark_scan",
            "ramsey_frequency",
        ),
        surrogate=dataclasses.replace(sur, table=bad_field),
    )
    t = report.table
    for name in ("micromotion_scan", "sideband_spectroscopy", "rabi_scan", "stark_scan", "ramsey_frequency"):
        assert name in report.refused, (name, report.refused)
    assert report.results == {}, "no experiment ran"
    for group in ("modes", "nbar", "rabi", "stark", "qubit_freq"):
        entries = getattr(t, group)
        assert entries, group
        assert all(e.status == "uncalibrated" for e in entries.values()), (group, entries)
    # each refusal names the first upstream group it could not read, the field for all five here (this fixture has no rf
    # record and the test asks for no shim ranges, so the micromotion scan writes no entry of its own to be marked)
    assert t.micromotion == {}
    assert all("field" in reason for reason in report.refused.values()), report.refused
    with pytest.raises(ScheduleError):
        schedule(BELL, fx.device, t)


def test_a_subset_calibration_leaves_the_other_entries_as_seeds_not_uncalibrated(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The distinction the fix has to keep: a group that is a seed because the caller did not ASK for its experiment stays
    a seed (the Section 7.5 bootstrap and ``calibrate(experiments=...)``), only the refusal path marks."""
    fx, sur = two_ion
    report = full_calibration(
        fx.device, experiments=("field_scan",), surrogate=sur, scans=CalibrationScans(shots=None)
    )
    assert not report.refused, report.refused
    assert report.table.field.status == "calibrated"
    assert all(e.status == "seed" for e in report.table.rabi.values())
    assert report.table.uncalibrated() == ()


def test_produces_covers_every_experiment_and_every_entry_group_has_one_producer() -> None:
    """``PRODUCES`` is the map the refusal path walks: every name in ``ORDER`` appears, every group it names is a real
    ``CalibrationTable`` attribute, and no group has two producers (the inverse map must be well defined)."""
    from qutip_trap.control.table import CalibrationTable

    assert set(PRODUCES) == set(ORDER) == set(UPSTREAM)
    fields = {f.name for f in dataclasses.fields(CalibrationTable)}
    seen: dict[str, str] = {}
    for name, groups in PRODUCES.items():
        for group in groups:
            assert group in fields, (name, group)
            assert group not in seen, (group, name, seen[group])
            seen[group] = name
    # every group any experiment READS is written by some experiment (otherwise a refusal cannot cascade)
    for name, needs in UPSTREAM.items():
        for group in needs:
            assert group in seen, (name, group)


# ---- B3: an uncalibrated qubit frequency ------------------------------------------------------------------------------------


def test_run_refuses_an_uncalibrated_qubit_frequency(two_ion) -> None:  # type: ignore[no-untyped-def]
    """``qubit_shifts_hz`` is the ONLY channel by which the table's frequency error reaches the physics, so the old
    ``shifts[i] = 0.0`` fallback put the frame exactly on the true transition and made an uncalibrated qubit frequency
    error-free by construction - the same defect the ledger's ``conv.played_chain`` records for M2-M7."""
    fx, sur = two_ion
    bad = dataclasses.replace(
        sur.table,
        qubit_freq={
            **sur.table.qubit_freq,
            0: dataclasses.replace(sur.table.qubit_freq[0], status="uncalibrated"),
        },
    )
    with pytest.raises(RunError, match="7.3"):
        run(GPI, fx.device, 10, table=bad)


# ---- B4: the micromotion loop -----------------------------------------------------------------------------------------------


def _rf_two_ion(stray_x_v_per_m: float):  # type: ignore[no-untyped-def]
    """The two-ion fixture with an rf record and a stray field along x (excess micromotion to compensate)."""
    from qutip_trap.trap.crystal import solve_crystal

    fx = circuit_fixture(2)
    dev = dataclasses.replace(
        fx.device,
        trap=dataclasses.replace(
            fx.device.trap,
            rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
            stray_field_v_per_m=(float(stray_x_v_per_m), 0.0, 0.0),
        ),
    )
    return fx, dataclasses.replace(dev, crystal=solve_crystal(dev.trap, dev.crystal.species))


def test_the_calibrated_shims_are_programmed_onto_the_device_the_run_evolves(two_ion) -> None:  # type: ignore[no-untyped-def]
    """PLAN 7.5 asks the closed loop: 'stores shims and beta in the table, SO THAT compensation is calibrated, drifts with
    the stray field between calibrations and is re-nulled like a laboratory re-nulls it'. Nothing in run/, dynamics/ or
    control/ read ``table.micromotion`` before this: the entries were inert bookkeeping and the run always used the
    device's own stray field. A ``calibrated`` shim now reaches the device the run evolves (through
    ``device_with_compensation``, which re-solves the crystal), a ``seed`` one does not (it is the device's own setting)
    and an ``uncalibrated`` one is reported."""
    from qutip_trap.experiments.micromotion import device_with_compensation, signed_beta
    from qutip_trap.light.raman import derive_raman_drive

    fx, dev = _rf_two_ion(20.0)
    dk = np.asarray(derive_raman_drive(dev, 0, fx.gate_drives[0].beams, scattering=False).delta_k)
    beta_uncompensated = signed_beta(dev, 0, dk)
    assert abs(beta_uncompensated) > 1e-3, "the stray field gives the fixture excess micromotion to null"
    # the compensating field is minus the stray one: with it programmed the residual index vanishes
    assert abs(signed_beta(device_with_compensation(dev, {"Ex": -20.0}), 0, dk)) < 1e-9

    def table_with(status: str, value: float):  # type: ignore[no-untyped-def]
        sur = surrogate_table(
            dev, pairs=[(0, 1)], detection_records=200, detection_windows_s=(20e-6,), spot_check=False
        )
        return dataclasses.replace(
            sur.table,
            micromotion={
                "shim[Ex]": CalEntry(
                    value, 0.1, status, "micromotion_scan", "anchor.trap.berkeland_excess_micromotion", 0.0, 0
                )
            },
        )

    kw = dict(
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    good = run(GPI, dev, 5, table=table_with("calibrated", -20.0), **kw)  # type: ignore[arg-type]
    assert any("micromotion compensation applied" in n for n in last_record(good).notes)
    seeded = run(GPI, dev, 5, table=table_with("seed", -20.0), **kw)  # type: ignore[arg-type]
    assert not any("micromotion compensation applied" in n for n in last_record(seeded).notes)
    refused = run(GPI, dev, 5, table=table_with("uncalibrated", -20.0), **kw)  # type: ignore[arg-type]
    assert any("micromotion compensation uncalibrated" in n for n in last_record(refused).notes)


@pytest.mark.slow
def test_a_stale_micromotion_calibration_against_a_drifted_stray_field_leaves_a_growing_residual() -> None:
    """The failure mode PLAN 7.5 names, which was not reproducible while the entries were inert: calibrate the shims at
    t0 = 0 against one stray field, then run an hour later on a trap whose stray field has drifted. The compensation the
    table still carries no longer nulls the field, the residual beta grows from the calibration's own to the drift's, and
    the gate error grows with it (the J_1 micromotion sidebands the compensated run does not have)."""
    from qutip_trap.experiments.micromotion import device_with_compensation, signed_beta
    from qutip_trap.light.raman import derive_raman_drive

    fx, at_t0 = _rf_two_ion(20.0)
    # an hour of stray-field drift: the trap the run evolves is not the one the shims were nulled on
    drifted = dataclasses.replace(
        at_t0, trap=dataclasses.replace(at_t0.trap, stray_field_v_per_m=(35.0, 0.0, 0.0))
    )
    from qutip_trap.trap.crystal import solve_crystal

    drifted = dataclasses.replace(drifted, crystal=solve_crystal(drifted.trap, drifted.crystal.species))
    beam = 0
    from qutip_trap.experiments import micromotion_scan

    scan = micromotion_scan(at_t0, 0, beam, {"Ex": (-40.0, 0.0)}, method="sideband_ratio", points=5)
    assert scan.converged, scan.notes
    shim = float(scan.fitted["shim[Ex]"][0])
    assert shim == pytest.approx(-20.0, abs=1.0), "the scan nulls the field it was calibrated against"
    dk = np.asarray(derive_raman_drive(at_t0, 0, fx.gate_drives[0].beams, scattering=False).delta_k)
    residual_fresh = abs(signed_beta(device_with_compensation(at_t0, {"Ex": shim}), 0, dk))
    residual_stale = abs(signed_beta(device_with_compensation(drifted, {"Ex": shim}), 0, dk))
    uncompensated = abs(signed_beta(drifted, 0, dk))
    assert residual_fresh < 1e-2 * uncompensated, (residual_fresh, uncompensated)
    assert residual_stale > 10.0 * residual_fresh, (residual_stale, residual_fresh)
    # the residual index the stale table leaves is the 15 V/m of drift, not the 35 V/m of stray field: compensation still
    # helps, it just no longer nulls
    assert residual_stale == pytest.approx(uncompensated * 15.0 / 35.0, rel=0.05)
    # and the gate error grows with it: the carrier is suppressed by J_0(beta) and the first sideband carries J_1(beta)
    from scipy.special import jv

    assert 1.0 - float(jv(0, residual_stale)) > 10.0 * (1.0 - float(jv(0, max(residual_fresh, 1e-12))))


# ---- B5: the entangling scans' frame and mode beliefs ------------------------------------------------------------------------


def test_the_entangling_setup_refuses_to_swallow_the_mode_frequencies_it_would_discard(two_ion) -> None:  # type: ignore[no-untyped-def]
    """``mode_frequencies_hz`` and ``qubit_shifts_hz`` were built by ``full_calibration``, landed in ``**kw`` and were
    discarded: the entangling scans were the only experiments in the whole calibration running in a PERFECT qubit frame at
    the crystal's true mode frequencies. A ``**kw`` that silently swallows a physics argument is how that survived, so a
    supplied ``modes`` together with ``mode_frequencies_hz`` is now refused rather than one of them dropped."""
    from qutip_trap.control.shaping import gate_modes
    from qutip_trap.experiments.entangling import ms_scan

    fx, sur = two_ion
    beams = fx.entangling_drives[0].beams
    modes = gate_modes(fx.device, (0, 1), (beams[0], beams[1]), nbar={})
    with pytest.raises(ValueError, match="mode_frequencies_hz"):
        ms_scan(fx.device, (0, 1), [1.0], [0.0], table=sur.table, modes=modes, mode_frequencies_hz={2: 2.8e6})
    # and the frequencies the table believes reach the GateModes the scans build
    from qutip_trap.experiments.entangling import _entangling_setup

    kw = dict(
        table=sur.table,
        mode_frequencies_hz={m: e.value + 1234.0 for m, e in sur.table.modes.items()},
    )
    _wf, _ent, _sq, _t, built, _space = _entangling_setup(fx.device, (0, 1), dict(kw))
    for m, w in zip(built.modes, built.omega_rad_s):
        assert w / TWO_PI == pytest.approx(fx.device.crystal.modes[m].omega_hz + 1234.0, rel=1e-12)


@pytest.mark.slow
def test_a_wrong_qubit_frequency_shifts_the_ms_phase_scans_correction_by_the_frame_phase(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The frame error the table's qubit frequency leaves is a sigma_z rotation at 2 pi delta_f, so the entangling axis the
    phase scan measures rotates during the gate and the correction it writes moves with it. With ``qubit_shifts_hz``
    discarded the corrections were IDENTICAL for any table frequency and omitted exactly the error ``run()`` then applies.

    The reported offset is in spin-phase units - the spin-phase offset that would give the same parity fringe shift - so the
    coefficient is not 2 pi delta_f t_gate itself: a drift that grows across the pulse shifts the fringe by less than a
    constant spin-phase offset of its final size, and the measured factor is 0.611 t_gate on this five-segment AM waveform
    (which the amplitude-weighted time of its segments and the analysis pulse's own timing set; not derived here). What IS a
    physics statement and is asserted: the shift is LINEAR in the frequency error, of the order of 2 pi delta_f t_gate, and
    lands on the shifted ion alone, because the per-ion corrections are the half-sum and half-difference of the |00> and
    |01> scans' offsets."""
    from qutip_trap.experiments.entangling import ms_phase_scan

    fx, sur = two_ion
    wf = sur.table.waveform_for((0, 1))
    assert wf is not None
    phases = [float(x) for x in np.linspace(0.0, math.pi, 4, endpoint=False)]
    kw = dict(
        table=sur.table,
        nbar={m: e.value for m, e in sur.table.nbar.items()},
        inputs=("00", "01"),
        shots=None,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    df = 1e3
    ref = ms_phase_scan(fx.device, (0, 1), phases, **kw)  # type: ignore[arg-type]
    one = ms_phase_scan(fx.device, (0, 1), phases, qubit_shifts_hz={0: df}, **kw)  # type: ignore[arg-type]
    two = ms_phase_scan(fx.device, (0, 1), phases, qubit_shifts_hz={0: 2.0 * df}, **kw)  # type: ignore[arg-type]
    assert ref.converged and one.converged and two.converged, (ref.notes, one.notes, two.notes)

    def delta(res, q: int) -> float:  # type: ignore[no-untyped-def]
        return float(res.fitted[f"correction_rad[{q}]"][0] - ref.fitted[f"correction_rad[{q}]"][0])

    d0, d0_two, d1 = delta(one, 0), delta(two, 0), delta(one, 1)
    scale = TWO_PI * df * float(wf.duration_s)
    assert 0.4 * scale < abs(d0) < 0.8 * scale, (d0, scale)
    assert d0_two / d0 == pytest.approx(2.0, rel=0.05), (d0, d0_two)
    assert abs(d1) < 0.05 * abs(d0), (d0, d1)


# ---- B8 and C: honest entries, and every advertised name runs something ------------------------------------------------------


def test_every_advertised_experiment_name_produces_a_result_a_refusal_or_a_reason(two_ion) -> None:  # type: ignore[no-untyped-def]
    """``calibrate`` advertised ``thermometry``, ``mode_spectroscopy``, ``crosstalk_phase`` and ``ms_phase_scan``;
    ``full_calibration`` honoured none of them as a standalone name, so ``calibrate(method="experiments",
    experiments=('mode_spectroscopy',))`` returned a pure surrogate table with no error and NO NOTE. Each is now an alias
    of the experiment that runs it, and every accepted name leaves one of three honest traces: a result, a refusal with its
    upstream reason, or a note saying why the scan does not exist on this device (``heating_rate`` on the quiet fixture,
    whose 10/ndot delay span is unbounded at ndot = 0)."""
    fx, sur = two_ion
    assert set(EXPERIMENTS) == set(ORDER) | set(ALIASES)
    # the point of this test is that a NAME leaves a trace, not that the physics is right, so the entries that gate the
    # expensive scans are left uncalibrated and the rest of the scans are as small as they go: the field refuses the
    # micromotion, mode, Rabi, Stark and Ramsey fits, and the light shift refuses the entangling ones (whose UPSTREAM does
    # not name the field, so with the field alone they ran their full amplitude, detuning, phase and parity scans once per
    # name, sixteen times over). What is left running is the crosstalk scan (reduced to four durations and three phases),
    # crystal_image, field_scan, detection_histogram and heating_rate, which is the name that reaches the third outcome.
    blocked = dataclasses.replace(
        sur.table,
        field=dataclasses.replace(sur.table.field, status="uncalibrated"),
        stark={k: dataclasses.replace(e, status="uncalibrated") for k, e in sur.table.stark.items()},
    )
    scans = CalibrationScans(
        shots=None,
        ramsey_delays_s=(0.0, 5e-4, 1e-3, 1.5e-3),
        crosstalk_points=4,
        crosstalk_phase_points=3,
        detection_records=100,
        detection_windows_s=(20e-6,),
        micromotion_ranges={},
    )
    for name in EXPERIMENTS:
        resolved = ALIASES.get(name, name)
        report = full_calibration(
            fx.device, experiments=(name,), surrogate=dataclasses.replace(sur, table=blocked), scans=scans
        )
        assert report.experiments == (resolved,), (name, report.experiments)
        ran = any(k == resolved or k.startswith(f"{resolved}[") for k in report.results)
        said = any(resolved in n for n in report.notes)
        assert ran or resolved in report.refused or said, (
            name,
            report.results.keys(),
            report.refused,
            report.notes,
        )


def test_a_device_with_no_entangling_drive_calibrates_and_runs(two_ion) -> None:  # type: ignore[no-untyped-def]
    """``surrogate_table`` seeded the ENTANGLING drives' own carrier Rabi frequencies with ``for i in range(n): ent[i]``,
    which assumed every ion has one: the 40Ca+ single-qubit preset has gate drives and no entangling drives at all and
    raised ``KeyError: 0`` before ``run()`` could schedule a single GPi2. The loop now walks the drives that exist, so a
    device with none gets no entangling seed and no waveform, and a chain whose pairs are a subset seeds only those."""
    from qutip_trap.device.presets import ca40_optical

    preset = ca40_optical()
    assert preset.entangling_drives == {}, "the 40Ca+ optical preset carries no entangling pair"
    sur = surrogate_table(preset.device, detection_records=200, detection_windows_s=(20e-6,))
    assert sur.table.ms == {}, "no entangling drive, no waveform"
    assert sur.table.rabi, "the single-qubit drives are still seeded"
    circuit = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
    res = run(circuit, preset.device, 20, table=sur.table)
    assert res.shots == 20 and sum(res.probabilities.values()) == pytest.approx(1.0)
    # and a partially addressed chain seeds only the ions its entangling drive names
    fx, _sur2 = two_ion
    partial = surrogate_table(
        dataclasses.replace(
            fx.device, roles=dataclasses.replace(fx.device.roles, entangling={0: fx.entangling_drives[0]})
        ),
        pairs=[],
        detection_records=200,
        detection_windows_s=(20e-6,),
    )
    assert partial.table.ms == {}
    ent_key = (1, fx.entangling_drives[1].table_key_beam)
    assert (
        ent_key not in partial.table.rabi or partial.table.rabi[ent_key].experiment != "derived_raman_drive"
    )


@pytest.mark.slow
def test_the_heating_experiment_runs_end_to_end_on_a_device_with_electric_field_noise() -> None:
    """On the quiet fixture the derived rate is zero, the delay scan (span 10/ndot) does not exist and the entry is left as
    the surrogate's SEED: the first M8 build wrote a ``calibrated`` 0 +- 0 under the name of an experiment that never ran,
    and the M8 end-to-end test pinned it, so the heating experiment was not exercised by the full calibration at all. With
    a non-zero S_E the scan runs and the entry is a real measurement with an uncertainty.

    The seed the span 10/ndot is taken from has to be the rate the noise model will actually heat at: the surrogate read
    only ``S_E``'s TABULATED arrays and missed ``NoiseSpectrum.white_level``, so this device - whose S_E is a pure white
    level - was seeded with zero heating while the engine heated its modes at tens of quanta per second, and the
    experiment had no scan to run on exactly the devices that heat (found here, not in the M8 audit).

    Exact populations, so the reported sigma is the fit's residual scatter and not a statistical error bar: the agreement
    is asserted relatively, as ``tests/test_calibration_experiments.py``'s exact heating row does."""
    fx = circuit_fixture(2)
    noisy = dataclasses.replace(
        fx.device,
        noise=dataclasses.replace(
            fx.device.noise, S_E=white_spectrum(2e-12, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    truth = noisy.noise.heating_rates_quanta_per_s(noisy)
    report = full_calibration(
        noisy,
        seed=11,
        experiments=("heating_rate",),
        pairs=[(0, 1)],
        scans=CalibrationScans(
            shots=None, heating_delays=4, heating_span_over_ndot=1.0, micromotion_ranges={}
        ),
        detection_records=200,
        detection_windows_s=(20e-6,),
    )
    assert not report.refused, report.refused
    # the seed the scan is built from now carries the white level: non-zero on every mode the model heats
    assert all(report.surrogate.table.heating[m].value > 1.0 for m in truth), report.surrogate.table.heating
    measured = [m for m in report.table.heating if report.table.heating[m].status == "calibrated"]
    assert measured, report.notes
    for m in measured:
        e = report.table.heating[m]
        assert e.experiment == "heating_rate" and e.uncertainty > 0.0
        assert e.value == pytest.approx(truth[m], rel=0.03), (m, e, truth[m])


# ---- P2-8: the run-level over-rotation --------------------------------------------------------------------------------------


def test_a_five_percent_rabi_error_in_the_table_over_rotates_the_run(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.10's played chain at the run level: a table whose Rabi belief is 5 % HIGH schedules a GPi 1/1.05 too
    short, the ion sees Omega_phys = Omega_req x Omega_derived/Omega_table = Omega_req/1.05, and the achieved rotation is
    pi/1.05, so the excited population misses by cos^2(pi/2/1.05) = 5.58e-3. The M8 suite pinned the played envelope but
    nothing at the run level, where a chain that silently played the request would have shown no error at all."""
    fx, sur = two_ion
    key = (0, fx.gate_drives[0].table_key_beam)
    high = dataclasses.replace(
        sur.table,
        rabi={
            **sur.table.rabi,
            key: dataclasses.replace(sur.table.rabi[key], value=1.05 * sur.table.rabi[key].value),
        },
    )
    # the schedule: the pi time shortens by exactly 1/1.05
    ref_sched = schedule(GPI, fx.device, sur.table)
    high_sched = schedule(GPI, fx.device, high)
    assert high_sched.pulses[0].duration_s / ref_sched.pulses[0].duration_s == pytest.approx(
        1.0 / 1.05, rel=1e-12
    )

    # the physics: the excited population of ion 0 after the GPi
    def loss(table) -> float:  # type: ignore[no-untyped-def]
        import qutip as qt

        res = run(
            GPI,
            fx.device,
            5,
            table=table,
            keep_final_state=True,
            numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-3)),
        )
        rho = last_record(res).register_state
        assert rho is not None
        proj = qt.tensor(qt.basis(2, 1).proj(), qt.qeye(2))
        return 1.0 - float(np.real(qt.expect(proj, rho)))

    analytic = math.cos(math.pi / 2.0 / 1.05) ** 2
    assert analytic == pytest.approx(5.58e-3, rel=2e-2), "1 - sin^2(pi/2/1.05)"
    baseline, over = loss(sur.table), loss(high)
    assert baseline < 1e-3, baseline
    assert over > 0.8 * analytic, (over, analytic, baseline)
    assert over < 2.0 * analytic, (over, analytic, baseline)
