"""The calibration layer without the long experiments (PLAN.md Section 7): the played chain, the scheduler's Stark
compensation, the servo, the cache, the lineshape fit, the crystal image, ``Device.derived()``, the dependency refusal and
table edits as proposals."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.calibration import CalibrationCache, cached_surrogate, calibrate
from qutip_trap.calibration.entangling import frame_rotated, spot_check_space
from qutip_trap.calibration.experiments import UPSTREAM, full_calibration, upstream_status
from qutip_trap.control.compiler import Circuit, Operation, compile_report
from qutip_trap.control.hardware import physical_schedule
from qutip_trap.control.native import gpi2
from qutip_trap.control.schedule import (
    PhaseFrame,
    Schedule,
    carrier_rabi_hz,
    compensation_phase_rad,
    crosstalk_beliefs,
    entangling_pulses,
    frame_after,
    ms_spin_phases,
    resolve_drives,
    schedule,
    single_qubit_pulse,
    stark_phase_rad,
)
from qutip_trap.control.shaping import gate_modes
from qutip_trap.control.table import ENTRY_KINDS, CalEntry, CalibrationTable
from qutip_trap.device.model import BeamRoles
from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec
from qutip_trap.experiments.fitting import (
    Observation,
    fit_lineshape,
    half_rabi_lineshape,
    lineshape_model,
    sideband_lineshape,
)
from qutip_trap.experiments.imaging import crystal_image
from qutip_trap.experiments.result import (
    CrystalImage,
    ExperimentResult,
    HeatingRateFit,
    ParityScan,
    RabiScan,
    RamseyFringe,
)
from qutip_trap.experiments.single_ion import rabi_scan, sub_stream
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive
from qutip_trap.machine import Machine
from qutip_trap.noise.model import NoiseModel, servo_residual
from qutip_trap.noise.sampling import KEY_FIELD_OFFSET_T, correlated_normals, quiet_sample
from qutip_trap.noise.spectra import Drift
from qutip_trap.options import Numerics
from qutip_trap.provenance import load_ledger
from qutip_trap.run.results import RunState
from qutip_trap.units import TWO_PI
from tests.fixtures import BELL, WINDOWS, make_device, single_ion_raman_device, two_ion_surrogate


@pytest.fixture(scope="module")
def two_ion():
    fx = yb171_chain(2)
    sur = two_ion_surrogate(1000)
    return fx, sur


# ---- the sideband lineshape fit ---------------------------------------------------------------------------------------------------


def test_sideband_lineshape_row_pi_time_half_depth_and_the_half_rabi_negative_control() -> None:
    """The sideband lineshape gives full transfer at Omega t = pi and half at pi/2; fitted to a synthetic scan the full form
    returns Omega and pi/Omega to 1e-6 and the half-Rabi form Omega/2."""
    omega = TWO_PI * 50e3
    t_pi = math.pi / omega
    assert sideband_lineshape(0.0, omega, t_pi) == pytest.approx(1.0)
    assert sideband_lineshape(0.0, omega, 0.5 * t_pi) == pytest.approx(0.5)
    # the half-Rabi form is the full form at twice its Omega
    d = TWO_PI * np.linspace(-150e3, 150e3, 61)
    assert np.allclose(
        half_rabi_lineshape(d, 0.5 * omega, t_pi), sideband_lineshape(d, omega, t_pi), atol=1e-14
    )
    detunings_hz = np.linspace(-120e3, 120e3, 49)
    y = lineshape_model(np.array([0.0, 50e3, 1.0, 0.0]), detunings_hz, t_pi)
    fit_plan = fit_lineshape(detunings_hz, y, t_pi, guess=(0.0, 40e3))
    fit_half = fit_lineshape(detunings_hz, y, t_pi, guess=(0.0, 40e3), form="half_rabi")
    assert fit_plan.converged and fit_plan.params[1] == pytest.approx(50e3, rel=1e-6)
    assert math.pi / (TWO_PI * fit_plan.params[1]) == pytest.approx(t_pi, rel=1e-6), (
        "the fitted pi time is pi/Omega"
    )
    assert fit_half.params[1] == pytest.approx(25e3, rel=1e-6), (
        "the half-Rabi form returns Omega/2 (negative control)"
    )


# ---- the played chain and the Stark compensation ------------------------------------------------------------------------------------


def test_played_chain_is_the_identity_on_a_surrogate_table_and_scales_a_miscalibrated_rabi_entry(
    two_ion,
) -> None:
    fx, sur = two_ion
    rep = compile_report(BELL)
    sched = schedule(rep.circuit, fx.device, sur.table)
    played, notes = physical_schedule(fx.device, sched, sur.table)
    assert all(p.drive.programmed for p in sched.pulses) and not any(
        p.drive.programmed for p in played.pulses
    )
    for a, b in zip(sched.pulses, played.pulses):
        for ta, tb in zip(a.drive.tones, b.drive.tones):
            assert float(tb.envelope_hz) == pytest.approx(float(ta.envelope_hz), rel=1e-12)
        assert float(b.drive.stark_shift_hz) == pytest.approx(float(a.drive.stark_shift_hz), rel=1e-9)
        assert b.drive.crosstalk.keys() == a.drive.crosstalk.keys()
    assert not any("no usable Rabi entry" in n for n in notes)
    # a table whose Rabi belief for ion 0 is 1 % low plays 1 % MORE light than requested (Omega_true/Omega_table), and the light
    # shift follows the played intensity, whatever the table believes
    key = (0, fx.gate_drives[0].table_key_beam)
    low = dataclasses.replace(
        sur.table,
        rabi={
            **sur.table.rabi,
            key: dataclasses.replace(sur.table.rabi[key], value=0.99 * sur.table.rabi[key].value),
        },
    )
    sched_low = schedule(rep.circuit, fx.device, low)
    played_low, _ = physical_schedule(fx.device, sched_low, low)
    single = [
        (a, b)
        for a, b in zip(sched_low.pulses, played_low.pulses)
        if a.drive.ions == (0,) and a.drive.beams == fx.gate_drives[0].beams
    ]
    assert single
    for a, b in single:
        assert float(b.drive.tones[0].envelope_hz) == pytest.approx(
            float(a.drive.tones[0].envelope_hz) / 0.99, rel=1e-9
        )
        derived = derive_raman_drive(fx.device, 0, fx.gate_drives[0].beams, scattering=False)
        assert float(b.drive.stark_shift_hz) == pytest.approx(
            derived.stark_shift_hz * (float(b.drive.tones[0].envelope_hz) / derived.carrier_rabi_hz), rel=1e-9
        )
        assert abs(b.drive.crosstalk[1]) == pytest.approx(
            abs(crosstalk_ratios(fx.device, 0, fx.gate_drives[0].beams)[1])
        )


def test_scheduler_compensates_the_believed_stark_shift_on_every_tone(two_ion) -> None:
    """Every tone is detuned by the table's Stark shift for its amplitude (to 1e-9), both MS legs by the same amount, none with
    stark_compensation=False, and the crosstalk belief carries the table's phase."""
    fx, sur = two_ion
    rep = compile_report(BELL)
    on = schedule(rep.circuit, fx.device, sur.table)
    off = schedule(rep.circuit, fx.device, sur.table, stark_compensation=False)
    for p_on, p_off in zip(on.pulses, off.pulses):
        belief = p_on.drive.stark_shift_hz
        assert belief == p_off.drive.stark_shift_hz
        if len(p_on.drive.tones) == 1:
            shift = float(belief)
            assert shift != 0.0
            assert float(p_on.drive.tones[0].detuning_hz) - float(
                p_off.drive.tones[0].detuning_hz
            ) == pytest.approx(shift, rel=1e-9)
        else:
            shifts = [
                float(a.detuning_hz) - float(b.detuning_hz)
                for a, b in zip(p_on.drive.tones, p_off.drive.tones)
            ]
            assert shifts[0] == pytest.approx(shifts[1], rel=1e-9) and shifts[0] != 0.0
            assert shifts[0] == pytest.approx(float(belief), rel=1e-9)
    # the crosstalk belief carries a phase when the table has one
    table = dataclasses.replace(
        sur.table,
        crosstalk_phase={
            (0, 1): CalEntry(0.3, 0.01, "calibrated", "crosstalk_scan", "conv.crosstalk_ratio", 0.0, 0)
        },
    )
    xt = crosstalk_beliefs(table, 0)
    assert abs(xt[1]) == pytest.approx(sur.table.crosstalk[(0, 1)].value) and np.angle(
        xt[1]
    ) == pytest.approx(0.3)


# ---- the servo ----------------------------------------------------------------------------------------------------------------------


def test_servo_high_passes_a_slow_drift_into_its_residual_band() -> None:
    """A servo of bandwidth f_s leaves an OU drift the residual variance sigma^2/(1 + 2 pi f_s tau) to 50 %, zero at the first
    sample and the drift itself at f_s = 0; through the noise model it holds field offsets below 0.2 of the rms."""
    rng = np.random.default_rng(1)
    times = np.linspace(0.0, 100.0, 4001)
    tau = 5.0
    x = correlated_normals(rng, times, tau, 1)[:, 0]
    for f_s in (0.02, 0.2, 2.0):
        r = servo_residual(x[:, None], times, f_s)[:, 0]
        assert r[0] == 0.0
        expected = 1.0 / (1.0 + TWO_PI * f_s * tau)
        assert np.var(r) == pytest.approx(expected, rel=0.5), (f_s, np.var(r), expected)
    assert np.array_equal(servo_residual(x[:, None], times, 0.0)[:, 0], x)
    # through the noise model: a field drift with a servo leaves the later samples' offsets far below the rms
    dev = make_device()
    noisy = dataclasses.replace(NoiseModel(), field_drift=Drift(1e-6, 10.0, 5.0))
    free = dataclasses.replace(NoiseModel(), field_drift=Drift(1e-6, 10.0, None))
    ts = np.linspace(0.0, 20.0, 201)
    seq_servo = noisy.sample_sequence(np.random.default_rng(0), ts, device=dev)
    seq_free = free.sample_sequence(np.random.default_rng(0), ts, device=dev)
    off_servo = np.array([s.values[KEY_FIELD_OFFSET_T] for s in seq_servo])
    off_free = np.array([s.values[KEY_FIELD_OFFSET_T] for s in seq_free])
    # the residual is the drift's increment between re-locks, sigma sqrt(2 dt/tau) = 0.14 sigma here; the free chain's std over a
    # window of two correlation times underestimates its rms, so the comparison is with the rms itself
    assert off_servo[0] == 0.0 and np.std(off_servo[1:]) < 0.2 * 1e-6 < 3.0 * np.std(off_free)


# ---- the cache ------------------------------------------------------------------------------------------------------------------------


def test_calibration_cache_hits_the_same_device_and_misses_a_changed_one(two_ion) -> None:
    fx, _sur = two_ion
    cache = CalibrationCache()
    kw = dict(
        pairs=[(0, 1)],
        detection_records=200,
        detection_windows_s=(20e-6,),
        spot_check=False,
    )
    t1 = calibrate(Machine(fx.device), cache=cache, **kw).table
    t2 = calibrate(Machine(fx.device), cache=cache, **kw).table
    assert t1 is t2 and len(cache.reports) == 1
    assert t1.is_current_for(fx.device.hash())
    changed = dataclasses.replace(
        fx.device,
        beams=fx.device.beams[:-1]
        + (dataclasses.replace(fx.device.beams[-1], power_w=fx.device.beams[-1].power_w * 1.1),),
    )
    t3 = calibrate(Machine(changed), cache=cache, **kw).table
    assert t3 is not t1 and not t1.is_current_for(changed.hash()) and t3.is_current_for(changed.hash())
    assert calibrate(Machine(changed), cache=cache, **kw).table is t3 and len(cache.reports) == 2
    # the roles are not in the device hash but they are in the key: another entangling assignment is another table
    single = dataclasses.replace(
        fx.device, roles=dataclasses.replace(fx.device.roles, entangling={0: fx.entangling_drives[0]})
    )
    assert calibrate(Machine(single), cache=cache, **kw).table is not t1


def test_a_run_and_a_caller_asking_for_the_same_pairs_share_one_cache_entry() -> None:
    """A run's request (its circuit's pairs as written, the explicit drive maps) and a caller's (a list of the same pairs in
    another order and orientation) are one request: one surrogate is built for both, its pairs keyed (lower, higher);
    no pairs is every pair, and a pair that is not two distinct ions of the crystal is refused."""
    fx = yb171_chain(3)
    machine = Machine(fx.device)
    cheap = dict(detection_records=200, detection_windows_s=(20e-6,), spot_check=False)
    cache = CalibrationCache()
    circuit = Circuit(
        3, (Operation("ms", (1, 2), (0.0, 0.0, 1.0)), Operation("ms", (1, 0), (0.0, 0.0, 1.0))), (0, 1, 2)
    )
    drives, ent = resolve_drives(fx.device)
    ran = cached_surrogate(
        fx.device,
        seed=0,
        t0_s=machine.physics.t0_s,
        cache=cache,
        gate_drives=drives,
        entangling_drives=ent,
        options=machine.numerics,
        builder_options=machine.physics.builder,
        hardware_chain=machine.physics.hardware_chain,
        pairs=circuit.entangling_pairs(),
        **cheap,
    )
    asked = calibrate(machine, cache=cache, pairs=[[0, 1], [2, 1]], **cheap)
    assert asked.surrogate is ran and len(cache.reports) == 1
    assert set(ran.table.ms) == set(ran.mode_classes) == {(0, 1), (1, 2)}
    every = calibrate(machine, cache=cache, **cheap).surrogate
    assert calibrate(machine, cache=cache, pairs=[(0, 2), (1, 2), (0, 1)], **cheap).surrogate is every
    assert len(cache.reports) == 2
    with pytest.raises(ValueError, match="two distinct ions"):
        calibrate(machine, cache=cache, pairs=[(1, 1)], **cheap)


# ---- the dependency graph ---------------------------------------------------------------------------------------------------------------


def test_a_mode_frequency_fit_with_micromotion_uncalibrated_refuses_to_run(two_ion) -> None:
    fx, sur = two_ion
    bad = dataclasses.replace(
        sur.table,
        micromotion={
            "beta[2]": CalEntry(
                0.0,
                0.0,
                "uncalibrated",
                "micromotion_scan",
                "anchor.trap.berkeland_excess_micromotion",
                0.0,
                0,
            )
        },
    )
    assert upstream_status(bad, UPSTREAM["sideband_spectroscopy"]) == "micromotion"
    assert upstream_status(sur.table, UPSTREAM["sideband_spectroscopy"]) is None
    report = full_calibration(
        fx.device, experiments=("sideband_spectroscopy",), surrogate=dataclasses.replace(sur, table=bad)
    )
    assert (
        "sideband_spectroscopy" in report.refused and "micromotion" in report.refused["sideband_spectroscopy"]
    )
    assert report.results == {}
    # what the refused fit could not establish is uncalibrated, NOT the derived seed it started from
    t = report.table
    assert all(
        e.status == "uncalibrated" and e.experiment == "sideband_spectroscopy" for e in t.modes.values()
    )
    assert all(e.status == "uncalibrated" for e in t.nbar.values())
    assert {k for k in t.uncalibrated() if k.startswith("modes")} == {
        f"modes[{m!r}]" for m in sur.table.modes
    }
    assert all(e.value == sur.table.modes[m].value for m, e in t.modes.items()), (
        "the value is kept for the record; only the status refuses"
    )
    for name in ("stark_scan", "crosstalk_scan", "field_scan", "crystal_image"):
        assert name in UPSTREAM


# ---- the crystal image --------------------------------------------------------------------------------------------------------------


def test_crystal_image_sees_the_nominal_chain_and_a_dark_ion(two_ion) -> None:
    fx, _sur = two_ion
    img = crystal_image(Machine(fx.device), shots=1)
    assert img.converged and img.fitted["n_bright"][0] == 2.0 and img.fitted["n_dark"][0] == 0.0
    assert img.fitted["counts[0]"][0] > 50.0
    dark = crystal_image(
        Machine(fx.device), run_state=RunState((0, 1), frozenset({1}), frozenset(), ()), shots=1
    )
    assert not dark.converged and dark.fitted["n_dark"][0] == 1.0 and dark.fitted["bright[1]"][0] == 0.0
    assert any("read dark" in n for n in dark.notes)
    lost = crystal_image(Machine(fx.device), run_state=RunState((0, 1), frozenset(), frozenset({0}), ()))
    assert lost.fitted["n_lost"][0] == 1.0 and lost.fitted["bright[0]"][0] == 0.0


# ---- Device.derived() -----------------------------------------------------------------------------------------------------------------


def test_device_derived_reports_the_calibration_seeds_with_ledger_ids(two_ion) -> None:
    fx, sur = two_ion
    ledger = load_ledger()
    d = fx.device.derived()
    assert set(d.values) == set(d.provenance)
    assert all(pid in ledger for pid in d.provenance.values()), sorted(
        set(d.provenance.values()) - set(ledger)
    )
    assert d.values["qubit_freq_hz[0]"] == pytest.approx(sur.table.qubit_freq[0].value)
    assert d.values["mode_hz[3]"] == pytest.approx(sur.table.modes[3].value)
    assert d.values["R_bright_per_s[0]"] > 1e6
    # the preset's roles name which of its six far-detuned beams play which gates, so the device derives the addressing
    # pairs' Rabi frequencies: the numbers the surrogate seeds its table with
    assert d.values["rabi_hz[(0, 2)]"] == pytest.approx(sur.table.rabi[(0, 2)].value)
    assert d.values["rabi_hz[(1, 4)]"] == pytest.approx(sur.table.rabi[(1, 4)].value)
    # without the roles the same beams do not identify ONE single-qubit drive: the device says so instead of guessing
    bare = dataclasses.replace(fx.device, roles=BeamRoles()).derived()
    assert not any(k.startswith("rabi_hz") for k in bare.values) and any(
        "gate drives" in n for n in bare.notes
    )
    # an unambiguous device derives its drive: the Rabi frequency, the light shift, the Lamb-Dicke parameters
    single = single_ion_raman_device()
    dd = derive_raman_drive(single, 0, (0, 1), scattering=False)
    ds = single.derived()
    assert ds.values["rabi_hz[(0, 0)]"] == pytest.approx(dd.carrier_rabi_hz)
    assert ds.values["stark_hz[(0, 0)]"] == pytest.approx(dd.stark_shift_hz)
    assert (
        ds.values["eta[(0, 1)]"] == pytest.approx(dd.etas[1])
        and ds.provenance["eta[(0, 1)]"] == "conv.lamb_dicke"
    )
    assert ds.values["secular_hz[x]"] == pytest.approx(3.0e6)


# ---- the phase reference of a compensation detuning ---------------------------------------------------------------------------------


def test_compensated_tones_are_referenced_to_the_pulse_start_and_the_frame_inside_a_gate(two_ion) -> None:
    """Every compensated tone carries compensation_phase_rad = 2 pi delta_s t_s at its own start (to 1e-12), minus the frame the
    gate's earlier segments accumulated, so a GPi2 at t = 1 ms keeps its t = 0 fidelity to 1e-6 where the uncorrected axis
    error would exceed 1e-2."""
    fx, sur = two_ion
    table = sur.table
    spec = fx.gate_drives[0]
    rabi = carrier_rabi_hz(table, 0, spec)
    shift = table.stark[(0, spec.table_key_beam)].value
    assert shift != 0.0

    def wrapped(x: float) -> float:
        return float((x + math.pi) % TWO_PI - math.pi)

    # the single-qubit tone: phi + 2 pi delta_s t_s
    t_s = 1e-3
    p = single_qubit_pulse(0, math.pi / 2, 0.4, spec, rabi, t_s, stark_shift_hz=shift)
    assert wrapped(float(p.drive.tones[0].phase_rad) - 0.4 - TWO_PI * shift * t_s) == pytest.approx(
        0.0, abs=1e-12
    )
    assert compensation_phase_rad(0.0, t_s) == 0.0 and compensation_phase_rad(shift, 0.0) == 0.0
    # the segments of the amplitude-modulated waveform: each at its own start, minus the frame the gate accumulated so far
    wf = table.waveform_for((0, 1))
    assert wf is not None and len(wf.segments) > 1
    spins, _ = ms_spin_phases(wf, (0, 1), (math.pi, 0.0), PhaseFrame())
    kw = dict(spin_phases_rad=spins, t_start_s=t_s, table=table, gate_id="ms")
    on = entangling_pulses(wf, fx.entangling_drives, **kw)
    off = entangling_pulses(wf, fx.entangling_drives, stark_compensation=False, **kw)
    accumulated = {0: 0.0, 1: 0.0}
    distinct = set()
    for p_on, p_off in zip(on, off):
        ion = p_on.drive.ions[0]
        delta = p_on.drive.stark_shift_hz
        distinct.add(round(float(delta), 6))
        expected = compensation_phase_rad(delta, p_on.t_start_s) - accumulated[ion]
        for a, b in zip(p_on.drive.tones, p_off.drive.tones):
            assert wrapped(float(a.phase_rad) - float(b.phase_rad) - expected) == pytest.approx(
                0.0, abs=1e-12
            )
        accumulated[ion] += stark_phase_rad(p_on)
    assert len(distinct) > 1, "the AM waveform's segments carry different shifts"
    assert frame_after(on, PhaseFrame()).as_dict(2)[0] == pytest.approx(accumulated[0])
    # the physics: a compensated GPi2 at t = 0 and one millisecond later reach the same frame-rotated fidelity
    modes = gate_modes(
        fx.device, (0, 1), fx.entangling_drives[0].beams, nbar={m: e.value for m, e in table.nbar.items()}
    )
    space = spot_check_space(fx.device, modes, wf, (0, 1), Numerics())[0]
    engine = JointExactEngine(table=table)
    ideal = qt.Qobj(np.kron(gpi2(0.0), np.eye(2)), dims=[[2, 2], [2, 2]])
    ket0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    losses = []
    for start in (0.0, t_s):
        pulse = single_qubit_pulse(0, math.pi / 2, 0.0, spec, rabi, start, stark_shift_hz=shift, gate_id="g")
        frame = frame_after([pulse], PhaseFrame()).as_dict(2)
        sched = Schedule((pulse,), ((0.0, start),) if start > 0 else (), (), frame)
        traces = engine.run_pulses(
            fx.device, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), Numerics()
        )
        losses.append(
            1.0 - float(np.real(qt.expect(traces.final.internal, frame_rotated(ideal, frame) * ket0)))
        )
    assert losses[0] < 1e-4 and abs(losses[1] - losses[0]) < 1e-6, losses
    assert (TWO_PI * shift * t_s) ** 2 / 4.0 > 1e-2, (
        "the uncorrected axis error this fixture would have shown"
    )


# ---- independent shot noise per experiment and sub-run --------------------------------------------------------------------------------


def test_every_experiment_and_sub_run_draws_its_own_shot_noise() -> None:
    """Observations with different ``stream`` labels draw independent shot noise, the same label replays the same draws,
    ``sub_stream`` composes labels, and exact observations ignore them."""
    base = Observation(shots=400, seed=3)
    beam0 = Observation(shots=400, seed=3, stream="stark_scan[0]/beam2")
    beam1 = Observation(shots=400, seed=3, stream="stark_scan[0]/beam3")
    draws = [[o.p1(0.5, 0, "ramsey", k)[0] for k in range(12)] for o in (base, beam0, beam1)]
    assert draws[0] != draws[1] and draws[1] != draws[2] and draws[0] != draws[2]
    # deterministic and order-free: the same stream replays the same draws
    assert draws[1] == [
        Observation(shots=400, seed=3, stream="stark_scan[0]/beam2").p1(0.5, 0, "ramsey", k)[0]
        for k in range(12)
    ]
    # the label composes from the parent experiment's stream
    assert sub_stream({"seed": 3}, "beam2")["stream"] == "beam2"
    assert sub_stream({"seed": 3, "stream": "stark_scan[0]"}, "beam2")["stream"] == "stark_scan[0]/beam2"
    # exact observations are untouched by the label
    assert Observation(stream="x").p1(0.3, 0, "ramsey", 0) == (0.3, None)


# ---- table edits are proposals ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def machine() -> Machine:
    return Machine(yb171_chain(2).device).calibrated(
        pairs=[(0, 1)], detection_records=300, detection_windows_s=WINDOWS
    )


def test_with_params_merges_mapping_fields_and_refuses_unknown_names(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    entry = table.rabi[(0, 2)]
    wrong = table.with_params(
        rabi={(0, 2): dataclasses.replace(entry, value=1.02 * entry.value, status="calibrated")}
    )
    assert (
        wrong.rabi[(0, 2)].value == pytest.approx(1.02 * entry.value)
        and wrong.rabi[(0, 2)].status == "calibrated"
    )
    assert (
        wrong.rabi[(1, 4)] is table.rabi[(1, 4)] and wrong.device_hash == table.device_hash
    )  # merged, not replaced
    assert (
        wrong.modes == table.modes and wrong is not table and table.rabi[(0, 2)] is entry
    )  # the original is untouched
    with pytest.raises(TypeError, match="unknown or unsettable field"):
        table.with_params(rabbi={})
    with pytest.raises(TypeError, match="device_hash"):
        table.with_params(device_hash="0" * 64)
    with pytest.raises(TypeError, match="takes a mapping"):
        table.with_params(rabi=3.0)
    # a waveform under the other key order replaces the stored pair rather than adding a second one
    wf = table.waveform_for((0, 1))
    assert wf is not None
    marked = dataclasses.replace(wf, phi_s=dataclasses.replace(wf.phi_s, value=wf.phi_s.value + 0.1))
    swapped = table.with_params(ms={(1, 0): marked})
    assert set(swapped.ms) == set(table.ms) and swapped.waveform_for((0, 1)) is marked
    # a scalar field is replaced
    assert (
        table.with_params(fitted_at_s=7.5).fitted_at_s == 7.5
        and table.with_params(surrogate=False).surrogate is False
    )


def test_updated_with_a_rabi_scan_changes_exactly_the_rabi_entry_with_the_stamps(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    scan = rabi_scan(machine, 0, np.linspace(0.0, 20e-6, 9), shots=200, seed=1)
    assert isinstance(scan, RabiScan) and scan.converged and scan.subject == {"ion": 0, "beam": 2}
    proposal = table.updated_with(scan, fitted_at_s=12.5, sample_id=3)
    changed = {k for k in table.entries() if table.entries()[k] != proposal.entries().get(k)}
    assert changed == {"rabi[(0, 2)]"}
    new = proposal.rabi[(0, 2)]
    assert new.value == scan.f_rabi_hz and new.uncertainty == scan.uncertainty("f_rabi_hz")
    assert (
        new.status == "calibrated"
        and new.experiment == "rabi_scan"
        and new.provenance_id == scan.provenance_id
    )
    assert new.fitted_at_s == 12.5 and new.sample_id == 3 and proposal.fitted_at_s == 12.5
    assert proposal.kind_of("rabi[(0, 2)]") == "setpoint" and proposal.ms == table.ms
    # the default time is the table's own; a failed fit proposes an uncalibrated entry
    assert table.updated_with(scan).rabi[(0, 2)].fitted_at_s == table.fitted_at_s
    failed = dataclasses.replace(scan, converged=False)
    assert table.updated_with(failed).rabi[(0, 2)].status == "uncalibrated" and failed.quality == "failed"
    # the results that set no entry refuse rather than returning the table unchanged
    for bare in (
        ParityScan(data=np.zeros((0, 2)), fitted={}, model="parity_oscillation", provenance_id="p"),
        CrystalImage(data=np.zeros((0, 2)), fitted={}, model="crystal_image", provenance_id="p"),
        RamseyFringe(
            data=np.zeros((0, 2)), fitted={"delta_hz": (1.0, 0.1)}, model="ramsey_fringe", provenance_id="p"
        ),
        ExperimentResult(data=np.zeros((0, 2)), fitted={}, model="x", provenance_id="p"),
    ):
        with pytest.raises(ValueError, match="sets no|only ramsey_frequency"):
            table.updated_with(bare)
    heat = HeatingRateFit(
        data=np.zeros((0, 2)),
        fitted={"ndot_per_s": (12.0, 1.0)},
        model="heating_rate_sideband_asymmetry",
        provenance_id="anchor.trap.heating_dynamics",
        subject={"mode": 3},
    )
    assert table.updated_with(heat).heating[3].value == 12.0 and heat.experiment == "heating_rate"


def test_entry_kinds_partition_the_table(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    setpoints = table.entries(kind="setpoint")
    characterisation = table.entries(kind="characterisation")
    assert set(setpoints) | set(characterisation) == set(table.entries()) and not (
        set(setpoints) & set(characterisation)
    )
    assert "rabi[(0, 2)]" in setpoints and "field" in characterisation and "ms[(0, 1)].phi_s" in setpoints
    assert all(
        table.kind_of(k) == "characterisation" for k in table.entries() if k.startswith(("nbar[", "heating["))
    )
    assert set(ENTRY_KINDS) == {f.name for f in dataclasses.fields(table)} - {
        "device_hash",
        "seed",
        "surrogate",
        "fitted_at_s",
    }
    with pytest.raises(KeyError):
        table.kind_of("rabi[(9, 9)]")


def test_entries_and_the_table_survive_the_json_round_trip(machine: Machine) -> None:
    table = machine.table
    assert table is not None
    entry = table.rabi[(0, 2)]
    assert CalEntry.from_dict(entry.to_dict()) == entry
    back = CalibrationTable.from_dict(table.to_dict())
    assert back.rabi[(0, 2)] == entry and back.ms == {}
    assert {k: e for k, e in back.entries().items()} == {
        k: e for k, e in table.entries().items() if not k.startswith("ms[")
    }
