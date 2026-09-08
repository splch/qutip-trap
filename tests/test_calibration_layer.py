"""The M8 calibration layer without the long experiments (PLAN.md Sections 7.3, 7.5, 7.10, 9.17): the played chain
(requested -> physical through the device), the scheduler's Stark compensation, the servo of Section 7.5, the calibration
cache, the sideband-lineshape row of Section 9.17 as a pure fit test, the crystal image, ``Device.derived()`` and the
dependency-graph refusal."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.api import CalEntry, Circuit, Drift, Operation, calibrate, compile_with_report, schedule
from qutip_trap.calibration import DEFAULT_CACHE, CalibrationCache, calibrate_with_report
from qutip_trap.calibration.experiments import UPSTREAM, full_calibration, upstream_status
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.played import physical_schedule
from qutip_trap.control.schedule import crosstalk_beliefs
from qutip_trap.experiments import crystal_image
from qutip_trap.experiments.fitting import (
    fit_lineshape,
    half_rabi_lineshape,
    lineshape_model,
    sideband_lineshape,
)
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive
from qutip_trap.noise.model import servo_residual
from qutip_trap.run.results import RunState
from qutip_trap.units import TWO_PI
from tests.m6_fixtures import circuit_fixture

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(
        fx.device,
        pairs=[(0, 1)],
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        detection_records=1000,
        detection_windows_s=WINDOWS,
    )
    return fx, sur


# ---- Section 9.17, "Sideband lineshape fit" -------------------------------------------------------------------------------------------


def test_sideband_lineshape_row_pi_time_half_depth_and_the_half_rabi_negative_control() -> None:
    """P = [Omega^2/(Omega^2 + delta^2)] sin^2((t/2) sqrt(Omega^2 + delta^2)): a carrier pi pulse at Omega t = pi, the Lorentzian weight
    at half depth for delta = Omega; the plan's form fitted to a synthetic scan returns Omega, the half-Rabi form returns Omega/2."""
    omega = TWO_PI * 50e3
    t_pi = math.pi / omega
    assert sideband_lineshape(0.0, omega, t_pi) == pytest.approx(1.0)
    assert sideband_lineshape(0.0, omega, 0.5 * t_pi) == pytest.approx(0.5)
    delta = np.array([omega])
    weight = omega**2 / (omega**2 + delta**2)
    assert weight[0] == pytest.approx(0.5)
    # the half-Rabi form is the plan's form at twice its Omega (Section 7.9: doubled on ingest)
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


# ---- the played chain and the Stark compensation (Sections 7.3, 7.10) ----------------------------------------------------------------


def test_played_chain_is_the_identity_on_a_surrogate_table_and_scales_a_miscalibrated_rabi_entry(
    two_ion,
) -> None:  # type: ignore[no-untyped-def]
    fx, sur = two_ion
    rep = compile_with_report(BELL, fx.device)
    sched = schedule(
        rep.circuit, fx.device, sur.table, gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives
    )
    played, notes = physical_schedule(fx.device, sched, sur.table)
    assert all(p.drive.programmed for p in sched.pulses) and not any(
        p.drive.programmed for p in played.pulses
    )
    for a, b in zip(sched.pulses, played.pulses):
        for ta, tb in zip(a.drive.tones, b.drive.tones):
            assert float(tb.envelope_hz) == pytest.approx(float(ta.envelope_hz), rel=1e-12)  # type: ignore[arg-type]
        assert float(b.drive.stark_shift_hz) == pytest.approx(float(a.drive.stark_shift_hz), rel=1e-9)  # type: ignore[arg-type]
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
    sched_low = schedule(
        rep.circuit, fx.device, low, gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives
    )
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
        )  # type: ignore[arg-type]
        derived = derive_raman_drive(fx.device, 0, fx.gate_drives[0].beams, scattering=False)
        assert float(b.drive.stark_shift_hz) == pytest.approx(
            derived.stark_shift_hz * (float(b.drive.tones[0].envelope_hz) / derived.carrier_rabi_hz), rel=1e-9
        )  # type: ignore[arg-type]
        assert abs(b.drive.crosstalk[1]) == pytest.approx(
            abs(crosstalk_ratios(fx.device, 0, fx.gate_drives[0].beams)[1])
        )


def test_scheduler_compensates_the_believed_stark_shift_on_every_tone(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5 item 7: every tone is detuned by the shift the table predicts for the played amplitude; both legs of an MS
    segment move together (the spin and motion phases are untouched); the flag switches it off."""
    fx, sur = two_ion
    rep = compile_with_report(BELL, fx.device)
    kw = dict(gate_drives=fx.gate_drives, entangling_drives=fx.entangling_drives)
    on = schedule(rep.circuit, fx.device, sur.table, **kw)  # type: ignore[arg-type]
    off = schedule(rep.circuit, fx.device, sur.table, stark_compensation=False, **kw)  # type: ignore[arg-type]
    for p_on, p_off in zip(on.pulses, off.pulses):
        belief = p_on.drive.stark_shift_hz
        assert belief == p_off.drive.stark_shift_hz
        if len(p_on.drive.tones) == 1:
            shift = float(belief)  # type: ignore[arg-type]
            assert shift != 0.0
            assert float(p_on.drive.tones[0].detuning_hz) - float(
                p_off.drive.tones[0].detuning_hz
            ) == pytest.approx(shift, rel=1e-9)  # type: ignore[arg-type]
        else:
            shifts = [
                float(a.detuning_hz) - float(b.detuning_hz)
                for a, b in zip(p_on.drive.tones, p_off.drive.tones)
            ]  # type: ignore[arg-type]
            assert shifts[0] == pytest.approx(shifts[1], rel=1e-9) and shifts[0] != 0.0
            assert shifts[0] == pytest.approx(float(belief), rel=1e-9)  # type: ignore[arg-type]
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


# ---- the servo of Section 7.5 --------------------------------------------------------------------------------------------------------


def test_servo_high_passes_a_slow_drift_into_its_residual_band() -> None:
    """A first-order lock of bandwidth f_s tracks an OU drift of correlation time tau: the residual variance falls to about
    sigma^2/(1 + 2 pi f_s tau), the first sample carries no offset (the calibration measured it), and a quiet servo changes nothing."""
    from qutip_trap.noise.processes import correlated_normals

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
    from tests.fixtures import make_device, make_noise

    dev = make_device()
    noisy = dataclasses.replace(make_noise(), field_drift=Drift(1e-6, 10.0, 5.0))
    free = dataclasses.replace(make_noise(), field_drift=Drift(1e-6, 10.0, None))
    ts = np.linspace(0.0, 20.0, 201)
    seq_servo = noisy.sample_sequence(np.random.default_rng(0), ts, device=dev)
    seq_free = free.sample_sequence(np.random.default_rng(0), ts, device=dev)
    off_servo = np.array([s.values.get("field_offset_t", 0.0) for s in seq_servo])
    off_free = np.array([s.values.get("field_offset_t", 0.0) for s in seq_free])
    # the residual is the drift's increment between re-locks, sigma sqrt(2 dt/tau) = 0.14 sigma here; the free chain's std over a
    # window of two correlation times underestimates its rms, so the comparison is with the rms itself
    assert off_servo[0] == 0.0 and np.std(off_servo[1:]) < 0.2 * 1e-6 < 3.0 * np.std(off_free)


# ---- the cache (Section 7.5) ----------------------------------------------------------------------------------------------------------


def test_calibration_cache_hits_the_same_device_and_misses_a_changed_one(two_ion) -> None:  # type: ignore[no-untyped-def]
    fx, _sur = two_ion
    cache = CalibrationCache()
    kw = dict(
        pairs=[(0, 1)],
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        detection_records=200,
        detection_windows_s=(20e-6,),
        spot_check=False,
    )
    t1 = calibrate(fx.device, cache=cache, **kw)  # type: ignore[arg-type]
    t2 = calibrate(fx.device, cache=cache, **kw)  # type: ignore[arg-type]
    assert t1 is t2 and cache.hits == 1 and cache.misses == 1
    assert t1.is_current_for(fx.device.hash())
    changed = dataclasses.replace(
        fx.device,
        beams=fx.device.beams[:-1]
        + (dataclasses.replace(fx.device.beams[-1], power_w=fx.device.beams[-1].power_w * 1.1),),
    )
    t3 = calibrate(changed, cache=cache, **kw)  # type: ignore[arg-type]
    assert t3 is not t1 and not t1.is_current_for(changed.hash()) and t3.is_current_for(changed.hash())
    assert cache.invalidate(fx.device) == 1 and cache.tables_for(fx.device) == ()
    report = calibrate_with_report(changed, cache=cache, **kw)  # type: ignore[arg-type]
    assert report.table is t3, "the report is cached beside its table"
    assert isinstance(DEFAULT_CACHE, CalibrationCache)


# ---- the dependency graph (Section 9.17, "Calibration dependency graph") --------------------------------------------------------------


def test_a_mode_frequency_fit_with_micromotion_uncalibrated_refuses_to_run(two_ion) -> None:  # type: ignore[no-untyped-def]
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
        fx.device,
        experiments=("sideband_spectroscopy",),
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        surrogate=dataclasses.replace(sur, table=bad),
    )
    assert (
        "sideband_spectroscopy" in report.refused and "micromotion" in report.refused["sideband_spectroscopy"]
    )
    assert report.results == {}
    # Section 7.3: what the refused fit could not establish is uncalibrated, NOT the derived seed it started from. The
    # first M8 build left the mode frequencies as schedulable seeds and this row pinned that (`report.table.modes ==
    # sur.table.modes`), which is exactly the fallback 7.3 forbids (M8 audit B2).
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


# ---- the crystal image (Section 6.7) ---------------------------------------------------------------------------------------------------


def test_crystal_image_sees_the_nominal_chain_and_a_dark_ion(two_ion) -> None:  # type: ignore[no-untyped-def]
    fx, _sur = two_ion
    img = crystal_image(fx.device, shots=1)
    assert img.converged and img.fitted["n_bright"][0] == 2.0 and img.fitted["n_dark"][0] == 0.0
    assert img.fitted["counts[0]"][0] > 50.0
    dark = crystal_image(fx.device, run_state=RunState((0, 1), frozenset({1}), frozenset(), ()), shots=1)
    assert not dark.converged and dark.fitted["n_dark"][0] == 1.0 and dark.fitted["bright[1]"][0] == 0.0
    assert any("read dark" in n for n in dark.notes)
    lost = crystal_image(fx.device, run_state=RunState((0, 1), frozenset(), frozenset({0}), ()))
    assert lost.fitted["n_lost"][0] == 1.0 and lost.fitted["bright[0]"][0] == 0.0


# ---- Device.derived() -----------------------------------------------------------------------------------------------------------------


def test_device_derived_reports_the_calibration_seeds_with_ledger_ids(two_ion) -> None:  # type: ignore[no-untyped-def]
    from qutip_trap.provenance import load_ledger
    from tests.m2_fixtures import single_ion_raman_device

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
    # six far-detuned beams do not identify ONE single-qubit drive: the device says so instead of guessing (Section 7.3)
    assert not any(k.startswith("rabi_hz") for k in d.values) and any("gate drives" in n for n in d.notes)
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


# ---- the phase reference of a compensation detuning (M8 finding) --------------------------------------------------------------------


def test_compensated_tones_are_referenced_to_the_pulse_start_and_the_frame_inside_a_gate(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The builder plays phase-continuous tones e^{-i(2 pi mu t - phi)} in absolute time, so a compensation detuning delta_s
    programmed without a phase reference rotates the pulse axis by 2 pi delta_s t_s (an uncorrected GPi2 one millisecond into a
    schedule lost 1.5e-2 on this fixture, (2 pi delta_s t_s)^2/4 for delta_s = -38 Hz): every compensated tone carries
    ``compensation_phase_rad`` at its own start and, inside a multi-segment gate, minus the frame the earlier segments
    accumulated; a compensated GPi2 then has the same fidelity at t = 0 and at t = 1 ms."""
    import qutip as qt

    from qutip_trap.api import SeedSpec, SolverOptions
    from qutip_trap.calibration.entangling import frame_rotated, gate_space
    from qutip_trap.control.native import gpi2
    from qutip_trap.control.schedule import (
        Schedule,
        carrier_rabi_hz,
        compensation_phase_rad,
        entangling_pulses,
        frame_after,
        ms_spin_phases,
        single_qubit_pulse,
        stark_phase_rad,
    )
    from qutip_trap.control.shaping import gate_modes
    from qutip_trap.dynamics.engine import JointExactEngine
    from qutip_trap.dynamics.frames import PhaseFrame
    from qutip_trap.noise.sampling import quiet_sample

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
    assert wf is not None and wf.segments is not None and len(wf.segments) > 1
    spins, _ = ms_spin_phases(wf, (0, 1), (math.pi, 0.0), PhaseFrame())
    kw = dict(spin_phases_rad=spins, t_start_s=t_s, table=table, gate_id="ms")
    on = entangling_pulses(wf, fx.entangling_drives, **kw)  # type: ignore[arg-type]
    off = entangling_pulses(wf, fx.entangling_drives, stark_compensation=False, **kw)  # type: ignore[arg-type]
    accumulated = {0: 0.0, 1: 0.0}
    distinct = set()
    for p_on, p_off in zip(on, off):
        ion = p_on.drive.ions[0]
        delta = p_on.drive.stark_shift_hz
        distinct.add(round(float(delta), 6))  # type: ignore[arg-type]
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
    space = gate_space(modes, 2, waveform=wf)
    engine = JointExactEngine(table=table)
    ideal = qt.Qobj(np.kron(gpi2(0.0), np.eye(2)), dims=[[2, 2], [2, 2]])
    ket0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    losses = []
    for start in (0.0, t_s):
        pulse = single_qubit_pulse(0, math.pi / 2, 0.0, spec, rabi, start, stark_shift_hz=shift, gate_id="g")
        frame = frame_after([pulse], PhaseFrame()).as_dict(2)
        sched = Schedule((pulse,), ((0.0, start),) if start > 0 else (), (), frame)
        traces = engine.run_pulses(
            fx.device, sched, space.initial_state([0, 0]), space, quiet_sample(), SeedSpec(0), SolverOptions()
        )
        losses.append(
            1.0 - float(np.real(qt.expect(traces.final.internal, frame_rotated(ideal, frame) * ket0)))
        )
    assert losses[0] < 1e-4 and abs(losses[1] - losses[0]) < 1e-6, losses
    assert (TWO_PI * shift * t_s) ** 2 / 4.0 > 1e-2, (
        "the uncorrected axis error this fixture would have shown"
    )


# ---- independent shot noise per experiment and sub-run (Section 3.4) -----------------------------------------------------------------


def test_every_experiment_and_sub_run_draws_its_own_shot_noise() -> None:
    """The observation model keys its draws by (sample, point index, ion, outcome); a calibration that runs the same experiment
    twice, or an experiment that repeats a scan (the Stark scan's Ramsey per beam), would otherwise replay identical noise and
    report a correlated pair as two independent measurements (the first M8 build read -12.67 Hz for both beams of the fixture's
    drive, exactly). ``stream`` labels keep the runs independent and compose through ``sub_stream``."""
    from qutip_trap.experiments.fitting import Observation
    from qutip_trap.experiments.single_ion import sub_stream

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
