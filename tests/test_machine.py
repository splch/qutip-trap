"""``Machine``: its run equals a run on the bare device at the same seed, its schedule is the scheduler's on the same table,
``calibrated`` pins a table, a forced level shows in the diagnostics, ``estimate`` matches the run that follows, ``hash``
moves when and only when the device, the table or the policy moves; the ``progress`` callback; ``RunSpec`` and the ``Job``
of ``Machine.submit``."""

from __future__ import annotations

import dataclasses
import json
import math
import time
import warnings

import numpy as np
import pytest

import qutip_trap as trap
from qutip_trap.control.compiler import Circuit, Operation, compile_to_native
from qutip_trap.control.schedule import schedule
from qutip_trap.device.model import BeamRoles
from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.dynamics.truncation import TruncationWarning, warn_if_boundary_exceeds
from qutip_trap.machine import Estimate, Machine
from qutip_trap.options import Numerics, Parallel, Physics, Readout, Truncation
from qutip_trap.readout.discriminate import ThresholdDiscriminator
from qutip_trap.run.job import last_record
from qutip_trap.run.levels import FidelityLevel
from qutip_trap.run.results import Progress
from qutip_trap.run.spec import SPEC_SCHEMA_VERSION, Job, JobCancelled, RunSpec, submit
from tests.fixtures import BELL, FAST, WINDOWS, run

ONE = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
SERIAL = Numerics(truncation=Truncation(branch_weight_min=1e-3), parallel=Parallel(map="serial"))


@pytest.fixture(scope="module")
def machine():  # type: ignore[no-untyped-def]
    preset = yb171_chain(2)
    pinned = preset.machine().calibrated(pairs=[(0, 1)], detection_records=2000, detection_windows_s=WINDOWS)
    return preset, dataclasses.replace(pinned, numerics=FAST)


@pytest.fixture(scope="module")
def bell(machine):  # type: ignore[no-untyped-def]
    _preset, m = machine
    return m.run(BELL, 400, seed=0)


def test_the_preset_bridges_to_a_machine_on_its_device() -> None:
    preset = yb171_chain(2)
    m = preset.machine()
    assert isinstance(m, Machine) and m.device is preset.device and m.name == preset.name
    assert m.table is None and m.level is FidelityLevel.AUTO
    assert (m.physics, m.numerics, m.readout) == (Physics(), Numerics(), Readout())


def test_calibrated_pins_a_table_for_the_device(machine) -> None:  # type: ignore[no-untyped-def]
    preset, m = machine
    assert m.table is not None and m.table.device_hash == preset.device.hash() and m.table.surrogate
    assert (0, 2) in m.table.rabi and m.table.waveform_for((0, 1)) is not None
    with pytest.raises(ValueError, match="closed_form"):
        m.calibrated("guess")  # type: ignore[arg-type]


def test_machine_run_equals_run_field_for_field(machine, bell) -> None:  # type: ignore[no-untyped-def]
    preset, m = machine
    reference = run(BELL, preset.device, 400, table=m.table, numerics=FAST, seed=0)
    assert reference.machine_hash == bell.machine_hash
    assert np.array_equal(bell.bitstrings, reference.bitstrings) and np.array_equal(
        bell.heralds, reference.heralds
    )
    for name in (
        "counts",
        "probabilities",
        "error_bars",
        "spam",
        "discarded_shots",
        "run_state",
        "bit_order",
    ):
        assert getattr(bell, name) == getattr(reference, name), name
    assert bell.probabilities["00"] + bell.probabilities["11"] > 0.98
    for f in dataclasses.fields(bell.diagnostics):
        if f.name == "wall_clock_span_s":
            continue
        assert getattr(bell.diagnostics, f.name) == getattr(reference.diagnostics, f.name), f.name


def test_schedule_equals_the_scheduler_on_the_same_table(machine) -> None:  # type: ignore[no-untyped-def]
    preset, m = machine
    mine = m.schedule(BELL)
    theirs = schedule(compile_to_native(BELL), preset.device, m.table)
    assert [p.gate_id for p in mine.pulses] == [p.gate_id for p in theirs.pulses]
    assert [(p.t_start_s, p.t_end_s) for p in mine.pulses] == [
        (p.t_start_s, p.t_end_s) for p in theirs.pulses
    ]
    assert mine.events == theirs.events and mine.phase_frame == theirs.phase_frame
    assert len(mine.gates) == len(theirs.gates) == 1 and mine.gates[0].pair == theirs.gates[0].pair
    assert m.compile(BELL).n_entangling == 1


def test_a_forced_level_reaches_the_diagnostics(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    deeper = dataclasses.replace(m, level="GATE_LOCAL")  # type: ignore[arg-type]  (strings are accepted)
    assert deeper.level is FidelityLevel.GATE_LOCAL
    res = deeper.run(ONE, 20)
    assert res.diagnostics.level == "GATE_LOCAL" and res.diagnostics.gate_local is not None
    assert res.diagnostics.level_reason.startswith("GATE_LOCAL forced by the caller")


def test_estimate_matches_the_diagnostics_of_the_run_that_follows(machine, bell) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    est = m.estimate(BELL)
    assert isinstance(est, Estimate)
    assert est.level == bell.diagnostics.level and est.reason == bell.diagnostics.level_reason
    assert est.space == bell.diagnostics.space and est.mode_class == bell.diagnostics.mode_class
    assert est.dimension == bell.diagnostics.space.dimension and est.nnz > 0
    assert est.n_entangling == 1 and est.n_pulses >= 5
    assert 0.0 < est.duration_s < 1e-3 and math.isfinite(est.wall_time_s) and est.wall_time_s > 0.0
    # a forced GATE_LOCAL estimate reports that level with the reason auto would have given
    forced = dataclasses.replace(m, level=FidelityLevel.GATE_LOCAL).estimate(BELL)
    assert forced.level is FidelityLevel.GATE_LOCAL and "would choose JOINT_EXACT" in forced.reason


def test_hash_changes_when_and_only_when_device_table_or_policy_change(machine) -> None:  # type: ignore[no-untyped-def]
    preset, m = machine
    h = m.hash()
    assert len(h) == 64 and h == m.hash() == dataclasses.replace(m, name="another label").hash()
    assert dataclasses.replace(m, physics=Physics(noise=False)).hash() != h
    assert dataclasses.replace(m, numerics=Numerics()).hash() != h
    assert dataclasses.replace(m, readout=Readout(mode="full")).hash() != h
    assert dataclasses.replace(m, level=FidelityLevel.GATE_LOCAL).hash() != h
    assert dataclasses.replace(m, table=None).hash() != h
    # the roles move the machine's identity even though they leave the device digest alone
    bare = dataclasses.replace(
        preset.device, roles=BeamRoles(gate=preset.gate_drives, entangling=preset.entangling_drives)
    )
    assert bare.hash() == preset.device.hash()
    assert dataclasses.replace(m, device=bare).hash() != h
    assert Machine(preset.device).hash() == Machine(preset.device).hash()


def test_the_engine_is_configured_from_the_machine(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    engine = m.engine
    assert isinstance(engine, JointExactEngine) and engine.table is m.table and engine.device_channels
    assert dataclasses.replace(m, physics=Physics(noise=False)).engine.device_channels is False


def test_specs_reports_the_derived_quantities_the_roles_the_table_and_the_level(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    text = m.specs()
    assert text.startswith(f"device {m.device.hash()[:12]}") and "rabi_hz[(0, 2)] =" in text
    assert "gate drives = {0: GateDrive(" in text and "table = pinned (" in text and "level = auto" in text


# ---- progress and the truncation warnings ---------------------------------------------------------------------------------


def test_progress_is_a_frozen_record() -> None:
    p = Progress("pulse", 2, 6, 0.5)
    assert (p.stage, p.done, p.total, p.elapsed_s) == ("pulse", 2, 6, 0.5) and p.fraction == pytest.approx(
        1 / 3
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.done = 3  # type: ignore[misc]
    with pytest.raises(ValueError):
        Progress("pulse", 7, 6, 0.0)


def test_the_callback_sequence_is_monotone_per_stage_and_complete(machine) -> None:  # type: ignore[no-untyped-def]
    """The serial map keeps every (sample, branch) engine run in-process, where the pulses are reported."""
    _preset, m = machine
    serial = dataclasses.replace(m, numerics=SERIAL)
    seen: list[Progress] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)  # the Bell run is silent
        result = serial.run(BELL, 200, seed=1, progress=seen.append)
    assert result.shots == 200
    stages = {p.stage for p in seen}
    assert {"pulse", "branch", "sample", "readout"} <= stages, stages
    for stage in stages:
        events = [p for p in seen if p.stage == stage]
        assert [p.done for p in events] == sorted(p.done for p in events), stage
        assert len({p.total for p in events}) == 1 and events[-1].done == events[-1].total, stage
        assert all(0 <= p.done <= p.total for p in events)
    elapsed = [p.elapsed_s for p in seen]
    assert elapsed == sorted(elapsed) and elapsed[-1] > 0.0
    seen_fn: list[Progress] = []
    same = run(BELL, serial.device, 200, table=serial.table, numerics=SERIAL, seed=1, progress=seen_fn.append)
    assert np.array_equal(same.bitstrings, result.bitstrings)
    assert [(p.stage, p.done, p.total) for p in seen_fn] == [(p.stage, p.done, p.total) for p in seen]


class Abort(RuntimeError):
    pass


def test_a_raising_callback_aborts_the_run_and_leaves_the_machine_usable(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine

    def boom(p: Progress) -> None:
        raise Abort(p.stage)

    with pytest.raises(Abort):
        m.run(ONE, 20, progress=boom)
    assert m.run(ONE, 20).shots == 20


def test_a_boundary_excess_warns_and_a_bounded_one_does_not() -> None:
    with pytest.warns(
        TruncationWarning,
        match=r"mode 3: boundary population 2\.000e-05 exceeds boundary_population_max = 1\.0e-06",
    ):
        warn_if_boundary_exceeds({2: 1e-7, 3: 2e-5}, 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        warn_if_boundary_exceeds({2: 1e-7, 3: 1e-6}, 1e-6)
    assert issubclass(TruncationWarning, UserWarning)


# ---- RunSpec: the record of a request ---------------------------------------------------------------------------------------


def test_spec_of_a_machine_carries_its_hash_and_policy(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    spec = m.spec(BELL, 200, seed=3, label="bell")
    assert spec == RunSpec.of(m, BELL, 200, seed=3, label="bell")
    assert spec.machine_hash == m.hash() and spec.level is m.level
    assert spec.physics == m.physics and spec.numerics == m.numerics and spec.readout == m.readout
    assert spec.shots == 200 and spec.seed == 3 and spec.circuit == BELL and not spec.keep_final_state
    with pytest.raises(ValueError, match="shots must be positive"):
        RunSpec(BELL, 0)


def test_spec_round_trips_through_json(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    numerics = dataclasses.replace(
        m.numerics,
        truncation={"branch_weight_min": 1e-3, "caps": {2: 12, 3: 14}, "enr_group": ((4, 5), 2)},
        integration={"atol": 1e-9, "integrators": ("vern9",), "store_marginals": True},
        parallel={"map": "serial", "samples": 3},
    )
    physics = trap.Physics(noise=False, internal_levels=3, entangler="zz", shot_period_s=2e-3)
    spec = RunSpec(
        BELL.measured(1),
        shots=7,
        seed=11,
        machine_hash=m.hash(),
        physics=physics,
        numerics=numerics,
        readout=trap.Readout(mode="full", povm_samples=99),
        level=trap.FidelityLevel.GATE_LOCAL,
        keep_final_state=True,
        label="round trip",
    )
    d = spec.to_dict()
    text = json.dumps(d)  # plain JSON: tuples became lists, integer keys strings
    assert d["schema_version"] == SPEC_SCHEMA_VERSION and d["qutip_trap_version"] == trap.__version__
    assert d["numerics"]["truncation"]["caps"] == {"2": 12, "3": 14}
    assert d["numerics"]["truncation"]["enr_group"] == [[4, 5], 2]
    assert d["level"] == "GATE_LOCAL" and d["circuit"]["measure"] == [1]
    back = RunSpec.from_dict(json.loads(text))
    assert back == spec, "exact: the option objects, the level and the circuit rebuilt as the same values"
    assert back.numerics.truncation.caps == {2: 12, 3: 14} and back.numerics.truncation.enr_group == (
        (4, 5),
        2,
    )
    assert back.numerics.integration.integrators == ("vern9",)
    with pytest.raises(ValueError, match="schema version"):
        RunSpec.from_dict({**d, "schema_version": 2})


def test_spec_refuses_by_name_what_a_record_cannot_carry(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    space = m.estimate(BELL).space
    with pytest.raises(ValueError, match="truncation.space"):
        dataclasses.replace(m.spec(BELL, 1), numerics=trap.Numerics(truncation={"space": space})).to_dict()
    with pytest.raises(ValueError, match="discriminator"):
        dataclasses.replace(
            m.spec(BELL, 1),
            readout=trap.Readout(discriminator=ThresholdDiscriminator(n_c=1.5, window_s=1e-5)),
        ).to_dict()


# ---- Job: the run in a worker process ---------------------------------------------------------------------------------------


def test_submit_result_equals_run_and_carries_its_record(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    job = m.submit(BELL, 200, seed=5, label="bell")
    assert isinstance(job, Job) and job.status() in ("running", "done") and job.spec.label == "bell"
    assert job.spec == m.spec(BELL, 200, seed=5, label="bell")
    direct = m.run(BELL, 200, seed=5)
    result = job.result(timeout_s=600.0)
    assert job.status() == "done" and repr(job).startswith("Job('done'")
    assert np.array_equal(result.bitstrings, direct.bitstrings), (
        "the same keyed streams, whichever process ran them"
    )
    assert result.counts == direct.counts and result.spam == direct.spam
    assert result.machine_hash == m.hash() == job.spec.machine_hash
    assert result.diagnostics.root_seed == 5 and result.diagnostics.level == direct.diagnostics.level
    record = job.record()  # the RunRecord travelled back on the Result
    assert record.schedule.pulses and record.compile.n_pulses == last_record(direct).compile.n_pulses
    progress = job.progress
    assert progress is not None and progress.stage == "readout" and progress.done == progress.total
    assert job.result() is result, "a finished job returns the same result again"


def test_cancel_stops_the_worker_within_one_pulse(machine) -> None:  # type: ignore[no-untyped-def]
    """Under a serial map the engine runs in-process and reports every pulse, so the cancel lands within one pulse."""
    _preset, m = machine
    job = submit(dataclasses.replace(m, numerics=SERIAL), BELL, 2000, seed=1)
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        p = job.progress
        if p is not None and p.stage == "pulse" and p.done >= 1:
            break
        if job.status() != "running":
            pytest.fail(f"the run ended before its first pulse report: {job.status()}")
        time.sleep(0.05)
    else:
        pytest.fail("no pulse progress arrived")
    first = job.progress
    assert first is not None
    job.cancel()
    assert job.cancel_requested and job.status() == "cancelled"
    t0 = time.monotonic()
    with pytest.raises(JobCancelled, match="cancelled"):
        job.result(timeout_s=300.0)
    stopped_after_s = time.monotonic() - t0
    last = job.progress
    assert last is not None and last.done <= first.done + 1, (first, last)
    assert not job._process.is_alive() and stopped_after_s < 120.0
    with pytest.raises(JobCancelled):
        job.result()


def test_a_job_started_twice_and_a_timeout_are_refused(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    job = m.submit(BELL, 200, seed=0)
    with pytest.raises(RuntimeError, match="already started"):
        job.start()
    with pytest.raises(TimeoutError):
        job.result(timeout_s=0.0)
    job.cancel(terminate_after_s=0.0)
    assert job.status() == "cancelled"
    with pytest.raises(JobCancelled):
        job.result(timeout_s=60.0)
