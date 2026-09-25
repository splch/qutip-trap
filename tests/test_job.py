"""``RunSpec`` and ``Job`` (docs/api_implementation_plan.md 3.1; 0.4.0): the spec round-trips through its JSON form, ``Machine.submit().result()`` equals ``Machine.run()`` shot for shot, the record behind it is
reachable both ways, and a cancel stops the worker within one pulse."""

from __future__ import annotations

import dataclasses
import json
import time

import numpy as np
import pytest

import qutip_trap as trap
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.machine import Machine
from qutip_trap.run.job import last_record
from qutip_trap.run.spec import SPEC_SCHEMA_VERSION, Job, JobCancelled, RunSpec, submit
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
FAST = trap.Numerics(truncation={"branch_weight_min": 1e-3})


@pytest.fixture(scope="module")
def machine() -> Machine:
    """The two-ion fixture as a machine with its surrogate table pinned, so the worker process does not recalibrate."""
    fx = circuit_fixture(2)
    return Machine(fx.device, numerics=FAST).calibrated(pairs=[(0, 1)], detection_records=500)


# ---- the record ---------------------------------------------------------------------------------------------------------


def test_spec_of_a_machine_carries_its_hash_and_policy(machine: Machine) -> None:
    spec = machine.spec(BELL, 200, seed=3, label="bell")
    assert spec == RunSpec.of(machine, BELL, 200, seed=3, label="bell")
    assert spec.machine_hash == machine.hash() and spec.level is machine.level
    assert (
        spec.physics == machine.physics
        and spec.numerics == machine.numerics
        and spec.readout == machine.readout
    )
    assert spec.shots == 200 and spec.seed == 3 and spec.circuit == BELL and not spec.keep_final_state
    with pytest.raises(ValueError, match="shots must be positive"):
        RunSpec(BELL, 0)


def test_spec_round_trips_through_json(machine: Machine) -> None:
    numerics = dataclasses.replace(
        machine.numerics,
        truncation={"branch_weight_min": 1e-3, "caps": {2: 12, 3: 14}, "enr_group": ((4, 5), 2)},
        integration={"atol": 1e-9, "integrators": ("vern9",), "store_marginals": True},
        parallel={"map": "serial", "samples": 3},
    )
    physics = trap.Physics(noise=False, internal_levels=3, entangler="zz", shot_period_s=2e-3)
    spec = RunSpec(
        BELL.measured(1),
        shots=7,
        seed=11,
        machine_hash=machine.hash(),
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


def test_spec_refuses_by_name_what_a_record_cannot_carry(machine: Machine) -> None:
    from qutip_trap.readout.discriminate import ThresholdDiscriminator

    space = machine.estimate(BELL).space
    with pytest.raises(ValueError, match="truncation.space"):
        dataclasses.replace(
            machine.spec(BELL, 1), numerics=trap.Numerics(truncation={"space": space})
        ).to_dict()
    with pytest.raises(ValueError, match="discriminator"):
        dataclasses.replace(
            machine.spec(BELL, 1),
            readout=trap.Readout(discriminator=ThresholdDiscriminator(n_c=1.5, window_s=1e-5)),
        ).to_dict()


# ---- the handle ---------------------------------------------------------------------------------------------------------


def test_submit_result_equals_run_and_leaves_its_record_reachable(machine: Machine) -> None:
    job = machine.submit(BELL, 200, seed=5, label="bell")
    assert isinstance(job, Job) and job.status() in ("running", "done") and job.spec.label == "bell"
    direct = machine.run(BELL, 200, seed=5)
    result = job.result(timeout_s=600.0)
    assert job.status() == "done" and repr(job).startswith("Job('done'")
    assert np.array_equal(result.bitstrings, direct.bitstrings), (
        "the same keyed streams, whichever process ran them"
    )
    assert result.counts == direct.counts and result.spam == direct.spam
    assert result.machine_hash == machine.hash() == job.spec.machine_hash
    assert result.diagnostics.root_seed == 5 and result.diagnostics.level == direct.diagnostics.level
    record = job.record()
    assert last_record(result) is record, "RunRecord and last_record are the old names of Job.record()"
    assert record.schedule.pulses and record.compile.n_pulses == last_record(direct).compile.n_pulses
    progress = job.progress
    assert progress is not None and progress.stage == "readout" and progress.done == progress.total
    assert job.result() is result, "a finished job returns the same result again"


def test_cancel_stops_the_worker_within_one_pulse(machine: Machine) -> None:
    """Under a serial map the engine runs in-process and reports every pulse, so the cancel lands within one pulse; under
    the parallel map the branches report when the map returns (the docstring of ``run.spec`` says so)."""
    serial = dataclasses.replace(
        machine, numerics=dataclasses.replace(machine.numerics, parallel={"map": "serial"})
    )
    job = submit(serial, BELL, 2000, seed=1)
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


def test_a_job_started_twice_and_a_timeout_are_refused(machine: Machine) -> None:
    job = machine.submit(BELL, 200, seed=0)
    with pytest.raises(RuntimeError, match="already started"):
        job.start()
    with pytest.raises(TimeoutError):
        job.result(timeout_s=0.0)
    job.cancel(terminate_after_s=0.0)
    assert job.status() == "cancelled"
    with pytest.raises(JobCancelled):
        job.result(timeout_s=60.0)
