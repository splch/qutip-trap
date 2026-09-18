"""``Machine`` (docs/api_implementation_plan.md 1.3): the executor's run equals ``run`` field for field at the same seed, its
schedule equals the scheduler's on the same table, ``calibrated`` pins a table for the device, a forced level shows in the
diagnostics, ``estimate`` matches the run that follows, and ``hash`` moves when and only when the device, the table or the
policy moves."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap._compat import QutipTrapDeprecationWarning
from qutip_trap.api import (
    Circuit,
    FidelityLevel,
    Operation,
    SolverOptions,
    compile_to_native,
    run,
    schedule,
    yb171_chain,
)
from qutip_trap.device.model import BeamRoles
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.machine import Estimate, Machine
from qutip_trap.options import Numerics, Physics, Readout, Truncation

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
ONE = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = Numerics(truncation=Truncation(branch_weight_min=1e-3))


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
    # the 0.1.0 call shape is rewritten onto the same machine, with the warning naming the keyword and its new home (2.1)
    with pytest.warns(
        QutipTrapDeprecationWarning, match=r"'options' argument of qutip_trap.run.job.run .* v0.5"
    ):
        legacy = run(
            BELL, preset.device, 400, table=m.table, options=SolverOptions(branch_weight_min=1e-3), seed=0
        )
    assert np.array_equal(legacy.bitstrings, reference.bitstrings) and legacy.counts == reference.counts
    assert legacy.machine_hash == reference.machine_hash == bell.machine_hash
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
        assert getattr(legacy.diagnostics, f.name) == getattr(reference.diagnostics, f.name), f.name


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


def test_mappings_are_accepted_and_the_engine_is_configured_from_the_machine(machine) -> None:  # type: ignore[no-untyped-def]
    preset, m = machine
    loose = Machine(
        preset.device,
        physics={"noise": False},
        numerics={"convergence_check": True},
        readout={"mode": "full"},
    )  # type: ignore[arg-type]
    assert (
        loose.physics == Physics(noise=False)
        and loose.numerics.convergence_check
        and loose.readout.mode == "full"
    )
    engine = m.engine
    assert isinstance(engine, JointExactEngine) and engine.table is m.table and engine.device_channels
    assert dataclasses.replace(m, physics=Physics(noise=False)).engine.device_channels is False


def test_the_later_phases_name_themselves(machine) -> None:  # type: ignore[no-untyped-def]
    _preset, m = machine
    # 3.1 (0.4.0): submit is implemented, a Job in a worker process whose spec records the call (tests/test_job.py runs it)
    job = m.submit(BELL, 10)
    try:
        assert job.status() in ("running", "done") and job.spec.shots == 10
        assert job.spec.machine_hash == m.hash() and m.spec(BELL, 10) == job.spec
    finally:
        job.cancel(terminate_after_s=0.0)
    assert job.status() == "cancelled"
    # 2.5: specs is implemented, on the device's derived quantities plus the machine's roles, table and level
    text = m.specs()
    assert text.startswith(f"device {m.device.hash()[:12]}") and "rabi_hz[(0, 2)] =" in text
    assert "gate drives = {0: GateDrive(" in text and "table = pinned (" in text and "level = auto" in text
