"""``progress`` on ``run`` and ``Machine.run`` (docs/api_implementation_plan.md 1.8) and the truncation warnings (1.9): the
callback sequence of a run is monotone per stage and ends at done == total, a callback that raises aborts the run and leaves
the machine usable, a clamped cap and a boundary excess warn with the mode and both numbers, and the Bell run is silent."""

from __future__ import annotations

import dataclasses
import re
import warnings

import numpy as np
import pytest

from qutip_trap._compat import QutipTrapWarning
from qutip_trap.api import Circuit, Operation, SolverOptions, run, yb171_chain
from qutip_trap.hilbert.truncation import TruncationWarning, warn_if_boundary_exceeds
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Parallel, Truncation
from qutip_trap.run.results import Progress
from qutip_trap.run.space import select_space

BELL = Circuit(2).h(0).cnot(0, 1)
ONE = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))


@pytest.fixture(scope="module")
def machine() -> Machine:
    pinned = (
        yb171_chain(2)
        .machine()
        .calibrated(pairs=[(0, 1)], detection_records=1000, detection_windows_s=WINDOWS)
    )
    # the serial map keeps every (sample, branch) engine run in-process, where the pulses can be reported
    return dataclasses.replace(
        pinned,
        numerics=Numerics(truncation=Truncation(branch_weight_min=1e-3), parallel=Parallel(map="serial")),
    )


def test_progress_is_a_frozen_record() -> None:
    p = Progress("pulse", 2, 6, 0.5)
    assert (p.stage, p.done, p.total, p.elapsed_s) == ("pulse", 2, 6, 0.5) and p.fraction == pytest.approx(
        1 / 3
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.done = 3  # type: ignore[misc]
    with pytest.raises(ValueError):
        Progress("pulse", 7, 6, 0.0)


def test_the_callback_sequence_is_monotone_per_stage_and_complete(machine: Machine) -> None:
    seen: list[Progress] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", QutipTrapWarning)  # 1.9: the Bell run is silent
        result = machine.run(BELL, 200, seed=1, progress=seen.append)
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
    # the same run through the function
    seen_fn: list[Progress] = []
    same = run(
        BELL,
        machine.device,
        200,
        table=machine.table,
        numerics=Numerics(truncation=Truncation(branch_weight_min=1e-3), parallel=Parallel(map="serial")),
        seed=1,
        progress=seen_fn.append,
    )
    assert np.array_equal(same.bitstrings, result.bitstrings)
    assert [(p.stage, p.done, p.total) for p in seen_fn] == [(p.stage, p.done, p.total) for p in seen]


class Abort(RuntimeError):
    pass


def test_a_raising_callback_aborts_the_run_and_leaves_the_machine_usable(machine: Machine) -> None:
    def boom(p: Progress) -> None:
        raise Abort(p.stage)

    with pytest.raises(Abort):
        machine.run(ONE, 20, progress=boom)
    assert machine.run(ONE, 20).shots == 20


def test_a_clamped_cap_warns_with_the_mode_and_both_numbers(machine: Machine) -> None:
    sched = machine.schedule(BELL)
    clamped = SolverOptions(mode_dimension_max=4, branch_weight_min=1e-3)
    with pytest.warns(TruncationWarning) as caught:
        selection = select_space(machine.device, sched, clamped, nbar={m: 0.01 for m in range(6)})
    assert all(t.d == 4 for t in selection.space.resolved)
    messages = [str(w.message) for w in caught if issubclass(w.category, TruncationWarning)]
    assert len(messages) == len(selection.resolved_modes), (
        messages
    )  # one warning per clamped mode, none twice
    assert any(
        re.search(r"mode 2: the cap rule asks for d = \d+ .* mode_dimension_max = 4 clamps it to d = 4", m)
        for m in messages
    ), messages
    with warnings.catch_warnings():
        warnings.simplefilter("error", QutipTrapWarning)
        select_space(
            machine.device, sched, SolverOptions(branch_weight_min=1e-3), nbar={m: 0.01 for m in range(6)}
        )


def test_a_boundary_excess_warns_and_a_bounded_one_does_not() -> None:
    with pytest.warns(
        TruncationWarning,
        match=r"mode 3: boundary population 2\.000e-05 exceeds boundary_population_max = 1\.0e-06",
    ):
        warn_if_boundary_exceeds({2: 1e-7, 3: 2e-5}, 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", QutipTrapWarning)
        warn_if_boundary_exceeds({2: 1e-7, 3: 1e-6}, 1e-6)
    assert issubclass(TruncationWarning, QutipTrapWarning) and issubclass(QutipTrapWarning, UserWarning)
