"""The GATE_LOCAL per-step register (docs/api_implementation_plan.md 3.2; 0.4.0): every step of the first sample's walk
reports the register after it and the channels it applied; composing the channels from one step's register gives the
next step's, and the last equals the run's recombined register."""

from __future__ import annotations

import numpy as np
import pytest

import qutip_trap as trap
from qutip_trap.api import Circuit, Operation, SolverOptions, last_record
from qutip_trap.dynamics.tomography import apply_kraus_dm, kraus_operators
from qutip_trap.machine import Machine
from qutip_trap.run.gate_local import REGISTER_STORE_DIM_MAX, AppliedChannel
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))


@pytest.fixture(scope="module")
def gate_local_run() -> trap.Result:
    """The Bell job routed to GATE_LOCAL by a joint-dimension guard far below its dimension (Section 11.5)."""
    fx = circuit_fixture(2)
    machine = Machine(
        fx.device,
        numerics=trap.Numerics.from_solver_options(
            SolverOptions(branch_weight_min=1e-3, joint_dimension_max=8)
        ),
    ).calibrated(pairs=[(0, 1)], detection_records=500)
    return machine.run(BELL, 100, seed=0)


def test_every_step_reports_its_register_and_channels(gate_local_run: trap.Result) -> None:
    rec = last_record(gate_local_run)
    assert gate_local_run.diagnostics.level == "GATE_LOCAL" and rec.gate_local is not None
    steps = rec.gate_local.steps
    assert steps and 4 <= REGISTER_STORE_DIM_MAX
    for st in steps:
        assert st.register_after is not None and st.register_after.shape == (4, 4)
        assert abs(np.trace(st.register_after) - 1.0) < 1e-9
        assert np.allclose(st.register_after, st.register_after.conj().T, atol=1e-12)
        assert st.channels and all(isinstance(c, AppliedChannel) for c in st.channels)
        if st.kind == "gate":
            assert len(st.channels) == 1 and st.channels[0].ions == st.ions
            assert st.summary is not None and np.allclose(st.channels[0].choi, st.summary.choi, atol=1e-12)
        else:
            assert [c.ions for c in st.channels] == [(0,), (1,)]
            assert all(c.choi.shape == (4, 4) for c in st.channels)


def test_the_channels_compose_the_register_step_by_step(gate_local_run: trap.Result) -> None:
    rec = last_record(gate_local_run)
    assert rec.gate_local is not None
    steps = rec.gate_local.steps
    dims = list(steps[0].space_dims) if steps[0].kind == "idle" else [2, 2]
    for before, after in zip(steps, steps[1:]):
        rho = np.asarray(before.register_after, dtype=complex)
        for ch in after.channels:
            rho = apply_kraus_dm(rho, kraus_operators(ch.choi), dims, list(ch.ions))
        assert after.register_after is not None
        assert np.max(np.abs(rho - after.register_after)) < 1e-10, after.gate_id
    assert rec.register_state is not None
    assert np.max(np.abs(steps[-1].register_after - np.asarray(rec.register_state.full()))) < 1e-10, (
        "the last step's register is the run's recombined one"
    )
