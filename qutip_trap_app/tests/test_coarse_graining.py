"""The coarse-graining identity: the register Level 1 shows after gate k, read from the recorded traces, equals the partial
trace of the joint state Level 3 re-simulation reaches at the gate's end, to 1e-12."""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap_app.record import BoundaryState, LiveRun, Record
from qutip_trap_app.resim import boundary_key, start_state
from qutip_trap_app.viewmodel.circuit import (
    bloch_vectors,
    pauli_expectations,
    populations,
    register_after,
    register_from_joint,
    timeline,
)

TOL = 1e-12


@pytest.fixture(scope="module")
def chained(bell: tuple[Record, LiveRun]) -> Record:
    record, live = bell
    for b in range(record.n_branches):
        record, _state = start_state(record, live, len(live.steps), 0, b)
    return record


def _joint_register(record: Record, step: int, branch: int) -> np.ndarray:
    boundary = record.cached(boundary_key(step, 0, branch), BoundaryState)
    assert boundary is not None and boundary.joint is not None
    rho = register_from_joint(boundary.joint, boundary.joint_dims, record.n_ions)
    assert np.max(np.abs(boundary.internal - rho)) < TOL, "the engine's own reduced state agrees"
    return rho


def test_level1_equals_the_partial_trace_of_level3_per_branch(chained: Record) -> None:
    n = chained.n_ions
    for gate in timeline(chained):
        for b in range(chained.n_branches):
            rho = _joint_register(chained, gate.step_index + 1, b)
            view = register_after(chained, gate.index, sample_index=0, branch=b)
            assert np.max(np.abs(view.rho - rho)) < TOL, (gate.gate_id, b)
            for k, v in populations(rho, n).items():
                assert abs(view.populations[k] - v) < TOL
            for k, v in pauli_expectations(rho, n).items():
                assert abs(view.pauli[k] - v) < TOL


def test_level1_weights_the_branches(chained: Record) -> None:
    total = sum(b.weight for b in chained.branches)
    for gate in timeline(chained):
        rho = sum(
            b.weight / total * _joint_register(chained, gate.step_index + 1, b.index)
            for b in chained.branches
        )
        assert np.max(np.abs(register_after(chained, gate.index).rho - rho)) < TOL


def test_the_bell_state_reads_off_the_record(chained: Record) -> None:
    gates = timeline(chained)
    after_ms = register_after(chained, next(g.index for g in gates if g.name == "ms"))
    for vec in bloch_vectors(after_ms.rho, 2).values():
        assert np.linalg.norm(vec) < 0.1, "each qubit alone is nearly maximally mixed after XX(pi/4)"
    assert float(after_ms.purity.value or 0.0) > 0.98
    final = register_after(chained, len(gates) - 1)
    assert abs(final.populations["00"] - 0.5) < 0.02 and abs(final.populations["11"] - 0.5) < 0.02
    fidelity = chained.results.register_fidelity
    assert fidelity is not None and float(final.fidelity.value or 0.0) > 0.99
    assert abs(float(final.fidelity.value or 0.0) - fidelity) < 1e-6, (
        "the view's ideal state (the gate targets with their frames) agrees with the core's"
    )
