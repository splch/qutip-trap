"""Populations and Pauli expectations shown at Level 1 after gate k, computed
from the recorded Level 3 joint state, equal the reduced-density-matrix values to 1e-12. Also the record's own ladder."""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.resim import boundary_states
from qutip_trap_app.viewmodel.circuit import (
    bloch_vectors,
    pauli_expectations,
    phase_register,
    populations,
    register_after,
    timeline,
)

TOL = 1e-12


def register_from_joint(joint: np.ndarray, joint_dims: tuple[int, ...], n_ions: int) -> np.ndarray:
    """The oracle: the reduced register density matrix (register order) of a joint ket or density matrix over ion and mode factors."""
    dims = list(joint_dims)
    d_int = int(np.prod(dims[:n_ions]))
    d_mot = int(np.prod(dims[n_ions:])) if len(dims) > n_ions else 1
    arr = np.asarray(joint, dtype=complex)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        psi = arr.reshape(d_int, d_mot)
        return np.asarray(psi @ psi.conj().T)
    rho = arr.reshape(d_int, d_mot, d_int, d_mot)
    return np.asarray(np.einsum("iaja->ij", rho))


def test_record_holds_the_ladder(bell: tuple[Record, LiveRun]) -> None:
    """Section 14.3: job -> compiled circuit -> schedule -> traces -> readout -> results, all present and consistent."""
    record, live = bell
    assert record.job.circuit.n_qubits == 2 and record.results.bitstrings.shape == (200, 2)
    assert record.compiled.native.ops and all(
        op.name in ("gpi", "gpi2", "ms", "zz") for op in record.compiled.native.ops
    )
    assert record.schedule.pulses and record.schedule.steps and record.schedule.targets
    assert {s.kind for s in record.schedule.steps} == {"gate", "idle"}
    assert record.diagnostics.level == "JOINT_EXACT" and record.space.dimension == live.space.dimension
    assert len(record.traces) == record.n_samples * record.n_branches
    assert record.readout.levels.shape == (200, 2)
    assert set(record.results.target_probabilities) == {"00", "11"}
    assert record.results.probabilities["00"] + record.results.probabilities["11"] > 0.98
    assert record.results.register_fidelity is not None and record.results.register_fidelity > 0.99
    assert record.device_card.n_ions == 2 and record.device_hash == record.job.device.hash
    assert record.table.device_hash == record.device_hash and record.table.entries


@pytest.fixture(scope="module")
def chained(bell: tuple[Record, LiveRun]) -> tuple[Record, LiveRun]:
    record, live = bell
    for b in range(record.n_branches):
        record = boundary_states(record, live, 0, b)
    return record, live


def test_level1_equals_partial_trace_of_level3_per_branch(chained: tuple[Record, LiveRun]) -> None:
    record, _ = chained
    n = record.n_qubits
    for gate in timeline(record):
        end_step = gate.step_index + 1
        for b in range(record.n_branches):
            boundary = record.boundary(end_step, 0, b)
            assert boundary is not None and boundary.joint is not None
            rho_joint = register_from_joint(boundary.joint, boundary.joint_dims, n)
            view = register_after(record, gate.index, sample_index=0, branch=b)
            assert np.max(np.abs(view.rho - rho_joint)) < TOL, (gate.gate_id, b)
            for k, v in populations(rho_joint, n).items():
                assert abs(view.populations[k].value - v) < TOL
            for k, v in pauli_expectations(rho_joint, n).items():
                assert abs(view.pauli[k].value - v) < TOL
            assert np.max(np.abs(boundary.internal - rho_joint)) < TOL, (
                "the engine's own reduced state agrees too"
            )


def test_level1_weighted_over_branches(chained: tuple[Record, LiveRun]) -> None:
    record, _ = chained
    n = record.n_qubits
    weights = {b.index: b.weight for b in record.branches}
    total = sum(weights.values())
    for gate in timeline(record):
        rho = np.zeros((2**n, 2**n), dtype=complex)
        for b in range(record.n_branches):
            boundary = record.boundary(gate.step_index + 1, 0, b)
            assert boundary is not None and boundary.joint is not None
            rho += weights[b] / total * register_from_joint(boundary.joint, boundary.joint_dims, n)
        view = register_after(record, gate.index)
        assert np.max(np.abs(view.rho - rho)) < TOL


def test_bell_physics_reads_correctly_from_the_record(chained: tuple[Record, LiveRun]) -> None:
    """The Bloch arrows vanish after the entangling gate and the final register is the Bell state."""
    record, _ = chained
    gates = timeline(record)
    ms = next(g for g in gates if g.name.value == "ms")
    after_ms = register_after(record, ms.index)
    for vec in bloch_vectors(after_ms.rho, 2).values():
        assert np.linalg.norm(vec) < 0.1, (
            "reduced single-qubit states are near maximally mixed after XX(pi/4)"
        )
    assert after_ms.purity.value is not None and float(after_ms.purity.value) > 0.98
    final = register_after(record, len(gates) - 1)
    assert final.fidelity.value is not None and float(final.fidelity.value) > 0.99
    assert abs(final.populations["00"].value - 0.5) < 0.02 and abs(final.populations["11"].value - 0.5) < 0.02
    assert record.results.register_fidelity is not None
    assert abs(float(final.fidelity.value) - record.results.register_fidelity) < 1e-6, (
        "the view's target state (the gate targets with their frames) agrees with the core's ideal_register_state"
    )
    assert 1.0 - record.results.register_fidelity < record.diagnostics.intrinsic_budget.total, (
        "inside the closed-form budget (Section 9.6)"
    )
    frame = phase_register(record)
    assert set(frame.final_frame) == {0, 1}
