"""Helpers shared by the tests (importable because the app's pytest config puts ``tests`` on the path)."""

from __future__ import annotations

from dataclasses import dataclass

from qutip_trap_app import core
from qutip_trap_app.record import (
    HamiltonianRecord,
    Job,
    LiveRun,
    ProcessMatrixRecord,
    Recheck,
    Record,
    ZoomTrace,
)
from qutip_trap_app.viewmodel.editor import BELL_QASM, parse_circuit
from qutip_trap_app.views.state import FAST

BELL = parse_circuit(BELL_QASM)
SHOTS = 200
SEED = 7


def job(
    circuit: core.Circuit = BELL,
    *,
    numerics: core.Numerics = FAST,
    shots: int = SHOTS,
    **device_kwargs: float,
) -> Job:
    """The job the app submits for ``circuit``, at the tests' seed."""
    spec = core.RunSpec(circuit, shots, seed=SEED, numerics=numerics, keep_final_state=True)
    return Job(spec, max(2, circuit.n_qubits), dict(device_kwargs))


def truncation(**fields: object) -> core.Numerics:
    """The app's numerics with other truncation settings."""
    return core.Numerics(truncation=core.Truncation(**{"branch_weight_min": 1e-3, **fields}))


def ms_step(record: Record) -> int:
    return next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))


@dataclass(frozen=True)
class BellResimulated:
    """The Bell run with its entangling step re-simulated, its Hamiltonian listed, its process matrix taken and both
    re-checks run, each cached on ``record``."""

    record: Record
    live: LiveRun
    step: int
    zoom: ZoomTrace
    hamiltonian: HamiltonianRecord
    process: ProcessMatrixRecord
    tolerance: Recheck
    truncation: Recheck
