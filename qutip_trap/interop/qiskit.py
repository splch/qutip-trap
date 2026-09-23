"""A Qiskit ``BackendV2`` over a :class:`qutip_trap.machine.Machine`, and a provider of backends on the presets
(``QutipTrapProvider().get_backend("yb171_chain", n_ions=2)``). Circuits cross as OpenQASM 2, so the ``Target`` advertises
the gates the importer accepts. Bitstrings need no conversion: both put qubit 0 in the least-significant (rightmost)
position, provided qubit i is measured into clbit i. Per-call options override the machine's; jobs come back finished."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from qiskit.circuit import Measure, Parameter, QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate,
    CXGate,
    CZGate,
    HGate,
    IGate,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXGate,
    TdgGate,
    TGate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.qasm2 import dumps
from qiskit.result import Result as QiskitResult
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

from qutip_trap import __version__
from qutip_trap.control.table import CalibrationTable
from qutip_trap.device.presets import ca40_optical, yb171_chain
from qutip_trap.io.openqasm import load_openqasm2
from qutip_trap.machine import Machine
from qutip_trap.run.results import Result

PRESETS: dict[str, Any] = {"yb171_chain": yb171_chain, "ca40_optical": ca40_optical}
"""The example devices of ``device/presets.py`` by name; ``QutipTrapProvider.get_backend(name, **knobs)`` builds one."""


def qutip_trap_target(n_qubits: int) -> Target:
    """The gates the OpenQASM 2 importer and the compiler accept, on every qubit and every ordered pair (all-to-all)."""
    theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")
    one = {(q,): None for q in range(n_qubits)}
    two = {(a, b): None for a in range(n_qubits) for b in range(n_qubits) if a != b}
    target = Target(num_qubits=n_qubits, description="qutip-trap")
    single = (IGate(), XGate(), YGate(), ZGate(), HGate(), SGate(), SdgGate(), TGate(), TdgGate(), SXGate())
    for gate in (*single, RXGate(theta), RYGate(theta), RZGate(theta), U3Gate(theta, phi, lam), Measure()):
        target.add_instruction(gate, one)
    for gate in (CXGate(), CZGate(), SwapGate(), CPhaseGate(theta), RXXGate(theta), RZZGate(theta)):
        target.add_instruction(gate, two)
    return target


class QutipTrapJob(JobV1):
    """A finished job: ``run`` simulated every circuit before the job was created."""

    def __init__(
        self, backend: QutipTrapBackend, job_id: str, result: QiskitResult, results: Sequence[Result]
    ) -> None:
        super().__init__(backend, job_id)
        self._result = result
        self.results: tuple[Result, ...] = tuple(results)
        """The qutip-trap ``Result`` of every circuit, in order."""

    def submit(self) -> None:
        """Nothing to do: the simulation ran when the backend created the job."""

    def result(self) -> QiskitResult:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE


class QutipTrapBackend(BackendV2):
    """A ``Machine`` as a Qiskit backend. ``table`` pins a calibration on the machine and ``run_options`` are defaults for
    every ``run``: ``shots``, ``seed``, and any of the machine's ``table``, ``level``, ``physics``, ``numerics`` and
    ``readout``."""

    def __init__(
        self, machine: Machine, *, table: CalibrationTable | None = None, **run_options: Any
    ) -> None:
        if table is not None:
            machine = replace(machine, table=table)
        label = machine.name or machine.device.hash()[:12]
        super().__init__(name=label, description=label, backend_version=__version__)
        self.machine: Machine = machine
        self.run_options: dict[str, Any] = dict(run_options)
        self._target = qutip_trap_target(machine.device.crystal.n_ions)

    @property
    def table(self) -> CalibrationTable | None:
        return self.machine.table

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, seed=0)

    def run(self, run_input: QuantumCircuit | Iterable[QuantumCircuit], **options: Any) -> QutipTrapJob:
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        kwargs = {**self.run_options, **options}
        shots = int(kwargs.pop("shots", self.options.shots))
        seed = int(kwargs.pop("seed", self.options.seed))
        changes = {
            k: v
            for k in ("table", "level", "physics", "numerics", "readout")
            if (v := kwargs.pop(k, None)) is not None
        }
        if kwargs:
            raise TypeError(f"QutipTrapBackend.run got unexpected options {sorted(kwargs)}")
        machine = replace(self.machine, **changes)
        results = [machine.run(load_openqasm2(dumps(c)), shots, seed=seed) for c in circuits]
        job_id = str(uuid.uuid4())
        experiments = [_experiment(c, r) for c, r in zip(circuits, results)]
        result = QiskitResult(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            success=True,
            results=experiments,
        )
        return QutipTrapJob(self, job_id, result, results)


class QutipTrapProvider:
    """The example devices as backends (Qiskit 2 has no provider base class)."""

    def backends(self, name: str | None = None) -> list[str]:
        return [n for n in PRESETS if name is None or n == name]

    def get_backend(self, name: str = "yb171_chain", **preset_kwargs: Any) -> QutipTrapBackend:
        """A backend on a preset: ``get_backend("yb171_chain", n_ions=2)`` builds ``presets.yb171_chain(n_ions=2)``, the
        keywords being the preset's device knobs."""
        if name not in PRESETS:
            raise KeyError(f"unknown backend {name!r}; known: {sorted(PRESETS)}")
        return QutipTrapBackend(PRESETS[name](**preset_kwargs).machine())


def _experiment(circuit: QuantumCircuit, res: Result) -> ExperimentResult:
    """One Qiskit experiment from one qutip-trap ``Result``: hex keys in the same bit order, a header that pads them."""
    counts = {hex(int(key, 2)): n for key, n in res.counts.items()}
    header = {
        "name": circuit.name,
        "memory_slots": res.n_qubits,
        "creg_sizes": [["c", res.n_qubits]],
        "n_qubits": circuit.num_qubits,
        "metadata": circuit.metadata,
    }
    return ExperimentResult(
        shots=res.shots,
        success=True,
        data=ExperimentResultData(counts=counts),
        header=header,
        seed=res.diagnostics.root_seed,
    )


__all__ = ["PRESETS", "QutipTrapBackend", "QutipTrapJob", "QutipTrapProvider", "qutip_trap_target"]
