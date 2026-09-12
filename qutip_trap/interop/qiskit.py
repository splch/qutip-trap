"""A Qiskit ``BackendV2`` over :func:`qutip_trap.run.job.run`: the gate-model SDKs' door into the simulator (PLAN.md
Section 1.4 names the importers and the ``run`` entry point; this is the same door with Qiskit on the other side).

    from qiskit import QuantumCircuit, transpile
    from qutip_trap.interop.qiskit import QutipTrapProvider

    backend = QutipTrapProvider().get_backend("yb171_chain", n_ions=2)
    bell = QuantumCircuit(2); bell.h(0); bell.cx(0, 1); bell.measure_all()
    job = backend.run(transpile(bell, backend), shots=2000)
    job.result().get_counts()          # {'00': ..., '11': ..., '01': ..., '10': ...}
    job.results[0].diagnostics.level   # the qutip-trap Result behind every experiment, with its diagnostics

The circuit crosses as OpenQASM 2 text (``qiskit.qasm2.dumps`` -> :func:`load_openqasm2`), so the ``Target`` advertises
exactly the gates the importer and the Section 7.7 compiler accept, and ``transpile(circuit, backend)`` rewrites anything
else into them. Bit order needs no conversion: qutip-trap's bitstrings and Qiskit's both put qubit 0 in the
least-significant (rightmost) position, provided qubit i is measured into clbit i, as ``measure_all`` does (the importer
flattens classical registers in declaration order). Run options beyond ``shots``, ``seed``, ``level`` and ``table`` go to
``run`` unchanged (``options=SolverOptions(...)``, ``readout="full"``, ``noise=False``, ...), Aer's convention for per-call
overrides. The simulation happens inside ``run`` (Section 3.4 is synchronous), so the job comes back finished.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
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
from qutip_trap.device.presets import DevicePreset, ca40_optical, yb171_chain
from qutip_trap.io.openqasm import load_openqasm2
from qutip_trap.run.job import run
from qutip_trap.run.results import Result

PRESETS: dict[str, Any] = {"yb171_chain": yb171_chain, "ca40_optical": ca40_optical}
"""The example devices of ``device/presets.py`` by name; ``QutipTrapProvider.get_backend(name, **knobs)`` builds one."""


def qutip_trap_target(n_qubits: int) -> Target:
    """The gates the OpenQASM 2 importer and the compiler accept (``STANDARD_GATES`` of ``control/compiler.py``), on every
    qubit and every ordered pair: the crystal is all-to-all and the compiler carries the Section 7.7 templates."""
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
        """The qutip-trap ``Result`` of every circuit, in order: probabilities, SPAM, photon records, diagnostics."""

    def submit(self) -> None:
        """Nothing to do: the simulation ran when the backend created the job."""

    def result(self) -> QiskitResult:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE


class QutipTrapBackend(BackendV2):
    """One example device (a ``DevicePreset``) as a Qiskit backend. ``table`` pins a calibration (else the closed-form
    surrogate, cached per device); ``run_kwargs`` are defaults for every ``run``; the drive maps come from the device's
    ``roles`` (0.2.0)."""

    def __init__(
        self, preset: DevicePreset, *, table: CalibrationTable | None = None, **run_kwargs: Any
    ) -> None:
        super().__init__(name=preset.name, description=preset.name, backend_version=__version__)
        self.preset = preset
        self.table = table
        self.run_kwargs: dict[str, Any] = dict(run_kwargs)
        self._target = qutip_trap_target(preset.n_ions)

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, seed=0, level="auto")

    def run(self, run_input: QuantumCircuit | Iterable[QuantumCircuit], **options: Any) -> QutipTrapJob:
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        kwargs = {**self.run_kwargs, **options}
        shots = int(kwargs.pop("shots", self.options.shots))
        seed = int(kwargs.pop("seed", self.options.seed))
        level = kwargs.pop("level", self.options.level)
        table = kwargs.pop("table", self.table)
        results = [
            run(
                load_openqasm2(dumps(c)),
                self.preset.device,
                shots,
                table=table,
                seed=seed,
                level=level,
                **kwargs,
            )
            for c in circuits
        ]
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
    """The example devices as backends. Qiskit 2 has no provider base class; this is the conventional pair of methods."""

    def backends(self, name: str | None = None) -> list[str]:
        return [n for n in PRESETS if name is None or n == name]

    def get_backend(self, name: str = "yb171_chain", **preset_kwargs: Any) -> QutipTrapBackend:
        """``get_backend("yb171_chain", n_ions=2)``: the keyword arguments are the preset's device knobs (``noise=``,
        ``hardware=``, ``omega_hz=``, ...). A backend around your own preset is ``QutipTrapBackend(preset)``."""
        if name not in PRESETS:
            raise KeyError(f"unknown backend {name!r}; known: {sorted(PRESETS)}")
        return QutipTrapBackend(PRESETS[name](**preset_kwargs))


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
