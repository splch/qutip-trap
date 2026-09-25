"""``RunSpec`` and ``Job``: the JSON-serialisable record of a run request, and ``Machine.run`` in a worker process.

``Machine.submit(circuit, shots, seed=)`` returns a ``Job``: the run in a spawned interpreter (so the run's own parallel maps
fork inside it as they do in-process), its ``Progress`` streamed back, cancellable, with the ``Result`` (and the
``RunRecord`` on it) handed back when done. The cancel is cooperative: the worker checks the flag at every progress report,
which the in-process engines make after every pulse, so a cancelled job stops within one pulse; a parallel map reports when
it returns (``Job.cancel(terminate_after_s=...)`` is the hard stop). A script that submits jobs runs under
``if __name__ == "__main__":`` like every user of ``multiprocessing``'s spawn start method; jobs still running when the
interpreter exits are terminated.
"""

from __future__ import annotations

import atexit
import dataclasses
import multiprocessing as mp
import queue
import time
import traceback
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.run.levels import FidelityLevel

if TYPE_CHECKING:
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.machine import Machine
    from qutip_trap.run.job import RunRecord
    from qutip_trap.run.results import Progress, Result

SPEC_SCHEMA_VERSION = 1
"""The ``schema_version`` ``RunSpec.to_dict`` writes."""

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
"""What ``Job.status()`` returns: ``queued`` before the worker started, ``running`` while it runs, then one of the three
terminal states (``done`` with a ``Result``, ``failed`` with the worker's traceback, ``cancelled`` by ``Job.cancel``)."""

TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})


class JobCancelled(RuntimeError):
    """``Job.result()`` on a cancelled job; raised inside the worker, at the next progress report, to stop the run."""


class JobError(RuntimeError):
    """``Job.result()`` on a job whose run failed; the message is the worker's traceback."""


# ---- the record ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """What ``Machine.submit`` runs, as a frozen record a JSON document can carry: the request (circuit, shots, seed,
    ``keep_final_state``), ``machine_hash`` (``Machine.hash()`` of the machine it was made for), the policy spelled out (the
    three option objects and the level) and ``label``, the caller's name for it."""

    circuit: Circuit
    shots: int
    seed: int = 0
    machine_hash: str = ""
    """``Machine.hash()`` of the machine the spec was made for; empty when the spec was written by hand."""
    physics: Physics = Physics()
    numerics: Numerics = Numerics()
    readout: Readout = Readout()
    level: FidelityLevel = FidelityLevel.AUTO
    keep_final_state: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if int(self.shots) <= 0:
            raise ValueError("shots must be positive")
        object.__setattr__(self, "shots", int(self.shots))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "level", FidelityLevel(self.level))

    @classmethod
    def of(
        cls,
        machine: Machine,
        circuit: Circuit,
        shots: int,
        *,
        seed: int = 0,
        keep_final_state: bool = False,
        label: str = "",
    ) -> RunSpec:
        """The spec of ``machine.run(circuit, shots, seed=seed, keep_final_state=keep_final_state)``."""
        return cls(
            circuit=circuit,
            shots=shots,
            seed=seed,
            machine_hash=machine.hash(),
            physics=machine.physics,
            numerics=machine.numerics,
            readout=machine.readout,
            level=machine.level,
            keep_final_state=keep_final_state,
            label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        """The spec as plain JSON-able values (schema version 1): tuples become lists and integer keys strings. A
        ``Physics.extra_channels`` with collapse operators, a ``Numerics.truncation.space`` or a ``Readout.discriminator``
        object is refused by name, because the record cannot carry it."""
        from qutip_trap import __version__

        return {
            "schema_version": SPEC_SCHEMA_VERSION,
            "qutip_trap_version": __version__,
            "machine_hash": str(self.machine_hash),
            "label": str(self.label),
            "circuit": _circuit_to_dict(self.circuit),
            "shots": int(self.shots),
            "seed": int(self.seed),
            "keep_final_state": bool(self.keep_final_state),
            "level": self.level.value,
            "physics": _physics_to_dict(self.physics),
            "numerics": _numerics_to_dict(self.numerics),
            "readout": _readout_to_dict(self.readout),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunSpec:
        """The inverse of :meth:`to_dict`; exact: ``RunSpec.from_dict(spec.to_dict()) == spec``."""
        if d["schema_version"] != SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"RunSpec.from_dict reads schema version {SPEC_SCHEMA_VERSION}, got {d['schema_version']!r}"
            )
        return cls(
            circuit=_circuit_from_dict(d["circuit"]),
            shots=int(d["shots"]),
            seed=int(d["seed"]),
            machine_hash=str(d["machine_hash"]),
            physics=_physics_from_dict(d["physics"]),
            numerics=_numerics_from_dict(d["numerics"]),
            readout=Readout(mode=d["readout"]["mode"], povm_samples=int(d["readout"]["povm_samples"])),
            level=FidelityLevel(d["level"]),
            keep_final_state=bool(d["keep_final_state"]),
            label=str(d["label"]),
        )


def _circuit_to_dict(circuit: Circuit) -> dict[str, Any]:
    return {
        "n_qubits": int(circuit.n_qubits),
        "ops": [
            {
                "name": str(op.name),
                "qubits": [int(q) for q in op.qubits],
                "params": [float(x) for x in op.params],
            }
            for op in circuit.ops
        ],
        "measure": [int(q) for q in circuit.measure],
        "registers": {str(k): [int(q) for q in v] for k, v in circuit.registers.items()},
    }


def _circuit_from_dict(d: Mapping[str, Any]) -> Circuit:
    return Circuit(
        int(d["n_qubits"]),
        tuple(
            Operation(
                str(op["name"]), tuple(int(q) for q in op["qubits"]), tuple(float(x) for x in op["params"])
            )
            for op in d["ops"]
        ),
        tuple(int(q) for q in d["measure"]),
        {str(k): tuple(int(q) for q in v) for k, v in dict(d["registers"]).items()},
    )


_SCALARS = (bool, int, float, str, type(None))


def _builder_to_dict(builder: BuilderOptions) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(builder):
        value = getattr(builder, f.name)
        if f.name == "curvature":
            out[f.name] = {
                str(int(ion)): {
                    "kappa_per_m2": float(spec.kappa_per_m2),
                    "axis": [float(x) for x in spec.axis],
                }
                for ion, spec in dict(value).items()
            }
        elif isinstance(value, _SCALARS):
            out[f.name] = value
        else:
            raise ValueError(
                f"RunSpec.to_dict: Physics.builder.{f.name} holds a {type(value).__name__}, which the record cannot carry"
            )
    return out


def _builder_from_dict(d: Mapping[str, Any]) -> BuilderOptions:
    from qutip_trap.dynamics.hamiltonian import BuilderOptions, CurvatureSpec

    fields = dict(d)
    fields["curvature"] = {
        int(ion): CurvatureSpec(
            float(spec["kappa_per_m2"]),
            (float(spec["axis"][0]), float(spec["axis"][1]), float(spec["axis"][2])),
        )
        for ion, spec in dict(fields["curvature"]).items()
    }
    return BuilderOptions(**fields)


def _physics_to_dict(physics: Physics) -> dict[str, Any]:
    if physics.extra_channels:
        raise ValueError(
            "RunSpec.to_dict: Physics.extra_channels holds collapse operators (Qobj), which the record cannot carry"
        )
    out = physics.asdict()
    out["extra_channels"] = []
    out["builder"] = None if physics.builder is None else _builder_to_dict(physics.builder)
    return out


def _physics_from_dict(d: Mapping[str, Any]) -> Physics:
    fields = dict(d)
    fields["extra_channels"] = ()
    fields["builder"] = None if fields["builder"] is None else _builder_from_dict(fields["builder"])
    return Physics.from_mapping(fields)


def _numerics_to_dict(numerics: Numerics) -> dict[str, Any]:
    tr = numerics.truncation
    if tr.space is not None:
        raise ValueError(
            "RunSpec.to_dict: Numerics.truncation.space is a declared HilbertSpace, which the record cannot carry; give "
            "caps or an enr_group and let the run select the space"
        )
    out: dict[str, Any] = {
        "integration": numerics.integration.asdict(),
        "truncation": tr.asdict(),
        "trajectories": numerics.trajectories.asdict(),
        "gate_local": numerics.gate_local.asdict(),
        "parallel": numerics.parallel.asdict(),
        "convergence_check": numerics.convergence_check,
    }
    out["integration"]["integrators"] = [str(x) for x in numerics.integration.integrators]
    out["truncation"]["caps"] = None if tr.caps is None else {str(int(m)): int(v) for m, v in tr.caps.items()}
    out["truncation"]["enr_group"] = (
        None if tr.enr_group is None else [[int(m) for m in tr.enr_group[0]], int(tr.enr_group[1])]
    )
    return out


def _numerics_from_dict(d: Mapping[str, Any]) -> Numerics:
    fields = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in dict(d).items()}
    integration = fields["integration"]
    integration["integrators"] = tuple(str(x) for x in integration["integrators"])
    truncation = fields["truncation"]
    caps = truncation["caps"]
    truncation["caps"] = None if caps is None else {int(m): int(v) for m, v in dict(caps).items()}
    enr = truncation["enr_group"]
    truncation["enr_group"] = None if enr is None else (tuple(int(m) for m in enr[0]), int(enr[1]))
    return Numerics.from_mapping(fields)


def _readout_to_dict(readout: Readout) -> dict[str, Any]:
    if readout.discriminator is not None:
        raise ValueError(
            "RunSpec.to_dict: Readout.discriminator is a discriminator object, which the record cannot carry; the table's "
            "threshold (discriminator=None) is the serialisable choice"
        )
    return {"mode": str(readout.mode), "discriminator": None, "povm_samples": int(readout.povm_samples)}


# ---- the handle ------------------------------------------------------------------------------------------------------------


def _worker_main(machine: Machine, spec: RunSpec, events: Any, cancel: Any) -> None:
    """The worker process: ``machine.run`` with a progress callback that streams every ``Progress`` to the parent and stops
    the run at the first report after a cancel; the ``Result`` (its ``RunRecord`` on it) goes back on the same queue."""

    def progress(p: Progress) -> None:
        events.put(("progress", p))
        if cancel.is_set():
            raise JobCancelled(f"cancelled at {p.stage} {p.done}/{p.total}")

    try:
        if cancel.is_set():
            events.put(("cancelled", None))
            return
        result = machine.run(
            spec.circuit,
            spec.shots,
            seed=spec.seed,
            keep_final_state=spec.keep_final_state,
            progress=progress,
        )
        events.put(("done", result))
    except JobCancelled:
        events.put(("cancelled", None))
    except BaseException:  # noqa: BLE001  (the worker reports every failure to the parent instead of dying silently)
        events.put(("error", traceback.format_exc()))


class Job:
    """A run in a worker process, the handle ``Machine.submit`` returns (a service object with state, not a record).

    ``status()`` is one of ``JobStatus``; ``progress`` the latest ``Progress`` the run reported; ``result(timeout_s=)``
    blocks for the ``Result`` (``JobCancelled`` after a cancel, ``JobError`` with the worker's traceback after a failure,
    ``TimeoutError`` past the timeout); ``record()`` the ``RunRecord`` behind it; ``cancel()`` stops the run at its next
    progress report, ``cancel(terminate_after_s=t)`` kills the worker if it has not stopped by then. ``spec`` is the
    ``RunSpec`` the job runs and ``machine`` the machine it runs on."""

    def __init__(self, machine: Machine, spec: RunSpec) -> None:
        self.machine = machine
        self.spec = spec
        self.submitted_at: str = datetime.now(UTC).isoformat(timespec="seconds")
        """When the job was created, ISO 8601 in UTC."""
        self._ctx = mp.get_context("spawn")
        self._events: Any = None
        self._cancel: Any = None
        self._process: Any = None
        self._progress: Progress | None = None
        self._status: JobStatus = "queued"
        self._result: Result | None = None
        self._error: str | None = None
        self._cancel_requested = False

    def __repr__(self) -> str:
        return f"Job({self.status()!r}, shots={self.spec.shots}, label={self.spec.label!r})"

    def start(self) -> Job:
        """Start the worker process (``submit`` does this; a job starts once)."""
        if self._process is not None:
            raise RuntimeError("the job was already started")
        self._events = self._ctx.Queue()
        self._cancel = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self.machine, self.spec, self._events, self._cancel),
            daemon=False,  # the run's own parallel maps fork inside the worker (Section 11.3 item 9)
            name="qutip-trap-job",
        )
        self._process.start()
        self._status = "running"
        _LIVE.add(self)
        return self

    def status(self) -> JobStatus:
        """The job's state now (drains the worker's events first)."""
        self._drain()
        if self._status in TERMINAL_STATES:
            return self._status
        return "cancelled" if self._cancel_requested else self._status

    @property
    def progress(self) -> Progress | None:
        """The latest ``Progress`` the run reported; None before the first."""
        self._drain()
        return self._progress

    @property
    def cancel_requested(self) -> bool:
        """Whether ``cancel`` was called (the worker acknowledges at its next progress report)."""
        return self._cancel_requested

    def result(self, timeout_s: float | None = None) -> Result:
        """Block until the run is done and return its ``Result``, the one ``machine.run`` returns at the same seed."""
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while self._status not in TERMINAL_STATES:
            self._drain(timeout_s=0.2)
            if deadline is not None and self._status not in TERMINAL_STATES and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"the job has not finished after {timeout_s:g} s (status {self.status()!r})"
                )
        if self._status == "done":
            assert self._result is not None
            return self._result
        if self._status == "cancelled":
            p = self._progress
            where = f" at {p.stage} {p.done}/{p.total}" if p is not None else ""
            raise JobCancelled(f"the job was cancelled{where}")
        raise JobError(self._error or "the worker failed without a traceback")

    def record(self) -> RunRecord:
        """The ``RunRecord`` behind the result, waiting for the run."""
        from qutip_trap.run.job import last_record

        return last_record(self.result())

    def cancel(self, *, terminate_after_s: float | None = None) -> None:
        """Ask the worker to stop at its next progress report. With ``terminate_after_s`` the worker is killed if it is still
        alive that many seconds later (0 kills it at once); a killed worker reports nothing, so the job is marked cancelled
        here. A job that finished before the flag was seen keeps its result."""
        self._cancel_requested = True
        if self._cancel is not None:
            self._cancel.set()
        if terminate_after_s is not None and self._process is not None:
            self._process.join(timeout=max(0.0, float(terminate_after_s)))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
                # a message that raced the kill still counts; a silent death after OUR kill is the cancel, not a failure
                self._drain(check_exit=False)
                if self._status not in TERMINAL_STATES:
                    self._status = "cancelled"
                    self._finish()

    def _drain(self, timeout_s: float = 0.0, *, check_exit: bool = True) -> None:
        """Read the worker's events: every progress report and the one terminal message (a first blocking wait of
        ``timeout_s`` when asked); a worker that died without a word marks the job failed, unless ``check_exit`` is off
        (the caller killed it and knows why)."""
        if self._events is None or self._status in TERMINAL_STATES:
            return
        block = timeout_s > 0.0
        while True:
            try:
                kind, payload = self._events.get(timeout=timeout_s) if block else self._events.get_nowait()
            except queue.Empty:
                break
            block = False
            if kind == "progress":
                self._progress = payload
            elif kind == "done":
                self._result = payload
                self._status = "done"
                self._finish()
                return
            elif kind == "cancelled":
                self._status = "cancelled"
                self._finish()
                return
            elif kind == "error":
                self._error = str(payload)
                self._status = "failed"
                self._finish()
                return
        if check_exit and self._process is not None and not self._process.is_alive():
            # the worker exited: its queue feeder flushed before the exit, so one more (short) blocking read sees a
            # terminal message that raced the exit check; nothing there means a crash (a signal, a hard kill)
            try:
                kind, payload = self._events.get(timeout=1.0)
            except queue.Empty:
                self._error = (
                    f"the worker process exited with code {self._process.exitcode} before reporting a result"
                )
                self._status = "failed"
                self._finish()
                return
            self._events.put((kind, payload))
            self._drain()

    def _finish(self) -> None:
        if self._process is not None:
            self._process.join(timeout=5.0)
        _LIVE.discard(self)


_LIVE: weakref.WeakSet[Job] = weakref.WeakSet()
"""The jobs whose worker may still be running; terminated at interpreter exit so that a forgotten job never holds it."""


def _terminate_live_jobs() -> None:
    for job in list(_LIVE):
        job.cancel(terminate_after_s=0.0)


atexit.register(_terminate_live_jobs)


def submit(
    machine: Machine,
    circuit: Circuit,
    shots: int,
    *,
    seed: int = 0,
    keep_final_state: bool = False,
    label: str = "",
) -> Job:
    """``Machine.submit``: the ``RunSpec`` of the call and a started ``Job`` running it in a worker process."""
    spec = RunSpec.of(machine, circuit, shots, seed=seed, keep_final_state=keep_final_state, label=label)
    return Job(machine, spec).start()
