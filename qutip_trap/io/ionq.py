"""IonQ circuit JSON importer and exporter (PLAN.md Sections 7.6, 8.6, 9.13).

Encoding (verified against the IonQ OpenAPI v0.4 specification and the qiskit-ionq, ionq-core-python and
PennyLane-IonQ clients, Section 7.6): the native gate names are ``gpi``, ``gpi2``, ``ms``, ``zz`` and ``nop``;
``gpi``/``gpi2`` take a scalar ``target`` and a scalar ``phase`` in TURNS; ``ms`` takes ``targets`` (two),
``phases`` (two, turns) and ``angle`` (turns, default 0.25, maximally entangling); ``zz`` takes ``targets`` and
``angle`` and no phases. The job body is ``{"input": {"gateset": "native" | "qis", "qubits": N, "circuit":
[...]}}``; the qis gateset's rotations (``rotation``) are radians. Internally every parameter is radians
(Section 13, "Gate parameters"); exported turns are rounded to 12 significant digits so that a value such as
0.75 round-trips exactly (the 9.13 test).

Result formats (Section 8.6; the exporters live on ``Result``). v1 (``ionq.result.probabilities.json.v1`` and its histogram
and shots forms): decimal-integer keys, qubit 0 the least-significant bit. v2 (``ionq.result.probabilities.json.v2``): the
envelope ``{"probabilities": {"registers": {<register>: {<bitstring>: p}}}}`` with bitstrings per named classical register
and ``output_all`` the whole-circuit register, in WIRE order, q[0] the leftmost character, the reverse of this package's
``bitstring_key``: ``x q[0]`` on three qubits is ``"1"`` in v1 and ``"100"`` in v2. Sources: the v0.4 API reference's
OpenQASM 3 page under "Reading results" (https://docs.ionq.com/api-reference/v0.4/openqasm3: "q[0] is the leftmost character
of the bitstring"), and qiskit-ionq 1.1.1's ``_decode_distribution_artifact``, which reverses every ``output_all`` key.

Job bodies (the v0.4 ``CircuitJobCreationPayload``): ``dump_job`` writes ``{"type": "ionq.circuit.v1", "backend": ...,
"shots": ..., "input": {...}}`` plus the optional ``name``, ``metadata``, ``noise`` (``{"model": ..., "seed": ...}``),
``settings`` (``compilation``, ``error_mitigation``) and ``dry_run``; the spec sets ``additionalProperties: false`` on the
body, the input and the settings, so every key is checked here. ``load_job`` reads a v0.4 body or a v0.3 one (``target``
for ``backend``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from qutip_trap.control.compiler import NATIVE_GATES, STANDARD_GATES, Circuit, Operation
from qutip_trap.control.native import rad_from_turns, turns_from_rad

_TURN_GATES: Final[frozenset[str]] = frozenset({"gpi", "gpi2", "ms", "zz"})
_QIS_ALIASES: Final[dict[str, str]] = {"cx": "cnot"}

JOB_TYPE: Final[str] = "ionq.circuit.v1"
"""The ``type`` of a JSON gate-list job (the other v0.4 type, ``ionq.qasm3.v1``, carries OpenQASM 3 text)."""
JOB_KEYS: Final[frozenset[str]] = frozenset(
    {"type", "backend", "input", "shots", "name", "metadata", "noise", "settings", "dry_run", "session_id"}
)
"""The keys of the v0.4 ``CircuitJobCreationPayload`` (``additionalProperties: false``); ``target`` is the v0.3 spelling of
``backend``, accepted on input and never written."""
NOISE_KEYS: Final[frozenset[str]] = frozenset({"model", "seed"})
"""The keys of a job's ``noise`` object: the model name (``ideal``, ``aria-1``, ``forte-1``, ...) and an optional seed."""
SETTINGS_KEYS: Final[dict[str, frozenset[str]]] = {
    "compilation": frozenset({"precision", "opt", "gate_basis", "service_version"}),
    "error_mitigation": frozenset({"debiasing", "symmetry_verification"}),
}
"""The two settings groups of the v0.4 spec and the keys each allows."""


def _round12(x: float) -> float:
    return float(f"{x:.11e}")


def _qubits_of(item: dict[str, Any]) -> tuple[int, ...]:
    controls: list[int] = []
    if "controls" in item:
        controls = [int(q) for q in item["controls"]]
    elif "control" in item:
        controls = [int(item["control"])]
    targets: list[int]
    if "targets" in item:
        targets = [int(q) for q in item["targets"]]
    elif "target" in item:
        targets = [int(item["target"])]
    else:
        raise ValueError(f"IonQ gate {item.get('gate')!r} has no target(s)")
    return tuple(controls + targets)


def load_ionq_json(obj: dict[str, Any]) -> Circuit:
    """Import an IonQ circuit (a job body with ``input``, or the ``input`` object itself) into the IR."""
    inp = obj["input"] if "input" in obj else obj
    n_qubits = int(inp["qubits"])
    ops: list[Operation] = []
    for item in inp["circuit"]:
        name = str(item["gate"]).lower()
        if name == "nop":
            continue
        name = _QIS_ALIASES.get(name, name)
        qubits = _qubits_of(item)
        params: tuple[float, ...]
        if name == "gpi" or name == "gpi2":
            params = (rad_from_turns(float(item["phase"])),)
        elif name == "ms":
            phases = item.get("phases", [0.0, 0.0])
            if len(phases) != 2:
                raise ValueError("ms takes two phases")
            angle = float(item.get("angle", 0.25))
            params = (
                rad_from_turns(float(phases[0])),
                rad_from_turns(float(phases[1])),
                rad_from_turns(angle),
            )
        elif name == "zz":
            if "phases" in item:
                raise ValueError("zz carries angle only (Section 7.6)")
            params = (rad_from_turns(float(item["angle"])),)
        elif name in STANDARD_GATES or name == "rz":
            n_params = (NATIVE_GATES["rz"] if name == "rz" else STANDARD_GATES[name])[1]
            if n_params == 0:
                params = ()
            elif "rotation" in item:
                params = (float(item["rotation"]),)
            elif "rotations" in item:
                params = tuple(float(x) for x in item["rotations"])
            else:
                raise ValueError(f"IonQ gate {name!r} needs a rotation")
        else:
            raise ValueError(f"unsupported IonQ gate {name!r}")
        ops.append(Operation(name, qubits, params))
    return Circuit(n_qubits=n_qubits, ops=tuple(ops), measure=tuple(range(n_qubits)))


def dump_ionq_json(circuit: Circuit) -> dict[str, Any]:
    """Export a native circuit as the IonQ ``input`` object (gateset native); non-native gates are refused."""
    items: list[dict[str, Any]] = []
    for op in circuit.ops:
        if op.name in ("measure", "reset", "recool"):
            raise ValueError(
                f"the IonQ circuit JSON has no {op.name!r} operation; measurement is implicit and terminal"
            )
        if op.name not in _TURN_GATES:
            raise ValueError(f"{op.name!r} is not an exported native gate (gpi, gpi2, ms, zz); compile first")
        if op.name in ("gpi", "gpi2"):
            items.append(
                {"gate": op.name, "target": op.qubits[0], "phase": _round12(turns_from_rad(op.params[0]))}
            )
        elif op.name == "ms":
            items.append(
                {
                    "gate": "ms",
                    "targets": list(op.qubits),
                    "phases": [
                        _round12(turns_from_rad(op.params[0])),
                        _round12(turns_from_rad(op.params[1])),
                    ],
                    "angle": _round12(turns_from_rad(op.params[2])),
                }
            )
        else:
            items.append(
                {"gate": "zz", "targets": list(op.qubits), "angle": _round12(turns_from_rad(op.params[0]))}
            )
    return {"gateset": "native", "qubits": circuit.n_qubits, "circuit": items}


@dataclass(frozen=True)
class IonQJob:
    """A circuit job as IonQ's REST API takes it (v0.4 ``CircuitJobCreationPayload``; ``target`` of v0.3 read as
    ``backend``): the circuit (radians; the wire carries turns), the backend name, the shots, the optional name, metadata
    (string values), noise (``{"model": ..., "seed": ...}``), settings and dry-run flag."""

    circuit: Circuit
    backend: str | None = None
    shots: int | None = None
    name: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    noise: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None
    dry_run: bool | None = None
    type: str = JOB_TYPE


def _check_keys(obj: Mapping[str, Any], allowed: frozenset[str], what: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ValueError(
            f"{what} has no key {unknown} (the v0.4 schema sets additionalProperties: false); allowed: {sorted(allowed)}"
        )


def load_job(obj: str | Mapping[str, Any]) -> IonQJob:
    """A v0.3 or v0.4 job body (text or mapping) -> :class:`IonQJob`: the circuit from ``input`` and the fields around it,
    ``target`` (v0.3) or ``backend`` (v0.4) as the backend."""
    body = json.loads(obj) if isinstance(obj, str) else dict(obj)
    if "input" not in body:
        raise ValueError("a job body carries its circuit under 'input'")
    job_type = str(body.get("type", JOB_TYPE))
    if job_type != JOB_TYPE:
        raise ValueError(f"load_job reads {JOB_TYPE!r} jobs (a JSON gate list), got type {job_type!r}")
    backend = body.get("backend", body.get("target"))
    metadata = {str(k): str(v) for k, v in dict(body.get("metadata") or {}).items()}
    return IonQJob(
        circuit=load_ionq_json(body["input"]),
        backend=None if backend is None else str(backend),
        shots=None if body.get("shots") is None else int(body["shots"]),
        name=None if body.get("name") is None else str(body["name"]),
        metadata=metadata,
        noise=None if body.get("noise") is None else dict(body["noise"]),
        settings=None if body.get("settings") is None else dict(body["settings"]),
        dry_run=None if body.get("dry_run") is None else bool(body["dry_run"]),
        type=job_type,
    )


def dump_job(
    circuit: Circuit,
    *,
    backend: str,
    shots: int = 100,
    noise: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    name: str | None = None,
    metadata: Mapping[str, str] | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """The v0.4 job body for a native circuit: ``type``, ``backend`` (``simulator``, ``qpu.forte-1``, ...), ``shots`` and
    ``input``, plus ``name``, ``metadata``, ``noise`` (``{"model": "aria-1", "seed": 7}``: the model is required when noise
    is given), ``settings`` (``compilation`` and ``error_mitigation`` groups) and ``dry_run`` when given; every key is checked
    against the spec, which rejects unknown ones."""
    if shots < 1:
        raise ValueError("shots is a positive count")
    body: dict[str, Any] = {
        "type": JOB_TYPE,
        "backend": str(backend),
        "shots": int(shots),
        "input": dump_ionq_json(circuit),
    }
    if name is not None:
        body["name"] = str(name)
    if metadata is not None:
        body["metadata"] = {str(k): str(v) for k, v in dict(metadata).items()}
    if noise is not None:
        noise_d = dict(noise)
        _check_keys(noise_d, NOISE_KEYS, "noise")
        if "model" not in noise_d:
            raise ValueError("noise needs its 'model' ('ideal', 'aria-1', 'forte-1', ...)")
        body["noise"] = {"model": str(noise_d["model"])}
        if noise_d.get("seed") is not None:
            body["noise"]["seed"] = int(noise_d["seed"])
    if settings is not None:
        settings_d = dict(settings)
        _check_keys(settings_d, frozenset(SETTINGS_KEYS), "settings")
        body["settings"] = {}
        for group, value in settings_d.items():
            group_d = dict(value)
            _check_keys(group_d, SETTINGS_KEYS[group], f"settings.{group}")
            body["settings"][group] = group_d
    if dry_run is not None:
        body["dry_run"] = bool(dry_run)
    _check_keys(body, JOB_KEYS, "the job body")
    return body
