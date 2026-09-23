"""IonQ circuit JSON importer and exporter, and IonQ v0.4 job bodies. IonQ gate phases and angles are in TURNS (``ms``
defaults to ``angle`` 0.25, maximally entangling; the qis gateset's ``rotation`` is in radians) and the IR in radians;
exported turns are rounded to 12 significant digits so values such as 0.75 round-trip exactly.

Result bit order (the exporters live on ``Result``): v1 keys are decimal integers with qubit 0 the least-significant bit;
v2 bitstrings are in wire order, q[0] the leftmost character (IonQ v0.4 API reference, "Reading results"), so ``x q[0]``
on three qubits is ``"1"`` in v1 and ``"100"`` in v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from qutip_trap.control.compiler import NATIVE_GATES, STANDARD_GATES, Circuit, Operation
from qutip_trap.control.native import rad_from_turns, turns_from_rad

_TURN_GATES: Final[frozenset[str]] = frozenset({"gpi", "gpi2", "ms", "zz"})
_QIS_ALIASES: Final[dict[str, str]] = {"cx": "cnot", "rz": "rz"}

JOB_TYPE: Final[str] = "ionq.circuit.v1"
"""The ``type`` of a JSON gate-list job (the other v0.4 type, ``ionq.qasm3.v1``, carries OpenQASM 3 text)."""
JOB_KEYS: Final[frozenset[str]] = frozenset(
    {"type", "backend", "input", "shots", "name", "metadata", "noise", "settings", "dry_run", "session_id"}
)
"""The keys of the v0.4 ``CircuitJobCreationPayload``, which allows no others; the v0.3 ``target`` is read, never written."""
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
            n_params = STANDARD_GATES.get(name, NATIVE_GATES.get(name, (0, 0)))[1]
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
    """A circuit job as IonQ's REST API takes it (v0.4 ``CircuitJobCreationPayload``), the circuit in radians."""

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


def loads(obj: str | Mapping[str, Any]) -> Circuit:
    """IonQ circuit JSON (a job body or its ``input``), as text or as a mapping -> ``Circuit`` (``load_ionq_json``)."""
    data = json.loads(obj) if isinstance(obj, str) else dict(obj)
    return load_ionq_json(data)


def dumps(circuit: Circuit, *, indent: int | None = None) -> str:
    """``Circuit`` -> the IonQ ``input`` object as JSON text (``dump_ionq_json``)."""
    return json.dumps(dump_ionq_json(circuit), indent=indent)


def load_job(obj: str | Mapping[str, Any]) -> IonQJob:
    """A v0.3 or v0.4 job body (text or mapping) -> :class:`IonQJob`, the v0.3 ``target`` read as the backend."""
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
    """The v0.4 job body for a native circuit: ``type``, ``backend``, ``shots`` and ``input``, plus the optional fields that
    are given (``noise`` needs its ``model``); every key is checked against the spec, which rejects unknown ones."""
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


__all__ = [
    "JOB_KEYS",
    "JOB_TYPE",
    "NOISE_KEYS",
    "SETTINGS_KEYS",
    "IonQJob",
    "dump_ionq_json",
    "dump_job",
    "dumps",
    "load_ionq_json",
    "load_job",
    "loads",
]
