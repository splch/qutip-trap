"""IonQ circuit JSON importer and exporter (PLAN.md Sections 7.6, 8.6, 9.13).

Encoding (verified against the IonQ OpenAPI v0.4 specification and the qiskit-ionq, ionq-core-python and
PennyLane-IonQ clients, Section 7.6): the native gate names are ``gpi``, ``gpi2``, ``ms``, ``zz`` and ``nop``;
``gpi``/``gpi2`` take a scalar ``target`` and a scalar ``phase`` in TURNS; ``ms`` takes ``targets`` (two),
``phases`` (two, turns) and ``angle`` (turns, default 0.25, maximally entangling); ``zz`` takes ``targets`` and
``angle`` and no phases. The job body is ``{"input": {"gateset": "native" | "qis", "qubits": N, "circuit":
[...]}}``; the qis gateset's rotations (``rotation``) are radians. Internally every parameter is radians
(Section 13, "Gate parameters"); exported turns are rounded to 12 significant digits so that a value such as
0.75 round-trips exactly (the 9.13 test).

Result formats (Section 8.6; the exporters live on ``Result`` in ``run/results.py``). v1, the format
``ionq.result.probabilities.json.v1`` and its histogram and shots forms: decimal-integer keys, qubit 0 the
least-significant bit (``to_ionq_json``, ``to_ionq_histogram``, ``to_ionq_shots``). v2, the format
``ionq.result.probabilities.json.v2``: the envelope ``{"probabilities": {"registers": {<register>: {<bitstring>: p}}}}``
with zero-padded bitstrings per named classical register and ``output_all`` the whole-circuit register (the histogram
form under ``"histogram"`` with counts, the shots form as per-shot bit arrays). Character order of a v2 bitstring,
settled 2026-09-11 (docs/api_implementation_plan.md item 0.4): WIRE order, q[0] the first (leftmost) character, the
reverse of this package's ``bitstring_key``, so ``x q[0]`` on three qubits is ``"1"`` in v1 and ``"100"`` in v2.
Sources: IonQ's documentation, the OpenQASM 3 page of the v0.4 API reference under "Reading results"
(https://docs.ionq.com/api-reference/v0.4/openqasm3: "q[0] is the leftmost character of the bitstring. A three-qubit
circuit applying x q[0]; and measuring all three returns {"100": 1.0}"), and qiskit-ionq 1.1.1, whose
``_decode_distribution_artifact`` in ``ionq_job.py`` reverses every ``output_all`` key before converting it to the
decimal form ("Result artifacts use wire-order bitstrings (qubit 0 first)"). The OpenAPI v0.4 document itself (spec
dated 2026-09-10) does not state the order and its example is the symmetric Bell state. ``Result.to_ionq_v2`` today
emits neither the envelope nor this order; the exporters that do are Phase 1.7 of the plan, and
``tests/test_m6_results_export.py`` holds the fixture.
"""

from __future__ import annotations

from typing import Any, Final

from qutip_trap.control.compiler import NATIVE_GATES, STANDARD_GATES, Circuit, Operation
from qutip_trap.control.native import rad_from_turns, turns_from_rad

_TURN_GATES: Final[frozenset[str]] = frozenset({"gpi", "gpi2", "ms", "zz"})
_QIS_ALIASES: Final[dict[str, str]] = {"cx": "cnot", "rz": "rz"}


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


__all__ = ["dump_ionq_json", "load_ionq_json"]
