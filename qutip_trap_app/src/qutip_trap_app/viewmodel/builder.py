"""The circuit builder's model (PLAN.md Section 14.2 row 0; DESIGN.md Sections 5 and 11, R16): the gates a learner can
place, the layout of a circuit on its wires, the edits, and the OpenQASM 2 text the store keeps.

The store's ``circuit_text`` stays the one source of truth (the presets load it, the prediction prompt is keyed to it, the
worker parses it), so the builder is a structured editor over that text: every render parses it into the IR, every edit
re-serialises the edited IR to OpenQASM 2 (the worked example is a fixed point of that round trip, so opening the builder
never rewrites a circuit), and an import of OpenQASM 2 or IonQ JSON keeps the pasted text verbatim until the next edit.
Angles are radians in the IR and are written as fractions of pi when they are one (``pi/2``, ``-3*pi/4``) and as Python's
shortest round-trip float otherwise, so the text a physicist reads under Code is the text the parser reads back.

Pure Python: no Flet import (``views/builder.py`` draws what this module lays out) and the core only through
``qutip_trap_app.core``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qutip_trap_app.core import Circuit, Operation, load_ionq_json, load_openqasm2

CircuitFormat = Literal["openqasm2", "ionq_json"]
Family = Literal["fixed", "rotation", "pair", "native", "other"]

MAX_QUBITS = 4
"""Every ion adds motional modes and joint dimension; four keeps a full run within minutes on the example device."""
MIN_QUBITS = 1

FAMILY_TITLES: dict[Family, str] = {
    "fixed": "one qubit",
    "rotation": "rotations",
    "pair": "two qubits",
    "native": "native",
    "other": "other",
}
"""The palette's group captions, in display order."""


@dataclass(frozen=True)
class GateSpec:
    """One gate of the palette: what the tile prints, what the inspector edits, what the tooltip says."""

    name: str
    """The IR name (``Operation.name``)."""
    label: str
    """The tile's text."""
    family: Family
    arity: int
    params: tuple[str, ...] = ()
    """Parameter names in the inspector, in the IR's order."""
    defaults: tuple[float, ...] = ()
    """The parameters a freshly placed gate carries (radians)."""
    title: str = ""
    """One line on what the gate does, for the tooltip and the inspector."""
    ends: tuple[str, str] | None = None
    """A two-qubit gate's glyph at each end: ``dot``, ``plus``, ``cross`` or a tile label."""
    shown_param: int | None = None
    """Which parameter the tile prints under its label, if any."""
    qasm: str | None = None
    """The OpenQASM 2 name when it differs from the IR name (``cnot`` is written ``cx``)."""

    @property
    def qasm_name(self) -> str:
        return self.qasm or self.name


_HALF_PI = math.pi / 2.0

PALETTE: tuple[GateSpec, ...] = (
    GateSpec("h", "H", "fixed", 1, title="Hadamard: |0> to the equal superposition (|0> + |1>)/sqrt 2"),
    GateSpec("x", "X", "fixed", 1, title="X: the bit flip, |0> and |1> exchanged"),
    GateSpec("y", "Y", "fixed", 1, title="Y: a bit flip and a phase flip together (i X Z)"),
    GateSpec("z", "Z", "fixed", 1, title="Z: the phase flip, |1> to -|1>"),
    GateSpec("s", "S", "fixed", 1, title="S: a quarter turn about Z (the square root of Z)"),
    GateSpec("sdg", "S†", "fixed", 1, title="S dagger: the inverse quarter turn about Z"),
    GateSpec("t", "T", "fixed", 1, title="T: an eighth turn about Z (the square root of S)"),
    GateSpec("tdg", "T†", "fixed", 1, title="T dagger: the inverse eighth turn about Z"),
    GateSpec("sx", "√X", "fixed", 1, title="Square root of X: half a bit flip, a pi/2 turn about X"),
    GateSpec("rx", "Rx", "rotation", 1, ("θ",), (_HALF_PI,), "Rx(θ): a turn of θ about X", shown_param=0),
    GateSpec("ry", "Ry", "rotation", 1, ("θ",), (_HALF_PI,), "Ry(θ): a turn of θ about Y", shown_param=0),
    GateSpec(
        "rz",
        "Rz",
        "rotation",
        1,
        ("θ",),
        (_HALF_PI,),
        "Rz(θ): a turn of θ about Z (a virtual frame update on the machine)",
        shown_param=0,
    ),
    GateSpec(
        "u3",
        "U",
        "rotation",
        1,
        ("θ", "φ", "λ"),
        (_HALF_PI, 0.0, math.pi),
        "U(θ, φ, λ): the general single-qubit rotation (OpenQASM's U3)",
        shown_param=0,
    ),
    GateSpec(
        "cnot",
        "CNOT",
        "pair",
        2,
        title="CNOT: flips the target when the control is |1>",
        ends=("dot", "plus"),
        qasm="cx",
    ),
    GateSpec("cz", "CZ", "pair", 2, title="CZ: a phase flip on |11> (symmetric)", ends=("dot", "dot")),
    GateSpec(
        "swap", "SWAP", "pair", 2, title="SWAP: exchanges the two qubits' states", ends=("cross", "cross")
    ),
    GateSpec(
        "cp",
        "CP",
        "pair",
        2,
        ("θ",),
        (_HALF_PI,),
        "CP(θ): a phase of θ on |11> (CZ at θ = pi)",
        ends=("dot", "P"),
        shown_param=0,
    ),
    GateSpec(
        "rxx",
        "Rxx",
        "pair",
        2,
        ("θ",),
        (_HALF_PI,),
        "Rxx(θ) = exp(-i θ XX/2): maximally entangling at θ = pi/2",
        ends=("Rxx", "Rxx"),
        shown_param=0,
    ),
    GateSpec(
        "rzz",
        "Rzz",
        "pair",
        2,
        ("θ",),
        (_HALF_PI,),
        "Rzz(θ) = exp(-i θ ZZ/2): maximally entangling at θ = pi/2",
        ends=("Rzz", "Rzz"),
        shown_param=0,
    ),
    GateSpec(
        "gpi",
        "GPi",
        "native",
        1,
        ("φ",),
        (0.0,),
        "GPi(φ): the native pi pulse about the equatorial axis at azimuth φ",
        shown_param=0,
    ),
    GateSpec(
        "gpi2",
        "GPi2",
        "native",
        1,
        ("φ",),
        (0.0,),
        "GPi2(φ): the native pi/2 pulse about the equatorial axis at azimuth φ",
        shown_param=0,
    ),
    GateSpec(
        "ms",
        "MS",
        "native",
        2,
        ("φ0", "φ1", "θ"),
        (0.0, 0.0, _HALF_PI),
        "MS(φ0, φ1, θ) = exp[-i (θ/2) GPi(φ0) GPi(φ1)]: the Mølmer-Sørensen gate, fully entangling at θ = pi/2",
        ends=("MS", "MS"),
        shown_param=2,
    ),
    GateSpec(
        "zz",
        "ZZ",
        "native",
        2,
        ("θ",),
        (_HALF_PI,),
        "ZZ(θ) = exp(-i (θ/2) ZZ): the native Ising gate, fully entangling at θ = pi/2",
        ends=("ZZ", "ZZ"),
        shown_param=0,
    ),
)
"""Every gate the builder offers, in palette order; each is a name the compiler accepts (``STANDARD_GATES`` or the native set)."""

GATES: dict[str, GateSpec] = {g.name: g for g in PALETTE}
GATES["cx"] = GATES["cnot"]

_OTHER_LABELS: dict[str, str] = {"measure": "M", "reset": "reset", "recool": "cool", "id": "I"}


def families() -> tuple[Family, ...]:
    """The palette's groups in display order (only those with a gate)."""
    seen: list[Family] = []
    for g in PALETTE:
        if g.family not in seen:
            seen.append(g.family)
    return tuple(seen)


def palette_group(family: Family) -> tuple[GateSpec, ...]:
    return tuple(g for g in PALETTE if g.family == family)


def spec_of(op: Operation) -> GateSpec:
    """The palette's spec for an operation, or a plain grey one for what the palette does not offer (a mid-circuit
    measure or reset from imported code, the identity)."""
    known = GATES.get(op.name)
    if known is not None:
        return known
    return GateSpec(
        op.name,
        _OTHER_LABELS.get(op.name, op.name),
        "other",
        len(op.qubits),
        tuple(f"p{k}" for k in range(len(op.params))),
        tuple(op.params),
        title=f"{op.name}: not in the palette; imported from code",
        ends=None if len(op.qubits) < 2 else (op.name, op.name),
    )


# ---- angles --------------------------------------------------------------------------------------------------------------------------

_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16)


def pi_fraction(x: float) -> tuple[int, int] | None:
    """``(k, d)`` with ``x = k pi / d`` for a small denominator, or None. Zero is ``(0, 1)``."""
    if not math.isfinite(x):
        return None
    for d in _DENOMINATORS:
        k = round(x * d / math.pi)
        if math.isclose(x, k * math.pi / d, rel_tol=0.0, abs_tol=1e-12):
            return int(k), d
    return None


def _fraction_text(k: int, d: int, pi: str, times: str) -> str:
    if k == 0:
        return "0"
    sign = "-" if k < 0 else ""
    k = abs(k)
    num = pi if k == 1 else f"{k}{times}{pi}"
    return f"{sign}{num}" if d == 1 else f"{sign}{num}/{d}"


def format_angle(x: float) -> str:
    """An angle for the screen: ``π/2``, ``-3π/4``, ``0`` or three significant digits."""
    frac = pi_fraction(x)
    if frac is not None:
        return _fraction_text(frac[0], frac[1], "π", "")
    return f"{x:.3g}"


def qasm_angle(x: float) -> str:
    """An angle for the OpenQASM 2 text: ``pi/2``, ``-3*pi/4``, ``0`` or Python's shortest round-trip float."""
    frac = pi_fraction(x)
    if frac is not None:
        return _fraction_text(frac[0], frac[1], "pi", "*")
    return repr(float(x))


_ANGLE_CHARS = re.compile(r"^[0-9a-z+\-*/^(). ]*$")


def parse_angle(text: str) -> float:
    """An angle typed by the learner, in radians: any OpenQASM 2 parameter expression (``pi/2``, ``-3*pi/4``, ``0.35``,
    ``sqrt(2)``), with ``π`` accepted for ``pi``. The same grammar the Code import reads, so nothing is accepted here
    that the text would refuse."""
    t = text.strip().replace("π", "pi").replace("−", "-")
    if not t:
        raise ValueError("enter an angle in radians, for example pi/2")
    if not _ANGLE_CHARS.match(t):
        raise ValueError(f"{text.strip()!r} is not an angle: use numbers, pi, + - * / ^ and parentheses")
    try:
        circuit = load_openqasm2(f"qreg q[1]; rz({t}) q[0];")
    except Exception as exc:  # the parser's own message names the offending token
        raise ValueError(f"{text.strip()!r} is not an angle: {exc}") from exc
    value = float(circuit.ops[0].params[0])
    if not math.isfinite(value):
        raise ValueError(f"{text.strip()!r} is not a finite angle")
    return value


# ---- text --------------------------------------------------------------------------------------------------------------------------

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _measures_everything(circuit: Circuit) -> bool:
    """Whether the terminal measurement is the worked example's: every qubit into the one register ``c`` in qubit order."""
    every = tuple(range(circuit.n_qubits))
    return tuple(circuit.measure) == every and dict(circuit.registers) == {"c": every}


def to_openqasm2(circuit: Circuit) -> str:
    """The circuit as OpenQASM 2 in the form of the worked example: one register ``q``, one statement per operation, and the
    terminal measurement last: ``creg c[n]`` with ``measure q -> c;`` when every qubit is measured into ``c`` in order, else
    the circuit's own classical registers, one ``creg`` each and one ``measure q[i] -> name[k];`` per written bit, so that
    a program measuring a subset of its qubits (or into several registers) keeps that through every builder edit. A
    ``recool`` has no OpenQASM form and is refused."""
    n = circuit.n_qubits
    lines = [f"{HEADER}qreg q[{n}];"]
    # a register nothing writes into has no OpenQASM form (a creg of size 0 is refused by the loader): a program that
    # measures nothing stays a program that measures nothing, and the machine then reads every ion out (the 0.1.0 rule)
    registers = {name: tuple(qubits) for name, qubits in circuit.registers.items() if qubits}
    if _measures_everything(circuit):
        lines.append(f"creg c[{n}];")
    else:
        lines.extend(f"creg {name}[{len(qubits)}];" for name, qubits in registers.items())
    for op in circuit.ops:
        if op.name == "measure":
            lines.extend(f"measure q[{q}] -> c[{q}];" for q in op.qubits)
        elif op.name == "reset":
            lines.extend(f"reset q[{q}];" for q in op.qubits)
        elif op.name == "recool":
            raise ValueError("recool has no OpenQASM 2 form; remove it before editing the circuit here")
        else:
            name = spec_of(op).qasm_name
            args = ",".join(f"q[{q}]" for q in op.qubits)
            params = f"({', '.join(qasm_angle(p) for p in op.params)})" if op.params else ""
            lines.append(f"{name}{params} {args};")
    if _measures_everything(circuit):
        lines.append("measure q -> c;")
    else:
        for name, qubits in registers.items():
            lines.extend(f"measure q[{q}] -> {name}[{k}];" for k, q in enumerate(qubits))
    return "\n".join(lines) + "\n"


def detect_format(text: str) -> CircuitFormat:
    """IonQ JSON when the text is a JSON object, else OpenQASM 2."""
    return "ionq_json" if text.lstrip().startswith("{") else "openqasm2"


def parse_circuit_text(text: str, fmt: CircuitFormat | None = None) -> Circuit:
    """The IR of a circuit text in either format (detected when not given); the loaders' errors propagate."""
    f = fmt if fmt is not None else detect_format(text)
    if f == "ionq_json":
        return load_ionq_json(json.loads(text))
    return load_openqasm2(text)


# ---- layout --------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Placed:
    """One operation on the grid: its position in the sequence, the column the layout gave it, and the wires it spans."""

    index: int
    op: Operation
    spec: GateSpec
    column: int

    @property
    def lo(self) -> int:
        return min(self.op.qubits)

    @property
    def hi(self) -> int:
        return max(self.op.qubits)

    def spans(self, wire: int) -> bool:
        """Whether the wire is one of the gate's qubits or lies between them (the connector's path)."""
        return self.lo <= wire <= self.hi


@dataclass(frozen=True)
class Layout:
    n_qubits: int
    placed: tuple[Placed, ...]
    n_columns: int

    def on_wire(self, wire: int) -> tuple[Placed, ...]:
        return tuple(p for p in self.placed if wire in p.op.qubits)


def layout(circuit: Circuit) -> Layout:
    """Greedy left packing in sequence order: each gate takes the first column free on every wire it spans, and a
    two-qubit gate occupies the wires between its qubits too, so no connector ever crosses another gate."""
    next_free = [0] * circuit.n_qubits
    placed: list[Placed] = []
    for k, op in enumerate(circuit.ops):
        lo, hi = min(op.qubits), max(op.qubits)
        col = max(next_free[lo : hi + 1])
        for w in range(lo, hi + 1):
            next_free[w] = col + 1
        placed.append(Placed(k, op, spec_of(op), col))
    return Layout(circuit.n_qubits, tuple(placed), max(next_free) if placed else 0)


def insert_index(lay: Layout, wire: int, column: int) -> int:
    """Where a gate dropped at (wire, column) goes in the sequence: before the first gate that spans the wire at or after
    that column, else at the end."""
    for p in lay.placed:
        if p.spans(wire) and p.column >= column:
            return p.index
    return len(lay.placed)


def qubits_for_new(spec: GateSpec, wire: int, n_qubits: int) -> tuple[int, ...]:
    """The qubits a gate placed on a wire takes: the wire, and for a two-qubit gate the wire below it (above it on the
    last wire)."""
    if spec.arity == 1:
        return (wire,)
    if n_qubits < 2:
        raise ValueError("a two-qubit gate needs two qubits")
    other = wire + 1 if wire + 1 < n_qubits else wire - 1
    return (wire, other)


def qubits_for_move(op: Operation, wire: int, n_qubits: int) -> tuple[int, ...]:
    """The qubits of a gate dragged so that its first qubit lands on ``wire``, the shape kept and the whole shifted back
    inside the register when it would overhang."""
    lo = min(op.qubits)
    shift = wire - lo
    hi = max(op.qubits) + shift
    if hi >= n_qubits:
        shift -= hi - (n_qubits - 1)
    if lo + shift < 0:
        shift = -lo
    return tuple(q + shift for q in op.qubits)


# ---- edits (every function returns a new circuit; the store keeps the text of the result) ------------------------------------------


def _with_ops(circuit: Circuit, ops: Sequence[Operation], n_qubits: int | None = None) -> Circuit:
    """The circuit with other operations (and, for a wire added or removed, another qubit count), its terminal measurement
    kept: a program measuring a subset of its qubits, or into named registers, still does after every edit. When the
    wire count changes, a circuit that measured every qubit measures every qubit of the new count; a subset loses the
    qubits that no longer exist."""
    n = circuit.n_qubits if n_qubits is None else n_qubits
    if n == circuit.n_qubits:
        return Circuit(n, tuple(ops), circuit.measure, circuit.registers)
    if _measures_everything(circuit):
        return Circuit(n, tuple(ops), tuple(range(n)))
    measure = tuple(q for q in circuit.measure if q < n)
    registers = {name: tuple(q for q in qubits if q < n) for name, qubits in circuit.registers.items()}
    kept = {name: qubits for name, qubits in registers.items() if qubits}
    return Circuit(n, tuple(ops), measure, kept or {"c": measure})


def empty_circuit(n_qubits: int = 2) -> Circuit:
    return Circuit(n_qubits, (), tuple(range(n_qubits)))


def new_operation(spec: GateSpec, qubits: Sequence[int], params: Sequence[float] | None = None) -> Operation:
    return Operation(
        spec.name, tuple(int(q) for q in qubits), tuple(spec.defaults if params is None else params)
    )


def insert_gate(
    circuit: Circuit, at: int, spec: GateSpec, qubits: Sequence[int], params: Sequence[float] | None = None
) -> Circuit:
    ops = list(circuit.ops)
    ops.insert(max(0, min(at, len(ops))), new_operation(spec, qubits, params))
    return _with_ops(circuit, ops)


def append_gate(
    circuit: Circuit, spec: GateSpec, qubits: Sequence[int], params: Sequence[float] | None = None
) -> Circuit:
    return insert_gate(circuit, len(circuit.ops), spec, qubits, params)


def remove_gate(circuit: Circuit, index: int) -> Circuit:
    ops = list(circuit.ops)
    del ops[index]
    return _with_ops(circuit, ops)


def replace_gate(circuit: Circuit, index: int, op: Operation) -> Circuit:
    ops = list(circuit.ops)
    ops[index] = op
    return _with_ops(circuit, ops)


def set_qubits(circuit: Circuit, index: int, qubits: Sequence[int]) -> Circuit:
    """The gate's qubits as given; when two coincide the other slot takes the qubit that was displaced, so a dropdown
    choice never produces an invalid gate."""
    op = circuit.ops[index]
    new = [int(q) for q in qubits]
    if len(new) == 2 and new[0] == new[1]:
        old = list(op.qubits)
        changed = 0 if new[0] != old[0] else 1
        new[1 - changed] = old[changed]
    return replace_gate(circuit, index, Operation(op.name, tuple(new), op.params))


def set_param(circuit: Circuit, index: int, which: int, value: float) -> Circuit:
    op = circuit.ops[index]
    params = list(op.params)
    params[which] = float(value)
    return replace_gate(circuit, index, Operation(op.name, op.qubits, tuple(params)))


def _spans_intersect(a: Operation, b: Operation) -> bool:
    return min(a.qubits) <= max(b.qubits) and min(b.qubits) <= max(a.qubits)


def neighbours(circuit: Circuit, index: int) -> tuple[int | None, int | None]:
    """The nearest earlier and later gates whose wire spans meet this gate's: the two it can trade places with."""
    op = circuit.ops[index]
    before = next((j for j in range(index - 1, -1, -1) if _spans_intersect(op, circuit.ops[j])), None)
    after = next(
        (j for j in range(index + 1, len(circuit.ops)) if _spans_intersect(op, circuit.ops[j])), None
    )
    return before, after


def move_gate(circuit: Circuit, index: int, direction: int) -> tuple[Circuit, int]:
    """The gate one step earlier (``direction`` negative) or later in time, past the nearest gate on its wires; the
    circuit and the gate's new index (unchanged when there is nothing to pass)."""
    before, after = neighbours(circuit, index)
    target = before if direction < 0 else after
    if target is None:
        return circuit, index
    ops = list(circuit.ops)
    op = ops.pop(index)
    ops.insert(target, op)
    return _with_ops(circuit, ops), target


def move_to(circuit: Circuit, index: int, insert_at: int, qubits: Sequence[int]) -> tuple[Circuit, int]:
    """A dragged gate re-placed: removed from its position, put at ``insert_at`` (an index into the sequence before the
    removal) on the given qubits; the circuit and the gate's new index."""
    ops = list(circuit.ops)
    op = ops.pop(index)
    at = insert_at - 1 if insert_at > index else insert_at
    at = max(0, min(at, len(ops)))
    ops.insert(at, Operation(op.name, tuple(int(q) for q in qubits), op.params))
    return _with_ops(circuit, ops), at


def add_qubit(circuit: Circuit) -> Circuit:
    if circuit.n_qubits >= MAX_QUBITS:
        raise ValueError(f"at most {MAX_QUBITS} qubits: every ion adds modes and joint dimension")
    return _with_ops(circuit, circuit.ops, circuit.n_qubits + 1)


def remove_qubit(circuit: Circuit) -> Circuit:
    """The last wire removed, and with it every gate that touched it."""
    if circuit.n_qubits <= MIN_QUBITS:
        raise ValueError(f"a circuit has at least {MIN_QUBITS} qubit")
    last = circuit.n_qubits - 1
    return _with_ops(circuit, [op for op in circuit.ops if last not in op.qubits], last)


def clear(circuit: Circuit) -> Circuit:
    return _with_ops(circuit, ())


# ---- words -------------------------------------------------------------------------------------------------------------------------


def qubits_text(qubits: Sequence[int]) -> str:
    items = [f"q{q}" for q in qubits]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def describe(op: Operation) -> str:
    """One line for a placed gate: its name with its angles and its qubits in words (R15)."""
    spec = spec_of(op)
    name = spec.label
    if op.params:
        name += "(" + ", ".join(format_angle(p) for p in op.params) + ")"
    if spec.ends == ("dot", "plus"):
        return f"{name}: control q{op.qubits[0]}, target q{op.qubits[1]}"
    return f"{name} on {qubits_text(op.qubits)}"


def summary(lay: Layout) -> str:
    """The status line under the wires: the size of the circuit."""
    n = len(lay.placed)
    gates = "1 gate" if n == 1 else f"{n} gates"
    cols = "1 column" if lay.n_columns == 1 else f"{lay.n_columns} columns"
    return f"{gates} in {cols}" if n else "no gates yet"


__all__ = [
    "FAMILY_TITLES",
    "GATES",
    "HEADER",
    "MAX_QUBITS",
    "MIN_QUBITS",
    "PALETTE",
    "CircuitFormat",
    "Family",
    "GateSpec",
    "Layout",
    "Placed",
    "add_qubit",
    "append_gate",
    "clear",
    "describe",
    "detect_format",
    "empty_circuit",
    "families",
    "format_angle",
    "insert_gate",
    "insert_index",
    "layout",
    "move_gate",
    "move_to",
    "neighbours",
    "new_operation",
    "palette_group",
    "parse_angle",
    "parse_circuit_text",
    "pi_fraction",
    "qasm_angle",
    "qubits_for_move",
    "qubits_for_new",
    "qubits_text",
    "remove_gate",
    "remove_qubit",
    "replace_gate",
    "set_param",
    "set_qubits",
    "spec_of",
    "summary",
    "to_openqasm2",
]
