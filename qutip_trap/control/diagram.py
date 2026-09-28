"""The text diagram of a circuit (``str(circuit)``): one wire per qubit, the gates in the columns of a greedy layout, a
two-qubit gate's wires joined by a vertical line, and the terminal measurement at the end.

    q0: ─H──●──M─
            │
    q1: ────X──M─

The layout is the application builder's (each gate takes the first column free on every wire it spans, so no vertical
line crosses a gate); angles are printed as fractions of pi where they are one. A diagram wider than ``width`` characters
is folded into consecutive blocks of columns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit, Operation

WIRE: Final[str] = "─"
VERTICAL: Final[str] = "│"
CROSSING: Final[str] = "┼"
CONTROL: Final[str] = "●"

NAMES: Final[dict[str, str]] = {
    "id": "I",
    "sdg": "S†",
    "tdg": "T†",
    "sx": "√X",
}
"""Gate names whose label is not the upper-cased name."""
ENDPOINTS: Final[dict[str, tuple[str, str]]] = {
    "cnot": (CONTROL, "X"),
    "cx": (CONTROL, "X"),
    "cz": (CONTROL, CONTROL),
    "swap": ("×", "×"),
}
"""Two-qubit gates drawn by their endpoint symbols (first qubit, second qubit) rather than a label on both wires."""
NON_UNITARY_LABELS: Final[dict[str, str]] = {"measure": "M", "reset": "|0>", "recool": "cool"}


def angle_text(x: float) -> str:
    """An angle in radians as a fraction of pi where it is one with a denominator up to 8 (``π/2``, ``-3π/4``), else four
    significant digits."""
    for k in (1, 2, 3, 4, 6, 8):
        n = x * k / math.pi
        if round(n) != 0 and abs(n - round(n)) < 1e-9:
            m = int(round(n))
            num = "π" if m == 1 else "-π" if m == -1 else f"{m}π"
            return num if k == 1 else f"{num}/{k}"
    return f"{x:.4g}"


def label(op: Operation) -> str:
    """The label a gate is drawn with: its name upper-cased (``NAMES`` otherwise) with its angles in parentheses."""
    name = NAMES.get(op.name, op.name.upper())
    return f"{name}({', '.join(angle_text(p) for p in op.params)})" if op.params else name


def draw(circuit: Circuit, *, width: int = 100) -> str:
    """The text diagram of ``circuit`` (module docstring), folded into blocks of at most ``width`` characters."""
    n = circuit.n_qubits
    free = [0] * n
    # per column: wire -> cell text, and the (low, high) wire span of every vertical line in it
    cells: list[dict[int, str]] = []
    spans: list[list[tuple[int, int]]] = []

    def place(wires: range | tuple[int, ...]) -> int:
        col = max(free[w] for w in wires)
        while len(cells) <= col:
            cells.append({})
            spans.append([])
        for w in wires:
            free[w] = col + 1
        return col

    for op in circuit.ops:
        if op.is_non_unitary:
            col = place(op.qubits)
            for q in op.qubits:
                cells[col][q] = NON_UNITARY_LABELS[op.name]
        elif len(op.qubits) == 1:
            cells[place(op.qubits)][op.qubits[0]] = label(op)
        else:
            a, b = op.qubits
            lo, hi = min(a, b), max(a, b)
            col = place(range(lo, hi + 1))
            first, second = ENDPOINTS.get(op.name, (label(op), label(op)))
            cells[col][a], cells[col][b] = first, second
            spans[col].append((lo, hi))
    if circuit.measure:
        col = place(range(n))
        for q in circuit.measure:
            cells[col][q] = NON_UNITARY_LABELS["measure"]

    prefix = [f"q{q}: " for q in range(n)]
    pad = max(len(p) for p in prefix)
    blocks: list[list[int]] = [[]]
    used = pad
    for c in range(len(cells)):
        w = max((len(t) for t in cells[c].values()), default=1) + 2
        if blocks[-1] and used + w > width:
            blocks.append([])
            used = pad
        blocks[-1].append(c)
        used += w
    out: list[str] = []
    for block in blocks:
        rows: list[str] = []
        for q in range(n):
            wire = [prefix[q].ljust(pad)]
            gap = [" " * pad]
            for c in block:
                w = max((len(t) for t in cells[c].values()), default=1)
                if q in cells[c]:
                    text = cells[c][q]
                elif any(lo < q < hi for lo, hi in spans[c]):
                    text = CROSSING
                else:
                    text = WIRE
                wire.append(WIRE + text.center(w, WIRE) + WIRE)
                below = any(lo <= q < hi for lo, hi in spans[c])
                gap.append(" " + (VERTICAL if below else " ").center(w) + " ")
            rows.append("".join(wire))
            if q < n - 1:
                rows.append("".join(gap).rstrip())
        out.append("\n".join(rows))
    return "\n\n".join(out)
