"""OpenQASM 2 both ways, shaped like ``json``: ``loads`` is the importer of ``qutip_trap.io.openqasm`` and ``dumps`` the
exporter (qelib1.inc names, one ``creg`` per register, parameters written with ``repr``, ``recool`` refused). By default
``dumps`` declares the native gates it uses over ``u3``, ``rz``, ``rxx`` and ``rzz`` (the same unitary up to a global
phase, for any reader); ``declare_native=False`` leaves bare native names, which this package reads back exactly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from qutip_trap.io.openqasm import load_openqasm2

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit

QELIB_NAMES: Final[dict[str, str]] = {"cnot": "cx"}
"""IR gate name -> qelib1.inc name where the two differ."""

NATIVE_DECLARATIONS: Final[dict[str, str]] = {
    "gpi": "gate gpi(phi) a { u3(pi, phi, pi - phi) a; }",
    "gpi2": "gate gpi2(phi) a { u3(pi/2, phi - pi/2, pi/2 - phi) a; }",
    "ms": "gate ms(phi0, phi1, theta) a, b { rz(-phi0) a; rz(-phi1) b; rxx(theta) a, b; rz(phi0) a; rz(phi1) b; }",
    "zz": "gate zz(theta) a, b { rzz(theta) a, b; }",
}
"""The native gates as qelib1.inc definitions (radians), each equal to ``control.native``'s matrix up to a global phase
(MS as RXX conjugated by RZ(phi0) (x) RZ(phi1); ZZ(theta) = rzz(theta) up to e^{i theta/2})."""


def loads(text: str) -> Circuit:
    """OpenQASM 2 text -> ``Circuit`` (``qutip_trap.io.openqasm.load_openqasm2``)."""
    return load_openqasm2(text)


def dumps(circuit: Circuit, *, declare_native: bool = True) -> str:
    """``Circuit`` -> OpenQASM 2 text, the terminal measurements last, into their registers."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    used = {op.name for op in circuit.ops}
    if declare_native:
        lines.extend(NATIVE_DECLARATIONS[name] for name in NATIVE_DECLARATIONS if name in used)
    lines.append(f"qreg q[{circuit.n_qubits}];")
    for name, qubits in circuit.registers.items():
        lines.append(f"creg {name}[{len(qubits)}];")
    bit_of: dict[int, tuple[str, int]] = {}
    for name, qubits in circuit.registers.items():
        for k, q in enumerate(qubits):
            bit_of.setdefault(q, (name, k))  # a qubit in two registers is written into the first
    for op in circuit.ops:
        if op.name == "measure":
            for q in op.qubits:
                if q not in bit_of:
                    raise ValueError(
                        f"a mid-circuit measure of qubit {q} needs a classical register that holds it (Circuit.registers)"
                    )
                name, bit = bit_of[q]
                lines.append(f"measure q[{q}] -> {name}[{bit}];")
        elif op.name == "reset":
            lines.extend(f"reset q[{q}];" for q in op.qubits)
        elif op.name == "recool":
            raise ValueError("OpenQASM 2 has no recool operation")
        else:
            gate = QELIB_NAMES.get(op.name, op.name)
            params = f"({', '.join(repr(float(x)) for x in op.params)})" if op.params else ""
            lines.append(f"{gate}{params} {', '.join(f'q[{q}]' for q in op.qubits)};")
    measured = set(circuit.measure)
    for name, qubits in circuit.registers.items():
        for k, q in enumerate(qubits):
            if q in measured:
                lines.append(f"measure q[{q}] -> {name}[{k}];")
    return "\n".join(lines) + "\n"
