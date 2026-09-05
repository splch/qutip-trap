"""Circuit IR and the compiler entry point (PLAN.md Sections 3.3, 7.1, 7.2; Appendix E; milestone M6).

The IR is a list of operations (name, qubits, params) supporting the native set (gpi, gpi2, ms, zz and the
virtual rz) plus the standard set of Section 7.2 and the non-unitary ``measure``, ``reset`` and ``recool``
(Section 7.2 item 4: schedulable from the first release; the scheduler refuses mid-circuit measurement with a
clear error until Section 8.5 is implemented). Parameters are RADIANS; IonQ's turns are converted at the
boundary (``qutip_trap.io.ionq``; Section 13, "Gate parameters").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

M6 = "milestone M6 (control/compiler.py, PLAN.md Section 7.2)"

# name -> (number of qubits, number of parameters)
NATIVE_GATES: Final[dict[str, tuple[int, int]]] = {
    "gpi": (1, 1),
    "gpi2": (1, 1),
    "ms": (2, 3),
    "zz": (2, 1),
    "rz": (1, 1),  # virtual: no pulse, a frame update (Section 7.1)
}
STANDARD_GATES: Final[dict[str, tuple[int, int]]] = {
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "cnot": (2, 0),
    "cx": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
    "cp": (2, 1),
    "u3": (1, 3),
}
NON_UNITARY: Final[frozenset[str]] = frozenset({"measure", "reset", "recool"})


@dataclass(frozen=True)
class Operation:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...]
    """Radians; turns at the IonQ boundary."""

    def __post_init__(self) -> None:
        name = self.name
        if name in NATIVE_GATES or name in STANDARD_GATES:
            arity, n_params = NATIVE_GATES.get(name) or STANDARD_GATES[name]
            if len(self.qubits) != arity:
                raise ValueError(f"{name} acts on {arity} qubit(s), got {self.qubits}")
            if len(self.params) != n_params:
                raise ValueError(f"{name} takes {n_params} parameter(s), got {len(self.params)}")
        elif name in NON_UNITARY:
            if not self.qubits:
                raise ValueError(f"{name} needs at least one target")
            if self.params:
                raise ValueError(f"{name} takes no parameters")
        else:
            raise ValueError(f"unknown operation {name!r}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"{name}: qubit indices must be distinct")
        if any(q < 0 for q in self.qubits):
            raise ValueError(f"{name}: qubit indices must be non-negative")

    @property
    def is_native(self) -> bool:
        return self.name in NATIVE_GATES

    @property
    def is_non_unitary(self) -> bool:
        return self.name in NON_UNITARY


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    ops: tuple[Operation, ...]
    measure: tuple[int, ...]
    """The terminal measurement targets; mid-circuit measure and reset live in ``ops``."""

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("a circuit has at least one qubit")
        for op in self.ops:
            if any(q >= self.n_qubits for q in op.qubits):
                raise ValueError(f"operation {op.name} addresses a qubit outside range({self.n_qubits})")
        if len(set(self.measure)) != len(self.measure) or any(
            q < 0 or q >= self.n_qubits for q in self.measure
        ):
            raise ValueError("measure targets must be distinct qubits of the circuit")

    @property
    def is_native(self) -> bool:
        return all(op.is_native or op.is_non_unitary for op in self.ops)


def compile_to_native(circuit: Circuit, device: Device) -> Circuit:
    """Standard gates -> native gates with phase tracking, verified against target unitaries (Section 7.2)."""
    raise NotImplementedError(f"compile_to_native is {M6}")


__all__ = ["NATIVE_GATES", "NON_UNITARY", "STANDARD_GATES", "Circuit", "Operation", "compile_to_native"]
