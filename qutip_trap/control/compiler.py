"""Circuit IR and the compiler (PLAN.md Section 7.2).

The IR is a list of operations (name, qubits, parameters in RADIANS; IonQ's turns are converted at the boundary,
``qutip_trap.io.ionq``): the native set (gpi, gpi2, ms, zz and the virtual rz), the standard set (x, y, z, h, s, sdg, t,
tdg, sx, rx, ry, cnot/cx, cz, swap, cp, u3, rxx, rzz) and the non-unitary measure, reset and recool.

Compilation turns every standard gate into native gates plus virtual RZ frame updates, verifies every block and the whole
circuit against the target matrix up to a global phase, and absorbs the frame into the phases of every later pulse
(phi -> phi - theta, time order), so the compiled circuit carries only gpi, gpi2, ms and zz plus a residual per-qubit Z
frame that the computational-basis measurement discards (``CompileReport.final_frame_rad``).

- Single-qubit gates: the ZYZ angles of the target give one virtual RZ, one GPi or GPi2 pulse, or the ZXZXZ form
  RZ(alpha) GPi2(0) RZ(beta) GPi2(0) RZ(gamma), complete because GPi2(0) RZ(beta) GPi2(0) = RY(-beta) RX(pi) up to a phase.
- CNOT: Maslov's one-XX template RY(v pi/2)_c, XX(s pi/4), RX(-s pi/2)_c RX(-v s pi/2)_t, RY(-v pi/2)_c, equal to
  e^{i pi v s/4} CNOT for all four signs; the scheduler realizes either sign of a pair's calibrated angle, so the physical
  sign never reaches the compiler.
- CP(theta) = [RZ(theta/2) (x) RZ(theta/2)] ZZ(-theta/2) up to a global phase, with ZZ the native zz or the wrapper
  W XX(-theta/4) W_in, W = GPi2(pi/2)^(x2), W_in = GPi2(3 pi/2)^(x2) (Debnath's fixed RZ(sgn(theta) pi/2) pair is exact
  for CZ only).
- CZ = CP(pi); SWAP = three CNOTs; rxx(theta) = XX(theta/2) = MS(0, 0, theta); rzz(theta) = ZZ(theta).
"""

from __future__ import annotations

import cmath
import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from qutip_trap.control import native

if TYPE_CHECKING:
    pass

NATIVE_GATES: Final[dict[str, tuple[int, int]]] = {
    "gpi": (1, 1),
    "gpi2": (1, 1),
    "ms": (2, 3),
    "zz": (2, 1),
    "rz": (1, 1),
}
"""The native set: name -> (qubits, parameters in radians); ``rz`` is virtual, a frame update without a pulse."""
STANDARD_GATES: Final[dict[str, tuple[int, int]]] = {
    "id": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "cnot": (2, 0),
    "cx": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
    "cp": (2, 1),
    "rxx": (2, 1),
    "rzz": (2, 1),
    "u3": (1, 3),
}
"""The standard set the compiler expands into native gates: name -> (qubits, parameters in radians)."""
NON_UNITARY: Final[frozenset[str]] = frozenset({"measure", "reset", "recool"})
"""Schedulable at the end of a circuit; mid-circuit they are refused by the scheduler."""
EXPORTED_NATIVE: Final[frozenset[str]] = frozenset({"gpi", "gpi2", "ms", "zz"})
"""The native gates the IonQ JSON exporter carries; rz is absorbed by the compiler."""

BLOCK_TOLERANCE: Final[float] = 1e-9
"""Residual (max |U_block - e^{i alpha} U_target|) above which a compiled block is refused."""

Entangler = Literal["ms", "zz"]


class CompileError(ValueError):
    """A compiled block failed its numerical verification, or the circuit cannot be compiled."""


@dataclass(frozen=True)
class Operation:
    """One operation of the IR: a gate name of the native or standard set (or ``measure``, ``reset``, ``recool``), the
    qubits in the gate's own order (the first is the first tensor factor of its matrix) and the parameters in radians."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...]

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

    @property
    def is_exported_native(self) -> bool:
        return self.name in EXPORTED_NATIVE


@dataclass(frozen=True)
class Circuit:
    """A circuit: ``n_qubits`` labels 0 to n - 1, the operations in time order, the terminal measurement targets (every
    qubit unless narrowed) and the classical registers a result is reported under. The run maps the labels onto ions, and
    the histogram keys of a ``Result`` put qubit 0 rightmost.

    Also a persistent builder: one method per gate name of ``NATIVE_GATES`` and ``STANDARD_GATES``, qubits first
    (``q``, or ``q0`` and ``q1`` in the gate's own order) and then the parameters in radians, each returning a new circuit
    with the operation appended (``Circuit(2).h(0).cnot(0, 1)``); ``measured`` sets the terminal targets."""

    n_qubits: int
    ops: tuple[Operation, ...] = ()
    measure: tuple[int, ...] = None  # type: ignore[assignment]  # omitted (None) means every qubit; a tuple after __post_init__
    """The terminal measurement targets; omitted, every qubit. Mid-circuit measure and reset live in ``ops``."""
    registers: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]  # omitted (None) means {"c": measure}
    """Name -> the qubits of each classical register in bit order (bit 0 first); omitted, one register ``"c"`` over
    ``measure``. The OpenQASM 2 importer fills it from the ``creg`` declarations and the exporter writes them back."""

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("a circuit has at least one qubit")
        object.__setattr__(self, "ops", tuple(self.ops))
        for op in self.ops:
            if any(q >= self.n_qubits for q in op.qubits):
                raise ValueError(f"operation {op.name} addresses a qubit outside range({self.n_qubits})")
        given_measure: object = self.measure
        measure = (
            tuple(range(self.n_qubits)) if given_measure is None else tuple(int(q) for q in self.measure)
        )
        if len(set(measure)) != len(measure) or any(q < 0 or q >= self.n_qubits for q in measure):
            raise ValueError("measure targets must be distinct qubits of the circuit")
        object.__setattr__(self, "measure", measure)
        given_registers: object = self.registers
        if given_registers is None:
            registers = {"c": measure}
        else:
            registers = {
                str(name): tuple(int(q) for q in qubits) for name, qubits in dict(self.registers).items()
            }
        for name, qubits in registers.items():
            if len(set(qubits)) != len(qubits) or any(q < 0 or q >= self.n_qubits for q in qubits):
                raise ValueError(f"register {name!r} names a qubit outside the circuit, or one qubit twice")
        object.__setattr__(self, "registers", registers)

    @property
    def is_native(self) -> bool:
        return all(op.is_native or op.is_non_unitary for op in self.ops)

    @property
    def is_exported_native(self) -> bool:
        """Only gpi, gpi2, ms, zz (and non-unitary operations): what the IonQ JSON exporter accepts."""
        return all(op.is_exported_native or op.is_non_unitary for op in self.ops)

    def entangling_pairs(self) -> tuple[tuple[int, int], ...]:
        """The pairs (ordered as written) of every two-qubit gate, each once."""
        seen: list[tuple[int, int]] = []
        for op in self.ops:
            if len(op.qubits) == 2 and not op.is_non_unitary:
                pair = (int(op.qubits[0]), int(op.qubits[1]))
                if pair not in seen and (pair[1], pair[0]) not in seen:
                    seen.append(pair)
        return tuple(seen)

    def measured(self, *qubits: int, registers: Mapping[str, Sequence[int]] | None = None) -> Circuit:
        """This circuit measuring exactly ``qubits`` at the end, reported under ``registers`` (default: one register ``"c"``
        over them in the order given)."""
        measure = tuple(int(q) for q in qubits)
        regs = (
            {"c": measure}
            if registers is None
            else {name: tuple(int(q) for q in qs) for name, qs in registers.items()}
        )
        return dataclasses.replace(self, measure=measure, registers=regs)

    @classmethod
    def from_openqasm(cls, text: str) -> Circuit:
        """The OpenQASM 2 importer (``qutip_trap.io.openqasm``): angles in radians, ``creg`` names into ``registers``."""
        from qutip_trap.io.openqasm import load_openqasm2

        return load_openqasm2(text)

    @classmethod
    def from_ionq(cls, obj: Mapping[str, Any]) -> Circuit:
        """The IonQ circuit JSON importer: a job body with ``input`` or the ``input`` object, turns converted to radians."""
        from qutip_trap.io.ionq import load_ionq_json

        return load_ionq_json(dict(obj))

    def to_openqasm(self, *, declare_native: bool = True) -> str:
        """OpenQASM 2 text, the native gates declared as qelib1.inc definitions unless told not to."""
        from qutip_trap.io.openqasm import dumps

        return dumps(self, declare_native=declare_native)

    def to_ionq(self) -> dict[str, Any]:
        """The IonQ ``input`` object; the circuit must be native (compile first)."""
        from qutip_trap.io.ionq import dump_ionq_json

        return dump_ionq_json(self)

    # ---- the builder -------------------------------------------------------------------------------------------------------

    def _op(self, name: str, qubits: tuple[int, ...], params: tuple[float, ...] = ()) -> Circuit:
        op = Operation(name, tuple(int(q) for q in qubits), tuple(float(p) for p in params))
        return dataclasses.replace(self, ops=self.ops + (op,))

    def gpi(self, q: int, phase: float) -> Circuit:
        return self._op("gpi", (q,), (phase,))

    def gpi2(self, q: int, phase: float) -> Circuit:
        return self._op("gpi2", (q,), (phase,))

    def ms(self, q0: int, q1: int, phi0: float, phi1: float, theta: float) -> Circuit:
        return self._op("ms", (q0, q1), (phi0, phi1, theta))

    def zz(self, q0: int, q1: int, theta: float) -> Circuit:
        return self._op("zz", (q0, q1), (theta,))

    def rz(self, q: int, theta: float) -> Circuit:
        return self._op("rz", (q,), (theta,))

    def id(self, q: int) -> Circuit:
        return self._op("id", (q,))

    def x(self, q: int) -> Circuit:
        return self._op("x", (q,))

    def y(self, q: int) -> Circuit:
        return self._op("y", (q,))

    def z(self, q: int) -> Circuit:
        return self._op("z", (q,))

    def h(self, q: int) -> Circuit:
        return self._op("h", (q,))

    def s(self, q: int) -> Circuit:
        return self._op("s", (q,))

    def sdg(self, q: int) -> Circuit:
        return self._op("sdg", (q,))

    def t(self, q: int) -> Circuit:
        return self._op("t", (q,))

    def tdg(self, q: int) -> Circuit:
        return self._op("tdg", (q,))

    def sx(self, q: int) -> Circuit:
        return self._op("sx", (q,))

    def rx(self, q: int, theta: float) -> Circuit:
        return self._op("rx", (q,), (theta,))

    def ry(self, q: int, theta: float) -> Circuit:
        return self._op("ry", (q,), (theta,))

    def cnot(self, q0: int, q1: int) -> Circuit:
        return self._op("cnot", (q0, q1))

    def cx(self, q0: int, q1: int) -> Circuit:
        return self._op("cx", (q0, q1))

    def cz(self, q0: int, q1: int) -> Circuit:
        return self._op("cz", (q0, q1))

    def swap(self, q0: int, q1: int) -> Circuit:
        return self._op("swap", (q0, q1))

    def cp(self, q0: int, q1: int, theta: float) -> Circuit:
        return self._op("cp", (q0, q1), (theta,))

    def rxx(self, q0: int, q1: int, theta: float) -> Circuit:
        return self._op("rxx", (q0, q1), (theta,))

    def rzz(self, q0: int, q1: int, theta: float) -> Circuit:
        return self._op("rzz", (q0, q1), (theta,))

    def u3(self, q: int, theta: float, phi: float, lam: float) -> Circuit:
        return self._op("u3", (q,), (theta, phi, lam))


# ---- ideal matrices: the compiler's definition of what a gate is supposed to do ------------------------------------------


def cp_matrix(theta: float) -> np.ndarray:
    """CP(theta) = diag(1, 1, 1, e^{i theta})."""
    return np.diag([1.0, 1.0, 1.0, cmath.exp(1j * theta)]).astype(complex)


CNOT_MATRIX: Final[np.ndarray] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
"""Basis |c t>, the control the first (more significant) factor."""
SWAP_MATRIX: Final[np.ndarray] = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)

_FIXED_MATRICES: Final[dict[str, np.ndarray]] = {
    "id": native.IDENTITY_2,
    "x": native.PAULI_X,
    "y": native.PAULI_Y,
    "z": native.PAULI_Z,
    "h": np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0),
    "s": np.diag([1.0, 1j]).astype(complex),
    "sdg": np.diag([1.0, -1j]).astype(complex),
    "t": np.diag([1.0, cmath.exp(1j * math.pi / 4.0)]).astype(complex),
    "tdg": np.diag([1.0, cmath.exp(-1j * math.pi / 4.0)]).astype(complex),
    "sx": 0.5 * np.array([[1.0 + 1j, 1.0 - 1j], [1.0 - 1j, 1.0 + 1j]], dtype=complex),
    "cnot": CNOT_MATRIX,
    "cx": CNOT_MATRIX,
    "cz": cp_matrix(math.pi),
    "swap": SWAP_MATRIX,
}
"""The matrices of the parameterless gates."""


def gate_matrix(op: Operation) -> np.ndarray:
    """The ideal matrix of a unitary operation in the qubit order of ``op.qubits`` (first qubit = first tensor factor)."""
    n, p = op.name, op.params
    if n in _FIXED_MATRICES:
        return _FIXED_MATRICES[n].copy()
    if n == "gpi":
        return native.gpi(p[0])
    if n == "gpi2":
        return native.gpi2(p[0])
    if n == "rz":
        return native.rz(p[0])
    if n == "ms":
        return native.ms(p[0], p[1], p[2])
    if n in ("zz", "rzz"):
        return native.zz(p[0])
    if n == "rx":
        return native.r_phi(p[0], 0.0)
    if n == "ry":
        return native.r_phi(p[0], math.pi / 2.0)
    if n == "cp":
        return cp_matrix(p[0])
    if n == "rxx":
        return native.xx(0.5 * p[0])
    if n == "u3":
        # OpenQASM 2: [[cos(theta/2), -e^{i lam} sin(theta/2)], [e^{i phi} sin(theta/2), e^{i(phi + lam)} cos(theta/2)]]
        c, s = math.cos(p[0] / 2.0), math.sin(p[0] / 2.0)
        return np.array(
            [[c, -cmath.exp(1j * p[2]) * s], [cmath.exp(1j * p[1]) * s, cmath.exp(1j * (p[1] + p[2])) * c]],
            dtype=complex,
        )
    raise ValueError(f"{n!r} has no unitary matrix")


def embed(mat: np.ndarray, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    """``mat`` on ``qubits`` (its first factor the first listed qubit) as a 2^n x 2^n matrix in the result bit order:
    qubit 0 is the LEAST-significant bit of the basis index, so the tensor axis of qubit j is n - 1 - j."""
    k = len(qubits)
    if mat.shape != (2**k, 2**k):
        raise ValueError("matrix and qubit count disagree")
    dim = 2**n_qubits
    ident = np.eye(dim, dtype=complex).reshape([2] * (2 * n_qubits))
    axes_in = [n_qubits - 1 - q for q in qubits]
    tensor = np.tensordot(mat.reshape([2] * (2 * k)), ident, axes=(list(range(k, 2 * k)), axes_in))
    # tensordot put the gate's output axes first; move them back to the qubits' axis positions
    out = np.moveaxis(tensor, list(range(k)), axes_in)
    return np.asarray(out.reshape(dim, dim))


def circuit_unitary(circuit: Circuit, ops: Sequence[Operation] | None = None) -> np.ndarray:
    """The ideal unitary of the circuit's operations in time order; refuses a non-unitary operation."""
    dim = 2**circuit.n_qubits
    u = np.eye(dim, dtype=complex)
    for op in circuit.ops if ops is None else ops:
        if op.is_non_unitary:
            raise ValueError(f"the circuit contains a non-unitary {op.name!r}: no unitary is defined")
        u = embed(gate_matrix(op), op.qubits, circuit.n_qubits) @ u
    return u


def ideal_probabilities(circuit: Circuit) -> dict[str, float]:
    """The target distribution |<b|U|0...0>|^2 keyed by the bitstring (qubit 0 rightmost), over the measured qubits only
    when ``circuit.measure`` is a subset."""
    u = circuit_unitary(circuit)
    amps = u[:, 0]
    n = circuit.n_qubits
    measured = tuple(sorted(circuit.measure)) if circuit.measure else tuple(range(n))
    out: dict[str, float] = {}
    for index, a in enumerate(amps):
        p = float(abs(a) ** 2)
        if p < 1e-15:
            continue
        bits = [(index >> q) & 1 for q in measured]
        key = "".join(str(b) for b in reversed(bits))
        out[key] = out.get(key, 0.0) + p
    return out


def verify_operations(
    ops: Sequence[Operation], target: np.ndarray, qubits: Sequence[int], *, tol: float = BLOCK_TOLERANCE
) -> float:
    """max |U_ops - e^{i alpha} target| of ``ops`` on ``qubits`` (the target's first factor the first listed qubit);
    ``CompileError`` when the operations do not reproduce the target up to a global phase within ``tol``."""
    local = {q: k for k, q in enumerate(qubits)}
    n = len(qubits)
    u = np.eye(2**n, dtype=complex)
    for op in ops:
        u = embed(gate_matrix(op), [local[q] for q in op.qubits], n) @ u
    want = embed(target, list(range(n)), n)
    phase = native.global_phase(u, want, atol=tol)
    if phase is None:
        raise CompileError("the operations do not reproduce their target matrix up to a global phase")
    return float(np.max(np.abs(u - cmath.exp(1j * phase) * want)))


# ---- decompositions ---------------------------------------------------------------------------------------------------------


def zyz_angles(u: np.ndarray) -> tuple[float, float, float, float]:
    """(a, b, c, delta) with U = e^{i delta} RZ(a) RY(b) RZ(c), b in [0, pi]; degenerate cases take c = 0."""
    u = np.asarray(u, dtype=complex)
    if u.shape != (2, 2):
        raise ValueError("a 2 x 2 matrix")
    det = np.linalg.det(u)
    if (
        not np.all(np.isfinite(u))
        or abs(abs(det) - 1.0) > 1e-9
        or np.max(np.abs(u.conj().T @ u - np.eye(2))) > 1e-9
    ):
        raise CompileError("not a unitary matrix")
    b = 2.0 * math.atan2(abs(u[1, 0]), abs(u[0, 0]))
    if abs(u[1, 0]) < 1e-12:  # b = 0: a pure Z rotation, a + c determined
        c = 0.0
        a = cmath.phase(u[1, 1]) - cmath.phase(u[0, 0])
    elif abs(u[0, 0]) < 1e-12:  # b = pi: a - c determined; split it evenly so that a + c = 0 (no residual rz)
        a = 0.5 * (cmath.phase(u[1, 0]) - cmath.phase(u[0, 1]) + math.pi)
        c = -a
    else:
        # arg V10 - arg V00 = a and arg V11 - arg V10 = c, each modulo 2 pi with no half-angle branch to choose
        a = cmath.phase(u[1, 0]) - cmath.phase(u[0, 0])
        c = cmath.phase(u[1, 1]) - cmath.phase(u[1, 0])
    phase = native.global_phase(u, native.rz(a) @ native.r_phi(b, math.pi / 2.0) @ native.rz(c))
    if phase is None:
        raise CompileError("ZYZ decomposition failed to reproduce the matrix")
    return float(a), float(b), float(c), float(phase)


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def decompose_single_qubit(u: np.ndarray, qubit: int, *, tol: float = 1e-10) -> list[Operation]:
    """Native operations (time order, with virtual rz) for the 2 x 2 unitary ``u`` on ``qubit``: a pure Z rotation is one
    rz, a pi/2 or pi equatorial rotation one GPi2 or GPi pulse plus an rz, anything else the ZXZXZ form with two GPi2
    pulses; verified against ``u`` up to a global phase."""
    a, b, c, _delta = zyz_angles(u)
    ops: list[Operation]
    if abs(math.sin(b / 2.0)) < tol:
        ops = [] if abs(_wrap(a + c)) < tol else [Operation("rz", (qubit,), (_wrap(a + c),))]
    elif abs(math.cos(b / 2.0)) < tol:
        # RY(pi) = -i GPi(pi/2); RZ(a) G(phi) = G(phi + a) RZ(a): U ~ GPi(pi/2 + a) RZ(a + c)
        ops = [
            Operation("rz", (qubit,), (_wrap(a + c),)),
            Operation("gpi", (qubit,), (_wrap(math.pi / 2.0 + a),)),
        ]
    elif abs(b - math.pi / 2.0) < tol:
        # RY(pi/2) = GPi2(pi/2): U ~ GPi2(pi/2 + a) RZ(a + c)
        ops = [
            Operation("rz", (qubit,), (_wrap(a + c),)),
            Operation("gpi2", (qubit,), (_wrap(math.pi / 2.0 + a),)),
        ]
    else:
        # ZXZXZ: U ~ RZ(alpha) GPi2(0) RZ(beta) GPi2(0) RZ(gamma) with (alpha, -beta, -gamma) the ZYZ angles of U X
        a2, b2, c2, _ = zyz_angles(np.asarray(u, dtype=complex) @ native.PAULI_X)
        alpha, beta, gamma = a2, -b2, -c2
        ops = [
            Operation("rz", (qubit,), (_wrap(gamma),)),
            Operation("gpi2", (qubit,), (0.0,)),
            Operation("rz", (qubit,), (_wrap(beta),)),
            Operation("gpi2", (qubit,), (0.0,)),
            Operation("rz", (qubit,), (_wrap(alpha),)),
        ]
    ops = [op for op in ops if not (op.name == "rz" and abs(op.params[0]) < tol)]
    verify_operations(ops, np.asarray(u, dtype=complex), (qubit,))
    return ops


def _xx_ops(chi: float, pair: tuple[int, int]) -> list[Operation]:
    """XX(chi) = exp(-i chi sigma_x sigma_x) as one native ms: MS(0, 0, 2 chi) for chi >= 0, MS(0, pi, -2 chi) for chi < 0
    (GPi(pi) = -X flips the generator's sign; the exported angle stays in [0, pi/2])."""
    if chi >= 0.0:
        return [Operation("ms", pair, (0.0, 0.0, 2.0 * chi))]
    return [Operation("ms", pair, (0.0, math.pi, -2.0 * chi))]


def _zz_ops(theta: float, pair: tuple[int, int], entangler: Entangler) -> list[Operation]:
    """ZZ(theta) = exp(-i (theta/2) Z Z): the native zz, or the wrapper W XX(theta/2) W_in on an MS-only device."""
    if entangler == "zz":
        return [Operation("zz", pair, (theta,))]
    a, b = pair
    return (
        [Operation("gpi2", (a,), (1.5 * math.pi,)), Operation("gpi2", (b,), (1.5 * math.pi,))]
        + _xx_ops(0.5 * theta, pair)
        + [Operation("gpi2", (a,), (0.5 * math.pi,)), Operation("gpi2", (b,), (0.5 * math.pi,))]
    )


def cnot_template(control: int, target: int, *, s: int = 1, v: int = 1) -> list[Operation]:
    """Maslov's CNOT from one XX and four pulses, time order: RY(v pi/2)_c; XX(s pi/4); RX(-s pi/2)_c with
    RX(-v s pi/2)_t; RY(-v pi/2)_c; equal to e^{i pi v s/4} CNOT. RY(+-pi/2) = GPi2(+-pi/2), RX(-pi/2) = GPi2(pi),
    RX(pi/2) = GPi2(0)."""
    if s not in (1, -1) or v not in (1, -1):
        raise ValueError("s and v are signs")

    def ry(q: int, sign: int) -> Operation:  # RY(sign pi/2) = GPi2(sign pi/2)
        return Operation("gpi2", (q,), (_wrap(sign * math.pi / 2.0),))

    def rx(q: int, sign: int) -> Operation:  # RX(sign pi/2): GPi2(0) for +, GPi2(pi) for -
        return Operation("gpi2", (q,), (0.0 if sign > 0 else math.pi,))

    return (
        [ry(control, v)]
        + _xx_ops(s * math.pi / 4.0, (control, target))
        + [rx(control, -s), rx(target, -v * s), ry(control, -v)]
    )


def cp_template(theta: float, pair: tuple[int, int], entangler: Entangler) -> list[Operation]:
    """CP(theta) = [RZ(theta/2) (x) RZ(theta/2)] ZZ(-theta/2) up to a global phase (exact): time order ZZ then the rz pair."""
    th = _wrap(theta)
    a, b = pair
    return _zz_ops(-0.5 * th, pair, entangler) + [
        Operation("rz", (a,), (0.5 * th,)),
        Operation("rz", (b,), (0.5 * th,)),
    ]


def decompose_two_qubit(op: Operation, *, entangler: Entangler) -> list[Operation]:
    """Native operations for a standard two-qubit gate; verified against the target up to a global phase."""
    a, b = op.qubits
    pair = (a, b)
    name = op.name
    if name in ("cnot", "cx"):
        if entangler == "ms":
            ops = cnot_template(a, b)
        else:
            # CNOT = (I (x) H) CZ (I (x) H) on a zz device
            h = gate_matrix(Operation("h", (b,), ()))
            ops = (
                decompose_single_qubit(h, b) + cp_template(math.pi, pair, "zz") + decompose_single_qubit(h, b)
            )
    elif name == "cz":
        ops = cp_template(math.pi, pair, entangler)
    elif name == "cp":
        ops = cp_template(op.params[0], pair, entangler)
    elif name == "swap":
        ops = (
            decompose_two_qubit(Operation("cnot", (a, b), ()), entangler=entangler)
            + decompose_two_qubit(Operation("cnot", (b, a), ()), entangler=entangler)
            + decompose_two_qubit(Operation("cnot", (a, b), ()), entangler=entangler)
        )
    elif name == "rxx":
        ops = _xx_ops(0.5 * op.params[0], pair)
    elif name == "rzz":
        ops = _zz_ops(op.params[0], pair, entangler)
    else:
        raise ValueError(f"{name!r} is not a standard two-qubit gate")
    verify_operations(ops, gate_matrix(op), pair)
    return ops


# ---- frame propagation and the compiler entry point -------------------------------------------------------------------------


def propagate_frames(ops: Sequence[Operation], n_qubits: int) -> tuple[list[Operation], dict[int, float]]:
    """Absorb every rz into the phases of the later pulses (phi -> phi - theta, time order) and return the rz-free
    operations with the residual per-qubit frame, which a computational-basis measurement discards."""
    frame = {q: 0.0 for q in range(n_qubits)}
    out: list[Operation] = []
    for op in ops:
        if op.name == "rz":
            frame[op.qubits[0]] += op.params[0]
        elif op.name in ("gpi", "gpi2"):
            out.append(Operation(op.name, op.qubits, (_wrap(op.params[0] - frame[op.qubits[0]]),)))
        elif op.name == "ms":
            a, b = op.qubits
            out.append(
                Operation(
                    "ms",
                    op.qubits,
                    (_wrap(op.params[0] - frame[a]), _wrap(op.params[1] - frame[b]), op.params[2]),
                )
            )
        else:  # zz commutes with the frame; non-unitary operations are untouched
            out.append(op)
    return out, {q: _wrap(v) for q, v in frame.items()}


def frame_unitary(frame_rad: dict[int, float], n_qubits: int) -> np.ndarray:
    """The residual frame as the matrix (x)_q RZ(theta_q)."""
    u = np.eye(2**n_qubits, dtype=complex)
    for q, theta in frame_rad.items():
        if theta != 0.0:
            u = embed(native.rz(theta), (q,), n_qubits) @ u
    return u


@dataclass(frozen=True)
class CompileReport:
    """What the compiler did and verified."""

    circuit: Circuit
    """The compiled circuit: gpi, gpi2, ms, zz and the non-unitary operations only."""
    final_frame_rad: dict[int, float]
    """The residual virtual-Z frame per qubit at the end (discarded by the measurement)."""
    n_pulses: int
    n_entangling: int
    block_residuals: tuple[float, ...]
    """max |U_block - e^{i alpha} U_target| per compiled block."""
    circuit_residual: float | None
    """max |RZ(frame) U_compiled - e^{i alpha} U_target| over the whole circuit (None when a mid-circuit operation or more
    than ten qubits prevent it)."""
    entangler: Entangler
    notes: tuple[str, ...] = field(default_factory=tuple)


def compile_report(circuit: Circuit, *, entangler: Entangler = "ms") -> CompileReport:
    """Standard gates -> native gates with phase tracking, every block and the whole circuit verified; native circuits pass
    through with their rz absorbed."""
    native_ops: list[Operation] = []
    residuals: list[float] = []
    notes: list[str] = []
    for op in circuit.ops:
        if op.is_non_unitary or op.is_native:
            native_ops.append(op)
            continue
        if len(op.qubits) == 1:
            block = decompose_single_qubit(gate_matrix(op), op.qubits[0])
        else:
            block = decompose_two_qubit(op, entangler=entangler)
        residuals.append(verify_operations(block, gate_matrix(op), op.qubits))
        native_ops.extend(block)
    propagated, frame = propagate_frames(native_ops, circuit.n_qubits)
    compiled = Circuit(circuit.n_qubits, tuple(propagated), circuit.measure, circuit.registers)
    circuit_residual: float | None = None
    if any(op.is_non_unitary for op in circuit.ops):
        notes.append(
            "mid-circuit non-unitary operation: blocks verified, the whole-circuit unitary is undefined"
        )
    elif circuit.n_qubits > 10:
        notes.append("more than 10 qubits: the whole-circuit matrix check is skipped, blocks verified")
    else:
        target = circuit_unitary(circuit)
        got = frame_unitary(frame, circuit.n_qubits) @ circuit_unitary(compiled)
        phase = native.global_phase(got, target, atol=1e-8)
        if phase is None:
            raise CompileError(
                "the compiled circuit does not reproduce the target unitary up to the residual frame"
            )
        circuit_residual = float(np.max(np.abs(got - cmath.exp(1j * phase) * target)))
    return CompileReport(
        circuit=compiled,
        final_frame_rad=frame,
        n_pulses=sum(1 for op in propagated if op.is_exported_native),
        n_entangling=sum(1 for op in propagated if op.name in ("ms", "zz")),
        block_residuals=tuple(residuals),
        circuit_residual=circuit_residual,
        entangler=entangler,
        notes=tuple(notes),
    )


def compile_to_native(circuit: Circuit) -> Circuit:
    """The native circuit of :func:`compile_report`."""
    return compile_report(circuit).circuit
