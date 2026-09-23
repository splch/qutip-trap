"""Circuit IR and the compiler: standard gates -> native gpi, gpi2, ms and zz, with virtual-Z frame tracking.

Parameters are radians (IonQ turns are converted in ``qutip_trap.io.ionq``). Every compiled block is verified up to a
global phase; rz updates are absorbed into later pulse phases (phi -> phi - theta, time order), leaving a residual Z frame
per qubit that the computational-basis measurement discards. The one place ideal gate matrices are used.
"""

from __future__ import annotations

import cmath
import dataclasses
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from qutip_trap.control import native

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

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
"""The standard gates the compiler expands into native ones: name -> (qubits, parameters in radians)."""
NON_UNITARY: Final[frozenset[str]] = frozenset({"measure", "reset", "recool"})
"""The non-unitary operations, schedulable at the end of a circuit; mid-circuit they are refused."""
EXPORTED_NATIVE: Final[frozenset[str]] = frozenset({"gpi", "gpi2", "ms", "zz"})
"""The native gates the IonQ JSON exporter carries; rz is absorbed by the compiler."""

BLOCK_TOLERANCE: Final[float] = 1e-9
"""Residual (max |U_block - e^{i alpha} U_target|) above which a compiled block is refused."""

Entangler = Literal["ms", "zz"]


class CompileError(ValueError):
    """A compiled block failed its numerical verification, or the circuit cannot be compiled."""


@dataclass(frozen=True)
class Operation:
    """One IR operation: a gate (or non-unitary) name, its qubits in the gate's own order and its parameters in radians."""

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
    """A circuit of the IR: qubit labels 0 to ``n_qubits`` - 1, the operations in time order, the terminal measurement
    targets and the classical registers; also a persistent builder, one method per gate (``Circuit(2).h(0).cnot(0, 1)``)."""

    n_qubits: int
    ops: tuple[Operation, ...] = ()
    measure: tuple[int, ...] = None  # type: ignore[assignment]  # omitted (None) means every qubit; a tuple after __post_init__
    """The terminal measurement targets (omitted: every qubit); mid-circuit measure and reset live in ``ops``."""
    registers: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]  # omitted (None) means {"c": measure}
    """Name -> the qubits of each classical register, its bit 0 first; omitted, one register ``"c"`` over ``measure``."""

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("a circuit has at least one qubit")
        object.__setattr__(self, "ops", tuple(self.ops))
        for op in self.ops:
            if any(q >= self.n_qubits for q in op.qubits):
                raise ValueError(f"operation {op.name} addresses a qubit outside range({self.n_qubits})")
        # the two defaults arrive as None (the omitted argument) and leave as the declared types
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

    @property
    def unitary_ops(self) -> tuple[Operation, ...]:
        return tuple(op for op in self.ops if not op.is_non_unitary)

    def entangling_pairs(self) -> tuple[tuple[int, int], ...]:
        """The (ordered as written) pairs of every ms/zz/two-qubit gate, each once."""
        seen: list[tuple[int, int]] = []
        for op in self.ops:
            if len(op.qubits) == 2 and not op.is_non_unitary:
                pair = (int(op.qubits[0]), int(op.qubits[1]))
                if pair not in seen and (pair[1], pair[0]) not in seen:
                    seen.append(pair)
        return tuple(seen)

    # ---- the builder -----------------------------------------------------------------------------------------------

    def _with(self, op: Operation) -> Circuit:
        return dataclasses.replace(self, ops=self.ops + (op,))

    def measured(self, *qubits: int, registers: Mapping[str, Sequence[int]] | None = None) -> Circuit:
        """This circuit measuring exactly ``qubits`` at the end, under ``registers`` (default: one register ``"c"``)."""
        measure = tuple(int(q) for q in qubits)
        regs = (
            {"c": measure}
            if registers is None
            else {name: tuple(int(q) for q in qs) for name, qs in registers.items()}
        )
        return dataclasses.replace(self, measure=measure, registers=regs)

    @classmethod
    def from_openqasm(cls, text: str) -> Circuit:
        """Parse OpenQASM 2 text: angles in radians, ``creg`` names into ``registers``."""
        from qutip_trap.io.openqasm import load_openqasm2

        return load_openqasm2(text)

    @classmethod
    def from_ionq(cls, obj: Mapping[str, Any]) -> Circuit:
        """Parse IonQ circuit JSON: a job body with ``input`` or the ``input`` object, native or qis gates, in turns."""
        from qutip_trap.io.ionq import load_ionq_json

        return load_ionq_json(dict(obj))

    def to_openqasm(self, *, declare_native: bool = True) -> str:
        """OpenQASM 2 text, the native gates declared as qelib1.inc definitions unless ``declare_native`` is False."""
        from qutip_trap.io.qasm2 import dumps

        return dumps(self, declare_native=declare_native)

    def to_ionq(self) -> dict[str, Any]:
        """The IonQ ``input`` object; the circuit must be native (compile first)."""
        from qutip_trap.io.ionq import dump_ionq_json

        return dump_ionq_json(self)

    if TYPE_CHECKING:
        # the gate methods are attached below by ``_builder_method``; these stubs give type checkers their signatures
        def gpi(self, q: int, phase: float) -> Circuit: ...
        def gpi2(self, q: int, phase: float) -> Circuit: ...
        def ms(self, q0: int, q1: int, phi0: float, phi1: float, theta: float) -> Circuit: ...
        def zz(self, q0: int, q1: int, theta: float) -> Circuit: ...
        def rz(self, q: int, theta: float) -> Circuit: ...
        def id(self, q: int) -> Circuit: ...
        def x(self, q: int) -> Circuit: ...
        def y(self, q: int) -> Circuit: ...
        def z(self, q: int) -> Circuit: ...
        def h(self, q: int) -> Circuit: ...
        def s(self, q: int) -> Circuit: ...
        def sdg(self, q: int) -> Circuit: ...
        def t(self, q: int) -> Circuit: ...
        def tdg(self, q: int) -> Circuit: ...
        def sx(self, q: int) -> Circuit: ...
        def rx(self, q: int, theta: float) -> Circuit: ...
        def ry(self, q: int, theta: float) -> Circuit: ...
        def cnot(self, q0: int, q1: int) -> Circuit: ...
        def cx(self, q0: int, q1: int) -> Circuit: ...
        def cz(self, q0: int, q1: int) -> Circuit: ...
        def swap(self, q0: int, q1: int) -> Circuit: ...
        def cp(self, q0: int, q1: int, theta: float) -> Circuit: ...
        def rxx(self, q0: int, q1: int, theta: float) -> Circuit: ...
        def rzz(self, q0: int, q1: int, theta: float) -> Circuit: ...
        def u3(self, q: int, theta: float, phi: float, lam: float) -> Circuit: ...


GATE_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    "gpi": ("phase",),
    "gpi2": ("phase",),
    "ms": ("phi0", "phi1", "theta"),
    "zz": ("theta",),
    "rz": ("theta",),
    "rx": ("theta",),
    "ry": ("theta",),
    "cp": ("theta",),
    "rxx": ("theta",),
    "rzz": ("theta",),
    "u3": ("theta", "phi", "lam"),
}
"""The builder methods' parameter names in the order of ``Operation.params``; a gate absent here takes no parameter."""


def _builder_method(name: str, arity: int, n_params: int) -> Callable[..., Circuit]:
    """The builder method of one gate: qubits (``q``, or ``q0`` and ``q1``), then its ``GATE_PARAMETERS`` in radians."""
    qubit_names = ("q",) if arity == 1 else ("q0", "q1")
    param_names = GATE_PARAMETERS.get(name, ())
    if len(param_names) != n_params:
        raise ValueError(
            f"GATE_PARAMETERS names {len(param_names)} parameter(s) for {name!r}, the table {n_params}"
        )
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    signature = inspect.Signature(
        [
            inspect.Parameter("self", kind),
            *(inspect.Parameter(q, kind, annotation=int) for q in qubit_names),
            *(inspect.Parameter(p, kind, annotation=float) for p in param_names),
        ],
        return_annotation="Circuit",
    )

    def method(self: Circuit, *args: Any, **kwargs: Any) -> Circuit:
        bound = signature.bind(self, *args, **kwargs)
        qubits = tuple(int(bound.arguments[q]) for q in qubit_names)
        params = tuple(float(bound.arguments[p]) for p in param_names)
        return self._with(Operation(name, qubits, params))

    method.__dict__["__signature__"] = signature
    method.__name__ = name
    method.__qualname__ = f"Circuit.{name}"
    where = "qubit ``q``" if arity == 1 else "qubits ``(q0, q1)``, in the gate's own order"
    with_params = f" with ``{'``, ``'.join(param_names)}`` in radians" if param_names else ""
    method.__doc__ = f"Append ``{name}`` on {where}{with_params}; returns a new ``Circuit``."
    return method


for _gate, (_arity, _n_params) in {**NATIVE_GATES, **STANDARD_GATES}.items():
    setattr(Circuit, _gate, _builder_method(_gate, _arity, _n_params))


# ---- ideal matrices ------------------------------------------------------------------------------------------------


def _rx(theta: float) -> np.ndarray:
    return native.r_phi(theta, 0.0)


def _ry(theta: float) -> np.ndarray:
    return native.r_phi(theta, math.pi / 2.0)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """OpenQASM 2 U3(theta, phi, lambda) = [[cos(theta/2), -e^{i lambda} sin(theta/2)], [e^{i phi} sin(theta/2), e^{i(phi + lambda)} cos(theta/2)]]."""
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array(
        [[c, -cmath.exp(1j * lam) * s], [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


CNOT_MATRIX: Final[np.ndarray] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
"""Basis |c t>, the control the first (more significant) factor."""
SWAP_MATRIX: Final[np.ndarray] = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def cp_matrix(theta: float) -> np.ndarray:
    """CP(theta) = diag(1, 1, 1, e^{i theta})."""
    return np.diag([1.0, 1.0, 1.0, cmath.exp(1j * theta)]).astype(complex)


def gate_matrix(op: Operation) -> np.ndarray:
    """The ideal matrix of a unitary operation in the qubit order of ``op.qubits`` (first qubit = first tensor factor)."""
    n = op.name
    p = op.params
    if n == "gpi":
        return native.gpi(p[0])
    if n == "gpi2":
        return native.gpi2(p[0])
    if n == "rz":
        return native.rz(p[0])
    if n == "ms":
        return native.ms(p[0], p[1], p[2])
    if n == "zz":
        return native.zz(p[0])
    if n == "id":
        return native.IDENTITY_2.copy()
    if n == "x":
        return native.PAULI_X.copy()
    if n == "y":
        return native.PAULI_Y.copy()
    if n == "z":
        return native.PAULI_Z.copy()
    if n == "h":
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
    if n == "s":
        return np.diag([1.0, 1j]).astype(complex)
    if n == "sdg":
        return np.diag([1.0, -1j]).astype(complex)
    if n == "t":
        return np.diag([1.0, cmath.exp(1j * math.pi / 4.0)]).astype(complex)
    if n == "tdg":
        return np.diag([1.0, cmath.exp(-1j * math.pi / 4.0)]).astype(complex)
    if n == "sx":
        return 0.5 * np.array([[1.0 + 1j, 1.0 - 1j], [1.0 - 1j, 1.0 + 1j]], dtype=complex)
    if n == "rx":
        return _rx(p[0])
    if n == "ry":
        return _ry(p[0])
    if n in ("cnot", "cx"):
        return CNOT_MATRIX.copy()
    if n == "cz":
        return cp_matrix(math.pi)
    if n == "swap":
        return SWAP_MATRIX.copy()
    if n == "cp":
        return cp_matrix(p[0])
    if n == "rxx":
        return native.xx(0.5 * p[0])
    if n == "rzz":
        return native.zz(p[0])
    if n == "u3":
        return _u3(p[0], p[1], p[2])
    raise ValueError(f"{n!r} has no unitary matrix")


def embed(mat: np.ndarray, qubits: Sequence[int], n_qubits: int) -> np.ndarray:
    """``mat`` on ``qubits`` (its first factor the first listed qubit) as a 2^n x 2^n matrix; qubit 0 is the
    least-significant bit of the basis index, so qubit j is tensor axis n - 1 - j."""
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
    """The ideal unitary of the circuit's operations (or ``ops``) in time order; ValueError on a non-unitary operation."""
    dim = 2**circuit.n_qubits
    u = np.eye(dim, dtype=complex)
    for op in circuit.ops if ops is None else ops:
        if op.is_non_unitary:
            raise ValueError(f"the circuit contains a non-unitary {op.name!r}: no unitary is defined")
        u = embed(gate_matrix(op), op.qubits, circuit.n_qubits) @ u
    return u


@dataclass(frozen=True)
class CircuitCost:
    """The cost model of a native circuit: the duration and the errors the templates predict."""

    duration_s: float
    """sum over the single-qubit pulses of |theta| tau_1q/pi plus tau_2q per entangling gate."""
    n_single: int
    n_entangling: int
    errors: tuple[float, ...]
    """Per gate, in circuit order: |sin theta| eps_1q for a single-qubit rotation of area theta, |sin 2 chi| E_2q for an
    entangling gate of angle chi (theta = 2 chi in the native MS parameters)."""
    fidelity: float
    """prod_i (1 - e_i) over ``errors``, not 1 - sum e_i."""


def cost_of(
    circuit: Circuit,
    *,
    tau_1q_s: float,
    tau_2q_s: float,
    eps_1q: float,
    eps_2q: float,
) -> CircuitCost:
    """The ``CircuitCost`` of a native circuit; theta is each pulse's rotation area (pi for gpi, pi/2 for gpi2, theta = 2 chi
    for ms). A virtual rz and a non-unitary operation cost nothing; a non-native gate raises CompileError."""
    if tau_1q_s < 0.0 or tau_2q_s < 0.0 or eps_1q < 0.0 or eps_2q < 0.0:
        raise ValueError("durations and per-gate errors are non-negative")
    duration = 0.0
    errors: list[float] = []
    n_single = n_entangling = 0
    for op in circuit.ops:
        if op.is_non_unitary or op.name == "rz":
            continue
        if len(op.qubits) == 1:
            area = NATIVE_AREAS_RAD.get(op.name)
            if area is None:
                raise CompileError(
                    f"cost_of takes a native circuit; {op.name!r} is not a native single-qubit gate (compile first)"
                )
            duration += abs(area) * tau_1q_s / math.pi
            errors.append(abs(math.sin(area)) * eps_1q)
            n_single += 1
        else:
            if op.name not in ("ms", "zz"):
                raise CompileError(
                    f"cost_of takes a native circuit; {op.name!r} is not a native entangling gate"
                )
            theta = op.params[-1] if op.name == "ms" else op.params[0]
            duration += tau_2q_s
            errors.append(abs(math.sin(float(theta))) * eps_2q)
            n_entangling += 1
    fidelity = 1.0
    for e in errors:
        fidelity *= 1.0 - e
    return CircuitCost(duration, n_single, n_entangling, tuple(errors), float(fidelity))


NATIVE_AREAS_RAD: Final[dict[str, float]] = {"gpi": math.pi, "gpi2": math.pi / 2.0}
"""The rotation area of each native single-qubit pulse."""


def ideal_probabilities(circuit: Circuit) -> dict[str, float]:
    """The ideal |<b|U|0...0>|^2 over the measured qubits, keyed by bitstring with the lowest qubit rightmost."""
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


# ---- decompositions ---------------------------------------------------------------------------------------------------------


def zyz_angles(u: np.ndarray) -> tuple[float, float, float, float]:
    """(a, b, c, delta) with U = e^{i delta} RZ(a) RY(b) RZ(c), b in [0, pi] (c = 0 at b = 0, c = -a at b = pi)."""
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
        a = cmath.phase(u[1, 0]) - cmath.phase(u[0, 0])
        c = cmath.phase(u[1, 1]) - cmath.phase(u[1, 0])
    rec = native.rz(a) @ _ry(b) @ native.rz(c)
    phase = _global_phase(u, rec)
    if phase is None:
        raise CompileError("ZYZ decomposition failed to reproduce the matrix")
    return float(a), float(b), float(c), float(phase)


def _global_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-9) -> float | None:
    """alpha with a = e^{i alpha} b, or None."""
    idx = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
    if abs(b[idx]) < atol:
        return 0.0 if np.allclose(a, b, atol=atol) else None
    ratio = a[idx] / b[idx]
    if abs(abs(ratio) - 1.0) > atol:
        return None
    if not np.allclose(a, ratio * b, atol=atol):
        return None
    return float(cmath.phase(ratio))


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


PHYSICAL_RZ_AXIS_RAD = 0.0
"""The reference phase x of the physical RZ, GPi(x) GPi(x - theta/2) = RZ(theta): any x gives the same RZ."""


def physical_rz(theta_rad: float, qubit: int, *, axis_rad: float = PHYSICAL_RZ_AXIS_RAD) -> list[Operation]:
    """RZ(theta) = GPi(x) GPi(x - theta/2) exactly, as two real pulses in time order [GPi(x - theta/2), GPi(x)], for where
    the virtual-Z frame cannot carry the rotation."""
    x = float(axis_rad)
    return [
        Operation("gpi", (qubit,), (_wrap(x - 0.5 * float(theta_rad)),)),
        Operation("gpi", (qubit,), (_wrap(x),)),
    ]


def decompose_single_qubit(
    u: np.ndarray, qubit: int, *, tol: float = 1e-10, physical_z: bool = False
) -> list[Operation]:
    """Native operations (time order, virtual rz) for the 2 x 2 unitary ``u`` on ``qubit``, verified up to a global phase:
    one rz for a Z rotation, one GPi2 or GPi plus an rz for a pi/2 or pi equatorial rotation, else ZXZXZ with two GPi2;
    ``physical_z=True`` replaces every rz by :func:`physical_rz`."""
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
    if physical_z:
        physical: list[Operation] = []
        for op in ops:
            physical.extend(physical_rz(op.params[0], qubit) if op.name == "rz" else [op])
        ops = physical
    _verify_block(ops, np.asarray(u, dtype=complex), (qubit,), "single-qubit")
    return ops


def _xx_ops(chi: float, pair: tuple[int, int]) -> list[Operation]:
    """XX(chi) as one native ms: MS(0, 0, 2 chi) for chi >= 0, else MS(0, pi, -2 chi) (GPi(pi) = -X)."""
    if chi >= 0.0:
        return [Operation("ms", pair, (0.0, 0.0, 2.0 * chi))]
    return [Operation("ms", pair, (0.0, math.pi, -2.0 * chi))]


def _zz_ops(theta: float, pair: tuple[int, int], entangler: Entangler) -> list[Operation]:
    """ZZ(theta): the native zz, or GPi2(3 pi/2)^(x2), XX(theta/2), GPi2(pi/2)^(x2) in time order on an MS-only device."""
    if entangler == "zz":
        return [Operation("zz", pair, (theta,))]
    a, b = pair
    return (
        [Operation("gpi2", (a,), (1.5 * math.pi,)), Operation("gpi2", (b,), (1.5 * math.pi,))]
        + _xx_ops(0.5 * theta, pair)
        + [Operation("gpi2", (a,), (0.5 * math.pi,)), Operation("gpi2", (b,), (0.5 * math.pi,))]
    )


def cnot_template(control: int, target: int, *, s: int = 1, v: int = 1) -> list[Operation]:
    """Maslov's CNOT from one XX and four pulses, time order: RY(v pi/2)_c; XX(s pi/4); RX(-s pi/2)_c with RX(-v s pi/2)_t;
    RY(-v pi/2)_c, where RY(+-pi/2) = GPi2(+-pi/2), RX(-pi/2) = GPi2(pi), RX(pi/2) = GPi2(0). ``s`` is the template's XX
    sign; the scheduler plays either sign of the calibrated chi."""
    if s not in (1, -1) or v not in (1, -1):
        raise ValueError("s and v are signs")

    def ry(q: int, sign: int) -> Operation:
        return Operation("gpi2", (q,), (_wrap(sign * math.pi / 2.0),))

    def rx(q: int, sign: int) -> Operation:
        return Operation("gpi2", (q,), (0.0 if sign > 0 else math.pi,))

    return (
        [ry(control, v)]
        + _xx_ops(s * math.pi / 4.0, (control, target))
        + [rx(control, -s), rx(target, -v * s), ry(control, -v)]
    )


def cnot_global_phase(s: int, v: int) -> float:
    """The global phase pi v s/4 of Maslov's template: it equals e^{i pi v s/4} CNOT."""
    return math.pi * v * s / 4.0


def cp_template(theta: float, pair: tuple[int, int], entangler: Entangler) -> list[Operation]:
    """CP(theta) = [RZ(theta/2) (x) RZ(theta/2)] ZZ(-theta/2) up to a global phase (exact): time order ZZ then the rz pair."""
    th = _wrap(theta)
    a, b = pair
    return _zz_ops(-0.5 * th, pair, entangler) + [
        Operation("rz", (a,), (0.5 * th,)),
        Operation("rz", (b,), (0.5 * th,)),
    ]


def debnath_cp_template(theta: float, pair: tuple[int, int], entangler: Entangler) -> list[Operation]:
    """Debnath's Fig. 2b CP(theta) as drawn: the XX(|theta|/4) block with the fixed RZ(sgn(theta) pi/2) pair, which equals
    CP(theta) times RZ((sgn(theta) pi - theta)/2) on both qubits; exact for CZ only."""
    th = _wrap(theta)
    a, b = pair
    half = math.copysign(math.pi / 2.0, th)
    return _zz_ops(-0.5 * th, pair, entangler) + [
        Operation("rz", (a,), (half,)),
        Operation("rz", (b,), (half,)),
    ]


def cp_template_local_defect_rad(theta: float) -> float:
    """The RZ angle (sgn(theta) pi - theta)/2 on both qubits that Debnath's template carries against CP(theta)."""
    th = _wrap(theta)
    return 0.5 * (math.copysign(math.pi, th) - th)


def cp_template_overlap(theta: float) -> float:
    """|Tr(CP^dagger T)|/4 of Debnath's template T against CP(theta): cos^2((sgn(theta) pi - theta)/4), 0.854 at pi/2, 0.691 at pi/4."""
    x = cp_template_local_defect_rad(theta)
    return math.cos(0.5 * x) ** 2


def decompose_two_qubit(op: Operation, *, entangler: Entangler, s: int = 1, v: int = 1) -> list[Operation]:
    """Native operations for a standard two-qubit gate; verified against the target up to a global phase."""
    a, b = op.qubits
    pair = (a, b)
    name = op.name
    if name in ("cnot", "cx"):
        if entangler == "ms":
            ops = cnot_template(a, b, s=s, v=v)
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
            decompose_two_qubit(Operation("cnot", (a, b), ()), entangler=entangler, s=s, v=v)
            + decompose_two_qubit(Operation("cnot", (b, a), ()), entangler=entangler, s=s, v=v)
            + decompose_two_qubit(Operation("cnot", (a, b), ()), entangler=entangler, s=s, v=v)
        )
    elif name == "rxx":
        ops = _xx_ops(0.5 * op.params[0], pair)
    elif name == "rzz":
        ops = _zz_ops(op.params[0], pair, entangler)
    else:
        raise ValueError(f"{name!r} is not a standard two-qubit gate")
    _verify_block(ops, gate_matrix(op), pair, name)
    return ops


def _verify_block(ops: Sequence[Operation], target: np.ndarray, qubits: tuple[int, ...], label: str) -> float:
    """The block's residual against the target up to a global phase; CompileError beyond ``BLOCK_TOLERANCE``."""
    local = {q: k for k, q in enumerate(qubits)}
    n = len(qubits)
    u = np.eye(2**n, dtype=complex)
    for op in ops:
        mat = gate_matrix(op)
        u = embed(mat, [local[q] for q in op.qubits], n) @ u
    want = embed(target, list(range(n)), n)  # the same bit order as u (qubit 0 least significant)
    phase = _global_phase(u, want, atol=BLOCK_TOLERANCE)
    if phase is None:
        raise CompileError(f"the {label} template does not reproduce its target matrix up to a global phase")
    return float(np.max(np.abs(u - cmath.exp(1j * phase) * want)))


# ---- frame propagation and the compiler entry point -------------------------------------------------------------------------


def propagate_frames(ops: Sequence[Operation], n_qubits: int) -> tuple[list[Operation], dict[int, float]]:
    """Absorb every rz into the phases of the later pulses (phi -> phi - theta, time order); return the rz-free
    operations and the residual per-qubit frame, which a computational-basis measurement discards."""
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
    """max |RZ(frame) U_compiled - e^{i alpha} U_target| over the whole circuit (None when a mid-circuit operation prevents it)."""
    entangler: Entangler
    notes: tuple[str, ...] = field(default_factory=tuple)


def compile_report(
    circuit: Circuit,
    device: Device | None = None,
    *,
    entangler: Entangler = "ms",
    cnot_signs: tuple[int, int] = (1, 1),
    verify_circuit: bool = True,
) -> CompileReport:
    """Compile to native gates with phase tracking, verifying every block and the whole circuit (up to 10 qubits, with no
    mid-circuit operation); native circuits pass through with their rz absorbed. ``device`` is unused (the templates depend
    on ``entangler`` only); ``cnot_signs`` = (s, v) of Maslov's template."""
    s, v = cnot_signs
    native_ops: list[Operation] = []
    residuals: list[float] = []
    notes: list[str] = []
    for op in circuit.ops:
        if op.is_non_unitary or op.name in ("gpi", "gpi2", "ms", "zz", "rz"):
            native_ops.append(op)
            continue
        if op.name not in STANDARD_GATES:
            raise CompileError(f"unknown gate {op.name!r}")
        if len(op.qubits) == 1:
            block = decompose_single_qubit(gate_matrix(op), op.qubits[0])
        else:
            block = decompose_two_qubit(op, entangler=entangler, s=s, v=v)
        residuals.append(_verify_block(block, gate_matrix(op), op.qubits, op.name))
        native_ops.extend(block)
    propagated, frame = propagate_frames(native_ops, circuit.n_qubits)
    compiled = Circuit(circuit.n_qubits, tuple(propagated), circuit.measure, circuit.registers)
    circuit_residual: float | None = None
    has_mid = any(op.is_non_unitary for op in circuit.ops)
    if verify_circuit and not has_mid and circuit.n_qubits <= 10:
        target = circuit_unitary(circuit)
        got = frame_unitary(frame, circuit.n_qubits) @ circuit_unitary(compiled)
        phase = _global_phase(got, target, atol=1e-8)
        if phase is None:
            raise CompileError(
                "the compiled circuit does not reproduce the target unitary up to the residual frame"
            )
        circuit_residual = float(np.max(np.abs(got - cmath.exp(1j * phase) * target)))
    elif has_mid:
        notes.append(
            "mid-circuit non-unitary operation: blocks verified, the whole-circuit unitary is undefined"
        )
    elif verify_circuit:
        notes.append("more than 10 qubits: the whole-circuit matrix check is skipped, blocks verified")
    n_pulses = sum(1 for op in propagated if op.name in ("gpi", "gpi2")) + sum(
        1 for op in propagated if op.name in ("ms", "zz")
    )
    n_ent = sum(1 for op in propagated if op.name in ("ms", "zz"))
    return CompileReport(
        circuit=compiled,
        final_frame_rad=frame,
        n_pulses=n_pulses,
        n_entangling=n_ent,
        block_residuals=tuple(residuals),
        circuit_residual=circuit_residual,
        entangler=entangler,
        notes=tuple(notes),
    )


def compile_to_native(circuit: Circuit, device: Device | None = None, **kwargs: object) -> Circuit:
    """The compiled circuit of :func:`compile_report`."""
    return compile_report(circuit, device, **kwargs).circuit  # type: ignore[arg-type]


__all__ = [
    "BLOCK_TOLERANCE",
    "CNOT_MATRIX",
    "GATE_PARAMETERS",
    "PHYSICAL_RZ_AXIS_RAD",
    "EXPORTED_NATIVE",
    "NATIVE_AREAS_RAD",
    "NATIVE_GATES",
    "NON_UNITARY",
    "STANDARD_GATES",
    "SWAP_MATRIX",
    "Circuit",
    "CompileError",
    "CircuitCost",
    "CompileReport",
    "Entangler",
    "Operation",
    "circuit_unitary",
    "cnot_global_phase",
    "cnot_template",
    "compile_report",
    "compile_to_native",
    "cost_of",
    "cp_matrix",
    "cp_template",
    "cp_template_local_defect_rad",
    "cp_template_overlap",
    "debnath_cp_template",
    "decompose_single_qubit",
    "decompose_two_qubit",
    "embed",
    "frame_unitary",
    "gate_matrix",
    "ideal_probabilities",
    "physical_rz",
    "propagate_frames",
    "zyz_angles",
]
