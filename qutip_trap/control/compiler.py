"""Circuit IR and the compiler (PLAN.md Sections 3.3, 7.1, 7.2, 7.6, 7.7; Section 13 rows "Gate parameters", "Operator order
in templates", "Virtual-Z propagation"; Appendix E; milestone M6).

The IR is a list of operations (name, qubits, params) supporting the native set (gpi, gpi2, ms, zz and the virtual rz)
plus the standard set of Section 7.2 (x, y, z, h, s, sdg, t, tdg, rx, ry, rz, cnot/cx, cz, swap, cp, u3, and the
two-qubit rotations rxx, rzz the OpenQASM exporters of the client SDKs use) and the non-unitary ``measure``, ``reset`` and
``recool`` (Section 7.2 item 4: schedulable from the first release; the scheduler refuses mid-circuit measurement with a
clear error until Section 8.5 is implemented). Parameters are RADIANS; IonQ's turns are converted at the boundary
(``qutip_trap.io.ionq``).

Compilation (Section 7.2), the ONE place ideal gate matrices are used (Section 3.1): every standard gate becomes native
gates plus virtual RZ frame updates, each block is verified numerically against its target matrix up to a global phase
before it is accepted, and the frame updates are then propagated into the phases of every later pulse (phi -> phi - theta
in time order, Section 7.6) so that the compiled circuit carries only gpi, gpi2, ms and zz, the exported native set, and a
residual per-qubit Z frame that the computational-basis measurement discards (``CompileReport.final_frame_rad``).

- Single-qubit gates: the ZYZ Euler angles of the target give one virtual RZ (a pure Z rotation), one GPi or GPi2 pulse
  (a pi or pi/2 equatorial rotation), or otherwise the ZXZXZ form RZ(alpha) GPi2(0) RZ(beta) GPi2(0) RZ(gamma), complete
  because GPi2(0) RZ(beta) GPi2(0) = RY(-beta) RX(pi) up to a phase, so (alpha, -beta, -gamma) are the ZYZ angles of U X
  (Section 7.2 item 1; the 20-random-target check of that item is a test).
- CNOT: Maslov's one-XX template in time order RY(v pi/2)_c, XX(s pi/4), RX(-s pi/2)_c RX(-v s pi/2)_t, RY(-v pi/2)_c, equal
  to CNOT up to the global phase e^{i pi v s/4} for all four signs (Section 7.7). ``s`` is the template's sign convention:
  the scheduler realizes either sign from the pair's calibrated waveform by a pi on the second ion's tones (Section 4.4.2),
  so the physical sign of chi never reaches the compiler.
- Controlled phase CP(theta) = RZ(theta/2) (x) RZ(theta/2) . ZZ(-theta/2) up to a global phase, exact; ZZ is emitted as the
  native zz gate or expanded into the wrapper W XX(-theta/4) W_in with W = GPi2(pi/2)^(x2), W_in = GPi2(3 pi/2)^(x2)
  (the M4 identity), depending on ``entangler``. Debnath's Fig. 2b template (``debnath_cp_template``: the same partially
  entangling XX with the FIXED RZ(sgn(theta) pi/2) pair that is right for CZ only) is diagonal with the right conditional
  phase but carries an extra RZ((sgn(theta) pi - theta)/2) on both qubits, overlap cos^2((sgn(theta) pi - theta)/4) with
  CP(theta): 0.854 at pi/2 and 0.691 at pi/4 (Section 7.7), which appending RZ((theta - sgn(theta) pi)/2) on both fixes.
- CZ = CP(pi); SWAP = three CNOTs; rxx(theta) = XX(theta/2) = MS(0, 0, theta); rzz(theta) = ZZ(theta).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from qutip_trap.control import native

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

# name -> (number of qubits, number of parameters)
NATIVE_GATES: Final[dict[str, tuple[int, int]]] = {
    "gpi": (1, 1),
    "gpi2": (1, 1),
    "ms": (2, 3),
    "zz": (2, 1),
    "rz": (1, 1),  # virtual: no pulse, a frame update (Section 7.1)
}
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
NON_UNITARY: Final[frozenset[str]] = frozenset({"measure", "reset", "recool"})
EXPORTED_NATIVE: Final[frozenset[str]] = frozenset({"gpi", "gpi2", "ms", "zz"})
"""The native gates the IonQ JSON exporter carries (Section 7.2 item 1); rz is absorbed by the compiler."""

BLOCK_TOLERANCE: Final[float] = 1e-9
"""Residual (max |U_block - e^{i alpha} U_target|) above which a compiled block is refused (Section 7.2 item 3)."""

Entangler = Literal["ms", "zz"]


class CompileError(ValueError):
    """A compiled block failed its numerical verification, or the circuit cannot be compiled."""


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

    @property
    def is_exported_native(self) -> bool:
        return self.name in EXPORTED_NATIVE


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


# ---- ideal matrices: the compiler's definition of what a gate is supposed to do (Section 3.1) ----------------------------


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
    """``mat`` on ``qubits`` (its first factor the first listed qubit) as a 2^n x 2^n matrix in the Section 13 bit order:
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
    """The ideal unitary of the circuit's unitary operations in time order (the target the compiler verifies against).

    Non-unitary operations are refused: the unitary of a circuit with a mid-circuit measurement is not defined.
    """
    dim = 2**circuit.n_qubits
    u = np.eye(dim, dtype=complex)
    for op in circuit.ops if ops is None else ops:
        if op.is_non_unitary:
            raise ValueError(f"the circuit contains a non-unitary {op.name!r}: no unitary is defined")
        u = embed(gate_matrix(op), op.qubits, circuit.n_qubits) @ u
    return u


def ideal_probabilities(circuit: Circuit) -> dict[str, float]:
    """The compiler's target distribution (Section 14.5: shown beside the simulated one, never in its place): |<b|U|0...0>|^2 keyed
    by the Section 13 bitstring (qubit 0 rightmost), over the measured qubits only when ``circuit.measure`` is a subset."""
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
    # verify; the global phase comes out of the comparison
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


def decompose_single_qubit(u: np.ndarray, qubit: int, *, tol: float = 1e-10) -> list[Operation]:
    """Native operations (time order, with virtual rz) for the 2 x 2 unitary ``u`` on ``qubit`` (Section 7.2 item 1).

    A pure Z rotation is one rz; a pi/2 or pi equatorial rotation is one GPi2 or GPi pulse plus an rz; anything else is the
    ZXZXZ form with two GPi2 pulses. Verified against ``u`` up to a global phase before it is returned.
    """
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
    _verify_block(ops, np.asarray(u, dtype=complex), (qubit,), "single-qubit")
    return ops


def _xx_ops(chi: float, pair: tuple[int, int]) -> list[Operation]:
    """XX(chi) = exp(-i chi sigma_x sigma_x) as one native ms: MS(0, 0, 2 chi) for chi >= 0, MS(0, pi, -2 chi) for chi < 0
    (GPi(pi) = -X flips the generator's sign; the exported angle stays in [0, pi/2])."""
    if chi >= 0.0:
        return [Operation("ms", pair, (0.0, 0.0, 2.0 * chi))]
    return [Operation("ms", pair, (0.0, math.pi, -2.0 * chi))]


def _zz_ops(theta: float, pair: tuple[int, int], entangler: Entangler) -> list[Operation]:
    """ZZ(theta) = exp(-i (theta/2) Z Z): the native zz, or the wrapper W XX(theta/2) W_in on an MS-only device (M4 identity)."""
    if entangler == "zz":
        return [Operation("zz", pair, (theta,))]
    a, b = pair
    return (
        [Operation("gpi2", (a,), (1.5 * math.pi,)), Operation("gpi2", (b,), (1.5 * math.pi,))]
        + _xx_ops(0.5 * theta, pair)
        + [Operation("gpi2", (a,), (0.5 * math.pi,)), Operation("gpi2", (b,), (0.5 * math.pi,))]
    )


def cnot_template(control: int, target: int, *, s: int = 1, v: int = 1) -> list[Operation]:
    """Maslov's CNOT from one XX and four pulses, time order (Section 7.7): RY(v pi/2)_c; XX(s pi/4); RX(-s pi/2)_c with
    RX(-v s pi/2)_t; RY(-v pi/2)_c. RY(+-pi/2) = GPi2(+-pi/2), RX(-pi/2) = GPi2(pi), RX(pi/2) = GPi2(0)."""
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


def cnot_global_phase(s: int, v: int) -> float:
    """The template equals e^{i pi v s/4} CNOT (recomputed in Section 7.7; the source prints (-1)^{-vs/4})."""
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
    """Debnath's CP(theta) as drawn (Section 7.7): the partially entangling XX(|theta|/4) block with the fixed RZ(sgn(theta) pi/2)
    pair, which equals CP(theta) times RZ((sgn(theta) pi - theta)/2) on both qubits; exact for CZ only."""
    th = _wrap(theta)
    a, b = pair
    half = math.copysign(math.pi / 2.0, th)
    return _zz_ops(-0.5 * th, pair, entangler) + [
        Operation("rz", (a,), (half,)),
        Operation("rz", (b,), (half,)),
    ]


def cp_template_local_defect_rad(theta: float) -> float:
    """The RZ angle (sgn(theta) pi - theta)/2 on both qubits that Debnath's template carries against CP(theta) (Section 7.7)."""
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
    """Section 7.2 item 3: the block's matrix equals the target up to a global phase, else the compiler refuses."""
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
    """Absorb every rz into the phases of the later pulses (phi -> phi - theta, time order; Section 7.6) and return the
    rz-free operations with the residual per-qubit frame, which a computational-basis measurement discards."""
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
    """What the compiler did and verified (Section 7.2)."""

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


def compile_with_report(
    circuit: Circuit,
    device: Device | None = None,
    *,
    entangler: Entangler = "ms",
    cnot_signs: tuple[int, int] = (1, 1),
    verify_circuit: bool = True,
) -> CompileReport:
    """Standard gates -> native gates with phase tracking, every block and the whole circuit verified (Section 7.2).

    ``device`` is accepted for the Appendix E signature (the template choice depends on the entangler, not on the device's
    hidden values); ``cnot_signs`` = (s, v) of Maslov's template. Native circuits pass through with their rz absorbed.
    """
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
    compiled = Circuit(circuit.n_qubits, tuple(propagated), circuit.measure)
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
    """Standard gates -> native gates with phase tracking, verified against target unitaries (Section 7.2; Appendix E)."""
    return compile_with_report(circuit, device, **kwargs).circuit  # type: ignore[arg-type]


__all__ = [
    "BLOCK_TOLERANCE",
    "CNOT_MATRIX",
    "EXPORTED_NATIVE",
    "NATIVE_GATES",
    "NON_UNITARY",
    "STANDARD_GATES",
    "SWAP_MATRIX",
    "Circuit",
    "CompileError",
    "CompileReport",
    "Entangler",
    "Operation",
    "circuit_unitary",
    "cnot_global_phase",
    "cnot_template",
    "compile_to_native",
    "compile_with_report",
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
    "propagate_frames",
    "zyz_angles",
]
