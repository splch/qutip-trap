"""Arbitrary two-qubit unitaries as native gates: the KAK (Cartan) decomposition in the magic basis (PLAN.md Section 7.2).

Every U in U(4) is, up to a global phase, U = (A (x) B) exp[-i (a XX + b YY + c ZZ)] (C (x) D) with A, B, C, D in SU(2) and
(a, b, c) in the Weyl chamber pi/4 >= a >= b >= |c| >= 0 (Khaneja-Glaser; Kraus-Cirac 2001). In the magic basis M the local
group is SO(4) and the canonical exponential is diagonal, so Q = M^dag U M = O_1 D O_2^T with O_1, O_2 real orthogonal:
O_2 and D^2 diagonalize the complex-symmetric Q^T Q, O_1 = Q O_2 D^-1, the determinants are fixed to +1, and (a, b, c)
follow from the four phases of D through the +-1 table of XX, YY and ZZ on the Bell states.

The canonical part is emitted as verified standard gates: exp(-i a XX) = rxx(2a); exp(-i b YY) = (S (x) S) rxx(2b)
(S^dag (x) S^dag), in time order sdg, sdg, rxx(2b), s, s; exp(-i c ZZ) = rzz(2c). In the chamber every rxx angle is at
most pi/2, so a generic SU(4) costs three entangling gates, the CNOT class one and a local unitary none.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from qutip_trap.control import native
from qutip_trap.control.compiler import (
    CompileError,
    Operation,
    decompose_single_qubit,
    gate_matrix,
    verify_operations,
)

SQRT_HALF: Final[float] = 1.0 / math.sqrt(2.0)
MAGIC: Final[np.ndarray] = (
    np.array(
        [[1.0, 0.0, 0.0, 1.0j], [0.0, 1.0j, 1.0, 0.0], [0.0, 1.0j, -1.0, 0.0], [1.0, 0.0, 0.0, -1.0j]],
        dtype=complex,
    )
    * SQRT_HALF
)
"""The magic (Bell) basis as columns: M^dag (A (x) B) M is real orthogonal for A, B in SU(2)."""

_XX: Final[np.ndarray] = np.kron(native.PAULI_X, native.PAULI_X)
_YY: Final[np.ndarray] = np.kron(native.PAULI_Y, native.PAULI_Y)
_ZZ: Final[np.ndarray] = np.kron(native.PAULI_Z, native.PAULI_Z)
_I4: Final[np.ndarray] = np.eye(4, dtype=complex)

# the +-1 eigenvalues of XX, YY, ZZ on the four magic vectors: angle_k(D) = phase - sum_j T[k, j] coef_j
_T: Final[np.ndarray] = np.array(
    [[float(np.real(MAGIC[:, k].conj() @ p @ MAGIC[:, k])) for p in (_XX, _YY, _ZZ)] for k in range(4)]
)
_SOLVE: Final[np.ndarray] = np.linalg.inv(np.hstack([-_T, np.ones((4, 1))]))

_S: Final[np.ndarray] = gate_matrix(Operation("s", (0,), ()))
_H: Final[np.ndarray] = gate_matrix(Operation("h", (0,), ()))
_RX_HALF: Final[np.ndarray] = native.r_phi(math.pi / 2.0, 0.0)
"""exp(-i pi/4 X): conjugation maps Y -> Z and Z -> -Y, so (R (x) R) swaps the YY and ZZ coefficients."""

_DIAGONALIZATION_MIXES: Final[tuple[float, ...]] = (0.7132, 0.3117, 1.9143, 0.1289, 2.7731)


def canonical_unitary(a: float, b: float, c: float) -> np.ndarray:
    """exp[-i (a XX + b YY + c ZZ)] (the three terms commute), first qubit the first tensor factor."""
    out = _I4.copy()
    for coef, p in ((a, _XX), (b, _YY), (c, _ZZ)):
        out = (math.cos(coef) * _I4 - 1j * math.sin(coef) * p) @ out
    return out


def kron_factor(u: np.ndarray, *, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray] | None:
    """(A, B) with u = A (x) B up to a global phase, both unitary, or None when ``u`` is not a tensor product: the matrix
    reshaped as R[(i, k), (j, l)] = u[(i, j), (k, l)] has rank one exactly for a product A[i, k] B[j, l]."""
    m = np.asarray(u, dtype=complex)
    if m.shape != (4, 4):
        raise ValueError("a 4 x 4 matrix")
    r = m.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    uu, s, vh = np.linalg.svd(r)
    if s[0] <= 0.0 or s[1] > tol * s[0]:
        return None
    a = math.sqrt(s[0]) * uu[:, 0].reshape(2, 2)
    b = math.sqrt(s[0]) * vh[0, :].reshape(2, 2)
    scale = math.sqrt(abs(np.linalg.det(a)))
    if scale == 0.0:
        return None
    return a / scale, b * scale


def global_phase(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-9) -> float | None:
    """alpha with a = e^{i alpha} b, or None (``native.global_phase``)."""
    return native.global_phase(a, b, atol=atol)


@dataclass(frozen=True)
class KAK:
    """u = e^{i phase} (A (x) B) exp[-i (a XX + b YY + c ZZ)] (C (x) D) with (a, b, c) in the Weyl chamber."""

    left: tuple[np.ndarray, np.ndarray]
    """(A, B), applied after the canonical part (last in time)."""
    right: tuple[np.ndarray, np.ndarray]
    """(C, D), applied before it (first in time)."""
    coefficients: tuple[float, float, float]
    phase: float

    def matrix(self) -> np.ndarray:
        a, b, c = self.coefficients
        return np.asarray(
            cmath.exp(1j * self.phase)
            * np.kron(*self.left)
            @ canonical_unitary(a, b, c)
            @ np.kron(*self.right)
        )

    @property
    def entangling_count(self) -> int:
        """The Moelmer-Soerensen gates the canonical class costs: one per non-zero coefficient."""
        return sum(1 for x in self.coefficients if abs(x) > 1e-12)


def _wrap_quarter(x: float) -> tuple[float, int]:
    """(x - k pi/2, k) with the result in (-pi/4, pi/4]."""
    k = math.floor((x + math.pi / 4.0) / (math.pi / 2.0))
    y = x - k * math.pi / 2.0
    if y <= -math.pi / 4.0:  # round-off at the lower edge
        y += math.pi / 2.0
        k -= 1
    return y, int(k)


def _bidiagonalize(q: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q = O1 diag(lam) O2^T with O1, O2 real special orthogonal and |lam| = 1."""
    s = q.T @ q
    o2: np.ndarray | None = None
    for mix in _DIAGONALIZATION_MIXES:
        _w, cand = np.linalg.eigh(s.real + mix * s.imag)
        d2 = cand.T @ s @ cand
        if np.max(np.abs(d2 - np.diag(np.diag(d2)))) < tol:
            o2 = cand
            break
    if o2 is None:
        raise CompileError(
            "KAK: the magic-basis Gram matrix could not be diagonalized by a real orthogonal matrix"
        )
    lam2 = np.diag(o2.T @ s @ o2)
    if np.max(np.abs(np.abs(lam2) - 1.0)) > 1e3 * tol:
        raise CompileError("KAK: the input is not unitary")
    lam = np.exp(0.5j * np.angle(lam2))
    o1 = q @ o2 @ np.diag(1.0 / lam)
    if np.max(np.abs(o1.imag)) > 1e3 * tol:
        raise CompileError("KAK: the left factor is not real orthogonal (branch failure)")
    o1 = o1.real
    if np.max(np.abs(o1.T @ o1 - np.eye(4))) > 1e3 * tol:
        raise CompileError("KAK: the left factor is not orthogonal")
    if np.linalg.det(o2) < 0.0:
        o2 = o2.copy()
        o1 = o1.copy()
        o2[:, 0] *= -1.0
        o1[:, 0] *= -1.0
    if np.linalg.det(o1) < 0.0:
        o1 = o1.copy()
        lam = lam.copy()
        o1[:, 0] *= -1.0
        lam[0] *= -1.0
    return o1, lam, o2


def kak_decomposition(u: np.ndarray, *, tol: float = 1e-9) -> KAK:
    """The KAK decomposition of a 4 x 4 unitary with the canonical class in the Weyl chamber; verified to ``tol``."""
    m = np.asarray(u, dtype=complex)
    if m.shape != (4, 4):
        raise ValueError("a 4 x 4 matrix")
    if np.max(np.abs(m.conj().T @ m - _I4)) > 1e3 * tol:
        raise CompileError("KAK: not a unitary matrix")
    q = MAGIC.conj().T @ m @ MAGIC
    o1, lam, o2 = _bidiagonalize(q, tol)
    left = kron_factor(MAGIC @ o1 @ MAGIC.conj().T, tol=1e3 * tol)
    right = kron_factor(MAGIC @ o2.T @ MAGIC.conj().T, tol=1e3 * tol)
    if left is None or right is None:
        raise CompileError("KAK: a special orthogonal factor did not map to a local unitary")
    a, b, c, _phi = (_SOLVE @ np.angle(lam)).tolist()
    left_a, left_b = left
    right_c, right_d = right
    coefs = [float(a), float(b), float(c)]
    paulis = (native.PAULI_X, native.PAULI_Y, native.PAULI_Z)
    # 1. each coefficient into (-pi/4, pi/4]: K(x + pi/2 ...) = -i (P (x) P) K(x ...), absorbed into the left factor
    for j in range(3):
        coefs[j], k = _wrap_quarter(coefs[j])
        if k % 2 == 1 or k % 2 == -1:
            left_a = left_a @ paulis[j]
            left_b = left_b @ paulis[j]
        # k even (k = 2n): K(x + n pi) = (-1)^n K(x), a global phase only

    def conjugate(w: np.ndarray, v: np.ndarray, perm: tuple[int, int, int]) -> None:
        """K(c) = (W (x) V)^dag K(c') (W (x) V) with c' = c permuted: left <- left (W^dag (x) V^dag), right <- (W (x) V) right."""
        nonlocal left_a, left_b, right_c, right_d
        left_a = left_a @ w.conj().T
        left_b = left_b @ v.conj().T
        right_c = w @ right_c
        right_d = v @ right_d
        coefs[:] = [coefs[perm[0]], coefs[perm[1]], coefs[perm[2]]]

    # 2. sort |a| >= |b| >= |c| by the local Cliffords that permute the Pauli axes
    #    (S (x) S): XX <-> YY;  (H (x) H): XX <-> ZZ;  (R_x(pi/2) (x) R_x(pi/2)): YY <-> ZZ
    swaps = {(0, 1): (_S, (1, 0, 2)), (0, 2): (_H, (2, 1, 0)), (1, 2): (_RX_HALF, (0, 2, 1))}
    for _ in range(3):
        for i, j in ((0, 1), (1, 2)):
            if abs(coefs[j]) > abs(coefs[i]) + 1e-15:
                w, perm = swaps[(i, j)]
                conjugate(w, w, perm)
    # 3. a >= 0 and b >= 0 by conjugation with one Pauli on the FIRST qubit, which negates two coefficients:
    #    (Z (x) 1) negates (a, b); (Y (x) 1) negates (a, c); (X (x) 1) negates (b, c)
    if coefs[0] < -1e-15 and coefs[1] < -1e-15:
        conjugate(native.PAULI_Z, np.eye(2, dtype=complex), (0, 1, 2))
        coefs[0], coefs[1] = -coefs[0], -coefs[1]
    elif coefs[0] < -1e-15:
        conjugate(native.PAULI_Y, np.eye(2, dtype=complex), (0, 1, 2))
        coefs[0], coefs[2] = -coefs[0], -coefs[2]
    elif coefs[1] < -1e-15:
        conjugate(native.PAULI_X, np.eye(2, dtype=complex), (0, 1, 2))
        coefs[1], coefs[2] = -coefs[1], -coefs[2]
    coefs = [0.0 if abs(x) < 1e-13 else x for x in coefs]
    kak = KAK(
        left=(left_a, left_b),
        right=(right_c, right_d),
        coefficients=(coefs[0], coefs[1], coefs[2]),
        phase=0.0,
    )
    phase = native.global_phase(m, kak.matrix(), atol=1e3 * tol)
    if phase is None:
        raise CompileError("KAK: the decomposition does not reproduce its target up to a global phase")
    return KAK(kak.left, kak.right, kak.coefficients, float(phase))


def canonical_operations(a: float, b: float, c: float, pair: tuple[int, int]) -> list[Operation]:
    """Standard-gate operations (time order) for exp[-i (a XX + b YY + c ZZ)] on ``pair``: rxx(2a); sdg, sdg, rxx(2b), s, s;
    rzz(2c); zero coefficients emit nothing."""
    q0, q1 = pair
    ops: list[Operation] = []
    if abs(a) > 1e-13:
        ops.append(Operation("rxx", pair, (2.0 * a,)))
    if abs(b) > 1e-13:
        ops += [
            Operation("sdg", (q0,), ()),
            Operation("sdg", (q1,), ()),
            Operation("rxx", pair, (2.0 * b,)),
            Operation("s", (q0,), ()),
            Operation("s", (q1,), ()),
        ]
    if abs(c) > 1e-13:
        ops.append(Operation("rzz", pair, (2.0 * c,)))
    return ops


def decompose_two_qubit_unitary(
    u: np.ndarray, pair: tuple[int, int], *, tol: float = 1e-9
) -> list[Operation]:
    """Native single-qubit operations and standard two-qubit gates (time order) for the 4 x 4 unitary ``u`` on ``pair`` (its
    first factor the first listed qubit); verified against ``u`` up to a global phase."""
    kak = kak_decomposition(u, tol=tol)
    q0, q1 = pair
    ops: list[Operation] = []
    ops += decompose_single_qubit(kak.right[0], q0)
    ops += decompose_single_qubit(kak.right[1], q1)
    ops += canonical_operations(*kak.coefficients, pair)
    ops += decompose_single_qubit(kak.left[0], q0)
    ops += decompose_single_qubit(kak.left[1], q1)
    verify_operations(ops, np.asarray(u, dtype=complex), pair, tol=1e3 * tol)
    return ops


def haar_random_unitary(rng: np.random.Generator, dim: int) -> np.ndarray:
    """A Haar-random unitary from the QR decomposition of a complex Ginibre matrix with the phases of R's diagonal removed
    (Mezzadri 2007)."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) * SQRT_HALF
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return np.asarray(q * (d / np.abs(d)))
