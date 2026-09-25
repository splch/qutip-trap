"""The Clifford groups behind randomized benchmarking (PLAN.md Section 7.9): the 24- and 11520-element groups by closure,
the four entangling classes and their sizes from the stabilizers, uniform sampling by the double-coset construction, the
class recognition of an arbitrary two-qubit Clifford, and the closure of a random sequence by the inverse of its product.
The 11520-element group, the stabilizers and the symplectic class are oracles computed here, independently of the
sampler and the recognizer they check."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np
import pytest

from qutip_trap.benchmarks.clifford import (
    _LOCALS,
    CLASS_SIZES,
    CORES,
    ISWAP_MATRIX,
    SINGLE_QUBIT_CLIFFORDS,
    TWO_QUBIT_GROUP_ORDER,
    CoreName,
    TwoQubitClifford,
    core_operations,
    decompose_two_qubit_clifford,
    group_closure,
    matrix_key,
    random_two_qubit_clifford,
    single_qubit_sequence_unitary,
)
from qutip_trap.benchmarks.rb import single_qubit_sequences, two_qubit_sequence
from qutip_trap.control import native
from qutip_trap.control.compiler import (
    CNOT_MATRIX,
    Circuit,
    Operation,
    circuit_unitary,
    compile_report,
    decompose_single_qubit,
)
from qutip_trap.control.two_qubit import global_phase

ENTANGLING_COUNT: dict[CoreName, int] = {"identity": 0, "cnot": 1, "iswap": 2, "swap": 3}
"""The Moelmer-Soerensen gates each core compiles to."""
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
_S = np.diag([1.0, 1.0j]).astype(complex)
_I2 = np.eye(2, dtype=complex)


@functools.cache
def two_qubit_clifford_group() -> list[np.ndarray]:
    """The 11520 two-qubit Cliffords modulo global phase: the closure of {H (x) 1, 1 (x) H, S (x) 1, 1 (x) S, CNOT}."""
    return group_closure(
        [np.kron(_H, _I2), np.kron(_I2, _H), np.kron(_S, _I2), np.kron(_I2, _S), CNOT_MATRIX]
    )


def stabilizer_size(core: CoreName) -> int:
    """|L intersect g L g^-1|, the local l with g l g^-1 local, by enumeration; 576^2 over it is the class size."""
    g = CORES[core]
    conj = np.einsum("ij,njk,kl->nil", g, _LOCALS, g.conj().T)
    s = np.linalg.svd(
        conj.reshape(-1, 2, 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 4, 4), compute_uv=False
    )
    return int(np.count_nonzero(s[:, 1] <= 1e-9 * s[:, 0]))


_PAULI_LABELS = tuple((a, b) for a in "IXYZ" for b in "IXYZ")
_PAULI = {"I": _I2, "X": native.PAULI_X, "Y": native.PAULI_Y, "Z": native.PAULI_Z}
_PAULI_PAIRS = np.array([np.kron(_PAULI[a], _PAULI[b]) for a, b in _PAULI_LABELS])


def pauli_frame_of(u: np.ndarray) -> CoreName:
    """The symplectic class of a two-qubit Clifford by its action on the Pauli generators: 'identity' (block diagonal),
    'cnot' (off-diagonal block of rank 1), 'iswap' (rank 2, not anti-block-diagonal) or 'swap' (anti-block-diagonal)."""
    m = np.asarray(u, dtype=complex)

    def image(p: np.ndarray) -> tuple[str, str]:
        # the two-qubit Pauli P with |Tr(P^dag Q)|/4 = 1 for Q = m p m^dag
        overlaps = np.abs(np.einsum("nji,ij->n", _PAULI_PAIRS.conj(), m @ p @ m.conj().T)) / 4.0
        k = int(np.argmax(overlaps))
        assert overlaps[k] > 1.0 - 1e-7, "a Pauli maps outside the Pauli group"
        return _PAULI_LABELS[k]

    gens = [np.kron(native.PAULI_X, _I2), np.kron(native.PAULI_Z, _I2)]
    bits = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
    rows = [image(g) for g in gens]  # the images of X1, Z1
    a_block = np.array([bits[r[0]] for r in rows])  # (X1, Z1) -> qubit-1 (x, z) bits
    rank_b = int(np.linalg.matrix_rank(np.array([bits[r[1]] for r in rows], dtype=float)))  # -> qubit-2 bits
    if rank_b == 0:
        return "identity"
    if rank_b == 1:
        return "cnot"
    return "swap" if not a_block.any() else "iswap"


def operations_unitary(ops: Sequence[Operation], qubits: tuple[int, ...]) -> np.ndarray:
    """The ideal unitary of ``ops`` on ``qubits``, the first listed qubit the first tensor factor."""
    local = {q: k for k, q in enumerate(qubits)}
    n = len(qubits)
    relabelled = tuple(Operation(op.name, tuple(local[q] for q in op.qubits), op.params) for op in ops)
    u_lsb = circuit_unitary(Circuit(n, relabelled, tuple(range(n))))
    # circuit_unitary makes qubit 0 the least-significant bit; undo the embedding to first-factor order
    perm = np.array([int("".join(reversed(format(i, f"0{n}b"))), 2) for i in range(2**n)])
    return np.asarray(u_lsb[np.ix_(perm, perm)])


def test_single_qubit_group_has_24_elements_closed_under_products_and_inverses() -> None:
    keys = {matrix_key(m) for m in SINGLE_QUBIT_CLIFFORDS}
    assert len(SINGLE_QUBIT_CLIFFORDS) == 24 and len(keys) == 24
    for a in SINGLE_QUBIT_CLIFFORDS[::5]:
        assert matrix_key(a.conj().T) in keys
        for b in SINGLE_QUBIT_CLIFFORDS:
            assert matrix_key(a @ b) in keys
    # the pulse cost of the group (Section 7.2 item 1): 20 of the 24 cost exactly one pulse (16 gpi2, 4 gpi) and the
    # order-4 Z-rotation subgroup {1, S, Z, S^dag} none; the two-GPi2 branch of the decomposer is never reached
    pulses = []
    n_gpi2 = n_gpi = 0
    for m in SINGLE_QUBIT_CLIFFORDS:
        ops = decompose_single_qubit(m, 0)
        pulses.append(sum(1 for op in ops if op.name in ("gpi", "gpi2")))
        n_gpi2 += sum(1 for op in ops if op.name == "gpi2")
        n_gpi += sum(1 for op in ops if op.name == "gpi")
        assert all(op.name in ("gpi", "gpi2", "rz") for op in ops)
    assert max(pulses) == 1 and pulses.count(0) == 4 and pulses.count(1) == 20, pulses
    assert (n_gpi2, n_gpi) == (16, 4)
    assert sum(pulses) / 24 == pytest.approx(0.8333333333333334, rel=1e-12)
    for m, k in zip(SINGLE_QUBIT_CLIFFORDS, pulses):
        if k == 0:
            assert np.max(np.abs(m - np.diag(np.diag(m)))) < 1e-12, (
                "the pulse-free elements are the Z rotations"
            )


def test_two_qubit_group_order_class_sizes_and_stabilizers() -> None:
    """|C2| = 11520 by closure; the classes L g L have sizes 576^2 / |L intersect g L g^-1| = 576, 5184, 5184, 576."""
    group = two_qubit_clifford_group()
    assert len(group) == TWO_QUBIT_GROUP_ORDER
    assert {c: stabilizer_size(c) for c in CORES} == {"identity": 576, "cnot": 64, "iswap": 64, "swap": 576}
    assert {c: 576 * 576 // stabilizer_size(c) for c in CORES} == CLASS_SIZES
    assert sum(CLASS_SIZES.values()) == TWO_QUBIT_GROUP_ORDER
    counts = {c: 0 for c in CORES}
    for m in group:
        counts[pauli_frame_of(m)] += 1
    assert counts == CLASS_SIZES
    # 1.5 entangling gates per Clifford on average (the count RB papers quote)
    assert sum(
        ENTANGLING_COUNT[c] * n for c, n in CLASS_SIZES.items()
    ) / TWO_QUBIT_GROUP_ORDER == pytest.approx(1.5)


def test_core_templates_reproduce_their_matrices() -> None:
    assert np.allclose(ISWAP_MATRIX @ ISWAP_MATRIX.conj().T, np.eye(4))
    for name, mat in CORES.items():
        ops = core_operations(name, (0, 1))
        assert global_phase(operations_unitary(ops, (0, 1)), mat, atol=1e-9) is not None, name
        assert compile_report(Circuit(2, tuple(ops), (0, 1))).n_entangling == ENTANGLING_COUNT[name], name


def test_uniform_sampling_covers_the_group_with_the_class_frequencies() -> None:
    group_keys = {matrix_key(m) for m in two_qubit_clifford_group()}
    rng = np.random.default_rng(7)
    n = 3000
    counts = {c: 0 for c in CORES}
    seen: set[bytes] = set()
    for _ in range(n):
        c = random_two_qubit_clifford(rng)
        key = matrix_key(c.matrix)
        assert key in group_keys
        seen.add(key)
        counts[c.core] += 1
        d = decompose_two_qubit_clifford(c.matrix)
        assert d.core == c.core == pauli_frame_of(c.matrix)
        assert global_phase(d.matrix, c.matrix) is not None
        assert global_phase(operations_unitary(c.operations((0, 1)), (0, 1)), c.matrix, atol=1e-8) is not None
    for core, size in CLASS_SIZES.items():
        expected = n * size / TWO_QUBIT_GROUP_ORDER
        assert abs(counts[core] - expected) < 5.0 * math.sqrt(expected), (core, counts[core], expected)
    assert len(seen) > 0.2 * n, "3000 draws from 11520 elements hit many distinct elements"


def test_every_group_element_is_recognized_with_its_class() -> None:
    for m in two_qubit_clifford_group()[::23]:
        d = decompose_two_qubit_clifford(m)
        assert isinstance(d, TwoQubitClifford) and d.core == pauli_frame_of(m)
        assert global_phase(d.matrix, m) is not None
    with pytest.raises(ValueError):
        decompose_two_qubit_clifford(np.diag([1.0, 1.0, 1.0, np.exp(0.3j)]))


def test_random_sequences_close_to_the_identity_and_measure_their_qubits() -> None:
    rng = np.random.default_rng(3)
    for m in (1, 5, 12):
        seq = single_qubit_sequences(rng, (1,), m, 2)
        assert seq.circuit.measure == (1,) and all(op.qubits == (1,) for op in seq.circuit.ops)
        assert (
            global_phase(seq.inverse[0] @ single_qubit_sequence_unitary(seq.cliffords[0]), np.eye(2))
            is not None
        )
        assert global_phase(operations_unitary(list(seq.circuit.ops), (1,)), np.eye(2), atol=1e-8) is not None
        two = two_qubit_sequence(rng, (0, 1), m, 2)
        assert two.circuit.measure == (0, 1)
        assert (
            global_phase(operations_unitary(list(two.circuit.ops), (0, 1)), np.eye(4), atol=1e-8) is not None
        )
        expected = (
            sum(ENTANGLING_COUNT[c.core] for c in two.cliffords[0]) + ENTANGLING_COUNT[two.inverse.core]
        )
        assert compile_report(two.circuit).n_entangling == expected
    sim = single_qubit_sequences(rng, (0, 1), 4, 2)
    assert sim.circuit.measure == (0, 1) and len(sim.cliffords) == 2
    assert global_phase(operations_unitary(list(sim.circuit.ops), (0, 1)), np.eye(4), atol=1e-8) is not None
