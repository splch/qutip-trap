"""The Clifford groups behind randomized benchmarking (PLAN.md Section 7.9; Section 13 row "RB error rate"; M10): the 24- and
11520-element groups by closure, the four entangling classes and their sizes from the stabilizers, uniform sampling by the
double-coset construction, the class recognition of an arbitrary two-qubit Clifford, and the closure of a random sequence by
the inverse of its product."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.benchmarks.clifford import (
    CLASS_SIZES,
    CORES,
    ENTANGLING_COUNT,
    ISWAP_MATRIX,
    SINGLE_QUBIT_CLIFFORDS,
    TWO_QUBIT_GROUP_ORDER,
    TwoQubitClifford,
    core_operations,
    decompose_two_qubit_clifford,
    matrix_key,
    operations_unitary,
    pauli_frame_of,
    random_two_qubit_clifford,
    single_qubit_sequence_unitary,
    stabilizer_size,
    two_qubit_clifford_group,
)
from qutip_trap.benchmarks.rb import single_qubit_sequences, two_qubit_sequence
from qutip_trap.control.compiler import compile_report, decompose_single_qubit
from qutip_trap.control.two_qubit import global_phase


def test_single_qubit_group_has_24_elements_closed_under_products_and_inverses() -> None:
    keys = {matrix_key(m) for m in SINGLE_QUBIT_CLIFFORDS}
    assert len(SINGLE_QUBIT_CLIFFORDS) == 24 and len(keys) == 24
    for a in SINGLE_QUBIT_CLIFFORDS[::5]:
        assert matrix_key(a.conj().T) in keys
        for b in SINGLE_QUBIT_CLIFFORDS:
            assert matrix_key(a @ b) in keys
    # the pulse cost of the group (Section 7.2 item 1): 20 of the 24 Cliffords cost exactly ONE pulse (16 gpi2, 4 gpi)
    # and the remaining 4 -- the order-4 Z-rotation subgroup {1, S, Z, S^dag}, one sixth of the group, not "a third" and
    # not "eight" -- cost none at all; the two-GPi2 ZXZXZ branch of the decomposer is never reached by a Clifford
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
    assert n_gpi2 / 24 == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert n_gpi / 24 == pytest.approx(1.0 / 6.0, rel=1e-12)
    # the four pulse-free elements are exactly the diagonal (Z-rotation) Cliffords
    free = [m for m, k in zip(SINGLE_QUBIT_CLIFFORDS, pulses) if k == 0]
    assert len(free) == 4
    for m in free:
        assert np.max(np.abs(m - np.diag(np.diag(m)))) < 1e-12


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
    avg = sum(ENTANGLING_COUNT[c] * n for c, n in CLASS_SIZES.items()) / TWO_QUBIT_GROUP_ORDER
    assert avg == pytest.approx(1.5)


def test_core_templates_reproduce_their_matrices() -> None:
    assert np.allclose(ISWAP_MATRIX @ ISWAP_MATRIX.conj().T, np.eye(4))
    for name, mat in CORES.items():
        got = operations_unitary(core_operations(name, (0, 1)), (0, 1))
        assert global_phase(got, mat, atol=1e-9) is not None, name
        rep = compile_report(
            __import__("qutip_trap.control.compiler", fromlist=["Circuit"]).Circuit(
                2, tuple(core_operations(name, (0, 1))), (0, 1)
            )
        )
        assert rep.n_entangling == ENTANGLING_COUNT[name], name


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
        assert d.entangling_count == ENTANGLING_COUNT[c.core]
        ops = c.operations((0, 1))
        assert global_phase(operations_unitary(ops, (0, 1)), c.matrix, atol=1e-8) is not None
    for core, size in CLASS_SIZES.items():
        expected = n * size / TWO_QUBIT_GROUP_ORDER
        assert abs(counts[core] - expected) < 5.0 * math.sqrt(expected), (core, counts[core], expected)
    assert len(seen) > 0.2 * n, "3000 draws from 11520 elements hit many distinct elements"


def test_every_group_element_is_recognized_with_its_class() -> None:
    group = two_qubit_clifford_group()
    for m in group[::23]:
        d = decompose_two_qubit_clifford(m)
        assert d.core == pauli_frame_of(m)
        assert global_phase(d.matrix, m) is not None
        assert isinstance(d, TwoQubitClifford)
    with pytest.raises(ValueError):
        decompose_two_qubit_clifford(np.diag([1.0, 1.0, 1.0, np.exp(0.3j)]))


def test_random_sequences_close_to_the_identity_and_measure_their_qubits() -> None:
    rng = np.random.default_rng(3)
    for m in (1, 5, 12):
        seq = single_qubit_sequences(rng, (1,), m, 2)
        assert seq.circuit.measure == (1,) and all(op.qubits == (1,) for op in seq.circuit.ops)
        u = single_qubit_sequence_unitary(seq.cliffords[0])
        assert global_phase(seq.inverse[0] @ u, np.eye(2)) is not None
        got = operations_unitary(list(seq.circuit.ops), (1,))
        assert global_phase(got, np.eye(2), atol=1e-8) is not None
        two = two_qubit_sequence(rng, (0, 1), m, 2)
        assert two.circuit.measure == (0, 1)
        got2 = operations_unitary(list(two.circuit.ops), (0, 1))
        assert global_phase(got2, np.eye(4), atol=1e-8) is not None
        rep = compile_report(two.circuit)
        assert (
            rep.n_entangling
            == sum(c.entangling_count for c in two.cliffords[0]) + two.inverse.entangling_count
        )
    sim = single_qubit_sequences(rng, (0, 1), 4, 2)
    assert sim.circuit.measure == (0, 1) and len(sim.cliffords) == 2
    assert global_phase(operations_unitary(list(sim.circuit.ops), (0, 1)), np.eye(4), atol=1e-8) is not None
