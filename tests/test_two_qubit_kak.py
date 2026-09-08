"""The KAK decomposition of arbitrary two-qubit unitaries into native gates (PLAN.md Section 7.2 extended to SU(4); Section 13
rows "Rotation generators" and "Entangling angle"; M10): the named gates' canonical classes, Haar-random SU(4) in the Weyl
chamber at three entangling gates, local unitaries at none, and the compiled block verified against its target."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.compiler import (
    CNOT_MATRIX,
    SWAP_MATRIX,
    Circuit,
    CompileError,
    compile_with_report,
    cp_matrix,
)
from qutip_trap.control.two_qubit import (
    canonical_operations,
    canonical_unitary,
    decompose_two_qubit_unitary,
    global_phase,
    haar_random_unitary,
    is_local,
    kak_decomposition,
    kron_factor,
    verify_operations,
)

ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)


@pytest.mark.parametrize(
    ("name", "u", "coefficients", "count"),
    [
        ("identity", np.eye(4), (0.0, 0.0, 0.0), 0),
        ("CNOT", CNOT_MATRIX, (math.pi / 4, 0.0, 0.0), 1),
        ("CZ", cp_matrix(math.pi), (math.pi / 4, 0.0, 0.0), 1),
        ("CP(0.3)", cp_matrix(0.3), (0.075, 0.0, 0.0), 1),
        ("iSWAP", ISWAP, (math.pi / 4, math.pi / 4, 0.0), 2),
        ("SWAP", SWAP_MATRIX, (math.pi / 4, math.pi / 4, math.pi / 4), 3),
    ],
)
def test_named_gates_land_in_their_canonical_class(
    name: str, u: np.ndarray, coefficients: tuple[float, float, float], count: int
) -> None:
    k = kak_decomposition(u)
    a, b, c = k.coefficients
    assert (a, b, abs(c)) == pytest.approx(coefficients, abs=1e-9), name
    assert k.entangling_count == count
    assert np.max(np.abs(k.matrix() - u)) < 1e-9
    ops = decompose_two_qubit_unitary(u, (0, 1))
    rep = compile_with_report(Circuit(2, tuple(ops), (0, 1)))
    assert rep.n_entangling == count and rep.circuit_residual is not None and rep.circuit_residual < 1e-8


def test_haar_random_su4_costs_three_entangling_gates_in_the_weyl_chamber() -> None:
    rng = np.random.default_rng(11)
    for _ in range(60):
        u = haar_random_unitary(rng, 4)
        assert np.max(np.abs(u.conj().T @ u - np.eye(4))) < 1e-12
        k = kak_decomposition(u)
        a, b, c = k.coefficients
        assert math.pi / 4 + 1e-12 >= a >= b >= abs(c) >= 0.0
        assert np.max(np.abs(k.matrix() - u)) < 1e-10
        assert k.entangling_count == 3
        for m in k.left + k.right:
            assert np.max(np.abs(m.conj().T @ m - np.eye(2))) < 1e-9
        ops = decompose_two_qubit_unitary(u, (1, 0))
        rep = compile_with_report(Circuit(2, tuple(ops), (0, 1)))
        assert rep.n_entangling == 3 and rep.circuit_residual is not None and rep.circuit_residual < 1e-8
        assert all(op.name in ("gpi", "gpi2", "ms", "zz") for op in rep.circuit.ops)
        # every Moelmer-Soerensen gate is played at an angle in [0, pi/2] (Section 7.1)
        for op in rep.circuit.ops:
            if op.name == "ms":
                assert 0.0 <= op.params[2] <= math.pi / 2 + 1e-12


def test_local_unitaries_and_kron_factor() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        a = haar_random_unitary(rng, 2)
        b = haar_random_unitary(rng, 2)
        u = np.kron(a, b) * np.exp(0.4j)
        assert is_local(u)
        fac = kron_factor(u)
        assert fac is not None
        assert global_phase(np.kron(*fac), u) is not None
        k = kak_decomposition(u)
        assert k.entangling_count == 0 and np.max(np.abs(k.matrix() - u)) < 1e-10
        assert decompose_two_qubit_unitary(u, (0, 1)) == decompose_two_qubit_unitary(u, (0, 1))
        assert not is_local(CNOT_MATRIX @ np.kron(a, b))


def test_canonical_operations_match_the_canonical_unitary_and_verify() -> None:
    rng = np.random.default_rng(5)
    for _ in range(10):
        a, b, c = rng.uniform(-math.pi / 4, math.pi / 4, size=3)
        ops = canonical_operations(a, b, c, (0, 1))
        assert verify_operations(ops, canonical_unitary(a, b, c), (0, 1)) < 1e-9
    with pytest.raises(CompileError):
        verify_operations(
            canonical_operations(0.3, 0.0, 0.0, (0, 1)), canonical_unitary(0.4, 0.0, 0.0), (0, 1)
        )
    with pytest.raises(CompileError):
        kak_decomposition(np.ones((4, 4)))
