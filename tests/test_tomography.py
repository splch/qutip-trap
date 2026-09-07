"""State-based process tomography, the Choi reconstruction, the CP/TP projection and the Kraus application (PLAN.md Section 5.4
item (a); Section 6.8; Section 9.17 row "GATE_LOCAL tomography"; M9a), on synthetic channels and against the ideal unitaries the
scheduler records (``GateTarget``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control import native
from qutip_trap.control.schedule import GateTarget
from qutip_trap.dynamics.tomography import (
    apply_kraus_dm,
    apply_kraus_ket,
    choi_least_squares,
    computational_labels,
    cp_residual,
    expansion_coefficients,
    ideal_unitary_on,
    input_states,
    kraus_operators,
    project_cptp,
    regrid_reduced,
    single_qudit_inputs,
    tp_residual,
)
from qutip_trap.noise.summary import (
    apply_choi,
    choi_from_unitary,
    depolarizing_choi,
    entanglement_infidelity,
    pauli_twirl,
)


def _dms(dims: tuple[int, ...]) -> list[np.ndarray]:
    return [np.outer(k, k.conj()) for _lab, k in input_states(dims)]


def test_input_states_span_the_operator_space() -> None:
    """d^2 pure inputs per qudit, linearly independent over the complex numbers (the least squares needs full rank)."""
    for dims in ((2,), (2, 2), (3,), (2, 3)):
        states = input_states(dims)
        d = int(np.prod(dims))
        assert len(states) == d * d
        r = np.array([np.outer(k, k.conj()).reshape(-1) for _l, k in states])
        assert np.linalg.matrix_rank(r) == d * d
        assert all(abs(np.linalg.norm(k) - 1.0) < 1e-12 for _l, k in states)
    assert [lab for lab, _k in single_qudit_inputs(2)] == ["0", "1", "+", "+i"]
    assert computational_labels((2, 2)) == ["0,0", "0,1", "1,0", "1,1"]
    with pytest.raises(ValueError):
        single_qudit_inputs(1)


def test_unitary_channel_is_reconstructed_exactly_and_returns_one_kraus_operator() -> None:
    """Sixteen inputs through MS(0.3, -0.7, pi/2): the Choi matrix of ``noise.summary.choi_from_unitary`` to round-off, both
    residuals at round-off after the projection, one Kraus operator equal to the unitary up to a phase."""
    u = native.ms(0.3, -0.7, math.pi / 2.0)
    rhos = _dms((2, 2))
    outs = [u @ r @ u.conj().T for r in rhos]
    c = choi_least_squares(rhos, outs)
    assert np.max(np.abs(c - choi_from_unitary(u))) < 1e-13
    cp, cpr, tpr, its = project_cptp(c)
    assert cpr < 1e-13 and tpr < 1e-13 and its >= 1
    ks = kraus_operators(cp)
    assert len(ks) == 1
    phase = ks[0][0, 0] / u[0, 0] if abs(u[0, 0]) > 1e-9 else ks[0][0, 3] / u[0, 3]
    assert abs(abs(phase) - 1.0) < 1e-12 and np.max(np.abs(ks[0] - phase * u)) < 1e-12
    # E(rho) from the Choi matrix agrees with U rho U^dag on a state outside the input set
    psi = np.array([0.6, 0.0, 0.0, 0.8j])
    rho = np.outer(psi, psi.conj())
    assert np.max(np.abs(apply_choi(cp, rho) - u @ rho @ u.conj().T)) < 1e-12


def test_dykstra_projection_restores_trace_preservation_and_positivity_under_noise() -> None:
    """Section 9.17: the projection onto CP and TP leaves ||Tr_out(Choi) - 1|| < 1e-10 and reports both residuals; a PSD
    projection alone would not (the noisy reconstruction violates TP at the noise level)."""
    cd = depolarizing_choi(0.05, 2)
    rhos = _dms((2, 2))
    rng = np.random.default_rng(1)
    outs = []
    for r in rhos:
        o = apply_choi(cd, r) + 3e-4 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        outs.append(0.5 * (o + o.conj().T))
    raw = choi_least_squares(rhos, outs)
    assert tp_residual(raw) > 1e-4, "the noise breaks trace preservation"
    assert cp_residual(raw) >= 0.0
    cp, cpr, tpr, its = project_cptp(raw)
    assert tpr < 1e-10 and cpr < 1e-10, (cpr, tpr)
    assert its >= 1
    assert np.max(np.abs(cp - cd)) < 5e-3
    assert abs(np.trace(cp) - 1.0) < 1e-12
    w = np.linalg.eigvalsh(cp)
    assert w.min() > -1e-12
    # the summary numbers of Section 6.8 on the projected matrix: the depolarizing rate IS the entanglement infidelity
    eps = entanglement_infidelity(cp, choi_from_unitary(np.eye(4, dtype=complex)))
    assert eps == pytest.approx(0.05, abs=2e-3)
    twirl = pauli_twirl(cp, 2)
    assert twirl["II"] == pytest.approx(0.95, abs=2e-3)
    assert all(abs(v - 0.05 / 15.0) < 2e-3 for k, v in twirl.items() if k != "II")


def test_kraus_application_on_a_register_matches_the_embedded_unitary() -> None:
    """The map on factors (0, 2) of a three-qubit register, on a density matrix and by Kraus sampling on a ket (a unitary has
    one Kraus operator, so the sample is deterministic), against the embedded unitary."""
    u = native.ms(0.3, -0.7, math.pi / 2.0)
    ks = kraus_operators(choi_from_unitary(u))
    psi = np.zeros(8, dtype=complex)
    psi[0] = psi[7] = 1.0 / math.sqrt(2.0)
    rho = np.outer(psi, psi.conj())
    u3 = ideal_unitary_on([((0, 2), u)], (0, 1, 2), (2, 2, 2))
    assert np.max(np.abs(u3.conj().T @ u3 - np.eye(8))) < 1e-12
    out = apply_kraus_dm(rho, ks, (2, 2, 2), (0, 2))
    assert np.max(np.abs(out - u3 @ rho @ u3.conj().T)) < 1e-12
    assert np.trace(out).real == pytest.approx(1.0, abs=1e-12)
    v, a = apply_kraus_ket(psi, ks, (2, 2, 2), (0, 2), np.random.default_rng(0))
    assert a == 0
    ref = u3 @ psi
    phase = v[np.argmax(np.abs(ref))] / ref[np.argmax(np.abs(ref))]
    assert np.max(np.abs(v - phase * ref)) < 1e-12
    # the embedding on a subset in reversed order: the first listed ion is the first factor of the gate matrix
    u_rev = ideal_unitary_on([((2, 0), u)], (0, 1, 2), (2, 2, 2))
    swap = np.zeros((4, 4))
    swap[0, 0] = swap[3, 3] = swap[1, 2] = swap[2, 1] = 1.0
    assert np.max(np.abs(u_rev - ideal_unitary_on([((0, 2), swap @ u @ swap)], (0, 1, 2), (2, 2, 2)))) < 1e-12
    # a depolarizing channel is applied with its full Kraus set and stays trace one
    kd = kraus_operators(depolarizing_choi(0.1, 2))
    assert len(kd) == 16
    out_d = apply_kraus_dm(rho, kd, (2, 2, 2), (1, 2))
    assert np.trace(out_d).real == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.eigvalsh(out_d).min() > -1e-12


def test_expansion_coefficients_and_regridding() -> None:
    rhos = _dms((2, 2))
    target = 0.3 * rhos[3] + 0.7 * rhos[7]
    c = expansion_coefficients(target, rhos)
    assert c == pytest.approx(np.eye(16)[3] * 0.3 + np.eye(16)[7] * 0.7, abs=1e-12)
    # a generic Hermitian matrix: the coefficients are real and reproduce it
    rng = np.random.default_rng(2)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)
    c = expansion_coefficients(h, rhos)
    assert np.max(np.abs(sum(ck * r for ck, r in zip(c, rhos)) - h)) < 1e-10
    rho_m = np.diag([0.7, 0.2, 0.1]).astype(complex)
    padded, dropped = regrid_reduced(rho_m, 5)
    assert padded.shape == (5, 5) and dropped == 0.0 and padded[1, 1] == 0.2
    cut, dropped2 = regrid_reduced(rho_m, 2)
    assert dropped2 == pytest.approx(0.1) and np.trace(cut).real == pytest.approx(1.0)


def test_gate_target_unitary_follows_the_virtual_z_rule() -> None:
    """The ideal physical unitary of a played gate: the native gate at its frame-applied phase, then RZ(-theta) for the Stark
    frame the scheduler absorbed (Section 7.6: a virtual RZ(theta) leaves the state as RZ(-theta) times the ideal one)."""
    t = GateTarget("gpi2[0]", (1,), ("gpi2", (0.4,)), {1: 0.0}, ("gpi2[0]",), 0.0, 1e-6)
    assert np.max(np.abs(t.unitary() - native.gpi2(0.4))) < 1e-14
    t2 = GateTarget("gpi2[0]", (1,), ("gpi2", (0.4,)), {1: 0.25}, ("gpi2[0]",), 0.0, 1e-6)
    assert np.max(np.abs(t2.unitary() - native.rz(-0.25) @ native.gpi2(0.4))) < 1e-14
    ms = GateTarget(
        "ms[2]", (0, 1), ("ms", (0.1, -0.2, math.pi / 2)), {0: 0.05, 1: -0.03}, ("a", "b"), 0.0, 1e-4
    )
    expected = np.kron(native.rz(-0.05), native.rz(0.03)) @ native.ms(0.1, -0.2, math.pi / 2)
    assert np.max(np.abs(ms.unitary() - expected)) < 1e-14
    zz = GateTarget("zz[3]/loop1", (0, 1), ("zz", (0.3,)), {}, ("c",), 0.0, 1e-4)
    assert np.max(np.abs(zz.unitary() - native.zz(0.3))) < 1e-14
