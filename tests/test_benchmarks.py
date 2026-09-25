"""Benchmark emulation (PLAN.md Sections 7.9, 13 'RB error rate'): randomized benchmarking (single-qubit, simultaneous,
two-qubit and Knill-style), GHZ fidelity and a quantum-volume style run on the two-ion example device with the simulator's
own budget alongside, and the algebra behind them on synthetic inputs: the decay fit and its units, the channel
reduction, the GHZ parity bound, the heavy outputs and Cross et al.'s Eq. (32) criterion, the intrinsic-budget keys and
the depolarizing conversions."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    gate_piece_of,
    intrinsic_by_kind,
    kinds_of_schedule,
    one_gate_circuit,
    reduced_choi,
    reduced_ideal,
)
from qutip_trap.benchmarks.ghz import ghz_circuit, ghz_fidelity, parity_circuit
from qutip_trap.benchmarks.rb import (
    fit_decay,
    knill_sequences,
    marginal_survival,
    mean_survival_sigma,
    randomized_benchmarking,
    rb_model,
)
from qutip_trap.benchmarks.volume import (
    CONFIDENCE_Z,
    PROTOCOL_CIRCUITS,
    cross_confidence_sigma,
    heavy_output_pass,
    quantum_volume,
    random_square_circuit,
)
from qutip_trap.control import native
from qutip_trap.control.compiler import CNOT_MATRIX, Circuit, Operation, compile_report, ideal_probabilities
from qutip_trap.control.native import global_phase
from qutip_trap.control.two_qubit import haar_random_unitary
from qutip_trap.device.presets import yb171_chain
from qutip_trap.experiments.fitting import fit_fringe
from qutip_trap.machine import Machine
from qutip_trap.noise.summary import (
    apply_choi,
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_choi,
    depolarizing_entanglement_infidelity,
    depolarizing_kraus,
    entanglement_infidelity,
    qiskit_depolarizing_lambda,
    rb_error_per_clifford,
)
from qutip_trap.options import Numerics, Truncation
from qutip_trap.run.job import last_record
from tests.fixtures import FAST, two_ion_surrogate


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    return Machine(yb171_chain(2).device, table=two_ion_surrogate(1000).table, numerics=FAST)


# ---- the decay fit and its units -------------------------------------------------------------------------------------------------


def test_decay_fit_recovers_p_and_the_section_13_conversions() -> None:
    lengths = (1, 16, 64, 256)
    a, p, b = 0.49, 0.9995, 0.5
    y = rb_model(np.array([a, p, b]), np.array(lengths, dtype=float))
    fit, ok, _notes = fit_decay(lengths, y, np.full(4, 1e-4), 1)
    assert ok and fit["p"][0] == pytest.approx(p, abs=1e-6) and fit["B"][0] == pytest.approx(b, abs=1e-4)
    fit2, ok2, notes2 = fit_decay(lengths[:2], y[:2], np.full(2, 1e-4), 1)
    assert ok2 and "fixed" in notes2[0] and fit2["p"][0] == pytest.approx(p, abs=1e-6)
    fit3, _ok3, notes3 = fit_decay(lengths, y, np.full(4, 1e-4), 1, fix_offset=True)
    assert "fix_offset" in notes3[0] and fit3["B"] == (0.5, 0.0)
    assert rb_error_per_clifford(p, 1) == pytest.approx((1 - p) / 2)
    assert rb_error_per_clifford(p, 2) == pytest.approx(0.75 * (1 - p))
    assert depolarizing_entanglement_infidelity(p, 1) == pytest.approx(0.75 * (1 - p))
    assert depolarizing_entanglement_infidelity(p, 2) == pytest.approx(15.0 / 16.0 * (1 - p))


def test_marginal_fit_recovers_the_per_qubit_rate_and_the_joint_fit_their_sum() -> None:
    """Two independent qubits, each decaying as 0.5 p_q^m + 0.5 with r_q = (1 - p_q)/2: the marginal fit returns r_q, the
    joint fit's (1 - p)(2^n - 1)/2^n returns sum_q r_q to first order, an error per LAYER of n Cliffords and a factor n
    away from any per-Clifford rate."""
    ls = (1, 128, 512)
    for r_q in (2.0e-5, 1.0e-4, 3.426e-4):
        marginal = np.array([0.5 * (1.0 - 2.0 * r_q) ** m + 0.5 for m in ls])
        m_fit, m_ok, _ = fit_decay(ls, marginal, np.full(3, 1e-6), 1, fix_offset=True)
        assert m_ok and rb_error_per_clifford(m_fit["p"][0], 1) == pytest.approx(r_q, rel=1e-6)
        j_fit, j_ok, _ = fit_decay(ls, marginal**2, np.full(3, 1e-6), 2, fix_offset=True)
        joint_r = rb_error_per_clifford(j_fit["p"][0], 2)
        assert j_ok and joint_r == pytest.approx(2.0 * r_q, rel=0.07)
        assert joint_r > 1.8 * r_q, "the joint decay is the SUM over the qubits, never a per-Clifford rate"


def test_marginal_survival_reads_the_histogram_in_the_section_13_bit_order() -> None:
    """The histogram key sorts the measured qubits with the lowest-indexed one RIGHTMOST (``conv.result_bit_order``), so a
    marginal read off position i belongs to the (n - 1 - i)-th sorted qubit."""

    class _Diag:
        effective_sample_size = 1000.0

    class _Res:
        diagnostics = _Diag()
        probabilities = {"01": 0.7, "11": 0.2, "00": 0.1}

    got = marginal_survival(_Res(), "01", (0, 1))  # type: ignore[arg-type]
    # key "01" means qubit 1 reads 0 and qubit 0 reads 1
    assert got[1][0] == pytest.approx(0.8)  # keys "01" and "00" have a 0 at position 0 -> qubit 1
    assert got[0][0] == pytest.approx(0.9)  # keys "01" and "11" have a 1 at position 1 -> qubit 0
    with pytest.raises(ValueError):
        marginal_survival(_Res(), "0", (0, 1))  # type: ignore[arg-type]


def test_mean_survival_sigma_does_not_double_count_the_shot_noise() -> None:
    """The between-sequence sample variance already contains each point's shot noise; adding the within-sequence variance
    to it would inflate the error bar by up to sqrt 2 where shot noise dominates."""
    values = np.array([[0.90, 0.92, 0.88]])
    sigma = np.full((1, 3), 0.01)
    got = mean_survival_sigma(values, sigma, 3, 100)
    assert got[0] == pytest.approx(float(values[0].std(ddof=1)) / math.sqrt(3.0), rel=1e-12)
    assert math.sqrt(float(values[0].var(ddof=1)) / 3.0 + float((sigma[0] ** 2).mean()) / 3.0) > got[0]
    # at one sequence the shot noise is the whole error bar
    assert mean_survival_sigma(values[:, :1], sigma[:, :1], 1, 100)[0] == pytest.approx(0.01, rel=1e-12)
    flat = np.full((1, 4), 0.5)
    assert mean_survival_sigma(flat, np.full((1, 4), 0.02), 4, 100)[0] == pytest.approx(
        1.0 / (100 * 4), rel=1e-12
    )


def test_knill_sequences_end_on_their_own_target_and_cost_one_pulse_per_gate() -> None:
    """Section 7.9's first RB variant: pi/2 gates with an interleaved pi or identity and one final pi/2. The ideal state
    never leaves the six Pauli eigenstates, so the sequence's own target bitstring carries the whole ideal probability."""
    rng = np.random.default_rng(11)
    for length in (1, 3, 9, 24):
        for qubits in ((0,), (0, 1), (0, 1, 2), (2, 0)):
            seq = knill_sequences(rng, qubits, length, 3)
            assert ideal_probabilities(seq.circuit).get(seq.key, 0.0) == pytest.approx(1.0, abs=1e-9), (
                length,
                qubits,
            )
            assert seq.circuit.measure == tuple(int(q) for q in qubits)
            assert set(seq.key) <= {"0", "1"} and len(seq.key) == len(qubits)
            pulses = [op for op in seq.circuit.ops if op.name in ("gpi", "gpi2")]
            # per qubit: length Clifford pulses, at most length Pauli pulses, at most one closing pi/2
            assert len(qubits) * length <= len(pulses) <= len(qubits) * (2 * length + 1)
            assert all(op.name in ("gpi", "gpi2", "rz") for op in seq.circuit.ops)
    assert {knill_sequences(rng, (0,), 5, 1).key for _ in range(40)} == {"0", "1"}, (
        "every target bit is reachable"
    )


def test_pair_true_is_refused_on_any_qubit_count_but_two() -> None:
    machine = yb171_chain(2).machine()
    for qubits in ((0,), (0, 1)):
        with pytest.raises(ValueError, match="two qubits"):
            randomized_benchmarking(machine, qubits, (1,), pair=True, variant="knill")
    with pytest.raises(ValueError, match="two qubits"):
        randomized_benchmarking(machine, (0,), (1,), pair=True)
    with pytest.raises(ValueError, match="variant"):
        randomized_benchmarking(machine, (0,), (1,), variant="direct")


# ---- the channel reduction and the intrinsic budget ------------------------------------------------------------------------


def test_reduced_choi_traces_the_spectator_in_zero_and_keeps_the_kept_channel() -> None:
    rng = np.random.default_rng(1)
    u, v = haar_random_unitary(rng, 2), haar_random_unitary(rng, 2)
    c = choi_from_unitary(np.kron(u, v))
    assert entanglement_infidelity(reduced_choi(c, 2, [0]), choi_from_unitary(u)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert entanglement_infidelity(reduced_choi(c, 2, [1]), choi_from_unitary(v)) == pytest.approx(
        0.0, abs=1e-12
    )
    # a depolarizing channel on two qubits reduces to a depolarizing channel on one with a smaller rate
    eps = 0.04
    red = reduced_choi(depolarizing_choi(eps, 2), 2, [0])
    e1 = entanglement_infidelity(red, choi_from_unitary(np.eye(2)))
    assert 0.0 < e1 < eps and abs(np.trace(red) - 1.0) < 1e-12 and np.min(np.linalg.eigvalsh(red)) > -1e-12
    # the reduced map applied to a state equals the full map on rho (x) |0><0| traced over the spectator
    w = choi_from_unitary(haar_random_unitary(rng, 4))
    rho = np.array([[0.6, 0.3j], [-0.3j, 0.4]])
    direct = apply_choi(w, np.kron(rho, np.diag([1.0, 0.0]))).reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)
    assert np.max(np.abs(apply_choi(reduced_choi(w, 2, [0]), rho) - direct)) < 1e-12
    assert average_gate_infidelity(0.01, 2) == pytest.approx(2.0 / 3.0 * 0.01)


def test_reduced_ideal_factors_a_step_s_target_instead_of_taking_its_zero_block() -> None:
    """A GATE_LOCAL step's ideal is a product over its ions of each ion's own target, so the factor on the benchmarked
    qubits is what the reduced channel is compared with; the <0_rest| U |0_rest> block would be off by 1/sqrt 2 for a GPi2
    on the other ion and exactly zero for a GPi."""
    eye = np.eye(2, dtype=complex)
    a, b = native.gpi2(0.3), native.gpi(1.1)
    haar = haar_random_unitary(np.random.default_rng(5), 2)
    for u, keep, want, label in (
        (np.kron(a, eye), [0], a, "the addressed ion"),
        (np.kron(eye, b), [0], eye, "a GPi on the OTHER ion: the identity here, and its |0> block is zero"),
        (np.kron(eye, a), [0], eye, "a GPi2 on the other ion: the identity, not 1/sqrt 2 times it"),
        (np.kron(a, b), [1], b, "the second factor"),
        (np.kron(haar, eye), [0], haar, "an arbitrary local target"),
    ):
        got = reduced_ideal(u, 2, keep)
        assert np.max(np.abs(got.conj().T @ got - eye)) < 1e-10, label
        assert global_phase(got, want, atol=1e-8) is not None, label
    u3 = np.kron(np.kron(a, eye), b)
    assert global_phase(reduced_ideal(u3, 3, [0, 1]), np.kron(a, eye), atol=1e-8) is not None
    assert global_phase(reduced_ideal(u3, 3, [1]), eye, atol=1e-8) is not None
    assert np.max(np.abs(reduced_ideal(np.kron(a, b), 2, [0, 1]) - np.kron(a, b))) < 1e-12
    with pytest.raises(ValueError, match="factor"):  # an entangling ideal does not factor
        reduced_ideal(CNOT_MATRIX, 2, [0])


def test_gate_piece_of_finds_multi_piece_gate_ids_by_longest_prefix() -> None:
    """``kinds_of_schedule`` keys on ``Schedule.target.gate_id``, and the ZZ wrapper's ids contain a slash of their own
    (``zz[k]/ms``, ``zz[k]/loop1``, ``zz[k]/loop2``), so the piece is the longest gate id prefixing the key."""
    kinds = {
        "gpi2[0]": "gpi2[0]",
        "ms[2]": "ms[2,3]",
        "zz[2]/ms": "ms[2,3]",
        "zz[2]/loop1": "gpi2[2]",
        "zz[2]/loop2": "gpi2[3]",
    }
    assert gate_piece_of("zz[2]/loop1.residual_displacement", kinds) == "zz[2]/loop1"
    assert gate_piece_of("zz[2]/ms/seg0/ion0.ion0.P_raman", kinds) == "zz[2]/ms"
    assert gate_piece_of("ms[2]/seg0/ion0.ion0.P_raman", kinds) == "ms[2]"
    assert gate_piece_of("gpi2[0].crosstalk", kinds) == "gpi2[0]"
    assert gate_piece_of("ms[2]", kinds) == "ms[2]"
    assert gate_piece_of("gpi2[1].crosstalk", kinds) is None
    assert gate_piece_of("total", kinds) is None


def test_depolarizing_kraus_normalizations_and_the_qiskit_lambda_conversion() -> None:
    """The depolarizing conversions: sum K^dag K = 1 for the p/3 and p/15 weights, the entanglement
    infidelity is p, the average gate infidelity (2/3)p and (4/5)p, and Qiskit's lambda = p 4^n/(4^n - 1) turns Chen et
    al.'s Forte medians 2.0e-4 and 46.4e-4 into 2.67e-4 and 49.5e-4."""
    for n, weight in ((1, 3), (2, 15)):
        d = 2**n
        for p in (0.0, 1e-4, 0.1, 1.0):
            kraus = depolarizing_kraus(p, n)
            assert len(kraus) == 4**n
            assert np.max(np.abs(sum(k.conj().T @ k for k in kraus) - np.eye(d))) < 1e-12, (n, p)
            # the non-identity Kraus weights are exactly p/3 (one qubit) and p/15 (two qubits)
            assert float(np.real(kraus[1].conj().T @ kraus[1])[0, 0]) == pytest.approx(p / weight, abs=1e-15)
        eps = 0.05
        choi = depolarizing_choi(eps, n)
        assert entanglement_infidelity(choi, choi_from_unitary(np.eye(d))) == pytest.approx(eps, abs=1e-12)
        assert average_gate_infidelity(eps, d) == pytest.approx(
            (2.0 / 3.0 if n == 1 else 4.0 / 5.0) * eps, rel=1e-12
        )
    assert qiskit_depolarizing_lambda(2.0e-4, 1) == pytest.approx(2.6667e-4, rel=1e-4)
    assert qiskit_depolarizing_lambda(46.4e-4, 2) == pytest.approx(49.493e-4, rel=1e-4)
    assert f"{qiskit_depolarizing_lambda(2.0e-4, 1):.2e}" == "2.67e-04"
    assert f"{qiskit_depolarizing_lambda(46.4e-4, 2) * 1e4:.3g}" == "49.5"
    with pytest.raises(ValueError):
        depolarizing_kraus(1.5, 1)


# ---- GHZ and quantum volume on synthetic inputs ---------------------------------------------------------------------------


def _ghz_theta_rho(n: int, theta: float, mix: float = 0.0) -> np.ndarray:
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[-1] = np.exp(1j * theta) / math.sqrt(2.0)
    return (1.0 - mix) * np.outer(psi, psi.conj()) + mix * np.eye(2**n, dtype=complex) / 2**n


def _kron_n(u: np.ndarray, n: int) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for _ in range(n):
        out = np.kron(out, u)
    return out


def _parity_under_analysis(rho: np.ndarray, n: int, phi: float) -> float:
    u = _kron_n(native.gpi2(phi), n)
    return float(np.real(np.trace(_kron_n(native.PAULI_Z, n) @ u @ rho @ u.conj().T)))


def test_parity_fit_and_ghz_circuits() -> None:
    n = 3
    phases = np.linspace(0.0, 2 * math.pi / n, 8, endpoint=False)
    fringe = fit_fringe(phases, 0.93 * np.cos(n * phases + 0.4) + 0.01, np.full(8, 1e-3), n)
    assert fringe.converged and fringe.contrast[0] == pytest.approx(0.93, abs=1e-6)
    assert fringe.phase_rad[0] == pytest.approx(0.4, abs=1e-6)
    assert ideal_probabilities(ghz_circuit((0, 1, 2), 3)) == pytest.approx({"000": 0.5, "111": 0.5})
    # the analysis pulses turn the GHZ state's parity into cos(N phi + phi_0) with unit contrast; phi_0 carries the
    # rotation axis (N pi/2) and the compiled circuit's residual frames
    ideal_parity = []
    for phi in phases:
        p = ideal_probabilities(parity_circuit((0, 1, 2), 3, float(phi)))
        ideal_parity.append(sum((-1) ** k.count("1") * v for k, v in p.items()))
    ideal = fit_fringe(phases, np.array(ideal_parity), np.full(8, 1e-6), n)
    assert ideal.converged and ideal.contrast[0] == pytest.approx(1.0, abs=1e-6)
    assert ideal.offset[0] == pytest.approx(0.0, abs=1e-6)
    rep = compile_report(parity_circuit((0, 1), 2, 0.7))
    assert rep.n_entangling == 1 and rep.circuit_residual is not None and rep.circuit_residual < 1e-8


def test_the_parity_bound_equals_the_phase_optimised_ghz_fidelity_and_exceeds_the_fixed_phase_one() -> None:
    """(P_0 + P_1 + C)/2 = (rho_00 + rho_11)/2 + |rho_{0...0,1...1}| = max_theta <GHZ_theta| rho |GHZ_theta> exactly, the
    fitted contrast being 2|rho_{0...0,1...1}|; against a FIXED-phase GHZ state the same number is an upper bound."""
    for n in (2, 3, 4):
        phases = np.linspace(0.0, 2.0 * math.pi / n, 8, endpoint=False)
        for theta, mix in ((0.0, 0.0), (0.7, 0.0), (math.pi, 0.0), (0.4, 0.15), (2.0, 0.05)):
            rho = _ghz_theta_rho(n, theta, mix)
            parity = np.array([_parity_under_analysis(rho, n, float(p)) for p in phases])
            fringe = fit_fringe(phases, parity, np.full(8, 1e-9), n)
            assert fringe.converged
            contrast = fringe.contrast[0]
            assert contrast == pytest.approx(2.0 * abs(rho[0, -1]), abs=1e-8), (n, theta, mix)
            bound = 0.5 * (float(np.real(rho[0, 0])) + float(np.real(rho[-1, -1])) + contrast)
            # the phase-optimised fidelity, in closed form and by scanning theta
            f_max = 0.5 * (float(np.real(rho[0, 0])) + float(np.real(rho[-1, -1]))) + abs(rho[0, -1])
            scan = max(
                float(np.real(np.vdot(g, rho @ g)))
                for g in (
                    np.array([1.0, *([0.0] * (2**n - 2)), np.exp(1j * t)]) / math.sqrt(2.0)
                    for t in np.linspace(0.0, 2.0 * math.pi, 4001)
                )
            )
            assert bound == pytest.approx(f_max, abs=1e-8)
            assert scan == pytest.approx(f_max, abs=1e-5)
            g0 = np.array([1.0, *([0.0] * (2**n - 2)), 1.0]) / math.sqrt(2.0)
            assert float(np.real(np.vdot(g0, rho @ g0))) <= f_max + 1e-12, (
                "the fixed-phase fidelity is never above it"
            )


def test_random_square_circuits_have_heavy_sets_of_half_the_strings_and_three_gates_per_su4() -> None:
    rng = np.random.default_rng(4)
    for n in (2, 3):
        qc = random_square_circuit(rng, tuple(range(n)), n, n)
        assert len(qc.heavy) == 2 ** (n - 1) and 0.5 < qc.heavy_ideal_probability <= 1.0
        assert abs(sum(qc.ideal.values()) - 1.0) < 1e-9 and qc.entangling_count == 3 * n * (n // 2)
        rep = compile_report(qc.circuit)
        assert rep.n_entangling == qc.entangling_count and rep.circuit_residual is not None
        assert rep.circuit_residual < 1e-7
    assert one_gate_circuit("ms[0,1]", 2).ops[0].params == (0.0, 0.0, math.pi / 2)
    assert one_gate_circuit("gpi2[1]", 2).measure == (1,)
    with pytest.raises(ValueError):
        one_gate_circuit("cnot[0,1]", 2)


def test_cross_confidence_sigma_reduces_appendix_c_equation_32_exactly() -> None:
    """Eq. (32) is [n_h - z sqrt(n_h (n_s - n_h/n_c))]/(n_c n_s) > 2/3; with n_h = h n_c n_s it is h - z sqrt(h(1-h)/n_c)
    identically, so the criterion's sigma is the per-circuit binomial one, not the standard error of the mean."""
    for n_c, n_s, h in ((4, 400, 0.7163), (100, 400, 0.7163), (7, 33, 0.61), (100, 1000, 0.68)):
        n_h = h * n_c * n_s
        equation_32 = (n_h - CONFIDENCE_Z * math.sqrt(n_h * (n_s - n_h / n_c))) / (n_c * n_s)
        assert h - CONFIDENCE_Z * cross_confidence_sigma(h, n_c) == pytest.approx(equation_32, abs=1e-12)
    with pytest.raises(ValueError):
        cross_confidence_sigma(0.7, 0)


def test_a_pass_needs_the_confidence_bound_and_the_hundred_circuits() -> None:
    sigma, cleared, met, passed = heavy_output_pass(0.71625, 4)
    assert sigma == pytest.approx(cross_confidence_sigma(0.71625, 4), rel=1e-12)
    assert not cleared and not met and not passed
    # a machine well above threshold still clears nothing below the protocol's circuit count
    _s, cleared_99, met_99, passed_99 = heavy_output_pass(0.85, PROTOCOL_CIRCUITS - 1)
    assert cleared_99 and not met_99 and not passed_99
    _s, cleared_100, met_100, passed_100 = heavy_output_pass(0.85, PROTOCOL_CIRCUITS)
    assert cleared_100 and met_100 and passed_100


# ---- the protocols on the example device ------------------------------------------------------------------------------------


def test_single_qubit_rb_decays_at_the_channel_scale_with_the_budget_alongside(two_ion: Machine) -> None:
    """Section 13 'RB error rate': r = (1 - p)/2 from the fit; the survival decays over hundreds of Cliffords at the 1e-5 level
    the channel of the carrier pulses predicts (the first-order composition is an estimate: coherent errors of one
    Clifford's pulses partly cancel); the SPAM offset F(0) matches the readout and preparation errors."""
    rb = randomized_benchmarking(two_ion, (0,), (1, 128, 512), n_sequences=1, shots=2000, fix_offset=True)
    assert rb.n_qubits == 1 and rb.survival.shape == (3, 1) and rb.converged
    assert rb.mean_survival[0] > 0.998 and rb.mean_survival[-1] < rb.mean_survival[0]
    r, sr = rb.error_per_clifford
    assert 0.0 < r < 1e-4 and sr < 1e-4
    assert rb.fit["B"] == (0.5, 0.0) and rb.fit["p"][0] == pytest.approx(1.0 - 2.0 * r)
    assert rb.depolarizing_entanglement_infidelity[0] == pytest.approx(1.5 * r)
    assert 0.5 < rb.pulses_per_clifford < 1.5 and rb.entangling_per_clifford == 0.0
    b = rb.budget
    assert isinstance(b, BenchmarkBudget) and b.unit == "clifford" and set(b.counts) <= {"gpi2[0]", "gpi[0]"}
    assert sum(b.counts.values()) == pytest.approx(rb.pulses_per_clifford)
    for kind, ch in b.channels.items():
        step = ch.steps[0]
        assert step.ions == (0, 1), "the crosstalk neighbour joins the gate-local space"
        # reduced to qubit 0 the neighbour's crosstalk rotation is not an error; on the full step it is
        assert 0.0 < b.channel_infidelity[kind] < step.summary.average_gate_infidelity
    r_channel = b.predicted["r_channel"]
    assert 1e-6 < r_channel < 1e-4 and 0.2 * r_channel < r < 5.0 * r_channel, (r, r_channel)
    assert b.predicted["r_intrinsic"] > r_channel, "the intrinsic scales bound the coherent errors"
    f0 = b.predicted["F0_spam"]
    assert 0.999 < f0 < 1.0 and abs(rb.mean_survival[0] - f0) < 3.0 * rb.mean_sigma[0] + 2e-4
    assert "first-order composition" in b.notes[0]
    d = rb.results[0].diagnostics
    assert d.level == "JOINT_EXACT" and d.propagator_cache_hits >= 0
    assert rb.results[0].n_qubits == 1 and rb.sequences[0].circuit.measure == (0,)


@pytest.mark.slow
def test_two_qubit_rb_and_the_entangling_channel(two_ion: Machine) -> None:
    """Two-qubit Clifford RB on the pair: 1.5 entangling gates per Clifford on average, r = (3/4)(1 - p) at the 1e-3 level of
    the composed channels (five pulses per Clifford at 2.6e-4 each on the pair dominate the 1.1e-4 entangling gate)."""
    rb = randomized_benchmarking(two_ion, (0, 1), (1, 6, 16), n_sequences=1, shots=400, fix_offset=True)
    assert rb.n_qubits == 2 and rb.fit["B"] == (0.25, 0.0)
    assert rb.mean_survival[0] > 0.97 and rb.mean_survival[-1] < rb.mean_survival[0]
    assert 0.5 <= rb.entangling_per_clifford <= 3.0 and rb.pulses_per_clifford > rb.entangling_per_clifford
    r, sr = rb.error_per_clifford
    assert 1e-4 < r < 3e-2
    b = rb.budget
    assert b is not None and "ms[0,1]" in b.channels
    ms = b.channels["ms[0,1]"].steps[0]
    # the isometry route: four basis columns per motional branch (the sixteen inputs follow by linearity)
    assert ms.local_dimension > 100 and ms.ions == (0, 1) and ms.engine_runs >= 4
    assert 1e-5 < ms.summary.average_gate_infidelity < 1e-3
    assert ms.summary.depolarizing_rate == pytest.approx(1.25 * ms.summary.average_gate_infidelity, rel=1e-6)
    assert 0.2 * b.predicted["r_channel"] < r < 5.0 * b.predicted["r_channel"] + 3.0 * sr, (r, b.predicted)


@pytest.mark.slow
def test_simultaneous_rb_reports_marginals_in_the_budget_s_own_unit(two_ion: Machine) -> None:
    """Gambetta et al. 2012 on the two-ion example device: the per-qubit marginal r_q, the per-qubit composition
    r_channel.q{i} (each kind's channel reduced to THAT ONE qubit) in the same unit, and the joint decay beside them as the
    correlation diagnostic, in the unit of r_channel_joint_layer."""
    rb = randomized_benchmarking(
        two_ion, (0, 1), (1, 32, 128), n_sequences=1, shots=600, pair=False, fix_offset=True
    )
    assert rb.n_qubits == 2 and rb.variant == "clifford"
    assert rb.marginal_survival is not None and rb.marginal_survival.shape == (2, 3, 1)
    assert len(rb.marginal_fit) == 2 and len(rb.marginal_error_per_clifford) == 2
    r, _sr = rb.error_per_clifford
    assert r == pytest.approx(float(np.mean([v for v, _ in rb.marginal_error_per_clifford])), rel=1e-12)
    assert 1e-4 < r < 1e-3, r
    joint = rb.joint_error_per_layer
    assert joint is not None and joint[0] > 1.4 * r, "the joint decay is the sum over the two qubits"
    b = rb.budget
    assert b is not None
    assert set(b.counts) == {"gpi2[0]", "gpi2[1]", "gpi[0]", "gpi[1]"}
    per_qubit = [b.predicted["r_channel.q0"], b.predicted["r_channel.q1"]]
    assert b.predicted["r_channel"] == pytest.approx(float(np.mean(per_qubit)), rel=1e-12)
    # the per-qubit composition is the marginal's own unit; three lengths to 128 at 600 shots on one sequence is a cheap
    # fit, so the band is a factor three
    assert b.predicted["r_channel"] / 3.0 < r < 3.0 * b.predicted["r_channel"], (r, b.predicted)
    # the joint-layer composition is n_q times the composition reduced to ALL benchmarked qubits: the joint decay's unit,
    # above the per-qubit one (a two-qubit average gate infidelity per layer against a one-qubit one per Clifford)
    assert b.predicted["r_channel_joint_layer"] == pytest.approx(
        2.0 * sum(b.counts[k] * b.channel_infidelity[k] for k in b.channel_infidelity), rel=1e-12
    )
    assert b.predicted["r_channel_joint_layer"] > b.predicted["r_channel"]
    assert b.predicted["r_channel_joint_layer"] / 4.0 < joint[0] < 4.0 * b.predicted["r_channel_joint_layer"]
    assert 1.4 * r < joint[0] < 3.0 * r, "the measured joint is the measured marginals summed"
    assert any("Gambetta" in n for n in rb.notes)


@pytest.mark.slow
def test_simultaneous_rb_runs_on_three_ions() -> None:
    """Simultaneous single-qubit RB on a three-ion chain, whose middle ion sees crosstalk from both sides."""
    preset = yb171_chain(3, address_waist_m=2.0e-6)
    rb = randomized_benchmarking(
        Machine(preset.device, numerics=Numerics(truncation=Truncation(branch_weight_min=3e-3))),
        (0, 1, 2),
        (1, 8),
        n_sequences=1,
        shots=200,
        pair=False,
        budget=False,
        fix_offset=True,
    )
    assert rb.n_qubits == 3 and rb.marginal_survival is not None
    assert rb.marginal_survival.shape == (3, 2, 1) and len(rb.marginal_error_per_clifford) == 3
    assert rb.error_per_clifford[0] == pytest.approx(
        float(np.mean([v for v, _ in rb.marginal_error_per_clifford])), rel=1e-12
    )
    assert rb.joint_error_per_layer is not None
    assert all(0.0 <= v < 1e-2 for v, _ in rb.marginal_error_per_clifford)
    assert rb.survival.shape == (2, 1) and rb.results[0].n_qubits == 3
    assert rb.sequences[0].circuit.measure == (0, 1, 2)


@pytest.mark.slow
def test_the_intrinsic_budget_keeps_the_zz_wrapper_s_section_9_6_scales(two_ion: Machine) -> None:
    """With ``entangler="zz"`` the CNOT template's pieces carry gate ids containing a slash (``zz[0]/ms``, ``zz[0]/loop1``,
    ``zz[0]/loop2``); every intrinsic entry of the run belongs to a piece and the wrapper's pieces contribute."""
    circuit = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    zz = dataclasses.replace(two_ion, physics=dataclasses.replace(two_ion.physics, entangler="zz"))
    res = zz.run(circuit, 1)
    kinds = kinds_of_schedule(last_record(res).schedule)
    assert any("/" in gid for gid in kinds), kinds
    by_kind = intrinsic_by_kind(res.diagnostics, kinds)
    assert by_kind, "the per-kind budget is not empty"
    orphans = [
        k
        for k in res.diagnostics.intrinsic_budget
        if k != "total" and gate_piece_of(k, kinds) is None and not k.startswith(("prep", "readout"))
    ]
    assert not orphans, orphans
    assert [k for k in res.diagnostics.intrinsic_budget if k.startswith("zz[")], list(
        res.diagnostics.intrinsic_budget
    )
    assert sum(by_kind.values()) > 0.0


@pytest.mark.slow
def test_ghz_fidelity_bound_and_exact_register_fidelity(two_ion: Machine) -> None:
    """Section 7.9: the Bell/GHZ bound (P_00 + P_11 + C)/2 from populations and a parity scan through run(), and the two
    exact fidelities it sits between: it estimates max_theta <GHZ_theta| rho |GHZ_theta> (``conv.ghz_parity_bound``) and so
    bounds the fixed-phase <GHZ| rho |GHZ> from above; the predicted floor from the channels beside them."""
    g = ghz_fidelity(two_ion, (0, 1), shots=500)
    assert g.n_qubits == 2 and len(g.results) == 9 and g.converged
    p0, p1 = g.populations["P0"][0], g.populations["P1"][0]
    assert p0 + p1 > 0.98 and abs(p0 - p1) < 0.1
    c, sc = g.fit["contrast"]
    assert 0.95 < c < 1.02 and sc < 0.05
    f, sf = g.fidelity_bound
    assert 0.98 < g.register_fidelity < 1.0
    assert g.register_fidelity_max_phase >= g.register_fidelity - 1e-12, (
        "the max-phase fidelity is never lower"
    )
    assert abs(f - g.register_fidelity_max_phase) < 3.0 * sf + 2e-3, (f, sf, g.register_fidelity_max_phase)
    assert f + 3.0 * sf > g.register_fidelity, "and so bounds the fixed-phase fidelity from ABOVE"
    assert any("UPPER bound" in n for n in g.notes)
    b = g.budget
    assert b is not None and b.unit == "circuit" and "ms[0,1]" in b.channels and b.counts["ms[0,1]"] == 1.0
    assert 0.99 < b.predicted["F_gates"] < 1.0 and b.predicted["fidelity_bound"] < b.predicted["F_gates"]
    assert abs(b.predicted["F_gates"] - g.register_fidelity) < 5e-3
    assert g.parity.shape == (8, 3) and np.all(np.abs(g.parity[:, 1]) <= 1.0)


@pytest.mark.slow
def test_quantum_volume_style_run_at_width_two(two_ion: Machine) -> None:
    """Cross et al. 2019 at width two: heavy outputs from the ideal distribution, the measured heavy-output probability near
    the ideal one on this device, three entangling gates per SU(4), the depolarizing prediction from the channels."""
    qv = quantum_volume(two_ion, (0, 1), n_circuits=2, shots=300)
    assert qv.depth == 2 and qv.n_circuits == 2 and qv.entangling_per_circuit == 6.0
    assert np.all(qv.ideal_heavy_probability > 0.5) and np.all(qv.register_fidelity > 0.95)
    assert np.all(
        np.abs(qv.heavy_output_probability - qv.ideal_heavy_probability) < 4.0 * qv.heavy_sigma + 0.05
    )
    assert not qv.protocol_circuit_count_met and any("100" in n for n in qv.notes)
    # Eq. (32)'s sigma = sqrt(h(1 - h)/n_c) sits far above the standard error of the mean, and a pass needs the 100
    # circuits as well as the two-sigma bound, so a two-circuit run clears no quantum volume
    assert qv.sigma == pytest.approx(math.sqrt(qv.mean * (1.0 - qv.mean) / qv.n_circuits), rel=1e-12)
    assert qv.sigma > 2.0 * qv.standard_error_of_the_mean
    assert not qv.passed and qv.log2_quantum_volume is None
    assert qv.passed == (qv.threshold_cleared and qv.protocol_circuit_count_met)
    b = qv.budget
    assert (
        b is not None
        and b.predicted["heavy_output_probability"] < b.predicted["ideal_heavy_output_probability"]
    )
    assert 0.0 < b.predicted["eps_gates"] < 0.05 and b.predicted["register_fidelity"] > 0.95
    for qc in qv.circuits:
        assert len(qc.heavy) == 2 and len(qc.layers) == 2
        assert all(u.shape == (4, 4) for _pair, u in qc.layers[0])
        assert native.equal_up_to_global_phase(
            qc.layers[0][0][1] @ qc.layers[0][0][1].conj().T, np.eye(4), atol=1e-9
        )
