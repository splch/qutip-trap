"""The M10 defects the 2026-09-07 audit found, each pinned by the identity or the published criterion it broke.

1. Simultaneous RB reported the JOINT decay's (1 - p)(2^n - 1)/2^n -- the error per LAYER of n Cliffords -- beside a
   budget composed per single Clifford, a factor n apart, and the ledger recorded that mismatch as a physical "factor 2"
   (errors adding in amplitude). Here: the per-qubit marginal fit of Gambetta et al. 2012 recovers r_q exactly, the joint
   fit recovers sum_q r_q, and two INDEPENDENT qubits at r_q = r_channel already reproduce the committed joint decay to
   3.5 %, so there is no factor 2 to explain (`anchor.m10.simultaneous_rb_crosstalk`, `conv.simultaneous_rb_units`).
2. The quantum-volume confidence was the standard error of the mean, four times smaller than Cross et al. 2019 Appendix C
   Eq. (32) and decision-changing at the protocol's own circuit count, and `passed` ignored the 100-circuit requirement
   (`conv.heavy_output_criterion`).
3. (P_0 + P_1 + C)/2 was called a lower bound on the fidelity printed beside it; it is exactly
   max_theta <GHZ_theta| rho |GHZ_theta> and therefore an UPPER bound on any fixed-phase GHZ fidelity
   (`conv.ghz_parity_bound`).
4. `pair=` never reached its `pop` on any qubit count but two and leaked into `run`; the intrinsic budget lost the
   Section 9.6 scales of every gate whose own id contains a slash (the ZZ wrapper); the Section 9.10 depolarizing
   conversions were unimplemented.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.benchmarks.budget import (
    gate_piece_of,
    intrinsic_by_kind,
    kinds_of_schedule,
    reduced_ideal,
)
from qutip_trap.benchmarks.ghz import fit_parity
from qutip_trap.benchmarks.rb import (
    fit_decay,
    knill_sequences,
    marginal_survival,
    mean_survival_sigma,
    randomized_benchmarking,
)
from qutip_trap.benchmarks.volume import (
    CONFIDENCE_Z,
    HEAVY_OUTPUT_THRESHOLD,
    PROTOCOL_CIRCUITS,
    cross_confidence_sigma,
    heavy_output_pass,
)
from qutip_trap.control import native
from qutip_trap.control.compiler import CNOT_MATRIX, Circuit, Operation, ideal_probabilities
from qutip_trap.noise.summary import (
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_choi,
    depolarizing_kraus,
    entanglement_infidelity,
    qiskit_depolarizing_lambda,
    rb_error_per_clifford,
)

# the committed check_benchmarks.out numbers these tests hold the code to
COMMITTED_SIMULTANEOUS_JOINT_P = 0.999080
COMMITTED_SIMULTANEOUS_R_CHANNEL = 3.426e-4
COMMITTED_QV_H = (0.795, 0.6475, 0.820, 0.6025)


# ---- 1. simultaneous RB: the marginal and the joint decay, and their units --------------------------------------------------


def test_marginal_fit_recovers_the_per_qubit_rate_and_the_joint_fit_their_sum() -> None:
    """Two independent qubits, each decaying as 0.5 p_q^m + 0.5 with r_q = (1 - p_q)/2. The marginal fit returns r_q; the
    JOINT fit's (1 - p)(2^n - 1)/2^n returns sum_q r_q to first order, so it is an error per LAYER of n Cliffords and a
    factor n away from any per-Clifford rate. Reporting the joint number beside a per-Clifford composition is the unit
    mismatch the audit found, not a physical factor 2."""
    ls = (1, 128, 512)
    for r_q in (2.0e-5, 1.0e-4, 3.426e-4):
        p_s = 1.0 - 2.0 * r_q
        marginal = np.array([0.5 * p_s**m + 0.5 for m in ls])
        joint = marginal**2
        m_fit, m_ok, _ = fit_decay(ls, marginal, np.full(3, 1e-6), 1, fix_offset=True)
        assert m_ok and rb_error_per_clifford(m_fit["p"][0], 1) == pytest.approx(r_q, rel=1e-6)
        j_fit, j_ok, _ = fit_decay(ls, joint, np.full(3, 1e-6), 2, fix_offset=True)
        joint_r = rb_error_per_clifford(j_fit["p"][0], 2)
        assert j_ok and joint_r == pytest.approx(2.0 * r_q, rel=0.07)
        assert joint_r > 1.8 * r_q, "the joint decay is the SUM over the qubits, never a per-Clifford rate"


def test_the_committed_simultaneous_decay_is_the_first_order_composition_without_a_factor_two() -> None:
    """The withdrawn finding said the measured joint decay was TWICE the first-order composition and read that as
    crosstalk errors adding in amplitude. Feeding the composition's own r_channel = 3.426e-4 into two INDEPENDENT qubits
    reproduces the committed joint fit p = 0.999080 to 3.5 % in (1 - p): the composition already predicts it."""
    ls = (1, 128, 512)
    p_s = 1.0 - 2.0 * COMMITTED_SIMULTANEOUS_R_CHANNEL
    joint = np.array([(0.5 * p_s**m + 0.5) ** 2 for m in ls])
    fit, ok, _ = fit_decay(ls, joint, np.full(3, 1e-6), 2, fix_offset=True)
    assert ok
    assert fit["p"][0] == pytest.approx(0.999112, abs=5e-7), fit["p"]
    model_decay = 1.0 - fit["p"][0]
    measured_decay = 1.0 - COMMITTED_SIMULTANEOUS_JOINT_P
    assert abs(measured_decay - model_decay) / measured_decay < 0.05
    # and the committed per-layer r stays far above the isolated one: the surviving claim
    assert 0.75 * measured_decay / (2.0 * 2.0e-5) > 15.0


def test_marginal_survival_reads_the_histogram_in_the_section_13_bit_order() -> None:
    """The histogram key sorts the measured qubits and puts the lowest-indexed one RIGHTMOST
    (``conv.result_bit_order``), so a marginal read off position i belongs to the (n - 1 - i)-th sorted qubit."""

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
    """The between-sequence sample variance already contains each point's shot noise; the old form added the
    within-sequence variance to it and inflated the error bar by up to sqrt 2 when shot noise dominated."""
    values = np.array([[0.90, 0.92, 0.88]])
    sigma = np.full((1, 3), 0.01)
    got = mean_survival_sigma(values, sigma, 3, 100)
    assert got[0] == pytest.approx(float(values[0].std(ddof=1)) / math.sqrt(3.0), rel=1e-12)
    double_counted = math.sqrt(float(values[0].var(ddof=1)) / 3.0 + float((sigma[0] ** 2).mean()) / 3.0)
    assert double_counted > got[0]
    # at one sequence there is no between-sequence variance and the shot noise is the whole error bar
    one = mean_survival_sigma(values[:, :1], sigma[:, :1], 1, 100)
    assert one[0] == pytest.approx(0.01, rel=1e-12)
    # shot-noise-dominated: the double count is exactly sqrt 2 too large
    flat = np.full((1, 4), 0.5)
    assert mean_survival_sigma(flat, np.full((1, 4), 0.02), 4, 100)[0] == pytest.approx(
        1.0 / (100 * 4), rel=1e-12
    )


# ---- 2. quantum volume: Cross et al. Appendix C Eq. (32) and the hundred circuits -------------------------------------------


def test_cross_confidence_sigma_reduces_appendix_c_equation_32_exactly() -> None:
    """Eq. (32) is [n_h - z sqrt(n_h (n_s - n_h/n_c))]/(n_c n_s) > 2/3; with n_h = h n_c n_s it is h - z sqrt(h(1-h)/n_c)
    identically, so the criterion's sigma is the per-CIRCUIT binomial one, not the standard error of the mean."""
    for n_c, n_s, h in ((4, 400, 0.7163), (100, 400, 0.7163), (7, 33, 0.61), (100, 1000, 0.68)):
        n_h = h * n_c * n_s
        equation_32 = (n_h - CONFIDENCE_Z * math.sqrt(n_h * (n_s - n_h / n_c))) / (n_c * n_s)
        assert h - CONFIDENCE_Z * cross_confidence_sigma(h, n_c) == pytest.approx(equation_32, abs=1e-12)
    with pytest.raises(ValueError):
        cross_confidence_sigma(0.7, 0)


def test_the_standard_error_of_the_mean_would_pass_a_run_cross_et_al_reject() -> None:
    """The committed four-circuit run's own spread, extrapolated to the protocol's 100 circuits: the standard error of
    the mean clears 2/3 (0.6943) while Eq. (32) does not (0.6261), so the old sigma was decision-changing."""
    h = np.array(COMMITTED_QV_H)
    mean = float(h.mean())
    assert mean == pytest.approx(0.71625, abs=1e-9)
    sigma_cross_4 = cross_confidence_sigma(mean, 4)
    sem_4 = math.sqrt(float(h.var(ddof=1)) / 4.0)
    assert sigma_cross_4 == pytest.approx(0.225408, abs=1e-6)
    assert sigma_cross_4 / sem_4 == pytest.approx(4.20, abs=0.02)
    for n_c in (4, 100):
        sem = math.sqrt(float(h.var(ddof=1)) / n_c)
        cross = cross_confidence_sigma(mean, n_c)
        assert cross > sem
        if n_c == 100:
            assert mean - CONFIDENCE_Z * sem > HEAVY_OUTPUT_THRESHOLD
            assert mean - CONFIDENCE_Z * cross < HEAVY_OUTPUT_THRESHOLD
            assert mean - CONFIDENCE_Z * cross == pytest.approx(0.626087, abs=1e-6)


def test_a_pass_needs_the_confidence_bound_and_the_hundred_circuits() -> None:
    sigma, cleared, met, passed = heavy_output_pass(0.71625, 4)
    assert sigma == pytest.approx(cross_confidence_sigma(0.71625, 4), rel=1e-12)
    assert not cleared and not met and not passed
    # a machine well above threshold still clears nothing below the protocol's circuit count
    _s, cleared_99, met_99, passed_99 = heavy_output_pass(0.85, PROTOCOL_CIRCUITS - 1)
    assert cleared_99 and not met_99 and not passed_99
    _s, cleared_100, met_100, passed_100 = heavy_output_pass(0.85, PROTOCOL_CIRCUITS)
    assert cleared_100 and met_100 and passed_100


# ---- 3. the GHZ bound sits on the phase-optimised fidelity ------------------------------------------------------------------


def _ghz_theta_rho(n: int, theta: float, mix: float = 0.0) -> np.ndarray:
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[-1] = np.exp(1j * theta) / math.sqrt(2.0)
    rho = np.outer(psi, psi.conj())
    return (1.0 - mix) * rho + mix * np.eye(2**n, dtype=complex) / 2**n


def _kron_n(u: np.ndarray, n: int) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for _ in range(n):
        out = np.kron(out, u)
    return out


def _parity_under_analysis(rho: np.ndarray, n: int, phi: float) -> float:
    u = _kron_n(native.gpi2(phi), n)
    z = _kron_n(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), n)
    return float(np.real(np.trace(z @ u @ rho @ u.conj().T)))


def test_the_parity_bound_equals_the_phase_optimised_ghz_fidelity_and_exceeds_the_fixed_phase_one() -> None:
    """(P_0 + P_1 + C)/2 = (rho_00 + rho_11)/2 + |rho_{0...0,1...1}| = max_theta <GHZ_theta| rho |GHZ_theta> exactly,
    because the fitted contrast is 2|rho_{0...0,1...1}|. Against a FIXED-phase GHZ state the same number is therefore an
    upper bound, which the committed three-ion run shows (bound 0.9935 against the exact 0.98520)."""
    for n in (2, 3, 4):
        phases = np.linspace(0.0, 2.0 * math.pi / n, 8, endpoint=False)
        for theta, mix in ((0.0, 0.0), (0.7, 0.0), (math.pi, 0.0), (0.4, 0.15), (2.0, 0.05)):
            rho = _ghz_theta_rho(n, theta, mix)
            parity = np.array([_parity_under_analysis(rho, n, float(p)) for p in phases])
            fit, ok = fit_parity(phases, parity, np.full(8, 1e-9), n)
            assert ok
            contrast = fit["contrast"][0]
            assert contrast == pytest.approx(2.0 * abs(rho[0, -1]), abs=1e-8), (n, theta, mix)
            bound = 0.5 * (float(np.real(rho[0, 0])) + float(np.real(rho[-1, -1])) + contrast)
            # the phase-optimised fidelity, both in closed form and by scanning theta
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
            # and the fixed-phase (theta = 0) fidelity is never above it
            g0 = np.array([1.0, *([0.0] * (2**n - 2)), 1.0]) / math.sqrt(2.0)
            f_fixed = float(np.real(np.vdot(g0, rho @ g0)))
            assert f_fixed <= f_max + 1e-12


def test_the_audit_table_of_bound_minus_fixed_phase_fidelity() -> None:
    """The three rows the audit printed for the ideal three-ion |GHZ_theta>: contrast 1, bound 1, and a fixed-phase
    fidelity cos^2(theta/2) that the bound exceeds by 0, 0.117579 and 1.0000 at theta = 0, 0.7 and pi."""
    n = 3
    phases = np.linspace(0.0, 2.0 * math.pi / n, 8, endpoint=False)
    for theta, gap in ((0.0, 0.0), (0.7, 0.117579), (math.pi, 1.0)):
        rho = _ghz_theta_rho(n, theta)
        parity = np.array([_parity_under_analysis(rho, n, float(p)) for p in phases])
        fit, ok = fit_parity(phases, parity, np.full(8, 1e-9), n)
        assert ok and fit["contrast"][0] == pytest.approx(1.0, abs=1e-8)
        bound = 0.5 * (1.0 + fit["contrast"][0])
        f_fixed = math.cos(0.5 * theta) ** 2
        assert bound == pytest.approx(1.0, abs=1e-8)
        assert bound - f_fixed == pytest.approx(gap, abs=1e-6)
    # the committed three-ion run: the bound is 0.008 ABOVE the fixed-phase fidelity it was compared with
    assert 0.9935 - 0.98520 == pytest.approx(0.0083, abs=1e-4)


# ---- 4. the pair short-circuit, the multi-piece budget keys and the Section 9.10 conversions --------------------------------


def test_gate_piece_of_finds_multi_piece_gate_ids_by_longest_prefix() -> None:
    """``kinds_of_schedule`` keys on ``Schedule.target.gate_id``, and the ZZ wrapper's ids contain a slash of their own
    (``zz[k]/ms``, ``zz[k]/loop1``, ``zz[k]/loop2``), so ``key.split("/")[0]`` yielded ``zz[k]``, which is no key at all,
    and every intrinsic entry of those pieces was dropped from the per-kind budget."""
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
    # the old rule would have looked "zz[2]" up and found nothing
    assert "zz[2]" not in kinds


def test_reduced_ideal_factors_a_step_s_target_instead_of_taking_its_zero_block() -> None:
    """A GATE_LOCAL step's ideal is a product over the step's ions of that ion's own target, so the factor on the
    benchmarked qubits is what the reduced channel must be compared with. The <0_rest| U |0_rest> block this took before
    is that factor scaled by prod_rest <0|U_rest|0>: correct only when the traced-out ions' ideal is the identity, off by
    1/sqrt 2 for a GPi2 on the other ion and exactly zero for a GPi -- which is why asking for the crosstalk channel a
    NEIGHBOUR's pulse inflicts on a benchmarked qubit raised 'not the identity on the traced-out ions'."""
    from qutip_trap.control.two_qubit import global_phase, haar_random_unitary

    eye = np.eye(2, dtype=complex)
    a, b = native.gpi2(0.3), native.gpi(1.1)
    rng = np.random.default_rng(5)
    haar = haar_random_unitary(rng, 2)
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
    # three factors, and a subset of two
    u3 = np.kron(np.kron(a, eye), b)
    assert global_phase(reduced_ideal(u3, 3, [0, 1]), np.kron(a, eye), atol=1e-8) is not None
    assert global_phase(reduced_ideal(u3, 3, [1]), eye, atol=1e-8) is not None
    assert np.max(np.abs(reduced_ideal(np.kron(a, b), 2, [0, 1]) - np.kron(a, b))) < 1e-12
    # an entangling ideal does not factor, and the reduction refuses rather than inventing one
    with pytest.raises(ValueError, match="factor"):
        reduced_ideal(CNOT_MATRIX, 2, [0])


def test_depolarizing_kraus_normalizations_and_the_qiskit_lambda_conversion() -> None:
    """Section 9.10 row "Depolarizing conversions": sum K^dag K = 1 for the p/3 and p/15 weights, the entanglement
    infidelity is p, the average gate infidelity (2/3)p and (4/5)p, and Qiskit's lambda = p 4^n/(4^n - 1) turns Chen et
    al.'s Forte medians 2.0e-4 and 46.4e-4 into 2.67e-4 and 49.5e-4."""
    for n, weight in ((1, 3), (2, 15)):
        d = 2**n
        for p in (0.0, 1e-4, 0.1, 1.0):
            kraus = depolarizing_kraus(p, n)
            assert len(kraus) == 4**n
            total = sum(k.conj().T @ k for k in kraus)
            assert np.max(np.abs(total - np.eye(d))) < 1e-12, (n, p)
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


def test_knill_sequences_end_on_their_own_target_and_cost_one_pulse_per_gate() -> None:
    """Section 7.9's first RB variant: pi/2 gates with an interleaved pi or identity and one final pi/2. The ideal state
    never leaves the six Pauli eigenstates, so the sequence's own target bitstring carries the whole ideal probability."""
    rng = np.random.default_rng(11)
    for length in (1, 3, 9, 24):
        for qubits in ((0,), (0, 1), (0, 1, 2), (2, 0)):
            seq = knill_sequences(rng, qubits, length, 3)
            probs = ideal_probabilities(seq.circuit)
            assert probs.get(seq.key, 0.0) == pytest.approx(1.0, abs=1e-9), (length, qubits, seq.key)
            assert seq.circuit.measure == tuple(int(q) for q in qubits)
            assert set(seq.key) <= {"0", "1"} and len(seq.key) == len(qubits)
            pulses = [op for op in seq.circuit.ops if op.name in ("gpi", "gpi2")]
            # per qubit: length Clifford pulses, at most length Pauli pulses, at most one closing pi/2
            assert len(qubits) * length <= len(pulses) <= len(qubits) * (2 * length + 1)
            assert all(op.name in ("gpi", "gpi2", "rz") for op in seq.circuit.ops)
    # every target bit is reachable
    keys = {knill_sequences(rng, (0,), 5, 1).key for _ in range(40)}
    assert keys == {"0", "1"}


# ---- the device paths -------------------------------------------------------------------------------------------------------


def test_pair_true_is_refused_on_any_qubit_count_but_two() -> None:
    """``two_qubit = len(qs) == 2 and bool(run_kwargs.pop("pair", True))`` short-circuited past the pop, so ``pair``
    stayed in the kwargs splatted into ``run``, which has no such parameter: ``pair=False`` on three ions -- the natural
    use of a variant whose point is chain crosstalk -- raised TypeError. The pop is now unconditional and an explicit
    ``pair=True`` on the wrong qubit count is refused up front."""
    from qutip_trap.api import yb171_chain

    device = yb171_chain(2).device
    for qubits in ((0,), (0, 1)):
        with pytest.raises(ValueError, match="two qubits"):
            randomized_benchmarking(device, qubits, (1,), pair=True, variant="knill")
    with pytest.raises(ValueError, match="two qubits"):
        randomized_benchmarking(device, (0,), (1,), pair=True)
    with pytest.raises(ValueError, match="variant"):
        randomized_benchmarking(device, (0,), (1,), variant="direct")


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    from qutip_trap.api import SolverOptions, yb171_chain
    from qutip_trap.calibration.surrogate import surrogate_table

    preset = yb171_chain(2)
    windows = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
    sur = surrogate_table(
        preset.device,
        pairs=[(0, 1)],
        gate_drives=preset.gate_drives,
        entangling_drives=preset.entangling_drives,
        detection_records=800,
        detection_windows_s=windows,
    )
    kw = dict(table=sur.table, options=SolverOptions(branch_weight_min=1e-3), **preset.run_kwargs())
    return preset, kw


@pytest.mark.slow
def test_simultaneous_rb_reports_marginals_in_the_budget_s_own_unit(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Gambetta et al. 2012 on the two-ion example device: the per-qubit marginal r_q, the per-qubit composition
    r_channel.q{i} (each kind's channel reduced to THAT ONE qubit) in the same unit, and the joint decay beside them as
    the correlation diagnostic, in the unit of r_channel_joint_layer."""
    preset, kw = two_ion
    rb = randomized_benchmarking(
        preset.device, (0, 1), (1, 32, 128), n_sequences=1, shots=600, pair=False, fix_offset=True, **kw
    )
    assert rb.n_qubits == 2 and rb.variant == "clifford"
    assert rb.marginal_survival is not None and rb.marginal_survival.shape == (2, 3, 1)
    assert len(rb.marginal_fit) == 2 and len(rb.marginal_error_per_clifford) == 2
    r, sr = rb.error_per_clifford
    assert r == pytest.approx(float(np.mean([v for v, _ in rb.marginal_error_per_clifford])), rel=1e-12)
    assert 1e-4 < r < 1e-3, r
    joint = rb.joint_error_per_layer
    assert joint is not None and joint[0] > 1.4 * r, "the joint decay is the sum over the two qubits"
    b = rb.budget
    assert b is not None
    assert set(b.counts) == {"gpi2[0]", "gpi2[1]", "gpi[0]", "gpi[1]"}
    per_qubit = [b.predicted["r_channel.q0"], b.predicted["r_channel.q1"]]
    assert b.predicted["r_channel"] == pytest.approx(float(np.mean(per_qubit)), rel=1e-12)
    # the per-qubit composition is the marginal's own unit, so the two are the same size; three lengths to 128 at 600
    # shots on one sequence is a deliberately cheap fit, so the band here is a factor three and the quantitative
    # agreement (r/r_channel and joint/r_channel_joint_layer both within a factor two of one) is pinned on the check
    # script's much longer run, validation/scripts/outputs/check_benchmarks.out
    assert b.predicted["r_channel"] / 3.0 < r < 3.0 * b.predicted["r_channel"], (r, b.predicted)
    # the joint-layer composition is exactly n_q times the composition reduced to ALL benchmarked qubits: the joint
    # decay's unit, and necessarily above the per-qubit one (a two-qubit average gate infidelity per layer against a
    # one-qubit one per Clifford)
    assert b.predicted["r_channel_joint_layer"] == pytest.approx(
        2.0 * sum(b.counts[k] * b.channel_infidelity[k] for k in b.channel_infidelity), rel=1e-12
    )
    assert b.predicted["r_channel_joint_layer"] > b.predicted["r_channel"]
    assert b.predicted["r_channel_joint_layer"] / 4.0 < joint[0] < 4.0 * b.predicted["r_channel_joint_layer"]
    # and the measured joint is the measured marginals SUMMED, which is the whole point of the two units
    assert 1.4 * r < joint[0] < 3.0 * r
    assert any("Gambetta" in n for n in rb.notes)


@pytest.mark.slow
def test_simultaneous_rb_runs_on_three_ions(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The path the `pair` short-circuit made unreachable: simultaneous single-qubit RB on a three-ion chain, whose
    middle ion sees crosstalk from both sides."""
    from qutip_trap.api import SolverOptions, yb171_chain

    del two_ion
    preset = yb171_chain(3, address_waist_m=2.0e-6)
    rb = randomized_benchmarking(
        preset.device,
        (0, 1, 2),
        (1, 8),
        n_sequences=1,
        shots=200,
        pair=False,
        budget=False,
        fix_offset=True,
        options=SolverOptions(branch_weight_min=3e-3),
        **preset.run_kwargs(),
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
def test_the_intrinsic_budget_keeps_the_zz_wrapper_s_section_9_6_scales(two_ion) -> None:  # type: ignore[no-untyped-def]
    """With ``entangler="zz"`` the CNOT template's pieces carry gate ids containing a slash (``zz[0]/ms``,
    ``zz[0]/loop1``, ``zz[0]/loop2``), whose scales the old ``key.split("/")[0]`` lookup dropped from every per-kind
    budget."""
    from qutip_trap.run.job import last_record, run

    preset, kw = two_ion
    circuit = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    res = run(circuit, preset.device, 1, entangler="zz", **kw)
    schedule = last_record(res).schedule
    kinds = kinds_of_schedule(schedule)
    assert any("/" in gid for gid in kinds), kinds
    by_kind = intrinsic_by_kind(res.diagnostics, kinds)
    assert by_kind, "the per-kind budget is not empty"
    orphans = [
        k
        for k in res.diagnostics.intrinsic_budget
        if k != "total" and gate_piece_of(k, kinds) is None and not k.startswith(("prep", "readout"))
    ]
    assert not orphans, orphans
    # the wrapper's own pieces contribute, not only the bare kinds
    wrapper_keys = [k for k in res.diagnostics.intrinsic_budget if k.startswith("zz[")]
    assert wrapper_keys, list(res.diagnostics.intrinsic_budget)
    assert sum(by_kind.values()) > 0.0
