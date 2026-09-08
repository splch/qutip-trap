"""Benchmark emulation (PLAN.md milestone M10; Sections 7.9, 9.12, 13 'RB error rate', 6.8): randomized benchmarking, GHZ
fidelity and a quantum-volume style run on the two-ion example device with the simulator's own budget alongside, plus the
algebra behind them (the decay fit, the channel reduction, the parity fit, the heavy outputs) on synthetic inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import (
    SolverOptions,
    ghz_fidelity,
    quantum_volume,
    randomized_benchmarking,
    yb171_chain,
)
from qutip_trap.benchmarks.budget import BenchmarkBudget, one_gate_circuit, reduced_choi
from qutip_trap.benchmarks.ghz import fit_parity, ghz_circuit, parity_circuit
from qutip_trap.benchmarks.rb import fit_decay, rb_model
from qutip_trap.benchmarks.volume import random_square_circuit
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control import native
from qutip_trap.control.compiler import compile_with_report, ideal_probabilities
from qutip_trap.control.two_qubit import haar_random_unitary
from qutip_trap.noise.summary import (
    apply_choi,
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_choi,
    entanglement_infidelity,
)

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = SolverOptions(branch_weight_min=1e-3)


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    preset = yb171_chain(2)
    sur = surrogate_table(
        preset.device,
        pairs=[(0, 1)],
        gate_drives=preset.gate_drives,
        entangling_drives=preset.entangling_drives,
        detection_records=1000,
        detection_windows_s=WINDOWS,
    )
    return preset, dict(table=sur.table, options=FAST, **preset.run_kwargs())


# ---- algebra on synthetic inputs -------------------------------------------------------------------------------------------


def test_decay_fit_recovers_p_and_the_section_13_conversions() -> None:
    lengths = (1, 16, 64, 256)
    a, p, b = 0.49, 0.9995, 0.5
    y = rb_model(np.array([a, p, b]), np.array(lengths, dtype=float))
    fit, ok, notes = fit_decay(lengths, y, np.full(4, 1e-4), 1)
    assert ok and fit["p"][0] == pytest.approx(p, abs=1e-6) and fit["B"][0] == pytest.approx(b, abs=1e-4)
    fit2, ok2, notes2 = fit_decay(lengths[:2], y[:2], np.full(2, 1e-4), 1)
    assert ok2 and "fixed" in notes2[0] and fit2["p"][0] == pytest.approx(p, abs=1e-6)
    fit3, _ok3, notes3 = fit_decay(lengths, y, np.full(4, 1e-4), 1, fix_offset=True)
    assert "fix_offset" in notes3[0] and fit3["B"] == (0.5, 0.0)
    from qutip_trap.noise.summary import depolarizing_entanglement_infidelity, rb_error_per_clifford

    assert rb_error_per_clifford(p, 1) == pytest.approx((1 - p) / 2)
    assert rb_error_per_clifford(p, 2) == pytest.approx(0.75 * (1 - p))
    assert depolarizing_entanglement_infidelity(p, 1) == pytest.approx(0.75 * (1 - p))
    assert depolarizing_entanglement_infidelity(p, 2) == pytest.approx(15.0 / 16.0 * (1 - p))


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


def test_parity_fit_and_ghz_circuits() -> None:
    n = 3
    phases = np.linspace(0.0, 2 * math.pi / n, 8, endpoint=False)
    par = 0.93 * np.cos(n * phases + 0.4) + 0.01
    fit, ok = fit_parity(phases, par, np.full(8, 1e-3), n)
    assert (
        ok
        and fit["contrast"][0] == pytest.approx(0.93, abs=1e-6)
        and fit["phi0_rad"][0] == pytest.approx(0.4, abs=1e-6)
    )
    circ = ghz_circuit((0, 1, 2), 3)
    assert ideal_probabilities(circ) == pytest.approx({"000": 0.5, "111": 0.5})
    # the analysis pulses turn the GHZ state's parity into cos(N phi + phi_0) with unit contrast; phi_0 carries the rotation
    # axis (N pi/2) and the compiled circuit's residual frames, so the parity at phi = 0 is not the maximum in general
    ideal_parity = []
    for phi in phases:
        p = ideal_probabilities(parity_circuit((0, 1, 2), 3, float(phi)))
        ideal_parity.append(sum((-1) ** k.count("1") * v for k, v in p.items()))
    fit_ideal, ok_ideal = fit_parity(phases, np.array(ideal_parity), np.full(8, 1e-6), n)
    assert ok_ideal and fit_ideal["contrast"][0] == pytest.approx(1.0, abs=1e-6)
    assert fit_ideal["offset"][0] == pytest.approx(0.0, abs=1e-6)
    rep = compile_with_report(parity_circuit((0, 1), 2, 0.7))
    assert rep.n_entangling == 1 and rep.circuit_residual is not None and rep.circuit_residual < 1e-8


def test_random_square_circuits_have_heavy_sets_of_half_the_strings_and_three_gates_per_su4() -> None:
    rng = np.random.default_rng(4)
    for n in (2, 3):
        qc = random_square_circuit(rng, tuple(range(n)), n, n)
        assert len(qc.heavy) == 2 ** (n - 1) and 0.5 < qc.heavy_ideal_probability <= 1.0
        assert abs(sum(qc.ideal.values()) - 1.0) < 1e-9 and qc.entangling_count == 3 * n * (n // 2)
        rep = compile_with_report(qc.circuit)
        assert rep.n_entangling == qc.entangling_count and rep.circuit_residual is not None
        assert rep.circuit_residual < 1e-7
    assert one_gate_circuit("ms[0,1]", 2).ops[0].params == (0.0, 0.0, math.pi / 2)
    assert one_gate_circuit("gpi2[1]", 2).measure == (1,)
    with pytest.raises(ValueError):
        one_gate_circuit("cnot[0,1]", 2)


# ---- the protocols on the example device ------------------------------------------------------------------------------------


def test_single_qubit_rb_decays_at_the_channel_scale_with_the_budget_alongside(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 13 'RB error rate': r = (1 - p)/2 from the fit; the survival decays over hundreds of Cliffords at the 1e-5 level
    the Section 6.8 channel of the carrier pulses predicts (the first-order composition is an estimate, coherent errors of one
    Clifford's pulses partly cancel); the SPAM offset F(0) matches the readout and preparation errors."""
    preset, kw = two_ion
    rb = randomized_benchmarking(
        preset.device, (0,), (1, 128, 512), n_sequences=1, shots=2000, fix_offset=True, **kw
    )
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
    assert b.predicted["r_intrinsic"] > r_channel, "the Section 9.6 scales bound the coherent errors"
    f0 = b.predicted["F0_spam"]
    assert 0.999 < f0 < 1.0 and abs(rb.mean_survival[0] - f0) < 3.0 * rb.mean_sigma[0] + 2e-4
    assert "first-order composition" in b.notes[0]
    d = rb.results[0].diagnostics
    assert d.level == "JOINT_EXACT" and d.propagator_cache_hits >= 0
    assert rb.results[0].n_qubits == 1 and rb.sequences[0].circuit.measure == (0,)


@pytest.mark.slow
def test_two_qubit_rb_and_the_entangling_channel(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Two-qubit Clifford RB on the pair: 1.5 entangling gates per Clifford on average, r = (3/4)(1 - p) at the 1e-3 level of
    the composed channels (five pulses per Clifford at 2.6e-4 each on the pair dominate the 1.1e-4 entangling gate)."""
    preset, kw = two_ion
    rb = randomized_benchmarking(
        preset.device, (0, 1), (1, 6, 16), n_sequences=1, shots=400, fix_offset=True, **kw
    )
    assert rb.n_qubits == 2 and rb.fit["B"] == (0.25, 0.0)
    assert rb.mean_survival[0] > 0.97 and rb.mean_survival[-1] < rb.mean_survival[0]
    assert 0.5 <= rb.entangling_per_clifford <= 3.0 and rb.pulses_per_clifford > rb.entangling_per_clifford
    r, sr = rb.error_per_clifford
    assert 1e-4 < r < 3e-2
    b = rb.budget
    assert b is not None and "ms[0,1]" in b.channels
    ms = b.channels["ms[0,1]"].steps[0]
    assert ms.local_dimension > 100 and ms.ions == (0, 1) and ms.engine_runs >= 16
    assert 1e-5 < ms.summary.average_gate_infidelity < 1e-3
    assert ms.summary.depolarizing_rate == pytest.approx(1.25 * ms.summary.average_gate_infidelity, rel=1e-6)
    assert 0.2 * b.predicted["r_channel"] < r < 5.0 * b.predicted["r_channel"] + 3.0 * sr, (r, b.predicted)


@pytest.mark.slow
def test_ghz_fidelity_bound_and_exact_register_fidelity(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.9: the Bell/GHZ bound (P_00 + P_11 + C)/2 from populations and a parity scan through run(), and the two
    exact fidelities it sits between -- it estimates max_theta <GHZ_theta| rho |GHZ_theta> (an identity, conv.ghz_parity_bound)
    and is therefore an UPPER bound on the fixed-phase <GHZ| rho |GHZ> -- plus the predicted floor from the channels."""
    preset, kw = two_ion
    g = ghz_fidelity(preset.device, (0, 1), shots=500, **kw)
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
    assert abs(f - g.register_fidelity_max_phase) < 3.0 * sf + 2e-3, (
        "the measured bound estimates the max-phase fidelity",
        f,
        sf,
        g.register_fidelity_max_phase,
    )
    assert f + 3.0 * sf > g.register_fidelity, "and so bounds the fixed-phase fidelity from ABOVE"
    assert any("UPPER bound" in n for n in g.notes)
    b = g.budget
    assert b is not None and b.unit == "circuit" and "ms[0,1]" in b.channels and b.counts["ms[0,1]"] == 1.0
    assert 0.99 < b.predicted["F_gates"] < 1.0 and b.predicted["fidelity_bound"] < b.predicted["F_gates"]
    assert abs(b.predicted["F_gates"] - g.register_fidelity) < 5e-3
    assert g.parity.shape == (8, 3) and np.all(np.abs(g.parity[:, 1]) <= 1.0)


@pytest.mark.slow
def test_quantum_volume_style_run_at_width_two(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Cross et al. 2019 at width two: heavy outputs from the ideal distribution, the measured heavy-output probability near the
    ideal one on this device, three entangling gates per SU(4), the depolarizing prediction from the channels."""
    preset, kw = two_ion
    qv = quantum_volume(preset.device, (0, 1), n_circuits=2, shots=300, **kw)
    assert qv.depth == 2 and qv.n_circuits == 2 and qv.entangling_per_circuit == 6.0
    assert np.all(qv.ideal_heavy_probability > 0.5) and np.all(qv.register_fidelity > 0.95)
    assert np.all(
        np.abs(qv.heavy_output_probability - qv.ideal_heavy_probability) < 4.0 * qv.heavy_sigma + 0.05
    )
    assert not qv.protocol_circuit_count_met and any("100" in n for n in qv.notes)
    # Cross et al. Appendix C Eq. (32): sigma = sqrt(h(1 - h)/n_c), far above the standard error of the mean, and a pass
    # needs the 100 circuits as well as the two-sigma bound, so a two-circuit run clears no quantum volume
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
