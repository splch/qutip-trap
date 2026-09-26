"""A quantum-volume style run on the simulated device (PLAN.md Section 7.9 "Application-level").

The protocol of Cross, Bishop, Sheldon, Nation and Gambetta (PRA 100, 032328, 2019): a square circuit of width n and depth
d = n, every layer a uniformly random permutation of the qubits followed by a Haar-random SU(4) on each of the floor(n/2)
pairs it defines; the heavy outputs are the bitstrings whose ideal probability exceeds the median of the ideal
distribution, and a circuit's heavy-output probability h_U is the fraction of shots landing on them. The width passes when
the mean h over the circuits clears 2/3 with two-sigma confidence AND at least the protocol's 100 circuits were run. The
confidence is Appendix C Eq. (32),

    [n_h - z sqrt(n_h (n_s - n_h/n_c))] / (n_c n_s) > 2/3   with z = 2,

n_h the total heavy count over n_c circuits of n_s shots, which with h = n_h/(n_c n_s) reduces exactly to mean - 2 sigma >
2/3 with the per-circuit binomial sigma = sqrt(h(1 - h)/n_c) (``cross_confidence_sigma``); the much smaller standard error
of the mean over circuits is reported beside it as a diagnostic, never as the criterion. Every SU(4) is compiled by the KAK
decomposition of ``control.two_qubit`` (three Moelmer-Soerensen gates) with the local unitaries between layers merged per
qubit, every circuit is one ``run``, and the simulator adds the exact register fidelity of each.

The budget alongside: the intrinsic scales per circuit and the depolarizing prediction h_pred = h_ideal (1 - eps) + eps/2
(the heavy set holds half the strings) with eps composed from the channels of the full-angle native gate kinds (every
partially entangling MS charged at the full gate's channel) and the readout errors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    _shot_sigma,
    channels_for,
    gather_counts_and_intrinsic,
    spam_of,
)
from qutip_trap.control.compiler import Circuit, Operation, decompose_single_qubit, ideal_probabilities
from qutip_trap.control.two_qubit import canonical_operations, haar_random_unitary, kak_decomposition
from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.run.job import last_record, register_fidelity
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.machine import Machine

HEAVY_OUTPUT_THRESHOLD = 2.0 / 3.0
PROTOCOL_CIRCUITS = 100
"""Cross et al. 2019, Appendix C: "we set n_c >= 100 circuits"; a run with fewer clears no quantum volume."""
CONFIDENCE_Z = 2.0
"""The z of Eq. (32): a two-sigma one-sided confidence bound."""


def cross_confidence_sigma(mean_heavy: float, n_circuits: int) -> float:
    """sigma = sqrt(h(1 - h)/n_c), the per-circuit binomial confidence of Cross et al. 2019 Appendix C Eq. (32), to which
    the equation reduces identically with n_h = h n_c n_s (as the Qiskit Experiments reference implementation computes)."""
    if n_circuits < 1:
        raise ValueError("the confidence needs at least one circuit")
    h = float(mean_heavy)
    return math.sqrt(max(h * (1.0 - h), 0.0) / n_circuits)


def heavy_output_pass(mean_heavy: float, n_circuits: int) -> tuple[float, bool, bool, bool]:
    """(sigma, threshold_cleared, protocol_circuit_count_met, passed) of Cross et al. 2019's criterion: Eq. (32)'s sigma,
    whether mean - 2 sigma clears 2/3, whether the run has the protocol's 100 circuits, and the pass, which needs both."""
    sigma = cross_confidence_sigma(mean_heavy, n_circuits)
    cleared = float(mean_heavy) - CONFIDENCE_Z * sigma > HEAVY_OUTPUT_THRESHOLD
    circuits_met = n_circuits >= PROTOCOL_CIRCUITS
    return sigma, cleared, circuits_met, (cleared and circuits_met)


@dataclass(frozen=True)
class QVCircuit:
    """One random square circuit: per layer ((pair, SU(4) matrix with the pair's first qubit the first factor), ...), the
    standard/native-gate circuit, its ideal distribution, heavy outputs and their ideal probability, and its entangling
    gate count."""

    qubits: tuple[int, ...]
    depth: int
    layers: tuple[tuple[tuple[tuple[int, int], np.ndarray], ...], ...]
    circuit: Circuit
    ideal: dict[str, float]
    heavy: frozenset[str]
    heavy_ideal_probability: float
    entangling_count: int


def random_square_circuit(
    rng: np.random.Generator, qubits: Sequence[int], depth: int, n_qubits: int
) -> QVCircuit:
    """A random square circuit on ``qubits`` with adjacent single-qubit unitaries merged per qubit."""
    qs = [int(q) for q in qubits]
    n = len(qs)
    pending: dict[int, np.ndarray] = {q: np.eye(2, dtype=complex) for q in qs}
    ops: list[Operation] = []
    layers: list[tuple[tuple[tuple[int, int], np.ndarray], ...]] = []
    n_ent = 0

    def flush(q: int) -> None:
        if np.max(np.abs(pending[q] - np.eye(2))) > 1e-12:
            ops.extend(decompose_single_qubit(pending[q], q))
        pending[q] = np.eye(2, dtype=complex)

    for _layer in range(depth):
        perm = [qs[int(k)] for k in rng.permutation(n)]
        layer: list[tuple[tuple[int, int], np.ndarray]] = []
        for pair in [(perm[2 * k], perm[2 * k + 1]) for k in range(n // 2)]:
            u = haar_random_unitary(rng, 4)
            kak = kak_decomposition(u)
            pending[pair[0]] = kak.right[0] @ pending[pair[0]]
            pending[pair[1]] = kak.right[1] @ pending[pair[1]]
            flush(pair[0])
            flush(pair[1])
            ops.extend(canonical_operations(*kak.coefficients, pair))
            n_ent += kak.entangling_count
            pending[pair[0]], pending[pair[1]] = kak.left
            layer.append((pair, u))
        layers.append(tuple(layer))
    for q in qs:
        flush(q)
    circuit = Circuit(n_qubits, tuple(ops), tuple(qs))
    ideal = ideal_probabilities(circuit)
    keys = ["".join(str((i >> (n - 1 - k)) & 1) for k in range(n)) for i in range(2**n)]
    probs = np.array([ideal.get(k, 0.0) for k in keys])
    median = float(np.median(probs))
    heavy = frozenset(k for k, p in zip(keys, probs) if p > median)
    return QVCircuit(
        qubits=tuple(qs),
        depth=depth,
        layers=tuple(layers),
        circuit=circuit,
        ideal={k: float(p) for k, p in zip(keys, probs)},
        heavy=heavy,
        heavy_ideal_probability=float(sum(ideal.get(k, 0.0) for k in heavy)),
        entangling_count=n_ent,
    )


def heavy_output_probability(result: Result, heavy: frozenset[str]) -> tuple[float, float]:
    """(fraction of shots on the heavy outputs, sigma from the effective sample size)."""
    h = float(sum(result.probabilities.get(k, 0.0) for k in heavy))
    return h, _shot_sigma(result, h * (1.0 - h))


@dataclass(frozen=True)
class QVPrediction:
    """What the channels and the readout predict for the quantum-volume circuits (``BenchmarkBudget.predicted``; the
    budget's notes give the rules)."""

    eps_gates: float
    """1 - the product over the circuits' native pieces of (1 - the kind's reduced channel infidelity)."""
    eps_readout: float
    """sum over the qubits of (eps_B + eps_D)/2."""
    eps_circuit: float
    """1 - (1 - eps_gates)(1 - eps_readout)."""
    heavy_output_probability: float
    """The depolarizing composition, mean over circuits of h_ideal (1 - eps_circuit) + eps_circuit/2."""
    register_fidelity: float
    """1 - eps_gates."""
    ideal_heavy_output_probability: float
    """The circuits' mean ideal heavy-output probability."""


@dataclass(frozen=True)
class QVResult:
    """A quantum-volume run (Cross et al. 2019): per circuit the heavy-output probability, its sigma and its ideal value;
    the mean with Eq. (32)'s ``sigma`` (the criterion) and the standard error of the mean (a diagnostic: the
    between-circuit sample variance, which already contains the shot noise, or the shot noise at one circuit); the two
    halves of the pass; the exact register fidelity and the pulse counts per circuit; the circuits, their results and
    the budget alongside."""

    qubits: tuple[int, ...]
    depth: int
    n_circuits: int
    shots: int
    heavy_output_probability: np.ndarray
    heavy_sigma: np.ndarray
    ideal_heavy_probability: np.ndarray
    mean: float
    sigma: float
    standard_error_of_the_mean: float
    threshold: float
    threshold_cleared: bool
    """mean - 2 sigma > 2/3 with Eq. (32)'s sigma."""
    passed: bool
    """``threshold_cleared and protocol_circuit_count_met``."""
    protocol_circuit_count_met: bool
    register_fidelity: np.ndarray
    entangling_per_circuit: float
    pulses_per_circuit: float
    circuits: tuple[QVCircuit, ...]
    results: tuple[Result, ...]
    budget: BenchmarkBudget[QVPrediction] | None = None
    notes: tuple[str, ...] = ()

    @property
    def log2_quantum_volume(self) -> int | None:
        """The width when the run passes the whole protocol (the quantum volume is then 2^width), None otherwise."""
        return len(self.qubits) if self.passed else None


def quantum_volume(
    machine: Machine,
    qubits: Sequence[int],
    *,
    n_circuits: int = 4,
    shots: int = 200,
    depth: int | None = None,
    seed: int = 0,
    budget: bool = True,
) -> QVResult:
    """The quantum-volume style run of the module docstring on ``qubits`` (two or more), depth = width by default, every
    circuit a ``Machine.run``."""
    qs = tuple(int(q) for q in qubits)
    if len(qs) < 2 or len(set(qs)) != len(qs):
        raise ValueError("a quantum-volume circuit needs at least two distinct qubits")
    n_ions = machine.device.crystal.n_ions
    if any(q < 0 or q >= n_ions for q in qs):
        raise ValueError("qubits index the device's ions")
    if n_circuits < 1 or shots < 1:
        raise ValueError("n_circuits and shots are positive")
    d = len(qs) if depth is None else int(depth)
    if d < 1:
        raise ValueError("depth is positive")
    root = SeedSpec(int(seed))
    rng = np.random.default_rng(root.child(0, 0, 0, 0, "qv_circuits"))
    circuits: list[QVCircuit] = []
    results: list[Result] = []
    h = np.zeros(n_circuits)
    sg = np.zeros(n_circuits)
    fid = np.zeros(n_circuits)
    n_pulses = n_ent = 0
    for k in range(n_circuits):
        qc = random_square_circuit(rng, qs, d, n_ions)
        run_seed = int(root.child(0, 0, k, 0, "qv_runs").generate_state(1)[0])
        res = machine.run(qc.circuit, shots, seed=run_seed, keep_final_state=True)
        rec = last_record(res)
        n_pulses += rec.compile.n_pulses
        n_ent += rec.compile.n_entangling
        h[k], sg[k] = heavy_output_probability(res, qc.heavy)
        fid[k] = register_fidelity(res)
        circuits.append(qc)
        results.append(res)
    mean = float(h.mean())
    sigma, threshold_cleared, circuits_met, passed = heavy_output_pass(mean, n_circuits)
    sem = math.sqrt(float(h.var(ddof=1)) / n_circuits if n_circuits > 1 else float((sg**2).mean()))
    notes: list[str] = []
    if not circuits_met:
        notes.append(
            f"{n_circuits} circuits: the protocol of Cross et al. 2019 asks for at least {PROTOCOL_CIRCUITS} "
            f"('we set n_c >= 100 circuits'), so this run clears no quantum volume whatever its confidence bound "
            f"(threshold_cleared = {threshold_cleared} at mean - 2 sigma = {mean - CONFIDENCE_Z * sigma:.4f} with "
            f"Eq. (32)'s sigma = {sigma:.4f})"
        )
    ideal_h = np.array([c.heavy_ideal_probability for c in circuits])
    return QVResult(
        qubits=qs,
        depth=d,
        n_circuits=n_circuits,
        shots=shots,
        heavy_output_probability=h,
        heavy_sigma=sg,
        ideal_heavy_probability=ideal_h,
        mean=mean,
        sigma=sigma,
        standard_error_of_the_mean=sem,
        threshold=HEAVY_OUTPUT_THRESHOLD,
        threshold_cleared=bool(threshold_cleared),
        passed=bool(passed),
        protocol_circuit_count_met=circuits_met,
        register_fidelity=fid,
        entangling_per_circuit=n_ent / n_circuits,
        pulses_per_circuit=n_pulses / n_circuits,
        circuits=tuple(circuits),
        results=tuple(results),
        budget=_qv_budget(machine, results, qs, ideal_h) if budget else None,
        notes=tuple(notes),
    )


def _qv_budget(
    machine: Machine, results: Sequence[Result], qs: tuple[int, ...], ideal_h: np.ndarray
) -> BenchmarkBudget[QVPrediction]:
    """The QV budget: the depolarizing prediction of the heavy-output probability from the channels and the readout."""
    counts, intrinsic = gather_counts_and_intrinsic(results, [1] * len(results))
    spam = spam_of(results[0], qs)
    kinds = list(counts)
    channels, infid = channels_for(machine, kinds, qs)
    eps_gates = 1.0 - float(np.prod([(1.0 - infid[k]) ** counts[k] for k in kinds])) if kinds else 0.0
    eps_ro = sum(0.5 * (spam.get(f"q{q}", (0.0, 0.0))[0] + spam.get(f"q{q}", (0.0, 0.0))[1]) for q in qs)
    eps = 1.0 - (1.0 - eps_gates) * (1.0 - eps_ro)
    return BenchmarkBudget(
        unit="circuit",
        qubits=qs,
        counts=counts,
        intrinsic=intrinsic,
        spam=spam,
        channels=channels,
        channel_infidelity=infid,
        predicted=QVPrediction(
            eps_gates=eps_gates,
            eps_readout=eps_ro,
            eps_circuit=eps,
            heavy_output_probability=float(np.mean(ideal_h * (1.0 - eps) + 0.5 * eps)),
            register_fidelity=1.0 - eps_gates,
            ideal_heavy_output_probability=float(ideal_h.mean()),
        ),
        notes=(
            "eps_gates = 1 - prod over the circuit's native pieces of (1 - reduced average infidelity of the kind's GATE_LOCAL"
            " channel at the FULL entangling angle): every partially entangling MS is charged at the full gate's channel",
            "heavy_output_probability = mean over circuits of h_ideal (1 - eps_circuit) + eps_circuit/2, the depolarizing"
            " composition (a depolarized output lands on the heavy half of the strings with probability 1/2)",
            "intrinsic_total = the closed-form intrinsic scales per circuit at the played amplitudes, summed over"
            " every native piece's whole schedule entry (the crosstalk on ions outside the benchmarked set included),"
            " so it bounds a larger error than eps_gates measures",
        ),
    )
