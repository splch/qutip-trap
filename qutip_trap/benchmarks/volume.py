"""A quantum-volume style run on the simulated device (PLAN.md Section 10 M10; Section 7.9 "Application-level"; Section 4.6
"Schedule, wall clock and the circuit-level error model" for the four-qubit quantum-volume circuit Pino et al. 2021 ran).

The protocol of Cross, Bishop, Sheldon, Nation and Gambetta (PRA 100, 032328, 2019): a square circuit of width n and depth d
= n, every layer a uniformly random permutation of the qubits followed by a Haar-random SU(4) on each of the floor(n/2)
pairs it defines; the heavy outputs of a circuit are the bitstrings whose IDEAL probability exceeds the median of the ideal
distribution; the circuit's heavy-output probability h_U is the measured fraction of shots landing on them, and the width
passes when the mean over circuits of h_U exceeds 2/3 with two-sigma confidence AND at least the protocol's 100 circuits were
run ("we set n_c >= 100 circuits"); log2 of the quantum volume is then n. The confidence is Appendix C Eq. (32) of the paper,

    [n_h - z sqrt(n_h (n_s - n_h/n_c))] / (n_c n_s) > 2/3   with z = 2,

n_h the total heavy count, n_c the circuits and n_s the shots per circuit, which with h = n_h/(n_c n_s) reduces exactly to
mean - 2 sigma > 2/3 with the per-CIRCUIT binomial sigma = sqrt(h(1 - h)/n_c) (``cross_confidence_sigma``). The standard error
of the mean over circuits is much smaller (the between-circuit spread of h_U sits far below h(1 - h)) and is reported as a
separate diagnostic, ``QVResult.standard_error_of_the_mean``, never as the criterion. A run with fewer than 100 circuits says
so, reports the same statistics and clears no quantum volume (``threshold_cleared`` without ``passed``). Every SU(4) is compiled by the
KAK decomposition of ``control.two_qubit`` (three Moelmer-Soerensen gates in the Weyl chamber) with the local unitaries
between layers merged per qubit, and every circuit is one ``run`` of ``shots``. The simulator adds the exact register
fidelity of every circuit.

The budget alongside: the Section 9.6 scales per circuit and, under a depolarizing composition, the predicted heavy-output
probability h_pred = h_ideal (1 - eps) + eps/2 with eps the circuit's composed error (the heavy set holds half the strings, so
a fully depolarized output lands on it with probability 1/2), from the Section 6.8 channels of the full-angle native gate
kinds (every partially entangling MS charged at the full gate's channel, stated as such) and the readout errors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    channels_for,
    gather_counts_and_intrinsic,
    spam_of,
)
from qutip_trap.control.compiler import Circuit, Operation, decompose_single_qubit, ideal_probabilities
from qutip_trap.control.two_qubit import canonical_operations, haar_random_unitary, kak_decomposition
from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.run.job import last_record, register_fidelity, run
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

HEAVY_OUTPUT_THRESHOLD = 2.0 / 3.0
PROTOCOL_CIRCUITS = 100
"""Cross et al. 2019, Appendix C: "we set n_c >= 100 circuits"; a run with fewer clears no quantum volume."""
CONFIDENCE_Z = 2.0
"""The z of Eq. (32): a two-sigma one-sided confidence bound."""


def heavy_output_pass(mean_heavy: float, n_circuits: int) -> tuple[float, bool, bool, bool]:
    """(sigma, threshold_cleared, protocol_circuit_count_met, passed) of Cross et al. 2019's criterion: Eq. (32)'s sigma,
    whether mean - 2 sigma clears 2/3, whether the run has the protocol's 100 circuits, and the pass, which needs BOTH."""
    sigma = cross_confidence_sigma(mean_heavy, n_circuits)
    cleared = float(mean_heavy) - CONFIDENCE_Z * sigma > HEAVY_OUTPUT_THRESHOLD
    circuits_met = n_circuits >= PROTOCOL_CIRCUITS
    return sigma, cleared, circuits_met, (cleared and circuits_met)


def cross_confidence_sigma(mean_heavy: float, n_circuits: int) -> float:
    """sigma = sqrt(h(1 - h)/n_c), the per-circuit binomial confidence of Cross et al. 2019 Appendix C Eq. (32).

    Eq. (32) reads [n_h - z sqrt(n_h (n_s - n_h/n_c))]/(n_c n_s) > 2/3 with n_h the total heavy count over n_c circuits of
    n_s shots; substituting n_h = h n_c n_s turns it into h - z sqrt(h(1 - h)/n_c) > 2/3 identically, so this is the sigma
    the criterion is built on (and what the Qiskit Experiments reference implementation computes). It is NOT the standard
    error of the mean over circuits, which is smaller by the ratio of the between-circuit spread of h_U to h(1 - h).
    """
    if n_circuits < 1:
        raise ValueError("the confidence needs at least one circuit")
    h = float(mean_heavy)
    return math.sqrt(max(h * (1.0 - h), 0.0) / n_circuits)


@dataclass(frozen=True)
class QVCircuit:
    """One random square circuit: its layers (pairs and SU(4)s), the standard/native-gate circuit and its heavy outputs."""

    qubits: tuple[int, ...]
    depth: int
    layers: tuple[tuple[tuple[tuple[int, int], np.ndarray], ...], ...]
    """Per layer, ((pair, SU(4) matrix), ...); the matrix acts with the pair's first qubit as the first factor."""
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
        mat = pending[q]
        if np.max(np.abs(mat - np.eye(2))) > 1e-12:
            ops.extend(decompose_single_qubit(mat, q))
        pending[q] = np.eye(2, dtype=complex)

    for _layer in range(depth):
        perm = [qs[int(k)] for k in rng.permutation(n)]
        pairs = [(perm[2 * k], perm[2 * k + 1]) for k in range(n // 2)]
        layer: list[tuple[tuple[int, int], np.ndarray]] = []
        for pair in pairs:
            u = haar_random_unitary(rng, 4)
            kak = kak_decomposition(u)
            c, d = kak.right
            a, b = kak.left
            pending[pair[0]] = c @ pending[pair[0]]
            pending[pair[1]] = d @ pending[pair[1]]
            flush(pair[0])
            flush(pair[1])
            ops.extend(canonical_operations(*kak.coefficients, pair))
            n_ent += kak.entangling_count
            pending[pair[0]] = a
            pending[pair[1]] = b
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
    n_eff = max(float(result.diagnostics.effective_sample_size), 1.0)
    return h, math.sqrt(max(h * (1.0 - h), 0.0) / n_eff) or 1.0 / n_eff


@dataclass(frozen=True)
class QVResult:
    """A quantum-volume run (Cross et al. 2019) on the simulated device: the heavy-output probability of every circuit with
    its ideal value, the mean with the paper's Eq. (32) confidence ``sigma`` (the criterion) beside the standard error of
    the mean (a diagnostic), the two halves of the pass (``threshold_cleared`` and the 100-circuit count), the exact
    register fidelity per circuit, the circuits, their ``Result`` records and the budget alongside."""

    qubits: tuple[int, ...]
    depth: int
    n_circuits: int
    shots: int
    heavy_output_probability: np.ndarray
    """Per circuit."""
    heavy_sigma: np.ndarray
    ideal_heavy_probability: np.ndarray
    mean: float
    sigma: float
    """The confidence of Cross et al. 2019 Appendix C Eq. (32), sqrt(h(1 - h)/n_c) with h = ``mean`` (the per-circuit
    binomial form the criterion is built on), from :func:`cross_confidence_sigma`."""
    standard_error_of_the_mean: float
    """A diagnostic beside it, never the criterion: the sample standard error of the mean of h_U over the circuits (the
    between-circuit variance alone when more than one circuit is run, which already contains the shot noise; the shot
    noise alone at one circuit). Much smaller than ``sigma`` because the between-circuit spread of h_U sits far below
    h(1 - h)."""
    threshold: float
    threshold_cleared: bool
    """mean - 2 sigma > 2/3 with Eq. (32)'s sigma: the confidence half of the criterion, on its own."""
    passed: bool
    """The protocol's pass: ``threshold_cleared and protocol_circuit_count_met`` (Cross et al. 2019 require the two-sigma
    bound AND at least 100 circuits)."""
    protocol_circuit_count_met: bool
    register_fidelity: np.ndarray
    """Exact <psi_ideal| rho |psi_ideal> per circuit (simulator only)."""
    entangling_per_circuit: float
    pulses_per_circuit: float
    circuits: tuple[QVCircuit, ...]
    results: tuple[Result, ...]
    budget: BenchmarkBudget | None = None
    notes: tuple[str, ...] = ()

    @property
    def log2_quantum_volume(self) -> int | None:
        """The width when the run passes the whole protocol -- the confidence bound AND the 100 circuits (the quantum
        volume is then 2^width) -- None otherwise."""
        return len(self.qubits) if self.passed else None


def quantum_volume(
    device: Device,
    qubits: Sequence[int],
    *,
    n_circuits: int = 4,
    shots: int = 200,
    depth: int | None = None,
    seed: int = 0,
    budget: bool = True,
    **run_kwargs: Any,
) -> QVResult:
    """The quantum-volume style run of the module docstring on ``qubits`` (two or more), depth = width by default."""
    qs = tuple(int(q) for q in qubits)
    if len(qs) < 2 or len(set(qs)) != len(qs):
        raise ValueError("a quantum-volume circuit needs at least two distinct qubits")
    n_ions = device.crystal.n_ions
    if any(q < 0 or q >= n_ions for q in qs):
        raise ValueError("qubits index the device's ions")
    if n_circuits < 1 or shots < 1:
        raise ValueError("n_circuits and shots are positive")
    d = len(qs) if depth is None else int(depth)
    if d < 1:
        raise ValueError("depth is positive")
    root = SeedSpec(int(seed))
    rng = np.random.default_rng(root.child(0, 0, 0, 0, "qv_circuits"))
    kw = {k: v for k, v in run_kwargs.items() if k != "keep_final_state"}
    circuits: list[QVCircuit] = []
    results: list[Result] = []
    h = np.zeros(n_circuits)
    sg = np.zeros(n_circuits)
    fid = np.zeros(n_circuits)
    n_pulses = 0
    n_ent = 0
    for k in range(n_circuits):
        qc = random_square_circuit(rng, qs, d, n_ions)
        run_seed = int(root.child(0, 0, k, 0, "qv_runs").generate_state(1)[0])
        res = run(qc.circuit, device, shots, seed=run_seed, keep_final_state=True, **kw)
        rec = last_record(res)
        n_pulses += rec.compile.n_pulses
        n_ent += rec.compile.n_entangling
        h[k], sg[k] = heavy_output_probability(res, qc.heavy)
        fid[k] = register_fidelity(res)
        circuits.append(qc)
        results.append(res)
    mean = float(h.mean())
    sigma, threshold_cleared, circuits_met, passed = heavy_output_pass(mean, n_circuits)
    # the between-circuit sample variance already contains each circuit's shot noise, so the two are never added
    sem = math.sqrt(float(h.var(ddof=1)) / n_circuits if n_circuits > 1 else float((sg**2).mean()))
    notes: list[str] = []
    if not circuits_met:
        notes.append(
            f"{n_circuits} circuits: the protocol of Cross et al. 2019 asks for at least {PROTOCOL_CIRCUITS} "
            f"('we set n_c >= 100 circuits'), so this run clears no quantum volume whatever its confidence bound "
            f"(threshold_cleared = {threshold_cleared} at mean - 2 sigma = {mean - CONFIDENCE_Z * sigma:.4f} with "
            f"Eq. (32)'s sigma = {sigma:.4f})"
        )
    bud: BenchmarkBudget | None = None
    if budget:
        counts, intrinsic = gather_counts_and_intrinsic(results, [1] * n_circuits)
        spam = spam_of(results[0], qs)
        kinds = [k for k in counts if not k.startswith("total")]
        channels, infid = channels_for(device, kinds, qs, **kw)
        eps_gates = 1.0 - float(np.prod([(1.0 - infid[k]) ** counts[k] for k in kinds])) if kinds else 0.0
        eps_ro = sum(0.5 * (spam.get(f"q{q}", (0.0, 0.0))[0] + spam.get(f"q{q}", (0.0, 0.0))[1]) for q in qs)
        eps = 1.0 - (1.0 - eps_gates) * (1.0 - eps_ro)
        ideal_h = np.array([c.heavy_ideal_probability for c in circuits])
        h_pred = float(np.mean(ideal_h * (1.0 - eps) + 0.5 * eps))
        predicted = {
            "eps_gates": eps_gates,
            "eps_readout": eps_ro,
            "eps_circuit": eps,
            "heavy_output_probability": h_pred,
            "register_fidelity": 1.0 - eps_gates,
            "ideal_heavy_output_probability": float(ideal_h.mean()),
            "intrinsic_total": float(intrinsic.get("total", 0.0)),
        }
        bud = BenchmarkBudget(
            unit="circuit",
            qubits=qs,
            counts=counts,
            intrinsic=intrinsic,
            spam=spam,
            channels=channels,
            channel_infidelity=infid,
            predicted=predicted,
            notes=(
                "eps_gates = 1 - prod over the circuit's native pieces of (1 - reduced average infidelity of the kind's GATE_LOCAL"
                " channel at the FULL entangling angle): every partially entangling MS is charged at the full gate's channel",
                "heavy_output_probability = mean over circuits of h_ideal (1 - eps_circuit) + eps_circuit/2, the depolarizing"
                " composition (a depolarized output lands on the heavy half of the strings with probability 1/2)",
                "intrinsic_total = the Section 9.6 closed-form scales per circuit at the played amplitudes, summed over"
                " every native piece's whole schedule entry (the crosstalk on ions outside the benchmarked set included),"
                " so it bounds a larger error than eps_gates measures",
            ),
        )
    return QVResult(
        qubits=qs,
        depth=d,
        n_circuits=n_circuits,
        shots=shots,
        heavy_output_probability=h,
        heavy_sigma=sg,
        ideal_heavy_probability=np.array([c.heavy_ideal_probability for c in circuits]),
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
        budget=bud,
        notes=tuple(notes),
    )


__all__ = [
    "CONFIDENCE_Z",
    "HEAVY_OUTPUT_THRESHOLD",
    "PROTOCOL_CIRCUITS",
    "QVCircuit",
    "QVResult",
    "cross_confidence_sigma",
    "heavy_output_pass",
    "heavy_output_probability",
    "quantum_volume",
    "random_square_circuit",
]
