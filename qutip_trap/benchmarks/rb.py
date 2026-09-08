"""Randomized benchmarking on the simulated device (PLAN.md Section 10 M10; Section 7.9 "Randomized benchmarking variants";
Section 9.12 row "Wright reproduction"; Section 13 row "RB error rate").

Clifford RB (Magesan et al. 2011; Egan 2021's and Mount 2015's variant in Section 7.9): for every sequence length m and every
random sequence, m uniform Cliffords followed by the inverse of THEIR PRODUCT (never the inverse of each Clifford), compiled
to native gates by the compiler and played as pulses by ``run`` from the prepared |0...0>, measured in the computational basis;
the survival probability is the fraction of shots in which every benchmarked qubit reads 0. The mean survival over the
sequences is fitted to F(m) = A p^m + B (the SPAM-independent decay), with B fixed at 1/2^n when fewer than three lengths are
scanned, and the reported error per Clifford is Section 13's r = (1 - p)(2^n - 1)/2^n (the average error per Clifford RB papers
report: (1 - p)/2 for one qubit, (3/4)(1 - p) for two), beside the entanglement infidelity (4^n - 1)(1 - p)/4^n of the
depolarizing channel with the same p, kept under that name. Single-qubit RB on several qubits at once (``pair=False``)
interleaves independent sequences (simultaneous RB) and fits the JOINT survival P(every benchmarked qubit reads 0), which
exposes the addressing crosstalk that single-ion RB cannot see; two-qubit RB runs the 11520-element group on a pair.

The budget reported alongside (``benchmarks.budget``): the Section 9.6 closed-form scales per Clifford, the Section 6.8
channels of the native gate kinds the sequences compiled to, reduced to the benchmarked qubits and composed to first order into
a predicted r and p, and the SPAM offsets A + B = F(0) that the preparation and readout errors predict.
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
from qutip_trap.benchmarks.clifford import (
    SINGLE_QUBIT_CLIFFORDS,
    TwoQubitClifford,
    decompose_two_qubit_clifford,
    random_single_qubit_clifford,
    random_two_qubit_clifford,
    single_qubit_sequence_unitary,
)
from qutip_trap.control.compiler import Circuit, Operation, decompose_single_qubit
from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.experiments.fitting import weighted_fit
from qutip_trap.noise.summary import depolarizing_entanglement_infidelity, rb_error_per_clifford
from qutip_trap.run.job import last_record, run
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.device.model import Device


@dataclass(frozen=True)
class RBSequence:
    """One random sequence: its Cliffords (indices for one qubit, decompositions for a pair), the inverse and the circuit."""

    length: int
    qubits: tuple[int, ...]
    cliffords: tuple[Any, ...]
    """Per benchmarked qubit a tuple of single-qubit Clifford indices, or one tuple of ``TwoQubitClifford`` for a pair."""
    inverse: Any
    circuit: Circuit
    """The standard/native-gate circuit as built (the compiler compiles and verifies it inside ``run``)."""


def single_qubit_sequences(
    rng: np.random.Generator, qubits: Sequence[int], length: int, n_qubits: int
) -> RBSequence:
    """Independent random single-qubit sequences on ``qubits`` interleaved Clifford by Clifford, each closed by its inverse."""
    qs = tuple(int(q) for q in qubits)
    indices = {q: tuple(random_single_qubit_clifford(rng) for _ in range(length)) for q in qs}
    ops: list[Operation] = []
    for k in range(length):
        for q in qs:
            ops += decompose_single_qubit(SINGLE_QUBIT_CLIFFORDS[indices[q][k]], q)
    inverses: dict[int, np.ndarray] = {}
    for q in qs:
        u = single_qubit_sequence_unitary(indices[q])
        inverses[q] = u.conj().T
        ops += decompose_single_qubit(inverses[q], q)
    return RBSequence(
        length=length,
        qubits=qs,
        cliffords=tuple(indices[q] for q in qs),
        inverse=tuple(inverses[q] for q in qs),
        circuit=Circuit(n_qubits, tuple(ops), qs),
    )


def two_qubit_sequence(
    rng: np.random.Generator, pair: tuple[int, int], length: int, n_qubits: int
) -> RBSequence:
    """A random two-qubit Clifford sequence on ``pair`` closed by the inverse of its product (its class recognized)."""
    cliffords = tuple(random_two_qubit_clifford(rng) for _ in range(length))
    u = np.eye(4, dtype=complex)
    ops: list[Operation] = []
    for c in cliffords:
        ops += c.operations(pair)
        u = c.matrix @ u
    inverse = decompose_two_qubit_clifford(u.conj().T)
    ops += inverse.operations(pair)
    return RBSequence(
        length=length,
        qubits=(int(pair[0]), int(pair[1])),
        cliffords=(cliffords,),
        inverse=inverse,
        circuit=Circuit(n_qubits, tuple(ops), (int(pair[0]), int(pair[1]))),
    )


def rb_model(params: np.ndarray, m: np.ndarray) -> np.ndarray:
    """F(m) = A p^m + B."""
    a, p, b = (float(x) for x in params)
    return np.asarray(a * np.power(p, np.asarray(m, dtype=float)) + b)


def fit_decay(
    lengths: Sequence[int],
    survival: np.ndarray,
    sigma: np.ndarray,
    n_qubits: int,
    *,
    fix_offset: bool | None = None,
) -> tuple[dict[str, tuple[float, float]], bool, tuple[str, ...]]:
    """Fit the mean survival per length to A p^m + B. ``fix_offset`` pins B at 1/2^n (the depolarizing floor, the usual
    remedy when a tiny decay leaves A and B degenerate); None fixes it only when fewer than three lengths are scanned."""
    m = np.asarray(lengths, dtype=float)
    y = np.asarray(survival, dtype=float)
    sg = np.asarray(sigma, dtype=float)
    b0 = 1.0 / 2**n_qubits
    a0 = min(max(float(y[0]) - b0, 1e-3), 1.0 - b0)
    p0 = 0.99
    if len(m) >= 2 and y[0] - b0 > 1e-6 and y[-1] - b0 > 1e-6 and m[-1] > m[0]:
        ratio = (y[-1] - b0) / (y[0] - b0)
        if 0.0 < ratio < 1.0:
            p0 = float(ratio ** (1.0 / (m[-1] - m[0])))
    p0 = min(max(p0, 0.5), 0.99999)
    notes: list[str] = []
    fixed_b = len(m) < 3 if fix_offset is None else bool(fix_offset)
    if fixed_b:

        def model2(params: np.ndarray, x: np.ndarray) -> np.ndarray:
            return rb_model(np.array([params[0], params[1], b0]), x)

        fit = weighted_fit(model2, [a0, p0], m, y, sigma=sg, bounds=([0.0, 0.0], [1.0, 1.0]))
        a, sa = fit.value(0)
        p, sp = fit.value(1)
        b, sb = b0, 0.0
        notes.append(
            f"B fixed at 1/2^n = {b0:g} ("
            + ("fewer than three sequence lengths" if fix_offset is None else "fix_offset=True")
            + ")"
        )
    else:
        fit = weighted_fit(rb_model, [a0, p0, b0], m, y, sigma=sg, bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
        a, sa = fit.value(0)
        p, sp = fit.value(1)
        b, sb = fit.value(2)
    out = {
        "A": (a, sa),
        "p": (p, sp),
        "B": (b, sb),
        "chi2_per_dof": (fit.chi2_per_dof, 0.0),
    }
    return out, fit.converged, tuple(notes)


@dataclass(frozen=True)
class RBResult:
    qubits: tuple[int, ...]
    lengths: tuple[int, ...]
    n_sequences: int
    shots: int
    survival: np.ndarray
    """(n_lengths, n_sequences) survival probabilities P(every benchmarked qubit reads 0)."""
    survival_sigma: np.ndarray
    """Shot-noise standard error per point, from the run's effective sample size."""
    mean_survival: np.ndarray
    mean_sigma: np.ndarray
    fit: dict[str, tuple[float, float]]
    """A, p, B and chi2_per_dof with uncertainties."""
    error_per_clifford: tuple[float, float]
    """r = (1 - p)(2^n - 1)/2^n (Section 13 'RB error rate')."""
    depolarizing_entanglement_infidelity: tuple[float, float]
    """(4^n - 1)(1 - p)/4^n: the entanglement infidelity of the depolarizing channel with the same p (Section 13), NOT r."""
    pulses_per_clifford: float
    entangling_per_clifford: float
    converged: bool
    sequences: tuple[RBSequence, ...]
    results: tuple[Result, ...]
    """The runs, in (length, sequence) order."""
    budget: BenchmarkBudget | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def fidelity_form(self) -> str:
        """The decay in the form Wright et al. 2019 print it, B p^L + 1/2 (Section 9.12), with this run's fitted B."""
        p, _ = self.fit["p"]
        a, _ = self.fit["A"]
        b, _ = self.fit["B"]
        return f"F(L) = {a:.4f} x {p:.5f}^L + {b:.4f}"


def _survival(res: Result, n: int) -> tuple[float, float]:
    key = "0" * n
    p = float(res.probabilities.get(key, 0.0))
    sigma = float(res.error_bars.get(key, math.sqrt(max(p * (1.0 - p), 0.0) / max(res.shots, 1))))
    if sigma <= 0.0:
        sigma = 1.0 / max(res.shots, 1)
    return p, sigma


def randomized_benchmarking(
    device: Device,
    qubits: Sequence[int],
    lengths: Sequence[int],
    *,
    n_sequences: int = 4,
    shots: int = 200,
    seed: int = 0,
    budget: bool = True,
    fix_offset: bool | None = None,
    **run_kwargs: Any,
) -> RBResult:
    """Clifford randomized benchmarking of one qubit (or several at once, simultaneous RB) or of a pair, through ``run``.

    ``lengths`` are the Clifford counts m (the inverse not counted); every (length, sequence) is one ``run`` of ``shots`` with its
    own keyed seed. ``run_kwargs`` go to ``run`` (``table``, ``gate_drives``, ``entangling_drives``, ``options``, ``level``,
    ``noise``, ...). ``budget=True`` adds the Section 6.8 channels of the native gate kinds used (one GATE_LOCAL tomography per
    kind, cached per device) and the predictions composed from them. ``fix_offset`` pins the fit's B at 1/2^n.
    """
    qs = tuple(int(q) for q in qubits)
    if not qs or len(set(qs)) != len(qs):
        raise ValueError(
            "qubits are one or more distinct ions: one qubit, several at once (simultaneous RB) or a pair"
        )
    ls = tuple(int(m) for m in lengths)
    if not ls or any(m < 1 for m in ls) or len(set(ls)) != len(ls):
        raise ValueError("sequence lengths are distinct positive Clifford counts")
    if n_sequences < 1 or shots < 1:
        raise ValueError("n_sequences and shots are positive")
    n_ions = device.crystal.n_ions
    if any(q < 0 or q >= n_ions for q in qs):
        raise ValueError("qubits index the device's ions")
    two_qubit = len(qs) == 2 and bool(run_kwargs.pop("pair", True))
    root = SeedSpec(int(seed))
    rng = np.random.default_rng(root.child(0, 0, 0, 0, "rb_sequences"))
    sequences: list[RBSequence] = []
    results: list[Result] = []
    survival = np.zeros((len(ls), n_sequences))
    sigma = np.zeros_like(survival)
    n_pulses = 0
    n_ent = 0
    n_cliffords = 0
    k_run = 0
    for i, m in enumerate(ls):
        for j in range(n_sequences):
            seq = (
                two_qubit_sequence(rng, (qs[0], qs[1]), m, n_ions)
                if two_qubit
                else single_qubit_sequences(rng, qs, m, n_ions)
            )
            run_seed = int(root.child(0, 0, k_run, 0, "rb_runs").generate_state(1)[0])
            k_run += 1
            res = run(seq.circuit, device, shots, seed=run_seed, **run_kwargs)
            rec = last_record(res)
            n_pulses += rec.compile.n_pulses
            n_ent += rec.compile.n_entangling
            n_cliffords += (m + 1) * (1 if two_qubit else len(qs))
            survival[i, j], sigma[i, j] = _survival(res, len(qs))
            sequences.append(seq)
            results.append(res)
    mean = survival.mean(axis=1)
    if n_sequences > 1:
        between = survival.var(axis=1, ddof=1) / n_sequences
    else:
        between = np.zeros(len(ls))
    within = (sigma**2).mean(axis=1) / n_sequences
    mean_sigma = np.sqrt(between + within)
    mean_sigma = np.where(mean_sigma > 0.0, mean_sigma, 1.0 / (shots * n_sequences))
    n_q = len(qs)
    fit, converged, fit_notes = fit_decay(ls, mean, mean_sigma, n_q, fix_offset=fix_offset)
    p, sp = fit["p"]
    r = rb_error_per_clifford(p, n_q)
    sr = sp * (2**n_q - 1) / 2**n_q
    eps_dep = depolarizing_entanglement_infidelity(p, n_q)
    s_dep = sp * (4**n_q - 1) / 4**n_q
    notes = list(fit_notes)
    units = [(m + 1) * (1 if two_qubit else len(qs)) for m in ls for _ in range(n_sequences)]
    bud: BenchmarkBudget | None = None
    if budget:
        counts, intrinsic = gather_counts_and_intrinsic(results, units)
        spam = spam_of(results[0], qs)
        kinds = [k for k in counts if not k.startswith("total")]
        channels, infid = channels_for(device, kinds, qs, **run_kwargs)
        r_channel = float(sum(counts[k] * infid[k] for k in kinds))
        p_channel = 1.0 - r_channel * 2**n_q / (2**n_q - 1)
        f0 = 1.0
        for q in qs:
            eb, ed = spam.get(f"q{q}", (0.0, 0.0))
            prep = spam.get(f"q{q}.state_preparation", (0.0, 0.0))[0]
            # survival to |0>: the prepared |0> read as 1 (eps_D for a dark |0>) or prepared wrong and read as 0
            f0 *= (1.0 - prep) * (1.0 - ed) + prep * eb
        predicted = {
            "r_channel": r_channel,
            "p_channel": p_channel,
            "r_intrinsic": float(intrinsic.get("total", 0.0)),
            "F0_spam": f0,
            "B_depolarizing": 1.0 / 2**n_q,
            "A_spam": f0 - 1.0 / 2**n_q,
        }
        bud = BenchmarkBudget(
            unit="clifford",
            qubits=qs,
            counts=counts,
            intrinsic=intrinsic,
            spam=spam,
            channels=channels,
            channel_infidelity=infid,
            predicted=predicted,
            notes=(
                "r_channel = sum over native gate kinds of (pieces per Clifford) x (average infidelity of the kind's "
                "GATE_LOCAL channel reduced to the benchmarked qubits), the first-order composition; p_channel = 1 - r 2^n/(2^n - 1)",
                "r_intrinsic = the Section 9.6 closed-form scales per Clifford (bounds on the coherent errors, not their values)",
                "F0_spam = prod_q [(1 - eps_prep)(1 - eps_D) + eps_prep eps_B]: the survival at m = 0 that SPAM alone predicts;"
                " B_depolarizing = 1/2^n",
            ),
        )
    return RBResult(
        qubits=qs,
        lengths=ls,
        n_sequences=n_sequences,
        shots=shots,
        survival=survival,
        survival_sigma=sigma,
        mean_survival=mean,
        mean_sigma=mean_sigma,
        fit=fit,
        error_per_clifford=(r, sr),
        depolarizing_entanglement_infidelity=(eps_dep, s_dep),
        pulses_per_clifford=n_pulses / n_cliffords,
        entangling_per_clifford=n_ent / n_cliffords,
        converged=converged,
        sequences=tuple(sequences),
        results=tuple(results),
        budget=bud,
        notes=tuple(notes),
    )


__all__ = [
    "RBResult",
    "RBSequence",
    "TwoQubitClifford",
    "fit_decay",
    "randomized_benchmarking",
    "rb_model",
    "single_qubit_sequences",
    "two_qubit_sequence",
]
