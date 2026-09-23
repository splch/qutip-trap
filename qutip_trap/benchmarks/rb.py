"""Randomized benchmarking on the simulated device. Clifford RB (Magesan et al. 2011) closes m uniform Cliffords by the
inverse of their product, fits F(m) = A p^m + B and reports r = (1 - p)(2^n - 1)/2^n beside the depolarizing entanglement
infidelity (4^n - 1)(1 - p)/4^n; simultaneous RB (Gambetta et al. 2012) fits each qubit's marginal, so its r is per
Clifford on one qubit; the Knill-style variant (Knill et al. 2008) plays a random Pauli then a random pi/2 Clifford per
computational gate, closes with one pi/2 into the computational basis and fits the authors' B p^L + 1/2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

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
from qutip_trap.control import native
from qutip_trap.control.compiler import Circuit, Operation, decompose_single_qubit
from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.experiments.fitting import weighted_fit
from qutip_trap.noise.summary import depolarizing_entanglement_infidelity, rb_error_per_clifford
from qutip_trap.run.job import last_record
from qutip_trap.run.results import Result

if TYPE_CHECKING:
    from qutip_trap.machine import Machine


@dataclass(frozen=True)
class RBSequence:
    """One random sequence: its Cliffords (indices for one qubit, decompositions for a pair), the inverse and the circuit."""

    length: int
    qubits: tuple[int, ...]
    cliffords: tuple[Any, ...]
    """Per qubit the Clifford indices (Knill: the Pauli and Clifford codes), or one tuple of ``TwoQubitClifford`` for a pair."""
    inverse: Any
    circuit: Circuit
    target_key: str = ""
    """The ideal outcome in histogram order (the lowest-indexed measured qubit rightmost); ``""`` means all zeros."""

    @property
    def key(self) -> str:
        """``target_key``, defaulted to ``"0" * len(qubits)``."""
        return self.target_key or "0" * len(self.qubits)


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


_KNILL_AZIMUTH: Final[tuple[float, ...]] = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
"""The azimuths +-x, +-y of the Knill-style variant: pi/2 and pi rotations about them keep the ideal state on the six Pauli
eigenstates."""


def _knill_final(u: np.ndarray) -> tuple[float | None, int]:
    """(azimuth of the final pi/2 pulse, or None when u|0> is already a basis state; the bit the ideal state then reads)."""
    psi = np.asarray(u, dtype=complex)[:, 0]
    for bit in (0, 1):
        if abs(psi[bit]) > 1.0 - 1e-9:
            return None, bit
    for phi in _KNILL_AZIMUTH:
        out = native.gpi2(phi) @ psi
        for bit in (0, 1):
            if abs(out[bit]) > 1.0 - 1e-9:
                return phi, bit
    raise RuntimeError(
        "no single pi/2 pulse takes the Knill sequence's ideal state to a computational basis state"
    )


def knill_sequences(
    rng: np.random.Generator, qubits: Sequence[int], length: int, n_qubits: int
) -> RBSequence:
    """Knill-style sequences (Knill et al. 2008), one per qubit, interleaved: per computational gate a random Pauli (GPi
    about +-x or +-y, a virtual RZ(+-pi), or nothing) then a GPi2 about +-x or +-y, and one final pi/2 into the
    computational basis where needed; ``target_key`` is the outcome the ideal sequence ends on."""
    qs = tuple(int(q) for q in qubits)
    paulis = {q: tuple(int(rng.integers(0, 7)) for _ in range(length)) for q in qs}
    cliffords = {q: tuple(int(rng.integers(0, 4)) for _ in range(length)) for q in qs}
    ops: list[Operation] = []
    ideal = {q: np.eye(2, dtype=complex) for q in qs}
    for k in range(length):
        for q in qs:
            code = paulis[q][k]
            if code < 4:  # pi about +-x, +-y: one GPi pulse
                phi = _KNILL_AZIMUTH[code]
                ops.append(Operation("gpi", (q,), (phi,)))
                ideal[q] = native.gpi(phi) @ ideal[q]
            elif code < 6:  # +-z: a frame rotation, no pulse
                theta = math.pi if code == 4 else -math.pi
                ops.append(Operation("rz", (q,), (theta,)))
                ideal[q] = native.rz(theta) @ ideal[q]
            # code 6 is the identity Pauli: no operation
            phi_c = _KNILL_AZIMUTH[cliffords[q][k]]
            ops.append(Operation("gpi2", (q,), (phi_c,)))
            ideal[q] = native.gpi2(phi_c) @ ideal[q]
    finals: list[float | None] = []
    bits: dict[int, int] = {}
    for q in qs:
        phi_f, bit = _knill_final(ideal[q])
        finals.append(phi_f)
        bits[q] = bit
        if phi_f is not None:
            ops.append(Operation("gpi2", (q,), (phi_f,)))
    # histogram order: the sorted measured qubits, the lowest-indexed rightmost
    target = "".join(str(bits[q]) for q in sorted(qs)[::-1])
    return RBSequence(
        length=length,
        qubits=qs,
        cliffords=tuple((paulis[q], cliffords[q]) for q in qs),
        inverse=tuple(finals),
        circuit=Circuit(n_qubits, tuple(ops), qs),
        target_key=target,
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
    """Fit the mean survival per length to A p^m + B. ``fix_offset`` pins B at the depolarizing floor 1/2^n; None pins it
    only when fewer than three lengths are scanned."""
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
    """A randomized-benchmarking run: the joint survival per (length, sequence), the A p^m + B fit, the error per Clifford,
    the per-qubit marginals of simultaneous RB, the sequences, every ``Result`` and the budget."""

    qubits: tuple[int, ...]
    lengths: tuple[int, ...]
    n_sequences: int
    shots: int
    variant: str
    """``clifford`` or ``knill``."""
    survival: np.ndarray
    """(n_lengths, n_sequences) joint survival P(every benchmarked qubit reads its target bit)."""
    survival_sigma: np.ndarray
    mean_survival: np.ndarray
    mean_sigma: np.ndarray
    fit: dict[str, tuple[float, float]]
    """A, p, B and chi2_per_dof of the joint survival, with uncertainties."""
    error_per_clifford: tuple[float, float]
    """r = (1 - p)(2^n - 1)/2^n for one qubit or a pair; for simultaneous RB the mean of the marginal r_q = (1 - p_q)/2,
    in the unit of ``budget.predicted['r_channel']``."""
    depolarizing_entanglement_infidelity: tuple[float, float]
    """(4^n - 1)(1 - p)/4^n of the depolarizing channel with the same p (not r); the marginal mean for simultaneous RB."""
    joint_error_per_layer: tuple[float, float] | None
    """Simultaneous RB only: (1 - p)(2^n - 1)/2^n of the joint survival, the error per layer of n Cliffords (sum_q r_q to
    first order), not an n-qubit Clifford error rate."""
    pulses_per_clifford: float
    entangling_per_clifford: float
    converged: bool
    sequences: tuple[RBSequence, ...]
    results: tuple[Result, ...]
    """The runs, in (length, sequence) order."""
    marginal_survival: np.ndarray | None = None
    """(n_qubits, n_lengths, n_sequences) marginal survival P(qubit q reads its target bit); None for a pair."""
    marginal_fit: tuple[dict[str, tuple[float, float]], ...] = ()
    marginal_error_per_clifford: tuple[tuple[float, float], ...] = ()
    """Per benchmarked qubit, r_q = (1 - p_q)/2 from its marginal decay."""
    budget: BenchmarkBudget | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def fidelity_form(self) -> str:
        """The fitted decay as text, ``F(L) = A x p^L + B`` (Wright et al. 2019 print it as B p^L + 1/2)."""
        p, _ = self.fit["p"]
        a, _ = self.fit["A"]
        b, _ = self.fit["B"]
        return f"F(L) = {a:.4f} x {p:.5f}^L + {b:.4f}"


def _survival(res: Result, key: str) -> tuple[float, float]:
    p = float(res.probabilities.get(key, 0.0))
    sigma = float(res.error_bars.get(key, math.sqrt(max(p * (1.0 - p), 0.0) / max(res.shots, 1))))
    if sigma <= 0.0:
        sigma = 1.0 / max(res.shots, 1)
    return p, sigma


def marginal_survival(res: Result, key: str, qubits: Sequence[int]) -> dict[int, tuple[float, float]]:
    """Per benchmarked qubit, (P(that qubit reads its target bit), shot-noise sigma); ``key`` is the target bitstring in
    histogram order (the sorted measured qubits, the lowest-indexed rightmost)."""
    order = sorted(int(q) for q in qubits)[::-1]
    if len(order) != len(key):
        raise ValueError("the target key and the benchmarked qubits disagree in length")
    n_eff = max(float(res.diagnostics.effective_sample_size), 1.0)
    out: dict[int, tuple[float, float]] = {}
    for i, q in enumerate(order):
        p = float(sum(v for k, v in res.probabilities.items() if k[i] == key[i]))
        out[q] = (p, math.sqrt(max(p * (1.0 - p), 0.0) / n_eff) or 1.0 / n_eff)
    return out


def mean_survival_sigma(values: np.ndarray, sigma: np.ndarray, n_sequences: int, shots: int) -> np.ndarray:
    """The standard error of the mean survival per length: the between-sequence variance alone for several sequences (it
    already contains the shot noise), the shot noise for one, and one count of the run when both vanish."""
    if n_sequences > 1:
        var = values.var(axis=1, ddof=1) / n_sequences
    else:
        var = (sigma**2).mean(axis=1)
    out = np.sqrt(var)
    return np.asarray(np.where(out > 0.0, out, 1.0 / (shots * n_sequences)))


def randomized_benchmarking(
    machine: Machine,
    qubits: Sequence[int],
    lengths: Sequence[int],
    *,
    n_sequences: int = 4,
    shots: int = 200,
    seed: int = 0,
    budget: bool = True,
    fix_offset: bool | None = None,
    variant: str = "clifford",
    pair: bool | None = None,
) -> RBResult:
    """Randomized benchmarking of one qubit, several at once (simultaneous RB) or a pair, through ``Machine.run``.

    ``lengths`` count Cliffords m, or computational gates L for ``variant="knill"`` (the closing gate not counted); each
    (length, sequence) is one run of ``shots`` with its own keyed seed. Two qubits run the two-qubit Clifford group unless
    ``pair=False``; ``pair=True`` needs two qubits and the Clifford variant. ``fix_offset`` pins B at 1/2^n (by default
    for Knill)."""
    mach = machine
    device = mach.device
    qs = tuple(int(q) for q in qubits)
    if not qs or len(set(qs)) != len(qs):
        raise ValueError(
            "qubits are one or more distinct ions: one qubit, several at once (simultaneous RB) or a pair"
        )
    if variant not in ("clifford", "knill"):
        raise ValueError(
            "variant is 'clifford' (the Clifford group) or 'knill' (Section 7.9's Knill-style sequences)"
        )
    ls = tuple(int(m) for m in lengths)
    if not ls or any(m < 1 for m in ls) or len(set(ls)) != len(ls):
        raise ValueError("sequence lengths are distinct positive Clifford counts")
    if n_sequences < 1 or shots < 1:
        raise ValueError("n_sequences and shots are positive")
    n_ions = device.crystal.n_ions
    if any(q < 0 or q >= n_ions for q in qs):
        raise ValueError("qubits index the device's ions")
    pair_requested = pair
    if pair_requested is not None and bool(pair_requested) and (len(qs) != 2 or variant != "clifford"):
        raise ValueError(
            "pair=True runs the 11520-element two-qubit Clifford group and needs exactly two qubits and "
            f"variant='clifford'; got {len(qs)} qubit(s) and variant={variant!r}. Pass pair=False for simultaneous "
            "single-qubit RB on several ions"
        )
    two_qubit = variant == "clifford" and len(qs) == 2 and (pair_requested is None or bool(pair_requested))
    n_q = len(qs)
    simultaneous = not two_qubit and n_q > 1
    offset = True if (variant == "knill" and fix_offset is None) else fix_offset
    root = SeedSpec(int(seed))
    rng = np.random.default_rng(root.child(0, 0, 0, 0, "rb_sequences"))
    sequences: list[RBSequence] = []
    results: list[Result] = []
    survival = np.zeros((len(ls), n_sequences))
    sigma = np.zeros_like(survival)
    marginal = np.zeros((n_q, len(ls), n_sequences))
    marginal_sigma = np.zeros_like(marginal)
    n_pulses = 0
    n_ent = 0
    n_cliffords = 0
    k_run = 0

    def per_unit(m: int) -> int:
        """Benchmark units in a sequence of length m: Cliffords (per qubit in single-qubit RB) or Knill computational gates."""
        if variant == "knill":
            return m * n_q  # the closing pi/2 is not a computational gate
        return (m + 1) * (1 if two_qubit else n_q)  # the closing inverse Clifford counts

    for i, m in enumerate(ls):
        for j in range(n_sequences):
            if variant == "knill":
                seq = knill_sequences(rng, qs, m, n_ions)
            elif two_qubit:
                seq = two_qubit_sequence(rng, (qs[0], qs[1]), m, n_ions)
            else:
                seq = single_qubit_sequences(rng, qs, m, n_ions)
            run_seed = int(root.child(0, 0, k_run, 0, "rb_runs").generate_state(1)[0])
            k_run += 1
            res = mach.run(seq.circuit, shots, seed=run_seed)
            rec = last_record(res)
            n_pulses += rec.compile.n_pulses
            n_ent += rec.compile.n_entangling
            n_cliffords += per_unit(m)
            survival[i, j], sigma[i, j] = _survival(res, seq.key)
            if not two_qubit:
                per_q = marginal_survival(res, seq.key, qs)
                for iq, q in enumerate(qs):
                    marginal[iq, i, j], marginal_sigma[iq, i, j] = per_q[q]
            sequences.append(seq)
            results.append(res)
    mean = survival.mean(axis=1)
    mean_sigma = mean_survival_sigma(survival, sigma, n_sequences, shots)
    fit, converged, fit_notes = fit_decay(ls, mean, mean_sigma, n_q, fix_offset=offset)
    notes = list(fit_notes)
    p, sp = fit["p"]
    joint_r = (rb_error_per_clifford(p, n_q), sp * (2**n_q - 1) / 2**n_q)
    marginal_fits: list[dict[str, tuple[float, float]]] = []
    marginal_r: list[tuple[float, float]] = []
    if not two_qubit:
        for iq in range(n_q):
            m_fit, m_ok, _ = fit_decay(
                ls,
                marginal[iq].mean(axis=1),
                mean_survival_sigma(marginal[iq], marginal_sigma[iq], n_sequences, shots),
                1,
                fix_offset=offset,
            )
            converged = converged and m_ok
            pq, spq = m_fit["p"]
            marginal_fits.append(m_fit)
            marginal_r.append((rb_error_per_clifford(pq, 1), 0.5 * spq))
    if simultaneous:
        r = float(np.mean([v for v, _ in marginal_r]))
        sr = float(math.sqrt(sum(s * s for _, s in marginal_r)) / n_q)
        eps_dep = float(np.mean([depolarizing_entanglement_infidelity(f["p"][0], 1) for f in marginal_fits]))
        s_dep = float(math.sqrt(sum((0.75 * f["p"][1]) ** 2 for f in marginal_fits)) / n_q)
        joint: tuple[float, float] | None = joint_r
        notes.append(
            "simultaneous RB (Gambetta et al. 2012): error_per_clifford is the mean over the benchmarked qubits of the"
            " per-qubit MARGINAL r_q = (1 - p_q)/2 (marginal_error_per_clifford), one Clifford on one qubit the unit and"
            " the same unit as budget.predicted['r_channel'] and as the r a qubit measures alone; joint_error_per_layer"
            " = (1 - p)(2^n - 1)/2^n of the JOINT survival is the error per LAYER of n Cliffords (sum_q r_q to first"
            " order), a correlation diagnostic and not an n-qubit Clifford error rate"
        )
    else:
        r, sr = joint_r
        eps_dep = depolarizing_entanglement_infidelity(p, n_q)
        s_dep = sp * (4**n_q - 1) / 4**n_q
        joint = None
    if variant == "knill":
        notes.append(
            "Knill-style variant (Section 7.9): the unit is one computational gate (a random Pauli then a random"
            " Clifford), the closing pi/2 not counted; p is the per-pi/2-gate decay in Wright et al. 2019's convention"
            " and r = (1 - p)/2 its Section 13 conversion"
        )
    units = [per_unit(m) for m in ls for _ in range(n_sequences)]
    bud: BenchmarkBudget | None = None
    if budget:
        counts, intrinsic = gather_counts_and_intrinsic(results, units)
        spam = spam_of(results[0], qs)
        kinds = [k for k in counts if not k.startswith("total")]
        channels, infid = channels_for(mach, kinds, qs)
        r_channel_joint = float(sum(counts[k] * infid[k] for k in kinds))
        per_qubit = (
            {q: float(n_q * sum(counts[k] * channels[k].infidelity_on((q,)) for k in kinds)) for q in qs}
            if simultaneous
            else {}
        )
        r_channel = float(np.mean(list(per_qubit.values()))) if simultaneous else r_channel_joint
        n_channel = 1 if simultaneous else n_q
        p_channel = 1.0 - r_channel * 2**n_channel / (2**n_channel - 1)
        f0 = 1.0
        for q in qs:
            eb, ed = spam.get(f"q{q}", (0.0, 0.0))
            prep = spam.get(f"q{q}.state_preparation", (0.0, 0.0))[0]
            # prepared and read right, or both wrong; Clifford RB ends on the dark |0>, Knill on |0> or |1> equally often
            e_target = 0.5 * (eb + ed) if variant == "knill" else ed
            e_other = 0.5 * (eb + ed) if variant == "knill" else eb
            f0 *= (1.0 - prep) * (1.0 - e_target) + prep * e_other
        predicted = {
            "r_channel": r_channel,
            "p_channel": p_channel,
            "r_intrinsic": float(intrinsic.get("total", 0.0)),
            "F0_spam": f0,
            "B_depolarizing": 1.0 / 2**n_q,
            "A_spam": f0 - 1.0 / 2**n_q,
        }
        budget_notes = [
            "r_channel = sum over native gate kinds of (pieces per unit) x (average infidelity of the kind's GATE_LOCAL "
            "channel reduced to the benchmarked qubits), the first-order composition; p_channel = 1 - r 2^n/(2^n - 1)",
            "r_intrinsic = the Section 9.6 closed-form scales per unit summed over each kind's WHOLE schedule entry, "
            "including the crosstalk it inflicts on ions outside the benchmarked set, so it bounds a LARGER error than "
            "r_channel (which is reduced to the benchmarked qubits) measures, and it bounds the coherent errors rather "
            "than valuing them",
            "F0_spam = prod_q [(1 - eps_prep)(1 - eps_read) + eps_prep eps_read_other]: the survival at m = 0 that SPAM "
            "alone predicts; B_depolarizing = 1/2^n",
        ]
        if simultaneous:
            predicted["r_channel_joint_layer"] = float(n_q * r_channel_joint)
            for q, v in per_qubit.items():
                predicted[f"r_channel.q{q}"] = v
            budget_notes[0] = (
                "r_channel is the mean over the benchmarked qubits of r_channel.q{i} = sum over native gate kinds of "
                "(pieces per Clifford of the sequence) x (average infidelity of the kind's GATE_LOCAL channel reduced to "
                "THAT ONE qubit), a one-qubit average gate infidelity per Clifford: the unit of the marginal r_q that "
                "simultaneous RB fits. r_channel_joint_layer composes the channels reduced to ALL benchmarked qubits over "
                "one layer of n Cliffords, the unit of joint_error_per_layer"
            )
        bud = BenchmarkBudget(
            unit="computational gate" if variant == "knill" else "clifford",
            qubits=qs,
            counts=counts,
            intrinsic=intrinsic,
            spam=spam,
            channels=channels,
            channel_infidelity=infid,
            predicted=predicted,
            notes=tuple(budget_notes),
        )
    return RBResult(
        qubits=qs,
        lengths=ls,
        n_sequences=n_sequences,
        shots=shots,
        variant=variant,
        survival=survival,
        survival_sigma=sigma,
        mean_survival=mean,
        mean_sigma=mean_sigma,
        fit=fit,
        error_per_clifford=(r, sr),
        depolarizing_entanglement_infidelity=(eps_dep, s_dep),
        joint_error_per_layer=joint,
        pulses_per_clifford=n_pulses / n_cliffords,
        entangling_per_clifford=n_ent / n_cliffords,
        converged=converged,
        sequences=tuple(sequences),
        results=tuple(results),
        marginal_survival=None if two_qubit else marginal,
        marginal_fit=tuple(marginal_fits),
        marginal_error_per_clifford=tuple(marginal_r),
        budget=bud,
        notes=tuple(notes),
    )


__all__ = [
    "RBResult",
    "RBSequence",
    "TwoQubitClifford",
    "fit_decay",
    "knill_sequences",
    "marginal_survival",
    "mean_survival_sigma",
    "randomized_benchmarking",
    "rb_model",
    "single_qubit_sequences",
    "two_qubit_sequence",
]
