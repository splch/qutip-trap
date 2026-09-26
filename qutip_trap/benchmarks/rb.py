"""Randomized benchmarking on the simulated device (PLAN.md Section 7.9).

Clifford RB (Magesan et al. 2011): for every sequence length m and random sequence, m uniform Cliffords followed by the
inverse of their product, compiled to native gates and played as pulses by ``run`` from |0...0>; the survival is the
fraction of shots in which every benchmarked qubit reads 0. The mean survival is fitted to F(m) = A p^m + B, with B fixed
at 1/2^n below three lengths, and the error per Clifford is r = (1 - p)(2^n - 1)/2^n, beside the entanglement
infidelity (4^n - 1)(1 - p)/4^n of the depolarizing channel with the same p.

Single-qubit RB on several qubits at once (``pair=False``, simultaneous RB; Gambetta et al., PRL 109, 240504, 2012) fits
every qubit's own marginal survival and reports its r_q (``error_per_clifford`` the mean over the qubits), in the unit of
the budget's ``r_channel``: one Clifford on one qubit. The joint survival's (1 - p)(2^n - 1)/2^n is the error per LAYER of
n Cliffords (``joint_error_per_layer``), the correlation diagnostic, never an n-qubit Clifford error rate, which the
product group C1 x ... x C1 does not define. Two-qubit RB (``pair=True``, the default on two qubits) runs the 11520-element
group on the pair.

The Knill-style variant (``variant="knill"``; Knill et al., PRA 77, 012307, 2008; Harty et al. 2014; Wright et al. 2019)
plays per computational gate a random Pauli (a pi rotation about +-x or +-y as one GPi pulse, about +-z as a virtual RZ, or
the identity) then a random pi/2 about +-x or +-y, and closes with one pi/2 into the computational basis; the ideal state
stays on the six Pauli eigenstates, so the survival is the probability of the sequence's own target and the decay is
fitted in the authors' form B p^L + 1/2.

The budget alongside (``benchmarks.budget``): the intrinsic scales per Clifford, the channels of the native gate kinds
composed to first order into a predicted r and p, and the SPAM offset F(0) the preparation and readout errors predict.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from qutip_trap.benchmarks.budget import (
    BenchmarkBudget,
    _shot_sigma,
    channels_for,
    gather_counts_and_intrinsic,
    spam_of,
)
from qutip_trap.benchmarks.clifford import (
    SINGLE_QUBIT_CLIFFORDS,
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
    """One random sequence: its Cliffords (per qubit a tuple of single-qubit Clifford indices, or one tuple of
    ``TwoQubitClifford`` for a pair), the closing inverse, the circuit as built (compiled and verified inside ``run``) and
    the bitstring its ideal state reads in the histogram's order (the sorted measured qubits, the lowest-indexed one
    rightmost; ``""`` for |0...0>)."""

    length: int
    qubits: tuple[int, ...]
    cliffords: tuple[Any, ...]
    inverse: Any
    circuit: Circuit
    target_key: str = ""

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
    inverses = {q: single_qubit_sequence_unitary(indices[q]).conj().T for q in qs}
    for q in qs:
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
    qs = (int(pair[0]), int(pair[1]))
    return RBSequence(
        length=length,
        qubits=qs,
        cliffords=(cliffords,),
        inverse=inverse,
        circuit=Circuit(n_qubits, tuple(ops), qs),
    )


_KNILL_AZIMUTH: Final[tuple[float, ...]] = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
"""The azimuths +-x, +-y of the Knill-style variant: a pi/2 or pi rotation about one maps a Pauli eigenstate to a Pauli
eigenstate."""


def _knill_final(u: np.ndarray) -> tuple[float | None, int]:
    """(the azimuth of the one final pi/2 pulse, the computational bit the ideal state then reads), or (None, bit) when the
    ideal state u|0> already is a computational basis state."""
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
    """Knill-style sequences (Knill et al. 2008; Harty et al. 2014; Wright et al. 2019), one per qubit of ``qubits``,
    interleaved gate by gate: ``length`` computational gates per qubit, each a random Pauli (a pi rotation about +-x or +-y
    as one GPi pulse, about +-z as a virtual RZ, or the identity) then a random pi/2 about +-x or +-y as one GPi2 pulse,
    and one final pi/2 per qubit into a computational basis state (none when the ideal state already is one)."""
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
            elif code < 6:  # +-z: a rotation of the logical frame, no pulse
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
        phi_f, bits[q] = _knill_final(ideal[q])
        finals.append(phi_f)
        if phi_f is not None:
            ops.append(Operation("gpi2", (q,), (phi_f,)))
    return RBSequence(
        length=length,
        qubits=qs,
        cliffords=tuple((paulis[q], cliffords[q]) for q in qs),
        inverse=tuple(finals),
        circuit=Circuit(n_qubits, tuple(ops), qs),
        target_key="".join(str(bits[q]) for q in sorted(qs)[::-1]),  # the histogram puts qubit 0 rightmost
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
    """Fit the mean survival per length to A p^m + B: (A, p, B and chi2_per_dof with uncertainties, converged, notes).
    ``fix_offset`` pins B at 1/2^n (the depolarizing floor, the usual remedy when a tiny decay leaves A and B degenerate);
    None pins it only below three lengths."""
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
    if len(m) < 3 if fix_offset is None else fix_offset:

        def model2(params: np.ndarray, x: np.ndarray) -> np.ndarray:
            return rb_model(np.array([params[0], params[1], b0]), x)

        fit = weighted_fit(model2, [a0, p0], m, y, sigma=sg, bounds=([0.0, 0.0], [1.0, 1.0]))
        b = (b0, 0.0)
        why = "fewer than three sequence lengths" if fix_offset is None else "fix_offset=True"
        notes.append(f"B fixed at 1/2^n = {b0:g} ({why})")
    else:
        fit = weighted_fit(rb_model, [a0, p0, b0], m, y, sigma=sg, bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
        b = fit.value(2)
    out = {"A": fit.value(0), "p": fit.value(1), "B": b, "chi2_per_dof": (fit.chi2_per_dof, 0.0)}
    return out, fit.converged, tuple(notes)


@dataclass(frozen=True)
class RBPrediction:
    """What the channels and the SPAM predict for a benchmark's decay (``BenchmarkBudget.predicted``; the budget's notes give
    the rules)."""

    r_channel: float
    """The channels composed to first order per unit, reduced to the benchmarked qubits; for simultaneous RB the mean of
    ``r_channel_per_qubit``."""
    p_channel: float
    """1 - r_channel 2^n/(2^n - 1)."""
    F0_spam: float
    """The survival at m = 0 that SPAM alone predicts."""
    B_depolarizing: float
    """1/2^n."""
    A_spam: float
    """F0_spam - B_depolarizing."""
    r_channel_per_qubit: dict[int, float] = field(default_factory=dict)
    """Simultaneous RB: per benchmarked qubit, the channels reduced to that one qubit, per Clifford."""
    r_channel_joint_layer: float | None = None
    """Simultaneous RB: the channels reduced to every benchmarked qubit over one layer of n Cliffords."""


@dataclass(frozen=True)
class RBResult:
    """A randomized-benchmarking run (module docstring): the joint survival P(every benchmarked qubit reads its target
    bit) per (length, sequence) with its shot-noise error, the mean per length with its error, the A p^m + B fit, the
    error per Clifford beside the depolarizing entanglement infidelity, the per-qubit marginals of simultaneous RB, the
    pulse counts, the sequences, every ``Result`` in (length, sequence) order and the budget alongside."""

    qubits: tuple[int, ...]
    lengths: tuple[int, ...]
    n_sequences: int
    shots: int
    variant: str
    """``clifford`` (the Clifford group) or ``knill`` (the Knill-style sequences)."""
    survival: np.ndarray
    """(n_lengths, n_sequences)."""
    survival_sigma: np.ndarray
    mean_survival: np.ndarray
    mean_sigma: np.ndarray
    fit: dict[str, tuple[float, float]]
    """A, p, B and chi2_per_dof of the joint survival."""
    error_per_clifford: tuple[float, float]
    """r in the unit of ``budget.predicted.r_channel``: (1 - p)(2^n - 1)/2^n for one qubit and for a pair; for
    simultaneous RB the mean over the qubits of the marginal r_q = (1 - p_q)/2."""
    depolarizing_entanglement_infidelity: tuple[float, float]
    """(4^n - 1)(1 - p)/4^n, not r; for simultaneous RB the mean of the marginal (4 - 1)(1 - p_q)/4."""
    joint_error_per_layer: tuple[float, float] | None
    """Simultaneous RB only: (1 - p)(2^n - 1)/2^n of the joint survival, sum_q r_q to first order."""
    pulses_per_clifford: float
    entangling_per_clifford: float
    converged: bool
    sequences: tuple[RBSequence, ...]
    results: tuple[Result, ...]
    marginal_survival: np.ndarray | None = None
    """(n_qubits, n_lengths, n_sequences) marginal survival per qubit; None for a pair."""
    marginal_fit: tuple[dict[str, tuple[float, float]], ...] = ()
    marginal_error_per_clifford: tuple[tuple[float, float], ...] = ()
    """Per qubit r_q = (1 - p_q)/2 from its marginal decay: its gates' cost under simultaneous operation."""
    budget: BenchmarkBudget[RBPrediction] | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def fidelity_form(self) -> str:
        """The decay in the form Wright et al. 2019 print it, B p^L + 1/2, with this run's fitted B."""
        return f"F(L) = {self.fit['A'][0]:.4f} x {self.fit['p'][0]:.5f}^L + {self.fit['B'][0]:.4f}"


def marginal_survival(res: Result, key: str, qubits: Sequence[int]) -> dict[int, tuple[float, float]]:
    """Per benchmarked qubit, (P(it reads its target bit), shot-noise sigma) from the histogram, ``key`` the target in the
    histogram's order (the sorted measured qubits, the lowest-indexed one rightmost)."""
    order = sorted(int(q) for q in qubits)[::-1]
    if len(order) != len(key):
        raise ValueError("the target key and the benchmarked qubits disagree in length")
    out: dict[int, tuple[float, float]] = {}
    for i, q in enumerate(order):
        p = float(sum(v for k, v in res.probabilities.items() if k[i] == key[i]))
        out[q] = (p, _shot_sigma(res, p * (1.0 - p)))
    return out


def mean_survival_sigma(values: np.ndarray, sigma: np.ndarray, n_sequences: int, shots: int) -> np.ndarray:
    """The standard error of the mean survival over sequences, per length: the between-sequence sample variance alone when
    more than one sequence is drawn (it already contains each point's shot noise), the shot noise alone at one sequence,
    and one count out of the whole run when both vanish."""
    var = values.var(axis=1, ddof=1) / n_sequences if n_sequences > 1 else (sigma**2).mean(axis=1)
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
    """Randomized benchmarking of one qubit, of several at once (simultaneous RB) or of a pair, through ``Machine.run``.

    ``lengths`` are the Clifford counts m (the closing inverse not counted), or for ``variant="knill"`` the computational
    gate counts L (the closing pi/2 not counted); every (length, sequence) is one run of ``shots`` with its own keyed seed.
    On exactly two qubits ``pair=True`` (the default there) runs the two-qubit Clifford group and ``pair=False``
    simultaneous single-qubit RB; ``pair=True`` on any other count is refused. ``budget`` adds the channels of the native
    gate kinds used and the predictions composed from them; ``fix_offset`` pins B at 1/2^n (always, for the Knill-style
    variant, whose published form is B p^L + 1/2)."""
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
    n_ions = machine.device.crystal.n_ions
    if any(q < 0 or q >= n_ions for q in qs):
        raise ValueError("qubits index the device's ions")
    if pair and (len(qs) != 2 or variant != "clifford"):
        raise ValueError(
            "pair=True runs the 11520-element two-qubit Clifford group and needs exactly two qubits and "
            f"variant='clifford'; got {len(qs)} qubit(s) and variant={variant!r}. Pass pair=False for simultaneous "
            "single-qubit RB on several ions"
        )
    knill = variant == "knill"
    two_qubit = not knill and len(qs) == 2 and (pair is None or pair)
    n_q = len(qs)
    simultaneous = not two_qubit and n_q > 1
    offset = True if (knill and fix_offset is None) else fix_offset
    # the sequence family and the benchmark units of a length-m sequence (one Clifford on one qubit, or one Knill
    # computational gate; the closing inverse Clifford counts, the closing pi/2 does not)
    draw: Callable[[np.random.Generator, int], RBSequence]
    per_unit: Callable[[int], int]
    if knill:
        draw, per_unit = (lambda g, m: knill_sequences(g, qs, m, n_ions)), (lambda m: m * n_q)
    elif two_qubit:
        draw, per_unit = (lambda g, m: two_qubit_sequence(g, (qs[0], qs[1]), m, n_ions)), (lambda m: m + 1)
    else:
        draw, per_unit = (lambda g, m: single_qubit_sequences(g, qs, m, n_ions)), (lambda m: (m + 1) * n_q)
    root = SeedSpec(int(seed))
    rng = np.random.default_rng(root.child(0, 0, 0, 0, "rb_sequences"))
    sequences: list[RBSequence] = []
    results: list[Result] = []
    survival = np.zeros((len(ls), n_sequences))
    sigma = np.zeros_like(survival)
    marginal = np.zeros((n_q, len(ls), n_sequences))
    marginal_sigma = np.zeros_like(marginal)
    n_pulses = n_ent = n_units = 0
    for i, m in enumerate(ls):
        for j in range(n_sequences):
            seq = draw(rng, m)
            run_seed = int(root.child(0, 0, len(results), 0, "rb_runs").generate_state(1)[0])
            res = machine.run(seq.circuit, shots, seed=run_seed)
            rec = last_record(res)
            n_pulses += rec.compile.n_pulses
            n_ent += rec.compile.n_entangling
            n_units += per_unit(m)
            p = float(res.probabilities.get(seq.key, 0.0))
            s = float(res.error_bars.get(seq.key, math.sqrt(max(p * (1.0 - p), 0.0) / max(res.shots, 1))))
            survival[i, j], sigma[i, j] = p, s if s > 0.0 else 1.0 / max(res.shots, 1)
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
            m_sigma = mean_survival_sigma(marginal[iq], marginal_sigma[iq], n_sequences, shots)
            m_fit, m_ok, _ = fit_decay(ls, marginal[iq].mean(axis=1), m_sigma, 1, fix_offset=offset)
            converged = converged and m_ok
            marginal_fits.append(m_fit)
            marginal_r.append((rb_error_per_clifford(m_fit["p"][0], 1), 0.5 * m_fit["p"][1]))
    if simultaneous:
        r = float(np.mean([v for v, _ in marginal_r]))
        sr = float(math.sqrt(sum(s * s for _, s in marginal_r)) / n_q)
        eps_dep = float(np.mean([depolarizing_entanglement_infidelity(f["p"][0], 1) for f in marginal_fits]))
        s_dep = float(math.sqrt(sum((0.75 * f["p"][1]) ** 2 for f in marginal_fits)) / n_q)
        joint: tuple[float, float] | None = joint_r
        notes.append(
            "simultaneous RB (Gambetta et al. 2012): error_per_clifford is the mean over the benchmarked qubits of the"
            " per-qubit MARGINAL r_q = (1 - p_q)/2 (marginal_error_per_clifford), one Clifford on one qubit the unit and"
            " the same unit as budget.predicted.r_channel and as the r a qubit measures alone; joint_error_per_layer"
            " = (1 - p)(2^n - 1)/2^n of the JOINT survival is the error per LAYER of n Cliffords (sum_q r_q to first"
            " order), a correlation diagnostic and not an n-qubit Clifford error rate"
        )
    else:
        r, sr = joint_r
        eps_dep = depolarizing_entanglement_infidelity(p, n_q)
        s_dep = sp * (4**n_q - 1) / 4**n_q
        joint = None
    if knill:
        notes.append(
            "Knill-style variant (Section 7.9): the unit is one computational gate (a random Pauli then a random"
            " Clifford), the closing pi/2 not counted; p is the per-pi/2-gate decay in Wright et al. 2019's convention"
            " and r = (1 - p)/2 its Section 13 conversion"
        )
    bud = None
    if budget:
        bud = _rb_budget(
            machine, results, [per_unit(m) for m in ls for _ in range(n_sequences)], qs, knill, simultaneous
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
        pulses_per_clifford=n_pulses / n_units,
        entangling_per_clifford=n_ent / n_units,
        converged=converged,
        sequences=tuple(sequences),
        results=tuple(results),
        marginal_survival=None if two_qubit else marginal,
        marginal_fit=tuple(marginal_fits),
        marginal_error_per_clifford=tuple(marginal_r),
        budget=bud,
        notes=tuple(notes),
    )


def _rb_budget(
    machine: Machine,
    results: Sequence[Result],
    units: Sequence[int],
    qs: tuple[int, ...],
    knill: bool,
    simultaneous: bool,
) -> BenchmarkBudget[RBPrediction]:
    """The RB budget: r_channel composed from the channels to first order, the intrinsic scales per unit and the SPAM
    survival F0 at m = 0 (module docstring)."""
    n_q = len(qs)
    counts, intrinsic = gather_counts_and_intrinsic(results, units)
    spam = spam_of(results[0], qs)
    kinds = list(counts)
    channels, infid = channels_for(machine, kinds, qs)
    r_channel_joint = float(sum(counts[k] * infid[k] for k in kinds))
    per_qubit = (
        {q: float(n_q * sum(counts[k] * channels[k].infidelity_on((q,)) for k in kinds)) for q in qs}
        if simultaneous
        else {}
    )
    r_channel = float(np.mean(list(per_qubit.values()))) if simultaneous else r_channel_joint
    n_channel = 1 if simultaneous else n_q
    f0 = 1.0
    for q in qs:
        eb, ed = spam.get(f"q{q}", (0.0, 0.0))
        prep = spam.get(f"q{q}.state_preparation", (0.0, 0.0))[0]
        # survival to the target bit: prepared right and read right, or prepared wrong and read wrong; Clifford RB returns
        # to the dark |0> (eps_D), the Knill-style variant to |0> or |1> with equal probability (the mean)
        e_target = 0.5 * (eb + ed) if knill else ed
        e_other = 0.5 * (eb + ed) if knill else eb
        f0 *= (1.0 - prep) * (1.0 - e_target) + prep * e_other
    predicted = RBPrediction(
        r_channel=r_channel,
        p_channel=1.0 - r_channel * 2**n_channel / (2**n_channel - 1),
        F0_spam=f0,
        B_depolarizing=1.0 / 2**n_q,
        A_spam=f0 - 1.0 / 2**n_q,
        r_channel_per_qubit=per_qubit,
        r_channel_joint_layer=float(n_q * r_channel_joint) if simultaneous else None,
    )
    notes = [
        "r_channel = sum over native gate kinds of (pieces per unit) x (average infidelity of the kind's GATE_LOCAL "
        "channel reduced to the benchmarked qubits), the first-order composition; p_channel = 1 - r 2^n/(2^n - 1)",
        "intrinsic_total = the closed-form intrinsic scales per unit summed over each kind's WHOLE schedule entry, "
        "including the crosstalk it inflicts on ions outside the benchmarked set, so it bounds a LARGER error than "
        "r_channel (which is reduced to the benchmarked qubits) measures, and it bounds the coherent errors rather "
        "than valuing them",
        "F0_spam = prod_q [(1 - eps_prep)(1 - eps_read) + eps_prep eps_read_other]: the survival at m = 0 that SPAM "
        "alone predicts; B_depolarizing = 1/2^n",
    ]
    if simultaneous:
        notes[0] = (
            "r_channel is the mean over the benchmarked qubits of r_channel_per_qubit[i] = sum over native gate kinds of "
            "(pieces per Clifford of the sequence) x (average infidelity of the kind's GATE_LOCAL channel reduced to "
            "THAT ONE qubit), a one-qubit average gate infidelity per Clifford: the unit of the marginal r_q that "
            "simultaneous RB fits. r_channel_joint_layer composes the channels reduced to ALL benchmarked qubits over "
            "one layer of n Cliffords, the unit of joint_error_per_layer"
        )
    return BenchmarkBudget(
        unit="computational gate" if knill else "clifford",
        qubits=qs,
        counts=counts,
        intrinsic=intrinsic,
        spam=spam,
        channels=channels,
        channel_infidelity=infid,
        predicted=predicted,
        notes=tuple(notes),
    )
