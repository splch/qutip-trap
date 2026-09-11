"""Results, diagnostics and the persistent machine state (PLAN.md Sections 3.4, 6.7, 8.6; Appendix E).

Bit order (Section 13, row "Result bit order"; the one sentence, stated in the same words on docs/conventions.md):
in every bitstring key qubit 0 is the least-significant bit, the rightmost character, so "101" on three qubits is
qubit0 = 1, qubit1 = 0, qubit2 = 1 and the IonQ v1 decimal key "5"; ``Result.bit_order`` says so explicitly and the
exporters convert. IonQ's v2 result strings run the other way, q[0] first (``qutip_trap.io.ionq``); PennyLane
reverses to big-endian on its side; Qiskit does not (Section 8.6).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.evolve import ConvergenceReport
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.noise.sampling import NoiseSample
    from qutip_trap.run.gate_local import GateLocalReport


def bitstring_key(bits: np.ndarray) -> str:
    """The histogram key of one shot: qubit 0 rightmost (least significant)."""
    b = np.asarray(bits).astype(int)
    return "".join(str(int(x)) for x in b[::-1])


def decimal_key(bits: np.ndarray) -> str:
    """The IonQ decimal key of one shot: the integer sum_j b_j 2^j with qubit 0 least significant."""
    b = np.asarray(bits).astype(int)
    return str(int(sum(int(x) << j for j, x in enumerate(b))))


def bits_from_decimal(key: str, n_qubits: int) -> np.ndarray:
    """Inverse of :func:`decimal_key`: qubit j is bit j of the integer."""
    value = int(key)
    if value < 0 or value >= 2**n_qubits:
        raise ValueError(f"decimal key {key!r} does not fit {n_qubits} qubits")
    return np.array([(value >> j) & 1 for j in range(n_qubits)], dtype=np.uint8)


def aggregate(bitstrings: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    """Counts and probabilities keyed by :func:`bitstring_key`."""
    arr = np.asarray(bitstrings)
    if arr.ndim != 2:
        raise ValueError("bitstrings must be a (shots, n_qubits) array")
    counts = Counter(bitstring_key(row) for row in arr)
    n = arr.shape[0]
    probabilities = {k: v / n for k, v in counts.items()} if n else {}
    return dict(counts), probabilities


@dataclass(frozen=True)
class RunState:
    """Machine state that PERSISTS across shots (Sections 6.7, 8.5)."""

    order: tuple[int, ...]
    """Physical position -> qubit label; a reorder permutes it for every later shot."""
    dark: frozenset[int]
    lost: frozenset[int]
    events: tuple[tuple[int, str], ...]
    """(shot index, event) log."""

    def recrystallize(self, shot: int = -1) -> RunState:
        """Bookkeeping of a recrystallization after a melt (Section 6.7): the order and the dark/lost flags persist (a reorder
        is not undone and a dark ion stays dark until repumped or reloaded); the event is logged."""
        return RunState(self.order, self.dark, self.lost, self.events + ((int(shot), "recrystallize"),))

    def reload(self, device: Device, shot: int = -1) -> RunState:
        """A reload restores the device's nominal crystal: identity order, no dark or lost ions, the event logged."""
        n = device.crystal.n_ions
        return RunState(tuple(range(n)), frozenset(), frozenset(), self.events + ((int(shot), "reload"),))

    @classmethod
    def nominal(cls, n_ions: int) -> RunState:
        return cls(tuple(range(n_ions)), frozenset(), frozenset(), ())


@dataclass(frozen=True)
class Diagnostics:
    """What a run did and what it approximated (Sections 3.4, 5.5, 8.6): the level that ran and the space it used, the class
    of every mode, the truncation monitors, the integrator and its tolerances, the realized (samples, trajectories, shots
    per sample) triple with the effective sample size, the root seed that reruns it, the calibration table it believed,
    the approximations it made and the closed-form error budget reported beside the result; the GATE_LOCAL report when
    that level ran."""

    level: Literal["JOINT_EXACT", "GATE_LOCAL"]
    """The level actually run."""
    space: HilbertSpace
    mode_class: dict[int, Literal["resolved", "frozen", "dropped", "enr"]]
    run_state: RunState
    wall_clock_span_s: float
    """t0 to the last shot: shot k is evaluated at t0 + k T_rep (Section 7.5)."""
    boundary_population: dict[int, float]
    margin_levels: dict[int, int]
    dropped_modes: tuple[int, ...]
    frozen_contribution: dict[int, tuple[float, float]]
    """(|alpha|^2 (2n + 1), chi) per frozen mode."""
    integrator: str
    tolerances: tuple[float, float]
    samples: int
    trajectories: int
    shots_per_sample: int
    effective_sample_size: float
    root_seed: int
    calibration: CalibrationTable
    approximations: tuple[str, ...]
    intrinsic_budget: dict[str, float] = field(default_factory=dict)
    """Per played gate, the closed-form error scales of Section 9.6 reported beside the result (M6): residual displacement
    eps_ent = sum |alpha|^2 (2 nbar + 1), the n = 0-referenced Debye-Waller loss, the off-resonant carrier scale (Omega/nu)^2,
    the frozen spectators' chi loss; and 'total' their sum."""
    dropped_branch_weight: float = 0.0
    """Weight of the initial-mixture branches below SolverOptions.branch_weight_min that were not evolved (M6)."""
    frozen_excitation_bound: dict[int, float] = field(default_factory=dict)
    """Per frozen mode, the Section 5.2 off-resonant excitation bound summed over the schedule's pulses (M9a)."""
    dropped_contribution: tuple[float, float] = (0.0, 0.0)
    """(sum |alpha|^2 (2n + 1), sum |chi|) over the dropped modes: the summed dropped contribution of Section 11.3 item 2 (M9a)."""
    margin_reached: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the smallest margin (levels) the cap kept above the populated range during a pulse (Section 5.5; M9a)."""
    populated_n_max: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the highest Fock index populated above the boundary threshold during a pulse (M9a)."""
    cap_growth: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the levels the truncation monitor added to the cap during the run (Section 5.5; M9a)."""
    gate_local: GateLocalReport | None = None
    """The GATE_LOCAL walk's report (Section 5.4; M9a): per step the local space, the tomography, the channel summary, the
    residual displacement and the motional bookkeeping; None for a JOINT_EXACT run. ``space`` is then the joint space the run
    would have needed (the one above the guards), whose mode classes ``mode_class`` reports."""
    kernel: str = "none"
    """How the drive operators were held (Section 11.3 item 4; M9b): ``factorized`` (the matrix-free kernel on some segment),
    ``assembled`` (CSR everywhere), ``mixed``, or ``none`` (no drive term integrated in this process: GATE_LOCAL's runs report
    theirs in ``gate_local``)."""
    workers: int = 1
    """Processes the parallel maps of Section 11.3 item 9 used (1 = everything in-process)."""
    propagator_cache_hits: int = 0
    """Segments served from the engines' propagator caches (Section 11.3 item 5; M9b)."""
    branches: int = 1
    """Branches of the initial mixture (Section 5.3's Fock sum) that were evolved. ``trajectories`` counts branches TIMES the
    engine's trajectory count, the plan's realized triple being (samples, trajectories, shots per sample); this field
    separates the two so a reader can tell a Fock branch from a quantum-jump trajectory (M6 fix, ``conv.fock_sum_branches``)."""
    convergence: ConvergenceReport | None = None
    """Section 5.5's tolerance-convergence report, present when ``SolverOptions.convergence_check`` was set: the change in the
    register populations when atol and rtol are tightened by ten (M2's ``dynamics.evolve.convergence_check``; the run-path
    plumbing is M9's, recorded as ``conv.appendix_e_signatures``). ``None`` means the check was not asked for, never that it
    passed."""
    shots_per_sample_realized: tuple[int, ...] = ()
    """The shots each dynamical sample actually took, in sample order (Section 3.4 / 8.6): ``shots_per_sample`` is the floor
    ``shots // samples``, so with shots = 2000 over 64 samples it reports 31 while 32 samples took 32 shots. This tuple is
    also the shot -> sample map Section 8.6 asks the result to make recoverable: sample k owns the contiguous shot block
    [sum_{j<k} M_j, sum_{j<=k} M_j) (see ``Result.sample_of_shot``)."""


def binomial_error_bars(probabilities: Mapping[str, float], n_eff: float) -> dict[str, float]:
    """sqrt(p (1 - p)/n_eff) per key: the histogram error bars from the effective sample size (Section 3.4)."""
    if n_eff <= 0.0:
        return {k: float("nan") for k in probabilities}
    return {k: math.sqrt(max(p * (1.0 - p), 0.0) / n_eff) for k, p in probabilities.items()}


@dataclass(frozen=True)
class Result:
    """The outcome of a run (Sections 3.4, 8.6): the per-shot bitstrings (column j = qubit j) and their aggregation into counts
    and probabilities with binomial error bars from the effective sample size, the photon records and posteriors when the
    record was read in full, the noise sample of every dynamical sample, the herald flags, the discarded shots, the
    persistent machine state, the SPAM errors per qubit, the recombined register state when kept, and the
    ``Diagnostics``. Histogram keys follow the Section 13 bit order (module docstring)."""

    bitstrings: np.ndarray
    """(shots, n_qubits) array of 0/1, column j = qubit j."""
    bit_order: Literal["qubit0_lsb"]
    counts: dict[str, int]
    probabilities: dict[str, float]
    error_bars: dict[str, float]
    photon_records: np.ndarray | None
    """(shots, n_qubits) total detected counts per ion per shot; None on the fast readout path."""
    posteriors: np.ndarray | None
    noise_samples: tuple[NoiseSample, ...]
    heralds: np.ndarray
    """Per-shot flags, bit 0 = collision, bit 1 = a dark or lost ion, bit 2 = a count anomaly (a record whose total lies
    outside the [1e-6, 1 - 1e-6] quantile band of BOTH the bright and the dark count distribution of the ion's model, so
    neither hypothesis explains it: a cosmic ray, an afterpulse burst or a stray-light flash); Sections 6.7, 8.6."""
    discarded_shots: int
    run_state: RunState
    spam: dict[str, tuple[float, float]]
    """Per qubit (eps_B, eps_D) with the definition used (Section 13, "Readout figure of merit")."""
    final_state: Qobj | None
    """The recombined register density matrix in QuTiP's tensor order (ion 0 the FIRST factor, the most-significant index bit);
    ``bitstrings``, ``counts`` and ``probabilities`` use the Section 13 order (qubit 0 the least-significant bit), and
    ``run.job.to_register_order`` converts a compiler-order ket to this one."""
    diagnostics: Diagnostics
    sub_bin_records: np.ndarray | None = None
    """(shots, n_qubits, n_sub_bins) counts per sub-bin when the discriminator is time resolved, which is Section 8.6's
    "the photon-count record per ion AND PER SUB-BIN when time-resolved"; None for a threshold discriminator, whose record
    carries no sub-bin structure, and on the fast path."""
    arrival_times_s: tuple[tuple[np.ndarray, ...], ...] | None = None
    """Per shot, per ion, the photon arrival times when the discriminator asked for them (Noek's first-photon protocol,
    Crain's stop-on-first-photon); None otherwise. Ragged, so a tuple of arrays rather than one array."""

    def __post_init__(self) -> None:
        arr = np.asarray(self.bitstrings)
        if arr.ndim != 2:
            raise ValueError("bitstrings must be a (shots, n_qubits) array")
        if self.bit_order != "qubit0_lsb":
            raise ValueError("the only supported bit order is qubit0_lsb (Section 13)")
        counts, probabilities = aggregate(arr)
        if counts != self.counts:
            raise ValueError("counts must aggregate the bitstrings with qubit 0 as the least-significant bit")
        if set(probabilities) != set(self.probabilities) or any(
            abs(probabilities[k] - self.probabilities[k]) > 1e-12 for k in probabilities
        ):
            raise ValueError("probabilities must be counts / shots")
        if len(self.heralds) != arr.shape[0]:
            raise ValueError("one herald flag per shot")

    @property
    def n_qubits(self) -> int:
        return int(np.asarray(self.bitstrings).shape[1])

    @property
    def shots(self) -> int:
        return int(np.asarray(self.bitstrings).shape[0])

    def to_ionq_json(self) -> dict[str, float]:
        """The legacy IonQ probability result: decimal-integer keys, qubit 0 least significant (Section 8.6)."""
        return {str(int(key, 2)): p for key, p in self.probabilities.items()}

    def to_ionq_histogram(self) -> dict[str, int]:
        """The histogram variant: shot counts with the same decimal keys."""
        return {str(int(key, 2)): c for key, c in self.counts.items()}

    def to_ionq_shots(self) -> list[str]:
        """The per-shot format: an ordered list of decimal strings ("6", "1", "0", "7" is 110, 001, 000, 111 on three qubits)."""
        return [decimal_key(row) for row in np.asarray(self.bitstrings)]

    @property
    def sample_of_shot(self) -> np.ndarray:
        """Per shot, the index of the dynamical sample it was drawn from (Section 8.6: "a Result carries per shot ... the
        sampled quasi-static noise parameters of that shot's dynamical sample"). ``noise_samples[sample_of_shot[k]]`` is
        shot k's parameter draw. Shots are allocated in contiguous blocks (``conv.shot_blocks_per_sample``), so this is the
        block index built from ``Diagnostics.shots_per_sample_realized``; a run with one sample maps every shot to 0."""
        realized = self.diagnostics.shots_per_sample_realized
        n = self.shots
        if not realized:
            return np.zeros(n, dtype=np.int64)
        out = np.concatenate([np.full(int(m), s, dtype=np.int64) for s, m in enumerate(realized)])
        # discarded shots (heralded collisions, Section 6.7) shorten the kept array; the map covers what was kept
        return (
            out[:n] if out.shape[0] >= n else np.concatenate([out, np.full(n - out.shape[0], -1, np.int64)])
        )

    def to_ionq_v2(self, registers: Mapping[str, Sequence[int]] | None = None) -> dict[str, dict[str, float]]:
        """Section 8.6's v2 register-nested result: ``{register name: {bitstring: probability}}``.

        ``registers`` names the classical registers and the qubits each covers, in the register's own bit order (qubit 0 of
        the register least significant, the Section 13 convention). With no registers given the whole measured set is one
        register named ``"c"``, the default name qelib1.inc-style exports use, so the format is always available. The
        OpenQASM 2 importer does not yet carry its ``creg`` names into the ``Circuit`` (it flattens them in declaration
        order), so a caller who imported a multi-register program supplies the map itself."""
        bits = np.asarray(self.bitstrings)
        regs = dict(registers) if registers else {"c": tuple(range(self.n_qubits))}
        out: dict[str, dict[str, float]] = {}
        for name, qubits in regs.items():
            idx = [int(q) for q in qubits]
            if any(q < 0 or q >= self.n_qubits for q in idx):
                raise ValueError(f"register {name!r} names qubits outside the result's {self.n_qubits}")
            counts: dict[str, int] = {}
            for row in bits[:, idx] if idx else np.zeros((bits.shape[0], 0), dtype=bits.dtype):
                key = bitstring_key(row)
                counts[key] = counts.get(key, 0) + 1
            total = float(sum(counts.values())) or 1.0
            out[name] = {k: c / total for k, c in sorted(counts.items())}
        return out


__all__ = [
    "Diagnostics",
    "Result",
    "RunState",
    "aggregate",
    "binomial_error_bars",
    "bits_from_decimal",
    "bitstring_key",
    "decimal_key",
]
