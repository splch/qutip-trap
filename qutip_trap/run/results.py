"""Results, diagnostics and the persistent machine state of a run.

Bit order: in every bitstring key qubit 0 is the least-significant bit, the rightmost character, so "101" on three
qubits is qubit0 = 1, qubit1 = 0, qubit2 = 1 and the IonQ v1 decimal key "5". IonQ's v2 result strings run the other
way, q[0] first (``qutip_trap.io.ionq``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
    """Machine state that persists across shots."""

    order: tuple[int, ...]
    """Physical position -> qubit label; a reorder permutes it for every later shot."""
    dark: frozenset[int]
    lost: frozenset[int]
    events: tuple[tuple[int, str], ...]
    """(shot index, event) log."""

    def recrystallize(self, shot: int = -1) -> RunState:
        """Log a recrystallization after a melt; the order and the dark/lost flags persist."""
        return RunState(self.order, self.dark, self.lost, self.events + ((int(shot), "recrystallize"),))

    def reload(self, device: Device, shot: int = -1) -> RunState:
        """Restore the device's nominal crystal (identity order, no dark or lost ions) and log the event."""
        n = device.crystal.n_ions
        return RunState(tuple(range(n)), frozenset(), frozenset(), self.events + ((int(shot), "reload"),))

    @classmethod
    def nominal(cls, n_ions: int) -> RunState:
        return cls(tuple(range(n_ions)), frozenset(), frozenset(), ())

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able values."""
        return {
            "order": [int(i) for i in self.order],
            "dark": sorted(int(i) for i in self.dark),
            "lost": sorted(int(i) for i in self.lost),
            "events": [[int(shot), str(event)] for shot, event in self.events],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunState:
        return cls(
            tuple(int(i) for i in d["order"]),
            frozenset(int(i) for i in d["dark"]),
            frozenset(int(i) for i in d["lost"]),
            tuple((int(shot), str(event)) for shot, event in d["events"]),
        )


@dataclass(frozen=True)
class Diagnostics:
    """What a run did and what it approximated, reported beside the result."""

    level: Literal["JOINT_EXACT", "GATE_LOCAL"]
    """The level actually run."""
    space: HilbertSpace
    mode_class: dict[int, Literal["resolved", "frozen", "dropped", "enr"]]
    run_state: RunState
    wall_clock_span_s: float
    """t0 to the last shot: shot k is evaluated at t0 + k T_rep."""
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
    """The closed-form error scales of ``run.job.intrinsic_budget``, keyed ``<gate>.<term>``, and their ``total``."""
    dropped_branch_weight: float = 0.0
    """Weight of the initial-mixture branches below ``branch_weight_min`` that were not evolved."""
    frozen_excitation_bound: dict[int, float] = field(default_factory=dict)
    """Per frozen mode, the off-resonant excitation bound summed over the schedule's pulses."""
    dropped_contribution: tuple[float, float] = (0.0, 0.0)
    """(sum |alpha|^2 (2n + 1), sum |chi|) over the dropped modes."""
    margin_reached: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the smallest margin (levels) the cap kept above the populated range during a pulse."""
    populated_n_max: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the highest Fock index populated above the boundary threshold during a pulse."""
    cap_growth: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the levels the truncation monitor added to the cap during the run."""
    gate_local: GateLocalReport | None = None
    """The GATE_LOCAL walk's per-step report; None for a JOINT_EXACT run. With it, ``space`` is the joint space the run
    would have needed."""
    kernel: str = "none"
    """How the drive operators were held: ``factorized`` (the matrix-free kernel on some segment), ``assembled`` (CSR
    everywhere), ``mixed``, or ``none`` (nothing integrated here; a GATE_LOCAL run reports in ``gate_local``)."""
    workers: int = 1
    """Processes the parallel maps used (1 = everything in-process)."""
    propagator_cache_hits: int = 0
    """Segments served from the engines' propagator caches."""
    branches: int = 1
    """Branches of the initial mixture (the Fock sum) that were evolved; ``trajectories`` counts branches times the
    engine's trajectory count."""
    convergence: ConvergenceReport | None = None
    """The change in the register populations when atol and rtol are tightened by ten (``convergence_check``); None
    means the check was not asked for, never that it passed."""
    shots_per_sample_realized: tuple[int, ...] = ()
    """The shots each dynamical sample took, in order (``shots_per_sample`` is the floor ``shots // samples``)."""
    level_reason: str = ""
    """Why the run integrated at ``level``: ``LevelDecision.reason``, or the forced level and what AUTO would choose."""

    def to_dict(self) -> dict[str, Any]:
        """The run summary as JSON-able values (the GATE_LOCAL report as a flag, the convergence one as its summary)."""

        space: HilbertSpace = self.space
        return {
            "level": self.level,
            "level_reason": self.level_reason,
            "space": {
                "ion_dims": [int(d) for d in space.ion_dims],
                "resolved": [
                    {
                        "mode": int(t.mode),
                        "d": int(t.d),
                        "expected_n_range": [int(t.expected_n_range[0]), int(t.expected_n_range[1])],
                        "eta_max": float(t.eta_max),
                        "element_tol": None if t.element_tol is None else float(t.element_tol),
                    }
                    for t in space.resolved
                ],
                "enr_group": (
                    None
                    if space.enr_group is None
                    else [[int(m) for m in space.enr_group[0]], int(space.enr_group[1])]
                ),
                "frozen": [int(m) for m in space.frozen],
                "ions": [int(i) for i in space.ions],
                "dropped": [int(m) for m in space.dropped],
            },
            "mode_class": {str(m): c for m, c in self.mode_class.items()},
            "run_state": self.run_state.to_dict(),
            "wall_clock_span_s": float(self.wall_clock_span_s),
            "boundary_population": {str(m): float(v) for m, v in self.boundary_population.items()},
            "margin_levels": {str(m): int(v) for m, v in self.margin_levels.items()},
            "dropped_modes": [int(m) for m in self.dropped_modes],
            "frozen_contribution": {
                str(m): [float(a), float(b)] for m, (a, b) in self.frozen_contribution.items()
            },
            "integrator": self.integrator,
            "tolerances": [float(self.tolerances[0]), float(self.tolerances[1])],
            "samples": int(self.samples),
            "trajectories": int(self.trajectories),
            "shots_per_sample": int(self.shots_per_sample),
            "effective_sample_size": float(self.effective_sample_size),
            "root_seed": int(self.root_seed),
            "calibration": self.calibration.to_dict(),
            "approximations": [str(a) for a in self.approximations],
            "intrinsic_budget": {str(k): float(v) for k, v in self.intrinsic_budget.items()},
            "dropped_branch_weight": float(self.dropped_branch_weight),
            "frozen_excitation_bound": {str(m): float(v) for m, v in self.frozen_excitation_bound.items()},
            "dropped_contribution": [
                float(self.dropped_contribution[0]),
                float(self.dropped_contribution[1]),
            ],
            "margin_reached": {str(m): int(v) for m, v in self.margin_reached.items()},
            "populated_n_max": {str(m): int(v) for m, v in self.populated_n_max.items()},
            "cap_growth": {str(m): int(v) for m, v in self.cap_growth.items()},
            "gate_local": self.gate_local is not None,
            "kernel": self.kernel,
            "workers": int(self.workers),
            "propagator_cache_hits": int(self.propagator_cache_hits),
            "branches": int(self.branches),
            "convergence": None if self.convergence is None else self.convergence.summary(),
            "shots_per_sample_realized": [int(m) for m in self.shots_per_sample_realized],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Diagnostics:
        """The inverse of :meth:`to_dict`, with both reports None and the calibration table without its waveforms."""
        from qutip_trap.control.table import CalibrationTable
        from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation

        sp = d["space"]
        space = HilbertSpace(
            tuple(int(x) for x in sp["ion_dims"]),
            tuple(
                ModeTruncation(
                    int(t["mode"]),
                    int(t["d"]),
                    (int(t["expected_n_range"][0]), int(t["expected_n_range"][1])),
                    float(t["eta_max"]),
                    None if t.get("element_tol") is None else float(t["element_tol"]),
                )
                for t in sp["resolved"]
            ),
            None
            if sp.get("enr_group") is None
            else (tuple(int(m) for m in sp["enr_group"][0]), int(sp["enr_group"][1])),
            tuple(int(m) for m in sp["frozen"]),
            tuple(int(i) for i in sp.get("ions", ())),
            tuple(int(m) for m in sp.get("dropped", ())),
        )
        return cls(
            level=d["level"],
            space=space,
            mode_class={int(m): c for m, c in dict(d["mode_class"]).items()},
            run_state=RunState.from_dict(d["run_state"]),
            wall_clock_span_s=float(d["wall_clock_span_s"]),
            boundary_population={int(m): float(v) for m, v in dict(d["boundary_population"]).items()},
            margin_levels={int(m): int(v) for m, v in dict(d["margin_levels"]).items()},
            dropped_modes=tuple(int(m) for m in d["dropped_modes"]),
            frozen_contribution={
                int(m): (float(v[0]), float(v[1])) for m, v in dict(d["frozen_contribution"]).items()
            },
            integrator=str(d["integrator"]),
            tolerances=(float(d["tolerances"][0]), float(d["tolerances"][1])),
            samples=int(d["samples"]),
            trajectories=int(d["trajectories"]),
            shots_per_sample=int(d["shots_per_sample"]),
            effective_sample_size=float(d["effective_sample_size"]),
            root_seed=int(d["root_seed"]),
            calibration=CalibrationTable.from_dict(d["calibration"]),
            approximations=tuple(str(a) for a in d["approximations"]),
            intrinsic_budget={str(k): float(v) for k, v in dict(d["intrinsic_budget"]).items()},
            dropped_branch_weight=float(d["dropped_branch_weight"]),
            frozen_excitation_bound={int(m): float(v) for m, v in dict(d["frozen_excitation_bound"]).items()},
            dropped_contribution=(float(d["dropped_contribution"][0]), float(d["dropped_contribution"][1])),
            margin_reached={int(m): int(v) for m, v in dict(d["margin_reached"]).items()},
            populated_n_max={int(m): int(v) for m, v in dict(d["populated_n_max"]).items()},
            cap_growth={int(m): int(v) for m, v in dict(d["cap_growth"]).items()},
            gate_local=None,
            kernel=str(d["kernel"]),
            workers=int(d["workers"]),
            propagator_cache_hits=int(d["propagator_cache_hits"]),
            branches=int(d["branches"]),
            convergence=None,
            shots_per_sample_realized=tuple(int(m) for m in d["shots_per_sample_realized"]),
            level_reason=str(d.get("level_reason", "")),
        )

    @classmethod
    def external(cls, n_qubits: int, *, approximations: Sequence[str] = ()) -> Diagnostics:
        """The diagnostics of a result from outside the simulator: a register-only space, no modes or samples, an empty
        table, and ``approximations`` saying where the shots came from."""
        from qutip_trap.control.table import CalibrationTable
        from qutip_trap.hilbert.space import HilbertSpace

        return cls(
            level="JOINT_EXACT",
            space=HilbertSpace(tuple([2] * int(n_qubits)), (), None, ()),
            mode_class={},
            run_state=RunState.nominal(int(n_qubits)),
            wall_clock_span_s=0.0,
            boundary_population={},
            margin_levels={},
            dropped_modes=(),
            frozen_contribution={},
            integrator="none",
            tolerances=(math.nan, math.nan),
            samples=0,
            trajectories=0,
            shots_per_sample=0,
            effective_sample_size=0.0,
            root_seed=0,
            calibration=CalibrationTable.empty(),
            approximations=tuple(approximations),
            level_reason="no simulation: the shots came from outside",
        )


@dataclass(frozen=True)
class Progress:
    """One step of a run's progress, handed to the ``progress`` callback: ``stage`` is ``pulse`` (one integrated pulse
    segment), ``branch`` (one (sample, branch) engine run), ``sample`` or ``readout``. A stage's counts are monotone and
    end at ``done == total``; a parallel map reports its branches when it returns and no pulses, and a cap-raising
    retry or a convergence check repeats the pulse counts."""

    stage: str
    done: int
    total: int
    elapsed_s: float

    def __post_init__(self) -> None:
        if self.total < 0 or not 0 <= self.done <= self.total:
            raise ValueError(
                f"progress counts 0 <= done <= total, got done = {self.done}, total = {self.total}"
            )

    @property
    def fraction(self) -> float:
        """``done / total``, 1 when the stage has no steps."""
        return self.done / self.total if self.total else 1.0


def binomial_error_bars(probabilities: Mapping[str, float], n_eff: float) -> dict[str, float]:
    """sqrt(p (1 - p)/n_eff) per key: the histogram error bars from the effective sample size."""
    if n_eff <= 0.0:
        return {k: float("nan") for k in probabilities}
    return {k: math.sqrt(max(p * (1.0 - p), 0.0) / n_eff) for k, p in probabilities.items()}


@dataclass(frozen=True)
class Result:
    """The outcome of a run: the per-shot bitstrings with their counts, probabilities and binomial error bars, the
    photon records, noise samples, heralds, SPAM errors, the register state when kept, and the ``Diagnostics``."""

    bitstrings: np.ndarray
    """(shots, n_qubits) array of 0/1: row k is shot k, column j the qubit ``qubits[j]``."""
    bit_order: Literal["qubit0_lsb", "qubit0_msb"]
    """``qubit0_lsb`` for every run; ``qubit0_msb`` only on the result of ``reversed_bits()``."""
    counts: dict[str, int]
    """Shots per bitstring key (qubit 0 the rightmost character)."""
    probabilities: dict[str, float]
    error_bars: dict[str, float]
    photon_records: np.ndarray | None
    """(shots, n_qubits) total detected counts per ion per shot; None on the fast readout path."""
    posteriors: np.ndarray | None
    noise_samples: tuple[NoiseSample, ...]
    heralds: np.ndarray
    """Per-shot flags: bit 0 = collision, bit 1 = a dark or lost ion, bit 2 = a count anomaly (a total outside the
    [1e-6, 1 - 1e-6] quantile band of both the bright and the dark count distribution)."""
    discarded_shots: int
    run_state: RunState
    spam: dict[str, tuple[float, float]]
    """Per qubit ``q<i>``, (eps_B, eps_D); ``q<i>.state_preparation`` holds the preparation error."""
    final_state: Qobj | None
    """The recombined register density matrix in QuTiP's tensor order (ion 0 the first factor, the most-significant
    index bit); ``run.job.to_register_order`` converts a compiler-order ket to it."""
    diagnostics: Diagnostics
    sub_bin_records: np.ndarray | None = None
    """(shots, n_qubits, n_sub_bins) counts per sub-bin when the discriminator is time resolved; None otherwise."""
    arrival_times_s: tuple[tuple[np.ndarray, ...], ...] | None = None
    """Per shot, per ion, the photon arrival times (ragged) when the discriminator asked for them; None otherwise."""
    qubits: tuple[int, ...] | None = None
    """The circuit qubit each column of ``bitstrings`` holds (the measured qubits in ascending order); None means
    column j is qubit j."""
    registers: dict[str, tuple[int, ...]] | None = None
    """The circuit's classical registers, name -> qubits in bit order, which the IonQ v2 exporters report beside
    ``output_all``; None means one register ``"c"`` over every measured qubit."""
    machine_hash: str | None = None
    """``Machine.hash()`` of the machine that ran the circuit; None for an imported result."""
    created_at: str = ""
    """When the run started, ISO 8601 in UTC; empty when unknown."""
    duration_s: float = 0.0
    """Wall time of the run, seconds; 0 when unknown."""

    def __post_init__(self) -> None:
        arr = np.asarray(self.bitstrings)
        if arr.ndim != 2:
            raise ValueError("bitstrings must be a (shots, n_qubits) array")
        if self.bit_order not in ("qubit0_lsb", "qubit0_msb"):
            raise ValueError("the bit order is qubit0_lsb (Section 13) or qubit0_msb (reversed_bits only)")
        counts, probabilities = aggregate(arr)
        if counts != self.counts:
            raise ValueError("counts must aggregate the bitstrings with qubit 0 as the least-significant bit")
        if set(probabilities) != set(self.probabilities) or any(
            abs(probabilities[k] - self.probabilities[k]) > 1e-12 for k in probabilities
        ):
            raise ValueError("probabilities must be counts / shots")
        if len(self.heralds) != arr.shape[0]:
            raise ValueError("one herald flag per shot")
        if self.qubits is not None:
            qubits = tuple(int(q) for q in self.qubits)
            if len(qubits) != arr.shape[1] or len(set(qubits)) != len(qubits):
                raise ValueError("qubits names each column of bitstrings once")
            object.__setattr__(self, "qubits", qubits)
        if self.registers is not None:
            object.__setattr__(
                self, "registers", {str(k): tuple(int(q) for q in v) for k, v in dict(self.registers).items()}
            )

    @property
    def n_qubits(self) -> int:
        return int(np.asarray(self.bitstrings).shape[1])

    @property
    def shots(self) -> int:
        return int(np.asarray(self.bitstrings).shape[0])

    @property
    def column_qubits(self) -> tuple[int, ...]:
        """The circuit qubit of every column of ``bitstrings``: ``qubits`` when given, else 0, 1, ..., n - 1."""
        return self.qubits if self.qubits is not None else tuple(range(self.n_qubits))

    # ---- the IonQ v1 formats: decimal keys, qubit 0 the 2^0 bit -----------------------------------------------------------

    def to_ionq_v1_probabilities(self) -> dict[str, float]:
        """IonQ's v1 probabilities: decimal-integer keys, qubit 0 the least-significant bit."""
        return {str(int(key, 2)): p for key, p in self.probabilities.items()}

    def to_ionq_v1_histogram(self) -> dict[str, int]:
        """The v1 histogram: shot counts with the same decimal keys."""
        return {str(int(key, 2)): c for key, c in self.counts.items()}

    def to_ionq_v1_shots(self) -> list[str]:
        """The v1 per-shot format: one decimal key per shot, in shot order; ``from_ionq_v1_shots`` reads it back."""
        return [decimal_key(row) for row in np.asarray(self.bitstrings)]

    @classmethod
    def from_ionq_v1_shots(
        cls, shots: Sequence[str | int], n_qubits: int, *, source: str = "IonQ v1 shots"
    ) -> Result:
        """A ``Result`` from a v1 per-shot list (decimal strings or integers), with no SPAM and ``Diagnostics.external``
        saying that no simulation stands behind it."""
        rows = [bits_from_decimal(str(int(k)), int(n_qubits)) for k in shots]
        bits = (
            np.asarray(rows, dtype=np.uint8).reshape(-1, int(n_qubits))
            if rows
            else np.zeros((0, int(n_qubits)), dtype=np.uint8)
        )
        counts, probabilities = aggregate(bits)
        return cls(
            bitstrings=bits,
            bit_order="qubit0_lsb",
            counts=counts,
            probabilities=probabilities,
            error_bars=binomial_error_bars(probabilities, float(len(rows))),
            photon_records=None,
            posteriors=None,
            noise_samples=(),
            heralds=np.zeros(len(rows), dtype=np.uint8),
            discarded_shots=0,
            run_state=RunState.nominal(int(n_qubits)),
            spam={},
            final_state=None,
            diagnostics=Diagnostics.external(
                int(n_qubits),
                approximations=(
                    f"imported from {source}: no simulation behind these shots, no SPAM, no diagnostics",
                ),
            ),
        )

    # ---- the IonQ v2 formats: the v0.4 envelope, bitstrings in wire order (qutip_trap.io.ionq) ------------------------------

    def _v2_registers(self) -> dict[str, list[int]]:
        """Register name -> the columns of ``bitstrings`` in the register's bit order, ``output_all`` first (every measured
        qubit, ascending), then the circuit's registers; a register naming an unmeasured qubit is refused."""
        column_of = {q: j for j, q in enumerate(self.column_qubits)}
        out: dict[str, list[int]] = {"output_all": [column_of[q] for q in sorted(column_of)]}
        named = self.registers if self.registers is not None else {"c": tuple(sorted(column_of))}
        for name, qubits in named.items():
            missing = [q for q in qubits if q not in column_of]
            if missing:
                raise ValueError(f"register {name!r} names qubits {missing} that this result did not measure")
            out[name] = [column_of[q] for q in qubits]
        return out

    def _v2_strings(self, columns: Sequence[int]) -> list[str]:
        """One string per shot: the bit of the register's first qubit is the FIRST character (IonQ's wire order)."""
        bits = (
            np.asarray(self.bitstrings)[:, list(columns)]
            if columns
            else np.zeros((self.shots, 0), dtype=np.uint8)
        )
        return ["".join(str(int(b)) for b in row) for row in bits]

    def to_ionq_v2_probabilities(self) -> dict[str, Any]:
        """``ionq.result.probabilities.json.v2``: ``{"probabilities": {"registers": {"output_all": {bitstring: p}, ...}}}``
        with strings in IonQ's wire order (the reverse of ``counts``' keys)."""
        registers: dict[str, dict[str, float]] = {}
        for name, columns in self._v2_registers().items():
            tally = Counter(self._v2_strings(columns))
            total = float(sum(tally.values())) or 1.0
            registers[name] = {key: n / total for key, n in sorted(tally.items())}
        return {"probabilities": {"registers": registers}}

    def to_ionq_v2_histogram(self) -> dict[str, Any]:
        """``ionq.result.histogram.json.v2``: the same envelope under ``"histogram"`` with shot counts."""
        registers: dict[str, dict[str, int]] = {}
        for name, columns in self._v2_registers().items():
            registers[name] = dict(sorted(Counter(self._v2_strings(columns)).items()))
        return {"histogram": {"registers": registers}}

    def to_ionq_v2_shots(self) -> dict[str, Any]:
        """``ionq.result.shots.json.v2``: ``{"shots": [{"registers": {name: [bits...]}}, ...]}``, one bit array per register
        per shot in wire order (the register's first qubit first)."""
        columns = self._v2_registers()
        bits = np.asarray(self.bitstrings)
        return {
            "shots": [
                {"registers": {name: [int(bits[k, j]) for j in cols] for name, cols in columns.items()}}
                for k in range(bits.shape[0])
            ]
        }

    def reversed_bits(self) -> Result:
        """The same result with every key and every column of ``bitstrings`` reversed (``bit_order == "qubit0_msb"``),
        for comparisons with Cirq, Braket and PennyLane; ``qubits`` and ``registers`` keep naming the qubits, and a
        second call restores the order."""
        from dataclasses import replace

        bits = np.asarray(self.bitstrings)[:, ::-1]
        counts, probabilities = aggregate(bits)
        order: Literal["qubit0_lsb", "qubit0_msb"] = (
            "qubit0_msb" if self.bit_order == "qubit0_lsb" else "qubit0_lsb"
        )
        return replace(
            self,
            bitstrings=np.ascontiguousarray(bits),
            bit_order=order,
            counts=counts,
            probabilities=probabilities,
            error_bars={key[::-1]: v for key, v in self.error_bars.items()},
            photon_records=None
            if self.photon_records is None
            else np.ascontiguousarray(self.photon_records[:, ::-1]),
            posteriors=None if self.posteriors is None else np.ascontiguousarray(self.posteriors[:, ::-1]),
            sub_bin_records=None
            if self.sub_bin_records is None
            else np.ascontiguousarray(self.sub_bin_records[:, ::-1]),
            arrival_times_s=None
            if self.arrival_times_s is None
            else tuple(tuple(reversed(shot)) for shot in self.arrival_times_s),
            qubits=tuple(reversed(self.column_qubits)),
        )

    # ---- the versioned record ---------------------------------------------------------------------------------------------

    def to_dict(self, *, per_shot: bool = False) -> dict[str, Any]:
        """The result as JSON-able values (schema version 1); ``per_shot=True`` adds the per-shot arrays. Not carried:
        the noise samples, the final state, the calibration waveforms and the GATE_LOCAL report."""
        from qutip_trap import __version__

        heralds = np.asarray(self.heralds, dtype=int)
        out: dict[str, Any] = {
            "schema_version": 1,
            "qutip_trap_version": __version__,
            "device_hash": self.diagnostics.calibration.device_hash,
            "machine_hash": self.machine_hash,
            "created_at": self.created_at,
            "duration_s": float(self.duration_s),
            "shots": int(self.shots),
            "n_qubits": int(self.n_qubits),
            "qubits": [int(q) for q in self.column_qubits],
            "bit_order": self.bit_order,
            "root_seed": int(self.diagnostics.root_seed),
            "counts": {k: int(v) for k, v in self.counts.items()},
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "error_bars": {k: float(v) for k, v in self.error_bars.items()},
            "spam": {k: [float(a), float(b)] for k, (a, b) in self.spam.items()},
            "discarded_shots": int(self.discarded_shots),
            "registers": None
            if self.registers is None
            else {k: [int(q) for q in v] for k, v in self.registers.items()},
            "heralds": {
                "collision": int(np.count_nonzero(heralds & 1)),
                "dark_or_lost": int(np.count_nonzero(heralds & 2)),
                "count_anomaly": int(np.count_nonzero(heralds & 4)),
            },
            "run_state": self.run_state.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }
        if per_shot:
            out["per_shot"] = {
                "bitstrings": np.asarray(self.bitstrings, dtype=int).tolist(),
                "heralds": heralds.tolist(),
                "photon_records": None
                if self.photon_records is None
                else np.asarray(self.photon_records).tolist(),
                "posteriors": None
                if self.posteriors is None
                else np.asarray(self.posteriors, dtype=float).tolist(),
            }
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Result:
        """The inverse of :meth:`to_dict`; without per-shot arrays the bitstrings are rebuilt from the counts (in key
        order) with zero heralds."""
        if int(d.get("schema_version", -1)) != 1:
            raise ValueError(f"Result.from_dict reads schema version 1, got {d.get('schema_version')!r}")
        n = int(d["n_qubits"])
        per_shot = d.get("per_shot")
        if per_shot is not None:
            bits = np.asarray(per_shot["bitstrings"], dtype=np.uint8).reshape(-1, n)
            heralds = np.asarray(per_shot["heralds"], dtype=np.uint8)
            records = per_shot.get("photon_records")
            posteriors = per_shot.get("posteriors")
        else:
            rows = [
                np.array([int(ch) for ch in key[::-1]], dtype=np.uint8)
                for key, count in dict(d["counts"]).items()
                for _ in range(int(count))
            ]
            bits = (
                np.asarray(rows, dtype=np.uint8).reshape(-1, n) if rows else np.zeros((0, n), dtype=np.uint8)
            )
            heralds = np.zeros(bits.shape[0], dtype=np.uint8)
            records = None
            posteriors = None
        return cls(
            bitstrings=bits,
            bit_order=d["bit_order"],
            counts={k: int(v) for k, v in dict(d["counts"]).items()},
            probabilities={k: float(v) for k, v in dict(d["probabilities"]).items()},
            error_bars={k: float(v) for k, v in dict(d["error_bars"]).items()},
            photon_records=None if records is None else np.asarray(records, dtype=int),
            posteriors=None if posteriors is None else np.asarray(posteriors, dtype=float),
            noise_samples=(),
            heralds=heralds,
            discarded_shots=int(d["discarded_shots"]),
            run_state=RunState.from_dict(d["run_state"]),
            spam={k: (float(v[0]), float(v[1])) for k, v in dict(d["spam"]).items()},
            final_state=None,
            diagnostics=Diagnostics.from_dict(d["diagnostics"]),
            qubits=tuple(int(q) for q in d["qubits"]),
            registers=None
            if d.get("registers") is None
            else {k: tuple(int(q) for q in v) for k, v in dict(d["registers"]).items()},
            machine_hash=d.get("machine_hash"),
            created_at=str(d.get("created_at", "")),
            duration_s=float(d.get("duration_s", 0.0)),
        )

    @property
    def sample_of_shot(self) -> np.ndarray:
        """Per shot, the dynamical sample it was drawn from (``noise_samples[sample_of_shot[k]]`` is shot k's draw)."""
        realized = self.diagnostics.shots_per_sample_realized
        n = self.shots
        if not realized:
            return np.zeros(n, dtype=np.int64)
        out = np.concatenate([np.full(int(m), s, dtype=np.int64) for s, m in enumerate(realized)])
        # discarded shots (heralded collisions) shorten the kept array; the map covers what was kept
        return (
            out[:n] if out.shape[0] >= n else np.concatenate([out, np.full(n - out.shape[0], -1, np.int64)])
        )


__all__ = [
    "Diagnostics",
    "Progress",
    "Result",
    "RunState",
    "aggregate",
    "binomial_error_bars",
    "bits_from_decimal",
    "bitstring_key",
    "decimal_key",
]
