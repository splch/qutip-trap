"""The run record: every screen is a view of one of these frozen dataclasses of plain values and arrays (PLAN.md Section 14.3).

A job runs once (:func:`execute`); what re-simulation computes later (boundary states, zooms, Hamiltonian listings, process
matrices) is cached on the record by key. Conventions are the core's: bitstring keys read qubit 0 rightmost,
reduced internal states are in register order (ion 0 the first tensor factor), a mode index is a position in
``Crystal.modes``, and frequencies are in Hz.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from qutip_trap_app import core, knobs
from qutip_trap_app.codec import digest, dumps, encode, to_document

# ---- the job -----------------------------------------------------------------------------------------------------------------

PresetArg = bool | float | tuple[float, float, float]
FidelityLevelRequest = Literal["JOINT_EXACT", "GATE_LOCAL", "auto"]
FidelityLevelRun = Literal["JOINT_EXACT", "GATE_LOCAL", "CHANNEL_REPLAY"]
"""The two core levels of Section 5.4 plus the app-side channel replay (labelled derived)."""

PRESETS: dict[str, Callable[..., core.DevicePreset]] = {"yb171_chain": core.yb171_chain}
"""The device presets a record can name and rebuild."""

NATIVE_GATE_SET: tuple[str, ...] = ("gpi", "gpi2", "ms", "zz", "rz (virtual)")
"""The native gates the device card lists (Section 7.1); rz is a frame update, never a pulse."""

DEFAULT_DETECTION_WINDOWS_S: tuple[float, ...] = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
"""The detection windows the app's calibration scans."""


Progress = Callable[[str, float | None, str], None]
"""A worker progress callback: progress(stage, fraction or None, message)."""


class RecordError(ValueError):
    """The record cannot be built or rebuilt from what it was given."""


@dataclass(frozen=True)
class OpRecord:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...]
    """Radians (Section 13, "Gate parameters": turns only at the IonQ boundary)."""


@dataclass(frozen=True)
class CircuitRecord:
    n_qubits: int
    ops: tuple[OpRecord, ...]
    measure: tuple[int, ...]
    registers: dict[str, tuple[int, ...]]
    """The classical registers a result is reported under, name -> the qubits in bit order (``Circuit.registers``)."""

    def to_core(self) -> core.Circuit:
        return core.Circuit(
            self.n_qubits,
            tuple(core.Operation(op.name, op.qubits, op.params) for op in self.ops),
            self.measure,
            self.registers,
        )

    @classmethod
    def from_core(cls, circuit: core.Circuit) -> CircuitRecord:
        return cls(
            circuit.n_qubits,
            tuple(
                OpRecord(op.name, tuple(op.qubits), tuple(float(p) for p in op.params)) for op in circuit.ops
            ),
            tuple(circuit.measure),
            {str(name): tuple(int(q) for q in qubits) for name, qubits in circuit.registers.items()},
        )


@dataclass(frozen=True)
class DeviceRef:
    """The device a job runs on: a public preset with its arguments, the Level 4 knob overrides applied to it (Section
    14.4; ``knobs.apply_overrides``) and the canonical hash of the result (empty until the worker builds it)."""

    hash: str
    preset: str
    n_ions: int
    kwargs: dict[str, PresetArg]
    overrides: dict[str, float] = field(default_factory=dict)
    """Knob id -> value (``knobs.knobs_for``); empty for the preset as published."""

    def build(self, *, check: bool = True) -> core.DevicePreset:
        """Rebuild the preset with its overrides and check its hash against the recorded one (Section 7.5: a table or
        record is invalidated, never silently reused, when a device parameter changes)."""
        if self.preset not in PRESETS:
            raise RecordError(f"unknown device preset {self.preset!r}; known: {sorted(PRESETS)}")
        # the preset functions take keyword arguments of several types; the record's PresetArg union covers them
        preset = PRESETS[self.preset](self.n_ions, **cast(dict[str, Any], self.kwargs))
        if self.overrides:
            preset = knobs.apply_overrides(preset, self.overrides)
        if check and preset.device.hash() != self.hash:
            raise RecordError(
                f"the rebuilt device's hash {preset.device.hash()[:12]} differs from the recorded {self.hash[:12]}: "
                "the preset or the core changed since the record was made"
            )
        return preset

    def cache_key(self) -> str:
        """What identifies the built preset (the worker caches built presets by it)."""
        return f"{self.preset}/{self.n_ions}/{sorted(self.kwargs.items())}/{sorted(self.overrides.items())}"


@dataclass(frozen=True)
class DriveRef:
    """Which drive kind and beams play an ion's gates (``GateDrive``)."""

    kind: str
    beams: tuple[int, ...]

    @classmethod
    def from_core(cls, drive: core.GateDrive) -> DriveRef:
        return cls(str(drive.kind), tuple(drive.beams))


@dataclass(frozen=True)
class CalibrationRef:
    """The arguments of the job's surrogate calibration (Section 7.5's cache key)."""

    seed: int
    pairs: tuple[tuple[int, int], ...]
    detection_records: int
    detection_windows_s: tuple[float, ...]


@dataclass(frozen=True)
class JobSpec:
    """What the user asked for, re-runnable: the device, the seed and the convergence policy (``options``) of Section 14.3."""

    circuit: CircuitRecord
    shots: int
    seed: int
    device: DeviceRef
    gate_drives: dict[int, DriveRef]
    entangling_drives: dict[int, DriveRef]
    calibration: CalibrationRef
    options: core.Numerics
    """The numerics the run integrates with, its explicit Fock caps too (None lets the Section 5.5 cap rule decide)."""
    level: FidelityLevelRequest = "auto"
    readout: core.ReadoutMode = "fast"
    """``fast`` applies the readout POVM to each shot's outcome; ``full`` generates every ion's photon record and reads it
    (Section 5.7)."""
    waveform_overrides: dict[str, float] = field(default_factory=dict)
    """A detuning set by hand at Level 2 (Section 14.4): per entangling pair ``"a,b"``, the beat-note offset in Hz added to
    every blue leg and subtracted from every red leg of the pair's calibrated waveform, which the scheduler plays as written."""
    requests: tuple[str, ...] = ()
    """The requests made at a shallower level that produced this job (Section 14.4), in plain words."""
    label: str = ""
    """A name for the job when a preset or an exercise made it."""

    def machine(self, device: core.Device, table: core.CalibrationTable | None = None) -> core.Machine:
        """The ``Machine`` this job runs on: the device (which carries the drive roles), the table, the job's numerics and
        the default physics."""
        return core.Machine(
            device,
            table=table,
            numerics=self.options,
            readout=core.Readout(mode=self.readout),
            level=core.FidelityLevel(self.level),
        )


def options_digest(options: core.Numerics) -> str:
    """A short stable digest of the numerics, for cache keys."""
    return hashlib.sha256(dumps(encode(options, {}))).hexdigest()[:16]


def complete_job(job: JobSpec, preset: core.DevicePreset) -> JobSpec:
    """The job with the built preset's device hash and drive maps (the UI submits jobs without building a device)."""
    return dataclasses.replace(
        job,
        device=dataclasses.replace(job.device, hash=preset.device.hash()),
        gate_drives={int(i): DriveRef.from_core(d) for i, d in preset.gate_drives.items()},
        entangling_drives={int(i): DriveRef.from_core(d) for i, d in preset.entangling_drives.items()},
    )


def job_for_preset(
    preset_name: str,
    n_ions: int,
    circuit: core.Circuit,
    shots: int,
    *,
    seed: int = 0,
    options: core.Numerics | None = None,
    pairs: Sequence[tuple[int, int]] | None = None,
    detection_records: int = 2000,
    detection_windows_s: Sequence[float] = DEFAULT_DETECTION_WINDOWS_S,
    preset_kwargs: Mapping[str, PresetArg] | None = None,
    overrides: Mapping[str, float] | None = None,
    preset: core.DevicePreset | None = None,
    build: bool = True,
    **run_fields: Any,
) -> tuple[JobSpec, core.DevicePreset | None]:
    """A job on a public preset with ``overrides`` (the Level 4 knobs of Section 14.4) and the built preset (or ``preset``,
    already built for these arguments). With ``build=False`` nothing is built: the device hash and the drive maps stay
    empty until the worker completes the job (:func:`complete_job`), so the UI never waits for a device build."""
    if preset_name not in PRESETS:
        raise RecordError(f"unknown device preset {preset_name!r}; known: {sorted(PRESETS)}")
    kwargs = dict(preset_kwargs or {})
    ov = {k: float(v) for k, v in (overrides or {}).items()}
    if preset is None and build:
        preset = PRESETS[preset_name](n_ions, **cast(dict[str, Any], kwargs))
        if ov:
            preset = knobs.apply_overrides(preset, ov)
    pairs_t = tuple((int(a), int(b)) for a, b in (pairs if pairs is not None else circuit.entangling_pairs()))
    job = JobSpec(
        circuit=CircuitRecord.from_core(circuit),
        shots=int(shots),
        seed=int(seed),
        device=DeviceRef("", preset_name, int(n_ions), kwargs, ov),
        gate_drives={},
        entangling_drives={},
        calibration=CalibrationRef(
            seed=int(seed),
            pairs=pairs_t,
            detection_records=int(detection_records),
            detection_windows_s=tuple(float(w) for w in detection_windows_s),
        ),
        options=options or core.Numerics(),
        **run_fields,
    )
    return (job if preset is None else complete_job(job, preset)), preset


# ---- the device card (Level 0) and the calibration table -----------------------------------------------------------------------


@dataclass(frozen=True)
class ModeRecord:
    index: int
    family: str
    family_index: int
    omega_hz: float


@dataclass(frozen=True)
class BeamRecord:
    index: int
    wavelength_m: float
    k_hat: tuple[float, float, float]


@dataclass(frozen=True)
class DetectorRecord:
    efficiency: float
    window_s: float


@dataclass(frozen=True)
class DerivedRecord:
    """``Device.derived()``: every computed number with its ledger id (Section 3.3)."""

    values: dict[str, float]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CalEntryRecord:
    value: float
    status: str
    experiment: str
    provenance_id: str


@dataclass(frozen=True)
class SampledFn:
    """A tone function as the record can hold it: a constant, or values on a uniform grid over the pulse."""

    kind: Literal["constant", "sampled", "array"]
    value: float | None
    samples: np.ndarray | None

    def at(self, tau_s: np.ndarray, duration_s: float) -> np.ndarray:
        if self.kind == "constant":
            assert self.value is not None
            return np.full(np.asarray(tau_s).shape, self.value, dtype=float)
        assert self.samples is not None
        grid = np.linspace(0.0, duration_s, self.samples.size)
        return np.asarray(np.interp(np.asarray(tau_s, dtype=float), grid, self.samples))


@dataclass(frozen=True)
class SegmentRecord:
    duration_s: float
    amplitude_hz: dict[str, SampledFn]
    """``"ion,leg"`` -> Omega (Section 7.4; the key is the flattened (ion, leg) pair)."""
    detuning_hz: dict[str, SampledFn]
    """``leg`` -> detuning from the carrier."""


@dataclass(frozen=True)
class WaveformRecord:
    duration_s: float
    chi_m: dict[int, float]
    """Signed per-mode entangling angle at closure (conv.entangling_angle, conv.entangling_sign)."""
    alpha_m: dict[int, complex]
    """Per-mode residual displacement at closure (Section 4.4.3)."""
    chi_total_rad: float
    phi_s: CalEntryRecord
    phi_m: CalEntryRecord
    segments: tuple[SegmentRecord, ...]


@dataclass(frozen=True)
class TableRecord:
    device_hash: str
    entries: dict[str, CalEntryRecord]
    waveforms: dict[str, WaveformRecord]
    """``"a,b"`` -> the pair's entangling waveform."""


@dataclass(frozen=True)
class DeviceCard:
    """Level 0's device card (Section 14.2): species, ion count, native gate set, calibrated and estimated gate errors, SPAM."""

    species: tuple[str, ...]
    n_ions: int
    modes: tuple[ModeRecord, ...]
    trap_omega_hz: tuple[float, float, float] | None
    field_gauss: float
    field_direction: tuple[float, float, float]
    beams: tuple[BeamRecord, ...]
    detector: DetectorRecord
    heating_quanta_per_s: dict[int, float]
    derived: DerivedRecord
    native_gates: tuple[str, ...]
    spam: dict[str, tuple[float, float]]
    """Per qubit ``q{i}`` -> (eps_B, eps_D), plus ``q{i}.state_preparation`` (Section 13, "Readout figure of merit")."""
    gate_error_estimates: dict[str, float]
    """Per played gate, the summed closed-form error scales of Section 9.6 (an ESTIMATE, labelled so on the card)."""
    gate_error_tomography: dict[str, float]
    """Per gate step, the average gate infidelity from process tomography where the run produced one (GATE_LOCAL)."""


# ---- compiled circuit, schedule, space, preparation -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledRecord:
    native: CircuitRecord
    final_frame_rad: dict[int, float]
    block_residuals: tuple[float, ...]
    circuit_residual: float | None


@dataclass(frozen=True)
class ToneRecord:
    detuning_hz: SampledFn
    phase_rad: SampledFn
    envelope_hz: SampledFn


@dataclass(frozen=True)
class PulseRecord:
    index: int
    gate_id: str | None
    ions: tuple[int, ...]
    kind: str
    beams: tuple[int, ...]
    t_start_s: float
    t_end_s: float
    tones: tuple[ToneRecord, ...]
    stark_shift_hz: SampledFn
    crosstalk: dict[int, complex]
    """Rabi (amplitude) ratio onto neighbours (conv.crosstalk_ratio)."""
    closes_modes: tuple[int, ...]

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


@dataclass(frozen=True)
class EventRecord:
    kind: str
    ions: tuple[int, ...]
    t_start_s: float
    t_end_s: float


@dataclass(frozen=True)
class PlayedGateRecord:
    gate_id: str
    pair: tuple[int, int]
    waveform: WaveformRecord


@dataclass(frozen=True)
class TargetRecord:
    """The ideal physical unitary of one played gate piece (Section 5.4; ``GateTarget``)."""

    gate_id: str
    ions: tuple[int, ...]
    native_name: str
    native_params: tuple[float, ...]
    stark_frame_rad: dict[int, float]
    pulse_indices: tuple[int, ...]
    t_start_s: float
    t_end_s: float
    unitary: np.ndarray
    """On ``ions`` in matrix order (first ion the first factor)."""


@dataclass(frozen=True)
class StepRecord:
    """One unit of the zoom ladder's time axis: a gate piece with all its pulses, or an idle interval (``gate_steps``)."""

    index: int
    kind: Literal["gate", "idle"]
    gate_id: str
    t_start_s: float
    t_end_s: float
    pulse_indices: tuple[int, ...]
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleRecord:
    pulses: tuple[PulseRecord, ...]
    idle: tuple[tuple[float, float], ...]
    events: tuple[EventRecord, ...]
    phase_frame: dict[int, float]
    gates: tuple[PlayedGateRecord, ...]
    targets: tuple[TargetRecord, ...]
    steps: tuple[StepRecord, ...]
    t0_s: float
    pulses_end_s: float
    duration_s: float


@dataclass(frozen=True)
class TruncationRecord:
    mode: int
    d: int
    expected_n_range: tuple[int, int]
    eta_max: float


@dataclass(frozen=True)
class SpaceRecord:
    resolved: tuple[TruncationRecord, ...]
    dropped: tuple[int, ...]
    enr_group: tuple[tuple[int, ...], int] | None
    dims: tuple[int, ...]
    dimension: int
    mode_class: dict[int, str]
    nbar: dict[int, float]


# ---- dynamics: samples, branches, traces -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseSampleRecord:
    sample_id: int
    t_s: float
    values: dict[str, float]
    grids: dict[str, np.ndarray]
    """Fixed-grid (2, N) realizations of the fast processes (Section 5.5)."""

    def to_core(self, extra: Mapping[str, float] | None = None) -> core.NoiseSample:
        values = dict(self.values)
        if extra:
            values.update(extra)
        return core.NoiseSample(
            sample_id=self.sample_id, values=values, ou_grids=dict(self.grids), t_s=self.t_s
        )


@dataclass(frozen=True)
class BranchRecord:
    index: int
    weight: float
    levels: tuple[int, ...]
    fock: dict[int, int]


@dataclass(frozen=True)
class BranchLoopRecord:
    """The phase-space trajectory alpha_im(t) of ONE spin branch (ion ``ion`` in the +1 eigenstate of its sigma_phi) for one
    mode of a played entangling waveform, integrated on the run's own modes and Lamb-Dicke parameters by the closed-form
    kernel (Sections 4.4.1, 4.4.3): the loop the pulse solver closed, whose swept area is the entangling angle. The exact
    trace's <a_m>(t) averages the branches, whose displacements cancel for a register with <S_phi> = 0."""

    gate_id: str
    ion: int
    mode: int
    eta: float
    """eta_{ion, mode} the loop was integrated with (C0 inside, Section 4.1.1)."""
    alpha: np.ndarray
    """Complex alpha_im(t) on the integration grid decimated to ``LOOP_STORE`` points, the first and last kept."""
    chi_m_rad: float
    """The played waveform's booked per-mode entangling angle: the table's exact spot-check angle (Section 7.8) times the
    scale the scheduler applied to reach the requested gate angle."""
    chi_closed_form_rad: float
    """2 Im int conj(alpha_a) d alpha_b over the pair's two loops on this mode (Section 4.4.3): the closed-form angle at the
    played amplitude; its gap to ``chi_m_rad`` is what the exact spot check corrected (Debye-Waller, beyond Lamb-Dicke)."""

    @property
    def closes(self) -> float:
        """|alpha(end) - alpha(start)|: how far the loop failed to close."""
        return float(abs(self.alpha[-1] - self.alpha[0])) if self.alpha.size else 0.0

    @property
    def excursion(self) -> float:
        """max |alpha(t) - alpha(start)|: the loop's reach."""
        return float(np.max(np.abs(self.alpha - self.alpha[0]))) if self.alpha.size else 0.0


@dataclass(frozen=True)
class TraceRecord:
    """One (sample, branch) evolution of the run at the times the run stored (Section 14.3)."""

    sample_index: int
    sample_id: int
    branch: int
    weight: float
    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    """``P1[i]`` and ``n[m]`` as the engine keys them."""
    reduced_internal: np.ndarray
    """(T, d, d) reduced register density matrices in the register order."""
    ion_dims: tuple[int, ...]
    mode_nbar: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
    boundary_population: dict[int, float]
    final_internal: np.ndarray
    final_mode_reduced: dict[int, np.ndarray]
    final_nbar: dict[int, float]
    joint_dims: tuple[int, ...]
    """The joint state at the last stored time; the app does not store it (None)."""
    mode_marginal: dict[int, np.ndarray] | None = None
    """Per resolved mode the (T, d) Fock populations at the stored times (``core.Traces.mode_marginal``) when the run stored
    them (``store_marginals``, which a zoom turns on); None otherwise."""

    def index_at(self, t_s: float, tolerance_s: float = 1e-12) -> int:
        """The LAST stored index at time ``t_s`` (a segment end)."""
        hits = np.flatnonzero(np.abs(self.times_s - t_s) <= tolerance_s)
        if hits.size == 0:
            raise KeyError(f"no stored point at t = {t_s:.9g} s")
        return int(hits[-1])


@dataclass(frozen=True)
class ChannelSummaryRecord:
    average_gate_infidelity: float
    depolarizing_rate: float


@dataclass(frozen=True)
class GateLocalStepRecord:
    gate_id: str
    summary: ChannelSummaryRecord | None
    register_after: np.ndarray | None
    """The register density matrix after this step for the first sample (``core.GateLocalStep.register_after``), ion 0 the
    first factor; None for a pure-state ensemble or a register above the core's store cap."""


@dataclass(frozen=True)
class GateLocalRecord:
    steps: tuple[GateLocalStepRecord, ...]
    discrepancy_bound: float
    largest_local_dimension: int


# ---- readout, results, diagnostics ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutRecord:
    window_s: float
    threshold: float | None
    levels: np.ndarray
    """(shots, n_ions) the projectively sampled internal level per ion (the joint outcome behind the declared bits)."""
    time_used_s: np.ndarray
    photon_records: np.ndarray | None
    """(shots, n_ions) the photon count of every ion's record, ion 0 first, when the run kept its records (the full
    readout), the declared bits being the discriminator's reading of the measured ions' records; None on the fast path."""
    posteriors: np.ndarray | None


@dataclass(frozen=True)
class ResultsRecord:
    bitstrings: np.ndarray
    heralds: np.ndarray
    counts: dict[str, int]
    probabilities: dict[str, float]
    error_bars: dict[str, float]
    target_probabilities: dict[str, float]
    """The compiler's target distribution (ideal statevector of the source circuit), shown BESIDE the simulated one."""
    effective_sample_size: float
    discarded_shots: int
    sample_of_shot: np.ndarray
    final_state: np.ndarray | None
    """The recombined register density matrix (register order) when kept."""
    register_fidelity: float | None
    """<ideal| rho |ideal> against the compiled circuit's state with its frame absorbed (``register_fidelity``)."""


@dataclass(frozen=True)
class ConvergenceRecord:
    tolerances: tuple[float, float]
    tightened_tolerances: tuple[float, float]
    changes: dict[str, float]
    tol: float
    converged: bool
    max_change: float


@dataclass(frozen=True)
class DiagnosticsRecord:
    level: FidelityLevelRun
    integrator: str
    tolerances: tuple[float, float]
    samples: int
    trajectories: int
    branches: int
    shots_per_sample_realized: tuple[int, ...]
    boundary_population: dict[int, float]
    boundary_population_max: float
    """The policy threshold the run was judged against (``Numerics.boundary_population_max``)."""
    margin_levels: dict[int, int]
    margin_reached: dict[int, int]
    dropped_branch_weight: float
    dropped_contribution: tuple[float, float]
    frozen_contribution: dict[int, tuple[float, float]]
    intrinsic_budget: dict[str, float]
    approximations: tuple[str, ...]
    kernel: str
    workers: int
    wall_time_s: float
    convergence: ConvergenceRecord | None


# ---- the channel replay (app-side, derived) ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelPieceRecord:
    """One gate step's channel at phase zero, as the core's GATE_LOCAL tomography extracted it (Section 5.4 (a))."""

    gate_id: str
    ions: tuple[int, ...]
    """The step's ions in matrix order: the addressed ions and the crosstalk neighbours above the threshold."""
    choi: np.ndarray
    ideal: np.ndarray
    """The target unitary on ``ions`` at phase zero (identity on the neighbours)."""
    summary: ChannelSummaryRecord
    residual_bound: float
    frozen_excitation: float
    dropped_crosstalk: float
    cp_residual: float
    tp_residual: float

    @property
    def derivation_residual(self) -> float:
        """What replaying this piece cannot account for: the traced-out correlations (Section 9.8's bound), the frozen
        spectators' excitation, the dropped crosstalk and the CP/TP projection residuals."""
        return (
            self.residual_bound
            + self.frozen_excitation
            + self.dropped_crosstalk
            + self.cp_residual
            + self.tp_residual
        )


@dataclass(frozen=True)
class ChannelEntryRecord:
    """The channel of one gate kind on its ions at one angle, with its measured frame covariance."""

    key: str
    kind: str
    addressed: tuple[int, ...]
    angle: float | None
    pieces: tuple[ChannelPieceRecord, ...]
    covariance_residual: float | None
    """max |Choi(phi) - R Choi(0) R^dag| at the covariance-check phase; None until checked."""

    @property
    def derivation_residual(self) -> float:
        return sum(p.derivation_residual for p in self.pieces) + (self.covariance_residual or 0.0)

    @property
    def average_gate_infidelity(self) -> float:
        return float(sum(p.summary.average_gate_infidelity for p in self.pieces))

    @property
    def depolarizing_rate(self) -> float:
        return float(sum(p.summary.depolarizing_rate for p in self.pieces))


@dataclass(frozen=True)
class ReplayGate:
    gate_id: str
    key: str
    average_gate_infidelity: float


@dataclass(frozen=True)
class ReplayRecord:
    """What the channel replay did: which channel each gate was played as, the register after each gate, and the
    channel-derivation residual the replay reports."""

    gates: tuple[ReplayGate, ...]
    register_after: np.ndarray
    """(n_gates, 2^n, 2^n): the register after each gate piece, register order."""
    channels: dict[str, ChannelEntryRecord]
    residual_terms: dict[str, float]
    residual_total: float


# ---- what re-simulation caches on the record ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryState:
    """The joint state at a gate-step boundary for one (sample, branch), recomputed by chaining and cached (Section 14.3)."""

    step_index: int
    """The state is at the START of this step (step ``len(steps)`` = the end of the schedule)."""
    sample_index: int
    branch: int
    joint: np.ndarray | None
    joint_dims: tuple[int, ...]
    internal: np.ndarray
    mode_reduced: dict[int, np.ndarray]
    nbar: dict[int, float]


@dataclass(frozen=True)
class ZoomTrace:
    """A re-simulated gate step at fine time resolution (Level 3), cached under its key."""

    key: str
    step_index: int
    sample_index: int
    branch: int
    n_store: int
    options_digest: str
    trace: TraceRecord
    fock_end: dict[int, np.ndarray]
    """Per resolved mode, the Fock distribution at the END of the step (from the final reduced motional state)."""
    fock_start: dict[int, np.ndarray]
    integrators: tuple[str, ...]
    method: str
    approximations: tuple[str, ...]
    wall_time_s: float
    engine_dimension: int


@dataclass(frozen=True)
class DriveTermRecord:
    """One (pulse, ion) drive term the builder assembled: sigma_+^ion (x) prod_m D_m(i eta_im) with its scalar coefficient
    (Section 5.2), and what went into it (``DriveRecord`` of the core plus the tone summary of the record's pulse)."""

    pulse: str
    ion: int
    primary_ion: int
    kind: str
    beams: tuple[int, ...]
    etas: dict[int, float]
    frozen_n: dict[int, int]
    debye_waller: float
    micromotion_beta: float
    carrier_factor: float
    crosstalk: complex
    rabi_scale: float
    omega_peak_hz: float
    tone_detunings_hz: tuple[float, ...]
    tone_phases_rad: tuple[float, ...]
    tone_peaks_hz: tuple[float, ...]
    operator_nnz: int
    matrix_elements: dict[int, np.ndarray]
    """Per resolved mode, the (d, d) table Omega_{n',n}/Omega = |<n'|D(i eta)|n>| at this term's eta (Section 4.3.1)."""
    frozen_debye_waller: dict[int, float]
    """Per frozen mode, e^{-eta^2/2} L_n(eta^2) at the sample's Fock state n (the factor the builder multiplied in)."""


@dataclass(frozen=True)
class CollapseRecord:
    channel: str
    rate_hz: float
    ion: int | None
    mode: int | None
    time_dependent: bool
    operator_nnz: int
    active_in_run: bool
    """Whether the recorded run integrated this channel (a scattering channel exists only when the run switched it on)."""
    note: str


@dataclass(frozen=True)
class SegmentSummary:
    t_start_s: float
    t_end_s: float
    pulses: tuple[str, ...]
    n_drive_terms: int
    omega_max_hz: float
    kernel: str


@dataclass(frozen=True)
class HamiltonianRecord:
    """The H(t) terms and collapse operators the engine assembles for one gate step and one (sample, branch), listed rather
    than integrated (Sections 5.7, 14.2 row 4)."""

    key: str
    gate_id: str
    t_start_s: float
    t_end_s: float
    frame: str
    dims: tuple[int, ...]
    dimension: int
    mode_frequencies_hz: dict[int, float]
    mode_offsets_hz: dict[int, float]
    qubit_offsets_hz: dict[int, float]
    mode_classes: dict[int, str]
    caps: dict[int, int]
    segments: tuple[SegmentSummary, ...]
    drives: tuple[DriveTermRecord, ...]
    """The drive terms of the FIRST segment (the operators are the same on every segment of a step; the coefficients differ)."""
    collapse: tuple[CollapseRecord, ...]
    approximations: tuple[str, ...]
    n_drive_terms: int
    omega_max_hz: float
    kernel: str
    fingerprint: str
    free_term_nnz: int
    stark_shifts_hz: dict[int, float]
    """Per addressed ion, the differential light shift the pulse's own beams impose (H_Stark of Section 5.7), at the pulse start."""


@dataclass(frozen=True)
class ProcessMatrixRecord:
    """Process tomography of one gate step from its recorded initial motional state (Section 5.4 (a)), for one (sample,
    branch)."""

    key: str
    gate_id: str
    ions: tuple[int, ...]
    choi: np.ndarray
    ideal: np.ndarray
    n_inputs: int
    cp_tp_residual: tuple[float, float]
    average_gate_infidelity: float
    entanglement_infidelity: float
    depolarizing_rate: float
    pauli_twirled: dict[str, float]
    n_traj: int
    wall_time_s: float
    method: str


# ---- the record -----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """The run record of Section 14.3: everything the five levels read, and nothing else."""

    job: JobSpec
    device_hash: str
    table: TableRecord
    device_card: DeviceCard
    compiled: CompiledRecord
    schedule: ScheduleRecord
    space: SpaceRecord
    noise_samples: tuple[NoiseSampleRecord, ...]
    branches: tuple[BranchRecord, ...]
    traces: tuple[TraceRecord, ...]
    gate_local: GateLocalRecord | None
    readout: ReadoutRecord
    results: ResultsRecord
    diagnostics: DiagnosticsRecord
    branch_loops: tuple[BranchLoopRecord, ...]
    """The spin-branch loops of every played entangling waveform (Level 3's phase-space drawing)."""
    replay: ReplayRecord | None = None
    """Present when the record was made by the app-side channel replay (Level 0's default engine)."""
    boundaries: tuple[BoundaryState, ...] = ()
    zooms: tuple[ZoomTrace, ...] = ()
    hamiltonians: tuple[HamiltonianRecord, ...] = ()
    process_matrices: tuple[ProcessMatrixRecord, ...] = ()

    def key(self) -> str:
        """The identity of the RUN behind the record: the digest without the caches and the wall time, so a record with more
        zooms cached keeps its key (the key of the UI's store and of the worker's live state)."""
        bare = dataclasses.replace(
            self,
            boundaries=(),
            zooms=(),
            hamiltonians=(),
            process_matrices=(),
            diagnostics=dataclasses.replace(self.diagnostics, wall_time_s=0.0),
        )
        doc, arrays = to_document(bare)
        return digest(doc, arrays)[:24]

    @property
    def n_qubits(self) -> int:
        """The circuit's qubits (its wires). The REGISTER the run carries spans every ion: see :attr:`n_ions`."""
        return self.job.circuit.n_qubits

    @property
    def n_ions(self) -> int:
        """The ions of the crystal: the size of every register-order state on the record. A circuit on fewer qubits than
        ions addresses the leading ions and leaves the rest in |0>."""
        return self.device_card.n_ions

    @property
    def n_samples(self) -> int:
        return len(self.noise_samples)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    def trace(self, sample_index: int, branch: int) -> TraceRecord:
        for tr in self.traces:
            if tr.sample_index == sample_index and tr.branch == branch:
                return tr
        raise KeyError(f"no trace for sample {sample_index}, branch {branch} (a GATE_LOCAL run stores none)")

    def step(self, index: int) -> StepRecord:
        return self.schedule.steps[index]

    def step_of_gate(self, gate_id: str) -> StepRecord:
        for st in self.schedule.steps:
            if gate_id in st.target_ids or st.gate_id == gate_id:
                return st
        raise KeyError(f"no gate step carries gate {gate_id!r}")

    def boundary(self, step_index: int, sample_index: int, branch: int) -> BoundaryState | None:
        return next(
            (
                b
                for b in self.boundaries
                if (b.step_index, b.sample_index, b.branch) == (step_index, sample_index, branch)
            ),
            None,
        )

    def zoom(self, key: str) -> ZoomTrace | None:
        return next((z for z in self.zooms if z.key == key), None)

    def hamiltonian(self, key: str) -> HamiltonianRecord | None:
        return next((h for h in self.hamiltonians if h.key == key), None)

    def process_matrix(self, key: str) -> ProcessMatrixRecord | None:
        return next((pm for pm in self.process_matrices if pm.key == key), None)

    def with_boundaries(self, new: Sequence[BoundaryState]) -> Record:
        at = {(b.step_index, b.sample_index, b.branch): b for b in (*self.boundaries, *new)}
        ordered = sorted(at.values(), key=lambda b: (b.sample_index, b.branch, b.step_index))
        return dataclasses.replace(self, boundaries=tuple(ordered))

    def with_zoom(self, zoom: ZoomTrace) -> Record:
        return dataclasses.replace(self, zooms=(*(z for z in self.zooms if z.key != zoom.key), zoom))

    def with_hamiltonian(self, ham: HamiltonianRecord) -> Record:
        return dataclasses.replace(
            self, hamiltonians=(*(h for h in self.hamiltonians if h.key != ham.key), ham)
        )

    def with_process_matrix(self, pm: ProcessMatrixRecord) -> Record:
        return dataclasses.replace(
            self, process_matrices=(*(x for x in self.process_matrices if x.key != pm.key), pm)
        )


# ---- building a record from a core Result ----------------------------------------------------------------------------------------

TONE_SAMPLES = 257
"""Grid points at which a callable tone function is sampled for the record (a display resolution, not a solver one)."""


def sampled_fn(value: object, duration_s: float) -> SampledFn:
    """A tone function (constant, callable of the time since the pulse start, or a uniformly sampled array) for the record."""
    if isinstance(value, np.ndarray):
        return SampledFn("array", None, np.asarray(value, dtype=float))
    if callable(value):
        grid = np.linspace(0.0, duration_s, TONE_SAMPLES)
        return SampledFn("sampled", None, np.array([float(value(float(t))) for t in grid], dtype=float))
    return SampledFn("constant", float(cast(float, value)), None)


def vec3(values: Any) -> tuple[float, float, float]:
    """Three components as floats (a direction, a position, the three secular frequencies)."""
    return (float(values[0]), float(values[1]), float(values[2]))


def embed_on_register(op: np.ndarray, ions: Sequence[int], n_ions: int) -> np.ndarray:
    """``op`` on ``ions`` (its first factor the first listed ion) as an operator on the whole two-level register in register
    order (ion 0 the first factor), identity elsewhere."""
    return core.embed(np.asarray(op, dtype=complex), [n_ions - 1 - int(i) for i in ions], n_ions)


def _cal_entry(entry: core.CalEntry) -> CalEntryRecord:
    return CalEntryRecord(
        value=float(entry.value),
        status=str(entry.status),
        experiment=str(entry.experiment),
        provenance_id=str(entry.provenance_id),
    )


def waveform_record(wf: core.Waveform) -> WaveformRecord:
    segments = tuple(
        SegmentRecord(
            duration_s=float(s.duration_s),
            amplitude_hz={
                f"{ion},{leg}": sampled_fn(v, s.duration_s) for (ion, leg), v in s.amplitude_hz.items()
            },
            detuning_hz={str(leg): sampled_fn(v, s.duration_s) for leg, v in s.detuning_hz.items()},
        )
        for s in wf.segments
    )
    return WaveformRecord(
        duration_s=float(wf.duration_s),
        chi_m={int(m): float(v) for m, v in wf.chi_m.items()},
        alpha_m={int(m): complex(v) for m, v in wf.alpha_m.items()},
        chi_total_rad=float(wf.chi_total_rad),
        phi_s=_cal_entry(wf.phi_s),
        phi_m=_cal_entry(wf.phi_m),
        segments=segments,
    )


def table_record(table: core.CalibrationTable) -> TableRecord:
    return TableRecord(
        device_hash=str(table.device_hash),
        entries={k: _cal_entry(e) for k, e in table.entries().items()},
        waveforms={f"{a},{b}": waveform_record(wf) for (a, b), wf in table.ms.items()},
    )


def device_card(
    device: core.Device,
    *,
    spam: Mapping[str, tuple[float, float]],
    intrinsic_budget: Mapping[str, float],
    tomography: Mapping[str, float],
) -> DeviceCard:
    """The Level 0 device card from the device and the run's SPAM, closed-form error scales and tomography infidelities."""
    derived = device.derived()
    crystal = device.crystal
    modes = tuple(
        ModeRecord(
            index=k,
            family=str(m.family),
            family_index=int(m.index),
            omega_hz=float(m.omega_hz),
        )
        for k, m in enumerate(crystal.modes)
    )
    beams = tuple(
        BeamRecord(
            index=k,
            wavelength_m=float(b.wavelength_m),
            k_hat=vec3(b.k_hat),
        )
        for k, b in enumerate(device.beams)
    )
    det = device.detector
    estimates: dict[str, float] = {}
    for key, value in intrinsic_budget.items():
        if key == "total" or "." not in key:
            continue
        gate = key.split(".")[0]
        estimates[gate] = estimates.get(gate, 0.0) + float(value)
    omega = device.trap.omega_hz
    return DeviceCard(
        species=tuple(s.name for s in crystal.species),
        n_ions=int(crystal.n_ions),
        modes=modes,
        trap_omega_hz=None if omega is None else vec3(omega),
        field_gauss=float(device.field.B_gauss),
        field_direction=vec3(device.field.direction),
        beams=beams,
        detector=DetectorRecord(
            efficiency=float(det.efficiency),
            window_s=float(det.window_s),
        ),
        heating_quanta_per_s={
            int(m): float(v) for m, v in device.noise.heating_rates_quanta_per_s(device).items()
        },
        derived=DerivedRecord(
            values={k: float(v) for k, v in derived.values.items()},
            notes=tuple(derived.notes),
        ),
        native_gates=NATIVE_GATE_SET,
        spam={str(k): (float(v[0]), float(v[1])) for k, v in spam.items()},
        gate_error_estimates=estimates,
        gate_error_tomography={str(k): float(v) for k, v in tomography.items() if not math.isnan(v)},
    )


def compiled_record(report: core.CompileReport) -> CompiledRecord:
    return CompiledRecord(
        native=CircuitRecord.from_core(report.circuit),
        final_frame_rad={int(q): float(v) for q, v in report.final_frame_rad.items()},
        block_residuals=tuple(float(r) for r in report.block_residuals),
        circuit_residual=None if report.circuit_residual is None else float(report.circuit_residual),
    )


def schedule_record(sched: core.Schedule) -> ScheduleRecord:
    index_of = {id(p): k for k, p in enumerate(sched.pulses)}
    pulses = tuple(
        PulseRecord(
            index=k,
            gate_id=p.gate_id,
            ions=tuple(int(i) for i in p.drive.ions),
            kind=str(p.drive.kind),
            beams=tuple(int(b) for b in p.drive.beams),
            t_start_s=float(p.t_start_s),
            t_end_s=float(p.t_end_s),
            tones=tuple(
                ToneRecord(
                    detuning_hz=sampled_fn(t.detuning_hz, p.duration_s),
                    phase_rad=sampled_fn(t.phase_rad, p.duration_s),
                    envelope_hz=sampled_fn(t.envelope_hz, p.duration_s),
                )
                for t in p.drive.tones
            ),
            stark_shift_hz=sampled_fn(p.drive.stark_shift_hz, p.duration_s),
            crosstalk={int(j): complex(e) for j, e in p.drive.crosstalk.items()},
            closes_modes=tuple(int(m) for m in p.closes_modes),
        )
        for k, p in enumerate(sched.pulses)
    )
    gates = tuple(
        PlayedGateRecord(
            gate_id=str(g.gate_id),
            pair=(int(g.pair[0]), int(g.pair[1])),
            waveform=waveform_record(g.waveform),
        )
        for g in sched.gates
    )
    pulse_by_gate_id: dict[str, list[int]] = {}
    for k, p in enumerate(sched.pulses):
        if p.gate_id is not None:
            pulse_by_gate_id.setdefault(p.gate_id, []).append(k)
    targets = tuple(
        TargetRecord(
            gate_id=str(t.gate_id),
            ions=tuple(int(i) for i in t.ions),
            native_name=str(t.native[0]),
            native_params=tuple(float(x) for x in t.native[1]),
            stark_frame_rad={int(q): float(v) for q, v in t.stark_frame_rad.items()},
            pulse_indices=tuple(idx for pid in t.pulse_ids for idx in pulse_by_gate_id.get(pid, ())),
            t_start_s=float(t.t_start_s),
            t_end_s=float(t.t_end_s),
            unitary=np.asarray(t.unitary(), dtype=complex),
        )
        for t in sched.targets
    )
    steps = tuple(
        StepRecord(
            index=k,
            kind=s.kind,
            gate_id=str(s.gate_id),
            t_start_s=float(s.t_start_s),
            t_end_s=float(s.t_end_s),
            pulse_indices=tuple(index_of[id(p)] for p in s.pulses),
            target_ids=tuple(str(t.gate_id) for t in s.targets),
        )
        for k, s in enumerate(core.gate_steps(sched))
    )
    starts = [p.t_start_s for p in sched.pulses] + [a for a, _ in sched.idle]
    t0 = float(sched.t0_s) if sched.t0_s is not None else min(starts + [0.0])
    return ScheduleRecord(
        pulses=pulses,
        idle=tuple((float(a), float(b)) for a, b in sched.idle),
        events=tuple(
            EventRecord(str(e.kind), tuple(int(i) for i in e.ions), float(e.t_start_s), float(e.t_end_s))
            for e in sched.events
        ),
        phase_frame={int(q): float(v) for q, v in sched.phase_frame.items()},
        gates=gates,
        targets=targets,
        steps=steps,
        t0_s=t0,
        pulses_end_s=float(sched.pulses_end_s),
        duration_s=float(sched.duration_s),
    )


def space_record(space: core.HilbertSpace, selection: core.SpaceSelection) -> SpaceRecord:
    return SpaceRecord(
        resolved=tuple(
            TruncationRecord(
                int(m.mode),
                int(m.d),
                (int(m.expected_n_range[0]), int(m.expected_n_range[1])),
                float(m.eta_max),
            )
            for m in space.resolved
        ),
        dropped=tuple(int(m) for m in space.dropped),
        enr_group=None
        if space.enr_group is None
        else (tuple(int(m) for m in space.enr_group[0]), int(space.enr_group[1])),
        dims=tuple(int(d) for d in space.dims),
        dimension=int(space.dimension),
        mode_class={int(m): str(c) for m, c in selection.mode_class.items()},
        nbar={int(m): float(v) for m, v in selection.nbar.items()},
    )


def _qobj_array(q: Any) -> np.ndarray:
    return np.asarray(q.full(), dtype=complex)


LOOP_GRID = 20001
"""Points of the uniform Simpson grid a spin-branch loop is integrated on (the core's own default for sampled envelopes)."""
LOOP_STORE = 1001
"""Points of a stored loop: every twentieth grid point, the first and last always kept."""


def branch_loops(
    sched: core.Schedule, device: core.Device, nbar: Mapping[int, float]
) -> tuple[BranchLoopRecord, ...]:
    """The spin-branch loops of every played entangling waveform on the run's own modes (Section 4.4.1).

    alpha_im(t) by the closed-form kernel the solver closed the pulse with ("choi": the force cos(mu t) with its
    counter-rotating part), on a uniform Simpson grid of ``LOOP_GRID`` points, stored at ``LOOP_STORE`` points with both ends
    kept. The pair's closed-form angle per mode, 2 Im int conj(alpha_a) d alpha_b, comes from the analytic segment integrals
    when the waveform is segmented and from the grid otherwise. A Fourier-parameterized waveform carries no explicit envelope
    and gets no loop."""
    out: list[BranchLoopRecord] = []
    for g in sched.gates:
        wf = g.waveform
        if len(g.beams) != 2:
            continue
        pair = (int(g.pair[0]), int(g.pair[1]))
        gm = core.gate_modes(device, pair, (int(g.beams[0]), int(g.beams[1])), nbar=nbar)
        env = core.envelope_of(wf, pair)
        if isinstance(env, core.SegmentedEnvelope):
            analytic = core.integrals_segmented(env, gm, "choi").chi_by_mode
            sampled = env.sampled(LOOP_GRID)
        else:
            analytic = None
            sampled = env
        n = int(sampled.times_s.size)
        stride = max(1, (n - 1) // max(1, LOOP_STORE - 1))
        keep = np.arange(0, n, stride)
        if keep[-1] != n - 1:
            keep = np.append(keep, n - 1)
        for k, m in enumerate(gm.modes):
            paths = {ion: core.trajectory_sampled(sampled, gm, ion, int(m), kernel="choi") for ion in pair}
            a, b = paths[pair[0]], paths[pair[1]]
            if analytic is not None:
                chi_closed = float(
                    analytic.get((pair[0], pair[1], int(m)), analytic.get((pair[1], pair[0], int(m)), 0.0))
                )
            else:
                chi_closed = 2.0 * float(np.sum(np.imag(np.conj(a[:-1]) * np.diff(b))))
            for ion in pair:
                out.append(
                    BranchLoopRecord(
                        gate_id=str(g.gate_id),
                        ion=int(ion),
                        mode=int(m),
                        eta=float(gm.eta[ion][k]),
                        alpha=np.asarray(paths[ion][keep], dtype=complex),
                        chi_m_rad=float(wf.chi_m.get(int(m), 0.0)),
                        chi_closed_form_rad=chi_closed,
                    )
                )
    return tuple(out)


def trace_record(
    tr: core.Traces,
    *,
    sample_index: int,
    sample_id: int,
    branch: int,
    weight: float,
    ion_dims: Sequence[int],
    joint_dims: Sequence[int],
) -> TraceRecord:
    reduced = (
        np.stack([_qobj_array(r) for r in tr.reduced_internal])
        if tr.reduced_internal
        else np.zeros((0, 1, 1), complex)
    )
    return TraceRecord(
        sample_index=int(sample_index),
        sample_id=int(sample_id),
        branch=int(branch),
        weight=float(weight),
        times_s=np.asarray(tr.times_s, dtype=float),
        expectations={str(k): np.asarray(v) for k, v in tr.expectations.items()},
        reduced_internal=reduced,
        ion_dims=tuple(int(d) for d in ion_dims),
        mode_nbar={int(m): np.asarray(np.real(v), dtype=float) for m, v in tr.mode_occupations.items()},
        alpha_m={int(m): np.asarray(v, dtype=complex) for m, v in tr.alpha_m.items()},
        jumps=tuple((float(t), str(c)) for t, c in tr.jumps),
        boundary_population={int(m): float(v) for m, v in tr.boundary_population.items()},
        final_internal=_qobj_array(tr.final.internal),
        final_mode_reduced={int(m): _qobj_array(r) for m, r in tr.final.motional.reduced.items()},
        final_nbar={int(m): float(v) for m, v in tr.final.motional.nbar.items()},
        joint_dims=tuple(int(d) for d in joint_dims),
        mode_marginal=None
        if tr.mode_marginal is None
        else {int(m): np.asarray(v, dtype=float) for m, v in tr.mode_marginal.items()},
    )


def channel_summary_record(s: core.ChannelSummary) -> ChannelSummaryRecord:
    return ChannelSummaryRecord(
        average_gate_infidelity=float(s.average_gate_infidelity),
        depolarizing_rate=float(s.depolarizing_rate),
    )


def gate_local_record(report: core.GateLocalReport) -> GateLocalRecord:
    steps = tuple(
        GateLocalStepRecord(
            gate_id=str(s.gate_id),
            summary=None if s.summary is None else channel_summary_record(s.summary),
            register_after=None if s.register_after is None else np.asarray(s.register_after, dtype=complex),
        )
        for s in report.steps
    )
    return GateLocalRecord(
        steps=steps,
        discrepancy_bound=float(report.discrepancy_bound),
        largest_local_dimension=int(report.largest_local_dimension),
    )


def readout_record(result: core.Result, rec: core.RunRecord) -> ReadoutRecord:
    stage = rec.readout
    disc = stage.discriminator
    outcome = rec.outcome
    return ReadoutRecord(
        window_s=float(disc.window_s),
        threshold=float(disc.n_c) if isinstance(disc, core.ThresholdDiscriminator) else None,
        levels=np.asarray(outcome.levels),
        time_used_s=np.asarray(outcome.time_used_s, dtype=float),
        # the RunRecord's records cover every ion (the Result's only the measured ones), like the levels beside them
        photon_records=None
        if outcome.records is None
        else np.array([[r.total for r in shot] for shot in outcome.records], dtype=int),
        posteriors=None if result.posteriors is None else np.asarray(result.posteriors, dtype=float),
    )


def results_record(result: core.Result, circuit: core.Circuit) -> ResultsRecord:
    return ResultsRecord(
        bitstrings=np.asarray(result.bitstrings, dtype=np.uint8),
        heralds=np.asarray(result.heralds, dtype=np.uint8),
        counts={str(k): int(v) for k, v in result.counts.items()},
        probabilities={str(k): float(v) for k, v in result.probabilities.items()},
        error_bars={str(k): float(v) for k, v in result.error_bars.items()},
        target_probabilities={str(k): float(v) for k, v in core.ideal_probabilities(circuit).items()},
        effective_sample_size=float(result.diagnostics.effective_sample_size),
        discarded_shots=int(result.discarded_shots),
        sample_of_shot=np.asarray(result.sample_of_shot, dtype=np.int64),
        final_state=None if result.final_state is None else _qobj_array(result.final_state),
        register_fidelity=core.register_fidelity(result) if result.final_state is not None else None,
    )


def diagnostics_record(
    diag: core.Diagnostics, options: core.Numerics, wall_time_s: float
) -> DiagnosticsRecord:
    conv = None
    if diag.convergence is not None:
        c = diag.convergence
        conv = ConvergenceRecord(
            tolerances=(float(c.tolerances[0]), float(c.tolerances[1])),
            tightened_tolerances=(float(c.tightened_tolerances[0]), float(c.tightened_tolerances[1])),
            changes={str(k): float(v) for k, v in c.changes.items()},
            tol=float(c.tol),
            converged=bool(c.converged),
            max_change=float(c.max_change),
        )
    return DiagnosticsRecord(
        level=diag.level,
        integrator=str(diag.integrator),
        tolerances=(float(diag.tolerances[0]), float(diag.tolerances[1])),
        samples=int(diag.samples),
        trajectories=int(diag.trajectories),
        branches=int(diag.branches),
        shots_per_sample_realized=tuple(int(m) for m in diag.shots_per_sample_realized),
        boundary_population={int(m): float(v) for m, v in diag.boundary_population.items()},
        boundary_population_max=float(options.boundary_population_max),
        margin_levels={int(m): int(v) for m, v in diag.margin_levels.items()},
        margin_reached={int(m): int(v) for m, v in diag.margin_reached.items()},
        dropped_branch_weight=float(diag.dropped_branch_weight),
        dropped_contribution=(float(diag.dropped_contribution[0]), float(diag.dropped_contribution[1])),
        frozen_contribution={int(m): (float(c[0]), float(c[1])) for m, c in diag.frozen_contribution.items()},
        intrinsic_budget={str(k): float(v) for k, v in diag.intrinsic_budget.items()},
        approximations=tuple(str(a) for a in diag.approximations),
        kernel=str(diag.kernel),
        workers=int(diag.workers),
        wall_time_s=float(wall_time_s),
        convergence=conv,
    )


def build_record(
    job: JobSpec, device: core.Device, table: core.CalibrationTable, result: core.Result
) -> Record:
    """The record of one run (Section 14.3), from the Result and the core's own RunRecord behind it."""
    rec = core.last_record(result)
    space = result.diagnostics.space
    n_branches = len(rec.branches)
    if rec.traces and (n_branches == 0 or len(rec.traces) != len(result.noise_samples) * n_branches):
        raise RecordError(
            f"{len(rec.traces)} traces do not factor as {len(result.noise_samples)} samples x {n_branches} branches"
        )
    traces = tuple(
        trace_record(
            tr,
            sample_index=k // n_branches,
            sample_id=int(result.noise_samples[k // n_branches].sample_id),
            branch=k % n_branches,
            weight=float(rec.branches[k % n_branches].weight),
            ion_dims=space.ion_dims,
            joint_dims=space.dims,
        )
        for k, tr in enumerate(rec.traces)
    )
    tomography = (
        {sid: s.average_gate_infidelity for sid, s in rec.gate_local.summaries.items()}
        if rec.gate_local is not None
        else {}
    )
    return Record(
        job=job,
        device_hash=device.hash(),
        table=table_record(table),
        device_card=device_card(
            device,
            spam=result.spam,
            intrinsic_budget=result.diagnostics.intrinsic_budget,
            tomography=tomography,
        ),
        compiled=compiled_record(rec.compile),
        schedule=schedule_record(rec.schedule),
        space=space_record(space, rec.selection),
        noise_samples=tuple(
            NoiseSampleRecord(
                sample_id=int(s.sample_id),
                t_s=float(s.t_s),
                values={str(k): float(v) for k, v in s.values.items()},
                grids={str(k): np.asarray(v, dtype=float) for k, v in s.ou_grids.items()},
            )
            for s in result.noise_samples
        ),
        branches=tuple(
            BranchRecord(
                k,
                float(b.weight),
                tuple(int(x) for x in b.levels),
                {int(m): int(n) for m, n in b.fock.items()},
            )
            for k, b in enumerate(rec.branches)
        ),
        traces=traces,
        gate_local=None if rec.gate_local is None else gate_local_record(rec.gate_local),
        readout=readout_record(result, rec),
        results=results_record(result, job.circuit.to_core()),
        diagnostics=diagnostics_record(result.diagnostics, job.options, result.duration_s),
        branch_loops=branch_loops(
            rec.schedule, device, {int(m): float(v) for m, v in rec.preparation.nbar.items()}
        ),
    )


# ---- executing a job ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveRun:
    """The in-process objects behind a record that re-simulation needs and the record does not carry (Section 14.3)."""

    device: core.Device
    table: core.CalibrationTable
    result: core.Result
    core_record: core.RunRecord
    machine: core.Machine
    """The machine the run ran on: its numerics and the engine it built (``Machine.engine``)."""

    @property
    def space(self) -> core.HilbertSpace:
        return self.result.diagnostics.space

    @property
    def schedule(self) -> core.Schedule:
        return self.core_record.schedule

    @property
    def steps(self) -> tuple[core.GateStep, ...]:
        return core.gate_steps(self.core_record.schedule)


def calibrate_for(job: JobSpec, device: core.Device) -> core.CalibrationTable:
    """The job's calibration table: the surrogate table of Section 7.5 for the device, with the job's hand-set detunings
    (``JobSpec.waveform_overrides``) applied to the pairs they name."""
    cal = job.calibration
    report = core.calibrate(
        job.machine(device),
        seed=cal.seed,
        pairs=[tuple(p) for p in cal.pairs],
        detection_records=cal.detection_records,
        detection_windows_s=cal.detection_windows_s,
    )
    return table_with_overrides(report.table, job.waveform_overrides)


def table_with_overrides(
    table: core.CalibrationTable, overrides: Mapping[str, float]
) -> core.CalibrationTable:
    """The table with each named pair's waveform shifted (``JobSpec.waveform_overrides``: every blue leg raised by the
    offset, every red leg lowered, the symmetric detuning scan of Section 7.5); a pair the table does not carry is an error,
    never a silent no-op. ``chi_m`` and ``alpha_m`` stay as the table measured them at closure: the scheduler rescales by
    them, so the pulse plays as written with its amplitude unchanged, and the run's branch loops show what the detuned loops
    do (Section 14.4)."""
    if not overrides:
        return table
    shifted: dict[tuple[int, int], core.Waveform] = {}
    for key, offset in overrides.items():
        a, b = (int(x) for x in key.split(","))
        wf = table.waveform_for((a, b))
        if wf is None:
            raise RecordError(f"no entangling waveform for pair {key} in the calibration table to shift")
        shifted[(a, b)] = core.shift_detuning(wf, float(offset))
    return table.with_params(ms=shifted)


def execute(
    job: JobSpec,
    preset: core.DevicePreset | None = None,
    *,
    progress: Callable[[core.Progress], None] | None = None,
) -> tuple[Record, LiveRun]:
    """Calibrate, run and record one job; returns the record and the live handle re-simulation uses. ``progress`` receives
    the core's per-pulse, per-branch, per-sample and per-readout ``Progress``."""
    pre = preset if preset is not None else job.device.build()
    if pre.device.hash() != job.device.hash:
        raise RecordError("the preset given does not match the job's device hash")
    table = calibrate_for(job, pre.device)
    machine = job.machine(pre.device, table)
    result = machine.run(
        job.circuit.to_core(),
        job.shots,
        seed=job.seed,
        keep_final_state=True,
        progress=progress,
    )
    live = LiveRun(pre.device, table, result, core.last_record(result), machine)
    return build_record(job, pre.device, table, result), live
