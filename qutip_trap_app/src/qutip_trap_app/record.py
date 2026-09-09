"""The run record: one data structure, many views (PLAN.md Section 14.3; milestone M11.1).

Section 14.3: "The run record is the single data structure the levels read. Job (device-model hash, seed, package version,
convergence policy) -> compiled circuit -> schedule -> per-sample dynamics traces -> readout records -> results." Every
class here is a frozen dataclass of plain values and ``numpy`` arrays, so the whole record is JSON plus binary arrays
(:mod:`qutip_trap_app.codec`, :mod:`qutip_trap_app.storage`) and nothing on any screen can come from an object the record
does not hold.

Storage policy (Section 14.3, "stored" against "recomputed on demand"):

- **Always stored.** The job as submitted (a re-runnable specification, with the device as a preset reference plus its
  canonical hash), the compiled native circuit and the compiler's verification numbers, the schedule with every pulse's
  tones sampled (constants kept as constants), the played waveforms with their segment tables, the ideal target unitary of
  every gate piece, the gate steps, the joint space and the mode classes, the preparation's occupations, every noise
  sample, every branch of the initial mixture, the per-(sample, branch) dynamics traces at the times the run itself
  stored (the segment boundaries: expectation values, reduced internal states, <n_m>(t), <a_m>(t), the jump list, the
  boundary populations and the final reduced states), the final joint state while the joint dimension is at or below
  ``joint_store_dimension_max``, the readout records (declared bits, the sampled levels behind them, photon counts and
  posteriors where the run produced them), the results with the compiler's target distribution beside them, the
  diagnostics with the convergence report, the device card, the calibration table's entries, and the list of core gaps.
- **Recomputed on demand and then cached in the record.** The joint state at every pulse boundary (by chaining the engine
  over the gate steps from the recorded initial state; ``resim.boundary_states``), the fine-grained traces of a zoomed
  pulse (``resim.zoom``), and the convergence re-checks of a zoomed pulse (``resim.tolerance_recheck``,
  ``resim.truncation_recheck``). A cached zoom is keyed by (step, sample, branch, stored points, options digest), so
  zooming twice recomputes once (Section 9.11).
- **Not available from the core** (Section 14.6: recorded as a gap, shown as unavailable): per-time Fock distributions
  inside a pulse and the per-gate register states of a GATE_LOCAL run; see ``core.CORE_GAPS``.

Conventions carried verbatim from the core: bitstring keys read qubit 0 rightmost (``conv.result_bit_order``); reduced
internal states are in the register order (ion 0 the first tensor factor); every mode index is a position in
``Crystal.modes`` (``conv.mode_index``); frequencies are in Hz at this boundary (``conv.frequencies``).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from qutip_trap_app import __version__ as app_version
from qutip_trap_app import core
from qutip_trap_app.codec import digest, to_document

# ---- the job -----------------------------------------------------------------------------------------------------------------

OptionValue = bool | int | float | str | tuple[str, ...] | None
PresetArg = bool | float | tuple[float, float, float]
FidelityLevelRequest = Literal["JOINT_EXACT", "GATE_LOCAL", "auto"]
FidelityLevelRun = Literal["JOINT_EXACT", "GATE_LOCAL", "CHANNEL_REPLAY"]
"""The two core levels of Section 5.4 plus the app-side channel replay of Section 14.2 row 0 (labelled derived)."""

PRESETS: dict[str, Callable[..., core.DevicePreset]] = {
    "yb171_chain": core.yb171_chain,
    "ca40_optical": core.ca40_optical,
}
"""The device presets a record can name and rebuild (the public presets of ``qutip_trap.api``)."""

NATIVE_GATE_SET: tuple[str, ...] = ("gpi", "gpi2", "ms", "zz", "rz (virtual)")
DEFAULT_DETECTION_WINDOWS_S: tuple[float, ...] = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
"""The detection windows the surrogate calibration scans by default (docs/examples.md)."""
"""Section 7.1: the native gates the device card lists; rz is a frame update, never a pulse."""


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

    def to_core(self) -> core.Circuit:
        return core.Circuit(
            self.n_qubits,
            tuple(core.Operation(op.name, op.qubits, op.params) for op in self.ops),
            self.measure,
        )

    @classmethod
    def from_core(cls, circuit: core.Circuit) -> CircuitRecord:
        return cls(
            circuit.n_qubits,
            tuple(
                OpRecord(op.name, tuple(op.qubits), tuple(float(p) for p in op.params)) for op in circuit.ops
            ),
            tuple(circuit.measure),
        )


@dataclass(frozen=True)
class DeviceRef:
    """Which device the job ran on: a public preset with its arguments (rebuildable) and always the canonical hash."""

    hash: str
    preset: str | None
    n_ions: int
    kwargs: dict[str, PresetArg]

    def build(self) -> core.DevicePreset:
        """Rebuild the preset and check its hash against the recorded one (Section 7.5: a table or record is invalidated,
        never silently reused, when a device parameter changes)."""
        if self.preset is None:
            raise RecordError("the record names no device preset; the device cannot be rebuilt from it")
        if self.preset not in PRESETS:
            raise RecordError(f"unknown device preset {self.preset!r}; known: {sorted(PRESETS)}")
        # the preset functions take keyword arguments of several types; the record's PresetArg union covers them
        preset = PRESETS[self.preset](self.n_ions, **cast(dict[str, Any], self.kwargs))
        if preset.device.hash() != self.hash:
            raise RecordError(
                f"the rebuilt device's hash {preset.device.hash()[:12]} differs from the recorded {self.hash[:12]}: "
                "the preset or the core changed since the record was made"
            )
        return preset


@dataclass(frozen=True)
class DriveRef:
    kind: str
    beams: tuple[int, ...]

    def to_core(self) -> core.GateDrive:
        # GateDrive.kind is a Literal of the drive kinds; the record stores the same strings
        return core.GateDrive(cast(Any, self.kind), self.beams)

    @classmethod
    def from_core(cls, drive: core.GateDrive) -> DriveRef:
        if drive.light_shift is not None:
            raise RecordError(
                "light-shift gate drives are not recorded in M11.1 (their couplings are not JSON values)"
            )
        return cls(str(drive.kind), tuple(drive.beams))


@dataclass(frozen=True)
class CalibrationRef:
    """How the calibration table was built, so that an imported record can rebuild it (Section 7.5's cache key)."""

    seed: int
    surrogate: bool
    pairs: tuple[tuple[int, int], ...]
    detection_records: int
    detection_windows_s: tuple[float, ...]


@dataclass(frozen=True)
class JobSpec:
    """What the user asked for, re-runnable: the Section 14.3 "Job" with the device model hash, seed, package version
    and convergence policy (``options``)."""

    circuit: CircuitRecord
    shots: int
    seed: int
    device: DeviceRef
    gate_drives: dict[int, DriveRef]
    entangling_drives: dict[int, DriveRef]
    calibration: CalibrationRef
    options: dict[str, OptionValue]
    """``SolverOptions`` as a mapping: the run's convergence policy."""
    level: FidelityLevelRequest = "auto"
    samples: int | None = None
    readout: Literal["fast", "full"] = "fast"
    noise: bool = True
    keep_final_state: bool = True
    entangler: Literal["ms", "zz"] = "ms"
    t0_s: float = 0.0
    qubit_to_ion: tuple[int, ...] = ()
    """Physical ion per circuit qubit; empty = identity."""
    internal_levels: int = 2
    stark_compensation: bool = True
    caps: dict[int, int] | None = None
    """Explicit Fock caps per mode (``run(caps=...)``); None lets the Section 5.5 cap rule decide."""

    def solver_options(self) -> core.SolverOptions:
        # the mapping was produced by dataclasses.asdict on a SolverOptions and carries exactly its fields
        return core.SolverOptions(**cast(dict[str, Any], self.options))

    def run_kwargs(self) -> dict[str, Any]:
        return {
            "gate_drives": {i: d.to_core() for i, d in self.gate_drives.items()},
            "entangling_drives": {i: d.to_core() for i, d in self.entangling_drives.items()},
        }


def options_record(options: core.SolverOptions) -> dict[str, OptionValue]:
    """``SolverOptions`` as the JSON mapping the job stores."""
    out: dict[str, OptionValue] = {}
    for f in dataclasses.fields(options):
        value = getattr(options, f.name)
        if isinstance(value, tuple):
            out[f.name] = tuple(str(v) for v in value)
        else:
            out[f.name] = value
    return out


def job_for_preset(
    preset_name: str,
    n_ions: int,
    circuit: core.Circuit,
    shots: int,
    *,
    seed: int = 0,
    options: core.SolverOptions | None = None,
    pairs: Sequence[tuple[int, int]] | None = None,
    detection_records: int = 2000,
    detection_windows_s: Sequence[float] = DEFAULT_DETECTION_WINDOWS_S,
    preset_kwargs: Mapping[str, PresetArg] | None = None,
    **run_fields: Any,
) -> tuple[JobSpec, core.DevicePreset]:
    """A JobSpec on a public preset, with the preset built so the caller can go on to :func:`execute`."""
    if preset_name not in PRESETS:
        raise RecordError(f"unknown device preset {preset_name!r}; known: {sorted(PRESETS)}")
    kwargs = dict(preset_kwargs or {})
    preset = PRESETS[preset_name](n_ions, **cast(dict[str, Any], kwargs))
    opts = options or core.SolverOptions()
    pairs_t = tuple((int(a), int(b)) for a, b in (pairs if pairs is not None else circuit.entangling_pairs()))
    job = JobSpec(
        circuit=CircuitRecord.from_core(circuit),
        shots=int(shots),
        seed=int(seed),
        device=DeviceRef(preset.device.hash(), preset_name, int(n_ions), kwargs),
        gate_drives={i: DriveRef.from_core(d) for i, d in preset.gate_drives.items()},
        entangling_drives={i: DriveRef.from_core(d) for i, d in preset.entangling_drives.items()},
        calibration=CalibrationRef(
            seed=int(seed),
            surrogate=True,
            pairs=pairs_t,
            detection_records=int(detection_records),
            detection_windows_s=tuple(float(w) for w in detection_windows_s),
        ),
        options=options_record(opts),
        **run_fields,
    )
    return job, preset


# ---- the device card (Level 0) and the calibration table -----------------------------------------------------------------------


@dataclass(frozen=True)
class ModeRecord:
    index: int
    family: str
    family_index: int
    omega_hz: float
    e_hat: tuple[float, float, float]
    eigenvector: np.ndarray
    """Mass-weighted c_{i,m}, unit norm, last component positive (conv.mode_eigenvector_gauge)."""


@dataclass(frozen=True)
class BeamRecord:
    index: int
    wavelength_m: float
    k_hat: tuple[float, float, float]
    polarization: tuple[complex, complex, complex]
    waist_m: float
    power_w: float
    pointing_m: tuple[float, float, float]


@dataclass(frozen=True)
class DetectorRecord:
    kind: str
    efficiency: float
    background_cps: float
    window_s: float
    psf_leakage: dict[int, float]
    numerical_aperture: float | None


@dataclass(frozen=True)
class DerivedRecord:
    """``Device.derived()``: every computed number with its ledger id (Section 3.3)."""

    values: dict[str, float]
    provenance: dict[str, str]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CalEntryRecord:
    key: str
    value: float
    uncertainty: float
    status: str
    experiment: str
    provenance_id: str
    fitted_at_s: float
    sample_id: int


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
    phase_rad: dict[str, float]
    detuning_hz: dict[str, SampledFn]
    """``leg`` -> detuning from the carrier."""


@dataclass(frozen=True)
class WaveformRecord:
    pair: tuple[int, int]
    kind: str
    duration_s: float
    chi_m: dict[int, float]
    """Signed per-mode entangling angle at closure (conv.entangling_angle, conv.entangling_sign)."""
    alpha_m: dict[int, complex]
    """Per-mode residual displacement at closure (Section 4.4.3)."""
    chi_total_rad: float
    phi_s: CalEntryRecord
    phi_m: CalEntryRecord
    segments: tuple[SegmentRecord, ...] | None
    fourier: tuple[complex, ...] | None


@dataclass(frozen=True)
class TableRecord:
    device_hash: str
    seed: int
    surrogate: bool
    fitted_at_s: float
    entries: dict[str, CalEntryRecord]
    waveforms: dict[str, WaveformRecord]
    """``"a,b"`` -> the pair's entangling waveform."""
    uncalibrated: tuple[str, ...]


@dataclass(frozen=True)
class DeviceCard:
    """Level 0's device card (Section 14.2): species, ion count, native gate set, calibrated and estimated gate errors, SPAM."""

    species: tuple[str, ...]
    n_ions: int
    positions_m: np.ndarray
    modes: tuple[ModeRecord, ...]
    trap_omega_hz: tuple[float, float, float] | None
    trap_axis_angle_rad: float
    field_gauss: float
    field_direction: tuple[float, float, float]
    beams: tuple[BeamRecord, ...]
    detector: DetectorRecord
    hardware: tuple[str, ...]
    """``HardwareChain.describe()``."""
    noise_quiet: bool
    heating_quanta_per_s: dict[int, float]
    motional_dephasing_tau_s: dict[int, float]
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
    n_pulses: int
    n_entangling: int
    block_residuals: tuple[float, ...]
    circuit_residual: float | None
    entangler: str
    notes: tuple[str, ...]
    target_unitary: np.ndarray | None
    """U of the source circuit in the compiler's order (qubit 0 the least-significant index bit); None above 6 qubits."""


@dataclass(frozen=True)
class ToneRecord:
    detuning_hz: SampledFn
    phase_rad: SampledFn
    envelope_hz: SampledFn
    theta_bessel_rad: float | None


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
    rf_locked: bool
    programmed: bool

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
    kind: str
    pair: tuple[int, int]
    beams: tuple[int, ...]
    t_start_s: float
    t_end_s: float
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
    ions: tuple[int, ...]


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
    ion_dims: tuple[int, ...]
    ions: tuple[int, ...]
    resolved: tuple[TruncationRecord, ...]
    frozen: tuple[int, ...]
    dropped: tuple[int, ...]
    enr_group: tuple[tuple[int, ...], int] | None
    dims: tuple[int, ...]
    dimension: int
    mode_class: dict[int, str]
    contribution: dict[int, tuple[float, float]]
    """Per mode touched by an entangling gate: (max |alpha|^2 (2 nbar + 1), max |chi|)."""
    nbar: dict[int, float]
    notes: tuple[str, ...]
    budget: tuple[bool, int, int]


@dataclass(frozen=True)
class PreparationRecord:
    nbar: dict[int, float]
    duration_s: float
    preparation_error: dict[int, float]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]


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
class TraceRecord:
    """One dynamics trace: one (sample, branch) evolution of the run, at the times the run stored (Section 14.3)."""

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
    final_joint: np.ndarray | None
    """The joint ket (D,) or density matrix (D, D) when D <= the store cap; None otherwise."""
    joint_dims: tuple[int, ...]

    def index_at(self, t_s: float, tolerance_s: float = 1e-12) -> int:
        """The LAST stored index at time ``t_s`` (a segment end)."""
        hits = np.flatnonzero(np.abs(self.times_s - t_s) <= tolerance_s)
        if hits.size == 0:
            raise KeyError(f"no stored point at t = {t_s:.9g} s")
        return int(hits[-1])


@dataclass(frozen=True)
class ChannelSummaryRecord:
    choi: np.ndarray
    cp_tp_residual: tuple[float, float]
    n_traj: int
    average_gate_infidelity: float
    pauli_twirled: dict[str, float]
    depolarizing_rate: float


@dataclass(frozen=True)
class GateLocalStepRecord:
    gate_id: str
    kind: str
    t_start_s: float
    t_end_s: float
    ions: tuple[int, ...]
    space_dims: tuple[int, ...]
    resolved: tuple[int, ...]
    n_inputs: int
    n_traj: int
    method: str
    residual_displacement: dict[int, float]
    residual_bound: float
    nbar_after: dict[int, float]
    boundary_population: dict[int, float]
    summary: ChannelSummaryRecord | None


@dataclass(frozen=True)
class GateLocalRecord:
    steps: tuple[GateLocalStepRecord, ...]
    residual_bound_total: float
    frozen_excitation_total: float
    dropped_crosstalk_total: float
    discrepancy_bound: float
    register: str
    ensemble_size: int
    largest_local_dimension: int
    engine_runs: int
    cache_hits: int
    notes: tuple[str, ...]


# ---- readout, results, diagnostics ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IonRatesRecord:
    ion: int
    R_bright_per_s: float
    R_dark_pumping_per_s: float
    R_bright_pumping_per_s: float
    detected_bright_per_s: float
    background_per_s: float


@dataclass(frozen=True)
class ReadoutRecord:
    mode: str
    window_s: float
    discriminator: str
    threshold: float | None
    levels: np.ndarray
    """(shots, n_ions) the projectively sampled internal level per ion (the joint outcome behind the declared bits)."""
    bits_declared: np.ndarray
    time_used_s: np.ndarray
    photon_records: np.ndarray | None
    posteriors: np.ndarray | None
    sub_bin_records: np.ndarray | None
    arrival_offsets: np.ndarray | None
    """Ragged arrival times flattened: offsets (shots * n_ions + 1,) into ``arrival_times_s``."""
    arrival_times_s: np.ndarray | None
    rates: tuple[IonRatesRecord, ...]
    leakage: dict[int, float]
    crosstalk_discrepancy: float | None


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
    spam: dict[str, tuple[float, float]]
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
    root_seed: int
    boundary_population: dict[int, float]
    boundary_population_max: float
    """The policy threshold the run was judged against (``SolverOptions.boundary_population_max``)."""
    margin_levels: dict[int, int]
    margin_reached: dict[int, int]
    populated_n_max: dict[int, int]
    cap_growth: dict[int, int]
    dropped_branch_weight: float
    frozen_excitation_bound: dict[int, float]
    dropped_contribution: tuple[float, float]
    frozen_contribution: dict[int, tuple[float, float]]
    intrinsic_budget: dict[str, float]
    approximations: tuple[str, ...]
    kernel: str
    workers: int
    propagator_cache_hits: int
    wall_clock_span_s: float
    wall_time_s: float
    """Measured by the application around ``run()`` (the core reports none)."""
    convergence: ConvergenceRecord | None
    run_state_order: tuple[int, ...]
    run_state_events: tuple[tuple[int, str], ...]


# ---- the channel replay (Section 14.2 row 0; app-side, derived) ---------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelPieceRecord:
    gate_id: str
    ions: tuple[int, ...]
    choi: np.ndarray
    ideal: np.ndarray
    summary: ChannelSummaryRecord
    residual_bound: float
    frozen_excitation: float
    dropped_crosstalk: float
    cp_residual: float
    tp_residual: float
    residual_displacement: dict[int, float]
    nbar_after: dict[int, float]
    local_dimension: int
    engine_runs: int


@dataclass(frozen=True)
class ChannelEntryRecord:
    key: str
    kind: str
    addressed: tuple[int, ...]
    angle: float | None
    pieces: tuple[ChannelPieceRecord, ...]
    covariance_residual: float | None
    wall_time_s: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ReplayGate:
    gate_id: str
    key: str
    ions: tuple[int, ...]
    phases: dict[int, float]
    residual: float
    average_gate_infidelity: float
    depolarizing_rate: float


@dataclass(frozen=True)
class ReplayRecord:
    """What the channel replay did: which channel each gate was played as, the register after each gate, and the
    channel-derivation residual the replay reports (Section 9.11 row "Channel derivation")."""

    engine: str
    gates: tuple[ReplayGate, ...]
    register_after: np.ndarray
    """(n_gates, 2^n, 2^n): the register after each gate piece, register order."""
    channels: dict[str, ChannelEntryRecord]
    residual_terms: dict[str, float]
    residual_total: float
    bright_levels: tuple[int, ...]
    spam_used: dict[str, tuple[float, float]]
    notes: tuple[str, ...]


# ---- the zoom cache -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryState:
    """The joint state at a gate-step boundary for one (sample, branch), recomputed by chaining and cached (Section 14.3)."""

    step_index: int
    """The state is at the START of this step (step ``len(steps)`` = the end of the schedule)."""
    sample_index: int
    branch: int
    t_s: float
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


# ---- the record -----------------------------------------------------------------------------------------------------------------

RECORD_FORMAT = "qutip-trap-app/record/1"


@dataclass(frozen=True)
class Record:
    """The run record of Section 14.3: everything the five levels read, and nothing else."""

    format: str
    created_utc: str
    versions: dict[str, str]
    job: JobSpec
    device_hash: str
    table: TableRecord
    device_card: DeviceCard
    compiled: CompiledRecord
    schedule: ScheduleRecord
    space: SpaceRecord
    preparation: PreparationRecord
    noise_samples: tuple[NoiseSampleRecord, ...]
    branches: tuple[BranchRecord, ...]
    traces: tuple[TraceRecord, ...]
    gate_local: GateLocalRecord | None
    readout: ReadoutRecord
    results: ResultsRecord
    diagnostics: DiagnosticsRecord
    notes: tuple[str, ...]
    core_gaps: tuple[str, ...]
    joint_store_dimension_max: int
    boundaries: tuple[BoundaryState, ...] = ()
    zooms: tuple[ZoomTrace, ...] = ()
    replay: ReplayRecord | None = None
    """Present when the record was made by the app-side channel replay (Level 0's default engine)."""

    # -- identity --

    def digest(self) -> str:
        """The bitwise identity of the record's data (the ``record_id`` of an export)."""
        doc, arrays = to_document(self)
        return digest(doc, arrays)

    def key(self) -> str:
        """The identity of the RUN behind the record: the digest with the caches, the clock and the wall times left out, so
        that a record and the same record with more zooms cached share one key (what the worker keys its live state by)."""
        bare = dataclasses.replace(
            self,
            created_utc="",
            versions={},
            boundaries=(),
            zooms=(),
            diagnostics=dataclasses.replace(self.diagnostics, wall_time_s=0.0),
        )
        doc, arrays = to_document(bare)
        return digest(doc, arrays)[:24]

    # -- lookups the view-models share --

    @property
    def n_qubits(self) -> int:
        return self.job.circuit.n_qubits

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
        for b in self.boundaries:
            if (b.step_index, b.sample_index, b.branch) == (step_index, sample_index, branch):
                return b
        return None

    def zoom(self, key: str) -> ZoomTrace | None:
        for z in self.zooms:
            if z.key == key:
                return z
        return None

    def with_boundaries(self, new: Sequence[BoundaryState]) -> Record:
        keep = [b for b in self.boundaries if not any(_same_boundary(b, n) for n in new)]
        ordered = sorted(keep + list(new), key=lambda b: (b.sample_index, b.branch, b.step_index))
        return dataclasses.replace(self, boundaries=tuple(ordered))

    def with_zoom(self, zoom: ZoomTrace) -> Record:
        keep = tuple(z for z in self.zooms if z.key != zoom.key)
        return dataclasses.replace(self, zooms=keep + (zoom,))


def _same_boundary(a: BoundaryState, b: BoundaryState) -> bool:
    return (a.step_index, a.sample_index, a.branch) == (b.step_index, b.sample_index, b.branch)


# ---- building a record from a core Result ----------------------------------------------------------------------------------------

TONE_SAMPLES = 257
"""Grid points at which a callable tone function is sampled for the record (a display resolution, not a solver one)."""


def sampled_fn(value: object, duration_s: float, n: int = TONE_SAMPLES) -> SampledFn:
    """A tone function (constant, callable of the time since the pulse start, or a uniformly sampled array) for the record."""
    if isinstance(value, np.ndarray):
        return SampledFn("array", None, np.asarray(value, dtype=float))
    if callable(value):
        grid = np.linspace(0.0, duration_s, n)
        return SampledFn("sampled", None, np.array([float(value(float(t))) for t in grid], dtype=float))
    return SampledFn("constant", float(cast(float, value)), None)


def _cal_entry(key: str, entry: core.CalEntry) -> CalEntryRecord:
    return CalEntryRecord(
        key=key,
        value=float(entry.value),
        uncertainty=float(entry.uncertainty),
        status=str(entry.status),
        experiment=str(entry.experiment),
        provenance_id=str(entry.provenance_id),
        fitted_at_s=float(entry.fitted_at_s),
        sample_id=int(entry.sample_id),
    )


def waveform_record(pair: tuple[int, int], wf: core.Waveform) -> WaveformRecord:
    segments: tuple[SegmentRecord, ...] | None = None
    if wf.segments is not None:
        segs = []
        for s in wf.segments:
            segs.append(
                SegmentRecord(
                    duration_s=float(s.duration_s),
                    amplitude_hz={
                        f"{ion},{leg}": sampled_fn(v, s.duration_s)
                        for (ion, leg), v in s.amplitude_hz.items()
                    },
                    phase_rad={f"{ion},{leg}": float(v) for (ion, leg), v in s.phase_rad.items()},
                    detuning_hz={str(leg): sampled_fn(v, s.duration_s) for leg, v in s.detuning_hz.items()},
                )
            )
        segments = tuple(segs)
    return WaveformRecord(
        pair=(int(pair[0]), int(pair[1])),
        kind=str(wf.kind),
        duration_s=float(wf.duration_s),
        chi_m={int(m): float(v) for m, v in wf.chi_m.items()},
        alpha_m={int(m): complex(v) for m, v in wf.alpha_m.items()},
        chi_total_rad=float(wf.chi_total_rad),
        phi_s=_cal_entry("phi_s", wf.phi_s),
        phi_m=_cal_entry("phi_m", wf.phi_m),
        segments=segments,
        fourier=None if wf.fourier is None else tuple(complex(c) for c in wf.fourier),
    )


def table_record(table: core.CalibrationTable) -> TableRecord:
    entries = {k: _cal_entry(k, e) for k, e in table.entries().items()}
    waveforms = {f"{a},{b}": waveform_record((a, b), wf) for (a, b), wf in table.ms.items()}
    return TableRecord(
        device_hash=str(table.device_hash),
        seed=int(table.seed),
        surrogate=bool(table.surrogate),
        fitted_at_s=float(table.fitted_at_s),
        entries=entries,
        waveforms=waveforms,
        uncalibrated=tuple(table.uncalibrated()),
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
            e_hat=(float(m.e_hat[0]), float(m.e_hat[1]), float(m.e_hat[2])),
            eigenvector=np.asarray(m.eigenvector, dtype=float),
        )
        for k, m in enumerate(crystal.modes)
    )
    beams = tuple(
        BeamRecord(
            index=k,
            wavelength_m=float(b.wavelength_m),
            k_hat=(float(b.k_hat[0]), float(b.k_hat[1]), float(b.k_hat[2])),
            polarization=(complex(b.polarization[0]), complex(b.polarization[1]), complex(b.polarization[2])),
            waist_m=float(b.waist_m),
            power_w=float(b.power_w),
            pointing_m=(float(b.pointing_m[0]), float(b.pointing_m[1]), float(b.pointing_m[2])),
        )
        for k, b in enumerate(device.beams)
    )
    det = device.detector
    detector = DetectorRecord(
        kind=str(det.kind),
        efficiency=float(det.efficiency),
        background_cps=float(det.background_cps),
        window_s=float(det.window_s),
        psf_leakage={int(k): float(v) for k, v in det.psf_leakage.items()},
        numerical_aperture=None if det.numerical_aperture is None else float(det.numerical_aperture),
    )
    estimates: dict[str, float] = {}
    for key, value in intrinsic_budget.items():
        if key == "total" or "." not in key:
            continue
        gate = key.split(".")[0]
        estimates[gate] = estimates.get(gate, 0.0) + float(value)
    trap = device.trap
    omega = None if trap.omega_hz is None else tuple(float(x) for x in trap.omega_hz)
    return DeviceCard(
        species=tuple(s.name for s in crystal.species),
        n_ions=int(crystal.n_ions),
        positions_m=np.asarray(crystal.positions_m, dtype=float),
        modes=modes,
        trap_omega_hz=None if omega is None else (omega[0], omega[1], omega[2]),
        trap_axis_angle_rad=float(trap.axis_angle_rad),
        field_gauss=float(device.field.B_gauss),
        field_direction=(
            float(device.field.direction[0]),
            float(device.field.direction[1]),
            float(device.field.direction[2]),
        ),
        beams=beams,
        detector=detector,
        hardware=tuple(device.hardware.describe()),
        noise_quiet=bool(device.noise.is_quiet(device)),
        heating_quanta_per_s={
            int(m): float(v) for m, v in device.noise.heating_rates_quanta_per_s(device).items()
        },
        motional_dephasing_tau_s={
            int(m): float(v) for m, v in device.noise.motional_dephasing_tau_s(device).items()
        },
        derived=DerivedRecord(
            values={k: float(v) for k, v in derived.values.items()},
            provenance={k: str(v) for k, v in derived.provenance.items()},
            notes=tuple(derived.notes),
        ),
        native_gates=NATIVE_GATE_SET,
        spam={str(k): (float(v[0]), float(v[1])) for k, v in spam.items()},
        gate_error_estimates=estimates,
        gate_error_tomography={str(k): float(v) for k, v in tomography.items() if not math.isnan(v)},
    )


def schedule_record(sched: core.Schedule) -> ScheduleRecord:
    index_of = {id(p): k for k, p in enumerate(sched.pulses)}
    pulses = []
    for k, p in enumerate(sched.pulses):
        d = p.drive
        pulses.append(
            PulseRecord(
                index=k,
                gate_id=p.gate_id,
                ions=tuple(int(i) for i in d.ions),
                kind=str(d.kind),
                beams=tuple(int(b) for b in d.beams),
                t_start_s=float(p.t_start_s),
                t_end_s=float(p.t_end_s),
                tones=tuple(
                    ToneRecord(
                        detuning_hz=sampled_fn(t.detuning_hz, p.duration_s),
                        phase_rad=sampled_fn(t.phase_rad, p.duration_s),
                        envelope_hz=sampled_fn(t.envelope_hz, p.duration_s),
                        theta_bessel_rad=None if t.theta_bessel_rad is None else float(t.theta_bessel_rad),
                    )
                    for t in d.tones
                ),
                stark_shift_hz=sampled_fn(d.stark_shift_hz, p.duration_s),
                crosstalk={int(j): complex(e) for j, e in d.crosstalk.items()},
                closes_modes=tuple(int(m) for m in p.closes_modes),
                rf_locked=bool(d.rf_locked),
                programmed=bool(d.programmed),
            )
        )
    gates = tuple(
        PlayedGateRecord(
            gate_id=str(g.gate_id),
            kind=str(g.kind),
            pair=(int(g.pair[0]), int(g.pair[1])),
            beams=tuple(int(b) for b in g.beams),
            t_start_s=float(g.t_start_s),
            t_end_s=float(g.t_end_s),
            waveform=waveform_record((int(g.pair[0]), int(g.pair[1])), g.waveform),
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
            ions=tuple(int(i) for i in s.ions),
        )
        for k, s in enumerate(core.gate_steps(sched))
    )
    starts = [p.t_start_s for p in sched.pulses] + [a for a, _ in sched.idle]
    t0 = float(sched.t0_s) if sched.t0_s is not None else min(starts + [0.0])
    return ScheduleRecord(
        pulses=tuple(pulses),
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


def space_record(space: core.HilbertSpace, selection: Any) -> SpaceRecord:
    return SpaceRecord(
        ion_dims=tuple(int(d) for d in space.ion_dims),
        ions=tuple(int(i) for i in space.ion_labels),
        resolved=tuple(
            TruncationRecord(
                int(m.mode),
                int(m.d),
                (int(m.expected_n_range[0]), int(m.expected_n_range[1])),
                float(m.eta_max),
            )
            for m in space.resolved
        ),
        frozen=tuple(int(m) for m in space.frozen),
        dropped=tuple(int(m) for m in space.dropped),
        enr_group=None
        if space.enr_group is None
        else (tuple(int(m) for m in space.enr_group[0]), int(space.enr_group[1])),
        dims=tuple(int(d) for d in space.dims),
        dimension=int(space.dimension),
        mode_class={int(m): str(c) for m, c in selection.mode_class.items()},
        contribution={int(m): (float(c[0]), float(c[1])) for m, c in selection.contribution.items()},
        nbar={int(m): float(v) for m, v in selection.nbar.items()},
        notes=tuple(selection.notes),
        budget=(bool(selection.budget[0]), int(selection.budget[1]), int(selection.budget[2])),
    )


def _qobj_array(q: Any) -> np.ndarray:
    return np.asarray(q.full(), dtype=complex)


def trace_record(
    tr: core.Traces,
    *,
    sample_index: int,
    sample_id: int,
    branch: int,
    weight: float,
    ion_dims: Sequence[int],
    joint_dims: Sequence[int],
    joint_cap: int,
) -> TraceRecord:
    reduced = (
        np.stack([_qobj_array(r) for r in tr.reduced_internal])
        if tr.reduced_internal
        else np.zeros((0, 1, 1), complex)
    )
    joint: np.ndarray | None = None
    if tr.final.joint is not None and tr.final.joint.shape[0] <= joint_cap:
        arr = _qobj_array(tr.final.joint)
        joint = arr.reshape(-1) if tr.final.joint.isket else arr
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
        final_joint=joint,
        joint_dims=tuple(int(d) for d in joint_dims),
    )


def channel_summary_record(s: Any) -> ChannelSummaryRecord:
    return ChannelSummaryRecord(
        choi=np.asarray(s.choi, dtype=complex),
        cp_tp_residual=(float(s.cp_tp_residual[0]), float(s.cp_tp_residual[1])),
        n_traj=int(s.n_traj),
        average_gate_infidelity=float(s.average_gate_infidelity),
        pauli_twirled={str(k): float(v) for k, v in s.pauli_twirled.items()},
        depolarizing_rate=float(s.depolarizing_rate),
    )


def gate_local_record(report: Any) -> GateLocalRecord:
    steps = tuple(
        GateLocalStepRecord(
            gate_id=str(s.gate_id),
            kind=str(s.kind),
            t_start_s=float(s.t_start_s),
            t_end_s=float(s.t_end_s),
            ions=tuple(int(i) for i in s.ions),
            space_dims=tuple(int(d) for d in s.space_dims),
            resolved=tuple(int(m) for m in s.resolved),
            n_inputs=int(s.n_inputs),
            n_traj=int(s.n_traj),
            method=str(s.method),
            residual_displacement={int(m): float(v) for m, v in s.residual_displacement.items()},
            residual_bound=float(s.residual_bound),
            nbar_after={int(m): float(v) for m, v in s.nbar_after.items()},
            boundary_population={int(m): float(v) for m, v in s.boundary_population.items()},
            summary=None if s.summary is None else channel_summary_record(s.summary),
        )
        for s in report.steps
    )
    return GateLocalRecord(
        steps=steps,
        residual_bound_total=float(report.residual_bound_total),
        frozen_excitation_total=float(report.frozen_excitation_total),
        dropped_crosstalk_total=float(report.dropped_crosstalk_total),
        discrepancy_bound=float(report.discrepancy_bound),
        register=str(report.register),
        ensemble_size=int(report.ensemble_size),
        largest_local_dimension=int(report.largest_local_dimension),
        engine_runs=int(report.engine_runs),
        cache_hits=int(report.cache_hits),
        notes=tuple(report.notes),
    )


def readout_record(result: core.Result, rec: core.RunRecord) -> ReadoutRecord:
    stage = rec.readout
    outcome = rec.outcome
    disc = stage.discriminator
    threshold = getattr(disc, "n_c", None)
    offsets: np.ndarray | None = None
    arrivals: np.ndarray | None = None
    if result.arrival_times_s is not None:
        flat: list[np.ndarray] = []
        offs = [0]
        for shot in result.arrival_times_s:
            for per_ion in shot:
                arr = np.asarray(per_ion, dtype=float)
                flat.append(arr)
                offs.append(offs[-1] + arr.size)
        offsets = np.asarray(offs, dtype=np.int64)
        arrivals = np.concatenate(flat) if flat else np.zeros(0, dtype=float)
    rates = []
    for i, (fr, model) in enumerate(zip(stage.rates, stage.models)):
        rates.append(
            IonRatesRecord(
                ion=i,
                R_bright_per_s=float(fr.R_bright_per_s),
                R_dark_pumping_per_s=float(fr.R_dark_pumping_per_s),
                R_bright_pumping_per_s=float(fr.R_bright_pumping_per_s),
                detected_bright_per_s=float(model.detected_bright_per_s),
                background_per_s=float(model.background_per_s),
            )
        )
    return ReadoutRecord(
        mode=str(outcome.mode),
        window_s=float(disc.window_s),
        discriminator=type(disc).__name__,
        threshold=None if threshold is None else float(threshold),
        levels=np.asarray(outcome.levels),
        bits_declared=np.asarray(outcome.bits),
        time_used_s=np.asarray(outcome.time_used_s, dtype=float),
        photon_records=None if result.photon_records is None else np.asarray(result.photon_records),
        posteriors=None if result.posteriors is None else np.asarray(result.posteriors, dtype=float),
        sub_bin_records=None if result.sub_bin_records is None else np.asarray(result.sub_bin_records),
        arrival_offsets=offsets,
        arrival_times_s=arrivals,
        rates=tuple(rates),
        leakage={int(k): float(v) for k, v in stage.leakage.items()},
        crosstalk_discrepancy=None
        if stage.crosstalk_discrepancy is None
        else float(stage.crosstalk_discrepancy),
    )


def results_record(result: core.Result, circuit: core.Circuit, register_fid: float | None) -> ResultsRecord:
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
        spam={str(k): (float(v[0]), float(v[1])) for k, v in result.spam.items()},
        final_state=None if result.final_state is None else _qobj_array(result.final_state),
        register_fidelity=register_fid,
    )


def diagnostics_record(
    diag: core.Diagnostics, options: core.SolverOptions, wall_time_s: float
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
        root_seed=int(diag.root_seed),
        boundary_population={int(m): float(v) for m, v in diag.boundary_population.items()},
        boundary_population_max=float(options.boundary_population_max),
        margin_levels={int(m): int(v) for m, v in diag.margin_levels.items()},
        margin_reached={int(m): int(v) for m, v in diag.margin_reached.items()},
        populated_n_max={int(m): int(v) for m, v in diag.populated_n_max.items()},
        cap_growth={int(m): int(v) for m, v in diag.cap_growth.items()},
        dropped_branch_weight=float(diag.dropped_branch_weight),
        frozen_excitation_bound={int(m): float(v) for m, v in diag.frozen_excitation_bound.items()},
        dropped_contribution=(float(diag.dropped_contribution[0]), float(diag.dropped_contribution[1])),
        frozen_contribution={int(m): (float(c[0]), float(c[1])) for m, c in diag.frozen_contribution.items()},
        intrinsic_budget={str(k): float(v) for k, v in diag.intrinsic_budget.items()},
        approximations=tuple(str(a) for a in diag.approximations),
        kernel=str(diag.kernel),
        workers=int(diag.workers),
        propagator_cache_hits=int(diag.propagator_cache_hits),
        wall_clock_span_s=float(diag.wall_clock_span_s),
        wall_time_s=float(wall_time_s),
        convergence=conv,
        run_state_order=tuple(int(i) for i in diag.run_state.order),
        run_state_events=tuple((int(s), str(e)) for s, e in diag.run_state.events),
    )


def build_record(
    job: JobSpec,
    device: core.Device,
    table: core.CalibrationTable,
    result: core.Result,
    *,
    wall_time_s: float,
    joint_store_dimension_max: int = 4096,
) -> Record:
    """The record of one ``run()`` (Section 14.3), from the Result and the core's own RunRecord behind it."""
    rec = core.last_record(result)
    circuit = job.circuit.to_core()
    opts = job.solver_options()
    space = result.diagnostics.space
    n_branches = len(rec.branches)
    traces: list[TraceRecord] = []
    if rec.traces:
        if n_branches == 0 or len(rec.traces) != len(result.noise_samples) * n_branches:
            raise RecordError(
                f"{len(rec.traces)} traces do not factor as {len(result.noise_samples)} samples x {n_branches} branches"
            )
        for k, tr in enumerate(rec.traces):
            s_idx, b_idx = divmod(k, n_branches)
            traces.append(
                trace_record(
                    tr,
                    sample_index=s_idx,
                    sample_id=int(result.noise_samples[s_idx].sample_id),
                    branch=b_idx,
                    weight=float(rec.branches[b_idx].weight),
                    ion_dims=space.ion_dims,
                    joint_dims=space.dims,
                    joint_cap=joint_store_dimension_max,
                )
            )
    register_fid = core.register_fidelity(result) if result.final_state is not None else None
    versions = {
        "qutip_trap": str(core.core_version),
        "qutip_trap_app": str(app_version),
        "python": platform.python_version(),
    }
    try:
        import qutip
        import scipy

        versions["qutip"] = str(qutip.__version__)
        versions["numpy"] = str(np.__version__)
        versions["scipy"] = str(scipy.__version__)
    except ImportError:  # pragma: no cover - the physics stack is a hard dependency
        pass
    return Record(
        format=RECORD_FORMAT,
        created_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        versions=versions,
        job=job,
        device_hash=device.hash(),
        table=table_record(table),
        device_card=device_card(
            device,
            spam=result.spam,
            intrinsic_budget=result.diagnostics.intrinsic_budget,
            tomography=(
                {sid: summ.average_gate_infidelity for sid, summ in rec.gate_local.summaries.items()}
                if rec.gate_local is not None
                else {}
            ),
        ),
        compiled=CompiledRecord(
            native=CircuitRecord.from_core(rec.compile.circuit),
            final_frame_rad={int(q): float(v) for q, v in rec.compile.final_frame_rad.items()},
            n_pulses=int(rec.compile.n_pulses),
            n_entangling=int(rec.compile.n_entangling),
            block_residuals=tuple(float(r) for r in rec.compile.block_residuals),
            circuit_residual=None
            if rec.compile.circuit_residual is None
            else float(rec.compile.circuit_residual),
            entangler=str(rec.compile.entangler),
            notes=tuple(rec.compile.notes),
            target_unitary=np.asarray(core.circuit_unitary(circuit), dtype=complex)
            if circuit.n_qubits <= 6
            else None,
        ),
        schedule=schedule_record(rec.schedule),
        space=space_record(space, rec.selection),
        preparation=PreparationRecord(
            nbar={int(m): float(v) for m, v in rec.preparation.nbar.items()},
            duration_s=float(rec.preparation.duration_s),
            preparation_error={
                int(i): float(rec.preparation.preparation_error(i)) for i in rec.preparation.pumps
            },
            provenance=tuple(rec.preparation.provenance),
            notes=tuple(rec.preparation.notes),
        ),
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
        traces=tuple(traces),
        gate_local=None if rec.gate_local is None else gate_local_record(rec.gate_local),
        readout=readout_record(result, rec),
        results=results_record(result, circuit, register_fid),
        diagnostics=diagnostics_record(result.diagnostics, opts, wall_time_s),
        notes=tuple(rec.notes),
        core_gaps=core.CORE_GAPS,
        joint_store_dimension_max=int(joint_store_dimension_max),
    )


# ---- executing a job ---------------------------------------------------------------------------------------------------------------


@dataclass
class LiveRun:
    """The in-process objects behind a record that re-simulation needs and the record cannot carry (Section 14.3)."""

    device: core.Device
    table: core.CalibrationTable
    result: core.Result
    core_record: core.RunRecord
    options: core.SolverOptions
    run_kwargs: dict[str, Any]

    @property
    def space(self) -> core.HilbertSpace:
        return self.result.diagnostics.space

    @property
    def schedule(self) -> core.Schedule:
        return self.core_record.schedule

    @property
    def steps(self) -> tuple[core.GateStep, ...]:
        return core.gate_steps(self.core_record.schedule)


def calibrate_for(job: JobSpec, preset: core.DevicePreset) -> core.CalibrationTable:
    cal = job.calibration
    if not cal.surrogate:
        raise RecordError(
            "M11.1 records rebuild surrogate tables only; the simulated-experiment path is M11.3's stale-badge job"
        )
    return core.calibrate(
        preset.device,
        seed=cal.seed,
        pairs=[tuple(p) for p in cal.pairs],
        detection_records=cal.detection_records,
        detection_windows_s=cal.detection_windows_s,
        **job.run_kwargs(),
    )


def execute(
    job: JobSpec, preset: core.DevicePreset | None = None, *, joint_store_dimension_max: int = 4096
) -> tuple[Record, LiveRun]:
    """Calibrate, run and record one job; returns the record and the live handle re-simulation uses."""
    pre = preset if preset is not None else job.device.build()
    if pre.device.hash() != job.device.hash:
        raise RecordError("the preset given does not match the job's device hash")
    table = calibrate_for(job, pre)
    circuit = job.circuit.to_core()
    opts = job.solver_options()
    kwargs = job.run_kwargs()
    t0 = time.perf_counter()
    result = core.run(
        circuit,
        pre.device,
        job.shots,
        table=table,
        t0_s=job.t0_s,
        samples=job.samples,
        level=job.level,
        seed=job.seed,
        options=opts,
        readout=job.readout,
        entangler=job.entangler,
        keep_final_state=job.keep_final_state,
        noise=job.noise,
        internal_levels=job.internal_levels,
        stark_compensation=job.stark_compensation,
        caps=job.caps,
        **kwargs,
    )
    wall = time.perf_counter() - t0
    record = build_record(
        job, pre.device, table, result, wall_time_s=wall, joint_store_dimension_max=joint_store_dimension_max
    )
    live = LiveRun(
        device=pre.device,
        table=table,
        result=result,
        core_record=core.last_record(result),
        options=opts,
        run_kwargs=kwargs,
    )
    return record, live


__all__ = [
    "DEFAULT_DETECTION_WINDOWS_S",
    "NATIVE_GATE_SET",
    "PRESETS",
    "RECORD_FORMAT",
    "BeamRecord",
    "BoundaryState",
    "BranchRecord",
    "CalEntryRecord",
    "CalibrationRef",
    "ChannelEntryRecord",
    "ChannelPieceRecord",
    "ChannelSummaryRecord",
    "CircuitRecord",
    "CompiledRecord",
    "ConvergenceRecord",
    "DerivedRecord",
    "DetectorRecord",
    "DeviceCard",
    "DeviceRef",
    "DiagnosticsRecord",
    "DriveRef",
    "EventRecord",
    "GateLocalRecord",
    "GateLocalStepRecord",
    "IonRatesRecord",
    "JobSpec",
    "LiveRun",
    "ModeRecord",
    "NoiseSampleRecord",
    "OpRecord",
    "PlayedGateRecord",
    "PreparationRecord",
    "PulseRecord",
    "ReadoutRecord",
    "Record",
    "RecordError",
    "ReplayGate",
    "ReplayRecord",
    "ResultsRecord",
    "SampledFn",
    "ScheduleRecord",
    "SegmentRecord",
    "SpaceRecord",
    "StepRecord",
    "TableRecord",
    "TargetRecord",
    "ToneRecord",
    "TraceRecord",
    "TruncationRecord",
    "WaveformRecord",
    "ZoomTrace",
    "build_record",
    "calibrate_for",
    "execute",
    "job_for_preset",
    "options_record",
    "sampled_fn",
]
