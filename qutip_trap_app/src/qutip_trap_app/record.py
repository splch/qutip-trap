"""The run record: what one run produced, as plain values and numpy arrays, for the screens to read; ``execute`` runs a
:class:`Job` and returns its record with the :class:`LiveRun` that re-simulation needs."""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qutip_trap_app import core

# Conventions of every record: bitstring keys read qubit 0 rightmost; register-order states have ion 0 as the first tensor
# factor; mode indices are positions in Crystal.modes; frequencies are in Hz.

DETECTION_RECORDS = 500
"""Photon records per detection window in the calibration: the core's default of 10,000 costs two more seconds per run."""
TONE_SAMPLES = 201
"""Points at which a tone's envelope is sampled over its pulse (a display resolution)."""
JOINT_STORE_DIMENSION_MAX = 4096
"""The largest joint dimension whose state a boundary keeps as an array."""
LOOP_GRID = 20001
"""Points of the uniform grid a spin-branch loop is integrated on."""
LOOP_STORE = 1001
"""Points of a stored loop (the first and last always kept)."""


# ---- the job -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """A run request: the core's ``RunSpec`` on the example 171Yb+ chain of ``n_ions`` ions, built with the preset's
    keyword arguments ``device_kwargs``."""

    spec: core.RunSpec
    n_ions: int
    device_kwargs: dict[str, float] = field(default_factory=dict)

    def machine(self) -> core.Machine:
        """The machine the job runs on: the preset with the spec's option objects and level."""
        spec = self.spec
        return dataclasses.replace(
            core.yb171_chain(self.n_ions, **self.device_kwargs),
            physics=spec.physics,
            numerics=spec.numerics,
            readout=spec.readout,
            level=spec.level,
        )


# ---- the schedule ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeRecord:
    index: int
    family: str
    family_index: int
    omega_hz: float


@dataclass(frozen=True)
class ToneRecord:
    """One tone of a pulse: its detuning from the carrier and its phase at the pulse start, and its envelope Omega(t)/2 pi
    at ``TONE_SAMPLES`` points over the pulse."""

    detuning_hz: float
    phase_rad: float
    envelope_hz: np.ndarray


@dataclass(frozen=True)
class PulseRecord:
    index: int
    gate_id: str | None
    ions: tuple[int, ...]
    kind: str
    t_start_s: float
    t_end_s: float
    tones: tuple[ToneRecord, ...]
    stark_shift_hz: float
    """The light shift of the pulse's own beams at its start."""
    crosstalk: dict[int, complex]
    """Rabi (amplitude) ratio onto each neighbour."""

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


@dataclass(frozen=True)
class PlayedGateRecord:
    """An entangling gate as played: its pair and its waveform's per-mode angle and residual displacement at closure."""

    gate_id: str
    pair: tuple[int, int]
    duration_s: float
    chi_m: dict[int, float]
    alpha_m: dict[int, complex]
    chi_total_rad: float


@dataclass(frozen=True)
class TargetRecord:
    """The ideal unitary of one played gate piece, on ``ions`` in matrix order (the first ion the first factor)."""

    gate_id: str
    ions: tuple[int, ...]
    native_name: str
    native_params: tuple[float, ...]
    t_start_s: float
    t_end_s: float
    unitary: np.ndarray


@dataclass(frozen=True)
class StepRecord:
    """One unit of the time axis a zoom re-simulates: a gate piece with all its pulses, or an idle interval."""

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
    gates: tuple[PlayedGateRecord, ...]
    targets: tuple[TargetRecord, ...]
    steps: tuple[StepRecord, ...]
    measure: tuple[float, float] | None
    """Start and end of the detection window."""
    t0_s: float
    pulses_end_s: float
    duration_s: float


@dataclass(frozen=True)
class SpaceRecord:
    """The joint space the run integrated on, with the class of every mode."""

    ion_dims: tuple[int, ...]
    dims: tuple[int, ...]
    dimension: int
    caps: dict[int, int]
    """Fock dimension of every resolved mode."""
    enr_group: tuple[tuple[int, ...], int] | None
    mode_class: dict[int, str]
    nbar: dict[int, float]


# ---- the dynamics ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchRecord:
    """One branch of the initial mixture: the Fock states of the modes, with its weight."""

    index: int
    weight: float
    fock: dict[int, int]


@dataclass(frozen=True)
class TraceRecord:
    """One (sample, branch) evolution at the times the engine stored."""

    sample_index: int
    branch: int
    weight: float
    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    """``P1[i]`` and ``n[m]`` as the engine keys them."""
    reduced_internal: np.ndarray
    """(T, d, d) register density matrices in the register order."""
    mode_nbar: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
    boundary_population: dict[int, float]
    mode_marginal: dict[int, np.ndarray] | None
    """Per carried mode, the (T, d) Fock populations at the stored times when the engine stored them."""

    def index_at(self, t_s: float, tolerance_s: float = 1e-12) -> int:
        """The last stored index at time ``t_s`` (a segment end)."""
        hits = np.flatnonzero(np.abs(self.times_s - t_s) <= tolerance_s)
        if hits.size == 0:
            raise KeyError(f"no stored point at t = {t_s:.9g} s")
        return int(hits[-1])


@dataclass(frozen=True)
class GateLocalStepRecord:
    gate_id: str
    register_after: np.ndarray | None
    """The register density matrix after the step for the first sample; None when the core kept none (a pure-state
    ensemble, or a register above its store cap)."""


@dataclass(frozen=True)
class GateLocalRecord:
    steps: tuple[GateLocalStepRecord, ...]
    largest_local_dimension: int


@dataclass(frozen=True)
class BranchLoop:
    """The phase-space trajectory alpha_im(t) of one spin branch on one mode of a played entangling waveform, by the core's
    closed-form kernel: the loop the pulse closes, whose swept area is the entangling angle."""

    gate_id: str
    ion: int
    mode: int
    alpha: np.ndarray
    chi_m_rad: float
    """The angle the played waveform books on this mode."""
    chi_closed_form_rad: float
    """2 Im int conj(alpha_a) d alpha_b over the pair's two loops on this mode, at the played amplitude."""


# ---- results and diagnostics ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultsRecord:
    bitstrings: np.ndarray
    heralds: np.ndarray
    levels: np.ndarray
    """(shots, n_ions): the sampled internal level of every ion behind each kept shot."""
    sample_of_shot: np.ndarray
    counts: dict[str, int]
    probabilities: dict[str, float]
    error_bars: dict[str, float]
    target_probabilities: dict[str, float]
    """The ideal circuit's distribution, shown beside the simulated one."""
    discarded_shots: int
    register_fidelity: float | None
    """<ideal| rho |ideal> of the recombined register against the compiled circuit's state."""


@dataclass(frozen=True)
class DiagnosticsRecord:
    level: Literal["JOINT_EXACT", "GATE_LOCAL"]
    integrator: str
    tolerances: tuple[float, float]
    samples: int
    trajectories: int
    branches: int
    shots_per_sample: tuple[int, ...]
    boundary_population: dict[int, float]
    boundary_population_max: float
    """The threshold the run was judged against."""
    margin_levels: dict[int, int]
    margin_reached: dict[int, int]
    frozen_contribution: dict[int, tuple[float, float]]
    """Per frozen mode: (|alpha|^2 (2 nbar + 1), chi)."""
    wall_time_s: float


# ---- what re-simulation adds to a record -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryState:
    """The joint state at the start of a step for one (sample, branch), from chaining the engine over the steps before it."""

    key: str
    step_index: int
    sample_index: int
    branch: int
    joint: np.ndarray | None
    """The joint ket (D,) or density matrix (D, D); None above ``JOINT_STORE_DIMENSION_MAX``."""
    joint_dims: tuple[int, ...]
    internal: np.ndarray
    fock: dict[int, np.ndarray]
    """Per carried mode, the Fock populations."""


@dataclass(frozen=True)
class ZoomTrace:
    """A gate step re-simulated at fine time resolution."""

    key: str
    step_index: int
    sample_index: int
    branch: int
    n_store: int
    trace: TraceRecord
    fock_start: dict[int, np.ndarray]
    fock_end: dict[int, np.ndarray]
    integrators: tuple[str, ...]
    method: str
    wall_time_s: float
    engine_dimension: int


@dataclass(frozen=True)
class Recheck:
    """A convergence re-check of one zoomed step: the largest change of every stored observable when the numerics are
    tightened from ``before`` to ``after``."""

    key: str
    what: str
    before: dict[str, float]
    after: dict[str, float]
    changes: dict[str, float]
    tol: float

    @property
    def max_change(self) -> float:
        return max(self.changes.values(), default=0.0)

    @property
    def converged(self) -> bool:
        return self.max_change < self.tol


@dataclass(frozen=True)
class DriveTerm:
    """One (pulse, ion) term sigma_+ (x) prod_m D_m(i eta_im) of H(t) with what went into its coefficient."""

    pulse: str
    ion: int
    primary_ion: int
    etas: dict[int, float]
    frozen_n: dict[int, int]
    debye_waller: float
    carrier_factor: float
    crosstalk: complex
    rabi_scale: float
    omega_peak_hz: float
    tones: tuple[tuple[float, float, float], ...]
    """Per tone of the pulse: (detuning, phase at the start, peak Omega/2 pi of the addressed ion)."""
    operator_nnz: int
    matrix_elements: dict[int, np.ndarray]
    """Per resolved mode, the (d, d) table |<n'|D(i eta)|n>| at this term's eta."""
    frozen_debye_waller: dict[int, float]


@dataclass(frozen=True)
class CollapseTerm:
    channel: str
    rate_hz: float
    ion: int | None
    mode: int | None
    integrated: bool
    """Whether the run integrated this channel."""
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
    """The terms of H(t) and the collapse operators the engine assembles for one step and one (sample, branch)."""

    key: str
    step_index: int
    sample_index: int
    branch: int
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
    drives: tuple[DriveTerm, ...]
    """The drive terms of the first segment (the operators are the same on every segment; the coefficients differ)."""
    collapse: tuple[CollapseTerm, ...]
    approximations: tuple[str, ...]
    n_drive_terms: int
    omega_max_hz: float
    kernel: str
    stark_shifts_hz: dict[int, float]


@dataclass(frozen=True)
class ProcessMatrixRecord:
    """Process tomography of one step from its recorded initial motional state, for one (sample, branch)."""

    key: str
    step_index: int
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


Cached = BoundaryState | ZoomTrace | Recheck | HamiltonianRecord | ProcessMatrixRecord


# ---- the record -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """Everything the screens read about one run; ``cache`` holds what re-simulation added, by key."""

    key: str
    job: Job
    n_ions: int
    modes: tuple[ModeRecord, ...]
    schedule: ScheduleRecord
    space: SpaceRecord
    sample_values: tuple[dict[str, float], ...]
    """The quasi-static noise values drawn for each dynamical sample."""
    branches: tuple[BranchRecord, ...]
    traces: tuple[TraceRecord, ...]
    """Sample-major, branch-minor; empty for a GATE_LOCAL run."""
    gate_local: GateLocalRecord | None
    results: ResultsRecord
    diagnostics: DiagnosticsRecord
    branch_loops: tuple[BranchLoop, ...]
    cache: dict[str, Cached] = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        """The circuit's qubits; the register spans every ion (``n_ions``)."""
        return self.job.spec.circuit.n_qubits

    @property
    def n_samples(self) -> int:
        return len(self.sample_values)

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

    def cached[T](self, key: str, kind: type[T]) -> T | None:
        item = self.cache.get(key)
        return item if isinstance(item, kind) else None

    def with_cached(self, *items: Cached) -> Record:
        return dataclasses.replace(self, cache={**self.cache, **{item.key: item for item in items}})


# ---- building a record ----------------------------------------------------------------------------------------------------------


ToneFn = Callable[[float], float] | np.ndarray | float
"""A tone field of the core's pulses: a constant, an array sampled uniformly over the pulse, or a callable of the time
since the pulse start."""


def _at_start(value: ToneFn) -> float:
    if isinstance(value, np.ndarray):
        return float(value[0])
    if callable(value):
        return float(value(0.0))
    return float(value)


def _sampled(value: ToneFn, duration_s: float, n: int = TONE_SAMPLES) -> np.ndarray:
    tau = np.linspace(0.0, duration_s, n)
    if isinstance(value, np.ndarray):
        return np.asarray(np.interp(tau, np.linspace(0.0, duration_s, value.size), value), dtype=float)
    if callable(value):
        return np.array([float(value(float(t))) for t in tau], dtype=float)
    return np.full(n, float(value))


def schedule_record(sched: core.Schedule) -> ScheduleRecord:
    index_of = {id(p): k for k, p in enumerate(sched.pulses)}
    pulses = tuple(
        PulseRecord(
            index=k,
            gate_id=p.gate_id,
            ions=tuple(int(i) for i in p.drive.ions),
            kind=str(p.drive.kind),
            t_start_s=float(p.t_start_s),
            t_end_s=float(p.t_end_s),
            tones=tuple(
                ToneRecord(
                    detuning_hz=_at_start(t.detuning_hz),
                    phase_rad=_at_start(t.phase_rad),
                    envelope_hz=_sampled(t.envelope_hz, p.duration_s),
                )
                for t in p.drive.tones
            ),
            stark_shift_hz=_at_start(p.drive.stark_shift_hz),
            crosstalk={int(j): complex(e) for j, e in p.drive.crosstalk.items()},
        )
        for k, p in enumerate(sched.pulses)
    )
    gates = tuple(
        PlayedGateRecord(
            gate_id=str(g.gate_id),
            pair=(int(g.pair[0]), int(g.pair[1])),
            duration_s=float(g.waveform.duration_s),
            chi_m={int(m): float(v) for m, v in g.waveform.chi_m.items()},
            alpha_m={int(m): complex(v) for m, v in g.waveform.alpha_m.items()},
            chi_total_rad=float(g.waveform.chi_total_rad),
        )
        for g in sched.gates
    )
    targets = tuple(
        TargetRecord(
            gate_id=str(t.gate_id),
            ions=tuple(int(i) for i in t.ions),
            native_name=str(t.native[0]),
            native_params=tuple(float(x) for x in t.native[1]),
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
    measure = next(
        ((float(e.t_start_s), float(e.t_end_s)) for e in sched.events if e.kind == "measure"), None
    )
    starts = [p.t_start_s for p in sched.pulses] + [a for a, _ in sched.idle]
    return ScheduleRecord(
        pulses=pulses,
        gates=gates,
        targets=targets,
        steps=steps,
        measure=measure,
        t0_s=float(sched.t0_s) if sched.t0_s is not None else min(starts + [0.0]),
        pulses_end_s=float(sched.pulses_end_s),
        duration_s=float(sched.duration_s),
    )


def trace_record(tr: core.Traces, *, sample_index: int, branch: int, weight: float) -> TraceRecord:
    reduced = (
        np.stack([np.asarray(r.full(), dtype=complex) for r in tr.reduced_internal])
        if tr.reduced_internal
        else np.zeros((0, 1, 1), dtype=complex)
    )
    return TraceRecord(
        sample_index=sample_index,
        branch=branch,
        weight=float(weight),
        times_s=np.asarray(tr.times_s, dtype=float),
        expectations={str(k): np.asarray(v) for k, v in tr.expectations.items()},
        reduced_internal=reduced,
        mode_nbar={int(m): np.asarray(np.real(v), dtype=float) for m, v in tr.mode_occupations.items()},
        alpha_m={int(m): np.asarray(v, dtype=complex) for m, v in tr.alpha_m.items()},
        jumps=tuple((float(t), str(c)) for t, c in tr.jumps),
        boundary_population={int(m): float(v) for m, v in tr.boundary_population.items()},
        mode_marginal=None
        if tr.mode_marginal is None
        else {int(m): np.asarray(v, dtype=float) for m, v in tr.mode_marginal.items()},
    )


def branch_loops(
    sched: core.Schedule, device: core.Device, nbar: Mapping[int, float]
) -> tuple[BranchLoop, ...]:
    """The spin-branch loops of every played entangling waveform on the run's own modes, by the closed-form kernel the
    pulse solver closed the pulse with. A waveform without segments carries no explicit envelope and gets no loop."""
    out: list[BranchLoop] = []
    for g in sched.gates:
        wf = g.waveform
        if wf.segments is None or len(g.beams) != 2:
            continue
        pair = (int(g.pair[0]), int(g.pair[1]))
        gm = core.gate_modes(device, pair, (int(g.beams[0]), int(g.beams[1])), nbar=nbar)
        env = core.envelope_of(wf, pair)
        analytic: Mapping[tuple[int, int, int], float] | None = None
        if isinstance(env, core.SegmentedEnvelope):
            analytic = core.integrals_segmented(env, gm, "choi").chi_by_mode
            sampled = env.sampled(LOOP_GRID)
        else:
            sampled = env
        n = int(sampled.times_s.size)
        keep = np.arange(0, n, max(1, (n - 1) // (LOOP_STORE - 1)))
        if keep[-1] != n - 1:
            keep = np.append(keep, n - 1)
        for m in gm.modes:
            paths = {ion: core.trajectory_sampled(sampled, gm, ion, int(m), kernel="choi") for ion in pair}
            if analytic is not None:
                chi = float(
                    analytic.get((pair[0], pair[1], int(m)), analytic.get((pair[1], pair[0], int(m)), 0.0))
                )
            else:
                a, b = paths[pair[0]], paths[pair[1]]
                chi = 2.0 * float(np.sum(np.imag(np.conj(a[:-1]) * np.diff(b))))
            for ion in pair:
                out.append(
                    BranchLoop(
                        gate_id=str(g.gate_id),
                        ion=int(ion),
                        mode=int(m),
                        alpha=np.asarray(paths[ion][keep], dtype=complex),
                        chi_m_rad=float(wf.chi_m.get(int(m), 0.0)),
                        chi_closed_form_rad=chi,
                    )
                )
    return tuple(out)


def build_record(job: Job, live: LiveRun) -> Record:
    """The record of one run, from the core's Result and the RunRecord behind it."""
    result, run = live.result, live.run
    diag = result.diagnostics
    space = diag.space
    n_branches = len(run.branches)
    if run.traces and len(run.traces) != len(result.noise_samples) * n_branches:
        raise ValueError(
            f"{len(run.traces)} traces do not factor as {len(result.noise_samples)} samples x {n_branches} branches"
        )
    traces = tuple(
        trace_record(
            tr,
            sample_index=k // n_branches,
            branch=k % n_branches,
            weight=run.branches[k % n_branches].weight,
        )
        for k, tr in enumerate(run.traces)
    )
    gate_local = None
    if run.gate_local is not None:
        gate_local = GateLocalRecord(
            steps=tuple(
                GateLocalStepRecord(
                    str(s.gate_id),
                    None if s.register_after is None else np.asarray(s.register_after, dtype=complex),
                )
                for s in run.gate_local.steps
            ),
            largest_local_dimension=int(run.gate_local.largest_local_dimension),
        )
    nbar = {int(m): float(v) for m, v in run.preparation.nbar.items()}
    device = live.machine.device
    options = live.options
    return Record(
        key=uuid.uuid4().hex[:12],
        job=job,
        n_ions=int(device.crystal.n_ions),
        modes=tuple(
            ModeRecord(k, str(m.family), int(m.index), float(m.omega_hz))
            for k, m in enumerate(device.crystal.modes)
        ),
        schedule=schedule_record(run.schedule),
        space=SpaceRecord(
            ion_dims=tuple(int(d) for d in space.ion_dims),
            dims=tuple(int(d) for d in space.dims),
            dimension=int(space.dimension),
            caps={int(t.mode): int(t.d) for t in space.resolved},
            enr_group=None
            if space.enr_group is None
            else (tuple(int(m) for m in space.enr_group[0]), int(space.enr_group[1])),
            mode_class={int(m): str(c) for m, c in diag.mode_class.items()},
            nbar={int(m): float(v) for m, v in run.selection.nbar.items()},
        ),
        sample_values=tuple({str(k): float(v) for k, v in s.values.items()} for s in result.noise_samples),
        branches=tuple(
            BranchRecord(k, float(b.weight), {int(m): int(n) for m, n in b.fock.items()})
            for k, b in enumerate(run.branches)
        ),
        traces=traces,
        gate_local=gate_local,
        results=ResultsRecord(
            bitstrings=np.asarray(result.bitstrings, dtype=np.uint8),
            heralds=np.asarray(result.heralds, dtype=np.uint8),
            levels=np.asarray(run.outcome.levels),
            sample_of_shot=np.asarray(result.sample_of_shot, dtype=np.int64),
            counts={str(k): int(v) for k, v in result.counts.items()},
            probabilities={str(k): float(v) for k, v in result.probabilities.items()},
            error_bars={str(k): float(v) for k, v in result.error_bars.items()},
            target_probabilities={
                str(k): float(v) for k, v in core.ideal_probabilities(job.spec.circuit).items()
            },
            discarded_shots=int(result.discarded_shots),
            register_fidelity=None if result.final_state is None else float(core.register_fidelity(result)),
        ),
        diagnostics=DiagnosticsRecord(
            level=diag.level,
            integrator=str(diag.integrator),
            tolerances=(float(diag.tolerances[0]), float(diag.tolerances[1])),
            samples=int(diag.samples),
            trajectories=int(diag.trajectories),
            branches=int(diag.branches),
            shots_per_sample=tuple(int(m) for m in diag.shots_per_sample_realized),
            boundary_population={int(m): float(v) for m, v in diag.boundary_population.items()},
            boundary_population_max=float(options.boundary_population_max),
            margin_levels={int(m): int(v) for m, v in diag.margin_levels.items()},
            margin_reached={int(m): int(v) for m, v in diag.margin_reached.items()},
            frozen_contribution={
                int(m): (float(c[0]), float(c[1])) for m, c in diag.frozen_contribution.items()
            },
            wall_time_s=float(result.duration_s),
        ),
        branch_loops=branch_loops(run.schedule, device, nbar),
    )


# ---- running a job ------------------------------------------------------------------------------------------------------------------


@dataclass
class LiveRun:
    """The core objects behind a record that re-simulation needs: the machine with its calibration table pinned, the result
    and the core's RunRecord behind it, and the joint states re-simulation reached, by boundary key."""

    machine: core.Machine
    result: core.Result
    run: core.RunRecord
    states: dict[str, core.State] = field(default_factory=dict)

    @property
    def space(self) -> core.HilbertSpace:
        return self.result.diagnostics.space

    @property
    def options(self) -> core.SolverOptions:
        return self.machine.numerics.to_solver_options(self.machine.physics)

    @property
    def steps(self) -> tuple[core.GateStep, ...]:
        return core.gate_steps(self.run.schedule)


def execute(job: Job, progress: Callable[[core.Progress], None] | None = None) -> tuple[Record, LiveRun]:
    """Calibrate the job's machine for its circuit's pairs, run it and record the run; ``progress`` is the core's callback
    per pulse, branch, sample and readout."""
    spec = job.spec
    machine = job.machine().calibrated(
        seed=spec.seed, pairs=spec.circuit.entangling_pairs(), detection_records=DETECTION_RECORDS
    )
    result = machine.run(
        spec.circuit, spec.shots, seed=spec.seed, keep_final_state=spec.keep_final_state, progress=progress
    )
    live = LiveRun(machine, result, core.last_record(result))
    return build_record(job, live), live
