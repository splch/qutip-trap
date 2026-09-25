"""On-demand re-simulation of one gate step of a recorded JOINT_EXACT run, cached on the record (PLAN.md Sections 14.3, 14.7).

The core's ``Traces`` hold the joint state only at the end of an evolution, so the state at the start of a step is obtained
by chaining the engine over the gate steps from the run's own initial state, each step played as a sub-schedule declared at
its own start; the engine is built as the run built its own (same table, drives, noise and seeds), so the chain reproduces
the run. A zoom replays one step with ``n_store`` stored points per integration segment and the Fock marginals on, keyed by
(step, sample, branch, points, solver options). Two convergence re-checks run on a zoomed step (Sections 5.5, 9.9):
tolerances tightened by ten, and every resolved cap raised by two (re-chained from the initial state on the grown space).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import qutip as qt

from qutip_trap_app import core
from qutip_trap_app.record import (
    BoundaryState,
    CollapseRecord,
    ConvergenceRecord,
    DriveTermRecord,
    HamiltonianRecord,
    LiveRun,
    ProcessMatrixRecord,
    Progress,
    Record,
    RecordError,
    SegmentSummary,
    ZoomTrace,
    embed_on_register,
    options_digest,
    trace_record,
)

DEFAULT_ZOOM_POINTS = 201
"""Stored points per integration segment of a zoomed step: the Level 3 time resolution."""

CONVERGENCE_TOL = 1e-6
"""Section 5.5: the change in a population under a tolerance tightening or a cap raise that still counts as converged."""

TIGHTEN_FACTOR = 10.0
CAP_RAISE = 2

BOUNDARY_JOINT_MAX = 4096
"""The largest joint dimension whose state a cached boundary keeps (above it only the reduced states are kept)."""


TWO_PI = 2.0 * math.pi


# ---- the engine as the run built it ----------------------------------------------------------------------------------------------


def engine_for(live: LiveRun, *, store_per_segment: int = 2) -> core.JointExactEngine:
    """The JOINT_EXACT engine with the settings the run gave its own."""
    return core.JointExactEngine(
        builder_options=None,
        store_per_segment=int(store_per_segment),
        channels=(),
        qubit_shifts_hz=dict(live.core_record.qubit_shifts_hz),
        device_channels=True,
        levels_by_ion=None,
        hardware_chain=True,
        table=live.table,
    )


def initial_state(
    record: Record, live: LiveRun, sample_index: int, branch: int, *, space: core.HilbertSpace | None = None
) -> tuple[core.State, core.NoiseSample]:
    """The (sample, branch) initial state and noise sample exactly as the run prepared them (Section 5.3's Fock sum)."""
    if record.diagnostics.level != "JOINT_EXACT":
        raise RecordError(f"a {record.diagnostics.level} run has no joint initial state to re-simulate from")
    sp = space if space is not None else live.space
    state0 = core.prepare(live.device, sp, live.table, preparation=live.core_record.preparation)
    br = record.branches[branch]
    total = sum(b.weight for b in record.branches)
    state = sp.initial_state(
        list(br.levels),
        fock={m: n for m, n in br.fock.items() if sp.mode_class(m) in ("resolved", "enr")},
        thermal={m: float(state0.motional.nbar.get(m, 0.0)) for m in sp.frozen},
        provenance=tuple(state0.provenance) + (f"branch[{branch}]",),
    )
    extra = {core.key_frozen_n(m): float(n) for m, n in br.fock.items() if sp.mode_class(m) == "frozen"}
    extra[core.KEY_BRANCH_WEIGHT] = float(br.weight / total)
    return state, record.noise_samples[sample_index].to_core(extra)


def sub_schedule(live: LiveRun, step: core.GateStep) -> core.Schedule:
    """One gate step as a schedule declared at its own start (the engine plays it exactly as the whole run did)."""
    sched = live.schedule
    idle = tuple(iv for iv in sched.idle if iv[0] >= step.t_start_s - 1e-15 and iv[1] <= step.t_end_s + 1e-15)
    return core.Schedule(
        pulses=step.pulses,
        idle=idle,
        events=(),
        phase_frame=dict(sched.phase_frame),
        gates=step.played,
        targets=step.targets,
        t0_s=step.t_start_s,
    )


def _boundary(
    state: core.State, *, step_index: int, sample_index: int, branch: int, dims: tuple[int, ...]
) -> BoundaryState:
    joint: np.ndarray | None = None
    if state.joint is not None and state.joint.shape[0] <= BOUNDARY_JOINT_MAX:
        arr = np.asarray(state.joint.full(), dtype=complex)
        joint = arr.reshape(-1) if state.joint.isket else arr
    return BoundaryState(
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        joint=joint,
        joint_dims=dims,
        internal=np.asarray(state.internal.full(), dtype=complex),
        mode_reduced={int(m): np.asarray(r.full(), dtype=complex) for m, r in state.motional.reduced.items()},
        nbar={int(m): float(v) for m, v in state.motional.nbar.items()},
    )


def _state_from_boundary(b: BoundaryState, space: core.HilbertSpace) -> core.State:
    if b.joint is None:
        raise RecordError(
            "the cached boundary state holds no joint state (above the store cap); recompute the chain"
        )
    dims = (
        [list(b.joint_dims), [1] * len(b.joint_dims)]
        if b.joint.ndim == 1
        else [list(b.joint_dims), list(b.joint_dims)]
    )
    joint = qt.Qobj(b.joint.reshape(-1, 1) if b.joint.ndim == 1 else b.joint, dims=dims)
    internal = qt.Qobj(b.internal, dims=[list(space.ion_dims), list(space.ion_dims)])
    reduced = {m: qt.Qobj(r, dims=[[r.shape[0]], [r.shape[0]]]) for m, r in b.mode_reduced.items()}
    motional = core.MotionalModel(reduced=reduced, nbar=dict(b.nbar), frozen=tuple(space.frozen))
    return core.State(internal=internal, motional=motional, joint=joint, provenance=("app.boundary",))


class _Chain:
    """Chained evolution over the gate steps for one (sample, branch), on a given space."""

    def __init__(
        self,
        record: Record,
        live: LiveRun,
        sample_index: int,
        branch: int,
        *,
        space: core.HilbertSpace | None = None,
    ) -> None:
        self.record = record
        self.live = live
        self.space = space if space is not None else live.space
        self.steps = live.steps
        self.state, self.sample = initial_state(record, live, sample_index, branch, space=self.space)
        self.seeds = core.SeedSpec(record.job.seed)
        self.position = 0
        self.engine_calls = 0

    def play(self, engine: core.JointExactEngine, options: core.SolverOptions) -> core.Traces:
        """Evolve the chain's state through its next step."""
        tr = engine.run_pulses(
            self.live.device,
            sub_schedule(self.live, self.steps[self.position]),
            self.state,
            self.space,
            self.sample,
            self.seeds,
            options,
        )
        self.engine_calls += 1
        self.state = tr.final
        self.position += 1
        return tr


def boundary_states(
    record: Record, live: LiveRun, sample_index: int = 0, branch: int = 0, *, up_to: int | None = None
) -> Record:
    """Cache the joint state at the start of every step up to ``up_to`` (default: every step and the schedule's end)."""
    steps = live.steps
    last = len(steps) if up_to is None else int(up_to)
    if not 0 <= last <= len(steps):
        raise IndexError(f"step index {last} outside 0..{len(steps)}")
    if all(record.boundary(k, sample_index, branch) is not None for k in range(last + 1)):
        return record
    engine = engine_for(live)
    chain = _Chain(record, live, sample_index, branch)
    dims = tuple(int(d) for d in chain.space.dims)
    new = [_boundary(chain.state, step_index=0, sample_index=sample_index, branch=branch, dims=dims)]
    for k in range(last):
        chain.play(engine, live.options)
        new.append(
            _boundary(chain.state, step_index=k + 1, sample_index=sample_index, branch=branch, dims=dims)
        )
    return record.with_boundaries(new)


def zoom_key(
    step_index: int, sample_index: int, branch: int, n_store: int, options: core.SolverOptions
) -> str:
    """The cache key of a zoom made with ``options``; a zoom always stores the Fock marginals, so the key is taken over the
    options with ``store_marginals`` on."""
    opts = dataclasses.replace(options, store_marginals=True)
    return f"step{step_index}/s{sample_index}/b{branch}/n{n_store}/{options_digest(opts)}"


@dataclass(frozen=True)
class ZoomStats:
    """What a zoom cost: engine calls made (0 = served from the cache) and wall time."""

    engine_calls: int
    wall_time_s: float
    cached: bool


def zoom(
    record: Record,
    live: LiveRun,
    step_index: int,
    sample_index: int = 0,
    branch: int = 0,
    *,
    n_store: int = DEFAULT_ZOOM_POINTS,
    options: core.SolverOptions | None = None,
) -> tuple[Record, ZoomTrace, ZoomStats]:
    """Re-simulate one gate step at fine resolution from its recorded initial state, with the per-time Fock populations of
    every resolved mode (``core.Traces.mode_marginal``); cached on the record by key."""
    opts = dataclasses.replace(options if options is not None else live.options, store_marginals=True)
    key = zoom_key(step_index, sample_index, branch, n_store, opts)
    cached = record.zoom(key)
    if cached is not None:
        return record, cached, ZoomStats(engine_calls=0, wall_time_s=0.0, cached=True)
    t0 = time.perf_counter()
    rec = boundary_states(record, live, sample_index, branch, up_to=step_index)
    start = rec.boundary(step_index, sample_index, branch)
    assert start is not None
    space = live.space
    engine = engine_for(live, store_per_segment=n_store)
    _st, sample = initial_state(rec, live, sample_index, branch)
    tr = engine.run_pulses(
        live.device,
        sub_schedule(live, live.steps[step_index]),
        _state_from_boundary(start, space),
        space,
        sample,
        core.SeedSpec(rec.job.seed),
        opts,
    )
    report = engine.last_report
    assert report is not None
    wall = time.perf_counter() - t0
    trace = trace_record(
        tr,
        sample_index=sample_index,
        sample_id=rec.noise_samples[sample_index].sample_id,
        branch=branch,
        weight=rec.branches[branch].weight,
        ion_dims=space.ion_dims,
        joint_dims=space.dims,
    )
    z = ZoomTrace(
        key=key,
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        n_store=n_store,
        options_digest=options_digest(opts),
        trace=trace,
        fock_end={m: np.real(np.diag(r)).astype(float) for m, r in trace.final_mode_reduced.items()},
        fock_start={m: np.real(np.diag(r)).astype(float) for m, r in start.mode_reduced.items()},
        integrators=tuple(dict.fromkeys(s.integrator for s in report.segments)),
        method=str(report.method),
        approximations=tuple(report.approximations),
        wall_time_s=wall,
        engine_dimension=int(space.dimension),
    )
    return rec.with_zoom(z), z, ZoomStats(engine_calls=1, wall_time_s=wall, cached=False)


# ---- convergence re-checks (Sections 5.5, 9.9) ------------------------------------------------------------------------------------


def _changes(trace_a: Mapping[str, np.ndarray], trace_b: Mapping[str, np.ndarray]) -> dict[str, float]:
    """max |a - b| per stored observable of two traces of the same step."""
    if set(trace_a) != set(trace_b):
        raise RecordError("the two zooms store different observables")
    out: dict[str, float] = {}
    for k in sorted(trace_a):
        a = np.asarray(trace_a[k], dtype=float)
        b = np.asarray(trace_b[k], dtype=float)
        if a.shape != b.shape:
            raise RecordError(f"observable {k!r}: {a.shape} against {b.shape} stored points")
        out[k] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return out


def tolerance_recheck(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, ConvergenceRecord]:
    """Section 5.5's tolerance arm on one zoomed step: atol and rtol tightened by ten, the change per observable."""
    rec, base, _ = zoom(record, live, step_index, sample_index, branch)
    tight = dataclasses.replace(
        live.options, atol=live.options.atol / TIGHTEN_FACTOR, rtol=live.options.rtol / TIGHTEN_FACTOR
    )
    rec, fine, _ = zoom(rec, live, step_index, sample_index, branch, options=tight)
    changes = _changes(base.trace.expectations, fine.trace.expectations)
    worst = max(changes.values()) if changes else 0.0
    return rec, ConvergenceRecord(
        tolerances=(live.options.atol, live.options.rtol),
        tightened_tolerances=(tight.atol, tight.rtol),
        changes=changes,
        tol=CONVERGENCE_TOL,
        converged=worst < CONVERGENCE_TOL,
        max_change=worst,
    )


@dataclass(frozen=True)
class TruncationCheck:
    """Section 9.9's cap arm on one zoomed step: every resolved cap raised by two, re-chained from the initial state."""

    caps: dict[int, int]
    grown_caps: dict[int, int]
    changes: dict[str, float]
    tol: float
    max_change: float
    converged: bool
    engine_calls: int


def truncation_recheck(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, TruncationCheck]:
    rec, base, _ = zoom(record, live, step_index, sample_index, branch)
    space = live.space
    grown = space
    for m in space.resolved:
        grown = grown.grown(m.mode, CAP_RAISE)
    if grown.dimension > live.options.joint_dimension_max:
        raise RecordError(
            f"raising every cap by {CAP_RAISE} takes the joint dimension to {grown.dimension}, above "
            f"joint_dimension_max = {live.options.joint_dimension_max}; the check would need the reduced mode set (Section 9.9)"
        )
    chain = _Chain(rec, live, sample_index, branch, space=grown)
    coarse = engine_for(live)
    while chain.position < step_index:
        chain.play(coarse, live.options)
    tr = chain.play(engine_for(live, store_per_segment=DEFAULT_ZOOM_POINTS), live.options)
    changes = _changes(base.trace.expectations, tr.expectations)
    worst = max(changes.values()) if changes else 0.0
    return rec, TruncationCheck(
        caps={m.mode: m.d for m in space.resolved},
        grown_caps={m.mode: m.d for m in grown.resolved},
        changes=changes,
        tol=CONVERGENCE_TOL,
        max_change=worst,
        converged=worst < CONVERGENCE_TOL,
        engine_calls=chain.engine_calls,
    )


# ---- Level 3 on demand: the Hamiltonian record, the process matrix ---------------------------------------------------


def _segments_of(step: core.GateStep) -> list[tuple[float, float, list[core.Pulse]]]:
    """The step cut at every pulse boundary, the engine's own segmentation (Section 5.2): (start, end, active pulses)."""
    edges = sorted({float(p.t_start_s) for p in step.pulses} | {float(p.t_end_s) for p in step.pulses})
    out: list[tuple[float, float, list[core.Pulse]]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        active = [p for p in step.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
        if active:
            out.append((a, b, active))
    return out


def _nnz(op: qt.Qobj | qt.QobjEvo) -> int:
    """Stored non-zeros of an operator in CSR form (a time-dependent one at t = 0)."""
    q = op(0.0) if isinstance(op, qt.QobjEvo) else op
    return int(q.to("CSR").data.as_scipy().nnz)


def _collapse(c: core.CollapseOp, *, active: bool, note: str) -> CollapseRecord:
    return CollapseRecord(
        channel=str(c.channel),
        rate_hz=float(c.rate_hz),
        ion=None if c.ion is None else int(c.ion),
        mode=None if c.mode is None else int(c.mode),
        time_dependent=bool(c.time_dependent),
        operator_nnz=_nnz(c.op),
        active_in_run=active,
        note=note,
    )


def hamiltonian_key(step_index: int, sample_index: int, branch: int) -> str:
    return f"ham/step{step_index}/s{sample_index}/b{branch}"


def hamiltonian_record(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, HamiltonianRecord]:
    """The terms of H(t) and the collapse operators the engine assembles for one step (Section 5.7), listed with their
    numbers: the builder is called as the engine calls it (same space, sample, frozen Fock states and qubit shifts), on every
    segment of the step for the summary table and on the first for the term list."""
    key = hamiltonian_key(step_index, sample_index, branch)
    cached = record.hamiltonian(key)
    if cached is not None:
        return record, cached
    space = live.space
    device = live.device
    _st, sample = initial_state(record, live, sample_index, branch)
    step = live.steps[step_index]
    frozen_n = {m: int(sample.get(core.key_frozen_n(m), 0.0)) for m in space.frozen if m not in space.dropped}
    shifts = dict(live.core_record.qubit_shifts_hz)
    summaries: list[SegmentSummary] = []
    first: core.BuiltHamiltonian | None = None
    first_pulses: list[core.Pulse] = []
    for a, b, active in _segments_of(step):
        built = core.build_hamiltonian(
            device, active, space, sample=sample, qubit_shifts_hz=shifts, frozen_n=frozen_n
        )
        summaries.append(
            SegmentSummary(
                t_start_s=a,
                t_end_s=b,
                pulses=tuple(str(p.gate_id) for p in active),
                n_drive_terms=int(built.n_drive_terms),
                omega_max_hz=float(built.omega_max_rad_s / TWO_PI),
                kernel=str(built.kernel),
            )
        )
        if first is None:
            first, first_pulses = built, list(active)
    if first is None:  # an idle step: the free Hamiltonian alone
        first = core.build_hamiltonian(
            device, [], space, sample=sample, qubit_shifts_hz=shifts, frozen_n=frozen_n
        )
    pulse_records = {p.gate_id: p for p in record.schedule.pulses if p.gate_id is not None}
    resolved_modes = {t.mode for t in space.resolved}
    drives: list[DriveTermRecord] = []
    for r in first.records:
        pr = pulse_records.get(r.pulse or "")
        core_pulse = next((p for p in first_pulses if p.gate_id == r.pulse), None)
        etas = {int(m): float(e) for m, e in r.etas.items()}
        op = space.drive_operator(int(r.ion), {m: e for m, e in etas.items() if m in resolved_modes})
        tones_mu: tuple[float, ...] = ()
        tones_phi: tuple[float, ...] = ()
        tones_peak: tuple[float, ...] = ()
        if pr is not None:
            tones_mu = tuple(float(t.detuning_hz.at(np.zeros(1), pr.duration_s)[0]) for t in pr.tones)
            tones_phi = tuple(float(t.phase_rad.at(np.zeros(1), pr.duration_s)[0]) for t in pr.tones)
            tones_peak = tuple(
                float(np.max(np.abs(t.envelope_hz.at(np.linspace(0.0, pr.duration_s, 65), pr.duration_s))))
                for t in pr.tones
            )
        drives.append(
            DriveTermRecord(
                pulse=str(r.pulse),
                ion=int(r.ion),
                primary_ion=int(r.primary_ion),
                kind=str(core_pulse.drive.kind) if core_pulse is not None else "",
                beams=tuple(int(b) for b in core_pulse.drive.beams) if core_pulse is not None else (),
                etas=etas,
                frozen_n={int(m): int(n) for m, n in r.frozen_n.items()},
                debye_waller=float(r.debye_waller),
                micromotion_beta=float(r.micromotion_beta),
                carrier_factor=float(r.carrier_factor),
                crosstalk=complex(r.crosstalk),
                rabi_scale=float(r.rabi_scale),
                omega_peak_hz=float(r.omega_peak_rad_s / TWO_PI),
                tone_detunings_hz=tones_mu,
                tone_phases_rad=tones_phi,
                tone_peaks_hz=tones_peak,
                operator_nnz=_nnz(op),
                matrix_elements={
                    m.mode: np.asarray(core.rabi_table(int(m.d), abs(etas.get(m.mode, 0.0))), dtype=float)
                    for m in space.resolved
                },
                frozen_debye_waller={
                    m: float(core.debye_waller_factor(int(frozen_n.get(m, 0)), abs(etas.get(m, 0.0))))
                    for m in space.frozen
                    if m not in space.dropped
                },
            )
        )
    collapse = [
        _collapse(
            c,
            active=True,
            note="device channel of Section 5.7 (heating, motional dephasing, qubit dephasing)",
        )
        for c in device.noise.channels(device, space)
    ]
    scattering_on = bool(live.options.scattering_channels)
    if first_pulses:
        try:
            ops, notes = core.scattering_channels(device, first_pulses[0], space)
        except Exception as exc:  # a drive kind without scattering (microwave) or a level set the model lacks
            ops, notes = (), (f"scattering channels unavailable: {exc}",)
        note = "photon-scattering channel of Section 6.5" + (
            ""
            if scattering_on
            else "; not integrated in this run (SolverOptions.scattering_channels is off: the per-pulse error is estimated instead)"
        )
        collapse += [_collapse(c, active=scattering_on, note=note) for c in ops]
        collapse += [
            CollapseRecord("scattering note", 0.0, None, None, False, 0, scattering_on, str(n)) for n in notes
        ]
    const = next((x for x in first.H.to_list() if not isinstance(x, list)), None)
    stark: dict[int, float] = {}
    for p in first_pulses:
        pr = pulse_records.get(p.gate_id or "")
        if pr is not None:
            for ion in pr.ions:
                stark[int(ion)] = float(pr.stark_shift_hz.at(np.zeros(1), pr.duration_s)[0])
    n_modes = len(device.crystal.modes)
    ham = HamiltonianRecord(
        key=key,
        gate_id=str(step.gate_id),
        t_start_s=float(step.t_start_s),
        t_end_s=float(step.t_end_s),
        frame=str(first.frame),
        dims=tuple(int(d) for d in space.dims),
        dimension=int(space.dimension),
        mode_frequencies_hz={int(m): float(w / TWO_PI) for m, w in first.mode_frequencies_rad_s.items()},
        mode_offsets_hz={m: float(sample.get(core.key_mode_offset_hz(m), 0.0)) for m in range(n_modes)},
        qubit_offsets_hz={
            int(i): float(sample.get(core.key_qubit_offset_hz(int(i)), 0.0)) for i in space.ion_labels
        },
        mode_classes={m: str(space.mode_class(m)) for m in range(n_modes)},
        caps={int(m.mode): int(m.d) for m in space.resolved},
        segments=tuple(summaries),
        drives=tuple(drives),
        collapse=tuple(collapse),
        approximations=tuple(str(a) for a in first.approximations),
        n_drive_terms=int(first.n_drive_terms),
        omega_max_hz=float(first.omega_max_rad_s / TWO_PI),
        kernel=str(first.kernel),
        fingerprint=str(first.fingerprint),
        free_term_nnz=_nnz(const) if const is not None else 0,
        stark_shifts_hz=stark,
    )
    return record.with_hamiltonian(ham), ham


def process_matrix_key(step_index: int, sample_index: int, branch: int) -> str:
    return f"pm/step{step_index}/s{sample_index}/b{branch}"


def process_matrix(
    record: Record,
    live: LiveRun,
    step_index: int,
    sample_index: int = 0,
    branch: int = 0,
    *,
    progress: Progress | None = None,
) -> tuple[Record, ProcessMatrixRecord]:
    """Process tomography of one step from its recorded initial motional state (Section 5.4 (a)), projected onto CP and TP
    and summarized against the step's ideal unitary (Section 6.8)."""
    key = process_matrix_key(step_index, sample_index, branch)
    cached = record.process_matrix(key)
    if cached is not None:
        return record, cached
    t0 = time.perf_counter()
    rec = boundary_states(record, live, sample_index, branch, up_to=step_index)
    start = rec.boundary(step_index, sample_index, branch)
    assert start is not None
    space = live.space
    _st, sample = initial_state(rec, live, sample_index, branch)
    step = live.steps[step_index]
    labels = tuple(int(i) for i in space.ion_labels)
    n = len(labels)
    ideal = np.eye(2**n, dtype=complex)
    for tg in step.targets:
        ideal = embed_on_register(tg.unitary(), [labels.index(int(i)) for i in tg.ions], n) @ ideal
    n_inputs = int(np.prod([d * d for d in space.ion_dims]))
    if progress is not None:
        progress(
            "tomography",
            None,
            f"{n_inputs} input states through the {step.gate_id} pulses at dimension {space.dimension}",
        )
    summary = engine_for(live).process_tomography(
        live.device,
        step.pulses,
        space,
        _state_from_boundary(start, space).motional,
        sample,
        core.SeedSpec(rec.job.seed),
        live.options,
        ideal=ideal,
    )
    choi = np.asarray(summary.choi, dtype=complex)
    pm = ProcessMatrixRecord(
        key=key,
        gate_id=str(step.gate_id),
        ions=labels,
        choi=choi,
        ideal=ideal,
        n_inputs=n_inputs,
        cp_tp_residual=(float(summary.cp_tp_residual[0]), float(summary.cp_tp_residual[1])),
        average_gate_infidelity=float(summary.average_gate_infidelity),
        entanglement_infidelity=float(core.entanglement_infidelity(choi, core.choi_from_unitary(ideal))),
        depolarizing_rate=float(summary.depolarizing_rate),
        pauli_twirled={str(k): float(v) for k, v in summary.pauli_twirled.items()},
        n_traj=int(summary.n_traj),
        wall_time_s=time.perf_counter() - t0,
        method="state-based process tomography from the recorded boundary motional state (Section 5.4 (a))",
    )
    return rec.with_process_matrix(pm), pm
