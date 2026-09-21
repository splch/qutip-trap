"""On-demand re-simulation of a zoomed pulse, with caching (PLAN.md Sections 14.3, 14.7; Section 9.11 rows
"Re-simulation cache" and "Convergence badge"; milestone M11.1).

Section 14.3: "zooming into a pulse re-simulates that pulse alone from its recorded initial state and its recorded noise
sample". The core's ``Traces`` hold the joint state only at the end of a whole evolution, so the recorded initial state of
a pulse is obtained by chaining the engine over the schedule's gate steps (``gate_steps``) from the run's own initial
state: each step is played as a sub-schedule declared at its own start (``Schedule.t0_s``, the GATE_LOCAL convention of
Section 5.4), and the state at every step boundary is cached in the record (:class:`BoundaryState`). The engine is built
exactly as ``run()`` built its own, from the same table, drives, noise switches and seeds, and the engine itself segments a
schedule at every pulse boundary, so the chained evolution is the run's evolution: the M11.1 prototype found the chained
final state bitwise equal to the recorded one on the two-ion Bell circuit.

A zoom replays one step with a fine store (``n_store`` points per integration segment) and caches the fine trace under a
key of (step, sample, branch, stored points, solver-options digest), so zooming twice recomputes once. Two convergence
re-checks of Section 5.5 / 9.9 run on a zoomed step: tolerances tightened by ten, and every resolved cap raised by two
(the second re-chains from the initial state on the grown space, since a state cannot be regridded through the public API).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.codec import dumps
from qutip_trap_app.record import (
    BoundaryState,
    CollapseRecord,
    ConvergenceRecord,
    DriveTermRecord,
    FockMovie,
    HamiltonianRecord,
    LiveRun,
    ProcessMatrixRecord,
    Record,
    RecordError,
    SegmentSummary,
    ZoomTrace,
    options_record,
    trace_record,
)

DEFAULT_ZOOM_POINTS = 201
"""Stored points per integration segment of a zoomed step: the Level 3 time resolution."""

CONVERGENCE_TOL = 1e-6
"""Section 5.5: the change in a population under a tolerance tightening or a cap raise that still counts as converged."""


# ---- the engine as run() built it ----------------------------------------------------------------------------------------------


def engine_for(record: Record, live: LiveRun, *, store_per_segment: int = 2) -> core.JointExactEngine:
    """The JOINT_EXACT engine with the knobs ``run()`` gave its own (``run.gate_local.EngineSetup.engine``)."""
    if record.job.internal_levels != 2:
        raise RecordError(
            "M11.1 re-simulates two-level registers only (the leakage level maps are not public)"
        )
    return core.JointExactEngine(
        builder_options=None,
        store_per_segment=int(store_per_segment),
        channels=(),
        qubit_shifts_hz=dict(live.core_record.qubit_shifts_hz),
        device_channels=bool(record.job.noise),
        levels_by_ion=None,
        hardware_chain=True,
        table=live.table,
    )


def initial_state(
    record: Record, live: LiveRun, sample_index: int, branch: int, *, space: core.HilbertSpace | None = None
) -> tuple[core.State, core.NoiseSample]:
    """The (sample, branch) initial state and noise sample exactly as ``run()`` prepared them (Section 5.3's Fock sum)."""
    sp = space if space is not None else live.space
    if record.diagnostics.level != "JOINT_EXACT":
        raise RecordError(
            "a GATE_LOCAL run has no joint initial state to re-simulate from (M11.2 replays its channels)"
        )
    state0 = core.prepare(live.device, sp, live.table, preparation=live.core_record.preparation)
    branches = record.branches
    if not branches:
        raise RecordError("the record has no branches")
    br = branches[branch]
    total = sum(b.weight for b in branches)
    fock_res = {m: n for m, n in br.fock.items() if sp.mode_class(m) in ("resolved", "enr")}
    thermal_frozen = {m: float(state0.motional.nbar.get(m, 0.0)) for m in sp.frozen}
    state = sp.initial_state(
        list(br.levels),
        fock=fock_res,
        thermal=thermal_frozen,
        provenance=tuple(state0.provenance) + (f"m6.branch[{branch}]",),
    )
    extra: dict[str, float] = {
        core.key_frozen_n(m): float(n) for m, n in br.fock.items() if sp.mode_class(m) == "frozen"
    }
    extra[core.KEY_BRANCH_WEIGHT] = float(br.weight / total)
    sample = record.noise_samples[sample_index].to_core(extra)
    return state, sample


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
    state: core.State,
    *,
    step_index: int,
    sample_index: int,
    branch: int,
    t_s: float,
    cap: int,
    dims: tuple[int, ...],
) -> BoundaryState:
    joint: np.ndarray | None = None
    if state.joint is not None and state.joint.shape[0] <= cap:
        arr = np.asarray(state.joint.full(), dtype=complex)
        joint = arr.reshape(-1) if state.joint.isket else arr
    return BoundaryState(
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        t_s=float(t_s),
        joint=joint,
        joint_dims=dims,
        internal=np.asarray(state.internal.full(), dtype=complex),
        mode_reduced={int(m): np.asarray(r.full(), dtype=complex) for m, r in state.motional.reduced.items()},
        nbar={int(m): float(v) for m, v in state.motional.nbar.items()},
    )


def _state_from_boundary(b: BoundaryState, space: core.HilbertSpace) -> core.State:
    import qutip as qt

    if b.joint is None:
        raise RecordError(
            "the cached boundary state holds no joint ket (above the store cap); recompute the chain"
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
        self.sample_index = sample_index
        self.branch = branch
        self.space = space if space is not None else live.space
        self.steps = live.steps
        self.state, self.sample = initial_state(record, live, sample_index, branch, space=self.space)
        self.seeds = core.SeedSpec(record.job.seed)
        self.position = 0
        self.engine_calls = 0

    def advance_to(self, step_index: int, engine: core.JointExactEngine, options: core.SolverOptions) -> None:
        while self.position < step_index:
            step = self.steps[self.position]
            tr = engine.run_pulses(
                self.live.device,
                sub_schedule(self.live, step),
                self.state,
                self.space,
                self.sample,
                self.seeds,
                options,
            )
            self.engine_calls += 1
            self.state = tr.final
            self.position += 1

    def play(
        self, step_index: int, engine: core.JointExactEngine, options: core.SolverOptions
    ) -> core.Traces:
        if self.position != step_index:
            raise RecordError(f"chain is at step {self.position}, asked to play step {step_index}")
        tr = engine.run_pulses(
            self.live.device,
            sub_schedule(self.live, self.steps[step_index]),
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
    engine = engine_for(record, live)
    chain = _Chain(record, live, sample_index, branch)
    dims = tuple(int(d) for d in chain.space.dims)
    cap = record.joint_store_dimension_max
    new: list[BoundaryState] = []
    t0 = live.schedule.t0_s if live.schedule.t0_s is not None else record.schedule.t0_s
    new.append(
        _boundary(
            chain.state,
            step_index=0,
            sample_index=sample_index,
            branch=branch,
            t_s=float(t0),
            cap=cap,
            dims=dims,
        )
    )
    for k in range(last):
        chain.advance_to(k + 1, engine, live.options)
        new.append(
            _boundary(
                chain.state,
                step_index=k + 1,
                sample_index=sample_index,
                branch=branch,
                t_s=steps[k].t_end_s,
                cap=cap,
                dims=dims,
            )
        )
    return record.with_boundaries(new)


def options_digest(options: core.SolverOptions) -> str:
    return hashlib.sha256(dumps(options_record(options))).hexdigest()[:16]


def zoom_key(
    step_index: int, sample_index: int, branch: int, n_store: int, options: core.SolverOptions
) -> str:
    """The cache key of a zoom made with ``options``. A zoom always stores the Fock marginals (0.4.0), so the key is taken
    over the options WITH ``store_marginals`` on, whatever the caller passes: the run's own options name the same zoom as
    the options the zoom actually ran with (Level 3 looks its zoom up with the record's options; until this was folded in
    here the two digests never agreed and the fine zoom was computed but never found)."""
    opts = dataclasses.replace(options, store_marginals=True)
    return f"step{step_index}/s{sample_index}/b{branch}/n{n_store}/{options_digest(opts)}"


@dataclass
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
    force: bool = False,
) -> tuple[Record, ZoomTrace, ZoomStats]:
    """Re-simulate one gate step at fine resolution from its recorded initial state; cached in the record by key. The zoom
    stores the per-time Fock populations of every resolved mode (``core.Traces.mode_marginal``; 0.4.0), so the Fock movie
    of the step is read off it rather than re-simulated."""
    opts = options if options is not None else live.options
    opts = dataclasses.replace(opts, store_marginals=True)
    key = zoom_key(step_index, sample_index, branch, n_store, opts)
    cached = record.zoom(key)
    if cached is not None and not force:
        return record, cached, ZoomStats(engine_calls=0, wall_time_s=0.0, cached=True)
    t0 = time.perf_counter()
    rec = boundary_states(record, live, sample_index, branch, up_to=step_index)
    start = rec.boundary(step_index, sample_index, branch)
    assert start is not None
    space = live.space
    engine = engine_for(rec, live, store_per_segment=n_store)
    state = _state_from_boundary(start, space)
    _st, sample = initial_state(rec, live, sample_index, branch)
    step = live.steps[step_index]
    tr = engine.run_pulses(
        live.device, sub_schedule(live, step), state, space, sample, core.SeedSpec(rec.job.seed), opts
    )
    report = engine.last_report
    wall = time.perf_counter() - t0
    trace = trace_record(
        tr,
        sample_index=sample_index,
        sample_id=rec.noise_samples[sample_index].sample_id,
        branch=branch,
        weight=rec.branches[branch].weight,
        ion_dims=space.ion_dims,
        joint_dims=space.dims,
        joint_cap=rec.joint_store_dimension_max,
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
        integrators=tuple(dict.fromkeys(s.integrator for s in report.segments)) if report is not None else (),
        method=str(report.method) if report is not None else "",
        approximations=tuple(report.approximations) if report is not None else (),
        wall_time_s=wall,
        engine_dimension=int(space.dimension),
    )
    return rec.with_zoom(z), z, ZoomStats(engine_calls=1, wall_time_s=wall, cached=False)


# ---- convergence re-checks (Section 5.5 / 9.9) ---------------------------------------------------------------------------------------


def _observables(trace_a: Mapping[str, np.ndarray], trace_b: Mapping[str, np.ndarray]) -> dict[str, float]:
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


def tightened(options: core.SolverOptions, factor: float = 10.0) -> core.SolverOptions:
    return dataclasses.replace(options, atol=options.atol / factor, rtol=options.rtol / factor)


def tolerance_recheck(
    record: Record,
    live: LiveRun,
    step_index: int,
    sample_index: int = 0,
    branch: int = 0,
    *,
    n_store: int = DEFAULT_ZOOM_POINTS,
    factor: float = 10.0,
    tol: float = CONVERGENCE_TOL,
) -> tuple[Record, ConvergenceRecord]:
    """Section 5.5's tolerance arm on one zoomed step: atol and rtol tightened by ``factor``, the change per observable."""
    rec, base, _ = zoom(record, live, step_index, sample_index, branch, n_store=n_store)
    tight = tightened(live.options, factor)
    rec, fine, _ = zoom(rec, live, step_index, sample_index, branch, n_store=n_store, options=tight)
    changes = _observables(base.trace.expectations, fine.trace.expectations)
    worst = max(changes.values()) if changes else 0.0
    return rec, ConvergenceRecord(
        tolerances=(live.options.atol, live.options.rtol),
        tightened_tolerances=(tight.atol, tight.rtol),
        changes=changes,
        tol=tol,
        converged=worst < tol,
        max_change=worst,
    )


@dataclass(frozen=True)
class TruncationCheck:
    """Section 9.9's cap arm on one zoomed step: every resolved cap raised by ``add``, re-chained from the initial state."""

    caps: dict[int, int]
    grown_caps: dict[int, int]
    changes: dict[str, float]
    tol: float
    max_change: float
    converged: bool
    engine_calls: int


def truncation_recheck(
    record: Record,
    live: LiveRun,
    step_index: int,
    sample_index: int = 0,
    branch: int = 0,
    *,
    n_store: int = DEFAULT_ZOOM_POINTS,
    add: int = 2,
    tol: float = CONVERGENCE_TOL,
) -> tuple[Record, TruncationCheck]:
    rec, base, _ = zoom(record, live, step_index, sample_index, branch, n_store=n_store)
    space = live.space
    grown = space
    for m in space.resolved:
        grown = grown.grown(m.mode, add)
    if grown.dimension > live.options.joint_dimension_max:
        raise RecordError(
            f"raising every cap by {add} takes the joint dimension to {grown.dimension}, above joint_dimension_max = "
            f"{live.options.joint_dimension_max}; the check would need the reduced mode set (Section 9.9)"
        )
    chain = _Chain(rec, live, sample_index, branch, space=grown)
    coarse = engine_for(rec, live)
    chain.advance_to(step_index, coarse, live.options)
    fine = engine_for(rec, live, store_per_segment=n_store)
    tr = chain.play(step_index, fine, live.options)
    changes = _observables(base.trace.expectations, {k: v for k, v in tr.expectations.items()})
    worst = max(changes.values()) if changes else 0.0
    return rec, TruncationCheck(
        caps={m.mode: m.d for m in space.resolved},
        grown_caps={m.mode: m.d for m in grown.resolved},
        changes=changes,
        tol=tol,
        max_change=worst,
        converged=worst < tol,
        engine_calls=chain.engine_calls,
    )


# ---- Level 3 on demand: Fock movies, the process matrix, the Hamiltonian record (Section 14.2 rows 3 and 4; M11.3) ------------

DEFAULT_FOCK_FRAMES = 8
"""Frames of a Fock movie after the step's start: the truncation points t_k = t_start + k/K (t_end - t_start)."""

TWO_PI = 2.0 * math.pi


def fock_movie_key(
    step_index: int, sample_index: int, branch: int, n_frames: int, options: core.SolverOptions
) -> str:
    return f"fock/step{step_index}/s{sample_index}/b{branch}/k{n_frames}/{options_digest(options)}"


def fock_movie(
    record: Record,
    live: LiveRun,
    step_index: int,
    sample_index: int = 0,
    branch: int = 0,
    *,
    n_frames: int = DEFAULT_FOCK_FRAMES,
    options: core.SolverOptions | None = None,
    progress: Callable[[str, float | None, str], None] | None = None,
) -> tuple[Record, FockMovie]:
    """Per-time Fock distributions of every resolved mode inside one step, read off the step's zoom (0.4.0), cached by key.

    The zoom's fine trace stores the Fock populations at every one of its points (``core.Traces.mode_marginal``, which the
    core added in 0.4.0 for exactly this view), so the movie is K + 1 of those rows at the frame times: frame 0 the step's
    start, frame K its end. Until 0.4.0 the core's traces carried <n_m>(t) only and the movie re-simulated K truncated
    copies of the pulse (``core.CORE_GAPS`` recorded the gap); the zoom is computed once when it is not cached yet.
    """
    if n_frames < 1:
        raise ValueError("a Fock movie has at least one frame after the start")
    opts = options if options is not None else live.options
    key = fock_movie_key(step_index, sample_index, branch, n_frames, opts)
    cached = record.fock_movie(key)
    if cached is not None:
        return record, cached
    t0 = time.perf_counter()
    if progress is not None:
        progress("fock movie", 0.0, "the step's zoom, whose stored Fock populations are the frames")
    rec, z, stats = zoom(record, live, step_index, sample_index, branch, options=opts)
    tr = z.trace
    if tr.mode_marginal is None:
        raise RecordError(
            "the zoom carries no Fock populations: re-run the zoom (its options store them since 0.4.0)"
        )
    n_points = int(tr.times_s.size)
    frames = [round(k * (n_points - 1) / n_frames) for k in range(n_frames + 1)]
    movie = FockMovie(
        key=key,
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        options_digest=options_digest(opts),
        times_s=np.asarray([float(tr.times_s[i]) for i in frames], dtype=float),
        distributions={int(m): np.asarray(dist[frames], dtype=float) for m, dist in tr.mode_marginal.items()},
        nbar={int(m): np.asarray(tr.mode_nbar[m][frames], dtype=float) for m in tr.mode_marginal},
        engine_calls=stats.engine_calls,
        wall_time_s=time.perf_counter() - t0,
        method="read off the step's zoom: the engine stored the Fock populations at every point (Traces.mode_marginal, 0.4.0)",
    )
    return rec.with_fock_movie(movie), movie


def _segments_of(step: core.GateStep) -> list[tuple[float, float, list[core.Pulse]]]:
    """The step cut at every pulse boundary, the engine's own segmentation (Section 5.2): (start, end, active pulses)."""
    edges = sorted({float(p.t_start_s) for p in step.pulses} | {float(p.t_end_s) for p in step.pulses})
    out: list[tuple[float, float, list[core.Pulse]]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        active = [p for p in step.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
        if active:
            out.append((a, b, active))
    return out


def _nnz(op: object) -> int:
    obj = op(0.0) if callable(op) and not hasattr(op, "to") else op
    try:
        return int(obj.to("CSR").data.as_scipy().nnz)  # type: ignore[union-attr]
    except Exception:  # a Dense operator or another data layer: count the non-zeros of the full matrix
        arr = np.asarray(obj.full())  # type: ignore[union-attr]
        return int(np.count_nonzero(arr))


def hamiltonian_key(step_index: int, sample_index: int, branch: int) -> str:
    return f"ham/step{step_index}/s{sample_index}/b{branch}"


def hamiltonian_record(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, HamiltonianRecord]:
    """The terms of H(t) and the collapse operators the engine assembled for one step (Section 5.7), listed with their numbers:
    the builder is called as the engine calls it (same space, sample, frozen Fock states and qubit shifts), on every segment of
    the step for the summary table and on the first for the term list."""
    key = hamiltonian_key(step_index, sample_index, branch)
    cached = record.hamiltonian(key)
    if cached is not None:
        return record, cached
    t0 = time.perf_counter()
    space = live.space
    device = live.device
    _st, sample = initial_state(record, live, sample_index, branch)
    step = live.steps[step_index]
    frozen_n = {m: int(sample.get(core.key_frozen_n(m), 0.0)) for m in space.frozen if m not in space.dropped}
    segments = _segments_of(step)
    summaries: list[SegmentSummary] = []
    first: core.BuiltHamiltonian | None = None
    first_pulses: list[core.Pulse] = []
    for a, b, active in segments:
        built = core.build_hamiltonian(
            device,
            active,
            space,
            sample=sample,
            qubit_shifts_hz=dict(live.core_record.qubit_shifts_hz),
            frozen_n=frozen_n,
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
    if first is None:
        # an idle step: the free Hamiltonian alone
        first = core.build_hamiltonian(
            device,
            [],
            space,
            sample=sample,
            qubit_shifts_hz=dict(live.core_record.qubit_shifts_hz),
            frozen_n=frozen_n,
        )
    pulse_records = {p.gate_id: p for p in record.schedule.pulses if p.gate_id is not None}
    drives: list[DriveTermRecord] = []
    for r in first.records:
        pr = pulse_records.get(r.pulse or "")
        core_pulse = next((p for p in first_pulses if p.gate_id == r.pulse), None)
        etas = {int(m): float(e) for m, e in r.etas.items()}
        tables = {
            m.mode: np.asarray(core.rabi_table(int(m.d), abs(etas.get(m.mode, 0.0))), dtype=float)
            for m in space.resolved
        }
        dw = {
            m: float(core.debye_waller_factor(int(frozen_n.get(m, 0)), abs(etas.get(m, 0.0))))
            for m in space.frozen
            if m not in space.dropped
        }
        resolved_modes = {t.mode for t in space.resolved}
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
                matrix_elements=tables,
                frozen_debye_waller=dw,
            )
        )
    collapse: list[CollapseRecord] = []
    noise_active = bool(record.job.noise)
    for c in device.noise.channels(device, space):
        collapse.append(
            CollapseRecord(
                channel=str(c.channel),
                rate_hz=float(c.rate_hz),
                ion=None if c.ion is None else int(c.ion),
                mode=None if c.mode is None else int(c.mode),
                time_dependent=bool(c.time_dependent),
                operator_nnz=_nnz(c.op),
                active_in_run=noise_active,
                note="device channel of Section 5.7 (heating, motional dephasing, qubit dephasing)"
                + ("" if noise_active else "; the run was made with noise=False, so it was not integrated"),
            )
        )
    scattering_on = bool(getattr(live.options, "scattering_channels", False))
    if first_pulses:
        try:
            ops, notes = core.scattering_channels(device, first_pulses[0], space)
        except Exception as exc:  # a drive kind without scattering (microwave) or a level set the model lacks
            ops, notes = (), (f"scattering channels unavailable: {exc}",)
        for c in ops:
            collapse.append(
                CollapseRecord(
                    channel=str(c.channel),
                    rate_hz=float(c.rate_hz),
                    ion=None if c.ion is None else int(c.ion),
                    mode=None if c.mode is None else int(c.mode),
                    time_dependent=bool(c.time_dependent),
                    operator_nnz=_nnz(c.op),
                    active_in_run=scattering_on,
                    note="photon-scattering channel of Section 6.5"
                    + (
                        ""
                        if scattering_on
                        else "; not integrated in this run (SolverOptions.scattering_channels is off: the per-pulse error is estimated instead)"
                    ),
                )
            )
        for n in notes:
            collapse.append(
                CollapseRecord("scattering note", 0.0, None, None, False, 0, scattering_on, str(n))
            )
    const = next((x for x in first.H.to_list() if not isinstance(x, list)), None)
    stark: dict[int, float] = {}
    for p in first_pulses:
        pr = pulse_records.get(p.gate_id or "")
        if pr is not None:
            for ion in pr.ions:
                stark[int(ion)] = float(pr.stark_shift_hz.at(np.zeros(1), pr.duration_s)[0])
    ham = HamiltonianRecord(
        key=key,
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        gate_id=str(step.gate_id),
        t_start_s=float(step.t_start_s),
        t_end_s=float(step.t_end_s),
        frame=str(first.frame),
        dims=tuple(int(d) for d in space.dims),
        dimension=int(space.dimension),
        ion_labels=tuple(int(i) for i in space.ion_labels),
        mode_frequencies_hz={int(m): float(w / TWO_PI) for m, w in first.mode_frequencies_rad_s.items()},
        mode_offsets_hz={
            m: float(sample.get(core.key_mode_offset_hz(m), 0.0)) for m in range(len(device.crystal.modes))
        },
        qubit_offsets_hz={
            int(i): float(sample.get(core.key_qubit_offset_hz(int(i)), 0.0)) for i in space.ion_labels
        },
        mode_classes={m: str(space.mode_class(m)) for m in range(len(device.crystal.modes))},
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
        wall_time_s=time.perf_counter() - t0,
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
    progress: Callable[[str, float | None, str], None] | None = None,
) -> tuple[Record, ProcessMatrixRecord]:
    """Process tomography of one step from its recorded initial motional state (Section 5.4 (a)): the step's channel on the
    register's product inputs (from the propagated internal basis when the step is unitary, every input propagated and the Choi
    matrix fit by least squares otherwise; ``TomographyRecord.route``), projected onto CP and TP, summarized against the step's
    ideal unitary (Section 6.8)."""
    from qutip_trap_app.viewmodel.circuit import embed_operator

    key = process_matrix_key(step_index, sample_index, branch)
    cached = record.process_matrix(key)
    if cached is not None:
        return record, cached
    t0 = time.perf_counter()
    rec = boundary_states(record, live, sample_index, branch, up_to=step_index)
    start = rec.boundary(step_index, sample_index, branch)
    assert start is not None
    space = live.space
    state = _state_from_boundary(start, space)
    _st, sample = initial_state(rec, live, sample_index, branch)
    step = live.steps[step_index]
    labels = tuple(int(i) for i in space.ion_labels)
    n = len(labels)
    ideal = np.eye(2**n, dtype=complex)
    for tg in step.targets:
        pos = tuple(labels.index(int(i)) for i in tg.ions)
        ideal = embed_operator(np.asarray(tg.unitary(), dtype=complex), pos, n) @ ideal
    n_inputs = int(np.prod([d * d for d in space.ion_dims]))
    if progress is not None:
        progress(
            "tomography",
            None,
            f"{n_inputs} input states through the {step.gate_id} pulses at dimension {space.dimension}",
        )
    engine = engine_for(rec, live)
    summary = engine.process_tomography(
        live.device,
        step.pulses,
        space,
        state.motional,
        sample,
        core.SeedSpec(rec.job.seed),
        live.options,
        ideal=ideal,
    )
    choi = np.asarray(summary.choi, dtype=complex)
    pm = ProcessMatrixRecord(
        key=key,
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
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


__all__ = [
    "CONVERGENCE_TOL",
    "DEFAULT_FOCK_FRAMES",
    "DEFAULT_ZOOM_POINTS",
    "TruncationCheck",
    "ZoomStats",
    "boundary_states",
    "engine_for",
    "fock_movie",
    "fock_movie_key",
    "hamiltonian_key",
    "hamiltonian_record",
    "initial_state",
    "options_digest",
    "process_matrix",
    "process_matrix_key",
    "sub_schedule",
    "tightened",
    "tolerance_recheck",
    "truncation_recheck",
    "zoom",
    "zoom_key",
]
