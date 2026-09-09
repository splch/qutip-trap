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
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.codec import dumps
from qutip_trap_app.record import (
    BoundaryState,
    ConvergenceRecord,
    LiveRun,
    Record,
    RecordError,
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
    return f"step{step_index}/s{sample_index}/b{branch}/n{n_store}/{options_digest(options)}"


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
    """Re-simulate one gate step at fine resolution from its recorded initial state; cached in the record by key."""
    opts = options if options is not None else live.options
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


__all__ = [
    "CONVERGENCE_TOL",
    "DEFAULT_ZOOM_POINTS",
    "TruncationCheck",
    "ZoomStats",
    "boundary_states",
    "engine_for",
    "initial_state",
    "options_digest",
    "sub_schedule",
    "tightened",
    "tolerance_recheck",
    "truncation_recheck",
    "zoom",
    "zoom_key",
]
