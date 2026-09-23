"""Re-simulation of one gate step from its recorded initial state, cached on the record: the Level 3 zoom, the
convergence re-checks, the terms of H(t) of a step and its process matrix."""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping

import numpy as np
import qutip as qt

from qutip_trap_app import core
from qutip_trap_app.record import (
    JOINT_STORE_DIMENSION_MAX,
    BoundaryState,
    CollapseTerm,
    DriveTerm,
    HamiltonianRecord,
    LiveRun,
    ProcessMatrixRecord,
    Recheck,
    Record,
    SegmentSummary,
    ZoomTrace,
    trace_record,
)
from qutip_trap_app.viewmodel.circuit import embed_operator

DEFAULT_ZOOM_POINTS = 201
"""Stored points per integration segment of a zoomed step."""
CONVERGENCE_TOL = 1e-6
"""The change of a stored observable under a re-check that still counts as converged."""
CAP_RAISE = 2
"""Fock levels the truncation re-check adds to every resolved mode."""
TWO_PI = 2.0 * math.pi


class ResimError(ValueError):
    """The run cannot be re-simulated as asked."""


# ---- the run's engine, states and samples ------------------------------------------------------------------------------------


def _require_joint(live: LiveRun) -> None:
    if live.result.diagnostics.level != "JOINT_EXACT":
        raise ResimError("a GATE_LOCAL run has no joint state to re-simulate from")


def engine_for(live: LiveRun, store_per_segment: int = 2) -> core.JointExactEngine:
    """The run's own JOINT_EXACT engine (``Machine.engine`` with the run's qubit-frequency shifts)."""
    return dataclasses.replace(
        live.machine.engine,
        store_per_segment=store_per_segment,
        qubit_shifts_hz=dict(live.run.qubit_shifts_hz),
    )


def tightened(options: core.SolverOptions, factor: float = 10.0) -> core.SolverOptions:
    return dataclasses.replace(options, atol=options.atol / factor, rtol=options.rtol / factor)


def branch_sample(
    live: LiveRun, sample_index: int, branch: int, space: core.HilbertSpace
) -> core.NoiseSample:
    """The run's noise sample with the branch's frozen Fock states and weight, as the run gave it to the engine."""
    branches = live.run.branches
    br = branches[branch]
    smp = live.result.noise_samples[sample_index]
    values = dict(smp.values)
    values.update(
        {core.key_frozen_n(m): float(n) for m, n in br.fock.items() if space.mode_class(m) == "frozen"}
    )
    values[core.KEY_BRANCH_WEIGHT] = float(br.weight / sum(b.weight for b in branches))
    return core.NoiseSample(sample_id=smp.sample_id, values=values, ou_grids=dict(smp.ou_grids), t_s=smp.t_s)


def initial_state(live: LiveRun, branch: int, space: core.HilbertSpace) -> core.State:
    """The branch's initial joint state as the run prepared it (one term of the Fock sum of the initial mixture)."""
    _require_joint(live)
    state0 = core.prepare(live.machine.device, space, live.machine.table, preparation=live.run.preparation)
    br = live.run.branches[branch]
    return space.initial_state(
        list(br.levels),
        fock={m: n for m, n in br.fock.items() if space.mode_class(m) in ("resolved", "enr")},
        thermal={m: float(state0.motional.nbar.get(m, 0.0)) for m in space.frozen},
        provenance=(*state0.provenance, f"m6.branch[{branch}]"),
    )


def sub_schedule(schedule: core.Schedule, step: core.GateStep) -> core.Schedule:
    """One gate step as a schedule declared at its own start."""
    idle = tuple(
        iv for iv in schedule.idle if iv[0] >= step.t_start_s - 1e-15 and iv[1] <= step.t_end_s + 1e-15
    )
    return core.Schedule(
        pulses=step.pulses,
        idle=idle,
        events=(),
        phase_frame=dict(schedule.phase_frame),
        gates=step.played,
        targets=step.targets,
        t0_s=step.t_start_s,
    )


def _play(
    live: LiveRun,
    engine: core.JointExactEngine,
    step_index: int,
    state: core.State,
    space: core.HilbertSpace,
    sample: core.NoiseSample,
    options: core.SolverOptions,
) -> core.Traces:
    return engine.run_pulses(
        live.machine.device,
        sub_schedule(live.run.schedule, live.steps[step_index]),
        state,
        space,
        sample,
        core.SeedSpec(live.result.diagnostics.root_seed),
        options,
    )


def _fock(model: core.MotionalModel) -> dict[int, np.ndarray]:
    return {int(m): np.real(np.diag(r.full())).astype(float) for m, r in model.reduced.items()}


# ---- boundary states ---------------------------------------------------------------------------------------------------------------


def boundary_key(step_index: int, sample_index: int, branch: int) -> str:
    return f"boundary/step{step_index}/s{sample_index}/b{branch}"


def _boundary(
    key: str, state: core.State, step_index: int, sample_index: int, branch: int, dims: tuple[int, ...]
) -> BoundaryState:
    joint: np.ndarray | None = None
    if state.joint is not None and state.joint.shape[0] <= JOINT_STORE_DIMENSION_MAX:
        arr = np.asarray(state.joint.full(), dtype=complex)
        joint = arr.reshape(-1) if state.joint.isket else arr
    return BoundaryState(
        key=key,
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        joint=joint,
        joint_dims=dims,
        internal=np.asarray(state.internal.full(), dtype=complex),
        fock=_fock(state.motional),
    )


def start_state(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, core.State]:
    """The joint state at the start of step ``step_index`` (``len(steps)`` is the end of the schedule), chained from the
    latest state already reached; every boundary on the way is kept on the live run and cached on the record."""
    # the run's own engine chained over the steps before this one, each played as a schedule declared at its own start: the
    # engine cuts a schedule at every pulse boundary, so the chained evolution is the run's
    steps = live.steps
    if not 0 <= step_index <= len(steps):
        raise IndexError(f"step {step_index} outside 0..{len(steps)}")
    space = live.space
    keys = [boundary_key(k, sample_index, branch) for k in range(step_index + 1)]
    start = max((k for k, key in enumerate(keys) if key in live.states), default=-1)
    if start < 0:
        live.states[keys[0]] = initial_state(live, branch, space)
        start = 0
    state = live.states[keys[start]]
    if start < step_index:
        engine = engine_for(live)
        sample = branch_sample(live, sample_index, branch, space)
        for k in range(start, step_index):
            state = _play(live, engine, k, state, space, sample, live.options).final
            live.states[keys[k + 1]] = state
    dims = tuple(int(d) for d in space.dims)
    new = [
        _boundary(key, live.states[key], k, sample_index, branch, dims)
        for k, key in enumerate(keys)
        if key not in record.cache
    ]
    return record.with_cached(*new), state


# ---- the zoom and its re-checks -------------------------------------------------------------------------------------------------


def zoom_key(step_index: int, sample_index: int, branch: int, n_store: int, *, tight: bool = False) -> str:
    return f"zoom/step{step_index}/s{sample_index}/b{branch}/n{n_store}" + ("/tight" if tight else "")


def zoom(
    record: Record,
    live: LiveRun,
    step_index: int,
    sample_index: int = 0,
    branch: int = 0,
    *,
    n_store: int = DEFAULT_ZOOM_POINTS,
    tight: bool = False,
    force: bool = False,
) -> tuple[Record, ZoomTrace, bool]:
    """Re-simulate one step at fine resolution from its recorded initial state, storing the Fock populations of every
    carried mode at every point; ``tight`` tightens the tolerances by ten. The flag says whether the zoom was cached."""
    key = zoom_key(step_index, sample_index, branch, n_store, tight=tight)
    cached = record.cached(key, ZoomTrace)
    if cached is not None and not force:
        return record, cached, True
    t0 = time.perf_counter()
    record, state = start_state(record, live, step_index, sample_index, branch)
    space = live.space
    options = dataclasses.replace(tightened(live.options) if tight else live.options, store_marginals=True)
    engine = engine_for(live, n_store)
    tr = _play(
        live, engine, step_index, state, space, branch_sample(live, sample_index, branch, space), options
    )
    report = engine.last_report
    z = ZoomTrace(
        key=key,
        step_index=step_index,
        sample_index=sample_index,
        branch=branch,
        n_store=n_store,
        trace=trace_record(
            tr, sample_index=sample_index, branch=branch, weight=record.branches[branch].weight
        ),
        fock_start=_fock(state.motional),
        fock_end=_fock(tr.final.motional),
        integrators=() if report is None else tuple(dict.fromkeys(s.integrator for s in report.segments)),
        method="" if report is None else str(report.method),
        wall_time_s=time.perf_counter() - t0,
        engine_dimension=int(space.dimension),
    )
    return record.with_cached(z), z, False


def recheck_key(what: str, step_index: int, sample_index: int, branch: int) -> str:
    return f"recheck/{what}/step{step_index}/s{sample_index}/b{branch}"


def _changes(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, float]:
    """The largest difference of every stored observable between two traces on the same time grid."""
    if set(a) != set(b):
        raise ResimError("the two traces store different observables")
    out: dict[str, float] = {}
    for k in sorted(a):
        x, y = np.real(np.asarray(a[k])), np.real(np.asarray(b[k]))
        if x.shape != y.shape:
            raise ResimError(f"observable {k!r}: {x.shape} against {y.shape} stored points")
        out[k] = float(np.max(np.abs(x - y))) if x.size else 0.0
    return out


def tolerance_recheck(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, Recheck]:
    """The zoomed step again with the tolerances tightened by ten."""
    record, base, _ = zoom(record, live, step_index, sample_index, branch)
    record, fine, _ = zoom(record, live, step_index, sample_index, branch, tight=True)
    opts = live.options
    tight = tightened(opts)
    check = Recheck(
        key=recheck_key("tolerance", step_index, sample_index, branch),
        what="tolerances tightened by ten",
        before={"atol": opts.atol, "rtol": opts.rtol},
        after={"atol": tight.atol, "rtol": tight.rtol},
        changes=_changes(base.trace.expectations, fine.trace.expectations),
        tol=CONVERGENCE_TOL,
    )
    return record.with_cached(check), check


def truncation_recheck(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, Recheck]:
    """The zoomed step again with every resolved cap raised by ``CAP_RAISE``, chained from the initial state on the
    grown space (a state cannot be carried onto a larger space)."""
    record, base, _ = zoom(record, live, step_index, sample_index, branch)
    space = live.space
    grown = space
    for m in space.resolved:
        grown = grown.grown(m.mode, CAP_RAISE)
    if grown.dimension > live.options.joint_dimension_max:
        raise ResimError(
            f"raising every cap by {CAP_RAISE} takes the joint dimension to {grown.dimension}, above "
            f"joint_dimension_max = {live.options.joint_dimension_max}"
        )
    sample = branch_sample(live, sample_index, branch, grown)
    state = initial_state(live, branch, grown)
    coarse = engine_for(live)
    for k in range(step_index):
        state = _play(live, coarse, k, state, grown, sample, live.options).final
    tr = _play(live, engine_for(live, DEFAULT_ZOOM_POINTS), step_index, state, grown, sample, live.options)
    check = Recheck(
        key=recheck_key("truncation", step_index, sample_index, branch),
        what=f"every cap raised by {CAP_RAISE}",
        before={f"cap {m.mode}": float(m.d) for m in space.resolved},
        after={f"cap {m.mode}": float(m.d) for m in grown.resolved},
        changes=_changes(base.trace.expectations, tr.expectations),
        tol=CONVERGENCE_TOL,
    )
    return record.with_cached(check), check


# ---- the Hamiltonian of a step -------------------------------------------------------------------------------------------------------


def hamiltonian_key(step_index: int, sample_index: int, branch: int) -> str:
    return f"hamiltonian/step{step_index}/s{sample_index}/b{branch}"


def _segments_of(step: core.GateStep) -> list[tuple[float, float, list[core.Pulse]]]:
    """The step cut at every pulse boundary, as the engine integrates it: (start, end, the pulses playing)."""
    edges = sorted({float(p.t_start_s) for p in step.pulses} | {float(p.t_end_s) for p in step.pulses})
    out: list[tuple[float, float, list[core.Pulse]]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        active = [p for p in step.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
        if active:
            out.append((a, b, active))
    return out


def _nnz(op: qt.Qobj) -> int:
    return int(op.to("CSR").data.as_scipy().nnz)


def hamiltonian_record(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, HamiltonianRecord]:
    """The terms of H(t) and the collapse operators the engine assembles for one step: the builder called as the engine
    calls it, on every segment of the step for the summary and on the first for the terms."""
    key = hamiltonian_key(step_index, sample_index, branch)
    cached = record.cached(key, HamiltonianRecord)
    if cached is not None:
        return record, cached
    _require_joint(live)
    space, device, physics = live.space, live.machine.device, live.machine.physics
    sample = branch_sample(live, sample_index, branch, space)
    step = live.steps[step_index]
    frozen_n = {m: int(sample.get(core.key_frozen_n(m), 0.0)) for m in space.frozen if m not in space.dropped}

    def build(pulses: list[core.Pulse]) -> core.BuiltHamiltonian:
        return core.build_hamiltonian(
            device,
            pulses,
            space,
            sample=sample,
            options=physics.builder,
            qubit_shifts_hz=dict(live.run.qubit_shifts_hz),
            frozen_n=frozen_n,
        )

    segments = [(a, b, active, build(active)) for a, b, active in _segments_of(step)]
    first, first_pulses = (segments[0][3], segments[0][2]) if segments else (build([]), [])
    pulse_records = {p.gate_id: p for p in record.schedule.pulses if p.gate_id is not None}
    caps = {t.mode: t.d for t in space.resolved}
    drives: list[DriveTerm] = []
    for r in first.records:
        etas = {int(m): float(e) for m, e in r.etas.items()}
        pr = pulse_records.get(r.pulse or "")
        drives.append(
            DriveTerm(
                pulse=str(r.pulse),
                ion=int(r.ion),
                primary_ion=int(r.primary_ion),
                etas=etas,
                frozen_n={int(m): int(n) for m, n in r.frozen_n.items()},
                debye_waller=float(r.debye_waller),
                carrier_factor=float(r.carrier_factor),
                crosstalk=complex(r.crosstalk),
                rabi_scale=float(r.rabi_scale),
                omega_peak_hz=float(r.omega_peak_rad_s / TWO_PI),
                tones=()
                if pr is None
                else tuple(
                    (t.detuning_hz, t.phase_rad, float(np.max(np.abs(t.envelope_hz)))) for t in pr.tones
                ),
                operator_nnz=_nnz(
                    space.drive_operator(int(r.ion), {m: e for m, e in etas.items() if m in caps})
                ),
                matrix_elements={
                    m: np.asarray(core.rabi_table(d, abs(etas.get(m, 0.0))), dtype=float)
                    for m, d in caps.items()
                },
                frozen_debye_waller={
                    m: core.debye_waller_factor(n, abs(etas.get(m, 0.0))) for m, n in frozen_n.items()
                },
            )
        )
    noise = bool(physics.noise)
    collapse = [
        CollapseTerm(
            str(c.channel),
            float(c.rate_hz),
            c.ion,
            c.mode,
            noise,
            "device noise" + ("" if noise else "; this run was made without noise"),
        )
        for c in device.noise.channels(device, space)
    ]
    if first_pulses:
        scattering = physics.scattering == "channels"
        ops, _notes = core.scattering_channels(device, first_pulses[0], space)
        collapse += [
            CollapseTerm(
                str(c.channel),
                float(c.rate_hz),
                c.ion,
                c.mode,
                scattering,
                "photon scattering" + ("" if scattering else "; this run estimated it instead"),
            )
            for c in ops
        ]
    stark: dict[int, float] = {}
    for p in first_pulses:
        pr = pulse_records.get(p.gate_id or "")
        if pr is not None:
            stark.update({int(i): pr.stark_shift_hz for i in pr.ions})
    n_modes = len(device.crystal.modes)
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
        mode_frequencies_hz={int(m): float(w / TWO_PI) for m, w in first.mode_frequencies_rad_s.items()},
        mode_offsets_hz={m: float(sample.get(core.key_mode_offset_hz(m), 0.0)) for m in range(n_modes)},
        qubit_offsets_hz={
            int(i): float(sample.get(core.key_qubit_offset_hz(int(i)), 0.0)) for i in space.ion_labels
        },
        mode_classes={m: str(space.mode_class(m)) for m in range(n_modes)},
        caps=caps,
        segments=tuple(
            SegmentSummary(
                t_start_s=a,
                t_end_s=b,
                pulses=tuple(str(p.gate_id) for p in active),
                n_drive_terms=int(built.n_drive_terms),
                omega_max_hz=float(built.omega_max_rad_s / TWO_PI),
                kernel=str(built.kernel),
            )
            for a, b, active, built in segments
        ),
        drives=tuple(drives),
        collapse=tuple(collapse),
        approximations=tuple(str(a) for a in first.approximations),
        n_drive_terms=int(first.n_drive_terms),
        omega_max_hz=float(first.omega_max_rad_s / TWO_PI),
        kernel=str(first.kernel),
        stark_shifts_hz=stark,
    )
    return record.with_cached(ham), ham


# ---- the process matrix of a step ---------------------------------------------------------------------------------------------------


def process_matrix_key(step_index: int, sample_index: int, branch: int) -> str:
    return f"process/step{step_index}/s{sample_index}/b{branch}"


def process_matrix(
    record: Record, live: LiveRun, step_index: int, sample_index: int = 0, branch: int = 0
) -> tuple[Record, ProcessMatrixRecord]:
    """Process tomography of one step from its recorded initial motional state, projected onto the physical channels and
    summarized against the step's ideal unitary."""
    key = process_matrix_key(step_index, sample_index, branch)
    cached = record.cached(key, ProcessMatrixRecord)
    if cached is not None:
        return record, cached
    t0 = time.perf_counter()
    record, state = start_state(record, live, step_index, sample_index, branch)
    space = live.space
    step = live.steps[step_index]
    labels = tuple(int(i) for i in space.ion_labels)
    n = len(labels)
    ideal = np.eye(2**n, dtype=complex)
    for tg in step.targets:
        positions = tuple(labels.index(int(i)) for i in tg.ions)
        ideal = embed_operator(np.asarray(tg.unitary(), dtype=complex), positions, n) @ ideal
    summary = engine_for(live).process_tomography(
        live.machine.device,
        step.pulses,
        space,
        state.motional,
        branch_sample(live, sample_index, branch, space),
        core.SeedSpec(live.result.diagnostics.root_seed),
        live.options,
        ideal=ideal,
    )
    choi = np.asarray(summary.choi, dtype=complex)
    pm = ProcessMatrixRecord(
        key=key,
        step_index=step_index,
        gate_id=str(step.gate_id),
        ions=labels,
        choi=choi,
        ideal=ideal,
        n_inputs=int(np.prod([d * d for d in space.ion_dims])),
        cp_tp_residual=(float(summary.cp_tp_residual[0]), float(summary.cp_tp_residual[1])),
        average_gate_infidelity=float(summary.average_gate_infidelity),
        entanglement_infidelity=float(core.entanglement_infidelity(choi, core.choi_from_unitary(ideal))),
        depolarizing_rate=float(summary.depolarizing_rate),
        pauli_twirled={str(k): float(v) for k, v in summary.pauli_twirled.items()},
        n_traj=int(summary.n_traj),
        wall_time_s=time.perf_counter() - t0,
    )
    return record.with_cached(pm), pm
