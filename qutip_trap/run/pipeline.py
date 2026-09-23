"""The run pipeline behind ``Machine.run`` (``execute``) and its compile-calibrate-schedule prefix.

compile -> calibrate -> program the calibrated micromotion shims -> schedule -> select the space -> prepare -> evolve
(JOINT_EXACT's branches, or the gate-local walk) -> read out, with background-gas collisions per shot -> ``Result``.
``compile_calibrate_schedule`` is the prefix that ``Machine.schedule`` and ``Machine.estimate`` share."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import qutip as qt

from qutip_trap.control.compiler import CompileReport, compile_report
from qutip_trap.control.schedule import CrosstalkSuppression, GateDrive, Schedule, resolve_drives, schedule
from qutip_trap.dynamics.engine import EngineReport, MotionalModel, SeedSpec, SolverOptions, State, Traces
from qutip_trap.dynamics.evolve import ConvergenceReport, convergence_check
from qutip_trap.dynamics.parallel import map_tasks, worker_count
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.hilbert.truncation import warn_if_boundary_exceeds
from qutip_trap.noise.collisions import (
    collision_rate_per_ion,
    sample_collisions,
    sample_kick_quanta,
    sample_reorder,
)
from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, NoiseSample, key_frozen_n, quiet_sample
from qutip_trap.prep.recipe import recipe_of, run_preparation
from qutip_trap.readout.detection import PhotonRecord, count_anomaly_band
from qutip_trap.readout.discriminate import Discriminator, ReadoutOutcome, measure
from qutip_trap.run.gate_local import EngineSetup, GateLocalReport, evolve_gate_local
from qutip_trap.run.job import (
    _LAST_RECORD,
    Branch,
    ReadoutMode,
    RunError,
    RunRecord,
    _raman_pair_hint,
    effective_sample_size,
    enumerate_branches,
    internal_probabilities,
    intrinsic_budget,
    level_maps,
    prepare,
    readout_stage,
)
from qutip_trap.run.levels import FidelityLevel, decide_level, within_budget
from qutip_trap.run.results import Diagnostics, Progress, Result, RunState, aggregate, binomial_error_bars
from qutip_trap.run.space import SpaceSelection, select_space

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.machine import Machine


@dataclass(frozen=True)
class Prefix:
    """What the prefix produced; ``device`` carries the table's calibrated shims and ``options`` the ``internal_levels``
    adjustment."""

    report: CompileReport
    compiled: Circuit
    table: CalibrationTable
    device: Device
    schedule: Schedule
    gate_drives: dict[int, GateDrive]
    entangling_drives: dict[int, GateDrive]
    options: SolverOptions
    notes: tuple[str, ...]


def compile_calibrate_schedule(
    circuit: Circuit,
    device: Device,
    *,
    table: CalibrationTable | None = None,
    seed: int = 0,
    t0_s: float = 0.0,
    options: SolverOptions | None = None,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
    builder_options: BuilderOptions | None = None,
    caps: Mapping[int, int] | None = None,
    calibrate_kwargs: Mapping[str, Any] | None = None,
    entangler: Literal["ms", "zz"] = "ms",
    parallel: bool | None = None,
    crosstalk_suppression: CrosstalkSuppression = "none",
    stark_compensation: bool = True,
    internal_levels: int = 2,
) -> Prefix:
    """Compile, calibrate (the cached surrogate when ``table`` is None), program the calibrated shims and schedule."""
    opts = options or SolverOptions()
    # the drive maps: the call's keyword arguments first, then the device's roles, then the inference
    drives, ent_drives = resolve_drives(device, gate_drives, entangling_drives)
    notes: list[str] = []
    # only the scattering channels populate d > 2 leakage levels, so internal_levels > 2 turns them on
    if internal_levels > 2 and not opts.scattering_channels:
        # without the recoil displacements (30 to 50x the cost) unless the caller asked for the vector quadrature
        recoil = opts.scattering_recoil if opts.scattering_recoil == "vector" else "off"
        opts = replace(opts, scattering_channels=True, scattering_recoil=recoil)
        notes.append(
            f"internal_levels = {internal_levels} > 2: scattering_channels turned ON with scattering_recoil={recoil!r} "
            "(Section 4.5.5, 'leakage is simulated, not estimated, whenever d > 2'; the recoil displacements are an "
            "explicit choice: pass scattering_channels=True with scattering_recoil='minimal' or 'vector'); pass "
            "internal_levels=2 for the d = 2 estimate path instead"
        )
    # 1. compile
    report = compile_report(circuit, device, entangler=entangler)
    compiled = report.circuit
    # 2. calibrate (the surrogate, cached per device and seed) when no table is given
    if table is None:
        from qutip_trap.calibration.cache import cached_surrogate

        kw = dict(calibrate_kwargs or {})
        kw.setdefault("pairs", compiled.entangling_pairs())
        sur = cached_surrogate(
            device,
            seed=seed,
            t0_s=t0_s,
            gate_drives=drives,
            entangling_drives=ent_drives,
            options=opts,
            builder_options=builder_options,
            caps=caps,
            **kw,
        )
        table = sur.table
        notes.extend(sur.notes)
    elif not table.is_current_for(device.hash()):
        notes.append(
            "calibration table fitted for another device configuration (hash mismatch): played as given, never regenerated "
            "silently (Section 7.5)"
        )
    # 2b. the table's shims are what the machine has programmed: the run evolves the compensated device (crystal
    # re-solved); the hash the table is checked against stays the uncompensated device's (a voltage is not a parameter)
    shim_entries = {
        name: entry
        for name, entry in table.micromotion.items()
        if name.startswith("shim[") and name.endswith("]")
    }
    # only measured shims are programmed: a seed shim is already the device's setting, an uncalibrated one is left alone
    shims = {
        name[len("shim[") : -1]: float(entry.value)
        for name, entry in shim_entries.items()
        if entry.status == "calibrated"
    }
    unresolved = sorted(name for name, entry in shim_entries.items() if entry.status == "uncalibrated")
    if unresolved:
        notes.append(
            f"micromotion compensation uncalibrated for {unresolved}: the run keeps the device's own shim settings and "
            "carries whatever excess micromotion they leave (Section 7.5)"
        )
    if shims:
        from qutip_trap.experiments.micromotion import device_with_compensation

        compensated = device_with_compensation(device, shims)
        if compensated.trap != device.trap:
            device = compensated
            notes.append(
                "calibrated micromotion compensation applied to the run: "
                + ", ".join(f"{k} = {v:.6g}" for k, v in sorted(shims.items()))
                + " (Section 7.5; the reported device hash is the uncompensated device's)"
            )
    # 3. schedule
    sched = schedule(
        compiled,
        device,
        table,
        gate_drives=drives,
        entangling_drives=ent_drives,
        t0_s=0.0,
        parallel=parallel,
        crosstalk_suppression=crosstalk_suppression,
        stark_compensation=stark_compensation,
    )
    return Prefix(
        report=report,
        compiled=compiled,
        table=table,
        device=device,
        schedule=sched,
        gate_drives=drives,
        entangling_drives=ent_drives,
        options=opts,
        notes=tuple(notes),
    )


def _diagnostics_level(level: FidelityLevel) -> Literal["JOINT_EXACT", "GATE_LOCAL"]:
    """The level a run ran at, as the literal ``Diagnostics.level`` carries (never the AUTO policy)."""
    if level is FidelityLevel.JOINT_EXACT:
        return "JOINT_EXACT"
    if level is FidelityLevel.GATE_LOCAL:
        return "GATE_LOCAL"
    raise ValueError("a run reports the level it ran at, not the AUTO policy")


def _kernel_kind(kinds: set[str]) -> str:
    if not kinds:
        return "none"
    if kinds == {"assembled"}:
        return "assembled"
    if kinds == {"factorized"}:
        return "factorized"
    return "mixed"


def _engine_task(
    payload: tuple[Any, Device, Schedule, State, HilbertSpace, NoiseSample, SeedSpec, SolverOptions],
) -> tuple[Traces, EngineReport]:
    """One (sample, branch) engine run as a map task (module-level so that it pickles under ``map="parallel"``)."""
    engine, device, sched, state, space, smp, seeds, opts = payload
    traces = engine.run_pulses(device, sched, state, space, smp, seeds, opts)
    rep = engine.last_report
    assert rep is not None
    return traces, rep


class _Reporter:
    """The ``progress`` callback of one run, on the run's own clock: a call with (stage, done, total) hands on a
    ``Progress``; a callback of None makes every call a no-op."""

    def __init__(self, callback: Callable[[Progress], None] | None, started: float) -> None:
        self.callback = callback
        self.started = started

    @property
    def active(self) -> bool:
        return self.callback is not None

    def __call__(self, stage: str, done: int, total: int) -> None:
        if self.callback is not None:
            self.callback(Progress(stage, int(done), int(total), time.perf_counter() - self.started))


def _run_engine_tasks(
    engine: Any,
    device: Device,
    sched: Schedule,
    space: HilbertSpace,
    payloads: Sequence[tuple[int, int, State, NoiseSample]],
    seeds: SeedSpec,
    opts: SolverOptions,
    report: _Reporter | None = None,
) -> tuple[list[tuple[Traces, EngineReport]], int]:
    """The JOINT_EXACT engine runs, in-process when the map is serial, one worker is available or there is one run,
    else spread over the workers; returns the (traces, report) pairs in order and the workers used. In-process runs
    report every pulse and run, a parallel map its runs once, when it returns."""
    workers = worker_count(opts)
    n_runs = len(payloads)
    if opts.map == "serial" or workers <= 1 or n_runs < 2:
        out: list[tuple[Traces, EngineReport]] = []
        for k, (_s_idx, _k, st, smp) in enumerate(payloads):
            if report is not None and report.active:

                def per_pulse(p: Progress, k: int = k) -> None:
                    assert report is not None
                    report("pulse", k * p.total + p.done, n_runs * p.total)

                engine.progress = per_pulse
            traces = engine.run_pulses(device, sched, st, space, smp, seeds, opts)
            rep = engine.last_report
            assert rep is not None
            out.append((traces, rep))
            if report is not None:
                report("branch", k + 1, n_runs)
        engine.progress = None
        used = max((r.workers for _t, r in out), default=1)
        return out, used
    inner = replace(opts, map="serial")
    items = [(engine, device, sched, st, space, smp, seeds, inner) for _s_idx, _k, st, smp in payloads]
    results = map_tasks(_engine_task, items, map_kind=opts.map, workers=workers)
    if report is not None:
        report("branch", n_runs, n_runs)
    return results, min(workers, n_runs)


def _supplied_class(space: HilbertSpace, mode: int) -> Literal["resolved", "frozen", "dropped", "enr"]:
    cls = space.mode_class(mode)
    if cls == "resolved":
        return "resolved"
    if cls == "enr":
        return "enr"
    return "frozen"


def _allocate(total: int, weights: Sequence[float]) -> list[int]:
    """Largest-remainder allocation of ``total`` shots over members with the given weights (one member takes them all)."""
    if len(weights) == 1:
        return [int(total)]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = w * total
    base = np.floor(raw).astype(int)
    rest = int(total - base.sum())
    order = np.argsort(-(raw - base))
    for k in order[:rest]:
        base[k] += 1
    return [int(x) for x in base]


def execute(
    machine: Machine,
    circuit: Circuit,
    shots: int,
    *,
    seed: int = 0,
    keep_final_state: bool = False,
    progress: Callable[[Progress], None] | None = None,
) -> Result:
    """The pipeline for one machine, ``circuit`` to ``Result`` (which carries ``machine.hash()``); ``seed`` roots every
    keyed stream and ``progress`` is called with a ``Progress`` per pulse, branch, sample and readout."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    started = time.perf_counter()
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    notify = _Reporter(progress, started)
    physics, numerics, reading = machine.physics, machine.numerics, machine.readout
    device = machine.device
    trunc = numerics.truncation
    level: FidelityLevel = machine.level
    t0_s: float = physics.t0_s
    shot_period_s: float | None = physics.shot_period_s
    samples: int | None = numerics.parallel.samples
    space: HilbertSpace | None = trunc.space
    caps: Mapping[int, int] | None = trunc.caps
    enr_group: tuple[Sequence[int], int] | None = trunc.enr_group
    noise: bool = physics.noise
    internal_levels: int = physics.internal_levels
    channels: Sequence[CollapseOp] = physics.extra_channels
    builder_options: BuilderOptions | None = physics.builder
    readout: ReadoutMode = reading.mode
    discriminator: Discriminator | None = reading.discriminator
    povm_samples: int = reading.povm_samples
    # 1 to 3: compile, calibrate, program the shims and schedule (the prefix Machine.schedule and .estimate share)
    prefix = compile_calibrate_schedule(
        circuit,
        device,
        table=machine.table,
        seed=seed,
        t0_s=t0_s,
        options=numerics.to_solver_options(physics),
        builder_options=builder_options,
        caps=caps,
        entangler=physics.entangler,
        parallel=numerics.parallel.addressing,
        crosstalk_suppression=physics.crosstalk_suppression,
        stark_compensation=physics.stark_compensation,
        internal_levels=internal_levels,
    )
    opts = prefix.options
    ent_drives = prefix.entangling_drives
    notes: list[str] = list(prefix.notes)
    report = prefix.report
    compiled = prefix.compiled
    table = prefix.table
    device = prefix.device
    sched = prefix.schedule
    # 4. preparation (the physics of the recipe) and the space
    cooling_pair = _raman_pair_hint(ent_drives)
    prep_run = run_preparation(device, recipe_of(device, raman_pair=cooling_pair))
    if device.preparation is None:
        notes.append("preparation recipe inferred by prep.recipe.standard_recipe (the device carries none)")
    notes.extend(prep_run.notes)
    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    if internal_levels < 2:
        raise ValueError("internal_levels is at least 2")
    if space is None:
        selection = select_space(
            device,
            sched,
            opts,
            nbar=prep_run.nbar,
            caps=caps,
            ion_dims=[int(internal_levels)] * n_ions,
            enr=enr_group,
        )
    else:
        if enr_group is not None:
            raise ValueError("give the ENR group inside the supplied space or as enr_group, not both")
        selection = SpaceSelection(
            space,
            {m: _supplied_class(space, m) for m in range(n_modes)},
            {},
            {m: float(prep_run.nbar.get(m, 0.0)) for m in range(n_modes)},
            ("space supplied by the caller",),
            budget=within_budget(space, opts),
        )
    joint_space = selection.space
    levels = level_maps(device, joint_space)
    if levels:
        notes.append(
            "register factors with leakage levels: "
            + "; ".join(f"ion {i}: {', '.join(m.labels)}" for i, m in levels.items())
        )
    # the size-guard verdict the selection reached on its declaration, before any operator was allocated
    _ok, dim, nnz = selection.budget
    # the run's actual space makes the guards exact; the numbers compared become Diagnostics.level_reason
    decision = decide_level(device, compiled, opts, space=joint_space)
    ok = decision.level is FidelityLevel.JOINT_EXACT
    requested = FidelityLevel(level)
    run_level: FidelityLevel = decision.level if requested is FidelityLevel.AUTO else requested
    level_reason = (
        decision.reason
        if requested is FidelityLevel.AUTO
        else f"{run_level.value} forced by the caller; level='auto' would choose {decision.reason}"
    )
    if run_level == "JOINT_EXACT" and not ok:
        # an explicit JOINT_EXACT above the guards is refused rather than built: the guards are the configuration
        raise RunError(
            f"level='JOINT_EXACT' asks for a joint space of dimension {dim} with {nnz} drive non-zeros, above the Section 11.5 "
            f"guards (joint_dimension_max = {opts.joint_dimension_max}, nnz_max = {opts.nnz_max}); raise them in SolverOptions "
            "to build it deliberately, reduce the caps or the resolved modes, or let level='auto' route the run to GATE_LOCAL"
        )
    if run_level == "GATE_LOCAL":
        notes.append(
            f"GATE_LOCAL (Section 5.4): the joint space would have dimension {dim} and {nnz} drive non-zeros against the "
            f"guards ({opts.joint_dimension_max}, {opts.nnz_max})"
            + ("" if not ok else "; requested below the guards")
        )
        if space is not None:
            notes.append(
                "GATE_LOCAL builds its own gate-local spaces; the supplied space sets the mode classes reported"
            )
    seeds = SeedSpec(int(seed))
    # the prepared state on the joint space (JOINT_EXACT) or the register alone (GATE_LOCAL: the joint one did not fit)
    prep_space = (
        joint_space
        if run_level == "JOINT_EXACT"
        else HilbertSpace(tuple(joint_space.ion_dims), (), None, tuple(range(n_modes)))
    )
    state0 = prepare(device, prep_space, table, quiet_sample(0), seeds, preparation=prep_run, levels=levels)
    # 5. the branches of the initial mixture (JOINT_EXACT: the Fock sum)
    probs_int = internal_probabilities(state0, prep_space)
    branches: list[Branch] = []
    dropped_weight = 0.0
    if run_level == "JOINT_EXACT":
        from qutip_trap.light.raman import lamb_dicke_parameters

        coupled_frozen: set[int] = set()
        for pulse in sched.pulses:
            dk = pulse.drive.delta_k(device.beams)
            if float(np.linalg.norm(dk)) == 0.0:
                continue
            for ion in pulse.drive.ions:
                etas, _ = lamb_dicke_parameters(device, ion, dk)
                # only frozen spectators become Fock branches (for their Debye-Waller factor); a dropped mode costs none
                coupled_frozen.update(
                    m for m in joint_space.frozen if abs(etas[m]) > 1e-12 and m not in joint_space.dropped
                )
        mode_nbar = {m.mode: float(state0.motional.nbar.get(m.mode, 0.0)) for m in joint_space.resolved}
        enr_modes = list(joint_space.enr_group[0]) if joint_space.enr_group is not None else []
        mode_nbar.update({m: float(state0.motional.nbar.get(m, 0.0)) for m in enr_modes})
        mode_nbar.update({m: float(state0.motional.nbar.get(m, 0.0)) for m in sorted(coupled_frozen)})
        branches, dropped_weight = enumerate_branches(probs_int, mode_nbar, opts.branch_weight_min)
        if joint_space.enr_group is not None:
            # an ENR Fock tuple lives inside the excitation cap; branches above it are dropped and reported
            n_exc = joint_space.enr_group[1]
            kept = [b for b in branches if sum(b.fock.get(m, 0) for m in enr_modes) <= n_exc]
            over = sum(b.weight for b in branches) - sum(b.weight for b in kept)
            if over > 0.0:
                notes.append(
                    f"initial-mixture branches above the ENR cap N_exc = {n_exc} dropped: weight {over:.3e} (renormalized)"
                )
                dropped_weight += over
            branches = kept
        if dropped_weight > 0.0:
            notes.append(
                f"initial-mixture branches below branch_weight_min = {opts.branch_weight_min:g} dropped: total weight "
                f"{dropped_weight:.3e} (renormalized)"
            )
    # 6. the qubit-frequency shifts: the true transition minus the table's frame
    shifts: dict[int, float] = {}
    for i in range(n_ions):
        sp = device.crystal.species[i]
        f_true, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
        entry = table.qubit_freq.get(i)
        if entry is not None and entry.status == "uncalibrated":
            # the shift is the only path by which a frequency error reaches the physics: a zero would make it error-free
            raise RunError(
                f"ion {i}: the calibration table's qubit frequency is uncalibrated (fitted by {entry.experiment!r}); the "
                "frame cannot be programmed and the run refuses rather than taking it at the true transition "
                "(Section 7.3). Re-calibrate the ion's Ramsey-frequency experiment or pass a table that carries it."
            )
        if entry is None:
            # no qubit-frequency entry (the surrogate always seeds one): no believed frame, so the frame is the true one
            shifts[i] = 0.0
            notes.append(
                f"ion {i}: the table has no qubit frequency; the frame is taken at the true transition"
            )
        else:
            shifts[i] = float(f_true - entry.value)
    # 7. timing and the dynamical samples: one per contiguous block of shots, taken at its first shot's time
    stage = readout_stage(
        device,
        table,
        discriminator=discriminator,
        levels=levels or None,
        povm_samples=povm_samples,
        seed=seed,
        need_povm=(readout == "fast"),
    )
    window = float(stage.discriminator.window_s)
    t_rep = (
        float(shot_period_s)
        if shot_period_s is not None
        else prep_run.duration_s + sched.pulses_end_s + window + float(device.hardware.dead_time_s)
    )
    quiet = (not noise) or device.noise.is_quiet(device)
    if quiet:
        n_samples = 1
        if samples not in (None, 1):
            notes.append(
                f"samples={samples} requested but the noise model has no quasi-static or sampled content (or noise=False): "
                "one nominal sample"
            )
    else:
        n_samples = int(samples) if samples is not None else min(int(shots), 64)
        n_samples = max(1, min(n_samples, int(shots)))
    counts_per_sample = [shots // n_samples + (1 if k < shots % n_samples else 0) for k in range(n_samples)]
    first_shots = [sum(counts_per_sample[:k]) for k in range(n_samples)]
    sample_times = [t0_s + f * t_rep for f in first_shots]
    if quiet:
        samples_seq: tuple[NoiseSample, ...] = tuple(quiet_sample(k, t) for k, t in enumerate(sample_times))
    else:
        rng_noise = np.random.default_rng(seeds.child(0, 0, 0, 0, "noise_samples"))
        samples_seq = device.noise.sample_sequence(
            rng_noise, sample_times, device=device, duration_s=sched.pulses_end_s, t0_s=t0_s
        )
    # 8. evolve every sample: every branch of the initial mixture on the joint space (JOINT_EXACT), or the gate-local walk
    setup = EngineSetup(
        builder_options=builder_options,
        channels=tuple(channels),
        qubit_shifts_hz=shifts,
        device_channels=bool(noise),
        levels_by_ion=levels or None,
        hardware_chain=True,
        table=table,
    )
    dims_int = [list(joint_space.ion_dims), list(joint_space.ion_dims)]
    d_int = int(np.prod(joint_space.ion_dims))
    traces_all: list[Traces] = []
    boundary: dict[int, float] = {m.mode: 0.0 for m in joint_space.resolved}
    margin_reached: dict[int, int] = {}
    populated_max: dict[int, int] = {}
    integrators: list[str] = []
    approximations: list[str] = []
    methods: list[str] = []
    n_traj_max = 1
    register_states: list[list[tuple[float, qt.Qobj]]] = []
    gl_report: GateLocalReport | None = None
    space_initial = joint_space
    kernel_kinds: set[str] = set()
    workers_used = 1
    propagator_hits = 0
    convergence: ConvergenceReport | None = None
    if run_level == "JOINT_EXACT":
        engine = setup.engine()
        total_weight = sum(b.weight for b in branches)
        # every (sample, branch) run is independent: prepare all on the selected space, map them, accumulate in order
        payloads: list[tuple[int, int, State, NoiseSample]] = []
        for s_idx, smp in enumerate(samples_seq):
            for k, br in enumerate(branches):
                fock_res = {
                    m: n for m, n in br.fock.items() if joint_space.mode_class(m) in ("resolved", "enr")
                }
                thermal_frozen = {m: float(state0.motional.nbar.get(m, 0.0)) for m in joint_space.frozen}
                st = joint_space.initial_state(
                    list(br.levels),
                    fock=fock_res,
                    thermal={m: v for m, v in thermal_frozen.items()},
                    provenance=tuple(state0.provenance) + (f"m6.branch[{k}]",),
                )
                values = dict(smp.values)
                values.update(
                    {
                        key_frozen_n(m): float(n)
                        for m, n in br.fock.items()
                        if joint_space.mode_class(m) == "frozen"
                    }
                )
                values[KEY_BRANCH_WEIGHT] = float(br.weight / total_weight)
                sample_b = NoiseSample(
                    sample_id=smp.sample_id, values=values, ou_grids=dict(smp.ou_grids), t_s=smp.t_s
                )
                payloads.append((s_idx, k, st, sample_b))
        results, workers_used = _run_engine_tasks(
            engine, device, sched, joint_space, payloads, seeds, opts, report=notify
        )
        if opts.convergence_check:
            # compare the first sample's register populations, deterministic where the sampled histogram is not
            def _register_populations(o: SolverOptions) -> dict[str, np.ndarray]:
                res, _w = _run_engine_tasks(engine, device, sched, joint_space, payloads, seeds, o)
                rho = np.zeros((d_int, d_int), dtype=complex)
                for (p_idx, k, _st, _sb), (tr_c, _rep_c) in zip(payloads, res):
                    if p_idx != 0:
                        continue
                    rho += (branches[k].weight / total_weight) * np.asarray(tr_c.final.internal.full())
                return {"register_populations": np.real(np.diag(rho))}

            convergence = convergence_check(_register_populations, opts)
            notes.append(convergence.summary())
        grown_space = joint_space
        for s_idx in range(len(samples_seq)):
            rho_int = np.zeros((d_int, d_int), dtype=complex)
            for (p_idx, k, _st, _sb), (tr, rep) in zip(payloads, results):
                if p_idx != s_idx:
                    continue
                br = branches[k]
                traces_all.append(tr)
                rho_int += (br.weight / total_weight) * np.asarray(tr.final.internal.full())
                for m, v in tr.boundary_population.items():
                    boundary[m] = max(boundary.get(m, 0.0), float(v))
                for seg in rep.segments:
                    if seg.integrator not in integrators:
                        integrators.append(seg.integrator)
                for a in rep.approximations:
                    if a not in approximations:
                        approximations.append(a)
                for n in rep.notes:
                    if n not in notes:
                        notes.append(n)
                if rep.method not in methods:
                    methods.append(rep.method)
                n_traj_max = max(n_traj_max, rep.trajectories)
                for m, v in rep.margin_reached.items():
                    margin_reached[m] = min(margin_reached.get(m, int(v)), int(v))
                for m, v in rep.populated_n_max.items():
                    populated_max[m] = max(populated_max.get(m, 0), int(v))
                if rep.kernel != "none":
                    kernel_kinds.add(rep.kernel)
                propagator_hits += rep.propagator_cache_hits
                if rep.space != grown_space and rep.space.dimension > grown_space.dimension:
                    # the truncation monitor grew the caps on this branch: the diagnostics report the largest space
                    grown_space = rep.space
            register_states.append([(1.0, qt.Qobj(rho_int, dims=dims_int))])
            notify("sample", s_idx + 1, len(samples_seq))
        joint_space = grown_space
        dims_int = [list(joint_space.ion_dims), list(joint_space.ion_dims)]
    else:
        register_states, gl_report, _models = evolve_gate_local(
            device,
            sched,
            samples_seq,
            seeds,
            opts,
            register0=state0.internal,
            nbar0=state0.motional.nbar,
            ion_dims=joint_space.ion_dims,
            setup=setup,
            caps=caps,
            progress=(lambda done, total: notify("sample", done, total)) if notify.active else None,
        )
        for step in gl_report.steps:
            for m, v in step.boundary_population.items():
                boundary[m] = max(boundary.get(m, 0.0), float(v))
            for m, v in step.margin_reached.items():
                margin_reached[m] = min(margin_reached.get(m, int(v)), int(v))
            for integ in step.integrators:
                if integ not in integrators:
                    integrators.append(integ)
            if step.method not in methods:
                methods.append(step.method)
            n_traj_max = max(n_traj_max, step.n_traj)
            for n in step.notes:
                if n not in notes:
                    notes.append(n)
        notes.extend(n for n in gl_report.notes if n not in notes)
        # the workers the walk actually used, not worker_count(opts)
        workers_used = gl_report.workers
        if opts.convergence_check:

            def _gate_local_populations(o: SolverOptions) -> dict[str, np.ndarray]:
                states, _rep, _mods = evolve_gate_local(
                    device,
                    sched,
                    samples_seq[:1],
                    seeds,
                    o,
                    register0=state0.internal,
                    nbar0=state0.motional.nbar,
                    ion_dims=joint_space.ion_dims,
                    setup=setup,
                    caps=caps,
                )
                rho = sum(w * np.asarray((st if st.isoper else qt.ket2dm(st)).full()) for w, st in states[0])
                return {"register_populations": np.real(np.diag(np.asarray(rho)))}

            # no cache clearing: the extraction cache is keyed on atol and rtol, so only the tightened pass recomputes
            convergence = convergence_check(_gate_local_populations, opts)
            notes.append(convergence.summary())
        approximations.append(
            f"GATE_LOCAL: {len([s for s in gl_report.steps if s.kind == 'gate'])} gate steps and "
            f"{len([s for s in gl_report.steps if s.kind == 'idle'])} idle steps through exact gate-local spaces (largest dimension "
            f"{gl_report.largest_local_dimension}); spin-motion and mode-mode correlations traced out between steps, the residual "
            f"displacement bound {gl_report.residual_bound_total:.2e}, the frozen excitation bound {gl_report.frozen_excitation_total:.2e}, "
            f"the dropped crosstalk {gl_report.dropped_crosstalk_total:.2e}, the dropped motional branches' bound "
            f"{gl_report.branch_error_total:.2e} and the keyed tolerance's convergence change {gl_report.tolerance_change_total:.2e} "
            "reported (Section 5.4)"
        )
    cap_growth = {
        t.mode: t.d - space_initial.truncation(t.mode).d
        for t in joint_space.resolved
        if t.d != space_initial.truncation(t.mode).d
    }
    if cap_growth:
        notes.append(
            "the truncation monitor raised the caps (Section 5.5): "
            + ", ".join(f"mode {m} by {add} level(s)" for m, add in sorted(cap_growth.items()))
        )
    warn_if_boundary_exceeds(boundary, opts.boundary_population_max)
    dm_states = [[(w, st) for w, st in members if st.isoper] for members in register_states]
    rho_register: qt.Qobj | None = None
    if all(len(ms) == len(all_) for ms, all_ in zip(dm_states, register_states)):
        acc = sum((w * st for members in dm_states for w, st in members), 0.0 * register_states[0][0][1])
        rho_register = acc / len(register_states)
    # 9. readout per sample on its register state(s), the collision process per shot
    register_space = HilbertSpace(tuple(joint_space.ion_dims), (), None, ())
    run_state = RunState.nominal(n_ions)
    collisions = device.noise.collisions if noise else None
    coll_rates: dict[int, float] = {}
    if collisions is not None and collisions.pressure_pa > 0.0:
        coll_rates = {
            i: collision_rate_per_ion(collisions, float(device.crystal.masses_kg[i])) for i in range(n_ions)
        }
    # the measured set: the circuit's targets and every measure op, the set the scheduler's terminal event uses
    declared: list[int] = list(compiled.measure)
    for op in compiled.ops:
        if op.name == "measure":
            declared.extend(q for q in op.qubits if q not in declared)
    measured = tuple(sorted(declared)) if declared else tuple(range(n_ions))
    bits_kept: list[np.ndarray] = []
    levels_kept: list[np.ndarray] = []
    heralds_kept: list[int] = []
    posteriors_kept: list[np.ndarray] = []
    records_kept: list[list[int]] = []
    sub_bins_kept: list[np.ndarray] = []
    arrivals_kept: list[tuple[np.ndarray, ...]] = []
    anomaly_bands = (
        [
            count_anomaly_band(stage.models[q], window, ("bright", stage.schemes[q].dark_class))
            for q in measured
        ]
        if readout == "full"
        else []
    )
    bits_per_sample: list[np.ndarray] = []
    discarded = 0
    prep_duration = prep_run.duration_s
    # the kick heats the SOFTEST mode most (delta nbar = E_kick/(hbar omega_m)), so that mode sets the reported quanta
    soft_mode_omega = (
        2.0 * math.pi * min(float(m.omega_hz) for m in device.crystal.modes) if device.crystal.modes else 0.0
    )
    kick_quanta: list[float] = []
    reorders = 0
    outcome: ReadoutOutcome | None = None
    # the RunRecord's outcome: every kept shot in the Result's row order, over every ion
    out_bits_kept: list[np.ndarray] = []
    out_levels_kept: list[np.ndarray] = []
    out_times_kept: list[np.ndarray] = []
    out_posteriors_kept: list[np.ndarray] = []
    out_records_kept: list[tuple[PhotonRecord, ...]] = []
    for s_idx, (smp, members) in enumerate(zip(samples_seq, register_states)):
        n_s = counts_per_sample[s_idx]
        if n_s == 0:
            bits_per_sample.append(np.zeros((0, len(measured)), dtype=np.uint8))
            continue
        sample_bits: list[np.ndarray] = []
        shot = first_shots[s_idx]
        for (_w, rho_s), n_k in zip(members, _allocate(n_s, [w for w, _s in members])):
            if n_k == 0:
                continue
            reg_state = State(
                internal=rho_s,
                motional=MotionalModel(reduced={}, nbar={}, frozen=()),
                joint=rho_s,
                provenance=tuple(state0.provenance) + ("m6.register_mixture",),
            )
            outcome = measure(
                register_space,
                reg_state,
                stage.schemes,
                stage.models,
                stage.discriminator,
                seeds,
                shots=n_k,
                sample=smp.sample_id,
                trajectory=0,
                first_shot=shot,
                leakage=stage.leakage or None,
                depumping=stage.depumping or None,
                mode=readout,
                povm=stage.povm if readout == "fast" else None,
                keep_records=(readout == "full"),
            )
            for j in range(n_k):
                herald = 0
                keep = True
                if coll_rates:
                    rng_c = np.random.default_rng(seeds.child(smp.sample_id, 0, shot, 0, "collisions"))
                    assert collisions is not None
                    for ev in sample_collisions(rng_c, collisions, coll_rates, t_rep):
                        herald |= 1
                        run_state = RunState(
                            run_state.order,
                            run_state.dark,
                            run_state.lost,
                            run_state.events + ((shot, f"collision:{ev.outcome}:ion{ev.ion}"),),
                        )
                        if ev.outcome == "heating_kick":
                            # drawn per event and recorded in quanta of the softest mode, the one it heats most
                            kick = 0.0
                            if soft_mode_omega > 0.0:
                                kick = sample_kick_quanta(
                                    rng_c,
                                    collisions,
                                    float(device.crystal.masses_kg[ev.ion]),
                                    soft_mode_omega,
                                )
                                kick_quanta.append(kick)
                            run_state = RunState(
                                run_state.order,
                                run_state.dark,
                                run_state.lost,
                                run_state.events[:-1]
                                + ((shot, f"collision:heating_kick:ion{ev.ion}:dnbar={kick:.3g}"),),
                            )
                            if ev.time_s >= prep_duration:
                                keep = False  # the crystal melted during the sequence; the Doppler stage recools only before it
                        elif ev.outcome == "reorder":
                            keep = False
                            run_state = RunState(
                                sample_reorder(rng_c, collisions, run_state.order, ev.ion),
                                run_state.dark,
                                run_state.lost,
                                run_state.events,
                            )
                            reorders += 1
                        elif ev.outcome == "loss":
                            keep = False
                            run_state = RunState(
                                run_state.order, run_state.dark, run_state.lost | {ev.ion}, run_state.events
                            )
                        else:  # dark_ion
                            keep = False
                            run_state = RunState(
                                run_state.order, run_state.dark | {ev.ion}, run_state.lost, run_state.events
                            )
                row = np.asarray(outcome.bits[j, list(measured)], dtype=np.uint8).copy()
                unusable = run_state.dark | run_state.lost
                if unusable:
                    herald |= 2
                    for col, q in enumerate(measured):
                        if q in unusable:
                            row[col] = stage.schemes[q].bit_of_class("dark")
                if outcome.records is not None and any(
                    not (lo <= outcome.records[j][q].total <= hi)
                    for q, (lo, hi) in zip(measured, anomaly_bands)
                ):
                    herald |= 4
                shot += 1
                if not keep:
                    discarded += 1
                    continue
                bits_kept.append(row)
                levels_kept.append(np.asarray(outcome.levels[j, list(measured)], dtype=np.uint8).copy())
                out_bits_kept.append(np.asarray(outcome.bits[j], dtype=np.uint8).copy())
                out_levels_kept.append(np.asarray(outcome.levels[j], dtype=np.uint8).copy())
                out_times_kept.append(np.asarray(outcome.time_used_s[j], dtype=float).copy())
                out_posteriors_kept.append(
                    np.asarray(outcome.posteriors[j], dtype=float).copy()
                    if outcome.posteriors is not None
                    else np.full(n_ions, np.nan)
                )
                if outcome.records is not None:
                    out_records_kept.append(tuple(outcome.records[j]))
                sample_bits.append(row)
                heralds_kept.append(herald)
                if outcome.posteriors is not None:
                    posteriors_kept.append(np.asarray(outcome.posteriors[j]))
                if readout == "full" and outcome.records is not None:
                    recs_j = outcome.records[j]
                    records_kept.append([recs_j[q].total for q in measured])
                    if all(recs_j[q].sub_bins is not None for q in measured):
                        sub_bins_kept.append(
                            np.stack([np.asarray(recs_j[q].sub_bins, dtype=int) for q in measured])
                        )
                    if all(recs_j[q].arrivals_s is not None for q in measured):
                        arrivals_kept.append(
                            tuple(np.asarray(recs_j[q].arrivals_s, dtype=float) for q in measured)
                        )
        bits_per_sample.append(
            np.asarray(sample_bits, dtype=np.uint8).reshape(-1, len(measured))
            if sample_bits
            else np.zeros((0, len(measured)), dtype=np.uint8)
        )
        notify("readout", s_idx + 1, len(samples_seq))
    assert outcome is not None
    posts_all = np.asarray(out_posteriors_kept, dtype=float).reshape(-1, n_ions)
    outcome_all = ReadoutOutcome(
        bits=np.asarray(out_bits_kept, dtype=np.uint8).reshape(-1, n_ions),
        levels=np.asarray(out_levels_kept, dtype=np.uint8).reshape(-1, n_ions),
        posteriors=None if posts_all.size == 0 or np.all(np.isnan(posts_all)) else posts_all,
        time_used_s=np.asarray(out_times_kept, dtype=float).reshape(-1, n_ions),
        records=(
            tuple(out_records_kept)
            if out_records_kept and len(out_records_kept) == len(out_bits_kept)
            else None
        ),
        mode=outcome.mode,
    )
    bits = np.asarray(bits_kept, dtype=np.uint8).reshape(-1, len(measured))
    counts, probabilities = aggregate(bits)
    n_eff = effective_sample_size(bits_per_sample)
    error_bars = binomial_error_bars(probabilities, n_eff)
    spam: dict[str, tuple[float, float]] = {}
    if stage.product is not None:
        model_errors = stage.product.per_ion_errors()
    else:
        # without a POVM, (eps_B, eps_D) is estimated from this run's own records; an unpopulated level reports nan
        lv = np.asarray(levels_kept, dtype=np.uint8).reshape(-1, len(measured))
        errs: list[tuple[float, float]] = []
        for col, q in enumerate(measured):
            bright = stage.schemes[q].bright_level
            pair: list[float] = []
            for lev, wrong_bit in ((bright, 1 - bright), (1 - bright, bright)):
                mask = lv[:, col] == lev
                pair.append(float(np.mean(bits[mask, col] == wrong_bit)) if np.any(mask) else math.nan)
            errs.append((pair[0], pair[1]))
        model_errors = tuple(errs)
    for i, (eps_b, eps_d) in enumerate(model_errors):
        spam[f"q{i}"] = (float(eps_b), float(eps_d))
        spam[f"q{i}.state_preparation"] = (float(prep_run.preparation_error(i)), 0.0)
    photon_records = np.array(records_kept, dtype=int) if (readout == "full" and records_kept) else None
    sub_bin_records = (
        np.stack(sub_bins_kept) if (sub_bins_kept and len(sub_bins_kept) == len(bits_kept)) else None
    )
    arrival_times = tuple(arrivals_kept) if (arrivals_kept and len(arrivals_kept) == len(bits_kept)) else None
    posteriors = np.array(posteriors_kept) if posteriors_kept else None
    if not noise:
        approximations.append(
            "noise: the nominal sample without the device's collapse operators (noise=False)"
        )
    elif quiet and any(m != "sesolve" for m in methods):
        approximations.append(
            "noise: the nominal sample (no quasi-static or sampled content) with the device's collapse operators, solver "
            + "/".join(methods)
        )
    elif quiet:
        approximations.append("noise: the nominal sample; the noise model is quiet")
    else:
        approximations.append(
            f"noise: {n_samples} dynamical samples at the shot clock (T_rep = {t_rep:.4g} s), solver {'/'.join(methods)}"
        )
    if run_state.dark or run_state.lost:
        approximations.append(
            "collisions: after a dark-ion or loss event the remaining ions' dynamics stay on the nominal crystal; the flagged "
            "ions read dark, and the crystal_image experiment of Section 6.7 detects the event for a recalibration on the "
            "reduced device (a Device with N - 1 ions is a different device and gets its own table, Section 7.5)"
        )
    if kick_quanta:
        approximations.append(
            f"collisions: {len(kick_quanta)} heating kick(s) drawn from the Section 6.7 energy distribution "
            f"(k_B T m_gas/m_ion, mean {sum(kick_quanta) / len(kick_quanta):.3g} quanta of the softest mode, max "
            f"{max(kick_quanta):.3g}); a kick landing before the preparation is recooled by the cooling stage and the shot "
            "is kept, but a kick landing after it DISCARDS the shot rather than adding its nbar to the already-evolved "
            "sample, because the dynamical samples are propagated before the shot loop (the melt and recrystallization "
            "dynamics are a Section 1.3 non-goal)"
        )
    if reorders:
        approximations.append(
            f"collisions: {reorders} reorder event(s) permuted RunState.order from the configured permutation "
            "distribution and DISCARDED the shot; the permutation does not re-derive b_{i,m}, eta_{i,m} or the pair sign "
            "s of Section 7.7 for the shots that follow, so Section 6.7's 'runs the rest of the sequence with the wrong "
            "mode structure' is met at the herald/discard level only (the sample loop is not re-entered after a reorder)"
        )
    if noise:
        sentence = device.noise.provenance_sentence()
        if sentence:
            approximations.append(f"noise provenance (Section 6.1): {sentence}")
    disc_name = type(stage.discriminator).__name__
    if stage.product is not None:
        approximations.append(
            f"SPAM definition: readout (eps_B, eps_D) of the {disc_name} discriminator's product POVM at zero crosstalk "
            "(Section 13 row 'Readout figure of merit'); state preparation 1 - P(target) of the optical pump (Section 4.2.6)"
        )
        if stage.product.uncertainty > 0.0:
            approximations.append(
                f"readout POVM: {disc_name} has no closed-form confusion, so it was estimated from {povm_samples} sampled "
                f"records per level per ion; every POVM entry carries a statistical uncertainty of "
                f"{stage.product.uncertainty:.2e} (Section 8.4)"
            )
    else:
        approximations.append(
            f"SPAM definition: readout (eps_B, eps_D) of the {disc_name} discriminator estimated from this run's own "
            "photon records (readout='full' replaces the POVM rather than preceding it, Section 5.7); a qubit level the "
            "circuit never populated reports nan; state preparation 1 - P(target) of the optical pump (Section 4.2.6)"
        )
    if readout == "fast":
        approximations.append(
            f"readout fast path: the {'register-wide confusion' if stage.leakage else 'product POVM'} applied to the joint outcome "
            "(Section 5.7)"
        )
    else:
        approximations.append(
            f"readout full path: one photon record per ion per shot generated from the {disc_name} discriminator's window "
            f"({stage.discriminator.window_s:.4g} s) and discriminated, the neighbour coupling of Section 8.5 "
            f"{'applied at the configured PSF leakage' if stage.leakage else 'inactive (no PSF leakage configured)'} "
            "(Section 5.7)"
        )
    if stage.depumping:
        approximations.append(
            "readout crosstalk (depumping half, Section 8.5): one bright neighbour raises an ion's (R_d, R_b) by "
            + ", ".join(
                f"d={d}: ({dd:.4g}, {db:.4g}) s^-1" for d, (dd, db) in sorted(stage.depumping.items())
            )
            + " through the leaked resonant light bounded by I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2), the neighbours' "
            "classes frozen at their start (the first-order form)"
        )
    if stage.crosstalk_discrepancy is not None:
        approximations.append(
            f"readout crosstalk: the register-wide confusion at the configured PSF leakage {stage.leakage} differs from the "
            f"product POVM by at most {stage.crosstalk_discrepancy:.4f} in a per-ion declared-bright probability (the "
            "bounded, reported discrepancy of Sections 9.5 and 9.17)"
        )
    for i, (beta, omega_rf) in sorted(stage.micromotion.items()):
        approximations.append(
            f"readout micromotion: ion {i}'s detection rates carry J_0({beta:.4g})^2 on the carrier and J_1^2 on each "
            f"first sideband at Omega_rf/2pi = {omega_rf / (2.0 * math.pi):.4g} Hz (Section 8.8)"
        )
    approximations.extend(device.hardware.describe())
    if selection.guard_violations:
        notes.extend(v for v in selection.guard_violations if v not in notes)
    diagnostics = Diagnostics(
        level=_diagnostics_level(run_level),
        space=joint_space,
        mode_class=dict(selection.mode_class),
        run_state=run_state,
        wall_clock_span_s=float((shots - 1) * t_rep),
        boundary_population=boundary,
        margin_levels={m.mode: m.margin_levels for m in joint_space.resolved},
        dropped_modes=selection.dropped_modes,
        frozen_contribution=selection.frozen_contribution,
        integrator=",".join(integrators) if integrators else "none",
        tolerances=(opts.atol, opts.rtol),
        samples=n_samples,
        trajectories=(len(branches) if branches else 1) * n_traj_max,
        branches=len(branches) if branches else 1,
        shots_per_sample=int(shots // n_samples),
        shots_per_sample_realized=tuple(int(m) for m in counts_per_sample),
        effective_sample_size=n_eff,
        root_seed=int(seed),
        calibration=table,
        approximations=tuple(approximations) + tuple(notes) + tuple(selection.notes),
        intrinsic_budget=intrinsic_budget(device, sched, selection),
        dropped_branch_weight=dropped_weight,
        frozen_excitation_bound=dict(selection.frozen_excitation),
        dropped_contribution=selection.dropped_contribution,
        margin_reached=margin_reached,
        populated_n_max=populated_max,
        cap_growth=cap_growth,
        gate_local=gl_report,
        kernel=_kernel_kind(kernel_kinds),
        workers=workers_used,
        propagator_cache_hits=propagator_hits,
        convergence=convergence,
        level_reason=level_reason,
    )
    result = Result(
        bitstrings=bits,
        bit_order="qubit0_lsb",
        counts=counts,
        probabilities=probabilities,
        error_bars=error_bars,
        photon_records=photon_records,
        posteriors=posteriors,
        noise_samples=tuple(samples_seq),
        heralds=np.asarray(heralds_kept, dtype=np.uint8),
        discarded_shots=int(discarded),
        run_state=run_state,
        spam=spam,
        final_state=rho_register if keep_final_state else None,
        diagnostics=diagnostics,
        sub_bin_records=sub_bin_records,
        arrival_times_s=arrival_times,
        qubits=tuple(int(q) for q in measured),
        registers=dict(compiled.registers),
        machine_hash=machine.hash(),
        created_at=created_at,
        duration_s=float(time.perf_counter() - started),
    )
    _LAST_RECORD[id(result)] = RunRecord(
        compile=report,
        schedule=sched,
        selection=selection,
        preparation=prep_run,
        branches=tuple(branches),
        traces=tuple(traces_all),
        register_state=rho_register,
        readout=stage,
        outcome=outcome_all,
        table=table,
        qubit_shifts_hz=shifts,
        notes=tuple(notes),
        gate_local=gl_report,
    )
    return result
