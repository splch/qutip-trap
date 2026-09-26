"""The pipeline of PLAN.md Section 3.4 on a ``Machine``: the compile-calibrate-schedule prefix that ``Machine.run``,
``Machine.schedule`` and ``Machine.estimate`` share, and the whole run behind ``Machine.run``.

    compile: native gates, phase-tracked                     [control.compiler]
      -> calibrate: the surrogate table, cached per device   [calibration.surrogate]
      -> the calibrated micromotion shims programmed         [experiments.micromotion]
      -> schedule: pulses with absolute times                [control.schedule]
      -> space: resolved / frozen / dropped / ENR modes      [run.space]
      -> prepare: Doppler -> sideband -> pump                [prep.recipe, run.job.prepare]
      -> evolve: every branch of the initial mixture on the joint space (JOINT_EXACT), or the gate-local walk
         of Section 5.4 (GATE_LOCAL)                         [dynamics.engine, run.gate_local]
      -> read out every dynamical sample: the POVM or the photon records, the collisions per shot
      -> Result
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

import numpy as np
import qutip as qt

from qutip_trap.control.compiler import CompileReport, compile_report
from qutip_trap.control.schedule import GateDrive, Schedule, resolve_drives, schedule
from qutip_trap.dynamics.engine import EngineReport, MotionalModel, SeedSpec, State, Traces
from qutip_trap.dynamics.evolve import ConvergenceReport, convergence_check
from qutip_trap.dynamics.parallel import map_tasks, worker_count
from qutip_trap.dynamics.space import HilbertSpace
from qutip_trap.dynamics.truncation import warn_if_boundary_exceeds
from qutip_trap.noise.collisions import (
    collision_rate_per_ion,
    sample_collisions,
    sample_kick_quanta,
    sample_reorder,
)
from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, NoiseSample, key_frozen_n, quiet_sample
from qutip_trap.options import Numerics
from qutip_trap.prep.recipe import recipe_of, run_preparation
from qutip_trap.readout.detection import PhotonRecord, count_anomaly_band
from qutip_trap.readout.discriminate import ReadoutOutcome, measure
from qutip_trap.run.gate_local import EngineSetup, GateLocalReport, evolve_gate_local
from qutip_trap.run.job import (
    Branch,
    ReadoutStage,
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
from qutip_trap.run.levels import FidelityLevel, decide_level
from qutip_trap.run.results import Diagnostics, Progress, Result, RunState, aggregate, binomial_error_bars
from qutip_trap.run.space import SpaceSelection, coupled_modes, select_space

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.machine import Machine
    from qutip_trap.options import Physics, ReadoutMode


@dataclass(frozen=True)
class Prefix:
    """What the prefix produced: the compile report and its native circuit, the table (built or given), the device the run
    evolves (the table's calibrated shims programmed onto it), the schedule, the resolved entangling drives, the physics
    after the ``internal_levels`` adjustment, and the notes so far."""

    report: CompileReport
    compiled: Circuit
    table: CalibrationTable
    device: Device
    schedule: Schedule
    entangling_drives: dict[int, GateDrive]
    physics: Physics
    notes: tuple[str, ...]


def compile_calibrate_schedule(machine: Machine, circuit: Circuit, *, seed: int = 0) -> Prefix:
    """Compile, calibrate (the cached surrogate when the machine carries no table), program the calibrated micromotion shims
    and schedule: the prefix of Section 3.4's pipeline."""
    physics, numerics = machine.physics, machine.numerics
    device = machine.device
    drives, ent_drives = resolve_drives(device)
    notes: list[str] = []
    # Section 4.5.5: "leakage is simulated, not estimated, whenever d > 2", so a d > 2 register turns the scattering
    # channels on; the recoil displacements (30 to 50x the cost of the register-only operators) stay an explicit choice
    if physics.internal_levels > 2 and physics.scattering == "estimate":
        recoil = physics.scattering_recoil if physics.scattering_recoil == "vector" else "off"
        physics = replace(physics, scattering="channels", scattering_recoil=recoil)
        notes.append(
            f"internal_levels = {physics.internal_levels} > 2: scattering='channels' turned on with scattering_recoil={recoil!r} "
            "(Section 4.5.5, 'leakage is simulated, not estimated, whenever d > 2'; the recoil displacements are an "
            "explicit choice: pass scattering='channels' with scattering_recoil='minimal' or 'vector'); pass "
            "internal_levels=2 for the d = 2 estimate path instead"
        )
    report = compile_report(circuit, entangler=physics.entangler)
    compiled = report.circuit
    table = machine.table
    if table is None:
        from qutip_trap.calibration import cached_surrogate

        sur = cached_surrogate(
            device,
            seed=seed,
            t0_s=physics.t0_s,
            gate_drives=drives,
            entangling_drives=ent_drives,
            options=numerics,
            builder_options=physics.builder,
            hardware_chain=physics.hardware_chain,
            pairs=compiled.entangling_pairs(),
        )
        table = sur.table
        notes.extend(sur.notes)
    elif not table.is_current_for(device.hash()):
        notes.append(
            "calibration table fitted for another device configuration (hash mismatch): played as given, never regenerated "
            "silently (Section 7.5)"
        )
    # the shim settings the table carries are what the machine has PROGRAMMED (Section 7.5): the run evolves the compensated
    # device, so a stale calibration against a drifted stray field leaves the residual excess micromotion; the hash the table
    # is compared against stays the uncompensated device's (no device parameter changed, only a programmed voltage)
    shim_entries = {
        name: entry
        for name, entry in table.micromotion.items()
        if name.startswith("shim[") and name.endswith("]")
    }
    # only what the compensation experiment MEASURED is programmed; a seed shim is already the device's own setting
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
    sched = schedule(
        compiled,
        device,
        table,
        gate_drives=drives,
        entangling_drives=ent_drives,
        t0_s=0.0,
        parallel=numerics.addressing,
        crosstalk_suppression=physics.crosstalk_suppression,
        stark_compensation=physics.stark_compensation,
    )
    return Prefix(
        report=report,
        compiled=compiled,
        table=table,
        device=device,
        schedule=sched,
        entangling_drives=ent_drives,
        physics=physics,
        notes=tuple(notes),
    )


def _kernel_kind(kinds: set[str]) -> str:
    if not kinds:
        return "none"
    if kinds == {"assembled"}:
        return "assembled"
    if kinds == {"factorized"}:
        return "factorized"
    return "mixed"


class _BranchRun(NamedTuple):
    """One (sample, branch) engine run of a JOINT_EXACT run: its indices, the branch's initial state and its noise sample."""

    sample: int
    branch: int
    state: State
    noise: NoiseSample


def _engine_task(
    payload: tuple[Any, Device, Schedule, State, HilbertSpace, NoiseSample, SeedSpec, Numerics],
) -> tuple[Traces, EngineReport]:
    """One (sample, branch) engine run as a map task (module-level so that it pickles under ``map="parallel"``)."""
    engine, device, sched, state, space, smp, seeds, opts = payload
    traces = engine.run_pulses(device, sched, state, space, smp, seeds, opts)
    rep = engine.last_report
    assert rep is not None
    return traces, rep


class _Reporter:
    """The ``progress`` callback of one run with the run's own clock; a callback of None makes every report a no-op."""

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
    payloads: Sequence[_BranchRun],
    seeds: SeedSpec,
    opts: Numerics,
    report: _Reporter | None = None,
) -> tuple[list[tuple[Traces, EngineReport]], int]:
    """The (sample, branch) engine runs of a JOINT_EXACT run: in-process on one engine when the map is serial, one worker is
    available or there is a single run (the engine's trajectory map then takes the workers), else spread over the workers
    (Section 11.3 item 9). Returns the (traces, report) pairs in order and the workers used. In-process runs report every
    pulse and every run; a parallel map reports its runs once, when it returns."""
    workers = worker_count(opts)
    n_runs = len(payloads)
    if opts.map == "serial" or workers <= 1 or n_runs < 2:
        out: list[tuple[Traces, EngineReport]] = []
        for k, run in enumerate(payloads):
            if report is not None and report.active:

                def per_pulse(p: Progress, k: int = k) -> None:
                    assert report is not None
                    report("pulse", k * p.total + p.done, n_runs * p.total)

                engine.progress = per_pulse
            traces = engine.run_pulses(device, sched, run.state, space, run.noise, seeds, opts)
            rep = engine.last_report
            assert rep is not None
            out.append((traces, rep))
            if report is not None:
                report("branch", k + 1, n_runs)
        engine.progress = None
        used = max((r.workers for _t, r in out), default=1)
        return out, used
    inner = replace(opts, map="serial")
    items = [(engine, device, sched, run.state, space, run.noise, seeds, inner) for run in payloads]
    results = map_tasks(_engine_task, items, map_kind=opts.map, workers=workers)
    if report is not None:
        report("branch", n_runs, n_runs)
    return results, min(workers, n_runs)


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


# ---- the level: JOINT_EXACT or GATE_LOCAL, resolved once per run ------------------------------------------------------------
@dataclass(frozen=True)
class _Walk:
    """What an evolution reads: the device, the schedule, the dynamical samples, the seeds, the solver options, the prepared
    state, the engine setup and the progress reporter; ``notes`` is the run's note list, appended in order."""

    device: Device
    sched: Schedule
    samples: tuple[NoiseSample, ...]
    seeds: SeedSpec
    opts: Numerics
    state0: State
    setup: EngineSetup
    notify: _Reporter
    notes: list[str]


@dataclass
class _Evolved:
    """Every dynamical sample evolved: per sample the weighted register states, and what the diagnostics report."""

    register_states: list[list[tuple[float, qt.Qobj]]]
    space: HilbertSpace
    """The joint space the diagnostics report: grown by the truncation monitor (JOINT_EXACT), or the one above the guards."""
    boundary: dict[int, float]
    traces: list[Traces] = field(default_factory=list)
    margin_reached: dict[int, int] = field(default_factory=dict)
    populated_max: dict[int, int] = field(default_factory=dict)
    integrators: list[str] = field(default_factory=list)
    approximations: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    n_traj_max: int = 1
    kernel_kinds: set[str] = field(default_factory=set)
    workers: int = 1
    propagator_hits: int = 0
    convergence: ConvergenceReport | None = None
    gate_local: GateLocalReport | None = None


class _Level(ABC):
    """How a run integrates (Section 5.4), resolved once per run from the level decision: the space the preparation starts
    on, the branches of the initial mixture, and the evolution of every dynamical sample."""

    name: Literal["JOINT_EXACT", "GATE_LOCAL"]

    def __init__(self, space: HilbertSpace) -> None:
        self.space = space
        self.branches: list[Branch] = []
        self.dropped_weight = 0.0

    def prep_space(self, n_modes: int) -> HilbertSpace:
        return self.space

    def enumerate(
        self, device: Device, sched: Schedule, state0: State, opts: Numerics, notes: list[str]
    ) -> None:
        """The branches of the initial mixture; none when the register is carried whole."""
        self.branches, self.dropped_weight = [], 0.0

    @abstractmethod
    def evolve(self, walk: _Walk) -> _Evolved: ...


class _JointExact(_Level):
    """JOINT_EXACT: every (sample, branch) of the initial mixture through the engine on the joint space (Section 5.3)."""

    name = "JOINT_EXACT"

    def enumerate(
        self, device: Device, sched: Schedule, state0: State, opts: Numerics, notes: list[str]
    ) -> None:
        """The Fock-sum branches: the internal levels, the resolved and ENR modes, and the frozen spectators a pulse couples
        to (their Debye-Waller factor is the physics; a dropped mode costs no branch)."""
        space = self.space
        nbar = state0.motional.nbar
        frozen = {
            m for m in coupled_modes(device, sched.pulses) if m in space.frozen and m not in space.dropped
        }
        enr_modes = list(space.enr_group[0]) if space.enr_group is not None else []
        mode_nbar = {m.mode: float(nbar.get(m.mode, 0.0)) for m in space.resolved}
        mode_nbar.update({m: float(nbar.get(m, 0.0)) for m in [*enr_modes, *sorted(frozen)]})
        branches, dropped = enumerate_branches(
            internal_probabilities(state0, space), mode_nbar, opts.branch_weight_min
        )
        if space.enr_group is not None:
            # an ENR Fock tuple lives inside the excitation cap; branches above it are dropped and reported (Section 5.1)
            n_exc = space.enr_group[1]
            kept = [b for b in branches if sum(b.fock.get(m, 0) for m in enr_modes) <= n_exc]
            over = sum(b.weight for b in branches) - sum(b.weight for b in kept)
            if over > 0.0:
                notes.append(
                    f"initial-mixture branches above the ENR cap N_exc = {n_exc} dropped: weight {over:.3e} (renormalized)"
                )
                dropped += over
            branches = kept
        if dropped > 0.0:
            notes.append(
                f"initial-mixture branches below branch_weight_min = {opts.branch_weight_min:g} dropped: total weight "
                f"{dropped:.3e} (renormalized)"
            )
        self.branches, self.dropped_weight = branches, dropped

    def evolve(self, walk: _Walk) -> _Evolved:
        space, branches, state0, notes = self.space, self.branches, walk.state0, walk.notes
        engine = walk.setup.engine()
        total_weight = sum(b.weight for b in branches)
        payloads: list[_BranchRun] = []
        for s_idx, smp in enumerate(walk.samples):
            for k, br in enumerate(branches):
                st = space.initial_state(
                    list(br.levels),
                    fock={m: n for m, n in br.fock.items() if space.mode_class(m) in ("resolved", "enr")},
                    thermal={m: float(state0.motional.nbar.get(m, 0.0)) for m in space.frozen},
                    provenance=tuple(state0.provenance) + (f"m6.branch[{k}]",),
                )
                values = dict(smp.values)
                values.update(
                    {key_frozen_n(m): float(n) for m, n in br.fock.items() if space.mode_class(m) == "frozen"}
                )
                values[KEY_BRANCH_WEIGHT] = float(br.weight / total_weight)
                sample_b = NoiseSample(
                    sample_id=smp.sample_id, values=values, ou_grids=dict(smp.ou_grids), t_s=smp.t_s
                )
                payloads.append(_BranchRun(s_idx, k, st, sample_b))

        def runs(
            options: Numerics, report: _Reporter | None = None
        ) -> tuple[list[tuple[Traces, EngineReport]], int]:
            return _run_engine_tasks(
                engine, walk.device, walk.sched, space, payloads, walk.seeds, options, report
            )

        results, workers = runs(walk.opts, walk.notify)
        d_int = int(np.prod(space.ion_dims))
        convergence: ConvergenceReport | None = None
        if walk.opts.convergence_check:
            # Section 5.5: the evolution again at atol and rtol tightened by ten; the change in the FIRST sample's register
            # populations, which are deterministic where the sampled histogram is not
            def _register_populations(o: Numerics) -> dict[str, np.ndarray]:
                rho = np.zeros((d_int, d_int), dtype=complex)
                for run, (tr_c, _rep_c) in zip(payloads, runs(o)[0]):
                    if run.sample != 0:
                        continue
                    rho += (branches[run.branch].weight / total_weight) * np.asarray(
                        tr_c.final.internal.full()
                    )
                return {"register_populations": np.real(np.diag(rho))}

            convergence = convergence_check(_register_populations, walk.opts)
            notes.append(convergence.summary())
        evo = _Evolved(
            register_states=[],
            space=space,
            boundary={m.mode: 0.0 for m in space.resolved},
            workers=workers,
            convergence=convergence,
        )
        dims_int = [list(space.ion_dims), list(space.ion_dims)]
        for s_idx in range(len(walk.samples)):
            rho_int = np.zeros((d_int, d_int), dtype=complex)
            for run, (tr, rep) in zip(payloads, results):
                if run.sample != s_idx:
                    continue
                evo.traces.append(tr)
                rho_int += (branches[run.branch].weight / total_weight) * np.asarray(tr.final.internal.full())
                for m, v in tr.boundary_population.items():
                    evo.boundary[m] = max(evo.boundary.get(m, 0.0), float(v))
                for seg in rep.segments:
                    if seg.integrator not in evo.integrators:
                        evo.integrators.append(seg.integrator)
                for a in rep.approximations:
                    if a not in evo.approximations:
                        evo.approximations.append(a)
                for n in rep.notes:
                    if n not in notes:
                        notes.append(n)
                if rep.method not in evo.methods:
                    evo.methods.append(rep.method)
                evo.n_traj_max = max(evo.n_traj_max, rep.trajectories)
                for m, v in rep.margin_reached.items():
                    evo.margin_reached[m] = min(evo.margin_reached.get(m, int(v)), int(v))
                for m, v in rep.populated_n_max.items():
                    evo.populated_max[m] = max(evo.populated_max.get(m, 0), int(v))
                if rep.kernel != "none":
                    evo.kernel_kinds.add(rep.kernel)
                evo.propagator_hits += rep.propagator_cache_hits
                if rep.space != evo.space and rep.space.dimension > evo.space.dimension:
                    # the truncation monitor grew the caps on this branch (Section 5.5): report the largest space
                    evo.space = rep.space
            evo.register_states.append([(1.0, qt.Qobj(rho_int, dims=dims_int))])
            walk.notify("sample", s_idx + 1, len(walk.samples))
        return evo


class _GateLocal(_Level):
    """GATE_LOCAL: the gate-local walk of Section 5.4 from the pumped register and the recipe's occupations."""

    name = "GATE_LOCAL"

    def __init__(self, space: HilbertSpace, caps: Mapping[int, int] | None) -> None:
        super().__init__(space)
        self.caps = caps

    def prep_space(self, n_modes: int) -> HilbertSpace:
        return HilbertSpace(tuple(self.space.ion_dims), (), None, tuple(range(n_modes)))

    def evolve(self, walk: _Walk) -> _Evolved:
        notes = walk.notes

        def gate_local(
            options: Numerics,
            samples: Sequence[NoiseSample],
            progress: Callable[[int, int], None] | None = None,
        ) -> tuple[list[list[tuple[float, qt.Qobj]]], GateLocalReport, list[MotionalModel]]:
            return evolve_gate_local(
                walk.device,
                walk.sched,
                samples,
                walk.seeds,
                options,
                register0=walk.state0.internal,
                nbar0=walk.state0.motional.nbar,
                ion_dims=self.space.ion_dims,
                setup=walk.setup,
                caps=self.caps,
                progress=progress,
            )

        register_states, report, _models = gate_local(
            walk.opts,
            walk.samples,
            (lambda done, total: walk.notify("sample", done, total)) if walk.notify.active else None,
        )
        evo = _Evolved(
            register_states=register_states,
            space=self.space,
            boundary={m.mode: 0.0 for m in self.space.resolved},
            workers=report.workers,
            gate_local=report,
        )
        for step in report.steps:
            for m, v in step.boundary_population.items():
                evo.boundary[m] = max(evo.boundary.get(m, 0.0), float(v))
            for m, v in step.margin_reached.items():
                evo.margin_reached[m] = min(evo.margin_reached.get(m, int(v)), int(v))
            for integ in step.integrators:
                if integ not in evo.integrators:
                    evo.integrators.append(integ)
            if step.method not in evo.methods:
                evo.methods.append(step.method)
            evo.n_traj_max = max(evo.n_traj_max, step.n_traj)
            for n in step.notes:
                if n not in notes:
                    notes.append(n)
        notes.extend(n for n in report.notes if n not in notes)
        if walk.opts.convergence_check:

            def _gate_local_populations(o: Numerics) -> dict[str, np.ndarray]:
                states = gate_local(o, walk.samples[:1])[0]
                rho = sum(w * np.asarray((st if st.isoper else qt.ket2dm(st)).full()) for w, st in states[0])
                return {"register_populations": np.real(np.diag(np.asarray(rho)))}

            # the extraction cache is keyed on atol and rtol, so the base-tolerance pass hits what the walk computed
            convergence = convergence_check(_gate_local_populations, walk.opts)
            evo.convergence = convergence
            notes.append(convergence.summary())
        evo.approximations.append(
            f"GATE_LOCAL: {len([s for s in report.steps if s.kind == 'gate'])} gate steps and "
            f"{len([s for s in report.steps if s.kind == 'idle'])} idle steps through exact gate-local spaces (largest dimension "
            f"{report.largest_local_dimension}); spin-motion and mode-mode correlations traced out between steps, the residual "
            f"displacement bound {report.residual_bound_total:.2e}, the frozen excitation bound {report.frozen_excitation_total:.2e}, "
            f"the dropped crosstalk {report.dropped_crosstalk_total:.2e}, the dropped motional branches' bound "
            f"{report.branch_error_total:.2e} and the keyed tolerance's convergence change {report.tolerance_change_total:.2e} "
            "reported (Section 5.4)"
        )
        return evo


# ---- the readout path: the POVM or the photon records (``Readout.mode``), resolved once per run -----------------------------


class _ReadoutPath(Protocol):
    mode: ReadoutMode
    need_povm: bool
    """Build the POVM (the fast path samples it); the full path replaces it (Section 5.7)."""

    def anomaly_bands(
        self, stage: ReadoutStage, window_s: float, measured: Sequence[int]
    ) -> list[tuple[int, int]]: ...

    def spam_errors(
        self, stage: ReadoutStage, bits: np.ndarray, levels: np.ndarray, measured: Sequence[int]
    ) -> tuple[tuple[float, float], ...]: ...

    def describe(self, stage: ReadoutStage, povm_samples: int) -> list[str]: ...


class _FastReadout:
    """``readout="fast"``: the POVM applied to the joint outcome (Section 5.7)."""

    mode: ReadoutMode = "fast"
    need_povm = True

    def anomaly_bands(
        self, stage: ReadoutStage, window_s: float, measured: Sequence[int]
    ) -> list[tuple[int, int]]:
        return []

    def spam_errors(
        self, stage: ReadoutStage, bits: np.ndarray, levels: np.ndarray, measured: Sequence[int]
    ) -> tuple[tuple[float, float], ...]:
        """(eps_B, eps_D) per ion of the product POVM at zero crosstalk."""
        assert stage.product is not None  # the fast path builds the POVM
        return stage.product.per_ion_errors()

    def describe(self, stage: ReadoutStage, povm_samples: int) -> list[str]:
        assert stage.product is not None
        name = type(stage.discriminator).__name__
        out = [
            f"SPAM definition: readout (eps_B, eps_D) of the {name} discriminator's product POVM at zero crosstalk "
            "(Section 13 row 'Readout figure of merit'); state preparation 1 - P(target) of the optical pump (Section 4.2.6)"
        ]
        if stage.product.uncertainty > 0.0:
            out.append(
                f"readout POVM: {name} has no closed-form confusion, so it was estimated from {povm_samples} "
                f"sampled records per level per ion; every POVM entry carries a statistical uncertainty of "
                f"{stage.product.uncertainty:.2e} (Section 8.4)"
            )
        out.append(
            f"readout fast path: the {'register-wide confusion' if stage.leakage else 'product POVM'} applied to the joint outcome "
            "(Section 5.7)"
        )
        return out


class _FullReadout:
    """``readout="full"``: one photon record per ion per shot, generated and discriminated (Section 5.7)."""

    mode: ReadoutMode = "full"
    need_povm = False

    def anomaly_bands(
        self, stage: ReadoutStage, window_s: float, measured: Sequence[int]
    ) -> list[tuple[int, int]]:
        """Per measured ion, the count band both the bright and the dark distributions explain (heralds bit 2)."""
        return [
            count_anomaly_band(stage.models[q], window_s, ("bright", stage.schemes[q].dark_class))
            for q in measured
        ]

    def spam_errors(
        self, stage: ReadoutStage, bits: np.ndarray, levels: np.ndarray, measured: Sequence[int]
    ) -> tuple[tuple[float, float], ...]:
        """(eps_B, eps_D) per measured column from this run's own records (Section 8.6), nan for a level the circuit never
        populated: the record path built no POVM."""
        errs: list[tuple[float, float]] = []
        for col, q in enumerate(measured):
            bright = stage.schemes[q].bright_level
            pair: list[float] = []
            for lev, wrong_bit in ((bright, 1 - bright), (1 - bright, bright)):
                mask = levels[:, col] == lev
                pair.append(float(np.mean(bits[mask, col] == wrong_bit)) if np.any(mask) else math.nan)
            errs.append((pair[0], pair[1]))
        return tuple(errs)

    def describe(self, stage: ReadoutStage, povm_samples: int) -> list[str]:
        name = type(stage.discriminator).__name__
        return [
            f"SPAM definition: readout (eps_B, eps_D) of the {name} discriminator estimated from this run's own "
            "photon records (readout='full' replaces the POVM rather than preceding it, Section 5.7); a qubit level the "
            "circuit never populated reports nan; state preparation 1 - P(target) of the optical pump (Section 4.2.6)",
            f"readout full path: one photon record per ion per shot generated from the {name} discriminator's window "
            f"({stage.discriminator.window_s:.4g} s) and discriminated, the neighbour coupling of Section 8.5 "
            f"{'applied at the configured PSF leakage' if stage.leakage else 'inactive (no PSF leakage configured)'} "
            "(Section 5.7)",
        ]


def _qubit_shifts(device: Device, table: CalibrationTable, notes: list[str]) -> dict[int, float]:
    """Per ion the true transition minus the table's frame (Section 7.3): the ONLY channel by which the table's frequency
    error reaches the physics, so an uncalibrated entry refuses rather than taking the frame at the true transition."""
    shifts: dict[int, float] = {}
    for i in range(device.crystal.n_ions):
        sp = device.crystal.species[i]
        f_true, _d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
        entry = table.qubit_freq.get(i)
        if entry is not None and entry.status == "uncalibrated":
            raise RunError(
                f"ion {i}: the calibration table's qubit frequency is uncalibrated (fitted by {entry.experiment!r}); the "
                "frame cannot be programmed and the run refuses rather than taking it at the true transition "
                "(Section 7.3). Re-calibrate the ion's Ramsey-frequency experiment or pass a table that carries it."
            )
        if entry is None:
            # no believed frame to be wrong about (never the surrogate, which always seeds one)
            shifts[i] = 0.0
            notes.append(
                f"ion {i}: the table has no qubit frequency; the frame is taken at the true transition"
            )
        else:
            shifts[i] = float(f_true - entry.value)
    return shifts


def execute(
    machine: Machine,
    circuit: Circuit,
    shots: int,
    *,
    seed: int = 0,
    keep_final_state: bool = False,
    progress: Callable[[Progress], None] | None = None,
) -> Result:
    """Section 3.4's pipeline on one machine (``Machine.run``): compile, calibrate, schedule, prepare, evolve, read out every
    dynamical sample with the collision process per shot, and assemble the ``Result`` with its ``Diagnostics`` and
    ``RunRecord``. ``seed`` is the root of every keyed stream."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    started = time.perf_counter()
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    notify = _Reporter(progress, started)
    opts, reading = machine.numerics, machine.readout
    path: _ReadoutPath = _FastReadout() if reading.mode == "fast" else _FullReadout()
    prefix = compile_calibrate_schedule(machine, circuit, seed=seed)
    physics = prefix.physics
    notes: list[str] = list(prefix.notes)
    compiled = prefix.compiled
    table = prefix.table
    device = prefix.device
    sched = prefix.schedule
    prep_run = run_preparation(
        device, recipe_of(device, raman_pair=_raman_pair_hint(prefix.entangling_drives))
    )
    if device.preparation is None:
        notes.append("preparation recipe inferred by prep.recipe.standard_recipe (the device carries none)")
    notes.extend(prep_run.notes)
    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    if opts.space is None:
        selection = select_space(
            device, sched, opts, nbar=prep_run.nbar, ion_dims=[int(physics.internal_levels)] * n_ions
        )
    else:
        selection = SpaceSelection.supplied(opts.space, opts, prep_run.nbar, n_modes)
    joint_space = selection.space
    levels = level_maps(device, joint_space)
    if levels:
        notes.append(
            "register factors with leakage levels: "
            + "; ".join(f"ion {i}: {', '.join(m.labels)}" for i, m in levels.items())
        )
    decision = decide_level(selection.budget, opts, machine.level)
    dim, nnz = decision.dimension, decision.nnz
    level: _Level
    if decision.level is FidelityLevel.JOINT_EXACT:
        if not decision.inside:
            # Section 11.5: the monitor "refuses to build joint spaces above a configurable dimension"
            raise RunError(
                f"level='JOINT_EXACT' asks for a joint space of dimension {dim} with {nnz} drive non-zeros, above the Section "
                f"11.5 guards (joint_dimension_max = {opts.joint_dimension_max}, nnz_max = {opts.nnz_max}); raise them in "
                "Numerics to build it deliberately, reduce the caps or the resolved modes, or let level='auto' route the "
                "run to GATE_LOCAL"
            )
        level = _JointExact(joint_space)
    else:
        notes.append(
            f"GATE_LOCAL (Section 5.4): the joint space would have dimension {dim} and {nnz} drive non-zeros against the "
            f"guards ({opts.joint_dimension_max}, {opts.nnz_max})"
            + ("" if not decision.inside else "; requested below the guards")
        )
        if opts.space is not None:
            notes.append(
                "GATE_LOCAL builds its own gate-local spaces; the supplied space sets the mode classes reported"
            )
        level = _GateLocal(joint_space, opts.caps)
    seeds = SeedSpec(int(seed))
    state0 = prepare(device, level.prep_space(n_modes), preparation=prep_run, levels=levels)
    level.enumerate(device, sched, state0, opts, notes)
    shifts = _qubit_shifts(device, table, notes)
    # timing (Section 7.5) and the dynamical samples, one per contiguous block of the shot clock (Section 3.4)
    stage = readout_stage(
        device,
        table,
        discriminator=reading.discriminator,
        levels=levels or None,
        povm_samples=reading.povm_samples,
        seed=seed,
        need_povm=path.need_povm,
    )
    window = float(stage.discriminator.window_s)
    t_rep = (
        float(physics.shot_period_s)
        if physics.shot_period_s is not None
        else prep_run.duration_s + sched.pulses_end_s + window + float(device.hardware.dead_time_s)
    )
    samples = opts.samples
    quiet = (not physics.noise) or device.noise.is_quiet()
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
    sample_times = [physics.t0_s + f * t_rep for f in first_shots]
    if quiet:
        samples_seq: tuple[NoiseSample, ...] = tuple(quiet_sample(k, t) for k, t in enumerate(sample_times))
    else:
        rng_noise = np.random.default_rng(seeds.child(0, 0, 0, 0, "noise_samples"))
        samples_seq = device.noise.sample_sequence(
            rng_noise, sample_times, device=device, duration_s=sched.pulses_end_s, t0_s=physics.t0_s
        )
    setup = EngineSetup(
        builder_options=physics.builder,
        channels=tuple(physics.extra_channels),
        qubit_shifts_hz=shifts,
        device_channels=bool(physics.noise),
        levels_by_ion=levels or None,
        hardware_chain=physics.hardware_chain,
        scattering_channels=physics.scattering == "channels",
        scattering_recoil=physics.scattering_recoil,
        intensity_noise_channels=physics.intensity_noise_channels,
        table=table,
    )
    evo = level.evolve(_Walk(device, sched, samples_seq, seeds, opts, state0, setup, notify, notes))
    cap_growth = {
        t.mode: t.d - joint_space.truncation(t.mode).d
        for t in evo.space.resolved
        if t.d != joint_space.truncation(t.mode).d
    }
    if cap_growth:
        notes.append(
            "the truncation monitor raised the caps (Section 5.5): "
            + ", ".join(f"mode {m} by {add} level(s)" for m, add in sorted(cap_growth.items()))
        )
    warn_if_boundary_exceeds(evo.boundary, opts.boundary_population_max)
    register_states = evo.register_states
    dm_states = [[(w, st) for w, st in members if st.isoper] for members in register_states]
    rho_register: qt.Qobj | None = None
    if all(len(ms) == len(all_) for ms, all_ in zip(dm_states, register_states)):
        acc = sum((w * st for members in dm_states for w, st in members), 0.0 * register_states[0][0][1])
        rho_register = acc / len(register_states)
    # the readout per sample on its register state(s), the collision process per shot (Section 6.7)
    register_space = HilbertSpace(tuple(evo.space.ion_dims), (), None, ())
    run_state = RunState.nominal(n_ions)
    collisions = device.noise.collisions if physics.noise else None
    coll_rates: dict[int, float] = {}
    if collisions is not None and collisions.pressure_pa > 0.0:
        coll_rates = {
            i: collision_rate_per_ion(collisions, float(device.crystal.masses_kg[i])) for i in range(n_ions)
        }
    # the measured set is the circuit's targets and every trailing measure operation: the scheduler's terminal event
    declared: list[int] = list(compiled.measure)
    for op in compiled.ops:
        if op.name == "measure":
            declared.extend(q for q in op.qubits if q not in declared)
    measured = tuple(sorted(declared)) if declared else tuple(range(n_ions))
    anomaly_bands = path.anomaly_bands(stage, window, measured)
    bits_kept: list[np.ndarray] = []
    sample_kept: list[int] = []
    levels_kept: list[np.ndarray] = []
    heralds_kept: list[int] = []
    posteriors_kept: list[np.ndarray] = []
    records_kept: list[list[int]] = []
    sub_bins_kept: list[np.ndarray] = []
    arrivals_kept: list[tuple[np.ndarray, ...]] = []
    bits_per_sample: list[np.ndarray] = []
    discarded = 0
    # a kick heats the SOFTEST mode most (delta nbar = E_kick/(hbar omega_m)), so that mode sets the reported quanta
    soft_mode_omega = (
        2.0 * math.pi * min(float(m.omega_hz) for m in device.crystal.modes) if device.crystal.modes else 0.0
    )
    kick_quanta: list[float] = []
    reorders = 0
    outcome: ReadoutOutcome | None = None
    # the RunRecord's outcome of every KEPT shot in the Result's row order over every ion
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
                mode=path.mode,
                povm=stage.povm,
                keep_records=not path.need_povm,
            )
            for j in range(n_k):
                herald = 0
                keep = True
                if coll_rates:
                    assert collisions is not None
                    rng_c = np.random.default_rng(seeds.child(smp.sample_id, 0, shot, 0, "collisions"))
                    for ev in sample_collisions(rng_c, collisions, coll_rates, t_rep):
                        herald |= 1
                        label = f"collision:{ev.outcome}:ion{ev.ion}"
                        if ev.outcome == "heating_kick":
                            # the neutral's thermal energy times the mass ratio (Section 6.7), in quanta of the softest mode
                            kick = 0.0
                            if soft_mode_omega > 0.0:
                                kick = sample_kick_quanta(
                                    rng_c,
                                    collisions,
                                    float(device.crystal.masses_kg[ev.ion]),
                                    soft_mode_omega,
                                )
                                kick_quanta.append(kick)
                            label = f"collision:heating_kick:ion{ev.ion}:dnbar={kick:.3g}"
                            if ev.time_s >= prep_run.duration_s:
                                keep = False  # the crystal melted during the sequence; Doppler recools only before it
                        else:
                            keep = False
                        run_state = replace(run_state, events=run_state.events + ((shot, label),))
                        if ev.outcome == "reorder":
                            run_state = replace(
                                run_state, order=sample_reorder(rng_c, collisions, run_state.order, ev.ion)
                            )
                            reorders += 1
                        elif ev.outcome == "loss":
                            run_state = replace(run_state, lost=run_state.lost | {ev.ion})
                        elif ev.outcome != "heating_kick":
                            run_state = replace(run_state, dark=run_state.dark | {ev.ion})
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
                sample_kept.append(s_idx)
                levels_kept.append(np.asarray(outcome.levels[j, list(measured)], dtype=np.uint8).copy())
                out_bits_kept.append(np.asarray(outcome.bits[j], dtype=np.uint8).copy())
                out_levels_kept.append(np.asarray(outcome.levels[j], dtype=np.uint8).copy())
                out_times_kept.append(np.asarray(outcome.time_used_s[j], dtype=float).copy())
                out_posteriors_kept.append(
                    np.asarray(outcome.posteriors[j], dtype=float).copy()
                    if outcome.posteriors is not None
                    else np.full(n_ions, np.nan)
                )
                sample_bits.append(row)
                heralds_kept.append(herald)
                if outcome.posteriors is not None:
                    posteriors_kept.append(np.asarray(outcome.posteriors[j]))
                if outcome.records is not None:
                    recs_j = outcome.records[j]
                    out_records_kept.append(tuple(recs_j))
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
    spam: dict[str, tuple[float, float]] = {}
    levels_arr = np.asarray(levels_kept, dtype=np.uint8).reshape(-1, len(measured))
    for i, (eps_b, eps_d) in enumerate(path.spam_errors(stage, bits, levels_arr, measured)):
        spam[f"q{i}"] = (float(eps_b), float(eps_d))
        spam[f"q{i}.state_preparation"] = (float(prep_run.preparation_error(i)), 0.0)
    approximations = list(evo.approximations)
    methods = evo.methods
    if not physics.noise:
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
    if physics.noise:
        sentence = device.noise.provenance_sentence()
        if sentence:
            approximations.append(f"noise provenance (Section 6.1): {sentence}")
        approximations.extend(device.noise.approximations(device))
    approximations.extend(path.describe(stage, reading.povm_samples))
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
            "bounded, reported discrepancy of Section 9.5)"
        )
    for i, (beta, omega_rf) in sorted(stage.micromotion.items()):
        approximations.append(
            f"readout micromotion: ion {i}'s detection rates carry J_0({beta:.4g})^2 on the carrier and J_1^2 on each "
            f"first sideband at Omega_rf/2pi = {omega_rf / (2.0 * math.pi):.4g} Hz (Section 8.8)"
        )
    approximations.extend(device.hardware.describe())
    notes.extend(v for v in selection.guard_violations if v not in notes)
    branches = level.branches
    diagnostics = Diagnostics(
        level=level.name,
        space=evo.space,
        mode_class=dict(selection.mode_class),
        run_state=run_state,
        wall_clock_span_s=float((shots - 1) * t_rep),
        boundary_population=evo.boundary,
        margin_levels={m.mode: m.margin_levels for m in evo.space.resolved},
        dropped_modes=selection.dropped_modes,
        frozen_contribution=selection.frozen_contribution,
        integrator=",".join(evo.integrators) if evo.integrators else "none",
        tolerances=(opts.atol, opts.rtol),
        samples=n_samples,
        trajectories=(len(branches) or 1) * evo.n_traj_max,
        branches=len(branches) or 1,
        shots_per_sample=int(shots // n_samples),
        shots_per_sample_realized=tuple(int(m) for m in counts_per_sample),
        effective_sample_size=n_eff,
        root_seed=int(seed),
        calibration=table,
        approximations=tuple(approximations) + tuple(notes) + tuple(selection.notes),
        intrinsic_budget=intrinsic_budget(device, sched, selection),
        dropped_branch_weight=level.dropped_weight,
        frozen_excitation_bound=dict(selection.frozen_excitation),
        dropped_contribution=selection.dropped_contribution,
        margin_reached=evo.margin_reached,
        populated_n_max=evo.populated_max,
        cap_growth=cap_growth,
        gate_local=evo.gate_local,
        kernel=_kernel_kind(evo.kernel_kinds),
        workers=evo.workers,
        propagator_cache_hits=evo.propagator_hits,
        convergence=evo.convergence,
        level_reason=decision.reason,
    )
    record = RunRecord(
        compile=prefix.report,
        schedule=sched,
        selection=selection,
        preparation=prep_run,
        branches=tuple(branches),
        traces=tuple(evo.traces),
        register_state=rho_register,
        readout=stage,
        outcome=outcome_all,
        table=table,
        qubit_shifts_hz=shifts,
        notes=tuple(notes),
        gate_local=evo.gate_local,
    )
    return Result(
        bitstrings=bits,
        bit_order="qubit0_lsb",
        counts=counts,
        probabilities=probabilities,
        error_bars=binomial_error_bars(probabilities, n_eff),
        photon_records=np.array(records_kept, dtype=int) if records_kept else None,
        posteriors=np.array(posteriors_kept) if posteriors_kept else None,
        noise_samples=tuple(samples_seq),
        heralds=np.asarray(heralds_kept, dtype=np.uint8),
        discarded_shots=int(discarded),
        run_state=run_state,
        spam=spam,
        final_state=rho_register if keep_final_state else None,
        diagnostics=diagnostics,
        sub_bin_records=(
            np.stack(sub_bins_kept) if (sub_bins_kept and len(sub_bins_kept) == len(bits_kept)) else None
        ),
        arrival_times_s=tuple(arrivals_kept)
        if (arrivals_kept and len(arrivals_kept) == len(bits_kept))
        else None,
        qubits=tuple(int(q) for q in measured),
        registers=dict(compiled.registers),
        machine_hash=machine.hash(),
        created_at=created_at,
        duration_s=float(time.perf_counter() - started),
        record=record,
        sample_of_shot=np.asarray(sample_kept, dtype=np.int64),
    )
