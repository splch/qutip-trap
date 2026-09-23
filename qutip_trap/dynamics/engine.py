"""The pulse engine protocol, the run-time state records and the JOINT_EXACT engine.

Randomness comes from one root ``SeedSequence`` per run, spawned by (sample, trajectory, shot, ion, channel), so every
variate is independent of execution order, worker count and truncation retries. With collapse operators the state is a
density matrix (``mesolve``) up to ``SolverOptions.mesolve_dimension_max`` and keyed ``mcsolve`` trajectories above it.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import qutip as qt

from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.parallel import worker_count
from qutip_trap.dynamics.rotating import RotatingSegment, expectation_phase, rotating_frame
from qutip_trap.hilbert.operators import required_margin

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.tomography import TomographyRecord
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.noise.levels import InternalLevels
    from qutip_trap.noise.sampling import NoiseSample
    from qutip_trap.run.results import Progress

# QuTiP multistep integrators, never used (they damp the oscillatory spectrum of -iH)
MULTISTEP_INTEGRATORS: frozenset[str] = frozenset({"adams", "bdf", "lsoda", "vode", "zvode"})
ALLOWED_INTEGRATORS: frozenset[str] = frozenset(
    {"dop853", "vern7", "vern9", "tsit5", "explicit_rk", "krylov", "diag"}
)

LindbladMethod = Literal["auto", "mesolve", "mcsolve"]
RecoilOption = Literal["off", "minimal", "vector"]


@dataclass(frozen=True)
class MotionalModel:
    """Per-mode state carried between pulses."""

    reduced: dict[int, Qobj]
    """mode -> (n_max + 1)^2 reduced density matrix."""
    nbar: dict[int, float]
    frozen: tuple[int, ...]


@dataclass(frozen=True)
class State:
    """What ``prepare()`` returns and ``run_pulses()`` advances."""

    internal: Qobj
    """Register density matrix or ket on the ion factors of ``space``."""
    motional: MotionalModel
    joint: Qobj | None
    """The joint ket or density matrix; None after a trajectory ensemble too large to average into one."""
    provenance: tuple[str, ...]
    """Ids of the preparation stages that produced it, in order."""


@dataclass(frozen=True)
class SeedSpec:
    """One root SeedSequence per run, spawned by key."""

    root: int

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError("the root seed is a non-negative integer")

    @staticmethod
    def channel_key(channel: str) -> int:
        """A deterministic 32-bit key for a channel name (never Python's salted ``hash``)."""
        return int.from_bytes(hashlib.sha256(channel.encode("utf-8")).digest()[:4], "big")

    def child(
        self, sample: int, trajectory: int, shot: int, ion: int, channel: str
    ) -> np.random.SeedSequence:
        """The keyed child stream for one (sample, trajectory, shot, ion, channel), independent of execution order."""
        if min(sample, trajectory, shot, ion) < 0:
            raise ValueError("sample, trajectory, shot and ion indices are non-negative")
        return np.random.SeedSequence(
            self.root, spawn_key=(sample, trajectory, shot, ion, self.channel_key(channel))
        )


@dataclass(frozen=True)
class SolverOptions:
    """The numerical policy of a run: tolerances, size guards, truncation, trajectories, parallelism, physics switches."""

    atol: float = 1e-10
    rtol: float = 1e-8
    nsteps: int = 10**7
    integrators: tuple[str, ...] = ("dop853", "vern9")
    """The escalation ladder; never a multistep method."""
    joint_dimension_max: int = 4096
    nnz_max: int = 2 * 10**7
    """With ``joint_dimension_max``, the size guards that route a run to GATE_LOCAL."""
    mode_dimension_max: int = 64
    """Ceiling on one resolved mode's Fock dimension; a clamp is reported in ``SpaceSelection.notes``."""
    boundary_population_max: float = 1e-6
    freeze_chi_max_rad: float = 0.05
    map: Literal["serial", "parallel", "loky"] = "parallel"
    """The QuTiP map the parallel work runs through."""
    e_ops_for_target_tol: bool = True
    """Allow the ``trajectory_target_tol`` estimate, which needs the population e_ops."""
    trajectory_target_tol: float | None = None
    """Tolerance on the population e_ops from which ``mcsolve``'s ``target_tol`` estimates the trajectory count (capped by
    ``ntraj``), then replayed from keyed seeds; None keeps ``ntraj`` (``target_tol`` can stop on a zero-variance batch)."""
    improved_sampling: bool = True
    """``mcsolve``'s no-jump trajectory as a weighted deterministic member, when exactly one segment is on trajectories."""
    freeze_alpha_max: float = 1e-4
    """|alpha_m|^2 (2 nbar_m + 1) below which a spectator may be frozen rather than resolved."""
    branch_weight_min: float = 1e-6
    """Weight below which a branch of the initial thermal and preparation mixture is dropped (and reported)."""
    lindblad_method: LindbladMethod = "auto"
    """``mesolve``, ``mcsolve`` (``ntraj`` per pure initial state) or ``auto`` (mesolve up to ``mesolve_dimension_max``)."""
    mesolve_dimension_max: int = 128
    ntraj: int = 64
    """Trajectories per pure initial state on the mcsolve path (a fixed keyed seed list)."""
    scattering_channels: bool = False
    """Build the photon-scattering collapse operators of every pulse instead of reporting the estimate."""
    scattering_recoil: RecoilOption = "minimal"
    intensity_noise_channels: bool = True
    """The white part of the laser-intensity spectrum as the channel sqrt(D) H_drive(t)."""
    hardware_chain: bool = True
    margin_check: bool = True
    """Raise the cap and rerun when a resolved mode's margin above its populated range is less than its eta needs."""
    convergence_check: bool = False
    """``run()`` repeats at ten-times-tighter tolerances and reports ``Diagnostics.convergence`` (triples the cost)."""
    map_accuracy: float = 1e-3
    """epsilon_map of the GATE_LOCAL tomography (ceil(1/epsilon_map) trajectories per input on the trajectory path)."""
    crosstalk_threshold: float = 1e-3
    """GATE_LOCAL: the crosstalk |epsilon| at which a neighbour joins the gate-local space (below, it is dropped)."""
    register_dm_max_qubits: int = 12
    """GATE_LOCAL register: a density matrix up to this many qubits, a pure-state ensemble beyond."""
    register_ensemble: int = 64
    """Members of the pure-state ensemble above ``register_dm_max_qubits``."""
    workers: int | None = None
    """Processes for the parallel maps; None = every CPU QuTiP sees, 1 = in-process (as is ``map="serial"``)."""
    propagator_cache: bool = True
    """Integrate an internal-state-only segment's propagator once and reuse it (keyed on the Hamiltonian's fingerprint)."""
    tomography_isometry: bool = True
    """GATE_LOCAL tomography of a unitary step from the propagated internal basis instead of every input state."""
    tomography_dropped_weight_max: float | None = None
    """GATE_LOCAL tomography: the total motional-branch weight w a step may drop beyond ``branch_weight_min`` (a 2w
    diamond-norm bound); None derives ``map_accuracy / 4``, 0.0 keeps every branch above the floor."""
    tomography_tolerance_keyed: bool = True
    """GATE_LOCAL: loosen a unitary step's default tolerances to the map accuracy (``tomography.keyed_tolerances``)."""
    margin_element_tol: float | None = None
    """The interior-element tolerance a resolved mode's margin is derived from (``hilbert.operators.required_margin``);
    None keeps the fixed margins, and the GATE_LOCAL walk then derives ``map_accuracy * 1e-5`` for its step spaces."""
    rotating_frame: bool = True
    """Integrate ket segments in the exact rotating frame of their diagonal H_0 where it applies."""
    store_marginals: bool = False
    """Store every carried mode's Fock populations at every stored time as ``Traces.mode_marginal``."""

    def __post_init__(self) -> None:
        if self.atol <= 0.0 or self.rtol <= 0.0 or self.nsteps <= 0:
            raise ValueError("tolerances and nsteps must be positive")
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers is a positive process count or None (every CPU)")
        if not 0.0 < self.map_accuracy < 1.0 or not 0.0 <= self.crosstalk_threshold <= 1.0:
            raise ValueError(
                "map_accuracy is a fraction in (0, 1) and crosstalk_threshold a Rabi ratio in [0, 1]"
            )
        if self.register_dm_max_qubits < 1 or self.register_ensemble < 1:
            raise ValueError("register_dm_max_qubits and register_ensemble are positive")
        if not 0.0 < self.freeze_alpha_max < 1.0 or not 0.0 < self.branch_weight_min < 1.0:
            raise ValueError("freeze_alpha_max and branch_weight_min are fractions in (0, 1)")
        if (
            self.tomography_dropped_weight_max is not None
            and not 0.0 <= self.tomography_dropped_weight_max < 1.0
        ):
            raise ValueError("tomography_dropped_weight_max is a weight fraction in [0, 1) or None")
        if self.margin_element_tol is not None and self.margin_element_tol <= 0.0:
            raise ValueError("margin_element_tol is a positive tolerance or None")
        if not self.integrators:
            raise ValueError("at least one integrator is required")
        bad = [name for name in self.integrators if name in MULTISTEP_INTEGRATORS]
        if bad:
            raise ValueError(f"multistep integrators are never used (Section 5.3): {bad}")
        unknown = [name for name in self.integrators if name not in ALLOWED_INTEGRATORS]
        if unknown:
            raise ValueError(f"unknown QuTiP integrators: {unknown}")
        if self.joint_dimension_max < 2 or self.nnz_max < 1:
            raise ValueError("the size guards must be positive")
        if self.mode_dimension_max < 2:
            raise ValueError("mode_dimension_max is at least two Fock levels per resolved mode")
        if not 0.0 < self.boundary_population_max < 1.0:
            raise ValueError("boundary_population_max is a population fraction in (0, 1)")
        if self.ntraj < 1 or self.mesolve_dimension_max < 1:
            raise ValueError("ntraj and mesolve_dimension_max are positive")


@dataclass(frozen=True)
class Traces:
    """What ``run_pulses()`` returns."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    reduced_internal: tuple[Qobj, ...]
    mode_occupations: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
    """(time, "traj{k}:{channel}") of every quantum jump of the trajectory path; empty on the density-matrix paths."""
    final: State
    boundary_population: dict[int, float]
    mode_marginal: dict[int, np.ndarray] | None = None
    """Per carried mode, the (T, d_m) Fock populations P(n, t) (rows align with ``times_s``), weighted as
    ``expectations``; None unless ``SolverOptions.store_marginals``."""
    wall_time_s: dict[str, float] = field(default_factory=dict)
    """Wall seconds per pulse gate id (a segment's split equally among its pulses; idle segments under ``"idle"``)."""


@dataclass(frozen=True)
class ChannelSummary:
    """Process tomography summary of one pulse."""

    choi: np.ndarray
    cp_tp_residual: tuple[float, float]
    """Residual against the completely-positive cone and the trace-preserving affine set after Dykstra projection."""
    n_traj: int
    average_gate_infidelity: float
    pauli_twirled: dict[str, float]
    depolarizing_rate: float
    """epsilon with Lambda_eps(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P."""


class PulseEngine(Protocol):
    """The one entry point that run/, calibration/ and experiments/ share."""

    def run_pulses(
        self,
        device: Device,
        schedule: Schedule,
        state: State,
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions,
    ) -> Traces: ...

    def process_tomography(
        self,
        device: Device,
        pulse: Pulse,
        space: HilbertSpace,
        motional_model: MotionalModel,
        sample: NoiseSample,
        seeds: SeedSpec,
    ) -> ChannelSummary: ...


__all__ = [
    "ALLOWED_INTEGRATORS",
    "MULTISTEP_INTEGRATORS",
    "ChannelSummary",
    "LindbladMethod",
    "MotionalModel",
    "PulseEngine",
    "SeedSpec",
    "SolverOptions",
    "State",
    "Traces",
]


# ---- the joint-exact engine ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentReport:
    """What one integration segment did."""

    t_start_s: float
    t_end_s: float
    pulses: tuple[str | None, ...]
    integrator: str
    atol: float
    rhs_evaluations: int | None
    steps_per_period: float | None
    approximations: tuple[str, ...]
    boundary_population: dict[int, float]
    retries: tuple[str, ...]
    method: str = "sesolve"
    """sesolve, mesolve or mcsolve."""
    n_collapse_ops: int = 0
    channels: tuple[str, ...] = ()
    """The channel names active on the segment (deduplicated by kind)."""
    kernel: str = "none"
    """How the segment's drive operators were held: ``factorized``, ``assembled``, ``mixed`` or ``none``."""
    frame: str = "schrodinger"
    """``rotating`` or ``schrodinger``: the picture the evaluation counts refer to."""
    wall_time_s: float = 0.0


@dataclass(frozen=True)
class EngineReport:
    """What one ``run_pulses`` did: its segments, the space it ended on after any cap growth, method, retries and notes."""

    segments: tuple[SegmentReport, ...]
    frozen_n: dict[int, int]
    growth_retries: int
    space: HilbertSpace
    drive_records: tuple[object, ...]
    trajectories: int = 1
    """Trajectories carried at the end (1 on the deterministic paths)."""
    method: str = "sesolve"
    hardware_notes: tuple[str, ...] = ()
    schedule_played: Schedule | None = None
    """The schedule after the hardware chain (what the ions saw)."""
    notes: tuple[str, ...] = ()
    populated_n_max: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the highest Fock index populated above the boundary threshold during any pulse."""
    margin_reached: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the smallest margin (levels) the cap kept above the populated range during the pulses."""
    kernel: str = "none"
    """``factorized`` or ``assembled`` when every segment with drive terms agrees, else ``mixed``; ``none`` without any."""
    map: str = "serial"
    """The map the trajectories ran through."""
    workers: int = 1
    """Processes the trajectory map used (1 in-process)."""
    propagator_solves: int = 0
    propagator_cache_hits: int = 0
    trajectory_finals: tuple[Qobj, ...] = ()
    """On the trajectory path, the final ket of every trajectory after the last segment, in the keyed order."""
    trajectory_seeds: tuple[tuple[int, ...], ...] = ()
    """The spawn keys of the trajectories' seeds on the last trajectory segment, in the same order."""

    @property
    def approximations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for seg in self.segments:
            for a in seg.approximations:
                if a not in seen:
                    seen.append(a)
        for a in self.hardware_notes + self.notes:
            if a not in seen:
                seen.append(a)
        return tuple(seen)

    @property
    def channel_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for seg in self.segments:
            for c in seg.channels:
                if c not in seen:
                    seen.append(c)
        return tuple(seen)


def _channel_kind(name: str) -> str:
    return name.split("[")[0]


@dataclass
class JointExactEngine:
    """JOINT_EXACT pulse engine: every pulse through the one builder, the joint state evolved exactly.

    ``store_per_segment`` counts stored points per segment (endpoints included); ``device_channels`` adds the device's
    collapse operators and, per segment, the active pulses' scattering and intensity-noise ones; ``levels_by_ion`` maps the
    levels of ions with d > 2. ``last_report`` carries the diagnostics of the most recent run.
    """

    builder_options: object | None = None
    store_per_segment: int = 2
    store_times_s: tuple[float, ...] = ()
    """Extra absolute times at which the traces are stored (the experiments' scan points)."""
    channels: tuple[CollapseOp, ...] = ()
    qubit_shifts_hz: dict[int, float] = field(default_factory=dict)
    max_growth_retries: int = 3
    growth_levels: int = 4
    device_channels: bool = False
    levels_by_ion: dict[int, InternalLevels] | None = None
    hardware_chain: bool = True
    table: CalibrationTable | None = None
    """The CalibrationTable the schedule's drives were built from: when given, ``control.played`` converts the requested
    Rabi frequencies, believed Stark shifts and crosstalk into what the ions see; None plays the schedule as physical."""
    pure_branches: bool = True
    """Evolve a density-matrix input without collapse operators as its weighted pure eigen-branches through ``sesolve``,
    dropping (and reporting) branches below ``SolverOptions.branch_weight_min``."""
    closed_form_constant: bool = True
    """Propagate a segment whose Hamiltonian is constant by its exact propagator instead of the ODE ladder (with
    eigenoperator collapse operators, the master equation in the frame rotating with H); it reports integrator ``exact``."""
    last_report: EngineReport | None = None
    progress: Callable[[Progress], None] | None = field(default=None, repr=False, compare=False)
    """Called with ``Progress("pulse", done, total, elapsed_s)`` after every pulse segment; not shipped to workers."""
    _propagators: dict[tuple[object, ...], _Propagator] = field(
        default_factory=dict, repr=False, compare=False
    )
    """The propagator cache, keyed by the built Hamiltonian's fingerprint, the stored times and the tolerances."""

    def __getstate__(self) -> dict[str, object]:
        # an engine shipped to a worker (the parallel tomography and run() maps) carries neither its report nor its cache
        state = dict(self.__dict__)
        state["last_report"] = None
        state["_propagators"] = {}
        state["progress"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)

    def process_tomography(
        self,
        device: Device,
        pulse: Pulse | Sequence[Pulse] | Schedule,
        space: HilbertSpace,
        motional_model: MotionalModel,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions | None = None,
        *,
        ideal: np.ndarray | None = None,
    ) -> ChannelSummary:
        """Process tomography of a pulse, pulse group or Schedule on ``space``, summarized against ``ideal`` (the unitary on
        the space's ions in factor order; NaN infidelities without one); :meth:`tomography` returns the full record."""
        rec = self.tomography(device, pulse, space, motional_model, sample, seeds, options)
        return rec.summary(ideal)

    def tomography(
        self,
        device: Device,
        pulse: Pulse | Sequence[Pulse] | Schedule,
        space: HilbertSpace,
        motional_model: MotionalModel,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions | None = None,
    ) -> TomographyRecord:
        """The process tomography behind :meth:`process_tomography`, with everything GATE_LOCAL tracks."""
        from qutip_trap.dynamics.tomography import tomography as _tomography

        return _tomography(
            self, device, pulse, space, motional_model, sample, seeds, options or SolverOptions()
        )

    def run_pulses(
        self,
        device: Device,
        schedule: Schedule,
        state: State,
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions,
    ) -> Traces:
        from qutip_trap.hilbert.truncation import regrid_state

        current_space = space
        current_state = state
        retries = 0
        growth_notes: list[str] = []
        while True:
            try:
                return self._run(
                    device,
                    schedule,
                    current_state,
                    current_space,
                    sample,
                    seeds,
                    options,
                    retries,
                    tuple(growth_notes),
                )
            except _BoundaryTrip as trip:
                if retries >= self.max_growth_retries:
                    if trip.reason == "margin":
                        raise TruncationLimit(
                            f"the cap of mode {trip.mode} keeps {trip.add} level(s) less margin above the populated range than "
                            f"Section 5.1.1 requires after {retries} cap-raising retries"
                        ) from trip
                    raise TruncationLimit(
                        f"boundary population {trip.worst:.3e} on mode {trip.mode} still exceeds "
                        f"{options.boundary_population_max:.1e} after {retries} cap-raising retries"
                    ) from trip
                retries += 1
                # a margin trip knows its deficit exactly and grows by it; a boundary trip grows by the configured step
                grow = trip.add if trip.reason == "margin" and trip.add > 0 else self.growth_levels
                new_space = current_space.grown(trip.mode, grow)
                growth_notes.append(
                    f"cap-raising retry {retries} of {self.max_growth_retries}: {trip.reason} trip on mode {trip.mode} "
                    f"(population {trip.worst:.3e}) grows its cap by {grow} level(s); the run is integrated again on joint "
                    f"dimension {new_space.dimension} (was {current_space.dimension})"
                )
                if new_space.dimension > options.joint_dimension_max:
                    # the size guard holds for a grown space too (a hot mode's cap can keep growing)
                    raise TruncationLimit(
                        f"raising the cap of mode {trip.mode} after a {trip.reason} trip would take the joint space to "
                        f"dimension {new_space.dimension}, above joint_dimension_max = {options.joint_dimension_max} "
                        f"(Section 11.5); the {trip.reason} population was {trip.worst:.3e}. Cool or freeze the mode, raise "
                        "the guard deliberately, or let level='auto' route the run to GATE_LOCAL"
                    ) from trip
                joint = current_state.joint
                if joint is None:
                    raise
                new_joint = regrid_state(joint, current_space, new_space)
                current_state = replace(current_state, joint=new_joint)
                current_space = new_space

    # ---- internals ---------------------------------------------------------------------------------------------------

    def _segment_propagator(
        self, built: object, times: np.ndarray, options: SolverOptions, largest_mode: int
    ) -> tuple[_Propagator, bool]:
        """U(t_k, t_0) at every stored time of a segment on an internal-state-only space, from the cache or by one integration of
        the identity through the ladder. Returns (propagator, cache hit)."""
        from qutip_trap.dynamics.evolve import evolve

        h = built.H  # type: ignore[attr-defined]
        key = (
            built.fingerprint,  # type: ignore[attr-defined]
            tuple(float(x) for x in np.round(times, 15)),
            options.atol,
            options.rtol,
            options.integrators,
            largest_mode,
        )
        cached = self._propagators.get(key)
        if cached is not None:
            return cached, True
        dims = [int(d) for d in h.dims[0]]
        identity = qt.Qobj(np.eye(h.shape[0], dtype=complex), dims=[dims, dims])
        ev = evolve(
            h,
            identity,
            times,
            options=options,
            store_states=True,
            omega_max_rad_s=built.omega_max_rad_s or None,  # type: ignore[attr-defined]
            largest_mode_dimension=largest_mode,
            counter_calls=built.counter.count,  # type: ignore[attr-defined]
            calls_per_rhs=built.n_drive_terms,  # type: ignore[attr-defined]
            propagator=True,
        )
        assert ev.states is not None
        prop = _Propagator(
            unitaries=tuple(np.asarray(st.full()) for st in ev.states),
            integrator=ev.integrator,
            atol=ev.atol,
            rhs_evaluations=ev.rhs_evaluations,
            retries=ev.retries,
        )
        if len(self._propagators) >= PROPAGATOR_CACHE_MAX:
            self._propagators.clear()
        self._propagators[key] = prop
        return prop, False

    def _segment_channels(
        self,
        device: Device,
        active: list[Pulse],
        space: HilbertSpace,
        built: object,
        static: list[CollapseOp],
        options: SolverOptions,
        notes: list[str],
    ) -> list[CollapseOp]:
        from qutip_trap.dynamics.channels import intensity_noise_channels
        from qutip_trap.noise.scattering import ScatteringOptions, scattering_channels

        ops = list(static)
        if not self.device_channels:
            return ops
        if options.scattering_channels:
            sopts = ScatteringOptions(recoil=options.scattering_recoil)
            for p in active:
                more, more_notes = scattering_channels(
                    device, p, space, levels_by_ion=self.levels_by_ion, options=sopts
                )
                ops.extend(more)
                for n in more_notes:
                    if n not in notes:
                        notes.append(n)
        density = device.noise.intensity_white_density()
        parts = getattr(built, "drive_parts", {})
        if options.intensity_noise_channels and density > 0.0 and parts:
            dens: dict[str, float] = {}
            for p in active:
                key = p.gate_id or f"pulse@{p.t_start_s:.9g}"
                if p.drive.kind in ("raman", "light_shift"):
                    dens[key] = density
                elif p.drive.kind in ("optical_E1", "optical_E2"):
                    dens[key] = 0.25 * density
            ops.extend(intensity_noise_channels(parts, dens))
        return ops

    def _played_schedule(
        self,
        device: Device,
        schedule: Schedule,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions,
        notes: list[str],
    ) -> tuple[Schedule, tuple[str, ...]]:
        """The schedule the ions see: the played chain (requested -> physical, when a table is given), then the hardware
        chain. Returns (schedule, hardware notes); the played chain's notes are appended to ``notes``."""
        from qutip_trap.control.hardware import apply_hardware_chain

        sched = schedule
        hw_notes: tuple[str, ...] = ()
        if self.table is not None:
            from qutip_trap.control.played import physical_schedule

            sched, played_notes = physical_schedule(device, sched, self.table)
            notes.extend(played_notes)
        if self.hardware_chain and options.hardware_chain:
            rng_jitter = np.random.default_rng(seeds.child(sample.sample_id, 0, 0, 0, "timing_jitter"))
            sched, hw_notes = apply_hardware_chain(sched, device.hardware, rng=rng_jitter)
        return sched, hw_notes

    @staticmethod
    def _frozen_fock_states(
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        nbar: Mapping[int, float],
        notes: list[str],
    ) -> dict[int, int]:
        """The frozen spectators' Fock states for this evolution, from the sample where the caller put them (``run()``
        enumerates them as weighted branches); otherwise drawn thermally, keyed per sample, and noted."""
        from qutip_trap.hilbert.operators import thermal_populations
        from qutip_trap.noise.sampling import key_frozen_n

        frozen_n: dict[int, int] = {}
        for m in space.frozen:
            key = key_frozen_n(m)
            if key in sample.values:
                frozen_n[m] = int(round(sample.values[key]))
            else:
                nbar_m = float(nbar.get(m, 0.0))
                if nbar_m <= 0.0:
                    frozen_n[m] = 0
                else:
                    rng = np.random.default_rng(seeds.child(sample.sample_id, 0, 0, 0, key))
                    probs = thermal_populations(nbar_m, int(60 + 40 * nbar_m))
                    probs = probs / probs.sum()
                    frozen_n[m] = int(rng.choice(len(probs), p=probs))
                    notes.append(
                        f"frozen mode {m}: no Fock state given for this evolution, so n = {frozen_n[m]} was drawn from the "
                        f"thermal distribution at nbar = {nbar_m:.4g}, keyed per SAMPLE (run() enumerates the branches "
                        "instead, which is the per-shot mechanism of Section 5.2)"
                    )
        return frozen_n

    def _collapse_setup(
        self, device: Device, space: HilbertSpace, options: SolverOptions
    ) -> tuple[list[CollapseOp], bool]:
        """The state-independent collapse operators of a run on ``space``, and whether a pulse segment can carry pulse-built
        ones; decided before any build so that drive operators are assembled exactly where a Liouvillian is formed."""
        static_ops: list[CollapseOp] = list(self.channels)
        if self.device_channels:
            static_ops.extend(device.noise.channels(device, space))
        pulse_channels_possible = bool(self.device_channels) and (
            bool(options.scattering_channels)
            or (bool(options.intensity_noise_channels) and device.noise.intensity_white_density() > 0.0)
        )
        return static_ops, pulse_channels_possible

    def is_unitary(self, device: Device, space: HilbertSpace, options: SolverOptions) -> bool:
        """Whether every segment of a run on ``space`` evolves unitarily (no explicit, device or pulse-built channel), so
        the final state is linear in the initial ket: the condition for the tomography's isometry route."""
        static_ops, pulse_channels_possible = self._collapse_setup(device, space, options)
        return not static_ops and not pulse_channels_possible

    @staticmethod
    def _segment_edges(sched: Schedule) -> list[float]:
        """Segment cuts at every pulse and idle boundary, from the schedule's declared start (``Schedule.t0_s``) or
        min(0, the first cut) to ``pulses_end_s`` (a following measurement belongs to the readout stage)."""
        starts = [p.t_start_s for p in sched.pulses] + [a for a, _ in sched.idle]
        t0 = float(sched.t0_s) if sched.t0_s is not None else min(starts + [0.0])
        t_end = max(sched.pulses_end_s, t0)
        cuts = {t0, t_end}
        for p in sched.pulses:
            cuts.update((p.t_start_s, p.t_end_s))
        for a, b in sched.idle:
            cuts.update((a, b))
        return _merge_cuts(sorted(t for t in cuts if t0 <= t <= t_end))

    def _segment_times(self, a: float, b: float) -> np.ndarray:
        """The stored times of one segment: ``store_per_segment`` points from ``a`` to ``b`` plus the caller's ``store_times_s``
        inside it (the same array on every path, so the propagator cache's key is shared between them)."""
        n_store = max(int(self.store_per_segment), 2)
        extra = [t for t in self.store_times_s if a < t < b]
        return np.array(sorted(set(np.linspace(a, b, n_store).tolist()) | set(extra)))

    def propagator(
        self,
        device: Device,
        schedule: Schedule,
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions,
        *,
        motional_model: MotionalModel | None = None,
    ) -> tuple[np.ndarray, EngineReport]:
        """U(t_end, t_0) of ``schedule`` on an internal-state-only ``space`` as a D x D matrix, with the report of a run.

        ``motional_model`` supplies occupations for frozen modes the sample carries no Fock state for. Raises
        ``ValueError`` on a space with a resolved mode or an ENR group, or when a segment would carry a collapse operator.
        """
        from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian

        if space.resolved or space.enr_group is not None:
            raise ValueError(
                "propagator() takes an internal-state-only space (every mode frozen, no ENR group); a space with a resolved "
                "mode is a joint space, whose propagator is never formed (Section 11.3 item 5)"
            )
        bopts = self.builder_options if isinstance(self.builder_options, BuilderOptions) else BuilderOptions()
        notes: list[str] = []
        sched, hw_notes = self._played_schedule(device, schedule, sample, seeds, options, notes)
        nbar = motional_model.nbar if motional_model is not None else {}
        frozen_n = self._frozen_fock_states(space, sample, seeds, nbar, notes)
        static_ops, pulse_channels_possible = self._collapse_setup(device, space, options)
        edges = self._segment_edges(sched)
        u = np.eye(space.dimension, dtype=complex)
        segments: list[SegmentReport] = []
        records: list[object] = []
        solves = 0
        hits = 0
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            active = [p for p in sched.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
            if static_ops or (active and pulse_channels_possible):
                raise ValueError(
                    "the segment carries collapse operators, so its propagator is not a unitary: propagate states through "
                    "run_pulses (is_unitary() says which)"
                )
            built = build_hamiltonian(
                device,
                active,
                space,
                sample=sample,
                options=bopts,
                qubit_shifts_hz=self.qubit_shifts_hz,
                frozen_n=frozen_n,
            )
            records.extend(built.records)
            times = self._segment_times(a, b)
            u_seg: np.ndarray | None = None
            integrator = "exact"
            atol_used = options.atol
            rhs_evals: int | None = None
            retries_seg: tuple[str, ...] = ()
            if self.closed_form_constant and built.H.isconstant:
                u_seg = _constant_unitary(built.H(a), b - a)
            if u_seg is None:
                prop, hit = self._segment_propagator(built, times, options, 0)
                hits += int(hit)
                solves += int(not hit)
                u_seg = prop.unitaries[-1]
                integrator = "propagator[cached]" if hit else f"{prop.integrator}[propagator]"
                atol_used = prop.atol
                rhs_evals = None if hit else prop.rhs_evaluations
                retries_seg = prop.retries
            u = u_seg @ u
            segments.append(
                SegmentReport(
                    a,
                    b,
                    tuple(p.gate_id for p in active),
                    integrator,
                    atol_used,
                    rhs_evals,
                    _steps_per_period(rhs_evals, built.omega_max_rad_s, b - a),
                    built.approximations,
                    {},
                    retries_seg,
                    method="sesolve",
                    kernel=built.kernel,
                )
            )
        report = EngineReport(
            segments=tuple(segments),
            frozen_n=frozen_n,
            growth_retries=0,
            space=space,
            drive_records=tuple(records),
            method="sesolve",
            hardware_notes=hw_notes,
            schedule_played=sched,
            notes=tuple(notes),
            kernel=_kernel_summary(segments),
            propagator_solves=solves,
            propagator_cache_hits=hits,
        )
        self.last_report = report
        return u, report

    def _run(
        self,
        device: Device,
        schedule: Schedule,
        state: State,
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: SolverOptions,
        growth_retries: int,
        growth_notes: tuple[str, ...] = (),
    ) -> Traces:
        from qutip_trap.dynamics.evolve import LARGE_MODE_DIMENSION, evolve
        from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
        from qutip_trap.hilbert.truncation import boundary_populations
        from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT

        bopts = self.builder_options if isinstance(self.builder_options, BuilderOptions) else BuilderOptions()
        joint = state.joint
        if joint is None:
            raise ValueError("JOINT_EXACT needs a joint state; build one with HilbertSpace.initial_state")
        if joint.shape[0] != space.dimension:
            raise ValueError("the state does not live on the given space")
        notes: list[str] = list(growth_notes)
        sched, hw_notes = self._played_schedule(device, schedule, sample, seeds, options, notes)
        frozen_n = self._frozen_fock_states(space, sample, seeds, state.motional.nbar, notes)
        static_ops, pulse_channels_possible = self._collapse_setup(device, space, options)
        lindblad_resolved: str = options.lindblad_method
        if lindblad_resolved == "auto":
            lindblad_resolved = "mesolve" if space.dimension <= options.mesolve_dimension_max else "mcsolve"
        map_kind = options.map
        n_workers = worker_count(options)
        workers_used = 1
        map_used = "serial"
        propagator_solves = 0
        propagator_hits = 0
        trajectory_finals: list[qt.Qobj] = []
        trajectory_seeds: list[tuple[int, ...]] = []
        edges = self._segment_edges(sched)
        t0 = edges[0]
        e_keys: list[str] = []
        e_list: list[qt.Qobj] = []
        for i in space.ion_labels:
            e_keys.append(f"P1[{i}]")
            e_list.append(space.projector(i, 1))
        carried = [m.mode for m in space.resolved] + (list(space.enr_group[0]) if space.enr_group else [])
        for m in carried:
            e_keys.append(f"n[{m}]")
            e_list.append(space.number(m))
            e_keys.append(f"a[{m}]")
            e_list.append(space.annihilation(m))
        times_all: list[np.ndarray] = []
        expect_all: dict[str, list[np.ndarray]] = {k: [] for k in e_keys}
        reduced: list[qt.Qobj] = []
        marginals: dict[int, list[np.ndarray]] | None = (
            {m: [] for m in carried} if options.store_marginals else None
        )
        wall_by_pulse: dict[str, float] = {}
        segments: list[SegmentReport] = []
        records: list[object] = []
        jumps: list[tuple[float, str]] = []
        worst_boundary: dict[int, float] = {m: 0.0 for m in carried}
        populated_max: dict[int, int] = {}
        margin_reached: dict[int, int] = {}
        # the state: a list of weighted kets (pure branches, or equal-weight trajectories), or one density matrix
        kets: list[qt.Qobj] | None = [joint] if joint.isket else None
        weights: list[float] = [1.0] if joint.isket else []
        rho: qt.Qobj | None = None if joint.isket else joint
        if (
            rho is not None
            and self.pure_branches
            and not self.channels
            and not self.device_channels
            and rho.shape[0] <= EIGH_DIMENSION_MAX
        ):
            kets, weights, dropped = _pure_branches(rho, options.branch_weight_min)
            rho = None
            if dropped > 0.0:
                notes.append(
                    f"initial mixture evolved as {len(kets)} pure branches (Section 5.3): branches below branch_weight_min = "
                    f"{options.branch_weight_min:g} dropped, total weight {dropped:.3e} (renormalized)"
                )
        method_used = "sesolve" if kets is not None else "mesolve"
        first = True
        largest_mode = max([m.d for m in space.resolved], default=0)
        atol_mc = options.atol if largest_mode <= LARGE_MODE_DIMENSION else max(options.atol, 1e-8)
        # improved_sampling splits the whole evolution, so it applies only when one segment takes the trajectory path
        n_mc_segments = 0
        if lindblad_resolved == "mcsolve" and kets is not None:
            for a_e, b_e in zip(edges[:-1], edges[1:]):
                if b_e <= a_e:
                    continue
                act_e = [p for p in sched.pulses if p.t_start_s <= a_e + 1e-15 and p.t_end_s >= b_e - 1e-15]
                if bool(static_ops) or (bool(act_e) and pulse_channels_possible):
                    n_mc_segments += 1
        improved_run = bool(options.improved_sampling) and n_mc_segments == 1
        if bool(options.improved_sampling) and n_mc_segments > 1:
            notes.append(IMPROVED_SAMPLING_MULTI_SEGMENT.format(n=n_mc_segments))
        target_tol_estimate: int | None = None
        pulse_total = sum(
            1
            for a_e, b_e in zip(edges[:-1], edges[1:])
            if b_e > a_e
            and any(p.t_start_s <= a_e + 1e-15 and p.t_end_s >= b_e - 1e-15 for p in sched.pulses)
        )
        pulse_done = 0
        progress_started = time.perf_counter()
        for seg_index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if b <= a:
                continue
            seg_started = time.perf_counter()
            seg_marg: dict[int, np.ndarray] | None = None
            active = [p for p in sched.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
            dissipative_seg = bool(static_ops) or (bool(active) and pulse_channels_possible)
            mesolve_seg = kets is None or (dissipative_seg and lindblad_resolved == "mesolve")
            bopts_seg = (
                replace(bopts, kernel="assembled") if (mesolve_seg and bopts.kernel != "assembled") else bopts
            )
            built = build_hamiltonian(
                device,
                active,
                space,
                sample=sample,
                options=bopts_seg,
                qubit_shifts_hz=self.qubit_shifts_hz,
                frozen_n=frozen_n,
            )
            records.extend(built.records)
            seg_ops = self._segment_channels(device, active, space, built, static_ops, options, notes)
            c_ops = [c.op for c in seg_ops]
            times = self._segment_times(a, b)
            sel = slice(1, None) if not first else slice(0, None)
            seg_method = "sesolve"
            integrator = options.integrators[0]
            atol_used = options.atol
            rhs_evals: int | None = None
            retries_seg: tuple[str, ...] = ()
            # ---- integrate ----------------------------------------------------------------------------------------
            # ket paths integrate in the exact rotating frame; mesolve segments and closed forms do not
            rot: RotatingSegment | None = None
            if (
                options.rotating_frame
                and kets is not None
                and not mesolve_seg
                and not built.H.isconstant
                and (bool(space.resolved) or space.enr_group is not None)
            ):
                rot = rotating_frame(built.H, space.dims, c_ops)
                if rot is not None:
                    for n in rot.notes:
                        if n not in notes:
                            notes.append(n)
            # per e_op, the phase lambda back from the rotating frame; None: the trace is taken on the rotated-back states
            e_phases: dict[str, float | None] = (
                {k: expectation_phase(op, rot.frame) for k, op in zip(e_keys, e_list)}
                if rot is not None
                else {}
            )
            closed: _ClosedForm | None = None
            if (
                self.closed_form_constant
                and built.H.isconstant
                and all(isinstance(c, qt.Qobj) for c in c_ops)
            ):
                lindblad = options.lindblad_method
                if lindblad == "auto":
                    lindblad = "mesolve" if space.dimension <= options.mesolve_dimension_max else "mcsolve"
                closed = _closed_form_segment(
                    built.H(a),
                    c_ops,
                    kets,
                    weights,
                    rho,
                    times,
                    space,
                    e_keys,
                    e_list,
                    options,
                    lindblad,
                    largest_mode,
                )
            if closed is not None:
                kets, rho = closed.kets, closed.rho
                for k in e_keys:
                    expect_all[k].append(closed.expect[k][sel])
                reduced.extend(closed.reduced[sel])
                if marginals is not None:
                    assert closed.marginals is not None
                    seg_marg = {m: closed.marginals[m][sel] for m in carried}
                seg_method = closed.method
                integrator = closed.integrator
                atol_used = closed.atol
                rhs_evals = None
                retries_seg = closed.retries
            elif not c_ops:
                if (
                    kets is not None
                    and options.propagator_cache
                    and not space.resolved
                    and space.enr_group is None
                ):
                    # an internal-state-only space: one propagator serves every initial state
                    prop, hit = self._segment_propagator(built, times, options, largest_mode)
                    propagator_hits += int(hit)
                    propagator_solves += int(not hit)
                    kets_p: list[qt.Qobj] = []
                    exp_p = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
                    red_p: list[np.ndarray] = []
                    for psi, w_k in zip(kets, weights):
                        vec = np.asarray(psi.full()).reshape(-1)
                        states_k = [qt.Qobj((u @ vec).reshape(-1, 1), dims=psi.dims) for u in prop.unitaries]
                        kets_p.append(states_k[-1])
                        for k, op in zip(e_keys, e_list):
                            exp_p[k] += w_k * np.array([qt.expect(op, st) for st in states_k], dtype=complex)
                        red_p.append(
                            w_k * np.array([space.internal_marginal(st).full() for st in states_k[sel]])
                        )
                        if marginals is not None:
                            seg_marg = _weighted_marginals(
                                seg_marg, w_k, _marginals_of(space, states_k[sel], carried)
                            )
                    kets = kets_p
                    for k in e_keys:
                        expect_all[k].append(exp_p[k][sel])
                    dims_int = [list(space.ion_dims), list(space.ion_dims)]
                    for arr in np.sum(np.stack(red_p), axis=0):
                        reduced.append(qt.Qobj(arr, dims=dims_int))
                    integrator = "propagator[cached]" if hit else f"{prop.integrator}[propagator]"
                    atol_used = prop.atol
                    rhs_evals = None if hit else prop.rhs_evaluations
                    retries_seg = prop.retries
                    seg_method = "sesolve"
                elif kets is not None:
                    new_kets: list[qt.Qobj] = []
                    exp_acc = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
                    red_acc: list[np.ndarray] = []
                    for psi, w_k in zip(kets, weights):
                        ev = evolve(
                            built.H if rot is None else rot.H,
                            psi if rot is None else rot.frame.to_frame(psi, float(times[0])),
                            times,
                            e_ops=dict(zip(e_keys, e_list)),
                            options=options,
                            store_states=True,
                            omega_max_rad_s=built.omega_max_rad_s or None,
                            largest_mode_dimension=largest_mode,
                            counter_calls=built.counter.count,
                            calls_per_rhs=built.n_drive_terms,
                        )
                        assert ev.states is not None
                        if rot is None:
                            states_back: list[qt.Qobj] = list(ev.states)
                            final_k = ev.final
                        else:
                            states_back = [
                                rot.frame.from_frame(st, float(t)) for st, t in zip(ev.states, times)
                            ]
                            final_k = rot.frame.from_frame(ev.final, float(times[-1]))
                        new_kets.append(final_k)
                        for k, op in zip(e_keys, e_list):
                            exp_acc[k] += w_k * _expectation_back(
                                np.asarray(ev.expect[k]), e_phases.get(k, 0.0), times, op, states_back
                            )
                        red_acc.append(
                            w_k * np.array([space.internal_marginal(st).full() for st in states_back[sel]])
                        )
                        if marginals is not None:
                            seg_marg = _weighted_marginals(
                                seg_marg, w_k, _marginals_of(space, states_back[sel], carried)
                            )
                        integrator, atol_used, rhs_evals, retries_seg = (
                            ev.integrator,
                            ev.atol,
                            ev.rhs_evaluations,
                            ev.retries,
                        )
                    kets = new_kets
                    for k in e_keys:
                        expect_all[k].append(exp_acc[k][sel])
                    dims_int = [list(space.ion_dims), list(space.ion_dims)]
                    for arr in np.sum(np.stack(red_acc), axis=0):
                        reduced.append(qt.Qobj(arr, dims=dims_int))
                    seg_method = "sesolve"
                else:
                    assert rho is not None
                    ev = evolve(
                        built.H,
                        rho,
                        times,
                        e_ops=dict(zip(e_keys, e_list)),
                        options=options,
                        store_states=True,
                        omega_max_rad_s=built.omega_max_rad_s or None,
                        largest_mode_dimension=largest_mode,
                        counter_calls=built.counter.count,
                        calls_per_rhs=built.n_drive_terms,
                    )
                    rho = ev.final
                    for k in e_keys:
                        expect_all[k].append(np.asarray(ev.expect[k])[sel])
                    assert ev.states is not None
                    for st in ev.states[sel]:
                        reduced.append(space.internal_marginal(st))
                    if marginals is not None:
                        seg_marg = _marginals_of(space, list(ev.states[sel]), carried)
                    integrator, atol_used, rhs_evals, retries_seg = (
                        ev.integrator,
                        ev.atol,
                        ev.rhs_evaluations,
                        ev.retries,
                    )
                    seg_method = "mesolve"
            else:
                method: str = options.lindblad_method
                if method == "auto":
                    method = "mesolve" if space.dimension <= options.mesolve_dimension_max else "mcsolve"
                if method == "mesolve":
                    if kets is not None:
                        rho = _mixture(kets, weights)
                        kets = None
                    assert rho is not None
                    ev = evolve(
                        built.H,
                        rho,
                        times,
                        c_ops=c_ops,
                        e_ops=dict(zip(e_keys, e_list)),
                        options=options,
                        store_states=True,
                        omega_max_rad_s=built.omega_max_rad_s or None,
                        largest_mode_dimension=largest_mode,
                        counter_calls=built.counter.count,
                        calls_per_rhs=built.n_drive_terms,
                    )
                    rho = ev.final
                    for k in e_keys:
                        expect_all[k].append(np.asarray(ev.expect[k])[sel])
                    assert ev.states is not None
                    for st in ev.states[sel]:
                        reduced.append(space.internal_marginal(st))
                    if marginals is not None:
                        seg_marg = _marginals_of(space, list(ev.states[sel]), carried)
                    integrator, atol_used, rhs_evals, retries_seg = (
                        ev.integrator,
                        ev.atol,
                        ev.rhs_evaluations,
                        ev.retries,
                    )
                    seg_method = "mesolve"
                else:
                    if kets is None:
                        raise RuntimeError(
                            "the trajectory path (mcsolve) needs pure trajectories: the state is a density matrix; enumerate the "
                            "initial mixture into pure branches (run()) or raise mesolve_dimension_max"
                        )
                    improved_seg = improved_run and len(kets) == 1
                    n_traj_seg = options.ntraj
                    # placeholders for the phase-one probe; the map and workers are fixed once n_traj_seg is final
                    seg_map, seg_workers = "serial", 1
                    mc_opts = {
                        "method": options.integrators[0],
                        "atol": atol_mc,
                        "rtol": options.rtol,
                        "nsteps": options.nsteps,
                        "store_final_state": True,
                        "store_states": True,
                        "keep_runs_results": True,
                        "norm_t_tol": 1e-8 * (b - a),
                        "norm_tol": 1e-6,
                        "norm_steps": 50,
                        "progress_bar": "",
                        "map": seg_map,
                        "num_cpus": seg_workers,
                        "improved_sampling": improved_seg,
                    }
                    # phase one: the trajectory count from target_tol, under a serial map so that it is reproducible
                    if (
                        len(kets) == 1
                        and options.e_ops_for_target_tol
                        and options.trajectory_target_tol is not None
                        and e_list
                    ):
                        if target_tol_estimate is None:
                            probe = qt.MCSolver(
                                built.H if rot is None else rot.H,
                                c_ops if rot is None else list(rot.c_ops),
                                options={
                                    **mc_opts,
                                    "map": "serial",
                                    "num_cpus": 1,
                                    "keep_runs_results": False,
                                    "store_states": False,
                                    "improved_sampling": False,
                                },
                            ).run(
                                kets[0] if rot is None else rot.frame.to_frame(kets[0], float(times[0])),
                                times,
                                ntraj=options.ntraj,
                                e_ops=e_list,
                                target_tol=float(options.trajectory_target_tol),
                            )
                            target_tol_estimate = max(
                                int(probe.num_trajectories), TARGET_TOL_MIN_TRAJECTORIES
                            )
                            notes.append(
                                f"trajectory count from mcsolve target_tol = {options.trajectory_target_tol:g} on the "
                                f"population e_ops: {int(probe.num_trajectories)} estimated, "
                                f"{target_tol_estimate} replayed from the keyed seed list (Section 3.4, two phases)"
                            )
                        n_traj_seg = min(target_tol_estimate, options.ntraj)
                    if len(kets) == 1 and n_traj_seg > 1 and not improved_seg:
                        kets = [kets[0]] * n_traj_seg
                        weights = [1.0 / n_traj_seg] * n_traj_seg
                    n_stoch = n_traj_seg if improved_seg else len(kets)
                    # one worker runs in-process: a one-process pool would still fork a copy of this process per segment
                    seg_map = map_kind if (n_stoch > 1 and n_workers > 1) else "serial"
                    seg_workers = min(n_workers, n_stoch) if seg_map != "serial" else 1
                    mc_opts["map"], mc_opts["num_cpus"] = seg_map, seg_workers
                    workers_used = max(workers_used, seg_workers)
                    if seg_map != "serial":
                        map_used = seg_map
                    solver = qt.MCSolver(
                        built.H if rot is None else rot.H,
                        c_ops if rot is None else list(rot.c_ops),
                        options=mc_opts,
                    )
                    kets_in = kets if rot is None else [rot.frame.to_frame(k, float(times[0])) for k in kets]
                    # one keyed-seed trajectory per ket; results arrive in completion order and are matched back by seed
                    seeds_k = [
                        seeds.child(sample.sample_id, k_traj, 0, 0, f"mcsolve[{seg_index}]")
                        for k_traj in range(n_stoch)
                    ]
                    if improved_seg:
                        res = solver.run(kets_in[0], times, ntraj=n_stoch, e_ops=e_list, seeds=seeds_k)
                    elif len(kets) == 1:
                        res = solver.run(kets_in[0], times, ntraj=1, e_ops=e_list, seeds=seeds_k)
                    else:
                        res = solver.run(
                            [(psi, 1.0 / len(kets)) for psi in kets_in],
                            times,
                            ntraj=[1] * len(kets),
                            e_ops=e_list,
                            seeds=seeds_k,
                        )
                    by_seed = {tuple(int(x) for x in sd.spawn_key): j for j, sd in enumerate(res.seeds)}
                    new_kets = []
                    new_weights: list[float] = []
                    exp_acc = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
                    red_acc = []
                    trajectory_seeds = []
                    # (trajectory, weight, seed key or None for a deterministic member, index into res.col_* or None)
                    members: list[tuple[Any, float, tuple[int, ...] | None, int | None]] = []
                    if improved_seg:
                        # weighted: the no-jump member carries p_no-jump (a uniform draw would bias per-shot observables)
                        members.extend(
                            (traj, float(w), None, None)
                            for traj, w in zip(res.deterministic_trajectories, res.deterministic_weights)
                        )
                        for k_traj in range(n_stoch):
                            key_k = tuple(int(x) for x in seeds_k[k_traj].spawn_key)
                            j = by_seed[key_k]
                            members.append((res.trajectories[j], float(res.runs_weights[j]), key_k, j))
                    else:
                        for k_traj, w_k in enumerate(weights):
                            key_k = tuple(int(x) for x in seeds_k[k_traj].spawn_key)
                            j = by_seed[key_k]
                            members.append((res.trajectories[j], float(w_k), key_k, j))
                    for k_traj, (traj_m, w_m, key_m, j_m) in enumerate(members):
                        if rot is None:
                            states_m: list[qt.Qobj] = list(traj_m.states)
                            final_m = traj_m.final_state
                        else:
                            states_m = [
                                rot.frame.from_frame(st, float(t)) for st, t in zip(traj_m.states, times)
                            ]
                            final_m = rot.frame.from_frame(traj_m.final_state, float(times[-1]))
                        new_kets.append(final_m)
                        new_weights.append(w_m)
                        if key_m is not None:
                            trajectory_seeds.append(key_m)
                        for idx, (k, op) in enumerate(zip(e_keys, e_list)):
                            exp_acc[k] += w_m * _expectation_back(
                                np.asarray(traj_m.expect[idx]), e_phases.get(k, 0.0), times, op, states_m
                            )
                        red_acc.append(
                            w_m * np.array([space.internal_marginal(st).full() for st in states_m[sel]])
                        )
                        if marginals is not None:
                            seg_marg = _weighted_marginals(
                                seg_marg, w_m, _marginals_of(space, states_m[sel], carried)
                            )
                        if j_m is not None:
                            for t_c, which in zip(res.col_times[j_m], res.col_which[j_m]):
                                jumps.append((float(t_c), f"traj{k_traj}:{seg_ops[int(which)].channel}"))
                    kets = new_kets
                    weights = new_weights
                    for k in e_keys:
                        expect_all[k].append(exp_acc[k][sel])
                    dims_int = [list(space.ion_dims), list(space.ion_dims)]
                    for arr in np.sum(np.stack(red_acc), axis=0):
                        reduced.append(qt.Qobj(arr, dims=dims_int))
                    integrator, atol_used = options.integrators[0], atol_mc
                    # the coefficient counter lives in this process: under a parallel map the workers' calls are not seen
                    rhs_evals = built.rhs_evaluations if seg_map == "serial" else None
                    seg_method = "mcsolve"
                    method_used = "mcsolve"
            if seg_method == "mesolve":
                method_used = "mesolve" if method_used != "mcsolve" else method_used
            times_all.append(times[sel])
            if marginals is not None:
                assert seg_marg is not None or not carried
                for m in carried:
                    assert seg_marg is not None
                    marginals[m].append(seg_marg[m])
            first = False
            seg_wall = time.perf_counter() - seg_started
            for key_w in [p.gate_id or "pulse" for p in active] or ["idle"]:
                wall_by_pulse[key_w] = wall_by_pulse.get(key_w, 0.0) + seg_wall / max(1, len(active))
            # ---- boundary populations and the report ---------------------------------------------------------
            if kets is not None:
                bpop: dict[int, float] = {}
                for psi, w_k in zip(kets, weights):
                    for m, v in boundary_populations(psi, space).items():
                        bpop[m] = bpop.get(m, 0.0) + w_k * float(v)
            else:
                assert rho is not None
                bpop = boundary_populations(rho, space)
            for m, v in bpop.items():
                worst_boundary[m] = max(worst_boundary.get(m, 0.0), v)
            steps_per_period = _steps_per_period(rhs_evals, built.omega_max_rad_s, b - a)
            kinds = tuple(dict.fromkeys(_channel_kind(c.channel) for c in seg_ops))
            segments.append(
                SegmentReport(
                    a,
                    b,
                    tuple(p.gate_id for p in active),
                    integrator,
                    atol_used,
                    rhs_evals,
                    steps_per_period,
                    built.approximations,
                    bpop,
                    retries_seg,
                    method=seg_method,
                    n_collapse_ops=len(seg_ops),
                    channels=kinds,
                    kernel=built.kernel,
                    frame="rotating" if rot is not None else "schrodinger",
                    wall_time_s=seg_wall,
                )
            )
            if active and self.progress is not None:
                from qutip_trap.run.results import Progress

                pulse_done += 1
                self.progress(
                    Progress("pulse", pulse_done, pulse_total, time.perf_counter() - progress_started)
                )
            if active:
                # a normalized branch of weight w carries at most w of the population the pulse moves, so both checks
                # compare against boundary_population_max / w
                branch_w = min(max(sample.get(KEY_BRANCH_WEIGHT, 1.0), 1e-300), 1.0)
                tail = min(options.boundary_population_max / branch_w, 0.5)
                for m, v in bpop.items():
                    if v > tail and space.mode_class(m) in ("resolved", "enr"):
                        raise _BoundaryTrip(m, v)
                if options.margin_check and space.resolved:
                    # the cap's margin above the populated range must cover the margin the segment's eta needs
                    eta_seg: dict[int, float] = {}
                    for rec in built.records:
                        for m, e in rec.etas.items():
                            eta_seg[m] = max(eta_seg.get(m, 0.0), abs(float(e)))
                    for tr in space.resolved:
                        m = tr.mode
                        if eta_seg.get(m, 0.0) == 0.0:
                            continue
                        n_pop = _populated_max(space, kets, weights, rho, m, tail)
                        populated_max[m] = max(populated_max.get(m, 0), n_pop)
                        margin = tr.d - 1 - n_pop
                        margin_reached[m] = min(margin_reached.get(m, margin), margin)
                        need = required_margin_under(eta_seg[m], options, n_pop)
                        if margin < need:
                            raise _BoundaryTrip(m, bpop.get(m, 0.0), add=need - margin, reason="margin")
        if method_used == "mcsolve" and kets is not None:
            trajectory_finals = list(kets)
        times_arr = np.concatenate(times_all) if times_all else np.array([t0])
        expect = {k: np.concatenate(v) if v else np.array([]) for k, v in expect_all.items()}
        # ---- the final state ----------------------------------------------------------------------------------
        final_joint: qt.Qobj | None
        if kets is not None:
            if len(kets) == 1:
                final_joint = kets[0]
            elif space.dimension <= 1024:
                final_joint = _mixture(kets, weights)
            else:
                final_joint = None
                notes.append(
                    f"ensemble of {len(kets)} kets at dimension {space.dimension} not averaged into a joint density matrix; "
                    "the reduced states are the averages"
                )
            internal = sum(
                (w_k * space.internal_marginal(k) for k, w_k in zip(kets, weights)),
                0.0 * space.internal_marginal(kets[0]),
            )
            motional_reduced: dict[int, qt.Qobj] = {}
            nbar: dict[int, float] = {}
            for m in carried:
                rho_m = sum(
                    (w_k * space.mode_marginal(k, m) for k, w_k in zip(kets, weights)),
                    0.0 * space.mode_marginal(kets[0], m),
                )
                motional_reduced[m] = rho_m
                nbar[m] = float(np.real(qt.expect(qt.num(rho_m.shape[0]), rho_m)))
            n_traj = len(kets)
        else:
            assert rho is not None
            final_joint = rho
            internal = space.internal_marginal(rho)
            motional_reduced = {}
            nbar = {}
            for m in carried:
                rho_m = space.mode_marginal(rho, m)
                motional_reduced[m] = rho_m
                nbar[m] = float(np.real(qt.expect(qt.num(rho_m.shape[0]), rho_m)))
            n_traj = 1
        for m in space.frozen:
            nbar[m] = float(state.motional.nbar.get(m, 0.0))
        final = State(
            internal=internal,
            motional=MotionalModel(reduced=motional_reduced, nbar=nbar, frozen=tuple(space.frozen)),
            joint=final_joint,
            provenance=tuple(state.provenance) + ("m2.joint_exact_engine",),
        )
        self.last_report = EngineReport(
            segments=tuple(segments),
            frozen_n=frozen_n,
            growth_retries=growth_retries,
            space=space,
            drive_records=tuple(records),
            trajectories=n_traj,
            method=method_used,
            hardware_notes=hw_notes,
            schedule_played=sched,
            notes=tuple(notes),
            populated_n_max=populated_max,
            margin_reached=margin_reached,
            kernel=_kernel_summary(segments),
            map=map_used if method_used == "mcsolve" else "serial",
            workers=workers_used,
            propagator_solves=propagator_solves,
            propagator_cache_hits=propagator_hits,
            trajectory_finals=tuple(trajectory_finals),
            trajectory_seeds=tuple(trajectory_seeds),
        )
        return Traces(
            times_s=times_arr,
            expectations={
                k: np.real_if_close(v) if k.startswith(("P1", "n[")) else v
                for k, v in expect.items()
                if not k.startswith("a[")
            },
            reduced_internal=tuple(reduced),
            mode_occupations={m: np.real(expect[f"n[{m}]"]) for m in carried},
            alpha_m={m: expect[f"a[{m}]"] for m in carried},
            jumps=tuple(sorted(jumps)),
            final=final,
            boundary_population=worst_boundary,
            mode_marginal=None
            if marginals is None
            else {
                m: np.concatenate(v) if v else np.zeros((times_arr.size, 0), dtype=float)
                for m, v in marginals.items()
            },
            wall_time_s=wall_by_pulse,
        )


EIGH_DIMENSION_MAX = 4096
"""Above this joint dimension a constant but non-diagonal Hamiltonian goes through the ODE ladder rather than a dense eigh."""

TARGET_TOL_MIN_TRAJECTORIES = 8
"""The floor on the phase-one trajectory estimate: QuTiP 5.3.1's ``target_tol`` can stop at 2 trajectories on a
zero-variance first batch."""

IMPROVED_SAMPLING_MULTI_SEGMENT = (
    "mcsolve improved_sampling not used: the no-jump/jump split is a decomposition of the WHOLE evolution, and this schedule "
    "has {n} trajectory segments; applied per segment it would turn the ensemble into a jump expansion of 2^{n} pure members "
    "that cannot be merged (pure states) and cannot be pruned without dropping exactly the jump weight the channels are there "
    "to produce, so the segments run uniform-weight trajectories (Section 5.3; conv.improved_sampling_single_segment)"
)

PROPAGATOR_CACHE_MAX = 256
"""Segment propagators an engine keeps; the cache is cleared when full."""


@dataclass(frozen=True)
class _Propagator:
    """U(t_k, t_0) at the stored times of one segment on an internal-state-only space, with how it was integrated."""

    unitaries: tuple[np.ndarray, ...]
    integrator: str
    atol: float
    rhs_evaluations: int | None
    retries: tuple[str, ...]


MARGIN_LEAKAGE_FRACTION = 0.1
"""The derived margin (``SolverOptions.margin_element_tol``) keeps one displacement's leakage from the top populated level
below this fraction of ``boundary_population_max``, so that the boundary monitor does not trip first."""


def required_margin_under(eta: float, options: SolverOptions, n_hi: int) -> int:
    """The margin a run under ``options`` keeps above the top populated level ``n_hi`` at |eta|: the fixed margin of
    ``hilbert.operators.required_margin`` unless ``margin_element_tol`` is declared, which derives it from that tolerance
    and a tenth of the boundary threshold. ``run.gate_local.step_space`` and the engine's margin check both read it."""
    if options.margin_element_tol is None:
        return required_margin(eta)
    return required_margin(
        eta,
        tail=float(options.boundary_population_max) * MARGIN_LEAKAGE_FRACTION,
        n_hi=int(n_hi),
        element_tol=float(options.margin_element_tol),
    )


def _steps_per_period(rhs_evals: int | None, omega_max_rad_s: float, duration_s: float) -> float | None:
    """Integrator steps per period of the fastest mode: rhs_evals / 12 dop853 stages over the periods of
    ``omega_max_rad_s`` in ``duration_s``; None without a count or a frequency."""
    if not rhs_evals or omega_max_rad_s <= 0.0:
        return None
    periods = duration_s * omega_max_rad_s / (2.0 * math.pi)
    return float((rhs_evals / 12.0) / periods) if periods > 0 else None


def _constant_unitary(h: qt.Qobj, tau: float) -> np.ndarray | None:
    """e^{-i h tau} of a constant Hamiltonian: by its diagonal phases when ``h`` is diagonal in the joint basis, else by one
    Hermitian eigendecomposition up to ``EIGH_DIMENSION_MAX`` (None above it: the caller integrates the propagator)."""
    energies = _diagonal_energies(h)
    if energies is not None:
        return np.asarray(np.diag(np.exp(-1j * tau * energies)), dtype=complex)
    if h.shape[0] > EIGH_DIMENSION_MAX:
        return None
    w, vecs = np.linalg.eigh(np.asarray(h.full()))
    return np.asarray((vecs * np.exp(-1j * tau * w)) @ vecs.conj().T, dtype=complex)


def _kernel_summary(segments: Sequence[SegmentReport]) -> str:
    kinds = {seg.kernel for seg in segments if seg.kernel != "none"}
    if not kinds:
        return "none"
    if kinds == {"assembled"}:
        return "assembled"
    if kinds == {"factorized"}:
        return "factorized"
    return "mixed"


@dataclass(frozen=True)
class _ClosedForm:
    """What the exact propagation of one constant-Hamiltonian segment produced (all stored times of the segment)."""

    kets: list[qt.Qobj] | None
    rho: qt.Qobj | None
    expect: dict[str, np.ndarray]
    reduced: list[qt.Qobj]
    method: str
    integrator: str
    atol: float
    retries: tuple[str, ...]
    marginals: dict[int, np.ndarray] | None = None
    """Per carried mode the (T, d_m) Fock populations over the segment's stored times, when the options ask for them."""


def _expectation_back(
    values: np.ndarray,
    lam: float | None,
    times: np.ndarray,
    op: qt.Qobj,
    states: Sequence[qt.Qobj],
) -> np.ndarray:
    """A rotating-frame expectation trace in the Schroedinger picture: e^{i lambda t} <phi|O|phi> for an eigenoperator of
    ad_{H_0}, the trace over the rotated-back ``states`` when ``lam`` is None."""
    if lam is None:
        return np.array([qt.expect(op, st) for st in states], dtype=complex)
    if lam == 0.0:
        return np.asarray(values, dtype=complex)
    return np.asarray(values, dtype=complex) * np.exp(1j * lam * np.asarray(times, dtype=float))


def _mixture(kets: list[qt.Qobj], weights: list[float]) -> qt.Qobj:
    """sum_k w_k |psi_k><psi_k| of weighted kets (a single ket's projector when there is one)."""
    if len(kets) == 1:
        return kets[0].proj()
    return sum((w * k.proj() for k, w in zip(kets, weights)), 0.0 * kets[0].proj())


def _pure_branches(rho: qt.Qobj, weight_min: float) -> tuple[list[qt.Qobj], list[float], float]:
    """The eigen-decomposition of a density matrix into pure branches: (kets, weights >= ``weight_min`` renormalized, dropped
    weight), heaviest first."""
    mat = np.asarray(rho.full())
    mat = 0.5 * (mat + mat.conj().T)
    w, v = np.linalg.eigh(mat)
    trace = float(np.sum(w))
    order = [int(k) for k in np.argsort(-w)]
    keep = [k for k in order if w[k] >= weight_min * trace]
    if not keep:
        keep = [order[0]]
    total = float(np.sum(w[keep]))
    dims = [list(rho.dims[0]), [1] * len(rho.dims[0])]
    kets = [qt.Qobj(v[:, k].reshape(-1, 1), dims=dims) for k in keep]
    return kets, [float(w[k]) / total for k in keep], float(max(trace - total, 0.0))


def _diagonal_energies(h: qt.Qobj) -> np.ndarray | None:
    """The diagonal of a constant Hamiltonian (rad/s) when it is diagonal in the joint Fock x computational basis, else None."""
    coo = h.to("CSR").data.as_scipy().tocoo()
    if coo.nnz and bool(np.any(coo.row != coo.col)):
        return None
    return np.real(np.asarray(h.diag(), dtype=complex))


def _eigen_frequency(op: qt.Qobj, energies: np.ndarray) -> float | None:
    """lambda with [H, op] = lambda op for the diagonal H of ``energies``, or None."""
    coo = op.to("CSR").data.as_scipy().tocoo()
    if coo.nnz == 0:
        return 0.0
    lam = energies[coo.row] - energies[coo.col]
    tol = 1e-9 * max(float(np.max(np.abs(energies))), 1.0)
    if float(np.ptp(lam)) > tol:
        return None
    return float(lam[0])


def _carried_modes(space: HilbertSpace) -> list[int]:
    """The modes a run carries as tensor factors: the resolved ones and the ENR group's members (``Traces.mode_occupations``)."""
    return [m.mode for m in space.resolved] + (list(space.enr_group[0]) if space.enr_group else [])


def _marginals_of(
    space: HilbertSpace, states: Sequence[qt.Qobj], modes: Sequence[int]
) -> dict[int, np.ndarray]:
    """Per mode the (T, d_m) Fock populations of the stored ``states``."""
    return {m: np.array([space.fock_populations(st, m) for st in states], dtype=float) for m in modes}


def _weighted_marginals(
    acc: dict[int, np.ndarray] | None, weight: float, part: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """``acc + weight * part`` per mode (``acc`` None starts the sum): the branch and trajectory weighting of the traces."""
    if acc is None:
        return {m: weight * v for m, v in part.items()}
    return {m: acc[m] + weight * v for m, v in part.items()}


def _closed_form_segment(
    h: qt.Qobj,
    c_ops: list[qt.Qobj],
    kets: list[qt.Qobj] | None,
    weights: list[float],
    rho: qt.Qobj | None,
    times: np.ndarray,
    space: HilbertSpace,
    e_keys: list[str],
    e_list: list[qt.Qobj],
    options: SolverOptions,
    lindblad: str,
    largest_mode_dimension: int,
) -> _ClosedForm | None:
    """Propagate a segment with the constant Hamiltonian ``h`` exactly: e^{-i h tau}, or with eigenoperator collapse
    operators (mesolve, diagonal ``h``) the master equation in the frame rotating with ``h``. None when neither applies."""
    energies = _diagonal_energies(h)
    taus = np.asarray(times, dtype=float) - float(times[0])
    dims_int = [list(space.ion_dims), list(space.ion_dims)]
    carried = _carried_modes(space) if options.store_marginals else None
    if not c_ops:
        if energies is not None:
            phases = np.exp(-1j * np.outer(taus, energies))

            def propagate_ket(psi: qt.Qobj) -> list[qt.Qobj]:
                v = np.asarray(psi.full()).reshape(-1)
                return [qt.Qobj((ph * v).reshape(-1, 1), dims=psi.dims) for ph in phases]

            def propagate_dm(r: qt.Qobj) -> list[qt.Qobj]:
                m = np.asarray(r.full())
                return [qt.Qobj((ph[:, None] * m) * np.conj(ph)[None, :], dims=r.dims) for ph in phases]

        else:
            if h.shape[0] > EIGH_DIMENSION_MAX:
                return None
            w, vecs = np.linalg.eigh(np.asarray(h.full()))
            vecs_h = vecs.conj().T

            def propagate_ket(psi: qt.Qobj) -> list[qt.Qobj]:
                c = vecs_h @ np.asarray(psi.full()).reshape(-1)
                return [
                    qt.Qobj((vecs @ (np.exp(-1j * tau * w) * c)).reshape(-1, 1), dims=psi.dims)
                    for tau in taus
                ]

            def propagate_dm(r: qt.Qobj) -> list[qt.Qobj]:
                m = vecs_h @ np.asarray(r.full()) @ vecs
                out = []
                for tau in taus:
                    ph = np.exp(-1j * tau * w)
                    out.append(
                        qt.Qobj(vecs @ ((ph[:, None] * m) * np.conj(ph)[None, :]) @ vecs_h, dims=r.dims)
                    )
                return out

        if kets is not None:
            expect = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
            red_acc: np.ndarray | None = None
            new_kets: list[qt.Qobj] = []
            fock_acc: dict[int, np.ndarray] | None = None
            for psi, w_k in zip(kets, weights):
                states = propagate_ket(psi)
                new_kets.append(states[-1])
                for k, op in zip(e_keys, e_list):
                    expect[k] += w_k * np.array([qt.expect(op, s) for s in states], dtype=complex)
                marg = w_k * np.array([space.internal_marginal(s).full() for s in states])
                red_acc = marg if red_acc is None else red_acc + marg
                if carried is not None:
                    fock_acc = _weighted_marginals(fock_acc, w_k, _marginals_of(space, states, carried))
            assert red_acc is not None
            reduced = [qt.Qobj(arr, dims=dims_int) for arr in red_acc]
            return _ClosedForm(
                new_kets, None, expect, reduced, "sesolve", "exact", options.atol, (), marginals=fock_acc
            )
        assert rho is not None
        states = propagate_dm(rho)
        expect = {
            k: np.array([qt.expect(op, s) for s in states], dtype=complex) for k, op in zip(e_keys, e_list)
        }
        return _ClosedForm(
            None,
            states[-1],
            expect,
            [space.internal_marginal(s) for s in states],
            "mesolve",
            "exact",
            options.atol,
            (),
            marginals=None if carried is None else _marginals_of(space, states, carried),
        )
    # dissipative segment: the master equation in the frame rotating with the diagonal H (mesolve path only)
    if energies is None or lindblad != "mesolve":
        return None
    if any(_eigen_frequency(c, energies) is None for c in c_ops):
        return None
    from qutip_trap.dynamics.evolve import evolve

    if kets is not None:
        rho0 = _mixture(kets, weights)
    else:
        assert rho is not None
        rho0 = rho
    zero = qt.Qobj(np.zeros(h.shape, dtype=complex), dims=h.dims).to("CSR")
    ev = evolve(
        zero,
        rho0,
        times,
        c_ops=c_ops,
        e_ops=None,
        options=options,
        store_states=True,
        omega_max_rad_s=None,
        largest_mode_dimension=largest_mode_dimension,
    )
    assert ev.states is not None
    phases = np.exp(-1j * np.outer(taus, energies))
    states = [
        qt.Qobj((ph[:, None] * np.asarray(s.full())) * np.conj(ph)[None, :], dims=rho0.dims)
        for ph, s in zip(phases, ev.states)
    ]
    expect = {k: np.array([qt.expect(op, s) for s in states], dtype=complex) for k, op in zip(e_keys, e_list)}
    return _ClosedForm(
        None,
        states[-1],
        expect,
        [space.internal_marginal(s) for s in states],
        "mesolve",
        f"{ev.integrator}[rotating frame]",
        ev.atol,
        ev.retries,
        marginals=None if carried is None else _marginals_of(space, states, carried),
    )


def _merge_cuts(times: list[float], tolerance_s: float = 1e-12) -> list[float]:
    """Collapse cut points closer than a picosecond: a pulse end and an idle start that differ by float round-off must not
    open a segment on which one pulse has ended and its tail has begun (the builder takes one segment's pulses)."""
    out: list[float] = []
    for t in times:
        if out and t - out[-1] <= tolerance_s:
            continue
        out.append(t)
    return out


def _populated_max(
    space: HilbertSpace,
    kets: list[qt.Qobj] | None,
    weights: list[float],
    rho: qt.Qobj | None,
    mode: int,
    tail: float,
) -> int:
    """The highest Fock index of ``mode`` the (weighted) state populates above ``tail``."""
    if kets is not None:
        parts = [w * space.fock_populations(k, mode) for k, w in zip(kets, weights)]
        p = np.sum(np.stack(parts), axis=0)
    else:
        assert rho is not None
        p = space.fock_populations(rho, mode)
    above = np.cumsum(p[::-1])[::-1]
    for n in range(p.size):
        if n + 1 >= p.size or above[n + 1] < tail:
            return n
    return int(p.size - 1)


class _BoundaryTrip(Exception):
    """The truncation monitor tripped: the boundary population exceeded the threshold (``reason`` "boundary") or the
    cap's margin above the populated range fell short ("margin")."""

    def __init__(self, mode: int, worst: float, add: int = 0, reason: str = "boundary") -> None:
        super().__init__(
            f"{reason} trip on mode {mode}: boundary population {worst:.3e}, deficit {add} level(s)"
        )
        self.mode = mode
        self.worst = worst
        self.add = int(add)
        self.reason = reason


class TruncationLimit(RuntimeError):
    """Truncation could not be fixed: the cap-raising retries ran out, or growth would exceed ``joint_dimension_max``."""


__all__ += [
    "MARGIN_LEAKAGE_FRACTION",
    "EngineReport",
    "JointExactEngine",
    "SegmentReport",
    "TruncationLimit",
    "required_margin_under",
]
