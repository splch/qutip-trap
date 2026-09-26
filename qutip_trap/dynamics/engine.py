"""The JOINT_EXACT pulse engine and the run-time records it reads and writes (PLAN.md Section 5.4).

``JointExactEngine.run_pulses`` cuts a schedule at every pulse and idle boundary and evolves the joint state through each
segment exactly: a constant Hamiltonian by its closed-form propagator, a segment on an internal-state-only space by a cached
propagator, kets through ``sesolve`` (in the exact rotating frame of ``dynamics.rotating`` where it applies), a density
matrix or any collapse operator through ``mesolve`` up to ``Numerics.mesolve_dimension_max``, and above it an ensemble
of keyed quantum-jump trajectories through ``mcsolve``. A density-matrix input without collapse operators is evolved as the
weighted pure branches of its eigen-decomposition. After every pulse the boundary and margin monitors may raise a cap and
repeat the run.

Randomness comes from one root ``SeedSequence`` per run, spawned by (sample, trajectory, shot, ion, channel), so every
variate is independent of execution order and of the retries.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import qutip as qt

from qutip_trap.dynamics.channels import CollapseOp, RecoilOption
from qutip_trap.dynamics.operators import _highest_populated, _thermal_levels, required_margin
from qutip_trap.dynamics.parallel import worker_count
from qutip_trap.dynamics.rotating import RotatingSegment, _diagonal_energies, eigen_frequency, rotating_frame
from qutip_trap.options import Numerics

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.hamiltonian import BuilderOptions, BuiltHamiltonian
    from qutip_trap.dynamics.space import HilbertSpace
    from qutip_trap.dynamics.tomography import TomographyRecord
    from qutip_trap.noise.sampling import NoiseSample
    from qutip_trap.noise.scattering import InternalLevels
    from qutip_trap.run.results import Progress


@dataclass(frozen=True)
class MotionalModel:
    """Per-mode state carried between pulses (Section 5.4)."""

    reduced: dict[int, Qobj]
    """mode -> its reduced density matrix."""
    nbar: dict[int, float]
    frozen: tuple[int, ...]


@dataclass(frozen=True)
class State:
    """What ``prepare()`` returns and ``run_pulses()`` advances."""

    internal: Qobj
    """Register density matrix or ket on the ion factors of the space."""
    motional: MotionalModel
    joint: Qobj | None
    """The joint ket or density matrix; None after a trajectory ensemble too large to average into one density matrix."""
    provenance: tuple[str, ...]
    """Ids of the stages that produced it."""


@dataclass(frozen=True)
class SeedSpec:
    """One root SeedSequence per run, spawned by key (Section 3.4)."""

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
class Traces:
    """What ``run_pulses()`` returns."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    """``P1[i]`` per ion and ``n[m]`` per carried mode at every stored time."""
    reduced_internal: tuple[Qobj, ...]
    mode_occupations: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
    """(time, "traj{k}:{channel}") of every quantum jump of the trajectory path; empty on the density-matrix paths."""
    final: State
    boundary_population: dict[int, float]
    mode_marginal: dict[int, np.ndarray] | None = None
    """Per carried mode the (T, d_m) Fock populations at every stored time, weighted as ``expectations``; None unless
    ``Numerics.store_marginals``."""
    wall_time_s: dict[str, float] = field(default_factory=dict)
    """Integration wall seconds per pulse by gate id: a segment's time split equally among its active pulses, an idle
    segment's under ``"idle"``."""


@dataclass(frozen=True)
class ChannelSummary:
    """The process tomography of one pulse group (Sections 5.4, 6.8)."""

    choi: np.ndarray
    cp_tp_residual: tuple[float, float]
    """Residuals against the completely-positive cone and the trace-preserving set after the projection."""
    n_traj: int
    average_gate_infidelity: float
    pauli_twirled: dict[str, float]
    depolarizing_rate: float
    """epsilon with Lambda_eps(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P (Section 6.8)."""


@dataclass(frozen=True)
class SegmentReport:
    """What one integration segment did."""

    t_start_s: float
    t_end_s: float
    pulses: tuple[str | None, ...]
    integrator: str
    atol: float
    approximations: tuple[str, ...]
    boundary_population: dict[int, float]
    retries: tuple[str, ...]
    method: str = "sesolve"
    """sesolve, mesolve or mcsolve."""
    n_collapse_ops: int = 0
    channels: tuple[str, ...] = ()
    """The channel kinds active on the segment."""
    kernel: str = "none"
    """How the drive operators were held: factorized, assembled, mixed or none."""
    frame: str = "schrodinger"
    """The picture the segment was integrated in: ``rotating`` (``dynamics.rotating``) or ``schrodinger``."""
    wall_time_s: float = 0.0


@dataclass(frozen=True)
class EngineReport:
    """What one ``run_pulses`` did (Section 5.5): the segments, the space it ended on after any cap growth, the method and
    trajectory count, the margins, the workers and propagator-cache use, and the notes."""

    segments: tuple[SegmentReport, ...]
    frozen_n: dict[int, int]
    growth_retries: int
    space: HilbertSpace
    trajectories: int = 1
    """Trajectories carried at the end (1 on the deterministic paths)."""
    method: str = "sesolve"
    hardware_notes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    populated_n_max: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the highest Fock index populated above the boundary threshold during any pulse."""
    margin_reached: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the smallest margin (levels) the cap kept above the populated range during the pulses."""
    kernel: str = "none"
    map: str = "serial"
    """The map the trajectories ran through."""
    workers: int = 1
    """Processes the trajectory map used."""
    propagator_solves: int = 0
    propagator_cache_hits: int = 0

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


_MAX_GROWTH_RETRIES = 3
"""Cap-raising retries a run makes when the boundary or margin monitor trips (Section 5.5)."""
_GROWTH_LEVELS = 4
"""Fock levels a boundary trip adds to the mode's cap (a margin trip adds its deficit)."""


@dataclass
class JointExactEngine:
    """The JOINT_EXACT pulse engine (Section 5.4): every pulse through the one builder, the joint state evolved exactly.

    ``store_per_segment`` stored points per segment and ``store_times_s`` extra absolute store times; ``channels`` explicit
    collapse operators; ``device_channels`` adds the device's own (``NoiseModel.channels`` and, per segment, the scattering
    operators of the active pulses under ``scattering_channels`` and their intensity-noise operators under
    ``intensity_noise_channels``); ``levels_by_ion`` the register level maps of ions with d > 2; ``hardware_chain`` passes
    the schedule through the control hardware chain; ``table`` converts the programmed drives into what the ions see
    (``control.played``), None plays them as physical. ``last_report`` carries the diagnostics of the most recent run. The
    physics switches are the ``Physics`` fields of the same names (``Machine.engine``).
    """

    builder_options: BuilderOptions | None = None
    store_per_segment: int = 2
    store_times_s: tuple[float, ...] = ()
    channels: tuple[CollapseOp, ...] = ()
    qubit_shifts_hz: dict[int, float] = field(default_factory=dict)
    device_channels: bool = False
    levels_by_ion: dict[int, InternalLevels] | None = None
    hardware_chain: bool = True
    scattering_channels: bool = False
    scattering_recoil: RecoilOption = "minimal"
    intensity_noise_channels: bool = True
    table: CalibrationTable | None = None
    last_report: EngineReport | None = None
    progress: Callable[[Progress], None] | None = field(default=None, repr=False, compare=False)
    """Called after every integrated pulse segment with ``Progress("pulse", done, total, elapsed_s)``; dropped when the
    engine is shipped to a worker."""
    _propagators: dict[tuple[object, ...], _Propagator] = field(
        default_factory=dict, repr=False, compare=False
    )
    """The propagator cache, keyed by the built Hamiltonian's fingerprint and the stored times."""

    def __getstate__(self) -> dict[str, object]:
        # an engine shipped to a worker carries neither its report, its cache nor its progress hook
        state = dict(self.__dict__)
        state["last_report"] = None
        state["_propagators"] = {}
        state["progress"] = None
        return state

    def process_tomography(
        self,
        device: Device,
        pulse: Pulse | Sequence[Pulse] | Schedule,
        space: HilbertSpace,
        motional_model: MotionalModel,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: Numerics | None = None,
        *,
        ideal: np.ndarray | None = None,
    ) -> ChannelSummary:
        """The Section 6.8 summary of the channel of a pulse, a pulse group or a schedule on ``space`` against ``ideal`` (the
        ideal unitary on the space's ions in factor order; NaN infidelities without one); the full record is
        :meth:`tomography`."""
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
        options: Numerics | None = None,
    ) -> TomographyRecord:
        """State-based process tomography from the motional state of ``motional_model`` (``dynamics.tomography``)."""
        from qutip_trap.dynamics.tomography import tomography as _tomography

        return _tomography(self, device, pulse, space, motional_model, sample, seeds, options or Numerics())

    def run_pulses(
        self,
        device: Device,
        schedule: Schedule,
        state: State,
        space: HilbertSpace,
        sample: NoiseSample,
        seeds: SeedSpec,
        options: Numerics,
    ) -> Traces:
        """Evolve ``state`` through ``schedule`` on ``space``; a boundary or margin trip raises the cap, regrids the state and
        repeats the run, up to ``_MAX_GROWTH_RETRIES`` times and never past ``joint_dimension_max``."""
        from qutip_trap.dynamics.truncation import regrid_state

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
                if retries >= _MAX_GROWTH_RETRIES:
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
                grow = trip.add if trip.reason == "margin" and trip.add > 0 else _GROWTH_LEVELS
                new_space = current_space.grown(trip.mode, grow)
                growth_notes.append(
                    f"cap-raising retry {retries} of {_MAX_GROWTH_RETRIES}: {trip.reason} trip on mode {trip.mode} "
                    f"(population {trip.worst:.3e}) grows its cap by {grow} level(s); the run is integrated again on joint "
                    f"dimension {new_space.dimension} (was {current_space.dimension})"
                )
                if new_space.dimension > options.joint_dimension_max:
                    raise TruncationLimit(
                        f"raising the cap of mode {trip.mode} after a {trip.reason} trip would take the joint space to "
                        f"dimension {new_space.dimension}, above joint_dimension_max = {options.joint_dimension_max}; "
                        f"the {trip.reason} population was {trip.worst:.3e}. Cool or freeze the mode, raise "
                        "the guard deliberately, or let level='auto' route the run to GATE_LOCAL"
                    ) from trip
                joint = current_state.joint
                if joint is None:
                    raise
                new_joint = regrid_state(joint, current_space, new_space)
                current_state = replace(current_state, joint=new_joint)
                current_space = new_space

    # ---- internals ---------------------------------------------------------------------------------------------------

    def _builder_options(self) -> BuilderOptions:
        from qutip_trap.dynamics.hamiltonian import BuilderOptions

        return self.builder_options or BuilderOptions()

    def _segment_propagator(
        self, built: BuiltHamiltonian, times: np.ndarray, options: Numerics, largest_mode: int
    ) -> tuple[_Propagator, bool]:
        """U(t_k, t_0) at every stored time of a segment on an internal-state-only space, from the cache or by one integration
        of the identity through the ladder. Returns (propagator, cache hit)."""
        from qutip_trap.dynamics.evolve import evolve

        h = built.H
        key = (
            built.fingerprint,
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
            omega_max_rad_s=built.omega_max_rad_s or None,
            largest_mode_dimension=largest_mode,
            propagator=True,
        )
        prop = _Propagator(
            unitaries=tuple(np.asarray(st.full()) for st in ev.states),
            integrator=ev.integrator,
            atol=ev.atol,
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
        built: BuiltHamiltonian,
        static: list[CollapseOp],
        notes: list[str],
    ) -> list[CollapseOp]:
        """The collapse operators of one segment: ``static`` plus, with the device's channels on, the scattering and
        intensity-noise operators of the active pulses."""
        from qutip_trap.dynamics.channels import intensity_noise_channels
        from qutip_trap.noise.scattering import ScatteringOptions, scattering_channels

        ops = list(static)
        if not self.device_channels:
            return ops
        if self.scattering_channels:
            sopts = ScatteringOptions(recoil=self.scattering_recoil)
            for p in active:
                more, more_notes = scattering_channels(
                    device, p, space, levels_by_ion=self.levels_by_ion, options=sopts
                )
                ops.extend(more)
                for n in more_notes:
                    if n not in notes:
                        notes.append(n)
        density = device.noise.intensity_white_density()
        if self.intensity_noise_channels and density > 0.0 and built.drive_parts:
            dens: dict[str, float] = {}
            for p in active:
                key = p.gate_id or f"pulse@{p.t_start_s:.9g}"
                if p.drive.kind in ("raman", "light_shift"):
                    dens[key] = density
                elif p.drive.kind in ("optical_E1", "optical_E2"):
                    dens[key] = 0.25 * density
            ops.extend(intensity_noise_channels(built.drive_parts, dens))
        return ops

    def _played_schedule(
        self,
        device: Device,
        schedule: Schedule,
        sample: NoiseSample,
        seeds: SeedSpec,
        notes: list[str],
    ) -> tuple[Schedule, tuple[str, ...]]:
        """The schedule the ions see: the played chain (requested -> physical through the device's derived values, with a
        table) and then the hardware chain. Returns (schedule, hardware notes); the played chain's notes go to ``notes``."""
        from qutip_trap.control.hardware import apply_hardware_chain

        sched = schedule
        hw_notes: tuple[str, ...] = ()
        if self.table is not None:
            from qutip_trap.control.hardware import physical_schedule

            sched, played_notes = physical_schedule(device, sched, self.table)
            notes.extend(played_notes)
        if self.hardware_chain:
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
        """The frozen spectators' Fock states for this evolution (Section 5.2): the sample's, where ``run()`` put them (it
        enumerates them as weighted branches), else a thermal draw keyed per sample and reported."""
        from qutip_trap.dynamics.operators import thermal_populations
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
                    probs = thermal_populations(nbar_m, _thermal_levels(nbar_m))
                    probs = probs / probs.sum()
                    frozen_n[m] = int(rng.choice(len(probs), p=probs))
                    notes.append(
                        f"frozen mode {m}: no Fock state given for this evolution, so n = {frozen_n[m]} was drawn from the "
                        f"thermal distribution at nbar = {nbar_m:.4g}, keyed per sample (run() enumerates the branches "
                        "instead)"
                    )
        return frozen_n

    def _collapse_setup(self, device: Device, space: HilbertSpace) -> tuple[list[CollapseOp], bool]:
        """The state-independent collapse operators of a run on ``space`` and whether a segment with active pulses can carry
        pulse-built ones (scattering, intensity noise); decided before any build, so the drive operators are assembled
        exactly where a Liouvillian is formed."""
        static_ops: list[CollapseOp] = list(self.channels)
        if self.device_channels:
            static_ops.extend(device.noise.channels(device, space))
        pulse_channels_possible = bool(self.device_channels) and (
            self.scattering_channels
            or (self.intensity_noise_channels and device.noise.intensity_white_density() > 0.0)
        )
        return static_ops, pulse_channels_possible

    def is_unitary(self, device: Device, space: HilbertSpace) -> bool:
        """Whether every segment of a run on ``space`` evolves unitarily (no explicit, device or pulse-built channel), so that
        the final state is linear in the initial ket."""
        static_ops, pulse_channels_possible = self._collapse_setup(device, space)
        return not static_ops and not pulse_channels_possible

    @staticmethod
    def _segment_edges(sched: Schedule) -> list[float]:
        """Cuts at every pulse and idle boundary from the schedule's declared start (``Schedule.t0_s``, else min(0, the first
        cut)) to ``pulses_end_s``."""
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
        """The stored times of one segment: ``store_per_segment`` points from ``a`` to ``b`` plus the ``store_times_s`` inside
        it (the same array on every path, so the propagator cache key is shared)."""
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
        options: Numerics,
        *,
        motional_model: MotionalModel | None = None,
    ) -> tuple[np.ndarray, EngineReport]:
        """U(t_end, t_0) of ``schedule`` on the internal-state-only ``space`` (every mode frozen, no ENR group) as a D x D
        matrix, with the report of a run: the product of the segment propagators (the closed form of a constant segment,
        else the cached or once-integrated propagator). ``motional_model`` supplies the occupations of the frozen modes the
        sample carries no Fock state for. Raises ``ValueError`` on a space with a resolved mode or an ENR group and when a
        segment carries a collapse operator."""
        from qutip_trap.dynamics.hamiltonian import _kernel_label, build_hamiltonian

        if space.resolved or space.enr_group is not None:
            raise ValueError(
                "propagator() takes an internal-state-only space (every mode frozen, no ENR group); a space with a resolved "
                "mode is a joint space, whose propagator is never formed"
            )
        bopts = self._builder_options()
        notes: list[str] = []
        sched, hw_notes = self._played_schedule(device, schedule, sample, seeds, notes)
        nbar = motional_model.nbar if motional_model is not None else {}
        frozen_n = self._frozen_fock_states(space, sample, seeds, nbar, notes)
        static_ops, pulse_channels_possible = self._collapse_setup(device, space)
        u = np.eye(space.dimension, dtype=complex)
        segments: list[SegmentReport] = []
        solves = 0
        hits = 0
        for a, b, active in _segments(sched, self._segment_edges(sched)):
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
            times = self._segment_times(a, b)
            u_seg: np.ndarray | None = None
            integrator = "exact"
            atol_used = options.atol
            retries_seg: tuple[str, ...] = ()
            if built.H.isconstant:
                u_seg = _constant_unitary(built.H(a), b - a)
            if u_seg is None:
                prop, hit = self._segment_propagator(built, times, options, 0)
                hits += int(hit)
                solves += int(not hit)
                u_seg = prop.unitaries[-1]
                integrator = "propagator[cached]" if hit else f"{prop.integrator}[propagator]"
                atol_used = prop.atol
                retries_seg = prop.retries
            u = u_seg @ u
            segments.append(
                SegmentReport(
                    a,
                    b,
                    tuple(p.gate_id for p in active),
                    integrator,
                    atol_used,
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
            method="sesolve",
            hardware_notes=hw_notes,
            notes=tuple(notes),
            kernel=_kernel_label(seg.kernel for seg in segments),
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
        options: Numerics,
        growth_retries: int,
        growth_notes: tuple[str, ...] = (),
    ) -> Traces:
        from qutip_trap.dynamics.evolve import LARGE_MODE_ATOL, LARGE_MODE_DIMENSION, evolve
        from qutip_trap.dynamics.hamiltonian import _kernel_label, build_hamiltonian
        from qutip_trap.dynamics.truncation import boundary_populations
        from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT

        bopts = self._builder_options()
        joint = state.joint
        if joint is None:
            raise ValueError("JOINT_EXACT needs a joint state; build one with HilbertSpace.initial_state")
        if joint.shape[0] != space.dimension:
            raise ValueError("the state does not live on the given space")
        notes: list[str] = list(growth_notes)
        sched, hw_notes = self._played_schedule(device, schedule, sample, seeds, notes)
        frozen_n = self._frozen_fock_states(space, sample, seeds, state.motional.nbar, notes)
        static_ops, pulse_channels_possible = self._collapse_setup(device, space)
        lindblad = _lindblad_method(options, space.dimension)
        map_kind = options.map
        n_workers = worker_count(options)
        workers_used = 1
        map_used = "serial"
        propagator_solves = 0
        propagator_hits = 0
        edges = self._segment_edges(sched)
        t0 = edges[0]
        segments_of_run = _segments(sched, edges)
        carried = _carried_modes(space)
        e_ops: dict[str, qt.Qobj] = {f"P1[{i}]": space.projector(i, 1) for i in space.ion_labels}
        for m in carried:
            e_ops[f"n[{m}]"] = space.number(m)
            e_ops[f"a[{m}]"] = space.annihilation(m)
        stored_modes = (
            carried if options.store_marginals else None
        )  # the modes whose Fock marginals are stored
        times_all: list[np.ndarray] = []
        expect_all: dict[str, list[np.ndarray]] = {k: [] for k in e_ops}
        reduced: list[qt.Qobj] = []
        marginals: dict[int, list[np.ndarray]] | None = (
            {m: [] for m in carried} if options.store_marginals else None
        )
        wall_by_pulse: dict[str, float] = {}
        segments: list[SegmentReport] = []
        jumps: list[tuple[float, str]] = []
        worst_boundary: dict[int, float] = {m: 0.0 for m in carried}
        populated_max: dict[int, int] = {}
        margin_reached: dict[int, int] = {}
        # the state: weighted kets (pure branches, or trajectories), or one density matrix
        kets: list[qt.Qobj] | None = [joint] if joint.isket else None
        weights: list[float] = [1.0] if joint.isket else []
        rho: qt.Qobj | None = None if joint.isket else joint
        if rho is not None and self.is_unitary(device, space) and rho.shape[0] <= EIGH_DIMENSION_MAX:
            kets, weights, dropped = _pure_branches(rho, options.branch_weight_min)
            rho = None
            if dropped > 0.0:
                notes.append(
                    f"initial mixture evolved as {len(kets)} pure branches: branches below branch_weight_min = "
                    f"{options.branch_weight_min:g} dropped, total weight {dropped:.3e} (renormalized)"
                )
        method_used = "sesolve" if kets is not None else "mesolve"
        first = True
        largest_mode = max([m.d for m in space.resolved], default=0)
        atol_mc = options.atol if largest_mode <= LARGE_MODE_DIMENSION else max(options.atol, LARGE_MODE_ATOL)
        # improved_sampling splits the whole evolution into its no-jump member and the rest, so it applies only when the
        # trajectory path is entered exactly once
        n_mc_segments = 0
        if lindblad == "mcsolve" and kets is not None:
            n_mc_segments = sum(
                1 for _a, _b, act in segments_of_run if static_ops or (act and pulse_channels_possible)
            )
        improved_run = bool(options.improved_sampling) and n_mc_segments == 1
        if bool(options.improved_sampling) and n_mc_segments > 1:
            notes.append(IMPROVED_SAMPLING_MULTI_SEGMENT.format(n=n_mc_segments))
        target_tol_estimate: int | None = None
        pulse_total = sum(1 for _a, _b, act in segments_of_run if act)
        pulse_done = 0
        progress_started = time.perf_counter()
        for seg_index, (a, b, active) in enumerate(segments_of_run):
            seg_started = time.perf_counter()
            dissipative_seg = bool(static_ops) or (bool(active) and pulse_channels_possible)
            mesolve_seg = kets is None or (dissipative_seg and lindblad == "mesolve")
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
            seg_ops = self._segment_channels(device, active, space, built, static_ops, notes)
            c_ops = [c.op for c in seg_ops]
            times = self._segment_times(a, b)
            sel = slice(1, None) if not first else slice(0, None)
            seg_method = "sesolve"
            integrator = options.integrators[0]
            atol_used = options.atol
            retries_seg: tuple[str, ...] = ()
            # the exact rotating frame for the ket paths; mesolve forms its Liouvillian in the Schroedinger picture and a
            # constant segment takes the closed form
            rot: RotatingSegment | None = None
            if not mesolve_seg and (bool(space.resolved) or space.enr_group is not None):
                rot = rotating_frame(built.H, space.dims, c_ops)
                if rot is not None:
                    for n in rot.notes:
                        if n not in notes:
                            notes.append(n)
            # per e_op, the e^{i lambda t} an expectation picks up on the way back from the frame (0 for the populations,
            # -omega_m for a_m); None: the trace is taken on the rotated-back states
            e_phases: dict[str, float | None] = (
                {k: eigen_frequency(op, rot.frame.energies) for k, op in e_ops.items()}
                if rot is not None
                else {}
            )
            closed: _ClosedForm | None = None
            if built.H.isconstant and all(isinstance(c, qt.Qobj) for c in c_ops):
                closed = _closed_form_segment(
                    built.H(a),
                    c_ops,
                    kets,
                    weights,
                    rho,
                    times,
                    sel,
                    space,
                    e_ops,
                    options,
                    lindblad,
                    largest_mode,
                )
            seg: _SegmentTraces
            if closed is not None:
                kets, rho, seg = closed.kets, closed.rho, closed.traces
                seg_method, integrator, atol_used, retries_seg = (
                    closed.method,
                    closed.integrator,
                    closed.atol,
                    closed.retries,
                )
            elif (
                kets is not None
                and not c_ops
                and options.propagator_cache
                and not space.resolved
                and space.enr_group is None
            ):
                # an internal-state-only space: one propagator serves every initial state
                prop, hit = self._segment_propagator(built, times, options, largest_mode)
                propagator_hits += int(hit)
                propagator_solves += int(not hit)
                acc = _WeightedKets(space, e_ops, times.size, sel, stored_modes)
                new_kets: list[qt.Qobj] = []
                for psi, w_k in zip(kets, weights):
                    vec = np.asarray(psi.full()).reshape(-1)
                    states_k = [qt.Qobj((u @ vec).reshape(-1, 1), dims=psi.dims) for u in prop.unitaries]
                    new_kets.append(states_k[-1])
                    acc.add(w_k, states_k, _expectations(e_ops, states_k))
                kets, seg = new_kets, acc.traces()
                integrator = "propagator[cached]" if hit else f"{prop.integrator}[propagator]"
                atol_used = prop.atol
                retries_seg = prop.retries
            elif kets is not None and not c_ops:
                acc = _WeightedKets(space, e_ops, times.size, sel, stored_modes)
                new_kets = []
                for psi, w_k in zip(kets, weights):
                    ev = evolve(
                        built.H if rot is None else rot.H,
                        psi if rot is None else rot.frame.to_frame(psi, float(times[0])),
                        times,
                        e_ops=e_ops,
                        options=options,
                        omega_max_rad_s=built.omega_max_rad_s or None,
                        largest_mode_dimension=largest_mode,
                    )
                    if rot is None:
                        states_back: list[qt.Qobj] = list(ev.states)
                        final_k = ev.final
                    else:
                        states_back = [rot.frame.from_frame(st, float(t)) for st, t in zip(ev.states, times)]
                        final_k = rot.frame.from_frame(ev.final, float(times[-1]))
                    new_kets.append(final_k)
                    acc.add(
                        w_k,
                        states_back,
                        {
                            k: _expectation_back(
                                np.asarray(ev.expect[k]), e_phases.get(k, 0.0), times, op, states_back
                            )
                            for k, op in e_ops.items()
                        },
                    )
                    integrator, atol_used, retries_seg = ev.integrator, ev.atol, ev.retries
                kets, seg = new_kets, acc.traces()
            elif not c_ops or lindblad == "mesolve":
                if kets is not None:
                    rho = _mixture(kets, weights)
                    kets = None
                assert rho is not None
                ev = evolve(
                    built.H,
                    rho,
                    times,
                    c_ops=c_ops,
                    e_ops=e_ops,
                    options=options,
                    omega_max_rad_s=built.omega_max_rad_s or None,
                    largest_mode_dimension=largest_mode,
                )
                rho = ev.final
                seg = _dm_traces(space, ev.states, ev.expect, sel, stored_modes)
                integrator, atol_used, retries_seg = ev.integrator, ev.atol, ev.retries
                seg_method = "mesolve"
            else:
                if kets is None:
                    raise RuntimeError(
                        "the trajectory path (mcsolve) needs pure trajectories: the state is a density matrix; enumerate the "
                        "initial mixture into pure branches (run()) or raise mesolve_dimension_max"
                    )
                improved_seg = improved_run and len(kets) == 1
                n_traj_seg = options.ntraj
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
                    "map": "serial",
                    "num_cpus": 1,
                    "improved_sampling": improved_seg,
                }
                # phase one of Section 3.4: the trajectory count from target_tol on the population e_ops under a serial map,
                # capped by ntraj and floored; phase two replays a keyed seed list of that length
                if len(kets) == 1 and options.trajectory_target_tol is not None and e_ops:
                    if target_tol_estimate is None:
                        probe = qt.MCSolver(
                            built.H if rot is None else rot.H,
                            c_ops if rot is None else list(rot.c_ops),
                            options={
                                **mc_opts,
                                "keep_runs_results": False,
                                "store_states": False,
                                "improved_sampling": False,
                            },
                        ).run(
                            kets[0] if rot is None else rot.frame.to_frame(kets[0], float(times[0])),
                            times,
                            ntraj=options.ntraj,
                            e_ops=list(e_ops.values()),
                            target_tol=float(options.trajectory_target_tol),
                        )
                        target_tol_estimate = max(int(probe.num_trajectories), TARGET_TOL_MIN_TRAJECTORIES)
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
                # one trajectory per ket of the ensemble with its keyed seed; the results come back in completion order
                # and are matched to their kets by seed
                seeds_k = [
                    seeds.child(sample.sample_id, k_traj, 0, 0, f"mcsolve[{seg_index}]")
                    for k_traj in range(n_stoch)
                ]
                if improved_seg:
                    res = solver.run(
                        kets_in[0], times, ntraj=n_stoch, e_ops=list(e_ops.values()), seeds=seeds_k
                    )
                elif len(kets) == 1:
                    res = solver.run(kets_in[0], times, ntraj=1, e_ops=list(e_ops.values()), seeds=seeds_k)
                else:
                    res = solver.run(
                        [(psi, 1.0 / len(kets)) for psi in kets_in],
                        times,
                        ntraj=[1] * len(kets),
                        e_ops=list(e_ops.values()),
                        seeds=seeds_k,
                    )
                by_seed = {tuple(int(x) for x in sd.spawn_key): j for j, sd in enumerate(res.seeds)}
                # (trajectory, weight, index into res.col_* or None for the deterministic member)
                members: list[tuple[Any, float, int | None]] = []
                if improved_seg:
                    # the weighted mixture: the deterministic no-jump member carries p_no-jump (QuTiP's
                    # ``deterministic_weights``), the stochastic trajectories the residual ``runs_weights``
                    members.extend(
                        (traj, float(w), None)
                        for traj, w in zip(res.deterministic_trajectories, res.deterministic_weights)
                    )
                    for k_traj in range(n_stoch):
                        j = by_seed[tuple(int(x) for x in seeds_k[k_traj].spawn_key)]
                        members.append((res.trajectories[j], float(res.runs_weights[j]), j))
                else:
                    for k_traj, w_k in enumerate(weights):
                        j = by_seed[tuple(int(x) for x in seeds_k[k_traj].spawn_key)]
                        members.append((res.trajectories[j], float(w_k), j))
                acc = _WeightedKets(space, e_ops, times.size, sel, stored_modes)
                new_kets = []
                for k_traj, (traj_m, w_m, j_m) in enumerate(members):
                    if rot is None:
                        states_m: list[qt.Qobj] = list(traj_m.states)
                        new_kets.append(traj_m.final_state)
                    else:
                        states_m = [rot.frame.from_frame(st, float(t)) for st, t in zip(traj_m.states, times)]
                        new_kets.append(rot.frame.from_frame(traj_m.final_state, float(times[-1])))
                    acc.add(
                        w_m,
                        states_m,
                        {
                            k: _expectation_back(
                                np.asarray(traj_m.expect[idx]), e_phases.get(k, 0.0), times, op, states_m
                            )
                            for idx, (k, op) in enumerate(e_ops.items())
                        },
                    )
                    if j_m is not None:
                        for t_c, which in zip(res.col_times[j_m], res.col_which[j_m]):
                            jumps.append((float(t_c), f"traj{k_traj}:{seg_ops[int(which)].channel}"))
                kets, weights, seg = new_kets, [w_m for _t, w_m, _j in members], acc.traces()
                integrator, atol_used = options.integrators[0], atol_mc
                seg_method = "mcsolve"
                method_used = "mcsolve"
            for k in e_ops:
                expect_all[k].append(seg.expect[k])
            reduced.extend(seg.reduced)
            if seg_method == "mesolve":
                method_used = "mesolve" if method_used != "mcsolve" else method_used
            times_all.append(times[sel])
            if marginals is not None:
                assert seg.marginals is not None
                for m in carried:
                    marginals[m].append(seg.marginals[m])
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
            segments.append(
                SegmentReport(
                    a,
                    b,
                    tuple(p.gate_id for p in active),
                    integrator,
                    atol_used,
                    built.approximations,
                    bpop,
                    retries_seg,
                    method=seg_method,
                    n_collapse_ops=len(seg_ops),
                    channels=tuple(dict.fromkeys(_channel_kind(c.channel) for c in seg_ops)),
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
                # the threshold is 1e-6 of the population the pulse moves: a branch of weight w carries at most w of it, so
                # the branch-relative boundary population and populated range are compared against max / w
                branch_w = min(max(sample.get(KEY_BRANCH_WEIGHT, 1.0), 1e-300), 1.0)
                tail = min(options.boundary_population_max / branch_w, 0.5)
                for m, v in bpop.items():
                    if v > tail and space.mode_class(m) in ("resolved", "enr"):
                        raise _BoundaryTrip(m, v)
                if options.margin_check and space.resolved:
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
        times_arr = np.concatenate(times_all) if times_all else np.array([t0])
        expect = {k: np.concatenate(v) if v else np.array([]) for k, v in expect_all.items()}
        # ---- the final state ----------------------------------------------------------------------------------
        final_joint: qt.Qobj | None
        motional_reduced: dict[int, qt.Qobj] = {}
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
            for m in carried:
                motional_reduced[m] = sum(
                    (w_k * space.mode_marginal(k, m) for k, w_k in zip(kets, weights)),
                    0.0 * space.mode_marginal(kets[0], m),
                )
            n_traj = len(kets)
        else:
            assert rho is not None
            final_joint = rho
            internal = space.internal_marginal(rho)
            for m in carried:
                motional_reduced[m] = space.mode_marginal(rho, m)
            n_traj = 1
        nbar: dict[int, float] = {
            m: float(np.real(qt.expect(qt.num(rho_m.shape[0]), rho_m)))
            for m, rho_m in motional_reduced.items()
        }
        for m in space.frozen:
            nbar[m] = float(state.motional.nbar.get(m, 0.0))
        final = State(
            internal=internal,
            motional=MotionalModel(reduced=motional_reduced, nbar=nbar, frozen=tuple(space.frozen)),
            joint=final_joint,
            provenance=tuple(state.provenance) + ("joint_exact_engine",),
        )
        self.last_report = EngineReport(
            segments=tuple(segments),
            frozen_n=frozen_n,
            growth_retries=growth_retries,
            space=space,
            trajectories=n_traj,
            method=method_used,
            hardware_notes=hw_notes,
            notes=tuple(notes),
            populated_n_max=populated_max,
            margin_reached=margin_reached,
            kernel=_kernel_label(seg.kernel for seg in segments),
            map=map_used if method_used == "mcsolve" else "serial",
            workers=workers_used,
            propagator_solves=propagator_solves,
            propagator_cache_hits=propagator_hits,
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


def _lindblad_method(options: Numerics, dimension: int) -> Literal["mesolve", "mcsolve"]:
    """How a dissipative segment on a space of ``dimension`` is integrated: ``auto`` is mesolve up to
    ``mesolve_dimension_max``."""
    if options.lindblad_method != "auto":
        return options.lindblad_method
    return "mesolve" if dimension <= options.mesolve_dimension_max else "mcsolve"


EIGH_DIMENSION_MAX = 4096
"""Above this joint dimension a constant but non-diagonal Hamiltonian goes through the ODE ladder rather than a dense eigh,
and a density matrix is not eigen-decomposed into pure branches."""

TARGET_TOL_MIN_TRAJECTORIES = 8
"""The floor on the phase-one trajectory estimate (``Numerics.trajectory_target_tol``)."""

IMPROVED_SAMPLING_MULTI_SEGMENT = (
    "mcsolve improved_sampling not used: the no-jump/jump split is a decomposition of the WHOLE evolution, and this schedule "
    "has {n} trajectory segments; applied per segment it would turn the ensemble into a jump expansion of 2^{n} pure members "
    "that cannot be merged (pure states) and cannot be pruned without dropping exactly the jump weight the channels are there "
    "to produce, so the segments run uniform-weight trajectories"
)

PROPAGATOR_CACHE_MAX = 256
"""Segment propagators an engine keeps; the cache is cleared when full."""


@dataclass(frozen=True)
class _Propagator:
    """U(t_k, t_0) at the stored times of one segment on an internal-state-only space, with how it was integrated."""

    unitaries: tuple[np.ndarray, ...]
    integrator: str
    atol: float
    retries: tuple[str, ...]


MARGIN_LEAKAGE_FRACTION = 0.1
"""With a declared ``margin_element_tol`` one displacement's leakage from the top populated level is kept below this fraction
of ``boundary_population_max``, so the boundary monitor is not the first thing a derived cap meets."""


def required_margin_under(eta: float, options: Numerics, n_hi: int) -> int:
    """The Section 5.1.1 margin a run under ``options`` keeps above the top populated level ``n_hi`` at |eta|: the fixture
    unless ``margin_element_tol`` is declared, else derived from it and a tenth of the boundary threshold. The GATE_LOCAL cap
    rule and the engine's margin check read this one function, so a first attempt does not trip."""
    if options.margin_element_tol is None:
        return required_margin(eta)
    return required_margin(
        eta,
        tail=float(options.boundary_population_max) * MARGIN_LEAKAGE_FRACTION,
        n_hi=int(n_hi),
        element_tol=float(options.margin_element_tol),
    )


def _segments(sched: Schedule, edges: Sequence[float]) -> list[tuple[float, float, list[Pulse]]]:
    """(start, end, the pulses active over the whole of it) of every segment between consecutive (increasing) cuts."""
    return [
        (a, b, [p for p in sched.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15])
        for a, b in zip(edges[:-1], edges[1:])
    ]


def _constant_unitary(h: qt.Qobj, tau: float) -> np.ndarray | None:
    """e^{-i h tau} of a constant Hamiltonian: its diagonal phases when ``h`` is diagonal in the joint basis, else one
    Hermitian eigendecomposition up to ``EIGH_DIMENSION_MAX`` (None above it: the caller integrates the propagator)."""
    energies = _diagonal_energies(h)
    if energies is not None:
        return np.asarray(np.diag(np.exp(-1j * tau * energies)), dtype=complex)
    if h.shape[0] > EIGH_DIMENSION_MAX:
        return None
    w, vecs = np.linalg.eigh(np.asarray(h.full()))
    return np.asarray((vecs * np.exp(-1j * tau * w)) @ vecs.conj().T, dtype=complex)


class _SegmentTraces(NamedTuple):
    """One segment's traces at its kept stored times: the expectations, the reduced register and the stored Fock marginals."""

    expect: dict[str, np.ndarray]
    reduced: list[qt.Qobj]
    marginals: dict[int, np.ndarray] | None


class _WeightedKets:
    """The weighted sum over the branches or trajectories of one segment's stored ket states, at the kept times ``sel``."""

    def __init__(
        self,
        space: HilbertSpace,
        e_ops: Mapping[str, qt.Qobj],
        n_times: int,
        sel: slice,
        stored_modes: Sequence[int] | None,
    ) -> None:
        self.space = space
        self.sel = sel
        self.stored_modes = stored_modes
        self.expect = {k: np.zeros(n_times, dtype=complex) for k in e_ops}
        self.registers: list[np.ndarray] = []
        self.marginals: dict[int, np.ndarray] | None = None

    def add(self, weight: float, states: Sequence[qt.Qobj], expect: Mapping[str, np.ndarray]) -> None:
        """One branch or trajectory of weight ``weight``: its states and expectation traces at every stored time."""
        for k, v in expect.items():
            self.expect[k] += weight * v
        kept = states[self.sel]
        self.registers.append(weight * np.array([self.space.internal_marginal(st).full() for st in kept]))
        if self.stored_modes is not None:
            self.marginals = _weighted_marginals(
                self.marginals, weight, _marginals_of(self.space, kept, self.stored_modes)
            )

    def traces(self) -> _SegmentTraces:
        dims = [list(self.space.ion_dims), list(self.space.ion_dims)]
        return _SegmentTraces(
            {k: v[self.sel] for k, v in self.expect.items()},
            [qt.Qobj(arr, dims=dims) for arr in np.sum(np.stack(self.registers), axis=0)],
            self.marginals,
        )


def _dm_traces(
    space: HilbertSpace,
    states: Sequence[qt.Qobj],
    expect: Mapping[str, np.ndarray],
    sel: slice,
    stored_modes: Sequence[int] | None,
) -> _SegmentTraces:
    """The kept-time traces of a density-matrix segment."""
    kept = list(states[sel])
    return _SegmentTraces(
        {k: np.asarray(v)[sel] for k, v in expect.items()},
        [space.internal_marginal(st) for st in kept],
        None if stored_modes is None else _marginals_of(space, kept, stored_modes),
    )


def _expectations(e_ops: Mapping[str, qt.Qobj], states: Sequence[qt.Qobj]) -> dict[str, np.ndarray]:
    return {k: np.array([qt.expect(op, st) for st in states], dtype=complex) for k, op in e_ops.items()}


@dataclass(frozen=True)
class _ClosedForm:
    """What the exact propagation of one constant-Hamiltonian segment produced."""

    kets: list[qt.Qobj] | None
    rho: qt.Qobj | None
    traces: _SegmentTraces
    method: str
    integrator: str
    atol: float
    retries: tuple[str, ...]


def _expectation_back(
    values: np.ndarray,
    lam: float | None,
    times: np.ndarray,
    op: qt.Qobj,
    states: Sequence[qt.Qobj],
) -> np.ndarray:
    """An expectation trace taken in the rotating frame, back in the Schroedinger picture: e^{i lambda t} <phi|O|phi> for an
    eigenoperator of ad_{H_0} (``lam``; 0 leaves the populations untouched), the trace over the rotated-back ``states`` for
    anything else (``lam`` None)."""
    if lam is None:
        return np.array([qt.expect(op, st) for st in states], dtype=complex)
    if lam == 0.0:
        return np.asarray(values, dtype=complex)
    return np.asarray(values, dtype=complex) * np.exp(1j * lam * np.asarray(times, dtype=float))


def _mixture(kets: list[qt.Qobj], weights: list[float]) -> qt.Qobj:
    """sum_k w_k |psi_k><psi_k| (a single ket's projector when there is one)."""
    if len(kets) == 1:
        return kets[0].proj()
    return sum((w * k.proj() for k, w in zip(kets, weights)), 0.0 * kets[0].proj())


def _pure_branches(rho: qt.Qobj, weight_min: float) -> tuple[list[qt.Qobj], list[float], float]:
    """A density matrix as its eigen-decomposition into pure branches, heaviest first: (kets, weights >= ``weight_min``
    renormalized, dropped weight); a product of thermal states is a mixture of Fock states (the Fock sum of Section 5.3)."""
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


def _carried_modes(space: HilbertSpace) -> list[int]:
    """The modes a run carries as tensor factors: the resolved ones and the ENR group's members."""
    return [m.mode for m in space.resolved] + (list(space.enr_group[0]) if space.enr_group else [])


def _marginals_of(
    space: HilbertSpace, states: Sequence[qt.Qobj], modes: Sequence[int]
) -> dict[int, np.ndarray]:
    """Per mode the (T, d_m) Fock populations of the stored ``states``."""
    return {m: np.array([space.fock_populations(st, m) for st in states], dtype=float) for m in modes}


def _weighted_marginals(
    acc: dict[int, np.ndarray] | None, weight: float, part: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """``acc + weight * part`` per mode (``acc`` None starts the sum)."""
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
    sel: slice,
    space: HilbertSpace,
    e_ops: Mapping[str, qt.Qobj],
    options: Numerics,
    lindblad: str,
    largest_mode_dimension: int,
) -> _ClosedForm | None:
    """Propagate a segment with the constant Hamiltonian ``h`` over ``times`` without an ODE solve of its oscillation.

    Without collapse operators the propagator is e^{-i h tau}: the diagonal phases of a diagonal ``h`` (an idle: H_mot +
    H_int + a constant Stark shift), else one Hermitian eigendecomposition. With collapse operators the master equation is
    integrated in the frame rotating with the diagonal ``h``, where H vanishes and every collapse operator that is an
    eigenoperator of ad_H picks up only a phase its dissipator does not see; the stored states are rotated back. None when
    this does not apply (a non-diagonal H with collapse operators, a non-eigenoperator, the trajectory path).
    """
    energies = _diagonal_energies(h)
    taus = np.asarray(times, dtype=float) - float(times[0])
    stored_modes = _carried_modes(space) if options.store_marginals else None
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
            acc = _WeightedKets(space, e_ops, times.size, sel, stored_modes)
            new_kets: list[qt.Qobj] = []
            for psi, w_k in zip(kets, weights):
                states = propagate_ket(psi)
                new_kets.append(states[-1])
                acc.add(w_k, states, _expectations(e_ops, states))
            return _ClosedForm(new_kets, None, acc.traces(), "sesolve", "exact", options.atol, ())
        assert rho is not None
        states = propagate_dm(rho)
        traces = _dm_traces(space, states, _expectations(e_ops, states), sel, stored_modes)
        return _ClosedForm(None, states[-1], traces, "mesolve", "exact", options.atol, ())
    # dissipative segment: the master equation in the frame rotating with the diagonal H (mesolve path only)
    if energies is None or lindblad != "mesolve":
        return None
    if any(eigen_frequency(c, energies) is None for c in c_ops):
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
        omega_max_rad_s=None,
        largest_mode_dimension=largest_mode_dimension,
    )
    phases = np.exp(-1j * np.outer(taus, energies))
    states = [
        qt.Qobj((ph[:, None] * np.asarray(s.full())) * np.conj(ph)[None, :], dims=rho0.dims)
        for ph, s in zip(phases, ev.states)
    ]
    traces = _dm_traces(space, states, _expectations(e_ops, states), sel, stored_modes)
    return _ClosedForm(
        None, states[-1], traces, "mesolve", f"{ev.integrator}[rotating frame]", ev.atol, ev.retries
    )


def _merge_cuts(times: list[float], tolerance_s: float = 1e-12) -> list[float]:
    """Collapse cut points closer than a picosecond: a pulse end and an idle start that differ by round-off must not open a
    segment on which one pulse has ended and its tail has begun."""
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
    """The highest Fock index of ``mode`` the (weighted) state populates above ``tail`` (Section 5.5)."""
    if kets is not None:
        parts = [w * space.fock_populations(k, mode) for k, w in zip(kets, weights)]
        p = np.sum(np.stack(parts), axis=0)
    else:
        assert rho is not None
        p = space.fock_populations(rho, mode)
    return _highest_populated(p, tail)


class _BoundaryTrip(Exception):
    """The truncation monitor tripped: the boundary population exceeded the threshold (``reason`` "boundary") or the cap's
    margin above the populated range fell below the Section 5.1.1 requirement ("margin")."""

    def __init__(self, mode: int, worst: float, add: int = 0, reason: str = "boundary") -> None:
        super().__init__(
            f"{reason} trip on mode {mode}: boundary population {worst:.3e}, deficit {add} level(s)"
        )
        self.mode = mode
        self.worst = worst
        self.add = int(add)
        self.reason = reason


class TruncationLimit(RuntimeError):
    """The cap-raising retries of Section 5.5 were exhausted."""
