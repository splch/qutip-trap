"""The pulse engine protocol and the run-time state records (PLAN.md Sections 3.4, 5.3, 5.4, 6, 7.10, 11; Appendix E).

``PulseEngine.run_pulses`` is the one entry point that ``run/``, ``calibration/`` and ``experiments/``
share. Randomness comes from one root ``SeedSequence`` per run, spawned deterministically by (sample,
trajectory, shot, ion, channel) so that every variate's key is independent of execution order and of
truncation retries (Section 3.4); reproducibility over 1 and 18 workers is a tolerance test (10^-12), not a
bitwise one, because ``MultiTrajResult`` accumulates in completion order (Section 3.4).

Milestone M7 gives the JOINT_EXACT engine its dissipative paths. Per integration segment the collapse operators
are the explicit ``channels``, the device's state-independent set (heating, motional dephasing, white qubit
dephasing; ``NoiseModel.channels``), the scattering operators of the active pulses (``noise/scattering.py``) and the
white intensity-noise channel sqrt(D) H_drive(t) (Section 6.4). With collapse operators present the state is a
density matrix through ``mesolve`` up to ``SolverOptions.mesolve_dimension_max`` (the deterministic reference path
of Section 3.4) and an ensemble of ``SolverOptions.ntraj`` quantum-jump trajectories above it, each trajectory
advanced segment by segment with its own keyed seed (sample, trajectory, 0, 0, "mcsolve[segment]") so that a
trajectory is identical under the same seed whatever the worker count; the jump records are returned in
``Traces.jumps``. Before segmentation the schedule passes through the control hardware chain of Section 7.10
(``control/hardware.py``): quantized tone words, low-pass-filtered envelopes with their tails, jittered train starts.

Two exact shortcuts (M8) replace ODE solves where none is needed, both pinned against the ODE path in
``tests/test_engine_numerics.py``: a segment whose Hamiltonian is constant (an idle interval, a zero-envelope pulse) is
propagated by its diagonal phases, or in the frame rotating with H_mot when eigenoperator collapse operators are present
(``closed_form_constant``); a density-matrix input without collapse operators is evolved as the weighted pure branches of its
eigen-decomposition through ``sesolve`` (``pure_branches``, the Fock-sum path of Section 5.3), never as a D^2 Liouvillian.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import qutip as qt

from qutip_trap.dynamics.channels import CollapseOp
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

# QuTiP multistep integrators, never used (Section 5.3: the escalation ladder is dop853 then vern9)
MULTISTEP_INTEGRATORS: frozenset[str] = frozenset({"adams", "bdf", "lsoda", "vode", "zvode"})
ALLOWED_INTEGRATORS: frozenset[str] = frozenset(
    {"dop853", "vern7", "vern9", "tsit5", "explicit_rk", "krylov", "diag"}
)

LindbladMethod = Literal["auto", "mesolve", "mcsolve"]
RecoilOption = Literal["off", "minimal", "vector"]


@dataclass(frozen=True)
class MotionalModel:
    """Per-mode state carried between pulses (Section 5.4 b)."""

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
    """The joint ket/density matrix when the run holds one (JOINT_EXACT); None after a trajectory ensemble too large to
    average into one density matrix (the reduced states above are the averages)."""
    provenance: tuple[str, ...]
    """Ids of the preparation stages that produced it (Section 4.2.6 order)."""


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
class SolverOptions:
    atol: float = 1e-10
    rtol: float = 1e-8
    nsteps: int = 10**7
    integrators: tuple[str, ...] = ("dop853", "vern9")
    """The escalation ladder of Section 5.3; never a multistep method."""
    joint_dimension_max: int = 4096
    nnz_max: int = 2 * 10**7
    """The Section 11.5 guards that route to GATE_LOCAL."""
    boundary_population_max: float = 1e-6
    freeze_chi_max_rad: float = 0.05
    map: Literal["serial", "parallel", "loky"] = "parallel"
    """Coefficients are module-level functions or arrays, so they pickle."""
    e_ops_for_target_tol: bool = True
    """mcsolve needs e_ops to target a tolerance (Section 5.4)."""
    freeze_alpha_max: float = 1e-4
    """|alpha_m|^2 (2 nbar_m + 1) below which a spectator may be frozen rather than resolved (Section 5.2; M6)."""
    branch_weight_min: float = 1e-6
    """Weight below which a branch of the initial thermal and preparation mixture is dropped from the exact evolution and
    reported (the Fock-sum path of Section 5.3, M6; an approximation of the truncation kind like boundary_population_max)."""
    lindblad_method: LindbladMethod = "auto"
    """How collapse operators are integrated (M7): ``mesolve`` (the density matrix, exact and deterministic), ``mcsolve``
    (quantum-jump trajectories, ``ntraj`` per pure initial state), ``auto`` = mesolve up to ``mesolve_dimension_max`` (a
    Liouvillian of that size integrates in seconds; above it the density matrix costs dimension times a trajectory)."""
    mesolve_dimension_max: int = 128
    ntraj: int = 64
    """Trajectories per pure initial state on the mcsolve path (a fixed keyed seed list, Section 3.4)."""
    scattering_channels: bool = False
    """Build the photon-scattering collapse operators of every pulse (Sections 4.5.5, 6.5) instead of reporting the estimate."""
    scattering_recoil: RecoilOption = "minimal"
    intensity_noise_channels: bool = True
    """The white part of the laser-intensity spectrum as the channel sqrt(D) H_drive(t) (Section 6.4)."""
    hardware_chain: bool = True
    """Pass the schedule through the control hardware chain of Section 7.10 before integrating."""
    margin_check: bool = True
    """Section 5.5 (M9a): after every pulse the cap's margin above the POPULATED range of each resolved mode is compared with
    the Section 5.1.1 margin for the pulse's eta; a deficit raises the cap by it and repeats the run, like the boundary trip."""
    map_accuracy: float = 1e-3
    """epsilon_map of the GATE_LOCAL tomography (Section 5.4): on the trajectory path every input state is propagated with
    n_traj = ceil(1/epsilon_map) trajectories so that the multinomial error of each output's populations sits below it."""
    crosstalk_threshold: float = 1e-3
    """GATE_LOCAL (Section 5.4): a neighbour receiving crosstalk light with |epsilon| at or above this joins the gate-local
    space; below it the leaked light is dropped and its rotation sin^2(eps theta/2) added to the reported bound."""
    register_dm_max_qubits: int = 12
    """GATE_LOCAL carries the register as a density matrix up to this many qubits and as a stochastic pure-state ensemble
    beyond (Section 5.4), the extracted map applied by Kraus sampling."""
    register_ensemble: int = 64
    """Members of the pure-state ensemble of a GATE_LOCAL register above ``register_dm_max_qubits`` qubits."""

    def __post_init__(self) -> None:
        if self.atol <= 0.0 or self.rtol <= 0.0 or self.nsteps <= 0:
            raise ValueError("tolerances and nsteps must be positive")
        if not 0.0 < self.map_accuracy < 1.0 or not 0.0 <= self.crosstalk_threshold <= 1.0:
            raise ValueError(
                "map_accuracy is a fraction in (0, 1) and crosstalk_threshold a Rabi ratio in [0, 1]"
            )
        if self.register_dm_max_qubits < 1 or self.register_ensemble < 1:
            raise ValueError("register_dm_max_qubits and register_ensemble are positive")
        if not 0.0 < self.freeze_alpha_max < 1.0 or not 0.0 < self.branch_weight_min < 1.0:
            raise ValueError("freeze_alpha_max and branch_weight_min are fractions in (0, 1)")
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
        if not 0.0 < self.boundary_population_max < 1.0:
            raise ValueError("boundary_population_max is a population fraction in (0, 1)")
        if self.ntraj < 1 or self.mesolve_dimension_max < 1:
            raise ValueError("ntraj and mesolve_dimension_max are positive")


@dataclass(frozen=True)
class Traces:
    """What ``run_pulses()`` returns (Section 14.3)."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    reduced_internal: tuple[Qobj, ...]
    mode_occupations: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
    """(time, "traj{k}:{channel}") of every quantum jump of the trajectory path (M7); empty on the density-matrix paths."""
    final: State
    boundary_population: dict[int, float]


@dataclass(frozen=True)
class ChannelSummary:
    """Process tomography of one pulse (Section 5.4)."""

    choi: np.ndarray
    cp_tp_residual: tuple[float, float]
    """Residual against the completely-positive cone and the trace-preserving affine set after Dykstra projection."""
    n_traj: int
    average_gate_infidelity: float
    pauli_twirled: dict[str, float]
    depolarizing_rate: float
    """epsilon with Lambda_eps(rho) = (1 - eps) rho + eps/(4^n - 1) sum_{P != I} P rho P (Section 6.8)."""


class PulseEngine(Protocol):
    """The one entry point that run/, calibration/ and experiments/ share (Appendix E)."""

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


# ---- the joint-exact engine of milestone M2 (dissipative paths M7) -------------------------------------------------------------


@dataclass(frozen=True)
class SegmentReport:
    """What one integration segment did (Section 5.7 diagnostics, per segment)."""

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
    """sesolve, mesolve or mcsolve (M7)."""
    n_collapse_ops: int = 0
    channels: tuple[str, ...] = ()
    """The channel names active on the segment (deduplicated by kind)."""


@dataclass(frozen=True)
class EngineReport:
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
    """Per resolved mode, the highest Fock index populated above the boundary threshold during any pulse (Section 5.5; M9a)."""
    margin_reached: dict[int, int] = field(default_factory=dict)
    """Per resolved mode, the smallest margin (levels) the cap kept above the populated range during the pulses (Section 5.1.1)."""

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
    """JOINT_EXACT pulse engine (Section 5.4): every pulse through the one builder, the joint state evolved exactly.

    ``store_per_segment`` stored points per segment (endpoints included); ``channels`` explicit collapse operators;
    ``device_channels`` assembles the device's own (``NoiseModel.channels`` plus, per segment and under the SolverOptions
    switches, the scattering and intensity-noise operators of the active pulses); ``levels_by_ion`` the register level maps
    of ions with d > 2 (leakage); ``hardware_chain`` passes the schedule through Section 7.10's chain (also gated by
    ``SolverOptions.hardware_chain``); ``max_growth_retries`` cap-raising retries when the boundary monitor trips (Section
    5.5). ``last_report`` carries the diagnostics of the most recent run.
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
    """The CalibrationTable the schedule's programmed drives were built from (M8): when given, ``control.played`` converts every
    requested Rabi frequency, believed Stark shift and believed crosstalk into what the ions see through the device's derived
    values before the hardware chain; None plays the schedule's values as physical (the M2 to M7 behaviour, exact for a
    surrogate table whose seeds are the derived values)."""
    pure_branches: bool = True
    """Evolve a density-matrix input without collapse operators (no explicit ``channels``, ``device_channels`` False) as the
    weighted pure branches of its eigen-decomposition, each through ``sesolve`` (the Fock-sum path of Section 5.3: a thermal
    motional state is a mixture of Fock states), rather than through ``mesolve`` on the D^2 Liouvillian, which costs D
    times more per step (Section 5.3; 58 ms against 68 us per right-hand side at D = 400). Branches below
    ``SolverOptions.branch_weight_min`` are dropped, the weights renormalized and the dropped weight reported in the notes,
    exactly as ``run()`` does for the initial mixture. False keeps the density-matrix reference path."""
    closed_form_constant: bool = True
    """Propagate a segment whose Hamiltonian is constant (an idle interval, a zero-envelope pulse) by its exact propagator
    instead of the ODE ladder: e^{-iHt} by the diagonal phases (H_mot + H_int are diagonal in the Fock x computational basis)
    or by one Hermitian eigendecomposition, and with the device's collapse operators present, the master equation in the
    frame rotating with H, where every collapse operator that is an eigenoperator of ad_H (the heating ladder operators,
    sigma_z, a^dag a) keeps its dissipator unchanged and the Liouvillian loses its fast oscillation (``_closed_form_segment``).
    The result is the same state to the solver tolerance; the segment reports integrator ``exact``. False forces the ODE
    ladder on every segment (the Section 5.3 step-density measurements)."""
    last_report: EngineReport | None = None

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
        """State-based process tomography of one pulse, a gate's pulse group or a whole Schedule on ``space`` (Section 5.4; M9a).

        Every one of the prod_i d_i^2 linearly independent pure internal inputs of ``dynamics.tomography.input_states`` is
        propagated through ``run_pulses`` from the motional state of ``motional_model`` (the tracked reduced density matrices
        of the resolved modes, their thermal states where none is tracked, the frozen modes' Fock populations as weighted
        branches), the Choi matrix is reconstructed by least squares and projected onto CP and TP by Dykstra's alternating
        projection, and the summary of Section 6.8 is computed against ``ideal`` (the gate's ideal unitary on the space's ions
        in factor order; NaN infidelities without one). The full record, including the motional outputs the GATE_LOCAL model
        tracks, is :meth:`tomography`.
        """
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
        """The process tomography behind :meth:`process_tomography`, with everything GATE_LOCAL tracks (Section 5.4 (b))."""
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
        while True:
            try:
                return self._run(
                    device, schedule, current_state, current_space, sample, seeds, options, retries
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
                new_space = current_space.grown(
                    trip.mode, trip.add if trip.reason == "margin" and trip.add > 0 else self.growth_levels
                )
                joint = current_state.joint
                if joint is None:
                    raise
                new_joint = regrid_state(joint, current_space, new_space)
                current_state = replace(current_state, joint=new_joint)
                current_space = new_space

    # ---- internals ---------------------------------------------------------------------------------------------------

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
    ) -> Traces:
        from qutip_trap.control.hardware import apply_hardware_chain
        from qutip_trap.dynamics.evolve import LARGE_MODE_DIMENSION, evolve
        from qutip_trap.dynamics.hamiltonian import BuilderOptions, build_hamiltonian
        from qutip_trap.hilbert.operators import thermal_populations
        from qutip_trap.hilbert.truncation import boundary_populations
        from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, key_frozen_n

        bopts = self.builder_options if isinstance(self.builder_options, BuilderOptions) else BuilderOptions()
        joint = state.joint
        if joint is None:
            raise ValueError("JOINT_EXACT needs a joint state; build one with HilbertSpace.initial_state")
        if joint.shape[0] != space.dimension:
            raise ValueError("the state does not live on the given space")
        notes: list[str] = []
        # the played chain of Section 7.3 (M8): requested -> physical through the device's derived values
        sched = schedule
        hw_notes: tuple[str, ...] = ()
        if self.table is not None:
            from qutip_trap.control.played import physical_schedule

            sched, played_notes = physical_schedule(device, sched, self.table)
            notes.extend(played_notes)
        # the hardware chain of Section 7.10 (M7)
        if self.hardware_chain and options.hardware_chain:
            rng_jitter = np.random.default_rng(seeds.child(sample.sample_id, 0, 0, 0, "timing_jitter"))
            sched, hw_notes = apply_hardware_chain(sched, device.hardware, rng=rng_jitter)
        # frozen spectators: the shot's Fock states (Section 5.2), from the sample or drawn from the keyed seeds
        frozen_n: dict[int, int] = {}
        for m in space.frozen:
            key = key_frozen_n(m)
            if key in sample.values:
                frozen_n[m] = int(round(sample.values[key]))
            else:
                nbar_m = float(state.motional.nbar.get(m, 0.0))
                if nbar_m <= 0.0:
                    frozen_n[m] = 0
                else:
                    rng = np.random.default_rng(seeds.child(sample.sample_id, 0, 0, 0, key))
                    probs = thermal_populations(nbar_m, int(60 + 40 * nbar_m))
                    probs = probs / probs.sum()
                    frozen_n[m] = int(rng.choice(len(probs), p=probs))
        # the state-independent collapse operators
        static_ops: list[CollapseOp] = list(self.channels)
        if self.device_channels:
            static_ops.extend(device.noise.channels(device, space))
        # segments at every pulse boundary and idle boundary; the state is given at the schedule's declared start
        # (``Schedule.t0_s``, a GATE_LOCAL step's start) or at min(0, the first cut) as M2 to M8 did
        starts = [p.t_start_s for p in sched.pulses] + [a for a, _ in sched.idle]
        t0 = float(sched.t0_s) if sched.t0_s is not None else min(starts + [0.0])
        t_end = (
            sched.pulses_end_s
        )  # the measurement event that may follow is the readout stage's, not free evolution
        if t_end < t0:
            t_end = t0
        cuts = {t0, t_end}
        for p in sched.pulses:
            cuts.update((p.t_start_s, p.t_end_s))
        for a, b in sched.idle:
            cuts.update((a, b))
        edges = _merge_cuts(sorted(t for t in cuts if t0 <= t <= t_end))
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
        n_store = max(int(self.store_per_segment), 2)
        first = True
        largest_mode = max([m.d for m in space.resolved], default=0)
        atol_mc = options.atol if largest_mode <= LARGE_MODE_DIMENSION else max(options.atol, 1e-8)
        for seg_index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if b <= a:
                continue
            active = [p for p in sched.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
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
            seg_ops = self._segment_channels(device, active, space, built, static_ops, options, notes)
            c_ops = [c.op for c in seg_ops]
            extra = [t for t in self.store_times_s if a < t < b]
            times = np.array(sorted(set(np.linspace(a, b, n_store).tolist()) | set(extra)))
            sel = slice(1, None) if not first else slice(0, None)
            seg_method = "sesolve"
            integrator = options.integrators[0]
            atol_used = options.atol
            rhs_evals: int | None = None
            retries_seg: tuple[str, ...] = ()
            # ---- integrate ----------------------------------------------------------------------------------------
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
                # the exact propagator of a constant Hamiltonian (Section 5.3: no ODE where none is needed)
                kets, rho = closed.kets, closed.rho
                for k in e_keys:
                    expect_all[k].append(closed.expect[k][sel])
                reduced.extend(closed.reduced[sel])
                seg_method = closed.method
                integrator = closed.integrator
                atol_used = closed.atol
                rhs_evals = None
                retries_seg = closed.retries
            elif not c_ops:
                if kets is not None:
                    new_kets: list[qt.Qobj] = []
                    exp_acc = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
                    red_acc: list[np.ndarray] = []
                    for psi, w_k in zip(kets, weights):
                        ev = evolve(
                            built.H,
                            psi,
                            times,
                            e_ops=dict(zip(e_keys, e_list)),
                            options=options,
                            store_states=True,
                            omega_max_rad_s=built.omega_max_rad_s or None,
                            largest_mode_dimension=largest_mode,
                            counter_calls=built.counter.count,
                            calls_per_rhs=built.n_drive_terms,
                        )
                        new_kets.append(ev.final)
                        for k in e_keys:
                            exp_acc[k] += w_k * np.asarray(ev.expect[k])
                        assert ev.states is not None
                        red_acc.append(
                            w_k * np.array([space.internal_marginal(st).full() for st in ev.states[sel]])
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
                    if len(kets) == 1 and options.ntraj > 1:
                        kets = [kets[0]] * options.ntraj
                        weights = [1.0 / options.ntraj] * options.ntraj
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
                    }
                    solver = qt.MCSolver(built.H, c_ops, options=mc_opts)
                    new_kets = []
                    exp_acc = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
                    red_acc = []
                    for k_traj, (psi, w_k) in enumerate(zip(kets, weights)):
                        seed = seeds.child(sample.sample_id, k_traj, 0, 0, f"mcsolve[{seg_index}]")
                        res = solver.run(psi, times, ntraj=1, e_ops=e_list, seeds=[seed])
                        traj = res.trajectories[0]
                        new_kets.append(traj.final_state)
                        for idx, k in enumerate(e_keys):
                            exp_acc[k] += w_k * np.asarray(traj.expect[idx])
                        red_acc.append(
                            w_k * np.array([space.internal_marginal(st).full() for st in traj.states[sel]])
                        )
                        for t_c, which in zip(res.col_times[0], res.col_which[0]):
                            jumps.append((float(t_c), f"traj{k_traj}:{seg_ops[int(which)].channel}"))
                    kets = new_kets
                    for k in e_keys:
                        expect_all[k].append(exp_acc[k][sel])
                    dims_int = [list(space.ion_dims), list(space.ion_dims)]
                    for arr in np.sum(np.stack(red_acc), axis=0):
                        reduced.append(qt.Qobj(arr, dims=dims_int))
                    integrator, atol_used = options.integrators[0], atol_mc
                    rhs_evals = built.rhs_evaluations
                    seg_method = "mcsolve"
                    method_used = "mcsolve"
            if seg_method == "mesolve":
                method_used = "mesolve" if method_used != "mcsolve" else method_used
            times_all.append(times[sel])
            first = False
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
            steps_per_period = None
            if rhs_evals and built.omega_max_rad_s > 0.0:
                n_steps = rhs_evals / 12.0
                periods = (b - a) * built.omega_max_rad_s / (2.0 * math.pi)
                steps_per_period = float(n_steps / periods) if periods > 0 else None
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
                )
            )
            if active:
                for m, v in bpop.items():
                    if v > options.boundary_population_max and space.mode_class(m) in ("resolved", "enr"):
                        raise _BoundaryTrip(m, v)
                if options.margin_check and space.resolved:
                    # Section 5.5: the cap's margin above the populated range must stay above the Section 5.1.1 margin for
                    # the segment's eta on every resolved mode the segment drives; a deficit raises the cap by it and repeats.
                    # The range is measured at the boundary threshold as a fraction of the MIXTURE's population: a branch of
                    # weight w (run() and the tomography evolve the Fock branches one by one) reads tail/w (M9a)
                    branch_w = min(max(sample.get(KEY_BRANCH_WEIGHT, 1.0), 1e-300), 1.0)
                    tail = min(options.boundary_population_max / branch_w, 0.5)
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
                        need = required_margin(eta_seg[m])
                        if margin < need:
                            raise _BoundaryTrip(m, bpop.get(m, 0.0), add=need - margin, reason="margin")
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
        )


EIGH_DIMENSION_MAX = 4096
"""Above this joint dimension a constant but non-diagonal Hamiltonian goes through the ODE ladder rather than a dense eigh."""


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


def _mixture(kets: list[qt.Qobj], weights: list[float]) -> qt.Qobj:
    """sum_k w_k |psi_k><psi_k| of weighted kets (a single ket's projector when there is one)."""
    if len(kets) == 1:
        return kets[0].proj()
    return sum((w * k.proj() for k, w in zip(kets, weights)), 0.0 * kets[0].proj())


def _pure_branches(rho: qt.Qobj, weight_min: float) -> tuple[list[qt.Qobj], list[float], float]:
    """The eigen-decomposition of a density matrix into pure branches: (kets, weights >= ``weight_min`` renormalized, dropped
    weight), heaviest first. A product of thermal states is a mixture of Fock states, so this is the Fock sum of Section 5.3
    (``run()`` enumerates the same branches from the occupations before it builds any state)."""
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
    """lambda with [H, op] = lambda op for the diagonal H of ``energies``: E_row - E_col equal on every non-zero element (the
    heating operators a and a^dag, sigma_z, a^dag a); None when ``op`` is not an eigenoperator of ad_H."""
    coo = op.to("CSR").data.as_scipy().tocoo()
    if coo.nnz == 0:
        return 0.0
    lam = energies[coo.row] - energies[coo.col]
    tol = 1e-9 * max(float(np.max(np.abs(energies))), 1.0)
    if float(np.ptp(lam)) > tol:
        return None
    return float(lam[0])


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
    """Propagate a segment with the constant Hamiltonian ``h`` over ``times`` without an ODE solve of the oscillatory part.

    Without collapse operators the propagator is e^{-i h tau}: the diagonal phases when ``h`` is diagonal (idle intervals:
    H_mot + H_int + a constant Stark shift), else one Hermitian eigendecomposition (a constant anharmonic or curvature term).
    With collapse operators the master equation is integrated in the frame rotating with the diagonal ``h``, where H vanishes
    and every collapse operator that is an eigenoperator of ad_H picks up only a phase, which its dissipator does not see; the
    stored states are rotated back, so expectations and marginals are in the Schroedinger picture. Returns None when the
    closed form does not apply (a non-diagonal H with collapse operators, an operator that is not an eigenoperator, the
    trajectory path); the caller then integrates as before. Nothing here is an approximation: the frame change is unitary
    and the phases are exact.
    """
    energies = _diagonal_energies(h)
    taus = np.asarray(times, dtype=float) - float(times[0])
    dims_int = [list(space.ion_dims), list(space.ion_dims)]
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
            for psi, w_k in zip(kets, weights):
                states = propagate_ket(psi)
                new_kets.append(states[-1])
                for k, op in zip(e_keys, e_list):
                    expect[k] += w_k * np.array([qt.expect(op, s) for s in states], dtype=complex)
                marg = w_k * np.array([space.internal_marginal(s).full() for s in states])
                red_acc = marg if red_acc is None else red_acc + marg
            assert red_acc is not None
            reduced = [qt.Qobj(arr, dims=dims_int) for arr in red_acc]
            return _ClosedForm(new_kets, None, expect, reduced, "sesolve", "exact", options.atol, ())
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
    """The highest Fock index of ``mode`` the (weighted) state populates above ``tail`` (Section 5.5)."""
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
    """The truncation monitor tripped (Section 5.5): the boundary population exceeded the threshold (``reason``
    "boundary") or the cap's margin above the populated range fell below the Section 5.1.1 requirement ("margin")."""

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


__all__ += ["EngineReport", "JointExactEngine", "SegmentReport", "TruncationLimit"]
