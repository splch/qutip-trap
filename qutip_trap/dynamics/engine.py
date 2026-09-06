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
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import qutip as qt

from qutip_trap.dynamics.channels import CollapseOp

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
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

    def __post_init__(self) -> None:
        if self.atol <= 0.0 or self.rtol <= 0.0 or self.nsteps <= 0:
            raise ValueError("tolerances and nsteps must be positive")
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
    last_report: EngineReport | None = None

    def process_tomography(
        self,
        device: Device,
        pulse: Pulse,
        space: HilbertSpace,
        motional_model: MotionalModel,
        sample: NoiseSample,
        seeds: SeedSpec,
    ) -> ChannelSummary:
        raise NotImplementedError("process_tomography is milestone M9a (GATE_LOCAL, PLAN.md Section 5.4)")

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
                    raise TruncationLimit(
                        f"boundary population {trip.worst:.3e} on mode {trip.mode} still exceeds "
                        f"{options.boundary_population_max:.1e} after {retries} cap-raising retries"
                    ) from trip
                retries += 1
                new_space = current_space.grown(trip.mode, self.growth_levels)
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
        from qutip_trap.noise.sampling import key_frozen_n

        bopts = self.builder_options if isinstance(self.builder_options, BuilderOptions) else BuilderOptions()
        joint = state.joint
        if joint is None:
            raise ValueError("JOINT_EXACT needs a joint state; build one with HilbertSpace.initial_state")
        if joint.shape[0] != space.dimension:
            raise ValueError("the state does not live on the given space")
        notes: list[str] = []
        # the hardware chain of Section 7.10 (M7)
        sched = schedule
        hw_notes: tuple[str, ...] = ()
        if self.hardware_chain and options.hardware_chain:
            rng_jitter = np.random.default_rng(seeds.child(sample.sample_id, 0, 0, 0, "timing_jitter"))
            sched, hw_notes = apply_hardware_chain(schedule, device.hardware, rng=rng_jitter)
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
        # segments at every pulse boundary and idle boundary
        t0 = min([p.t_start_s for p in sched.pulses] + [a for a, _ in sched.idle] + [0.0])
        t_end = (
            sched.pulses_end_s
        )  # the measurement event that may follow is the readout stage's, not free evolution
        cuts = {t0, t_end}
        for p in sched.pulses:
            cuts.update((p.t_start_s, p.t_end_s))
        for a, b in sched.idle:
            cuts.update((a, b))
        edges = _merge_cuts(sorted(t for t in cuts if t0 <= t <= t_end))
        e_keys: list[str] = []
        e_list: list[qt.Qobj] = []
        for i in range(space.n_ions):
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
        # the state: a list of trajectory kets, or one density matrix
        kets: list[qt.Qobj] | None = [joint] if joint.isket else None
        rho: qt.Qobj | None = None if joint.isket else joint
        method_used = "sesolve" if joint.isket else "mesolve"
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
            if not c_ops:
                if kets is not None:
                    new_kets: list[qt.Qobj] = []
                    exp_acc = {k: np.zeros(times.size, dtype=complex) for k in e_keys}
                    red_acc: list[np.ndarray] = []
                    for psi in kets:
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
                            exp_acc[k] += np.asarray(ev.expect[k]) / len(kets)
                        assert ev.states is not None
                        red_acc.append(
                            np.array([space.internal_marginal(st).full() for st in ev.states[sel]])
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
                    for arr in np.mean(np.stack(red_acc), axis=0):
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
                        rho = (
                            kets[0].proj()
                            if len(kets) == 1
                            else sum((k.proj() for k in kets), 0.0 * kets[0].proj()) / len(kets)
                        )
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
                    for k_traj, psi in enumerate(kets):
                        seed = seeds.child(sample.sample_id, k_traj, 0, 0, f"mcsolve[{seg_index}]")
                        res = solver.run(psi, times, ntraj=1, e_ops=e_list, seeds=[seed])
                        traj = res.trajectories[0]
                        new_kets.append(traj.final_state)
                        for idx, k in enumerate(e_keys):
                            exp_acc[k] += np.asarray(traj.expect[idx]) / len(kets)
                        red_acc.append(
                            np.array([space.internal_marginal(st).full() for st in traj.states[sel]])
                        )
                        for t_c, which in zip(res.col_times[0], res.col_which[0]):
                            jumps.append((float(t_c), f"traj{k_traj}:{seg_ops[int(which)].channel}"))
                    kets = new_kets
                    for k in e_keys:
                        expect_all[k].append(exp_acc[k][sel])
                    dims_int = [list(space.ion_dims), list(space.ion_dims)]
                    for arr in np.mean(np.stack(red_acc), axis=0):
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
                for psi in kets:
                    for m, v in boundary_populations(psi, space).items():
                        bpop[m] = bpop.get(m, 0.0) + float(v) / len(kets)
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
                    if v > options.boundary_population_max and space.mode_class(m) == "resolved":
                        raise _BoundaryTrip(m, v)
        times_arr = np.concatenate(times_all) if times_all else np.array([t0])
        expect = {k: np.concatenate(v) if v else np.array([]) for k, v in expect_all.items()}
        # ---- the final state ----------------------------------------------------------------------------------
        final_joint: qt.Qobj | None
        if kets is not None:
            if len(kets) == 1:
                final_joint = kets[0]
            elif space.dimension <= 1024:
                final_joint = sum((k.proj() for k in kets), 0.0 * kets[0].proj()) / len(kets)
            else:
                final_joint = None
                notes.append(
                    f"trajectory ensemble of {len(kets)} at dimension {space.dimension} not averaged into a joint density matrix; "
                    "the reduced states are the averages"
                )
            internal = sum(
                (space.internal_marginal(k) for k in kets), 0.0 * space.internal_marginal(kets[0])
            ) / len(kets)
            motional_reduced: dict[int, qt.Qobj] = {}
            nbar: dict[int, float] = {}
            for m in carried:
                rho_m = sum(
                    (space.mode_marginal(k, m) for k in kets), 0.0 * space.mode_marginal(kets[0], m)
                ) / len(kets)
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


def _merge_cuts(times: list[float], tolerance_s: float = 1e-12) -> list[float]:
    """Collapse cut points closer than a picosecond: a pulse end and an idle start that differ by float round-off must not
    open a segment on which one pulse has ended and its tail has begun (the builder takes one segment's pulses)."""
    out: list[float] = []
    for t in times:
        if out and t - out[-1] <= tolerance_s:
            continue
        out.append(t)
    return out


class _BoundaryTrip(Exception):
    def __init__(self, mode: int, worst: float) -> None:
        super().__init__(f"boundary population {worst:.3e} on mode {mode}")
        self.mode = mode
        self.worst = worst


class TruncationLimit(RuntimeError):
    """The cap-raising retries of Section 5.5 were exhausted."""


__all__ += ["EngineReport", "JointExactEngine", "SegmentReport", "TruncationLimit"]
