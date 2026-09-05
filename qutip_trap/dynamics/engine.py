"""The pulse engine protocol and the run-time state records (PLAN.md Sections 3.4, 5.3, 5.4, 11; Appendix E).

``PulseEngine.run_pulses`` is the one entry point that ``run/``, ``calibration/`` and ``experiments/``
share. Randomness comes from one root ``SeedSequence`` per run, spawned deterministically by (sample,
trajectory, shot, ion, channel) so that every variate's key is independent of execution order and of
truncation retries (Section 3.4); reproducibility over 1 and 18 workers is a tolerance test (10^-12), not a
bitwise one, because ``MultiTrajResult`` accumulates in completion order (Section 3.4).
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
    from qutip_trap.noise.sampling import NoiseSample

# QuTiP multistep integrators, never used (Section 5.3: the escalation ladder is dop853 then vern9)
MULTISTEP_INTEGRATORS: frozenset[str] = frozenset({"adams", "bdf", "lsoda", "vode", "zvode"})
ALLOWED_INTEGRATORS: frozenset[str] = frozenset(
    {"dop853", "vern7", "vern9", "tsit5", "explicit_rk", "krylov", "diag"}
)


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
    """The joint ket/density matrix when the run holds one (JOINT_EXACT)."""
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

    def __post_init__(self) -> None:
        if self.atol <= 0.0 or self.rtol <= 0.0 or self.nsteps <= 0:
            raise ValueError("tolerances and nsteps must be positive")
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


@dataclass(frozen=True)
class Traces:
    """What ``run_pulses()`` returns (Section 14.3)."""

    times_s: np.ndarray
    expectations: dict[str, np.ndarray]
    reduced_internal: tuple[Qobj, ...]
    mode_occupations: dict[int, np.ndarray]
    alpha_m: dict[int, np.ndarray]
    jumps: tuple[tuple[float, str], ...]
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
    "MotionalModel",
    "PulseEngine",
    "SeedSpec",
    "SolverOptions",
    "State",
    "Traces",
]


# ---- the joint-exact engine of milestone M2 ---------------------------------------------------------------------------------


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


@dataclass(frozen=True)
class EngineReport:
    segments: tuple[SegmentReport, ...]
    frozen_n: dict[int, int]
    growth_retries: int
    space: HilbertSpace
    drive_records: tuple[object, ...]

    @property
    def approximations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for seg in self.segments:
            for a in seg.approximations:
                if a not in seen:
                    seen.append(a)
        return tuple(seen)


@dataclass
class JointExactEngine:
    """JOINT_EXACT pulse engine (Section 5.4): every pulse through the one builder, the joint state evolved exactly.

    ``store_per_segment`` stored points per segment (endpoints included); ``channels`` explicit collapse operators
    (M7 assembles them from the device); ``max_growth_retries`` cap-raising retries when the boundary monitor trips
    (Section 5.5). ``last_report`` carries the diagnostics of the most recent run.
    """

    builder_options: object | None = None
    store_per_segment: int = 2
    store_times_s: tuple[float, ...] = ()
    """Extra absolute times at which the traces are stored (the experiments' scan points)."""
    channels: tuple[CollapseOp, ...] = ()
    qubit_shifts_hz: dict[int, float] = field(default_factory=dict)
    max_growth_retries: int = 3
    growth_levels: int = 4
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
        from qutip_trap.dynamics.evolve import evolve
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
        # segments at every pulse boundary and idle boundary
        t0 = min([p.t_start_s for p in schedule.pulses] + [a for a, _ in schedule.idle] + [0.0])
        t_end = schedule.duration_s
        cuts = {t0, t_end}
        for p in schedule.pulses:
            cuts.update((p.t_start_s, p.t_end_s))
        for a, b in schedule.idle:
            cuts.update((a, b))
        edges = sorted(t for t in cuts if t0 <= t <= t_end)
        c_ops = [c.op for c in self.channels]
        e_ops: dict[str, qt.Qobj] = {}
        for i in range(space.n_ions):
            e_ops[f"P1[{i}]"] = space.projector(i, 1)
        carried = [m.mode for m in space.resolved] + (list(space.enr_group[0]) if space.enr_group else [])
        for m in carried:
            e_ops[f"n[{m}]"] = space.number(m)
            e_ops[f"a[{m}]"] = space.annihilation(m)
        times_all: list[np.ndarray] = []
        expect_all: dict[str, list[np.ndarray]] = {k: [] for k in e_ops}
        reduced: list[qt.Qobj] = []
        segments: list[SegmentReport] = []
        records: list[object] = []
        worst_boundary: dict[int, float] = {m: 0.0 for m in carried}
        current = joint
        n_store = max(int(self.store_per_segment), 2)
        first = True
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            active = [p for p in schedule.pulses if p.t_start_s <= a + 1e-15 and p.t_end_s >= b - 1e-15]
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
            extra = [t for t in self.store_times_s if a < t < b]
            times = np.array(sorted(set(np.linspace(a, b, n_store).tolist()) | set(extra)))
            ev = evolve(
                built.H,
                current,
                times,
                c_ops=c_ops,
                e_ops=e_ops,
                options=options,
                store_states=True,
                omega_max_rad_s=built.omega_max_rad_s or None,
                largest_mode_dimension=max([m.d for m in space.resolved], default=0),
                counter_calls=built.counter.count,
                calls_per_rhs=built.n_drive_terms,
            )
            current = ev.final
            assert ev.states is not None
            sel = slice(1, None) if not first else slice(0, None)
            times_all.append(times[sel])
            for k in e_ops:
                expect_all[k].append(np.asarray(ev.expect[k])[sel])
            for st in ev.states[sel]:
                reduced.append(space.internal_marginal(st))
            first = False
            bpop = boundary_populations(current, space)
            for m, v in bpop.items():
                worst_boundary[m] = max(worst_boundary.get(m, 0.0), v)
            steps_per_period = None
            if ev.rhs_evaluations and built.omega_max_rad_s > 0.0:
                n_steps = ev.rhs_evaluations / 12.0
                periods = (b - a) * built.omega_max_rad_s / (2.0 * math.pi)
                steps_per_period = float(n_steps / periods) if periods > 0 else None
            segments.append(
                SegmentReport(
                    a,
                    b,
                    tuple(p.gate_id for p in active),
                    ev.integrator,
                    ev.atol,
                    ev.rhs_evaluations,
                    steps_per_period,
                    built.approximations,
                    bpop,
                    ev.retries,
                )
            )
            if active:
                for m, v in bpop.items():
                    if v > options.boundary_population_max and space.mode_class(m) == "resolved":
                        raise _BoundaryTrip(m, v)
        times_arr = np.concatenate(times_all) if times_all else np.array([t0])
        expect = {k: np.concatenate(v) if v else np.array([]) for k, v in expect_all.items()}
        motional_reduced: dict[int, qt.Qobj] = {}
        nbar: dict[int, float] = {}
        for m in carried:
            rho = space.mode_marginal(current, m)
            motional_reduced[m] = rho
            nbar[m] = float(np.real(qt.expect(qt.num(rho.shape[0]), rho)))
        for m in space.frozen:
            nbar[m] = float(state.motional.nbar.get(m, 0.0))
        final = State(
            internal=space.internal_marginal(current),
            motional=MotionalModel(reduced=motional_reduced, nbar=nbar, frozen=tuple(space.frozen)),
            joint=current,
            provenance=tuple(state.provenance) + ("m2.joint_exact_engine",),
        )
        self.last_report = EngineReport(
            segments=tuple(segments),
            frozen_n=frozen_n,
            growth_retries=growth_retries,
            space=space,
            drive_records=tuple(records),
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
            jumps=(),
            final=final,
            boundary_population=worst_boundary,
        )


class _BoundaryTrip(Exception):
    def __init__(self, mode: int, worst: float) -> None:
        super().__init__(f"boundary population {worst:.3e} on mode {mode}")
        self.mode = mode
        self.worst = worst


class TruncationLimit(RuntimeError):
    """The cap-raising retries of Section 5.5 were exhausted."""


__all__ += ["EngineReport", "JointExactEngine", "SegmentReport", "TruncationLimit"]
