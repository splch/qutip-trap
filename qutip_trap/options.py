"""The option objects of a run: ``Physics`` (which effects are simulated), ``Numerics`` (how the integration is done,
nested by concern) and ``Readout`` (how the photon record is read). ``Machine`` holds one of each. ``SolverOptions`` is what
``Numerics`` is built into (``Numerics.to_solver_options``), and validation delegates to it, so the errors are the ones a
``SolverOptions`` raises.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from qutip_trap.control.schedule import CrosstalkSuppression
from qutip_trap.dynamics.engine import LindbladMethod, RecoilOption, SolverOptions

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.readout.discriminate import Discriminator

MapKind = Literal["serial", "parallel", "loky"]
ReadoutMode = Literal["fast", "full"]
Scattering = Literal["estimate", "channels"]


def _group_from(cls: type[Any], value: object, name: str) -> Any:
    """A nested group from an instance (kept) or a mapping (built); anything else is refused by name."""
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_mapping(value)
    raise TypeError(f"{name} takes a {cls.__name__} or a mapping of its fields, got {type(value).__name__}")


class _FromMapping:
    """``from_mapping`` for a frozen dataclass of scalars: unknown keys are refused (QuTiP's rule for its options dict)."""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        # every subclass is a frozen dataclass; the mixin itself is not, which the cast tells the type checker
        names = {f.name for f in dataclasses.fields(cast("type[DataclassInstance]", cls))}
        unknown = sorted(set(mapping) - names)
        if unknown:
            raise ValueError(f"{cls.__name__} has no field {unknown}; the fields are {sorted(names)}")
        return cls(**dict(mapping))

    def asdict(self) -> dict[str, Any]:
        """The fields as a plain dictionary (nested groups as dictionaries), the form ``from_mapping`` accepts."""
        return dataclasses.asdict(cast("DataclassInstance", self))


@dataclass(frozen=True)
class Integration(_FromMapping):
    """The ODE integration of every segment (Section 5.3): tolerances, the escalation ladder and the frame."""

    atol: float = 1e-10
    """Absolute tolerance of the integrator (dimensionless amplitude)."""
    rtol: float = 1e-8
    """Relative tolerance of the integrator."""
    nsteps: int = 10**7
    """The integrator's step budget per segment."""
    integrators: tuple[str, ...] = ("dop853", "vern9")
    """The escalation ladder of Section 5.3; never a multistep method."""
    rotating_frame: bool = True
    """Integrate ket segments in the exact rotating frame of the diagonal H_0 (Section 5.2)."""
    propagator_cache: bool = True
    """Cache the propagator of internal-state-only segments (Section 11.3 item 5)."""
    store_marginals: bool = False
    """Store the Fock populations of every carried mode at every stored time as ``Traces.mode_marginal`` (0.4.0)."""

    def __post_init__(self) -> None:
        SolverOptions(**self.asdict())  # the same rules and messages as SolverOptions


@dataclass(frozen=True)
class Truncation(_FromMapping):
    """The Fock caps, the guards and the mode classes (Sections 5.1, 5.2, 5.5, 11.5)."""

    joint_dimension_max: int = 4096
    """The joint dimension above which ``FidelityLevel.AUTO`` routes to GATE_LOCAL (Section 11.5)."""
    nnz_max: int = 2 * 10**7
    """The drive-operator non-zero count above which AUTO routes to GATE_LOCAL."""
    mode_dimension_max: int = 64
    """The ceiling on one resolved mode's Fock dimension; a clamp warns (``TruncationWarning``) and is reported."""
    boundary_population_max: float = 1e-6
    """The population the cap may leave at its boundary before the monitor raises it (Section 5.5)."""
    freeze_chi_max_rad: float = 0.05
    """|chi_m| (rad) below which a spectator mode may be frozen rather than resolved (Section 5.2)."""
    freeze_alpha_max: float = 1e-4
    """|alpha_m|^2 (2 nbar + 1) below which a spectator may be frozen (Section 5.2)."""
    branch_weight_min: float = 1e-6
    """Weight below which a branch of the initial mixture is dropped and reported (Section 5.3)."""
    margin_check: bool = True
    """Compare every resolved cap's margin with the Section 5.1.1 margin after each pulse."""
    margin_element_tol: float | None = None
    """The interior-element tolerance the margin is derived from; None keeps the fixture margin."""
    caps: Mapping[int, int] | None = None
    """Per mode, a Fock dimension that overrides the cap rule (mode index -> d)."""
    enr_group: tuple[Sequence[int], int] | None = None
    """(modes, N_exc): carry these modes as one excitation-number-restricted factor (Section 11.3 item 1)."""
    space: HilbertSpace | None = None
    """A declared joint space that replaces the selection of Section 5.2 entirely."""

    def __post_init__(self) -> None:
        if self.caps is not None:
            object.__setattr__(self, "caps", {int(m): int(d) for m, d in dict(self.caps).items()})
        if self.enr_group is not None:
            modes, n_exc = self.enr_group
            object.__setattr__(self, "enr_group", (tuple(int(m) for m in modes), int(n_exc)))
        if self.enr_group is not None and self.space is not None:
            raise ValueError("give the ENR group inside the supplied space or as enr_group, not both")
        SolverOptions(**{k: v for k, v in self.asdict().items() if k not in ("caps", "enr_group", "space")})

    def asdict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["caps"] = None if self.caps is None else dict(self.caps)
        return out


@dataclass(frozen=True)
class Trajectories(_FromMapping):
    """How collapse operators are integrated (Sections 3.4, 5.3): the density matrix or keyed quantum-jump trajectories."""

    lindblad_method: LindbladMethod = "auto"
    """``mesolve``, ``mcsolve``, or ``auto`` = mesolve up to ``mesolve_dimension_max``."""
    mesolve_dimension_max: int = 128
    """The joint dimension up to which the density matrix is integrated under ``auto``."""
    ntraj: int = 64
    """Trajectories per pure initial state on the mcsolve path (a fixed keyed seed list)."""
    improved_sampling: bool = True
    """The no-jump trajectory as a deterministic member of weight p_no-jump (Section 5.3)."""
    trajectory_target_tol: float | None = None
    """``mcsolve``'s ``target_tol`` on the population e_ops; None keeps the fixed ``ntraj``."""
    e_ops_for_target_tol: bool = True
    """Register the population e_ops ``target_tol`` needs (Section 5.4)."""

    def __post_init__(self) -> None:
        SolverOptions(**self.asdict())


@dataclass(frozen=True)
class GateLocal(_FromMapping):
    """The GATE_LOCAL walk of Section 5.4: the map accuracy, the neighbour rule, the register carrier, the tomography."""

    map_accuracy: float = 1e-3
    """The tolerance of the channel map; a fraction in (0, 1) that keys the tomography's tolerances."""
    crosstalk_threshold: float = 1e-3
    """The Rabi ratio |epsilon| at or above which a neighbour joins the gate-local space."""
    register_dm_max_qubits: int = 12
    """Carry the register as a density matrix up to this many qubits, as a pure-state ensemble above."""
    register_ensemble: int = 64
    """Members of the pure-state ensemble above ``register_dm_max_qubits``."""
    tomography_isometry: bool = True
    """Read a unitary step's channel off the Stinespring isometry rather than the least-squares fit."""
    tomography_dropped_weight_max: float | None = None
    """The total motional-branch weight a step may drop (None = map_accuracy / 4; 0 keeps every branch)."""
    tomography_tolerance_keyed: bool = True
    """Integrate a unitary step with resolved modes at the tolerance the map accuracy warrants."""

    def __post_init__(self) -> None:
        SolverOptions(**self.asdict())


@dataclass(frozen=True)
class Parallel(_FromMapping):
    """The parallel maps of Section 11.3 item 9 and the two per-run counts that shape them."""

    map: MapKind = "parallel"
    """QuTiP's serial, ``multiprocessing`` (``parallel``) or ``loky`` map."""
    workers: int | None = None
    """Processes for the maps; None = every CPU QuTiP sees, capped by the memory rule."""
    samples: int | None = None
    """Dynamical samples per run; None = min(shots, 64) when the noise model has quasi-static content, else 1."""
    addressing: bool | None = None
    """Schedule single-qubit gates in parallel (Section 7.3); None = the device's ``HardwareChain.parallel_addressing``."""

    def __post_init__(self) -> None:
        if self.samples is not None and self.samples < 1:
            raise ValueError("samples is a positive count or None")
        SolverOptions(map=self.map, workers=self.workers)


@dataclass(frozen=True)
class Numerics(_FromMapping):
    """How the integration is done, nested so that a physicist reads the truncation policy without the tomography knobs
    (docs/api_proposal.md Section 4.7). ``to_solver_options(physics)`` is the ``SolverOptions`` a run integrates with."""

    integration: Integration = Integration()
    """Tolerances, the escalation ladder, the frame and the propagator cache."""
    truncation: Truncation = Truncation()
    """The caps, the guards, the mode classes, the branch cutoff and any declared space."""
    trajectories: Trajectories = Trajectories()
    """The Lindblad method and the trajectory count."""
    gate_local: GateLocal = GateLocal()
    """The GATE_LOCAL walk's accuracy, neighbour rule, register carrier and tomography."""
    parallel: Parallel = Parallel()
    """The maps, the workers, the sample count and parallel addressing."""
    convergence_check: bool = False
    """Repeat the evolution at ten times tighter tolerances and report the change (Section 5.5)."""

    def __post_init__(self) -> None:
        for name, cls in (
            ("integration", Integration),
            ("truncation", Truncation),
            ("trajectories", Trajectories),
            ("gate_local", GateLocal),
            ("parallel", Parallel),
        ):
            object.__setattr__(self, name, _group_from(cls, getattr(self, name), f"Numerics.{name}"))

    def to_solver_options(self, physics: Physics | None = None) -> SolverOptions:
        """The ``SolverOptions`` of a run: these numerics plus the three physics switches ``SolverOptions`` still carries
        (``scattering_channels``, ``scattering_recoil``, ``intensity_noise_channels``, ``hardware_chain``)."""
        phys = physics if physics is not None else Physics()
        return SolverOptions(
            **self.integration.asdict(),
            **{k: v for k, v in self.truncation.asdict().items() if k not in ("caps", "enr_group", "space")},
            **self.trajectories.asdict(),
            **self.gate_local.asdict(),
            map=self.parallel.map,
            workers=self.parallel.workers,
            convergence_check=self.convergence_check,
            scattering_channels=(phys.scattering == "channels"),
            scattering_recoil=phys.scattering_recoil,
            intensity_noise_channels=phys.intensity_noise_channels,
            hardware_chain=phys.hardware_chain,
        )

    @classmethod
    def from_solver_options(
        cls,
        options: SolverOptions | None = None,
        *,
        caps: Mapping[int, int] | None = None,
        space: HilbertSpace | None = None,
        enr_group: tuple[Sequence[int], int] | None = None,
        samples: int | None = None,
        addressing: bool | None = None,
    ) -> Numerics:
        """The numerics a ``SolverOptions`` carries (its four physics switches go to ``Physics.from_solver_options``; None is
        the default options), plus the five numerics that never lived on ``SolverOptions``: the explicit caps, a declared
        space and an ENR group (``Truncation``), the sample count and the parallel-addressing switch (``Parallel``), the
        homes of ``run``'s 0.1.0 keywords ``caps``, ``space``, ``enr_group``, ``samples`` and ``parallel``."""
        opts = options if options is not None else SolverOptions()
        d = dataclasses.asdict(opts)

        def pick(group: type[DataclassInstance]) -> dict[str, Any]:
            return {f.name: d[f.name] for f in dataclasses.fields(group) if f.name in d}

        return cls(
            integration=Integration(**pick(Integration)),
            truncation=Truncation(**pick(Truncation), caps=caps, space=space, enr_group=enr_group),
            trajectories=Trajectories(**pick(Trajectories)),
            gate_local=GateLocal(**pick(GateLocal)),
            parallel=Parallel(map=opts.map, workers=opts.workers, samples=samples, addressing=addressing),
            convergence_check=opts.convergence_check,
        )


@dataclass(frozen=True)
class Physics(_FromMapping):
    """Which physical effects a run simulates (docs/api_proposal.md Section 4.7). Every default is what ``run`` did in 0.1.0:
    the device's noise on, two register levels per ion, scattering as an estimate, the hardware chain applied."""

    noise: bool = True
    """Draw the device's dynamical samples and assemble its collapse operators; False runs the quiet nominal sample."""
    internal_levels: int = 2
    """Register levels per ion; more than 2 adds leakage levels and turns the scattering channels on (Section 4.5.5)."""
    scattering: Scattering = "estimate"
    """Photon scattering as a reported estimate, or as collapse operators (``channels``; Sections 4.5.5, 6.5)."""
    scattering_recoil: RecoilOption = "minimal"
    """The recoil discretization of the scattering operators: ``off``, ``minimal`` or ``vector``."""
    intensity_noise_channels: bool = True
    """The white part of the laser-intensity spectrum as the channel sqrt(D) H_drive(t) (Section 6.4)."""
    hardware_chain: bool = True
    """Pass the schedule through the control electronics of Section 7.10 before integrating."""
    stark_compensation: bool = True
    """Detune every pulse by the light shift the table believes (Section 7.5 item 7)."""
    crosstalk_suppression: CrosstalkSuppression = "none"
    """Section 6.6's echo schemes on the MS gates: ``none``, ``neighbour`` or ``local``."""
    entangler: Literal["ms", "zz"] = "ms"
    """The native entangling gate the compiler expands two-qubit gates into."""
    extra_channels: tuple[CollapseOp, ...] = ()
    """Explicit collapse operators added to the device's own."""
    builder: BuilderOptions | None = None
    """The Hamiltonian builder's options (frame, Lamb-Dicke order, kernel); None = the defaults."""
    t0_s: float = 0.0
    """The laboratory time (s) of the first shot, where drifts and the mains phase are evaluated (Section 7.5)."""
    shot_period_s: float | None = None
    """T_rep (s); None derives it from the preparation, the schedule and the detection window."""

    def __post_init__(self) -> None:
        if self.internal_levels < 2:
            raise ValueError("internal_levels is at least 2")
        if self.scattering not in ("estimate", "channels"):
            raise ValueError("scattering is 'estimate' or 'channels'")
        if self.entangler not in ("ms", "zz"):
            raise ValueError("entangler is 'ms' or 'zz'")
        if self.crosstalk_suppression not in ("none", "neighbour", "local"):
            raise ValueError("crosstalk_suppression is 'none', 'neighbour' or 'local'")
        if not math.isfinite(self.t0_s):
            raise ValueError("t0_s is a finite laboratory time in seconds")
        if self.shot_period_s is not None and not self.shot_period_s > 0.0:
            raise ValueError("shot_period_s is a positive period in seconds or None")
        if self.scattering_recoil not in ("off", "minimal", "vector"):
            raise ValueError("scattering_recoil is 'off', 'minimal' or 'vector'")
        object.__setattr__(self, "extra_channels", tuple(self.extra_channels))

    def asdict(self) -> dict[str, Any]:
        """The scalar fields as a dictionary; ``extra_channels`` and ``builder`` are kept as the objects they are."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_solver_options(cls, options: SolverOptions | None = None, **fields: Any) -> Physics:
        """The physics switches a ``SolverOptions`` carries (None: the defaults), plus any other ``Physics`` field as a
        keyword (``noise=False``, ``entangler="zz"``, ...)."""
        opts = options if options is not None else SolverOptions()
        return cls(
            scattering="channels" if opts.scattering_channels else "estimate",
            scattering_recoil=opts.scattering_recoil,
            intensity_noise_channels=opts.intensity_noise_channels,
            hardware_chain=opts.hardware_chain,
            **fields,
        )


@dataclass(frozen=True)
class Readout(_FromMapping):
    """How the photon record is read (Sections 5.7, 8.3): the fast POVM path or the full record, and the discriminator."""

    mode: ReadoutMode = "fast"
    """``fast`` applies the POVM to the joint outcome; ``full`` generates every photon record and discriminates it."""
    discriminator: Discriminator | None = None
    """The strategy over the record (threshold, time-resolved ML, adaptive, first-photon); None = the table's threshold."""
    povm_samples: int = 20_000
    """Records sampled per level per ion when a discriminator has no closed-form confusion (Section 8.4)."""

    def __post_init__(self) -> None:
        if self.mode not in ("fast", "full"):
            raise ValueError("readout mode is 'fast' or 'full'")
        if self.povm_samples < 1:
            raise ValueError("povm_samples is a positive count")

    def asdict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
