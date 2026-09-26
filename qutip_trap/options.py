"""The option objects of a run: ``Physics`` (which effects are simulated), ``Numerics`` (how the integration is done) and
``Readout`` (how the photon record is read); ``Machine`` holds one of each. The engine and every lower-level routine take
the ``Numerics``; ``Machine.engine`` builds the engine from the ``Physics``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qutip_trap.control.schedule import CrosstalkSuppression
    from qutip_trap.dynamics.channels import CollapseOp, RecoilOption
    from qutip_trap.dynamics.hamiltonian import BuilderOptions
    from qutip_trap.dynamics.parallel import MapKind
    from qutip_trap.dynamics.space import HilbertSpace
    from qutip_trap.readout.discriminate import Discriminator

LindbladMethod = Literal["auto", "mesolve", "mcsolve"]
ReadoutMode = Literal["fast", "full"]
Scattering = Literal["estimate", "channels"]

MULTISTEP_INTEGRATORS: frozenset[str] = frozenset({"adams", "bdf", "lsoda", "vode", "zvode"})
"""QuTiP's multistep integrators, never used: their damping distorts the oscillatory spectrum of -iH (Section 5.3)."""
ALLOWED_INTEGRATORS: frozenset[str] = frozenset(
    {"dop853", "vern7", "vern9", "tsit5", "explicit_rk", "krylov", "diag"}
)


@dataclass(frozen=True)
class Numerics:
    """How a run integrates (Sections 5.1 to 5.5, 11.3, 11.5): the tolerances and the escalation ladder, the caps and the
    guards, the trajectory method, the GATE_LOCAL walk and the parallel maps."""

    # the integration of every segment (Section 5.3)
    atol: float = 1e-10
    """Absolute tolerance of the integrator (dimensionless amplitude)."""
    rtol: float = 1e-8
    """Relative tolerance of the integrator."""
    nsteps: int = 10**7
    """The integrator's step budget per segment."""
    integrators: tuple[str, ...] = ("dop853", "vern9")
    """The escalation ladder (``dynamics.evolve``); never a multistep method."""
    propagator_cache: bool = True
    """On an internal-state-only space integrate a segment's propagator once and apply it to every initial state; False
    integrates every state."""
    store_marginals: bool = False
    """Store the Fock populations of every carried mode at every stored time as ``Traces.mode_marginal``."""
    convergence_check: bool = False
    """``Machine.run`` repeats its evolution with atol and rtol tightened by ten and reports the change as
    ``Diagnostics.convergence`` (Section 5.5); off by default because it triples the cost."""
    # the truncation (Sections 5.1, 5.2, 5.5, 11.5)
    joint_dimension_max: int = 4096
    """The joint dimension above which ``FidelityLevel.AUTO`` routes a run to GATE_LOCAL (Section 11.5)."""
    nnz_max: int = 2 * 10**7
    """The drive-operator non-zero count above which AUTO routes a run to GATE_LOCAL."""
    mode_dimension_max: int = 64
    """The ceiling on one resolved mode's Fock dimension the cap rule may ask for; a clamp warns (``TruncationWarning``) and
    is reported, and the oracle and margin checks then cover the clamped range only."""
    boundary_population_max: float = 1e-6
    """The population the cap may leave at its boundary before the monitor raises it (Section 5.5)."""
    freeze_chi_max_rad: float = 0.05
    """|chi_m| (rad) below which a spectator mode may be frozen rather than resolved (Section 5.2)."""
    freeze_alpha_max: float = 1e-4
    """|alpha_m|^2 (2 nbar_m + 1) below which a spectator may be frozen rather than resolved (Section 5.2)."""
    branch_weight_min: float = 1e-6
    """Weight below which a branch of the initial mixture is dropped from the exact evolution (renormalized, reported)."""
    margin_check: bool = True
    """After every pulse compare each driven resolved mode's margin above its populated range with the Section 5.1.1 margin;
    a deficit raises the cap and repeats the run (Section 5.5)."""
    margin_element_tol: float | None = None
    """The interior-element tolerance the Section 5.1.1 margin of a resolved mode is derived from (``required_margin_under``);
    None keeps the fixture margins, and the GATE_LOCAL walk then derives ``map_accuracy * 1e-5`` for its step spaces."""
    caps: Mapping[int, int] | None = None
    """Per mode, a Fock dimension that overrides the cap rule (mode index -> d)."""
    enr_group: tuple[Sequence[int], int] | None = None
    """(modes, N_exc): carry these modes as one excitation-number-restricted factor (Section 11.3 item 1)."""
    space: HilbertSpace | None = None
    """A declared joint space that replaces the selection of Section 5.2 entirely."""
    # the collapse operators (Sections 3.4, 5.3)
    lindblad_method: LindbladMethod = "auto"
    """``mesolve`` (the density matrix), ``mcsolve`` (``ntraj`` quantum-jump trajectories per pure initial state) or
    ``auto``, mesolve up to ``mesolve_dimension_max``."""
    mesolve_dimension_max: int = 128
    """The joint dimension up to which the density matrix is integrated under ``auto``."""
    ntraj: int = 64
    """Trajectories per pure initial state on the mcsolve path (a fixed keyed seed list)."""
    improved_sampling: bool = True
    """``mcsolve``'s no-jump trajectory as a deterministic member of weight p_no-jump, the stochastic ones carrying the rest,
    so the mixture shots are drawn from is weighted (Section 5.3); applied when the run has exactly one trajectory segment."""
    trajectory_target_tol: float | None = None
    """The absolute tolerance ``mcsolve``'s ``target_tol`` targets on the population e_ops in phase one (Section 3.4), which
    fixes the trajectory count phase two replays from a keyed seed list (under a serial map, capped by ``ntraj``, floored by
    ``TARGET_TOL_MIN_TRAJECTORIES``). None keeps ``ntraj``: QuTiP's ``target_tol`` can stop on a zero-variance first batch
    and its firing point depends on scheduling."""
    # the GATE_LOCAL walk (Section 5.4)
    map_accuracy: float = 1e-3
    """epsilon_map of the tomography, a fraction in (0, 1) that keys its tolerances: ceil(1/epsilon_map) trajectories per
    input on the trajectory path."""
    crosstalk_threshold: float = 1e-3
    """A neighbour receiving crosstalk with |epsilon| at or above this joins the gate-local space; below it the light is
    dropped and its rotation sin^2(eps theta/2) added to the reported bound."""
    register_dm_max_qubits: int = 12
    """Carry the register as a density matrix up to this many qubits, as a pure-state ensemble above."""
    register_ensemble: int = 64
    """Members of the pure-state register ensemble."""
    tomography_dropped_weight_max: float | None = None
    """Total weight of the lightest motional branches a step may drop beyond ``branch_weight_min``, reported as the
    diamond-norm bound 2w; None = ``map_accuracy / 4``, 0.0 keeps every branch above the floor."""
    tomography_tolerance_keyed: bool = True
    """Integrate a unitary step with resolved modes at atol = 1e-5 and rtol = 1e-3 of the map accuracy where the caller left
    the defaults, and report the change a ten times tighter dominant branch makes."""
    # the parallel maps (Section 11.3 item 9) and the two per-run counts that shape them
    map: MapKind = "parallel"
    """QuTiP's serial, ``multiprocessing`` (``parallel``) or ``loky`` map; ``serial`` runs in-process."""
    workers: int | None = None
    """Processes for the maps: None = every CPU QuTiP sees, capped by the memory rule; 1 = in-process."""
    samples: int | None = None
    """Dynamical samples per run; None = min(shots, 64) when the noise model has quasi-static content, else 1."""
    addressing: bool | None = None
    """Schedule single-qubit gates in parallel (Section 7.3); None = the device's ``HardwareChain.parallel_addressing``."""

    def __post_init__(self) -> None:
        if self.caps is not None:
            object.__setattr__(self, "caps", {int(m): int(d) for m, d in dict(self.caps).items()})
        if self.enr_group is not None:
            modes, n_exc = self.enr_group
            object.__setattr__(self, "enr_group", (tuple(int(m) for m in modes), int(n_exc)))
        if self.enr_group is not None and self.space is not None:
            raise ValueError("give the ENR group inside the supplied space or as enr_group, not both")
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
        if self.mode_dimension_max < 2:
            raise ValueError("mode_dimension_max is at least two Fock levels per resolved mode")
        if not 0.0 < self.boundary_population_max < 1.0:
            raise ValueError("boundary_population_max is a population fraction in (0, 1)")
        if not 0.0 < self.freeze_alpha_max < 1.0 or not 0.0 < self.branch_weight_min < 1.0:
            raise ValueError("freeze_alpha_max and branch_weight_min are fractions in (0, 1)")
        if self.margin_element_tol is not None and self.margin_element_tol <= 0.0:
            raise ValueError("margin_element_tol is a positive tolerance or None")
        if self.ntraj < 1 or self.mesolve_dimension_max < 1:
            raise ValueError("ntraj and mesolve_dimension_max are positive")
        if not 0.0 < self.map_accuracy < 1.0 or not 0.0 <= self.crosstalk_threshold <= 1.0:
            raise ValueError(
                "map_accuracy is a fraction in (0, 1) and crosstalk_threshold a Rabi ratio in [0, 1]"
            )
        if self.register_dm_max_qubits < 1 or self.register_ensemble < 1:
            raise ValueError("register_dm_max_qubits and register_ensemble are positive")
        if (
            self.tomography_dropped_weight_max is not None
            and not 0.0 <= self.tomography_dropped_weight_max < 1.0
        ):
            raise ValueError("tomography_dropped_weight_max is a weight fraction in [0, 1) or None")
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers is a positive process count or None (every CPU)")
        if self.samples is not None and self.samples < 1:
            raise ValueError("samples is a positive count or None")


@dataclass(frozen=True)
class Physics:
    """Which physical effects a run simulates: by default the device's noise, two register levels per ion, scattering as
    an estimate and the hardware chain."""

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
    """Pass the schedule through the control electronics of Section 7.10 before integrating, the scheduler referencing the
    entangling tones to its response."""
    stark_compensation: bool = True
    """Detune every spin-flip tone by the light shift the table believes; a sigma_z force keeps its beat note (Section 7.5
    item 7)."""
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


@dataclass(frozen=True)
class Readout:
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
