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

Milestone M9b adds the scaling machinery of Section 11.3 items 4, 5 and 9. The drive operators are held factorized (the
matrix-free kernel of ``dynamics/kernels.py``) whenever ``BuilderOptions.kernel`` allows it and the segment is integrated as
kets (``sesolve`` or ``mcsolve``); every ``mesolve`` segment builds the assembled CSR operator, because there the Liouvillian is
formed from the matrix (Section 5.3). The trajectories of a segment run as ONE ``MCSolver.run`` over the ensemble of their
kets (mixed initial conditions, one trajectory per ket, the keyed seed list of Section 3.4) through QuTiP's serial or parallel
map (``SolverOptions.map``, ``SolverOptions.workers``); the per-trajectory results are matched back by seed, so the ensemble is
identical whatever the worker count and the per-trajectory final states are reported for the Section 9.9 test. On an
internal-state-only space (every mode frozen: a carrier pulse of a GATE_LOCAL step, an idle with a Stark shift) the segment
propagator U(t, t_0) is integrated once as a D x D operator and applied to every initial state, cached per engine on the built
Hamiltonian's fingerprint (``SolverOptions.propagator_cache``), never on a joint space (Section 11.3 item 5).

Two exact shortcuts (M8) replace ODE solves where none is needed, both pinned against the ODE path in
``tests/test_engine_numerics.py``: a segment whose Hamiltonian is constant (an idle interval, a zero-envelope pulse) is
propagated by its diagonal phases, or in the frame rotating with H_mot when eigenoperator collapse operators are present
(``closed_form_constant``); a density-matrix input without collapse operators is evolved as the weighted pure branches of its
eigen-decomposition through ``sesolve`` (``pure_branches``, the Fock-sum path of Section 5.3), never as a D^2 Liouvillian.
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
    """The numerical policy of a run (Sections 5.3, 5.5, 11.5): the integrator tolerances and the escalation ladder, the guards
    that route a run to GATE_LOCAL, the truncation caps and monitors, the trajectory method and count, the parallel map,
    and the physics switches a run still reads from here (scattering channels, intensity-noise channels, the hardware
    chain; docs/api_implementation_plan.md moves them to ``Physics`` in 0.3.0). Each field's docstring states its unit
    and its rule."""

    atol: float = 1e-10
    rtol: float = 1e-8
    nsteps: int = 10**7
    integrators: tuple[str, ...] = ("dop853", "vern9")
    """The escalation ladder of Section 5.3; never a multistep method."""
    joint_dimension_max: int = 4096
    nnz_max: int = 2 * 10**7
    """The Section 11.5 guards that route to GATE_LOCAL."""
    mode_dimension_max: int = 64
    """The ceiling on ONE resolved mode's Fock dimension d_m that the cap rule of Sections 5.1.1 and 5.5 may ask for (M9a
    audit D1). 64 is what M9a hard-coded; Section 5.3 states that a Doppler-cooled nbar ~ 20 mode needs d_m >~ 150 for a
    boundary population below 1e-4 and Section 5.2 that long-chain transverse spectators within tens of kHz of the tones
    "cannot be frozen and must be resolved", so the hot-mode regime raises this. When the rule wants more, the cap is
    clamped here AND the clamp is reported in ``SpaceSelection.notes`` (naming the mode, the range the rule asked for and
    the range that survives): the Section 5.1.1 oracle check and the Section 5.5 margin check are then evaluated over a
    narrower range than the physics, which was silent before."""
    boundary_population_max: float = 1e-6
    freeze_chi_max_rad: float = 0.05
    map: Literal["serial", "parallel", "loky"] = "parallel"
    """Coefficients are module-level functions or arrays, so they pickle."""
    e_ops_for_target_tol: bool = True
    """mcsolve needs e_ops to target a tolerance (Section 5.4): the gate on the two-phase trajectory count of Section 3.4."""
    trajectory_target_tol: float | None = None
    """Section 3.4's phase one: the absolute tolerance ``mcsolve``'s ``target_tol`` targets on the population e_ops, which
    fixes the trajectory count that phase two then replays from a keyed seed list of that length. ``None`` keeps the fixed
    ``ntraj``, the default, because QuTiP 5.3.1's ``target_tol`` stops on a zero-variance first batch (measured: 26
    trajectories at tol 0.1 and 604 at 0.02 on a damped two-level system, but 2 at 0.005, where the first two trajectories
    both jumped and the estimated error was exactly 0) and because its firing point is scheduling dependent (Section 3.4
    [corrected]). Set it and the estimate runs under a serial map with ``ntraj`` as its cap and
    ``TARGET_TOL_MIN_TRAJECTORIES`` as its floor."""
    improved_sampling: bool = True
    """Section 5.3: ``mcsolve``'s no-jump trajectory as a deterministic member of weight p_no-jump, the stochastic ones
    carrying the residual weight, so the mixture the shots are drawn from is WEIGHTED (never uniform over the stored
    trajectories, which biases per-shot observables by p_no-jump). Applied when the run has exactly ONE trajectory segment;
    a multi-segment schedule keeps uniform trajectories and says so (see ``IMPROVED_SAMPLING_MULTI_SEGMENT``)."""
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
    convergence_check: bool = False
    """Section 5.5's second bullet: ``run()`` repeats its evolution with atol and rtol tightened by ten and reports the change
    in the register populations as ``Diagnostics.convergence`` (``dynamics.evolve.convergence_check``, M2). Off by default
    because it triples the cost of a run: the comparison runs the base tolerances twice and the tightened ones once. The other
    two arms Section 9.9 asks for - loosening by ten for the integrator ladder, and every resolved cap + 2 - are
    ``hilbert.truncation.convergence_report``, which the ``convergence``-marked tests drive (M9a audit E6)."""
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
    workers: int | None = None
    """Processes for the parallel maps of Section 11.3 item 9 (M9b): the trajectories of a segment in the engine, the
    initial-mixture branches and dynamical samples of ``run()``, the tomography inputs of GATE_LOCAL. None = every CPU QuTiP sees
    (``qutip.settings.available_cpu_count``), 1 = in-process; ``map="serial"`` runs everything in-process whatever this says.
    A task that already runs in a worker runs its own trajectories in-process (never a nested pool)."""
    propagator_cache: bool = True
    """Internal-state-only spaces (every mode frozen; Section 11.3 item 5, M9b): the segment propagator is integrated once as a
    D x D operator through the same ladder and applied to every initial state, cached per engine on the built Hamiltonian's
    fingerprint; False integrates every state through the ODE ladder (the reference)."""
    tomography_isometry: bool = True
    """GATE_LOCAL tomography (Section 5.4; performance pass 2026-09-09): when a step's evolution is unitary (no collapse operator
    on any segment) the channel is read off the propagated INTERNAL BASIS, prod_i d_i kets per motional branch, as the Stinespring
    isometry V_b whose slices by motional output index are the Kraus operators (``dynamics.tomography.choi_from_isometry``), and on an
    internal-state-only space off the segment propagator itself (``JointExactEngine.propagator``: one engine call per branch, no state
    propagated); the prod_i d_i^2 input states the map is stated on follow by linearity. False propagates every input state and fits the
    Choi matrix by least squares (the M9a reference); the two agree to the solver tolerance (``tests/test_tomography.py``). A
    dissipative step takes the reference route whatever this says, because a trajectory is not linear in its initial ket."""
    tomography_dropped_weight_max: float | None = None
    """GATE_LOCAL tomography (Section 5.4; performance pass 2026-09-09): the total weight of motional branches a step may drop,
    lightest first, beyond the per-branch floor ``branch_weight_min``. A channel that is a convex mixture over branches changes by
    at most 2w in diamond norm when weight w is dropped and the rest renormalized, so None derives ``map_accuracy / 4`` (the bound
    2w stays at or below half the map accuracy) and 0.0 keeps every branch above the floor. The bound is reported per step
    (``TomographyRecord.branch_error_bound``, ``GateLocalStep.branch_error_bound``) and summed into
    ``GateLocalReport.discrepancy_bound``; the realized error is far below it (the dropped branches are high Fock states whose
    only effect is a slightly different Debye-Waller factor: 4e-5 measured against a 4e-4 bound on the four-qubit GHZ step)."""
    tomography_tolerance_keyed: bool = True
    """GATE_LOCAL tomography: integrate a unitary step with resolved modes at the tolerance the map accuracy warrants, atol =
    1e-5 map_accuracy and rtol = 1e-3 map_accuracy (1e-8 and 1e-6 at the default), instead of the engine's 1e-10 and 1e-8, when
    the caller left those at their defaults (a tolerance the caller chose is never overridden, the Section 5.3 precedent). The
    change the step's channel makes when the dominant branch is re-integrated ten times tighter is measured and reported
    (``TomographyRecord.tolerance_change``, the Section 5.5 convergence statement, d times the trace norm of the Choi
    difference, a bound on the diamond norm) and summed into ``GateLocalReport.discrepancy_bound``; measured 1.3e-4 for 1.8
    times fewer right-hand sides on the example device's entangling step, where the next loosening exceeds the map accuracy.
    False keeps the engine tolerances everywhere. Internal-state-only steps (propagators at dimension 4 to 16) and dissipative
    steps are never loosened."""
    margin_element_tol: float | None = None
    """The interior-element tolerance the Section 5.1.1 margin of a resolved mode is derived from
    (``hilbert.operators.required_margin``): None keeps the fixture (6 levels at |eta| <= 0.1, 10 at 0.5, 20 at 1, the
    margins at which the exponential's elements reach 10^-12), which every JOINT_EXACT run and every Section 9 validation case
    uses. The GATE_LOCAL walk derives ``map_accuracy * 1e-5`` for its step spaces when the caller leaves None (1e-8 at the
    default): the cap keeps the smallest margin at which the exponential's interior elements over the populated range are exact
    to it and one displacement from the top populated level leaks less than a tenth of ``boundary_population_max`` past the cap,
    never more than the fixture; the same rule is what the engine's margin check (Section 5.5) reads on those runs, the rule (ii)
    oracle asserts the declared tolerance at construction, and the step reports the measured element error
    (``GateLocalStep.element_error``). Two to three levels per resolved mode on the four-qubit GHZ circuit's entangling steps."""
    rotating_frame: bool = True
    """Integrate every ket segment (``sesolve`` and ``mcsolve``) in the exact rotating frame of its diagonal H_0 = H_mot + H_int
    (``dynamics.rotating``; Sections 5.2, 5.3): psi = e^{-i H_0 t} phi with the phases applied to the STATE and undone on the
    way out, so the same drive operators, tolerances and integrator ladder integrate a state that moves at the drive and
    detuning frequencies instead of the Fock energies. Nothing is expanded or dropped; the results equal the Schroedinger-picture
    integration to the solver tolerance (1e-7 in norm at atol 1e-10, rtol 1e-8, the Section 11.1 meaning of 'identical'), with
    3 to 8 times fewer right-hand-side evaluations on the Section 11.1 rows (performance pass 2026-09-09). False integrates in
    the Schroedinger picture, the M2 to M9b reference and the picture the Section 5.3 step-density band (27 to 52 steps per
    period of the highest mode) was measured in. A segment the frame does not cover (a non-diagonal static part such as an
    anharmonic term, a function-element term, a collapse operator that is not an eigenoperator of H_0 such as a recoil kick or
    the intensity-noise channel) integrates in the Schroedinger picture and says so in ``SegmentReport.frame``."""

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
    kernel: str = "none"
    """How the segment's drive operators were held (Section 11.3 item 4; M9b): ``factorized``, ``assembled``, ``mixed`` or
    ``none`` (no drive term)."""
    frame: str = "schrodinger"
    """The picture the segment was integrated in: ``rotating`` (the exact rotating frame of ``dynamics.rotating`` under
    ``SolverOptions.rotating_frame``) or ``schrodinger`` (the closed forms, the propagator path, every ``mesolve`` segment, and
    the ket segments the frame does not cover). ``rhs_evaluations`` and ``steps_per_period`` count the evaluations of the picture
    named here, so the Section 5.3 band applies to ``schrodinger`` rows only."""


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
    kernel: str = "none"
    """``factorized`` when any segment held its drive operators factorized (Section 11.3 item 4; M9b), ``assembled`` when every
    segment with drive terms assembled them, ``none`` without drive terms."""
    map: str = "serial"
    """The map the trajectories ran through (Section 11.3 item 9): ``serial``, ``parallel`` or ``loky``."""
    workers: int = 1
    """Processes the trajectory map used (1 in-process)."""
    propagator_solves: int = 0
    """Segment propagators integrated on internal-state-only spaces (Section 11.3 item 5)."""
    propagator_cache_hits: int = 0
    """Segments served from the engine's propagator cache."""
    trajectory_finals: tuple[Qobj, ...] = ()
    """On the trajectory path, the final ket of every trajectory after the last segment, in the keyed order (the per-trajectory
    identity of Section 9.9)."""
    trajectory_seeds: tuple[tuple[int, ...], ...] = ()
    """The spawn keys of the trajectories' seeds on the last trajectory segment, in the same order (every segment keys its own
    seeds by (sample, trajectory, 0, 0, "mcsolve[segment]"))."""

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
    progress: Callable[[Progress], None] | None = field(default=None, repr=False, compare=False)
    """Called after every integrated pulse segment of ``run_pulses`` with ``Progress("pulse", done, total, elapsed_s)``
    (0.2.0; docs/api_implementation_plan.md 1.8); ``run`` sets it and rebases the clock to the run's start. Dropped when the
    engine is shipped to a worker, so a parallel map reports no pulses."""
    _propagators: dict[tuple[object, ...], _Propagator] = field(
        default_factory=dict, repr=False, compare=False
    )
    """The propagator cache of Section 11.3 item 5 (M9b), keyed by the built Hamiltonian's fingerprint and the stored times."""

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
        """State-based process tomography of one pulse, a gate's pulse group or a whole Schedule on ``space`` (Section 5.4; M9a).

        The channel is extracted from the motional state of ``motional_model`` (the tracked reduced density matrices of the
        resolved modes, their thermal states where none is tracked, the frozen modes' Fock populations as weighted branches):
        when the evolution is unitary and ``SolverOptions.tomography_isometry`` holds, from the prod_i d_i internal basis kets
        propagated through ``run_pulses`` per branch (the Stinespring isometry; off the segment propagator itself on an
        internal-state-only space), otherwise from every one of the prod_i d_i^2 linearly independent pure inputs of
        ``dynamics.tomography.input_states`` with the Choi matrix reconstructed by least squares. The Choi matrix is projected
        onto CP and TP by Dykstra's alternating projection and the summary of Section 6.8 is computed against ``ideal`` (the
        gate's ideal unitary on the space's ions in factor order; NaN infidelities without one). The full record, including the
        motional outputs the GATE_LOCAL model tracks and the route taken, is :meth:`tomography`.
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
                if new_space.dimension > options.joint_dimension_max:
                    # Section 11.5's ceiling holds for a GROWN space too: a cap that keeps growing (a hot mode, or an ENR
                    # group of Doppler-limited modes whose top shell never empties) must not build past the guard the
                    # declaration was checked against (2026-09-08: an ENR group at N_exc = 2 grew to 6, dimension 16016)
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
        the identity through the Section 5.3 ladder (Section 11.3 item 5; M9b). Returns (propagator, cache hit)."""
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
        """The schedule the ions see: the played chain of Section 7.3 (M8: requested -> physical through the device's derived
        values, when a table is given) and then the hardware chain of Section 7.10 (M7). Returns (schedule, hardware notes); the
        played chain's notes are appended to ``notes``."""
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
        """Frozen spectators: this evolution's Fock states (Section 5.2), from the sample where the caller put them there.

        Section 5.2's "samples n_m once per shot" is realized by run(), which enumerates the frozen modes' Fock states as
        weighted branches (an exact quadrature over the same thermal distribution, better than sampling) and passes each
        branch's tuple in sample.values. The draw below is the fallback for a direct run_pulses caller, which has no shot
        index at all - one call is one evolution - so it is keyed PER SAMPLE and reported as such (M9b audit B11)."""
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
        """The state-independent collapse operators of a run on ``space`` (the explicit ``channels`` plus, under
        ``device_channels``, the device's own) and whether a segment with active pulses CAN carry collapse operators built from
        its pulses (scattering, intensity noise): decided before any build so that the drive operators are assembled exactly
        where a Liouvillian is formed (Section 5.3) and held factorized everywhere else (Section 11.3 item 4; M9b)."""
        static_ops: list[CollapseOp] = list(self.channels)
        if self.device_channels:
            static_ops.extend(device.noise.channels(device, space))
        pulse_channels_possible = bool(self.device_channels) and (
            bool(options.scattering_channels)
            or (bool(options.intensity_noise_channels) and device.noise.intensity_white_density() > 0.0)
        )
        return static_ops, pulse_channels_possible

    def is_unitary(self, device: Device, space: HilbertSpace, options: SolverOptions) -> bool:
        """Whether every segment of a run on ``space`` evolves unitarily: no explicit channel, no device channel on the space and
        no pulse-built channel possible, so the final state is LINEAR in the initial ket. This is the condition under which the
        GATE_LOCAL tomography may propagate the internal basis alone (``SolverOptions.tomography_isometry``); a trajectory of
        ``mcsolve`` is not linear in its initial ket and a ``mesolve`` segment evolves a density matrix."""
        static_ops, pulse_channels_possible = self._collapse_setup(device, space, options)
        return not static_ops and not pulse_channels_possible

    @staticmethod
    def _segment_edges(sched: Schedule) -> list[float]:
        """Segments at every pulse boundary and idle boundary; the state is given at the schedule's declared start
        (``Schedule.t0_s``, a GATE_LOCAL step's start) or at min(0, the first cut) as M2 to M8 did, and the last edge is
        ``pulses_end_s`` (the measurement event that may follow is the readout stage's, not free evolution)."""
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
        """U(t_end, t_0) of ``schedule`` on the internal-state-only ``space`` (every mode frozen, no ENR group) as a D x D
        matrix, with the report of a run (Section 11.3 item 5; performance pass 2026-09-09).

        The product over the segments of the segment propagators: the exact closed form of a constant segment (the diagonal
        phases of an idle, one eigendecomposition otherwise; ``closed_form_constant``) or the propagator the cache holds or
        integrates once through the Section 5.3 ladder (``_segment_propagator``, the same key ``run_pulses`` uses). On such a
        space the propagator IS the step's channel, so the GATE_LOCAL tomography reads the branch's Kraus operator off it
        instead of propagating states (``dynamics.tomography``, ``SolverOptions.tomography_isometry``). ``motional_model``
        supplies the occupations for the frozen modes whose Fock state the sample does not carry (``run_pulses`` reads them
        from its state). Raises ``ValueError`` on a space with a resolved mode or an ENR group, and when a segment would carry
        a collapse operator (a propagator of a dissipative segment is not a unitary: use ``run_pulses``).
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
        notes: list[str] = []
        # the played chain of Section 7.3 (M8) and the hardware chain of Section 7.10 (M7)
        sched, hw_notes = self._played_schedule(device, schedule, sample, seeds, options, notes)
        # the frozen spectators' Fock states for this evolution (Section 5.2; M9b audit B11)
        frozen_n = self._frozen_fock_states(space, sample, seeds, state.motional.nbar, notes)
        # the state-independent collapse operators and whether a pulse segment can carry pulse-built ones (Section 5.3)
        static_ops, pulse_channels_possible = self._collapse_setup(device, space, options)
        # the Lindblad method the dissipative segments will take
        lindblad_resolved: str = options.lindblad_method
        if lindblad_resolved == "auto":
            lindblad_resolved = "mesolve" if space.dimension <= options.mesolve_dimension_max else "mcsolve"
        map_kind = options.map
        n_workers = worker_count(options)
        workers_used = 1
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
        # Section 5.3: improved_sampling decomposes the WHOLE evolution into its no-jump member and the rest, so it is used
        # only when the trajectory path is entered exactly once (IMPROVED_SAMPLING_MULTI_SEGMENT says why)
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
            # the exact rotating frame of Section 5.2 for the ket paths (sesolve, mcsolve): the state carries e^{-i H_0 t}
            # and the drive terms are applied between the phases (dynamics/rotating.py); mesolve segments form the
            # Liouvillian from the Schroedinger-picture matrices (Section 5.3) and the closed forms need no frame
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
            # per e_op, the e^{i lambda t} an expectation value picks up on the way back from the rotating frame (0 for the
            # populations, -omega_m for a_m); None means the trace is taken on the rotated-back states
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
                if (
                    kets is not None
                    and options.propagator_cache
                    and not space.resolved
                    and space.enr_group is None
                ):
                    # an internal-state-only space (Section 11.3 item 5): one propagator serves every initial state
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
                    improved_seg = improved_run and len(kets) == 1
                    n_traj_seg = options.ntraj
                    # the map and the worker count are fixed once n_traj_seg is final (phase one may lower it); the
                    # placeholders here only make mc_opts constructible for the phase-one probe
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
                    # Section 3.4 phase one: the trajectory count from mcsolve's target_tol on the population e_ops, run
                    # under a SERIAL map so that its scheduling-dependent firing point is reproducible, capped by ntraj and
                    # floored by TARGET_TOL_MIN_TRAJECTORIES; phase two below replays a keyed seed list of that length
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
                    seg_map = map_kind if n_stoch > 1 else "serial"
                    seg_workers = min(n_workers, n_stoch) if seg_map != "serial" else 1
                    mc_opts["map"], mc_opts["num_cpus"] = seg_map, seg_workers
                    workers_used = max(workers_used, seg_workers)
                    solver = qt.MCSolver(
                        built.H if rot is None else rot.H,
                        c_ops if rot is None else list(rot.c_ops),
                        options=mc_opts,
                    )
                    kets_in = kets if rot is None else [rot.frame.to_frame(k, float(times[0])) for k in kets]
                    # one trajectory per ket of the ensemble, each with its keyed seed (Section 3.4), through QuTiP's map:
                    # mixed initial conditions with an explicit per-state trajectory count; the results come back in completion
                    # order and are matched to their kets by seed (Section 11.3 item 9; M9b)
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
                        # the WEIGHTED mixture of Section 5.3: the deterministic no-jump member carries p_no-jump
                        # (QuTiP 5.3.1 calls the plan's deterministic_weight_info `deterministic_weights`), the stochastic
                        # trajectories the residual weight `runs_weights`; drawing uniformly from the stored trajectories
                        # would bias every per-shot observable by p_no-jump
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
                )
            )
            if active and self.progress is not None:
                from qutip_trap.run.results import Progress

                pulse_done += 1
                self.progress(
                    Progress("pulse", pulse_done, pulse_total, time.perf_counter() - progress_started)
                )
            if active:
                # Section 5.5's ONE threshold, "1e-6 of the population the pulse moves": a branch of weight w (run() and the
                # tomography evolve the initial mixture's Fock branches one by one, each normalized) carries at most w of that
                # population, so the branch-relative quantities bpop and the populated range are both compared against
                # boundary_population_max / w. The trip used to read the unscaled threshold while the margin check read the
                # scaled one, so the two differed by 1/w on a low-weight branch (M9b audit B3); a single state has w = 1 and
                # neither number moves
                branch_w = min(max(sample.get(KEY_BRANCH_WEIGHT, 1.0), 1e-300), 1.0)
                tail = min(options.boundary_population_max / branch_w, 0.5)
                for m, v in bpop.items():
                    if v > tail and space.mode_class(m) in ("resolved", "enr"):
                        raise _BoundaryTrip(m, v)
                if options.margin_check and space.resolved:
                    # Section 5.5: the cap's margin above the populated range must stay above the Section 5.1.1 margin for
                    # the segment's eta on every resolved mode the segment drives; a deficit raises the cap by it and repeats
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
            # the trajectories' final kets after EVERY segment, in the keyed order (Section 9.9's per-trajectory identity)
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
            map=map_kind if method_used == "mcsolve" else "serial",
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
        )


EIGH_DIMENSION_MAX = 4096
"""Above this joint dimension a constant but non-diagonal Hamiltonian goes through the ODE ladder rather than a dense eigh."""

TARGET_TOL_MIN_TRAJECTORIES = 8
"""The floor on Section 3.4's phase-one estimate: QuTiP 5.3.1's ``target_tol`` accepts a zero-variance first batch and can
stop at 2 trajectories (measured, see ``SolverOptions.trajectory_target_tol``), so the estimate is never taken below this."""

IMPROVED_SAMPLING_MULTI_SEGMENT = (
    "mcsolve improved_sampling not used: the no-jump/jump split is a decomposition of the WHOLE evolution, and this schedule "
    "has {n} trajectory segments; applied per segment it would turn the ensemble into a jump expansion of 2^{n} pure members "
    "that cannot be merged (pure states) and cannot be pruned without dropping exactly the jump weight the channels are there "
    "to produce, so the segments run uniform-weight trajectories (Section 5.3; conv.improved_sampling_single_segment)"
)

PROPAGATOR_CACHE_MAX = 256
"""Segment propagators an engine keeps (Section 11.3 item 5); the cache is cleared when full."""


@dataclass(frozen=True)
class _Propagator:
    """U(t_k, t_0) at the stored times of one segment on an internal-state-only space, with how it was integrated."""

    unitaries: tuple[np.ndarray, ...]
    integrator: str
    atol: float
    rhs_evaluations: int | None
    retries: tuple[str, ...]


MARGIN_LEAKAGE_FRACTION = 0.1
"""The derived margin (``SolverOptions.margin_element_tol``) keeps one displacement's leakage from the top populated level below
this fraction of ``boundary_population_max``, so that the boundary monitor's own trip is not the first thing a derived cap meets."""


def required_margin_under(eta: float, options: SolverOptions, n_hi: int) -> int:
    """The Section 5.1.1 margin a run under ``options`` keeps above the top populated level ``n_hi`` at |eta|: the fixture
    (``hilbert.operators.required_margin``) unless ``margin_element_tol`` is declared, in which case the margin is derived from
    the declared element tolerance and a tenth of the boundary threshold. The cap rule of ``run.gate_local.step_space`` and the
    engine's margin check read this one function, so a first attempt does not trip."""
    if options.margin_element_tol is None:
        return required_margin(eta)
    return required_margin(
        eta,
        tail=float(options.boundary_population_max) * MARGIN_LEAKAGE_FRACTION,
        n_hi=int(n_hi),
        element_tol=float(options.margin_element_tol),
    )


def _steps_per_period(rhs_evals: int | None, omega_max_rad_s: float, duration_s: float) -> float | None:
    """Integrator steps per period of the fastest mode (Section 5.3's step-density band): rhs_evals / 12 dop853 stages over the
    periods of ``omega_max_rad_s`` in ``duration_s``; None without a count or a frequency."""
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


def _expectation_back(
    values: np.ndarray,
    lam: float | None,
    times: np.ndarray,
    op: qt.Qobj,
    states: Sequence[qt.Qobj],
) -> np.ndarray:
    """An expectation trace taken in the rotating frame, back in the Schroedinger picture: <psi|O|psi> = e^{i lambda t} <phi|O|phi>
    for an eigenoperator of ad_{H_0} (``lam``; 0 leaves the populations untouched), the trace over the rotated-back ``states`` for
    anything else (``lam`` None)."""
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


__all__ += [
    "MARGIN_LEAKAGE_FRACTION",
    "EngineReport",
    "JointExactEngine",
    "SegmentReport",
    "TruncationLimit",
    "required_margin_under",
]
