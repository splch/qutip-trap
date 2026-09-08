"""The ONE Hamiltonian builder (PLAN.md Sections 3.1, 4.3.1, 4.3.6, 5.2, 5.7; milestone M2; multi-level mode M3a).

H(t)/hbar = H_mot + H_int + sum_i H_drive,i(t) + sum_i H_Stark,i(t) + H_anh (+ H_curv), in rad/s on the joint space of
Section 5.1, every term switchable and every switch recorded in ``BuiltHamiltonian.approximations`` (Section 5.7):

- H_mot = sum_m omega_m a_m^dag a_m over the RESOLVED (and ENR) modes, omega_m from Crystal.modes plus the sample's
  mode offsets; frozen spectators have no operator and enter through their Debye-Waller factor.
- H_int = sum_i (Delta_i/2) sigma_z^i with sigma_z the ENERGY operator |1><1| - |0><0| and Delta_i the true transition
  frequency minus the frame frequency (the sample's qubit offset plus explicit shifts such as an ac Zeeman shift).
- H_drive,i(t) = (1/2) sum_tones Omega(t) e^{-i(mu t - phi(t))} sigma_+^i (x) prod_m D_m(i eta_im) + h.c. (Section 5.2)
  with the EXACT displacement operators (expm, Section 5.1.1) unless ``lamb_dicke_order`` truncates them; the same term
  on a neighbour j with Omega -> eps_ij Omega, j's own etas and the geometric phase Delta k . (X_j - X_i) is crosstalk
  (Section 6.6); the coefficient is multiplied by J_0(beta) (unlocked drives, the rf-phase average of Section 4.3.6)
  or by e^{i beta cos(Omega_rf t + delta)} (rf-locked, ``micromotion="modulated"``), by the frozen spectators'
  e^{-eta^2/2} L_n(eta^2) for the shot's Fock states n, and by the sample's Rabi scale. C0 is already inside every eta
  that Crystal.lamb_dicke delivers and is never reapplied here (Section 4.1.1).
- A ``light_shift`` drive (Section 4.4.4, M4) replaces sigma_+^i by the level-weighted projector w_dn P_0 + w_up P_1
  with (w_dn, w_up) = (Omega_dndn, Omega_upup)/Omega_LS, so the term is Omega_LS cos(mu t - phi + Delta k . x)[sigma_z +
  (w_up + w_dn)/2] (x) D: Zhu-Monroe-Duan's spin-dependent force plus the spin-independent one; the same beams' ordinary
  Raman coupling Omega_R = w_flip Omega_LS is kept as sigma_+ (x) D rotating at mu - omega_0 (the beat note sits near a
  mode, far from the qubit frequency) unless its off-resonant excitation (w_flip Omega_LS/(omega_0 - mu))^2 is below
  1e-12, in which case it is dropped and the drop recorded.
- H_Stark,i(t) = (delta_St,i(t)/2) sigma_z^i, proportional to the instantaneous intensity (Section 4.3.2).
- H_anh = sum over sorted mode tuples of coefficient x X_k X_l X_m, opt-in (Section 4.1.4).
- Beam-curvature coupling (Cetina 2022, Section 6.2): Omega -> Omega (1 + (Omega''/2 Omega) x_hat^2) with x_hat the
  ion's position operator along the curvature axis, applied symmetrically with the displacement.

Time conventions: the coefficient's beat-note phase is mu t in ABSOLUTE time for phase-continuous operation at the
nominal qubit frequency (Section 7.10, the mode the virtual-Z rule was pinned in) and mu (t - t_start) under
``phase_mode="reset"``; envelopes, phase schedules and detuning schedules are callables of the time since the pulse
START; a constant is a square pulse; an array is uniformly sampled over the pulse and cubic-spline interpolated
(Section 5.5). Coefficients are Python functions with the QuTiP 5.3 signature f(t, **kwargs) (never strings, Section 5.2);
the plain and conjugate terms of a drive, and every sideband term of the interaction picture, share one tone-sum evaluation
per time through a one-entry memo (the integrator evaluates every element at the same t), square tones are evaluated from
their three constants, and a drive whose tones are identically zero contributes no operator term (its Stark shift stays).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import qutip as qt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import eval_genlaguerre, jv

from qutip_trap.control.pulses import Drive, Pulse, as_time_function, fingerprint_pulse
from qutip_trap.device.model import Device
from qutip_trap.dynamics.frames import interaction_picture
from qutip_trap.dynamics.kernels import KernelChoice, prefer_factorized
from qutip_trap.dynamics.multilevel import (  # the multi-level mode of Section 4.2.8 (M3a): one builder module
    ModeSpec,
    MultiLevelBuild,
    MultiLevelOptions,
    assign_frames,
    build_multilevel,
)
from qutip_trap.hashing import canonical_digest
from qutip_trap.hilbert.operators import debye_waller_factor, qudit_projector, qudit_sigma_plus
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.light.comb import comb_build_notes
from qutip_trap.light.raman import lamb_dicke_parameters, mathieu_or_none
from qutip_trap.noise.processes import Trajectory
from qutip_trap.noise.sampling import (
    KEY_BRANCH_WEIGHT,
    KEY_INTENSITY_TRAJECTORY,
    KEY_LASER_OFFSET_HZ,
    KEY_LASER_PHASE_TRAJECTORY,
    KEY_RABI_SCALE,
    KEY_RF_FRACTION_TRAJECTORY,
    KEY_RF_PHASE,
    NoiseSample,
    key_beam_offset_m,
    key_beam_phase_rad,
    key_beam_phase_trajectory_rad,
    key_frozen_n,
    key_mode_offset_hz,
    key_position_offset_m,
    key_qubit_offset_hz,
    key_qubit_trajectory_hz,
    quiet_sample,
)
from qutip_trap.trap.anharmonic import anharmonic_estimate
from qutip_trap.units import TWO_PI

M2 = "milestone M2 (dynamics/hamiltonian.py, PLAN.md Section 4.3.1)"
M3A = "milestone M3a (the multi-level mode of Section 4.2.8, dynamics/multilevel.py)"

Frame = Literal["schrodinger", "interaction"]
MicromotionMode = Literal["none", "carrier_j0", "modulated"]
PhaseMode = Literal["continuous", "reset"]
GradientForm = Literal["bare", "dressed"]


@dataclass(frozen=True)
class CurvatureSpec:
    """Omega''/Omega of the addressing beam at the ion (1/m^2, negative for a Gaussian centre) and the curvature axis."""

    kappa_per_m2: float
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class BuilderOptions:
    frame: Frame = "schrodinger"
    lamb_dicke_order: int | None = None
    """None = exact D(i eta); 0, 1, 2, ... = the Taylor expansion of every displacement to that order (an approximation)."""
    rwa: bool = False
    """Interaction frame only: keep, per tone, the single sideband combination nearest its resonance (the textbook model)."""
    k_max: int | None = None
    """Interaction frame: keep |k_m| <= k_max per mode and report the dropped weight (Section 5.2)."""
    micromotion: MicromotionMode = "carrier_j0"
    phase_mode: PhaseMode = "continuous"
    include_stark: bool = True
    include_anharmonic: bool = False
    include_crosstalk: bool = True
    frozen_debye_waller: bool = True
    curvature: Mapping[int, CurvatureSpec] = field(default_factory=dict)
    """Per ion: the Cetina beam-curvature coupling (Section 6.2); empty = off."""
    gradient_form: GradientForm = "bare"
    """A ``gradient`` drive (Section 4.4.5, M4): ``bare`` builds the two microwave tones as ordinary carrier terms beside the
    laboratory sigma_z force w_{i,m} cos(omega_g t + phi_g), so the J_2(4 Omega_mu/delta) weight and the J_0 = 0 intrinsic
    dynamical decoupling emerge from the exact dynamics (the plan's "everything derived" preference); ``dressed`` builds the
    adiabatically eliminated effective force (w_{i,m}/2) J_2(4 Omega_mu/delta) sigma_z^i (a_m e^{i(...)} + h.c.) alone and
    records the elimination in ``approximations``."""
    kernel: KernelChoice = "auto"
    """How the drive operators sigma_+^i (x) prod_m D_m are held and applied (Section 11.3 item 4; M9b): ``assembled`` builds
    the CSR matrix (M2 to M9a), ``factorized`` holds the per-mode factors and applies them mode by mode as a matrix-free
    right-hand side, ``auto`` chooses by the Section 11.2 cost model per space (``dynamics.kernels.prefer_factorized``). Neither
    is an approximation: the two are the same operator to round-off. The factorized form needs a product space in the Schroedinger
    frame with exact displacements (no ``lamb_dicke_order``, no curvature factor, no coupling to an ENR group); the engine forces
    ``assembled`` on every ``mesolve`` path, where the Liouvillian is built from the matrix (Section 5.3)."""

    def __post_init__(self) -> None:
        if self.lamb_dicke_order is not None and self.lamb_dicke_order < 0:
            raise ValueError("lamb_dicke_order is None (exact) or a non-negative expansion order")
        if self.rwa and self.frame != "interaction":
            raise ValueError(
                "the sideband rotating-wave approximation is defined in the interaction frame only"
            )
        if self.k_max is not None and self.k_max < 0:
            raise ValueError("k_max is a non-negative sideband order")


@dataclass(frozen=True)
class DriveRecord:
    """What the builder derived for one (pulse, ion) pair, for provenance and the Section 5.7 diagnostics."""

    pulse: str | None
    ion: int
    primary_ion: int
    etas: dict[int, float]
    frozen_n: dict[int, int]
    debye_waller: float
    micromotion_beta: float
    carrier_factor: float
    crosstalk: complex
    rabi_scale: float
    omega_peak_rad_s: float


class _Counter:
    """Right-hand-side evaluation counter shared by a build's coefficients (Section 5.3 step-density budget)."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls = 0

    def count(self) -> int:
        return self.calls


@dataclass
class _ToneFn:
    envelope: Callable[[float], float]
    """Omega(tau) in rad/s, tau the time since the pulse start."""
    phase: Callable[[float], float]
    """phi(tau) in rad."""
    beat_phase: Callable[[float, float], float]
    """Theta(t, tau) = the accumulated beat-note phase mu t (continuous) or the pulse-local integral (reset/FM)."""
    constants: tuple[float, float, float] | None = None
    """(Omega in rad/s, phi in rad, mu in rad/s) of a square tone with constant detuning: the same three numbers the callables
    return, evaluated without the calls (the coefficient is called 10^4 to 10^6 times per pulse, Section 11.2)."""
    continuous: bool = True
    """The beat phase is mu t (phase_mode continuous) rather than mu tau (reset)."""


@dataclass(eq=False)
class _DriveCoefficient:
    """The scalar c(t) multiplying sigma_+^i (x) D_i: (1/2) sum_tones Omega(tau) e^{-i(Theta - phi)} times the scalar factors.

    Compared by identity (``eq=False``): ``QobjEvo`` merges elements whose coefficients compare equal, and two drives are
    two terms of H whatever their parameters happen to be.
    """

    t_start: float
    tones: list[_ToneFn]
    scale: complex
    """rabi_scale x crosstalk x Debye-Waller x J_0 (the static factors)."""
    modulation_beta: float
    modulation_omega: float
    modulation_delta: float
    k_dot_omega: float
    counter: _Counter
    phase_trajectory: Callable[[float], float] | None = None
    """phi_L(t) of the laser (rad), added to every tone's phase (single-photon optical drives, Section 6.3)."""
    amplitude_trajectory: Callable[[float], float] | None = None
    """dI/I(t) of the light; the envelope is multiplied by (1 + dI/I)^amplitude_power (Section 6.4)."""
    amplitude_power: float = 1.0
    _t_last: float = field(default=math.nan, init=False, repr=False)
    _c_last: complex = field(default=0j, init=False, repr=False)
    """The last (t, c(t)): the integrator evaluates every element of the QobjEvo at the same t, so the plain term, its
    conjugate and, in the interaction picture, every sideband term of the same drive share one tone-sum evaluation."""

    def __call__(self, t: float) -> complex:
        self.counter.calls += 1
        if t == self._t_last:
            return self._c_last
        tau = t - self.t_start
        extra_phase = 0.0 if self.phase_trajectory is None else float(self.phase_trajectory(t))
        total = 0j
        for tone in self.tones:
            k = tone.constants
            if k is not None:
                omega, phi, mu = k
                theta = mu * t if tone.continuous else mu * tau
                total += 0.5 * omega * cmath.exp(-1j * (theta - phi - extra_phase))
            else:
                total += (
                    0.5
                    * tone.envelope(tau)
                    * cmath.exp(-1j * (tone.beat_phase(t, tau) - tone.phase(tau) - extra_phase))
                )
        c = total * self.scale
        if self.amplitude_trajectory is not None:
            c = c * (1.0 + float(self.amplitude_trajectory(t))) ** self.amplitude_power
        if self.modulation_beta != 0.0:
            c = c * cmath.exp(
                1j * self.modulation_beta * math.cos(self.modulation_omega * t + self.modulation_delta)
            )
        if self.k_dot_omega != 0.0:
            c = c * cmath.exp(1j * self.k_dot_omega * t)
        self._t_last = t
        self._c_last = complex(c)
        return self._c_last


@dataclass(eq=False)
class _RotatedCoefficient:
    """c(t) e^{i k . omega t}: one sideband term of the interaction picture, sharing the drive's tone sum ``base``."""

    base: _DriveCoefficient
    rotation: float

    def __call__(self, t: float) -> complex:
        c = self.base(t)
        return c if self.rotation == 0.0 else c * cmath.exp(1j * self.rotation * t)


@dataclass(eq=False)
class _ScalarCoefficient:
    fn: Callable[[float], float]
    t_start: float
    counter: _Counter

    def __call__(self, t: float) -> float:
        self.counter.calls += 1
        return float(self.fn(t - self.t_start))


@dataclass(eq=False)
class _GradientCoefficient:
    """f(t) on sigma_z^i (x) a_m of a near-field microwave-gradient drive (Section 4.4.5; M4).

    ``dressed=False`` is the LABORATORY force w_{i,m} cos(omega_g t + phi_g), both rotating components present, which is
    what the bare two-tone build uses: the J_2(4 Omega_mu/delta) weight then emerges from the exact dynamics of the
    microwave dressing that the drive's own tones generate. ``dressed=True`` is the adiabatically eliminated
    co-rotating term (w_{i,m}/2) J_2(4 Omega_mu/delta) e^{i(rotation t + phi)}, an approximation recorded by the builder.
    """

    weight: float
    omega_g: float
    phase: float
    t_start: float
    counter: _Counter
    rotation: float = 0.0
    dressed: bool = False
    continuous: bool = True

    def __call__(self, t: float) -> complex:
        self.counter.calls += 1
        tt = t if self.continuous else t - self.t_start
        if self.dressed:
            return self.weight * cmath.exp(1j * (self.rotation * tt + self.phase))
        return complex(self.weight * math.cos(self.omega_g * tt + self.phase))


def _coef_plain(
    t: float,
    coef: _DriveCoefficient | _RotatedCoefficient | _ScalarCoefficient | _GradientCoefficient,
    **_: object,
) -> complex:
    return complex(coef(t))


def _traj_coef(t: float, traj: Callable[[float], float], scale: float, **_: object) -> float:
    """A sampled trajectory (Section 6.1 route d) as a coefficient: scale x traj(t), the fixed-grid interpolation of M7."""
    return float(scale * traj(t))


def _coef_conj(
    t: float, coef: _DriveCoefficient | _RotatedCoefficient | _GradientCoefficient, **_: object
) -> complex:
    return complex(coef(t)).conjugate()


@dataclass(frozen=True)
class BuiltHamiltonian:
    """The QobjEvo and everything the run's diagnostics need to know about how it was built (Section 5.7)."""

    H: qt.QobjEvo
    space: HilbertSpace
    t_start_s: float
    t_end_s: float
    frame: Frame
    omega_max_rad_s: float
    """The highest MODE (or beat-note) frequency in the frame; the step-density budget of Section 5.3 is quoted in steps
    per period of the highest mode (14 to 45), the |n> component of the state rotating n times faster."""
    n_drive_terms: int
    approximations: tuple[str, ...]
    records: tuple[DriveRecord, ...]
    dropped_weight: float
    mode_frequencies_rad_s: dict[int, float]
    counter: _Counter
    drive_parts: dict[str, qt.QobjEvo] = field(default_factory=dict)
    """Per pulse (gate_id), the QobjEvo of that pulse's drive terms alone: what the white intensity-noise channel
    sqrt(D) H_drive(t) of Section 6.4 multiplies (M7)."""
    kernel: Literal["assembled", "factorized", "mixed", "none"] = "none"
    """How the drive operators are held (Section 11.3 item 4; M9b): ``factorized`` (the matrix-free kernel on every drive term),
    ``assembled`` (CSR), ``mixed`` (a segment whose terms differ, e.g. an ENR-coupled drive beside a product-space one), ``none``
    (no drive term: an idle or a silent pulse)."""
    fingerprint: str = ""
    """A value digest of everything that determined H(t) (space, frequencies, the pulses' sampled tones, the sample's offsets
    and trajectories, the shifts, the options): the key of the engine's propagator cache (Section 11.3 item 5; M9b)."""

    @property
    def rhs_evaluations(self) -> int:
        """Right-hand-side evaluations so far: coefficient calls over the number of coefficient-bearing elements."""
        n = max(self.n_drive_terms, 1)
        return self.counter.calls // n


# ---- helpers -------------------------------------------------------------------------------------------------------------


def _as_time_function(
    value: Callable[[float], float] | np.ndarray | float, duration_s: float, *, scale: float
) -> Callable[[float], float]:
    """A pulse-local callable of tau from a constant, a callable or a uniformly sampled array over [0, duration] (the shared
    picklable conversion of ``control.pulses``; Section 11.3 item 9)."""
    return as_time_function(value, duration_s, scale=scale)


@dataclass(frozen=True)
class _LinearBeat:
    """Theta(t, tau) = mu t (phase_mode continuous) or mu tau (reset) for a constant detuning."""

    mu: float
    continuous: bool

    def __call__(self, t: float, tau: float) -> float:
        return self.mu * t if self.continuous else self.mu * tau


class _IntegratedBeat:
    """Theta(t, tau) = int_0^tau mu(tau') dtau' (+ mu(0) t_start under phase_mode continuous) for an FM detuning schedule,
    the integral splined over 4097 points of the pulse (picklable through its samples)."""

    __slots__ = ("_spline", "duration_s", "integral", "offset")

    def __init__(self, integral: np.ndarray, duration_s: float, offset: float) -> None:
        self.integral = np.asarray(integral, dtype=float)
        self.duration_s = float(duration_s)
        self.offset = float(offset)
        grid = np.linspace(0.0, self.duration_s, self.integral.size)
        self._spline = CubicSpline(grid, self.integral, extrapolate=True)

    def __call__(self, t: float, tau: float) -> float:
        return float(self._spline(min(max(tau, 0.0), self.duration_s))) + self.offset

    def __reduce__(self) -> tuple[object, ...]:
        return (_IntegratedBeat, (self.integral, self.duration_s, self.offset))


def _beat_phase_function(
    detuning: Callable[[float], float] | float, t_start_s: float, duration_s: float, mode: PhaseMode
) -> Callable[[float, float], float]:
    """Theta(t, tau): mu t (continuous, constant mu), mu tau (reset), or the FM integral int_0^tau mu dtau' (+ mu(0) t_start)."""
    if not callable(detuning):
        return _LinearBeat(TWO_PI * float(detuning), mode == "continuous")
    mu_fn = detuning
    grid = np.linspace(0.0, duration_s, 4097)
    mu_samples = np.array([TWO_PI * float(mu_fn(x)) for x in grid])
    integral = cumulative_trapezoid(mu_samples, grid, initial=0.0)
    offset = mu_samples[0] * t_start_s if mode == "continuous" else 0.0
    return _IntegratedBeat(np.asarray(integral), duration_s, float(offset))


def micromotion_index(device: Device, ion: int, delta_k: np.ndarray) -> tuple[float, float]:
    """(beta, rf-phase offset) of the ion's excess micromotion along delta_k (Section 4.3.6).

    ``MicromotionIndex.as_modulation``: beta = hypot(in_phase, out_of_phase) >= 0 and the offset -atan2(out_of_phase,
    in_phase) is what the modulated coefficient e^{i beta cos(Omega_rf t + delta)} needs added to the drive's rf phase in
    order to carry BOTH quadratures - and with a SIGNED in-phase index, the pi step across a compensating shim
    (Section 9.17). J_0 is even, so the ``carrier_j0`` route sees only beta.

    (0, 0) means one of the two things that make the micromotion index physically zero: a drive with no wavevector
    (a microwave drive), or a trap with NO rf record, where Section 4.1.1's beta = 0 and C0 = 1 hold by construction.
    Anything else - a trap that carries an rf drive but whose ``Trap.micromotion_beta`` cannot be evaluated, e.g. an rf
    phase imbalance without the rod geometry factors R_m and alpha, a rod record without its single endcap voltage, or a
    point outside the Mathieu stability region - RAISES. A missing geometry factor reported as beta = 0 is a
    default-value fallback that silently switches the micromotion comb off (M1 audit; found by the M5 fixer).
    """
    if float(np.linalg.norm(delta_k)) == 0.0 or device.trap.rf is None:
        return 0.0, 0.0
    return device.trap.micromotion_beta(device.crystal.species[ion], delta_k).as_peak().as_modulation()


def _truncated_exponential(space: HilbertSpace, mode: int, eta: float, order: int) -> qt.Qobj:
    """sum_{p <= order} (i eta (a + a^dag))^p / p! in the mode's own factor space (the Lamb-Dicke expansion option)."""
    d = space.truncation(mode).d
    a = qt.destroy(d)
    gen = 1j * eta * (a + a.dag())
    term = qt.qeye(d)
    total = qt.qeye(d)
    for p in range(1, order + 1):
        term = term * gen / p
        total = total + term
    return total


def _use_kernel(space: HilbertSpace, ion: int, etas: Mapping[int, float], options: BuilderOptions) -> bool:
    """Whether this drive term is held factorized (Section 11.3 item 4): the option, the structural conditions (exact
    displacements, no curvature factor, no ENR coupling, at least one resolved mode) and, under ``auto``, the cost model."""
    if (
        options.kernel == "assembled"
        or options.lamb_dicke_order is not None
        or options.curvature.get(ion) is not None
    ):
        return False
    coupled = [m for m, e in etas.items() if e != 0.0]
    if any(space.mode_class(m) == "enr" for m in coupled):
        return False
    mode_factors = [space.mode_factor(m) for m in coupled if space.mode_class(m) == "resolved"]
    if not mode_factors:
        return False  # a carrier on an all-frozen space: the assembled operator is a tiny CSR matrix
    if options.kernel == "factorized":
        return True
    return prefer_factorized(space.dims, space.ion_factor(ion), mode_factors)


@dataclass(frozen=True)
class _GradientSpec:
    """What the builder needs of a ``gradient`` drive (Section 4.4.5, M4): the per-(ion, mode) force coefficient, the
    gradient's angular frequency and phase, the Bessel weight of the dressed form and its extra rotation."""

    weights: dict[tuple[int, int], float]
    omega_g: float
    phase: float
    bessel: float
    rotation: float
    dressed: bool
    notes: tuple[str, ...]


def _gradient_terms(device: Device, drive: Drive, options: BuilderOptions) -> _GradientSpec:
    """The Section 4.4.5 sigma_z force of ``drive`` on the configured gradient electrodes, everything derived.

    The two tones are the microwave pair at -/+ delta from the (ac-Zeeman-shifted) qubit frequency, each carrying
    2 Omega_mu/2pi in the plan's (hbar Omega/2) convention; their half-difference phase phi_d enters the dressed force as
    e^{i(phi_g + 2 phi_d)} (a pi/2 shift of phi_d reverses the force, which is how the Walsh segments of Srinivas et al.
    close the loop), and the dressed term is resonant where 2 delta = omega_m - omega_g."""
    from qutip_trap.light.microwave import derive_gradient_drive

    grad = device.gradient
    if grad is None:
        raise ValueError(
            "a 'gradient' drive needs Device.gradient (the near-field electrodes' gradient amplitude, frequency and "
            "microwave field amplitude; Section 4.4.5)"
        )
    if len(drive.tones) != 2:
        raise ValueError(
            "a gradient drive carries the two microwave tones symmetrically detuned by +-delta from the qubit "
            "frequency (Section 4.4.5)"
        )
    detunings = []
    for tone in drive.tones:
        if callable(tone.detuning_hz):
            raise NotImplementedError("a gradient drive's microwave tones carry constant detunings +-delta")
        detunings.append(TWO_PI * float(tone.detuning_hz))
    if not math.isclose(detunings[0], -detunings[1], rel_tol=1e-9, abs_tol=0.0) or detunings[0] == 0.0:
        raise ValueError(
            f"the two microwave tones of a gradient drive sit at +-delta: got {detunings[0]:.6g} and "
            f"{detunings[1]:.6g} rad/s"
        )
    delta = abs(detunings[0])
    envelopes = []
    for tone in drive.tones:
        if callable(tone.envelope_hz) or isinstance(tone.envelope_hz, np.ndarray):
            raise NotImplementedError("a gradient drive's microwave tones carry constant amplitudes")
        envelopes.append(TWO_PI * float(tone.envelope_hz))
    # Srinivas's Omega_mu is HALF the tone Rabi frequency of the (hbar Omega/2) convention (light/microwave.py)
    omega_mu = 0.5 * float(np.mean(envelopes))
    phases = []
    for tone in drive.tones:
        if callable(tone.phase_rad):
            raise NotImplementedError("a gradient drive's microwave tones carry constant phases")
        phases.append(float(tone.phase_rad))
    plus = 0 if detunings[0] > 0.0 else 1
    phi_d = 0.5 * (phases[plus] - phases[1 - plus])
    derived = derive_gradient_drive(device, drive.ions)
    weights = {key: val for key, val in derived.coupling_rad_s.items() if key[0] in drive.ions}
    omega_g = TWO_PI * grad.frequency_hz
    bessel = float(jv(2, 4.0 * omega_mu / delta))
    notes: list[str] = []
    dressed = options.gradient_form == "dressed"
    if dressed:
        notes.append(
            "gradient drive: microwave dressing adiabatically eliminated, the force carries "
            f"J_2(4 Omega_mu/delta) = {bessel:.6f} at Omega_mu/delta = {omega_mu / delta:.4f} and the counter-rotating "
            "(omega_m + omega_g), J_0 and sigma_y dressed terms are dropped"
        )
    else:
        notes.append(
            f"gradient drive: the two microwave tones at +-{delta / TWO_PI:.6g} Hz are built as carrier terms and the "
            f"J_2(4 Omega_mu/delta) = {bessel:.6f} weight of the sigma_z force emerges from the dynamics "
            f"(J_0 = {float(jv(0, 4.0 * omega_mu / delta)):.6f}; intrinsic dynamical decoupling at J_0 = 0)"
        )
    return _GradientSpec(
        weights=weights,
        omega_g=omega_g,
        phase=grad.phase_rad + (2.0 * phi_d if dressed else 0.0),
        bessel=bessel,
        rotation=omega_g + 2.0 * delta if dressed else 0.0,
        dressed=dressed,
        notes=tuple(notes),
    )


def _drive_operator(
    space: HilbertSpace,
    ion: int,
    etas: Mapping[int, float],
    options: BuilderOptions,
    device: Device,
    ion_op: qt.Qobj | None = None,
) -> tuple[qt.Qobj, bool]:
    """sigma_+^ion (or ``ion_op``) (x) prod_m D_m (exact or expanded) with the optional symmetrized curvature factor, and whether
    it is held factorized (the matrix-free kernel of Section 11.3 item 4) or assembled (CSR)."""
    if options.lamb_dicke_order is None:
        if _use_kernel(space, ion, etas, options):
            return space.drive_operator_factorized(ion, etas, ion_op=ion_op), True
        op = space.drive_operator(ion, etas, ion_op=ion_op)
    else:
        ops: dict[int, qt.Qobj] = {
            space.ion_factor(ion): qudit_sigma_plus(space.ion_dim(ion)) if ion_op is None else ion_op
        }
        for mode, eta in etas.items():
            if eta == 0.0 or space.mode_class(mode) != "resolved":
                if eta != 0.0 and space.mode_class(mode) == "enr":
                    raise NotImplementedError("lamb_dicke_order with an ENR group is not supported")
                continue
            ops[space.mode_factor(mode)] = _truncated_exponential(space, mode, eta, options.lamb_dicke_order)
        op = space.embed_many(ops)
    spec = options.curvature.get(ion)
    if spec is not None:
        x_op = _curvature_position_operator(space, device, ion, spec)
        factor = space.identity() + 0.5 * spec.kappa_per_m2 * x_op * x_op
        op = 0.5 * (op * factor + factor * op)
    return op.to("CSR"), False


def _curvature_position_operator(
    space: HilbertSpace, device: Device, ion: int, spec: CurvatureSpec
) -> qt.Qobj:
    """x_hat_ion along the curvature axis over the resolved modes: sum_m c_im (e_m . axis) x0_m (a_m + a_m^dag)."""
    from qutip_trap.units import HBAR_J_S

    axis = np.asarray(spec.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mass = float(device.crystal.masses_kg[ion])
    x_op = 0.0 * space.identity()
    for tr in space.resolved:
        mode = device.crystal.modes[tr.mode]
        proj = float(np.dot(mode.displacement_pattern()[ion], axis))
        if proj == 0.0:
            continue
        x0 = math.sqrt(HBAR_J_S / (2.0 * mass * mode.omega_rad_s))
        x_op = x_op + proj * x0 * space.position(tr.mode)
    return x_op


def _frozen_fock_states(
    space: HilbertSpace, sample: NoiseSample, frozen_n: Mapping[int, int] | None
) -> dict[int, int]:
    out: dict[int, int] = {}
    for m in space.frozen:
        if frozen_n is not None and m in frozen_n:
            out[m] = int(frozen_n[m])
        elif key_frozen_n(m) in sample.values:
            out[m] = int(round(sample.values[key_frozen_n(m)]))
        else:
            out[m] = 0
    return out


# ---- the builder ---------------------------------------------------------------------------------------------------------


def _beam_phase_trajectory(smp: NoiseSample, drive: Drive) -> Trajectory | None:
    """Delta phi(t), the sampled beam-path phase of the drive's beat note: phi_2(t) - phi_1(t) for a Raman or light-shift
    pair (``Delta k = k_1 - k_2``) and -phi_1(t) for a single-beam optical drive, the convention of ``key_beam_phase_rad``;
    ``None`` when the sample carries no beam-phase trajectory for the drive's beams (a beam without one contributes zero)."""
    if drive.kind in ("raman", "light_shift"):
        t1 = smp.trajectory(key_beam_phase_trajectory_rad(drive.beams[1]))
        t0 = smp.trajectory(key_beam_phase_trajectory_rad(drive.beams[0]))
        if t1 is None and t0 is None:
            return None
        if t1 is None:
            assert t0 is not None
            return t0.scaled(-1.0)
        if t0 is None:
            return t1
        return _sum_trajectories(t1, t0.scaled(-1.0))
    if drive.kind in ("optical_E1", "optical_E2"):
        t0 = smp.trajectory(key_beam_phase_trajectory_rad(drive.beams[0]))
        return None if t0 is None else t0.scaled(-1.0)
    return None


def _sum_trajectories(a: Trajectory, b: Trajectory) -> Trajectory:
    """a(t) + b(t) on their common grid. ``NoiseModel`` synthesizes every trajectory of one sample on one ``time_grid``, so
    two grids that differ are a construction error, not a case to interpolate over."""
    if a.times_s.shape != b.times_s.shape or not np.array_equal(a.times_s, b.times_s):
        raise ValueError(
            "two noise trajectories of one sample sit on different time grids; NoiseModel synthesizes them on one grid"
        )
    return Trajectory(a.times_s, a.values + b.values)


def build_hamiltonian(
    device: Device,
    pulses: Sequence[Pulse],
    space: HilbertSpace,
    *,
    sample: NoiseSample | None = None,
    options: BuilderOptions | None = None,
    qubit_shifts_hz: Mapping[int, float] | None = None,
    frozen_n: Mapping[int, int] | None = None,
    mode_frequencies_hz: Mapping[int, float] | None = None,
) -> BuiltHamiltonian:
    """Assemble H(t) for the pulses active on one time segment (every pulse must span the whole segment).

    ``qubit_shifts_hz``: extra per-ion offsets of the true transition from the frame (an ac Zeeman shift, a static
    field error) added to the sample's; ``frozen_n``: the frozen spectators' Fock states for this shot (default: the
    sample's keys, else 0); ``mode_frequencies_hz``: overrides of the crystal's mode frequencies (calibrated values).
    """
    opts = options or BuilderOptions()
    smp = sample or quiet_sample()
    approximations: list[str] = []
    counter = _Counter()
    crystal = device.crystal
    n_modes = len(crystal.modes)

    if pulses:
        # the segment is the interval every pulse covers (the engine cuts the schedule at every pulse boundary, so the
        # pulses it passes all span the segment; simultaneous pulses of different lengths, two pi/2 pulses at two ions'
        # fitted Rabi frequencies, overlap without coinciding); every pulse's own clock (tau, its duration) is its own
        t_start = max(p.t_start_s for p in pulses)
        t_end = min(p.t_end_s for p in pulses)
        if t_end <= t_start:
            raise ValueError(
                "build_hamiltonian takes the pulses active on ONE segment (a common interval); split the schedule at pulse "
                "boundaries"
            )
    else:
        t_start, t_end = 0.0, 0.0

    # mode frequencies (rad/s) with the sample's quasi-static offsets
    omegas: dict[int, float] = {}
    for m in range(n_modes):
        base = (
            crystal.modes[m].omega_hz
            if mode_frequencies_hz is None or m not in mode_frequencies_hz
            else mode_frequencies_hz[m]
        )
        omegas[m] = TWO_PI * (float(base) + smp.get(key_mode_offset_hz(m), 0.0))

    terms: list[Any] = []
    omega_max = 0.0

    # H_mot
    if opts.frame == "schrodinger":
        h_mot = 0.0 * space.identity()
        for m in range(n_modes):
            if space.mode_class(m) != "frozen":
                h_mot = h_mot + omegas[m] * space.number(m)
                omega_max = max(omega_max, omegas[m])
        static = h_mot.to("CSR")
    else:
        static = (0.0 * space.identity()).to("CSR")
        approximations.append("frame=interaction: H_0 removed, drives decomposed into sideband operators")

    # H_int (the ions are DEVICE indices: a GATE_LOCAL space carries a subset of the crystal, Section 5.4)
    for i in space.ion_labels:
        delta = smp.get(key_qubit_offset_hz(i), 0.0) + (
            0.0 if qubit_shifts_hz is None else float(qubit_shifts_hz.get(i, 0.0))
        )
        if delta != 0.0:
            static = static + (0.5 * TWO_PI * delta) * space.sigma_z(i)
    # the sampled part of the transition offsets (S_B through the sensitivities plus the mains, Section 6.3): a
    # time-dependent (delta nu_i(t)/2) sigma_z^i on the fixed grid of the sample, whatever the integrator's steps
    trajectory_terms: list[Any] = []
    for i in space.ion_labels:
        traj = smp.trajectory(key_qubit_trajectory_hz(i))
        if traj is not None:
            trajectory_terms.append(
                [
                    (0.5 * TWO_PI * space.sigma_z(i)).to("CSR"),
                    qt.coefficient(_traj_coef, args={"traj": traj, "scale": 1.0}),
                ]
            )
            approximations.append(
                f"ion {i}: sampled qubit-frequency trajectory (rms {traj.rms():.3g} Hz) on {traj.times_s.size} grid points"
            )
    rf_traj = smp.trajectory(KEY_RF_FRACTION_TRAJECTORY)
    if rf_traj is not None:
        if opts.frame != "schrodinger":
            raise NotImplementedError(
                "a sampled rf-amplitude trajectory needs the Schroedinger frame (H_mot present)"
            )
        op_rf = 0.0 * space.identity()
        for m in range(n_modes):
            if space.mode_class(m) != "frozen" and crystal.modes[m].family in (
                "transverse_1",
                "transverse_2",
            ):
                op_rf = op_rf + omegas[m] * space.number(m)
        trajectory_terms.append(
            [op_rf.to("CSR"), qt.coefficient(_traj_coef, args={"traj": rf_traj, "scale": 1.0})]
        )
        approximations.append(
            f"sampled rf-amplitude trajectory (rms {rf_traj.rms():.3g}) on the transverse modes"
        )
    laser_phase_traj = smp.trajectory(KEY_LASER_PHASE_TRAJECTORY)
    intensity_traj = smp.trajectory(KEY_INTENSITY_TRAJECTORY)
    laser_offset_hz = smp.get(KEY_LASER_OFFSET_HZ, 0.0)

    # H_anh
    anh = device.trap.anharmonic()
    if opts.include_anharmonic and anh is not None:
        if opts.frame != "schrodinger":
            raise NotImplementedError("anharmonic terms are built in the Schroedinger frame only")
        for key3, coeff in anh.cubic_rad_s.items():
            if all(space.mode_class(m) == "resolved" for m in key3):
                op = space.identity()
                for m in key3:
                    op = op * space.position(m)
                static = static + coeff * op
            else:
                approximations.append(f"anharmonic term {key3} dropped: a mode is not resolved")
        for key4, coeff4 in anh.quartic_rad_s.items():
            if all(space.mode_class(m) == "resolved" for m in key4):
                op4 = space.identity()
                for m in key4:
                    op4 = op4 * space.position(m)
                static = static + coeff4 * op4
            else:
                approximations.append(f"anharmonic term {key4} dropped: a mode is not resolved")
    elif not opts.include_anharmonic and anh is not None:
        approximations.append(
            "anharmonic terms present on the trap but switched off (include_anharmonic=False)"
        )
    # Section 5.7: the cubic Hamiltonian is opt-in, but "the resonance checker and the estimated accumulated phase are
    # ON BY DEFAULT" - so this runs whatever include_anharmonic says, gated only by the record's own resonance_check
    # switch and by there being three modes for a triple to exist (Sections 4.1.4, 9.12; audit item E.5).
    if anh is not None and anh.resonance_check and n_modes >= 3:
        estimate = anharmonic_estimate(crystal, anh, t_end - t_start)
        if estimate is not None:
            approximations.append(estimate.summary())

    terms.append(static)
    terms.extend(trajectory_terms)
    records: list[DriveRecord] = []
    frozen_states = _frozen_fock_states(space, smp, frozen_n)
    rabi_scale = smp.get(KEY_RABI_SCALE, 1.0)
    dropped_total = 0.0
    n_drive_terms = 0
    drive_parts: dict[str, list[Any]] = {}
    kernel_flags: list[bool] = []

    for pulse in pulses:
        drive: Drive = pulse.drive
        duration = pulse.duration_s
        gradient = _gradient_terms(device, drive, opts) if drive.kind == "gradient" else None
        delta_k = drive.delta_k(device.beams)
        part_key = pulse.gate_id or f"pulse@{pulse.t_start_s:.9g}"
        part = drive_parts.setdefault(part_key, [])
        # quasi-static beam-path phases: Delta phi, the phase of the beat note written E ~ cos(omega_L t - Delta k . r
        # + Delta phi), so Delta phi = phi_2 - phi_1 for a Raman pair with Delta k = k_1 - k_2 and E_j ~ cos(k_j . r -
        # omega_j t + phi_j), and Delta phi = -phi_1 for a single beam (PLAN.md:808; Section 13 row "Optical phase factor
        # on sigma_+": the factor is e^{+i(Delta k . X - Delta phi)}, so the RELATIVE sign of the two is physical)
        beam_phase = 0.0
        if drive.kind in ("raman", "light_shift"):
            beam_phase = smp.get(key_beam_phase_rad(drive.beams[1]), 0.0) - smp.get(
                key_beam_phase_rad(drive.beams[0]), 0.0
            )
        elif drive.kind in ("optical_E1", "optical_E2"):
            beam_phase = -smp.get(key_beam_phase_rad(drive.beams[0]), 0.0)
        if beam_phase != 0.0:
            approximations.append(
                f"pulse {pulse.gate_id!r}: quasi-static beam-path phase Delta phi = {beam_phase:.3g} rad"
            )
        if drive.comb is not None:
            # the validity hierarchy of Section 4.3.7 is "a runtime guard logged before every solve": the clauses the
            # builder can evaluate from the pulse go into approximations here, the rest at construction time in
            # light.raman.comb_drive, which is where the species and the intensity chain are known (M2 audit E8)
            approximations.extend(
                comb_build_notes(
                    drive.comb,
                    duration,
                    len(drive.tones),
                    float(drive.stark_shift_hz) if not callable(drive.stark_shift_hz) else float("nan"),
                )
            )
        is_laser = drive.kind in ("raman", "light_shift", "optical_E1", "optical_E2")
        phase_traj = laser_phase_traj if drive.kind in ("optical_E1", "optical_E2") else None
        # the SAMPLED beam-path phase Delta phi(t) (Section 6.3 route (d), Section 7.10; conv.beam_phase_spectrum): the same
        # combination as the quasi-static Delta phi above, on the sample's shared grid. The static factor below carries
        # e^{-i Delta phi} and _DriveCoefficient adds its phase trajectory as e^{+i phi(t)}, so it enters as -Delta phi(t),
        # beside a single-photon drive's own laser phase (M7 hand-off, consolidated 2026-09-08)
        beam_phase_traj = _beam_phase_trajectory(smp, drive)
        if beam_phase_traj is not None:
            minus = beam_phase_traj.scaled(-1.0)
            phase_traj = minus if phase_traj is None else _sum_trajectories(phase_traj, minus)
            approximations.append(
                f"pulse {pulse.gate_id!r}: sampled beam-path phase trajectory Delta phi(t) "
                f"(rms {beam_phase_traj.rms():.3g} rad) enters the beat note as e^(-i Delta phi(t))"
            )
        amp_traj = intensity_traj if is_laser else None
        amp_power = 0.5 if drive.kind in ("optical_E1", "optical_E2") else 1.0
        laser_rotation = (
            -TWO_PI * laser_offset_hz
            if (drive.kind in ("optical_E1", "optical_E2") and laser_offset_hz != 0.0)
            else 0.0
        )
        if phase_traj is not None:
            approximations.append(
                f"pulse {pulse.gate_id!r}: sampled laser phase trajectory (rms {phase_traj.rms():.3g} rad)"
            )
        if amp_traj is not None:
            approximations.append(
                f"pulse {pulse.gate_id!r}: sampled intensity trajectory (rms {amp_traj.rms():.3g}) at power {amp_power:g}"
            )
        if laser_rotation != 0.0:
            approximations.append(f"pulse {pulse.gate_id!r}: laser frequency offset {laser_offset_hz:.3g} Hz")
        # micromotion modulation parameters
        rf_omega = 0.0
        rf_delta = 0.0
        if opts.micromotion == "modulated":
            if device.trap.rf is None:
                raise ValueError("micromotion='modulated' needs the trap's rf record")
            rf_omega = device.trap.rf.omega_rad_s
            if drive.rf_locked:
                rf_delta = float(drive.rf_phase_rad or 0.0)
            elif KEY_RF_PHASE in smp.values:
                rf_delta = smp.values[KEY_RF_PHASE]
            else:
                raise ValueError(
                    "an unlocked drive has no rf phase reference: use micromotion='carrier_j0' (the shot average, "
                    "Section 4.3.6) or supply the sample key rf_phase_rad"
                )
        targets: list[tuple[int, complex]] = [(i, 1.0 + 0.0j) for i in drive.ions]
        for ion_addr in drive.ions:
            if not space.has_ion(ion_addr):
                raise ValueError(
                    f"pulse {pulse.gate_id!r} addresses ion {ion_addr}, which the space does not carry (ions {space.ion_labels})"
                )
        if opts.include_crosstalk:
            for j, eps in drive.crosstalk.items():
                if j in drive.ions:
                    raise ValueError("crosstalk targets are neighbours, not the addressed ions")
                if not space.has_ion(j):
                    # a GATE_LOCAL space carries the addressed ions and the neighbours above the crosstalk threshold
                    # (Section 5.4); a neighbour outside it receives the light in reality and not here: recorded
                    approximations.append(
                        f"pulse {pulse.gate_id!r}: crosstalk |eps| = {abs(complex(eps)):.3g} onto ion {j} outside the space dropped"
                    )
                    continue
                targets.append((j, complex(eps)))
        elif drive.crosstalk:
            approximations.append(f"crosstalk of pulse {pulse.gate_id!r} switched off")
        primary = drive.ions[0]
        x_primary = np.asarray(crystal.positions_m[primary], dtype=float)
        # a drive whose every tone is the constant 0 (one beam of a pair on: a light shift with no two-photon coupling, the
        # Stark scan of Section 7.5 item 7) has an identically zero drive term; it is left out of H (exactly, its coefficient
        # is 0 at every t) so that the segment's Hamiltonian is the constant H_mot + H_int + H_Stark; the Stark term is kept
        silent = all(
            not callable(tone.envelope_hz)
            and not isinstance(tone.envelope_hz, np.ndarray)
            and float(tone.envelope_hz) == 0.0
            for tone in drive.tones
        )
        if gradient is not None and gradient.dressed:
            # the dressed effective form REPLACES the two microwave tones by their J_2 weight on the force, so the
            # spin-flip carrier terms they would otherwise build are not part of that Hamiltonian
            silent = True
        tone_fns: list[_ToneFn] = []
        for tone in drive.tones:
            constants: tuple[float, float, float] | None = None
            if not any(
                callable(v) or isinstance(v, np.ndarray) for v in (tone.envelope_hz, tone.phase_rad)
            ) and (not callable(tone.detuning_hz)):
                constants = (
                    TWO_PI * float(tone.envelope_hz),  # type: ignore[arg-type]
                    float(tone.phase_rad),  # type: ignore[arg-type]
                    TWO_PI * float(tone.detuning_hz),
                )
            tone_fns.append(
                _ToneFn(
                    envelope=_as_time_function(tone.envelope_hz, duration, scale=TWO_PI),
                    phase=_as_time_function(tone.phase_rad, duration, scale=1.0),
                    beat_phase=_beat_phase_function(
                        tone.detuning_hz, pulse.t_start_s, duration, opts.phase_mode
                    ),
                    constants=constants,
                    continuous=opts.phase_mode == "continuous",
                )
            )
            mu0 = abs(
                TWO_PI * float(tone.detuning_hz(0.0) if callable(tone.detuning_hz) else tone.detuning_hz)
            )
            omega_max = max(omega_max, mu0)
        for ion, eps in targets:
            if space.ion_dim(ion) > 2:
                approximations.append(
                    f"ion {ion} has d = {space.ion_dim(ion)}: the drive couples levels 0 and 1 only ({M3A})"
                )
            etas, c0_applied = lamb_dicke_parameters(device, ion, delta_k)
            if not c0_applied and float(np.linalg.norm(delta_k)) > 0.0:
                approximations.append(
                    f"ion {ion}: no rf record, C0 = 1 and beta = 0 in the Lamb-Dicke parameters"
                )
            beta, beta_phase = (
                micromotion_index(device, ion, delta_k) if opts.micromotion != "none" else (0.0, 0.0)
            )
            if opts.micromotion == "carrier_j0":
                carrier = float(jv(0, beta))
                if beta != 0.0:
                    approximations.append(
                        f"ion {ion}: unlocked micromotion, carrier x J_0({beta:.4f}) = {carrier:.6f} (Section 4.3.6)"
                    )
            else:
                carrier = 1.0
            dw = 1.0
            if opts.frozen_debye_waller:
                for m in space.frozen:
                    # a DROPPED mode is not modelled at all (Section 5.2: "nothing absorbs a dropped mode's loss"), so it
                    # gets no Debye-Waller factor either; only the frozen spectators do (M6 fix)
                    if m in space.dropped:
                        continue
                    dw *= debye_waller_factor(frozen_states[m], etas[m])
            # the optical phase carries the RELATIVE ion-position drift as well as the equilibrium geometry: a stray
            # field displaces ion i by u_0(i) = Q E_dc/(m omega^2) (Section 4.1.1; Berkeland 1998 Eq. 16), sampled as
            # key_position_offset_m, and Section 13's row "Optical phase factor on sigma_+" stores "beam-path AND
            # ion-position drift ... as one scalar per ion per pulse". The primary's own drift is a per-ion constant
            # that the frame alignment of Section 7.5 step 3 absorbs, so only Delta k . (u_0(j) - u_0(i)) survives.
            geometric = (
                0.0
                if ion == primary
                else float(
                    np.dot(
                        delta_k,
                        (np.asarray(crystal.positions_m[ion], dtype=float) + _position_offset(ion, smp))
                        - (x_primary + _position_offset(primary, smp)),
                    )
                )
            )
            drift_phase = float(np.dot(delta_k, _position_offset(ion, smp) - _position_offset(primary, smp)))
            if drift_phase != 0.0:
                approximations.append(
                    f"pulse {pulse.gate_id!r}, ion {ion}: sampled ion-position drift phase "
                    f"Delta k . (u_0({ion}) - u_0({primary})) = {drift_phase:.6g} rad"
                )
            pointing = _pointing_factor(device, drive, ion, smp)
            if pointing != 1.0:
                approximations.append(
                    f"pulse {pulse.gate_id!r}, ion {ion}: beam pointing/position factor {pointing:.6f} on Omega"
                )
            scale = rabi_scale * eps * carrier * dw * pointing * np.exp(1j * (geometric - beam_phase))
            active_etas = {m: e for m, e in etas.items() if space.mode_class(m) != "frozen"}
            peak = 0.0
            for tone in drive.tones:
                env = tone.envelope_hz
                if callable(env):
                    peak = max(peak, abs(TWO_PI * float(env(0.0))))
                elif isinstance(env, np.ndarray):
                    peak = max(peak, float(np.max(np.abs(env))) * TWO_PI)
                else:
                    peak = max(peak, abs(TWO_PI * float(env)))
            records.append(
                DriveRecord(
                    pulse=pulse.gate_id,
                    ion=ion,
                    primary_ion=primary,
                    etas=etas,
                    frozen_n={m: frozen_states[m] for m in space.frozen},
                    debye_waller=dw,
                    micromotion_beta=beta,
                    carrier_factor=carrier,
                    crosstalk=eps,
                    rabi_scale=rabi_scale,
                    omega_peak_rad_s=peak * abs(scale),
                )
            )
            # the ion operators this drive couples through: sigma_+ for a spin-flip drive; the level-weighted projector
            # (the force) plus the far-off-resonant sigma_+ (the same beams' Raman coupling) for a light-shift drive
            ion_terms: list[tuple[qt.Qobj | None, complex, float]] = [(None, 1.0 + 0.0j, 0.0)]
            if drive.kind == "light_shift":
                ls = drive.light_shift
                assert ls is not None
                d_ion = space.ion_dim(ion)
                w_dn, w_up = ls.level_weights
                force_op = w_dn * qudit_projector(d_ion, 0) + w_up * qudit_projector(d_ion, 1)
                ion_terms = [(force_op, 1.0 + 0.0j, 0.0)]
                omega_0 = TWO_PI * ls.qubit_freq_hz
                mu_beat = TWO_PI * float(
                    drive.tones[0].detuning_hz(0.0)
                    if callable(drive.tones[0].detuning_hz)
                    else drive.tones[0].detuning_hz
                )
                excitation = (
                    abs(ls.spin_flip_weight) * peak * abs(scale) / max(abs(omega_0 - mu_beat), 1e-300)
                ) ** 2
                if ls.spin_flip_weight != 0.0 and excitation >= 1e-12 and opts.frame == "schrodinger":
                    ion_terms.append((None, complex(ls.spin_flip_weight), omega_0))
                    omega_max = max(omega_max, abs(omega_0 - mu_beat))
                    approximations.append(
                        f"ion {ion}: light-shift drive keeps the off-resonant Raman spin flip (w = {abs(ls.spin_flip_weight):.3g}, "
                        f"excitation {excitation:.2e})"
                    )
                elif ls.spin_flip_weight != 0.0:
                    approximations.append(
                        f"ion {ion}: light-shift drive drops the off-resonant Raman spin flip (excitation {excitation:.2e})"
                    )
            for ion_op, weight, extra_rotation in ion_terms:
                if silent:
                    continue
                term_scale = complex(scale) * weight
                if opts.frame == "schrodinger":
                    op, factorized = _drive_operator(space, ion, active_etas, opts, device, ion_op)
                    kernel_flags.append(factorized)
                    coef = _DriveCoefficient(
                        pulse.t_start_s,
                        tone_fns,
                        term_scale,
                        beta if opts.micromotion == "modulated" else 0.0,
                        rf_omega,
                        # the micromotion index's own quadrature offset -atan2(beta_op, beta_ip) on top of the drive's rf
                        # phase, so that beta cos(Omega t + delta) is beta_ip cos + beta_op sin (Section 9.17)
                        rf_delta + beta_phase,
                        extra_rotation + laser_rotation,
                        counter,
                        phase_traj,
                        amp_traj,
                        amp_power,
                    )
                    t_plain = [op, qt.coefficient(_coef_plain, args={"coef": coef})]
                    t_conj = [op.dag(), qt.coefficient(_coef_conj, args={"coef": coef})]
                    terms.extend([t_plain, t_conj])
                    part.extend([t_plain, t_conj])
                    n_drive_terms += 2
                    continue
                kernel_flags.append(False)
                if opts.curvature.get(ion) is not None:
                    raise NotImplementedError("beam curvature is built in the Schroedinger frame only")
                mats = None
                if opts.lamb_dicke_order is not None:
                    mats = {
                        m: _truncated_exponential(space, m, e, opts.lamb_dicke_order).full()
                        for m, e in active_etas.items()
                        if e != 0.0 and space.mode_class(m) == "resolved"
                    }
                embedded = None if ion_op is None else space.embed(ion_op, space.ion_factor(ion))
                pic = interaction_picture(
                    space, ion, active_etas, k_max=opts.k_max, matrices=mats, ion_op=embedded
                )
                dropped_total += pic.dropped_weight
                kept = list(pic.terms)
                tones_by_term: dict[int, list[_ToneFn]] = {}
                if opts.rwa:
                    # the textbook model: each tone drives ONLY the sideband combination nearest its resonance, so a kept
                    # operator carries the coefficients of its resonant tones alone (no counter-rotating cross terms)
                    for tone, tone_fn in zip(drive.tones, tone_fns):
                        mu = TWO_PI * float(
                            tone.detuning_hz(0.0) if callable(tone.detuning_hz) else tone.detuning_hz
                        )
                        best = min(
                            range(len(kept)),
                            key=lambda idx: abs(
                                mu - sum(k * omegas[m] for k, m in zip(kept[idx].k, kept[idx].modes))
                            ),
                        )
                        tones_by_term.setdefault(best, []).append(tone_fn)
                    dropped_total += sum(
                        term.weight for idx, term in enumerate(kept) if idx not in tones_by_term
                    )
                    kept_pairs = [(kept[idx], tones_by_term[idx]) for idx in sorted(tones_by_term)]
                    approximations.append(
                        f"ion {ion}: rwa keeps {len(kept_pairs)} sideband term(s) of {len(pic.terms)}, "
                        "each with its resonant tone(s)"
                    )
                else:
                    kept_pairs = [(term, tone_fns) for term in kept]
                if opts.k_max is not None:
                    approximations.append(
                        f"ion {ion}: sideband sum truncated at |k| <= {opts.k_max}, dropped weight {pic.dropped_weight:.3e}"
                    )
                shared: dict[int, _DriveCoefficient] = {}
                for term, term_tones in kept_pairs:
                    k_dot_w = sum(k * omegas[m] for k, m in zip(term.k, term.modes))
                    omega_max = max(omega_max, abs(k_dot_w))
                    tone_sum = shared.get(id(term_tones))
                    if tone_sum is None:
                        tone_sum = _DriveCoefficient(
                            pulse.t_start_s,
                            term_tones,
                            term_scale,
                            0.0,
                            0.0,
                            0.0,
                            extra_rotation + laser_rotation,
                            counter,
                            phase_traj,
                            amp_traj,
                            amp_power,
                        )
                        shared[id(term_tones)] = tone_sum
                    rotated = _RotatedCoefficient(tone_sum, k_dot_w)
                    t_plain = [term.op, qt.coefficient(_coef_plain, args={"coef": rotated})]
                    t_conj = [term.op.dag(), qt.coefficient(_coef_conj, args={"coef": rotated})]
                    terms.extend([t_plain, t_conj])
                    part.extend([t_plain, t_conj])
                    n_drive_terms += 2
            if opts.lamb_dicke_order is not None:
                approximations.append(
                    f"ion {ion}: displacement operators expanded to order {opts.lamb_dicke_order} in eta"
                )
        # the near-field microwave-gradient force (Section 4.4.5, M4): sum_{i,m} w_{i,m} sigma_z^i (a_m + a_m^dag) x
        # cos(omega_g t + phi_g), the position-dependent Zeeman shift of the oscillating gradient, with w_{i,m} built
        # from the mode's mass-weighted displacement pattern by ``light.microwave.derive_gradient_drive``. The two
        # microwave tones of the same drive were built above as ordinary carrier terms (``bare``), so J_2(4 Omega_mu/delta)
        # and the J_0 = 0 decoupling come out of the dynamics; ``dressed`` builds the eliminated form instead.
        if gradient is not None:
            if opts.frame != "schrodinger":
                raise NotImplementedError("a gradient drive is built in the Schroedinger frame only")
            approximations.extend(gradient.notes)
            omega_max = max(omega_max, gradient.omega_g)
            for ion in drive.ions:
                if not space.has_ion(ion):
                    raise ValueError(
                        f"pulse {pulse.gate_id!r} addresses ion {ion}, which the space does not carry"
                    )
                sz = space.sigma_z(ion)
                for tr in space.resolved:
                    weight = gradient.weights.get((ion, tr.mode), 0.0)
                    if weight == 0.0:
                        continue
                    op = (sz * space.annihilation(tr.mode)).to("CSR")
                    grad_coef = _GradientCoefficient(
                        weight=0.5 * weight * gradient.bessel if gradient.dressed else weight,
                        omega_g=gradient.omega_g,
                        phase=gradient.phase,
                        t_start=pulse.t_start_s,
                        counter=counter,
                        rotation=gradient.rotation,
                        dressed=gradient.dressed,
                        continuous=opts.phase_mode == "continuous",
                    )
                    t_plain = [op, qt.coefficient(_coef_plain, args={"coef": grad_coef})]
                    t_conj = [op.dag(), qt.coefficient(_coef_conj, args={"coef": grad_coef})]
                    terms.extend([t_plain, t_conj])
                    part.extend([t_plain, t_conj])
                    n_drive_terms += 2
                    kernel_flags.append(False)
                for m in space.frozen:
                    if gradient.weights.get((ion, m), 0.0) != 0.0:
                        approximations.append(
                            f"pulse {pulse.gate_id!r}, ion {ion}: gradient force on frozen mode {m} dropped"
                        )
        # H_Stark
        if opts.include_stark:
            st = drive.stark_shift_hz
            if callable(st):
                fn = _as_time_function(st, duration, scale=0.5 * TWO_PI)
                sz = 0.0 * space.identity()
                for ion, eps in targets:
                    sz = sz + (abs(eps) ** 2) * space.sigma_z(ion)
                sc = _ScalarCoefficient(fn, pulse.t_start_s, counter)
                terms.append([sz.to("CSR"), qt.coefficient(_coef_plain, args={"coef": sc})])
                n_drive_terms += 1
            elif float(st) != 0.0:
                for ion, eps in targets:
                    terms[0] = terms[0] + (0.5 * TWO_PI * float(st) * abs(eps) ** 2) * space.sigma_z(ion)
        elif callable(drive.stark_shift_hz) or float(drive.stark_shift_hz) != 0.0:
            approximations.append(f"Stark shift of pulse {pulse.gate_id!r} switched off")
        if opts.micromotion == "modulated" and device.trap.rf is not None:
            omega_max = max(omega_max, device.trap.rf.omega_rad_s)
        elif opts.micromotion == "none":
            approximations.append("micromotion factors switched off")

    terms[0] = terms[0].to("CSR")
    H = qt.QobjEvo(terms)
    kernel: Literal["assembled", "factorized", "mixed", "none"]
    if not kernel_flags:
        kernel = "none"
    elif all(kernel_flags):
        kernel = "factorized"
    elif not any(kernel_flags):
        kernel = "assembled"
    else:
        kernel = "mixed"
    if kernel == "mixed":
        approximations.append(
            "drive terms held partly factorized and partly assembled (an ENR-coupled or curvature drive beside a product-space one)"
        )
    fingerprint = canonical_digest(
        (
            "build_hamiltonian",
            # H(t) depends on the device through every eta (lamb_dicke_parameters), the beam waists and pointing
            # (_pointing_factor) and the geometric phase Delta k . X_i, none of which any other entry of this digest
            # carries: without it the engine's propagator cache serves one device's propagator for another (M9b audit
            # B1: two devices differing only in beam 0's wavelength gave 0.4950194835 instead of 0.4948647699)
            device.hash(),
            tuple(space.ion_dims),
            tuple(space.dims),
            tuple(space.ion_labels),
            tuple(space.frozen),
            space.enr_group,
            tuple(sorted(omegas.items())),
            float(t_start),
            float(t_end),
            tuple(fingerprint_pulse(p) for p in pulses),
            tuple(sorted(frozen_states.items())),
            tuple(
                sorted((k, v) for k, v in smp.values.items() if k != KEY_BRANCH_WEIGHT)
            ),  # the weight steers no term of H
            tuple(sorted(smp.ou_grids.items())),
            float(smp.t_s),
            tuple(sorted((int(k), float(v)) for k, v in (qubit_shifts_hz or {}).items())),
            tuple(sorted((int(k), float(v)) for k, v in (mode_frequencies_hz or {}).items())),
            replace(opts, kernel="auto"),  # the same H(t) whichever way its drive operators are held
        )
    )
    return BuiltHamiltonian(
        H=H,
        space=space,
        t_start_s=t_start,
        t_end_s=t_end,
        frame=opts.frame,
        omega_max_rad_s=omega_max,
        n_drive_terms=n_drive_terms,
        approximations=tuple(approximations),
        records=tuple(records),
        dropped_weight=dropped_total,
        mode_frequencies_rad_s=omegas,
        counter=counter,
        drive_parts={k: qt.QobjEvo(v) for k, v in drive_parts.items() if v},
        kernel=kernel,
        fingerprint=fingerprint,
    )


def _position_offset(ion: int, sample: NoiseSample) -> np.ndarray:
    """The sampled quasi-static displacement u_0 of ``ion`` (m, laboratory axes): Section 4.1.1's Q E_dc/(m omega^2)
    under an uncompensated stray field (Berkeland 1998 Eq. 16), drawn by ``noise/model.py``'s ``stray_field_drift``.

    It enters the drive TWICE and in two different ways: in the optical phase, as Delta k . u_0 (Section 13 row
    "Optical phase factor on sigma_+"), and in the intensity, by moving the ion on the beams' profile
    (``_pointing_factor``). For 171Yb+ at 355 nm counter-propagating the phase term is the larger by six orders of
    magnitude (0.51 rad against 2.6e-7 in amplitude at u_0 = 14.3 nm), so dropping either is not symmetric.
    """
    return np.array([sample.get(key_position_offset_m(ion, ax), 0.0) for ax in range(3)])


def _pointing_factor(device: Device, drive: Drive, ion: int, sample: NoiseSample) -> float:
    """sqrt(prod_beams I_b(x_ion + dx_ion - d_b)/I_b(x_ion)) for the drive's beams: a beam pointing offset d_b or an ion
    displacement dx_ion (a stray field) moves the ion on the intensity profile, so Omega and the crosstalk ratios change
    together (Sections 6.6, 7.10); 1 for a microwave drive and for the nominal sample."""
    if not drive.beams:
        return 1.0
    x = np.asarray(device.crystal.positions_m[ion], dtype=float)
    dx = _position_offset(ion, sample)
    factor = 1.0
    for b in drive.beams:
        d_b = np.array([sample.get(key_beam_offset_m(b, ax), 0.0) for ax in range(3)])
        if not np.any(dx) and not np.any(d_b):
            continue
        beam = device.beams[b]
        i0 = beam.intensity_at(x)
        i1 = beam.intensity_at(x + dx - d_b)
        if i0 <= 0.0:
            continue
        factor *= math.sqrt(i1 / i0)
    return float(factor)


def free_hamiltonian(
    device: Device,
    space: HilbertSpace,
    *,
    sample: NoiseSample | None = None,
    qubit_shifts_hz: Mapping[int, float] | None = None,
    mode_frequencies_hz: Mapping[int, float] | None = None,
) -> BuiltHamiltonian:
    """H_mot + H_int alone: the idle intervals between pulses (Section 3.4)."""
    return build_hamiltonian(
        device,
        (),
        space,
        sample=sample,
        qubit_shifts_hz=qubit_shifts_hz,
        mode_frequencies_hz=mode_frequencies_hz,
    )


def carrier_debye_waller_frozen(etas: Mapping[int, float], frozen_n: Mapping[int, int]) -> float:
    """prod_{m frozen} e^{-eta_m^2/2} L_{n_m}(eta_m^2): the factor the builder applies for frozen spectators."""
    out = 1.0
    for m, n in frozen_n.items():
        e = etas[m]
        out *= float(math.exp(-(e**2) / 2.0) * eval_genlaguerre(int(n), 0, e**2))
    return out


__all__ = [
    "BuilderOptions",
    "BuiltHamiltonian",
    "CurvatureSpec",
    "DriveRecord",
    "ModeSpec",
    "MultiLevelBuild",
    "MultiLevelOptions",
    "assign_frames",
    "build_hamiltonian",
    "build_multilevel",
    "carrier_debye_waller_frozen",
    "free_hamiltonian",
    "lamb_dicke_parameters",
    "mathieu_or_none",
    "micromotion_index",
]
