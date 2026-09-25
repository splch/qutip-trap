"""The one Hamiltonian builder (PLAN.md Section 4.3.1).

H(t)/hbar = H_mot + H_int + sum_i H_drive,i(t) + sum_i H_Stark,i(t) (+ H_anh, + curvature) in rad/s on the joint space,
every term switchable and every switch recorded in ``BuiltHamiltonian.approximations``:

- H_mot = sum_m omega_m a_m^dag a_m over the resolved and ENR modes, omega_m from ``Crystal.modes`` plus the sample's
  mode offsets; frozen spectators enter only through their Debye-Waller factor.
- H_int = sum_i (Delta_i/2) sigma_z^i with the energy sigma_z = |1><1| - |0><0| and Delta_i the transition frequency minus
  the frame frequency (the sample's qubit offset plus explicit shifts such as an ac Zeeman shift).
- H_drive,i(t) = (1/2) sum_tones Omega(t) e^{-i(mu t - phi(t))} sigma_+^i (x) prod_m D_m(i eta_im) + h.c. with the exact
  displacements unless ``lamb_dicke_order`` expands them; crosstalk is the same term on a neighbour j with Omega -> eps_ij
  Omega, j's own etas and the phase Delta k . (X_j - X_i). The coefficient carries J_0(beta) (unlocked drives) or
  e^{i beta cos(Omega_rf t + delta)} (``micromotion="modulated"``), the frozen spectators' e^{-eta^2/2} L_n(eta^2) and the
  sample's Rabi scale; C0 is already inside every eta ``lamb_dicke_parameters`` delivers.
- A ``light_shift`` drive replaces sigma_+^i by the level-weighted projector w_dn P_0 + w_up P_1 (the Zhu-Monroe-Duan
  spin-dependent force plus the spin-independent one) and keeps the same beams' Raman spin flip w_flip Omega_LS
  sigma_+ (x) D unless its off-resonant excitation (w_flip Omega_LS/(omega_0 - mu))^2 is below 1e-12.
- H_Stark,i(t) = (delta_St,i(t)/2) sigma_z^i; H_anh the cubic and quartic mode couplings (opt-in); the beam curvature
  Omega -> Omega (1 + (Omega''/2 Omega) x_hat^2) of Cetina 2022.

Time conventions: the beat-note phase is mu t in absolute time (phase-continuous operation at the nominal qubit frequency)
or mu (t - t_start) under ``phase_mode="reset"``; envelopes, phases and detunings are functions of the time since the pulse
start (a constant is a square pulse, an array a uniform sampling cubic-spline interpolated). Coefficients are picklable
objects with the QuTiP f(t, **kwargs) signature; the plain and conjugate terms of a drive, and every sideband term of the
interaction picture, share one tone-sum evaluation per time, and a drive whose tones are identically zero adds no term.
"""

from __future__ import annotations

import cmath
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import qutip as qt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import jv

from qutip_trap.control.pulses import Drive, Pulse, as_time_function, fingerprint_pulse
from qutip_trap.device.model import Device
from qutip_trap.dynamics.kernels import KernelChoice, prefer_factorized
from qutip_trap.hashing import canonical_digest
from qutip_trap.hilbert.operators import (
    debye_waller_factor,
    qudit_projector,
    qudit_sigma_plus,
    sideband_operators,
)
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.light.comb import comb_build_notes
from qutip_trap.light.raman import lamb_dicke_parameters
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
    """How the builder writes a segment: the frame, the Lamb-Dicke expansion, the sideband RWA and cutoff, the micromotion
    and phase modes, the Stark, anharmonic, crosstalk and Debye-Waller terms, the beam curvature, the gradient-drive form
    and the kernel that holds the drive operators."""

    frame: Frame = "schrodinger"
    lamb_dicke_order: int | None = None
    """None = the exact D(i eta); 0, 1, 2, ... = every displacement Taylor-expanded to that order."""
    rwa: bool = False
    """Interaction frame only: each tone keeps the one sideband combination nearest its resonance (the textbook model)."""
    k_max: int | None = None
    """Interaction frame: keep |k_m| <= k_max per mode and report the dropped weight (Section 5.2)."""
    micromotion: MicromotionMode = "carrier_j0"
    phase_mode: PhaseMode = "continuous"
    include_stark: bool = True
    include_anharmonic: bool = False
    include_crosstalk: bool = True
    frozen_debye_waller: bool = True
    curvature: Mapping[int, CurvatureSpec] = field(default_factory=dict)
    """Per ion: the beam-curvature coupling (Section 6.2); empty = off."""
    gradient_form: GradientForm = "bare"
    """A ``gradient`` drive (Section 4.4.5): ``bare`` builds the two microwave tones as carrier terms beside the laboratory
    sigma_z force w_{i,m} cos(omega_g t + phi_g), so the J_2(4 Omega_mu/delta) weight emerges from the dynamics;
    ``dressed`` builds the adiabatically eliminated force (w_{i,m}/2) J_2(4 Omega_mu/delta) sigma_z^i (a_m e^{i(...)} + h.c.)
    alone and records the elimination."""
    kernel: KernelChoice = "auto"
    """How the drive operators are held: ``assembled`` CSR, ``factorized`` per-mode factors applied
    matrix-free, ``auto`` by the cost model of ``dynamics.kernels``. The same operator either way; the factorized form needs
    a product space in the Schroedinger frame with exact displacements and no curvature factor, and the engine assembles on
    every ``mesolve`` segment."""

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
    """What the builder derived for one (pulse, ion) pair (the Section 5.7 diagnostics)."""

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


@dataclass
class _ToneFn:
    envelope: Callable[[float], float]
    """Omega(tau) in rad/s, tau the time since the pulse start."""
    phase: Callable[[float], float]
    """phi(tau) in rad."""
    beat_phase: Callable[[float, float], float]
    """Theta(t, tau): mu t (continuous), mu tau (reset) or the integral of an FM detuning."""
    constants: tuple[float, float, float] | None = None
    """(Omega in rad/s, phi in rad, mu in rad/s) of a square tone with constant detuning, evaluated without the calls."""
    continuous: bool = True
    """The beat phase is mu t (phase_mode continuous) rather than mu tau (reset)."""


@dataclass(eq=False)
class _DriveCoefficient:
    """The scalar c(t) on sigma_+^i (x) D_i: (1/2) sum_tones Omega(tau) e^{-i(Theta - phi)} times the scalar factors.

    Compared by identity: ``QobjEvo`` merges elements whose coefficients compare equal, and two drives are two terms.
    """

    t_start: float
    tones: list[_ToneFn]
    scale: complex
    """rabi_scale x crosstalk x Debye-Waller x J_0 (the static factors)."""
    modulation_beta: float
    modulation_omega: float
    modulation_delta: float
    k_dot_omega: float
    phase_trajectory: Callable[[float], float] | None = None
    """phi_L(t) of the laser (rad), added to every tone's phase (single-photon optical drives, Section 6.3)."""
    amplitude_trajectory: Callable[[float], float] | None = None
    """dI/I(t) of the light; the envelope is multiplied by (1 + dI/I)^amplitude_power (Section 6.4)."""
    amplitude_power: float = 1.0
    _t_last: float = field(default=math.nan, init=False, repr=False)
    _c_last: complex = field(default=0j, init=False, repr=False)
    """The last (t, c(t)): every element of the QobjEvo is evaluated at the same t, so the terms of one drive share it."""

    def __call__(self, t: float) -> complex:
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

    def __call__(self, t: float) -> float:
        return float(self.fn(t - self.t_start))


@dataclass(eq=False)
class _GradientCoefficient:
    """f(t) on sigma_z^i (x) a_m of a near-field microwave-gradient drive (Section 4.4.5): the laboratory force
    w_{i,m} cos(omega_g t + phi_g) (``dressed=False``), or the adiabatically eliminated co-rotating term
    (w_{i,m}/2) J_2(4 Omega_mu/delta) e^{i(rotation t + phi)} (``dressed=True``)."""

    weight: float
    omega_g: float
    phase: float
    t_start: float
    rotation: float = 0.0
    dressed: bool = False
    continuous: bool = True

    def __call__(self, t: float) -> complex:
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
    """A sampled noise trajectory as a coefficient: scale x traj(t) on the sample's fixed grid (Section 6.1 route d)."""
    return float(scale * traj(t))


def _coef_conj(
    t: float, coef: _DriveCoefficient | _RotatedCoefficient | _GradientCoefficient, **_: object
) -> complex:
    return complex(coef(t)).conjugate()


_KernelLabel = Literal["assembled", "factorized", "mixed", "none"]


def _kernel_label(kinds: Iterable[str]) -> _KernelLabel:
    """``factorized`` or ``assembled`` when every drive term is held that way, ``mixed`` when both occur, ``none`` for none."""
    held = {k for k in kinds if k != "none"}
    if not held:
        return "none"
    if held == {"assembled"}:
        return "assembled"
    if held == {"factorized"}:
        return "factorized"
    return "mixed"


@dataclass(frozen=True)
class BuiltHamiltonian:
    """The QobjEvo of one segment and how it was built (Section 5.7)."""

    H: qt.QobjEvo
    space: HilbertSpace
    t_start_s: float
    t_end_s: float
    frame: Frame
    omega_max_rad_s: float
    """The highest mode or beat-note frequency in the frame (rad/s): the ladder's max_step is a fraction of its period."""
    n_drive_terms: int
    approximations: tuple[str, ...]
    records: tuple[DriveRecord, ...]
    dropped_weight: float
    mode_frequencies_rad_s: dict[int, float]
    drive_parts: dict[str, qt.QobjEvo] = field(default_factory=dict)
    """Per pulse (gate_id), the QobjEvo of its drive terms alone: what the intensity-noise channel sqrt(D) H_drive(t)
    multiplies."""
    kernel: _KernelLabel = "none"
    """How the drive operators are held: ``factorized``, ``assembled``, ``mixed`` or ``none`` (no drive term)."""
    fingerprint: str = ""
    """A digest of everything that determined H(t): the key of the engine's propagator cache."""


# ---- helpers -------------------------------------------------------------------------------------------------------------


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

    beta = hypot(in_phase, out_of_phase) >= 0 and the offset -atan2(out_of_phase, in_phase) is what the modulated
    coefficient e^{i beta cos(Omega_rf t + delta)} adds to the drive's rf phase to carry both quadratures (a signed in-phase
    index steps by pi across a compensating shim); J_0 is even, so ``carrier_j0`` sees only beta. (0, 0) for a drive with no
    wavevector or a trap with no rf record; a trap whose ``micromotion_beta`` cannot be evaluated raises.
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
    """Whether this drive term is held factorized: the option, the structural conditions (exact
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
    """What the builder needs of a ``gradient`` drive (Section 4.4.5): the per-(ion, mode) force coefficient, the
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
    it is held factorized (the matrix-free kernel) or assembled (CSR)."""
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
    """The frozen modes' Fock states: ``frozen_n``, else the sample's keys, else 0."""
    given = frozen_n or {}
    return {
        m: int(given[m]) if m in given else int(round(sample.values.get(key_frozen_n(m), 0.0)))
        for m in space.frozen
    }


# ---- the interaction picture ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SidebandTerm:
    """One sideband combination of the interaction picture: sigma_+ (x) A_k, its k-vector over ``modes`` and its weight
    (the product of the per-mode spectral norms)."""

    op: qt.Qobj
    k: tuple[int, ...]
    modes: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class InteractionPicture:
    terms: tuple[SidebandTerm, ...]
    dropped_weight: float
    """sum over the dropped combinations of their weights (Section 5.2)."""


def interaction_picture(
    space: HilbertSpace,
    ion: int,
    etas: Mapping[int, float],
    *,
    k_max: int | None = None,
    matrices: Mapping[int, np.ndarray] | None = None,
    ion_op: qt.Qobj | None = None,
) -> InteractionPicture:
    """sigma_+^ion (x) prod_m D_m(i eta_m) over the resolved modes decomposed into the sideband operators A_k (n' - n = k per
    mode), the picture of H_0 = sum_m omega_m a_m^dag a_m in which each term rotates at e^{i k . omega t} (Section 5.2).

    ``k_max`` drops combinations with any |k_m| > k_max and reports their weight; ``matrices`` supplies per-mode matrices
    (default: the space's exponentials); ``ion_op`` (embedded) replaces sigma_+. Product spaces only.
    """
    if space.enr_group is not None and any(space.mode_class(m) == "enr" for m in etas if etas[m] != 0.0):
        raise NotImplementedError("the interaction picture is defined on product spaces only (Section 5.1.1)")
    modes = [m.mode for m in space.resolved]
    per_mode: list[dict[int, np.ndarray]] = []
    norms: list[dict[int, float]] = []
    for m in modes:
        eta = float(etas.get(m, 0.0))
        d = space.truncation(m).d
        if eta == 0.0:
            parts = {0: np.eye(d, dtype=complex)}
        else:
            mat = (
                matrices[m]
                if matrices is not None and m in matrices
                else space.displacement_factor(m, eta).full()
            )
            parts = sideband_operators(np.asarray(mat))
        per_mode.append(parts)
        norms.append({k: float(np.linalg.norm(v, 2)) for k, v in parts.items()})
    kept: list[SidebandTerm] = []
    dropped = 0.0
    sp = space.sigma_plus(ion) if ion_op is None else ion_op
    for combo in itertools.product(*[sorted(p) for p in per_mode]):
        weight = math.prod(norms[i][k] for i, k in enumerate(combo)) if combo else 1.0
        if k_max is not None and any(abs(k) > k_max for k in combo):
            dropped += weight
            continue
        ops = {
            space.mode_factor(m): qt.Qobj(per_mode[i][combo[i]], dims=[[space.truncation(m).d]] * 2)
            for i, m in enumerate(modes)
        }
        op = space.embed_many(ops) * sp
        kept.append(SidebandTerm(op=op.to("CSR"), k=tuple(combo), modes=tuple(modes), weight=weight))
    return InteractionPicture(terms=tuple(kept), dropped_weight=dropped)


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
    crystal = device.crystal
    n_modes = len(crystal.modes)

    if pulses:
        # the segment is the interval every pulse covers; each pulse keeps its own clock (tau since its own start)
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

    # H_int (the ions are device indices: a GATE_LOCAL space carries a subset of the crystal)
    for i in space.ion_labels:
        delta = smp.get(key_qubit_offset_hz(i), 0.0) + (
            0.0 if qubit_shifts_hz is None else float(qubit_shifts_hz.get(i, 0.0))
        )
        if delta != 0.0:
            static = static + (0.5 * TWO_PI * delta) * space.sigma_z(i)
    # the sampled transition offsets (Section 6.3): (delta nu_i(t)/2) sigma_z^i on the sample's fixed grid
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
    # the resonance checker and the estimated accumulated phase run whatever include_anharmonic says (Section 5.7)
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
    kernels: list[str] = []

    for pulse in pulses:
        drive: Drive = pulse.drive
        duration = pulse.duration_s
        gradient = _gradient_terms(device, drive, opts) if drive.kind == "gradient" else None
        delta_k = drive.delta_k(device.beams)
        part_key = pulse.gate_id or f"pulse@{pulse.t_start_s:.9g}"
        part = drive_parts.setdefault(part_key, [])
        # quasi-static beam-path phase Delta phi of the beat note E ~ cos(omega_L t - Delta k . r + Delta phi): phi_2 - phi_1
        # for a Raman pair with Delta k = k_1 - k_2, -phi_1 for a single beam; the factor on sigma_+ is
        # e^{+i(Delta k . X - Delta phi)} (Section 13, "Optical phase factor on sigma_+")
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
            # the Section 4.3.7 validity clauses the pulse alone decides (the rest are checked by light.raman.comb_drive)
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
        # the sampled beam-path phase Delta phi(t) (Section 6.3 route (d)): the static factor carries e^{-i Delta phi} and
        # _DriveCoefficient adds its phase trajectory as e^{+i phi(t)}, so it enters as -Delta phi(t)
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
        # a drive whose every tone is the constant 0 (one beam of a pair on, the Stark scan of Section 7.5) adds no drive
        # term, so the segment's Hamiltonian stays constant; its Stark term is kept
        silent = all(
            not callable(tone.envelope_hz)
            and not isinstance(tone.envelope_hz, np.ndarray)
            and float(tone.envelope_hz) == 0.0
            for tone in drive.tones
        )
        if gradient is not None and gradient.dressed:
            # the dressed form replaces the two microwave tones by their J_2 weight on the force: no carrier terms
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
                    envelope=as_time_function(tone.envelope_hz, duration, scale=TWO_PI),
                    phase=as_time_function(tone.phase_rad, duration, scale=1.0),
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
                    f"ion {ion} has d = {space.ion_dim(ion)}: the drive couples levels 0 and 1 only"
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
                    # a dropped mode is not modelled at all (Section 5.2), so it has no Debye-Waller factor
                    if m in space.dropped:
                        continue
                    dw *= debye_waller_factor(frozen_states[m], etas[m])
            # the optical phase carries the relative ion-position drift u_0 = Q E_dc/(m omega^2) of a stray field
            # (Berkeland 1998 Eq. 16) besides the geometry; the primary's own drift is absorbed by the frame alignment,
            # so only Delta k . (u_0(j) - u_0(i)) survives
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
                    kernels.append("factorized" if factorized else "assembled")
                    coef = _DriveCoefficient(
                        pulse.t_start_s,
                        tone_fns,
                        term_scale,
                        beta if opts.micromotion == "modulated" else 0.0,
                        rf_omega,
                        # the index's quadrature offset on top of the drive's rf phase: beta_ip cos + beta_op sin
                        rf_delta + beta_phase,
                        extra_rotation + laser_rotation,
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
                kernels.append("assembled")
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
        # the near-field microwave-gradient force (Section 4.4.5): sum_{i,m} w_{i,m} sigma_z^i (a_m + a_m^dag)
        # cos(omega_g t + phi_g), w_{i,m} from ``light.microwave.derive_gradient_drive``
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
                        rotation=gradient.rotation,
                        dressed=gradient.dressed,
                        continuous=opts.phase_mode == "continuous",
                    )
                    t_plain = [op, qt.coefficient(_coef_plain, args={"coef": grad_coef})]
                    t_conj = [op.dag(), qt.coefficient(_coef_conj, args={"coef": grad_coef})]
                    terms.extend([t_plain, t_conj])
                    part.extend([t_plain, t_conj])
                    n_drive_terms += 2
                    kernels.append("assembled")
                for m in space.frozen:
                    if gradient.weights.get((ion, m), 0.0) != 0.0:
                        approximations.append(
                            f"pulse {pulse.gate_id!r}, ion {ion}: gradient force on frozen mode {m} dropped"
                        )
        # H_Stark
        if opts.include_stark:
            st = drive.stark_shift_hz
            if callable(st):
                fn = as_time_function(st, duration, scale=0.5 * TWO_PI)
                sz = 0.0 * space.identity()
                for ion, eps in targets:
                    sz = sz + (abs(eps) ** 2) * space.sigma_z(ion)
                sc = _ScalarCoefficient(fn, pulse.t_start_s)
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
    # compress=False: every drive term carries its own coefficient object, so QuTiP's merge pass could only compare the
    # drive operators pairwise and never merge them
    H = qt.QobjEvo(terms, compress=False)
    kernel = _kernel_label(kernels)
    if kernel == "mixed":
        approximations.append(
            "drive terms held partly factorized and partly assembled (an ENR-coupled or curvature drive beside a product-space one)"
        )
    fingerprint = canonical_digest(
        (
            "build_hamiltonian",
            # the device enters through every eta, the beam profiles and the geometric phase, which nothing else here
            # carries
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
        drive_parts={k: qt.QobjEvo(v, compress=False) for k, v in drive_parts.items() if v},
        kernel=kernel,
        fingerprint=fingerprint,
    )


def _position_offset(ion: int, sample: NoiseSample) -> np.ndarray:
    """The sampled quasi-static displacement u_0 of ``ion`` (m, laboratory axes), Q E_dc/(m omega^2) under a stray field
    (Berkeland 1998 Eq. 16): it enters the optical phase as Delta k . u_0 and the intensity through ``_pointing_factor``."""
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
