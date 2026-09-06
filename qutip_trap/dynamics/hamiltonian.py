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
(Section 5.5). Coefficients are Python functions with the QuTiP 5.3 signature f(t, **kwargs) (never strings, Section 5.2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import qutip as qt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import eval_genlaguerre, jv

from qutip_trap.control.pulses import Drive, Pulse
from qutip_trap.device.model import Device
from qutip_trap.dynamics.frames import interaction_picture
from qutip_trap.dynamics.multilevel import (  # the multi-level mode of Section 4.2.8 (M3a): one builder module
    ModeSpec,
    MultiLevelBuild,
    MultiLevelOptions,
    assign_frames,
    build_multilevel,
)
from qutip_trap.hilbert.operators import debye_waller_factor, qudit_projector, qudit_sigma_plus
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.light.raman import lamb_dicke_parameters, mathieu_or_none
from qutip_trap.noise.sampling import (
    KEY_RABI_SCALE,
    KEY_RF_PHASE,
    NoiseSample,
    key_frozen_n,
    key_mode_offset_hz,
    key_qubit_offset_hz,
    quiet_sample,
)
from qutip_trap.units import TWO_PI

M2 = "milestone M2 (dynamics/hamiltonian.py, PLAN.md Section 4.3.1)"
M3A = "milestone M3a (the multi-level mode of Section 4.2.8, dynamics/multilevel.py)"

Frame = Literal["schrodinger", "interaction"]
MicromotionMode = Literal["none", "carrier_j0", "modulated"]
PhaseMode = Literal["continuous", "reset"]


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


@dataclass
class _DriveCoefficient:
    """The scalar c(t) multiplying sigma_+^i (x) D_i: (1/2) sum_tones Omega(tau) e^{-i(Theta - phi)} times the scalar factors."""

    t_start: float
    tones: list[_ToneFn]
    scale: complex
    """rabi_scale x crosstalk x Debye-Waller x J_0 (the static factors)."""
    modulation_beta: float
    modulation_omega: float
    modulation_delta: float
    k_dot_omega: float
    counter: _Counter

    def __call__(self, t: float) -> complex:
        self.counter.calls += 1
        tau = t - self.t_start
        total = 0.0 + 0.0j
        for tone in self.tones:
            total += 0.5 * tone.envelope(tau) * np.exp(-1j * (tone.beat_phase(t, tau) - tone.phase(tau)))
        c = total * self.scale
        if self.modulation_beta != 0.0:
            c = c * np.exp(
                1j * self.modulation_beta * math.cos(self.modulation_omega * t + self.modulation_delta)
            )
        if self.k_dot_omega != 0.0:
            c = c * np.exp(1j * self.k_dot_omega * t)
        return complex(c)


@dataclass
class _ScalarCoefficient:
    fn: Callable[[float], float]
    t_start: float
    counter: _Counter

    def __call__(self, t: float) -> float:
        self.counter.calls += 1
        return float(self.fn(t - self.t_start))


def _coef_plain(t: float, coef: _DriveCoefficient | _ScalarCoefficient, **_: object) -> complex:
    return complex(coef(t))


def _coef_conj(t: float, coef: _DriveCoefficient, **_: object) -> complex:
    return complex(np.conj(coef(t)))


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

    @property
    def rhs_evaluations(self) -> int:
        """Right-hand-side evaluations so far: coefficient calls over the number of coefficient-bearing elements."""
        n = max(self.n_drive_terms, 1)
        return self.counter.calls // n


# ---- helpers -------------------------------------------------------------------------------------------------------------


def _as_time_function(
    value: Callable[[float], float] | np.ndarray | float, duration_s: float, *, scale: float
) -> Callable[[float], float]:
    """A pulse-local callable of tau from a constant, a callable or a uniformly sampled array over [0, duration]."""
    if callable(value):
        fn = value
        return lambda tau: scale * float(fn(tau))
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("a sampled envelope needs at least two samples")
        grid = np.linspace(0.0, duration_s, arr.size)
        spline = CubicSpline(grid, scale * arr, extrapolate=True)
        return lambda tau: float(spline(min(max(tau, 0.0), duration_s)))
    const = scale * float(value)
    return lambda tau: const


def _beat_phase_function(
    detuning: Callable[[float], float] | float, t_start_s: float, duration_s: float, mode: PhaseMode
) -> Callable[[float, float], float]:
    """Theta(t, tau): mu t (continuous, constant mu), mu tau (reset), or the FM integral int_0^tau mu dtau' (+ mu(0) t_start)."""
    if not callable(detuning):
        mu = TWO_PI * float(detuning)
        if mode == "continuous":
            return lambda t, tau: mu * t
        return lambda t, tau: mu * tau
    mu_fn = detuning
    grid = np.linspace(0.0, duration_s, 4097)
    mu_samples = np.array([TWO_PI * float(mu_fn(x)) for x in grid])
    integral = cumulative_trapezoid(mu_samples, grid, initial=0.0)
    spline = CubicSpline(grid, integral, extrapolate=True)
    offset = mu_samples[0] * t_start_s if mode == "continuous" else 0.0

    def theta(t: float, tau: float) -> float:
        return float(spline(min(max(tau, 0.0), duration_s))) + offset

    return theta


def micromotion_index(device: Device, ion: int, delta_k: np.ndarray) -> float:
    """beta_total of the ion's excess micromotion along delta_k (Section 4.3.6), 0 without an rf record or a field."""
    if float(np.linalg.norm(delta_k)) == 0.0:
        return 0.0
    try:
        return float(device.trap.micromotion_beta(device.crystal.species[ion], delta_k).as_peak().total)
    except ValueError:
        return 0.0


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


def _drive_operator(
    space: HilbertSpace,
    ion: int,
    etas: Mapping[int, float],
    options: BuilderOptions,
    device: Device,
    ion_op: qt.Qobj | None = None,
) -> qt.Qobj:
    """sigma_+^ion (or ``ion_op``) (x) prod_m D_m (exact or expanded) with the optional symmetrized curvature factor."""
    if options.lamb_dicke_order is None:
        op = space.drive_operator(ion, etas, ion_op=ion_op)
    else:
        ops: dict[int, qt.Qobj] = {
            space.ion_factor(ion): qudit_sigma_plus(space.ion_dims[ion]) if ion_op is None else ion_op
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
    return op.to("CSR")


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
        t_start = max(p.t_start_s for p in pulses)
        t_end = min(p.t_end_s for p in pulses)
        starts = {p.t_start_s for p in pulses}
        ends = {p.t_end_s for p in pulses}
        if len(starts) > 1 or len(ends) > 1:
            raise ValueError(
                "build_hamiltonian takes the pulses active on ONE segment; split the schedule at pulse boundaries"
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

    # H_int
    for i in range(space.n_ions):
        delta = smp.get(key_qubit_offset_hz(i), 0.0) + (
            0.0 if qubit_shifts_hz is None else float(qubit_shifts_hz.get(i, 0.0))
        )
        if delta != 0.0:
            static = static + (0.5 * TWO_PI * delta) * space.sigma_z(i)

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

    terms.append(static)
    records: list[DriveRecord] = []
    frozen_states = _frozen_fock_states(space, smp, frozen_n)
    rabi_scale = smp.get(KEY_RABI_SCALE, 1.0)
    dropped_total = 0.0
    n_drive_terms = 0
    duration = t_end - t_start

    for pulse in pulses:
        drive: Drive = pulse.drive
        delta_k = drive.delta_k(device.beams)
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
        if opts.include_crosstalk:
            for j, eps in drive.crosstalk.items():
                if j in drive.ions:
                    raise ValueError("crosstalk targets are neighbours, not the addressed ions")
                targets.append((j, complex(eps)))
        elif drive.crosstalk:
            approximations.append(f"crosstalk of pulse {pulse.gate_id!r} switched off")
        primary = drive.ions[0]
        x_primary = np.asarray(crystal.positions_m[primary], dtype=float)
        tone_fns: list[_ToneFn] = []
        for tone in drive.tones:
            tone_fns.append(
                _ToneFn(
                    envelope=_as_time_function(tone.envelope_hz, duration, scale=TWO_PI),
                    phase=_as_time_function(tone.phase_rad, duration, scale=1.0),
                    beat_phase=_beat_phase_function(
                        tone.detuning_hz, pulse.t_start_s, duration, opts.phase_mode
                    ),
                )
            )
            mu0 = abs(
                TWO_PI * float(tone.detuning_hz(0.0) if callable(tone.detuning_hz) else tone.detuning_hz)
            )
            omega_max = max(omega_max, mu0)
        for ion, eps in targets:
            if space.ion_dims[ion] > 2:
                approximations.append(
                    f"ion {ion} has d = {space.ion_dims[ion]}: the drive couples levels 0 and 1 only ({M3A})"
                )
            etas, c0_applied = lamb_dicke_parameters(device, ion, delta_k)
            if not c0_applied and float(np.linalg.norm(delta_k)) > 0.0:
                approximations.append(
                    f"ion {ion}: no rf record, C0 = 1 and beta = 0 in the Lamb-Dicke parameters"
                )
            beta = micromotion_index(device, ion, delta_k) if opts.micromotion != "none" else 0.0
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
                    dw *= debye_waller_factor(frozen_states[m], etas[m])
            geometric = (
                0.0
                if ion == primary
                else float(np.dot(delta_k, np.asarray(crystal.positions_m[ion]) - x_primary))
            )
            scale = rabi_scale * eps * carrier * dw * np.exp(1j * geometric)
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
                d_ion = space.ion_dims[ion]
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
                term_scale = complex(scale) * weight
                if opts.frame == "schrodinger":
                    op = _drive_operator(space, ion, active_etas, opts, device, ion_op)
                    coef = _DriveCoefficient(
                        pulse.t_start_s,
                        tone_fns,
                        term_scale,
                        beta if opts.micromotion == "modulated" else 0.0,
                        rf_omega,
                        rf_delta,
                        extra_rotation,
                        counter,
                    )
                    terms.append([op, qt.coefficient(_coef_plain, args={"coef": coef})])
                    terms.append([op.dag(), qt.coefficient(_coef_conj, args={"coef": coef})])
                    n_drive_terms += 2
                    continue
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
                for term, term_tones in kept_pairs:
                    k_dot_w = sum(k * omegas[m] for k, m in zip(term.k, term.modes))
                    omega_max = max(omega_max, abs(k_dot_w))
                    coef = _DriveCoefficient(
                        pulse.t_start_s,
                        term_tones,
                        term_scale,
                        0.0,
                        0.0,
                        0.0,
                        k_dot_w + extra_rotation,
                        counter,
                    )
                    terms.append([term.op, qt.coefficient(_coef_plain, args={"coef": coef})])
                    terms.append([term.op.dag(), qt.coefficient(_coef_conj, args={"coef": coef})])
                    n_drive_terms += 2
            if opts.lamb_dicke_order is not None:
                approximations.append(
                    f"ion {ion}: displacement operators expanded to order {opts.lamb_dicke_order} in eta"
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
    )


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
