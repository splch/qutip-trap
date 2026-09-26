"""Closed-form gate integrals and the AM/FM/PM/Fourier pulse solvers for multi-mode Molmer-Sorensen closure (PLAN.md
Section 4.4.3).

Conventions: Omega is the per-tone Rabi frequency of the (hbar Omega/2) convention; the bichromatic tones sit at -/+ mu from the carrier with phases
phi_s -/+ phi_m (Haljan's spin phase phi_s = (phi_b + phi_r)/2 and motion phase phi_m = (phi_b - phi_r)/2). Expanding the
builder's drive (hbar/2) sum_tones Omega e^{-i(mu t - phi)} sigma_+ D(i eta) + h.c. to first order in eta in the interaction
picture gives the spin-dependent force

    H/hbar = -sum_i sum_m (eta_im Omega_i(t)/2) sigma^i_{phi_s + pi/2} (a_m^dag e^{i psi_m(t)} + a_m e^{-i psi_m(t)}),
    psi_m(t) = omega_m t - Theta(t) + phi_m,   Theta(t) = int_0^t mu(t') dt',

whose Magnus series terminates: U = exp[sum_i sigma^i sum_m (alpha_im a_m^dag - alpha_im^* a_m)] exp[i sum_{i<j} chi_ij sigma^i
sigma^j] (a global phase aside) with

    alpha_im(tau) = i eta_im int_0^tau Omega_i(t) f_m(t) dt,
    chi_ij(tau)   = sum_m eta_im eta_jm int_0^tau dt' int_0^{t'} dt [Omega_i(t) Omega_j(t') + Omega_j(t) Omega_i(t')] g_m(t, t'),
    g_m(t, t')    = Im[f_m(t') conj(f_m(t))].

The force is written with the MINUS of -(hbar eta Omega/2) S_phi, the sign that makes
alpha = (eta Omega/2 eps)(e^{i eps t} - 1). The builder's own sign is the opposite (+i eta from the i of
i eta(a + a^dag), hence ``schedule.FORCE_AXIS_OFFSET_RAD = pi/2``), so a comparison against the exact propagator negates
alpha; everything downstream reads |alpha| only.

Two kernels f_m, never mixed: the exact first-order Lamb-Dicke form f_m = cos(Theta(t) - phi_m) e^{i omega_m t},
which keeps the counter-rotating omega_m + mu term (Choi 2014 Eq. 3 is its phi_m = pi/2 case, sin(Theta) e^{i omega t}, the
sine beat-note convention with the force zero at the pulse start), and the slow-envelope ("rwa") form f_m = (e^{i phi_m}/2)
e^{i(omega_m t - Theta(t))}, its resonant part, with g_m = (1/4) sin[psi_m(t') - psi_m(t)] (Leung 2018). The pair sum runs
over i < j, so the coefficient of sigma^a sigma^b is chi_ab itself and the kernel is SYMMETRIZED in the two envelopes
(Choi's printed 2 Omega_i(t) Omega_j(t') holds only for proportional envelopes). The motion phase is a convention for the
closed forms but not for the exact dynamics: the carrier Omega cos(mu t - phi_m) sigma_phi_s tilts the force axis out of
the equatorial plane by psi = (2 Omega/mu) sin phi_m (Roos 2008), so the waveforms default to phi_m = 0 (equal tone
phases), where the tilt vanishes. For a square pulse with (omega_m - mu) tau = 2 pi K the rwa kernel gives
chi = pi K (eta Omega/eps)^2, so eta Omega/eps = 1/(2 sqrt K) is the maximally entangling closure chi = pi/4. The sign of
chi is set by the detuning side, exp(+i chi sigma sigma) = XX(-chi), and the scheduler folds it into the gate phases.
Blumel 2021 write the kernel without the pair sum and target pi/8: double their kernel to convert.

The solvers act on the mode structure alone; the exact verification of a solution is ``qutip_trap.calibration.entangling``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.integrate import cumulative_simpson, simpson
from scipy.optimize import least_squares

from qutip_trap.control.pulses import ConstantFn, ScaledFn
from qutip_trap.control.table import CalEntry, Leg, Segment, Waveform
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

Kernel = Literal["rwa", "choi"]
CHI_MAXIMAL_RAD = math.pi / 4.0
"""|chi| of a maximally entangling XX(chi) = exp(-i chi sigma_x sigma_x)."""
SINE_MOTION_PHASE_RAD = math.pi / 2.0
"""phi_m = pi/2: the tone phases differ by pi, so the force is Omega sin(Theta) (Choi's sine beat-note convention)."""
DEFAULT_MOTION_PHASE_RAD = 0.0
"""phi_m = 0: equal tone phases, the force Omega cos(Theta) with no mean carrier rotation (the played default)."""


class ClosureError(ValueError):
    """The solver could not close every mode with the requested degrees of freedom, or the target angle is unreachable."""


# ---- the mode structure a gate sees -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateModes:
    """omega_m, eta_im and nbar_m of the modes an entangling pulse on ``ions`` couples to, in one fixed order."""

    ions: tuple[int, ...]
    modes: tuple[int, ...]
    """Crystal mode indices (positions in Crystal.modes)."""
    omega_rad_s: tuple[float, ...]
    eta: dict[int, tuple[float, ...]]
    """ion -> eta_{ion, m} in the order of ``modes`` (C0 inside)."""
    nbar: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.modes)
        if len(self.omega_rad_s) != n or len(self.nbar) != n:
            raise ValueError("omega_rad_s and nbar carry one entry per mode")
        if len(self.modes) != len(set(self.modes)):
            raise ValueError("modes are distinct")
        for ion in self.ions:
            if ion not in self.eta or len(self.eta[ion]) != n:
                raise ValueError(f"eta of ion {ion} must carry one entry per mode")
        if any(w <= 0.0 for w in self.omega_rad_s):
            raise ValueError("mode frequencies are positive")

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def index(self, mode: int) -> int:
        return self.modes.index(mode)

    def eta_of(self, ion: int, mode: int) -> float:
        return self.eta[ion][self.index(mode)]

    def pairs(self) -> list[tuple[int, int]]:
        return [(a, b) for k, a in enumerate(self.ions) for b in self.ions[k + 1 :]]

    def subset(self, modes: Sequence[int]) -> GateModes:
        idx = [self.index(m) for m in modes]
        return GateModes(
            self.ions,
            tuple(self.modes[i] for i in idx),
            tuple(self.omega_rad_s[i] for i in idx),
            {ion: tuple(self.eta[ion][i] for i in idx) for ion in self.ions},
            tuple(self.nbar[i] for i in idx),
        )


def gate_modes(
    device: Device,
    ions: Sequence[int],
    beams: tuple[int, int],
    *,
    nbar: Mapping[int, float] | None = None,
    modes: Sequence[int] | None = None,
    eta_min: float = 0.0,
    mode_frequencies_hz: Mapping[int, float] | None = None,
) -> GateModes:
    """The modes the Raman pair (beam 1, beam 2) couples to on ``ions``: every crystal mode with max_i |eta_im| > ``eta_min``
    (default: every mode with any coupling), or the given ``modes``; C0 from the trap's Mathieu record.
    ``mode_frequencies_hz``: believed (calibrated) frequencies used instead of the crystal's."""
    from qutip_trap.light.raman import lamb_dicke_parameters

    b1, b2 = beams
    delta_k = np.asarray(device.beams[b1].k_vector() - device.beams[b2].k_vector(), dtype=float)
    etas = {int(i): lamb_dicke_parameters(device, int(i), delta_k)[0] for i in ions}
    n_modes = len(device.crystal.modes)
    if modes is None:
        chosen = [m for m in range(n_modes) if max(abs(etas[i][m]) for i in etas) > eta_min]
    else:
        chosen = [int(m) for m in modes]
    if not chosen:
        raise ValueError("the beam pair couples the ions to no mode (Delta k = 0 or eta_min too high)")
    freqs = []
    for m in chosen:
        f = device.crystal.modes[m].omega_hz
        if mode_frequencies_hz is not None and m in mode_frequencies_hz:
            f = float(mode_frequencies_hz[m])
        freqs.append(TWO_PI * f)
    nb = dict(nbar or {})
    return GateModes(
        ions=tuple(int(i) for i in ions),
        modes=tuple(chosen),
        omega_rad_s=tuple(freqs),
        eta={i: tuple(etas[i][m] for m in chosen) for i in etas},
        nbar=tuple(float(nb.get(m, 0.0)) for m in chosen),
    )


# ---- envelopes ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentedEnvelope:
    """Piecewise-constant per-ion amplitudes at ONE beat note (the AM family): the integrals are analytic."""

    durations_s: tuple[float, ...]
    amplitude_rad_s: dict[int, tuple[float, ...]]
    """ion -> signed Omega per segment (a negative value is a pi phase on both tones, i.e. a reversed force)."""
    mu_rad_s: float
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD
    """The motion phase (phi_b - phi_r)/2: the force is Omega cos(mu t - phi_m)."""
    phase_rad: tuple[float, ...] | None = None
    """Per-segment motion-phase OFFSET on top of ``phi_m_rad`` (the PM family): the force of segment k is
    Omega_k cos(mu t - phi_m - phi_k), so the segment enters every integral through the complex weight
    z_k = Omega_k e^{i phi_k}. None = every offset zero (the AM family, real weights)."""

    def __post_init__(self) -> None:
        if not self.durations_s or any(d <= 0.0 for d in self.durations_s):
            raise ValueError("segment durations are positive")
        for ion, amps in self.amplitude_rad_s.items():
            if len(amps) != len(self.durations_s):
                raise ValueError(f"ion {ion}: one amplitude per segment")
        if self.phase_rad is not None and len(self.phase_rad) != len(self.durations_s):
            raise ValueError("one motion-phase offset per segment")

    @property
    def weights(self) -> dict[int, np.ndarray]:
        """z_{i,k} = Omega_{i,k} e^{i phi_k}: the complex per-segment weight every integral sees."""
        phase = (
            np.ones(len(self.durations_s), dtype=complex)
            if self.phase_rad is None
            else np.exp(1j * np.asarray(self.phase_rad, dtype=float))
        )
        return {ion: np.asarray(a, dtype=float) * phase for ion, a in self.amplitude_rad_s.items()}

    @property
    def duration_s(self) -> float:
        return float(sum(self.durations_s))

    @property
    def edges_s(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.durations_s)])

    def sampled(self, n: int = 20001) -> SampledEnvelope:
        """The same pulse on a uniform grid (for cross-checking the analytic path)."""
        t = np.linspace(0.0, self.duration_s, n)
        edges = self.edges_s
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(self.durations_s) - 1)
        amps = {ion: np.asarray(a, dtype=float)[idx] for ion, a in self.amplitude_rad_s.items()}
        phi_m: float | np.ndarray = self.phi_m_rad
        if self.phase_rad is not None:
            phi_m = self.phi_m_rad + np.asarray(self.phase_rad, dtype=float)[idx]
        return SampledEnvelope(t, amps, self.mu_rad_s * t, phi_m)


@dataclass(frozen=True)
class SampledEnvelope:
    """Per-ion amplitudes Omega_i(t) and the accumulated beat phase Theta(t) on a uniform grid (FM, Fourier AM, or anything)."""

    times_s: np.ndarray
    amplitude_rad_s: dict[int, np.ndarray]
    beat_phase_rad: np.ndarray
    phi_m_rad: float | np.ndarray = DEFAULT_MOTION_PHASE_RAD
    """The motion phase, a constant or sampled on the grid (a phase-modulated pulse)."""

    def __post_init__(self) -> None:
        t = np.asarray(self.times_s, dtype=float)
        if t.ndim != 1 or t.size < 3 or t.size % 2 == 0:
            raise ValueError("the grid has an odd number (>= 3) of points for Simpson's rule")
        if not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-9, atol=0.0):
            raise ValueError("the grid is uniform")
        if np.asarray(self.beat_phase_rad).shape != t.shape:
            raise ValueError("beat_phase_rad is sampled on the grid")
        if isinstance(self.phi_m_rad, np.ndarray) and self.phi_m_rad.shape != t.shape:
            raise ValueError("a sampled motion phase is sampled on the grid")
        for ion, a in self.amplitude_rad_s.items():
            if np.asarray(a).shape != t.shape:
                raise ValueError(f"ion {ion}: the amplitude is sampled on the grid")

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1] - self.times_s[0])


Envelope = SegmentedEnvelope | SampledEnvelope
"""A played waveform's envelope as the closed forms read it: piecewise constant (``SegmentedEnvelope``) or sampled by the
hardware chain (``SampledEnvelope``); ``envelope_of`` builds one from a ``Waveform``."""


# ---- the kernels -----------------------------------------------------------------------------------------------------------------


def _exp_integral(nu: float, a: float, b: float) -> complex:
    """int_a^b e^{i nu t} dt."""
    if nu == 0.0:
        return complex(b - a)
    return complex((np.exp(1j * nu * b) - np.exp(1j * nu * a)) / (1j * nu))


def _exp_sum(
    kernel: Kernel, omega: float, mu: float, phi_m: float = DEFAULT_MOTION_PHASE_RAD
) -> tuple[tuple[complex, ...], tuple[float, ...]]:
    """f_m(t) = sum_s c_s e^{i nu_s t} for a constant beat note: cos(mu t - phi_m) e^{i omega t} (exact first order; Choi's
    sin(mu t) e^{i omega t} at phi_m = pi/2) or the rwa (e^{i phi_m}/2) e^{i(omega - mu) t}."""
    if kernel == "choi":
        return ((0.5 * np.exp(-1j * phi_m), 0.5 * np.exp(1j * phi_m)), (omega + mu, omega - mu))
    if kernel == "rwa":
        return ((0.5 * np.exp(1j * phi_m),), (omega - mu,))
    raise ValueError("kernel is 'rwa' or 'choi'")


def _segment_f(cs: tuple[complex, ...], nus: tuple[float, ...], a: float, b: float) -> complex:
    return complex(sum(c * _exp_integral(nu, a, b) for c, nu in zip(cs, nus)))


def _segment_triangle(cs: tuple[complex, ...], nus: tuple[float, ...], a: float, b: float) -> float:
    """int_a^b dt' int_a^{t'} dt Im[f(t') conj f(t)] for f = sum_s c_s e^{i nu_s t}, in closed form."""
    total = 0.0 + 0.0j
    for c_s, nu_s in zip(cs, nus):
        if nu_s == 0.0:
            raise ClosureError("a tone exactly on a mode (omega_m = mu) has no closed-form geometric phase")
        inner = (np.exp(-1j * nu_s * a),)
        for c_t, nu_t in zip(cs, nus):
            total += (
                c_t
                * np.conj(c_s)
                / (-1j * nu_s)
                * (_exp_integral(nu_t - nu_s, a, b) - inner[0] * _exp_integral(nu_t, a, b))
            )
    return float(np.imag(total))


@dataclass(frozen=True)
class GateIntegrals:
    """The integrals of one pulse: alpha per (ion, mode), chi per pair, chi per (pair, mode)."""

    alpha: dict[tuple[int, int], complex]
    chi: dict[tuple[int, int], float]
    chi_by_mode: dict[tuple[int, int, int], float]
    kernel: Kernel

    def chi_of(self, a: int, b: int) -> float:
        return self.chi[(a, b)] if (a, b) in self.chi else self.chi[(b, a)]

    def residual_error(self, modes: GateModes, ions: Sequence[int] | None = None) -> float:
        """epsilon_ent = sum_{i,m} |alpha_im|^2 (2 nbar_m + 1), the entanglement infidelity from open loops."""
        ions_ = tuple(ions) if ions is not None else modes.ions
        return float(
            sum(
                abs(self.alpha[(i, m)]) ** 2 * (2.0 * modes.nbar[modes.index(m)] + 1.0)
                for i in ions_
                for m in modes.modes
            )
        )

    def alpha_of(self, ion: int) -> dict[int, complex]:
        return {m: a for (i, m), a in self.alpha.items() if i == ion}

    @property
    def chi_by_mode_pairs(self) -> set[tuple[int, int]]:
        """The (a, b) keys ``chi_by_mode`` carries, in the order the integrals were built with."""
        return {(a, b) for a, b, _m in self.chi_by_mode}


def _segment_matrices(
    env: SegmentedEnvelope, modes: GateModes, kernel: Kernel
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per mode: the vector F_k = int_seg_k f_m and the symmetric matrix G with G_kk = 2 T_k, G_kl = Im(F_l conj F_k) (k < l)."""
    edges = env.edges_s
    n = len(env.durations_s)
    fs: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    for omega in modes.omega_rad_s:
        cs, nus = _exp_sum(kernel, omega, env.mu_rad_s, env.phi_m_rad)
        f = np.array([_segment_f(cs, nus, edges[k], edges[k + 1]) for k in range(n)], dtype=complex)
        g = np.zeros((n, n))
        for k in range(n):
            g[k, k] = 2.0 * _segment_triangle(cs, nus, edges[k], edges[k + 1])
            for j in range(k + 1, n):
                g[k, j] = g[j, k] = float(np.imag(f[j] * np.conj(f[k])))
        fs.append(f)
        gs.append(g)
    return fs, gs


def integrals_segmented(env: SegmentedEnvelope, modes: GateModes, kernel: Kernel = "choi") -> GateIntegrals:
    """alpha and chi of a piecewise-constant pulse at one beat note, analytically (the AM and PM families).

    Each segment enters through its complex weight z_{i,k} = Omega_{i,k} e^{i phi_k} (``SegmentedEnvelope.weights``):
    alpha_{i,m} = i eta_{i,m} sum_k z_{i,k} F_k^{(m)} and, since the two ions share the segment's motion phase, the
    symmetrized double integral is
    chi_m = eta_a eta_b [sum_k Omega_a^k Omega_b^k (2 T_k) + sum_{k<l} Im((conj(z_a^k) z_b^l + conj(z_b^k) z_a^l) F_l conj(F_k))],
    which reduces to the real quadratic form amps_a^T G amps_b when every offset is zero (the AM family)."""
    fs, gs = _segment_matrices(env, modes, kernel)
    alpha: dict[tuple[int, int], complex] = {}
    chi: dict[tuple[int, int], float] = {}
    by_mode: dict[tuple[int, int, int], float] = {}
    amps = {ion: np.asarray(env.amplitude_rad_s[ion], dtype=float) for ion in modes.ions}
    zs = env.weights
    modulated = env.phase_rad is not None and any(p != 0.0 for p in env.phase_rad)
    n_seg = len(env.durations_s)
    for ion in modes.ions:
        for k, m in enumerate(modes.modes):
            alpha[(ion, m)] = complex(1j * modes.eta[ion][k] * np.dot(zs[ion], fs[k]))
    for a, b in modes.pairs():
        total = 0.0
        for k, m in enumerate(modes.modes):
            if modulated:
                za, zb, f = zs[a], zs[b], fs[k]
                val = float(np.sum(amps[a] * amps[b] * np.diag(gs[k])))
                for j in range(n_seg):
                    for i in range(j):
                        val += float(
                            np.imag((np.conj(za[i]) * zb[j] + np.conj(zb[i]) * za[j]) * f[j] * np.conj(f[i]))
                        )
                val *= modes.eta[a][k] * modes.eta[b][k]
            else:
                val = float(modes.eta[a][k] * modes.eta[b][k] * (amps[a] @ gs[k] @ amps[b]))
            by_mode[(a, b, m)] = val
            total += val
        chi[(a, b)] = total
    return GateIntegrals(alpha, chi, by_mode, kernel)


def _kernel_samples(
    kernel: Kernel,
    omega: float,
    t: np.ndarray,
    theta: np.ndarray,
    phi_m: float | np.ndarray = DEFAULT_MOTION_PHASE_RAD,
) -> np.ndarray:
    if kernel == "choi":
        return np.asarray(np.cos(theta - phi_m) * np.exp(1j * omega * t), dtype=complex)
    if kernel == "rwa":
        return np.asarray(0.5 * np.exp(1j * phi_m) * np.exp(1j * (omega * t - theta)), dtype=complex)
    raise ValueError("kernel is 'rwa' or 'choi'")


def integrals_sampled(env: SampledEnvelope, modes: GateModes, kernel: Kernel = "rwa") -> GateIntegrals:
    """alpha and chi of a sampled pulse by Simpson's rule: the outer integral over the cumulative inner one (FM, Fourier AM)."""
    t = np.asarray(env.times_s, dtype=float)
    theta = np.asarray(env.beat_phase_rad, dtype=float)
    alpha: dict[tuple[int, int], complex] = {}
    chi: dict[tuple[int, int], float] = {a: 0.0 for a in modes.pairs()}
    by_mode: dict[tuple[int, int, int], float] = {}
    for k, m in enumerate(modes.modes):
        f = _kernel_samples(kernel, modes.omega_rad_s[k], t, theta, env.phi_m_rad)
        cum = {
            ion: np.asarray(cumulative_simpson(env.amplitude_rad_s[ion] * f, x=t, initial=0.0), dtype=complex)
            for ion in modes.ions
        }
        for ion in modes.ions:
            alpha[(ion, m)] = complex(1j * modes.eta[ion][k] * cum[ion][-1])
        for a, b in modes.pairs():
            integrand = np.imag(
                env.amplitude_rad_s[b] * f * np.conj(cum[a]) + env.amplitude_rad_s[a] * f * np.conj(cum[b])
            )
            val = float(modes.eta[a][k] * modes.eta[b][k] * simpson(integrand, x=t))
            by_mode[(a, b, m)] = val
            chi[(a, b)] += val
    return GateIntegrals(alpha, chi, by_mode, kernel)


def integrals(env: Envelope, modes: GateModes, kernel: Kernel | None = None) -> GateIntegrals:
    """The integrals of any envelope; default kernel: Choi's for segmented pulses, rwa for sampled ones."""
    if isinstance(env, SegmentedEnvelope):
        return integrals_segmented(env, modes, kernel or "choi")
    return integrals_sampled(env, modes, kernel or "rwa")


def trajectory_sampled(
    env: SampledEnvelope, modes: GateModes, ion: int, mode: int, kernel: Kernel = "rwa"
) -> np.ndarray:
    """alpha_im(t) on the grid: the phase-space trajectory (Leung's robustness cost is its time average)."""
    cum = cumulative_simpson(
        trajectory_rate_sampled(env, modes, ion, mode, kernel), x=env.times_s, initial=0.0
    )
    return np.asarray(cum, dtype=complex)


def trajectory_rate_sampled(
    env: SampledEnvelope, modes: GateModes, ion: int, mode: int, kernel: Kernel = "rwa"
) -> np.ndarray:
    """d alpha_im/dt = i eta_im Omega_i(t) f_m(t) on the grid: the force whose running integral is
    ``trajectory_sampled``."""
    k = modes.index(mode)
    f = _kernel_samples(kernel, modes.omega_rad_s[k], env.times_s, env.beat_phase_rad, env.phi_m_rad)
    return np.asarray(1j * modes.eta[ion][k] * env.amplitude_rad_s[ion] * f, dtype=complex)


# ---- closed forms of the square pulse ---------------------------------------------------------------------------------------


def closure_rabi_rad_s(
    eta: float, epsilon_rad_s: float, loops: int = 1, chi_rad: float = CHI_MAXIMAL_RAD
) -> float:
    """The per-tone Omega that gives |chi| = chi_rad on ONE mode after K loops: chi = pi K (eta Omega/eps)^2."""
    if eta == 0.0 or epsilon_rad_s == 0.0:
        raise ValueError("the gate mode needs eta != 0 and a detuning eps != 0 from its sideband")
    return math.sqrt(abs(chi_rad) / (math.pi * loops)) * abs(epsilon_rad_s) / abs(eta)


def closure_duration_s(epsilon_rad_s: float, loops: int = 1) -> float:
    """tau = 2 pi K/|eps|."""
    return TWO_PI * loops / abs(epsilon_rad_s)


# ---- Waveform assembly -------------------------------------------------------------------------------------------------------------


def _phase_entry(value: float) -> CalEntry:
    """A solved waveform's spin or motion phase as a seed entry."""
    return CalEntry(float(value), 0.0, "seed", "pulse_solver", "conv.spin_motion_phases", 0.0, 0)


def _leg_phases(phi_s: float, phi_m: float) -> dict[Leg, float]:
    return {"blue": phi_s + phi_m, "red": phi_s - phi_m}


def waveform_from_segmented(
    env: SegmentedEnvelope,
    ints: GateIntegrals,
    modes: GateModes,
    *,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    kind: Literal["ms", "light_shift"] = "ms",
    pair: tuple[int, int] | None = None,
) -> Waveform:
    """A segmented ``Waveform`` from per-ion amplitudes at spin phase 0: legs red/blue at -/+ mu with phases
    -/+ (phi_m + phi_k), a negative amplitude played as |Omega| at phase + pi on BOTH legs (which moves the half-sum, the
    spin phase, and leaves the half-difference alone). ``env.phase_rad``, the PM family's per-segment motion-phase offset
    phi_k, goes into the legs' half-difference. A light-shift waveform has one leg ("blue", the beat note itself)."""
    mu_hz = env.mu_rad_s / TWO_PI
    segments: list[Segment] = []
    offsets = env.phase_rad if env.phase_rad is not None else (0.0,) * len(env.durations_s)
    if kind == "light_shift" and any(o != 0.0 for o in offsets):
        raise ValueError(
            "a one-leg light-shift waveform cannot separate the spin and motion phases: a per-segment phase offset "
            "is indistinguishable from a sign there"
        )
    for k, dur in enumerate(env.durations_s):
        amp: dict[tuple[int, Leg], float | Callable[[float], float]] = {}
        phase: dict[tuple[int, Leg], float] = {}
        phi_m_k = phi_m_rad + float(offsets[k])
        for ion, amps in env.amplitude_rad_s.items():
            omega_k = float(amps[k])
            extra = math.pi if omega_k < 0.0 else 0.0
            if kind == "light_shift":
                amp[(ion, "blue")] = abs(omega_k) / TWO_PI
                phase[(ion, "blue")] = phi_m_k + extra
            else:
                amp[(ion, "blue")] = abs(omega_k) / TWO_PI
                amp[(ion, "red")] = abs(omega_k) / TWO_PI
                for leg, ph in _leg_phases(0.0, phi_m_k).items():
                    phase[(ion, leg)] = ph + extra
        detuning: dict[Leg, float | Callable[[float], float]] = (
            {"blue": mu_hz} if kind == "light_shift" else {"blue": mu_hz, "red": -mu_hz}
        )
        segments.append(Segment(float(dur), amp, phase, detuning))
    return Waveform(
        segments=tuple(segments),
        duration_s=env.duration_s,
        phi_s=_phase_entry(0.0),
        phi_m=_phase_entry(phi_m_rad),
        chi_m=_chi_by_mode_for_pair(ints, modes, pair),
        alpha_m=_alpha_by_mode(ints, modes),
        kind=kind,
    )


def waveform_from_callables(
    duration_s: float,
    modes: GateModes,
    ints: GateIntegrals,
    *,
    amplitude_rad_s: Mapping[int, Callable[[float], float]],
    mu_rad_s: Callable[[float], float] | float,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    kind: Literal["ms", "light_shift"] = "ms",
    pair: tuple[int, int] | None = None,
) -> Waveform:
    """A single-segment ``Waveform`` at spin phase 0 with callable amplitudes (Fourier AM) or a callable beat note (FM), in
    Hz of tau."""

    def hz_of(fn: Callable[[float], float]) -> Callable[[float], float]:
        return ScaledFn(fn, 1.0 / TWO_PI)

    def neg_hz_of(fn: Callable[[float], float]) -> Callable[[float], float]:
        return ScaledFn(fn, -1.0 / TWO_PI)

    amp: dict[tuple[int, Leg], float | Callable[[float], float]] = {}
    phase: dict[tuple[int, Leg], float] = {}
    for ion, fn in amplitude_rad_s.items():
        if kind == "light_shift":
            amp[(ion, "blue")] = hz_of(fn)
            phase[(ion, "blue")] = 0.0 + phi_m_rad
        else:
            amp[(ion, "blue")] = hz_of(fn)
            amp[(ion, "red")] = hz_of(fn)
            for leg, ph in _leg_phases(0.0, phi_m_rad).items():
                phase[(ion, leg)] = ph
    detuning: dict[Leg, float | Callable[[float], float]]
    if callable(mu_rad_s):
        detuning = (
            {"blue": hz_of(mu_rad_s)}
            if kind == "light_shift"
            else {"blue": hz_of(mu_rad_s), "red": neg_hz_of(mu_rad_s)}
        )
    else:
        mu_hz = float(mu_rad_s) / TWO_PI
        detuning = {"blue": mu_hz} if kind == "light_shift" else {"blue": mu_hz, "red": -mu_hz}
    seg = Segment(float(duration_s), amp, phase, detuning)
    return Waveform(
        segments=(seg,),
        duration_s=float(duration_s),
        phi_s=_phase_entry(0.0),
        phi_m=_phase_entry(phi_m_rad),
        chi_m=_chi_by_mode_for_pair(ints, modes, pair),
        alpha_m=_alpha_by_mode(ints, modes),
        kind=kind,
    )


def _chi_by_mode_for_pair(
    ints: GateIntegrals, modes: GateModes, pair: tuple[int, int] | None = None
) -> dict[int, float]:
    """The per-mode entangling angles of the pair the pulse was SOLVED for: a ``GateModes`` on more than two ions carries an
    angle per pair and the scheduler rescales by the stored one, so a multi-pair ``GateModes`` with no pair named is
    refused rather than defaulted to the first pair."""
    pairs = modes.pairs()
    if pair is not None:
        a, b = int(pair[0]), int(pair[1])
        if (a, b) not in ints.chi_by_mode_pairs and (b, a) not in ints.chi_by_mode_pairs:
            raise ValueError(f"pair {(a, b)} is not a pair of the GateModes ions {modes.ions}")
        if (a, b) not in ints.chi_by_mode_pairs:
            a, b = b, a
    elif len(pairs) == 1:
        a, b = pairs[0]
    else:
        raise ValueError(
            f"a Waveform stores ONE pair's per-mode angles, and this GateModes carries {len(pairs)} pairs "
            f"{pairs}: name the pair the pulse targets"
        )
    return {m: ints.chi_by_mode[(a, b, m)] for m in modes.modes}


def _alpha_by_mode(ints: GateIntegrals, modes: GateModes) -> dict[int, complex]:
    """The largest residual over the gate ions per mode (what the calibration reports as the open-loop remainder)."""
    out: dict[int, complex] = {}
    for m in modes.modes:
        worst = max((ints.alpha[(i, m)] for i in modes.ions), key=abs)
        out[m] = complex(worst)
    return out


PHASE_SPLIT_TOLERANCE_RAD = 1e-9
"""How far a segment's spin-phase offset may sit from 0 or pi before ``envelope_of`` refuses to read it as a sign."""


def _split_phase(delta_rad: float) -> tuple[float, float]:
    """(sign, residual) of a phase offset: the nearest multiple of pi is a sign (+-1), the reversed force the AM solvers
    encode as a negative amplitude, and what is left is a genuine phase (which a PM waveform carries)."""
    residual = ((delta_rad + 0.5 * math.pi) % math.pi) - 0.5 * math.pi
    sign = 1.0 if math.cos(delta_rad - residual) >= 0.0 else -1.0
    return sign, residual


def envelope_of(waveform: Waveform, ions: Sequence[int], *, n_samples: int = 20001) -> Envelope:
    """Reconstruct the solver's envelope from a Waveform: segmented when every amplitude and detuning is constant, sampled
    otherwise; a leg imbalance enters through the mean leg amplitude.

    Per segment the two legs' half-SUM carries the ion's spin phase and their half-DIFFERENCE the motion phase:
    a pi offset of the half-sum relative to the first segment is the reversed force the AM solvers store as a negative
    amplitude, while an offset of the half-difference is a genuine per-segment motion phase (the PM family) and is returned
    in ``SegmentedEnvelope.phase_rad``. A half-sum offset that is neither 0 nor pi is a per-segment change of the FORCE AXIS,
    which no envelope can express, and is refused rather than collapsed to a sign."""
    segs = waveform.segments
    ions_ = tuple(int(i) for i in ions)
    one_leg = waveform.kind == "light_shift"
    legs: tuple[Leg, ...] = ("blue",) if one_leg else ("blue", "red")
    constant = all(
        not callable(s.detuning_hz[leg]) and all(not callable(s.amplitude_hz[(i, leg)]) for i in ions_)
        for s in segs
        for leg in legs
    )

    def spin_and_motion(seg: Segment, ion: int) -> tuple[float, float]:
        """(half-sum, half-difference) of the ion's leg phases; a one-leg (light-shift) force carries both in one phase."""
        blue = float(seg.phase_rad[(ion, "blue")])
        if one_leg:
            return blue, blue
        red = float(seg.phase_rad[(ion, "red")])
        return 0.5 * (blue + red), 0.5 * (blue - red)

    ref_spin = {i: spin_and_motion(segs[0], i)[0] for i in ions_}
    phi_m = spin_and_motion(segs[0], ions_[0])[1]
    signs: list[float] = []
    offsets: list[float] = []
    for seg in segs:
        sign, residual = _split_phase(spin_and_motion(seg, ions_[0])[0] - ref_spin[ions_[0]])
        if abs(residual) > PHASE_SPLIT_TOLERANCE_RAD:
            raise ValueError(
                f"segment spin phase offset {residual:.6g} rad is neither 0 nor pi: a per-segment force AXIS is not "
                "an envelope (a per-segment motion phase is, and belongs in the legs' half-difference)"
            )
        for i in ions_[1:]:
            sign_i, residual_i = _split_phase(spin_and_motion(seg, i)[0] - ref_spin[i])
            if sign_i != sign or abs(residual_i) > PHASE_SPLIT_TOLERANCE_RAD:
                raise ValueError("the gate ions' segment sign flips must agree for a single envelope")
        signs.append(sign)
        offsets.append(spin_and_motion(seg, ions_[0])[1] - phi_m if not one_leg else 0.0)
    modulated = any(abs(o) > PHASE_SPLIT_TOLERANCE_RAD for o in offsets)
    if constant:
        mus = {float(s.detuning_hz["blue"]) for s in segs if not callable(s.detuning_hz["blue"])}
        if len(mus) != 1:
            raise ValueError("a segmented envelope has one beat note")
        amps: dict[int, tuple[float, ...]] = {}
        for i in ions_:
            vals = []
            for k, s in enumerate(segs):
                mean = float(
                    np.mean([float(v) for leg in legs if not callable(v := s.amplitude_hz[(i, leg)])])
                )
                vals.append(TWO_PI * mean * signs[k])
            amps[i] = tuple(vals)
        return SegmentedEnvelope(
            tuple(s.duration_s for s in segs),
            amps,
            TWO_PI * mus.pop(),
            phi_m,
            tuple(offsets) if modulated else None,
        )
    t = np.linspace(0.0, waveform.duration_s, n_samples)
    edges = np.concatenate([[0.0], np.cumsum([s.duration_s for s in segs])])
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(segs) - 1)
    amp_arr = {i: np.zeros_like(t) for i in ions_}
    mu_arr = np.zeros_like(t)
    phi_arr = np.zeros_like(t)
    for k, s in enumerate(segs):
        sel = idx == k
        tau = t[sel] - edges[k]
        mu_leg = s.detuning_hz["blue"]
        mu_arr[sel] = (
            np.array([float(mu_leg(x)) if callable(mu_leg) else float(mu_leg) for x in tau]) * TWO_PI
        )
        for i in ions_:
            vals = []
            for x in tau:
                per_leg = []
                for leg in legs:
                    v = s.amplitude_hz[(i, leg)]
                    per_leg.append(float(v(x)) if callable(v) else float(v))
                vals.append(float(np.mean(per_leg)))
            amp_arr[i][sel] = TWO_PI * np.asarray(vals) * signs[k]
        phi_arr[sel] = phi_m + offsets[k]
    theta = np.asarray(cumulative_simpson(mu_arr, x=t, initial=0.0), dtype=float)
    return SampledEnvelope(t, amp_arr, theta, phi_arr if modulated else phi_m)


def waveform_integrals(waveform: Waveform, modes: GateModes, kernel: Kernel | None = None) -> GateIntegrals:
    """The integrals of a stored Waveform on the given mode structure."""
    return integrals(envelope_of(waveform, modes.ions), modes, kernel)


LIGHT_SHIFT_FORCE_WEIGHT = 2.0
"""The per-ion spectral radius of a light-shift force operator w_dn P_0 + w_up P_1 whose weights differ by exactly 2 and one of
which vanishes (the archetype (-2, 0): a force on one qubit level only)."""


def excursion_by_mode(
    waveform: Waveform,
    modes: GateModes,
    *,
    n_samples: int = 4001,
    kernel: Kernel = "choi",
    force_weight: float | None = None,
) -> dict[int, float]:
    """Per mode, the largest coherent excursion of the pulse's phase-space trajectory: max over time of
    ``force_weight`` sum_i |alpha_{i,m}(t)|, the displacement the extreme spin branch reaches (every ion's force adds on
    that branch), on a sampled grid: the populated range a Fock cap must hold.

    ``force_weight`` is the per-ion spectral radius of the FORCE OPERATOR the builder applies, which alpha does not carry:
    1 for a spin-flip MS or a sigma_z gradient force, 2 (:data:`LIGHT_SHIFT_FORCE_WEIGHT`) for the archetypal (-2, 0)
    light-shift force on one level only, a spin-dependent force PLUS an equal spin-independent one. ``None`` derives it
    from the waveform kind; pass it explicitly for level weights outside that archetype."""
    weight = (
        force_weight
        if force_weight is not None
        else (LIGHT_SHIFT_FORCE_WEIGHT if waveform.kind == "light_shift" else 1.0)
    )
    env = envelope_of(waveform, modes.ions, n_samples=n_samples)
    sampled = env.sampled(n_samples) if isinstance(env, SegmentedEnvelope) else env
    out: dict[int, float] = {}
    for m in modes.modes:
        total = np.zeros(sampled.times_s.size)
        for ion in modes.ions:
            total += np.abs(trajectory_sampled(sampled, modes, ion, m, kernel))
        out[m] = float(weight * np.max(total))
    return out


def scaled(waveform: Waveform, factor: float) -> Waveform:
    """Every amplitude times ``factor``: chi scales as factor^2 and every alpha as factor (the s^2 law)."""

    def scale_val(v: float | Callable[[float], float]) -> float | Callable[[float], float]:
        if callable(v):
            return ScaledFn(v, factor)
        return factor * float(v)

    segs = tuple(
        Segment(
            s.duration_s,
            {k: scale_val(v) for k, v in s.amplitude_hz.items()},
            dict(s.phase_rad),
            dict(s.detuning_hz),
        )
        for s in waveform.segments
    )
    return Waveform(
        segments=segs,
        duration_s=waveform.duration_s,
        phi_s=waveform.phi_s,
        phi_m=waveform.phi_m,
        chi_m={m: factor**2 * v for m, v in waveform.chi_m.items()},
        alpha_m={m: factor * v for m, v in waveform.alpha_m.items()},
        kind=waveform.kind,
    )


def phase_shifted(waveform: Waveform, offsets_rad: Mapping[int, float]) -> Waveform:
    """Every segment's tone phases of ion i shifted by ``offsets_rad[i]`` on BOTH legs: the ion's spin phase (the half-sum)
    moves by the offset, the motion phase (the half-difference) and every angle stay; what the MS phase scan stores."""
    segs = tuple(
        Segment(
            s.duration_s,
            dict(s.amplitude_hz),
            {k: float(v) + float(offsets_rad.get(k[0], 0.0)) for k, v in s.phase_rad.items()},
            dict(s.detuning_hz),
        )
        for s in waveform.segments
    )
    return replace(waveform, segments=segs)


# ---- the solvers -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapedPulse:
    """A solved entangling pulse: the Waveform the scheduler plays plus the closed-form record behind it."""

    waveform: Waveform
    modes: GateModes
    integrals: GateIntegrals
    envelope: Envelope
    chi_rad: float
    """Signed two-body angle of the pair (the kernel's sign s = sign(chi_rad))."""
    method: str
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return 1 if self.chi_rad >= 0.0 else -1

    @property
    def residual_error(self) -> float:
        return self.integrals.residual_error(self.modes)


def segment_count(n_modes: int) -> int:
    """2N + 1 segments close N modes with equal envelopes (Choi 2014); with both transverse families N doubles (4N + 1)."""
    return 2 * n_modes + 1


def _pair(modes: GateModes, pair: tuple[int, int] | None) -> tuple[int, int]:
    if pair is None:
        if len(modes.ions) != 2:
            raise ValueError("name the pair whose entangling angle is targeted")
        return modes.ions[0], modes.ions[1]
    return pair


def solve_amplitude_modulation(
    modes: GateModes,
    *,
    mu_hz: float,
    duration_s: float,
    n_segments: int | None = None,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    kernel: Kernel = "choi",
    pair: tuple[int, int] | None = None,
    amplitude_ratio: float = 1.0,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """Segmented AM (Zhu-Monroe-Duan 2006; Choi 2014): equal segments, one beat note mu, Omega_b = ``amplitude_ratio`` x Omega_a.

    Closure alpha_{a,m}(tau) = 0 for every mode is 2N real linear conditions on the n segment amplitudes (2N + 1 by
    default); within the null space the power-optimal direction (largest |chi| per unit sum Omega_k^2, the top eigenvector
    of the projected kernel) is taken and scaled to |chi| = ``chi_target_rad``; the sign is the kernel's.
    """
    a, b = _pair(modes, pair)
    n = segment_count(modes.n_modes) if n_segments is None else int(n_segments)
    if n < 1:
        raise ValueError("at least one segment")
    durations = tuple([duration_s / n] * n)
    unit = SegmentedEnvelope(durations, {i: tuple([1.0] * n) for i in modes.ions}, TWO_PI * mu_hz, phi_m_rad)
    fs, gs = _segment_matrices(unit, modes, kernel)
    rows = []
    for k in range(modes.n_modes):
        if any(modes.eta[i][k] != 0.0 for i in (a, b)):
            rows.append(np.real(fs[k]))
            rows.append(np.imag(fs[k]))
    if rows:
        amat = np.array(rows)
        # |F_k| <= the segment duration, so the natural scale of a row is the pulse duration: an already-closed pulse
        # (every F_k ~ 1e-16 tau) then has rank 0 rather than a spurious rank from its round-off
        _u, sing, vt = np.linalg.svd(amat / duration_s, full_matrices=True)
        rank = int(np.sum(sing > 1e-10))
        null = vt[rank:].T  # n x (n - rank)
    else:
        null = np.eye(n)
    if null.shape[1] == 0:
        raise ClosureError(
            f"{n} segments cannot close {modes.n_modes} modes (2N + 1 = {segment_count(modes.n_modes)} needed)"
        )
    gtot = sum(modes.eta[a][k] * modes.eta[b][k] * gs[k] for k in range(modes.n_modes)) * amplitude_ratio
    proj = null.T @ gtot @ null
    proj = 0.5 * (proj + proj.T)
    vals, vecs = np.linalg.eigh(proj)
    j = int(np.argmax(np.abs(vals)))
    if abs(vals[j]) <= 0.0:
        raise ClosureError("the closed pulses carry no entangling angle (every eta_a eta_b product vanishes)")
    direction = null @ vecs[:, j]
    direction = direction / np.linalg.norm(direction)
    # fix the overall sign so the first non-zero segment is positive (a convention; the physics is the same)
    first = direction[np.flatnonzero(np.abs(direction) > 1e-12)[0]]
    if first < 0.0:
        direction = -direction
    lam = float(vals[j])
    omega0 = math.sqrt(abs(chi_target_rad) / abs(lam))
    amps_a = tuple(float(x) for x in omega0 * direction)
    amps = {i: amps_a for i in modes.ions}
    amps[b] = tuple(amplitude_ratio * x for x in amps_a)
    env = SegmentedEnvelope(durations, amps, TWO_PI * mu_hz, phi_m_rad)
    ints = integrals_segmented(env, modes, kernel)
    chi = ints.chi_of(a, b)
    wf = waveform_from_segmented(env, ints, modes, phi_m_rad=phi_m_rad, kind=kind, pair=(a, b))
    return ShapedPulse(
        waveform=wf,
        modes=modes,
        integrals=ints,
        envelope=env,
        chi_rad=chi,
        method="am_segmented",
        diagnostics={
            "null_space_dimension": float(null.shape[1]),
            "segments": float(n),
            "peak_rabi_hz": float(max(abs(x) for x in amps_a) / TWO_PI),
            "power_integral_rad2_s": float(sum(x * x * d for x, d in zip(amps_a, durations))),
            "residual_error": ints.residual_error(modes),
        },
    )


def symmetric_pulse(
    modes: GateModes,
    *,
    gate_mode: int,
    loops: int = 1,
    epsilon_hz: float | None = None,
    duration_s: float | None = None,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    kernel: Kernel = "choi",
    pair: tuple[int, int] | None = None,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    detuning_side: Literal["inside", "outside"] = "inside",
    all_modes: bool = True,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """The square bichromatic pulse closed on ``gate_mode`` after ``loops`` loops: tau = 2 pi K/eps, tones at
    -/+ (omega_g - eps) ("inside", the sideband detuning eps toward the carrier) or -/+ (omega_g + eps) ("outside"), and
    the amplitude from |chi| = chi_target with chi = sum_m of the pair's per-mode angles (spectators included when
    ``all_modes``): for one mode eta Omega/eps = 1/(2 sqrt K) exactly."""
    a, b = _pair(modes, pair)
    k = modes.index(gate_mode)
    if epsilon_hz is None and duration_s is None:
        raise ValueError("give the sideband detuning epsilon_hz or the duration (tau = K/epsilon)")
    if epsilon_hz is None:
        assert duration_s is not None
        epsilon_hz = loops / duration_s
    eps = TWO_PI * abs(float(epsilon_hz))
    tau = closure_duration_s(eps, loops)
    omega_g = modes.omega_rad_s[k]
    mu = omega_g - eps if detuning_side == "inside" else omega_g + eps
    if mu <= 0.0:
        raise ValueError("the beat note must stay positive")
    unit = SegmentedEnvelope((tau,), {i: (1.0,) for i in modes.ions}, mu, phi_m_rad)
    sub = modes if all_modes else modes.subset([gate_mode])
    ints_unit = integrals_segmented(unit, sub, kernel)
    chi_unit = ints_unit.chi_of(a, b)
    if chi_unit == 0.0:
        raise ClosureError("the pair has no entangling angle on the gate mode (eta_a eta_b = 0)")
    omega0 = math.sqrt(abs(chi_target_rad) / abs(chi_unit))
    env = SegmentedEnvelope((tau,), {i: (omega0,) for i in modes.ions}, mu, phi_m_rad)
    ints = integrals_segmented(env, modes, kernel)
    wf = waveform_from_segmented(env, ints, modes, phi_m_rad=phi_m_rad, kind=kind, pair=(a, b))
    return ShapedPulse(
        waveform=wf,
        modes=modes,
        integrals=ints,
        envelope=env,
        chi_rad=ints.chi_of(a, b),
        method="symmetric_square",
        diagnostics={
            "loops": float(loops),
            "epsilon_hz": float(eps / TWO_PI),
            "rabi_hz": float(omega0 / TWO_PI),
            "closure_ratio": float(abs(modes.eta[a][k]) * omega0 / eps),
            "residual_error": ints.residual_error(modes),
        },
    )


def solve_phase_modulation(
    modes: GateModes,
    *,
    mu_hz: float,
    duration_s: float,
    n_segments: int | None = None,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    kernel: Kernel = "choi",
    pair: tuple[int, int] | None = None,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    max_nfev: int = 400,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """Segmented PM: FIXED amplitude and beat note, the per-segment MOTION phase phi_k free.

    The pulse is a square bichromatic drive whose tone phases step from segment to segment so that the legs'
    half-difference is phi_m + phi_k while their half-sum, the spin phase, never moves: the force axis therefore stays put
    and only the phase-space direction of each segment's push rotates. Every integral sees the complex segment weight
    z_k = Omega e^{i phi_k}, so closure is

        alpha_{i,m}(tau) = i eta_{i,m} Omega sum_k e^{i phi_k} F_k^{(m)} = 0,

    2N real conditions on the n phases (one of which is a global phase the conditions are blind to), and the two-body angle
    follows from the symmetrized double integral with the phases inside (``integrals_segmented``). The phases are solved by
    least squares like the FM family's vertices, then Omega is set by |chi| = chi_target (chi ~ Omega^2 at fixed phases).

    ``n_segments`` defaults to Choi's 2N + 1, which makes the system square in the n - 1 non-global phases. Milne et al.
    close N modes with N + 1 segments by imposing a time-symmetric phase profile, which this solver does not impose.
    Milne et al., PRApplied 13, 024022 (2020); Green and Biercuk, PRL 114, 120502 (2015); Lu et al., Nature 572, 363 (2019).
    """
    a, b = _pair(modes, pair)
    n = 2 * modes.n_modes + 1 if n_segments is None else int(n_segments)
    if n < 2:
        raise ValueError("a phase-modulated pulse needs at least two segments")
    tau = float(duration_s)
    durations = tuple([tau / n] * n)
    unit = SegmentedEnvelope(durations, {i: tuple([1.0] * n) for i in modes.ions}, TWO_PI * mu_hz, phi_m_rad)
    fs, _gs = _segment_matrices(unit, modes, kernel)
    rows: list[int] = [k for k in range(modes.n_modes) if any(modes.eta[i][k] != 0.0 for i in (a, b))]
    if not rows:
        raise ClosureError("the pair couples to no mode (every eta_a eta_b product vanishes)")

    def residuals(free: np.ndarray) -> np.ndarray:
        # the first segment's phase is the global phase the closure conditions cannot see: hold it at zero
        z = np.exp(1j * np.concatenate([[0.0], free]))
        out: list[float] = []
        for k in rows:
            val = np.dot(z, fs[k]) / tau
            out += [float(np.real(val)), float(np.imag(val))]
        return np.asarray(out)

    x0 = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)[1:]
    sol = least_squares(residuals, x0, max_nfev=max_nfev, ftol=1e-15, xtol=1e-15, gtol=None)
    phases = tuple(float(x) for x in np.concatenate([[0.0], sol.x]))
    unit_pm = SegmentedEnvelope(
        durations, {i: tuple([1.0] * n) for i in modes.ions}, TWO_PI * mu_hz, phi_m_rad, phases
    )
    ints_unit = integrals_segmented(unit_pm, modes, kernel)
    chi_unit = ints_unit.chi_of(a, b)
    if chi_unit == 0.0:
        raise ClosureError("the phase-modulated solution carries no entangling angle")
    omega0 = math.sqrt(abs(chi_target_rad) / abs(chi_unit))
    env = SegmentedEnvelope(
        durations, {i: tuple([omega0] * n) for i in modes.ions}, TWO_PI * mu_hz, phi_m_rad, phases
    )
    ints = integrals_segmented(env, modes, kernel)
    wf = waveform_from_segmented(env, ints, modes, phi_m_rad=phi_m_rad, kind=kind, pair=(a, b))
    return ShapedPulse(
        waveform=wf,
        modes=modes,
        integrals=ints,
        envelope=env,
        chi_rad=ints.chi_of(a, b),
        method="pm_segmented",
        diagnostics={
            "segments": float(n),
            "rabi_hz": float(omega0 / TWO_PI),
            "closure_cost": float(sol.cost),
            "residual_error": ints.residual_error(modes),
            "converged": float(bool(sol.success)),
            "phase_span_rad": float(np.max(phases) - np.min(phases)),
        },
    )


@dataclass(frozen=True, eq=False)
class CosineVertexFn:
    """Leung's vertex parameterization: mu(t) between equally spaced vertices with (1 - cos)/2 interpolation (picklable)."""

    vertices: tuple[float, ...]
    duration_s: float

    def __call__(self, tau: float) -> float:
        v = self.vertices
        n = len(v)
        if n < 2:
            return float(v[0])
        dt = self.duration_s / (n - 1)
        x = min(max(tau, 0.0), self.duration_s) / dt
        k = min(int(math.floor(x)), n - 2)
        s = x - k
        return float(v[k] + (v[k + 1] - v[k]) * 0.5 * (1.0 - math.cos(math.pi * s)))


@dataclass(frozen=True, eq=False)
class FourierSineFn:
    """Bluemel's Fourier-sine amplitude g(tau) = sum_n c_n sin(2 pi (n + 1) tau/T) in rad/s (picklable)."""

    coefficients: tuple[float, ...]
    duration_s: float

    def __call__(self, tau: float) -> float:
        return float(
            sum(
                c * math.sin(2.0 * math.pi * (n + 1) * tau / self.duration_s)
                for n, c in enumerate(self.coefficients)
            )
        )


def _cosine_interpolation(vertices: np.ndarray, duration_s: float) -> CosineVertexFn:
    return CosineVertexFn(tuple(float(x) for x in np.asarray(vertices, dtype=float)), float(duration_s))


def _sampled_from_callables(
    duration_s: float,
    amplitude_rad_s: Mapping[int, Callable[[float], float]],
    mu_rad_s: Callable[[float], float],
    n: int,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
) -> SampledEnvelope:
    t = np.linspace(0.0, duration_s, n)
    mu = np.array([mu_rad_s(x) for x in t])
    theta = np.asarray(cumulative_simpson(mu, x=t, initial=0.0), dtype=float)
    amps = {ion: np.array([fn(x) for x in t]) for ion, fn in amplitude_rad_s.items()}
    return SampledEnvelope(t, amps, theta, phi_m_rad)


def solve_frequency_modulation(
    modes: GateModes,
    *,
    duration_s: float,
    n_vertices: int,
    mu0_hz: float,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    kernel: Kernel = "rwa",
    pair: tuple[int, int] | None = None,
    robust: bool = True,
    n_samples: int = 4001,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    max_nfev: int = 400,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """Leung 2018's FM gate: constant Omega, a time-symmetric detuning mu(t) through ``n_vertices`` cosine-interpolated
    vertices, solved by least squares for alpha_m(tau) = 0 on every mode and, with ``robust``, for a vanishing time-averaged
    trajectory (d alpha_m/d delta_1 = 0 under a common detuning drift), then Omega from |chi| = chi_target (chi ~ Omega^2)."""
    a, b = _pair(modes, pair)
    n_free = (n_vertices + 1) // 2
    mu0 = TWO_PI * mu0_hz
    tau = float(duration_s)

    def full_vertices(free: np.ndarray) -> np.ndarray:
        return np.concatenate([free, free[: n_vertices - n_free][::-1]])

    def residuals(free: np.ndarray) -> np.ndarray:
        """Dimensionless closure (and robustness) residuals of the unit-amplitude pulse: alpha_m/(eta tau) is at most 1."""
        mu_fn = _cosine_interpolation(mu0 + free, tau)
        env = _sampled_from_callables(
            tau, {i: ConstantFn(1.0) for i in modes.ions}, mu_fn, n_samples, phi_m_rad
        )
        out: list[float] = []
        for k, m in enumerate(modes.modes):
            # Omega is constant and equal on both ions, so alpha_{i,m} ~ eta_{i,m} and one ion's trajectory closes the
            # other's, EXCEPT where that ion sits at a node of the mode (eta = 0, the centre ion of an odd chain in an
            # antisymmetric mode): drive the residual from the ion that couples more strongly, skip only when neither does
            eta_a, eta_b = modes.eta[a][k], modes.eta[b][k]
            if eta_a == 0.0 and eta_b == 0.0:
                continue
            ion, eta_ref = (a, eta_a) if abs(eta_a) >= abs(eta_b) else (b, eta_b)
            alpha_t = trajectory_sampled(env, modes, ion, m, kernel) / (abs(eta_ref) * tau)
            out += [float(np.real(alpha_t[-1])), float(np.imag(alpha_t[-1]))]
            if robust:
                mean = simpson(alpha_t, x=env.times_s) / tau
                out += [float(np.real(mean)), float(np.imag(mean))]
        return np.asarray(out)

    # start from a gentle symmetric sweep (a flat schedule is the square pulse, whose Jacobian is degenerate for equal vertices)
    x0 = TWO_PI * 0.01 * abs(mu0_hz) * np.cos(np.linspace(0.0, math.pi, n_free)) if n_free else np.zeros(0)
    swing = TWO_PI * max(abs(mu0_hz) * 0.05, 1e3)
    sol = least_squares(
        lambda x: residuals(full_vertices(x)),
        x0,
        max_nfev=max_nfev,
        x_scale=swing,
        # the closure residual is alpha/(eta tau) of the UNIT-amplitude pulse, so the played alpha carries the factor
        # Omega_0 ~ 1e6 rad/s: stopping at a relative cost reduction of 1e-12 leaves |alpha| ~ 1e-6 after the scale-up.
        # The gradient test is what stops the descent first here (a small residual on a well-conditioned Jacobian makes
        # J^T r tiny long before the residual is at its floor), so it is disabled and the step/cost tests do the work:
        # the normalized closure then reaches 2e-14 (|alpha| < 1e-12 played) in about 50 function evaluations.
        ftol=1e-15,
        xtol=1e-15,
        gtol=None,
    )
    verts = mu0 + full_vertices(sol.x)
    mu_fn = _cosine_interpolation(verts, tau)
    unit = _sampled_from_callables(tau, {i: ConstantFn(1.0) for i in modes.ions}, mu_fn, n_samples, phi_m_rad)
    ints_unit = integrals_sampled(unit, modes, kernel)
    chi_unit = ints_unit.chi_of(a, b)
    if chi_unit == 0.0:
        raise ClosureError("the FM solution carries no entangling angle")
    omega0 = math.sqrt(abs(chi_target_rad) / abs(chi_unit))
    env = SampledEnvelope(
        unit.times_s,
        {i: omega0 * unit.amplitude_rad_s[i] for i in modes.ions},
        unit.beat_phase_rad,
        phi_m_rad,
    )
    ints = integrals_sampled(env, modes, kernel)
    wf = waveform_from_callables(
        tau,
        modes,
        ints,
        amplitude_rad_s={i: ConstantFn(float(omega0)) for i in modes.ions},
        mu_rad_s=mu_fn,
        phi_m_rad=phi_m_rad,
        kind=kind,
        pair=(a, b),
    )
    return ShapedPulse(
        waveform=wf,
        modes=modes,
        integrals=ints,
        envelope=env,
        chi_rad=ints.chi_of(a, b),
        method="fm_vertices",
        diagnostics={
            "vertices": float(n_vertices),
            "rabi_hz": float(omega0 / TWO_PI),
            "closure_cost": float(sol.cost),
            "residual_error": ints.residual_error(modes),
            "mu_min_hz": float(np.min(verts) / TWO_PI),
            "mu_max_hz": float(np.max(verts) / TWO_PI),
            "converged": float(bool(sol.success)),
        },
    )


def solve_fourier_amplitude_modulation(
    modes: GateModes,
    *,
    mu_hz: float,
    duration_s: float,
    n_basis: int,
    stabilization_order: int = 0,
    chi_target_rad: float = CHI_MAXIMAL_RAD,
    kernel: Kernel = "rwa",
    pair: tuple[int, int] | None = None,
    n_samples: int = 8001,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """Blumel 2021's power-optimal stabilized AM: g(t) = sum_n A_n sin(2 pi n t/tau) at one beat note; closure and its first
    K mode-frequency derivatives (d^k alpha_m/d omega_m^k = 0, k <= ``stabilization_order``) form the homogeneous system M A
    = 0 (2N(K + 1) rows), the power-optimal solution is the top eigenvector of the kernel projected on the null space, and
    the scale follows from |chi| = chi_target in the pi/4 convention (their pi/8 kernel is half of ours)."""
    a, b = _pair(modes, pair)
    tau = float(duration_s)
    t = np.linspace(0.0, tau, n_samples)
    theta = TWO_PI * mu_hz * t
    basis = np.array(
        [np.sin(2.0 * math.pi * (n + 1) * t / tau) for n in range(n_basis)]
    )  # (n_basis, n_samples)
    rows: list[np.ndarray] = []
    for k in range(modes.n_modes):
        if all(modes.eta[i][k] == 0.0 for i in (a, b)):
            continue
        f = _kernel_samples(kernel, modes.omega_rad_s[k], t, theta, phi_m_rad)
        for order in range(stabilization_order + 1):
            weight = (1j * t) ** order * f
            col = np.array([simpson(basis[n] * weight, x=t) for n in range(n_basis)])
            rows.append(np.real(col))
            rows.append(np.imag(col))
    amat = np.array(rows) if rows else np.zeros((0, n_basis))
    if amat.shape[0]:
        scale = np.max(np.abs(amat)) or 1.0
        gamma = (amat / scale).T @ (amat / scale)
        evals, evecs = np.linalg.eigh(gamma)
        keep = evals < 1e-20 * max(float(evals[-1]), 1e-300)
        if not np.any(keep):
            raise ClosureError(
                f"{n_basis} basis functions cannot satisfy {amat.shape[0]} closure rows; add basis functions"
            )
        null = evecs[:, keep]
    else:
        null = np.eye(n_basis)
    # the kernel in the basis: chi = A^T K A with K_nn' the symmetrized double integral of basis n on ion a, n' on ion b
    kmat = np.zeros((n_basis, n_basis))
    for k in range(modes.n_modes):
        w = modes.eta[a][k] * modes.eta[b][k]
        if w == 0.0:
            continue
        f = _kernel_samples(kernel, modes.omega_rad_s[k], t, theta, phi_m_rad)
        cum = np.array([cumulative_simpson(basis[n] * f, x=t, initial=0.0) for n in range(n_basis)])
        for n in range(n_basis):
            for p in range(n, n_basis):
                integrand = np.imag(basis[p] * f * np.conj(cum[n]) + basis[n] * f * np.conj(cum[p]))
                val = w * float(simpson(integrand, x=t))
                kmat[n, p] += val
                if p != n:
                    kmat[p, n] += val
    proj = null.T @ kmat @ null
    proj = 0.5 * (proj + proj.T)
    vals, vecs = np.linalg.eigh(proj)
    j = int(np.argmax(np.abs(vals)))
    lam = float(vals[j])
    if lam == 0.0:
        raise ClosureError("the stabilized null space carries no entangling angle")
    direction = null @ vecs[:, j]
    direction = direction / np.linalg.norm(direction)
    coeffs = math.sqrt(abs(chi_target_rad) / abs(lam)) * direction
    g = coeffs @ basis
    env = SampledEnvelope(t, {i: g.copy() for i in modes.ions}, theta, phi_m_rad)
    ints = integrals_sampled(env, modes, kernel)

    g_fn = FourierSineFn(tuple(float(c) for c in coeffs), float(tau))
    wf = waveform_from_callables(
        tau,
        modes,
        ints,
        amplitude_rad_s={i: g_fn for i in modes.ions},
        mu_rad_s=TWO_PI * mu_hz,
        phi_m_rad=phi_m_rad,
        kind=kind,
        pair=(a, b),
    )
    return ShapedPulse(
        waveform=wf,
        modes=modes,
        integrals=ints,
        envelope=env,
        chi_rad=ints.chi_of(a, b),
        method="fourier_am_stabilized",
        diagnostics={
            "basis_functions": float(n_basis),
            "stabilization_order": float(stabilization_order),
            "null_space_dimension": float(null.shape[1]),
            "constraint_rows": float(amat.shape[0]),
            "peak_rabi_hz": float(np.max(np.abs(g)) / TWO_PI),
            "power_integral_rad2_s": float(simpson(g * g, x=t)),
            "residual_error": ints.residual_error(modes),
            # Blumel's f = (4/5) sum_p (|alpha_p^i|^2 + |alpha_p^j|^2) is Landsman's conversion of eps_ent at nbar = 0:
            # residual_error carries the (2 nbar_m + 1) thermal weight, so it is NOT this quantity at nbar != 0
            "zero_temperature_infidelity": 0.8
            * float(sum(abs(ints.alpha[(i, m)]) ** 2 for i in modes.ions for m in modes.modes)),
            **{f"A_{n + 1}": float(c) for n, c in enumerate(coeffs)},
        },
    )


def frequency_derivative_residuals(
    env: SampledEnvelope, modes: GateModes, ion: int, mode: int, *, orders: int, kernel: Kernel = "rwa"
) -> tuple[complex, ...]:
    """(d^k alpha/d omega_m^k)(tau) for k = 0..orders: the stabilization quantities Blumel's rows null."""
    k = modes.index(mode)
    f = _kernel_samples(kernel, modes.omega_rad_s[k], env.times_s, env.beat_phase_rad, env.phi_m_rad)
    out = []
    for order in range(orders + 1):
        val = simpson(env.amplitude_rad_s[ion] * (1j * env.times_s) ** order * f, x=env.times_s)
        out.append(complex(1j * modes.eta[ion][k] * val))
    return tuple(out)
