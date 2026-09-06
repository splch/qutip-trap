"""Closed-form gate integrals and the AM/FM/Fourier pulse solvers for multi-mode Molmer-Sorensen closure (PLAN.md
Sections 4.4.1, 4.4.3, 7.4, 13; milestone M4).

Conventions (Section 13, rows "Spin operator in MS formulas", "Entangling angle", "Waveform per (ion, leg)"): Omega is the
per-tone Rabi frequency of the (hbar Omega/2) convention; the bichromatic tones sit at -/+ mu from the carrier with phases
phi_s -/+ phi_m (Haljan's spin phase phi_s = (phi_b + phi_r)/2 and motion phase phi_m = (phi_b - phi_r)/2). Expanding the
builder's drive (hbar/2) sum_tones Omega e^{-i(mu t - phi)} sigma_+ D(i eta) + h.c. to first order in eta in the interaction
picture gives the spin-dependent force

    H/hbar = sum_i sum_m (eta_im Omega_i(t)/2) sigma^i_{phi_s + pi/2} (a_m^dag e^{i psi_m(t)} + a_m e^{-i psi_m(t)}),
    psi_m(t) = omega_m t - Theta(t) + phi_m,   Theta(t) = int_0^t mu(t') dt',

whose Magnus series terminates: U = exp[sum_i sigma^i sum_m (alpha_im a_m^dag - alpha_im^* a_m)] exp[i sum_{i<j} chi_ij sigma^i
sigma^j] (a global phase aside) with

    alpha_im(tau) = i eta_im int_0^tau Omega_i(t) f_m(t) dt,
    chi_ij(tau)   = sum_m eta_im eta_jm int_0^tau dt' int_0^{t'} dt [Omega_i(t) Omega_j(t') + Omega_j(t) Omega_i(t')] g_m(t, t'),
    g_m(t, t')    = Im[f_m(t') conj(f_m(t))].

Two kernels f_m, never mixed (Section 13): the exact first-order Lamb-Dicke form f_m = cos(Theta(t) - phi_m) e^{i omega_m t},
which keeps the counter-rotating omega_m + mu term (Choi 2014 Eq. 3 is its phi_m = pi/2 case, sin(Theta) e^{i omega t}, the
sine beat-note convention with the force zero at the pulse start), and the slow-envelope ("rwa") form f_m = (e^{i phi_m}/2)
e^{i(omega_m t - Theta(t))}, its resonant part, with g_m = (1/4) sin[psi_m(t') - psi_m(t)] (Leung 2018). The pair sum runs
over i < j, so the coefficient of sigma^a sigma^b is chi_ab itself and the kernel is SYMMETRIZED in the two envelopes:
Choi's printed 2 Omega_i(t) Omega_j(t') holds only for proportional envelopes (derivation audit 2026-09-04, Section 9.16 row
4.4-5). The motion phase is a convention for the closed forms but not for the exact dynamics: the carrier Omega cos(mu t -
phi_m) sigma_phi_s rotates the spin at F(t) = (2 Omega/mu)[sin(mu t - phi_m) + sin phi_m], whose mean 2 Omega sin(phi_m)/mu
tilts the force axis out of the equatorial plane by psi = (2 Omega/mu) sin phi_m (Roos 2008's psi = (4 Omega_Roos/delta)
sin zeta with Omega_Roos = Omega/2 and zeta = -phi_m). At phi_m = pi/2 the ``check_ms_closure.py`` fixture (Omega/mu = 0.1)
loses 4% of its concurrence (0.9597 against 0.9999); the waveforms therefore default to phi_m = 0 (phi_b = phi_r, the check
script's convention), where the tilt vanishes and the anchor reproduces (Section 9.4; M4 finding). For a square pulse of duration tau with (omega_m - mu) tau = 2 pi K the rwa kernel gives chi = pi K
(eta Omega/eps)^2, so eta Omega/eps = 1/(2 sqrt K) is the maximally entangling closure chi = pi/4 on sigma sigma
(``check_ms_closure.py``; Section 9.4). The sign s of chi is set by the detuning side: exp(+i chi sigma sigma) = XX(-chi), and
the scheduler folds s into the gate phases (a pi on one ion's tones flips it). Blumel 2021 write the kernel without the pair
sum and target pi/8: double their kernel to convert, never halve ours (Section 13).

The solvers act on the mode structure alone (omega_m, eta_im, nbar_m from the device's crystal and beams) and never import
dynamics/: the exact verification of a solution through the PulseEngine protocol is ``qutip_trap.calibration.entangling``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.integrate import cumulative_simpson, simpson
from scipy.optimize import least_squares

from qutip_trap.control.table import CalEntry, Leg, Segment, Waveform
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

Kernel = Literal["rwa", "choi"]
CHI_MAXIMAL_RAD = math.pi / 4.0
"""|chi| of a maximally entangling XX(chi) = exp(-i chi sigma_x sigma_x) (Section 13)."""
SINE_MOTION_PHASE_RAD = math.pi / 2.0
"""phi_m = pi/2: the tone phases differ by pi, so the force is Omega sin(Theta) (Choi's sine beat-note convention)."""
DEFAULT_MOTION_PHASE_RAD = 0.0
"""phi_m = 0: equal tone phases, the force Omega cos(Theta) with no mean carrier rotation (the played default, see above)."""


class ClosureError(ValueError):
    """The solver could not close every mode with the requested degrees of freedom, or the target angle is unreachable."""


# ---- the mode structure a gate sees -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateModes:
    """omega_m, eta_im and nbar_m of the modes an entangling pulse on ``ions`` couples to, in one fixed order."""

    ions: tuple[int, ...]
    modes: tuple[int, ...]
    """Crystal mode indices (positions in Crystal.modes, Section 13)."""
    omega_rad_s: tuple[float, ...]
    eta: dict[int, tuple[float, ...]]
    """ion -> eta_{ion, m} in the order of ``modes`` (C0 inside, Section 4.1.1)."""
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
    (default: every mode with any coupling), or the given ``modes``; C0 from the trap's Mathieu record (Section 4.1.1).

    ``mode_frequencies_hz`` overrides the crystal's frequencies with calibrated values (the solvers work at what the table
    says, never at the device's hidden truth, Section 7.3)."""
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

    def __post_init__(self) -> None:
        if not self.durations_s or any(d <= 0.0 for d in self.durations_s):
            raise ValueError("segment durations are positive")
        for ion, amps in self.amplitude_rad_s.items():
            if len(amps) != len(self.durations_s):
                raise ValueError(f"ion {ion}: one amplitude per segment")

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
        return SampledEnvelope(t, amps, self.mu_rad_s * t, self.phi_m_rad)


@dataclass(frozen=True)
class SampledEnvelope:
    """Per-ion amplitudes Omega_i(t) and the accumulated beat phase Theta(t) on a uniform grid (FM, Fourier AM, or anything)."""

    times_s: np.ndarray
    amplitude_rad_s: dict[int, np.ndarray]
    beat_phase_rad: np.ndarray
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD

    def __post_init__(self) -> None:
        t = np.asarray(self.times_s, dtype=float)
        if t.ndim != 1 or t.size < 3 or t.size % 2 == 0:
            raise ValueError("the grid has an odd number (>= 3) of points for Simpson's rule")
        if not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-9, atol=0.0):
            raise ValueError("the grid is uniform")
        if np.asarray(self.beat_phase_rad).shape != t.shape:
            raise ValueError("beat_phase_rad is sampled on the grid")
        for ion, a in self.amplitude_rad_s.items():
            if np.asarray(a).shape != t.shape:
                raise ValueError(f"ion {ion}: the amplitude is sampled on the grid")

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1] - self.times_s[0])


Envelope = SegmentedEnvelope | SampledEnvelope


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
    """The Section 4.4.3 integrals of one pulse: alpha per (ion, mode), chi per pair, chi per (pair, mode)."""

    alpha: dict[tuple[int, int], complex]
    chi: dict[tuple[int, int], float]
    chi_by_mode: dict[tuple[int, int, int], float]
    kernel: Kernel

    def chi_of(self, a: int, b: int) -> float:
        return self.chi[(a, b)] if (a, b) in self.chi else self.chi[(b, a)]

    def residual_error(self, modes: GateModes, ions: Sequence[int] | None = None) -> float:
        """epsilon_ent = sum_{i,m} |alpha_im|^2 (2 nbar_m + 1), the entanglement infidelity from open loops (Section 4.4.7 (8))."""
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
    """alpha and chi of a piecewise-constant pulse at one beat note, analytically (the AM family)."""
    fs, gs = _segment_matrices(env, modes, kernel)
    alpha: dict[tuple[int, int], complex] = {}
    chi: dict[tuple[int, int], float] = {}
    by_mode: dict[tuple[int, int, int], float] = {}
    amps = {ion: np.asarray(env.amplitude_rad_s[ion], dtype=float) for ion in modes.ions}
    for ion in modes.ions:
        for k, m in enumerate(modes.modes):
            alpha[(ion, m)] = complex(1j * modes.eta[ion][k] * np.dot(amps[ion], fs[k]))
    for a, b in modes.pairs():
        total = 0.0
        for k, m in enumerate(modes.modes):
            val = float(modes.eta[a][k] * modes.eta[b][k] * (amps[a] @ gs[k] @ amps[b]))
            by_mode[(a, b, m)] = val
            total += val
        chi[(a, b)] = total
    return GateIntegrals(alpha, chi, by_mode, kernel)


def _kernel_samples(
    kernel: Kernel, omega: float, t: np.ndarray, theta: np.ndarray, phi_m: float = DEFAULT_MOTION_PHASE_RAD
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
    """The Section 4.4.3 integrals of any envelope; default kernel: Choi's for segmented pulses, rwa for sampled ones."""
    if isinstance(env, SegmentedEnvelope):
        return integrals_segmented(env, modes, kernel or "choi")
    return integrals_sampled(env, modes, kernel or "rwa")


def trajectory_sampled(
    env: SampledEnvelope, modes: GateModes, ion: int, mode: int, kernel: Kernel = "rwa"
) -> np.ndarray:
    """alpha_im(t) on the grid: the phase-space trajectory (Leung's robustness cost is its time average)."""
    k = modes.index(mode)
    f = _kernel_samples(kernel, modes.omega_rad_s[k], env.times_s, env.beat_phase_rad, env.phi_m_rad)
    cum = cumulative_simpson(env.amplitude_rad_s[ion] * f, x=env.times_s, initial=0.0)
    return np.asarray(1j * modes.eta[ion][k] * cum, dtype=complex)


# ---- closed forms of the square pulse (Section 4.4.1) ---------------------------------------------------------------------------


def closure_ratio(loops: int) -> float:
    """eta Omega/eps = 1/(2 sqrt K): the maximally entangling closure in every spin normalization (Section 13)."""
    if loops < 1:
        raise ValueError("at least one loop")
    return 1.0 / (2.0 * math.sqrt(loops))


def closure_rabi_rad_s(
    eta: float, epsilon_rad_s: float, loops: int = 1, chi_rad: float = CHI_MAXIMAL_RAD
) -> float:
    """The per-tone Omega that gives |chi| = chi_rad on ONE mode after K loops: chi = pi K (eta Omega/eps)^2 (Section 4.4.1)."""
    if eta == 0.0 or epsilon_rad_s == 0.0:
        raise ValueError("the gate mode needs eta != 0 and a detuning eps != 0 from its sideband")
    return math.sqrt(abs(chi_rad) / (math.pi * loops)) * abs(epsilon_rad_s) / abs(eta)


def closure_duration_s(epsilon_rad_s: float, loops: int = 1) -> float:
    """tau = 2 pi K/|eps|."""
    return TWO_PI * loops / abs(epsilon_rad_s)


def square_pulse_chi(
    eta_a: float, eta_b: float, omega_rad_s: float, epsilon_rad_s: float, loops: int
) -> float:
    """chi of a square pulse closed on one mode, rwa kernel: pi K eta_a eta_b (Omega/eps)^2 with sign(eps) (Section 4.4.1)."""
    return float(
        math.pi
        * loops
        * eta_a
        * eta_b
        * (omega_rad_s / epsilon_rad_s) ** 2
        * (1.0 if epsilon_rad_s > 0 else -1.0)
    )


# ---- Waveform assembly -------------------------------------------------------------------------------------------------------------


def _seed_entry(value: float, provenance_id: str, experiment: str = "pulse_solver") -> CalEntry:
    return CalEntry(float(value), 0.0, "seed", experiment, provenance_id, 0.0, 0)


def _leg_phases(phi_s: float, phi_m: float) -> dict[Leg, float]:
    return {"blue": phi_s + phi_m, "red": phi_s - phi_m}


def waveform_from_segmented(
    env: SegmentedEnvelope,
    ints: GateIntegrals,
    modes: GateModes,
    *,
    phi_s_rad: float = 0.0,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    imbalance: float = 0.0,
    kind: Literal["ms", "light_shift"] = "ms",
    provenance_id: str = "conv.entangling_angle",
) -> Waveform:
    """A segmented ``Waveform`` from per-ion amplitudes: legs red/blue at -/+ mu with phases phi_s -/+ phi_m, a negative amplitude
    played as |Omega| at phase + pi on both legs; Kirchmair's imbalance Omega_b = Omega(1 + xi), Omega_r = Omega(1 - xi)
    (Section 4.4.1, Stark compensation) on request. A light-shift waveform has one leg ("blue", the beat note itself)."""
    mu_hz = env.mu_rad_s / TWO_PI
    segments: list[Segment] = []
    for k, dur in enumerate(env.durations_s):
        amp: dict[tuple[int, Leg], float | Callable[[float], float]] = {}
        phase: dict[tuple[int, Leg], float] = {}
        for ion, amps in env.amplitude_rad_s.items():
            omega_k = float(amps[k])
            extra = math.pi if omega_k < 0.0 else 0.0
            if kind == "light_shift":
                amp[(ion, "blue")] = abs(omega_k) / TWO_PI
                phase[(ion, "blue")] = phi_s_rad + phi_m_rad + extra
            else:
                amp[(ion, "blue")] = abs(omega_k) * (1.0 + imbalance) / TWO_PI
                amp[(ion, "red")] = abs(omega_k) * (1.0 - imbalance) / TWO_PI
                for leg, ph in _leg_phases(phi_s_rad, phi_m_rad).items():
                    phase[(ion, leg)] = ph + extra
        detuning: dict[Leg, float | Callable[[float], float]] = (
            {"blue": mu_hz} if kind == "light_shift" else {"blue": mu_hz, "red": -mu_hz}
        )
        segments.append(Segment(float(dur), amp, phase, detuning))
    return Waveform(
        segments=tuple(segments),
        fourier=None,
        duration_s=env.duration_s,
        phi_s=_seed_entry(phi_s_rad, "conv.spin_motion_phases"),
        phi_m=_seed_entry(phi_m_rad, "conv.spin_motion_phases"),
        chi_m=_chi_by_mode_for_pair(ints, modes),
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
    phi_s_rad: float = 0.0,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    kind: Literal["ms", "light_shift"] = "ms",
) -> Waveform:
    """A single-segment ``Waveform`` with callable amplitudes (Fourier AM) or a callable beat note (FM), in Hz of tau."""

    def hz_of(fn: Callable[[float], float]) -> Callable[[float], float]:
        return lambda tau: float(fn(tau)) / TWO_PI

    def neg_hz_of(fn: Callable[[float], float]) -> Callable[[float], float]:
        return lambda tau: -float(fn(tau)) / TWO_PI

    amp: dict[tuple[int, Leg], float | Callable[[float], float]] = {}
    phase: dict[tuple[int, Leg], float] = {}
    for ion, fn in amplitude_rad_s.items():
        if kind == "light_shift":
            amp[(ion, "blue")] = hz_of(fn)
            phase[(ion, "blue")] = phi_s_rad + phi_m_rad
        else:
            amp[(ion, "blue")] = hz_of(fn)
            amp[(ion, "red")] = hz_of(fn)
            for leg, ph in _leg_phases(phi_s_rad, phi_m_rad).items():
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
        fourier=None,
        duration_s=float(duration_s),
        phi_s=_seed_entry(phi_s_rad, "conv.spin_motion_phases"),
        phi_m=_seed_entry(phi_m_rad, "conv.spin_motion_phases"),
        chi_m=_chi_by_mode_for_pair(ints, modes),
        alpha_m=_alpha_by_mode(ints, modes),
        kind=kind,
    )


def _chi_by_mode_for_pair(ints: GateIntegrals, modes: GateModes) -> dict[int, float]:
    pairs = modes.pairs()
    if len(pairs) != 1:
        # a multi-pair pulse (a global beam on N > 2 ions) stores the first pair's per-mode angles
        pairs = pairs[:1]
    a, b = pairs[0]
    return {m: ints.chi_by_mode[(a, b, m)] for m in modes.modes}


def _alpha_by_mode(ints: GateIntegrals, modes: GateModes) -> dict[int, complex]:
    """The largest residual over the gate ions per mode (what the calibration reports as the open-loop remainder)."""
    out: dict[int, complex] = {}
    for m in modes.modes:
        worst = max((ints.alpha[(i, m)] for i in modes.ions), key=abs)
        out[m] = complex(worst)
    return out


def envelope_of(waveform: Waveform, ions: Sequence[int], *, n_samples: int = 20001) -> Envelope:
    """Reconstruct the solver's envelope from a Waveform: segmented when every amplitude and detuning is constant, sampled
    otherwise; a leg imbalance enters through the mean leg amplitude; a pi phase offset (relative to the first segment) is a sign."""
    if waveform.segments is None:
        raise ValueError(
            "Fourier-parameterized waveforms carry no explicit envelope; the solvers emit segments"
        )
    segs = waveform.segments
    ions_ = tuple(int(i) for i in ions)
    legs: tuple[Leg, ...] = ("blue",) if waveform.kind == "light_shift" else ("blue", "red")
    constant = all(
        not callable(s.detuning_hz[leg]) and all(not callable(s.amplitude_hz[(i, leg)]) for i in ions_)
        for s in segs
        for leg in legs
    )
    ref_phase = {i: float(segs[0].phase_rad[(i, "blue")]) for i in ions_}
    if waveform.kind == "light_shift":
        phi_m = float(segs[0].phase_rad[(ions_[0], "blue")])
    else:
        phi_m = 0.5 * float(segs[0].phase_rad[(ions_[0], "blue")] - segs[0].phase_rad[(ions_[0], "red")])
    if constant:
        mus = {float(s.detuning_hz["blue"]) for s in segs if not callable(s.detuning_hz["blue"])}
        if len(mus) != 1:
            raise ValueError("a segmented envelope has one beat note")
        amps: dict[int, tuple[float, ...]] = {}
        for i in ions_:
            vals = []
            for s in segs:
                mean = float(
                    np.mean([float(v) for leg in legs if not callable(v := s.amplitude_hz[(i, leg)])])
                )
                flip = math.cos(float(s.phase_rad[(i, "blue")]) - ref_phase[i])
                vals.append(TWO_PI * mean * (1.0 if flip >= 0.0 else -1.0))
            amps[i] = tuple(vals)
        return SegmentedEnvelope(tuple(s.duration_s for s in segs), amps, TWO_PI * mus.pop(), phi_m)
    t = np.linspace(0.0, waveform.duration_s, n_samples)
    edges = np.concatenate([[0.0], np.cumsum([s.duration_s for s in segs])])
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(segs) - 1)
    amp_arr = {i: np.zeros_like(t) for i in ions_}
    mu_arr = np.zeros_like(t)
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
            flip = math.cos(float(s.phase_rad[(i, "blue")]) - ref_phase[i])
            amp_arr[i][sel] = TWO_PI * np.asarray(vals) * (1.0 if flip >= 0.0 else -1.0)
    theta = np.asarray(cumulative_simpson(mu_arr, x=t, initial=0.0), dtype=float)
    return SampledEnvelope(t, amp_arr, theta, phi_m)


def waveform_integrals(waveform: Waveform, modes: GateModes, kernel: Kernel | None = None) -> GateIntegrals:
    """The Section 4.4.3 integrals of a stored Waveform on the given mode structure (the surrogate calibration's core)."""
    return integrals(envelope_of(waveform, modes.ions), modes, kernel)


def scaled(waveform: Waveform, factor: float) -> Waveform:
    """Every amplitude times ``factor``: chi scales as factor^2 and every alpha as factor (the s^2 law of Section 4.4.7 (7))."""
    if waveform.segments is None:
        raise ValueError("scaling a Fourier-parameterized waveform is not supported")

    def scale_val(v: float | Callable[[float], float]) -> float | Callable[[float], float]:
        if callable(v):
            fn = v
            return lambda tau: factor * float(fn(tau))
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
        fourier=None,
        duration_s=waveform.duration_s,
        phi_s=waveform.phi_s,
        phi_m=waveform.phi_m,
        chi_m={m: factor**2 * v for m, v in waveform.chi_m.items()},
        alpha_m={m: factor * v for m, v in waveform.alpha_m.items()},
        kind=waveform.kind,
    )


def total_chi(waveform: Waveform) -> float:
    """The signed two-body angle of a waveform, the sum of its per-mode angles (exp(+i chi sigma sigma))."""
    return float(sum(waveform.chi_m.values()))


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
    phi_s_rad: float = 0.0,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    imbalance: float = 0.0,
    null_tolerance: float = 1e-10,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """Segmented AM (Zhu-Monroe-Duan 2006; Choi 2014): equal segments, one beat note mu, Omega_b = ``amplitude_ratio`` x Omega_a.

    Closure alpha_{a,m}(tau) = 0 for every mode is 2N real linear conditions on the n segment amplitudes (2N + 1 by default,
    Section 4.4.3); within the null space the power-optimal direction (largest |chi| per unit sum Omega_k^2, the top
    eigenvector of the projected kernel) is taken and scaled to |chi| = ``chi_target_rad``; the sign is the kernel's.
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
        rank = int(np.sum(sing > null_tolerance))
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
    wf = waveform_from_segmented(
        env, ints, modes, phi_s_rad=phi_s_rad, phi_m_rad=phi_m_rad, imbalance=imbalance, kind=kind
    )
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
    phi_s_rad: float = 0.0,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    imbalance: float = 0.0,
    detuning_side: Literal["inside", "outside"] = "inside",
    all_modes: bool = True,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """The square bichromatic pulse of Section 4.4.1 closed on ``gate_mode`` after ``loops`` loops: tau = 2 pi K/eps, tones at
    -/+ (omega_g - eps) ("inside", the sideband detuning eps toward the carrier) or -/+ (omega_g + eps) ("outside"), and
    the amplitude from |chi| = chi_target with chi = sum_m of the pair's per-mode angles (spectators included when
    ``all_modes``): for one mode eta Omega/eps = 1/(2 sqrt K) exactly (``check_ms_closure.py``)."""
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
    wf = waveform_from_segmented(
        env, ints, modes, phi_s_rad=phi_s_rad, phi_m_rad=phi_m_rad, imbalance=imbalance, kind=kind
    )
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


def _constant(value: float) -> Callable[[float], float]:
    def fn(tau: float) -> float:
        return float(value)

    return fn


def _cosine_interpolation(vertices: np.ndarray, duration_s: float) -> Callable[[float], float]:
    """Leung's vertex parameterization: mu(t) between equally spaced vertices with (1 - cos)/2 interpolation."""
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    if n < 2:
        return lambda tau: float(v[0])
    dt = duration_s / (n - 1)

    def mu(tau: float) -> float:
        x = min(max(tau, 0.0), duration_s) / dt
        k = min(int(math.floor(x)), n - 2)
        s = x - k
        return float(v[k] + (v[k + 1] - v[k]) * 0.5 * (1.0 - math.cos(math.pi * s)))

    return mu


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
    robust_weight: float = 1.0,
    n_samples: int = 4001,
    phi_s_rad: float = 0.0,
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
            tau, {i: _constant(1.0) for i in modes.ions}, mu_fn, n_samples, phi_m_rad
        )
        out: list[float] = []
        for k, m in enumerate(modes.modes):
            eta_a = modes.eta[a][k]
            if eta_a == 0.0:
                continue
            alpha_t = trajectory_sampled(env, modes, a, m, kernel) / (abs(eta_a) * tau)
            out += [float(np.real(alpha_t[-1])), float(np.imag(alpha_t[-1]))]
            if robust:
                mean = simpson(alpha_t, x=env.times_s) / tau
                out += [robust_weight * float(np.real(mean)), robust_weight * float(np.imag(mean))]
        return np.asarray(out)

    # start from a gentle symmetric sweep (a flat schedule is the square pulse, whose Jacobian is degenerate for equal vertices)
    x0 = TWO_PI * 0.01 * abs(mu0_hz) * np.cos(np.linspace(0.0, math.pi, n_free)) if n_free else np.zeros(0)
    swing = TWO_PI * max(abs(mu0_hz) * 0.05, 1e3)
    sol = least_squares(
        lambda x: residuals(full_vertices(x)),
        x0,
        max_nfev=max_nfev,
        x_scale=swing,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    verts = mu0 + full_vertices(sol.x)
    mu_fn = _cosine_interpolation(verts, tau)
    unit = _sampled_from_callables(tau, {i: _constant(1.0) for i in modes.ions}, mu_fn, n_samples, phi_m_rad)
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
        amplitude_rad_s={i: _constant(omega0) for i in modes.ions},
        mu_rad_s=mu_fn,
        phi_s_rad=phi_s_rad,
        phi_m_rad=phi_m_rad,
        kind=kind,
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
    relaxed_null_dimension: int | None = None,
    phi_s_rad: float = 0.0,
    phi_m_rad: float = DEFAULT_MOTION_PHASE_RAD,
    kind: Literal["ms", "light_shift"] = "ms",
) -> ShapedPulse:
    """Blumel 2021's power-optimal stabilized AM: g(t) = sum_n A_n sin(2 pi n t/tau) at one beat note; closure and its first
    K mode-frequency derivatives (d^k alpha_m/d omega_m^k = 0, k <= ``stabilization_order``) form the homogeneous system M A
    = 0 (2N(K + 1) rows), the power-optimal solution is the top eigenvector of the kernel projected on the null space (or, with
    ``relaxed_null_dimension``, on the lowest eigenvectors of Gamma = M^T M), and the scale follows from |chi| = chi_target
    in the plan's pi/4 convention (their pi/8 kernel is half of ours: double it to convert, Section 13)."""
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
        if relaxed_null_dimension is None:
            keep = evals < 1e-20 * max(float(evals[-1]), 1e-300)
            if not np.any(keep):
                raise ClosureError(
                    f"{n_basis} basis functions cannot satisfy {amat.shape[0]} closure rows; add functions or relax"
                )
            null = evecs[:, keep]
        else:
            null = evecs[:, : int(relaxed_null_dimension)]
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

    def g_fn(tau_s: float, c: np.ndarray = coeffs) -> float:
        return float(sum(c[n] * math.sin(2.0 * math.pi * (n + 1) * tau_s / tau) for n in range(len(c))))

    wf = waveform_from_callables(
        tau,
        modes,
        ints,
        amplitude_rad_s={i: g_fn for i in modes.ions},
        mu_rad_s=TWO_PI * mu_hz,
        phi_s_rad=phi_s_rad,
        phi_m_rad=phi_m_rad,
        kind=kind,
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
            "zero_temperature_infidelity": 0.8 * ints.residual_error(modes.subset(modes.modes)),
            **{f"A_{n + 1}": float(c) for n, c in enumerate(coeffs)},
        },
    )


def frequency_derivative_residuals(
    env: SampledEnvelope, modes: GateModes, ion: int, mode: int, *, orders: int, kernel: Kernel = "rwa"
) -> tuple[complex, ...]:
    """(d^k alpha/d omega_m^k)(tau) for k = 0..orders: the stabilization quantities Blumel's rows null (Section 4.4.3)."""
    k = modes.index(mode)
    f = _kernel_samples(kernel, modes.omega_rad_s[k], env.times_s, env.beat_phase_rad, env.phi_m_rad)
    out = []
    for order in range(orders + 1):
        val = simpson(env.amplitude_rad_s[ion] * (1j * env.times_s) ** order * f, x=env.times_s)
        out.append(complex(1j * modes.eta[ion][k] * val))
    return tuple(out)


__all__ = [
    "CHI_MAXIMAL_RAD",
    "DEFAULT_MOTION_PHASE_RAD",
    "SINE_MOTION_PHASE_RAD",
    "ClosureError",
    "Envelope",
    "GateIntegrals",
    "GateModes",
    "Kernel",
    "SampledEnvelope",
    "SegmentedEnvelope",
    "ShapedPulse",
    "closure_duration_s",
    "closure_rabi_rad_s",
    "closure_ratio",
    "envelope_of",
    "frequency_derivative_residuals",
    "gate_modes",
    "integrals",
    "integrals_sampled",
    "integrals_segmented",
    "scaled",
    "segment_count",
    "solve_amplitude_modulation",
    "solve_fourier_amplitude_modulation",
    "solve_frequency_modulation",
    "square_pulse_chi",
    "symmetric_pulse",
    "total_chi",
    "trajectory_sampled",
    "waveform_from_callables",
    "waveform_from_segmented",
    "waveform_integrals",
]
