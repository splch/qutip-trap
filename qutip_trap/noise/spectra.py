"""Noise spectra and slow-parameter records (PLAN.md Section 6).

Every ``NoiseSpectrum`` is two-sided in angular frequency with the e^{-i omega t} kernel: the variance is
(1/2 pi) int_{-inf}^{inf} S d omega = (1/pi) int_0^inf S d omega, and the single-sided per-hertz density a laboratory
quotes is 2 S(omega = 2 pi f). The tabulated band is sampled as a trajectory, ``white_level`` (the flat density above the
band) becomes a Lindblad operator, and a ``Drift`` is a quasi-static parameter drawn once per dynamical sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NoiseSpectrum:
    """A two-sided power spectral density in angular frequency (e^{-i omega t} kernel)."""

    omega_rad_s: np.ndarray
    S: np.ndarray
    unit: str
    """The unit of S, e.g. "(V/m)^2/(rad/s)" or "(rad/s)^2/(rad/s)"."""
    white_level: float = 0.0
    """The flat two-sided density above the tabulated band (unit of S), routed to a Lindblad operator."""
    provenance: tuple[str, ...] = ()
    """The apparatus this density comes from, one entry each ("author year, species, N ions"); empty = undeclared."""

    def __post_init__(self) -> None:
        w = np.asarray(self.omega_rad_s, dtype=float)
        s = np.asarray(self.S, dtype=float)
        if w.ndim != 1 or w.shape != s.shape:
            raise ValueError("omega_rad_s and S must be one-dimensional arrays of the same length")
        if np.any(np.diff(w) <= 0.0):
            raise ValueError("omega_rad_s must be strictly increasing")
        if np.any(s < 0.0) or self.white_level < 0.0:
            raise ValueError("a power spectral density is non-negative")

    @property
    def omega_max_rad_s(self) -> float:
        """The top of the tabulated band, |omega| beyond which only ``white_level`` remains."""
        return float(np.max(np.abs(np.asarray(self.omega_rad_s, dtype=float))))

    def tabulated(self, omega_rad_s: np.ndarray | float) -> np.ndarray | float:
        """The tabulated density at |omega|, zero above the band, without the white level."""
        w = np.abs(np.asarray(omega_rad_s, dtype=float))
        grid = np.asarray(self.omega_rad_s, dtype=float)
        s = np.asarray(self.S, dtype=float)
        if np.any(grid < 0.0):  # a symmetric tabulation: fold onto |omega|
            order = np.argsort(np.abs(grid))
            grid, s = np.abs(grid)[order], s[order]
        out = np.interp(w, grid, s, left=float(s[0]), right=0.0)
        return float(out) if np.ndim(omega_rad_s) == 0 else np.asarray(out)

    def value(self, omega_rad_s: np.ndarray | float) -> np.ndarray | float:
        """The full density: the tabulated part plus the white level everywhere."""
        return self.tabulated(omega_rad_s) + self.white_level

    def variance(self) -> float:
        """(1/pi) int_0^inf S_tab d omega, the variance of the sampled (tabulated) process."""
        w = np.asarray(self.omega_rad_s, dtype=float)
        s = np.asarray(self.S, dtype=float)
        if np.any(w < 0.0):
            return float(np.trapezoid(s, w) / (2.0 * math.pi))
        return float(np.trapezoid(s, w) / math.pi)

    def is_zero(self) -> bool:
        return bool(np.all(np.asarray(self.S) == 0.0)) and self.white_level == 0.0


def white_spectrum(level: float, unit: str) -> NoiseSpectrum:
    """A purely white process: a two-point zero band and the density in ``white_level``."""
    return NoiseSpectrum(np.array([0.0, 1.0]), np.zeros(2), unit, white_level=float(level))


def ou_spectrum(
    variance: float, tau_c_s: float, unit: str, *, omega_max_rad_s: float | None = None, n: int = 4001
) -> NoiseSpectrum:
    """The Ornstein-Uhlenbeck (Lorentzian) density S = 2 sigma^2 tau_c/(1 + omega^2 tau_c^2), autocorrelation
    sigma^2 e^{-|t|/tau_c}, tabulated to ``omega_max`` (default 200/tau_c, 99.7 % of the variance)."""
    if variance < 0.0 or tau_c_s <= 0.0:
        raise ValueError("variance is non-negative and the correlation time positive")
    w_max = 200.0 / tau_c_s if omega_max_rad_s is None else float(omega_max_rad_s)
    w = np.linspace(0.0, w_max, int(n))
    s = 2.0 * variance * tau_c_s / (1.0 + (w * tau_c_s) ** 2)
    return NoiseSpectrum(w, s, unit)


def gaussian_spectrum(
    rms: float, sigma_rad_s: float, unit: str, *, omega_max_rad_s: float | None = None, n: int = 4001
) -> NoiseSpectrum:
    """S = sqrt(2 pi) (rms^2/sigma) exp[-omega^2/(2 sigma^2)], autocorrelation rms^2 exp(-sigma^2 t^2/2)."""
    if rms < 0.0 or sigma_rad_s <= 0.0:
        raise ValueError("rms is non-negative and sigma positive")
    w_max = 8.0 * sigma_rad_s if omega_max_rad_s is None else float(omega_max_rad_s)
    w = np.linspace(0.0, w_max, int(n))
    s = math.sqrt(2.0 * math.pi) * rms**2 / sigma_rad_s * np.exp(-(w**2) / (2.0 * sigma_rad_s**2))
    return NoiseSpectrum(w, s, unit)


def power_law_spectrum(
    level_at_ref: float,
    omega_ref_rad_s: float,
    alpha: float,
    unit: str,
    *,
    omega_min_rad_s: float,
    omega_max_rad_s: float,
    n: int = 2001,
) -> NoiseSpectrum:
    """S = S_ref (omega/omega_ref)^-alpha on a log-spaced band [omega_min, omega_max]; omega_min is the declared infrared
    cutoff."""
    if omega_min_rad_s <= 0.0 or omega_max_rad_s <= omega_min_rad_s or omega_ref_rad_s <= 0.0:
        raise ValueError("0 < omega_min < omega_max and omega_ref > 0")
    w = np.geomspace(omega_min_rad_s, omega_max_rad_s, int(n))
    s = level_at_ref * (w / omega_ref_rad_s) ** (-alpha)
    return NoiseSpectrum(w, s, unit)


@dataclass(frozen=True)
class Drift:
    """A slow parameter: rms amplitude, correlation time, optional servo bandwidth and ramp (PLAN.md Section 7.5).

    ``rms`` and ``rate_per_s`` are in the unit of the NoiseModel field that carries the record. Samples at shot-clock times
    t and t' correlate as exp(-|t - t'|/tau_s); ``rate_per_s`` adds a ramp rate x (t - t0); ``servo_bandwidth_hz``
    high-passes the drift through a first-order re-locking loop (``servo_residual``).
    """

    rms: float
    tau_s: float
    servo_bandwidth_hz: float | None
    rate_per_s: float = 0.0
    provenance: tuple[str, ...] = ()
    """The apparatus this record comes from, as on ``NoiseSpectrum``."""

    def __post_init__(self) -> None:
        if self.rms < 0.0 or self.tau_s <= 0.0:
            raise ValueError("rms is non-negative and the correlation time positive")

    @property
    def quiet(self) -> bool:
        return self.rms == 0.0 and self.rate_per_s == 0.0


@dataclass(frozen=True)
class Mains:
    """Mains pickup: the magnetic-field amplitude (T) and phase per harmonic, and the line-trigger policy."""

    line_hz: float
    amplitudes_t: dict[int, float]
    phases_rad: dict[int, float]
    trigger: Literal["free_running", "line_triggered"]

    def __post_init__(self) -> None:
        if self.line_hz <= 0.0:
            raise ValueError("line frequency must be positive")
        if set(self.amplitudes_t) != set(self.phases_rad):
            raise ValueError("every harmonic needs an amplitude and a phase")
        if any(h <= 0 for h in self.amplitudes_t):
            raise ValueError("harmonic numbers are positive integers")

    @property
    def omega_max_rad_s(self) -> float:
        return 2.0 * math.pi * self.line_hz * max(self.amplitudes_t) if self.amplitudes_t else 0.0

    def field_t(self, t_s: np.ndarray | float, trigger_phase_rad: float) -> np.ndarray | float:
        """B(t) = sum_h A_h cos(2 pi h f_line t + phi_h + phi_trigger) in tesla."""
        t = np.asarray(t_s, dtype=float)
        out = np.zeros_like(t)
        for h, amp in self.amplitudes_t.items():
            out = out + amp * np.cos(
                2.0 * math.pi * h * self.line_hz * t + self.phases_rad[h] + trigger_phase_rad
            )
        return float(out) if np.ndim(t_s) == 0 else out


@dataclass(frozen=True)
class Collisions:
    """Background-gas collisions as instantaneous events with an outcome distribution (PLAN.md Section 6.7)."""

    pressure_pa: float
    gas: dict[str, float]
    """Partial-pressure fractions by species, summing to 1."""
    outcome_probabilities: dict[Literal["heating_kick", "reorder", "loss", "dark_ion"], float]
    """Conditional on a collision, summing to 1."""
    temperature_k: float = 300.0
    kick_distribution: Literal["exponential", "thermal_maxwell"] = "exponential"
    """The shape of the kick-energy distribution, whose mean is k_B T m_gas/m_ion: an exponential or the chi^2_3 Maxwell
    energy distribution. No source quantifies the shape, so it is a device input."""
    kick_scale_multiplier: float = 1.0
    """Multiplies the k_B T m_gas/m_ion energy scale."""
    reorder_permutations: tuple[tuple[int, ...], ...] = ()
    """Candidate orders of the ion labels, drawn uniformly on a reorder; empty = the adjacent swap at the struck ion."""

    def __post_init__(self) -> None:
        if self.pressure_pa < 0.0 or self.temperature_k <= 0.0:
            raise ValueError("pressure is non-negative and the temperature positive")
        for name, table in (("gas", self.gas), ("outcome_probabilities", self.outcome_probabilities)):
            total = sum(table.values())
            if table and abs(total - 1.0) > 1e-9:
                raise ValueError(f"{name} fractions must sum to 1, got {total}")
        if self.kick_scale_multiplier <= 0.0:
            raise ValueError("kick_scale_multiplier is positive")
        for perm in self.reorder_permutations:
            if sorted(perm) != list(range(len(perm))):
                raise ValueError(f"reorder_permutations entries must be permutations of range(N), got {perm}")
