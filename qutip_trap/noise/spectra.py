"""Noise spectra and slow-parameter records (PLAN.md Sections 6.1 to 6.7, 6.9; Section 13; Appendix E).

Sidedness (Section 13, rows "Electric-field noise density" and "Filter-function normalization stack"): the
heating formula uses a SINGLE-sided S_E, while every ``NoiseSpectrum`` object is TWO-sided in angular
frequency with the e^{-i omega t} kernel; the conversion is explicit at the boundary and the field
``sidedness`` exists so that it cannot be silently mixed. Under that kernel the variance of the process is
(1/2 pi) int_{-inf}^{inf} S d omega = (1/pi) int_0^inf S d omega (``NoiseSpectrum.variance``), and the
single-sided per-hertz density a laboratory quotes is exactly 2 S(omega = 2 pi f) with no further 2 pi
(``trap/heating.single_sided_from_two_sided``).

Routing (Section 6.1, M7): the TABULATED band of a spectrum is the sampled-trajectory route (d), synthesized on
a fixed grid per dynamical sample (``noise/processes.py``); ``white_level`` is the flat two-sided density
above the tabulated band, which the model routes to a Lindblad collapse operator (route b) with the
Section 13 normalizations; quasi-static parameters (route c) are the ``Drift`` records of the NoiseModel.
Nothing is routed by a heuristic on the spectrum's shape: the split is declared by the two fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NoiseSpectrum:
    """ALWAYS two-sided, angular frequency, exp(-i omega t) kernel (Section 13)."""

    omega_rad_s: np.ndarray
    S: np.ndarray
    unit: str
    """The unit of S, e.g. "(V/m)^2/(rad/s)" or "(rad/s)^2/(rad/s)"."""
    sidedness: Literal["two-sided"] = "two-sided"
    white_level: float = 0.0
    """The flat two-sided density ABOVE the tabulated band (same unit as S): the white component that Section 6.1
    routes to a Lindblad operator; 0 when the spectrum has no white part (M7)."""
    provenance: tuple[str, ...] = ()
    """Where this density came from: one entry per apparatus, in the form "author year, species, N ions" (Section 6.1
    tags this **[extracted]**). Section 6.1 requires the report to say when a budget mixes machines, because "Fang's
    614 quanta/s ... Cetina's 88(6) quanta/s ... Trout's simulated 25 quanta/s" are different traps; ``apparatus_count``
    over every rate in play is what ``Diagnostics.approximations`` emits. Empty = undeclared, which the report says."""

    @property
    def apparatus_count(self) -> int:
        """The number of distinct apparatus this density is stitched from (0 when undeclared)."""
        return len(set(self.provenance))

    def __post_init__(self) -> None:
        w = np.asarray(self.omega_rad_s, dtype=float)
        s = np.asarray(self.S, dtype=float)
        if w.ndim != 1 or w.shape != s.shape:
            raise ValueError("omega_rad_s and S must be one-dimensional arrays of the same length")
        if np.any(np.diff(w) <= 0.0):
            raise ValueError("omega_rad_s must be strictly increasing")
        if np.any(s < 0.0) or self.white_level < 0.0:
            raise ValueError("a power spectral density is non-negative")
        if self.sidedness != "two-sided":
            raise ValueError(
                "NoiseSpectrum is always two-sided (Section 13); convert single-sided data at the boundary"
            )

    # ---- evaluation ------------------------------------------------------------------------------------------------

    @property
    def omega_max_rad_s(self) -> float:
        """The top of the tabulated band, |omega| beyond which only ``white_level`` remains."""
        return float(np.max(np.abs(np.asarray(self.omega_rad_s, dtype=float))))

    def tabulated(self, omega_rad_s: np.ndarray | float) -> np.ndarray | float:
        """The tabulated density at |omega| (S(-omega) = S(omega)), zero outside the band; no white level."""
        w = np.abs(np.asarray(omega_rad_s, dtype=float))
        grid = np.abs(np.asarray(self.omega_rad_s, dtype=float))
        if np.any(np.asarray(self.omega_rad_s, dtype=float) < 0.0):
            # a symmetric tabulation: fold onto |omega| and interpolate the non-negative half
            order = np.argsort(grid)
            grid, s = grid[order], np.asarray(self.S, dtype=float)[order]
            out = np.interp(w, grid, s, left=float(s[0]), right=0.0)
        else:
            out = np.interp(w, grid, np.asarray(self.S, dtype=float), left=float(self.S[0]), right=0.0)
        return float(out) if np.ndim(omega_rad_s) == 0 else np.asarray(out)

    def value(self, omega_rad_s: np.ndarray | float) -> np.ndarray | float:
        """The full two-sided density: the tabulated part plus the white level everywhere."""
        tab = self.tabulated(omega_rad_s)
        return tab + self.white_level

    def variance(self) -> float:
        """(1/pi) int_0^inf S_tab d omega: the variance of the sampled (tabulated) process (Section 13 kernel)."""
        w = np.asarray(self.omega_rad_s, dtype=float)
        s = np.asarray(self.S, dtype=float)
        if np.any(w < 0.0):
            return float(np.trapezoid(s, w) / (2.0 * math.pi))
        return float(np.trapezoid(s, w) / math.pi)

    def is_zero(self) -> bool:
        return bool(np.all(np.asarray(self.S) == 0.0)) and self.white_level == 0.0


# ---- constructors (two-sided, angular) ------------------------------------------------------------------------------------


def white_spectrum(level: float, unit: str) -> NoiseSpectrum:
    """A purely white process: no tabulated band (an empty two-point zero band), the density in ``white_level``."""
    return NoiseSpectrum(np.array([0.0, 1.0]), np.zeros(2), unit, white_level=float(level))


def ou_spectrum(
    variance: float, tau_c_s: float, unit: str, *, omega_max_rad_s: float | None = None, n: int = 4001
) -> NoiseSpectrum:
    """The Ornstein-Uhlenbeck (Lorentzian) two-sided density S(omega) = 2 sigma^2 tau_c/(1 + omega^2 tau_c^2), whose
    (1/pi) int_0^inf is exactly sigma^2 (the autocorrelation sigma^2 e^{-|t|/tau_c}); tabulated to ``omega_max``
    (default 200/tau_c, which holds 99.7 % of the variance)."""
    if variance < 0.0 or tau_c_s <= 0.0:
        raise ValueError("variance is non-negative and the correlation time positive")
    w_max = 200.0 / tau_c_s if omega_max_rad_s is None else float(omega_max_rad_s)
    w = np.linspace(0.0, w_max, int(n))
    s = 2.0 * variance * tau_c_s / (1.0 + (w * tau_c_s) ** 2)
    return NoiseSpectrum(w, s, unit)


def gaussian_spectrum(
    rms: float, sigma_rad_s: float, unit: str, *, omega_max_rad_s: float | None = None, n: int = 4001
) -> NoiseSpectrum:
    """S(omega) = sqrt(2 pi) (rms^2/sigma) exp[-omega^2/(2 sigma^2)], the Section 9.15 end-to-end fixture: C(0) = rms^2."""
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
    """S(omega) = S_ref (omega/omega_ref)^-alpha on a log-spaced band [omega_min, omega_max] (the 1/f^alpha family of
    Sections 6.3 and 6.9; the infrared end is the caller's declared cutoff, Section 12)."""
    if omega_min_rad_s <= 0.0 or omega_max_rad_s <= omega_min_rad_s or omega_ref_rad_s <= 0.0:
        raise ValueError("0 < omega_min < omega_max and omega_ref > 0")
    w = np.geomspace(omega_min_rad_s, omega_max_rad_s, int(n))
    s = level_at_ref * (w / omega_ref_rad_s) ** (-alpha)
    return NoiseSpectrum(w, s, unit)


def spectrum_from_single_sided_hz(
    f_hz: np.ndarray, s_single_sided_per_hz: np.ndarray, unit: str, *, white_level_per_hz: float = 0.0
) -> NoiseSpectrum:
    """Ingest a laboratory single-sided per-hertz density: S_two(omega = 2 pi f) = S_single(f)/2 (Section 13; the variance
    int_0^inf S_single df equals (1/pi) int_0^inf S_two d omega exactly, no further 2 pi)."""
    f = np.asarray(f_hz, dtype=float)
    s = np.asarray(s_single_sided_per_hz, dtype=float)
    return NoiseSpectrum(2.0 * math.pi * f, s / 2.0, unit, white_level=white_level_per_hz / 2.0)


@dataclass(frozen=True)
class Drift:
    """A slow parameter: rms amplitude, correlation time, optional servo bandwidth, optional ramp (Section 7.5).

    The unit of ``rms`` and ``rate_per_s`` is the unit of the NoiseModel field that carries the record (documented on
    each field). Successive dynamical samples at times t and t' are correlated as exp(-|t - t'|/tau_s) (an
    Ornstein-Uhlenbeck stationary process sampled at the shot clock, ``NoiseModel.sample_sequence``); ``rate_per_s`` adds a
    deterministic ramp rate x (t - t0) across a run (Section 9.17 row "Shot clock"). ``servo_bandwidth_hz`` (M8) high-passes
    the drift into its residual band: the machine re-locks the parameter with a first-order loop of that bandwidth over the
    shot clock (``NoiseModel.sample_sequence``, ``servo_residual``), the frequency-feedforward mechanism of Section 7.5.
    """

    rms: float
    tau_s: float
    servo_bandwidth_hz: float | None
    rate_per_s: float = 0.0
    """Deterministic ramp (a reference cavity in Hz/s is the dominant optical-qubit drift)."""
    provenance: tuple[str, ...] = ()
    """One entry per apparatus, as on ``NoiseSpectrum`` (Section 6.1 **[extracted]**)."""

    def __post_init__(self) -> None:
        if self.rms < 0.0 or self.tau_s <= 0.0:
            raise ValueError("rms is non-negative and the correlation time positive")

    @property
    def apparatus_count(self) -> int:
        return len(set(self.provenance))

    @property
    def quiet(self) -> bool:
        return self.rms == 0.0 and self.rate_per_s == 0.0


@dataclass(frozen=True)
class Mains:
    """Mains pickup: amplitude per harmonic and the line-trigger policy (Section 6.3)."""

    line_hz: float
    amplitudes_t: dict[int, float]
    """Harmonic number -> magnetic-field amplitude in tesla."""
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
        """B(t) = sum_h A_h cos(2 pi h f_line t + phi_h + phi_trigger), tesla; the per-shot trigger phase is uniform for a
        free-running sequence and zero for a line-triggered one (Section 6.3)."""
        t = np.asarray(t_s, dtype=float)
        out = np.zeros_like(t)
        for h, amp in self.amplitudes_t.items():
            out = out + amp * np.cos(
                2.0 * math.pi * h * self.line_hz * t + self.phases_rad[h] + trigger_phase_rad
            )
        return float(out) if np.ndim(t_s) == 0 else out


@dataclass(frozen=True)
class Collisions:
    """Background-gas collisions as instantaneous events with an outcome distribution (Section 6.7)."""

    pressure_pa: float
    gas: dict[str, float]
    """Partial-pressure fractions by species, summing to 1."""
    outcome_probabilities: dict[Literal["heating_kick", "reorder", "loss", "dark_ion"], float]
    """Conditional on a collision, summing to 1."""
    temperature_k: float = 300.0
    """The gas temperature that converts the pressure to a density (300 K unless the chamber is cryogenic)."""

    kick_distribution: Literal["exponential", "thermal_maxwell"] = "exponential"
    """The shape of the heating kick's energy distribution (Section 6.7: "a heating kick drawn from a configured energy
    distribution whose scale is the neutral's thermal energy times the mass ratio, tens to thousands of quanta").

    ``"exponential"`` draws E from an exponential of that mean (the maximum-entropy choice at fixed mean energy);
    ``"thermal_maxwell"`` draws it from the chi-squared-with-3-degrees-of-freedom Maxwell energy distribution of the same
    mean. No source quantifies the shape, so it is a device input **[background]** and the scale is the physics.
    """
    kick_scale_multiplier: float = 1.0
    """Multiplies the k_B T m_gas/m_ion energy scale, for a chamber whose measured kicks depart from the Langevin estimate."""
    reorder_permutations: tuple[tuple[int, ...], ...] = ()
    """The configured permutation distribution of Section 6.7 ("sampled from a configured permutation distribution"),
    as candidate orders of the ion labels; drawn uniformly. Empty = the adjacent transposition at the struck ion, the
    single-swap default a Langevin kick most often produces."""

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


__all__ = [
    "Collisions",
    "Drift",
    "Mains",
    "NoiseSpectrum",
    "gaussian_spectrum",
    "ou_spectrum",
    "power_law_spectrum",
    "spectrum_from_single_sided_hz",
    "white_spectrum",
]
