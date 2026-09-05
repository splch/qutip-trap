"""Noise spectra and slow-parameter records (PLAN.md Sections 6.1 to 6.7, 6.9; Section 13; Appendix E).

Sidedness (Section 13, rows "Electric-field noise density" and "Filter-function normalization stack"): the
heating formula uses a SINGLE-sided S_E, while every ``NoiseSpectrum`` object is TWO-sided in angular
frequency with the e^{-i omega t} kernel; the conversion is explicit at the boundary and the field
``sidedness`` exists so that it cannot be silently mixed.
"""

from __future__ import annotations

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

    def __post_init__(self) -> None:
        w = np.asarray(self.omega_rad_s, dtype=float)
        s = np.asarray(self.S, dtype=float)
        if w.ndim != 1 or w.shape != s.shape:
            raise ValueError("omega_rad_s and S must be one-dimensional arrays of the same length")
        if np.any(np.diff(w) <= 0.0):
            raise ValueError("omega_rad_s must be strictly increasing")
        if np.any(s < 0.0):
            raise ValueError("a power spectral density is non-negative")
        if self.sidedness != "two-sided":
            raise ValueError(
                "NoiseSpectrum is always two-sided (Section 13); convert single-sided data at the boundary"
            )


@dataclass(frozen=True)
class Drift:
    """A slow parameter: rms amplitude, correlation time, optional servo bandwidth, optional ramp (Section 7.5)."""

    rms: float
    tau_s: float
    servo_bandwidth_hz: float | None
    rate_per_s: float = 0.0
    """Deterministic ramp (a reference cavity in Hz/s is the dominant optical-qubit drift)."""

    def __post_init__(self) -> None:
        if self.rms < 0.0 or self.tau_s <= 0.0:
            raise ValueError("rms is non-negative and the correlation time positive")


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


@dataclass(frozen=True)
class Collisions:
    """Background-gas collisions as instantaneous events with an outcome distribution (Section 6.7)."""

    pressure_pa: float
    gas: dict[str, float]
    """Partial-pressure fractions by species, summing to 1."""
    outcome_probabilities: dict[Literal["heating_kick", "reorder", "loss", "dark_ion"], float]
    """Conditional on a collision, summing to 1."""

    def __post_init__(self) -> None:
        if self.pressure_pa < 0.0:
            raise ValueError("pressure is non-negative")
        for name, table in (("gas", self.gas), ("outcome_probabilities", self.outcome_probabilities)):
            total = sum(table.values())
            if table and abs(total - 1.0) > 1e-9:
                raise ValueError(f"{name} fractions must sum to 1, got {total}")


__all__ = ["Collisions", "Drift", "Mains", "NoiseSpectrum"]
