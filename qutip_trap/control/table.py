"""The calibration table and its entries (PLAN.md Sections 3.2, 7.5; Appendix E).

``CalibrationTable`` is plain data: ``control.schedule`` reads it and never writes it, and ``control`` never
imports ``calibration`` (Appendix E). Every entry carries its status, the experiment that produced it, its
provenance id, its age and the noise sample it was fitted under (Section 7.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Leg = Literal["red", "blue"]


@dataclass(frozen=True)
class CalEntry:
    value: float
    uncertainty: float
    status: Literal["seed", "calibrated", "uncalibrated"]
    experiment: str
    provenance_id: str
    fitted_at_s: float
    """Calibration age: the laboratory time the fit was made at (Section 7.5)."""
    sample_id: int
    """The noise sample the entry was fitted under."""

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")


@dataclass(frozen=True)
class Segment:
    """One segment of a calibrated entangling pulse (Appendix E, ``Waveform.segments``)."""

    duration_s: float
    amplitude_hz: dict[tuple[int, Leg], float]
    """Omega per (ion, leg), leg in {red, blue}; per-ion imbalance is a CalibrationTable entry."""
    phase_rad: dict[tuple[int, Leg], float]
    detuning_hz: dict[Leg, float]

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("segment duration must be positive")
        if set(self.amplitude_hz) != set(self.phase_rad):
            raise ValueError("amplitude_hz and phase_rad must be indexed by the same (ion, leg) pairs")


@dataclass(frozen=True)
class Waveform:
    """A calibrated entangling pulse: what the scheduler plays (Sections 4.4.3, 7.4)."""

    segments: tuple[Segment, ...] | None
    fourier: tuple[complex, ...] | None
    """Or Fourier coefficients of the modulation (Section 4.4.3)."""
    duration_s: float
    phi_s: CalEntry
    phi_m: CalEntry
    chi_m: dict[int, float]
    """Per-mode entangling angle at closure."""
    alpha_m: dict[int, complex]
    """Per-mode residual displacement at closure."""

    def __post_init__(self) -> None:
        if (self.segments is None) == (self.fourier is None):
            raise ValueError("a Waveform is either segmented or Fourier-parameterized, not both or neither")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.segments is not None:
            total = sum(s.duration_s for s in self.segments)
            if not math.isclose(total, self.duration_s, rel_tol=1e-9, abs_tol=1e-15):
                raise ValueError(f"segment durations sum to {total}, not duration_s = {self.duration_s}")

    @classmethod
    def symmetric(cls, *args: object, **kwargs: object) -> Waveform:
        """The equal-envelope symmetric-detuning constructor shortcut (Appendix E); milestone M4/M8."""
        raise NotImplementedError("Waveform.symmetric is milestone M4 (PLAN.md Section 4.4.1)")


@dataclass(frozen=True)
class CalibrationTable:
    device_hash: str
    seed: int
    surrogate: bool
    """Closed-form surrogate with spot checks, or full simulated experiments (Section 7.5)."""
    qubit_freq: dict[int, CalEntry]
    """From the Ramsey-frequency experiment; the scheduler never reads the true value."""
    rabi: dict[tuple[int, int], CalEntry]
    """(ion, beam)."""
    stark: dict[tuple[int, int], CalEntry]
    crosstalk: dict[tuple[int, int], CalEntry]
    modes: dict[int, CalEntry]
    nbar: dict[int, CalEntry]
    ms: dict[tuple[int, int], Waveform]
    field: CalEntry
    micromotion: dict[str, CalEntry]
    """Shim voltages and residual beta per beam direction."""
    detection: dict[str, CalEntry]
    heating: dict[int, CalEntry]


__all__ = ["CalEntry", "CalibrationTable", "Leg", "Segment", "Waveform"]
