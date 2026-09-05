"""rf drive and dc electrode voltages (PLAN.md Sections 4.1.1, 4.1.6; Section 13; milestone M1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RfDrive:
    """The rf drive of a Paul trap.

    ``voltage_peak_v`` is the PEAK of a cos(Omega_rf t) drive (Section 13, row "rf amplitude and
    pseudopotential"): reading a stored peak amplitude as rms doubles the depth and inflates every rf-set
    secular frequency by sqrt 2; reading it as peak-to-peak quarters the depth and halves those frequencies.
    ``frequency_hz`` is Omega_rf/2pi, an ordinary frequency (Section 5.6).
    """

    voltage_peak_v: float
    frequency_hz: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("rf frequency must be positive")


@dataclass(frozen=True)
class DcElectrodes:
    """dc voltages by electrode name (V)."""

    voltages_v: dict[str, float]


__all__ = ["DcElectrodes", "RfDrive"]
