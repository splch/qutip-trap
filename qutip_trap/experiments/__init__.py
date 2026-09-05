"""User-facing simulated experiments built on the PulseEngine protocol (PLAN.md Sections 7.5, 7.9; M8).

Each returns data plus the fitted parameters with uncertainties; ``calibration/`` uses them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.device.model import Device

M8 = "milestone M8 (experiments/, PLAN.md Section 7.5)"


@dataclass(frozen=True)
class ExperimentResult:
    data: np.ndarray
    fitted: dict[str, tuple[float, float]]
    """Parameter -> (value, uncertainty)."""
    model: str
    provenance_id: str


def rabi_scan(device: Device, ion: int, durations_s: Sequence[float], **kw: Any) -> ExperimentResult:
    raise NotImplementedError(f"rabi_scan is {M8}")


def ramsey(device: Device, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    raise NotImplementedError(f"ramsey is {M8}")


def ramsey_frequency(device: Device, ion: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    """Qubit frequency for the table; the scheduler never reads the true value (Section 7.5)."""
    raise NotImplementedError(f"ramsey_frequency is {M8}")


def micromotion_scan(
    device: Device,
    ion: int,
    beam: int,
    shim_ranges_v: dict[str, tuple[float, float]],
    method: str = "rf_photon_correlation",
    **kw: Any,
) -> ExperimentResult:
    raise NotImplementedError(f"micromotion_scan is {M8}")


def sideband_spectroscopy(
    device: Device, ion: int, detunings_hz: Sequence[float], **kw: Any
) -> ExperimentResult:
    raise NotImplementedError(f"sideband_spectroscopy is {M8}")


def ms_scan(
    device: Device,
    pair: tuple[int, int],
    amplitudes: Sequence[float],
    detunings_hz: Sequence[float],
    **kw: Any,
) -> ExperimentResult:
    raise NotImplementedError(f"ms_scan is {M8}")


def parity_scan(
    device: Device, pair: tuple[int, int], analysis_phases_rad: Sequence[float], **kw: Any
) -> ExperimentResult:
    raise NotImplementedError(f"parity_scan is {M8}")


def heating_rate(device: Device, mode: int, delays_s: Sequence[float], **kw: Any) -> ExperimentResult:
    raise NotImplementedError(f"heating_rate is {M8}")


def detection_histogram(device: Device, ion: int, n_records: int, **kw: Any) -> ExperimentResult:
    raise NotImplementedError(f"detection_histogram is {M8}")


__all__ = [
    "ExperimentResult",
    "detection_histogram",
    "heating_rate",
    "micromotion_scan",
    "ms_scan",
    "parity_scan",
    "rabi_scan",
    "ramsey",
    "ramsey_frequency",
    "sideband_spectroscopy",
]
