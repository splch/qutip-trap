"""Calibration emulation: simulated experiments -> CalibrationTable (PLAN.md Section 7.5; milestone M8).

Sits ABOVE control/ and dynamics/ (Section 3.2): it consumes the experiments package and produces the plain
data object ``control.table.CalibrationTable``; the dependency graph is field -> micromotion -> modes -> rabi,
stark -> crosstalk -> ms -> detection, heating, and a downstream fit whose upstream entry is ``uncalibrated``
is refused (Appendix E).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device

M8 = "milestone M8 (calibration/, PLAN.md Section 7.5)"

EXPERIMENTS: tuple[str, ...] = (
    "field_scan",
    "micromotion_scan",
    "sideband_spectroscopy",
    "rabi_scan",
    "stark_scan",
    "crosstalk_scan",
    "ms_scan",
    "parity_scan",
    "detection_histogram",
    "heating_rate",
    "crystal_image",
    "ramsey_frequency",
)


def calibrate(
    device: Device,
    *,
    seed: int = 0,
    experiments: tuple[str, ...] = ("all",),
    surrogate: bool = True,
    t0_s: float = 0.0,
) -> CalibrationTable:
    unknown = [e for e in experiments if e != "all" and e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown calibration experiments {unknown}; known: {EXPERIMENTS}")
    raise NotImplementedError(f"calibrate is {M8}")


__all__ = ["EXPERIMENTS", "calibrate"]
