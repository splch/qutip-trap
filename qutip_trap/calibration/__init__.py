"""Calibration emulation: simulated experiments -> CalibrationTable (PLAN.md Section 7.5; milestones M4, M5, M6, M8).

Sits ABOVE control/ and dynamics/ (Section 3.2): it consumes the experiments package and produces the plain data object
``control.table.CalibrationTable``. The default calibration is the SURROGATE of Section 7.5 (``surrogate=True``): the
device's derived values as ``seed`` entries, the closed-form entangling waveforms corrected by exact spot checks
(``calibration.entangling``, M4), and the detection threshold and window from the simulated readout model
(``calibration.readout``, M5), assembled by ``calibration.surrogate`` (M6). The full simulated-experiment path
(``surrogate=False``) follows the dependency graph field -> micromotion -> modes -> rabi, stark -> crosstalk -> ms ->
detection, heating, refusing a downstream fit whose upstream entry is ``uncalibrated`` (Appendix E); it is milestone M8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device

M8 = "milestone M8 (calibration/, PLAN.md Section 7.5: the full simulated-experiment calibration)"

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
    **kwargs: Any,
) -> CalibrationTable:
    """The CalibrationTable of ``device`` at ``t0_s``: the Section 7.5 surrogate (default), or the M8 simulated experiments.

    Keyword arguments beyond Appendix E's are passed to :func:`qutip_trap.calibration.surrogate.surrogate_table` (gate
    drives, the pairs to calibrate, the number of detection records, the spot-check options).
    """
    unknown = [e for e in experiments if e != "all" and e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown calibration experiments {unknown}; known: {EXPERIMENTS}")
    if not surrogate:
        raise NotImplementedError(f"calibrate(surrogate=False) is {M8}")
    from qutip_trap.calibration.surrogate import surrogate_table

    return surrogate_table(device, seed=seed, t0_s=t0_s, **kwargs).table


__all__ = ["EXPERIMENTS", "M8", "calibrate"]
