"""User-facing simulated experiments built on the PulseEngine protocol (PLAN.md Sections 7.5, 7.9; M2 for the single-ion
Rabi, Ramsey and sideband scans, M4 for the entangling-gate amplitude scan and the parity scan, M5 for the detection
histogram, M8 for the rest: thermometry, mode spectroscopy with the plan's lineshape, the heating-rate measurement, the
Stark, crosstalk and field scans, the MS detuning and phase scans, the micromotion compensation scan and the crystal image).

Each returns an ``ExperimentResult``: data plus the fitted parameters with uncertainties (Appendix E), a ``converged`` flag
the calibration turns into the entry's status, and notes. Every experiment drives the device through the JOINT_EXACT engine
with the drives ``light/`` derives from the beams (never a table value as physics) and reads its populations through the
observation model of ``experiments.fitting`` (``shots`` per point through the readout's errors), so the uncertainties are the
ones a laboratory would quote (Section 7.5); ``calibration/`` consumes them.
"""

from __future__ import annotations

from qutip_trap.experiments.entangling import ms_phase_scan, ms_scan, parity_scan
from qutip_trap.experiments.fitting import (
    FitResult,
    Observation,
    ReadoutErrors,
    fit_lineshape,
    half_rabi_lineshape,
    lineshape_model,
    readout_errors_for,
    sideband_lineshape,
    thermal_rabi_model,
    thermal_rabi_model_fixed_nbar,
    weighted_fit,
)
from qutip_trap.experiments.imaging import crystal_image
from qutip_trap.experiments.light import crosstalk_scan, field_scan, stark_scan
from qutip_trap.experiments.micromotion import (
    correlation_signal,
    device_with_compensation,
    micromotion_scan,
    periodic_scattering,
    signed_beta,
)
from qutip_trap.experiments.motion import heating_rate, mode_spectroscopy, thermometry
from qutip_trap.experiments.readout import detection_histogram
from qutip_trap.experiments.result import (
    RESULT_TYPES,
    CrosstalkScan,
    CrystalImage,
    DetectionHistogram,
    ExperimentResult,
    FieldScan,
    HeatingRateFit,
    MicromotionScan,
    MSScan,
    ParityScan,
    RabiScan,
    RamseyFringe,
    ScanParameters,
    SidebandSpectrum,
    StarkScan,
    ThermometryResult,
    realized_drive,
)
from qutip_trap.experiments.single_ion import rabi_scan, ramsey, ramsey_frequency, sideband_spectroscopy

__all__ = [
    "RESULT_TYPES",
    "CrosstalkScan",
    "CrystalImage",
    "DetectionHistogram",
    "ExperimentResult",
    "FieldScan",
    "FitResult",
    "HeatingRateFit",
    "MSScan",
    "MicromotionScan",
    "ParityScan",
    "RabiScan",
    "RamseyFringe",
    "ScanParameters",
    "SidebandSpectrum",
    "StarkScan",
    "ThermometryResult",
    "Observation",
    "ReadoutErrors",
    "correlation_signal",
    "crosstalk_scan",
    "crystal_image",
    "detection_histogram",
    "device_with_compensation",
    "field_scan",
    "fit_lineshape",
    "half_rabi_lineshape",
    "heating_rate",
    "lineshape_model",
    "micromotion_scan",
    "mode_spectroscopy",
    "ms_phase_scan",
    "ms_scan",
    "parity_scan",
    "periodic_scattering",
    "rabi_scan",
    "realized_drive",
    "ramsey",
    "ramsey_frequency",
    "readout_errors_for",
    "sideband_lineshape",
    "sideband_spectroscopy",
    "signed_beta",
    "stark_scan",
    "thermal_rabi_model",
    "thermal_rabi_model_fixed_nbar",
    "thermometry",
    "weighted_fit",
]
