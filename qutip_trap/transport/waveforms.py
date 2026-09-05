"""Voltage waveforms and the control-line filter chain (PLAN.md Section 4.6; Appendix E).

Naming (Appendix E, closing list): the M12 voltage record is ``VoltageWaveform``; the calibrated
entangling-pulse record of Sections 7 and 8 keeps the name ``Waveform`` (``qutip_trap.control.table``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.species.model import Species
    from qutip_trap.transport.budget import TransportBudget

M12 = "milestone M12 (PLAN.md Section 4.6; not scheduled for the first release)"


@dataclass(frozen=True)
class FilterStage:
    kind: Literal["zoh", "fir", "butterworth", "single_pole"]
    order: int | None
    corner_hz: float | None
    taps: np.ndarray | None

    def response(self, f_hz: np.ndarray) -> np.ndarray:
        """Complex transfer function, magnitude and phase."""
        raise NotImplementedError(f"FilterStage.response is {M12}")

    def group_delay_s(self) -> float:
        raise NotImplementedError(f"FilterStage.group_delay_s is {M12}")


@dataclass(frozen=True)
class VoltageWaveform:
    kind: Literal["transport", "split", "merge", "swap", "junction_traverse"]
    shape: Literal["sine", "erf", "blackman", "bezier", "linear", "sin2_distance", "quintic_distance"]
    duration_s: float
    params: dict[str, float]
    dac_step_s: float
    """Zero-order hold; the J*R_DAC = f_z knob."""
    filter_chain: tuple[FilterStage, ...]
    """FIR then analog low-pass, applied in order."""
    voltages_v: dict[str, np.ndarray] | None
    well_center_m: Callable[[float], float] | np.ndarray | None
    omega_axial_hz: Callable[[float], float] | np.ndarray | None
    distance_m: Callable[[float], float] | np.ndarray | None
    quartic_v_per_m4: float | None
    quadratic_v_per_m2: Callable[[float], float] | None
    tilt_v_per_m: float
    """gamma, including the stray offset gamma_prime; a coefficient of a VOLT potential (Section 13)."""

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or self.dac_step_s <= 0.0:
            raise ValueError("duration_s and dac_step_s must be positive")

    def delivered(self) -> VoltageWaveform:
        """The same object after the filter chain; x_well and omega are always derived from THIS."""
        raise NotImplementedError(f"VoltageWaveform.delivered is {M12}")

    def excitation(self, species: Species, mode_hz: float) -> TransportBudget:
        raise NotImplementedError(f"VoltageWaveform.excitation is {M12}")


__all__ = ["FilterStage", "VoltageWaveform"]
