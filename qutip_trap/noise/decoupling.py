"""Dynamical decoupling and filter functions (PLAN.md Section 6.9; Appendix E, Run 5 additions; milestone M7).

Normalization stack pinned by Section 13: S two-sided in angular frequency with the e^{-i omega t} kernel;
prefactor 1/pi one-sided; weight 1/omega^2 paired with the explicit -i omega inside R_ij(omega); the dephasing
variable b(t) sigma_z with no 1/2, so S_b = S_delta/4; chi = (2/pi) int (d omega/omega^2) S_b F, W = e^{-chi};
omega_min = 2 pi/T_total always reported with d ln chi/d ln omega_min beside chi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip_trap.control.composite import CompositePulse
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.experiments import ExperimentResult
    from qutip_trap.noise.spectra import NoiseSpectrum

M7 = "milestone M7 (noise/decoupling.py, PLAN.md Section 6.9)"

Timing = Literal["hahn", "cpmg", "udd", "xy4", "xy8", "kdd", "cdd", "custom"]


@dataclass(frozen=True)
class DecouplingSequence:
    """A net-identity pulse train, or a decoupled idle (Section 6.9)."""

    timing: Timing
    n_pulses: int
    tau_s: float
    """TOTAL duration, INCLUSIVE of the pi-pulse widths."""
    tau_pi_s: float
    """One pulse duration; delta_pi = tau_pi_s/tau_s."""
    deltas: tuple[float, ...]
    """Fractional pulse CENTRES in [0, 1]; len == n_pulses."""
    axes_rad: tuple[float, ...]
    """Per-pulse axis azimuth; all zero for single-axis families."""
    inner: CompositePulse | None = None
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if self.n_pulses < 0 or self.tau_s <= 0.0 or self.tau_pi_s < 0.0:
            raise ValueError("n_pulses >= 0, tau_s > 0, tau_pi_s >= 0")
        if len(self.deltas) != self.n_pulses or len(self.axes_rad) != self.n_pulses:
            raise ValueError("deltas and axes_rad have one entry per pulse")
        if any(not 0.0 <= d <= 1.0 for d in self.deltas):
            raise ValueError("pulse centres are fractions of the total duration")
        if any(b <= a for a, b in zip(self.deltas, self.deltas[1:])):
            raise ValueError("pulse centres must be strictly increasing")

    def is_single_axis(self) -> bool:
        """If False the scalar (-1)^l bookkeeping is INVALID and the full toggling machinery is needed."""
        return (
            all(math.isclose(a, self.axes_rad[0], abs_tol=1e-12) for a in self.axes_rad)
            if self.axes_rad
            else True
        )

    def feasible(self) -> bool:
        """delta_pi <= 2 sin^2[pi/(2n + 2)] and no pulse overlap (Section 6.9)."""
        if self.n_pulses == 0:
            return True
        delta_pi = self.tau_pi_s / self.tau_s
        if delta_pi > 2.0 * math.sin(math.pi / (2.0 * self.n_pulses + 2.0)) ** 2:
            return False
        centres = list(self.deltas)
        if centres[0] - delta_pi / 2.0 < 0.0 or centres[-1] + delta_pi / 2.0 > 1.0:
            return False
        return all(b - a >= delta_pi for a, b in zip(centres, centres[1:]))

    def moments(self, k_max: int = 2) -> tuple[float, ...]:
        """A_k = sum_j (-1)^j delta_j^k for k = 1..k_max (the low-frequency roll-off moments)."""
        return tuple(
            float(sum((-1) ** j * d**k for j, d in enumerate(self.deltas))) for k in range(1, k_max + 1)
        )

    def control_matrix(self, omega_rad_s: np.ndarray) -> np.ndarray:
        """(len(omega), 3, 3) complex R_ij(omega) from the full toggling machinery (Section 13, Green Eqs. 25-30)."""
        raise NotImplementedError(f"DecouplingSequence.control_matrix is {M7}")

    def filter_function(
        self,
        omega_rad_s: np.ndarray,
        quadrature: Literal["dephasing", "amplitude", "universal"] = "dephasing",
    ) -> np.ndarray:
        raise NotImplementedError(f"DecouplingSequence.filter_function is {M7}")

    def rounded_to_clock(self, clock_s: float) -> DecouplingSequence:
        """Snap pulse centres to a grid while preserving A_1; reports the residual moment error."""
        raise NotImplementedError(f"DecouplingSequence.rounded_to_clock is {M7}")


def decoupling_sequence(
    timing: str, n_pulses: int, tau_s: float, tau_pi_s: float, *, inner: CompositePulse | None = None
) -> DecouplingSequence:
    raise NotImplementedError(f"decoupling_sequence is {M7}")


def filter_function(
    device: Device,
    control: CompositePulse | DecouplingSequence | Schedule,
    *,
    quadrature: Literal["dephasing", "amplitude", "universal"] = "dephasing",
    omega_rad_s: np.ndarray | None = None,
    spectrum: NoiseSpectrum | None = None,
    omega_min_rad_s: float | None = None,
    dc_floor: bool = True,
    monte_carlo_samples: int = 0,
) -> ExperimentResult:
    """F(omega), 1 - F_av = (1/pi) int dw/w^2 S F, chi, W, the roll-off order, the dc floor (Section 6.9)."""
    raise NotImplementedError(f"filter_function is {M7}")


__all__ = ["DecouplingSequence", "Timing", "decoupling_sequence", "filter_function"]
