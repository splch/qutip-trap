"""Scattering-rate model, bright/dark pumping rates, dark states, shelving (PLAN.md Section 8.1; M3a/M5).

The scattering rate object is the multi-level optical-Bloch steady state of Section 4.2.8 (milestone M3a);
its photon rate is asserted against the saturation ceiling n_e/(n_e + n_g) of the closed manifold: Gamma/4 for
the 171Yb+ F = 1 -> F' = 0 detection cycle, Gamma/3 for a Lambda system, Gamma/2 for a two-level atom (Section 13).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M5 = "milestone M5 (readout/fluorescence.py, PLAN.md Section 8.1)"


def saturation_ceiling(n_ground: int, n_excited: int) -> float:
    """P_f^max = n_e/(n_e + n_g), the equal-population ceiling of a closed manifold (Section 13; Berkeland Eqs. 13-14, 21)."""
    if n_ground <= 0 or n_excited <= 0:
        raise ValueError("a closed manifold has at least one ground and one excited state")
    return n_excited / (n_excited + n_ground)


@dataclass(frozen=True)
class DarkStateReport:
    """Returned by the CPT solve so the ceiling is never assumed (Section 8.1)."""

    n_ground: int
    n_excited: int
    ceiling: float
    """n_e/(n_e + n_g); the photon rate must satisfy Gamma*P_f <= Gamma*ceiling."""
    dark_dimension: int
    dark_basis: np.ndarray
    """(dark_dimension, n_ground) complex amplitudes c_m, STRAIGHT pairing (Section 13)."""
    delta_over_omega: float
    theta_be_deg: float
    """Angle between the linear polarization and B; optimum 54.7356 degrees."""
    raman_zero_margin_hz: float | None

    def __post_init__(self) -> None:
        expected = saturation_ceiling(self.n_ground, self.n_excited)
        if abs(self.ceiling - expected) > 1e-12:
            raise ValueError(f"ceiling must be n_e/(n_e + n_g) = {expected}, got {self.ceiling}")


def scattering_rate(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"scattering_rate is {M5}")


__all__ = ["DarkStateReport", "saturation_ceiling", "scattering_rate"]
