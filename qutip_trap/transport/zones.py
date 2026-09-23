"""Trapping zones of a multi-zone array (PLAN.md Section 4.6; Appendix E, Run 5 additions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Zone:
    """One named trapping site or junction leg of a multi-zone array."""

    name: str
    position_m: tuple[float, float, float]
    kind: Literal["site", "junction_centre", "junction_apex", "channel"]
    omega_hz: tuple[float, float, float] | None
    barrier_ev: float | None
    """q*phi_ps at the apex, in eV (PEAK rf-amplitude convention, Section 13)."""
    axial_anticonfinement: bool
    """True where omega_rf,z is imaginary (apex)."""
    sum_rule_holds: bool
    """Assert omega_x^2 + omega_y^2 + omega_z^2 = 2 omega_rf^2 only where True; False everywhere in a junction."""
    micromotion_z1_m: float | None
    micromotion_convention: Literal["peak", "rms"]
    """Declared, never inferred (Section 13, "Micromotion amplitude convention")."""
    shim_split_hz: float | None
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "site" and self.sum_rule_holds:
            raise ValueError(
                f"zone {self.name!r}: the 2 omega_rf^2 sum rule holds only in straight channels/sites"
            )
