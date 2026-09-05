"""Composite pulses (PLAN.md Sections 4.3.5, 6.9; Appendix E, Run 5 additions; milestone M2).

Conventions (Section 13): a sequence is stored as (area, phase) in TIME order and folded right to left, the
target azimuth phi ADDED to every entry; R(theta, phi) carries theta/2 in the exponent; order n means a
propagator residual O(eps^{n+1}) and an infidelity slope 2(n + 1); durations tau = sum_l theta_l/Omega (SK1,
BB1 4 pi + theta; CORPSE 4 pi + theta - 4k; CinSK, CinBB 8 pi + theta - 4k; ``check_composite.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

M2 = "milestone M2 (control/composite.py, PLAN.md Section 4.3.5)"

Family = Literal[
    "primitive",
    "SK1",
    "SKn",
    "BB1",
    "NB1",
    "PB1",
    "P2j",
    "N2j",
    "B2j",
    "CORPSE",
    "short_CORPSE",
    "SCROFULOUS",
    "B2CORPSE",
    "CinSK",
    "CinBB",
    "PDn",
    "APn",
    "ToPn",
    "BBn",
]
FAMILIES: tuple[str, ...] = Family.__args__  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CompositePulse:
    """A static-error-compensating single-qubit sequence (Section 4.3.5)."""

    family: Family
    theta_rad: float
    phi_rad: float
    """Target axis azimuth; ADDED to every segment phase."""
    order: int
    """n; residual O(eps^{n+1}), infidelity slope 2(n+1)."""
    corrects: frozenset[Literal["amplitude", "pulse_length", "addressing", "detuning"]]
    segments: tuple[tuple[float, float], ...]
    """(area_rad, phase_rad) in TIME order, areas > 0."""
    n_rep: int = 1
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown composite-pulse family {self.family!r}")
        if self.order < 0 or self.n_rep < 1:
            raise ValueError("order must be >= 0 and n_rep >= 1")
        if not self.segments:
            raise ValueError("a composite pulse has at least one segment")
        if any(area <= 0.0 for area, _ in self.segments):
            raise ValueError(
                "segment areas are positive; a negative nominal area is emitted at phase + pi (Section 13)"
            )

    def total_rotation_rad(self) -> float:
        """Sum of the segment areas; the duration is this over Omega."""
        return float(sum(area for area, _ in self.segments))

    def propagator(self, eps_a: float = 0.0, eps_d: float = 0.0, eps_N: float | None = None) -> np.ndarray:
        """2x2, folded right-to-left; eps_N selects the addressing model with target = 1."""
        raise NotImplementedError(f"CompositePulse.propagator is {M2}")

    def infidelity(self, eps_a: float = 0.0, eps_d: float = 0.0, measure: str = "F_K") -> float:
        raise NotImplementedError(f"CompositePulse.infidelity is {M2}")

    def order_slope(
        self,
        channel: Literal["amplitude", "detuning", "simultaneous"],
        eps_range: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """(slope, fit residual) over an order-aware window eps >> 10^(-8/(n+1)) (Section 9.17)."""
        raise NotImplementedError(f"CompositePulse.order_slope is {M2}")

    def dc_polygon(self, rtol: float = 1e-10) -> tuple[np.ndarray, bool]:
        raise NotImplementedError(f"CompositePulse.dc_polygon is {M2}")

    def certificate(self, rtol: float = 1e-10) -> dict[int, tuple[complex, bool]]:
        raise NotImplementedError(f"CompositePulse.certificate is {M2}")

    def filter_function_amplitude(self, omega_rad_s: np.ndarray, omega_rabi_rad_s: float) -> np.ndarray:
        raise NotImplementedError(f"CompositePulse.filter_function_amplitude is {M2}")

    def dc_floor(self, moments: dict[str, float], omega_rabi_rad_s: float) -> float:
        raise NotImplementedError(f"CompositePulse.dc_floor is {M2}")


def composite_pulse(
    family: str, theta_rad: float, phi_rad: float = 0.0, *, order: int = 1, n_rep: int = 1
) -> CompositePulse:
    """Phases from the verified closed forms of Section 4.3.5; raises outside the arccos domain, never NaN."""
    raise NotImplementedError(f"composite_pulse is {M2}")


__all__ = ["FAMILIES", "CompositePulse", "Family", "composite_pulse"]
