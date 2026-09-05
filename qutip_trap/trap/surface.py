"""Electrode geometry: rod, blade and surface-electrode (gapless-plane) traps (PLAN.md Section 4.1.6; M1).

Conventions (Section 13, row "Surface-electrode geometry"): widths from gap centre to gap centre; the
ion at the minimum of the TOTAL potential, not the rf null; the escape point a saddle; the five-wire null
height h = sqrt(a(a + 2b))/2 for FULL widths a (centre) and b (rails) (M1 test list).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

M1 = "milestone M1 (trap/surface.py, PLAN.md Section 4.1.6)"


@dataclass(frozen=True)
class Electrodes:
    """Electrode geometry; ``parameters`` are the named lengths of the layout in metres."""

    kind: Literal["rod_quadrupole", "blade", "surface_five_wire", "surface_four_wire", "surface_general"]
    parameters: dict[str, float]
    electrode_names: tuple[str, ...] = ()


def five_wire_null_height_m(a_m: float, b_m: float) -> float:
    """h = sqrt(a(a + 2b))/2 for FULL widths a (centre) and b (rails) (House 2008; PLAN.md M1 tests).

    The first version of the plan wrote sqrt(a(a + b)), the same formula with a as a half-width, unstated
    (caught by the 2026-09-04 critique); ``check_surface_mixed.py`` gives y0 = 0.921954 for a = 1, b = 1.2.
    """
    if a_m <= 0.0 or b_m <= 0.0:
        raise ValueError("widths must be positive")
    return math.sqrt(a_m * (a_m + 2.0 * b_m)) / 2.0


def electrode_potential(*args: object, **kwargs: object) -> object:
    """Gapless-plane analytic electrode potentials, rf null, depth, principal axes (Section 4.1.6)."""
    raise NotImplementedError(f"electrode_potential is {M1}")


__all__ = ["Electrodes", "electrode_potential", "five_wire_null_height_m"]
