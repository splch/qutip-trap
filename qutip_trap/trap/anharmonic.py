"""Cubic and quartic mode couplings (PLAN.md Section 4.1.4; optional correction terms; milestone M1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnharmonicTerms:
    """Coupling coefficients of the optional H_anh and H_curv terms of Section 5.7, in rad/s.

    Keys are mode-index triples (cubic) and quadruples (quartic) into ``Crystal.modes``. The leading
    off-resonant effect of the cubic coupling is SECOND order, g^2 t/Delta, not g t (Section 4.1.4, caught by
    the 2026-09-04 experimentalist critique; ``check_anharmonic.py``).
    """

    cubic_rad_s: dict[tuple[int, int, int], float] = field(default_factory=dict)
    quartic_rad_s: dict[tuple[int, int, int, int], float] = field(default_factory=dict)
    resonance_check: bool = True


__all__ = ["AnharmonicTerms"]
