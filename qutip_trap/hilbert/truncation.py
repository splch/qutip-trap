"""Truncation policy: n_max caps, the boundary-population monitor and adaptive growth (PLAN.md Section 5.5; M2, M9a)."""

from __future__ import annotations

M2 = "milestone M2 (hilbert/truncation.py, PLAN.md Section 5.5)"


def boundary_population(*args: object, **kwargs: object) -> float:
    raise NotImplementedError(f"boundary_population is {M2}")


def halving_test(*args: object, **kwargs: object) -> bool:
    """The convergence test of Section 5.5 behind the convergence badge (Section 14.5)."""
    raise NotImplementedError(f"halving_test is {M2}")


__all__ = ["boundary_population", "halving_test"]
