"""Solver selection (sesolve/mesolve/mcsolve), step control and checkpoints (PLAN.md Section 5.3; milestone M2)."""

from __future__ import annotations

M2 = "milestone M2 (dynamics/evolve.py, PLAN.md Section 5.3)"


def evolve(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"evolve is {M2}")


__all__ = ["evolve"]
