"""Quasi-static parameter sampling per shot and OU processes for fast noise (PLAN.md Sections 5.5, 6.1; M7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseSample:
    """One draw of every quasi-static parameter (Section 6.1)."""

    sample_id: int
    values: dict[str, float]
    """Field offset, mode offsets, Rabi scale, beam phases, ..."""
    ou_grids: dict[str, np.ndarray]
    """Fixed-grid realizations of the fast processes (Section 5.5)."""


__all__ = ["NoiseSample"]
