"""The ``ExperimentResult`` record every simulated experiment returns (PLAN.md Appendix E; Sections 7.5, 7.9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentResult:
    data: np.ndarray
    """The scan as measured: one row per point (columns documented per experiment)."""
    fitted: dict[str, tuple[float, float]]
    """Parameter -> (value, uncertainty)."""
    model: str
    provenance_id: str
    converged: bool = True
    """False when a fit failed, landed at the edge of its scan range or violated the experiment's own consistency check: the
    entry it feeds is then ``uncalibrated`` (Section 7.5; M8)."""
    notes: tuple[str, ...] = ()
    sigma: np.ndarray | None = None
    """Per-row statistical uncertainty of the measured column when shots were drawn (None for exact populations)."""

    def value(self, key: str) -> float:
        return float(self.fitted[key][0])

    def uncertainty(self, key: str) -> float:
        return float(self.fitted[key][1])


__all__ = ["ExperimentResult"]
