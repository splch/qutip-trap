"""Photon-count sampling: window, efficiency, background, crosstalk (PLAN.md Sections 8.2, 8.5; Appendix E; M5).

Convention (Section 13, "Detection efficiency"): epsilon_sys applied once, at scattered -> detected rate;
the background rate is separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Detector:
    kind: Literal["pmt", "camera", "snspd"]
    efficiency: float
    """Total system detection efficiency epsilon_sys (Section 8.2)."""
    background_cps: float
    psf_leakage: dict[int, float]
    """Neighbour distance -> fraction of one ion's light landing on the neighbour's detector (Section 8.5)."""
    dead_time_s: float | None
    afterpulse_prob: float | None
    window_s: float

    def __post_init__(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("detection efficiency lies in (0, 1]")
        if self.background_cps < 0.0 or self.window_s <= 0.0:
            raise ValueError("background is non-negative and the window positive")
        if any(not 0.0 <= f <= 1.0 for f in self.psf_leakage.values()):
            raise ValueError("PSF leakage fractions lie in [0, 1]")


__all__ = ["Detector"]
