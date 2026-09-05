"""Excess micromotion: in-phase and out-of-phase residual indices (PLAN.md Section 4.1.1; Appendix E)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MicromotionIndex:
    """Residual modulation index beta = delta_k . u_1 for the FULL wavevector of a drive.

    ``in_phase`` is the stray-field part (nullable by shims) and ``out_of_phase`` the rf-quadrature part
    (Berkeland's phi_ac term, not nullable); they never collapse into one number (Section 4.1.1). The
    peak/rms tag is declared, never inferred (Section 13).
    """

    in_phase: float
    out_of_phase: float
    convention: Literal["peak", "rms"]


__all__ = ["MicromotionIndex"]
