"""Discriminators and the readout POVM fast path (PLAN.md Sections 5.7, 8.3, 8.4; Appendix E; milestone M5).

The product POVM is exact only at zero readout crosstalk; at configured crosstalk the fast path carries a
register-wide 2^N x 2^N confusion tensor with its own size guard (dense to N = 12), and the readout error is
never applied twice (Section 5.7, corrected after the 2026-09-04 critique).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

M5 = "milestone M5 (readout/, PLAN.md Section 8)"


@dataclass(frozen=True)
class POVM:
    per_ion: tuple[np.ndarray, ...] | None
    """Product form, valid only at zero readout crosstalk."""
    confusion: np.ndarray | None
    """Register-wide 2^N x 2^N tensor otherwise (guarded, Section 5.7)."""
    crosstalk_domain: Literal["zero", "configured"]

    def __post_init__(self) -> None:
        if (self.per_ion is None) == (self.confusion is None):
            raise ValueError("a POVM is either the product form or the register-wide confusion tensor")
        if self.crosstalk_domain == "zero" and self.per_ion is None:
            raise ValueError("the zero-crosstalk domain is the product form")


def threshold_discriminator(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"threshold_discriminator is {M5}")


def time_resolved_discriminator(*args: object, **kwargs: object) -> object:
    raise NotImplementedError(f"time_resolved_discriminator is {M5}")


__all__ = ["POVM", "threshold_discriminator", "time_resolved_discriminator"]
