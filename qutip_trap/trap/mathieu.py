"""Mathieu parameters and the micromotion factor (PLAN.md Section 4.1.1; Section 13; milestone M1).

Convention (Section 13, row "Mathieu equation"): d^2x/dxi^2 + [a - 2q cos 2xi] x = 0 with xi = omega_rf t/2,
a = 4 Z|e| U alpha/(m omega_rf^2), q = 2 Z|e| U~ alpha'/(m omega_rf^2) (Leibfried et al. 2003). Wineland 1998
and Berkeland 1998 use +2q cos, a half-period shift of the rf phase origin, which is why the sign is a
declared field of ``MathieuParameters`` and why the Floquet function reads u(t) ~ e^{i nu t}[1 - (q/2) cos
omega_rf t] with u_dot(0) = i nu (1 + q) in this origin (Section 13, "Floquet branch and normalization").

The micromotion factor on the Lamb-Dicke parameter is C0 = 1 + 3 q^2/16 + O(q^4), the e^{i nu t} Fourier
coefficient of the exact Floquet function under the Wronskian normalization Im(u* u_dot) = nu; the RMP's
(1 + q/2)^-1 is the u(0) = 1 normalization at one rf phase, an rf-clock artifact [corrected: derivation
audit, 2026-09-04] (``validation/scripts/check_c0_floquet.py``: 1.001890, 1.007741, 1.018161 at q = 0.1,
0.2, 0.3). C0 is applied once, in ``Crystal.lamb_dicke`` (Section 5.7), and nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

MATHIEU_SIGN_CONVENTION: Final = "a - 2q cos 2xi"
M1 = "milestone M1 (trap/mathieu.py, PLAN.md Section 4.1.1)"


@dataclass(frozen=True)
class MathieuParameters:
    a: np.ndarray
    q: np.ndarray
    beta: tuple[float, float, float]
    secular_hz: tuple[float, float, float]
    C0: tuple[float, float, float]
    """Wronskian-normalized rf-period Fourier coefficient per axis (Section 4.1.1)."""
    sign_convention: Literal["a - 2q cos 2xi"] = MATHIEU_SIGN_CONVENTION


def beta_lowest_order(a: float, q: float) -> float:
    """The lowest-order characteristic exponent beta ~ sqrt(a + q^2/2), a CHECK on the monodromy result (Section 13)."""
    radicand = a + q * q / 2.0
    if radicand <= 0.0:
        raise ValueError(f"a + q^2/2 = {radicand} <= 0: outside the lowest-order stable region")
    return math.sqrt(radicand)


def c0_series(q: float) -> float:
    """C0 = 1 + 3 q^2/16, the O(q^2) micromotion factor on eta (even in q, no first-order term; Section 13)."""
    return 1.0 + 3.0 * q * q / 16.0


def mathieu_parameters(*args: object, **kwargs: object) -> MathieuParameters:
    """a, q from voltages and geometry; beta by the monodromy matrix; secular frequencies; C0 (Section 4.1.1)."""
    raise NotImplementedError(f"mathieu_parameters is {M1}")


__all__ = [
    "MATHIEU_SIGN_CONVENTION",
    "MathieuParameters",
    "beta_lowest_order",
    "c0_series",
    "mathieu_parameters",
]
