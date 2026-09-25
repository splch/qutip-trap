"""Exact angular-momentum algebra: the Wigner 3j symbol and the spin matrices.

The 3j symbol is the Racah closed form in exact rational arithmetic (``fractions.Fraction``), so every value is correct
to the last floating-point digit; conventions are Edmonds' (Steck QAO Appendix).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial, sqrt

import numpy as np

Half = Fraction | int | float


@lru_cache(maxsize=4096)
def as_half_integer(x: Half) -> Fraction:
    """Coerce an angular-momentum quantum number to an exact integer or half-integer Fraction."""
    f = Fraction(x).limit_denominator(2)
    if abs(float(f) - float(x)) > 1e-12:
        raise ValueError(f"{x!r} is not an integer or half-integer")
    return f


def _fact(x: Fraction) -> int:
    if x.denominator != 1 or x < 0:
        raise ValueError(f"factorial of a non-integer or negative argument {x}")
    return factorial(int(x))


def _triangle_ok(a: Fraction, b: Fraction, c: Fraction) -> bool:
    return abs(a - b) <= c <= a + b and (a + b + c).denominator == 1


def _delta(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    """The triangle coefficient Delta(abc) = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!."""
    return Fraction(_fact(a + b - c) * _fact(a - b + c) * _fact(-a + b + c), _fact(a + b + c + 1))


def parity_sign(n: int) -> float:
    """(-1)^n as a float (mypy types int ** int as Any)."""
    return -1.0 if n % 2 else 1.0


def _signed_sqrt(square: Fraction, sign: int) -> float:
    return float(sign) * float(sqrt(float(square)))


@lru_cache(maxsize=65536)
def _wigner_3j_cached(
    j1: Fraction, j2: Fraction, j3: Fraction, m1: Fraction, m2: Fraction, m3: Fraction
) -> float:
    if m1 + m2 + m3 != 0:
        return 0.0
    if not _triangle_ok(j1, j2, j3):
        return 0.0
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        if abs(m) > j or (j - m).denominator != 1:
            return 0.0
    # Racah formula
    pref2 = _delta(j1, j2, j3) * (
        _fact(j1 + m1) * _fact(j1 - m1) * _fact(j2 + m2) * _fact(j2 - m2) * _fact(j3 + m3) * _fact(j3 - m3)
    )
    total = Fraction(0)
    k_min = max(Fraction(0), j2 - j3 - m1, j1 - j3 + m2)
    k_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    k = k_min
    while k <= k_max:
        denom = (
            _fact(k)
            * _fact(j1 + j2 - j3 - k)
            * _fact(j1 - m1 - k)
            * _fact(j2 + m2 - k)
            * _fact(j3 - j2 + m1 + k)
            * _fact(j3 - j1 - m2 + k)
        )
        total += Fraction((-1) ** int(k), denom)
        k += 1
    if total == 0:
        return 0.0
    phase = 1 if int(j1 - j2 - m3) % 2 == 0 else -1
    sign = phase * (1 if total > 0 else -1)
    return _signed_sqrt(pref2 * total * total, sign)


def wigner_3j(j1: Half, j2: Half, j3: Half, m1: Half, m2: Half, m3: Half) -> float:
    """The Wigner 3j symbol (j1 j2 j3; m1 m2 m3), exact to floating-point rounding."""
    return _wigner_3j_cached(*(as_half_integer(x) for x in (j1, j2, j3, m1, m2, m3)))


def m_values(j: Half) -> tuple[Fraction, ...]:
    """m = -j, -j + 1, ..., +j (ascending)."""
    jj = as_half_integer(j)
    return tuple(-jj + k for k in range(int(2 * jj) + 1))


def angular_momentum_matrices(j: Half) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Jx, Jy, Jz) in units of hbar on the basis m = -j..+j ascending; J+ |j m> = sqrt(j(j+1) - m(m+1)) |j m+1>."""
    ms = m_values(j)
    jj = float(as_half_integer(j))
    dim = len(ms)
    jz = np.diag([float(m) for m in ms]).astype(complex)
    jp = np.zeros((dim, dim), dtype=complex)
    for k, m in enumerate(ms[:-1]):
        mf = float(m)
        jp[k + 1, k] = sqrt(jj * (jj + 1.0) - mf * (mf + 1.0))
    jm = jp.conj().T
    jx = (jp + jm) / 2.0
    jy = (jp - jm) / 2.0j
    return jx, jy, jz
