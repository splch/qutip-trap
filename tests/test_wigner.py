"""The exact 3j symbol against sympy, and the angular-momentum matrices."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_3j as sympy_3j

from qutip_trap.species.wigner import angular_momentum_matrices, m_values, wigner_3j

HALVES = [Fraction(k, 2) for k in range(0, 8)]  # 0 .. 7/2
SMALL = HALVES[:5]  # 0 .. 2


def _r(x: Fraction) -> Rational:
    return Rational(x.numerator, x.denominator)


@pytest.mark.parametrize(
    "j1,j2,j3",
    [
        (a, b, c)
        for a in SMALL
        for b in SMALL
        for c in HALVES
        if abs(a - b) <= c <= a + b and (a + b + c).denominator == 1
    ],
)
def test_3j_matches_sympy(j1: Fraction, j2: Fraction, j3: Fraction) -> None:
    for m1 in m_values(j1):
        for m2 in m_values(j2):
            m3 = -(m1 + m2)
            if abs(m3) > j3:
                continue
            ours = wigner_3j(j1, j2, j3, m1, m2, m3)
            ref = float(sympy_3j(_r(j1), _r(j2), _r(j3), _r(m1), _r(m2), _r(m3)))
            assert ours == pytest.approx(ref, abs=1e-15)


def test_3j_selection_rules_return_zero() -> None:
    assert wigner_3j(1, 1, 3, 0, 0, 0) == 0.0, "triangle violated"
    assert wigner_3j(1, 1, 1, 1, 1, -1) == 0.0, "m sum not zero"
    assert wigner_3j(0.5, 0.5, 1, 0.5, -0.5, 0) != 0.0


def test_angular_momentum_algebra() -> None:
    for j in (0.5, 1, 1.5, 3.5):
        jx, jy, jz = angular_momentum_matrices(j)
        assert np.allclose(jx @ jy - jy @ jx, 1j * jz, atol=1e-14)
        j2 = jx @ jx + jy @ jy + jz @ jz
        assert np.allclose(j2, j * (j + 1) * np.eye(j2.shape[0]), atol=1e-13)
        assert np.allclose(np.diag(jz).real, [float(m) for m in m_values(j)])
