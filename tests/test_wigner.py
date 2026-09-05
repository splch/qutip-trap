"""Exact angular-momentum algebra against sympy (the dev-only oracle) and its identities."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan as sympy_cg
from sympy.physics.wigner import wigner_3j as sympy_3j
from sympy.physics.wigner import wigner_6j as sympy_6j

from qutip_trap.species.wigner import (
    angular_momentum_matrices,
    clebsch_gordan,
    m_values,
    wigner_3j,
    wigner_6j,
)

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


@pytest.mark.parametrize("js", list(product(HALVES[:5], repeat=6))[::37])
def test_6j_matches_sympy(js: tuple[Fraction, ...]) -> None:
    ours = wigner_6j(*js)
    try:
        ref = float(sympy_6j(*(_r(j) for j in js)))
    except ValueError:  # sympy raises on non-triangular triads; the symbol is zero there
        assert ours == 0.0
        return
    assert ours == pytest.approx(ref, abs=1e-15)


def test_specific_6j_values_of_section_9_13() -> None:
    """{1/2 1/2 1; 1 0 1/2}^2 = 1/6 (171Yb+), the entry behind the 1/3 : 2/3 branching."""
    assert wigner_6j(0.5, 0.5, 1, 1, 0, 0.5) ** 2 == pytest.approx(1.0 / 6.0, abs=1e-15)


def test_clebsch_gordan_matches_sympy_and_is_orthonormal() -> None:
    for j1, j2 in ((0.5, 0.5), (1.5, 0.5), (3.5, 0.5), (1, 1), (1.5, 1)):
        jmin, jmax = abs(j1 - j2), j1 + j2
        J = jmin
        while J <= jmax:
            for M in m_values(J):
                for m1 in m_values(j1):
                    m2 = M - m1
                    if abs(m2) > j2:
                        continue
                    ours = clebsch_gordan(j1, m1, j2, m2, J, M)
                    ref = float(
                        sympy_cg(
                            _r(Fraction(j1)),
                            _r(Fraction(j2)),
                            _r(Fraction(J)),
                            _r(m1),
                            _r(Fraction(m2)),
                            _r(M),
                        )
                    )
                    assert ours == pytest.approx(ref, abs=1e-15)
            J += 1
        # orthonormality: the recoupling matrix is orthogonal
        rows = []
        J = jmin
        while J <= jmax:
            for M in m_values(J):
                rows.append(
                    [clebsch_gordan(j1, m1, j2, m2, J, M) for m1 in m_values(j1) for m2 in m_values(j2)]
                )
            J += 1
        u = np.array(rows)
        assert np.allclose(u @ u.T, np.eye(len(rows)), atol=1e-14)


def test_angular_momentum_algebra() -> None:
    for j in (0.5, 1, 1.5, 3.5):
        jx, jy, jz = angular_momentum_matrices(j)
        assert np.allclose(jx @ jy - jy @ jx, 1j * jz, atol=1e-14)
        j2 = jx @ jx + jy @ jy + jz @ jz
        assert np.allclose(j2, j * (j + 1) * np.eye(j2.shape[0]), atol=1e-13)
        assert np.allclose(np.diag(jz).real, [float(m) for m in m_values(j)])
