"""Dipole elements, hyperfine factors and saturation (PLAN.md Section 4.5.2), with Steck's 6j factorization (sympy) as
the independent check of the uncoupled-basis operator."""

from __future__ import annotations

import math
from fractions import Fraction
from functools import cache
from itertools import product

import numpy as np
import pytest
from scipy.constants import physical_constants
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan as sympy_cg
from sympy.physics.wigner import wigner_3j as sympy_3j
from sympy.physics.wigner import wigner_6j as sympy_6j

from qutip_trap.species import species
from qutip_trap.species.dipole import (
    dipole_operator_uncoupled,
    reduced_element_from_partial_rate,
    stretched_element_factor,
    wigner_eckart_j,
)
from qutip_trap.species.model import Transition
from qutip_trap.species.wigner import m_values
from qutip_trap.units import C_M_PER_S, E_C, EPSILON_0_F_PER_M, HBAR_J_S, TWO_PI
from tests.fixtures import HALF

Fr = Fraction | int
E_A0 = E_C * physical_constants["Bohr radius"][0]

# Steck's 87Rb D-line data: vacuum wavelengths and lifetimes
RB87_D2_LAMBDA = 780.241209686e-9
RB87_D2_TAU = 26.2348e-9
RB87_D1_LAMBDA = 794.978851156e-9
RB87_D1_TAU = 27.679e-9


def _r(x: Fraction | int) -> Rational:
    f = Fraction(x)
    return Rational(f.numerator, f.denominator)


@cache
def _coupled(i_n: Fraction, J: Fraction, F: Fraction, mF: Fraction) -> np.ndarray:
    """|F mF> on the uncoupled basis (m_I outer, m_J inner) in the |(J I) F> coupling order, from sympy's CG."""
    return np.array(
        [
            float(sympy_cg(_r(J), _r(i_n), _r(F), _r(mj), _r(mi), _r(mF)))
            for mi in m_values(i_n)
            for mj in m_values(J)
        ]
    )


@cache
def _six_j(*js: Fraction) -> float:
    try:
        return float(sympy_6j(*(_r(j) for j in js)))
    except ValueError:  # sympy refuses a non-triangular triad; the symbol is zero there
        return 0.0


@cache
def _three_j(*js: Fraction) -> float:
    return float(sympy_3j(*(_r(j) for j in js)))


def _steck_element(F: Fr, mF: Fr, Fp: Fr, mFp: Fr, q: int, J: Fr, Jp: Fr, i_n: Fr) -> float:
    """<F mF|d_q|F' mF'>/<J||d||J'> in Steck's 6j factorization (Rb87 Eqs. 35-36), zero unless mF = mF' + q."""
    if mF != mFp + q:
        return 0.0
    reduced = (
        (-1) ** int(Fp + J + 1 + i_n) * math.sqrt((2 * Fp + 1) * (2 * J + 1)) * _six_j(J, Jp, 1, Fp, F, i_n)
    )
    return reduced * (-1) ** int(Fp - 1 + mF) * math.sqrt(2 * F + 1) * _three_j(Fp, 1, F, mFp, q, -mF)


def _manifolds(i_n: Fraction, J: Fraction) -> list[Fraction]:
    return [abs(J - i_n) + k for k in range(int(J + i_n - abs(J - i_n)) + 1)]


def _element(i_n: Fr, J: Fr, F: Fr, mF: Fr, Jp: Fr, Fp: Fr, mFp: Fr, q: int) -> float:
    """<F mF|T_q|F' mF'>/<J||d||J'> of the live uncoupled-basis operator between coupled states."""
    return float(
        _coupled(i_n, J, F, mF) @ dipole_operator_uncoupled(i_n, J, Jp, q) @ _coupled(i_n, Jp, Fp, mFp)
    )


def _absorption(i_n: Fraction, J: Fraction, F: int, Jp: Fraction, Fp: int) -> float:
    """S_FF' = sum_{mF', q} |<F mF|T_q|F' mF'>|^2 at one mF: the relative absorption strength."""
    mF = m_values(F)[0]
    return sum(_element(i_n, J, F, mF, Jp, Fp, mFp, q) ** 2 for mFp in m_values(Fp) for q in (-1, 0, 1))


def _emission(i_n: Fraction, J: Fraction, F: int, Jp: Fraction, Fp: int) -> float:
    """b(F' -> F) = (2J'+1)/(2J+1) sum_{mF, q} |<F mF|T_q|F' mF'>|^2 at one mF': the decay branching into F."""
    mFp = m_values(Fp)[0]
    total = sum(_element(i_n, J, F, mF, Jp, Fp, mFp, q) ** 2 for mF in m_values(F) for q in (-1, 0, 1))
    return float(2 * Jp + 1) / float(2 * J + 1) * total


def test_j_level_sum_rules() -> None:
    """The Wigner-Eckart sums are |d|^2 at fixed m and (2J+1)/(2J'+1)|d|^2 at fixed m', to 1e-14."""
    for J, Jp in (
        (HALF, HALF),
        (HALF, Fraction(3, 2)),
        (Fraction(3, 2), Fraction(3, 2)),
        (Fraction(3, 2), Fraction(5, 2)),
    ):
        for m in m_values(J):
            total = sum(wigner_eckart_j(J, m, Jp, mp, q) ** 2 for mp in m_values(Jp) for q in (-1, 0, 1))
            assert total == pytest.approx(1.0, abs=1e-14)
        for mp in m_values(Jp):
            total = sum(wigner_eckart_j(J, m, Jp, mp, q) ** 2 for m in m_values(J) for q in (-1, 0, 1))
            assert total == pytest.approx(float(2 * J + 1) / float(2 * Jp + 1), abs=1e-14)


def test_hyperfine_factors_of_87rb_and_171yb() -> None:
    """To 1e-14: 87Rb S_FF' rows {1/20, 1/4, 7/10} and {1/6, 5/12, 5/12} on D2 and {1/2, 1/2}, {1/6, 5/6} on D1, 171Yb+
    b(1 -> 0) = 1/3, b(1 -> 1) = 2/3, b(0 -> 0) = 0, and the |1,1> -> P3/2 cycling strength 1/2."""
    i_rb, three_half = Fraction(3, 2), Fraction(3, 2)
    d2 = {
        F: [_absorption(i_rb, HALF, F, three_half, Fp) for Fp in range(abs(F - 1), F + 2) if Fp <= 3]
        for F in (1, 2)
    }
    assert d2[2] == pytest.approx([1 / 20, 1 / 4, 7 / 10], abs=1e-14)
    assert d2[1] == pytest.approx([1 / 6, 5 / 12, 5 / 12], abs=1e-14)
    d1 = {F: [_absorption(i_rb, HALF, F, HALF, Fp) for Fp in (1, 2)] for F in (1, 2)}
    assert d1[2] == pytest.approx([1 / 2, 1 / 2], abs=1e-14)
    assert d1[1] == pytest.approx([1 / 6, 5 / 6], abs=1e-14)
    assert _emission(HALF, HALF, 0, HALF, 1) == pytest.approx(1 / 3, abs=1e-14)
    assert _emission(HALF, HALF, 1, HALF, 1) == pytest.approx(2 / 3, abs=1e-14)
    assert _emission(HALF, HALF, 0, HALF, 0) == pytest.approx(0.0, abs=1e-14)
    assert _emission(HALF, HALF, 1, HALF, 0) == pytest.approx(1.0, abs=1e-14)
    assert _element(HALF, HALF, 1, 1, three_half, 2, 2, -1) ** 2 == pytest.approx(0.5, abs=1e-14)
    assert sum(_absorption(HALF, HALF, 1, three_half, Fp) for Fp in (1, 2)) == pytest.approx(1.0, abs=1e-14)


def test_steck_6j_factorization_equals_the_uncoupled_basis_operator() -> None:
    """Steck's 6j form of <F mF|d_q|F' mF'> equals the uncoupled-basis element to 1e-13 for I up to 7/2 and four (J, J')
    pairs."""
    pairs = (
        (HALF, HALF),
        (HALF, Fraction(3, 2)),
        (Fraction(3, 2), Fraction(3, 2)),
        (Fraction(3, 2), Fraction(5, 2)),
    )
    checked = nonzero = 0
    for i_n in (Fraction(0), HALF, Fraction(1), Fraction(3, 2), Fraction(5, 2), Fraction(7, 2)):
        for J, Jp in pairs:
            ops = {q: dipole_operator_uncoupled(i_n, J, Jp, q) for q in (-1, 0, 1)}
            for F, Fp in product(_manifolds(i_n, J), _manifolds(i_n, Jp)):
                for mF, mFp in product(m_values(F), m_values(Fp)):
                    va, vb = _coupled(i_n, J, F, mF), _coupled(i_n, Jp, Fp, mFp)
                    for q in (-1, 0, 1):
                        steck = _steck_element(F, mF, Fp, mFp, q, J, Jp, i_n)
                        assert float(va @ ops[q] @ vb) == pytest.approx(steck, abs=1e-13)
                        checked += 1
                        nonzero += steck != 0.0
    assert checked > 20000 and nonzero > 1500


def test_reduced_element_and_saturation_of_87rb_d2_and_d1() -> None:
    """87Rb D2 and D1: |<J||er||J'>| = 4.227524 and 2.993125 e a0, I_sat = 1.669325 and 1.495851 mW/cm^2 and the
    stretched element 2.989311, all to 2e-6, and the rounded 26.24 ns lifetime already gives 4.227104."""
    for lam, tau, jp, d_ref, isat_ref in (
        (RB87_D2_LAMBDA, RB87_D2_TAU, Fraction(3, 2), 4.227524, 1.669325),
        (RB87_D1_LAMBDA, RB87_D1_TAU, HALF, 2.993125, 1.495851),
    ):
        d = reduced_element_from_partial_rate(1.0 / tau, TWO_PI * C_M_PER_S / lam, HALF, jp)
        assert d / E_A0 == pytest.approx(d_ref, abs=2e-6)
        line = Transition("S1/2", f"P{jp}", lam, 1.0 / (TWO_PI * tau), 1.0, "E1", ("Steck",))
        assert line.i_sat_w_m2 * 0.1 == pytest.approx(isat_ref, abs=2e-6)
    d2 = reduced_element_from_partial_rate(
        1.0 / RB87_D2_TAU, TWO_PI * C_M_PER_S / RB87_D2_LAMBDA, HALF, Fraction(3, 2)
    )
    assert stretched_element_factor(HALF, Fraction(3, 2)) * d2 / E_A0 == pytest.approx(2.989311, abs=2e-6)
    d_rounded = reduced_element_from_partial_rate(
        1.0 / 26.24e-9, TWO_PI * C_M_PER_S / RB87_D2_LAMBDA, HALF, Fraction(3, 2)
    )
    assert d_rounded / E_A0 == pytest.approx(4.227104, abs=2e-6)


def test_171yb_gamma_to_element_to_i_sat_chain() -> None:
    """171Yb+ 369.5 nm with the partial 19.62 MHz rate: 1.752 e a0 (1.5e-3) and 50.83 mW/cm^2 (0.01)."""
    yb = species("171Yb+").transition("S1/2-P1/2")
    omega = TWO_PI * C_M_PER_S / yb.wavelength_vac_m
    d = reduced_element_from_partial_rate(yb.partial_rate_rad_s, omega, HALF, HALF)
    assert d / E_A0 == pytest.approx(1.752, abs=1.5e-3)
    assert yb.i_sat_w_m2 * 0.1 == pytest.approx(50.83, abs=0.01)


def test_total_decay_rate_of_every_excited_sublevel_is_one_over_tau() -> None:
    """Every one of the 16 excited 87Rb D2 sublevels decays at omega^3/(3 pi eps0 hbar c^3) sum |d_q|^2 = 1/tau to
    1e-12."""
    i_rb = Fraction(3, 2)
    omega = TWO_PI * C_M_PER_S / RB87_D2_LAMBDA
    d = reduced_element_from_partial_rate(1.0 / RB87_D2_TAU, omega, HALF, Fraction(3, 2))
    ops = {q: d * dipole_operator_uncoupled(i_rb, HALF, Fraction(3, 2), q) for q in (-1, 0, 1)}
    pref = omega**3 / (3.0 * math.pi * EPSILON_0_F_PER_M * HBAR_J_S * C_M_PER_S**3)
    rates = [pref * sum(float(np.sum(np.abs(ops[q][:, col]) ** 2)) for q in (-1, 0, 1)) for col in range(16)]
    assert min(rates) == pytest.approx(1.0 / RB87_D2_TAU, rel=1e-12)
    assert max(rates) == pytest.approx(1.0 / RB87_D2_TAU, rel=1e-12)
