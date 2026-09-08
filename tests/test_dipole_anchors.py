"""Dipole elements, branchings and saturation: the anchors of PLAN.md Sections 4.5.2, 9.13 and the audit rows 4.5-1, 4.5-4."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from qutip_trap.species import species
from qutip_trap.species.dipole import (
    absorption_strength,
    coupled_state_vector,
    dipole_operator_uncoupled,
    emission_branching,
    hyperfine_element,
    reduced_element_from_partial_rate,
    resonant_cross_section_m2,
    saturation_intensity_random_orientation_w_m2,
    saturation_intensity_w_m2,
    stretched_element_factor,
    wigner_eckart_j,
)
from qutip_trap.species.wigner import m_values
from qutip_trap.units import A_0_M, C_M_PER_S, E_C, EPSILON_0_F_PER_M, HBAR_J_S, TWO_PI

HALF = Fraction(1, 2)
E_A0 = E_C * A_0_M

# Steck's 87Rb D-line data: vacuum wavelengths and lifetimes; the anchors below are the plan's Section 9.13 numbers
RB87_D2_LAMBDA = 780.241209686e-9
RB87_D2_TAU = 26.2348e-9
RB87_D1_LAMBDA = 794.978851156e-9
RB87_D1_TAU = 27.679e-9


def test_j_level_sum_rules() -> None:
    """sum_{m',q} |<J m|T_q|J' m'>|^2 = |d|^2 at fixed m; sum_{m,q} at fixed m' = (2J+1)/(2J'+1)|d|^2 (Section 4.5.2)."""
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


def test_hyperfine_factors_87rb_and_171yb() -> None:
    """S_FF' rows {1/20, 1/4, 7/10} (F = 2) and {1/6, 5/12, 5/12} (F = 1) on D2, D1 {1/2, 1/2} and {1/6, 5/6}; each sums to 1;
    171Yb+ b(1 -> 0) = 1/3, b(1 -> 1) = 2/3, b(0 -> 0) = 0 (Section 9.13)."""
    i_rb = Fraction(3, 2)
    d2 = {
        F: [
            absorption_strength(F, Fp, HALF, Fraction(3, 2), i_rb)
            for Fp in range(abs(F - 1), F + 2)
            if Fp <= 3
        ]
        for F in (1, 2)
    }
    assert d2[2] == pytest.approx([1 / 20, 1 / 4, 7 / 10], abs=1e-14)
    assert d2[1] == pytest.approx([1 / 6, 5 / 12, 5 / 12], abs=1e-14)
    d1 = {F: [absorption_strength(F, Fp, HALF, HALF, i_rb) for Fp in (1, 2)] for F in (1, 2)}
    assert d1[2] == pytest.approx([1 / 2, 1 / 2], abs=1e-14)
    assert d1[1] == pytest.approx([1 / 6, 5 / 6], abs=1e-14)
    for F in (1, 2):
        assert sum(d2[F]) == pytest.approx(1.0, abs=1e-14) and sum(d1[F]) == pytest.approx(1.0, abs=1e-14)
    assert emission_branching(1, 0, HALF, HALF, HALF) == pytest.approx(1 / 3, abs=1e-14)
    assert emission_branching(1, 1, HALF, HALF, HALF) == pytest.approx(2 / 3, abs=1e-14)
    assert emission_branching(0, 0, HALF, HALF, HALF) == 0.0
    assert emission_branching(0, 1, HALF, HALF, HALF) == pytest.approx(1.0, abs=1e-14)
    # both directed factors sum to one over their manifold; one "hyperfine factor" used in both roles would not
    for Fp in (0, 1):
        assert sum(emission_branching(Fp, F, HALF, HALF, HALF) for F in (0, 1)) == pytest.approx(
            1.0, abs=1e-14
        )
    for F in (0, 1):
        assert sum(absorption_strength(F, Fp, HALF, HALF, HALF) for Fp in (0, 1)) == pytest.approx(
            1.0, abs=1e-14
        )


def test_steck_6j_factorization_equals_the_uncoupled_basis_sum() -> None:
    """Audit row 4.5-4: <F mF|d_q|F' mF'> from the 6j form equals the uncoupled-basis (spectator m_I, no 6j) evaluation
    with |F mF> built in the |(J I) F> coupling order, for I up to 7/2 and four (J, J') pairs; the m_F = m_F' + q rule holds."""
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
            fs = [abs(J - i_n) + k for k in range(int(J + i_n - abs(J - i_n)) + 1)]
            fps = [abs(Jp - i_n) + k for k in range(int(Jp + i_n - abs(Jp - i_n)) + 1)]
            for F, Fp in product(fs, fps):
                for mF, mFp in product(m_values(F), m_values(Fp)):
                    va = coupled_state_vector(i_n, J, F, mF)
                    vb = coupled_state_vector(i_n, Jp, Fp, mFp)
                    for q in (-1, 0, 1):
                        steck = hyperfine_element(F, mF, Fp, mFp, q, J, Jp, i_n)
                        brute = float(va @ ops[q] @ vb)
                        assert brute == pytest.approx(steck, abs=1e-13)
                        checked += 1
                        nonzero += steck != 0.0
                        if mF != mFp + q:
                            assert steck == 0.0 and abs(brute) < 1e-13
    assert checked > 10000 and nonzero > 1000


def test_reduced_element_and_saturation_of_87rb_d2_and_d1() -> None:
    """|<J||er||J'>| = 4.227524 e a0 (D2), 2.993125 (D1); stretched element sqrt(1/2) x 4.227524 = 2.989311; I_sat cycling
    1.669325 mW/cm^2 and random-orientation 5.007975 (3.0000x) on D2, 1.495851 and 4.487552 on D1; sigma_0 = 2.9066929377205e-9 cm^2."""
    for lam, tau, jp, d_ref, isat_ref, isat_iso_ref in (
        (RB87_D2_LAMBDA, RB87_D2_TAU, Fraction(3, 2), 4.227524, 1.669325, 5.007975),
        (RB87_D1_LAMBDA, RB87_D1_TAU, HALF, 2.993125, 1.495851, 4.487552),
    ):
        omega = TWO_PI * C_M_PER_S / lam
        gamma = 1.0 / tau
        d = reduced_element_from_partial_rate(gamma, omega, HALF, jp)
        assert d / E_A0 == pytest.approx(d_ref, abs=2e-6)
        assert saturation_intensity_w_m2(gamma, lam) * 0.1 == pytest.approx(isat_ref, abs=2e-6)
        iso = saturation_intensity_random_orientation_w_m2(gamma, lam) * 0.1
        assert iso == pytest.approx(isat_iso_ref, abs=2e-6)
        assert iso / (saturation_intensity_w_m2(gamma, lam) * 0.1) == pytest.approx(3.0, abs=1e-12)
    d2 = reduced_element_from_partial_rate(
        1.0 / RB87_D2_TAU, TWO_PI * C_M_PER_S / RB87_D2_LAMBDA, HALF, Fraction(3, 2)
    )
    assert stretched_element_factor(HALF, Fraction(3, 2)) * d2 / E_A0 == pytest.approx(2.989311, abs=2e-6)
    assert resonant_cross_section_m2(RB87_D2_LAMBDA) * 1e4 == pytest.approx(2.9066929377205e-9, rel=1e-11)
    # the rounded 26.24 ns already moves the element to 4.227104 (Section 4.5.6)
    d_rounded = reduced_element_from_partial_rate(
        1.0 / 26.24e-9, TWO_PI * C_M_PER_S / RB87_D2_LAMBDA, HALF, Fraction(3, 2)
    )
    assert d_rounded / E_A0 == pytest.approx(4.227104, abs=2e-6)


def test_171yb_and_40ca_gamma_to_element_to_i_sat_chain() -> None:
    """171Yb+ 369.5 nm with the partial 19.62 MHz rate: 1.752 e a0 and 50.83 mW/cm^2; 40Ca+ 397/393 nm read as partial 21.57/23.4 MHz
    at the plan's inputs: 2.045 / 2.972 e a0 and 45.11 / 50.25 mW/cm^2 (Section 9.13, check_atomic.py).

    The 171Yb+ row is the TABLE's rate; the two 40Ca+ rows are closed forms at PLAN.md 9.13's own quoted linewidths and AIR
    wavelengths, and stay that way. The partial reading is now the ca40 table's too, and the 397 nm row comes out of it to
    4e-5, but the 393 nm one does not: the table's partial rate there is Meir et al. 2020's 22.4071 MHz, not 23.4
    (tests/test_species_tables.py pins both sides; ledger anchor.ca40.p32_linewidth_readings)."""
    yb = species("171Yb+").transition("S1/2-P1/2")
    omega = TWO_PI * C_M_PER_S / yb.wavelength_vac_m
    d = reduced_element_from_partial_rate(yb.partial_rate_rad_s, omega, HALF, HALF)
    assert d / E_A0 == pytest.approx(1.752, abs=1.5e-3)
    assert yb.i_sat_w_m2 * 0.1 == pytest.approx(50.83, abs=0.01)
    for lam, g_mhz, jp, d_ref, isat_ref in (
        (396.85e-9, 21.57, HALF, 2.045, 45.11),
        (393.37e-9, 23.4, Fraction(3, 2), 2.972, 50.25),
    ):
        gamma = TWO_PI * g_mhz * 1e6
        omega = TWO_PI * C_M_PER_S / lam
        assert reduced_element_from_partial_rate(gamma, omega, HALF, jp) / E_A0 == pytest.approx(
            d_ref, abs=1.5e-3
        )
        assert saturation_intensity_w_m2(gamma, lam) * 0.1 == pytest.approx(isat_ref, abs=0.01)


def test_total_decay_rate_of_every_excited_sublevel_is_one_over_tau() -> None:
    """Audit row 4.5-1: the 24-level 87Rb D2 manifold by |m_J, m_I> recoupling; omega^3/(3 pi eps0 hbar c^3) sum_{m,q}|<J_g m|d_q|J_e m'>|^2
    equals 1/tau for all 16 excited sublevels, m'-independent, with Steck's Gamma formula fed the plan's own reduced element."""
    i_rb = Fraction(3, 2)
    omega = TWO_PI * C_M_PER_S / RB87_D2_LAMBDA
    d = reduced_element_from_partial_rate(1.0 / RB87_D2_TAU, omega, HALF, Fraction(3, 2))
    ops = {q: d * dipole_operator_uncoupled(i_rb, HALF, Fraction(3, 2), q) for q in (-1, 0, 1)}
    pref = omega**3 / (3.0 * math.pi * EPSILON_0_F_PER_M * HBAR_J_S * C_M_PER_S**3)
    rates = [pref * sum(float(np.sum(np.abs(ops[q][:, col]) ** 2)) for q in (-1, 0, 1)) for col in range(16)]
    assert min(rates) == pytest.approx(1.0 / RB87_D2_TAU, rel=1e-12)
    assert max(rates) == pytest.approx(1.0 / RB87_D2_TAU, rel=1e-12)
    assert rates[0] * RB87_D2_TAU == pytest.approx(1.0, abs=1e-12), "Gamma_Steck * tau = 1, not 0.5"
