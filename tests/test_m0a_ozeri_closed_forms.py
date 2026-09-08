"""Ozeri 2005 Eq. 3, the Ozeri 2007 Table II power regression and Uys's 9Be+ 4.5 T anchor
(PLAN.md 447, 457, 683, 1182, 1410, 1411, 1529; Sections 9.10, 9.13; milestone M0a, audit item E11).

Every one of these was named-and-required by the plan and absent from the tree before 2026-09-07; the
closed forms live in :mod:`qutip_trap.validation.ozeri_closed_forms`.
"""

from __future__ import annotations

import math

import pytest

from qutip_trap.units import C_M_PER_S, TWO_PI
from qutip_trap.validation.atomic_closed_forms import wineland_clock_light_shift
from qutip_trap.validation.ozeri_closed_forms import (
    Uys2010Be9Anchor,
    equal_elastic_rate_detuning,
    free_electron_zeeman_splitting_hz,
    ozeri_2005_clock_stark_shift,
    ozeri_2005_scattering_rate,
    ozeri_epsilon_s_at_power,
    ozeri_rate_difference_estimate,
    ozeri_table_ii_power_printed_lambda_w,
    ozeri_table_ii_power_w,
)

# PLAN.md 457, "the species module stores Ozeri's Table I constants": 9Be+ I = 3/2, gamma/2pi = 19.6 MHz,
# omega_0/2pi = 1.25 GHz, omega_f/2pi = 0.198 THz, lambda = 313.1/313.0 nm
BE9_GAMMA = TWO_PI * 19.6e6
BE9_OMEGA_0 = TWO_PI * 1.25e9
BE9_OMEGA_F = TWO_PI * 0.198e12
BE9_LAMBDA_32_M = 313.0e-9

# the LS-symmetric clock-qubit diagonal amplitudes: a^(1/2) = 1/3, a^(3/2) = 2/3, the same for both states,
# so that a^(1/2) + a^(3/2) = 1 and one shared gamma is exact in LS coupling (PLAN.md 683)
A_HALF = 1.0 / 3.0
A_THREE_HALF = 2.0 / 3.0


# ---- Ozeri 2005 Eq. 3 -----------------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1e-3, 1e-4, 1e-5])
def test_ozeri_2005_eq_3_collapses_onto_wineland_eq_2_17(ratio: float) -> None:
    """With the LS-symmetric amplitudes Eq. 3 reduces, to first order in delta_hf/Delta, to Wineland Eq. 2.17
    -(g_b^2 + g_r^2)(omega_0/3)[1/D^2 + 2/(D - omega_F)^2].

    This is the cross-check that the four amplitudes were read as diagonal LINE STRENGTHS (each pair summing
    to 1) and not as branching ratios, and it ties the two independently transcribed closed forms together.
    """
    delta = BE9_OMEGA_0 / ratio
    exact = ozeri_2005_clock_stark_shift(
        1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, delta, BE9_OMEGA_0, BE9_OMEGA_F
    )
    wineland = wineland_clock_light_shift(1.0, 0.0, BE9_OMEGA_0, delta, BE9_OMEGA_F)
    assert exact == pytest.approx(wineland, rel=8.0 * ratio)
    # and the agreement tightens linearly in delta_hf/Delta: the neglected term is second order
    assert abs(exact / wineland - 1.0) < 8.0 * ratio


def test_ozeri_2005_eq_3_vanishes_for_a_degenerate_pair_and_is_first_order_in_the_splitting() -> None:
    """Delta_hf = 0 with equal amplitudes gives exactly zero (there is no clock qubit), and the shift is
    linear in Delta_hf: doubling the splitting doubles the shift to 1e-4."""
    delta = 100.0 * BE9_OMEGA_F
    zero = ozeri_2005_clock_stark_shift(
        1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, delta, 0.0, BE9_OMEGA_F
    )
    assert zero == 0.0
    one = ozeri_2005_clock_stark_shift(
        1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, delta, BE9_OMEGA_0, BE9_OMEGA_F
    )
    two = ozeri_2005_clock_stark_shift(
        1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, delta, 2.0 * BE9_OMEGA_0, BE9_OMEGA_F
    )
    assert two / one == pytest.approx(2.0, rel=1e-4)


def test_ozeri_2005_eq_3_falls_as_one_over_delta_squared() -> None:
    """Far from both fine-structure levels the leading 1/Delta term cancels by the LS sum rule and
    Delta_St ~ delta_hf/Delta^2: doubling Delta quarters the shift."""
    d1 = 200.0 * BE9_OMEGA_F
    s1 = ozeri_2005_clock_stark_shift(
        1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, d1, BE9_OMEGA_0, BE9_OMEGA_F
    )
    s2 = ozeri_2005_clock_stark_shift(
        1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, 2.0 * d1, BE9_OMEGA_0, BE9_OMEGA_F
    )
    assert s2 / s1 == pytest.approx(0.25, rel=1e-2)


def test_ozeri_2005_eq_3_refuses_a_resonant_denominator() -> None:
    for delta, delta_hf in ((0.0, BE9_OMEGA_0), (BE9_OMEGA_F, BE9_OMEGA_0), (-BE9_OMEGA_0, BE9_OMEGA_0)):
        with pytest.raises(ZeroDivisionError, match="no i gamma/2"):
            ozeri_2005_clock_stark_shift(
                1.0, A_HALF, A_THREE_HALF, A_HALF, A_THREE_HALF, delta, delta_hf, BE9_OMEGA_F
            )


def test_ozeri_2005_scattering_rate_signs_set_the_fall_off() -> None:
    """Opposite-sign amplitudes (a Raman channel) give 1/Delta^4; same-sign ones (elastic) give 1/Delta^2."""
    d1 = 100.0 * BE9_OMEGA_F
    raman = [
        ozeri_2005_scattering_rate(
            1.0, BE9_GAMMA, -math.sqrt(2.0) / 3.0, math.sqrt(2.0) / 3.0, d, BE9_OMEGA_F
        )
        for d in (d1, 2.0 * d1)
    ]
    elastic = [
        ozeri_2005_scattering_rate(1.0, BE9_GAMMA, 1.0 / 3.0, 2.0 / 3.0, d, BE9_OMEGA_F)
        for d in (d1, 2.0 * d1)
    ]
    assert raman[1] / raman[0] == pytest.approx(1.0 / 16.0, rel=2e-2)
    assert elastic[1] / elastic[0] == pytest.approx(0.25, rel=1e-2)


# ---- Ozeri 2007 Table II: the power regression -----------------------------------------------------------

W0_M = 20e-6
OMEGA_R_HALF = TWO_PI * 0.25e6


def test_table_ii_regression_round_trips_and_pins_the_eq_17_invariant() -> None:
    """eps_S x P is the Eq. 17 invariant, so the forward and inverse directions round trip exactly and the
    regression is power-independent (PLAN.md 1411)."""
    for eps in (0.337e-4, 1.0e-4, 4.0e-4):
        p = ozeri_table_ii_power_w(eps, OMEGA_R_HALF, BE9_LAMBDA_32_M, W0_M)
        assert ozeri_epsilon_s_at_power(p, OMEGA_R_HALF, BE9_LAMBDA_32_M, W0_M) == pytest.approx(
            eps, rel=1e-12
        )
        assert eps * p == pytest.approx(3.365441e-7, rel=1e-6)


def test_table_ii_plan_pair_for_be9_does_not_satisfy_eq_17() -> None:
    """PLAN.md 9.13 pairs eps_S = 0.337e-4 with P_0 = 3.37 mW for 9Be+ at w_0 = 20 um and
    Omega_R/2pi = 0.25 MHz. RECOMPUTED HERE: Eq. 17 as printed gives 9.986 mW at that eps_S, so the plan's
    own pair violates its own Eq. 17 by a factor 2.9633 (the prefactor would have to be 9, not 3).

    This is a PLAN inconsistency, recorded in the ledger as ``anchor.m0a.ozeri_table_ii`` [contested] and
    reported rather than absorbed. The 25Mg+ / 43Ca+ / 67Zn+ / 111Cd+ rows cannot be checked at all: their
    lambda_{3/2}, gamma and omega_f are not in PLAN.md (only the 9Be+ and 43Ca+ Table I rows are, and 43Ca+'s
    eps_S is not printed). The Yb+ row is excluded as the plan instructs.
    """
    plan_eps, plan_power_w = 0.337e-4, 3.37e-3
    ours = ozeri_table_ii_power_w(plan_eps, OMEGA_R_HALF, BE9_LAMBDA_32_M, W0_M)
    assert ours == pytest.approx(9.9865e-3, rel=1e-4)
    assert ours / plan_power_w == pytest.approx(2.9633, rel=1e-3)
    # with the prefactor 9 the plan's pair is reproduced to 1.2%, which localizes the slip
    assert ours / 3.0 == pytest.approx(plan_power_w, rel=1.3e-2)


def test_printed_lambda_reading_is_the_negative_control_at_two_pi_cubed() -> None:
    """Ozeri's printed "lambda_{3/2} = c/omega_{3/2}" read literally makes omega a factor 2 pi small, so
    Eq. 17's P_0 comes out exactly (2 pi)^3 = 248.05 times smaller (PLAN.md 457, [verified flag]).

    PLAN.md's own negative control, "the printed lambda = c/omega gives 133 mW for 9Be+", is 133/3.37 = 39.47
    = (2 pi)^2, which is NOT the ratio between the two readings of Eq. 17 under any grouping; it is reported
    here and not pinned.
    """
    good = ozeri_table_ii_power_w(0.337e-4, OMEGA_R_HALF, BE9_LAMBDA_32_M, W0_M)
    printed = ozeri_table_ii_power_printed_lambda_w(0.337e-4, OMEGA_R_HALF, BE9_LAMBDA_32_M, W0_M)
    assert good / printed == pytest.approx(TWO_PI**3, rel=1e-12)
    assert 133.0 / 3.37 == pytest.approx(TWO_PI**2, rel=1e-3)  # what the plan's pair actually is


def test_table_ii_regression_rejects_unphysical_inputs() -> None:
    for eps in (0.0, -1e-4):
        with pytest.raises(ValueError):
            ozeri_table_ii_power_w(eps, OMEGA_R_HALF, BE9_LAMBDA_32_M, W0_M)
    with pytest.raises(ValueError):
        ozeri_table_ii_power_w(1e-4, OMEGA_R_HALF, -1.0, W0_M)


# ---- Uys 2010: the 9Be+ 4.5 T anchor ---------------------------------------------------------------------


def test_uys_4_5_tesla_anchor_constants_and_the_free_electron_sanity_check() -> None:
    """PLAN.md 683 and 9.16 row "Signed fine-structure detunings": Omega_z/2pi = 124.1 GHz at 4.5 T with
    313 nm light, P3/2 resonances at -79.4 and -37.7 GHz from the cycling transition and equal elastic rates
    near -56 GHz. The plan's own check is 2 x 13.996 GHz/T x 4.5 T = 126.0 GHz, "order-of-magnitude sanity
    only" (1.5% above the printed Omega_z)."""
    a = Uys2010Be9Anchor
    assert free_electron_zeeman_splitting_hz(a.FIELD_T) == pytest.approx(125.964e9, rel=1e-6)
    assert free_electron_zeeman_splitting_hz(a.FIELD_T) / a.ZEEMAN_SPLITTING_HZ == pytest.approx(
        1.015, rel=1e-3
    )
    # the equal-rate detuning lies BETWEEN the two P3/2 resonances, as an amplitude crossing must
    assert a.P32_RESONANCE_LOW_HZ < a.EQUAL_ELASTIC_RATE_DETUNING_HZ < a.P32_RESONANCE_HIGH_HZ
    assert C_M_PER_S / (a.WAVELENGTH_NM * 1e-9) == pytest.approx(957.80e12, rel=1e-4)


def test_the_rate_difference_estimator_is_exactly_zero_where_gamma_el_is_largest() -> None:
    """PLAN.md 1410: "near -56 GHz in 9Be+ at 4.5 T the amplitude form gives about 5x the rate-difference
    estimate, which returns about zero".

    The plan's [corrected] reading of its own row is that this is a RATIO TO ZERO, so no finite factor may be
    pinned. What is testable, and pinned here, is the structure: at the detuning where the two elastic rates
    coincide the Ozeri estimator (Gamma_uu - Gamma_dd)^2/(Gamma_uu + Gamma_dd) is exactly 0 while Uys's
    Gamma_el -- the square of the amplitude DIFFERENCE -- is strictly positive.
    """
    import numpy as np

    from qutip_trap.validation.atomic_closed_forms import uys_bounds, uys_gamma_el, uys_gamma_ij

    # two states whose signed two-path amplitude sums are equal and OPPOSITE: equal rates, maximal dephasing
    a_up = np.array([[0.4, -0.9]])
    a_dn = -a_up
    g_uu = uys_gamma_ij(a_up, 1.0, 1.0)
    g_dd = uys_gamma_ij(a_dn, 1.0, 1.0)
    assert g_uu == pytest.approx(g_dd, rel=1e-15)
    assert ozeri_rate_difference_estimate(g_uu, g_dd) == 0.0
    gamma_el = uys_gamma_el(a_dn, a_up, 1.0, 1.0)
    assert gamma_el > 0.0
    lo, hi = uys_bounds(g_dd, g_uu)
    assert gamma_el == pytest.approx(hi, rel=1e-12), "equal and opposite amplitudes saturate the upper bound"
    assert lo == 0.0


def test_the_equal_elastic_rate_detuning_is_a_closed_form_crossing() -> None:
    """The crossing D = u D_f/(u + v) with u, v the summed a^(1/2) and a^(3/2) amplitudes: at that detuning
    the two states' amplitude sums are equal and opposite, so the elastic rates are equal by construction."""
    delta_f = BE9_OMEGA_F
    a_up = (0.5, -0.2)
    a_dn = (0.1, 0.4)
    d = equal_elastic_rate_detuning(a_up, a_dn, delta_f)
    r_up = ozeri_2005_scattering_rate(1.0, BE9_GAMMA, a_up[0], a_up[1], d, delta_f)
    r_dn = ozeri_2005_scattering_rate(1.0, BE9_GAMMA, a_dn[0], a_dn[1], d, delta_f)
    assert r_up == pytest.approx(r_dn, rel=1e-12)
    assert ozeri_rate_difference_estimate(r_up, r_dn) == pytest.approx(0.0, abs=1e-9 * r_up)
