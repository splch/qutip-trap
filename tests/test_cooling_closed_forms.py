"""The closed-form cooling rates: the Doppler force model and its alpha fork, Stenholm's coefficients and floors, the
weak-drive guard, the 40Ca+ Lamb-Dicke parameters and the Franck-Condon index."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import eval_genlaguerre, factorial

from qutip_trap.dynamics.operators import displacement_element_analytic
from qutip_trap.light.bloch import CoolingError
from qutip_trap.prep.closed_forms import (
    doppler_force_energy_j,
    doppler_force_nbar,
    lamb_dicke_parameter,
    lorentzian_scattering_rate,
    stenholm_coefficients,
    x0_m,
)
from qutip_trap.prep.validity import ValidityError, assert_weak_drive
from qutip_trap.species.metastable import bose_occupation
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, K_B_J_PER_K, TWO_PI


def test_doppler_limit_fork_and_bose_occupations() -> None:
    """At Delta = -Gamma/2 the force model gives k_B T = 0.480 and 0.336 mK (alpha = 1, 2/5) at Gamma/2pi = 20 MHz and
    1.680 and 1.176 mK at 70 MHz (1e-3 mK), a 1 MHz mode holding 9.5 and 6.5 quanta (0.02)."""
    g = TWO_PI * 20e6
    t1 = doppler_force_energy_j(g, -0.5 * g, 0.0, 1.0) / K_B_J_PER_K
    t2 = doppler_force_energy_j(g, -0.5 * g, 0.0, 0.4) / K_B_J_PER_K
    assert t1 * 1e3 == pytest.approx(0.480, abs=0.001) and t2 * 1e3 == pytest.approx(0.336, abs=0.001)
    assert t1 / t2 == pytest.approx(2.0 / 1.4, rel=1e-9)
    assert bose_occupation(1e6, t1) == pytest.approx(9.5, abs=0.02)
    assert bose_occupation(1e6, t2) == pytest.approx(6.5, abs=0.02)
    g70 = TWO_PI * 70e6
    assert doppler_force_energy_j(g70, -0.5 * g70, 0.0, 1.0) / K_B_J_PER_K * 1e3 == pytest.approx(
        1.680, abs=0.001
    )
    assert doppler_force_energy_j(g70, -0.5 * g70, 0.0, 0.4) / K_B_J_PER_K * 1e3 == pytest.approx(
        1.176, abs=0.001
    )


def test_force_model_optimum_and_the_zero_point_offset() -> None:
    """The force-model energy is minimal at Delta = -(Gamma/2) sqrt(1 + s) (2e-3), E_K is hbar Gamma/6 and
    7 hbar Gamma/40 for alpha = 1/3 and 2/5, nbar = 349.5 at Gamma/nu = 1e3 (1e-9), and a blue detuning raises."""
    for s in (0.0, 0.5, 3.0):
        grid = np.linspace(-3.0, -0.05, 4001)
        e = [doppler_force_energy_j(1.0, d, s, 0.4) for d in grid]
        assert grid[int(np.argmin(e))] == pytest.approx(-0.5 * math.sqrt(1.0 + s), abs=2e-3)
    assert doppler_force_energy_j(1.0, -0.5, 0.0, 1.0 / 3.0) / 2.0 == pytest.approx(HBAR_J_S / 6.0)
    assert doppler_force_energy_j(1.0, -0.5, 0.0, 0.4) / 2.0 == pytest.approx(7.0 * HBAR_J_S / 40.0)
    assert doppler_force_nbar(1.0, -0.5, 0.0, 1e-3, 0.4) == pytest.approx(349.5, abs=1e-9)
    with pytest.raises(CoolingError):
        doppler_force_energy_j(1.0, +0.5, 0.0, 0.4)


def test_stenholm_coefficients_carrier_term_cancels_in_the_rate_but_not_in_the_steady_state() -> None:
    """Stenholm's floor is (Gamma/2 nu)^2 [(eta~/eta)^2 + 1/4] (2e-3), alpha = 2/5 raising the alpha = 0 value 2.6 times
    (1e-3) at an unchanged rate; halving Omega leaves it unchanged and a blue detuning raises."""
    gamma, nu = 1.0, 50.0
    with_carrier = stenholm_coefficients(0.1, gamma, nu, -nu, 0.4)
    without = stenholm_coefficients(0.1, gamma, nu, -nu, 0.0)
    assert with_carrier.cooling_rate_bare_per_s == pytest.approx(without.cooling_rate_bare_per_s, rel=1e-12)
    assert with_carrier.nbar / without.nbar == pytest.approx(2.6, rel=1e-3)
    assert with_carrier.nbar == pytest.approx((gamma / (2.0 * nu)) ** 2 * (0.4 + 0.25), rel=2e-3)
    assert stenholm_coefficients(0.05, gamma, nu, -nu, 0.4).nbar == pytest.approx(
        with_carrier.nbar, rel=1e-12
    )
    assert with_carrier.cooling_rate_per_s(0.1) == pytest.approx(0.01 * with_carrier.cooling_rate_bare_per_s)
    assert lorentzian_scattering_rate(0.1, 1.0, 0.0) == pytest.approx(0.01)
    with pytest.raises(CoolingError):
        _ = stenholm_coefficients(0.05, gamma, 0.2, +0.5, 0.0).nbar


def test_morigi_walther_sympathetic_steady_states_are_stenholms_form_with_alpha_two_fifths() -> None:
    """Stenholm's form with alpha = 2/5 gives Morigi and Walther's A_- - A_+ = 1 - 1/(16 Omega^2/gamma^2 + 1) (1e-12)
    and nbar = 0.5, 0.1475, 0.039522, 0.0064703, 0.0016231 at Omega/gamma = 0.5 to 10 (2e-4), limit 0.1625
    (gamma/Omega)^2."""
    expected = {0.5: 0.5, 1.0: 0.1475, 2.0: 0.039522, 5.0: 0.0064703, 10.0: 0.0016231}
    for ratio, nbar in expected.items():
        nu = ratio  # in units of gamma = Gamma (full width) = 1
        rc = stenholm_coefficients(0.01, 1.0, nu, -nu, 0.4)
        assert rc.nbar == pytest.approx(nbar, rel=2e-4)
        normalized = rc.cooling_rate_bare_per_s / lorentzian_scattering_rate(0.01, 1.0, 0.0)
        assert normalized == pytest.approx(1.0 - 1.0 / (16.0 * ratio**2 + 1.0), rel=1e-12)
    assert stenholm_coefficients(0.01, 1.0, 100.0, -100.0, 0.4).nbar * 100.0**2 == pytest.approx(
        0.1625, rel=1e-3
    )


def test_the_closed_forms_refuse_a_saturated_drive_unless_the_caller_says_so() -> None:
    """The closed forms refuse Omega/Gamma above 0.1 unless allow_saturation is passed."""
    for bad in (0.11, 0.5, 1.0):
        with pytest.raises(ValidityError, match="Omega/Gamma"):
            lorentzian_scattering_rate(bad, 1.0, 0.0)
        with pytest.raises(ValidityError, match="Omega/Gamma"):
            stenholm_coefficients(bad, 1.0, 50.0, -50.0, 0.4)
    assert lorentzian_scattering_rate(0.5, 1.0, 0.0, allow_saturation=True) == pytest.approx(0.25)
    assert stenholm_coefficients(0.5, 1.0, 50.0, -50.0, 0.4, allow_saturation=True).nbar > 0.0
    assert_weak_drive(1.0, 1.0, allow_saturation=True)
    with pytest.raises(ValidityError, match="linewidth is positive"):
        assert_weak_drive(0.1, 0.0)


def test_ca40_lamb_dicke_parameters_at_1_mhz() -> None:
    """eta_729 = 0.0969 and eta_393 = 0.180 (5e-4) at 2 pi x 1 MHz with x0 = 11.24 nm for 40 u (Roos thesis p. 28)."""
    m = 40.0 * ATOMIC_MASS_KG
    assert x0_m(m, TWO_PI * 1e6) * 1e9 == pytest.approx(11.24, abs=0.005)
    assert lamb_dicke_parameter(TWO_PI / 729.147e-9, m, TWO_PI * 1e6) == pytest.approx(0.0969, abs=5e-4)
    assert lamb_dicke_parameter(TWO_PI / 393.366e-9, m, TWO_PI * 1e6) == pytest.approx(0.180, abs=5e-4)


def test_roos_eq_3_11_as_printed_overstates_the_lower_sidebands() -> None:
    """|<n+m|D|n>| = e^{-eta^2/2} eta^|m| L^(|m|)_{n<}(eta^2) sqrt(n<!/n>!) to 1e-10, and Roos's printed L_n overstates
    it 2.00, 3.3 and 2.9 times at (1, -1), (3, -2) and (8, -3) for eta = 0.05."""
    eta = 0.05
    ratios = {}
    for n, m in ((1, -1), (3, -2), (8, -3)):
        lo, hi = min(n, n + m), max(n, n + m)
        exact = abs(displacement_element_analytic(n + m, n, 1j * eta))
        corrected = (
            math.exp(-(eta**2) / 2)
            * eta ** abs(m)
            * abs(eval_genlaguerre(lo, abs(m), eta**2))
            * math.sqrt(factorial(lo) / factorial(hi))
        )
        printed = (
            math.exp(-(eta**2) / 2)
            * eta ** abs(m)
            * abs(eval_genlaguerre(n, abs(m), eta**2))
            * math.sqrt(factorial(lo) / factorial(hi))
        )
        assert corrected == pytest.approx(exact, rel=1e-10)
        ratios[(n, m)] = printed / exact
    assert ratios[(1, -1)] == pytest.approx(2.00, abs=0.01)
    assert ratios[(3, -2)] == pytest.approx(3.3, abs=0.15)
    assert ratios[(8, -3)] == pytest.approx(2.9, abs=0.15)
