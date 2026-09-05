"""The closed-form cooling oracles of PLAN.md Section 4.2 (M3): Section 9.3 rows "Doppler force", "Sideband limit", "Lamb-Dicke
rate framework", "Sideband floor decomposition", "Doppler limit with recoil", "V-system diffusion", "Effective two-level
parameters", "Lambda system with repumper", "Franck-Condon index", "40Ca+ Lamb-Dicke recomputation", "Cooling-rate
saturation"; Section 9.12 "Doppler-limit fork"; Section 9.13 row "Sympathetic-cooling optimum and steady state"; Section 9.15
"Repump recoil per cycle"."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import eval_genlaguerre, factorial

from qutip_trap.hilbert.operators import displacement_element_analytic
from qutip_trap.light.bloch import CoolingError
from qutip_trap.prep.closed_forms import (
    absorption_diffusion,
    bose_occupation,
    doppler_force_energy_j,
    doppler_force_minimum_energy_j,
    doppler_force_nbar,
    doppler_force_optimum_detuning,
    doppler_friction_kg_per_s,
    doppler_limit_nbar,
    doppler_temperature_k,
    effective_two_level,
    effective_two_level_rate,
    effective_two_level_valid,
    half_rate_emission_diffusion,
    lamb_dicke_parameter,
    lambda_repumper_minimum,
    lambda_repumper_nbar,
    lorentzian_scattering_rate,
    offresonant_scattering_rate_per_s,
    repump_recoil_quanta,
    sideband_cooling_rate_n,
    sideband_floor,
    stenholm_coefficients,
    two_level_excited_population,
    v_system_diffusion,
    x0_m,
)
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI

# ---- Doppler force model (Section 4.2.1; 9.12 "Doppler-limit fork") ------------------------------------------------------------


def test_doppler_limit_fork_and_bose_occupations() -> None:
    """k_B T = hbar Gamma/2 (alpha = 1) against 0.35 hbar Gamma (alpha = 2/5): 0.480 vs 0.336 mK at Gamma/2pi = 20 MHz (ratio 1.4286), 1.680 vs
    1.176 mK at 70 MHz (Berkeland's 1.7 mK), and a 1 MHz mode holds 9.5 against 6.5 quanta."""
    g = TWO_PI * 20e6
    t1 = doppler_temperature_k(doppler_force_minimum_energy_j(g, 0.0, 1.0))
    t2 = doppler_temperature_k(doppler_force_minimum_energy_j(g, 0.0, 0.4))
    assert t1 * 1e3 == pytest.approx(0.480, abs=0.001) and t2 * 1e3 == pytest.approx(0.336, abs=0.001)
    assert t1 / t2 == pytest.approx(2.0 / 1.4, rel=1e-9)
    assert bose_occupation(t1, TWO_PI * 1e6) == pytest.approx(9.5, abs=0.02)
    assert bose_occupation(t2, TWO_PI * 1e6) == pytest.approx(6.5, abs=0.02)
    g70 = TWO_PI * 70e6
    assert doppler_temperature_k(doppler_force_minimum_energy_j(g70, 0.0, 1.0)) * 1e3 == pytest.approx(
        1.680, abs=0.001
    )
    assert doppler_temperature_k(doppler_force_minimum_energy_j(g70, 0.0, 0.4)) * 1e3 == pytest.approx(
        1.176, abs=0.001
    )


def test_force_model_optimum_and_the_zero_point_offset() -> None:
    """Delta_opt = -(Gamma/2) sqrt(1 + s); E_K = E/2 = (1 + alpha) hbar Gamma/8 (hbar Gamma/6 for 1/3, 7 hbar Gamma/40 for 2/5); at Gamma/nu = 1e3
    the force model gives 349.5 against the (Gamma/4 nu)(1 + alpha) = 350, the 1/2 being the zero point (Section 9.3)."""
    for s in (0.0, 0.5, 3.0):
        assert doppler_force_optimum_detuning(1.0, s) == pytest.approx(-0.5 * math.sqrt(1.0 + s))
        grid = np.linspace(-3.0, -0.05, 4001)
        e = [doppler_force_energy_j(1.0, d, s, 0.4) for d in grid]
        assert grid[int(np.argmin(e))] == pytest.approx(-0.5 * math.sqrt(1.0 + s), abs=2e-3)
    assert doppler_force_minimum_energy_j(1.0, 0.0, 1.0 / 3.0) / 2.0 == pytest.approx(HBAR_J_S / 6.0)
    assert doppler_force_minimum_energy_j(1.0, 0.0, 0.4) / 2.0 == pytest.approx(7.0 * HBAR_J_S / 40.0)
    assert doppler_force_nbar(1.0, -0.5, 0.0, 1e-3, 0.4) == pytest.approx(349.5, abs=1e-9)
    assert doppler_limit_nbar(1.0, 1e-3, 0.4) == pytest.approx(350.0)
    with pytest.raises(CoolingError):
        doppler_force_energy_j(1.0, +0.5, 0.0, 0.4)
    assert (
        doppler_friction_kg_per_s(1.0, 1.0, 1.0, -0.5) < 0.0 < doppler_friction_kg_per_s(1.0, 1.0, 1.0, +0.5)
    )
    assert two_level_excited_population(1.0, 1.0 / math.sqrt(2.0), 0.0) == pytest.approx(0.25)


# ---- Stenholm coefficients and the floors (Sections 4.2.2, 4.2.8 iv; 9.13 row 127) ---------------------------------------------


def test_sideband_floor_coefficients_and_alpha_to_zero() -> None:
    """(gamma/nu)^2 (alpha + 1/4) with gamma = Gamma/2: 7/12 and 13/20 for alpha = 1/3, 2/5; alpha -> 0 leaves (Gamma/4 nu)^2 = 6.25e-6 at Gamma/nu = 0.01."""
    assert sideband_floor(2.0, 1.0, 1.0 / 3.0) == pytest.approx(7.0 / 12.0)
    assert sideband_floor(2.0, 1.0, 0.4) == pytest.approx(13.0 / 20.0)
    assert sideband_floor(0.01, 1.0, 0.0) == pytest.approx(6.25e-6)
    assert sideband_floor(0.01, 1.0, 0.4) / sideband_floor(0.01, 1.0, 0.0) == pytest.approx(2.6)


def test_stenholm_coefficients_carrier_term_cancels_in_the_rate_but_not_in_the_steady_state() -> None:
    gamma, nu = 1.0, 50.0
    with_carrier = stenholm_coefficients(0.1, gamma, nu, -nu, 0.4)
    without = stenholm_coefficients(0.1, gamma, nu, -nu, 0.0)
    assert with_carrier.cooling_rate_bare_per_s == pytest.approx(without.cooling_rate_bare_per_s, rel=1e-12)
    assert with_carrier.nbar / without.nbar == pytest.approx(2.6, rel=1e-3)
    assert with_carrier.nbar == pytest.approx(sideband_floor(gamma, nu, 0.4), rel=2e-3)
    # halving Omega leaves nbar unchanged (Section 9.3)
    assert stenholm_coefficients(0.05, gamma, nu, -nu, 0.4).nbar == pytest.approx(
        with_carrier.nbar, rel=1e-12
    )
    assert lorentzian_scattering_rate(0.1, 1.0, 0.0) == pytest.approx(0.01)


def test_morigi_walther_sympathetic_steady_states_are_stenholms_form_with_alpha_two_fifths() -> None:
    """Section 9.13 row 127: A_- - A_+ = 1 - 1/(16 Omega^2/gamma^2 + 1) with their Omega the trap frequency, and nbar_ss = 0.5, 0.1475,
    0.039522, 0.0064703, 0.0016231 at Omega/gamma = 0.5, 1, 2, 5, 10, limit 0.1625 (gamma/Omega)^2, independent of W_k and g."""
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


def test_cooling_rate_saturates_at_half_the_linewidth() -> None:
    """R_n -> Gamma~/2 for eta sqrt n Omega >> Gamma~ and (eta Omega)^2/Gamma~ at n = 1 for a weak drive (Roos p. 21)."""
    assert sideband_cooling_rate_n(1, 0.1, 0.01, 1.0) == pytest.approx(1e-6, rel=1e-5)
    assert sideband_cooling_rate_n(100, 1.0, 100.0, 1.0) == pytest.approx(0.5, rel=1e-4)
    assert sideband_cooling_rate_n(0, 0.1, 1.0, 1.0) == 0.0


# ---- Marzoli's effective two-level parameters (Section 4.2.8 v; Section 13 "Effective two-level roles") ------------------------


def test_effective_two_level_ratios_light_shift_and_role_assignment() -> None:
    """gamma'/Gamma' = (Gamma_10 + Gamma_12)/(2 Gamma_10) = 0.55 (Xi) and /(2 Gamma_12) = 5.5 (V) for Gamma_12 = 0.1 Gamma_10, never 1/2; the aux beam
    at Omega_12 = 0.2, delta_12 = -1 shifts the level by +0.0077 Gamma_10 so the optimum is delta_20 = -0.058 (Marzoli Fig. 3); V cools at +nu."""
    xi = effective_two_level(1.0, 0.1, 0.2, -1.0, "Xi")
    v = effective_two_level(1.0, 0.1, 0.2, -1.0, "V")
    assert xi.excitation_fraction == pytest.approx(0.01 / 1.3025, rel=1e-9)
    assert xi.ratio == pytest.approx(0.55) and v.ratio == pytest.approx(5.5)
    assert xi.light_shift_rad_s == pytest.approx(0.0077, abs=1e-4)
    assert xi.optimum_detuning_rad_s(0.05) == pytest.approx(-0.058, abs=5e-4)
    assert v.optimum_detuning_rad_s(0.05) > 0.0
    assert xi.gamma_prime_rad_s == pytest.approx(
        xi.excitation_fraction
    ) and v.gamma_prime_rad_s == pytest.approx(0.1 * v.excitation_fraction)
    assert xi.resolved(0.05) and not xi.resolved(0.005)
    assert effective_two_level_valid(1.0, 0.1, 0.2, -0.05) and not effective_two_level_valid(
        1.0, 0.1, 2.0, -0.05
    )
    # a two-level line with gamma' = Gamma/2 recovers the Lorentzian W(Delta)
    assert effective_two_level_rate(0.1, 0.5, 0.3) == pytest.approx(
        lorentzian_scattering_rate(0.1, 1.0, 0.3), rel=1e-12
    )
    with pytest.raises(ValueError):
        effective_two_level(1.0, 0.1, 0.2, -1.0, "Lambda")  # type: ignore[arg-type]


# ---- V-system diffusion (Marzoli Eq. B4) ---------------------------------------------------------------------------------------


def test_v_system_diffusion_is_half_rate_emission_plus_absorption_and_the_negative_controls_fail() -> None:
    s10, rho00, g12, g10, eta10, eta12, alpha = 0.3, 0.8, 0.4, 1.0, 0.1, 0.15, 0.4
    rho11 = 0.5 * s10 * rho00
    d = v_system_diffusion(s10, rho00, g12, g10, eta10, eta12, alpha)
    emission = half_rate_emission_diffusion([(rho11, g10, alpha, eta10), (rho11, g12, alpha, eta12)])
    absorption = absorption_diffusion(rho11 * (g10 + g12), eta10)
    assert d == pytest.approx(emission + absorption, rel=1e-12)
    # alpha on the absorbed photon, or the Gamma_10 branch carrying alpha alone (no 1), give different numbers
    wrong_absorbed = (
        s10 / 4.0 * rho00 * (g12 * alpha * (eta10**2 + eta12**2) + g10 * eta10**2 * (1.0 + alpha))
    )
    wrong_branch = s10 / 4.0 * rho00 * (g12 * (eta10**2 + alpha * eta12**2) + g10 * eta10**2 * alpha)
    assert wrong_absorbed != pytest.approx(d, rel=1e-3) and wrong_branch != pytest.approx(d, rel=1e-3)


# ---- Cirac's Lambda system with a repumper (Section 9.3) -------------------------------------------------------------------------


def test_cirac_lambda_repumper_minima_and_doppler_reference() -> None:
    om, nb = lambda_repumper_minimum(0.9, 0.1, 10.0)
    assert (om, nb) == (pytest.approx(2.05, abs=0.01), pytest.approx(0.554, abs=1e-3))
    om, nb = lambda_repumper_minimum(0.1, 0.1, 10.0)
    assert (om, nb) == (pytest.approx(3.56, abs=0.01), pytest.approx(2.66, abs=5e-3))
    assert lambda_repumper_nbar(0.1, 0.1, 10.0, om) == pytest.approx(nb, rel=1e-12)
    assert lambda_repumper_nbar(0.1, 0.1, 10.0, 2.0 * om) > nb
    assert 1.0 / (2.0 * 0.1) - 0.5 == pytest.approx(4.5)


# ---- repump recoil (Che Eq. 6; Section 9.15) -----------------------------------------------------------------------------------


def test_repump_recoil_per_cycle_and_the_2pi_trap() -> None:
    g, d = TWO_PI * 41.3e6, TWO_PI * 1.789e9
    dn = repump_recoil_quanta(10e-6, 1.0, g, 0.3, 0.3 * 0.0 + d)
    assert dn == pytest.approx(0.031117, rel=2e-5)
    assert repump_recoil_quanta(10e-6, 1.0, 41.3e6, 0.3, 1.789e9) == pytest.approx(0.0049524, rel=2e-5)
    assert dn / repump_recoil_quanta(10e-6, 1.0, 41.3e6, 0.3, 1.789e9) == pytest.approx(TWO_PI, rel=1e-9)
    r_sc = offresonant_scattering_rate_per_s(1.0, g, d)
    assert r_sc == pytest.approx(1.7287e4, rel=1e-4)
    assert r_sc * 10e-6 * 2.0 * 0.09 == pytest.approx(dn, rel=1e-12)


# ---- Lamb-Dicke recomputation and the Franck-Condon index (Section 9.3) --------------------------------------------------------


def test_ca40_lamb_dicke_parameters_at_1_mhz() -> None:
    """eta_729 = 0.0969 and eta_393 = 0.180 at 2 pi x 1 MHz with x0 = 11.24 nm for 40 u (Roos thesis p. 28: 0.096, 0.179)."""
    m = 40.0 * ATOMIC_MASS_KG
    assert x0_m(m, TWO_PI * 1e6) * 1e9 == pytest.approx(11.24, abs=0.005)
    assert lamb_dicke_parameter(TWO_PI / 729.147e-9, m, TWO_PI * 1e6) == pytest.approx(0.0969, abs=5e-4)
    assert lamb_dicke_parameter(TWO_PI / 393.366e-9, m, TWO_PI * 1e6) == pytest.approx(0.180, abs=5e-4)


def test_roos_eq_3_11_as_printed_overstates_the_lower_sidebands() -> None:
    """|<n+m|D|n>| = e^{-eta^2/2} eta^|m| L^(|m|)_{n<}(eta^2) sqrt(n<!/n>!); the printed L_n is 2.00x too large at (1, -1), 3.3x at (3, -2)
    and 2.9x at (8, -3) for eta = 0.05 (Section 9.3 "Franck-Condon index")."""
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
