"""EIT cooling closed forms and level C (PLAN.md Section 4.2.3; M3): Section 9.3 row "EIT", Section 9.13 row 123, Section 9.17 row
"Morigi EIT fixture in the plan's sign", Section 9.12 "EIT rate versus mode frequency", Section 13 "EIT detuning sign"."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from qutip_trap.light.bloch import CoolingError
from qutip_trap.prep.eit import (
    EitClosedForm,
    composed_coupling_rad_s,
    coupling_for_target_rad_s,
    dressed_linewidth_rad_s,
    eit_cooling_rate_per_s,
    eit_nbar_at_rmp_tuning,
    eit_nbar_at_tuning,
    eit_rate_coefficients,
    eit_steady_state_nbar,
    from_morigi_sign,
    light_shift_rad_s,
    two_photon_lamb_dicke,
)
from qutip_trap.trap.crystal import (
    alpha_critical,
    axial_modes_dimensionless,
    equilibrium_dimensionless,
    transverse_eigenvalues,
)
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI

NU, GAMMA, OM1, OM2, DELTA = 2.0068, 20.0, 17.0, 17.0, 70.0  # MHz, the plan's fixture


def test_morigi_fixture_in_the_plans_sign_and_its_negative_controls() -> None:
    """Delta = +70 MHz with Omega_r = sqrt(17^2 + 17^2) = 24.04 MHz gives <n>_S = 0.005102 = (gamma/4 Delta)^2; Delta = -70 trips the A_- - A_+ <= 0 guard;
    a single 17 in place of the quadrature sum returns 0.1467, 29x the target (Section 9.17)."""
    om_r = composed_coupling_rad_s([OM1, OM2])
    assert om_r == pytest.approx(24.0416, abs=1e-4)
    assert om_r**2 == pytest.approx(4.0 * NU * (NU + DELTA), rel=2e-3)
    nbar = eit_steady_state_nbar(om_r, NU, DELTA, GAMMA)
    assert nbar == pytest.approx(0.005102, abs=1e-6)
    assert nbar / eit_nbar_at_tuning(GAMMA, DELTA) == pytest.approx(1.0, abs=3e-5)
    with pytest.raises(CoolingError):
        eit_steady_state_nbar(om_r, NU, from_morigi_sign(+DELTA), GAMMA)
    assert from_morigi_sign(-DELTA) == DELTA
    single = eit_steady_state_nbar(17.0, NU, DELTA, GAMMA)
    assert single == pytest.approx(0.1467, abs=2e-4) and single / nbar == pytest.approx(29.0, abs=0.5)
    # the coefficients at the tuning: A_- = Omega_g^2/gamma exactly; the plan's Section 9.13 row 123 prints half of this (7.225 for 17)
    # Omega_g = 17 against gamma = 20: the plan's own fixture is outside the weak-probe regime, so the Section 4.2.8 (vii)
    # escape is passed explicitly (the guard itself is pinned in test_validity_conditions_raise below)
    rc = eit_rate_coefficients(OM1, om_r, NU, DELTA, GAMMA, allow_saturation=True)
    assert rc.A_minus_per_s == pytest.approx(OM1**2 / GAMMA, rel=2e-3)
    assert rc.A_plus_per_s == pytest.approx(0.0734, abs=2e-4)
    assert rc.carrier_weight == 0.0
    assert 0.02**2 * rc.cooling_rate_bare_per_s == pytest.approx(
        5.75e-3, rel=2e-3
    )  # MHz: 5.75 kHz, twice the row's 2.875


def test_rmp_tuning_exact_value_and_the_sign_of_the_bracket() -> None:
    """At Omega_r^2 = 4 nu Delta the exact steady state is (Gamma^2 + 4 nu^2)/(16 Delta (Delta - nu)) = 0.00361702 for gamma = 1, nu = 0.3, Delta = 5;
    the (Delta + nu) form printed before the critique gives 0.00320755 (Section 4.2.3)."""
    assert eit_nbar_at_rmp_tuning(1.0, 0.3, 5.0) == pytest.approx(0.00361702, abs=1e-8)
    assert eit_steady_state_nbar(math.sqrt(4.0 * 0.3 * 5.0), 0.3, 5.0, 1.0) == pytest.approx(
        0.00361702, abs=1e-8
    )
    assert (1.0 + 4.0 * 0.09) / (16.0 * 5.0 * 5.3) == pytest.approx(0.00320755, abs=1e-8)


def test_light_shift_tuning_rule_bandwidth_and_poles() -> None:
    """Roos 2000: Delta = 70 MHz and Omega_sigma = 21.4 MHz put the narrow dressed state 1.6 MHz up (1.599); the inverse rule returns 21.4 for
    nu = 1.6; the poles Delta = 0 and Omega_r = 2 nu raise; Delta < 0 raises; the dressed linewidth is about Gamma nu/Delta."""
    assert light_shift_rad_s(70.0, 21.4) == pytest.approx(1.599, abs=1e-3)
    assert coupling_for_target_rad_s(1.6, 70.0) == pytest.approx(21.41, abs=0.01)
    assert light_shift_rad_s(70.0, coupling_for_target_rad_s(1.6, 70.0)) == pytest.approx(1.6, rel=1e-12)
    with pytest.raises(CoolingError):
        eit_steady_state_nbar(2.0 * 0.3, 0.3, 5.0, 1.0)
    with pytest.raises(CoolingError):
        coupling_for_target_rad_s(1.6, -70.0)
    with pytest.raises(CoolingError):
        eit_steady_state_nbar(10.0, 0.3, 0.0, 1.0)
    bw = dressed_linewidth_rad_s(GAMMA, DELTA, composed_coupling_rad_s([OM1, OM2]))
    assert bw == pytest.approx(GAMMA * NU / DELTA, rel=0.1)
    assert two_photon_lamb_dicke(0.1, 1.0, 0.1, -1.0) == pytest.approx(0.2)
    assert two_photon_lamb_dicke(0.1, 1.0, 0.1, 1.0) == 0.0


def test_lechner_2016_rate_ratio_favours_the_mode_nearer_the_bright_resonance() -> None:
    """Omega_sigma = 30 MHz, Omega_pi = 6.2 MHz and a light shift of 2.2-2.3 MHz (Delta = 96-100 MHz): W = eta^2 (A_- - A_+) with eta^2 ∝ 1/nu gives
    R(3.29 MHz)/R(1.13 MHz) = 2.7-3.8 (3.2 at 2.25 MHz) against the measured 17/5 = 3.4; the plan's Section 9.12 row prints the ratio the
    other way round (M3 finding). The absolute 19e3 s^-1 needs eta = 0.126 at 3.29 MHz, a two-photon |Delta k| of 1.3 k for 40Ca+."""
    gamma, om_s, om_p = 21.57, 30.0, 6.2
    ratios = []
    for shift in (2.2, 2.25, 2.3):
        delta = brentq(lambda d, target=shift: light_shift_rad_s(d, om_s) - target, 10.0, 1000.0)
        # Omega_pi/gamma = 0.29: Lechner's own probe is not weak against gamma, so the escape is explicit
        r_hi = eit_cooling_rate_per_s(
            1.0 / math.sqrt(3.29), om_p, om_s, 3.29, delta, gamma, allow_saturation=True
        )
        r_lo = eit_cooling_rate_per_s(
            1.0 / math.sqrt(1.13), om_p, om_s, 1.13, delta, gamma, allow_saturation=True
        )
        ratios.append(r_hi / r_lo)
    assert ratios[1] == pytest.approx(3.19, abs=0.02)
    assert 2.6 < min(ratios) and max(ratios) < 3.9
    assert all(r > 1.0 for r in ratios)


def test_roos_operating_point_closed_form_is_far_below_his_measured_occupation() -> None:
    """Section 9.3 row "EIT": Roos's 40Ca+ operating point (Delta = 70 MHz, Omega_sigma = 21.4 MHz, so the narrow dressed
    state sits delta = 1.6 MHz up, on his 1.6 MHz radial mode) gives the closed-form floor (gamma/4 Delta)^2 = 0.00593 at
    gamma = 2 pi x 21.57 MHz. His MEASURED nbar_y = 0.18 is 30 times that, so 0.18 is not a floor of this model and is
    recorded as an anchor that reports rather than fails (the M3 finding; the value was previously only printed by
    check_cooling.py section D)."""
    gamma_ca, delta_roos, om_sigma = 21.57, 70.0, 21.4
    assert light_shift_rad_s(delta_roos, om_sigma) == pytest.approx(1.6, abs=2e-3)
    floor = eit_nbar_at_tuning(gamma_ca, delta_roos)
    assert floor == pytest.approx(0.00593, abs=1e-5)
    # the exact steady state at that tuning agrees with (gamma/4 Delta)^2 to the tuning's own precision
    nu = light_shift_rad_s(delta_roos, om_sigma)
    exact = eit_steady_state_nbar(om_sigma, nu, delta_roos, gamma_ca)
    assert exact == pytest.approx(floor, rel=2e-3)
    measured = 0.18
    assert measured / floor == pytest.approx(30.4, abs=0.5)
    assert measured > floor  # the anchor reports; it is not a target of the closed form


def test_lechner_eighteen_ion_radial_band_is_wider_than_the_dressed_cooling_bandwidth() -> None:
    """Section 9.3 row "EIT": Lechner et al. cool an 18-ion 40Ca+ string, quoting 0.01-0.02 on the radial modes in
    under 1 ms at Omega_sigma = 2 pi x 30 MHz, Omega_pi = 2 pi x 6.2 MHz and a light shift of 2.2-2.3 MHz.

    Built here as the actual 18-ion radial mode set (the band edges 3.29 and 1.13 MHz fix the trap anisotropy through
    gamma_p = 1/alpha + 1/2 - mu_p/2, Marquet 2003), with W_m = (|Delta k| x0,m)^2 (A_- - A_+) for a GLOBAL beam, where
    sum_i c_{i,m}^2 = 1 removes the participation exactly. The finding: 17 of the 18 modes relax in under 1 ms, but only
    the 7 modes within the dressed linewidth Gamma' = 0.485 MHz of the light shift reach nbar <= 0.02.
    The 2.16 MHz-wide radial band is 4.45 times that bandwidth, so a SINGLE-tone EIT configuration cannot deliver the
    row's uniform 0.01-0.02 across an 18-ion string; the numbers below are the regression (M3 finding).
    """
    n_ions = 18
    mu, _b = axial_modes_dimensionless(equilibrium_dimensionless(n_ions))
    mu_max = float(mu[-1])
    assert mu_max == pytest.approx(118.533, abs=1e-3)
    radial_com, radial_low = 3.29, 1.13
    omega_z = math.sqrt(2.0 * (radial_com**2 - radial_low**2) / (mu_max - 1.0))
    alpha = (omega_z / radial_com) ** 2
    assert alpha < alpha_critical(n_ions)  # the string is linear, not a zigzag
    radial = np.sort(np.sqrt(transverse_eigenvalues(mu, alpha)) * omega_z)
    assert radial[0] == pytest.approx(radial_low, rel=1e-9)
    assert radial[-1] == pytest.approx(radial_com, rel=1e-9)
    gamma_ca, om_sigma, om_pi = 21.57, 30.0, 6.2
    delta = brentq(lambda d: light_shift_rad_s(d, om_sigma) - 2.25, 10.0, 2000.0)
    assert delta == pytest.approx(97.75, abs=0.05)
    k_397 = TWO_PI / 397e-9
    nbars, times_ms = [], []
    for f_mhz in radial:
        nbars.append(eit_steady_state_nbar(om_sigma, float(f_mhz), delta, gamma_ca))
        x0 = math.sqrt(HBAR_J_S / (2.0 * 39.9626 * ATOMIC_MASS_KG * TWO_PI * f_mhz * 1e6))
        eta = 1.3 * k_397 * x0  # the two-photon |Delta k| = 1.3 k of Section 9.12's Lechner row
        w = 1e6 * eit_cooling_rate_per_s(
            eta, om_pi, om_sigma, float(f_mhz), delta, gamma_ca, allow_saturation=True
        )
        times_ms.append(1e3 / w)
    assert nbars[-1] == pytest.approx(0.04645, rel=2e-3)  # the 3.29 MHz radial centre of mass
    assert nbars[0] == pytest.approx(0.13129, rel=2e-3)  # the 1.13 MHz band edge
    assert min(nbars) == pytest.approx(0.00334, rel=2e-3)
    assert sum(1 for t in times_ms if t < 1.0) == 17
    assert times_ms[0] == pytest.approx(1.1708, rel=2e-3)
    assert sum(1 for n in nbars if n <= 0.02) == 7
    # why: the band is four times wider than the dressed cooling bandwidth at this operating point
    bandwidth = dressed_linewidth_rad_s(gamma_ca, delta, om_sigma)
    assert bandwidth == pytest.approx(0.4853, abs=1e-3)
    assert (radial[-1] - radial[0]) / bandwidth == pytest.approx(4.45, abs=0.05)


def test_closed_form_record_flags_the_weak_probe_regime() -> None:
    cf = EitClosedForm(NU, DELTA, OM1, composed_coupling_rad_s([OM1, OM2]), GAMMA, 0.02)
    assert (
        not cf.weak_probe()
    )  # Omega_g = 17 against Omega_r = 24: outside the closed form's regime (the M3a finding on the fixture)
    assert cf.nbar == pytest.approx(0.005102, abs=1e-6)
    assert cf.light_shift_rad_s == pytest.approx(NU, rel=2e-3)
    assert cf.cooling_rate_per_s == pytest.approx(0.02**2 * cf.coefficients.cooling_rate_bare_per_s)
    weak = EitClosedForm(NU, DELTA, 1.0, composed_coupling_rad_s([OM1, OM2]), GAMMA, 0.02)
    assert weak.weak_probe() and weak.nbar == pytest.approx(cf.nbar)
