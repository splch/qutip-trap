"""EIT cooling closed forms and level C (PLAN.md Section 4.2.3; M3): Section 9.3 row "EIT", Section 9.13 row 123, Section 9.17 row
"Morigi EIT fixture in the plan's sign", Section 9.12 "EIT rate versus mode frequency", Section 13 "EIT detuning sign"."""

from __future__ import annotations

import math

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
    rc = eit_rate_coefficients(OM1, om_r, NU, DELTA, GAMMA)
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
        r_hi = eit_cooling_rate_per_s(1.0 / math.sqrt(3.29), om_p, om_s, 3.29, delta, gamma)
        r_lo = eit_cooling_rate_per_s(1.0 / math.sqrt(1.13), om_p, om_s, 1.13, delta, gamma)
        ratios.append(r_hi / r_lo)
    assert ratios[1] == pytest.approx(3.19, abs=0.02)
    assert 2.6 < min(ratios) and max(ratios) < 3.9
    assert all(r > 1.0 for r in ratios)


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
