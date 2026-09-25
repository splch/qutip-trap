"""EIT cooling: the closed forms in the plan's sign (Morigi's fixture, the RMP tuning, the light-shift rule, the poles), the
Lechner and Roos anchors, the weak-probe guard, and the level-C Lambda system against the closed form."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions
from qutip_trap.light.bloch import BlochModel, CoolingError
from qutip_trap.prep.eit import (
    EitClosedForm,
    coupling_for_target_rad_s,
    dressed_linewidth_rad_s,
    eit_rate_coefficients,
    eit_steady_state_nbar,
    light_shift_rad_s,
)
from qutip_trap.prep.level_c import level_c_steady_state
from qutip_trap.prep.validity import ValidityError
from qutip_trap.trap.crystal import axial_modes_dimensionless, equilibrium_dimensionless
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI
from tests.bloch_fixtures import (
    LAMBDA_EXCITED,
    LAMBDA_GROUND_MINUS,
    LAMBDA_GROUND_PLUS,
    WAVELENGTH_M,
    circular_beam,
    gamma_rad_s,
    lambda_atom,
    structure,
)

NU, GAMMA, OM1, OM2, DELTA = 2.0068, 20.0, 17.0, 17.0, 70.0  # MHz, the plan's fixture
OM_R = math.sqrt(OM1**2 + OM2**2)
"""The two coupling Rabi frequencies compose as the quadrature sum."""


def test_morigi_fixture_in_the_plans_sign_and_its_negative_controls() -> None:
    """Delta = +70 MHz with Omega_r = sqrt(17^2 + 17^2) = 24.04 MHz gives <n>_S = 0.005102 = (gamma/4 Delta)^2;
    Delta = -70 (Morigi's sign) trips the A_- - A_+ <= 0 guard; a single 17 in place of the quadrature sum returns
    0.1467, 29x the target."""
    om_r = OM_R
    assert om_r == pytest.approx(24.0416, abs=1e-4)
    assert om_r**2 == pytest.approx(4.0 * NU * (NU + DELTA), rel=2e-3)
    nbar = eit_steady_state_nbar(om_r, NU, DELTA, GAMMA)
    assert nbar == pytest.approx(0.005102, abs=1e-6)
    assert nbar / (GAMMA / (4.0 * DELTA)) ** 2 == pytest.approx(1.0, abs=3e-5)
    with pytest.raises(CoolingError):
        eit_steady_state_nbar(om_r, NU, -DELTA, GAMMA)
    single = eit_steady_state_nbar(17.0, NU, DELTA, GAMMA)
    assert single == pytest.approx(0.1467, abs=2e-4) and single / nbar == pytest.approx(29.0, abs=0.5)
    # the coefficients at the tuning: A_- = Omega_g^2/gamma exactly (twice the 7.225 the plan prints for 17); Omega_g = 17
    # against gamma = 20 is outside the weak-probe regime, so the escape is passed explicitly
    rc = eit_rate_coefficients(OM1, om_r, NU, DELTA, GAMMA, allow_saturation=True)
    assert rc.A_minus_per_s == pytest.approx(OM1**2 / GAMMA, rel=2e-3)
    assert rc.A_plus_per_s == pytest.approx(0.0734, abs=2e-4)
    assert rc.carrier_weight == 0.0
    assert 0.02**2 * rc.cooling_rate_bare_per_s == pytest.approx(
        5.75e-3, rel=2e-3
    )  # MHz: 5.75 kHz, twice the row's 2.875


def test_rmp_tuning_exact_value() -> None:
    """At Omega_r^2 = 4 nu Delta the exact steady state is (Gamma^2 + 4 nu^2)/(16 Delta (Delta - nu)) = 0.00361702 for
    gamma = 1, nu = 0.3, Delta = 5 (the (Delta + nu) form would give 0.00320755)."""
    assert eit_steady_state_nbar(math.sqrt(4.0 * 0.3 * 5.0), 0.3, 5.0, 1.0) == pytest.approx(
        0.00361702, abs=1e-8
    )


def test_light_shift_tuning_rule_bandwidth_and_poles() -> None:
    """Roos 2000: Delta = 70 MHz and Omega_sigma = 21.4 MHz put the narrow dressed state 1.6 MHz up (1.599); the inverse
    rule returns 21.4 for nu = 1.6; the poles Delta = 0 and Omega_r = 2 nu raise; Delta < 0 raises; the dressed linewidth
    is about Gamma nu/Delta."""
    assert light_shift_rad_s(70.0, 21.4) == pytest.approx(1.599, abs=1e-3)
    assert coupling_for_target_rad_s(1.6, 70.0) == pytest.approx(21.41, abs=0.01)
    assert light_shift_rad_s(70.0, coupling_for_target_rad_s(1.6, 70.0)) == pytest.approx(1.6, rel=1e-12)
    with pytest.raises(CoolingError):
        eit_steady_state_nbar(2.0 * 0.3, 0.3, 5.0, 1.0)
    with pytest.raises(CoolingError):
        coupling_for_target_rad_s(1.6, -70.0)
    with pytest.raises(CoolingError):
        eit_steady_state_nbar(10.0, 0.3, 0.0, 1.0)
    bw = dressed_linewidth_rad_s(GAMMA, DELTA, OM_R)
    assert bw == pytest.approx(GAMMA * NU / DELTA, rel=0.1)


def test_lechner_2016_rate_ratio_favours_the_mode_nearer_the_bright_resonance() -> None:
    """Omega_sigma = 30 MHz, Omega_pi = 6.2 MHz and a light shift of 2.2-2.3 MHz (Delta = 96-100 MHz):
    W = eta^2 (A_- - A_+) with eta^2 proportional to 1/nu gives R(3.29 MHz)/R(1.13 MHz) = 2.7-3.8 (3.2 at 2.25 MHz)
    against the measured 17/5 = 3.4."""
    gamma, om_s, om_p = 21.57, 30.0, 6.2
    ratios = []
    for shift in (2.2, 2.25, 2.3):
        delta = brentq(lambda d, target=shift: light_shift_rad_s(d, om_s) - target, 10.0, 1000.0)
        # Omega_pi/gamma = 0.29: Lechner's own probe is not weak against gamma, so the escape is explicit
        r_hi = eit_rate_coefficients(
            om_p, om_s, 3.29, delta, gamma, allow_saturation=True
        ).cooling_rate_per_s(1.0 / math.sqrt(3.29))
        r_lo = eit_rate_coefficients(
            om_p, om_s, 1.13, delta, gamma, allow_saturation=True
        ).cooling_rate_per_s(1.0 / math.sqrt(1.13))
        ratios.append(r_hi / r_lo)
    assert ratios[1] == pytest.approx(3.19, abs=0.02)
    assert 2.6 < min(ratios) and max(ratios) < 3.9
    assert all(r > 1.0 for r in ratios)


def test_roos_operating_point_closed_form_is_far_below_his_measured_occupation() -> None:
    """Roos's 40Ca+ operating point (Delta = 70 MHz, Omega_sigma = 21.4 MHz, the narrow dressed state delta = 1.6 MHz up,
    on his 1.6 MHz radial mode) gives the floor (gamma/4 Delta)^2 = 0.00593 at gamma = 2 pi x 21.57 MHz; his measured
    nbar_y = 0.18 is 30 times that, an anchor that reports rather than a target of the closed form."""
    gamma_ca, delta_roos, om_sigma = 21.57, 70.0, 21.4
    nu = light_shift_rad_s(delta_roos, om_sigma)
    assert nu == pytest.approx(1.6, abs=2e-3)
    # the exact steady state at that tuning agrees with (gamma/4 Delta)^2 to the tuning's own precision
    exact = eit_steady_state_nbar(om_sigma, nu, delta_roos, gamma_ca)
    assert exact == pytest.approx((gamma_ca / (4.0 * delta_roos)) ** 2, rel=2e-3)
    assert exact == pytest.approx(0.00593, rel=3e-3)
    assert 0.18 / exact == pytest.approx(30.4, abs=0.5)


def test_lechner_eighteen_ion_radial_band_is_wider_than_the_dressed_cooling_bandwidth() -> None:
    """Lechner et al. cool an 18-ion 40Ca+ string, quoting 0.01-0.02 on the radial modes in under 1 ms at
    Omega_sigma = 2 pi x 30 MHz, Omega_pi = 2 pi x 6.2 MHz and a light shift of 2.2-2.3 MHz.

    The 18-ion radial mode set (the band edges 3.29 and 1.13 MHz fix the anisotropy through the transverse eigenvalues
    gamma_p = 1/alpha + 1/2 - mu_p/2, Marquet 2003) with W_m = (|Delta k| x0,m)^2 (A_- - A_+) for a global beam: 17 of
    the 18 modes relax in under 1 ms, but only the 7 within the dressed linewidth Gamma' = 0.485 MHz of the light shift
    reach nbar <= 0.02, the 2.16 MHz band being 4.45 times that bandwidth.
    """
    n_ions = 18
    mu, _b = axial_modes_dimensionless(equilibrium_dimensionless(n_ions))
    mu_max = float(mu[-1])
    assert mu_max == pytest.approx(118.533, abs=1e-3)
    radial_com, radial_low = 3.29, 1.13
    omega_z = math.sqrt(2.0 * (radial_com**2 - radial_low**2) / (mu_max - 1.0))
    alpha = (omega_z / radial_com) ** 2
    assert alpha < 2.0 / (mu_max - 1.0)  # the string is linear, not a zigzag
    radial = np.sort(np.sqrt(1.0 / alpha + 0.5 - np.asarray(mu, dtype=float) / 2.0) * omega_z)
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
        eta = 1.3 * k_397 * x0  # the two-photon |Delta k| = 1.3 k
        w = 1e6 * eit_rate_coefficients(
            om_pi, om_sigma, float(f_mhz), delta, gamma_ca, allow_saturation=True
        ).cooling_rate_per_s(eta)
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
    cf = EitClosedForm(NU, DELTA, OM1, OM_R, GAMMA, 0.02)
    assert not cf.weak_probe()  # Omega_g = 17 against Omega_r = 24: outside the closed form's regime
    assert cf.nbar == pytest.approx(0.005102, abs=1e-6)
    assert cf.light_shift_rad_s == pytest.approx(NU, rel=2e-3)
    assert cf.cooling_rate_per_s == pytest.approx(0.02**2 * cf.coefficients.cooling_rate_bare_per_s)
    weak = EitClosedForm(NU, DELTA, 1.0, OM_R, GAMMA, 0.02)
    assert weak.weak_probe() and weak.nbar == pytest.approx(cf.nbar)


def test_the_perturbative_coefficients_refuse_a_saturated_probe() -> None:
    for bad in (0.11, 0.5, 1.0):
        with pytest.raises(ValidityError, match="Omega/Gamma"):
            eit_rate_coefficients(bad, 24.0, 2.0, 70.0, 1.0)
    assert eit_rate_coefficients(17.0, 24.0416, 2.0068, 70.0, 20.0, allow_saturation=True).A_minus_per_s > 0.0


# ---- level C: the Lambda system with one mode (Morigi, Eschner and Keitel 2000) --------------------------------------

G = gamma_rad_s()


def _lambda_eit(
    om_g: float, om_r: float, nu: float, delta: float, eta_two_photon: float
) -> tuple[BlochModel, float]:
    stl = structure(lambda_atom())
    b_g = circular_beam(stl, LAMBDA_GROUND_MINUS, LAMBDA_EXCITED, om_g, delta, q=1, k_sign=1)
    b_r = circular_beam(stl, LAMBDA_GROUND_PLUS, LAMBDA_EXCITED, om_r, delta, q=-1, k_sign=-1)
    x0 = eta_two_photon / (2.0 * TWO_PI / WAVELENGTH_M)
    mass = HBAR_J_S / (2.0 * nu * x0**2)
    mode = ModeSpec(nu, mass, (0.0, 0.0, 1.0), d=30, expected_n_max=8)
    model = BlochModel(
        stl,
        [b_g, b_r],
        states=(LAMBDA_GROUND_MINUS, LAMBDA_GROUND_PLUS, LAMBDA_EXCITED),
        mode=mode,
        options=MultiLevelOptions(leak="renormalize", recoil="minimal"),
    )
    eta = mode.eta(b_g.k_vector()) - mode.eta(b_r.k_vector())
    return model, eta


@pytest.mark.slow
def test_morigi_2000_fig_3_ground_state_occupation() -> None:
    """Omega_r = gamma, Omega_g = gamma/20, nu = gamma/10, eta = 0.145, Delta = 2.5 gamma (the paper's caption): 99 %
    ground-state occupation; level C sits within 6 % of the rate-equation nbar_S = 0.01083 (Lamb-Dicke corrections at
    eta = 0.145)."""
    om_g, om_r, nu, delta = 0.05 * G, 1.0 * G, 0.1 * G, 2.5 * G
    nbar_s = eit_rate_coefficients(om_g, om_r, nu, delta, G).nbar
    assert nbar_s == pytest.approx(0.010833, rel=1e-3)
    model, eta = _lambda_eit(om_g, om_r, nu, delta, 0.145)
    assert eta == pytest.approx(0.145, rel=1e-6)  # the beam is 2.5 gamma off the nominal wavelength
    lc = level_c_steady_state(model.build)
    assert lc.fock_populations[0] > 0.985
    assert lc.nbar == pytest.approx(nbar_s, rel=0.06)
    assert lc.boundary_population < 1e-8


@pytest.mark.slow
def test_eit_closed_form_is_the_weak_probe_limit_of_level_c() -> None:
    """At the delta = nu tuning Omega_r^2 = 4 nu (nu + Delta) the closed form gives (gamma/4 Delta)^2 exactly and level C
    approaches it as the probe weakens, the residual being the O(nu/Delta) the closed form drops."""
    nu, delta = 0.1 * G, 3.5 * G
    om_r = math.sqrt(4.0 * nu * (nu + delta))
    nbar_s = eit_rate_coefficients(0.01 * G, om_r, nu, delta, G).nbar
    assert nbar_s == pytest.approx((G / (4.0 * delta)) ** 2, rel=1e-9)
    previous = math.inf
    for om_g in (0.1 * G, 0.03 * G):
        model, _eta = _lambda_eit(om_g, om_r, nu, delta, 0.13)
        nbar = level_c_steady_state(model.build).nbar
        assert nbar < previous
        previous = nbar
    assert previous == pytest.approx(nbar_s, rel=0.06)
    # a blue-detuned but mis-tuned coupling (Omega_r = 2 nu) is the pole where cooling vanishes
    with pytest.raises(CoolingError):
        _ = eit_rate_coefficients(0.01 * G, 2.0 * nu, nu, delta, G).nbar
