"""Level C (internal levels plus one mode) against the level-A/B closed forms in their stated regimes (PLAN.md Sections
4.2.1, 4.2.2, 4.2.8, 9.3; M3a).

Every closed form here is evaluated on the closed two-level fixture (J = 0 -> J' = 1 with sigma+ light along B), for which
Stenholm's A_+- = P(Delta -+ nu) + alpha P(Delta), the RMP/Roos sideband floor (Gamma/2 nu)^2 [alpha/cos^2 theta_L + 1/4]
and Itano's Doppler limit hold exactly; the recoil kernel enters through the three discretizations of Section 4.2.8, the
spectrum path A_+- = 2 Re[S(-+ nu) + D] through the internal Liouvillian, and level B through ``mesolve`` on the two
phonon collapse operators.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt
from scipy.optimize import minimize_scalar

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions, decay_sum_rule_residual
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import (
    WEAK_DRIVE_MAX,
    BlochModel,
    CoolingError,
    carrier_weight,
    rate_coefficients,
    rate_coefficients_from_model,
    rate_coefficients_from_spectrum,
)
from qutip_trap.light.recoil import angular_factor
from qutip_trap.prep.level_c import (
    level_c_relaxation_rate,
    level_c_steady_state,
    phonon_generator,
    phonon_mean_closed_form,
    phonon_rate_equation_mesolve,
    phonon_steady_state,
)
from qutip_trap.units import C_M_PER_S, TWO_PI
from tests.bloch_fixtures import (
    MASS_KG,
    TWO_LEVEL_EXCITED,
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    WAVELENGTH_M,
    gamma_rad_s,
    sigma_plus_beam,
    structure,
    two_level_atom,
)

Z = (0.0, 0.0, 1.0)
OBLIQUE = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
ST = structure(two_level_atom())
G = gamma_rad_s()
NU = 50.0 * G
"""Gamma/nu = 0.02: the resolved-sideband regime of the closed forms."""


def sideband_setup(axis: tuple[float, float, float], omega: float) -> tuple[Beam, ModeSpec, float, float]:
    mode = ModeSpec(NU, MASS_KG, axis, d=14, expected_n_max=3)
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega, -NU)
    cos_chi = float(np.dot(axis, Z))
    alpha = angular_factor(1, cos_chi)
    cw = carrier_weight(alpha, beam.k_rad_per_m, float(np.dot(beam.k_vector(), axis)))
    return beam, mode, alpha, cw


# ---- sideband floor (Section 9.3 "Sideband floor decomposition", "Lamb-Dicke rate framework") ---------------------------


@pytest.mark.parametrize("axis, expected_weight", [(Z, 0.4), (OBLIQUE, 0.7)])
@pytest.mark.parametrize("recoil", ["minimal", "marginal", "vector"])
def test_level_c_sideband_floor_is_gamma_over_2nu_squared_times_alpha_over_cos2_plus_quarter(
    axis: tuple[float, float, float], expected_weight: float, recoil: str
) -> None:
    """nbar_SB = (Gamma/2 nu)^2 [alpha (k_em/k_L)^2 / cos^2 theta_L + 1/4]: Roos Eq. 3.20 with (eta~/eta)^2 = alpha/cos^2 theta_L
    for one beam on the emitting line (alpha = 2/5 along B; 0.35/0.5 = 0.7 at 45 degrees). All three recoil discretizations
    agree, since the floor needs only the second moment (Section 4.2.8)."""
    beam, mode, _alpha, cw = sideband_setup(axis, 0.5 * G)
    assert cw == pytest.approx(expected_weight, abs=1e-12)
    build = BlochModel(ST, [beam], mode=mode, options=MultiLevelOptions(recoil=recoil)).build  # type: ignore[arg-type]
    assert decay_sum_rule_residual(build) < 1e-12
    lc = level_c_steady_state(build)
    expected = (G / (2.0 * NU)) ** 2 * (cw + 0.25)
    assert lc.nbar == pytest.approx(expected, rel=1e-4)
    assert lc.boundary_population < 1e-12


def test_alpha_to_zero_leaves_the_blue_sideband_floor_gamma_over_4nu_squared_not_zero() -> None:
    """Recoil off: nbar = (Gamma/4 nu)^2 from off-resonant blue-sideband excitation (Section 9.3), 6.25e-6 at Gamma/nu = 0.01
    scaled here to Gamma/nu = 0.02."""
    beam, mode, _alpha, _cw = sideband_setup(Z, 0.5 * G)
    lc = level_c_steady_state(BlochModel(ST, [beam], mode=mode).build)
    assert lc.nbar == pytest.approx((G / (4.0 * NU)) ** 2, rel=1e-4)
    assert lc.nbar > 0.0


def test_halving_omega_leaves_the_floor_and_doubling_eta_em_quadruples_the_recoil_part() -> None:
    """Section 9.3: 'halving Omega leaves nbar unchanged; doubling eta_em quadruples the alpha part'."""
    beam, mode, _alpha, cw = sideband_setup(Z, 0.5 * G)
    half = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.25 * G, -NU)
    n_full = level_c_steady_state(
        BlochModel(ST, [beam], mode=mode, options=MultiLevelOptions(recoil="minimal")).build
    ).nbar
    n_half = level_c_steady_state(
        BlochModel(ST, [half], mode=mode, options=MultiLevelOptions(recoil="minimal")).build
    ).nbar
    assert n_half == pytest.approx(n_full, rel=2e-3)
    # the recoil part scales as eta_em^2 = (k_em x0)^2: a four times lighter "ion" doubles eta_em
    light = ModeSpec(NU, MASS_KG / 4.0, Z, d=14, expected_n_max=3)
    n_light = level_c_steady_state(
        BlochModel(ST, [beam], mode=light, options=MultiLevelOptions(recoil="minimal")).build
    ).nbar
    base = (G / (2.0 * NU)) ** 2
    # the drive's eta doubles too, so (eta~/eta)^2 is unchanged: the floor is eta-independent (Section 4.2.8 iv)
    assert n_light == pytest.approx(base * (cw + 0.25), rel=1e-3)


# ---- the spectrum path and the relaxation rate (Section 4.2.8 iii; Section 9.3 "Phonon rate equation") -------------------


@pytest.mark.parametrize("axis", [Z, OBLIQUE])
def test_spectrum_path_reproduces_level_c_and_pins_qutips_sign_convention(
    axis: tuple[float, float, float],
) -> None:
    """A_+ = S(+nu) + 2D and A_- = S(-nu) + 2D with QuTiP's spectrum(omega) = W(Delta - omega); nbar and the relaxation rate
    agree with the full solve to 1e-4 at Omega = Gamma/2, where the W(Delta -+ nu) closed form with the saturated W is off by 1 + s."""
    beam, mode, alpha, cw = sideband_setup(axis, 0.5 * G)
    model = BlochModel(ST, [beam])
    sc = rate_coefficients_from_spectrum(model, mode)
    eta = mode.eta(beam.k_vector())
    w = model.w_of_offset(0)
    # the heating spectrum is the far-off-resonant W(Delta - nu), unsaturated, so it agrees with eta^2 W(Delta - nu)
    assert sc.spectrum_plus_nu_per_s == pytest.approx(eta**2 * w(-NU), rel=1e-3)
    assert sc.two_D_per_s == pytest.approx(alpha * (beam.k_rad_per_m * mode.x0_m) ** 2 * w(0.0), rel=1e-5)
    lc = level_c_steady_state(
        BlochModel(ST, [beam], mode=mode, options=MultiLevelOptions(recoil="minimal")).build
    )
    assert sc.nbar == pytest.approx(lc.nbar, rel=2e-4)
    w_c = level_c_relaxation_rate(
        BlochModel(ST, [beam], mode=mode, options=MultiLevelOptions(recoil="minimal")).build
    )
    assert w_c == pytest.approx(sc.cooling_rate_per_s, rel=5e-4)
    saturated = rate_coefficients(w, NU, cw)
    s = 2.0 * (0.5 * G) ** 2 / G**2
    assert saturated.nbar / sc.nbar == pytest.approx(1.0 + s, rel=2e-3)
    with pytest.raises(ValueError):
        rate_coefficients_from_model(model, 0, NU, cw)
    assert model.drive_saturation() == pytest.approx(0.5, rel=1e-6) and WEAK_DRIVE_MAX == 0.1


def test_weak_drive_closed_form_agrees_with_level_c_to_the_saturation_order() -> None:
    """Omega = Gamma/20 (s = 0.005): the W(Delta -+ nu) form agrees with the spectrum path to 0.5 % (Section 4.2.8 vii)."""
    beam_w = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, -NU)
    mode = ModeSpec(NU, MASS_KG, Z, d=14, expected_n_max=3)
    model = BlochModel(ST, [beam_w])
    cw = carrier_weight(0.4, beam_w.k_rad_per_m, beam_w.k_rad_per_m)
    rc = rate_coefficients_from_model(model, 0, NU, cw)
    sc = rate_coefficients_from_spectrum(model, mode)
    assert rc.nbar == pytest.approx(sc.nbar, rel=6e-3)
    assert rc.nbar == pytest.approx((G / (2.0 * NU)) ** 2 * (0.4 + 0.25), rel=6e-3)
    eta = mode.eta(beam_w.k_vector())
    assert rc.cooling_rate_per_s(eta) == pytest.approx(sc.cooling_rate_per_s, rel=6e-3)


def test_mixed_pi_and_sigma_channels_need_one_alpha_per_channel() -> None:
    """A beam at 45 degrees with pi and sigma components populates |e, 0> and |e, +-1>; the per-channel D of Section 4.2.8 (ii)
    reproduces level C while any single mean alpha does not (the 'one mean alpha is up to 37 % wrong' statement)."""
    k_hat = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
    pol = (1.0 / math.sqrt(2.0) + 0j, 0j, -1.0 / math.sqrt(2.0) + 0j)
    probe = Beam(WAVELENGTH_M, k_hat, pol, 20e-6, 1e-3, (0.0, 0.0, 0.0))
    om_pi = abs(
        ST.single_photon_coupling_rad_s(ST.state(TWO_LEVEL_GROUND), ST.state(TWO_LEVEL_EXCITED), probe)
    )
    power = 1e-3 * (0.5 * G / om_pi) ** 2
    omega_l = TWO_PI * (ST.state(TWO_LEVEL_EXCITED).energy_hz - ST.state(TWO_LEVEL_GROUND).energy_hz) - NU
    beam45 = Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, pol, 20e-6, power, (0.0, 0.0, 0.0))
    mode = ModeSpec(NU, MASS_KG, Z, d=14, expected_n_max=3)
    model = BlochModel(ST, [beam45])
    pops = model.steadystate().populations
    assert pops["P2/2 mJ=0"] > 0.0 and pops["P2/2 mJ=1"] > 0.0 and pops["P2/2 mJ=-1"] > 0.0
    sc = rate_coefficients_from_spectrum(model, mode)
    for recoil in ("minimal", "vector"):
        lc = level_c_steady_state(
            BlochModel(ST, [beam45], mode=mode, options=MultiLevelOptions(recoil=recoil)).build
        )
        assert lc.nbar == pytest.approx(sc.nbar, rel=1e-4)
    # a single mean alpha in the same steady state: 2D_single = alpha_mean eta_em^2 Gamma P_e
    rho = model.steadystate().rho
    rates = model.photon_rates(rho)
    total = sum(rates.values())
    eta_em = (TWO_PI / WAVELENGTH_M) * mode.x0_m
    for alpha_single in (0.2, 0.4, 1.0 / 3.0):
        two_d_single = alpha_single * eta_em**2 * total
        assert abs(two_d_single / sc.two_D_per_s - 1.0) > 0.1


def test_q_coherence_terms_of_the_vector_form_matter_for_a_mode_perpendicular_to_b() -> None:
    """With the excited sublevels driven coherently and the mode perpendicular to B, the vector form (Steck's f_qq' tensor)
    and the per-q scalar patterns differ; along B the azimuthal symmetry kills the cross terms and they coincide."""
    k_hat = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
    pol = (1.0 / math.sqrt(2.0) + 0j, 0j, -1.0 / math.sqrt(2.0) + 0j)
    probe = Beam(WAVELENGTH_M, k_hat, pol, 20e-6, 1e-3, (0.0, 0.0, 0.0))
    om_pi = abs(
        ST.single_photon_coupling_rad_s(ST.state(TWO_LEVEL_GROUND), ST.state(TWO_LEVEL_EXCITED), probe)
    )
    power = 1e-3 * (0.5 * G / om_pi) ** 2
    omega_l = TWO_PI * (ST.state(TWO_LEVEL_EXCITED).energy_hz - ST.state(TWO_LEVEL_GROUND).energy_hz) - NU
    beam45 = Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, pol, 20e-6, power, (0.0, 0.0, 0.0))
    results = {}
    for axis in (Z, (1.0, 0.0, 0.0)):
        mode = ModeSpec(NU, MASS_KG, axis, d=14, expected_n_max=3)
        results[axis] = tuple(
            level_c_steady_state(
                BlochModel(ST, [beam45], mode=mode, options=MultiLevelOptions(recoil=r)).build
            ).nbar
            for r in ("minimal", "vector")
        )
    n_min_z, n_vec_z = results[Z]
    n_min_x, n_vec_x = results[(1.0, 0.0, 0.0)]
    assert n_vec_z == pytest.approx(n_min_z, rel=1e-4)
    assert abs(n_vec_x / n_min_x - 1.0) > 1e-3


# ---- Doppler regime (Section 9.3 "Doppler limit with recoil") -------------------------------------------------------------


def test_doppler_limit_with_recoil_at_the_minimum_detuning() -> None:
    """nbar_D = (Gamma/4 nu)(1 + alpha) at Delta = -Gamma/2 up to the 1/2 zero-point offset (Itano-Wineland E_K = 7 hbar Gamma/40 for
    alpha = 2/5); level A and level C agree to 0.3 % at nu = Gamma/20 and the argmin over Delta is -Gamma/2."""
    nu_d = 0.05 * G
    alpha = angular_factor(1, 1.0)
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, -0.5 * G)
    cw = carrier_weight(alpha, beam.k_rad_per_m, beam.k_rad_per_m)
    rc = rate_coefficients_from_model(BlochModel(ST, [beam]), 0, nu_d, cw)
    closed = (G / (4.0 * nu_d)) * (1.0 + alpha)
    assert rc.nbar == pytest.approx(closed - 0.5, rel=8e-3)
    assert closed == pytest.approx(7.0)

    # the argmin is -(Gamma/2) sqrt(1 + s), i.e. -0.50125 Gamma at s = 2 x 0.05^2 = 0.005 (RMP Eq. 106); a 21-point
    # grid of spacing 0.025 Gamma cannot resolve that, so it is found with a bounded minimizer (the M3 finding)
    def nbar_at(detuning_over_gamma: float) -> float:
        beam_at = sigma_plus_beam(
            ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, detuning_over_gamma * G
        )
        return rate_coefficients_from_model(BlochModel(ST, [beam_at]), 0, nu_d, cw).nbar

    s = 2.0 * 0.05**2
    target = -0.5 * math.sqrt(1.0 + s)
    assert target == pytest.approx(-0.501248, abs=1e-6)
    sol = minimize_scalar(nbar_at, bounds=(-0.8, -0.3), method="bounded", options={"xatol": 1e-7})
    assert float(sol.x) == pytest.approx(-0.504442, abs=1e-5)  # the measured argmin at nu/Gamma = 0.05
    # -(Gamma/2) sqrt(1 + s) is the nu/Gamma -> 0 limit; the residual is first order in nu/Gamma
    residuals = {}
    for ratio in (0.05, 0.01, 0.002):
        nu_r = ratio * G

        def nbar_r(detuning_over_gamma: float, nu_r: float = nu_r) -> float:
            beam_r = sigma_plus_beam(
                ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, detuning_over_gamma * G
            )
            return rate_coefficients_from_model(BlochModel(ST, [beam_r]), 0, nu_r, cw).nbar

        best = minimize_scalar(nbar_r, bounds=(-0.8, -0.3), method="bounded", options={"xatol": 1e-7})
        residuals[ratio] = float(best.x) - target
    assert residuals[0.05] == pytest.approx(-3.19e-3, rel=0.02)
    assert residuals[0.01] == pytest.approx(-1.28e-4, rel=0.02)
    # a round-off-limited residual: -5.24e-6 on the Linux runner against -4.37e-6 here; the sign and the order are the point
    assert residuals[0.002] == pytest.approx(-4.37e-6, rel=0.35)
    assert abs(residuals[0.002]) < 1e-5  # the plan's argmin -0.5000 is the limit, reached from below
    # the 21-point grid of the earlier revision lands on its nearest node and cannot see any of that
    grid = np.linspace(-0.8 * G, -0.3 * G, 21)
    nbars = [nbar_at(float(d) / G) for d in grid]
    assert grid[int(np.argmin(nbars))] / G == pytest.approx(-0.5, abs=1e-9)
    assert abs(grid[int(np.argmin(nbars))] / G - float(sol.x)) > 1e-3


@pytest.mark.slow
@pytest.mark.heavy
def test_doppler_limit_level_c_matches_level_a() -> None:
    """The d = 70 four-level model's Liouvillian is 78400-square with 8 million non-zeros; its direct factorization peaks at
    7 GB of resident memory (measured 2026-09-19), so the test carries the ``heavy`` marker and CI runs it alone."""
    nu_d = 0.05 * G
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, -0.5 * G)
    cw = carrier_weight(0.4, beam.k_rad_per_m, beam.k_rad_per_m)
    rc = rate_coefficients_from_model(BlochModel(ST, [beam]), 0, nu_d, cw)
    mode = ModeSpec(nu_d, MASS_KG, Z, d=70, expected_n_max=40)
    lc = level_c_steady_state(
        BlochModel(ST, [beam], mode=mode, options=MultiLevelOptions(recoil="minimal")).build
    )
    assert lc.nbar == pytest.approx(rc.nbar, rel=3e-3)
    assert lc.boundary_population < 1e-4


# ---- EIT (Section 4.2.3; Morigi, Eschner and Keitel 2000) --------------------------------------------------------------


def _lambda_eit(
    om_g: float, om_r: float, nu: float, delta: float, eta_two_photon: float, recoil: str
) -> tuple[BlochModel, float]:
    from qutip_trap.units import HBAR_J_S
    from tests.bloch_fixtures import (
        LAMBDA_EXCITED,
        LAMBDA_GROUND_MINUS,
        LAMBDA_GROUND_PLUS,
        circular_beam,
        lambda_atom,
    )

    stl = structure(lambda_atom())
    b_g = circular_beam(stl, LAMBDA_GROUND_MINUS, LAMBDA_EXCITED, om_g, delta, q=1, k_sign=1)
    b_r = circular_beam(stl, LAMBDA_GROUND_PLUS, LAMBDA_EXCITED, om_r, delta, q=-1, k_sign=-1)
    x0 = eta_two_photon / (2.0 * TWO_PI / WAVELENGTH_M)
    mass = HBAR_J_S / (2.0 * nu * x0**2)
    mode = ModeSpec(nu, mass, Z, d=30, expected_n_max=8)
    model = BlochModel(
        stl,
        [b_g, b_r],
        states=(LAMBDA_GROUND_MINUS, LAMBDA_GROUND_PLUS, LAMBDA_EXCITED),
        mode=mode,
        options=MultiLevelOptions(leak="renormalize", recoil=recoil),  # type: ignore[arg-type]
    )
    eta = mode.eta(b_g.k_vector()) - mode.eta(b_r.k_vector())
    return model, eta


def eit_closed_form(
    om_g: float, om_r: float, nu: float, delta: float, gamma: float
) -> tuple[float, float, float]:
    """Section 4.2.3: A_+- = (Omega_g^2/gamma) gamma^2 nu^2 / {gamma^2 nu^2 + 4 [Omega_r^2/4 - nu (nu -+ Delta)]^2}."""

    def a(sign: float) -> float:
        return (
            (om_g**2 / gamma)
            * gamma**2
            * nu**2
            / (gamma**2 * nu**2 + 4.0 * (om_r**2 / 4.0 - nu * (nu - sign * delta)) ** 2)
        )

    a_plus, a_minus = a(+1.0), a(-1.0)
    return phonon_steady_state(a_plus, a_minus), a_plus, a_minus


@pytest.mark.slow
def test_morigi_2000_fig_3_ground_state_occupation() -> None:
    """Omega_r = gamma, Omega_g = gamma/20, nu = gamma/10, eta = 0.145, Delta = 2.5 gamma (the paper's caption, not the plan's
    fixture): 99 % ground-state occupation; level C sits within 5 % of the rate-equation nbar_S = 0.01083 (Lamb-Dicke
    corrections at eta = 0.145) and the closed form's (gamma/4 Delta)^2 = 0.01 is the delta = nu optimum it approaches."""
    om_g, om_r, nu, delta = 0.05 * G, 1.0 * G, 0.1 * G, 2.5 * G
    nbar_s, a_plus, a_minus = eit_closed_form(om_g, om_r, nu, delta, G)
    assert nbar_s == pytest.approx(0.010833, rel=1e-3)
    model, eta = _lambda_eit(om_g, om_r, nu, delta, 0.145, "minimal")
    assert eta == pytest.approx(0.145, rel=1e-6)  # the beam is 2.5 gamma off the nominal wavelength
    lc = level_c_steady_state(model.build)
    assert lc.fock_populations[0] > 0.985
    assert lc.nbar == pytest.approx(nbar_s, rel=0.06)
    assert lc.boundary_population < 1e-8


@pytest.mark.slow
def test_eit_closed_form_is_the_weak_probe_limit_of_level_c() -> None:
    """At the delta = nu tuning Omega_r^2 = 4 nu (nu + Delta) the closed form gives (gamma/4 Delta)^2 exactly and level C
    approaches it as the probe weakens, the residual being the O(nu/Delta) the closed form drops (Section 4.2.3)."""
    nu, delta = 0.1 * G, 3.5 * G
    om_r = math.sqrt(4.0 * nu * (nu + delta))
    nbar_s, _, _ = eit_closed_form(0.01 * G, om_r, nu, delta, G)
    assert nbar_s == pytest.approx((G / (4.0 * delta)) ** 2, rel=1e-9)
    previous = math.inf
    for om_g in (0.1 * G, 0.03 * G):
        model, _eta = _lambda_eit(om_g, om_r, nu, delta, 0.13, "minimal")
        nbar = level_c_steady_state(model.build).nbar
        assert nbar < previous
        previous = nbar
    assert previous == pytest.approx(nbar_s, rel=0.06)
    # a blue-detuned but mis-tuned coupling (Omega_r = 2 nu) is the pole where cooling vanishes
    with pytest.raises(CoolingError):
        eit_closed_form(0.01 * G, 2.0 * nu, nu, delta, G)


# ---- level B (Section 9.3 "Phonon rate equation") -------------------------------------------------------------------------


def test_phonon_rate_equation_mesolve_matches_the_closed_form_to_1e_12() -> None:
    """Section 9.3 "Phonon rate equation" pins 1e-12; the solver's own settings (atol 1e-13, rtol 1e-11, dop853) deliver
    max abs 3.73e-13 (max rel 3.95e-12) on this fixture."""
    a_plus, a_minus, eta = 3.0e4, 4.0e5, 0.1
    times = np.linspace(0.0, 3.0e-3, 31)
    mesolve = phonon_rate_equation_mesolve(a_plus, a_minus, eta, 5, 40, times)
    closed = phonon_mean_closed_form(a_plus, a_minus, eta, 5.0, times)
    assert np.max(np.abs(mesolve - closed)) < 1e-12
    assert phonon_steady_state(a_plus, a_minus) == pytest.approx(a_plus / (a_minus - a_plus))
    gen = phonon_generator(a_plus, a_minus, eta, 40)
    assert np.allclose(gen.sum(axis=0), 0.0, atol=1e-9)
    stationary = np.array([(a_plus / a_minus) ** n for n in range(40)])
    stationary /= stationary.sum()
    assert np.allclose(gen @ stationary, 0.0, atol=1e-9 * np.max(np.abs(gen)))


def test_the_birth_death_generator_conserves_probability_and_vanishes_without_coefficients() -> None:
    """Section 9.15: "Sum_n dP(n)/dt = 0 to machine precision for the canonical birth-death form, while Morigi's
    Eq. 29 as printed gives +5.56 ... and must be rejected"; Section 9.13: "W_k = 0 gives dP/dt = 0, no steady state".

    The canonical form's column sums vanish to machine precision. Morigi's printed Eq. 29 carries a plus for a minus
    (Section 4.2.8: "the arXiv v1 Eq. 29 has a plus for a minus that breaks probability conservation"), and PLAN.md
    does not say WHICH of the two loss terms is affected, so the printed +5.56 is not reproducible from the plan;
    what is testable, and is what the row is for, is that a sign flip on EITHER loss term breaks the conservation by
    an amount of order the cooling rate, so the printed form must be rejected rather than copied.
    """
    a_plus, a_minus, eta, d = 0.37, 1.0, 1.0, 60
    canonical = phonon_generator(a_plus, a_minus, eta, d)
    assert np.max(np.abs(canonical.sum(axis=0))) < 1e-14
    n = np.arange(d)
    for flipped in ("cooling", "heating"):
        wrong = np.zeros((d, d))
        for k in range(d):
            if k + 1 < d:
                wrong[k + 1, k] += (k + 1) * a_plus
                wrong[k, k] += (k + 1) * a_plus if flipped == "heating" else -(k + 1) * a_plus
            if k > 0:
                wrong[k - 1, k] += k * a_minus
                wrong[k, k] += k * a_minus if flipped == "cooling" else -k * a_minus
        stationary = np.array([(a_plus / a_minus) ** m for m in range(d)])
        stationary /= stationary.sum()
        leak = float(np.sum(wrong @ stationary))
        assert abs(leak) > 0.5  # of order the rate itself: the printed form is not a generator at all
        assert abs(float(np.sum(canonical @ stationary))) < 1e-14
    # A_+- = 0 gives the zero generator: no evolution and therefore no steady state to speak of
    assert np.all(phonon_generator(0.0, 0.0, eta, d) == 0.0)
    with pytest.raises(CoolingError):
        phonon_steady_state(0.0, 0.0)
    assert float(n @ stationary) == pytest.approx(a_plus / (a_minus - a_plus), rel=1e-9)


def test_the_printed_sideband_hamiltonian_is_not_hermitian_and_grows_the_norm() -> None:
    """Section 9.3 row "Sideband limit" / Section 9.16 row 4.2-3: RMP Eq. 109 is printed as i eta (sigma_+ a +
    sigma_- a^dagger), which is NOT Hermitian; the plan's corrected i eta (sigma_+ a - sigma_- a^dagger) is. The
    builder is Hermitian by construction (``multilevel``: h_static + c op + conj(c) op.dag()), so this is the
    negative control the plan's row asks for: the printed form has |H + H^dagger| > 0 and an evolution that leaves
    the norm, while the corrected one reproduces sin^2(eta Omega sqrt(n) t/2) at unit norm."""
    d, eta, omega = 8, 0.1, 1.0e6
    a = qt.tensor(qt.qeye(2), qt.destroy(d))
    sp = qt.tensor(qt.sigmap(), qt.qeye(d))
    sm = qt.tensor(qt.sigmam(), qt.qeye(d))
    printed = 0.5 * omega * eta * 1j * (sp * a + sm * a.dag())
    correct = 0.5 * omega * eta * 1j * (sp * a - sm * a.dag())
    assert (correct - correct.dag()).norm() < 1e-12
    assert (printed - printed.dag()).norm() > 1e-3 * omega * eta
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(d, 1))  # |down, n = 1>
    times = np.linspace(0.0, 4.0 * math.pi / (omega * eta), 41)
    opts = {"normalize_output": False, "progress_bar": "", "atol": 1e-12, "rtol": 1e-10}
    good = qt.sesolve(correct, psi0, times, options=opts)
    bad = qt.sesolve(printed, psi0, times, options=opts)
    up = qt.tensor(qt.basis(2, 0), qt.basis(d, 0))
    for t, state in zip(times, good.states):
        assert abs(state.norm() ** 2 - 1.0) < 1e-7
        expected = math.sin(0.5 * omega * eta * math.sqrt(1.0) * t) ** 2
        assert abs(abs(up.overlap(state)) ** 2 - expected) < 1e-6
    assert max(s.norm() ** 2 for s in bad.states) > 1.5


def test_heating_configuration_raises_instead_of_returning_a_negative_occupation() -> None:
    with pytest.raises(CoolingError):
        phonon_steady_state(2.0, 1.0)
    with pytest.raises(CoolingError):
        phonon_mean_closed_form(2.0, 1.0, 0.1, 1.0, np.array([0.0, 1.0]))
