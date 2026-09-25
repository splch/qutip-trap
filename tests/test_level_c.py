"""Level C (internal levels plus one mode) against the level-A spectrum path and the closed forms in their stated regimes.

Every closed form here is evaluated on the closed two-level fixture (J = 0 -> J' = 1 with sigma+ light along B), for which
Stenholm's A_+- = W(Delta -+ nu) + alpha W(Delta), the sideband floor (Gamma/2 nu)^2 [alpha/cos^2 theta_L + 1/4] and the
Doppler limit hold exactly; the recoil kernel enters through the three discretizations, and the spectrum path
A_+- = 2 Re[S(-+ nu) + D] through the internal Liouvillian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions, decay_sum_rule_residual
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import BlochModel, rate_coefficients_from_spectrum
from qutip_trap.light.recoil import angular_factor
from qutip_trap.prep.closed_forms import stenholm_coefficients
from qutip_trap.prep.level_c import level_c_relaxation_rate, level_c_steady_state
from qutip_trap.units import C_M_PER_S, TWO_PI
from tests.fixtures import (
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
    cw = alpha * (beam.k_rad_per_m / float(np.dot(beam.k_vector(), axis))) ** 2  # (eta~/eta)^2
    return beam, mode, alpha, cw


# ---- sideband floor ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("axis, expected_weight", [(Z, 0.4), (OBLIQUE, 0.7)])
@pytest.mark.parametrize("recoil", ["minimal", "marginal", "vector"])
def test_level_c_sideband_floor_is_gamma_over_2nu_squared_times_alpha_over_cos2_plus_quarter(
    axis: tuple[float, float, float], expected_weight: float, recoil: str
) -> None:
    """nbar_SB = (Gamma/2 nu)^2 [alpha (k_em/k_L)^2 / cos^2 theta_L + 1/4] (Roos Eq. 3.20) for one beam on the emitting
    line (alpha = 2/5 along B; 0.35/0.5 = 0.7 at 45 degrees); all three recoil discretizations agree, since the floor
    needs only the second moment."""
    beam, mode, _alpha, cw = sideband_setup(axis, 0.5 * G)
    assert cw == pytest.approx(expected_weight, abs=1e-12)
    build = BlochModel(ST, [beam], mode=mode, options=MultiLevelOptions(recoil=recoil)).build  # type: ignore[arg-type]
    assert decay_sum_rule_residual(build) < 1e-12
    lc = level_c_steady_state(build)
    expected = (G / (2.0 * NU)) ** 2 * (cw + 0.25)
    assert lc.nbar == pytest.approx(expected, rel=1e-4)
    assert lc.boundary_population < 1e-12


def test_alpha_to_zero_leaves_the_blue_sideband_floor_gamma_over_4nu_squared_not_zero() -> None:
    """Recoil off: nbar = (Gamma/4 nu)^2 from off-resonant blue-sideband excitation."""
    beam, mode, _alpha, _cw = sideband_setup(Z, 0.5 * G)
    lc = level_c_steady_state(BlochModel(ST, [beam], mode=mode).build)
    assert lc.nbar == pytest.approx((G / (4.0 * NU)) ** 2, rel=1e-4)
    assert lc.nbar > 0.0


def test_halving_omega_leaves_the_floor_and_doubling_eta_em_quadruples_the_recoil_part() -> None:
    """Halving Omega leaves nbar unchanged; the floor is eta independent."""
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
    # the drive's eta doubles too, so (eta~/eta)^2 is unchanged
    assert n_light == pytest.approx(base * (cw + 0.25), rel=1e-3)


# ---- the spectrum path and the relaxation rate -----------------------------------------------------------------------


@pytest.mark.parametrize("axis", [Z, OBLIQUE])
def test_spectrum_path_reproduces_level_c_and_pins_qutips_sign_convention(
    axis: tuple[float, float, float],
) -> None:
    """A_+ = S(+nu) + 2D and A_- = S(-nu) + 2D with QuTiP's spectrum(omega) = W(Delta - omega); nbar and the relaxation
    rate agree with the full solve to 1e-4 at Omega = Gamma/2, where the W(Delta -+ nu) form with the saturated W is off by
    1 + s."""
    beam, mode, alpha, cw = sideband_setup(axis, 0.5 * G)
    model = BlochModel(ST, [beam])
    sc = rate_coefficients_from_spectrum(model, mode)
    eta = mode.eta(beam.k_vector())

    def w(offset: float) -> float:
        return model.shifted(0, offset).scattering_rate_per_s()

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
    a_plus, a_minus = w(-NU) + cw * w(0.0), w(+NU) + cw * w(0.0)
    s = 2.0 * (0.5 * G) ** 2 / G**2
    assert a_plus / (a_minus - a_plus) / sc.nbar == pytest.approx(1.0 + s, rel=2e-3)


def test_weak_drive_spectrum_path_agrees_with_stenholm_to_the_saturation_order() -> None:
    """Omega = Gamma/20 (s = 0.005): Stenholm's weak-drive A_+- agree with the spectrum path to 0.6 %."""
    beam_w = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * G, -NU)
    mode = ModeSpec(NU, MASS_KG, Z, d=14, expected_n_max=3)
    rc = stenholm_coefficients(0.05 * G, G, NU, -NU, 0.4)
    sc = rate_coefficients_from_spectrum(BlochModel(ST, [beam_w]), mode)
    assert rc.nbar == pytest.approx(sc.nbar, rel=6e-3)
    assert sc.nbar == pytest.approx((G / (2.0 * NU)) ** 2 * (0.4 + 0.25), rel=6e-3)
    assert rc.cooling_rate_per_s(mode.eta(beam_w.k_vector())) == pytest.approx(
        sc.cooling_rate_per_s, rel=6e-3
    )


def test_mixed_pi_and_sigma_channels_need_one_alpha_per_channel() -> None:
    """A beam at 45 degrees with pi and sigma components populates |e, 0> and |e, +-1>; the per-channel D reproduces
    level C while any single mean alpha does not."""
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
