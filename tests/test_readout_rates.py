"""Section 9.5 targets of the atomic-rate layer (PLAN.md Sections 8.1, 8.8): the 171Yb+ rates from the Bloch solve against
the closed forms and the ceiling, Acton's factors and ceiling, the corrected I_sat, the efficiency chain, shelving and
the readout scheme's polarity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.light.bloch import CeilingViolation
from qutip_trap.readout.fluorescence import (
    ACTON_P12_FACTORS,
    FluorescenceRates,
    ReadoutScheme,
    ShelvingBranching,
    acton_angular_factors,
    acton_clock_state_ceiling,
    acton_leak_bright_to_dark,
    acton_leak_dark_to_bright,
    acton_mean_count,
    acton_optimal_light_level,
    camera_snr,
    crain_dark_pumping_form,
    detection_rates_for_ion,
    doppler_width_hz,
    emccd_effective_quantum_efficiency,
    fit_mean_count_curve,
    geometric_efficiency,
    mean_count_curve,
    micromotion_detection_rate,
    neighbour_intensity_ratio,
    rates_from_bloch,
    rates_from_detected,
    rms_velocity_m_per_s,
    saturation_ceiling,
    saturation_intensity_w_m2,
    scattering_rate,
    shelf_decay_error,
    system_efficiency,
    thermal_transfer_probability,
    yb171_bright_rate_closed,
    yb171_bright_rate_crain_form,
    yb171_bright_rate_noek_form,
    yb171_leakage_rates_closed,
    yb171_leakage_ratio,
)
from qutip_trap.species import species
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI
from tests.readout_fixtures import yb_detection_beam
from tests.test_bloch import BRIGHT, DARK, detection_model

YB = species("171Yb+")
GAMMA_S = YB.transition("S1/2-P1/2").partial_rate_rad_s
D_HFP = TWO_PI * 2.105e9
D_HFS = TWO_PI * 12_642_812_118.5


# ---- 171Yb+ rates from the Bloch solve (Section 9.5 "171Yb+ rates", "CPT") ----------------------------------------------------


def test_yb171_bloch_rates_sit_below_the_closed_form_ceiling_and_obey_gamma_over_4() -> None:
    """R_o from the exact four-level solve is within 0.6 % below (Gamma/18) s_o/[1 + (2/9) s_o] at s_o = 0.1, 1 G (the M3a
    finding: the closed form is the ceiling of the exact rate), the resonant manifold's ceiling is 1/4 and the excited
    population sits far below it."""
    fl = rates_from_bloch(detection_model(0.1, 1.0), BRIGHT, DARK, line="S1/2<-P1/2")
    closed = yb171_bright_rate_closed(0.1, GAMMA_S)
    assert 0.99 < fl.R_bright_per_s / closed < 1.0
    assert fl.ceiling == 0.25
    assert fl.excited_population is not None and fl.excited_population < 0.25
    assert fl.R_bright_per_s < GAMMA_S / 4.0
    assert "conv.scattering_rate_object" in fl.provenance


def test_yb171_leakage_prefactors_and_the_3_over_49_ratio() -> None:
    """R_d and R_b from the slow-manifold analysis reproduce Noek's (2/3)(1/3) and (2/3) prefactors with s = s_o/3 to 0.6 %
    and refute Crain's 1/3 (50 % high); R_b/R_d = 3/49 derived against 16.4/341 = 0.048 measured (30 % band)."""
    fl = rates_from_bloch(detection_model(0.1, 1.0), BRIGHT, DARK, line="S1/2<-P1/2")
    r_d, r_b = yb171_leakage_rates_closed(0.1, GAMMA_S, D_HFP, D_HFS)
    assert fl.R_dark_pumping_per_s == pytest.approx(r_d, rel=6e-3)
    assert fl.R_bright_pumping_per_s == pytest.approx(r_b, rel=6e-3)
    assert crain_dark_pumping_form(0.1, GAMMA_S, D_HFP) == pytest.approx(1.5 * r_d, rel=1e-12)
    ratio = fl.R_bright_pumping_per_s / fl.R_dark_pumping_per_s
    assert ratio == pytest.approx(yb171_leakage_ratio(D_HFP, D_HFS), rel=3e-3)
    assert yb171_leakage_ratio(D_HFP, D_HFS) == pytest.approx(3.0 / 49.0, rel=2e-3)
    assert abs(ratio - 16.4 / 341.0) / (16.4 / 341.0) < 0.30


def test_crain_operating_point_from_the_closed_form_and_efficiency() -> None:
    """Crain's measured 472 kcps at eps_sys = 4.356 % is R_o = 0.0880 Gamma, which the corrected form reaches at Noek's
    s = 0.815, that is s_o = 3 s = 2.45 (Section 8.8); the Section 9.5 "CPT" row prints "s_o = 0.815" for the same number,
    a slip between the two saturation parameters that the row itself warns against (M5 finding): at s_o = 0.815 the form
    gives 0.0383 Gamma."""
    r_o = yb171_bright_rate_closed(3.0 * 0.815, GAMMA_S)
    assert yb171_bright_rate_closed(0.815, GAMMA_S) / GAMMA_S == pytest.approx(0.0383, abs=2e-4)
    assert r_o / GAMMA_S == pytest.approx(0.0880, abs=2e-4)
    assert 0.04356 * r_o == pytest.approx(472e3, rel=0.01)


def test_the_two_saturation_parameters_are_the_same_rate() -> None:
    """Noek's (Gamma/6, 2/3) form at s = s_o/3 and the (Gamma/4, 1) form at s~ = (2/9) s_o equal the plan's form; the
    (Gamma/4, 1) form saturates at Gamma/4 with half maximum at s_o = 4.5."""
    for s_o in (0.01, 0.5, 2.45, 30.0):
        plan = yb171_bright_rate_closed(s_o, GAMMA_S, 0.3 * GAMMA_S)
        assert yb171_bright_rate_noek_form(s_o / 3.0, GAMMA_S, 0.3 * GAMMA_S) == pytest.approx(
            plan, rel=1e-12
        )
        assert yb171_bright_rate_crain_form(2.0 * s_o / 9.0, GAMMA_S, 0.3 * GAMMA_S) == pytest.approx(
            plan, rel=1e-12
        )
    assert yb171_bright_rate_closed(1e9, GAMMA_S) == pytest.approx(GAMMA_S / 4.0, rel=1e-6)
    assert yb171_bright_rate_closed(4.5, GAMMA_S) == pytest.approx(GAMMA_S / 8.0, rel=1e-12)


def test_scattering_rate_derives_the_bright_and_dark_manifolds_from_the_beams() -> None:
    """With no labels given the bright manifold is what the beam drives resonantly (the three F = 1 sublevels) and the dark
    manifold the remaining S1/2 sublevel (F = 0); the rates equal the explicit-label solve."""
    m = detection_model(0.1, 1.0)
    st = AtomicStructure(YB, 1.0, (0.0, 0.0, 1.0))
    fl, model = scattering_rate(st, m.beams, levels=("S1/2", "P1/2"))
    ground, excited = model.resonant_manifold()
    assert set(ground) == set(BRIGHT) and excited == ("P1/2 F=0 mF=0",)
    ref = rates_from_bloch(m, BRIGHT, DARK, line="S1/2<-P1/2")
    assert fl.R_bright_per_s == pytest.approx(ref.R_bright_per_s, rel=1e-9)
    assert fl.R_dark_pumping_per_s == pytest.approx(ref.R_dark_pumping_per_s, rel=1e-9)


def test_detection_rates_for_ion_builds_the_species_scheme_with_bright_is_1_polarity() -> None:
    beam = yb_detection_beam(0.5)
    fl, scheme, _model = detection_rates_for_ion(YB, 5.0, (1.0, 0.0, 0.0), [beam], levels=("S1/2", "P1/2"))
    assert (
        scheme.kind == "direct" and scheme.polarity == "bright_is_1" and scheme.classes == ("dark", "bright")
    )
    assert fl.R_bright_per_s > 0.0 and fl.shelf_decay_per_s == 0.0
    # the magic-angle beam along y at B = 5 G is destabilized: the rate is a sizeable fraction of the closed form
    assert fl.R_bright_per_s / yb171_bright_rate_closed(0.5, GAMMA_S) > 0.5


def test_ceiling_violation_is_raised_not_assumed() -> None:
    """A rate object cannot be built from a state whose resonant excited population exceeds n_e/(n_e + n_g)."""
    import qutip as qt

    m = detection_model(0.1, 1.0)
    hot = qt.basis(m.build.n_internal, m.build.index("P1/2 F=0 mF=0")).proj()
    with pytest.raises(CeilingViolation):
        m.ceiling_report(hot)
    assert saturation_ceiling(3, 1) == 0.25 and saturation_ceiling(2, 1) == pytest.approx(1.0 / 3.0)
    assert saturation_ceiling(1, 1) == 0.5


# ---- Acton 2006 closed forms (Section 9.5 rows "Clebsch-Gordan factors", "111Cd+ ceiling", "Corrected I_sat") ---------------------


def test_acton_clebsch_gordan_factors_for_i_one_half_to_three() -> None:
    expected_m1 = [2 / 9, 20 / 81, 1 / 4, 56 / 225, 20 / 81, 12 / 49]
    for i, m1 in zip((0.5, 1.0, 1.5, 2.0, 2.5, 3.0), expected_m1):
        f = acton_angular_factors(i)
        assert f.M1 == pytest.approx(m1, rel=1e-12)
    half = acton_angular_factors(0.5)
    assert (half.M1, half.M2pi, half.M2minus) == pytest.approx((2 / 9, 1 / 9, 1 / 9), rel=1e-12)
    assert ACTON_P12_FACTORS == (2 / 9, 2 / 9)


def test_cd111_clock_state_ceiling_and_the_p12_leak_ordering() -> None:
    """F_max = 1 - (4/9)(gamma/2 omega_HFP)^2 = 99.9375 % at gamma/2pi = 60 MHz, omega_HFP/2pi = 800 MHz; through P1/2 the
    bright -> dark leak always exceeds the dark -> bright one because Delta_1' > Delta_2'."""
    assert acton_clock_state_ceiling(TWO_PI * 60e6, TWO_PI * 800e6) == pytest.approx(0.999375, abs=1e-12)
    gamma = TWO_PI * 60e6
    a1 = acton_leak_dark_to_bright(2 / 9, 0.0, gamma, 0.0, TWO_PI * (14.5e9 + 2.0e9))
    a2 = acton_leak_dark_to_bright(2 / 9, 0.0, gamma, 0.0, TWO_PI * 2.0e9)
    assert a2 > a1
    f = acton_angular_factors(0.5)
    assert acton_leak_bright_to_dark(f, 0.0, 0.0, 0.25, gamma, 0.0, TWO_PI * 800e6) == 0.0
    assert acton_leak_bright_to_dark(f, 1e-3, 5e-4, 0.25, gamma, 0.0, TWO_PI * 800e6) > 0.0
    assert acton_mean_count(150e-6, 1.4e-3, 0.25, gamma) == pytest.approx(
        150e-6 * 1.4e-3 * 0.25 * gamma / 2 / 1.25
    )
    assert acton_optimal_light_level(1.1e-6 / 1e-3) == pytest.approx(math.log(1e-3 / 1.1e-6))


def test_corrected_saturation_intensity_and_neighbour_ratio() -> None:
    """87Rb D2: 1.669 mW/cm^2 (Steck); Cd+ I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2) = 1.09e-4 at 4 um (Acton's printed 7e-4 is
    2 pi too large); the 171Yb+ partial-rate I_sat is the species table's 50.83 mW/cm^2."""
    assert saturation_intensity_w_m2(TWO_PI * 6.0666e6, 780.241e-9) / 10.0 == pytest.approx(1.669, abs=1e-3)
    assert neighbour_intensity_ratio(214.5e-9, 4e-6) == pytest.approx(1.09e-4, abs=0.005e-4)
    assert neighbour_intensity_ratio(214.5e-9, 4e-6) * 2.0 * math.pi == pytest.approx(6.9e-4, abs=0.1e-4)
    line = YB.transition("S1/2-P1/2")
    assert saturation_intensity_w_m2(line.partial_rate_rad_s, line.wavelength_vac_m) == pytest.approx(
        line.i_sat_w_m2
    )
    assert line.i_sat_w_m2 / 10.0 == pytest.approx(50.83, abs=0.01)


# ---- efficiency chain, micromotion and Doppler factors (Sections 8.1, 8.8) ---------------------------------------------------------


def test_efficiency_chain_enters_once() -> None:
    assert geometric_efficiency(0.6) == pytest.approx(0.10, abs=1e-12)
    assert geometric_efficiency(0.25) == pytest.approx(0.5 * (1.0 - math.cos(math.asin(0.25))))
    assert emccd_effective_quantum_efficiency(0.48) == pytest.approx(0.24)
    eps = system_efficiency(0.10, 0.818, 0.731, 0.79)
    assert eps == pytest.approx(0.04724, abs=2e-4)
    assert camera_snr(100.0, 1.0, 0.0, 1) == pytest.approx(10.0)
    assert camera_snr(100.0, 50.0, 10.0, 4) < camera_snr(100.0, 50.0, 10.0, 1) < 10.0
    rates = rates_from_detected(472e3, 0.04356, dark_pumping_per_s=341.0, bright_pumping_per_s=16.4)
    from tests.readout_fixtures import snspd_detector

    detected, background = rates.detected(snspd_detector())
    assert detected == pytest.approx(472e3) and background == 4.2
    assert rates.ceiling is None and "apparatus" in rates.provenance[-1]


def test_micromotion_puts_j0_squared_on_the_carrier_and_j1_squared_on_the_sidebands() -> None:
    from scipy.special import j0, j1

    def rate(delta: float) -> float:
        return yb171_bright_rate_closed(0.5, GAMMA_S, delta)

    omega_rf = TWO_PI * 30e6
    beta = 0.4
    r = micromotion_detection_rate(rate, beta, omega_rf)
    expected = j0(beta) ** 2 * rate(0.0) + j1(beta) ** 2 * (rate(-omega_rf) + rate(omega_rf))
    assert r == pytest.approx(expected, rel=1e-12)
    assert micromotion_detection_rate(rate, 0.0, omega_rf) == pytest.approx(rate(0.0))
    assert r < rate(0.0)


def test_doppler_width_scaling_with_nbar_and_the_yb171_magnitude() -> None:
    """v_rms = sqrt(sum hbar omega (2 nbar + 1)/(2 m)); three modes at 3.0, 2.9 and 1.0 MHz give k v/2pi = 0.27 MHz at
    nbar = 0.1 and 3.4 MHz at nbar = 100, against the plan's 0.257 and 3.48 MHz (Section 8.8 [background], mode set not
    stated there) and the 19.7 MHz linewidth; the ratio follows sqrt((2 nbar + 1)) exactly."""
    m = YB.mass_u * ATOMIC_MASS_KG
    omegas = [TWO_PI * f for f in (3.0e6, 2.9e6, 1.0e6)]
    widths = {}
    for nbar in (0.1, 1.0, 20.0, 100.0):
        v = rms_velocity_m_per_s(m, omegas, [nbar] * 3)
        widths[nbar] = doppler_width_hz(v, 369.5e-9)
    assert widths[0.1] == pytest.approx(0.257e6, rel=0.05)
    assert widths[100.0] == pytest.approx(3.48e6, rel=0.05)
    assert widths[100.0] / widths[0.1] == pytest.approx(math.sqrt(201.0 / 1.2), rel=1e-9)
    assert widths[100.0] < 19.7e6 / 4.0
    projected = rms_velocity_m_per_s(m, omegas, [0.1] * 3, projections=[1.0, 0.0, 0.0])
    assert projected < rms_velocity_m_per_s(m, omegas, [0.1] * 3)


def test_thermal_transfer_probability_reduces_a_pi_pulse_with_doppler_width() -> None:
    """A resonant pi pulse transfers fully at rest and less as k v_rms grows past Omega (the untested motion-to-readout
    coupling of Section 8.8, derived from first principles)."""
    omega = TWO_PI * 100e3
    t_pi = math.pi / omega
    assert thermal_transfer_probability(omega, t_pi, TWO_PI / 393e-9, 0.0) == pytest.approx(1.0, abs=1e-12)
    cold = thermal_transfer_probability(omega, t_pi, TWO_PI / 393e-9, 0.01)
    hot = thermal_transfer_probability(omega, t_pi, TWO_PI / 393e-9, 0.1)
    assert 0.9 < cold < 1.0 and hot < cold


# ---- shelving and the mean-count curve (Sections 8.1, 8.2) -----------------------------------------------------------------------


def test_shelf_decay_error_and_the_christensen_branching() -> None:
    assert shelf_decay_error(4.5e-3, 30.0) == pytest.approx(1.5e-4, rel=2e-3)
    assert shelf_decay_error(420e-6, 1.168) == pytest.approx(3.596e-4, rel=1e-3)
    stranded = ShelvingBranching(to_shelf=0.23, to_bright=0.74, to_other_metastable=0.03)
    assert stranded.shelving_probability() == pytest.approx(0.23 / 0.26, rel=1e-12)
    assert round(stranded.shelving_probability(), 2) == 0.88
    repumped = ShelvingBranching(to_shelf=0.23, to_bright=0.74, to_other_metastable=0.03, repump_other=True)
    assert repumped.shelving_probability() == 1.0
    assert repumped.expected_excitations() == pytest.approx(1.0 / 0.23)
    leaky = ShelvingBranching(
        to_shelf=0.23, to_bright=0.7396, to_other_metastable=0.03, to_dark_ground=4e-4, repump_other=True
    )
    assert 0.998 < leaky.shelving_probability() < 0.9999


def test_mean_count_curve_is_the_two_state_rate_equation_and_its_fit_recovers_the_rates() -> None:
    """n̄(τ) = εR_o[(R_b/k) τ + (R_d/k^2)(1 - e^{-kτ})] equals the exact Markov-chain mean when only R_d and R_b act, and the
    least-squares fit recovers (εR_o, R_d, R_b) from noiseless samples."""
    from tests.readout_fixtures import crain_record_model

    rm = crain_record_model()
    taus = np.linspace(1e-4, 60e-3, 12)
    curve = np.asarray(mean_count_curve(taus, 472e3, 341.0, 16.4))
    for t, n in zip(taus, curve):
        assert rm.mean_counts("bright", float(t)) - rm.background_per_s * t == pytest.approx(
            float(n), rel=1e-9
        )
    (r0, rd, rb), _sigma = fit_mean_count_curve(taus, curve, guess=(4e5, 300.0, 10.0))
    assert (
        r0 == pytest.approx(472e3, rel=1e-6)
        and rd == pytest.approx(341.0, rel=1e-5)
        and rb == pytest.approx(16.4, rel=1e-5)
    )


# ---- the scheme: polarity as an enum, Pi_dark not rank one ------------------------------------------------------------------------


def test_scheme_polarity_direct_vs_shelving_and_the_non_rank_one_dark_operator() -> None:
    yb = ReadoutScheme.direct(1)
    ca = ReadoutScheme.shelving(1)
    assert yb.polarity == "bright_is_1" and ca.polarity == "bright_is_0"
    assert yb.bit_of_class("bright") == 1 and ca.bit_of_class("bright") == 0
    assert yb.dark_weights() == (1.0, 0.0) and ca.dark_weights() == (0.0, 1.0)
    harty = ReadoutScheme.shelving(0, transfer_probability=1.0 - 1.7e-4, off_resonant_shelving=3e-4)
    assert harty.transfer is not None
    assert harty.dark_weights() == pytest.approx((1.0 - 1.7e-4, 3e-4))
    assert harty.start_distribution(0)["shelf"] == pytest.approx(1.0 - 1.7e-4)
    assert harty.start_distribution(1)["bright"] == pytest.approx(1.0 - 3e-4)
    for_species = ReadoutScheme.for_species(species("40Ca+"), ("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"))
    assert for_species.kind == "shelving" and for_species.classes == ("bright", "shelf")
    with pytest.raises(ValueError):
        ReadoutScheme("direct", ("dark", "dark"))
    with pytest.raises(ValueError):
        FluorescenceRates(0.0, 1.0, 1.0)
