"""Section 9.5 targets of the photon-record layer (PLAN.md Section 8.2, 8.5, 8.8): the exact chain against the single-jump
closed forms, Crain's corrections, Acton's distributions, Wineland's zero-photon law, the mcsolve trajectory path, detector
non-idealities, neighbour-coupled records and the camera image model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.readout.detection import (
    CRAIN_SPECTATOR_ANCHORS_S,
    CameraGeometry,
    ClassPath,
    Detector,
    RecordModel,
    acton_bright_distribution,
    acton_dark_distribution,
    apply_detector_nonidealities,
    count_anomaly_band,
    crain_printed_bright_error,
    crain_spectator_alpha_s,
    first_photon_cutoff_s,
    mcsolve_records,
    neighbourhood_model,
    poisson_pmf,
    psf_leakage_from_geometry,
    sample_camera_image,
    sample_register_records,
    sample_spectator_offset_rad_s,
    single_jump_count_distribution,
    spectator_coherence,
    spectator_offset_sigma_rad_s,
    zero_photon_probability,
    zero_threshold_errors,
)
from tests.readout_fixtures import crain_record_model, myerson_record_model, two_state_rates

CRAIN = dict(detected=472e3, r_d=341.0, r_b=16.4, r_bg=4.2)


# ---- Wineland, Crain Eq. 1, Acton distributions (rows "Zero-photon law", "Crain corrections", "Acton distributions") --------------


def test_wineland_zero_photon_law() -> None:
    # the binomial (1 - eta)^N approaches e^{-n_d} only for eta << 1: at eta = 1e-3 the two agree to N eta^2/2 = 0.5 %
    assert zero_photon_probability(10_000, 1e-3) == pytest.approx(math.exp(-10.0), rel=6e-3)
    assert zero_photon_probability(1000, 0.01) == pytest.approx(math.exp(-10.0), rel=0.06)
    assert math.exp(-10.0) == pytest.approx(4.5400e-5, rel=1e-4)
    assert math.exp(-100.0) == pytest.approx(3.7201e-44, rel=1e-4)
    assert zero_photon_probability(100_000, 1e-3) == pytest.approx(math.exp(-100.0), rel=0.06)
    assert zero_photon_probability(0, 0.3) == 1.0


def test_crain_eq1_sums_to_one_only_with_the_no_jump_branch() -> None:
    t, r1, r2, rt = 20e-6, 472e3 + 4.2, 4.2, 341.0
    full = single_jump_count_distribution(80, t, r1, r2, rt)
    printed = single_jump_count_distribution(80, t, r1, r2, rt, include_no_jump=False)
    assert full.sum() == pytest.approx(1.0, abs=1e-12)
    assert printed.sum() == pytest.approx(1.0 - math.exp(-rt * t), abs=1e-12)
    # the paper's own regime (a 500 us record with R_d = 341 Hz) carries about 16 % of the probability without the branch,
    # 1 % at the 30 us scale of a fast record
    assert single_jump_count_distribution(
        400, 500e-6, r1, r2, rt, include_no_jump=False
    ).sum() == pytest.approx(1.0 - math.exp(-rt * 500e-6), abs=1e-9)
    assert single_jump_count_distribution(
        60, 30e-6, r1, r2, rt, include_no_jump=False
    ).sum() == pytest.approx(0.0102, abs=2e-4)


def test_exact_chain_reduces_to_the_single_jump_form_when_one_transition_acts() -> None:
    """With only R_d (bright start) the Markov chain has exactly one jump, so the matrix-exponential distribution equals the
    single-jump closed form to machine precision; the same for a dark start with only R_b."""
    det = Detector("snspd", 0.04356, 0.0, {}, None, None, 22e-6)
    bright_only = RecordModel.from_rates(two_state_rates(472e3, 0.04356, 341.0, 0.0), det)
    dist = bright_only.count_distribution("bright", 22e-6, n_max=60)
    closed = single_jump_count_distribution(60, 22e-6, 472e3, 0.0, 341.0)
    assert np.max(np.abs(dist.pmf - closed)) < 1e-13
    dark_only = RecordModel.from_rates(two_state_rates(472e3, 0.04356, 0.0, 16.4), det)
    dist_d = dark_only.count_distribution("dark", 22e-6, n_max=60)
    closed_d = single_jump_count_distribution(60, 22e-6, 0.0, 472e3, 16.4)
    assert np.max(np.abs(dist_d.pmf - closed_d)) < 1e-13
    assert dist.overflow < 1e-20 and dist_d.overflow < 1e-20


def test_acton_distributions_normalize_and_match_the_single_jump_quadrature() -> None:
    """Acton Eqs. 5-6 are the single-jump mixtures with R_2 = 0 (bright) or R_1 = 0 (dark) and no background: they normalize
    to 1e-12 and agree with the Gauss-Legendre quadrature of Crain's corrected Eq. 1 to the Section 9.5 row's 2e-16.

    The residual is pure double-precision round-off (2e-16 is one ulp at these magnitudes), so it grows slowly with the
    node count as the quadrature sums more terms: the row's 2e-16 is reached at 64 nodes and a few ulp at the default 96
    (M5 finding, 2026-09-07; the earlier bound of 1e-14 was 50x looser than the row).
    """
    lambda0, a1, a2 = 12.0, 0.02, 0.05
    dark = acton_dark_distribution(200, lambda0, a1)
    bright = acton_bright_distribution(200, lambda0, a2)
    assert dark.sum() == pytest.approx(1.0, abs=1e-12) and bright.sum() == pytest.approx(1.0, abs=1e-12)
    t = 1.0  # the mixtures depend only on lambda_0 and alpha/eta: choose unit window, detected rate lambda_0
    quad_dark = single_jump_count_distribution(200, t, 0.0, lambda0, a1 * lambda0, nodes=64)
    quad_bright = single_jump_count_distribution(200, t, lambda0, 0.0, a2 * lambda0, nodes=64)
    # two machine epsilons: the Linux runner's libm lands exactly one epsilon (2.22e-16) away from the quadrature
    assert np.max(np.abs(dark - quad_dark)) < 5e-16
    assert np.max(np.abs(bright - quad_bright)) < 5e-16
    # at the default node count the same identity holds within a few ulp
    assert np.max(np.abs(dark - single_jump_count_distribution(200, t, 0.0, lambda0, a1 * lambda0))) < 1e-15
    assert np.max(np.abs(bright - single_jump_count_distribution(200, t, lambda0, 0.0, a2 * lambda0))) < 1e-15
    assert dark[0] == pytest.approx(
        math.exp(-a1 * lambda0) * (1.0 + a1 / (1.0 - a1) * (1.0 - math.exp(-(1.0 - a1) * lambda0)))
    )
    assert np.argmax(bright) in (11, 12)  # Poisson(12) peaks at 11 and 12 equally; the smearing tips it to 11
    # no leakage: pure Poisson and pure delta
    assert np.allclose(acton_bright_distribution(60, 8.0, 0.0), np.asarray(poisson_pmf(np.arange(61), 8.0)))
    assert acton_dark_distribution(10, 8.0, 0.0)[0] == 1.0


# ---- Crain Eqs. 2-3 and the SNSPD operating point (rows "Crain corrections", "SNSPD operating point") ---------------------------


def test_crain_bright_error_at_11us_is_6e_3_corrected_against_7e_4_printed() -> None:
    eps_b, eps_d = zero_threshold_errors(11e-6, **_crain_args())
    assert eps_b == pytest.approx(6.3e-3, abs=0.1e-3)
    printed = crain_printed_bright_error(11e-6, 472e3, 472e3 / 0.04356, 341.0, 4.2)
    assert printed == pytest.approx(7.2e-4, abs=0.05e-4)
    assert eps_b / printed == pytest.approx(8.7, abs=0.3)
    # the pumped-bright ion still needs time to emit: eps_D = (R_bg + R_b) t - (R_b/eps R_o)(1 - e^{-eps R_o t}) to first order
    expected_d = (4.2 + 16.4) * 11e-6 - 16.4 / 472e3 * (1.0 - math.exp(-472e3 * 11e-6))
    assert eps_d == pytest.approx(expected_d, rel=2e-3)


def test_snspd_zero_threshold_optimum_near_22us_at_5_9e_4() -> None:
    """The average of the corrected Eqs. 2-3 has its interior minimum near 20 to 25 us at 5.85e-4 (the plan's 5.84e-4 closed
    form) against Crain's measured 6.9e-4 at 11 us average detection time; detector non-idealities are the named remainder."""
    windows = np.linspace(5e-6, 60e-6, 111)
    avg = np.array([0.5 * sum(zero_threshold_errors(t, **_crain_args())) for t in windows])
    k = int(np.argmin(avg))
    assert 20e-6 <= windows[k] <= 25e-6, "the Section 9.5 row's own window; the fine-grid argmin is 21.26 us"
    assert avg[k] == pytest.approx(5.85e-4, abs=0.03e-4)
    assert avg[k] < 6.9e-4
    zero_bg = np.array([0.5 * sum(zero_threshold_errors(t, 472e3, 341.0, 16.4, 0.0)) for t in windows])
    assert zero_bg.min() < avg[k]
    assert first_photon_cutoff_s(341.0, 4.2, 472e3) == pytest.approx(math.log(341.0 / 4.2) / 472e3)


def test_exact_chain_agrees_with_the_zero_threshold_closed_forms_to_first_order() -> None:
    """The closed forms neglect re-pumping after a jump (a second jump); the exact chain differs from them by that second-order
    amount only (relative 1e-3 at 22 us)."""
    rm = crain_record_model()
    eps_b, eps_d = zero_threshold_errors(22e-6, **_crain_args())
    exact_b = rm.count_distribution("bright", 22e-6).pmf[0]
    exact_d = rm.count_distribution("dark", 22e-6).probability_above(0.5)
    assert exact_b == pytest.approx(eps_b, rel=1e-3)
    assert exact_d == pytest.approx(eps_d, rel=1e-3)
    assert rm.mean_counts("bright", 22e-6) == pytest.approx(472e3 * 22e-6, rel=5e-3)


def _crain_args() -> dict[str, float]:
    return dict(
        detected_bright_per_s=CRAIN["detected"],
        dark_pumping_per_s=CRAIN["r_d"],
        bright_pumping_per_s=CRAIN["r_b"],
        background_per_s=CRAIN["r_bg"],
    )


# ---- sampled records against the exact distribution --------------------------------------------------------------------------------


def test_sampled_records_follow_the_exact_count_distribution() -> None:
    rm = crain_record_model()
    rng = np.random.default_rng(11)
    n = 20000
    totals = np.array([rm.sample_record("bright", 22e-6, rng).total for _ in range(n)])
    dist = rm.count_distribution("bright", 22e-6)
    assert totals.mean() == pytest.approx(dist.mean(), abs=4.0 * math.sqrt(dist.mean() / n))
    assert np.mean(totals == 0) == pytest.approx(dist.pmf[0], abs=4.0 * math.sqrt(dist.pmf[0] / n) + 1e-4)
    dark_totals = np.array([rm.sample_record("dark", 22e-6, rng).total for _ in range(n)])
    p_hit = rm.count_distribution("dark", 22e-6).probability_above(0.5)
    assert np.mean(dark_totals > 0) == pytest.approx(p_hit, abs=4.0 * math.sqrt(p_hit / n) + 1e-4)


def test_long_windows_carry_many_transitions_not_one() -> None:
    """Myerson's 420 us on a 1168 ms shelf is single-jump; a 100 ms window on Crain's 171Yb+ chain is not: the ion pumps dark
    in 1/R_d = 2.9 ms and back in 1/R_b = 61 ms, so the sampled paths carry a few cycles and the exact chain's mean count
    sits well above the single-jump (bright -> dark, no return) extrapolation."""
    rm = crain_record_model(window_s=100e-3)
    rng = np.random.default_rng(5)
    jumps = [rm.sample_path("bright", 100e-3, rng).n_jumps for _ in range(200)]
    assert 2.5 < np.mean(jumps) < 8.0
    single = single_jump_count_distribution(int(472e3 * 100e-3 * 1.2), 100e-3, 472e3 + 4.2, 4.2, 341.0)
    single_mean = float(np.dot(np.arange(single.size), single))
    assert rm.mean_counts("bright", 100e-3) > 1.2 * single_mean


def test_mcsolve_trajectory_path_agrees_with_the_chain() -> None:
    """The quantum-jump record of the reduced class space reproduces the mean count and the shelf decay statistics."""
    rm = myerson_record_model()
    recs = mcsolve_records(rm, "bright", 420e-6, 300, seed=3)
    mean = rm.mean_counts("bright", 420e-6)
    totals = np.array([r.total for r in recs])
    assert totals.mean() == pytest.approx(mean, abs=4.0 * math.sqrt(mean / 300))
    assert all(r.arrivals_s is not None and np.all(np.diff(r.arrivals_s) >= 0.0) for r in recs)
    dark = mcsolve_records(rm, "shelf", 420e-6, 300, seed=4, sub_bin_s=10e-6)
    p0 = rm.count_distribution("shelf", 420e-6).pmf[0]
    assert np.mean([r.total == 0 for r in dark]) == pytest.approx(p0, abs=0.08)
    assert all(r.sub_bins is not None and r.sub_bins.size == 42 for r in dark)


def test_class_path_bookkeeping_and_bright_occupancy() -> None:
    path = ClassPath(1.0, "bright", (0.25, 0.75), ("dark", "bright"))
    assert path.segments() == [(0.0, 0.25, "bright"), (0.25, 0.75, "dark"), (0.75, 1.0, "bright")]
    assert path.occupancy_s("bright") == pytest.approx(0.5) and path.class_at(0.5) == "dark"
    rm = crain_record_model()
    occ = rm.bright_occupancy_mean_s("bright", 22e-6)
    assert 0.99 * 22e-6 < occ < 22e-6
    assert rm.class_probabilities("bright", 0.0) == pytest.approx([1.0, 0.0, 0.0])
    probs = rm.class_probabilities("dark", 1.0)
    assert probs[0] == pytest.approx(16.4 / (16.4 + 341.0), rel=1e-6)


# ---- detector non-idealities (Section 8.8) ---------------------------------------------------------------------------------------


def test_dead_time_and_afterpulsing_act_on_arrival_times() -> None:
    rng = np.random.default_rng(0)
    arr = np.array([0.0, 0.5e-6, 1.1e-6, 3.0e-6, 3.2e-6])
    kept = apply_detector_nonidealities(arr, 1.0e-6, None, rng, 10e-6)
    assert kept.tolist() == [0.0, 1.1e-6, 3.0e-6]
    with_ap = apply_detector_nonidealities(np.array([1e-6]), 0.5e-6, 0.999999, rng, 10e-6)
    assert with_ap.size == 2 and with_ap[1] > with_ap[0] + 0.5e-6
    assert apply_detector_nonidealities(arr, None, None, rng, 10e-6).size == arr.size
    # a non-paralyzable dead time reduces the mean detected rate to lambda/(1 + lambda tau)
    det = Detector("snspd", 0.04356, 4.2, {}, 2e-6, None, 1e-3)
    rm = RecordModel.from_rates(two_state_rates(472e3, 0.04356, 0.0, 0.0), det)
    totals = [rm.sample_record("bright", 1e-3, rng).total for _ in range(300)]
    expected = 472e3 / (1.0 + 472e3 * 2e-6) * 1e-3
    assert np.mean(totals) == pytest.approx(expected, rel=0.03)


# ---- neighbour-coupled records and the camera (Section 8.5) ------------------------------------------------------------------------


def test_register_records_add_a_bright_neighbours_light_to_a_dark_ion() -> None:
    rm = crain_record_model()
    rngs = [np.random.default_rng(k) for k in range(2)]
    n = 4000
    dark_alone = np.array(
        [sample_register_records([rm, rm], ["dark", "dark"], 22e-6, rngs)[0].total for _ in range(n)]
    )
    dark_next_to_bright = np.array(
        [
            sample_register_records([rm, rm], ["dark", "bright"], 22e-6, rngs, leakage={1: 0.04})[0].total
            for _ in range(n)
        ]
    )
    assert dark_alone.mean() < 0.01
    assert dark_next_to_bright.mean() == pytest.approx(0.04 * 472e3 * 22e-6, rel=0.1)
    frozen = neighbourhood_model([rm, rm], 0, ["dark", "bright"], {1: 0.04})
    assert frozen.background_per_s == pytest.approx(4.2 + 0.04 * 472e3)
    no_leak = sample_register_records([rm, rm], ["dark", "bright"], 22e-6, rngs, leakage={})[0]
    assert no_leak.ion == 0


def test_camera_geometry_leakage_from_the_airy_psf_and_an_aberrated_gaussian() -> None:
    """A diffraction-limited NA 0.25 system at 397 nm leaks 0.25 % into a 28-pixel neighbour ROI at 14 um; Burrell's measured
    4.0 % nearest and 0.9 % next-nearest belong to an aberrated PSF (Section 8.3): a Gaussian of sigma 4.1 um reproduces the
    nearest-neighbour figure but not the 0.9 % wing, which a Gaussian has not."""
    airy = CameraGeometry(tuple(np.arange(4) * 14e-6), 2.6e-6, 50, 10, 397e-9, numerical_aperture=0.25)
    w = airy.weights(1)
    assert w.shape == (10, 50) and w.sum() == pytest.approx(1.0, abs=0.02)
    assert airy.leakage_fraction(1, 2, 28) < 0.005
    assert airy.leakage_fraction(1, 2, 10) < airy.leakage_fraction(1, 2, 60)
    roi_one_spacing = int(round(math.pi * 7.0**2 / 2.6**2))
    gauss = CameraGeometry(tuple(np.arange(4) * 14e-6), 2.6e-6, 50, 10, 397e-9, psf_sigma_m=4.1e-6)
    assert gauss.leakage_fraction(1, 2, roi_one_spacing) == pytest.approx(0.040, abs=0.006)
    assert gauss.leakage_fraction(1, 3, roi_one_spacing) < 0.001
    with pytest.raises(ValueError):
        CameraGeometry((0.0,), 2.6e-6, 10, 10, 397e-9)
    # Detector.psf_leakage is the ABSOLUTE fraction of the neighbour's total light, not Burrell's ROI-relative ratio: the
    # two differ by the ion's own ROI collection efficiency, so the conversion happens once (audit 2026-09-07 B6)
    own = gauss.own_roi_efficiency(1, roi_one_spacing)
    assert own == pytest.approx(0.757, abs=0.02)
    absolute = gauss.absolute_leakage_fraction(1, 2, roi_one_spacing)
    assert absolute == pytest.approx(gauss.leakage_fraction(1, 2, roi_one_spacing) * own, rel=1e-12)
    assert absolute == pytest.approx(0.0295, abs=0.004), (
        "feeding 4.0 % straight in overstates the leak by 32 %"
    )
    derived = psf_leakage_from_geometry(gauss, roi_one_spacing, max_distance=2)
    # the derived value averages the pairs at that distance: each ion sits at its own sub-pixel phase on the grid, so the
    # end pairs (0.0274 / 0.0209) straddle the interior pair's 0.0295 and the mean is 0.0260
    assert set(derived) == {1, 2} and derived[1] == pytest.approx(0.0260, abs=0.002)
    assert 0.7 * absolute < derived[1] <= absolute
    assert derived[2] < 0.1 * derived[1]
    assert all(0.0 <= v <= 1.0 for v in derived.values())
    # the derived dict is a valid Detector.psf_leakage without a hand-set free parameter (Section 8.3)
    det = Detector(
        kind="camera",
        efficiency=0.01,
        background_cps=0.0,
        psf_leakage=derived,
        dead_time_s=None,
        afterpulse_prob=None,
        window_s=400e-6,
        numerical_aperture=0.25,
        pixel_m=2.6e-6,
    )
    assert det.has_crosstalk and det.leakage(1) == pytest.approx(derived[1])


def test_camera_image_means_follow_the_weights() -> None:
    rm = myerson_record_model()
    geo = CameraGeometry((0.0, 14e-6), 2.6e-6, 30, 6, 397e-9, psf_sigma_m=4e-6)
    rng = np.random.default_rng(1)
    paths = [ClassPath(400e-6, "bright", (), ()), ClassPath(400e-6, "shelf", (), ())]
    images = np.array(
        [sample_camera_image(geo, [rm, rm], paths, 400e-6, rng) for _ in range(300)], dtype=float
    )
    mean = images.mean(axis=0)
    # the background is the DETECTOR's, spread over the pixels ONCE per exposure: a register of N ions used to collect
    # N x R_bg (audit 2026-09-07 B12)
    expected = (
        geo.weights(0) * rm.detected_bright_per_s * 400e-6 + rm.background_per_s * 400e-6 / geo.n_pixels
    )
    assert np.allclose(mean, expected, atol=0.6)
    assert mean.sum() == pytest.approx(expected.sum(), rel=0.05)
    four = [ClassPath(400e-6, "shelf", (), ())] * 4
    geo4 = CameraGeometry(tuple(np.arange(4) * 14e-6), 2.6e-6, 30, 6, 397e-9, psf_sigma_m=4e-6)
    dark_images = np.array(
        [sample_camera_image(geo4, [rm] * 4, four, 400e-6, rng) for _ in range(300)], dtype=float
    )
    assert dark_images.mean(axis=0).sum() == pytest.approx(rm.background_per_s * 400e-6, rel=0.15)


def test_count_anomaly_band_brackets_both_hypotheses() -> None:
    """Section 8.6's "count anomaly" herald: a total outside the [1e-6, 1 - 1e-6] band of BOTH the bright and the dark
    count distribution is explained by neither hypothesis (a cosmic ray, an afterpulse burst, a stray-light flash), so the
    run flags it. Before the 2026-09-07 M5 audit ``Result.heralds`` promised the flag and nothing set it (B10)."""
    rm = crain_record_model(window_s=22e-6)
    lo, hi = count_anomaly_band(rm, 22e-6)
    bright = rm.count_distribution("bright", 22e-6)
    dark = rm.count_distribution("dark", 22e-6)
    assert lo == 0, "the dark hypothesis explains zero counts"
    assert hi > bright.mean() > 0.0
    # the band covers all but 1e-6 of each hypothesis
    n = np.arange(len(bright.pmf))
    for dist in (bright, dark):
        inside = float(np.sum(dist.pmf[(n >= lo) & (n <= hi)]))
        assert inside > 1.0 - 3e-6, dist.mean()
    # a shelving scheme's non-bright hypothesis is the shelf, not "dark"
    my = myerson_record_model()
    lo_s, hi_s = count_anomaly_band(my, 420e-6, ("bright", "shelf"))
    assert lo_s == 0 and hi_s > my.mean_counts("bright", 420e-6)
    # a tighter quantile widens the band, and the quantile is checked
    wide = count_anomaly_band(rm, 22e-6, quantile=1e-9)
    assert wide[1] >= hi
    with pytest.raises(ValueError, match="quantile"):
        count_anomaly_band(rm, 22e-6, quantile=0.6)


def test_spectator_dephasing_is_gaussian_and_sampled_as_a_quasi_static_offset() -> None:
    assert spectator_coherence(814e-3, 814e-3) == pytest.approx(math.exp(-1.0))
    sigma = spectator_offset_sigma_rad_s(0.814)
    rng = np.random.default_rng(3)
    taus = np.array([0.1, 0.4, 0.8])
    draws = np.array([sample_spectator_offset_rad_s(0.814, rng) for _ in range(20000)])
    for tau in taus:
        assert np.mean(np.cos(draws * tau)) == pytest.approx(spectator_coherence(tau, 0.814), abs=0.02)
    assert sigma == pytest.approx(math.sqrt(2.0) / 0.814)
    (d1, a1), (d2, a2) = CRAIN_SPECTATOR_ANCHORS_S
    assert crain_spectator_alpha_s(d1) == pytest.approx(a1) and crain_spectator_alpha_s(d2) == pytest.approx(
        a2
    )
    assert crain_spectator_alpha_s(300e-6) > a1
