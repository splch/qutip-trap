"""Section 9.5 targets of the discriminator layer (PLAN.md Section 8.3): Myerson's recursion and optimum, the maximum-likelihood
asymptote, the adaptive method, Noek's and Crain's first-photon protocols, Burrell's camera discriminators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.readout.detection import (
    CameraGeometry,
    ClassPath,
    Detector,
    RecordModel,
    first_photon_cutoff_s,
    sample_camera_image,
)
from qutip_trap.readout.discriminate import (
    AdaptiveML,
    FirstPhoton,
    ThresholdDiscriminator,
    TimeResolvedML,
    average_detection_time_s,
    camera_log_likelihoods,
    camera_threshold_decode,
    decode_camera_image,
    exact_log_likelihoods,
    myerson_brute_force,
    myerson_log_likelihoods,
    optimize_threshold,
    spatio_temporal_log_likelihoods,
)
from tests.readout_fixtures import crain_record_model, myerson_record_model, two_state_rates

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---- Myerson 2008 (rows "Myerson recursion", "40Ca+ optimum") ----------------------------------------------------------------------


def test_myerson_recursion_equals_brute_force_and_the_weights_normalize() -> None:
    rm = myerson_record_model()
    rng = np.random.default_rng(1)
    for start in ("bright", "shelf"):
        for _ in range(5):
            rec = rm.sample_record(start, 420e-6, rng, sub_bin_s=10e-6)
            assert rec.sub_bins is not None
            lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            pb, pd = myerson_brute_force(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            assert lb == pytest.approx(math.log(pb), abs=1e-10)
            assert ld == pytest.approx(math.log(pd), abs=1e-10)
    # the mixture weights (1 - t_b/tau) + N (t_s/tau) = 1 exactly: with B = D the dark likelihood equals the bright one
    flat = RecordModel(rm.detected_bright_per_s, rm.detected_bright_per_s + 442.0, rm.rates)
    counts = np.array([20, 25, 22, 19])
    lam = (flat.detected_bright_per_s + flat.background_per_s) * 10e-6
    flat_b = RecordModel(1e-9, lam / 10e-6 - 1e-9, rm.rates)
    lb, ld = myerson_log_likelihoods(counts, 10e-6, flat_b, dark_class="shelf")
    assert lb == pytest.approx(ld, abs=1e-9)


def test_log_domain_survives_the_227_sub_bin_underflow() -> None:
    """M_N underflows near N ~ 227 sub-bins at the paper's rates (Section 8.3): the linear brute force returns p_D = 0 for a
    long bright record while the recursion returns a finite log-likelihood, and the two agree wherever both are finite."""
    rm = myerson_record_model()
    rng = np.random.default_rng(2)
    rec = rm.sample_record("bright", 2.27e-3, rng, sub_bin_s=10e-6)
    assert rec.sub_bins is not None and rec.sub_bins.size == 227
    lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
    pb, pd = myerson_brute_force(rec.sub_bins, 10e-6, rm, dark_class="shelf")
    assert math.isfinite(lb) and math.isfinite(ld) and lb > ld
    assert pb == 0.0 or lb == pytest.approx(math.log(pb), rel=1e-9)
    long_rec = rm.sample_record("bright", 6.0e-3, rng, sub_bin_s=10e-6)
    assert long_rec.sub_bins is not None
    lb2, ld2 = myerson_log_likelihoods(long_rec.sub_bins, 10e-6, rm, dark_class="shelf")
    assert math.isfinite(lb2) and math.isfinite(ld2)
    assert myerson_brute_force(long_rec.sub_bins, 10e-6, rm, dark_class="shelf")[1] < 1e-250


def test_exact_forward_likelihood_agrees_with_myerson_to_first_order_in_t_over_tau() -> None:
    """Records without a jump: the exact hidden-Markov likelihood equals Myerson's first-order form to 1e-5 in log (bright)
    and 1e-6 (shelf). Records with a decay inside the window: Myerson places the decay at the start of its sub-bin, the exact
    kernel integrates over its position, so ln p_D differs at the few-percent level with the SAME decision (the "similar
    expressions ... with negligible effect on the results" of the source)."""
    rm = myerson_record_model()
    rng = np.random.default_rng(3)
    seen_jump = False
    for start in ("bright", "shelf"):
        for _ in range(40):
            rec = rm.sample_record(start, 420e-6, rng, sub_bin_s=10e-6)
            assert rec.sub_bins is not None and rec.path is not None
            lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            eb, ed = exact_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            # the hypothesis matching the record's start class agrees to 1e-5 when no jump occurred; the OTHER hypothesis
            # must invoke a decay, whose position inside its sub-bin the exact kernel integrates over (percent-level)
            tight = start == "bright" and rec.path.n_jumps == 0
            assert eb == pytest.approx(lb, abs=1e-5 if tight or start == "shelf" else 0.2)
            if rec.path.n_jumps == 0 and start == "shelf":
                assert ed == pytest.approx(ld, abs=1e-5)
            else:
                # a decay inside a sub-bin: Myerson puts it at the sub-bin's start, the exact kernel integrates over its
                # position, so ln p_D differs by up to about one nat, and the decision agrees away from the boundary
                seen_jump = seen_jump or rec.path.n_jumps > 0
                assert ed == pytest.approx(ld, abs=1.5)
                if abs(lb - ld) > 2.0:
                    assert (eb >= ed) == (lb >= ld)
    # force a decay to exercise the second branch
    late = ClassPath(420e-6, "shelf", (150e-6,), ("bright",))
    rec = rm.sample_record("shelf", 420e-6, rng, sub_bin_s=10e-6, path=late)
    assert rec.sub_bins is not None
    lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
    eb, ed = exact_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
    assert ed == pytest.approx(ld, abs=1.5) and abs(ed - ld) > 1e-4


def test_ca40_threshold_optimum_from_the_ideal_poisson_model() -> None:
    """The exact chain's average error has an interior optimum in (n_c, t_b): 1.24e-4 at (3.5, 320 us) with ideal Poisson
    statistics; at Myerson's measured operating point (5.5, 420 us) it gives 1.37e-4 against the measured 1.8(1)e-4, whose
    dark-state error the paper attributes about 20 % to non-Poissonian PMT dark counts (cosmic rays) that the ideal model has
    not (Section 9.5 row "40Ca+ optimum"; M5 finding)."""
    rm = myerson_record_model()
    opt = optimize_threshold(rm, np.arange(100e-6, 1001e-6, 20e-6), dark_start="shelf")
    assert 280e-6 <= opt.best.window_s <= 360e-6
    assert opt.best.n_c in (2.5, 3.5, 4.5)
    assert opt.best.eps == pytest.approx(1.24e-4, abs=0.1e-4)
    assert opt.best.eps_B < opt.best.eps_D
    at_paper = ThresholdDiscriminator(5.5, 420e-6).error_rates(rm, dark_start="shelf")
    eps_paper = 0.5 * sum(at_paper)
    assert eps_paper == pytest.approx(1.37e-4, abs=0.05e-4)
    assert eps_paper < 1.8e-4 < 1.5 * eps_paper
    # the interior optimum exists: short windows lose to overlap, long ones to shelf decay
    scan = {p.window_s: p.eps for p in opt.scan}
    assert scan[100e-6] > 10 * opt.best.eps and scan[980e-6] > 1.5 * opt.best.eps
    assert at_paper[1] == pytest.approx(0.765 * 420e-6 / 1.168, rel=0.06)


@pytest.mark.slow
def test_time_resolved_ml_beats_the_threshold_on_shelf_decays() -> None:
    """Myerson: the ML method tends to 0.87(11)e-4 with eps_D = 1.5(2)e-4 (its Poisson simulation's asymptote 0.89e-4), the
    bright state detected to eps_B < 2e-6; Monte Carlo over 30 000 records per state at t_b = 1 ms."""
    rm = myerson_record_model()
    ml = TimeResolvedML(10e-6, 1000e-6, dark_class="shelf")
    rng = np.random.default_rng(4)
    n = 30_000
    err_b = sum(
        not ml.decide(rm.sample_record("bright", 1000e-6, rng, sub_bin_s=10e-6), rm).bright for _ in range(n)
    )
    err_d = sum(
        ml.decide(rm.sample_record("shelf", 1000e-6, rng, sub_bin_s=10e-6), rm).bright for _ in range(n)
    )
    assert err_b <= 1
    eps_d = err_d / n
    assert eps_d == pytest.approx(1.5e-4, abs=1.0e-4)
    assert 0.5 * (err_b / n + eps_d) < 1.37e-4


def test_ml_catches_a_late_decay_that_the_threshold_would_call_bright() -> None:
    """A shelf decay 200 us into a 420 us window leaves 12 bright counts: the threshold calls it bright; the ML weighs the 20
    empty sub-bins (20 x 0.56 nats) against the decay prior t_s/tau = e^-11.7 and catches most; at 300 us it catches nearly
    all, the decision boundary sitting near 21 empty sub-bins (Section 8.3)."""
    rm = myerson_record_model()
    rng = np.random.default_rng(5)
    ml = TimeResolvedML(10e-6, 420e-6, dark_class="shelf")
    for t_decay, min_caught, min_threshold_wrong in ((200e-6, 160, 190), (300e-6, 195, 100)):
        path = ClassPath(420e-6, "shelf", (t_decay,), ("bright",))
        caught = threshold_wrong = 0
        for _ in range(200):
            rec = rm.sample_record("shelf", 420e-6, rng, sub_bin_s=10e-6, path=path)
            d = ml.decide(rec, rm)
            caught += int(not d.bright)
            threshold_wrong += int(rec.total > 5.5)
            assert d.posterior_error is not None and 0.0 <= d.posterior_error <= 0.5
        assert threshold_wrong > min_threshold_wrong
        assert caught > min_caught


def test_adaptive_early_termination_is_faster_at_the_same_error() -> None:
    rm = myerson_record_model()
    rng = np.random.default_rng(6)
    ad = AdaptiveML(10e-6, 500e-6, 1e-4, dark_class="shelf")
    dec_b = [ad.decide(rm.sample_record("bright", 500e-6, rng, sub_bin_s=10e-6), rm) for _ in range(3000)]
    dec_d = [ad.decide(rm.sample_record("shelf", 500e-6, rng, sub_bin_s=10e-6), rm) for _ in range(3000)]
    t_b, t_d = average_detection_time_s(dec_b), average_detection_time_s(dec_d)
    assert 40e-6 < t_b < 90e-6 and 140e-6 < t_d < 260e-6
    assert 0.5 * (t_b + t_d) < 200e-6
    # the cutoff is an error PROBABILITY: bright records stopped early as dark occur at about the cutoff rate
    assert np.mean([not d.bright for d in dec_b]) < 1e-3
    assert np.mean([d.bright for d in dec_d]) < 3e-3
    assert all(d.time_used_s <= 500e-6 for d in dec_b + dec_d)


# ---- first-photon protocols (rows "Crain corrections", "SNSPD operating point", Noek) ---------------------------------------------


def test_crain_stop_on_first_photon_reaches_the_closed_form_errors_and_11us_average() -> None:
    rm = crain_record_model()
    fp = FirstPhoton(22e-6)
    rng = np.random.default_rng(7)
    n = 20_000
    dec_b = [fp.decide(rm.sample_record("bright", 22e-6, rng, arrivals=True), rm) for _ in range(n)]
    dec_d = [fp.decide(rm.sample_record("dark", 22e-6, rng, arrivals=True), rm) for _ in range(n)]
    eps_b = np.mean([not d.bright for d in dec_b])
    eps_d = np.mean([d.bright for d in dec_d])
    assert eps_b == pytest.approx(7.5e-4, abs=3e-4)
    assert eps_d == pytest.approx(4.2e-4, abs=2e-4)
    avg_time = 0.5 * (average_detection_time_s(dec_b) + average_detection_time_s(dec_d))
    assert avg_time == pytest.approx(12e-6, abs=1.5e-6)
    assert average_detection_time_s(dec_b) == pytest.approx(1.0 / 472e3, rel=0.1)


def test_noek_two_photon_rule_uses_the_cutoff_time() -> None:
    rm = crain_record_model()
    tau_c = first_photon_cutoff_s(341.0, 4.2, 472e3)
    fp = FirstPhoton(30e-6, cutoff_s=tau_c)
    import numpy as np

    from qutip_trap.readout.detection import PhotonRecord

    early = PhotonRecord(0, 30e-6, 1, arrivals_s=np.array([0.5 * tau_c]))
    late = PhotonRecord(0, 30e-6, 1, arrivals_s=np.array([2.0 * tau_c]))
    two = PhotonRecord(0, 30e-6, 2, arrivals_s=np.array([2.0 * tau_c, 2.5 * tau_c]))
    none = PhotonRecord(0, 30e-6, 0, arrivals_s=np.zeros(0))
    assert fp.decide(early, rm).bright and fp.decide(early, rm).time_used_s == pytest.approx(0.5 * tau_c)
    assert not fp.decide(late, rm).bright and fp.decide(late, rm).time_used_s == 30e-6
    assert fp.decide(two, rm).bright and fp.decide(two, rm).time_used_s == pytest.approx(2.5 * tau_c)
    assert not fp.decide(none, rm).bright
    assert tau_c == pytest.approx(math.log(341.0 / 4.2) / 472e3)
    with pytest.raises(ValueError):
        fp.decide(PhotonRecord(0, 30e-6, 3), rm)


# ---- camera (row "Camera") ----------------------------------------------------------------------------------------------------------


def _burrell_like() -> tuple[CameraGeometry, RecordModel]:
    det = Detector("camera", 0.010, 0.0, {}, None, None, 400e-6, numerical_aperture=0.25, pixel_m=2.6e-6)
    rates = two_state_rates(55_800.0 * 0.010 / 0.0019, 0.010, 0.0, 0.0).with_shelf(1.168)
    return CameraGeometry(
        tuple(np.arange(4) * 14e-6), 2.6e-6, 50, 10, 397e-9, psf_sigma_m=4.1e-6
    ), RecordModel.from_rates(rates, det)


def test_camera_eps_d_floor_is_t_exp_over_2_tau() -> None:
    """Without time resolution a shelf decay in the first half of the exposure reads bright: eps_D -> t_exp/(2 tau) = 1.7e-4 at
    400 us on the 1168 ms shelf as the bright count grows (Burrell; Section 9.5)."""
    det = Detector("camera", 0.010, 0.0, {}, None, None, 400e-6)
    for detected in (2e5, 1e6, 4e6):
        rm = RecordModel.from_rates(two_state_rates(detected, 0.010, 0.0, 0.0).with_shelf(1.168), det)
        pd = rm.count_distribution("shelf", 400e-6)
        pb = rm.count_distribution("bright", 400e-6)
        # the optimal threshold sits far from both means; take the midpoint
        n_c = math.floor(0.5 * pb.mean()) + 0.5
        eps_d = pd.probability_above(n_c)
        assert eps_d == pytest.approx(400e-6 / (2.0 * 1.168), rel=0.03 if detected > 1e6 else 0.15)
    assert 400e-6 / (2.0 * 1.168) == pytest.approx(1.7e-4, abs=0.02e-4)


def test_camera_time_resolution_gives_no_gain_with_coarse_exposures() -> None:
    """Burrell's 18 exposures of 200 us did not lower the error below the single 400 us exposure: with sub-bins of 200 us the
    spatio-temporal likelihood cannot place a decay better than the exposure, so the dark error stays at the t_exp/(2 tau)
    scale; with 10 us PMT sub-bins it drops (Myerson). Monte Carlo over 6000 shelf records."""
    det = Detector("pmt", 0.010, 200.0, {}, None, None, 400e-6)
    rm = RecordModel.from_rates(two_state_rates(4e6, 0.010, 0.0, 0.0).with_shelf(1.168), det)
    rng = np.random.default_rng(8)
    coarse = TimeResolvedML(200e-6, 400e-6, dark_class="shelf")
    fine = TimeResolvedML(10e-6, 400e-6, dark_class="shelf")
    n = 6000
    decay_times = rng.uniform(0.0, 400e-6, size=n)
    err_coarse = err_fine = 0
    for t_dec in decay_times:
        path = ClassPath(400e-6, "shelf", (float(t_dec),), ("bright",))
        rec = rm.sample_record("shelf", 400e-6, rng, sub_bin_s=10e-6, path=path)
        err_coarse += int(coarse.decide(rec, rm).bright)
        err_fine += int(fine.decide(rec, rm).bright)
    # conditioned on a decay inside the window: the coarse analysis misreads about half, the fine one far fewer
    assert 0.35 < err_coarse / n < 0.65
    assert err_fine / n < 0.2


def test_camera_spatial_and_neighbour_conditioned_decoding() -> None:
    geo, rm = _burrell_like()
    rng = np.random.default_rng(9)
    models = [rm] * 4
    n_trials = 60
    wrong_independent = wrong_neighbours = 0
    estimates: list[float] = []
    for _ in range(n_trials):
        truth = [bool(b) for b in rng.integers(0, 2, size=4)]
        paths = [ClassPath(400e-6, "bright" if b else "shelf", (), ()) for b in truth]
        img = sample_camera_image(geo, models, paths, 400e-6, rng, read_noise_counts=0.05)
        ind = decode_camera_image(
            img, geo, models, 400e-6, roi_pixels=60, neighbours=False, read_noise_counts=0.05
        )
        nb = decode_camera_image(
            img, geo, models, 400e-6, roi_pixels=60, neighbours=True, read_noise_counts=0.05
        )
        wrong_independent += sum(a != b for a, b in zip(ind.bright, truth))
        wrong_neighbours += sum(a != b for a, b in zip(nb.bright, truth))
        estimates.append(nb.register_error_estimate)
        assert len(nb.log_ratios) == 4 and nb.iterations >= 1
    assert wrong_neighbours <= wrong_independent
    assert wrong_neighbours == 0
    assert all(0.0 <= e < 1e-2 for e in estimates)
    # the neighbour-conditioned likelihood of a dark ion next to a bright one differs from the isolated one
    truth = [True, False, False, False]
    img = sample_camera_image(
        geo, models, [ClassPath(400e-6, "bright" if b else "shelf", (), ()) for b in truth], 400e-6, rng
    )
    roi = geo.roi(1, 60)
    lb0, ld0 = camera_log_likelihoods(img, geo, models, 400e-6, 1, roi, [False] * 4)
    lb1, ld1 = camera_log_likelihoods(img, geo, models, 400e-6, 1, roi, truth)
    assert ld1 > ld0
    flags = camera_threshold_decode(img, geo, 20, [30.0] * 4)
    assert flags[0] and not flags[3]


def test_spatio_temporal_recursion_matches_the_single_exposure_limit() -> None:
    """The decay mixture raises p_D when the LATE exposures look bright (a decay explains them) and lowers it slightly when
    every exposure looks dark (the weight (1 - M t_s/tau) < 1); without decay it is the plain product."""
    looks_dark = [(-3.0, -1.0), (-2.5, -1.2), (-3.2, -0.9)]
    lb, ld = spatio_temporal_log_likelihoods(looks_dark, 200e-6, 1.0 / 1.168)
    assert lb == pytest.approx(sum(p[0] for p in looks_dark))
    plain = sum(p[1] for p in looks_dark)
    assert plain - 1e-3 < ld < plain
    decayed = [(-3.0, -1.0), (-0.5, -8.0), (-0.4, -9.0)]
    _lb2, ld2 = spatio_temporal_log_likelihoods(decayed, 200e-6, 1.0 / 1.168)
    assert ld2 > sum(p[1] for p in decayed)
    lb0, ld0 = spatio_temporal_log_likelihoods(looks_dark, 200e-6, 0.0)
    assert ld0 == pytest.approx(plain)
    with pytest.raises(ValueError):
        spatio_temporal_log_likelihoods(looks_dark, 1.0, 1.0)
