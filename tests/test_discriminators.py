"""The discriminator layer and the detection calibration: Myerson's recursion, optimum and maximum-likelihood asymptote, the
adaptive method, Noek's and Crain's first-photon protocols, the camera-exposure limits, and the calibration that picks
(n_c, t_b) (PLAN.md Section 8.3)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.calibration.readout import calibrate_detection, histogram_error_rates
from qutip_trap.device.presets import crain_snspd_detector
from qutip_trap.experiments.readout import detection_histogram
from qutip_trap.machine import Machine
from qutip_trap.readout.detection import ClassPath, Detector, PhotonRecord, RecordModel, log_poisson_pmf
from qutip_trap.readout.discriminate import (
    AdaptiveML,
    FirstPhoton,
    ThresholdDiscriminator,
    TimeResolvedML,
    exact_log_likelihoods,
    myerson_log_likelihoods,
    optimize_threshold,
)
from tests.fixtures import (
    CA_OPTICAL,
    GAMMA_S,
    YB_DIRECT,
    chain_device,
    crain_record_model,
    myerson_record_model,
    two_state_rates,
    yb_detection_beam,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _myerson_direct_sum(counts: np.ndarray, sub_bin_s: float, model: RecordModel) -> tuple[float, float]:
    """(p_B, p_D) of Myerson Eq. 1 summed directly in the linear domain, the O(N^2) form the recursion evaluates."""
    b = np.exp(
        np.asarray(
            log_poisson_pmf(counts, (model.detected_bright_per_s + model.background_per_s) * sub_bin_s)
        )
    )
    d = np.exp(np.asarray(log_poisson_pmf(counts, model.background_per_s * sub_bin_s)))
    rate = model.rates.get(("shelf", "bright"), 0.0)
    p_d = (1.0 - rate * sub_bin_s * counts.size) * float(np.prod(d))
    for j in range(counts.size):
        p_d += rate * sub_bin_s * float(np.prod(d[:j])) * float(np.prod(b[j:]))
    return float(np.prod(b)), p_d


# ---- Myerson 2008 ----------------------------------------------------------------------------------------------------------


def test_myerson_recursion_equals_the_direct_sum_and_the_weights_normalize() -> None:
    rm = myerson_record_model()
    rng = np.random.default_rng(1)
    for start in ("bright", "shelf"):
        for _ in range(5):
            rec = rm.sample_record(start, 420e-6, rng, sub_bin_s=10e-6)
            assert rec.sub_bins is not None
            lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            pb, pd = _myerson_direct_sum(rec.sub_bins, 10e-6, rm)
            assert lb == pytest.approx(math.log(pb), abs=1e-10)
            assert ld == pytest.approx(math.log(pd), abs=1e-10)
    # the mixture weights (1 - t_b/tau) + N (t_s/tau) = 1 exactly: with B = D the dark likelihood equals the bright one
    lam = (rm.detected_bright_per_s + rm.detected_bright_per_s + 442.0) * 10e-6
    flat_b = RecordModel(1e-9, lam / 10e-6 - 1e-9, rm.rates)
    lb, ld = myerson_log_likelihoods(np.array([20, 25, 22, 19]), 10e-6, flat_b, dark_class="shelf")
    assert lb == pytest.approx(ld, abs=1e-9)


def test_the_log_domain_survives_the_227_sub_bin_underflow() -> None:
    """M_N underflows near N ~ 227 sub-bins at the paper's rates: the log-domain recursion stays finite for long records."""
    rm = myerson_record_model()
    rng = np.random.default_rng(2)
    for window_s in (2.27e-3, 6.0e-3):
        rec = rm.sample_record("bright", window_s, rng, sub_bin_s=10e-6)
        assert rec.sub_bins is not None
        lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
        assert math.isfinite(lb) and math.isfinite(ld) and lb > ld


def test_exact_forward_likelihood_agrees_with_myerson_to_first_order_in_t_over_tau() -> None:
    """Records without a jump: the exact hidden-Markov likelihood equals Myerson's first-order form to 1e-5 in log. Records
    with a decay inside the window: Myerson puts it at the start of its sub-bin while the exact kernel integrates over its
    position, so ln p_D differs by up to about a nat with the same decision away from the boundary."""
    rm = myerson_record_model()
    rng = np.random.default_rng(3)
    for start in ("bright", "shelf"):
        for _ in range(40):
            rec = rm.sample_record(start, 420e-6, rng, sub_bin_s=10e-6)
            assert rec.sub_bins is not None and rec.path is not None
            lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            eb, ed = exact_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
            jumps = len(rec.path.jump_times_s)
            tight = start == "bright" and jumps == 0
            assert eb == pytest.approx(lb, abs=1e-5 if tight or start == "shelf" else 0.2)
            if jumps == 0 and start == "shelf":
                assert ed == pytest.approx(ld, abs=1e-5)
            else:
                assert ed == pytest.approx(ld, abs=1.5)
                if abs(lb - ld) > 2.0:
                    assert (eb >= ed) == (lb >= ld)
    late = ClassPath(420e-6, "shelf", (150e-6,), ("bright",))
    rec = rm.sample_record("shelf", 420e-6, rng, sub_bin_s=10e-6, path=late)
    assert rec.sub_bins is not None
    _lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
    _eb, ed = exact_log_likelihoods(rec.sub_bins, 10e-6, rm, dark_class="shelf")
    assert ed == pytest.approx(ld, abs=1.5) and abs(ed - ld) > 1e-4


def test_ca40_threshold_optimum_from_the_ideal_poisson_model() -> None:
    """The exact chain's average error has an interior optimum in (n_c, t_b): 1.24e-4 at (3.5, 320 us); at Myerson's
    operating point (5.5, 420 us) it gives 1.37e-4 against the measured 1.8(1)e-4, whose dark error the paper attributes
    about 20 % to cosmic rays the ideal model has not."""
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
    """Myerson: the ML method tends to 0.87(11)e-4 with eps_D = 1.5(2)e-4, the bright state detected to eps_B < 2e-6;
    Monte Carlo over 30 000 records per state at t_b = 1 ms."""
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
    """A shelf decay 200 us into a 420 us window leaves 12 bright counts: the threshold calls it bright, the ML weighs the
    20 empty sub-bins against the decay prior and catches most; at 300 us it catches nearly all."""
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
    t_b = float(np.mean([d.time_used_s for d in dec_b]))
    t_d = float(np.mean([d.time_used_s for d in dec_d]))
    assert 40e-6 < t_b < 90e-6 and 140e-6 < t_d < 260e-6
    assert 0.5 * (t_b + t_d) < 200e-6
    # the cutoff is an error probability: bright records stopped early as dark occur at about the cutoff rate
    assert np.mean([not d.bright for d in dec_b]) < 1e-3
    assert np.mean([d.bright for d in dec_d]) < 3e-3
    assert all(d.time_used_s <= 500e-6 for d in dec_b + dec_d)


# ---- first-photon protocols ------------------------------------------------------------------------------------------------


def test_crain_stop_on_first_photon_reaches_the_closed_form_errors_and_12us_average() -> None:
    rm = crain_record_model()
    fp = FirstPhoton(22e-6)
    rng = np.random.default_rng(7)
    n = 20_000
    dec_b = [fp.decide(rm.sample_record("bright", 22e-6, rng, arrivals=True), rm) for _ in range(n)]
    dec_d = [fp.decide(rm.sample_record("dark", 22e-6, rng, arrivals=True), rm) for _ in range(n)]
    assert np.mean([not d.bright for d in dec_b]) == pytest.approx(7.5e-4, abs=3e-4)
    assert np.mean([d.bright for d in dec_d]) == pytest.approx(4.2e-4, abs=2e-4)
    t_b = float(np.mean([d.time_used_s for d in dec_b]))
    t_d = float(np.mean([d.time_used_s for d in dec_d]))
    assert 0.5 * (t_b + t_d) == pytest.approx(12e-6, abs=1.5e-6)
    assert t_b == pytest.approx(1.0 / 472e3, rel=0.1)


def test_noek_two_photon_rule_uses_the_cutoff_time() -> None:
    """tau_c = ln(R_d/R_dc)/(eps R_o) (Noek 2013 Eq. 6): a lone photon before it is bright, after it dark."""
    rm = crain_record_model()
    tau_c = math.log(341.0 / 4.2) / 472e3
    fp = FirstPhoton(30e-6, cutoff_s=tau_c)
    early = PhotonRecord(0, 30e-6, 1, arrivals_s=np.array([0.5 * tau_c]))
    late = PhotonRecord(0, 30e-6, 1, arrivals_s=np.array([2.0 * tau_c]))
    two = PhotonRecord(0, 30e-6, 2, arrivals_s=np.array([2.0 * tau_c, 2.5 * tau_c]))
    none = PhotonRecord(0, 30e-6, 0, arrivals_s=np.zeros(0))
    assert fp.decide(early, rm).bright and fp.decide(early, rm).time_used_s == pytest.approx(0.5 * tau_c)
    assert not fp.decide(late, rm).bright and fp.decide(late, rm).time_used_s == 30e-6
    assert fp.decide(two, rm).bright and fp.decide(two, rm).time_used_s == pytest.approx(2.5 * tau_c)
    assert not fp.decide(none, rm).bright
    with pytest.raises(ValueError):
        fp.decide(PhotonRecord(0, 30e-6, 3), rm)


# ---- a camera exposure ---------------------------------------------------------------------------------------------------------


def test_camera_eps_d_floor_is_t_exp_over_2_tau() -> None:
    """Without time resolution a shelf decay in the first half of the exposure reads bright: eps_D -> t_exp/(2 tau) =
    1.7e-4 at 400 us on the 1168 ms shelf as the bright count grows (Burrell)."""
    det = Detector("camera", 0.010, 0.0, {}, None, None, 400e-6)
    for detected in (2e5, 1e6, 4e6):
        rm = RecordModel.from_rates(two_state_rates(detected, 0.010, 0.0, 0.0).with_shelf(1.168), det)
        pd = rm.count_distribution("shelf", 400e-6)
        pb = rm.count_distribution("bright", 400e-6)
        n_c = math.floor(0.5 * pb.mean()) + 0.5
        assert pd.probability_above(n_c) == pytest.approx(
            400e-6 / (2.0 * 1.168), rel=0.03 if detected > 1e6 else 0.15
        )


def test_coarse_exposures_give_no_time_resolution_gain() -> None:
    """With 200 us sub-bins the likelihood cannot place a decay better than the exposure, so the dark error of records with
    a decay inside the window stays near one half; 10 us sub-bins catch most (Burrell against Myerson)."""
    det = Detector("pmt", 0.010, 200.0, {}, None, None, 400e-6)
    rm = RecordModel.from_rates(two_state_rates(4e6, 0.010, 0.0, 0.0).with_shelf(1.168), det)
    rng = np.random.default_rng(8)
    coarse = TimeResolvedML(200e-6, 400e-6, dark_class="shelf")
    fine = TimeResolvedML(10e-6, 400e-6, dark_class="shelf")
    n = 6000
    err_coarse = err_fine = 0
    for t_dec in rng.uniform(0.0, 400e-6, size=n):
        path = ClassPath(400e-6, "shelf", (float(t_dec),), ("bright",))
        rec = rm.sample_record("shelf", 400e-6, rng, sub_bin_s=10e-6, path=path)
        err_coarse += int(coarse.decide(rec, rm).bright)
        err_fine += int(fine.decide(rec, rm).bright)
    assert 0.35 < err_coarse / n < 0.65
    assert err_fine / n < 0.2


# ---- the detection calibration (Section 7.5 item 5) ---------------------------------------------------------------------------


def test_calibrate_detection_recovers_the_rates_and_picks_the_interior_optimum() -> None:
    """10^4 bright and dark records at Crain's operating point: the mean-count fit recovers eps R_o to 1 % and R_d within its
    uncertainty, and the chosen (n_c, t_b) sits at the fitted model's interior optimum."""
    rm = crain_record_model(window_s=60e-6)
    windows = tuple(float(x) for x in np.linspace(6e-6, 60e-6, 10))
    cal = calibrate_detection(
        rm, YB_DIRECT, windows_s=windows, n_records=10_000, rng=np.random.default_rng(0)
    )
    e = cal.entries
    assert e["R_bright_detected_per_s"].value == pytest.approx(472e3, rel=0.01)
    assert e["R_dark_pumping_per_s"].value == pytest.approx(
        341.0, abs=4.0 * max(e["R_dark_pumping_per_s"].uncertainty, 60.0)
    )
    assert e["threshold"].value == 0.5
    assert 15e-6 <= e["window_s"].value <= 30e-6
    assert e["eps_B"].status == "calibrated" and e["eps_B"].experiment == "detection_histogram"
    assert 0.5 * (e["eps_B"].value + e["eps_D"].value) == pytest.approx(5.9e-4, abs=0.6e-4)
    assert cal.povm.per_ion is not None and cal.discriminator.n_c == 0.5
    # the histograms report the laboratory's own (eps_B, eps_D) beside the model's, agreeing to counting statistics
    lab_eps_b, lab_eps_d = cal.histogram_errors
    assert lab_eps_b == pytest.approx(e["eps_B"].value, abs=4.0 * e["eps_B"].uncertainty + 1e-4)
    assert lab_eps_d == pytest.approx(e["eps_D"].value, abs=4.0 * e["eps_D"].uncertainty + 1e-4)
    assert histogram_error_rates(np.array([0, 1, 5, 7]), np.array([0, 0, 1, 0]), 0.5) == (0.25, 0.25)
    assert cal.bright_histogram.sum() == 10_000 and cal.dark_histogram.sum() == 10_000


def test_calibrate_detection_on_the_shelving_scheme_reports_the_shelf_start() -> None:
    cal = calibrate_detection(
        myerson_record_model(),
        CA_OPTICAL,
        windows_s=(200e-6, 300e-6, 400e-6, 500e-6, 700e-6),
        n_records=2000,
        rng=np.random.default_rng(1),
    )
    assert cal.entries["R_dark_pumping_per_s"].value < 50.0
    assert 250e-6 <= cal.entries["window_s"].value <= 500e-6
    assert cal.optimum.best.eps < 3e-4


@pytest.mark.slow
def test_detection_histogram_experiment_runs_the_bloch_model_of_the_device_beams() -> None:
    """The experiment finds the 369.5 nm beam of the device, solves the rates at the ion's position and returns histograms,
    the threshold, the window and the fitted rates; the scattered rate is the Bloch solve's, below Gamma/4."""
    base = chain_device(2)
    dev = dataclasses.replace(
        base,
        beams=base.beams + (yb_detection_beam(2.45, base.field.B_gauss),),
        detector=crain_snspd_detector(),
    )
    res = detection_histogram(Machine(dev), 0, 2000, windows_s=tuple(np.linspace(10e-6, 60e-6, 6)), seed=0)
    assert res.model == "detection_histogram" and res.data.shape[0] == 2
    assert res.data[0].sum() == 2000 and res.data[1].sum() == 2000
    fitted = res.fitted
    assert 0.0 < fitted["R_bright_scattered_per_s"][0] < GAMMA_S / 4.0
    assert fitted["R_bright_detected_per_s"][0] == pytest.approx(
        dev.detector.efficiency * fitted["R_bright_scattered_per_s"][0], rel=0.05
    )
    assert fitted["threshold"][0] >= 0.5 and 10e-6 <= fitted["window_s"][0] <= 60e-6
    with pytest.raises(ValueError):
        detection_histogram(Machine(chain_device(2)), 0, 500)
