"""The photon-record layer: the exact chain against the single-jump closed forms, sampled records, detector non-idealities,
neighbour-coupled records with both halves of Wineland's crosstalk, the camera PSF and the count-anomaly band (PLAN.md
Section 8.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.readout.detection import (
    CameraGeometry,
    ClassPath,
    Detector,
    RecordModel,
    apply_detector_nonidealities,
    count_anomaly_band,
    depumped_model,
    log_poisson_pmf,
    neighbourhood_model,
    sample_register_records,
)
from qutip_trap.readout.discriminate import ThresholdDiscriminator, register_confusion
from qutip_trap.readout.fluorescence import (
    neighbour_intensity_ratio,
    neighbour_pumping_rates,
    rates_from_detected,
)
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD
from tests.fixtures import YB_DIRECT, crain_record_model, myerson_record_model, two_state_rates

WINDOW_S = 22e-6
YB_WAVELENGTH_M = 369.5e-9
SPACING_M = 14e-6


def _single_jump_distribution(
    n_max: int, window_s: float, rate_before: float, rate_after: float, jump_rate: float
) -> np.ndarray:
    """Crain 2019 Eq. 1 with its no-jump branch restored: P(n; t) = e^{-R_T t} P_p(n; R_1 t) + int_0^t dtau R_T e^{-R_T tau}
    P_p(n; R_1 tau + R_2 (t - tau)), by Gauss-Legendre quadrature."""
    n = np.arange(n_max + 1)
    out = math.exp(-jump_rate * window_s) * np.exp(np.asarray(log_poisson_pmf(n, rate_before * window_s)))
    x, w = np.polynomial.legendre.leggauss(96)
    for t_k, w_k in zip(0.5 * window_s * (x + 1.0), 0.5 * window_s * w):
        mean = rate_before * t_k + rate_after * (window_s - t_k)
        out = out + w_k * jump_rate * math.exp(-jump_rate * t_k) * np.exp(
            np.asarray(log_poisson_pmf(n, mean))
        )
    return np.asarray(out)


def _zero_threshold_errors(t: float, r0: float, rd: float, rb: float, rbg: float) -> tuple[float, float]:
    """(eps_B, eps_D) of any-photon-is-bright to first order in the jumps (Crain 2019 Eqs. 2-3 with the detected rate in
    the no-jump exponent)."""
    k = r0 + rd
    no_jump = math.exp(-k * t)
    eps_b = math.exp(-rbg * t) * (no_jump + rd / k * (1.0 - no_jump))
    pumped = rb / (r0 - rb) * (math.exp(-rb * t) - math.exp(-r0 * t))
    return eps_b, 1.0 - math.exp(-rbg * t) * (math.exp(-rb * t) + pumped)


# ---- the exact chain and the sampled records -------------------------------------------------------------------------------


def test_exact_chain_reduces_to_the_single_jump_form_when_one_transition_acts() -> None:
    """With only R_d (bright start) the chain has one jump, so the matrix-exponential distribution equals the single-jump
    closed form to machine precision; the same for a dark start with only R_b."""
    det = Detector("snspd", 0.04356, 0.0, {}, None, None, 22e-6)
    bright_only = RecordModel.from_rates(two_state_rates(472e3, 0.04356, 341.0, 0.0), det)
    dist = bright_only.count_distribution("bright", 22e-6, n_max=60)
    assert np.max(np.abs(dist.pmf - _single_jump_distribution(60, 22e-6, 472e3, 0.0, 341.0))) < 1e-13
    dark_only = RecordModel.from_rates(two_state_rates(472e3, 0.04356, 0.0, 16.4), det)
    dist_d = dark_only.count_distribution("dark", 22e-6, n_max=60)
    assert np.max(np.abs(dist_d.pmf - _single_jump_distribution(60, 22e-6, 0.0, 472e3, 16.4))) < 1e-13
    assert dist.overflow < 1e-20 and dist_d.overflow < 1e-20


def test_exact_chain_agrees_with_the_zero_threshold_closed_forms_to_first_order() -> None:
    """The closed forms neglect re-pumping after a jump; the exact chain differs from them by that second-order amount only
    (1e-3 relative at 22 us)."""
    rm = crain_record_model()
    eps_b, eps_d = _zero_threshold_errors(22e-6, 472e3, 341.0, 16.4, 4.2)
    assert rm.count_distribution("bright", 22e-6).pmf[0] == pytest.approx(eps_b, rel=1e-3)
    assert rm.count_distribution("dark", 22e-6).probability_above(0.5) == pytest.approx(eps_d, rel=1e-3)
    assert rm.mean_counts("bright", 22e-6) == pytest.approx(472e3 * 22e-6, rel=5e-3)


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
    """A 100 ms window on Crain's 171Yb+ chain pumps dark in 1/R_d = 2.9 ms and back in 1/R_b = 61 ms, so the sampled paths
    carry a few cycles and the exact mean count sits well above the single-jump (no return) one."""
    rm = crain_record_model(window_s=100e-3)
    rng = np.random.default_rng(5)
    jumps = [len(rm.sample_path("bright", 100e-3, rng).jump_times_s) for _ in range(200)]
    assert 2.5 < np.mean(jumps) < 8.0
    single = _single_jump_distribution(int(472e3 * 100e-3 * 1.2), 100e-3, 472e3 + 4.2, 4.2, 341.0)
    assert rm.mean_counts("bright", 100e-3) > 1.2 * float(np.dot(np.arange(single.size), single))


def test_class_path_segments_and_the_bright_occupancy() -> None:
    path = ClassPath(1.0, "bright", (0.25, 0.75), ("dark", "bright"))
    assert path.segments() == [(0.0, 0.25, "bright"), (0.25, 0.75, "dark"), (0.75, 1.0, "bright")]
    with pytest.raises(ValueError):
        ClassPath(1.0, "bright", (0.75, 0.25), ("dark", "bright"))
    occ = crain_record_model().bright_occupancy_mean_s("bright", 22e-6)
    assert 0.99 * 22e-6 < occ < 22e-6


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
    assert np.mean(totals) == pytest.approx(472e3 / (1.0 + 472e3 * 2e-6) * 1e-3, rel=0.03)


def test_count_anomaly_band_brackets_both_hypotheses() -> None:
    """A total outside the [1e-6, 1 - 1e-6] band of both the bright and the dark count distribution is explained by
    neither hypothesis, so the run flags it."""
    rm = crain_record_model(window_s=22e-6)
    lo, hi = count_anomaly_band(rm, 22e-6)
    bright = rm.count_distribution("bright", 22e-6)
    dark = rm.count_distribution("dark", 22e-6)
    assert lo == 0, "the dark hypothesis explains zero counts"
    assert hi > bright.mean() > 0.0
    n = np.arange(len(bright.pmf))
    for dist in (bright, dark):
        assert float(np.sum(dist.pmf[(n >= lo) & (n <= hi)])) > 1.0 - 3e-6, dist.mean()
    # a shelving scheme's non-bright hypothesis is the shelf
    my = myerson_record_model()
    lo_s, hi_s = count_anomaly_band(my, 420e-6, ("bright", "shelf"))
    assert lo_s == 0 and hi_s > my.mean_counts("bright", 420e-6)
    assert count_anomaly_band(rm, 22e-6, quantile=1e-9)[1] >= hi
    with pytest.raises(ValueError, match="quantile"):
        count_anomaly_band(rm, 22e-6, quantile=0.6)


# ---- neighbour-coupled records (Section 8.5) ----------------------------------------------------------------------------------


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
    assert sample_register_records([rm, rm], ["dark", "bright"], 22e-6, rngs, leakage={})[0].ion == 0


def test_the_pumping_a_bright_neighbour_drives_is_the_bound_times_the_ions_own_rates() -> None:
    """R_d and R_b are linear in intensity, so the leaked light drives them at s_neighbour/s_beam, s_neighbour the bound
    3 lambda^2/(8 pi^2 x^2); at 14 um it adds a few mHz to Crain's 341 Hz."""
    rates = CRAIN_YB171_SNSPD.rates()
    s_beam = 2.45
    s_nb = neighbour_intensity_ratio(YB_WAVELENGTH_M, SPACING_M)
    assert s_nb == pytest.approx(3.0 * YB_WAVELENGTH_M**2 / (8.0 * math.pi**2 * SPACING_M**2), rel=1e-12)
    d_d, d_b = neighbour_pumping_rates(rates, s_beam, YB_WAVELENGTH_M, SPACING_M)
    scale = s_nb / s_beam
    assert d_d == pytest.approx(rates.R_dark_pumping_per_s * scale, rel=1e-12)
    assert d_b == pytest.approx(rates.R_bright_pumping_per_s * scale, rel=1e-12)
    assert 1e-6 < scale < 1e-4
    half = neighbour_pumping_rates(rates, s_beam, YB_WAVELENGTH_M, SPACING_M, polarization_purity=0.5)
    assert half[0] == pytest.approx(0.5 * d_d, rel=1e-12)
    with pytest.raises(ValueError, match="polarization share"):
        neighbour_pumping_rates(rates, s_beam, YB_WAVELENGTH_M, SPACING_M, polarization_purity=1.5)
    with pytest.raises(ValueError, match="saturation parameter"):
        neighbour_pumping_rates(rates, 0.0, YB_WAVELENGTH_M, SPACING_M)
    # measured rates feed the bound without a species constant, and it falls as 1/x^2
    ingested = rates_from_detected(472e3, 0.04356, dark_pumping_per_s=341.0, bright_pumping_per_s=16.4)
    d_d, d_b = neighbour_pumping_rates(ingested, 2.45, YB_WAVELENGTH_M, SPACING_M)
    assert d_d / d_b == pytest.approx(341.0 / 16.4, rel=1e-12)
    far = neighbour_pumping_rates(ingested, 2.45, YB_WAVELENGTH_M, 2.0 * SPACING_M)
    assert far[0] == pytest.approx(d_d / 4.0, rel=1e-12)


def test_depumped_model_touches_only_the_pumping_channels_and_only_for_bright_neighbours() -> None:
    rm = crain_record_model()
    extra = {1: (5.0e3, 2.0e2)}
    beside_bright = depumped_model([rm, rm], 0, ["dark", "bright"], extra)
    assert depumped_model([rm, rm], 0, ["dark", "dark"], extra) is rm, "a dark neighbour pumps nothing"
    assert beside_bright.detected_bright_per_s == rm.detected_bright_per_s
    assert beside_bright.background_per_s == rm.background_per_s, "the added counts are the other half"
    assert beside_bright.rates[("bright", "dark")] == pytest.approx(
        rm.rates[("bright", "dark")] + 5.0e3, rel=1e-12
    )
    assert beside_bright.rates[("dark", "bright")] == pytest.approx(
        rm.rates[("dark", "bright")] + 2.0e2, rel=1e-12
    )
    # several bright neighbours add, because the rates are linear in intensity
    three = depumped_model([rm] * 3, 1, ["bright", "dark", "bright"], extra)
    assert three.rates[("bright", "dark")] == pytest.approx(
        rm.rates[("bright", "dark")] + 2.0 * 5.0e3, rel=1e-12
    )
    assert depumped_model([rm, rm], 0, ["dark", "bright"], None) is rm


def test_a_bright_ion_beside_a_bright_one_loses_photons_the_depumping_reduced_n() -> None:
    """With the neighbour bright the ion is pumped dark during the window: dR_d t = 0.11 costs about 5 % of the mean count,
    and the threshold error grows."""
    rm = crain_record_model()
    extra = {1: (5.0e3, 0.0)}
    alone = neighbourhood_model([rm, rm], 0, ["bright", "dark"], {}, extra)
    beside = neighbourhood_model([rm, rm], 0, ["bright", "bright"], {}, extra)
    n_alone = alone.mean_counts("bright", WINDOW_S)
    n_beside = beside.mean_counts("bright", WINDOW_S)
    assert n_beside < n_alone
    assert n_beside / n_alone == pytest.approx(1.0 - 0.5 * 5.0e3 * WINDOW_S, rel=2e-2)
    disc = ThresholdDiscriminator(0.5, WINDOW_S)
    assert disc.error_rates(beside)[0] > disc.error_rates(alone)[0]


def test_the_register_confusion_and_the_sampled_records_both_carry_the_depumping() -> None:
    rm = crain_record_model()
    schemes = [YB_DIRECT, YB_DIRECT]
    extra = {1: (5.0e3, 0.0)}
    plain = register_confusion([rm, rm], schemes, ThresholdDiscriminator(0.5, WINDOW_S), {1: 0.04})
    pumped = register_confusion(
        [rm, rm], schemes, ThresholdDiscriminator(0.5, WINDOW_S), {1: 0.04}, depumping=extra
    )
    assert plain.factored is not None and pumped.factored is not None
    # ion 0 in the bright level (|1> for 171Yb+) beside a bright neighbour: declared bright less often once depumped
    assert float(pumped.factored.tables[0][1, 0, 0]) < float(plain.factored.tables[0][1, 0, 0])
    totals_bright_neighbour = []
    totals_dark_neighbour = []
    for s in range(4000):
        rngs = [np.random.default_rng((s, i, 17)) for i in range(2)]
        recs = sample_register_records(
            [rm, rm], ["bright", "bright"], WINDOW_S, rngs, leakage={}, depumping=extra
        )
        totals_bright_neighbour.append(recs[0].total)
        rngs = [np.random.default_rng((s, i, 17)) for i in range(2)]
        recs = sample_register_records(
            [rm, rm], ["bright", "dark"], WINDOW_S, rngs, leakage={}, depumping=extra
        )
        totals_dark_neighbour.append(recs[0].total)
    beside = neighbourhood_model([rm, rm], 0, ["bright", "bright"], {}, extra)
    alone = neighbourhood_model([rm, rm], 0, ["bright", "dark"], {}, extra)
    mean_b = float(np.mean(totals_bright_neighbour))
    mean_d = float(np.mean(totals_dark_neighbour))
    assert mean_b == pytest.approx(beside.mean_counts("bright", WINDOW_S), rel=0.05)
    assert mean_d == pytest.approx(alone.mean_counts("bright", WINDOW_S), rel=0.05)
    assert mean_b < mean_d


# ---- the camera point-spread function (Section 8.3) ---------------------------------------------------------------------------


def test_camera_psf_leakage_from_the_airy_pattern_and_an_aberrated_gaussian() -> None:
    """A diffraction-limited NA 0.25 system at 397 nm leaks under 0.5 % into a 28-pixel neighbour ROI at 14 um; Burrell's
    measured 4.0 % nearest-neighbour signal (relative to the ion's own) belongs to an aberrated PSF, which a Gaussian of
    sigma 4.1 um reproduces, collecting 0.757 of the ion's own light in an ROI of one spacing's diameter."""

    def leakage(geo: CameraGeometry, ion: int, other: int, n_pixels: int) -> float:
        roi = geo.roi(ion, n_pixels)
        return float(geo.weights(other).ravel()[roi].sum() / geo.weights(ion).ravel()[roi].sum())

    positions = tuple(np.arange(4) * 14e-6)
    airy = CameraGeometry(positions, 2.6e-6, 50, 10, 397e-9, numerical_aperture=0.25)
    w = airy.weights(1)
    assert w.shape == (10, 50) and w.sum() == pytest.approx(1.0, abs=0.02)
    assert leakage(airy, 1, 2, 28) < 0.005
    assert leakage(airy, 1, 2, 10) < leakage(airy, 1, 2, 60)
    roi_one_spacing = int(round(math.pi * 7.0**2 / 2.6**2))
    gauss = CameraGeometry(positions, 2.6e-6, 50, 10, 397e-9, psf_sigma_m=4.1e-6)
    assert leakage(gauss, 1, 2, roi_one_spacing) == pytest.approx(0.040, abs=0.006)
    assert leakage(gauss, 1, 3, roi_one_spacing) < 0.001
    own = float(gauss.weights(1).ravel()[gauss.roi(1, roi_one_spacing)].sum())
    assert own == pytest.approx(0.757, abs=0.02)
    with pytest.raises(ValueError):
        CameraGeometry((0.0,), 2.6e-6, 10, 10, 397e-9)
