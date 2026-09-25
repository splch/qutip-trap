"""The Section 5.5 periodogram test and the tau_c/10 grid rule (PLAN.md Section 5.5 "a test compares the realized
coefficient's periodogram with the configured spectrum over the band the pulse is sensitive to"; Sections 6.1, 6.3;
M7 audit E-6, E-12).

The audit found no periodogram anywhere (``grep -rni "periodogram|welch|rfft"`` returned nothing) and the grid step
7.9x coarser than the plan's ``Delta t <= tau_c/10``, with the constant ``tau_c/10`` appearing nowhere in the
repository. Both are pinned here, in the module's own two-sided angular convention: a laboratory single-sided
per-hertz density is ``S_1(f) = 2 S(omega = 2 pi f)``, so a periodogram estimate of ``S_1`` halves into ``S``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import welch

from qutip_trap.noise.processes import (
    MIN_GRID_POINTS,
    TAU_C_OVERSAMPLE,
    mains_trajectory,
    synthesize,
    time_grid,
)
from qutip_trap.noise.spectra import Mains, ou_spectrum


def test_the_grid_honours_delta_t_at_most_tau_c_over_ten() -> None:
    """Section 5.5's rule, applied to the band's FASTEST component (tau_c,min = 1/omega_max): dt <= 1/(10 omega_max),
    which ``time_grid``'s dt = pi/(omega_max x oversample) meets at oversample >= 10 pi = 31.4159. The retired default of
    4 gave dt = 0.785/omega_max, 7.9x too coarse."""
    assert TAU_C_OVERSAMPLE == pytest.approx(10.0 * math.pi)
    for w_max in (1e2, 1e4, 2.0 * math.pi * 3e3):
        tau_c_min = 1.0 / w_max
        t = time_grid(1e-2, w_max)
        dt = float(np.max(np.diff(t)))
        assert dt <= tau_c_min / 10.0 * (1.0 + 1e-12), (w_max, dt, tau_c_min / 10.0)
    # the retired default violates the rule by exactly 10 pi/4 = 7.854, wherever the dt formula binds rather than the
    # MIN_GRID_POINTS floor (at w_max = 1e2 over 10 ms the floor already over-resolves by 6.4x)
    for w_max, duration in ((1e4, 1e-2), (2.0 * math.pi * 3e3, 1e-2), (1e6, 1e-3)):
        tau_c_min = 1.0 / w_max
        dt_old = float(np.max(np.diff(time_grid(duration, w_max, oversample=4.0))))
        # 2 % tolerance: time_grid rounds the point count up, so the realized dt is duration/(n - 1), a little below
        # the nominal pi/(omega_max x oversample)
        assert dt_old / (tau_c_min / 10.0) == pytest.approx(10.0 * math.pi / 4.0, rel=2e-2), w_max
    # a band with no content still gets a smooth curve over the shot
    assert time_grid(1e-3, 0.0).size == MIN_GRID_POINTS
    with pytest.raises(ValueError):
        time_grid(0.0, 1e3)


def test_the_noise_model_grid_default_is_the_tau_c_rule() -> None:
    """``NoiseModel.grid_oversample`` carries the rule, so every synthesized trajectory in the simulator obeys it."""
    from tests.fixtures import make_noise

    assert make_noise().grid_oversample == pytest.approx(TAU_C_OVERSAMPLE)


def _periodogram_two_sided(values: np.ndarray, dt: float, n_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch's estimate of the SINGLE-sided per-hertz density, halved into the module's two-sided angular convention.

    ``welch`` with ``fs = 1/dt`` returns S_1(f) in units^2/Hz; the module's spectra are S(omega) = S_1(f)/2 at
    omega = 2 pi f (Section 13, no further 2 pi).
    """
    f, pxx = welch(values, fs=1.0 / dt, nperseg=max(64, values.size // n_seg), scaling="density")
    return 2.0 * math.pi * f, pxx / 2.0


@pytest.mark.slow
def test_the_realized_periodogram_reproduces_the_configured_ou_spectrum() -> None:
    """Section 5.5's own test: synthesize long realizations of a Lorentzian band and compare their periodogram with
    ``spectrum.tabulated`` in the two-sided angular convention, over the band the trajectory is sensitive to.

    Averaged over 16 seeds the mean ratio is 1 to better than 15 % from 0.4/tau_c to 6/tau_c, and the estimate rolls off
    WITH the Lorentzian rather than flattening - which is what a cubic-spline interpolation of a rough sample path would
    do, and is Section 5.5's stated reason for ``order=1``.

    Cost note: ``synthesize`` forms an outer product of the grid against its 512 spectral bins, so a 180k-point grid
    costs 92M elements per seed; the grid here is kept near 40k (a 20/tau_c band top over 200 tau_c) and the bin count at
    128, which is why the band is compared over one decade rather than two."""
    tau_c = 1e-3
    sp = ou_spectrum(4e-4, tau_c, "u^2/(rad/s)", omega_max_rad_s=20.0 / tau_c)
    t = time_grid(200.0 * tau_c, sp.omega_max_rad_s)
    dt = float(t[1] - t[0])
    assert dt <= tau_c / 10.0, "and the realization itself obeys the rule the first test pins"
    acc = None
    for seed in range(16):
        vals = synthesize(sp, t, np.random.default_rng(seed), n_bins=128).values
        w, pxx = _periodogram_two_sided(vals, dt, 6)
        acc = pxx if acc is None else acc + pxx
    assert acc is not None
    est = acc / 16.0
    band = (w >= 0.4 / tau_c) & (w <= 6.0 / tau_c)
    assert band.sum() > 8, band.sum()
    ratio = est[band] / np.asarray(sp.tabulated(w[band]), dtype=float)
    assert float(np.mean(ratio)) == pytest.approx(1.0, rel=0.15), float(np.mean(ratio))
    assert float(np.median(ratio)) == pytest.approx(1.0, rel=0.15), float(np.median(ratio))
    # a periodogram bin is chi^2-distributed, so the per-bin scatter is the estimator's and not the process's: assert a
    # ROBUST statistic (most bins within 50 %) rather than the maximum, which a single bin can dominate
    assert float(np.mean(np.abs(ratio - 1.0) < 0.5)) > 0.8, sorted(np.round(ratio, 3).tolist())
    # the realization's variance is the spectrum's, independently of the periodogram
    var = float(np.mean(synthesize(sp, t, np.random.default_rng(11), n_bins=128).values ** 2))
    assert var == pytest.approx(sp.variance(), rel=0.3)
    # the roll-off is real: a decade above the corner the density has fallen by about 100
    lo = est[(w >= 0.4 / tau_c) & (w <= 0.6 / tau_c)]
    hi = est[(w >= 8.0 / tau_c) & (w <= 12.0 / tau_c)]
    assert lo.size and hi.size and float(np.mean(lo)) / float(np.mean(hi)) > 20.0


def test_the_mains_harmonics_appear_as_lines_in_the_periodogram() -> None:
    """Section 6.3's deterministic periodic component: the mains trajectory's periodogram is a comb at h f_line, and the
    line powers follow the configured amplitudes as A_h^2/2 (the mean square of a cosine)."""
    line = 50.0
    amps = {1: 3e-9, 2: 1e-9, 3: 5e-10}
    mains = Mains(line, amps, dict.fromkeys(amps, 0.0), "line_triggered")
    duration = 2.0
    t = np.linspace(0.0, duration, 200001)
    dt = float(t[1] - t[0])
    values = mains_trajectory(mains, t, 0.0).values
    f, pxx = welch(values, fs=1.0 / dt, nperseg=100000, scaling="spectrum")
    for h, amp in amps.items():
        k = int(np.argmin(np.abs(f - h * line)))
        assert abs(f[k] - h * line) < 1.0, (h, f[k])
        assert pxx[k] == pytest.approx(amp**2 / 2.0, rel=0.05), (h, pxx[k], amp**2 / 2.0)
    # nothing between the harmonics
    between = pxx[(f > 1.2 * line) & (f < 1.8 * line)]
    assert float(np.max(between)) < 1e-6 * (amps[1] ** 2 / 2.0)
    # a free-running trigger phase moves the lines' phase, not their power
    shifted = mains_trajectory(mains, t, 1.1).values
    _f2, pxx2 = welch(shifted, fs=1.0 / dt, nperseg=100000, scaling="spectrum")
    k1 = int(np.argmin(np.abs(f - line)))
    assert pxx2[k1] == pytest.approx(pxx[k1], rel=0.05)


def test_noise_rates_carry_provenance_and_the_model_says_how_many_apparatus() -> None:
    """Section 6.1 **[extracted]**: "An N-ion error budget assembled from them is stitched from at least three apparatus,
    and the report says so". A zero rate contributes no apparatus (it contributes nothing to a budget); a non-zero rate
    with no tag is counted and named as undeclared."""
    import dataclasses

    from qutip_trap.noise.spectra import Drift, white_spectrum
    from tests.fixtures import make_noise

    quiet = make_noise()
    assert quiet.apparatus() == () or isinstance(quiet.apparatus(), tuple)
    s_b = white_spectrum(1e-24, "T^2/(rad/s)")
    assert s_b.apparatus_count == 0 and s_b.provenance == ()
    tagged = dataclasses.replace(s_b, provenance=("Fang 2022, 171Yb+, 5 ions",))
    assert tagged.apparatus_count == 1
    drift = Drift(1e-3, 1.0, None, provenance=("Cetina 2022, 171Yb+, 15 ions",))
    assert drift.apparatus_count == 1
    model = dataclasses.replace(
        quiet,
        S_B=tagged,
        rabi_drift=drift,
        laser_intensity=white_spectrum(1e-9, "(dI/I)^2/(rad/s)"),
    )
    assert model.apparatus() == ("Cetina 2022, 171Yb+, 15 ions", "Fang 2022, 171Yb+, 5 ions")
    assert model.undeclared_rate_count() == 1, "the untagged laser_intensity is counted"
    sentence = model.provenance_sentence()
    assert "stitched from 2 apparatus" in sentence and "Fang 2022" in sentence
    assert "1 non-zero rate(s) carry no apparatus tag" in sentence
    # three apparatus is Section 6.1's own example
    three = dataclasses.replace(
        model,
        laser_intensity=white_spectrum(
            1e-9,
            "u",
        ),
        rf_amplitude_noise=None,
    )
    three = dataclasses.replace(
        three,
        laser_intensity=dataclasses.replace(
            white_spectrum(1e-9, "u"), provenance=("Trout 2018, simulated, 2 ions",)
        ),
    )
    assert len(three.apparatus()) == 3 and three.undeclared_rate_count() == 0
    assert "stitched from 3 apparatus" in three.provenance_sentence()
