"""Spectra, fixed-grid trajectories and their synthesis (PLAN.md Section 6.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import welch

from qutip_trap.noise.sampling import (
    MAX_GRID_POINTS,
    MIN_GRID_POINTS,
    NoiseSample,
    Trajectory,
    correlated_normals,
    key_qubit_trajectory_hz,
    mains_trajectory,
    quiet_sample,
    synthesize,
    time_grid,
)
from qutip_trap.noise.spectra import (
    Mains,
    NoiseSpectrum,
    gaussian_spectrum,
    ou_spectrum,
    power_law_spectrum,
    white_spectrum,
)


def test_spectrum_records_are_two_sided_with_the_variance_kernel_and_a_white_level() -> None:
    """(1/pi) int_0^inf S d omega is the variance; the OU and Gaussian constructors integrate to their variances; a white
    level is a separate field, never a heuristic on the shape."""
    ou = ou_spectrum(0.02, 0.05, "u^2/(rad/s)")
    assert ou.variance() == pytest.approx(0.02, rel=5e-3)
    assert ou.value(0.0) == pytest.approx(2.0 * 0.02 * 0.05)
    g = gaussian_spectrum(0.3, 2.0, "u")
    assert g.variance() == pytest.approx(0.09, rel=1e-9)
    assert g.tabulated(0.0) == pytest.approx(math.sqrt(2.0 * math.pi) * 0.09 / 2.0)
    w = white_spectrum(3.0, "u")
    assert not w.is_zero() and w.value(1e9) == 3.0 and w.variance() == 0.0 and w.tabulated(5.0) == 0.0
    pl = power_law_spectrum(1.0, 1.0, 2.0, "u", omega_min_rad_s=0.1, omega_max_rad_s=10.0)
    assert pl.tabulated(2.0) == pytest.approx(0.25, rel=1e-6) and pl.omega_max_rad_s == 10.0
    with pytest.raises(ValueError):
        NoiseSpectrum(np.array([0.0, 1.0]), np.array([1.0, -1.0]), "u")


@pytest.mark.slow
def test_synthesized_process_reproduces_variance_and_autocorrelation() -> None:
    """The spectral method: <x^2> = (1/pi) int S d omega and <x(t) x(t + tau_c)> = sigma^2/e for the OU spectrum (ensemble
    over seeds; the grid resolves the band's top)."""
    sp = ou_spectrum(0.02, 0.05, "u", omega_max_rad_s=60.0 / 0.05)
    t = time_grid(20.0, sp.omega_max_rad_s)
    lag = int(round(0.05 / (t[1] - t[0])))
    var, ac = [], []
    for k in range(200):
        tr = synthesize(sp, t, np.random.default_rng(k))
        var.append(float(np.mean(tr.values**2)))
        ac.append(float(np.mean(tr.values[:-lag] * tr.values[lag:])))
    assert np.mean(var) == pytest.approx(sp.variance(), rel=0.03)
    assert np.mean(ac) == pytest.approx(0.02 * math.exp(-1.0), rel=0.05)
    assert synthesize(sp, t, np.random.default_rng(3)).values.shape == t.shape
    zero = synthesize(NoiseSpectrum(np.array([0.0, 1.0]), np.zeros(2), "u"), t, np.random.default_rng(0))
    assert np.all(zero.values == 0.0)


def test_correlated_normals_are_an_exact_chain_on_any_grid() -> None:
    z = correlated_normals(np.random.default_rng(1), np.array([0.0, 1.0, 100.0]), 1.0, 20000)
    assert np.mean(z[0] * z[1]) == pytest.approx(math.exp(-1.0), abs=0.03)
    assert abs(np.mean(z[0] * z[2])) < 0.03
    z_inf = correlated_normals(np.random.default_rng(2), np.array([0.0, 7.0]), math.inf, 5)
    assert np.array_equal(z_inf[0], z_inf[1]), "an infinite correlation time is one draw for the whole run"


def test_trajectory_interpolates_linearly_on_its_fixed_grid_and_round_trips_through_a_sample() -> None:
    t = np.linspace(0.0, 1.0, 101)
    tr = Trajectory(t, np.random.default_rng(0).standard_normal(t.size))
    assert tr(0.505) == pytest.approx(0.5 * (tr.values[50] + tr.values[51]))
    assert tr(-1.0) == tr.values[0] and tr(5.0) == tr.values[-1]
    sample = NoiseSample(0, {}, {key_qubit_trajectory_hz(0): tr.as_grid()})
    back = sample.trajectory(key_qubit_trajectory_hz(0))
    assert back is not None and np.array_equal(back.values, tr.values) and not sample.is_quiet
    assert quiet_sample().is_quiet and quiet_sample().trajectory("x") is None


def test_time_grid_honours_delta_t_at_most_tau_c_over_ten_or_refuses_with_the_remedy() -> None:
    """Section 5.5's Delta t <= tau_c/10 for the band's fastest component, tau_c = 1/omega_max."""
    for w_max in (1e2, 1e4, 2.0 * math.pi * 3e3, 2.0 * math.pi * 1e5):
        dt = float(np.max(np.diff(time_grid(1e-2, w_max))))
        assert dt <= 1.0 / w_max / 10.0 * (1.0 + 1e-12), (w_max, dt)
    assert time_grid(1e-3, 0.0).size == MIN_GRID_POINTS
    with pytest.raises(ValueError):
        time_grid(0.0, 1e3)
    with pytest.raises(ValueError, match="white_level"):
        time_grid(1.0, 2.0 * math.pi * 1e9)
    assert MAX_GRID_POINTS > 1000
    sp = ou_spectrum(1.0, 1.0, "u", omega_max_rad_s=100.0)
    with pytest.raises(ValueError, match="Nyquist"):
        synthesize(sp, np.linspace(0.0, 10.0, 11), np.random.default_rng(0))


def test_mains_trajectory_and_trigger_phase() -> None:
    m = Mains(60.0, {1: 1e-9, 3: 2e-10}, {1: 0.0, 3: 0.5}, "free_running")
    t = np.linspace(0.0, 1e-3, 201)
    tr = mains_trajectory(m, t, 0.0)
    assert tr.values[0] == pytest.approx(1e-9 + 2e-10 * math.cos(0.5))
    assert m.omega_max_rad_s == pytest.approx(2.0 * math.pi * 180.0)
    shifted = mains_trajectory(m, t, math.pi)
    assert shifted.values[0] == pytest.approx(-1e-9 + 2e-10 * math.cos(0.5 + 3.0 * math.pi))
    with pytest.raises(ValueError):
        Mains(60.0, {1: 1e-9}, {2: 0.0}, "free_running")


def _periodogram_two_sided(values: np.ndarray, dt: float, n_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch's single-sided per-hertz estimate, halved into the two-sided angular convention S(2 pi f) = S_1(f)/2."""
    f, pxx = welch(values, fs=1.0 / dt, nperseg=max(64, values.size // n_seg), scaling="density")
    return 2.0 * math.pi * f, pxx / 2.0


@pytest.mark.slow
def test_the_realized_periodogram_reproduces_the_configured_ou_spectrum() -> None:
    """Section 5.5's test: the periodogram of long realizations of a Lorentzian band matches ``spectrum.tabulated`` over the
    band the trajectory resolves (mean ratio 1 within 15 % from 0.4/tau_c to 6/tau_c over 16 seeds) and rolls off with the
    Lorentzian, as a linearly interpolated sample path must."""
    tau_c = 1e-3
    sp = ou_spectrum(4e-4, tau_c, "u^2/(rad/s)", omega_max_rad_s=20.0 / tau_c)
    t = time_grid(200.0 * tau_c, sp.omega_max_rad_s)
    dt = float(t[1] - t[0])
    assert dt <= tau_c / 10.0
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
    # a periodogram bin is chi^2-distributed: most bins within 50 %, not the maximum
    assert float(np.mean(np.abs(ratio - 1.0) < 0.5)) > 0.8, sorted(np.round(ratio, 3).tolist())
    var = float(np.mean(synthesize(sp, t, np.random.default_rng(11), n_bins=128).values ** 2))
    assert var == pytest.approx(sp.variance(), rel=0.3)
    # a decade above the corner the density has fallen by about 100
    lo = est[(w >= 0.4 / tau_c) & (w <= 0.6 / tau_c)]
    hi = est[(w >= 8.0 / tau_c) & (w <= 12.0 / tau_c)]
    assert lo.size and hi.size and float(np.mean(lo)) / float(np.mean(hi)) > 20.0


def test_the_mains_harmonics_appear_as_lines_in_the_periodogram() -> None:
    """The mains trajectory's periodogram is a comb at h f_line with line powers A_h^2/2; a free-running trigger phase moves
    the lines' phase, not their power."""
    line = 50.0
    amps = {1: 3e-9, 2: 1e-9, 3: 5e-10}
    mains = Mains(line, amps, dict.fromkeys(amps, 0.0), "line_triggered")
    t = np.linspace(0.0, 2.0, 200001)
    dt = float(t[1] - t[0])
    values = mains_trajectory(mains, t, 0.0).values
    f, pxx = welch(values, fs=1.0 / dt, nperseg=100000, scaling="spectrum")
    for h, amp in amps.items():
        k = int(np.argmin(np.abs(f - h * line)))
        assert abs(f[k] - h * line) < 1.0, (h, f[k])
        assert pxx[k] == pytest.approx(amp**2 / 2.0, rel=0.05), (h, pxx[k], amp**2 / 2.0)
    between = pxx[(f > 1.2 * line) & (f < 1.8 * line)]
    assert float(np.max(between)) < 1e-6 * (amps[1] ** 2 / 2.0)
    shifted = mains_trajectory(mains, t, 1.1).values
    _f2, pxx2 = welch(shifted, fs=1.0 / dt, nperseg=100000, scaling="spectrum")
    k1 = int(np.argmin(np.abs(f - line)))
    assert pxx2[k1] == pytest.approx(pxx[k1], rel=0.05)
