"""Spectra, trajectories and sampling records of the noise layer (PLAN.md Sections 5.5, 6.1, 6.3, 13; Section 9.17 row
'Seeds and reproducibility'; M7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.noise.processes import (
    MAX_GRID_POINTS,
    Trajectory,
    correlated_normals,
    mains_trajectory,
    ou_autocorrelation,
    ou_process,
    synthesize,
    time_grid,
)
from qutip_trap.noise.sampling import NoiseSample, key_qubit_trajectory_hz, quiet_sample
from qutip_trap.noise.spectra import (
    Mains,
    NoiseSpectrum,
    gaussian_spectrum,
    ou_spectrum,
    power_law_spectrum,
    spectrum_from_single_sided_hz,
    white_spectrum,
)
from qutip_trap.trap.heating import single_sided_from_two_sided


def test_spectrum_records_are_two_sided_with_the_variance_kernel_and_a_white_level() -> None:
    """(1/pi) int_0^inf S d omega is the variance (Section 13 kernel); the OU and Gaussian constructors integrate to their
    variances; a white level is a separate field, never a heuristic on the shape; single-sided per-hertz data ingests as S/2."""
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
    ss = spectrum_from_single_sided_hz(
        np.array([0.0, 1e3]), np.array([4.0, 4.0]), "u", white_level_per_hz=2.0
    )
    assert ss.tabulated(2.0 * math.pi * 500.0) == pytest.approx(2.0) and ss.white_level == 1.0
    assert single_sided_from_two_sided(ss.tabulated(1.0)) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        NoiseSpectrum(np.array([0.0, 1.0]), np.array([1.0, -1.0]), "u")
    with pytest.raises(ValueError):
        NoiseSpectrum(np.array([0.0, 1.0]), np.array([1.0, 1.0]), "u", sidedness="one-sided")  # type: ignore[arg-type]


def test_synthesized_process_reproduces_variance_and_autocorrelation() -> None:
    """The spectral method: <x^2> = (1/pi) int S d omega and <x(t) x(t + tau_c)> = sigma^2/e for the OU spectrum (ensemble over
    seeds; the grid resolves the band's top)."""
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


def test_ou_process_and_correlated_normals_are_exact_on_any_grid() -> None:
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0.0, 4000.0, 40000))
    t[0] = 0.0
    tr = ou_process(0.5, 2.0, t, rng)
    assert tr.rms() == pytest.approx(0.5, rel=0.1), (
        "2000 correlation times of an exact OU chain on an irregular grid"
    )
    assert ou_autocorrelation(0.5, 2.0, 2.0) == pytest.approx(0.25 / math.e)
    z = correlated_normals(np.random.default_rng(1), np.array([0.0, 1.0, 100.0]), 1.0, 20000)
    assert np.mean(z[0] * z[1]) == pytest.approx(math.exp(-1.0), abs=0.03)
    assert abs(np.mean(z[0] * z[2])) < 0.03
    z_inf = correlated_normals(np.random.default_rng(2), np.array([0.0, 7.0]), math.inf, 5)
    assert np.array_equal(z_inf[0], z_inf[1]), "an infinite correlation time is one draw for the whole run"


def test_trajectory_is_fixed_grid_and_integrator_independent() -> None:
    """Section 9.17: an OU realization is identical under two integrator step sequences because it lives on a fixed grid and is
    interpolated; the NoiseSample round trip preserves it."""
    t = np.linspace(0.0, 1.0, 101)
    tr = ou_process(1.0, 0.1, t, np.random.default_rng(0))
    fine = np.linspace(0.0, 1.0, 1001)
    coarse = np.linspace(0.0, 1.0, 51)
    assert np.allclose(tr.at(fine[::20]), tr.at(coarse), atol=0.0)
    assert tr(0.505) == pytest.approx(0.5 * (tr.values[50] + tr.values[51]))
    assert tr(-1.0) == tr.values[0] and tr(5.0) == tr.values[-1]
    sample = NoiseSample(0, {}, {key_qubit_trajectory_hz(0): tr.as_grid()})
    back = sample.trajectory(key_qubit_trajectory_hz(0))
    assert back is not None and np.array_equal(back.values, tr.values) and not sample.is_quiet
    assert quiet_sample().is_quiet and quiet_sample().trajectory("x") is None


def test_time_grid_resolves_the_band_or_refuses_with_the_remedy() -> None:
    g = time_grid(1e-3, 2.0 * math.pi * 1e5)
    assert g[1] - g[0] <= math.pi / (2.0 * math.pi * 1e5) / 4.0 * (1 + 1e-9)
    assert time_grid(1e-3, 0.0).size >= 65
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
    assert isinstance(Trajectory.from_grid(tr.as_grid()), Trajectory)
