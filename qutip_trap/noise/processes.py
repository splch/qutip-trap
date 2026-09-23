"""Stochastic time series for sampled noise.

A ``Trajectory`` is a realization on a fixed time grid with linear interpolation, so the values an integrator sees do
not depend on its step sequence. The grid must resolve the band's top (dt <= pi/omega_max); anything faster belongs in
the spectrum's ``white_level``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qutip_trap.noise.spectra import Mains, NoiseSpectrum

MAX_GRID_POINTS = 400_001
"""The largest grid one trajectory may hold; ``time_grid`` refuses a finer one."""


@dataclass(frozen=True)
class Trajectory:
    """A realization on a fixed grid; ``__call__`` interpolates linearly and holds the end values outside the grid."""

    times_s: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times_s, dtype=float)
        v = np.asarray(self.values, dtype=float)
        if t.ndim != 1 or t.shape != v.shape or t.size < 2:
            raise ValueError(
                "a trajectory needs matching one-dimensional time and value arrays of length >= 2"
            )
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")

    def __call__(self, t_s: float) -> float:
        return float(np.interp(t_s, self.times_s, self.values))

    def at(self, t_s: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(np.asarray(t_s, dtype=float), self.times_s, self.values))

    @property
    def dt_s(self) -> float:
        return float(self.times_s[1] - self.times_s[0])

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.values**2)))

    def as_grid(self) -> np.ndarray:
        """The (2, N) array a ``NoiseSample`` stores under one key: row 0 the times, row 1 the values."""
        return np.vstack([self.times_s, self.values])

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> Trajectory:
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("a stored trajectory is a (2, N) array")
        return cls(arr[0].copy(), arr[1].copy())

    def scaled(self, factor: float) -> Trajectory:
        return Trajectory(self.times_s, factor * self.values)

    def mapped(self, fn: np.ndarray) -> Trajectory:
        return Trajectory(self.times_s, np.asarray(fn, dtype=float))


MIN_GRID_POINTS = 65
"""The fewest points of a trajectory grid, so a slow process is still a smooth curve over the shot."""

TAU_C_OVERSAMPLE = 10.0 * math.pi
"""The ``time_grid`` oversampling that gives dt <= tau_c/10 for the band's fastest component (tau_c = 1/omega_max)."""


def time_grid(
    duration_s: float,
    omega_max_rad_s: float,
    *,
    t0_s: float = 0.0,
    oversample: float = TAU_C_OVERSAMPLE,
    min_points: int = MIN_GRID_POINTS,
) -> np.ndarray:
    """A uniform grid over [t0, t0 + duration] with ``oversample`` points per half period of omega_max (``oversample=1``
    is Nyquist) and at least ``min_points``; more than ``MAX_GRID_POINTS`` is refused."""
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if omega_max_rad_s <= 0.0:
        return np.linspace(t0_s, t0_s + duration_s, max(2, min_points))
    dt = math.pi / omega_max_rad_s / oversample
    n = max(int(math.ceil(duration_s / dt)) + 1, min_points)
    if n > MAX_GRID_POINTS:
        raise ValueError(
            f"a trajectory resolving omega_max = {omega_max_rad_s:.3g} rad/s over {duration_s:.3g} s needs {n} points "
            f"(> {MAX_GRID_POINTS}); declare the band above the affordable grid as the spectrum's white_level instead "
            "(Section 6.1 route b)"
        )
    return np.linspace(t0_s, t0_s + duration_s, max(n, 2))


def _check_resolves(times_s: np.ndarray, omega_max_rad_s: float) -> None:
    dt = float(np.max(np.diff(times_s)))
    if omega_max_rad_s > 0.0 and dt > math.pi / omega_max_rad_s * (1.0 + 1e-9):
        raise ValueError(
            f"grid step {dt:.3g} s does not resolve omega_max = {omega_max_rad_s:.3g} rad/s (Nyquist {math.pi / omega_max_rad_s:.3g} s)"
        )


_SYNTH_BLOCK_ELEMENTS = 1 << 20
"""Complex elements per block of the (times x bins) phase table ``_phased_sum`` evaluates, which bounds its memory."""


def _phased_sum(tau: np.ndarray, omega: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Re sum_k coef_k exp(i omega_k tau_j) on the grid ``tau``, block by block (one rotated phase table if uniform)."""
    n_t = int(tau.size)
    n_w = int(omega.size)
    out = np.empty(n_t)
    if n_t == 0 or n_w == 0:
        out[:] = 0.0
        return out
    block = max(1, min(n_t, _SYNTH_BLOCK_ELEMENTS // n_w))
    steps = np.diff(tau)
    uniform = n_t > 2 and bool(np.all(np.abs(steps - steps[0]) <= 1e-9 * abs(steps[0])))
    table = np.exp(1j * np.outer(np.arange(block) * steps[0], omega)) if uniform else None
    for start in range(0, n_t, block):
        stop = min(start + block, n_t)
        if table is not None:
            phase = table[: stop - start] * np.exp(1j * tau[start] * omega)
        else:
            phase = np.exp(1j * np.outer(tau[start:stop], omega))
        out[start:stop] = (phase @ coef).real
    return out


def synthesize(
    spectrum: NoiseSpectrum,
    times_s: np.ndarray,
    rng: np.random.Generator,
    *,
    n_bins: int = 512,
    omega_min_rad_s: float | None = None,
) -> Trajectory:
    """A Gaussian realization x(t) = sum_k sqrt(S(omega_k) Delta omega_k/pi) [a_k cos omega_k t + b_k sin omega_k t],
    a_k, b_k ~ N(0, 1), of the spectrum's tabulated band on ``times_s``, over bins log-spaced from ``omega_min``
    (default: the lowest positive frequency, else 2 pi/(1000 T); at most a tenth of the top) to the band's top."""
    t = np.asarray(times_s, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times_s must be a one-dimensional grid of at least two points")
    w_tab = np.abs(np.asarray(spectrum.omega_rad_s, dtype=float))
    w_hi = float(np.max(w_tab))
    if w_hi <= 0.0 or np.all(np.asarray(spectrum.S) == 0.0):
        return Trajectory(t, np.zeros_like(t))
    _check_resolves(t, w_hi)
    positive = w_tab[w_tab > 0.0]
    span = float(t[-1] - t[0])
    if omega_min_rad_s is not None:
        w_lo = float(omega_min_rad_s)
    elif positive.size and float(np.min(positive)) < w_hi:
        w_lo = float(np.min(positive))
    else:
        w_lo = 2.0 * math.pi / (1000.0 * span)
    w_lo = min(w_lo, w_hi / 10.0)
    edges = np.geomspace(w_lo, w_hi, int(n_bins) + 1)
    if float(np.min(w_tab)) == 0.0 and omega_min_rad_s is None:
        edges[0] = 0.0
    centres = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)
    s_c = np.asarray(spectrum.tabulated(centres), dtype=float)
    amp = np.sqrt(np.maximum(s_c * widths / math.pi, 0.0))
    a = rng.standard_normal(centres.size)
    b = rng.standard_normal(centres.size)
    # the process is stationary: the origin is the grid start; Re(coef e^{i w tau}) = amp (a cos w tau + b sin w tau)
    coef = amp * (a - 1j * b)
    return Trajectory(t, _phased_sum(t - t[0], centres, coef))


def ou_process(
    sigma: float, tau_c_s: float, times_s: np.ndarray, rng: np.random.Generator, *, x0: float | None = None
) -> Trajectory:
    """The exact Ornstein-Uhlenbeck realization with stationary rms ``sigma`` and correlation time ``tau_c`` on any grid."""
    t = np.asarray(times_s, dtype=float)
    if tau_c_s <= 0.0 or sigma < 0.0:
        raise ValueError("tau_c positive and sigma non-negative")
    x = np.empty_like(t)
    x[0] = rng.normal(0.0, sigma) if x0 is None else float(x0)
    for k in range(1, t.size):
        rho = math.exp(-(t[k] - t[k - 1]) / tau_c_s)
        x[k] = rho * x[k - 1] + sigma * math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal()
    return Trajectory(t, x)


def ou_autocorrelation(sigma: float, tau_c_s: float, lag_s: np.ndarray | float) -> np.ndarray | float:
    return sigma**2 * np.exp(-np.abs(np.asarray(lag_s, dtype=float)) / tau_c_s)


def mains_trajectory(mains: Mains, times_s: np.ndarray, trigger_phase_rad: float) -> Trajectory:
    """The deterministic mains field over the grid at the shot's line-trigger phase (tesla)."""
    t = np.asarray(times_s, dtype=float)
    return Trajectory(t, np.asarray(mains.field_t(t, trigger_phase_rad), dtype=float))


def correlated_normals(rng: np.random.Generator, times_s: np.ndarray, tau_s: float, size: int) -> np.ndarray:
    """Standard normals z[k, :] at ``times_s`` with <z_k z_l> = exp(-|t_k - t_l|/tau) per column, an OU chain on any
    grid (tau = inf holds the first draw)."""
    t = np.asarray(times_s, dtype=float)
    out = np.empty((t.size, size))
    out[0] = rng.standard_normal(size)
    for k in range(1, t.size):
        rho = math.exp(-abs(t[k] - t[k - 1]) / tau_s) if math.isfinite(tau_s) else 1.0
        out[k] = rho * out[k - 1] + math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(size)
    return out


__all__ = [
    "MAX_GRID_POINTS",
    "MIN_GRID_POINTS",
    "TAU_C_OVERSAMPLE",
    "Trajectory",
    "correlated_normals",
    "mains_trajectory",
    "ou_autocorrelation",
    "ou_process",
    "synthesize",
    "time_grid",
]
