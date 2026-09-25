"""Fixed-grid stochastic trajectories for the sampled route of PLAN.md Section 6.1.

A ``Trajectory`` is interpolated linearly between fixed grid points, so an integrator sees the same realization whatever
its step sequence. ``synthesize`` draws a zero-mean Gaussian process with a spectrum's tabulated band by the spectral
method x(t) = sum_k sqrt(S(omega_k) Delta omega_k/pi) [a_k cos omega_k t + b_k sin omega_k t], a_k and b_k standard
normal, so that <x^2> = (1/pi) int S d omega; the grid must resolve the band's top (dt <= pi/omega_max).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qutip_trap.noise.spectra import Mains, NoiseSpectrum

MAX_GRID_POINTS = 400_001
"""The largest grid one trajectory may hold (a 1 ms shot at 2.5 ns steps)."""
MIN_GRID_POINTS = 65
"""Every grid has at least this many points, so a slow process is still a smooth curve over the shot."""
TAU_C_OVERSAMPLE = 10.0 * math.pi
"""Points per half period of omega_max that meet Delta t <= tau_c/10 for the band's fastest component (tau_c = 1/omega_max):
``time_grid``'s dt = pi/(omega_max x oversample) meets it at oversample >= 10 pi."""


@dataclass(frozen=True)
class Trajectory:
    """A realization on a fixed grid; a call interpolates linearly and holds the end values outside the grid."""

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

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.values**2)))

    def as_grid(self) -> np.ndarray:
        """The (2, N) array a ``NoiseSample`` stores: row 0 the times, row 1 the values."""
        return np.vstack([self.times_s, self.values])

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> Trajectory:
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("a stored trajectory is a (2, N) array")
        return cls(arr[0].copy(), arr[1].copy())

    def scaled(self, factor: float) -> Trajectory:
        return Trajectory(self.times_s, factor * self.values)


def time_grid(
    duration_s: float, omega_max_rad_s: float, *, oversample: float = TAU_C_OVERSAMPLE
) -> np.ndarray:
    """A uniform grid over [0, duration] with ``oversample`` points per half period of omega_max (Nyquist at 1)."""
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if omega_max_rad_s <= 0.0:
        return np.linspace(0.0, duration_s, MIN_GRID_POINTS)
    dt = math.pi / omega_max_rad_s / oversample
    n = max(int(math.ceil(duration_s / dt)) + 1, MIN_GRID_POINTS)
    if n > MAX_GRID_POINTS:
        raise ValueError(
            f"a trajectory resolving omega_max = {omega_max_rad_s:.3g} rad/s over {duration_s:.3g} s needs {n} points "
            f"(> {MAX_GRID_POINTS}); declare the band above the affordable grid as the spectrum's white_level instead"
        )
    return np.linspace(0.0, duration_s, n)


def _check_resolves(times_s: np.ndarray, omega_max_rad_s: float) -> None:
    dt = float(np.max(np.diff(times_s)))
    if omega_max_rad_s > 0.0 and dt > math.pi / omega_max_rad_s * (1.0 + 1e-9):
        raise ValueError(
            f"grid step {dt:.3g} s does not resolve omega_max = {omega_max_rad_s:.3g} rad/s (Nyquist {math.pi / omega_max_rad_s:.3g} s)"
        )


_SYNTH_BLOCK_ELEMENTS = 1 << 20
"""Complex elements per block of the (times x bins) phase table, so a long grid never forms the whole matrix."""


def _phased_sum(tau: np.ndarray, omega: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Re sum_k coef_k exp(i omega_k tau_j), block by block; on a uniform grid one phase table is rotated per block."""
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
    spectrum: NoiseSpectrum, times_s: np.ndarray, rng: np.random.Generator, *, n_bins: int = 512
) -> Trajectory:
    """A Gaussian realization of the spectrum's tabulated band on ``times_s``.

    The bins are log-spaced from the band's lowest positive frequency (2 pi/(1000 T) for a band starting at zero, so that
    components slower than the grid appear as a constant offset with the right variance) to its top, the first bin reaching
    down to dc when the band does; each carries S at its centre times its width.
    """
    t = np.asarray(times_s, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times_s must be a one-dimensional grid of at least two points")
    w_tab = np.abs(np.asarray(spectrum.omega_rad_s, dtype=float))
    w_hi = float(np.max(w_tab))
    if w_hi <= 0.0 or np.all(np.asarray(spectrum.S) == 0.0):
        return Trajectory(t, np.zeros_like(t))
    _check_resolves(t, w_hi)
    positive = w_tab[w_tab > 0.0]
    if positive.size and float(np.min(positive)) < w_hi:
        w_lo = float(np.min(positive))
    else:
        w_lo = 2.0 * math.pi / (1000.0 * float(t[-1] - t[0]))
    edges = np.geomspace(min(w_lo, w_hi / 10.0), w_hi, int(n_bins) + 1)
    if float(np.min(w_tab)) == 0.0:
        edges[0] = 0.0
    centres = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)
    s_c = np.asarray(spectrum.tabulated(centres), dtype=float)
    amp = np.sqrt(np.maximum(s_c * widths / math.pi, 0.0))
    a = rng.standard_normal(centres.size)
    b = rng.standard_normal(centres.size)
    # stationary, so the origin is the grid start: Re(coef e^{i w tau}) = amp (a cos w tau + b sin w tau)
    coef = amp * (a - 1j * b)
    return Trajectory(t, _phased_sum(t - t[0], centres, coef))


def mains_trajectory(mains: Mains, times_s: np.ndarray, trigger_phase_rad: float) -> Trajectory:
    """The mains field over the grid at the shot's line-trigger phase (T)."""
    t = np.asarray(times_s, dtype=float)
    return Trajectory(t, np.asarray(mains.field_t(t, trigger_phase_rad), dtype=float))


def correlated_normals(rng: np.random.Generator, times_s: np.ndarray, tau_s: float, size: int) -> np.ndarray:
    """Standard normals z[k, :] at the times ``times_s`` with <z_k z_l> = exp(-|t_k - t_l|/tau) per column: an OU chain on an
    irregular grid, the raw draws of a Drift at the shot clock."""
    t = np.asarray(times_s, dtype=float)
    out = np.empty((t.size, size))
    out[0] = rng.standard_normal(size)
    for k in range(1, t.size):
        rho = math.exp(-abs(t[k] - t[k - 1]) / tau_s) if math.isfinite(tau_s) else 1.0
        out[k] = rho * out[k - 1] + math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(size)
    return out
