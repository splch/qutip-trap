"""Pulsed Raman sideband cooling: exact transfer matrices on the Fock populations (Che et al. 2017; Rasmusson et al. 2021).

One cycle, a k-th-order red-sideband pulse of duration t followed by fast pumping that erases the coherences, is the
column-stochastic matrix W_k(t) with diagonal a_n = cos^2(Omega_{n,n-k} t/2) and k-th upper diagonal
b_n = sin^2(Omega_{n,n-k} t/2), built from the exact Omega_{n,n-k} = Omega_0 |<n-k|D(i eta)|n>| (never the Lamb-Dicke
sqrt n); N pulses compose as an ordered product and the repump's photons kick through the recoil kernel. Omega_{n,n-k} is
proportional to L^{(k)}_{n-k}(eta^2), whose zeros strand population under single-order cooling, so higher orders are
applied first; a fixed pulse traps |n> wherever Omega_{n-1,n} t = 2 m pi in this plan's Rabi convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from qutip_trap.dynamics.operators import rabi_matrix_element, thermal_populations
from qutip_trap.light.recoil import Quadrature1D, recoil_kernel_matrix


@dataclass(frozen=True)
class SidebandPulse:
    """One k-th-order red-sideband pulse of the pulsed schedule (Che Eq. 2; Rasmusson Eqs. 6-7)."""

    order: int
    duration_s: float

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("the sideband order is a positive integer")
        if self.duration_s < 0.0:
            raise ValueError("the pulse duration is non-negative")


def sideband_rabi_rad_s(omega0_rad_s: float, eta: float, n: int, order: int) -> float:
    """Omega_{n,n-k} = Omega_0 |<n-k|D(i eta)|n>|, the exact k-th red-sideband Rabi frequency out of |n> (0 for n < k)."""
    if n < order:
        return 0.0
    return omega0_rad_s * rabi_matrix_element(n - order, n, eta)


def transfer_matrix(d: int, eta: float, omega0_rad_s: float, order: int, duration_s: float) -> np.ndarray:
    """W_k(t): column n holds cos^2(Omega_{n,n-k} t/2) at (n, n) and sin^2 at (n - k, n); columns n < k are the absorbing
    identity block."""
    if d < 2:
        raise ValueError("at least two Fock levels")
    w = np.eye(d)
    for n in range(order, d):
        half = 0.5 * sideband_rabi_rad_s(omega0_rad_s, eta, n, order) * duration_s
        b = math.sin(half) ** 2
        w[n, n] = 1.0 - b
        w[n - order, n] = b
    return w


def apply_pulses(
    p0: np.ndarray,
    pulses: Sequence[SidebandPulse],
    eta: float,
    omega0_rad_s: float,
    *,
    repump: np.ndarray | None = None,
) -> np.ndarray:
    """The Fock populations after the ordered pulses, each followed by the repump kernel (None: coherences discarded, no
    recoil)."""
    p = np.asarray(p0, dtype=float).copy()
    d = p.size
    if repump is not None and repump.shape != (d, d):
        raise ValueError("the repump kernel acts on the same Fock space as the populations")
    cache: dict[tuple[int, float], np.ndarray] = {}
    for pulse in pulses:
        key = (pulse.order, pulse.duration_s)
        w = cache.get(key)
        if w is None:
            w = transfer_matrix(d, eta, omega0_rad_s, pulse.order, pulse.duration_s)
            cache[key] = w
        p = w @ p
        if repump is not None:
            p = repump @ p
    return p


def mean_occupation(p: np.ndarray) -> float:
    """sum_n n p_n of a Fock distribution."""
    return float(np.dot(np.arange(p.size), p))


def thermal_distribution(nbar: float, d: int) -> np.ndarray:
    """P_n = nbar^n/(nbar + 1)^{n+1} on d levels (the truncated tail is [nbar/(nbar + 1)]^d)."""
    return thermal_populations(nbar, d)


def repump_kernel(d: int, eta_em: float, quad: Quadrature1D, mean_photons: float) -> np.ndarray:
    """The Fock kernel of a repump scattering a Poisson-distributed number of photons of mean ``mean_photons``, each
    kicking with the one-dimensional recoil quadrature: sum_k Poisson(k) K^k, truncated at 1e-12 of the Poisson weight."""
    k1 = recoil_kernel_matrix(d, eta_em, quad)
    out = np.zeros((d, d))
    power = np.eye(d)
    weight = math.exp(-mean_photons)
    k = 0
    while weight > 1e-12 or k <= mean_photons:
        out += weight * power
        k += 1
        weight *= mean_photons / k
        power = k1 @ power
        if k > 200:
            break
    return out


def pi_time_s(omega0_rad_s: float, eta: float, n: int, order: int) -> float:
    """pi/Omega_{n,n-k}: the pulse that empties |n> on the k-th sideband."""
    om = sideband_rabi_rad_s(omega0_rad_s, eta, n, order)
    if om <= 0.0:
        raise ValueError(f"|{n}> has no {order}-th red sideband")
    return math.pi / om


GRID_POINTS = 64
"""Coarse grid of the scalar searches: the occupation is oscillatory in the pulse time, so a bounded minimizer alone
lands in a local minimum (6.7 against the global 3.2 on Rasmusson's fixed-pulse schedule)."""


def _scalar_minimum(
    objective: Callable[[float], float], bounds_s: tuple[float, float]
) -> tuple[float, float]:
    """Coarse grid over the bounds, then a bounded refinement around the best grid point."""
    grid = np.linspace(bounds_s[0], bounds_s[1], GRID_POINTS)
    values = np.array([objective(float(t)) for t in grid])
    k = int(np.argmin(values))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, GRID_POINTS - 1)]
    if hi <= lo:
        return float(grid[k]), float(values[k])
    sol = minimize_scalar(objective, bounds=(float(lo), float(hi)), method="bounded")
    if float(sol.fun) <= values[k]:
        return float(sol.x), float(sol.fun)
    return float(grid[k]), float(values[k])


def optimize_per_order(
    p0: np.ndarray,
    counts: Mapping[int, int],
    eta: float,
    omega0_rad_s: float,
    bounds_s: tuple[float, float],
    *,
    repump: np.ndarray | None = None,
) -> tuple[list[SidebandPulse], float]:
    """One pulse time per sideband order, higher orders first, each chosen by scalar minimization of the occupation after
    its own block with the earlier blocks fixed. Returns (pulses, final nbar)."""
    pulses: list[SidebandPulse] = []
    p = np.asarray(p0, dtype=float)
    for order in sorted(counts, reverse=True):
        n_pulses = counts[order]
        if n_pulses <= 0:
            continue

        def block(t: float, order: int = order, n_pulses: int = n_pulses, p: np.ndarray = p) -> float:
            return mean_occupation(
                apply_pulses(p, [SidebandPulse(order, t)] * n_pulses, eta, omega0_rad_s, repump=repump)
            )

        t_best, _value = _scalar_minimum(block, bounds_s)
        pulses.extend([SidebandPulse(order, t_best)] * n_pulses)
        p = apply_pulses(p, [SidebandPulse(order, t_best)] * n_pulses, eta, omega0_rad_s, repump=repump)
    return pulses, mean_occupation(p)


def optimize_durations(
    p0: np.ndarray,
    orders: Sequence[int],
    eta: float,
    omega0_rad_s: float,
    initial_s: Sequence[float],
    bounds_s: tuple[float, float],
    *,
    repump: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Independent pulse times by L-BFGS-B (finite differences) from ``initial_s``: (durations, final nbar); colder than
    ``optimize_per_order`` at the cost of one objective evaluation per pulse per gradient."""

    def final_nbar(durations: np.ndarray) -> float:
        pulses = [SidebandPulse(k, float(t)) for k, t in zip(orders, durations)]
        return mean_occupation(apply_pulses(p0, pulses, eta, omega0_rad_s, repump=repump))

    sol = minimize(
        final_nbar, np.asarray(initial_s, dtype=float), method="L-BFGS-B", bounds=[bounds_s] * len(orders)
    )
    return np.asarray(sol.x), float(sol.fun)
