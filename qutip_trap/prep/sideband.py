"""Resolved-sideband and Raman sideband cooling: continuous closed forms, pulsed transfer matrices and thermometry.

Pulses use the exact Omega_{n,n-k} = Omega_0 |<n-k|D(i eta)|n>|, never the Lamb-Dicke sqrt n; its Laguerre zeros strand
population, so higher orders go first. Transfer is sin^2(Omega t/2) in the (hbar Omega/2) convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares, minimize, minimize_scalar, nnls
from scipy.special import eval_genlaguerre

from qutip_trap.hilbert.operators import rabi_matrix_element, thermal_populations
from qutip_trap.light.recoil import Quadrature1D, recoil_kernel_matrix
from qutip_trap.prep.closed_forms import EffectiveTwoLevel


def raman_two_photon_rabi_rad_s(omega_1_rad_s: float, omega_2_rad_s: float, delta_rad_s: float) -> float:
    """Omega = Omega_1 Omega_2/(2 Delta) in the (hbar Omega/2) convention: Monroe 1995's g_1 g_2/Delta with g = Omega/2."""
    if delta_rad_s == 0.0:
        raise ValueError("the Raman detuning is nonzero")
    return omega_1_rad_s * omega_2_rad_s / (2.0 * delta_rad_s)


def monroe_half_rabi(omega_plan_rad_s: float) -> float:
    """g = Omega/2: the single-photon coupling in the convention of Monroe 1995 and Wineland 1998."""
    return 0.5 * omega_plan_rad_s


def stenholm_floor_half_width(gamma_half_rad_s: float, nu_rad_s: float, carrier_weight: float) -> float:
    """nbar = (gamma/nu)^2 [(eta~/eta)^2 + 1/4] with the half width gamma (Stenholm 1986 Eqs. 5.49-5.54)."""
    return (gamma_half_rad_s / nu_rad_s) ** 2 * (carrier_weight + 0.25)


def quenched_floor(effective: EffectiveTwoLevel, nu_rad_s: float, carrier_weight: float) -> float:
    """The sideband floor of a quenched (Xi) or repumped (V) scheme, with Marzoli's gamma' (not Gamma'/2) as half width."""
    return stenholm_floor_half_width(effective.gamma_coherence_rad_s, nu_rad_s, carrier_weight)


def trapping_pulse_areas(n: int, eta: float, omega0_rad_s: float, duration_s: float) -> float:
    """Omega_{n-1,n} t / (2 pi): an integer means the fixed pulse traps the population of |n>."""
    return sideband_rabi_rad_s(omega0_rad_s, eta, n, 1) * duration_s / (2.0 * math.pi)


@dataclass(frozen=True)
class SidebandPulse:
    """One k-th-order red-sideband pulse of a pulsed schedule (Che et al. 2017 Eq. 2)."""

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
    """W_k(t): column n holds cos^2(Omega_{n,n-k} t/2) at (n, n) and sin^2 at (n - k, n); columns n < k are the identity
    (Rasmusson et al. 2021 Eqs. 6-7)."""
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
    """The Fock populations after the ordered pulses, each followed by the ``repump`` kernel (None: no recoil)."""
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
    """sum_n n p_n of a Fock distribution ``p``."""
    return float(np.dot(np.arange(p.size), p))


def thermal_distribution(nbar: float, d: int) -> np.ndarray:
    """P_n = nbar^n/(nbar + 1)^{n+1} on d levels, the truncated tail reported by ``truncated_tail``."""
    return thermal_populations(nbar, d)


def truncated_tail(nbar: float, d: int) -> float:
    """1 - sum_{n < d} P_n = [nbar/(nbar + 1)]^d: the probability the truncation drops."""
    return (nbar / (nbar + 1.0)) ** d if nbar > 0.0 else 0.0


def repump_kernel(d: int, eta_em: float, quad: Quadrature1D, mean_photons: float) -> np.ndarray:
    """sum_k Poisson(k) K^k: the Fock kernel of a repump scattering a Poisson number of photons of mean ``mean_photons``."""
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


def laguerre_first_zero(order: int, eta: float, *, n_max: float = 1e4) -> float:
    """The smallest continuous degree n' > 0 at which L^{(k)}_{n'}(eta^2) vanishes (scipy's real-degree Laguerre)."""
    x = eta * eta

    def f(n: float) -> float:
        return float(eval_genlaguerre(n, order, x))

    step = 0.5
    a = 0.0
    fa = f(a)
    while a < n_max:
        b = a + step
        fb = f(b)
        if fa * fb <= 0.0:
            return float(brentq(f, a, b, xtol=1e-10))
        a, fa = b, fb
    raise RuntimeError("no Laguerre zero found below n_max")


def stranded_index(order: int, eta: float) -> int:
    """The Fock state whose k-th red-sideband Rabi frequency sits at the first zero, round(zero + k)."""
    return int(round(laguerre_first_zero(order, eta) + order))


def stranded_population(p: np.ndarray, order: int, eta: float) -> float:
    """Population at and above the Rabi node of the given order, which that order's pulses cannot remove."""
    return float(np.sum(p[stranded_index(order, eta) :]))


def accumulation_centre(p: np.ndarray, order: int, eta: float) -> float | None:
    """Population-weighted mean Fock index above the node's lower shoulder (from the first zero's degree), None if empty."""
    lo = int(math.floor(laguerre_first_zero(order, eta)))
    tail = p[lo:]
    total = float(np.sum(tail))
    if total <= 0.0:
        return None
    return float(np.dot(np.arange(lo, p.size), tail)) / total


def _final_nbar(
    p0: np.ndarray,
    orders: Sequence[int],
    durations: Sequence[float],
    eta: float,
    omega0: float,
    repump: np.ndarray | None,
) -> float:
    pulses = [SidebandPulse(k, float(t)) for k, t in zip(orders, durations)]
    return mean_occupation(apply_pulses(p0, pulses, eta, omega0, repump=repump))


GRID_POINTS = 64
"""Coarse-grid size of the pulse-time searches (the occupation oscillates in the pulse time)."""


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


def optimize_shared_duration(
    p0: np.ndarray,
    orders: Sequence[int],
    eta: float,
    omega0_rad_s: float,
    bounds_s: tuple[float, float],
    *,
    repump: np.ndarray | None = None,
) -> tuple[float, float]:
    """One pulse time for every pulse: (t, final nbar) by a grid scan and bounded refinement."""
    return _scalar_minimum(
        lambda t: _final_nbar(p0, orders, [t] * len(orders), eta, omega0_rad_s, repump), bounds_s
    )


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
    """Independent pulse times by gradient descent (L-BFGS-B, finite differences) from ``initial_s``: (durations, final nbar)."""
    x0 = np.asarray(initial_s, dtype=float)
    sol = minimize(
        lambda t: _final_nbar(p0, orders, t, eta, omega0_rad_s, repump),
        x0,
        method="L-BFGS-B",
        bounds=[bounds_s] * len(orders),
    )
    return np.asarray(sol.x), float(sol.fun)


def optimize_per_order(
    p0: np.ndarray,
    counts: Mapping[int, int],
    eta: float,
    omega0_rad_s: float,
    bounds_s: tuple[float, float],
    *,
    repump: np.ndarray | None = None,
) -> tuple[list[SidebandPulse], float]:
    """One pulse time per sideband order, higher orders first, each minimizing the occupation after its own block with
    the earlier blocks fixed; returns (pulses, final nbar)."""
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


def pi_time_s(omega0_rad_s: float, eta: float, n: int, order: int) -> float:
    """pi/Omega_{n,n-k}: the pulse that empties |n> on the k-th sideband."""
    om = sideband_rabi_rad_s(omega0_rad_s, eta, n, order)
    if om <= 0.0:
        raise ValueError(f"|{n}> has no {order}-th red sideband")
    return math.pi / om


def sideband_excitations(
    p: np.ndarray, eta: float, omega0_rad_s: float, order: int, duration_s: float
) -> tuple[float, float]:
    """(P_rsb, P_bsb): the excitation after a pulse of ``duration_s`` on the k-th red and blue sidebands, exact in eta."""
    d = p.size
    rsb = 0.0
    bsb = 0.0
    for n in range(d):
        rsb += p[n] * math.sin(0.5 * sideband_rabi_rad_s(omega0_rad_s, eta, n, order) * duration_s) ** 2
        up = omega0_rad_s * rabi_matrix_element(n + order, n, eta)
        bsb += p[n] * math.sin(0.5 * up * duration_s) ** 2
    return rsb, bsb


def sideband_ratio(p: np.ndarray, eta: float, omega0_rad_s: float, order: int, duration_s: float) -> float:
    rsb, bsb = sideband_excitations(p, eta, omega0_rad_s, order, duration_s)
    if bsb <= 0.0:
        raise ValueError("no blue-sideband excitation: the ratio is undefined")
    return rsb / bsb


def thermal_ratio(nbar: float, order: int) -> float:
    """[nbar/(nbar + 1)]^k, the exact thermal sideband ratio for every pulse duration and eta (Turchette et al. 2000)."""
    return (nbar / (nbar + 1.0)) ** order


def nbar_from_ratio(ratio: float, order: int) -> float:
    """nbar = R^{1/k}/(1 - R^{1/k}), the inversion of the thermal ratio (thermal states only)."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("the ratio lies in [0, 1)")
    r = float(ratio ** (1.0 / order))
    return r / (1.0 - r)


def time_averaged_rsb_signal(p: np.ndarray, order: int) -> float:
    """(1/2) sum_{n >= m} p(n): the time-averaged m-th red-sideband excitation (Rasmusson et al. 2021)."""
    return 0.5 * float(np.sum(p[order:]))


def populations_from_tail_sums(signals: Sequence[float]) -> np.ndarray:
    """p(m) = 2 (S_m - S_{m+1}) from the time-averaged signals S_1, S_2, ... on successive orders; the last entry is 2 S_last."""
    s = np.asarray(signals, dtype=float)
    out = 2.0 * np.diff(-s)
    return np.concatenate([out, [2.0 * s[-1]]])


def blue_sideband_flopping(
    p: np.ndarray,
    eta: float,
    omega0_rad_s: float,
    times_s: np.ndarray,
    *,
    decoherence_per_s: Callable[[int], float] | None = None,
) -> np.ndarray:
    """P_down(tau) = (1/2)(1 + sum_n P_n e^{-gamma_n tau} cos(Omega_{n+1,n} tau)) (Wineland 1998 Eq. 42)."""
    t = np.asarray(times_s, dtype=float)
    out = np.full(t.shape, 0.5)
    for n in range(p.size):
        om = omega0_rad_s * rabi_matrix_element(n + 1, n, eta)
        g = 0.0 if decoherence_per_s is None else decoherence_per_s(n)
        out += 0.5 * p[n] * np.exp(-g * t) * np.cos(om * t)
    return out


def invert_flopping(
    times_s: np.ndarray,
    signal: np.ndarray,
    eta: float,
    omega0_rad_s: float,
    d: int,
    *,
    decoherence_per_s: Callable[[int], float] | None = None,
) -> np.ndarray:
    """Non-negative least squares for P_n from a blue-sideband flopping record on the cosine basis (normalized to sum one)."""
    t = np.asarray(times_s, dtype=float)
    basis = np.zeros((t.size, d))
    for n in range(d):
        om = omega0_rad_s * rabi_matrix_element(n + 1, n, eta)
        g = 0.0 if decoherence_per_s is None else decoherence_per_s(n)
        basis[:, n] = 0.5 * np.exp(-g * t) * np.cos(om * t)
    p, _ = nnls(basis, np.asarray(signal, dtype=float) - 0.5)
    total = float(np.sum(p))
    return np.asarray(p / total) if total > 0.0 else np.asarray(p)


def double_thermal_fit(
    p: np.ndarray, *, initial: tuple[float, float, float] = (0.5, 0.1, 5.0)
) -> tuple[float, float, float]:
    """(a, nbar_l, nbar_h) of p = a p_th(nbar_l) + (1 - a) p_th(nbar_h) by least squares (the post-cooling fit)."""
    d = p.size

    def resid(x: np.ndarray) -> np.ndarray:
        a, nl, nh = x
        model = a * thermal_populations(nl, d) + (1.0 - a) * thermal_populations(nh, d)
        return np.asarray(model - p)

    sol = least_squares(resid, np.asarray(initial), bounds=([0.0, 0.0, 0.0], [1.0, np.inf, np.inf]))
    a, nl, nh = (float(v) for v in sol.x)
    return a, nl, nh


__all__ = [
    "SidebandPulse",
    "accumulation_centre",
    "apply_pulses",
    "blue_sideband_flopping",
    "double_thermal_fit",
    "invert_flopping",
    "laguerre_first_zero",
    "mean_occupation",
    "monroe_half_rabi",
    "nbar_from_ratio",
    "optimize_durations",
    "optimize_per_order",
    "optimize_shared_duration",
    "pi_time_s",
    "populations_from_tail_sums",
    "quenched_floor",
    "raman_two_photon_rabi_rad_s",
    "repump_kernel",
    "sideband_excitations",
    "sideband_rabi_rad_s",
    "sideband_ratio",
    "stenholm_floor_half_width",
    "stranded_index",
    "stranded_population",
    "thermal_distribution",
    "thermal_ratio",
    "time_averaged_rsb_signal",
    "transfer_matrix",
    "trapping_pulse_areas",
    "truncated_tail",
]
