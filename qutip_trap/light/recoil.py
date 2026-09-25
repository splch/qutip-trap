"""Spontaneous-emission recoil: emission patterns, angular factors, quadratures and kernels (PLAN.md Section 4.2.8).

A photon of wavevector k_em emitted along k_hat displaces the motion by exp(-i k_em k_hat . x), so a decay's jump operator
carries D(-i eta_em u) with eta_em = k_em x0 and u = k_hat . e_m. Three discretizations: ``vector``, a direction quadrature
with two transverse polarizations and the jump operator sum_q conj(eps . e_q) T_q^- exp(-i k k_hat . x) (it sums to the
scalar patterns N_0 = (3/8 pi) sin^2 theta, N_+-1 = (3/16 pi)(1 + cos^2 theta)); ``marginal``, Gauss-Legendre nodes on the
one-dimensional marginal of a definite-q pattern along the mode axis; ``minimal``, the +-k_hat pair with weight alpha/2
each plus a no-recoil node, exact in the first two moments. alpha_m = int N (k_hat . e_m)^2 is 1/5 for pi along B, 2/5 for
sigma along B or pi across, 3/10 for sigma across, 1/3 isotropic, and each photon deposits alpha_m eta_em^2 quanta. In a
crystal the emitting ion's displacement resolves as sum_m c_{i,m} (k_hat . e_m) x0_{i,m} (a_m + a_m^dag), with the ion's
own mass in x0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss

from qutip_trap.species.polarization import Vec, spherical_basis
from qutip_trap.units import HBAR_J_S

if TYPE_CHECKING:
    from qutip_trap.trap.crystal import Crystal

RecoilMode = Literal["off", "minimal", "marginal", "vector"]

MOMENT_TOLERANCE = 1e-12
"""Tolerance of the quadrature identities sum p = 1 and sum p u^2 = alpha (a failure is a quadrature bug)."""


def _unit(v: Vec, what: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if not math.isclose(n, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{what} must be a unit vector (norm {n})")
    return arr / n


def pattern_density(q: int, cos_theta: np.ndarray | float) -> np.ndarray | float:
    """N_q(k_hat): (3/8 pi) sin^2 theta for pi (q = 0), (3/16 pi)(1 + cos^2 theta) for sigma (q = +-1); theta to B_hat."""
    c2 = np.square(cos_theta)
    if q == 0:
        return 3.0 / (8.0 * math.pi) * (1.0 - c2)
    if q in (-1, 1):
        return 3.0 / (16.0 * math.pi) * (1.0 + c2)
    raise ValueError("the polarization index of a decay is -1, 0 or +1")


def angular_factor(q: int | None, cos_chi: float) -> float:
    """alpha_m = alpha_par cos^2 chi + alpha_perp sin^2 chi for a mode axis at angle chi to B_hat: pi (1/5, 2/5), sigma
    (2/5, 3/10); ``q=None`` is isotropic (1/3)."""
    c2 = float(cos_chi) ** 2
    if not 0.0 <= c2 <= 1.0 + 1e-12:
        raise ValueError("cos chi must lie in [-1, 1]")
    if q is None:
        return 1.0 / 3.0
    if q == 0:
        return 0.2 * c2 + 0.4 * (1.0 - c2)
    if q in (-1, 1):
        return 0.4 * c2 + 0.3 * (1.0 - c2)
    raise ValueError("the polarization index of a decay is -1, 0 or +1")


def marginal(q: int | None, cos_chi: float, u: np.ndarray | float) -> np.ndarray | float:
    """f~_q(u): the density of u = k_hat . e_m on [-1, 1] for a mode axis at angle chi to B_hat (normalized to one)."""
    c2 = float(cos_chi) ** 2
    s2 = 1.0 - c2
    uu = np.square(u)
    if q is None:
        return 0.5 * np.ones_like(uu) if isinstance(uu, np.ndarray) else 0.5
    if q == 0:
        return 0.75 * (1.0 - uu * c2 - (1.0 - uu) * s2 / 2.0)
    if q in (-1, 1):
        return 0.375 * (1.0 + uu * c2 + (1.0 - uu) * s2 / 2.0)
    raise ValueError("the polarization index of a decay is -1, 0 or +1")


@dataclass(frozen=True)
class Quadrature1D:
    """Nodes u_j = k_hat . e_m and weights p_j of a one-dimensional recoil quadrature; ``alpha`` = sum p_j u_j^2."""

    nodes: np.ndarray
    weights: np.ndarray
    alpha: float

    def check(self, alpha_expected: float) -> None:
        total = float(np.sum(self.weights))
        second = float(np.sum(self.weights * self.nodes**2))
        if abs(total - 1.0) > MOMENT_TOLERANCE:
            raise AssertionError(f"recoil quadrature weights sum to {total}, not 1 (quadrature bug)")
        if abs(second - alpha_expected) > MOMENT_TOLERANCE:
            raise AssertionError(
                f"recoil quadrature second moment {second} differs from alpha = {alpha_expected} (quadrature bug)"
            )


def marginal_quadrature(q: int | None, cos_chi: float, n_nodes: int = 16) -> Quadrature1D:
    """Gauss-Legendre nodes on [-1, 1] with p_j = w_j f~_q(u_j); three nodes are the minimum for the second moment."""
    if n_nodes < 3:
        raise ValueError(
            "three Gauss-Legendre nodes are the minimum for the second moment (two return 1/3 for every quadratic marginal)"
        )
    u, w = leggauss(n_nodes)
    p = w * np.asarray(marginal(q, cos_chi, u), dtype=float)
    quad = Quadrature1D(nodes=u, weights=p, alpha=float(np.sum(p * u * u)))
    quad.check(angular_factor(q, cos_chi))
    return quad


def minimal_quadrature(alpha: float) -> Quadrature1D:
    """Nodes (-1, 0, +1) with weights (alpha/2, 1 - alpha, alpha/2): exact first and second moments."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha lies in (0, 1)")
    quad = Quadrature1D(
        nodes=np.array([-1.0, 0.0, 1.0]),
        weights=np.array([alpha / 2.0, 1.0 - alpha, alpha / 2.0]),
        alpha=alpha,
    )
    quad.check(alpha)
    return quad


@dataclass(frozen=True)
class DirectionQuadrature:
    """Directions k_hat_k (laboratory frame, unit rows), weights summing to 4 pi, and per direction two real orthonormal
    transverse polarizations (theta_hat, phi_hat), shape (K, 2, 3)."""

    directions: np.ndarray
    weights: np.ndarray
    polarizations: np.ndarray

    @property
    def size(self) -> int:
        return int(self.directions.shape[0])


def direction_quadrature(
    n_theta: int = 6, n_phi: int = 8, axis: Vec = (0.0, 0.0, 1.0)
) -> DirectionQuadrature:
    """Gauss-Legendre in cos theta times uniform in phi about ``axis``: exact to degree 2 n_theta - 1 and n_phi - 1."""
    if n_theta < 3 or n_phi < 5:
        raise ValueError("n_theta >= 3 and n_phi >= 5 are needed for the second moments of a dipole pattern")
    z = _unit(axis, "axis")
    seed = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = seed - np.dot(seed, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ct, wt = leggauss(n_theta)
    phis = 2.0 * math.pi * (np.arange(n_phi) + 0.5) / n_phi
    dirs: list[np.ndarray] = []
    weights: list[float] = []
    pols: list[np.ndarray] = []
    for c, w in zip(ct, wt):
        s = math.sqrt(max(0.0, 1.0 - c * c))
        for ph in phis:
            dirs.append(s * math.cos(ph) * x + s * math.sin(ph) * y + c * z)
            weights.append(float(w) * 2.0 * math.pi / n_phi)
            theta_hat = c * math.cos(ph) * x + c * math.sin(ph) * y - s * z
            pols.append(np.stack([theta_hat, -math.sin(ph) * x + math.cos(ph) * y]))
    return DirectionQuadrature(np.array(dirs), np.array(weights), np.array(pols))


@dataclass(frozen=True)
class VectorChannel:
    """One direction-and-polarization emission channel: weight w_k 3/(8 pi), amplitudes conj(eps . e_q) for q = -1, 0, +1,
    and u = k_hat . e_m."""

    weight: float
    amplitudes: tuple[complex, complex, complex]
    u: float


def vector_channels(
    b_hat: Vec, mode_axis: Vec, *, n_theta: int = 6, n_phi: int = 8, grid_axis: Vec | None = None
) -> tuple[VectorChannel, ...]:
    """The vector-form emission channels over a direction quadrature: the jump operator of direction k_hat and polarization
    eps is sqrt(w (3/8 pi) g) sum_q conj(eps . e_q) T_q^- exp(-i k k_hat . x), e_q the spherical basis about B_hat; summing
    |eps . e_q|^2 over the two polarizations gives 1 - |k_hat . e_q|^2, the scalar pattern of a pure q decay."""
    e_m = _unit(mode_axis, "mode_axis")
    quad = direction_quadrature(n_theta, n_phi, axis=grid_axis if grid_axis is not None else b_hat)
    basis = spherical_basis(b_hat)
    out: list[VectorChannel] = []
    for k, w, pols in zip(quad.directions, quad.weights, quad.polarizations):
        u = float(np.dot(k, e_m))
        for eps in pols:
            amps = tuple(complex(np.conj(np.dot(eps.astype(complex), e))) for e in basis)
            out.append(VectorChannel(float(w) * 3.0 / (8.0 * math.pi), amps, u))  # type: ignore[arg-type]
    return tuple(out)


def recoil_lamb_dicke(k_em_rad_per_m: float, x0_m: float) -> float:
    """eta_em = k_em x0, the emitted photon's Lamb-Dicke parameter before the angular projection."""
    if k_em_rad_per_m <= 0.0 or x0_m <= 0.0:
        raise ValueError("k_em and x0 are positive")
    return k_em_rad_per_m * x0_m


def free_recoil_energy_j(k_em_rad_per_m: float, mass_kg: float) -> float:
    """(hbar k)^2/(2 m): the free-ion recoil energy, shared by the modes of one axis as alpha."""
    return (HBAR_J_S * k_em_rad_per_m) ** 2 / (2.0 * mass_kg)


def recoil_projections(
    crystal: Crystal, ion: int, k_em_rad_per_m: float, k_hat: Vec, *, modes: Sequence[int] | None = None
) -> dict[int, float]:
    """eta_{i,m}(k_hat) = k_em c_{i,m} x0_{i,m} (k_hat . e_m) per mode: the kick each mode receives from a photon emitted
    along k_hat by ion i (no micromotion factor on emission)."""
    k = k_em_rad_per_m * _unit(k_hat, "k_hat")
    return {
        m: crystal.lamb_dicke(ion, m, k, micromotion=None)
        for m in (range(len(crystal.modes)) if modes is None else modes)
    }


def emission_lamb_dicke(crystal: Crystal, ion: int, k_em_rad_per_m: float, mode: int) -> float:
    """eta_em,{i,m} = k_em |c_{i,m}| x0_{i,m}, before the angular projection, so that eta~^2 = alpha_m eta_em^2."""
    e = np.asarray(crystal.modes[mode].e_hat, dtype=float)
    return abs(crystal.lamb_dicke(ion, mode, k_em_rad_per_m * e, micromotion=None))


def recoil_kernel_matrix(d: int, eta_em: float, quad: Quadrature1D) -> np.ndarray:
    """K[n, n'] = sum_j p_j |<n|D(-i eta_em u_j)|n'>|^2 on d Fock levels: the population kernel of one emitted photon, exact
    in eta; column stochastic up to what the truncation loses, with mean kick alpha eta_em^2 per column."""
    from qutip_trap.hilbert.operators import displacement_matrix_analytic

    if d < 2:
        raise ValueError("at least two Fock levels")
    kernel = np.zeros((d, d))
    for u, p in zip(quad.nodes, quad.weights):
        if p > 0.0:
            kernel += p * np.abs(displacement_matrix_analytic(d, -1j * eta_em * float(u))) ** 2
    return kernel
