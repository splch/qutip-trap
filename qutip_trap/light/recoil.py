"""Spontaneous-emission recoil kernel for one emitting ion and one motional mode (PLAN.md Section 4.2.8; M3a).

A photon of wavevector k_em emitted in the direction k_hat displaces the motional state by exp(-i k_em k_hat . x),
x = e_m x0 (a + a^dagger) along the mode axis e_m, so the jump operator of a decay carries D(-i eta_em u) with
eta_em = k_em x0 and u = k_hat . e_m. QuTiP needs a finite set of collapse operators, so the emission pattern is
discretized (Section 4.2.8, "Implementation as collapse operators"). Three discretizations are provided:

- ``vector``: a product quadrature over the sphere (Gauss-Legendre in cos theta, uniform in the azimuth) with, per
  direction, the two transverse polarizations, and the jump operator in the VECTOR form
  sum_q conj(eps_lambda . e_q) T_q^- exp(-i k k_hat . x), which sums over the two polarizations to the scalar
  patterns N_0 = (3/8 pi) sin^2 theta and N_{+-1} = (3/16 pi)(1 + cos^2 theta) when one q decays and carries the
  q-coherence tensor otherwise; no scalar alpha is hard-coded and every alpha below is DERIVED from it in the tests.
- ``marginal``: the one-dimensional reduction for a decay of definite polarization q, the marginal f~_q(u) of the
  pattern along the mode axis at angle chi to B_hat, integrated on 16 Gauss-Legendre nodes (exact for these
  quadratic marginals): f~_pi = (3/4)[1 - u^2 cos^2 chi - (1 - u^2) sin^2 chi / 2] and
  f~_sigma = (3/8)[1 + u^2 cos^2 chi + (1 - u^2) sin^2 chi / 2], which reduce to (3/4)(1 - u^2), (3/8)(1 + u^2),
  (9 - 3u^2)/16 and the isotropic 1/2 in the principal orientations (Section 4.2.8).
- ``minimal``: the +-k_hat pair with weights alpha/2 each plus the weight 1 - alpha on a no-recoil operator, exact in
  the first and second moments, the default for Lamb-Dicke-regime cooling and the sideband floor.

Conventions (Section 13, "Recoil angular factor", "Emission-pattern normalization"): the three-dimensional density
is normalized to one over the sphere, the one-dimensional marginal to one on [-1, 1], alpha_m = int dOmega N(k_hat)
(k_hat . e_m)^2 (1/5 for pi along B, 2/5 for sigma along B or pi perpendicular, 3/10 for sigma perpendicular, 1/3
isotropic), and the recoil heating per photon into the mode is alpha_m eta_em^2 quanta. Three assertions run before
any solve (Section 4.2.8): sum_j p_j = 1 and sum_j p_j u_j^2 = alpha to 1e-12 (a failure is a quadrature bug, never
a physics one), sum_k C_k^dagger C_k = Gamma |e><e| (x) 1 because the kicks are unitary, and the single-kick
expectation <n> = alpha eta_em^2 with <p> = 0. The per-ion participation b_{i,m} and the product over the modes of one
axis (the recoil-energy unit test) are milestone M3.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss

from qutip_trap.species.polarization import Vec, spherical_basis

RecoilMode = Literal["off", "minimal", "marginal", "vector"]
PatternQ = Literal[-1, 0, 1]

MOMENT_TOLERANCE = 1e-12
"""Tolerance of the quadrature identities sum p = 1 and sum p u^2 = alpha (Section 4.2.8)."""


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
    """alpha_m = int dOmega N_q(k_hat)(k_hat . e_m)^2 for a mode axis at angle chi to B_hat; ``q=None`` is isotropic (1/3).

    The emission tensor of a pattern is diagonal, diag(alpha_perp, alpha_perp, alpha_par) about B_hat with trace one,
    so alpha(chi) = alpha_par cos^2 chi + alpha_perp sin^2 chi: pi (1/5, 2/5), sigma (2/5, 3/10) (Section 4.2.8).
    """
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
    """f~_q(u): the density of u = k_hat . e_m on [-1, 1] for a mode axis at angle chi to B_hat (normalized to one).

    Obtained by integrating N_q over the azimuth about e_m with k_hat . B_hat = u cos chi + sqrt(1 - u^2) sin chi cos phi.
    """
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
    """Nodes u_j = k_hat . e_m and weights p_j of a one-dimensional recoil quadrature with its moments checked."""

    nodes: np.ndarray
    weights: np.ndarray
    alpha: float
    """The second moment sum_j p_j u_j^2, equal to alpha_m."""

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
            "three Gauss-Legendre nodes are the minimum for the second moment (two return 1/3 for every quadratic "
            "marginal, Section 4.2.8)"
        )
    u, w = leggauss(n_nodes)
    p = w * np.asarray(marginal(q, cos_chi, u), dtype=float)
    quad = Quadrature1D(nodes=u, weights=p, alpha=float(np.sum(p * u * u)))
    quad.check(angular_factor(q, cos_chi))
    return quad


def minimal_quadrature(alpha: float) -> Quadrature1D:
    """Nodes (-1, 0, +1) with weights (alpha/2, 1 - alpha, alpha/2): exact first and second moments (Section 4.2.8)."""
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
    """Directions k_hat_k (laboratory frame, unit rows) with weights summing to 4 pi, and two transverse polarizations each."""

    directions: np.ndarray
    weights: np.ndarray
    polarizations: np.ndarray
    """(K, 2, 3) real orthonormal transverse vectors (theta_hat, phi_hat) per direction."""

    @property
    def size(self) -> int:
        return int(self.directions.shape[0])


def direction_quadrature(
    n_theta: int = 6, n_phi: int = 8, axis: Vec = (0.0, 0.0, 1.0)
) -> DirectionQuadrature:
    """A product quadrature exact for polynomials of degree 2 n_theta - 1 in cos theta and trigonometric degree n_phi - 1.

    ``axis`` is the polar axis of the grid (any unit vector; the result is exact for the low-order patterns whatever
    the axis, and the tests use it to show the derived alpha is orientation independent).
    """
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
            k = s * math.cos(ph) * x + s * math.sin(ph) * y + c * z
            theta_hat = c * math.cos(ph) * x + c * math.sin(ph) * y - s * z
            phi_hat = -math.sin(ph) * x + math.cos(ph) * y
            dirs.append(k)
            weights.append(float(w) * 2.0 * math.pi / n_phi)
            pols.append(np.stack([theta_hat, phi_hat]))
    return DirectionQuadrature(np.array(dirs), np.array(weights), np.array(pols))


@dataclass(frozen=True)
class VectorChannel:
    """One direction-and-polarization emission channel: weight w_k (3/8 pi), the amplitudes conj(eps . e_q) per q, and u."""

    weight: float
    """w_k 3/(8 pi): multiplies g_LU |sum_q c_q T_q^-|^2 to the channel's rate share."""
    amplitudes: tuple[complex, complex, complex]
    """conj(eps_lambda . e_q) for q = -1, 0, +1."""
    u: float
    """k_hat . e_m, the recoil projection on the mode axis."""


def vector_channels(
    b_hat: Vec, mode_axis: Vec, *, n_theta: int = 6, n_phi: int = 8, grid_axis: Vec | None = None
) -> tuple[VectorChannel, ...]:
    """The vector-form emission channels over a direction quadrature (Section 4.2.8, the form the module builds).

    For each direction k_hat and transverse polarization eps the jump operator is
    sqrt(w (3/8 pi) g) sum_q conj(eps . e_q) T_q^- exp(-i k k_hat . x), with e_q the spherical basis about B_hat
    (Section 13 row "Polarization components") and g = omega^3/(3 pi eps0 hbar c^3). Summing |eps . e_q|^2 over the two
    polarizations gives 1 - |k_hat . e_q|^2, which is the scalar pattern of a pure q decay; the cross terms between
    different q are the q-coherence tensor that only a coherent superposition of upper sublevels feels.
    """
    e_m = _unit(mode_axis, "mode_axis")
    quad = direction_quadrature(n_theta, n_phi, axis=grid_axis if grid_axis is not None else b_hat)
    e_minus, e_zero, e_plus = spherical_basis(b_hat)
    basis = (e_minus, e_zero, e_plus)
    out: list[VectorChannel] = []
    for k, w, pols in zip(quad.directions, quad.weights, quad.polarizations):
        u = float(np.dot(k, e_m))
        for eps in pols:
            amps = tuple(complex(np.conj(np.dot(eps.astype(complex), e))) for e in basis)
            out.append(VectorChannel(float(w) * 3.0 / (8.0 * math.pi), amps, u))  # type: ignore[arg-type]
    return tuple(out)


def derived_angular_factors(channels: Sequence[VectorChannel]) -> dict[int, float]:
    """alpha_q = sum_channels weight |c_q|^2 u^2 for each pure q: the second moments the vector form implies (no scalar input)."""
    out: dict[int, float] = {}
    for idx, q in enumerate((-1, 0, 1)):
        out[q] = float(sum(ch.weight * abs(ch.amplitudes[idx]) ** 2 * ch.u**2 for ch in channels))
    return out


def derived_pattern_norms(channels: Sequence[VectorChannel]) -> dict[int, float]:
    """sum_channels weight |c_q|^2 for each q, which must be exactly one (the pattern normalization)."""
    return {
        q: float(sum(ch.weight * abs(ch.amplitudes[idx]) ** 2 for ch in channels))
        for idx, q in enumerate((-1, 0, 1))
    }


def recoil_lamb_dicke(k_em_rad_per_m: float, x0_m: float) -> float:
    """eta_em = k_em x0, the emitted photon's Lamb-Dicke parameter before the angular projection (Section 4.2.8)."""
    if k_em_rad_per_m <= 0.0 or x0_m <= 0.0:
        raise ValueError("k_em and x0 are positive")
    return k_em_rad_per_m * x0_m


def recoil_quanta_per_photon(alpha: float, eta_em: float) -> float:
    """alpha eta_em^2: the mean recoil heating per emitted photon into one mode (Section 4.2.8)."""
    return alpha * eta_em**2


def free_recoil_energy_j(k_em_rad_per_m: float, mass_kg: float) -> float:
    """(hbar k)^2 / (2 m), the free-ion recoil energy that the modes of one axis share as alpha (the M3 unit test)."""
    from qutip_trap.units import HBAR_J_S

    return (HBAR_J_S * k_em_rad_per_m) ** 2 / (2.0 * mass_kg)


__all__ = [
    "MOMENT_TOLERANCE",
    "DirectionQuadrature",
    "PatternQ",
    "Quadrature1D",
    "RecoilMode",
    "VectorChannel",
    "angular_factor",
    "derived_angular_factors",
    "derived_pattern_norms",
    "direction_quadrature",
    "free_recoil_energy_j",
    "marginal",
    "marginal_quadrature",
    "minimal_quadrature",
    "pattern_density",
    "recoil_lamb_dicke",
    "recoil_quanta_per_photon",
    "vector_channels",
]
