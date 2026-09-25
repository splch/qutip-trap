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
expectation <n> = alpha eta_em^2 with <p> = 0.

Milestone M3 adds the per-ion participation and the joint multi-mode kick (Section 4.2.8, "The recoil term"): the
emitting ion i's position resolves into modes as k_hat . x_i = sum_m c_{i,m} (k_hat . e_m) x0_{i,m} (a_m + a_m^dagger)
with the mass-weighted eigenvector component c_{i,m} of Section 4.1.3 (b_{i,m} for equal masses) and x0 with the
ion's OWN mass (Section 13, "Lamb-Dicke base"), so one sampled direction kicks every mode at once through the product
prod_m D_m(-i k_em c_{i,m} x0_{i,m} k_hat . e_m) (``multi_mode_kick``), the recoil heating per photon into mode m is
alpha_m c_{i,m}^2 (k_em x0_{i,m})^2 quanta (``recoil_heating_quanta``) and, summed over the N modes of one axis family,
the deposited energy equals alpha_axis (hbar k_em)^2/(2 m_i) exactly because the eigenvectors of a family are complete
(``recoil_energy_ratio``, the unit test the plan names). ``recoil_kernel_matrix`` is the Fock-basis kernel of one emitted
photon, sum_j p_j |<n|D(-i eta u_j)|n'>|^2, column stochastic, exact in eta through the Laguerre elements (Section 4.3.1);
``recoil_velocity_m_per_s`` and ``recoil_temperature_k`` are Steck's free-atom regressions (Section 9.3, "Recoil unit
regression": k_B T_r = (hbar k)^2/m = 2 hbar omega_r, no factor 2 in V_p -> D_p t).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss

from qutip_trap.species.polarization import Vec, spherical_basis

if TYPE_CHECKING:
    import qutip as qt

    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.trap.crystal import Crystal

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


def recoil_velocity_m_per_s(k_rad_per_m: float, mass_kg: float) -> float:
    """v_r = hbar k / m (Steck Eq. 1.94): 133Cs 852 nm 3.5 mm/s, 87Rb 780 nm 5.9 mm/s (Section 9.3)."""
    from qutip_trap.units import HBAR_J_S

    if k_rad_per_m <= 0.0 or mass_kg <= 0.0:
        raise ValueError("k and the mass are positive")
    return HBAR_J_S * k_rad_per_m / mass_kg


def recoil_temperature_k(k_rad_per_m: float, mass_kg: float) -> float:
    """k_B T_r = (hbar k)^2/m = 2 hbar omega_r (Steck Eqs. 1.113, 5.430; Section 13 "Momentum diffusion and recoil temperature").

    198 nK for 133Cs at 852 nm and 362 nK for 87Rb at 780 nm; the k_B T_r = hbar omega_r found elsewhere is half this.
    """
    from qutip_trap.units import HBAR_J_S, K_B_J_PER_K

    return (HBAR_J_S * k_rad_per_m) ** 2 / (mass_kg * K_B_J_PER_K)


# ---- per-ion participation and the joint multi-mode kick (M3) ---------------------------------------------------------------


def recoil_projections(
    crystal: Crystal, ion: int, k_em_rad_per_m: float, k_hat: Vec, *, modes: Sequence[int] | None = None
) -> dict[int, float]:
    """eta_{i,m}(k_hat) = k_em c_{i,m} x0_{i,m} (k_hat . e_m) for every mode: the displacement each mode receives from a photon
    of wavenumber k_em emitted along k_hat by ion i (Section 4.2.8, "The recoil term"; no micromotion factor on emission)."""
    k = k_em_rad_per_m * _unit(k_hat, "k_hat")
    which = range(len(crystal.modes)) if modes is None else modes
    return {m: crystal.lamb_dicke(ion, m, k, micromotion=None) for m in which}


def emission_lamb_dicke(crystal: Crystal, ion: int, k_em_rad_per_m: float, mode: int) -> float:
    """eta_em,{i,m} = k_em |c_{i,m}| x0_{i,m}: the emitted photon's Lamb-Dicke parameter on mode m BEFORE the angular
    projection, so that eta~^2 = alpha_m eta_em^2 (Section 4.2.8, "Angular factors")."""
    e = np.asarray(crystal.modes[mode].e_hat, dtype=float)
    return abs(crystal.lamb_dicke(ion, mode, k_em_rad_per_m * e, micromotion=None))


def recoil_heating_quanta(
    crystal: Crystal, ion: int, k_em_rad_per_m: float, q: int | None, b_hat: Vec
) -> dict[int, float]:
    """alpha_m c_{i,m}^2 (k_em x0_{i,m})^2 per mode: the mean recoil heating of one photon of polarization type q (None:
    isotropic) emitted by ion i, alpha_m evaluated at the angle between the mode axis and B_hat (Section 4.2.8)."""
    b = _unit(b_hat, "B_hat")
    out: dict[int, float] = {}
    for m, mode in enumerate(crystal.modes):
        cos_chi = float(np.dot(np.asarray(mode.e_hat, dtype=float), b))
        out[m] = angular_factor(q, cos_chi) * emission_lamb_dicke(crystal, ion, k_em_rad_per_m, m) ** 2
    return out


def recoil_energy_ratio(
    crystal: Crystal, ion: int, k_em_rad_per_m: float, q: int | None, b_hat: Vec, family: str
) -> float:
    """sum over the modes of one axis family of alpha_m c^2 (k x0)^2 hbar omega_m, over alpha_axis (hbar k)^2/(2 m_i): exactly 1.

    The modes of a family share the axis e_hat, so alpha_m = alpha_axis, and x0_{i,m}^2 hbar omega_m = hbar^2/(2 m_i) for
    every mode; completeness of the family's mass-weighted eigenvectors, sum_m c_{i,m}^2 = 1, then gives the free-ion recoil
    energy along that axis (Section 4.2.8: "the module's unit test"; Section 9.3 "Recoil quadrature identities").
    """
    from qutip_trap.units import HBAR_J_S

    quanta = recoil_heating_quanta(crystal, ion, k_em_rad_per_m, q, b_hat)
    members = [m for m, mode in enumerate(crystal.modes) if mode.family == family]
    if not members:
        raise ValueError(f"the crystal has no {family!r} modes")
    e = np.asarray(crystal.modes[members[0]].e_hat, dtype=float)
    alpha_axis = angular_factor(q, float(np.dot(e, _unit(b_hat, "B_hat"))))
    energy = sum(quanta[m] * HBAR_J_S * crystal.modes[m].omega_rad_s for m in members)
    free = alpha_axis * free_recoil_energy_j(k_em_rad_per_m, float(crystal.masses_kg[ion]))
    return float(energy / free)


def multi_mode_kick(space: HilbertSpace, etas_by_mode: Mapping[int, float]) -> qt.Qobj:
    """prod_m D_m(-i eta_m) on the space's motional factors, identity on the ions: the joint kick of one sampled direction
    (Section 4.2.8), each factor the exact expm displacement of Section 5.1.1 (ENR members through the summed generator)."""
    factors: dict[int, qt.Qobj] = {}
    enr: dict[int, float] = {}
    for mode, eta in etas_by_mode.items():
        if space.mode_class(mode) == "resolved":
            factors[space.mode_factor(mode)] = space.displacement_factor(mode, -float(eta))
        elif space.mode_class(mode) == "enr":
            enr[mode] = -float(eta)
        else:
            raise ValueError(
                f"mode {mode} is frozen: a frozen mode receives no kick operator (its nbar is bookkeeping)"
            )
    if enr:
        assert space.enr_factor is not None
        factors[space.enr_factor] = space.enr_displacement(enr)
    if not factors:
        return space.identity()
    return space.embed_many(factors)


def recoil_kernel_matrix(d: int, eta_em: float, quad: Quadrature1D) -> np.ndarray:
    """K[n, n'] = sum_j p_j |<n|D(-i eta_em u_j)|n'>|^2 on d Fock levels: the Fock-population kernel of one emitted photon
    along a mode, exact in eta through the Laguerre elements (Section 4.2.8, "the recoil kernel over Delta n").

    Column stochastic up to the population the truncation loses from the top columns (reported by the caller as the boundary
    population); the mean kick of column n' is sum_n (n - n') K[n, n'] = alpha eta_em^2 exactly for every n' in the
    infinite space (the Kraus map's single-kick expectation generalized to |n'>).
    """
    from qutip_trap.hilbert.operators import displacement_matrix_analytic

    if d < 2:
        raise ValueError("at least two Fock levels")
    kernel = np.zeros((d, d))
    for u, p in zip(quad.nodes, quad.weights):
        if p <= 0.0:
            continue
        kernel += p * np.abs(displacement_matrix_analytic(d, -1j * eta_em * float(u))) ** 2
    return kernel
