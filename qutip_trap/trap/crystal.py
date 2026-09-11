"""Ion crystal and normal modes (PLAN.md Sections 4.1.2, 4.1.3, 4.1.7; Appendix E; milestone M1).

Mode addressing (Section 13, row "Mode index"): every ``mode: int`` anywhere in the package is a position in
``Crystal.modes``, ordered axial, transverse_1, transverse_2, ascending frequency within a family, with
unit-norm mass-weighted eigenvectors whose last component is positive. The Lamb-Dicke parameter
eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 is computed HERE and nowhere else
(Section 5.7), with the ion's own mass and the mass-weighted eigenvector component (Section 4.1.7).

Physics, in the order the module applies it:

- Equilibrium (Section 4.1.2): N ions of charge Z_i e in the potential energy
  V = sum_i [(1/2) sum_a kappa_{i,a} r_{i,a}^2 - Z_i e E . r_i] + sum_{i<j} Z_i Z_j e^2/(4 pi eps0 |r_i - r_j|),
  with kappa_{i,a} = m_i omega_{i,a}^2 the per-ion spring constants along the trap's principal axes (x', y', z),
  solved by Newton iteration with the exact gradient and Hessian from the linear-chain start of James 1998
  (dimensionless u_m - sum_{n<m} 1/(u_m - u_n)^2 + sum_{n>m} 1/(u_m - u_n)^2 = 0, length scale
  l = (Z^2 e^2/(4 pi eps0 M nu^2))^{1/3}, closed forms u = -+(1/2)^{2/3} for N = 2 and -+(5/4)^{1/3}, 0 for N = 3).
  Statics are mass independent for equal charges (Section 4.1.7); a stray field displaces the ions. The
  Hessian at the solution must be positive definite: a linear chain whose transverse Hessian has a
  non-positive eigenvalue has buckled (zigzag) and the module refuses to build a linear-chain mode structure
  (alpha_crit = 2/(mu_N - 1), i.e. (omega_r/omega_z)_crit = 1 and sqrt(12/5) = 1.5492 at N = 2, 3; Marquet 2003).
- Normal modes (Sections 4.1.3, 4.1.7): the generalized eigenproblem K b = omega^2 M b is solved as the
  symmetric problem M^{-1/2} K M^{-1/2} c = omega^2 c with orthonormal MASS-WEIGHTED eigenvectors c (Home 2013
  Eqs. 7-9; Kielpinski 2000; Morigi and Walther 2001; Wubbena 2012); for equal masses c is James's b. The
  physical displacement of ion i in mode k is c_i^{(k)} sqrt(hbar/(2 m_i omega_k)) (a_k + a_k^dag), the standard
  x = x0 (a + a^dag) convention and never James's Kittel form with the factor i (Section 13). For a collinear
  chain the three Cartesian families separate exactly and are solved as three N x N problems, which keeps a
  degenerate radial pair (omega_x = omega_y) from mixing families; axially mu_1 = 1 (centre of mass, b ~ 1/sqrt N)
  and mu_2 = 3 (stretch, b ~ u) for every N, transversely gamma_p = 1/alpha + 1/2 - mu_p/2 with alpha = (omega_z/omega_r)^2
  per family (the two radial families share A's eigenvectors and differ in frequency whenever omega_x != omega_y),
  so the centre-of-mass mode is the HIGHEST transverse mode and the zigzag mode the lowest.
- Parity (Section 4.1.7): in a reflection-symmetric mass array every mode has a definite inversion parity;
  the modes with ANTISYMMETRIC displacement patterns (p = +1 in Morigi and Walther's convention, the central ion
  at rest for odd N) have sum_i c_i/sqrt(m_i) = 0, are exempt from uniform-field noise and uncoolable by a
  centre ion; the in-phase lowest mode is odd under that parity ("even" read as "in-phase" inverts the rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import zeta

from qutip_trap.trap.mathieu import MathieuParameters, c0_wronskian
from qutip_trap.units import ATOMIC_MASS_KG, E_C, EPSILON_0_F_PER_M, HBAR_J_S, TWO_PI

if TYPE_CHECKING:
    from qutip_trap.species.model import Species
    from qutip_trap.trap.model import Trap

Family = Literal["axial", "transverse_1", "transverse_2"]
FAMILY_ORDER: tuple[Family, ...] = ("axial", "transverse_1", "transverse_2")
FAMILY_AXIS: dict[str, int] = {"transverse_1": 0, "transverse_2": 1, "axial": 2}
"""Column of the principal frame (x', y', z) each family moves along."""

K_COULOMB_J_M: float = E_C * E_C / (4.0 * math.pi * EPSILON_0_F_PER_M)
"""e^2/(4 pi eps0) in J m: the Coulomb energy of two unit charges at 1 m."""

_NEWTON_TOL = 1e-13
_NEWTON_MAX = 200


class ZigzagError(ValueError):
    """The linear chain is not a stable configuration in this trap (Section 4.1.2: the module refuses to build a
    linear-chain mode structure when the zigzag criterion is violated)."""


# ---- James 1998: the dimensionless linear chain ------------------------------------------------------------------


def length_scale_m(mass_kg: float, omega_z_rad_s: float, *, charge: int = 1) -> float:
    """l = (Z^2 e^2/(4 pi eps0 M nu^2))^{1/3} (James 1998; Wineland 1998's s with s_2 = 2^{1/3} s, s_3 = (5/4)^{1/3} s)."""
    if mass_kg <= 0.0 or omega_z_rad_s <= 0.0:
        raise ValueError("mass and axial frequency must be positive")
    return float((charge * charge * K_COULOMB_J_M / (mass_kg * omega_z_rad_s**2)) ** (1.0 / 3.0))


def _chain_gradient(u: np.ndarray) -> np.ndarray:
    d = u[:, None] - u[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv2 = np.where(np.eye(len(u), dtype=bool), 0.0, 1.0 / d**2)
    # d/du_m [ (1/2) sum u^2 + sum_{n<m} 1/(u_m - u_n) ] = u_m - sum_{n<m} 1/(u_m-u_n)^2 + sum_{n>m} 1/(u_m-u_n)^2
    return np.asarray(u - np.sum(np.sign(d) * inv2, axis=1))


def axial_hessian_dimensionless(u: np.ndarray) -> np.ndarray:
    """A_mn = 1 + 2 sum_{p != m} 1/|u_m - u_p|^3 (m = n), -2/|u_m - u_n|^3 (m != n) (James Eq. 3.3; Marquet Eq. 2.9)."""
    u = np.asarray(u, dtype=float)
    d = np.abs(u[:, None] - u[None, :])
    with np.errstate(divide="ignore"):
        inv3 = np.where(np.eye(len(u), dtype=bool), 0.0, 1.0 / d**3)
    a = -2.0 * inv3
    a[np.diag_indices(len(u))] = 1.0 + 2.0 * inv3.sum(axis=1)
    return np.asarray(a)


def equilibrium_dimensionless(n: int, *, start: np.ndarray | None = None) -> np.ndarray:
    """The James equilibrium positions u_1 < ... < u_N of N equal charges, by Newton iteration from a scaled uniform start.

    The chain energy is strictly convex on the ordered domain, so the minimum is unique; steps are damped to
    preserve the ordering. Closed forms: N = 2 -> -+(1/2)^{2/3}; N = 3 -> -+(5/4)^{1/3}, 0 (Section 4.1.2).
    """
    if n < 1:
        raise ValueError("N >= 1")
    if n == 1:
        return np.zeros(1)
    if start is None:
        # James's empirical minimum spacing 2.018 N^{-0.559} sets the scale of a uniform start
        spacing = 2.018 * n ** (-0.559)
        u = spacing * (np.arange(n) - (n - 1) / 2.0)
    else:
        u = np.array(start, dtype=float)
    for _ in range(_NEWTON_MAX):
        g = _chain_gradient(u)
        if np.max(np.abs(g)) < _NEWTON_TOL:
            return np.asarray(u, dtype=float)
        step = np.linalg.solve(axial_hessian_dimensionless(u), -g)
        lam = 1.0
        while lam > 1e-6:
            trial = u + lam * step
            if np.all(np.diff(trial) > 0.0) and np.max(np.abs(_chain_gradient(trial))) < np.max(np.abs(g)):
                break
            lam *= 0.5
        u = u + lam * step
    raise RuntimeError("James equilibrium did not converge")


def _fix_sign(vectors: np.ndarray) -> np.ndarray:
    """Sign gauge (Section 13): the last component of every column positive (Marquet et al.'s convention)."""
    v = np.array(vectors, dtype=float)
    for k in range(v.shape[1]):
        last = v[np.max(np.flatnonzero(np.abs(v[:, k]) > 1e-12)), k]
        if last < 0.0:
            v[:, k] = -v[:, k]
    return v


def axial_modes_dimensionless(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu_p ascending, b^{(p)} as columns) of the dimensionless axial Hessian; nu_p = sqrt(mu_p) nu (James Eq. 3.5).

    mu_1 = 1 with b ~ (1, ..., 1)/sqrt N and mu_2 = 3 with b ~ u for every N; mu_3 = 29/5 at N = 3.
    """
    mu, b = np.linalg.eigh(axial_hessian_dimensionless(np.asarray(u, dtype=float)))
    return np.asarray(mu), _fix_sign(b)


def transverse_eigenvalues(mu: np.ndarray, alpha: float) -> np.ndarray:
    """gamma_p = 1/alpha + 1/2 - mu_p/2 with alpha = (omega_z/omega_r)^2 (Marquet 2003); Omega_p = omega_z sqrt(gamma_p)."""
    if alpha <= 0.0:
        raise ValueError("alpha = (omega_z/omega_r)^2 must be positive")
    return np.asarray(1.0 / alpha + 0.5 - np.asarray(mu, dtype=float) / 2.0)


def alpha_critical(n: int) -> float:
    """alpha_crit = 2/(mu_N - 1): the anisotropy at which the zigzag mode's gamma_N reaches 0 (Marquet 2003)."""
    if n < 2:
        raise ValueError("a single ion has no zigzag transition")
    mu, _ = axial_modes_dimensionless(equilibrium_dimensionless(n))
    return 2.0 / (float(mu[-1]) - 1.0)


def zigzag_ratio_critical(n: int) -> float:
    """(omega_r/omega_z)_crit = 1/sqrt(alpha_crit): 1.0000 and sqrt(12/5) = 1.5492 at N = 2, 3 (Section 9.1)."""
    return 1.0 / math.sqrt(alpha_critical(n))


def james_coupling(b: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """James's coupling constants s_m^{(p)} = sqrt(N) b_m^{(p)} mu_p^{-1/4} (Eqs. 3.18-3.19): s = 1 on the COM mode."""
    b = np.asarray(b, dtype=float)
    return np.asarray(math.sqrt(b.shape[0]) * b * np.asarray(mu, dtype=float)[None, :] ** (-0.25))


def infinite_chain_transverse_omega_rad_s(
    kappa_d: np.ndarray | float, omega_r_rad_s: float, epsilon: float, *, terms: int = 200000
) -> np.ndarray:
    """omega_kappa = omega_1 sqrt(1 - 2 eps [zeta(3) - S(kappa d)]), S(x) = sum_{j>=1} cos(jx)/j^3, eps = e^2/(4 pi eps0 d^3 m omega_1^2)
    (Landsman 2019): the transverse dispersion of an infinite chain with spacing d; kappa d = pi is the zigzag mode."""
    x = np.atleast_1d(np.asarray(kappa_d, dtype=float))
    j = np.arange(1, terms + 1, dtype=float)
    s = np.array([np.sum(np.cos(j * xi) / j**3) for xi in x])
    radicand = 1.0 - 2.0 * epsilon * (float(zeta(3.0)) - s)
    radicand = np.where((radicand < 0.0) & (radicand > -1e-9), 0.0, radicand)
    return np.asarray(omega_r_rad_s * np.sqrt(np.where(radicand >= 0.0, radicand, np.nan)))


def infinite_chain_epsilon(
    mass_kg: float, spacing_m: float, omega_r_rad_s: float, *, charge: int = 1
) -> float:
    """eps = Z^2 e^2/(4 pi eps0 d^3 m omega_1^2) of Landsman's dispersion relation."""
    return charge * charge * K_COULOMB_J_M / (spacing_m**3 * mass_kg * omega_r_rad_s**2)


def infinite_chain_zigzag_omega_r_rad_s(mass_kg: float, spacing_m: float, *, charge: int = 1) -> float:
    """omega_r^2 = (7 zeta(3)/(8 pi eps0)) e^2/(m s_c^3): the infinite-chain force balance (Wineland 1998), the zigzag mode
    of Landsman's dispersion at kappa d = pi where S(pi) = -(3/4) zeta(3); 7.806 MHz for 9Be+ at s_c = 3 um (Section 9.1)."""
    return math.sqrt(
        7.0 * float(zeta(3.0)) / 2.0 * charge * charge * K_COULOMB_J_M / (mass_kg * spacing_m**3)
    )


def two_ion_mixed_axial_squared(mu: float) -> tuple[float, float]:
    """omega_-+^2/omega_z1^2 = 1 + 1/mu -+ sqrt(1 - 1/mu + 1/mu^2), mu = m_2/m_1 (Wubbena 2012 Eqs. 12-14; Morigi-Walther Eq. 8);
    (1, 3) at mu = 1, (0.500847, 2.250636) for 9Be+/24Mg+ with isotope masses; omega_+^2 + omega_-^2 = 2(1 + 1/mu)."""
    if mu <= 0.0:
        raise ValueError("mass ratio must be positive")
    root_ = math.sqrt(1.0 - 1.0 / mu + 1.0 / mu**2)
    return 1.0 + 1.0 / mu - root_, 1.0 + 1.0 / mu + root_


def kielpinski_three_ion_axial(mu: float) -> tuple[float, float, float]:
    """zeta_{1,3} = [13/10 + (21 -+ sqrt(441 - 34 mu + 169 mu^2))/(10 mu)]^{1/2}, zeta_2 = sqrt 3 for an impurity of mass ratio mu at the
    centre of a three-ion chain (Kielpinski 2000 Eqs. 11-16), in units of the outer ions' single-ion axial frequency."""
    disc = math.sqrt(441.0 - 34.0 * mu + 169.0 * mu * mu)
    z1 = math.sqrt(1.3 + (21.0 - disc) / (10.0 * mu))
    z3 = math.sqrt(1.3 + (21.0 + disc) / (10.0 * mu))
    return z1, math.sqrt(3.0), z3


# ---- the general three-dimensional crystal ------------------------------------------------------------------------


def _pair_tensor(r: np.ndarray) -> np.ndarray:
    """T_ab = d_a d_b (1/r) = 3 r_a r_b/r^5 - delta_ab/r^3."""
    rr = float(np.dot(r, r))
    return np.asarray(3.0 * np.outer(r, r) / rr**2.5 - np.eye(3) / rr**1.5)


def coulomb_gradient_j_per_m(positions_m: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """d/dr_i of sum_{i<j} k Z_i Z_j/|r_i - r_j| = -sum_{j != i} k Z_i Z_j (r_i - r_j)/|r_i - r_j|^3, shape (N, 3)."""
    pos = np.asarray(positions_m, dtype=float)
    n = pos.shape[0]
    g = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[i] - pos[j]
            g[i] += -K_COULOMB_J_M * charges[i] * charges[j] * d / float(np.dot(d, d)) ** 1.5
    return g


def coulomb_hessian_j_per_m2(positions_m: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """The (3N, 3N) Hessian of the Coulomb energy: block (i, i) += k Z_i Z_j T(r_i - r_j), block (i, j) = -k Z_i Z_j T."""
    pos = np.asarray(positions_m, dtype=float)
    n = pos.shape[0]
    k = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            t = K_COULOMB_J_M * charges[i] * charges[j] * _pair_tensor(pos[i] - pos[j])
            k[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += t
            k[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += t
            k[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= t
            k[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] -= t
    return k


def potential_energy_j(
    positions_m: np.ndarray,
    spring_j_per_m2: np.ndarray,
    charges: np.ndarray,
    field_v_per_m: np.ndarray | None = None,
) -> float:
    pos = np.asarray(positions_m, dtype=float)
    v = 0.5 * float(np.sum(spring_j_per_m2 * pos * pos))
    if field_v_per_m is not None:
        v -= float(np.sum(charges[:, None] * E_C * pos * np.asarray(field_v_per_m, dtype=float)[None, :]))
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            v += K_COULOMB_J_M * charges[i] * charges[j] / float(np.linalg.norm(pos[i] - pos[j]))
    return v


def total_gradient(
    positions_m: np.ndarray, spring: np.ndarray, charges: np.ndarray, field: np.ndarray | None
) -> np.ndarray:
    pos = np.asarray(positions_m, dtype=float)
    g = spring * pos + coulomb_gradient_j_per_m(pos, charges)
    if field is not None:
        g -= charges[:, None] * E_C * np.asarray(field, dtype=float)[None, :]
    return np.asarray(g)


def total_hessian(positions_m: np.ndarray, spring: np.ndarray, charges: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.diag(np.asarray(spring, dtype=float).ravel()) + coulomb_hessian_j_per_m2(positions_m, charges)
    )


def equilibrium_positions_m(
    spring_j_per_m2: np.ndarray,
    charges: np.ndarray,
    *,
    field_v_per_m: np.ndarray | None = None,
    start_m: np.ndarray | None = None,
) -> np.ndarray:
    """Minimize the crystal energy by damped Newton iteration from the James linear chain (Section 4.1.2).

    ``spring_j_per_m2`` is the (N, 3) array kappa_{i,a} = m_i omega_{i,a}^2 in the principal frame (x', y', z).
    The start is the James chain along z at the scale set by ion 0's axial spring, each ion displaced by
    Z_i e E/kappa_i under a uniform field; the returned configuration is checked to be a minimum (positive
    definite Hessian) and ``ZigzagError`` is raised otherwise.
    """
    spring = np.asarray(spring_j_per_m2, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = spring.shape[0]
    if spring.shape != (n, 3) or np.any(spring <= 0.0):
        raise ValueError("spring constants must be an (N, 3) array of positive numbers")
    if start_m is None:
        scale = (K_COULOMB_J_M * charges[0] ** 2 / spring[0, 2]) ** (1.0 / 3.0)
        pos = np.zeros((n, 3))
        pos[:, 2] = equilibrium_dimensionless(n) * scale
        if field_v_per_m is not None:
            pos += charges[:, None] * E_C * np.asarray(field_v_per_m, dtype=float)[None, :] / spring
    else:
        pos = np.array(start_m, dtype=float)
    gscale = float(np.max(np.abs(spring))) * float(np.max(np.abs(pos)) + 1e-30)
    energy = potential_energy_j(pos, spring, charges, field_v_per_m)
    for _ in range(_NEWTON_MAX):
        g = total_gradient(pos, spring, charges, field_v_per_m)
        if np.max(np.abs(g)) < 1e-12 * gscale:
            break
        h = total_hessian(pos, spring, charges)
        try:
            step = np.linalg.solve(h, -g.ravel()).reshape(n, 3)
        except np.linalg.LinAlgError:
            step = -g / np.max(np.abs(spring))
        lam = 1.0
        while lam > 1e-8:
            trial = pos + lam * step
            e_trial = potential_energy_j(trial, spring, charges, field_v_per_m)
            if e_trial <= energy + 1e-15 * abs(energy):
                break
            lam *= 0.5
        pos = pos + lam * step
        energy = potential_energy_j(pos, spring, charges, field_v_per_m)
    else:
        raise RuntimeError("crystal equilibrium did not converge")
    evals = np.linalg.eigvalsh(total_hessian(pos, spring, charges))
    if np.min(evals) <= 1e-9 * np.max(evals):
        raise ZigzagError(
            "the linear chain is not a stable minimum in this trap (a transverse Hessian eigenvalue is not positive): "
            "the zigzag criterion alpha < alpha_crit = 2/(mu_N - 1) is violated (Section 4.1.2)"
        )
    return pos


def normal_modes(
    positions_m: np.ndarray, spring_j_per_m2: np.ndarray, masses_kg: np.ndarray, charges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(omega_k in rad/s ascending, c as columns of the (3N, 3N) mass-weighted orthonormal eigenvectors) of the full crystal."""
    m = np.repeat(np.asarray(masses_kg, dtype=float), 3)
    minv = 1.0 / np.sqrt(m)
    d = minv[:, None] * total_hessian(positions_m, spring_j_per_m2, charges) * minv[None, :]
    w2, c = np.linalg.eigh((d + d.T) / 2.0)
    if np.any(w2 <= 0.0):
        raise ZigzagError("the crystal Hessian is not positive definite")
    return np.sqrt(w2), c


def is_collinear(positions_m: np.ndarray, *, rtol: float = 1e-9) -> bool:
    """All ions on the z axis to rtol times the chain length."""
    pos = np.asarray(positions_m, dtype=float)
    extent = max(float(np.ptp(pos[:, 2])), float(np.max(np.abs(pos))), 1e-300)
    return bool(np.max(np.abs(pos[:, :2])) <= rtol * extent)


# ---- Appendix E records ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """One normal mode of the crystal (Section 4.1.3): its family and position within it (ascending frequency), its frequency
    as an ordinary frequency (Hz), its unit axis, the unit-norm mass-weighted eigenvector c_{i,m} (last component positive)
    and, for a non-collinear crystal, the (N, 3) displacement pattern the eigenvector is the projection of."""

    family: Family
    index: int
    """Position within the family, ascending frequency."""
    omega_hz: float
    e_hat: tuple[float, float, float]
    eigenvector: np.ndarray
    """Mass-weighted c_{i,m}, unit norm, last component positive (Section 4.1.3)."""
    pattern: np.ndarray | None = None
    """(N, 3) mass-weighted displacement pattern in the laboratory frame for a non-collinear crystal, whose modes
    are not along one axis; ``eigenvector`` is then its projection on ``e_hat``. None for a collinear chain."""

    def __post_init__(self) -> None:
        if self.omega_hz <= 0.0:
            raise ValueError("mode frequency must be positive")
        v = np.asarray(self.eigenvector, dtype=float)
        if v.ndim != 1:
            raise ValueError("eigenvector must be one-dimensional (one component per ion)")
        if self.pattern is None and not np.isclose(float(np.linalg.norm(v)), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("eigenvector must have unit norm")
        nz = np.flatnonzero(np.abs(v) > 1e-12)
        if len(nz) and v[nz[-1]] < 0.0:
            raise ValueError("eigenvector sign gauge: last component must be positive (Section 13)")
        e = np.asarray(self.e_hat, dtype=float)
        if not math.isclose(float(np.linalg.norm(e)), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("e_hat must be a unit vector")
        if self.pattern is not None:
            p = np.asarray(self.pattern, dtype=float)
            if p.shape != (len(v), 3) or not np.isclose(float(np.linalg.norm(p)), 1.0, atol=1e-9):
                raise ValueError("pattern must be an (N, 3) unit-norm array")

    @property
    def omega_rad_s(self) -> float:
        return TWO_PI * self.omega_hz

    def displacement_pattern(self) -> np.ndarray:
        """The (N, 3) mass-weighted pattern c_i^{(m)} e_hat (or the stored one for a non-collinear crystal)."""
        if self.pattern is not None:
            return np.asarray(self.pattern, dtype=float)
        return np.outer(np.asarray(self.eigenvector, dtype=float), np.asarray(self.e_hat, dtype=float))


@dataclass(frozen=True)
class LambDicke:
    """One Lamb-Dicke parameter with the factors it was built from (the provenance ``Crystal.lamb_dicke`` records)."""

    ion: int
    mode: int
    eta: float
    projection: float
    """(delta_k . e_hat) c_{i,m}, in 1/m: the wavevector projected on the ion's mass-weighted displacement."""
    x0_m: float
    """sqrt(hbar/(2 m_i omega_m)) with the ion's OWN mass (Section 13, "Lamb-Dicke base")."""
    C0: float
    """The micromotion factor applied here and nowhere else (Section 4.1.1); 1 without a Mathieu record."""
    mass_kg: float
    omega_rad_s: float


@dataclass(frozen=True)
class Crystal:
    """The ion crystal (Sections 4.1.2, 4.1.3, 4.1.7): the species of every ion, the equilibrium positions (m, laboratory
    frame, z the trap axis), the 3N normal modes in the one canonical order every ``mode: int`` of the package refers to,
    and the principal axes when they differ from the laboratory axes. Lamb-Dicke parameters are computed by this class
    and nowhere else (``lamb_dicke``, with the micromotion factor C0 inside)."""

    species: tuple[Species, ...]
    """One entry per ion (mixed species allowed)."""
    positions_m: np.ndarray
    """(N, 3) equilibrium positions in the laboratory frame (z the trap axis)."""
    modes: tuple[Mode, ...]
    """3N modes in ONE canonical order: axial, transverse_1, transverse_2, ascending frequency within a family."""
    axes: np.ndarray | None = None
    """Principal frame (x', y', z) as columns in the laboratory frame; None means the laboratory axes."""

    def __post_init__(self) -> None:
        n = len(self.species)
        pos = np.asarray(self.positions_m)
        if pos.shape != (n, 3):
            raise ValueError(f"positions_m must have shape ({n}, 3), got {pos.shape}")
        if len(self.modes) != 3 * n:
            raise ValueError(f"a crystal of {n} ions has 3N = {3 * n} modes, got {len(self.modes)}")
        fam_rank = {f: k for k, f in enumerate(FAMILY_ORDER)}
        keys = [(fam_rank[m.family], m.omega_hz) for m in self.modes]
        if keys != sorted(keys):
            raise ValueError(
                "modes must be ordered axial, transverse_1, transverse_2 and ascending in frequency"
            )
        for m in self.modes:
            if len(m.eigenvector) != n:
                raise ValueError("every mode eigenvector has one component per ion")
        for fam in FAMILY_ORDER:
            idx = [m.index for m in self.modes if m.family == fam]
            if idx != list(range(len(idx))):
                raise ValueError(f"{fam} mode indices must run 0, 1, ... in frequency order")
            # PLAN 4.1.3: "3N modes in THREE families", i.e. N per family. The non-collinear branch assigns a family by
            # the argmax of the per-axis weight, which for a strongly mixed crystal could return 4/1/1 and pass the
            # 3N count; a family that is not N modes means the assignment, not the eigenproblem, is wrong.
            if len(idx) != n:
                counts = {f: sum(1 for m in self.modes if m.family == f) for f in FAMILY_ORDER}
                raise ValueError(
                    f"each of the three mode families has exactly N = {n} modes (Section 4.1.3), got {counts}"
                )

    @property
    def n_ions(self) -> int:
        return len(self.species)

    @property
    def masses_kg(self) -> np.ndarray:
        return np.array([s.mass_u * ATOMIC_MASS_KG for s in self.species])

    @property
    def principal_axes(self) -> np.ndarray:
        return np.eye(3) if self.axes is None else np.asarray(self.axes, dtype=float)

    def family(self, name: Family) -> tuple[Mode, ...]:
        return tuple(m for m in self.modes if m.family == name)

    def mode_index(self, family: Family, index: int) -> int:
        """The position in ``modes`` of (family, index): the ONE mode-addressing convention (Section 13)."""
        for k, m in enumerate(self.modes):
            if m.family == family and m.index == index:
                return k
        raise IndexError(f"no mode ({family}, {index})")

    def c0_for(self, ion: int, mode: int, micromotion: MathieuParameters) -> float:
        """The ion's own C0 along the mode's axis: (a, q) of the record scaled by m_ref/m_i (a, q ~ 1/m at fixed voltages)."""
        e = np.asarray(self.modes[mode].e_hat, dtype=float)
        axes = micromotion.principal_axes
        k = int(np.argmax(np.abs(axes.T @ e)))
        a_k, q_k = float(micromotion.a[k, k]), float(micromotion.q_effective[k])
        if micromotion.mass_kg is not None:
            ratio = micromotion.mass_kg / float(self.masses_kg[ion])
            a_k, q_k = a_k * ratio, q_k * ratio
        return c0_wronskian(a_k, q_k)

    def lamb_dicke_record(
        self, ion: int, mode: int, delta_k: np.ndarray, *, micromotion: MathieuParameters | None
    ) -> LambDicke:
        """eta with every factor it was built from (Section 13, "Lamb-Dicke base"; Section 4.1.7 for mixed species)."""
        if not 0 <= ion < self.n_ions:
            raise IndexError(f"ion {ion} out of range for {self.n_ions} ions")
        if not 0 <= mode < len(self.modes):
            raise IndexError(f"mode {mode} out of range for {len(self.modes)} modes")
        m = self.modes[mode]
        dk = np.asarray(delta_k, dtype=float)
        if dk.shape != (3,):
            raise ValueError("delta_k must be a 3-vector in rad/m")
        mass = float(self.masses_kg[ion])
        w = m.omega_rad_s
        projection = float(np.dot(dk, m.displacement_pattern()[ion]))
        x0 = math.sqrt(HBAR_J_S / (2.0 * mass * w))
        c0 = 1.0 if micromotion is None else self.c0_for(ion, mode, micromotion)
        return LambDicke(
            ion=ion,
            mode=mode,
            eta=projection * x0 * c0,
            projection=projection,
            x0_m=x0,
            C0=c0,
            mass_kg=mass,
            omega_rad_s=w,
        )

    def lamb_dicke(
        self, ion: int, mode: int, delta_k: np.ndarray, *, micromotion: MathieuParameters | None
    ) -> float:
        """eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 from ``micromotion`` (Section 4.1.1).

        C0 is applied HERE and nowhere else (Section 5.7); ``lamb_dicke_record`` returns the factors for provenance.
        """
        return self.lamb_dicke_record(ion, mode, delta_k, micromotion=micromotion).eta

    def lamb_dicke_matrix(self, delta_k: np.ndarray, *, micromotion: MathieuParameters | None) -> np.ndarray:
        """eta_{i,m} for every ion and mode, shape (N, 3N)."""
        return np.array(
            [
                [self.lamb_dicke(i, k, delta_k, micromotion=micromotion) for k in range(len(self.modes))]
                for i in range(self.n_ions)
            ]
        )

    def uniform_field_weight(self, mode: int) -> float:
        """(sum_i c_i^{(k)}/sqrt(m_i))^2 in kg^-1: the coupling of a mode to a uniform electric field (Kielpinski Eq. 20).

        N/m for an equal-mass centre-of-mass mode, zero for every even-parity mode of a reflection-symmetric mass
        array and for every non-COM mode of an equal-mass chain (Sections 4.1.5, 4.1.7).
        """
        m = self.modes[mode]
        pattern = m.displacement_pattern()
        e = np.asarray(m.e_hat, dtype=float)
        return float(np.sum(pattern @ e / np.sqrt(self.masses_kg))) ** 2

    def collinear(self) -> bool:
        pos = np.asarray(self.positions_m, dtype=float)
        pos = (pos - pos.mean(axis=0)) @ self.principal_axes
        return is_collinear(pos)


# ---- building a Crystal -------------------------------------------------------------------------------------------------


def build_crystal(
    species: tuple[Species, ...],
    omega_rad_s: np.ndarray,
    *,
    axes: np.ndarray | None = None,
    field_v_per_m: np.ndarray | None = None,
    charges: np.ndarray | None = None,
    centre_m: np.ndarray | None = None,
) -> Crystal:
    """The crystal of ``species`` whose single-ion secular frequencies (N, 3) along the principal axes (x', y', z) are
    ``omega_rad_s``, in a uniform residual field ``field_v_per_m`` (laboratory frame); ``axes`` are the principal
    directions as columns (default the laboratory axes) and ``centre_m`` the trap centre (the rf null) in the
    laboratory frame about which the positions are reported (default the origin)."""
    n = len(species)
    w = np.asarray(omega_rad_s, dtype=float)
    if w.shape != (n, 3) or np.any(w <= 0.0):
        raise ValueError("omega_rad_s must be an (N, 3) array of positive single-ion secular frequencies")
    masses = np.array([s.mass_u * ATOMIC_MASS_KG for s in species])
    z = np.ones(n) if charges is None else np.asarray(charges, dtype=float)
    frame = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
    spring = masses[:, None] * w * w
    field_p = None if field_v_per_m is None else frame.T @ np.asarray(field_v_per_m, dtype=float)
    pos_p = equilibrium_positions_m(spring, z, field_v_per_m=field_p)
    modes: list[Mode] = []
    if is_collinear(pos_p):
        pos_axis = pos_p[:, 2]
        for fam in FAMILY_ORDER:
            ax = FAMILY_AXIS[fam]
            k = np.diag(spring[:, ax]).astype(float)
            for i in range(n):
                for j in range(i + 1, n):
                    d = pos_axis[i] - pos_axis[j]
                    t = K_COULOMB_J_M * z[i] * z[j] * (2.0 if ax == 2 else -1.0) / abs(d) ** 3
                    k[i, i] += t
                    k[j, j] += t
                    k[i, j] -= t
                    k[j, i] -= t
            minv = 1.0 / np.sqrt(masses)
            d_mat = minv[:, None] * k * minv[None, :]
            w2, c = np.linalg.eigh((d_mat + d_mat.T) / 2.0)
            if np.any(w2 <= 0.0):
                raise ZigzagError(
                    f"the {fam} family has a non-positive squared frequency: the chain has buckled"
                )
            c = _fix_sign(c)
            e_hat = frame[:, ax]
            for idx in range(n):
                modes.append(
                    Mode(
                        fam,
                        idx,
                        float(math.sqrt(w2[idx]) / TWO_PI),
                        (float(e_hat[0]), float(e_hat[1]), float(e_hat[2])),
                        c[:, idx].copy(),
                    )
                )
    else:
        w_all, c_all = normal_modes(pos_p, spring, masses, z)
        per_family: dict[str, list[tuple[float, np.ndarray]]] = {f: [] for f in FAMILY_ORDER}
        for k_idx in range(3 * n):
            pat = c_all[:, k_idx].reshape(n, 3)
            weights = np.sum(pat * pat, axis=0)
            ax = int(np.argmax(weights))
            fam = next(f for f in FAMILY_ORDER if FAMILY_AXIS[f] == ax)
            per_family[fam].append((float(w_all[k_idx]), pat))
        for fam in FAMILY_ORDER:
            ax = FAMILY_AXIS[fam]
            items = sorted(per_family[fam], key=lambda t: t[0])
            for idx, (w_k, pat) in enumerate(items):
                proj = pat[:, ax]
                nz = np.flatnonzero(np.abs(proj) > 1e-12)
                if len(nz) and proj[nz[-1]] < 0.0:
                    pat, proj = -pat, -proj
                pat_lab = pat @ frame.T
                e_hat = frame[:, ax]
                modes.append(
                    Mode(
                        fam,
                        idx,
                        float(w_k / TWO_PI),
                        (float(e_hat[0]), float(e_hat[1]), float(e_hat[2])),
                        proj.copy(),
                        pattern=pat_lab,
                    )
                )
    centre = np.zeros(3) if centre_m is None else np.asarray(centre_m, dtype=float)
    return Crystal(
        species=tuple(species), positions_m=pos_p @ frame.T + centre, modes=tuple(modes), axes=frame
    )


def solve_crystal(trap: Trap, species: tuple[Species, ...] | list[Species], *, reference: int = 0) -> Crystal:
    """Equilibrium positions, mass-weighted Hessian and normal modes of ``species`` in ``trap`` (Sections 4.1.2, 4.1.3, 4.1.7).

    ``reference`` names the ion whose species the trap's explicit secular frequencies describe; the other species'
    frequencies follow from the Mathieu parameters scaled by m_ref/m_i (Section 4.1.7: the rf part of omega^2
    scales as 1/m^2 and the static part as 1/m), which needs the trap's rf frequency. The residual field
    (stray plus shim response) displaces the crystal.
    """
    sp = tuple(species)
    if not sp:
        raise ValueError("a crystal needs at least one ion")
    if not 0 <= reference < len(sp):
        raise IndexError("reference ion out of range")
    omega, axes, field = trap.single_ion_frequencies_rad_s(sp, reference=reference)
    return build_crystal(sp, omega, axes=axes, field_v_per_m=field, centre_m=trap.rf_null_m())


__all__ = [
    "FAMILY_AXIS",
    "FAMILY_ORDER",
    "K_COULOMB_J_M",
    "Crystal",
    "LambDicke",
    "Mode",
    "ZigzagError",
    "alpha_critical",
    "axial_hessian_dimensionless",
    "axial_modes_dimensionless",
    "build_crystal",
    "coulomb_gradient_j_per_m",
    "coulomb_hessian_j_per_m2",
    "equilibrium_dimensionless",
    "equilibrium_positions_m",
    "infinite_chain_epsilon",
    "infinite_chain_transverse_omega_rad_s",
    "infinite_chain_zigzag_omega_r_rad_s",
    "is_collinear",
    "james_coupling",
    "kielpinski_three_ion_axial",
    "length_scale_m",
    "normal_modes",
    "potential_energy_j",
    "solve_crystal",
    "transverse_eigenvalues",
    "two_ion_mixed_axial_squared",
    "zigzag_ratio_critical",
]
