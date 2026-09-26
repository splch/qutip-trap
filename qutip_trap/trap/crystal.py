"""Ion crystal, normal modes and Lamb-Dicke parameters (PLAN.md Section 4.1.3).

Every ``mode: int`` of the package is a position in ``Crystal.modes``: axial, transverse_1, transverse_2, ascending
frequency within a family, unit-norm MASS-WEIGHTED eigenvectors whose last non-zero component is positive. The
equilibrium minimizes sum_i [(1/2) sum_a kappa_{i,a} r_{i,a}^2 - Z_i e E . r_i] + sum_{i<j} Z_i Z_j e^2/(4 pi eps0 r_ij)
by damped Newton from the James 1998 chain and must be a strict minimum (a buckled chain raises ``ZigzagError``). The
modes solve M^{-1/2} K M^{-1/2} c = omega^2 c (Home 2013; Morigi and Walther 2001): ion i moves by
c_i sqrt(hbar/(2 m_i omega)) (a + a^dag) along e_hat, so eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) C0
with the ion's own mass; a collinear chain's three Cartesian families separate exactly and are solved separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

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
    """The linear chain is not a stable minimum in this trap (the zigzag criterion is violated)."""


# ---- James 1998: the dimensionless linear chain ------------------------------------------------------------------


def length_scale_m(mass_kg: float, omega_z_rad_s: float, *, charge: int = 1) -> float:
    """l = (Z^2 e^2/(4 pi eps0 M nu^2))^{1/3} (James 1998): two ions sit 2^{1/3} l apart."""
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
    """A_mn = 1 + 2 sum_{p != m} 1/|u_m - u_p|^3 (m = n), -2/|u_m - u_n|^3 (m != n) (James Eq. 3.3)."""
    u = np.asarray(u, dtype=float)
    d = np.abs(u[:, None] - u[None, :])
    with np.errstate(divide="ignore"):
        inv3 = np.where(np.eye(len(u), dtype=bool), 0.0, 1.0 / d**3)
    a = -2.0 * inv3
    a[np.diag_indices(len(u))] = 1.0 + 2.0 * inv3.sum(axis=1)
    return np.asarray(a)


def equilibrium_dimensionless(n: int, *, start: np.ndarray | None = None) -> np.ndarray:
    """The James equilibrium u_1 < ... < u_N of N equal charges by damped Newton iteration (the minimum is unique).

    Closed forms: -+(1/2)^{2/3} for N = 2; -+(5/4)^{1/3}, 0 for N = 3.
    """
    if n < 1:
        raise ValueError("N >= 1")
    if n == 1:
        return np.zeros(1)
    if start is None:
        # James's empirical minimum spacing 2.018 N^{-0.559} sets the scale of a uniform start
        u = 2.018 * n ** (-0.559) * (np.arange(n) - (n - 1) / 2.0)
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
    """Sign gauge: the last non-zero component of every column positive (Marquet et al.'s convention)."""
    v = np.array(vectors, dtype=float)
    for k in range(v.shape[1]):
        last = v[np.max(np.flatnonzero(np.abs(v[:, k]) > 1e-12)), k]
        if last < 0.0:
            v[:, k] = -v[:, k]
    return v


def axial_modes_dimensionless(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu_p ascending, b^{(p)} as columns) of the dimensionless axial Hessian; nu_p = sqrt(mu_p) nu (James Eq. 3.5)."""
    mu, b = np.linalg.eigh(axial_hessian_dimensionless(np.asarray(u, dtype=float)))
    return np.asarray(mu), _fix_sign(b)


# ---- the general three-dimensional crystal ------------------------------------------------------------------------


def _pair_tensor(r: np.ndarray) -> np.ndarray:
    """T_ab = d_a d_b (1/r) = 3 r_a r_b/r^5 - delta_ab/r^3."""
    rr = float(np.dot(r, r))
    return np.asarray(3.0 * np.outer(r, r) / rr**2.5 - np.eye(3) / rr**1.5)


def coulomb_gradient_j_per_m(positions_m: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """d/dr_i of sum_{i<j} k Z_i Z_j/|r_i - r_j|, shape (N, 3)."""
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
    """Minimize the crystal energy by damped Newton iteration from the James chain along z.

    ``spring_j_per_m2`` is the (N, 3) array m_i omega_{i,a}^2 in the principal frame; the start is displaced by
    Z_i e E/kappa_i under a uniform field. Raises ``ZigzagError`` unless the result is a strict minimum.
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
            "the zigzag criterion alpha < alpha_crit = 2/(mu_N - 1) is violated"
        )
    return pos


def normal_modes(
    positions_m: np.ndarray, spring_j_per_m2: np.ndarray, masses_kg: np.ndarray, charges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(omega_k ascending in rad/s, the (3N, 3N) orthonormal mass-weighted eigenvectors as columns) of the crystal."""
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


# ---- the records ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """One normal mode: family, index within it (ascending frequency), ordinary frequency (Hz), unit axis and the
    mass-weighted eigenvector c_{i,m}; a non-collinear crystal also stores the (N, 3) laboratory-frame pattern, whose
    projection on ``e_hat`` the eigenvector then is."""

    family: Family
    index: int
    omega_hz: float
    e_hat: tuple[float, float, float]
    eigenvector: np.ndarray
    pattern: np.ndarray | None = None

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
            raise ValueError("eigenvector sign gauge: last component must be positive")
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
class Crystal:
    """The species of every ion, the equilibrium positions (m, laboratory frame, z the trap axis), the 3N modes in the
    canonical order, and the principal axes (None: the laboratory axes)."""

    species: tuple[Species, ...]
    positions_m: np.ndarray
    modes: tuple[Mode, ...]
    axes: np.ndarray | None = None

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
            # N per family: the non-collinear branch assigns families by the largest per-axis weight
            if len(idx) != n:
                counts = {f: sum(1 for m in self.modes if m.family == f) for f in FAMILY_ORDER}
                raise ValueError(f"each of the three mode families has exactly N = {n} modes, got {counts}")

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
        """The position in ``modes`` of (family, index)."""
        for k, m in enumerate(self.modes):
            if m.family == family and m.index == index:
                return k
        raise IndexError(f"no mode ({family}, {index})")

    def c0_for(self, ion: int, mode: int, micromotion: MathieuParameters) -> float:
        """The ion's own C0 along the mode's axis: the record's (a, q) scaled by m_ref/m_i (a, q ~ 1/m at fixed voltages)."""
        e = np.asarray(self.modes[mode].e_hat, dtype=float)
        k = int(np.argmax(np.abs(micromotion.principal_axes.T @ e)))
        mass = float(self.masses_kg[ion])
        if micromotion.mass_kg == mass:
            return micromotion.C0[k]  # the record's own C0: the same (a_kk, q_eff_k) at ratio 1
        a_k, q_k = float(micromotion.a[k, k]), float(micromotion.q_effective[k])
        if micromotion.mass_kg is not None:
            ratio = micromotion.mass_kg / mass
            a_k, q_k = a_k * ratio, q_k * ratio
        return c0_wronskian(a_k, q_k)

    def lamb_dicke(
        self, ion: int, mode: int, delta_k: np.ndarray, *, micromotion: MathieuParameters | None
    ) -> float:
        """eta_{i,m} = (delta_k . e_hat) c_{i,m} sqrt(hbar/(2 m_i omega_m)) times C0 from ``micromotion`` (1 without)."""
        if not 0 <= ion < self.n_ions:
            raise IndexError(f"ion {ion} out of range for {self.n_ions} ions")
        if not 0 <= mode < len(self.modes):
            raise IndexError(f"mode {mode} out of range for {len(self.modes)} modes")
        dk = np.asarray(delta_k, dtype=float)
        if dk.shape != (3,):
            raise ValueError("delta_k must be a 3-vector in rad/m")
        m = self.modes[mode]
        projection = float(np.dot(dk, m.displacement_pattern()[ion]))
        x0 = math.sqrt(HBAR_J_S / (2.0 * float(self.masses_kg[ion]) * m.omega_rad_s))
        c0 = 1.0 if micromotion is None else self.c0_for(ion, mode, micromotion)
        return projection * x0 * c0

    def field_displacement_m(self, field_v_per_m: np.ndarray) -> np.ndarray:
        """The (N, 3) laboratory-frame shift of every equilibrium position under an added uniform field, to first order:
        K^{-1} e E with K the Hessian the crystal was solved with, sum_m (c_i^{(m)}/sqrt(m_i)) sum_j (c_j^{(m)} . e E)/
        (sqrt(m_j) omega_m^2) over the mass-weighted mode patterns.

        An equal-mass chain moves rigidly by e E_a/(m omega_a^2) along each principal axis; in a mixed crystal the axial
        shift is still rigid (the dc curvature is mass-independent) while the unequal transverse springs are coupled by the
        Coulomb interaction.
        """
        e = np.asarray(field_v_per_m, dtype=float)
        if e.shape != (3,):
            raise ValueError("the field is a laboratory-frame 3-vector in V/m")
        force = E_C * e
        inv_sqrt_m = 1.0 / np.sqrt(self.masses_kg)
        shift = np.zeros((self.n_ions, 3))
        for m in self.modes:
            weighted = m.displacement_pattern() * inv_sqrt_m[:, None]
            shift += weighted * float(np.sum(weighted @ force)) / m.omega_rad_s**2
        return shift

    def uniform_field_weight(self, mode: int) -> float:
        """(sum_i c_i^{(k)}/sqrt(m_i))^2 in kg^-1: the mode's coupling to a uniform electric field (Kielpinski Eq. 20).

        N/m for an equal-mass centre-of-mass mode, zero for every other mode of an equal-mass chain and for every
        antisymmetric mode of a reflection-symmetric mass array.
        """
        m = self.modes[mode]
        e = np.asarray(m.e_hat, dtype=float)
        return float(np.sum(m.displacement_pattern() @ e / np.sqrt(self.masses_kg))) ** 2

    def collinear(self) -> bool:
        pos = np.asarray(self.positions_m, dtype=float)
        return is_collinear((pos - pos.mean(axis=0)) @ self.principal_axes)


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
    """The crystal of ``species`` with single-ion secular frequencies ``omega_rad_s`` (N, 3) along the principal axes
    ``axes`` (columns, default the laboratory axes), in a uniform residual field (laboratory frame), positions reported
    about ``centre_m`` (the rf null, default the origin)."""
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
    minv = 1.0 / np.sqrt(masses)
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
            ax = int(np.argmax(np.sum(pat * pat, axis=0)))
            per_family[next(f for f in FAMILY_ORDER if FAMILY_AXIS[f] == ax)].append(
                (float(w_all[k_idx]), pat)
            )
        for fam in FAMILY_ORDER:
            ax = FAMILY_AXIS[fam]
            for idx, (w_k, pat) in enumerate(sorted(per_family[fam], key=lambda t: t[0])):
                proj = pat[:, ax]
                nz = np.flatnonzero(np.abs(proj) > 1e-12)
                if len(nz) and proj[nz[-1]] < 0.0:
                    pat, proj = -pat, -proj
                e_hat = frame[:, ax]
                modes.append(
                    Mode(
                        fam,
                        idx,
                        float(w_k / TWO_PI),
                        (float(e_hat[0]), float(e_hat[1]), float(e_hat[2])),
                        proj.copy(),
                        pattern=pat @ frame.T,
                    )
                )
    centre = np.zeros(3) if centre_m is None else np.asarray(centre_m, dtype=float)
    return Crystal(
        species=tuple(species), positions_m=pos_p @ frame.T + centre, modes=tuple(modes), axes=frame
    )


def solve_crystal(trap: Trap, species: tuple[Species, ...] | list[Species]) -> Crystal:
    """Equilibrium and normal modes of ``species`` in ``trap``, displaced by the residual field.

    On the explicit path the trap's secular frequencies are those of the ion mass ``Trap.reference_mass_u``, and an ion of
    another mass follows from the Mathieu parameters scaled by m_ref/m_i, which needs the trap's rf frequency.
    """
    sp = tuple(species)
    if not sp:
        raise ValueError("a crystal needs at least one ion")
    omega, axes, field = trap.single_ion_frequencies_rad_s(sp)
    return build_crystal(sp, omega, axes=axes, field_v_per_m=field, centre_m=trap.rf_null_m())
