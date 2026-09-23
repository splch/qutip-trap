"""Cubic and quartic Coulomb mode couplings and the three-mode resonance check (Marquet, Schmidt-Kaler and James 2003).

H_anh/hbar = sum_{k<=l<=m} cubic_rad_s[(k, l, m)] X_k X_l X_m + sum_{k<=l<=m<=n} quartic_rad_s[(k, l, m, n)] X_k X_l X_m X_n
with X_k = a_k + a_k^dag, each SORTED index tuple's coefficient carrying its permutation multiplicity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, permutations

import numpy as np
from scipy.linalg import expm

from qutip_trap.trap.crystal import (
    K_COULOMB_J_M,
    Crystal,
    axial_hessian_dimensionless,
    length_scale_m,
)
from qutip_trap.units import HBAR_J_S, TWO_PI


@dataclass(frozen=True)
class AnharmonicTerms:
    """Coupling coefficients of the optional H_anh and H_curv terms in rad/s, keyed by sorted mode-index tuples into
    ``Crystal.modes`` (multiplicity included; module docstring)."""

    cubic_rad_s: dict[tuple[int, int, int], float] = field(default_factory=dict)
    quartic_rad_s: dict[tuple[int, int, int, int], float] = field(default_factory=dict)
    resonance_check: bool = True
    """The Hamiltonian builder reports ``anharmonic_estimate`` whether or not the cubic term is built; False silences it."""

    def __post_init__(self) -> None:
        for key in self.cubic_rad_s:
            if len(key) != 3 or list(key) != sorted(key):
                raise ValueError(f"cubic keys are sorted mode-index triples, got {key}")
        for key4 in self.quartic_rad_s:
            if len(key4) != 4 or list(key4) != sorted(key4):
                raise ValueError(f"quartic keys are sorted mode-index quadruples, got {key4}")

    def largest_cubic_rad_s(self) -> float:
        return max((abs(v) for v in self.cubic_rad_s.values()), default=0.0)


# ---- exact Coulomb derivatives (Cartesian, any crystal) ---------------------------------------------------------------------


def _third_pair_tensor(r: np.ndarray) -> np.ndarray:
    """d_a d_b d_c (1/r) = 3(delta_ab r_c + delta_ac r_b + delta_bc r_a)/r^5 - 15 r_a r_b r_c/r^7."""
    rr = float(np.dot(r, r))
    eye = np.eye(3)
    t = np.einsum("ab,c->abc", eye, r) + np.einsum("ac,b->abc", eye, r) + np.einsum("bc,a->abc", eye, r)
    return np.asarray(3.0 * t / rr**2.5 - 15.0 * np.einsum("a,b,c->abc", r, r, r) / rr**3.5)


def _fourth_pair_tensor(r: np.ndarray) -> np.ndarray:
    """d_a d_b d_c d_d (1/r) = 3(dd)/r^5 - 15(delta r r terms)/r^7 + 105 r r r r/r^9."""
    rr = float(np.dot(r, r))
    eye = np.eye(3)
    dd = (
        np.einsum("ab,cd->abcd", eye, eye)
        + np.einsum("ac,bd->abcd", eye, eye)
        + np.einsum("ad,bc->abcd", eye, eye)
    )
    drr = (
        np.einsum("ab,c,d->abcd", eye, r, r)
        + np.einsum("ac,b,d->abcd", eye, r, r)
        + np.einsum("ad,b,c->abcd", eye, r, r)
        + np.einsum("bc,a,d->abcd", eye, r, r)
        + np.einsum("bd,a,c->abcd", eye, r, r)
        + np.einsum("cd,a,b->abcd", eye, r, r)
    )
    return np.asarray(
        3.0 * dd / rr**2.5 - 15.0 * drr / rr**3.5 + 105.0 * np.einsum("a,b,c,d->abcd", r, r, r, r) / rr**4.5
    )


def coulomb_third_derivatives(positions_m: np.ndarray, charges: np.ndarray | None = None) -> np.ndarray:
    """V_abc over the 3N coordinates, exact: for each pair the ion labels carry a sign (-1)^{number of j's} on d(1/|r_i - r_j|)."""
    pos = np.asarray(positions_m, dtype=float)
    n = pos.shape[0]
    z = np.ones(n) if charges is None else np.asarray(charges, dtype=float)
    out = np.zeros((3 * n, 3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            t = K_COULOMB_J_M * z[i] * z[j] * _third_pair_tensor(pos[i] - pos[j])
            for ions in (
                (i, i, i),
                (i, i, j),
                (i, j, i),
                (j, i, i),
                (i, j, j),
                (j, i, j),
                (j, j, i),
                (j, j, j),
            ):
                sign = (-1.0) ** sum(1 for x in ions if x == j)
                sl = tuple(slice(3 * x, 3 * x + 3) for x in ions)
                out[sl] += sign * t
    return out


def coulomb_fourth_derivatives(positions_m: np.ndarray, charges: np.ndarray | None = None) -> np.ndarray:
    """V_abcd over the 3N coordinates, exact (the same sign rule with four labels)."""
    pos = np.asarray(positions_m, dtype=float)
    n = pos.shape[0]
    z = np.ones(n) if charges is None else np.asarray(charges, dtype=float)
    out = np.zeros((3 * n,) * 4)
    for i in range(n):
        for j in range(i + 1, n):
            t = K_COULOMB_J_M * z[i] * z[j] * _fourth_pair_tensor(pos[i] - pos[j])
            for bits in range(16):
                ions = tuple(j if (bits >> s) & 1 else i for s in range(4))
                sign = (-1.0) ** sum(1 for x in ions if x == j)
                sl = tuple(slice(3 * x, 3 * x + 3) for x in ions)
                out[sl] += sign * t
    return out


def zero_point_matrix(crystal: Crystal) -> np.ndarray:
    """e_{(i,a),k} = c_{(i,a)}^{(k)} sqrt(hbar/(2 m_i omega_k)): the Cartesian zero-point amplitude of coordinate (i, a) in mode k, (3N, 3N)."""
    masses = crystal.masses_kg
    n = crystal.n_ions
    e = np.zeros((3 * n, len(crystal.modes)))
    for k, mode in enumerate(crystal.modes):
        pattern = mode.displacement_pattern()  # (N, 3) in the lab frame
        x0 = np.sqrt(HBAR_J_S / (2.0 * masses * mode.omega_rad_s))
        e[:, k] = (pattern * x0[:, None]).ravel()
    return e


def coulomb_anharmonic_terms(
    crystal: Crystal, *, quartic: bool = False, cutoff_rad_s: float = 0.0
) -> AnharmonicTerms:
    """The cubic (and optionally quartic) Coulomb couplings of a crystal in mode space, keyed by sorted mode indices, in
    rad/s; coefficients below ``cutoff_rad_s`` in magnitude are dropped."""
    e = zero_point_matrix(crystal)
    n_modes = e.shape[1]
    v3 = coulomb_third_derivatives(crystal.positions_m)
    d3 = np.einsum("abc,ak,bl,cm->klm", v3, e, e, e) / (6.0 * HBAR_J_S)
    cubic: dict[tuple[int, int, int], float] = {}
    for key in combinations_with_replacement(range(n_modes), 3):
        mult = len(set(permutations(key)))
        val = float(mult * d3[key])
        if abs(val) > cutoff_rad_s:
            cubic[key] = val
    quart: dict[tuple[int, int, int, int], float] = {}
    if quartic:
        v4 = coulomb_fourth_derivatives(crystal.positions_m)
        d4 = np.einsum("abcd,ak,bl,cm,dn->klmn", v4, e, e, e, e) / (24.0 * HBAR_J_S)
        for key4 in combinations_with_replacement(range(n_modes), 4):
            mult = len(set(permutations(key4)))
            val = float(mult * d4[key4])
            if abs(val) > cutoff_rad_s:
                quart[key4] = val
    return AnharmonicTerms(cubic_rad_s=cubic, quartic_rad_s=quart)


# ---- Marquet 2003: the dimensionless equal-mass closed forms ---------------------------------------------------------------------


def marquet_c_tensor(u: np.ndarray) -> np.ndarray:
    """C_mnp = (1/6) d^3 F/du_m du_n du_p, F = sum_{m<n} 1/|u_m - u_n|, at the James equilibrium u (Marquet Eq. 2.11)."""
    u = np.asarray(u, dtype=float)
    n = len(u)
    c = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = u[i] - u[j]
            # f(d) = 1/|d|: f''' = -6 sign(d)/d^4, with d = u_i - u_j so that d/du_i = +1 and d/du_j = -1
            f3 = -6.0 * math.copysign(1.0, d) / d**4
            for ions in (
                (i, i, i),
                (i, i, j),
                (i, j, i),
                (j, i, i),
                (i, j, j),
                (j, i, j),
                (j, j, i),
                (j, j, j),
            ):
                sign = (-1.0) ** sum(1 for x in ions if x == j)
                c[ions] += sign * f3
    return c / 6.0


def marquet_d_coefficients(c_tensor: np.ndarray, b: np.ndarray) -> np.ndarray:
    """D_pqr = sum_mnp' C_mnp' b_m^(p) b_n^(q) b_p'^(r): the cubic coefficients in mode space (Marquet Eqs. 3.1-3.5)."""
    return np.asarray(np.einsum("mnp,mi,nj,pk->ijk", c_tensor, b, b, b))


def marquet_selection_rules(u: np.ndarray, b: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """(max |D_mn1|, max |D_mn2 - ((1 - mu_m)/(2|u|)) delta_mn|): Marquet's selection rules, both zero to round-off."""
    c = marquet_c_tensor(u)
    d = marquet_d_coefficients(c, b)
    n = len(u)
    norm_u = float(np.linalg.norm(u))
    rule2 = np.array(
        [[(1.0 - mu[m]) / (2.0 * norm_u) if m == k else 0.0 for k in range(n)] for m in range(n)]
    )
    return float(np.max(np.abs(d[:, :, 0]))), float(np.max(np.abs(d[:, :, 1] - rule2)))


def nonlinearity_epsilon(mass_kg: float, omega_z_rad_s: float, *, charge: int = 1) -> float:
    """eps = x0/(4 l) with x0 = sqrt(hbar/(2 M omega_z)) and l the James length scale."""
    x0 = math.sqrt(HBAR_J_S / (2.0 * mass_kg * omega_z_rad_s))
    return x0 / (4.0 * length_scale_m(mass_kg, omega_z_rad_s, charge=charge))


def coupling_g_rad_s(mass_kg: float, omega_z_rad_s: float, *, charge: int = 1) -> float:
    """g = eps omega_z (``nonlinearity_epsilon``)."""
    return nonlinearity_epsilon(mass_kg, omega_z_rad_s, charge=charge) * omega_z_rad_s


def axial_cubic_closed_form_rad_s(crystal: Crystal) -> dict[tuple[int, int, int], float]:
    """Marquet's axial coefficients 2 eps omega_z D_pqr (mu_p mu_q mu_r)^{-1/4} x multiplicity for an equal-mass collinear
    chain, keyed like ``AnharmonicTerms.cubic_rad_s`` (the axial modes are the first N of ``Crystal.modes``)."""
    masses = crystal.masses_kg
    if not np.allclose(masses, masses[0]):
        raise ValueError("the Marquet closed form is for equal masses")
    axial = crystal.family("axial")
    n = crystal.n_ions
    omega_z = axial[0].omega_rad_s  # COM = single-ion axial frequency
    pos = np.asarray(crystal.positions_m, dtype=float) @ crystal.principal_axes
    ell = length_scale_m(float(masses[0]), omega_z)
    u = pos[:, 2] / ell
    mu = np.array([(m.omega_rad_s / omega_z) ** 2 for m in axial])
    b = np.column_stack([m.eigenvector for m in axial])
    d = marquet_d_coefficients(marquet_c_tensor(u), b)
    eps = nonlinearity_epsilon(float(masses[0]), omega_z)
    out: dict[tuple[int, int, int], float] = {}
    for key in combinations_with_replacement(range(n), 3):
        mult = len(set(permutations(key)))
        out[key] = float(
            mult * 2.0 * eps * omega_z * d[key] * (mu[key[0]] * mu[key[1]] * mu[key[2]]) ** (-0.25)
        )
    return out


@dataclass(frozen=True)
class Resonance:
    modes: tuple[int, int, int]
    """(p, q, r) with omega_p + sign omega_q ~ omega_r."""
    sign: int
    mismatch_hz: float
    coupling_rad_s: float


def three_mode_resonances(
    crystal: Crystal, terms: AnharmonicTerms, *, width_hz: float
) -> tuple[Resonance, ...]:
    """Mode triples with |omega_p +- omega_q - omega_r| < width_hz and a non-zero cubic coupling (Marquet Eq. 4.10, a
    necessary condition), sorted by |mismatch|. Near such a triple the accumulated phase grows linearly."""
    w = np.array([m.omega_hz for m in crystal.modes])
    found: list[Resonance] = []
    for key, g in terms.cubic_rad_s.items():
        p, q, r = key
        for perm in set(permutations((p, q, r))):
            a, b, c = perm
            for sign in (+1, -1):
                mismatch = w[a] + sign * w[b] - w[c]
                if abs(mismatch) < width_hz and not (sign == -1 and a == b):
                    found.append(
                        Resonance(modes=(a, b, c), sign=sign, mismatch_hz=float(mismatch), coupling_rad_s=g)
                    )
    return tuple(sorted(found, key=lambda r_: abs(r_.mismatch_hz)))


def dispersive_phase_bound_rad(coupling_rad_s: float, mismatch_rad_s: float, duration_s: float) -> float:
    """g^2 t/Delta_res: a second-order upper bound on the off-resonant phase, not its scaling (valid only for Delta_res >>
    the mode-frequency spread)."""
    if mismatch_rad_s == 0.0:
        raise ValueError("on resonance the growth is linear in time, not dispersive")
    return coupling_rad_s**2 * duration_s / abs(mismatch_rad_s)


# ---- the default-on integrated-phase estimator -----------------------------------------------------------------------------


def integrated_cubic_phase_rad(
    coupling_rad_s: float,
    omega_a_rad_s: float,
    omega_b_rad_s: float,
    duration_s: float,
    *,
    state: tuple[int, int] = (1, 0),
    dimensions: tuple[int, int] = (8, 6),
) -> float:
    """The directly integrated phase of |n_a, n_b> under H = w_a a^dag a + w_b b^dag b + g (a + a^dag)^2 (b + b^dag),
    relative to |0, 0> and free evolution: arg[(<s|U|s>/<s|U_0|s>) / (<0|U|0>/<0|U_0|0>)]; ``dimensions`` are Fock caps."""
    if duration_s < 0.0:
        raise ValueError("a duration is non-negative")
    d_a, d_b = dimensions
    if state[0] >= d_a or state[1] >= d_b or state[0] < 0 or state[1] < 0:
        raise ValueError(f"the state {state} does not fit the truncations {dimensions}")
    a = np.kron(np.diag(np.sqrt(np.arange(1, d_a)), 1), np.eye(d_b))
    b = np.kron(np.eye(d_a), np.diag(np.sqrt(np.arange(1, d_b)), 1))
    h0 = omega_a_rad_s * (a.T @ a) + omega_b_rad_s * (b.T @ b)
    x_a, x_b = a + a.T, b + b.T
    h = h0 + coupling_rad_s * (x_a @ x_a) @ x_b
    u = expm(-1j * h * duration_s)
    u0 = expm(-1j * h0 * duration_s)
    i = state[0] * d_b + state[1]
    return float(np.angle((u[i, i] / u0[i, i]) / (u[0, 0] / u0[0, 0])))


@dataclass(frozen=True)
class AnharmonicEstimate:
    """The default-on anharmonic report: ``phase_rad`` for the worst off-resonant triple (largest g^2 t/Delta_res, the
    ``bound_rad``) in the two-mode 2:1 model, and the ``resonances`` inside the coupling width, where the estimate fails."""

    phase_rad: float
    coupling_rad_s: float
    mismatch_hz: float
    modes: tuple[int, int, int]
    bound_rad: float
    resonances: tuple[Resonance, ...]
    width_hz: float

    def summary(self) -> str:
        """The one-line note the Hamiltonian builder appends to ``BuiltHamiltonian.approximations``."""
        near = (
            f"{len(self.resonances)} mode triple(s) within the {self.width_hz:.3g} Hz coupling width "
            f"(nearest {abs(self.resonances[0].mismatch_hz):.3g} Hz: growth is LINEAR in time there)"
            if self.resonances
            else f"no mode triple within the {self.width_hz:.3g} Hz coupling width"
        )
        return (
            f"anharmonic resonance check (Section 5.7, on by default): {near}; integrated cubic phase "
            f"{self.phase_rad:+.3e} rad on modes {self.modes} at g/2pi = {self.coupling_rad_s / TWO_PI:.4g} Hz and "
            f"Delta_res = {self.mismatch_hz:.4g} Hz (the g^2 t/Delta bound alone would read {self.bound_rad:.3e} rad)"
        )


def anharmonic_estimate(
    crystal: Crystal, terms: AnharmonicTerms, duration_s: float, *, width_hz: float | None = None
) -> AnharmonicEstimate | None:
    """Run the resonance check and integrate the phase of the worst off-resonant triple; None without a cubic coupling or an
    off-resonant triple. ``width_hz`` defaults to the coupling width g_max/2pi."""
    if not terms.cubic_rad_s:
        return None
    width = terms.largest_cubic_rad_s() / TWO_PI if width_hz is None else float(width_hz)
    every = three_mode_resonances(
        crystal, terms, width_hz=math.inf
    )  # sorted by |mismatch|; one pass, then filtered
    resonances = tuple(r for r in every if abs(r.mismatch_hz) < width)
    off = [r for r in every if r.mismatch_hz != 0.0]
    if not off:
        return None
    worst = max(off, key=lambda r: r.coupling_rad_s**2 / abs(TWO_PI * r.mismatch_hz))
    omega_a = crystal.modes[worst.modes[0]].omega_rad_s
    phase = integrated_cubic_phase_rad(
        worst.coupling_rad_s, omega_a, 2.0 * omega_a - TWO_PI * worst.mismatch_hz, duration_s
    )
    return AnharmonicEstimate(
        phase_rad=phase,
        coupling_rad_s=worst.coupling_rad_s,
        mismatch_hz=worst.mismatch_hz,
        modes=worst.modes,
        bound_rad=dispersive_phase_bound_rad(worst.coupling_rad_s, TWO_PI * worst.mismatch_hz, duration_s),
        resonances=resonances,
        width_hz=width,
    )


def axial_hessian_check(u: np.ndarray) -> np.ndarray:
    """sum_p u_p C_mnp = (1/2)(delta_mn - A_mn) - (the Euler identity behind D_mn2): returns the residual matrix."""
    c = marquet_c_tensor(u)
    lhs = np.einsum("p,mnp->mn", np.asarray(u, dtype=float), c)
    rhs = 0.5 * (np.eye(len(u)) - axial_hessian_dimensionless(u))
    return np.asarray(lhs - rhs)


__all__ = [
    "AnharmonicEstimate",
    "AnharmonicTerms",
    "Resonance",
    "anharmonic_estimate",
    "axial_cubic_closed_form_rad_s",
    "axial_hessian_check",
    "coulomb_anharmonic_terms",
    "coulomb_fourth_derivatives",
    "coulomb_third_derivatives",
    "coupling_g_rad_s",
    "dispersive_phase_bound_rad",
    "integrated_cubic_phase_rad",
    "marquet_c_tensor",
    "marquet_d_coefficients",
    "marquet_selection_rules",
    "nonlinearity_epsilon",
    "three_mode_resonances",
    "zero_point_matrix",
]
