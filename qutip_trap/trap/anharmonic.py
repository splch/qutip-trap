"""Cubic and quartic Coulomb mode couplings and the three-mode resonance checker (PLAN.md Section 4.1.4; M1).

Expanding the Coulomb-plus-trap energy about the equilibrium to third and fourth order gives
V_3 = (1/6) sum_abc V_abc xi_a xi_b xi_c and V_4 = (1/24) sum_abcd V_abcd xi_a xi_b xi_c xi_d over the 3N Cartesian
displacements, with V_abc.. the exact derivatives of sum_{i<j} k Z_i Z_j/|r_i - r_j| (the harmonic trap contributes
nothing beyond second order; the electrode potential's own anharmonicity is the separate H_curv of Section 5.7).
With xi_{(i,a)} = sum_k e_{(i,a),k} (a_k + a_k^dag), e_{(i,a),k} = c_{(i,a)}^{(k)} sqrt(hbar/(2 m_i omega_k)) the mass-weighted
eigenvectors of ``trap/crystal.py`` (Section 4.1.7), the interaction in mode space is

    H_anh/hbar = sum_{k<=l<=m} cubic_rad_s[(k, l, m)] X_k X_l X_m + sum_{k<=l<=m<=n} quartic_rad_s[(k, l, m, n)] X_k X_l X_m X_n,

X = a + a^dag, the coefficient of each SORTED index tuple carrying its permutation multiplicity, which is the
convention ``AnharmonicTerms`` stores and the Hamiltonian builder of Section 5.7 consumes.

For an equal-mass linear chain this is Marquet, Schmidt-Kaler and James 2003: with the dimensionless tensor
C_mnp = (1/6) d^3 F/du_m du_n du_p of the Coulomb sum F = sum_{m<n} 1/|u_m - u_n| at the James equilibrium and
D_pqr = sum C_mnp b_m^(p) b_n^(q) b_r^(r) its mode-space form, sum_p C_mnp = 0 decouples the centre-of-mass mode
(D_mn1 = 0 for every N), sum_p u_p C_mnp = (1/2)(delta_mn - A_mn) gives D_mn2 = ((1 - mu_m)/(2 |u|)) delta_mn with
|u| = sqrt(sum u_p^2) (D_222 = -1.1225 at N = 2), and the axial coefficient of X_p X_q X_r is
2 eps omega_z D_pqr (mu_p mu_q mu_r)^{-1/4} times the multiplicity, with the dimensionless nonlinearity
eps = x0/(4 l) = sqrt(hbar/(2 M omega_z))/(4 l): 1.06e-3 for 9Be+ at omega_z = 2 pi x 5.0 MHz, 7.09e-4 for 40Ca+ at 2 MHz,
3.79e-4 for 171Yb+ at 0.2 MHz, and g = eps omega_z (g/2pi = 1419 Hz and 76 Hz for the last two). The general Cartesian
route above reproduces this closed form for equal masses and is what mixed crystals use.

The leading off-resonant effect of the cubic term is SECOND order, of order g^2/Delta_res with virtual population
(g/Delta_res)^2 and Delta_res = |omega_m +- omega_n - omega_p| the distance to the nearest three-mode resonance; the
first version of the plan multiplied g by the gate time, a first-order figure the 2026-09-04 critique withdrew
(``check_anharmonic.py``: 0.003 to 0.004 rad over 100 us at g/2pi = 1.419 kHz against 0.89 rad). The resonance checker
lists the mode triples within a coupling width of Marquet's condition (their Eq. 4.10), a necessary condition to be
confirmed by substitution; the cubic Hamiltonian itself is an opt-in term (Section 5.7).
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
    """Coupling coefficients of the optional H_anh and H_curv terms of Section 5.7, in rad/s.

    Keys are SORTED mode-index triples (cubic) and quadruples (quartic) into ``Crystal.modes``; each coefficient
    multiplies X_k X_l X_m (X = a + a^dag) once and already carries the permutation multiplicity (module docstring).
    The leading off-resonant effect of the cubic coupling is SECOND order, g^2 t/Delta, not g t (Section 4.1.4, caught
    by the 2026-09-04 experimentalist critique; ``check_anharmonic.py``).
    """

    cubic_rad_s: dict[tuple[int, int, int], float] = field(default_factory=dict)
    quartic_rad_s: dict[tuple[int, int, int, int], float] = field(default_factory=dict)
    resonance_check: bool = True
    """Section 5.7: the resonance checker and the integrated-phase estimate are ON BY DEFAULT, independently of whether
    the cubic Hamiltonian itself is built (``BuilderOptions.include_anharmonic``); the Hamiltonian builder reads this
    field and reports ``anharmonic_estimate`` in ``BuiltHamiltonian.approximations``. Set False to silence the report."""

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
    """The cubic (and optionally quartic) Coulomb couplings of a crystal in mode space, keyed by sorted mode indices (rad/s).

    Coefficients below ``cutoff_rad_s`` in magnitude are dropped. The centre-of-mass modes of an equal-mass chain decouple
    exactly (D_mn1 = 0), so their entries vanish to round-off.
    """
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
    """(max |D_mn1|, max |D_mn2 - ((1 - mu_m)/(2|u|)) delta_mn|): both zero to round-off (Section 9.1, "Anharmonic selection rules")."""
    c = marquet_c_tensor(u)
    d = marquet_d_coefficients(c, b)
    n = len(u)
    norm_u = float(np.linalg.norm(u))
    rule2 = np.array(
        [[(1.0 - mu[m]) / (2.0 * norm_u) if m == k else 0.0 for k in range(n)] for m in range(n)]
    )
    return float(np.max(np.abs(d[:, :, 0]))), float(np.max(np.abs(d[:, :, 1] - rule2)))


def nonlinearity_epsilon(mass_kg: float, omega_z_rad_s: float, *, charge: int = 1) -> float:
    """eps = x0/(4 l) with x0 = sqrt(hbar/(2 M omega_z)) and l the James length scale: 1.06e-3 for 9Be+ at 5 MHz (Section 4.1.4)."""
    x0 = math.sqrt(HBAR_J_S / (2.0 * mass_kg * omega_z_rad_s))
    return x0 / (4.0 * length_scale_m(mass_kg, omega_z_rad_s, charge=charge))


def coupling_g_rad_s(mass_kg: float, omega_z_rad_s: float, *, charge: int = 1) -> float:
    """g = eps omega_z: g/2pi = 1419 Hz for 40Ca+ at 2 MHz and 76 Hz for 171Yb+ at 0.2 MHz (Section 4.1.4)."""
    return nonlinearity_epsilon(mass_kg, omega_z_rad_s, charge=charge) * omega_z_rad_s


def axial_cubic_closed_form_rad_s(crystal: Crystal) -> dict[tuple[int, int, int], float]:
    """Marquet's axial coefficients 2 eps omega_z D_pqr (mu_p mu_q mu_r)^{-1/4} x multiplicity for an EQUAL-mass collinear chain,
    keyed like ``AnharmonicTerms.cubic_rad_s`` (axial modes are the first N entries of ``Crystal.modes``); the cross-check of
    the Cartesian route."""
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
    """Mode triples with |omega_p +- omega_q - omega_r| < width_hz and a non-zero cubic coupling: Marquet's Eq. 4.10 condition,
    necessary and to be confirmed by substitution (Section 4.1.4). Near such a triple the accumulated phase grows linearly."""
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
    """g^2 t/Delta_res: the second-order UPPER BOUND on the off-resonant phase, valid for Delta_res >> the mode-frequency spread only
    (Section 9.12: the directly integrated phases are the fixture; this label is a bound, not the scaling)."""
    if mismatch_rad_s == 0.0:
        raise ValueError("on resonance the growth is linear in time, not dispersive")
    return coupling_rad_s**2 * duration_s / abs(mismatch_rad_s)


# ---- the default-on integrated-phase estimator (Sections 4.1.4, 5.7; 9.12 row "Cubic anharmonicity gate") ------------------


def integrated_cubic_phase_rad(
    coupling_rad_s: float,
    omega_a_rad_s: float,
    omega_b_rad_s: float,
    duration_s: float,
    *,
    state: tuple[int, int] = (1, 0),
    dimensions: tuple[int, int] = (8, 6),
) -> float:
    """The DIRECTLY INTEGRATED phase of |n_a, n_b> under H = w_a a^dag a + w_b b^dag b + g (a + a^dag)^2 (b + b^dag).

    The phase is taken relative to |0, 0> and beyond free evolution, from the exact propagator:
    arg[(<s|U|s>/<s|U_0|s>) / (<0|U|0>/<0|U_0|0>)]. This is the fixture of Section 9.12's "Cubic anharmonicity gate"
    row, which the 2026-09-04 numerics critique made the reported quantity: at g/2pi = 1.419 kHz over 100 us, mismatches
    of 1, 0.3, 0.1 and 0.03 MHz (w_b = 2 w_a - 2 pi Delta, the 2:1 three-phonon resonance) give 0.0041, 0.0034, 0.0032
    and 0.0032 rad on |1, 0> - essentially FLAT, where the second-order label g^2 t/Delta_res would give 0.0013, 0.0042,
    0.0127 and 0.0422 rad (3x low at 1 MHz, 13x high at 0.03 MHz), and 0.016 rad on the resonant pair |2, 0> <-> |0, 1>
    at 0.1 MHz. The withdrawn first-order figure g t is 0.89 rad over the same 100 us
    (``validation/scripts/check_anharmonic.py``, whose two-mode model this reproduces).

    ``dimensions`` are the two modes' Fock truncations; the leaked population is below 1e-6 over the whole fixture, so
    the diagonal element is a phase and not an amplitude.
    """
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
    """What the default-on estimator of Section 5.7 reports: the nearest three-mode resonance and the integrated phase.

    ``phase_rad`` is ``integrated_cubic_phase_rad`` evaluated for the worst triple (the one with the largest second-order
    bound g^2 t/Delta_res) in the canonical two-mode 2:1 model at that triple's coupling and mismatch; ``bound_rad`` is
    that bound itself, which Section 9.12 keeps ONLY as an explicit upper bound. ``resonances`` are the triples inside the
    coupling width, where the growth is linear in time instead and the estimate does not apply.
    """

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
    """The Section 5.7 default-on report: run the resonance checker and integrate the phase; None with no cubic coupling.

    ``width_hz`` defaults to the coupling width g_max/2pi (Section 4.1.4: "the resonance checker lists the mode triples
    within a coupling width of Marquet's condition"). The triple the phase is quoted for is the one maximizing the
    second-order bound g^2/|Delta_res|, i.e. the worst case; an exactly resonant triple has no dispersive phase at all
    (the growth is linear in time) and is reported through ``resonances`` instead.
    """
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
