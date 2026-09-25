"""Cubic and quartic Coulomb mode couplings, the three-mode resonance checker and its phase estimate (PLAN.md 4.1.4).

The Coulomb energy expanded about the equilibrium gives V_3 = (1/6) sum V_abc xi_a xi_b xi_c (and V_4 likewise) over the
3N Cartesian displacements, with xi_(i,a) = sum_k c_(i,a),k sqrt(hbar/(2 m_i omega_k)) (a_k + a_k^dag), so in mode space
H_anh/hbar = sum_{k<=l<=m} cubic_rad_s[(k, l, m)] X_k X_l X_m + (quartic), X = a + a^dag, each SORTED key carrying its
permutation multiplicity. For an equal-mass chain this is Marquet, Schmidt-Kaler and James 2003 (the centre-of-mass mode
decouples). The leading off-resonant effect of the cubic term is second order, so the reported quantity is the directly
integrated phase of a two-mode model, with g^2 t/Delta_res only as an upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, permutations

import numpy as np
from scipy.linalg import expm

from qutip_trap.trap.crystal import K_COULOMB_J_M, Crystal
from qutip_trap.units import HBAR_J_S, TWO_PI


@dataclass(frozen=True)
class AnharmonicTerms:
    """Cubic and quartic couplings in rad/s keyed by SORTED mode-index tuples into ``Crystal.modes`` (multiplicity inside).

    ``resonance_check`` switches the builder's default-on resonance report (``anharmonic_estimate``) independently of
    whether the cubic Hamiltonian itself is built (``BuilderOptions.include_anharmonic``).
    """

    cubic_rad_s: dict[tuple[int, int, int], float] = field(default_factory=dict)
    quartic_rad_s: dict[tuple[int, int, int, int], float] = field(default_factory=dict)
    resonance_check: bool = True

    def __post_init__(self) -> None:
        for key in self.cubic_rad_s:
            if len(key) != 3 or list(key) != sorted(key):
                raise ValueError(f"cubic keys are sorted mode-index triples, got {key}")
        for key4 in self.quartic_rad_s:
            if len(key4) != 4 or list(key4) != sorted(key4):
                raise ValueError(f"quartic keys are sorted mode-index quadruples, got {key4}")

    def largest_cubic_rad_s(self) -> float:
        return max((abs(v) for v in self.cubic_rad_s.values()), default=0.0)


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
    """V_abc over the 3N coordinates: each pair's tensor with the sign (-1)^{number of j labels}."""
    pos = np.asarray(positions_m, dtype=float)
    n = pos.shape[0]
    z = np.ones(n) if charges is None else np.asarray(charges, dtype=float)
    out = np.zeros((3 * n, 3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            t = K_COULOMB_J_M * z[i] * z[j] * _third_pair_tensor(pos[i] - pos[j])
            for bits in range(8):
                ions = tuple(j if (bits >> s) & 1 else i for s in range(3))
                sign = (-1.0) ** sum(1 for x in ions if x == j)
                out[tuple(slice(3 * x, 3 * x + 3) for x in ions)] += sign * t
    return out


def coulomb_fourth_derivatives(positions_m: np.ndarray, charges: np.ndarray | None = None) -> np.ndarray:
    """V_abcd over the 3N coordinates (the same sign rule with four labels)."""
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
                out[tuple(slice(3 * x, 3 * x + 3) for x in ions)] += sign * t
    return out


def zero_point_matrix(crystal: Crystal) -> np.ndarray:
    """e_{(i,a),k} = c_{(i,a)}^{(k)} sqrt(hbar/(2 m_i omega_k)): coordinate (i, a)'s zero-point amplitude in mode k."""
    masses = crystal.masses_kg
    e = np.zeros((3 * crystal.n_ions, len(crystal.modes)))
    for k, mode in enumerate(crystal.modes):
        x0 = np.sqrt(HBAR_J_S / (2.0 * masses * mode.omega_rad_s))
        e[:, k] = (mode.displacement_pattern() * x0[:, None]).ravel()
    return e


def coulomb_anharmonic_terms(
    crystal: Crystal, *, quartic: bool = False, cutoff_rad_s: float = 0.0
) -> AnharmonicTerms:
    """The cubic (and optionally quartic) Coulomb couplings of a crystal in mode space (rad/s), below ``cutoff_rad_s`` dropped."""
    e = zero_point_matrix(crystal)
    n_modes = e.shape[1]
    d3 = np.einsum("abc,ak,bl,cm->klm", coulomb_third_derivatives(crystal.positions_m), e, e, e) / (
        6.0 * HBAR_J_S
    )
    cubic: dict[tuple[int, int, int], float] = {}
    for key in combinations_with_replacement(range(n_modes), 3):
        val = float(len(set(permutations(key))) * d3[key])
        if abs(val) > cutoff_rad_s:
            cubic[key] = val
    quart: dict[tuple[int, int, int, int], float] = {}
    if quartic:
        v4 = coulomb_fourth_derivatives(crystal.positions_m)
        d4 = np.einsum("abcd,ak,bl,cm,dn->klmn", v4, e, e, e, e) / (24.0 * HBAR_J_S)
        for key4 in combinations_with_replacement(range(n_modes), 4):
            val = float(len(set(permutations(key4))) * d4[key4])
            if abs(val) > cutoff_rad_s:
                quart[key4] = val
    return AnharmonicTerms(cubic_rad_s=cubic, quartic_rad_s=quart)


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
    """Mode triples with |omega_p +- omega_q - omega_r| < width_hz and a non-zero cubic coupling, nearest first
    (Marquet's Eq. 4.10 condition: necessary, to be confirmed by substitution)."""
    w = np.array([m.omega_hz for m in crystal.modes])
    found: list[Resonance] = []
    for key, g in terms.cubic_rad_s.items():
        for a, b, c in set(permutations(key)):
            for sign in (+1, -1):
                mismatch = w[a] + sign * w[b] - w[c]
                if abs(mismatch) < width_hz and not (sign == -1 and a == b):
                    found.append(
                        Resonance(modes=(a, b, c), sign=sign, mismatch_hz=float(mismatch), coupling_rad_s=g)
                    )
    return tuple(sorted(found, key=lambda r_: abs(r_.mismatch_hz)))


def dispersive_phase_bound_rad(coupling_rad_s: float, mismatch_rad_s: float, duration_s: float) -> float:
    """g^2 t/Delta_res: the second-order upper bound on the off-resonant phase (not its scaling)."""
    if mismatch_rad_s == 0.0:
        raise ValueError("on resonance the growth is linear in time, not dispersive")
    return coupling_rad_s**2 * duration_s / abs(mismatch_rad_s)


def integrated_cubic_phase_rad(
    coupling_rad_s: float,
    omega_a_rad_s: float,
    omega_b_rad_s: float,
    duration_s: float,
    *,
    state: tuple[int, int] = (1, 0),
    dimensions: tuple[int, int] = (8, 6),
) -> float:
    """The phase of |n_a, n_b> beyond free evolution, relative to |0, 0>, under H = w_a a^dag a + w_b b^dag b +
    g (a + a^dag)^2 (b + b^dag), from the exact propagator: arg[(<s|U|s>/<s|U_0|s>)/(<0|U|0>/<0|U_0|0>)].

    ``dimensions`` are the two Fock truncations (the leaked population is below 1e-6 on the 2:1 fixture).
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
    u = expm(-1j * (h0 + coupling_rad_s * (x_a @ x_a) @ x_b) * duration_s)
    u0 = expm(-1j * h0 * duration_s)
    i = state[0] * d_b + state[1]
    return float(np.angle((u[i, i] / u0[i, i]) / (u[0, 0] / u0[0, 0])))


@dataclass(frozen=True)
class AnharmonicEstimate:
    """The default-on report: the integrated phase for the worst triple (largest g^2 t/Delta_res) in the two-mode 2:1
    model, that bound itself, and the triples inside the coupling width, where growth is linear and the estimate fails."""

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
            f"anharmonic resonance check (on by default): {near}; integrated cubic phase "
            f"{self.phase_rad:+.3e} rad on modes {self.modes} at g/2pi = {self.coupling_rad_s / TWO_PI:.4g} Hz and "
            f"Delta_res = {self.mismatch_hz:.4g} Hz (the g^2 t/Delta bound alone would read {self.bound_rad:.3e} rad)"
        )


def anharmonic_estimate(
    crystal: Crystal, terms: AnharmonicTerms, duration_s: float, *, width_hz: float | None = None
) -> AnharmonicEstimate | None:
    """Run the resonance checker and integrate the phase; None with no cubic coupling or no off-resonant triple.

    ``width_hz`` defaults to the coupling width g_max/2pi; an exactly resonant triple has no dispersive phase and is
    reported through ``resonances`` instead.
    """
    if not terms.cubic_rad_s:
        return None
    width = terms.largest_cubic_rad_s() / TWO_PI if width_hz is None else float(width_hz)
    every = three_mode_resonances(crystal, terms, width_hz=math.inf)
    off = [r for r in every if r.mismatch_hz != 0.0]
    if not off:
        return None
    worst = max(off, key=lambda r: r.coupling_rad_s**2 / abs(TWO_PI * r.mismatch_hz))
    omega_a = crystal.modes[worst.modes[0]].omega_rad_s
    return AnharmonicEstimate(
        phase_rad=integrated_cubic_phase_rad(
            worst.coupling_rad_s, omega_a, 2.0 * omega_a - TWO_PI * worst.mismatch_hz, duration_s
        ),
        coupling_rad_s=worst.coupling_rad_s,
        mismatch_hz=worst.mismatch_hz,
        modes=worst.modes,
        bound_rad=dispersive_phase_bound_rad(worst.coupling_rad_s, TWO_PI * worst.mismatch_hz, duration_s),
        resonances=tuple(r for r in every if abs(r.mismatch_hz) < width),
        width_hz=width,
    )
