"""The displacement operator, its exact elements and truncation tolerances, Debye-Waller factors and qubit operators.

x = x0 (a + a^dagger), x0 = sqrt(hbar/(2 m omega)); a running-wave drive's motional operator is D(i eta), with
D(alpha) = exp(alpha a^dagger - alpha^* a). The Hamiltonian uses the matrix exponential of the truncated generator
(unitary, wrong near the cap); the analytic elements (exact, not unitary once truncated) are the oracle and give the
Rabi frequencies Omega |<n'|D(i eta)|n>|. Qubit index 0 is the lower level |0> = |down> (``Species.qubit[0]``);
sigma_+ = |1><0|, and the energy operator sigma_z = |1><1| - |0><0| is the negative of the computational Pauli Z.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import qutip as qt
from scipy.special import erf, eval_genlaguerre, gammaln

M2 = "milestone M2 (hilbert/operators.py, PLAN.md Section 5.1.1)"

# ---- truncation table, rows d, columns TABLE_ETA: max |D_expm - D_analytic| over n, n' < d/2, and the norm loss -------
TABLE_ETA: Final[tuple[float, ...]] = (0.1, 0.5, 1.0)
TABLE_ELEMENT_ERROR: Final[dict[int, tuple[float, ...]]] = {
    8: (2e-13, 2e-6, 1e-3),
    10: (3e-16, 7e-8, 2e-4),
    20: (1e-16, 1e-15, 3e-9),
    40: (7e-16, 8e-16, 7e-16),
}
TABLE_NORM_LOSS: Final[dict[int, tuple[float, ...]]] = {
    8: (3e-8, 6e-3, 0.3),
    10: (2e-10, 1e-3, 0.18),
    20: (1e-16, 3e-8, 3e-3),
    40: (1e-15, 4e-16, 3e-9),
}
ORACLE_FLOOR: Final[float] = 1e-12
"""The oracle tolerance floor: 1e-12 where the table reaches it, the table's value otherwise."""
MARGIN_POINTS: Final[tuple[tuple[float, int], ...]] = ((0.1, 6), (0.5, 10), (1.0, 20))
"""(|eta|, margin levels) points the table implies, interpolated in eta by ``required_margin``."""


# ---- analytic elements ---------------------------------------------------------------------------------------------------


def displacement_element_analytic(n_row: int, n_col: int, alpha: complex) -> complex:
    """<n_row|D(alpha)|n_col>, the exact infinite-space element: for n' = ``n_row`` >= n = ``n_col``,
    sqrt(n!/n'!) e^{-|alpha|^2/2} alpha^{n'-n} L^{(n'-n)}_n(|alpha|^2); for n' < n the factor is (-alpha^*)^{n-n'} with n
    and n' interchanged elsewhere. At alpha = i eta both carry (i eta)^{|n'-n|}."""
    if n_row < 0 or n_col < 0:
        raise ValueError("Fock indices are non-negative")
    a = complex(alpha)
    x = abs(a) ** 2
    lo, hi = min(n_row, n_col), max(n_row, n_col)
    k = hi - lo
    pref = math.exp(0.5 * (gammaln(lo + 1) - gammaln(hi + 1)) - x / 2.0)
    factor = a**k if n_row >= n_col else (-np.conj(a)) ** k
    return complex(pref * factor * float(eval_genlaguerre(lo, k, x)))


def displacement_matrix_analytic(d: int, alpha: complex) -> np.ndarray:
    """The d x d matrix of exact elements <n'|D(alpha)|n> (rows n', columns n); NOT unitary once truncated."""
    if d < 1:
        raise ValueError("d must be positive")
    a = complex(alpha)
    x = abs(a) ** 2
    rows, cols = np.meshgrid(np.arange(d), np.arange(d), indexing="ij")
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    k = hi - lo
    pref = np.exp(0.5 * (gammaln(lo + 1) - gammaln(hi + 1)) - x / 2.0)
    lag = eval_genlaguerre(lo, k, x)
    factor = np.where(rows >= cols, a ** k.astype(float), (-np.conj(a)) ** k.astype(float))
    out: np.ndarray = (pref * lag * factor).astype(complex)
    return out


def displacement_operator(d: int, alpha: complex, *, construction: str = "expm") -> qt.Qobj:
    """D(alpha) in a space of d Fock levels: ``expm`` of the truncated generator (the Hamiltonian's operator, unitary)
    or ``analytic`` (the oracle, exact element by element, non-unitary at the boundary); CSR either way."""
    if d < 2:
        raise ValueError("a displacement operator needs at least two Fock levels")
    a = complex(alpha)
    if construction == "expm":
        ann = qt.destroy(d)
        op = (a * ann.dag() - np.conj(a) * ann).expm()
    elif construction == "analytic":
        op = qt.Qobj(displacement_matrix_analytic(d, a), dims=[[d], [d]])
    else:
        raise ValueError("construction is 'expm' or 'analytic'")
    return op.to("CSR")


def oracle_check(d: int, alpha: complex, n_hi: int, tolerance: float) -> float:
    """Max |D_expm - D_analytic| over the block n, n' <= n_hi; raises ValueError if it exceeds ``tolerance``."""
    if not 0 <= n_hi < d:
        raise ValueError("the populated range must lie inside the truncated space")
    exp_ = displacement_operator(d, alpha).full()
    ana = displacement_matrix_analytic(d, alpha)
    diff = float(np.max(np.abs(exp_[: n_hi + 1, : n_hi + 1] - ana[: n_hi + 1, : n_hi + 1])))
    if diff > tolerance:
        raise ValueError(
            f"displacement oracle: expm and analytic elements differ by {diff:.2e} over n <= {n_hi} in a space of "
            f"d = {d} at alpha = {alpha}, above the Section 5.1.1 tolerance {tolerance:.1e}; raise the cap"
        )
    return diff


def analytic_norm_loss(d: int, alpha: complex, column: int) -> float:
    """1 - sum_n' |<n'|D|n = column>|^2 in the truncated analytic matrix: the probability the drive pushes out of the space."""
    ana = displacement_matrix_analytic(d, alpha)
    return float(1.0 - np.sum(np.abs(ana[:, column]) ** 2))


def _interp_log(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    lx = math.log10(x)
    lxs = [math.log10(v) for v in xs]
    lys = [math.log10(v) for v in ys]
    if lx <= lxs[0]:
        return 10 ** lys[0]
    if lx >= lxs[-1]:
        return 10 ** lys[-1]
    return float(10 ** np.interp(lx, lxs, lys))


def interior_tolerance(margin_levels: int, eta: float) -> float:
    """The oracle tolerance for interior elements ``margin_levels`` below the cap at Lamb-Dicke parameter |eta|: the table
    row d = 2 * margin (its block n, n' < d/2 sits d/2 below the cap), interpolated log-linearly in |eta| and d, clamped to
    the table's rows and floored at ``ORACLE_FLOOR``."""
    if margin_levels < 1:
        raise ValueError("an interior element needs at least one level of margin")
    e = max(abs(eta), 1e-3)
    ds = sorted(TABLE_ELEMENT_ERROR)
    d_eff = 2 * margin_levels
    per_row = [_interp_log(e, TABLE_ETA, TABLE_ELEMENT_ERROR[d]) for d in ds]
    if d_eff <= ds[0]:
        val = per_row[0]
    elif d_eff >= ds[-1]:
        val = per_row[-1]
    else:
        val = float(10 ** np.interp(d_eff, ds, [math.log10(v) for v in per_row]))
    return max(val, ORACLE_FLOOR)


def required_margin(
    eta: float, *, tail: float | None = None, n_hi: int = 0, element_tol: float | None = None
) -> int:
    """Levels above the populated range a resolved mode's cap keeps: the ``MARGIN_POINTS`` fixture, or with ``element_tol``
    and/or ``tail`` the smallest (memoized) margin at which the expm and analytic elements over n, n' <= ``n_hi`` agree to
    ``element_tol`` and one displacement from ``n_hi`` leaks less than ``tail`` past the cap, at most the fixture."""
    fixture = _fixture_margin(eta)
    if tail is None and element_tol is None:
        return fixture
    return _derived_margin(round(abs(float(eta)), 9), tail, int(n_hi), element_tol, fixture)


def _fixture_margin(eta: float) -> int:
    e = abs(eta)
    xs = [p[0] for p in MARGIN_POINTS]
    ys = [float(p[1]) for p in MARGIN_POINTS]
    if e <= xs[0]:
        return int(ys[0])
    if e > xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return int(math.ceil(ys[-1] + slope * (e - xs[-1])))
    return int(math.ceil(float(np.interp(e, xs, ys)) - 1e-9))


@functools.lru_cache(maxsize=4096)
def _derived_margin(
    eta: float, tail: float | None, n_hi: int, element_tol: float | None, fixture: int
) -> int:
    if n_hi < 0:
        raise ValueError("the populated range is non-negative")
    if tail is not None and not 0.0 < tail < 1.0:
        raise ValueError("tail is a population fraction in (0, 1)")
    if element_tol is not None and element_tol <= 0.0:
        raise ValueError("element_tol is a positive tolerance")
    for m in range(1, fixture):
        if tail is not None and displacement_leakage(eta, n_hi, m) >= tail:
            continue
        if element_tol is not None and interior_element_error(n_hi + 1 + m, eta, n_hi) > element_tol:
            continue
        return m
    return fixture


def displacement_leakage(eta: float, n_hi: int, margin: int, *, terms: int = 60) -> float:
    """sum_{dn > margin} |<n_hi + dn|D(i eta)|n_hi>|^2 over ``terms`` analytic elements: the population one displacement
    moves from level ``n_hi`` to more than ``margin`` levels above it."""
    if n_hi < 0 or margin < 0:
        raise ValueError("n_hi and margin are non-negative")
    alpha = 1j * abs(float(eta))
    return float(
        sum(
            abs(displacement_element_analytic(n_hi + dn, n_hi, alpha)) ** 2
            for dn in range(margin + 1, margin + 1 + terms)
        )
    )


def interior_element_error(d: int, eta: float, n_hi: int) -> float:
    """max |D_expm - D_analytic| over the block n, n' <= n_hi in a space of ``d`` levels at alpha = i eta."""
    if not 0 <= n_hi < d:
        raise ValueError("the populated range must lie inside the truncated space")
    exp_ = displacement_operator(d, 1j * abs(float(eta))).full()
    ana = displacement_matrix_analytic(d, 1j * abs(float(eta)))
    return float(np.max(np.abs(exp_[: n_hi + 1, : n_hi + 1] - ana[: n_hi + 1, : n_hi + 1])))


# ---- Rabi table and Debye-Waller factors -------------------------------------------------------------------------------


def rabi_matrix_element(n_row: int, n_col: int, eta: float) -> float:
    """Omega_{n', n}/Omega = |<n'|D(i eta)|n>| = e^{-eta^2/2} sqrt(n_<!/n_>!) |eta|^{|n'-n|} |L^{(|n'-n|)}_{n_<}(eta^2)| (Wineland 1998 Eq. 18)."""
    return abs(displacement_element_analytic(n_row, n_col, 1j * eta))


def rabi_table(d: int, eta: float) -> np.ndarray:
    """|<n'|D(i eta)|n>| for n, n' < d, from the analytic elements."""
    return np.abs(displacement_matrix_analytic(d, 1j * eta))


def sideband_phase_rad(n_row: int, n_col: int) -> float:
    """The (pi/2)|n' - n| the drive phase acquires on the |n' - n|-th sideband: the phase of (i eta)^{|n'-n|} (Wineland Eq. 21)."""
    return 0.5 * math.pi * abs(n_row - n_col)


def debye_waller_factor(n: int, eta: float) -> float:
    """The carrier element <n|D(i eta)|n> = e^{-eta^2/2} L_n(eta^2), signed (Monroe 2020 Eq. 11)."""
    return float(math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2))


def thermal_populations(nbar: float, d: int) -> np.ndarray:
    """P_n = nbar^n/(1 + nbar)^{n+1} for n < d (unnormalized by the truncation; the tail is reported by callers)."""
    if nbar < 0.0:
        raise ValueError("nbar is non-negative")
    n = np.arange(d)
    if nbar == 0.0:
        out = np.zeros(d)
        out[0] = 1.0
        return out
    log_p = n * math.log(nbar) - (n + 1) * math.log1p(nbar)
    return np.asarray(np.exp(log_p))


def displaced_thermal_populations(alpha_abs: float, nbar: float, d: int) -> np.ndarray:
    """P(n) of a thermal state at ``nbar`` displaced by |alpha|: sum_k P_th(k) |<n|D(alpha)|k>|^2 over ``d`` levels, from the
    analytic elements."""
    p_th = thermal_populations(nbar, d)
    p_th = p_th / p_th.sum()
    mat = np.abs(displacement_matrix_analytic(d, 1j * abs(alpha_abs))) ** 2
    return np.asarray(mat @ p_th, dtype=float)


def populated_range(alpha_abs: float, nbar: float, *, tail: float = 1e-6) -> int:
    """The highest Fock index a thermal mode displaced by |alpha| populates: the smallest n whose population above it is
    below ``tail``. The margin is measured from this index."""
    if alpha_abs < 0.0 or nbar < 0.0:
        raise ValueError("|alpha| and nbar are non-negative")
    if not 0.0 < tail < 1.0:
        raise ValueError("tail is a population fraction in (0, 1)")
    d = int(math.ceil(30.0 + 6.0 * (alpha_abs**2 + nbar) + 12.0 * math.sqrt(alpha_abs**2 + nbar)))
    p = displaced_thermal_populations(alpha_abs, nbar, d)
    above = np.cumsum(p[::-1])[::-1]
    for n in range(p.size):
        if n + 1 >= p.size or above[n + 1] < tail:
            return int(n)
    return int(p.size - 1)


def thermal_debye_waller_mean(eta: float, nbar: float, *, n_max: int | None = None) -> float:
    """sum_n P_n e^{-eta^2/2} L_n(eta^2), the thermal mean of the carrier Debye-Waller factor (Wineland 1998 Eq. 124)."""
    if n_max is None:
        n_max = int(50 + 40 * nbar)
    p = thermal_populations(nbar, n_max + 1)
    dw = np.array([debye_waller_factor(n, eta) for n in range(n_max + 1)])
    return float(np.sum(p * dw))


def thermal_debye_waller_approx(eta: float, nbar: float) -> float:
    """exp[-eta^2 (nbar + 1/2)], the Lamb-Dicke-regime approximation of the thermal mean (Wineland 1998 Eq. 124)."""
    return math.exp(-(eta**2) * (nbar + 0.5))


def debye_waller_rms_fraction(etas: Sequence[float], nbars: Sequence[float]) -> float:
    """rms fractional Rabi-frequency spread over frozen thermal spectators, sqrt(sum_p eta_p^4 nbar_p (nbar_p + 1))
    (Wineland 1998 Eq. 127)."""
    if len(etas) != len(nbars):
        raise ValueError("one nbar per spectator mode")
    return math.sqrt(sum(float(e) ** 4 * float(nb) * (float(nb) + 1.0) for e, nb in zip(etas, nbars)))


def probability_within(delta: float, rms: float) -> float:
    """P(|Omega/Omega_mean - 1| < delta) = erf(delta/(sqrt 2 rms)) for a Gaussian spread (Wineland 1998 Eq. 127)."""
    if rms <= 0.0:
        return 1.0
    return float(erf(delta / (math.sqrt(2.0) * rms)))


def sideband_operators(matrix: np.ndarray) -> dict[int, np.ndarray]:
    """Split a single-mode operator into its sideband parts A_k with n' - n = k (the interaction-picture decomposition)."""
    d = matrix.shape[0]
    out: dict[int, np.ndarray] = {}
    for k in range(-(d - 1), d):
        part = np.zeros_like(matrix)
        for n in range(d):
            m = n + k
            if 0 <= m < d:
                part[m, n] = matrix[m, n]
        if np.any(part):
            out[k] = part
    return out


# ---- qubit operators in the computational ordering ----------------------------------------------------------------------


def qudit_projector(d: int, level: int) -> qt.Qobj:
    if not 0 <= level < d:
        raise ValueError(f"level {level} outside a qudit of dimension {d}")
    return qt.basis(d, level).proj()


def qudit_transition(d: int, upper: int, lower: int) -> qt.Qobj:
    """|upper><lower|: the raising operator between two levels of a qudit (sigma_+ for (1, 0))."""
    return qt.basis(d, upper) * qt.basis(d, lower).dag()


def qudit_sigma_plus(d: int = 2) -> qt.Qobj:
    """sigma_+ = |1><0|: raises the lower qubit level (index 0) to the upper (index 1)."""
    return qudit_transition(d, 1, 0)


def qudit_sigma_minus(d: int = 2) -> qt.Qobj:
    return qudit_transition(d, 0, 1)


def qudit_sigma_z(d: int = 2) -> qt.Qobj:
    """The ENERGY sigma_z = |1><1| - |0><0| (upper level positive), the negative of the computational Pauli Z."""
    return qudit_projector(d, 1) - qudit_projector(d, 0)


def qudit_sigma_phi(phi_rad: float, d: int = 2) -> qt.Qobj:
    """sigma_phi = e^{i phi} sigma_+ + e^{-i phi} sigma_- = cos(phi) X + sin(phi) Y on the qubit pair."""
    return np.exp(1j * phi_rad) * qudit_sigma_plus(d) + np.exp(-1j * phi_rad) * qudit_sigma_minus(d)
