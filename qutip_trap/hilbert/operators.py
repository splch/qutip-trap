"""The exact displacement operator, its Laguerre oracle, the margin rule and the qudit operators (PLAN.md Section 5.1.1).

Conventions:

- x = x0 (a + a^dag) with x0 = sqrt(hbar/(2 m omega)); a running-wave drive carries exp[i eta (a + a^dag)] = D(i eta), the
  displacement D(alpha) = exp(alpha a^dag - alpha^* a) at alpha = i eta.
- <n'|D(alpha)|n> = sqrt(n!/n'!) e^{-|alpha|^2/2} alpha^{n'-n} L^{(n'-n)}_n(|alpha|^2) for n' >= n and
  sqrt(n'!/n!) e^{-|alpha|^2/2} (-alpha^*)^{n-n'} L^{(n-n')}_{n'}(|alpha|^2) for n' < n; at alpha = i eta both are
  (i eta)^{|n'-n|} times the real rest.
- The Rabi frequency of |down, n> <-> |up, n'> is Omega |<n'|D(i eta)|n>|; the Hamiltonian carries the signed element.
- The Hamiltonian's operator is the matrix exponential of the truncated generator (unitary, wrong near the cap); the
  analytic elements are the oracle it is checked against over the populated range (exact, not unitary).
- Qubit index 0 is the lower level |0> = |down>; sigma_+ = |1><0|; the energy operator sigma_z = |1><1| - |0><0| is the
  negative of the computational Pauli Z, so H_int = (Delta/2) sigma_z puts the upper level Delta/2 above the frame.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import qutip as qt
from scipy.special import eval_genlaguerre, gammaln

TABLE_ETA: Final[tuple[float, ...]] = (0.1, 0.5, 1.0)
TABLE_ELEMENT_ERROR: Final[dict[int, tuple[float, ...]]] = {
    8: (2e-13, 2e-6, 1e-3),
    10: (3e-16, 7e-8, 2e-4),
    20: (1e-16, 1e-15, 3e-9),
    40: (7e-16, 8e-16, 7e-16),
}
"""The Section 5.1.1 table: max |D_expm - D_analytic| over the block n, n' < d/2, per d and per |eta| of ``TABLE_ETA``."""
ORACLE_FLOOR: Final[float] = 1e-12
"""The oracle tolerance is 1e-12 where the table reaches it, the table's own value otherwise."""
MARGIN_POINTS: Final[tuple[tuple[float, int], ...]] = ((0.1, 6), (0.5, 10), (1.0, 20))
"""(|eta|, levels above the populated range) at which the table's interior elements reach the floor; interpolated in eta."""


# ---- analytic elements ---------------------------------------------------------------------------------------------------


def displacement_element_analytic(n_row: int, n_col: int, alpha: complex) -> complex:
    """<n_row|D(alpha)|n_col>, the exact infinite-space element."""
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
    """The d x d matrix of exact elements <n'|D(alpha)|n> (rows n', columns n); not unitary once truncated."""
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


def displacement_operator(d: int, alpha: complex) -> qt.Qobj:
    """D(alpha) in a space of d Fock levels as the matrix exponential of the truncated generator (CSR, unitary)."""
    if d < 2:
        raise ValueError("a displacement operator needs at least two Fock levels")
    a = complex(alpha)
    ann = qt.destroy(d)
    return (a * ann.dag() - np.conj(a) * ann).expm().to("CSR")


def _element_error(d: int, alpha: complex, n_hi: int) -> float:
    """max |D_expm - D_analytic| over the block n, n' <= n_hi in a space of d levels."""
    if not 0 <= n_hi < d:
        raise ValueError("the populated range must lie inside the truncated space")
    exp_ = displacement_operator(d, alpha).full()
    ana = displacement_matrix_analytic(d, alpha)
    return float(np.max(np.abs(exp_[: n_hi + 1, : n_hi + 1] - ana[: n_hi + 1, : n_hi + 1])))


def oracle_check(d: int, alpha: complex, n_hi: int, tolerance: float) -> float:
    """The element error over n, n' <= n_hi; raises if it exceeds ``tolerance`` (Section 5.1.1 rule ii)."""
    diff = _element_error(d, alpha, n_hi)
    if diff > tolerance:
        raise ValueError(
            f"displacement oracle: expm and analytic elements differ by {diff:.2e} over n <= {n_hi} in a space of "
            f"d = {d} at alpha = {alpha}, above the Section 5.1.1 tolerance {tolerance:.1e}; raise the cap"
        )
    return diff


def interior_element_error(d: int, eta: float, n_hi: int) -> float:
    """max |D_expm - D_analytic| over n, n' <= n_hi at alpha = i |eta| in a space of d levels."""
    return _element_error(d, 1j * abs(float(eta)), n_hi)


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
    """The Section 5.1.1 tolerance for interior elements ``margin_levels`` below the cap at |eta|.

    A margin m reads the table's row d = 2m (its block n, n' < d/2 sits d/2 below the cap), interpolated log-linearly in
    |eta| and d and floored at ``ORACLE_FLOOR``; margins below the smallest row take its d = 8 row.
    """
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
    """Levels a resolved mode's cap keeps above the populated range (Section 5.1.1 rule ii).

    Without keywords, the fixture ``MARGIN_POINTS`` (6 at |eta| <= 0.1, 10 at 0.5, 20 at 1.0, interpolated). With
    ``element_tol`` and/or ``tail`` the smallest margin m at which (a) the exponential's elements over n, n' <= ``n_hi``
    agree with the analytic ones to ``element_tol`` and (b) one displacement from ``n_hi`` leaks less than ``tail`` past
    the cap (``displacement_leakage``), never more than the fixture.
    """
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
        # beyond the table: the last slope extended
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
    """sum_{dn > margin} |<n_hi + dn|D(i eta)|n_hi>|^2: the population one displacement moves from ``n_hi`` to more than
    ``margin`` levels above it (analytic elements)."""
    if n_hi < 0 or margin < 0:
        raise ValueError("n_hi and margin are non-negative")
    alpha = 1j * abs(float(eta))
    return float(
        sum(
            abs(displacement_element_analytic(n_hi + dn, n_hi, alpha)) ** 2
            for dn in range(margin + 1, margin + 1 + terms)
        )
    )


# ---- Rabi table, Debye-Waller factors and populations (Section 4.3.1) -------------------------------------------------------


def rabi_matrix_element(n_row: int, n_col: int, eta: float) -> float:
    """Omega_{n', n}/Omega = |<n'|D(i eta)|n>| = e^{-eta^2/2} sqrt(n_<!/n_>!) |eta|^{|n'-n|} |L^{(|n'-n|)}_{n_<}(eta^2)|
    (Wineland 1998 Eq. 18)."""
    return abs(displacement_element_analytic(n_row, n_col, 1j * eta))


def rabi_table(d: int, eta: float) -> np.ndarray:
    """|<n'|D(i eta)|n>| for n, n' < d from the analytic elements."""
    return np.abs(displacement_matrix_analytic(d, 1j * eta))


def debye_waller_factor(n: int, eta: float) -> float:
    """The carrier element <n|D(i eta)|n> = e^{-eta^2/2} L_n(eta^2), signed (Monroe 2020 Eq. 11)."""
    return float(math.exp(-(eta**2) / 2.0) * eval_genlaguerre(n, 0, eta**2))


def thermal_populations(nbar: float, d: int) -> np.ndarray:
    """P_n = nbar^n/(1 + nbar)^{n+1} for n < d, not renormalized over the truncation."""
    if nbar < 0.0:
        raise ValueError("nbar is non-negative")
    n = np.arange(d)
    if nbar == 0.0:
        out = np.zeros(d)
        out[0] = 1.0
        return out
    log_p = n * math.log(nbar) - (n + 1) * math.log1p(nbar)
    return np.asarray(np.exp(log_p))


def _thermal_levels(nbar: float) -> int:
    """Fock levels that hold a thermal distribution at ``nbar`` for a draw or a branch enumeration."""
    return int(60 + 40 * nbar)


def displaced_thermal_populations(alpha_abs: float, nbar: float, d: int) -> np.ndarray:
    """P(n) over d levels of a thermal state at ``nbar`` displaced by |alpha|: sum_k P_th(k) |<n|D(alpha)|k>|^2 (analytic
    elements), the Fock distribution at the far point of a spin-dependent force's loop."""
    p_th = thermal_populations(nbar, d)
    p_th = p_th / p_th.sum()
    mat = np.abs(displacement_matrix_analytic(d, 1j * abs(alpha_abs))) ** 2
    return np.asarray(mat @ p_th, dtype=float)


def _highest_populated(p: np.ndarray, tail: float) -> int:
    """The smallest Fock index n of the populations ``p`` whose population above n is below ``tail``."""
    above = np.cumsum(p[::-1])[::-1]
    for n in range(p.size):
        if n + 1 >= p.size or above[n + 1] < tail:
            return n
    return int(p.size - 1)


def populated_range(alpha_abs: float, nbar: float, *, tail: float = 1e-6) -> int:
    """The highest Fock index a thermal mode at ``nbar`` displaced by |alpha| populates above ``tail`` (Section 5.5): the
    range the cap rule and the engine's margin check measure the Section 5.1.1 margin from."""
    if alpha_abs < 0.0 or nbar < 0.0:
        raise ValueError("|alpha| and nbar are non-negative")
    if not 0.0 < tail < 1.0:
        raise ValueError("tail is a population fraction in (0, 1)")
    d = int(math.ceil(30.0 + 6.0 * (alpha_abs**2 + nbar) + 12.0 * math.sqrt(alpha_abs**2 + nbar)))
    return int(_highest_populated(displaced_thermal_populations(alpha_abs, nbar, d), tail))


def sideband_operators(matrix: np.ndarray) -> dict[int, np.ndarray]:
    """A single-mode operator split into its sideband parts A_k with n' - n = k (the interaction-picture decomposition)."""
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


# ---- qudit operators in the computational ordering -----------------------------------------------------------------------


def qudit_projector(d: int, level: int) -> qt.Qobj:
    if not 0 <= level < d:
        raise ValueError(f"level {level} outside a qudit of dimension {d}")
    return qt.basis(d, level).proj()


def qudit_transition(d: int, upper: int, lower: int) -> qt.Qobj:
    """|upper><lower| on a qudit (sigma_+ for (1, 0))."""
    return qt.basis(d, upper) * qt.basis(d, lower).dag()


def qudit_sigma_plus(d: int = 2) -> qt.Qobj:
    """sigma_+ = |1><0|."""
    return qudit_transition(d, 1, 0)


def qudit_sigma_minus(d: int = 2) -> qt.Qobj:
    return qudit_transition(d, 0, 1)


def qudit_sigma_z(d: int = 2) -> qt.Qobj:
    """The energy operator |1><1| - |0><0|, the negative of the computational Pauli Z."""
    return qudit_projector(d, 1) - qudit_projector(d, 0)
