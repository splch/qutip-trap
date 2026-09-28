"""The exact displacement operator, its Laguerre oracle, the margin rule and the qudit operators (PLAN.md Section 5.1.1).

Conventions:

- x = x0 (a + a^dag) with x0 = sqrt(hbar/(2 m omega)); a running-wave drive carries exp[i eta (a + a^dag)] = D(i eta), the
  displacement D(alpha) = exp(alpha a^dag - alpha^* a) at alpha = i eta.
- <n'|D(alpha)|n> = sqrt(n!/n'!) e^{-|alpha|^2/2} alpha^{n'-n} L^{(n'-n)}_n(|alpha|^2) for n' >= n and
  sqrt(n'!/n!) e^{-|alpha|^2/2} (-alpha^*)^{n-n'} L^{(n-n')}_{n'}(|alpha|^2) for n' < n; at alpha = i eta both are
  (i eta)^{|n'-n|} times the real rest.
- The closed form is never evaluated as a product of its factors, which overflow and underflow together (inf * 0 = nan)
  once |alpha| or the indices grow: every analytic element comes from the Laguerre recurrence along its diagonal
  (``_laguerre_elements``), finite and exact to rounding wherever |alpha|^2 is a finite double.
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
from scipy.linalg.blas import dtbsv

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

_LN2: Final[float] = math.log(2.0)
_LN2_HI: Final[float] = 6.93147180369123816490e-01
_LN2_LO: Final[float] = 1.90821492927058770002e-10
"""ln 2 = _LN2_HI + _LN2_LO with 21 trailing zero bits in _LN2_HI (fdlibm's split), so q _LN2_HI is exact for |q| < 2^21."""
_HEADROOM_BITS: Final[float] = 1000.0
"""Octaves a running product (``_coherent_amplitudes``) or a diagonal (``_laguerre_elements``) may grow between two
renormalizations: a double's exponent reaches 1023."""
_EXPONENT_FLOOR: Final[float] = -(2.0**20)
"""The binary exponent the recurrence's exponents, unbounded below, are clamped to for ``np.ldexp``: far below a
subnormal's (-1074), so an element scaled to it is 0.0."""


def _coherent_amplitudes(count: int, alpha_abs: float) -> tuple[np.ndarray, np.ndarray]:
    """(m_k, e_k) with <k|alpha> = |alpha|^k e^{-|alpha|^2/2}/sqrt(k!) = m_k 2^{e_k}, m_k in [1/2, 1), for k < count at
    |alpha| > 0 (e_k an integral float, unbounded below): the running product <k|alpha> = <k - 1|alpha> |alpha|/sqrt(k)
    from e^{-|alpha|^2/2}, renormalized by a power of two between blocks too short for their factors to carry it out of
    range."""
    # e^{-|alpha|^2/2} = e^r 2^e with 0 <= r < ln 2, r formed to its own rounding while |e| < 2^21 (|alpha| < 1700); r
    # is clamped because past |alpha|^2 ~ 1e18 it is noise, and there every element within reach is 0.0 anyway
    half = -0.5 * alpha_abs * alpha_abs
    e = float(math.floor(half / _LN2))
    r = min(max((half - e * _LN2_HI) - e * _LN2_LO, 0.0), _LN2)
    # the factors e^r (times 2^e) and |alpha|/sqrt(k), overwritten block by block by their running product
    run = np.empty(count)
    run[0] = math.exp(r)
    run[1:] = alpha_abs / np.sqrt(np.arange(1.0, count))
    exponent = np.empty(count)
    # a block of factors of at most `worst` octaves each keeps a product started in [1/2, 2) inside a double's range
    log_alpha = math.log2(alpha_abs)
    worst = max(1.0, abs(log_alpha), abs(log_alpha - 0.5 * math.log2(max(count - 1, 1))))
    block = max(1, int(_HEADROOM_BITS // worst))
    carry = 1.0
    for start in range(0, count, block):
        stop = min(count, start + block)
        run[start] *= carry
        np.cumprod(run[start:stop], out=run[start:stop])
        exponent[start:stop] = e
        carry, shift = math.frexp(run[stop - 1])
        e += shift
    mantissa, octaves = np.frexp(run)
    return mantissa, exponent + octaves


def _laguerre_elements(rows: int, ks: np.ndarray, alpha_abs: float) -> np.ndarray:
    """T[n, j] = sqrt(n!/(n + k)!) |alpha|^k e^{-|alpha|^2/2} L^{(k)}_n(|alpha|^2) at k = ks[j] for n < rows: the element
    <n + k|D(|alpha|)|n>, which is <n + k|D(alpha)|n> up to its phase (Section 5.1.1), at |alpha| >= 0.

    Along each diagonal k, T_n = <k|alpha> P_n (``_coherent_amplitudes``), and P_n = T_n/T_0 runs forward from P_0 = 1,
    Q_0 = 0 in the difference form of the Laguerre recurrence (DLMF 18.9.13), the form ``scipy.special.eval_genlaguerre``
    runs on L/C(n + k, n), here carried with its increment Q_n:
        Q_{n+1} = (n Q_n - |alpha|^2 P_n)/sqrt((n + 1)(n + k + 1)),  P_{n+1} = sqrt((n + k + 1)/(n + 1)) P_n + Q_{n+1}.
    The three-term form loses about n^2 eps near |alpha| = 0, where its two solutions merge; carrying the increment does
    not. Forward is the stable direction: a diagonal starts outside the classically allowed band
    |sqrt(n + k) - sqrt(n)| <= |alpha| <= sqrt(n + k) + sqrt(n), where the element is the solution growing toward the band
    (the dominant one, so an error in the other decays relative to it), enters the band, where both solutions oscillate
    under one envelope, and never leaves it. All diagonals are one unit lower triangular banded system in the unknowns
    (Q_n, P_n), solved by forward substitution (BLAS dtbsv). The values span more than a double's range (e^{-|alpha|^2/2}
    underflows past |alpha| = 38.6; L^{(k)}_n reaches e^{|alpha|^2/2} and, through its binomial, C(n + k, n), past 1e308 at
    n + k = 1030), so each diagonal carries a binary exponent, and the system is solved in stretches of rows too short for
    the growth bound 1 + sqrt(k + 1) + |alpha|^2 per row to carry a pair started below 1 to overflow, renormalized by a
    power of two between them. A power-of-two scaling is exact: the elements are the unscaled recurrence's to the last bit,
    and those below the subnormal range are 0.0."""
    x = alpha_abs * alpha_abs
    if not math.isfinite(x):
        raise ValueError(f"|alpha| = {alpha_abs} has no finite |alpha|^2")
    width = ks.size
    if alpha_abs == 0.0:
        identity = np.zeros((rows, width))
        identity[:, ks == 0] = 1.0  # D(0) is the identity
        return identity
    out = np.empty((rows, width))
    k_max = int(ks.max())
    mantissa, exponent = _coherent_amplitudes(k_max + 1, alpha_abs)
    # a pair renormalized below 1 grows by less than 3 + sqrt(k + 1) + |alpha|^2 per row: `period` rows stay below 2^1000
    period = max(1, int(_HEADROOM_BITS // math.log2(3.0 + math.sqrt(k_max + 1.0) + x)) - 1)
    k1 = ks + 1.0
    carry = np.zeros((width, 2))  # (Q_n0, P_n0) of each diagonal, in units of 2^shift
    carry[:, 1] = mantissa[ks]
    shift = exponent[ks]
    n0 = 0
    while True:
        n1 = min(rows - 1, n0 + period)
        n = np.arange(float(n0), float(n1))[:, None]
        root = np.sqrt((n + 1.0) * (n + k1))
        # per diagonal the unknowns Q_n0, P_n0, ..., Q_n1, P_n1, each coupled to the two before it
        band = np.zeros((width, 2 * (n1 - n0 + 1), 3))
        band[:, 1:-1:2, 1] = (x / root).T  # P_n in Q_{n+1}
        band[:, 2::2, 1] = -1.0  # Q_{n+1} in P_{n+1}
        band[:, 0:-2:2, 2] = (-n / root).T  # Q_n in Q_{n+1}
        band[:, 1:-2:2, 2] = (root / (-1.0 - n)).T  # P_n in P_{n+1}
        rhs = np.zeros((width, band.shape[1]))
        rhs[:, :2] = carry
        solved = dtbsv(2, band.reshape(-1, 3).T, rhs.reshape(-1), lower=1, diag=1, overwrite_x=1)
        z = solved.reshape(width, -1)
        scale = np.maximum(shift, _EXPONENT_FLOOR).astype(np.intc)
        out[n0 : n1 + 1] = np.ldexp(z[:, 1::2], scale[:, None]).T
        if n1 == rows - 1:
            return out
        _, e = np.frexp(np.max(np.abs(z[:, -2:]), axis=1))
        carry = np.ldexp(z[:, -2:], -e[:, None])
        shift = shift + e
        n0 = n1


@functools.lru_cache(maxsize=4096)
def _diagonal(k: int, alpha_abs: float, length: int) -> np.ndarray:
    """T_n^{(k)} of ``_laguerre_elements`` for n < ``length``, read-only: the memo ``_real_element`` reads."""
    out = _laguerre_elements(length, np.array([k]), alpha_abs)[:, 0]
    out.flags.writeable = False
    return out


def _real_element(n_row: int, n_col: int, alpha_abs: float) -> float:
    """<n_>|D(|alpha|)|n_<>, the modulus of <n_row|D(alpha)|n_col> up to the Laguerre sign, read from its diagonal walked to
    the next power of two past it: a run along one diagonal (a sideband's Rabi frequencies, a thermal sum of Debye-Waller
    factors) costs one recurrence, not one per element."""
    if n_row < 0 or n_col < 0:
        raise ValueError("Fock indices are non-negative")
    lo = int(min(n_row, n_col))
    return float(_diagonal(abs(int(n_row) - int(n_col)), float(alpha_abs), max(16, 1 << lo.bit_length()))[lo])


def _unit_power(u: complex, k: int) -> complex:
    """u^k by repeated squaring: exact at u = +-1, +-i (a real or imaginary displacement) for every k."""
    out = 1.0 + 0.0j
    while k:
        if k & 1:
            out *= u
        u *= u
        k >>= 1
    return out


def displacement_element_analytic(n_row: int, n_col: int, alpha: complex) -> complex:
    """<n_row|D(alpha)|n_col>, the exact infinite-space element (Section 5.1.1): (alpha/|alpha|)^k below the diagonal and
    (-alpha^*/|alpha|)^k above it (k = |n_row - n_col|) times the element of D(|alpha|) from its diagonal's recurrence
    (``_laguerre_elements``)."""
    a = complex(alpha)
    r = abs(a)
    element = _real_element(n_row, n_col, r)
    u = a / r if r > 0.0 else 1.0 + 0.0j
    return _unit_power(u if n_row >= n_col else -u.conjugate(), abs(int(n_row) - int(n_col))) * element


def _fold(t: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """(t[min(m, n), |m - n|], m - n) over rows m < ``rows`` and columns n < ``cols``: the table of ``_laguerre_elements``
    (diagonal k in column k) laid out as a matrix, the element below the diagonal on both sides of it."""
    m = np.arange(rows)[:, None]
    n = np.arange(cols)
    offset = m - n
    return t[np.minimum(m, n), np.abs(offset)], offset


def displacement_matrix_analytic(d: int, alpha: complex) -> np.ndarray:
    """The d x d matrix of exact elements <n'|D(alpha)|n> (rows n', columns n), ``displacement_element_analytic``'s phases
    on the elements of all d diagonals from one recurrence (``_laguerre_elements``); not unitary once truncated."""
    if d < 1:
        raise ValueError("d must be positive")
    a = complex(alpha)
    r = abs(a)
    moduli, offset = _fold(_laguerre_elements(d, np.arange(d), r), d, d)
    u = a / r if r > 0.0 else 1.0 + 0.0j
    # phase[m - n + d - 1] = u^{m-n} below the diagonal and (-u^*)^{n-m} above it, by repeated multiplication
    phase = np.empty(2 * d - 1, dtype=complex)
    phase[: d - 1] = -u.conjugate()
    phase[d - 1] = 1.0
    phase[d:] = u
    np.cumprod(phase[d - 1 :], out=phase[d - 1 :])
    np.cumprod(phase[d - 2 :: -1], out=phase[d - 2 :: -1])
    out: np.ndarray = moduli * phase[offset + (d - 1)]
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
    ``margin`` levels above it (analytic elements, the ``terms`` diagonals past the margin in one recurrence)."""
    if n_hi < 0 or margin < 0:
        raise ValueError("n_hi and margin are non-negative")
    if terms < 1:
        raise ValueError("terms is positive")
    t = _laguerre_elements(n_hi + 1, np.arange(margin + 1, margin + 1 + terms), abs(float(eta)))[n_hi]
    return float(np.sum(t * t))


# ---- Rabi table, Debye-Waller factors and populations (Section 4.3.1) -------------------------------------------------------


def rabi_matrix_element(n_row: int, n_col: int, eta: float) -> float:
    """Omega_{n', n}/Omega = |<n'|D(i eta)|n>| = e^{-eta^2/2} sqrt(n_<!/n_>!) |eta|^{|n'-n|} |L^{(|n'-n|)}_{n_<}(eta^2)|
    (Wineland 1998 Eq. 18), from the Section 5.1.1 recurrence (``_laguerre_elements``)."""
    return abs(_real_element(n_row, n_col, abs(float(eta))))


def rabi_table(d: int, eta: float) -> np.ndarray:
    """|<n'|D(i eta)|n>| for n, n' < d from the analytic elements."""
    if d < 1:
        raise ValueError("d must be positive")
    return np.abs(_fold(_laguerre_elements(d, np.arange(d), abs(float(eta))), d, d)[0])


def debye_waller_factor(n: int, eta: float) -> float:
    """The carrier element <n|D(i eta)|n> = e^{-eta^2/2} L_n(eta^2), signed (Monroe 2020 Eq. 11), from the Section 5.1.1
    recurrence (``_laguerre_elements``)."""
    return _real_element(n, n, abs(float(eta)))


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
    elements), the Fock distribution at the far point of a spin-dependent force's loop. The sum runs over the thermal levels
    whose weight is not 0.0 (the weights fall monotonically and underflow past about 745/ln(1 + 1/nbar)), so a cold mode's
    columns are a few hundred however large d is."""
    if d < 1:
        raise ValueError("d must be positive")
    p_th = thermal_populations(nbar, d)
    p_th = p_th / p_th.sum()
    cols = int(np.count_nonzero(p_th))
    moduli = _fold(_laguerre_elements(cols, np.arange(d), abs(float(alpha_abs))), d, cols)[0]
    return np.asarray((moduli * moduli) @ p_th[:cols], dtype=float)


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
