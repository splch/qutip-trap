"""Mathieu equation, Floquet exponent, Floquet function and the micromotion factor (PLAN.md Section 4.1.1; 13; M1).

Convention (Section 13, row "Mathieu equation"): d^2x/dxi^2 + [a - 2q cos 2xi] x = 0 with xi = omega_rf t/2,
a = 4 Z|e| U alpha/(m omega_rf^2), q = 2 Z|e| U~ alpha'/(m omega_rf^2) (Leibfried et al. 2003). Wineland 1998
and Berkeland 1998 use +2q cos, a half-period shift of the rf phase origin, which is why the sign is a
declared field of ``MathieuParameters`` and why the Floquet function reads u(t) ~ e^{i nu t}[1 - (q/2) cos
omega_rf t] with u_dot(0) = i nu (1 + q) in this origin (Section 13, "Floquet function and rf phase origin";
``check_critique_v3.py``: c_{+-1}/c_0 = -q/((2 +- beta)^2 - a), -0.023322 and -0.026875 at q = 0.1).

The characteristic exponent is computed by the MONODROMY method (Section 4.1.1): the two fundamental
solutions are integrated over one period (pi in xi), cos(pi beta) = tr M/2, stability iff |tr M| < 2, beta on
the principal branch [0, 1]; the printed continued fractions of RMP 2003 Eqs. 10-11 carry an index error and
are not used. The closed forms sqrt(a + q^2/2) and the preprint form [(a + q^2/2)/(1 - 3q^2/8)]^{1/2} are
CHECKS on the monodromy result, never the computation. The 3 x 3 coupled (vector) Mathieu system of a trap
whose static and rf curvature matrices do not commute (Section 4.1.6) is integrated as the 6 x 6 companion
system F(tau) = [[0, 1], [-(A - 2Q cos 2tau), 0]]; House 2008 Eq. 7 prints the lower-left block with the
wrong sign and reports a stable trap as unstable (Section 13, "Mathieu sign in House 2008"; Section 9.13).

The micromotion factor on the Lamb-Dicke parameter is C0 = c_0, the e^{i nu t} Fourier coefficient of the
exact Floquet function u(t) = sum_n c_n e^{i(nu + n omega_rf)t} under the canonical Wronskian normalization
Im(u* u_dot) = nu, i.e. sum_n c_n^2 (1 + n omega_rf/nu) = 1; C0 = 1 + 3q^2/16 + O(q^4), even in q, with no
first-order term (1.001890, 1.007741, 1.018161 at q = 0.1, 0.2, 0.3; ``check_c0_floquet.py``). The RMP's
(1 + q/2)^-1 is the u(0) = 1 normalization at one rf phase, an rf-clock artifact [corrected: derivation
audit, 2026-09-04]. C0 is applied once, in ``Crystal.lamb_dicke`` (Section 5.7), and nowhere else. The
Fourier coefficients c_n are obtained from the three-term recursion the Mathieu equation imposes,
[a - (beta + 2n)^2] c_n = q (c_{n-1} + c_{n+1}), solved as the null vector of the truncated tridiagonal
matrix at the monodromy beta (Section 4.1.1, "D_{2n} = [a - (2n + beta)^2]/q"), so they are exact to the
truncation and never depend on an rf clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from qutip_trap.units import TWO_PI

MATHIEU_SIGN_CONVENTION: Final = "a - 2q cos 2xi"
PERIOD_XI: Final[float] = math.pi
"""One rf period in the dimensionless time xi = omega_rf t/2."""

_RTOL: Final[float] = 1e-12
_ATOL: Final[float] = 1e-14
_MARGINAL_TOL: Final[float] = 1e-10


class UnstableMathieuError(ValueError):
    """The (a, q) point lies outside the stability region (|tr M| >= 2; Section 4.1.1)."""


# ---- closed-form checks (Section 13, "Characteristic exponent") -------------------------------------


def beta_lowest_order(a: float, q: float) -> float:
    """The lowest-order characteristic exponent beta ~ sqrt(a + q^2/2), a CHECK on the monodromy result (Section 13)."""
    radicand = a + q * q / 2.0
    if radicand <= 0.0:
        raise ValueError(f"a + q^2/2 = {radicand} <= 0: outside the lowest-order stable region")
    return math.sqrt(radicand)


def beta_preprint(a: float, q: float) -> float:
    """The preprint form beta = [(a + q^2/2)/(1 - 3q^2/8)]^{1/2} of Wineland 1998, dropped from the published paper.

    Section 4.1.1: 0.12% from the exact exponent at q = 0.3 against 1.8% for the bare lowest-order form.
    """
    num = a + q * q / 2.0
    den = 1.0 - 3.0 * q * q / 8.0
    if num <= 0.0 or den <= 0.0:
        raise ValueError("outside the region where the preprint form is real")
    return math.sqrt(num / den)


def c0_series(q: float) -> float:
    """C0 = 1 + 3 q^2/16, the O(q^2) micromotion factor on eta (even in q, no first-order term; Section 13)."""
    return 1.0 + 3.0 * q * q / 16.0


def secular_rad_s(beta: float, omega_rf_rad_s: float) -> float:
    """nu = beta omega_rf/2 (Section 4.1.1), angular."""
    return beta * omega_rf_rad_s / 2.0


def pseudopotential_radial_rad_s(q: float, omega_rf_rad_s: float) -> float:
    """omega_r = q Omega/(2 sqrt 2), the a = 0 pseudopotential frequency with q the MATHIEU parameter (Section 4.1.1).

    Equivalently Q V0/(sqrt 2 Omega m R^2) with Q the charge (Wineland 1998 Eq. 6, whose q is the charge); the two
    symbols are separated in code as ``q_mathieu`` and ``Q_charge`` (Section 13).
    """
    return abs(q) * omega_rf_rad_s / (2.0 * math.sqrt(2.0))


# ---- scalar monodromy --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Monodromy:
    """The monodromy matrix M of x'' + (a - 2q cos 2xi) x = 0 over one period pi (Section 4.1.1)."""

    a: float
    q: float
    matrix: np.ndarray

    @property
    def trace(self) -> float:
        return float(self.matrix[0, 0] + self.matrix[1, 1])

    @property
    def stable(self) -> bool:
        """|tr M| < 2 (strict); the boundary |tr M| = 2 is the marginal edge of a stability region."""
        return abs(self.trace) < 2.0

    @property
    def marginal(self) -> bool:
        """|tr M| = 2 to round-off: beta = 0 (a free coordinate, a = q = 0) or beta = 1 (the stability edge)."""
        return abs(abs(self.trace) - 2.0) <= _MARGINAL_TOL

    @property
    def beta(self) -> float:
        """cos(pi beta) = tr M/2 on the principal branch beta in [0, 1]; raises outside the stability region.

        A marginal axis (|tr M| = 2 to round-off, e.g. an axis the electrodes do not confine, a = q = 0) returns
        the boundary value 0 or 1 rather than raising, so that a strip-model trap with no axial curvature reports
        beta_z = 0 and the crystal builder, not the Mathieu solver, refuses the missing confinement.
        """
        if abs(self.trace) > 2.0 + _MARGINAL_TOL:
            raise UnstableMathieuError(
                f"(a, q) = ({self.a}, {self.q}) is unstable: |tr M| = {abs(self.trace):.6g} > 2"
            )
        return math.acos(max(-1.0, min(1.0, self.trace / 2.0))) / math.pi


def _mathieu_rhs(a: float, q: float) -> object:
    def rhs(xi: float, y: np.ndarray) -> list[float]:
        return [y[1], -(a - 2.0 * q * math.cos(2.0 * xi)) * y[0]]

    return rhs


def monodromy(a: float, q: float, *, rtol: float = _RTOL, atol: float = _ATOL) -> Monodromy:
    """Integrate the two fundamental solutions over xi in [0, pi] and assemble M = [[x1, x2], [x1', x2']](pi)."""
    rhs = _mathieu_rhs(a, q)
    cols = []
    for y0 in ([1.0, 0.0], [0.0, 1.0]):
        sol = solve_ivp(rhs, (0.0, PERIOD_XI), y0, method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"monodromy integration failed: {sol.message}")
        cols.append(sol.y[:, -1])
    m = np.column_stack(cols)
    return Monodromy(a=float(a), q=float(q), matrix=m)


def beta_exact(a: float, q: float) -> float:
    """The characteristic exponent by the monodromy method (Section 4.1.1); raises when (a, q) is unstable."""
    return monodromy(a, q).beta


def is_stable(a: float, q: float) -> bool:
    return monodromy(a, q).stable


def stability_edge_q(a: float = 0.0, *, q_lo: float = 0.5, q_hi: float = 1.5) -> float:
    """The q at which beta reaches 1 for the given a (tr M = -2): 0.908 at a = 0 (Section 9.1; Mathieu tables)."""

    def f(q: float) -> float:
        return monodromy(a, q).trace + 2.0

    return float(brentq(f, q_lo, q_hi, xtol=1e-12))


# ---- Floquet function and C0 --------------------------------------------------------------------------


@dataclass(frozen=True)
class FloquetCoefficients:
    """u(t) = sum_n c_n e^{i(nu + n omega_rf) t} under the Wronskian normalization sum_n c_n^2 (1 + n omega_rf/nu) = 1.

    ``n`` runs over the harmonic indices, ``c`` are the real coefficients with c_0 > 0; ``beta`` is the exponent.
    In xi = omega_rf t/2 the same function is sum_n c_n e^{i(beta + 2n) xi} and the normalization reads
    sum_n c_n^2 (beta + 2n) = beta, i.e. Im(u* du/dxi) = beta.
    """

    a: float
    q: float
    beta: float
    n: np.ndarray
    c: np.ndarray

    @property
    def c0(self) -> float:
        """C0, the micromotion factor on eta (Section 4.1.1; Section 13, "Micromotion correction")."""
        return float(self.c[int(np.flatnonzero(self.n == 0)[0])])

    def ratio(self, n: int) -> float:
        """c_n / c_0."""
        return float(self.c[int(np.flatnonzero(self.n == n)[0])]) / self.c0

    def sideband_weights(self) -> dict[int, float]:
        """|c_n/c_0|^2 per harmonic: the Floquet weights of the micromotion-sideband heating sum (Section 4.1.5).

        Lowest order |c_{+-1}/c_0|^2 = q^2/16, Turchette's omega_m^2/(2 Omega_T^2) at a = 0; the two forms are one
        number and must never be added (Section 4.1.5). Brownnutt's index 2j is this module's n = j.
        """
        return {int(k): float(v) for k, v in zip(self.n, (self.c / self.c0) ** 2)}

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """u(xi) = sum_n c_n e^{i(beta + 2n) xi}."""
        xi = np.asarray(xi, dtype=float)
        return np.asarray(
            np.exp(1j * self.beta * xi)
            * (self.c[None, :] * np.exp(2j * self.n[None, :] * xi[:, None])).sum(-1)
        )

    def comb_phase_per_rf_period_rad(self) -> float:
        """nu T_rf = beta pi: the secular phase u accumulates over ONE rf period, i.e. the phase step between successive
        teeth of the micromotion comb (Section 9.10, row "Micromotion carrier and comb").

        At a = 0, q = 0.1 the exact exponent gives 0.222580, against 0.222144 from the pseudopotential exponent
        sqrt(a + q^2/2) - the two differ in the fourth digit, which is the O(q^4) correction to beta. The RMP prints
        beta omega_rf T, exactly TWICE this, because nu = beta omega_rf/2.
        """
        return float(self.beta * math.pi)

    def lowest_order_ratios(self) -> tuple[float, float]:
        """The plan's closed forms c_{+1}/c_0 = -q/((2 + beta)^2 - a) and c_{-1}/c_0 = -q/((2 - beta)^2 - a)."""
        return (
            -self.q / ((2.0 + self.beta) ** 2 - self.a),
            -self.q / ((2.0 - self.beta) ** 2 - self.a),
        )


def floquet_coefficients(a: float, q: float, *, n_max: int = 24) -> FloquetCoefficients:
    """Fourier coefficients of the Floquet function from the three-term recursion at the monodromy beta.

    [a - (beta + 2n)^2] c_n - q (c_{n-1} + c_{n+1}) = 0 for |n| <= n_max: the null vector of the tridiagonal
    matrix, taken as the right singular vector of its smallest singular value, then normalized by the
    Wronskian. The truncation is checked by the size of that singular value and of the edge coefficients.
    """
    beta = beta_exact(a, q)
    n = np.arange(-n_max, n_max + 1)
    diag = a - (beta + 2.0 * n) ** 2
    t = np.diag(diag) - q * (np.eye(len(n), k=1) + np.eye(len(n), k=-1))
    _, s, vh = np.linalg.svd(t)
    c = vh[-1].copy()
    scale = float(np.max(np.abs(diag)))
    if s[-1] > 1e-8 * scale:
        raise RuntimeError(f"no Floquet null vector at beta = {beta}: smallest singular value {s[-1]:.3e}")
    if abs(c[0]) > 1e-10 or abs(c[-1]) > 1e-10:
        raise RuntimeError("Floquet harmonic truncation too small: edge coefficients are not negligible")
    wronskian = float(np.sum(c * c * (beta + 2.0 * n)))
    if wronskian <= 0.0:
        raise RuntimeError("the Floquet solution has a non-positive Wronskian; not the e^{+i beta xi} branch")
    c *= math.sqrt(beta / wronskian)
    i0 = int(np.flatnonzero(n == 0)[0])
    if c[i0] < 0.0:
        c = -c
    return FloquetCoefficients(a=float(a), q=float(q), beta=beta, n=n, c=c)


def c0_wronskian(a: float, q: float) -> float:
    """The exact micromotion factor C0 (Section 13, "Micromotion correction"); 1 exactly at q = 0."""
    if q == 0.0:
        return 1.0
    return floquet_coefficients(a, q).c0


def floquet_function_by_integration(
    a: float, q: float, *, n_samples: int = 4096
) -> tuple[np.ndarray, np.ndarray, float]:
    """The Floquet eigen-solution u(xi) over one period by direct integration, Wronskian-normalized.

    An independent construction of the same function ``floquet_coefficients`` builds from the recursion
    (the route ``check_c0_floquet.py`` takes); returns (xi, u, beta). Used by the tests to cross-check.
    """
    mono = monodromy(a, q)
    beta = mono.beta
    w, v = np.linalg.eig(mono.matrix)
    k = int(np.argmin(np.abs(w - np.exp(1j * math.pi * beta))))
    vec = v[:, k]
    rhs = _mathieu_rhs(a, q)
    xi = np.linspace(0.0, PERIOD_XI, n_samples, endpoint=False)
    sols = [
        solve_ivp(rhs, (0.0, PERIOD_XI), y0, method="DOP853", rtol=_RTOL, atol=_ATOL, dense_output=True)
        for y0 in ([1.0, 0.0], [0.0, 1.0])
    ]
    u = vec[0] * sols[0].sol(xi)[0] + vec[1] * sols[1].sol(xi)[0]
    du = vec[0] * sols[0].sol(xi)[1] + vec[1] * sols[1].sol(xi)[1]
    wr = np.imag(np.conj(u) * du)
    u = u * math.sqrt(beta / float(np.mean(wr)))
    return xi, np.asarray(u), beta


# ---- the coupled 3 x 3 (vector) Mathieu system ------------------------------------------------------------


def companion_matrix(
    A: np.ndarray, Q: np.ndarray, tau: float, *, lower_left_sign: float = -1.0
) -> np.ndarray:
    """F(tau) = [[0, 1], [-(A - 2Q cos 2tau), 0]] for u'' + [A - 2Q cos 2tau] u = 0 (Section 13, "Floquet branch").

    ``lower_left_sign`` = -1 is the equation of motion; +1 reproduces House 2008's printed Eq. 7, the negative
    control of Section 9.13 (a stable trap reported unstable, |lambda| = 1.3244 and 0.7550 for a_z = 0.008).
    """
    n = A.shape[0]
    f = np.zeros((2 * n, 2 * n))
    f[:n, n:] = np.eye(n)
    f[n:, :n] = lower_left_sign * (A - 2.0 * Q * math.cos(2.0 * tau))
    return f


@dataclass(frozen=True)
class VectorMonodromy:
    """U(pi) of the coupled system, its eigenvalues and the exponents beta_j on the principal branch."""

    A: np.ndarray
    Q: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray

    @property
    def stable(self) -> bool:
        """Every Floquet multiplier on the unit circle (to 1e-6), i.e. every beta_j real."""
        return bool(np.all(np.abs(np.abs(self.eigenvalues) - 1.0) < 1e-6))

    @property
    def betas(self) -> np.ndarray:
        """beta_j = |arg lambda_j|/pi, one per conjugate pair, ascending; raises if unstable."""
        if not self.stable:
            raise UnstableMathieuError(
                f"coupled Mathieu system unstable: |lambda| = {np.abs(self.eigenvalues)}"
            )
        b = np.sort(np.abs(np.angle(self.eigenvalues)) / math.pi)
        return np.asarray(b[::2])  # conjugate pairs give equal |arg|

    def modes(self) -> tuple[np.ndarray, np.ndarray]:
        """(betas, directions): the position-space direction of each Floquet mode at tau = 0 (unit vectors)."""
        if not self.stable:
            raise UnstableMathieuError("coupled Mathieu system unstable")
        w, v = np.linalg.eig(self.matrix)
        n = self.A.shape[0]
        order = np.argsort(np.angle(w))
        w, v = w[order], v[:, order]
        keep = np.angle(w) >= 0.0
        betas = np.angle(w[keep]) / math.pi
        dirs = np.real(v[:n, keep])
        norms = np.linalg.norm(dirs, axis=0)
        dirs = dirs / np.where(norms > 0, norms, 1.0)
        srt = np.argsort(betas)
        return np.asarray(betas[srt]), np.asarray(dirs[:, srt])


def vector_monodromy(
    A: np.ndarray, Q: np.ndarray, *, lower_left_sign: float = -1.0, rtol: float = 1e-11, atol: float = 1e-13
) -> VectorMonodromy:
    """Integrate Y' = F(tau) Y from Y(0) = 1 over tau in [0, pi] (House 2008 Eqs. 4-10 with the sign fixed)."""
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if A.shape != Q.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and Q must be square matrices of the same shape")
    if not (np.allclose(A, A.T, atol=1e-12) and np.allclose(Q, Q.T, atol=1e-12)):
        raise ValueError("A and Q must be symmetric (Landa 2012 Eq. 1; mass-weighted coordinates)")
    n = A.shape[0]
    d = 2 * n

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        y_mat = y.reshape(d, d)
        return np.asarray(companion_matrix(A, Q, tau, lower_left_sign=lower_left_sign) @ y_mat).ravel()

    sol = solve_ivp(rhs, (0.0, PERIOD_XI), np.eye(d).ravel(), method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"vector monodromy integration failed: {sol.message}")
    u = sol.y[:, -1].reshape(d, d)
    return VectorMonodromy(A=A, Q=Q, matrix=u, eigenvalues=np.linalg.eigvals(u))


# ---- MathieuParameters --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MathieuParameters:
    """a, q matrices, exponents, secular frequencies and C0 per principal axis (Appendix E; Section 4.1.1).

    ``a`` and ``q`` are the 3 x 3 static and rf curvature matrices in the PRINCIPAL-AXIS frame of the
    pseudopotential (diagonal whenever the static and rf Hessians commute; a rod trap and a mirror-symmetric
    surface trap); ``axes`` holds those principal directions as columns in the laboratory frame (x, y, z with z
    the trap axis), so that ``beta``, ``secular_hz`` and ``C0`` are indexed like the columns of ``axes`` and
    like ``Trap.omega_hz`` (x', y', z). ``omega_rf_hz`` is the drive frequency the exponents refer to.
    """

    a: np.ndarray
    q: np.ndarray
    beta: tuple[float, float, float]
    secular_hz: tuple[float, float, float]
    C0: tuple[float, float, float]
    """Wronskian-normalized rf-period Fourier coefficient per axis (Section 4.1.1)."""
    sign_convention: Literal["a - 2q cos 2xi"] = MATHIEU_SIGN_CONVENTION
    axes: np.ndarray | None = None
    """Principal directions as columns in the laboratory frame; None means the laboratory axes."""
    omega_rf_hz: float | None = None
    mass_kg: float | None = None
    """The ion mass the parameters were computed for; a and q scale as 1/m at fixed voltages (Section 4.1.7), which is how
    ``Crystal.lamb_dicke`` derives another species' C0 from this record."""

    def __post_init__(self) -> None:
        if self.sign_convention != MATHIEU_SIGN_CONVENTION:
            raise ValueError("the simulator's Mathieu sign convention is fixed (Section 13)")
        for name in ("a", "q"):
            m = np.asarray(getattr(self, name), dtype=float)
            if m.shape != (3, 3):
                raise ValueError(f"MathieuParameters.{name} must be a 3 x 3 matrix, got {m.shape}")
        if any(not 0.0 <= b <= 1.0 for b in self.beta):
            raise ValueError(f"beta must lie on the principal branch [0, 1], got {self.beta}")

    @property
    def diagonal_a(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in np.diag(self.a))  # type: ignore[return-value]

    @property
    def diagonal_q(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in np.diag(self.q))  # type: ignore[return-value]

    @property
    def principal_axes(self) -> np.ndarray:
        return np.eye(3) if self.axes is None else np.asarray(self.axes, dtype=float)

    @property
    def q_effective(self) -> tuple[float, float, float]:
        """sqrt((Q^2)_ii): the rf modulation strength along each principal axis (|q_ii| when Q is diagonal here)."""
        q2 = np.asarray(self.q, dtype=float) @ np.asarray(self.q, dtype=float)
        return tuple(math.sqrt(max(float(q2[i, i]), 0.0)) for i in range(3))  # type: ignore[return-value]

    def pseudopotential_spring(self) -> np.ndarray:
        """(Omega/2)^2 (a + q^2/2): the static (period-averaged) spring matrix per unit mass in the principal frame, rad^2/s^2."""
        if self.omega_rf_hz is None:
            raise ValueError("the pseudopotential spring needs the rf frequency")
        a = np.asarray(self.a, dtype=float)
        q = np.asarray(self.q, dtype=float)
        return np.asarray((TWO_PI * self.omega_rf_hz / 2.0) ** 2 * (a + q @ q / 2.0))

    def secular_rad_s(self) -> tuple[float, float, float]:
        return tuple(TWO_PI * f for f in self.secular_hz)  # type: ignore[return-value]


def _off_diagonal(m: np.ndarray) -> float:
    return float(np.max(np.abs(m - np.diag(np.diag(m)))))


_DEGENERACY_BREAKER = 0.3712
"""An arbitrary non-zero, non-round coefficient of Q in the commuting branch of ``_principal_frame`` (see there)."""


def _principal_frame(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Columns: eigenvectors of the pseudopotential Hessian A + Q^2/2 (the lowest-order secular matrix), ordered
    so that the axis closest to z comes last and the other two follow a right-handed (x', y', z) triad."""
    commute = float(np.max(np.abs(A @ Q - Q @ A))) <= 1e-12 * max(
        float(np.max(np.abs(A))) * float(np.max(np.abs(Q))), 1e-300
    )
    if commute:
        # commuting matrices share eigenvectors: a generic combination breaks the degeneracies of A + Q^2/2 (a_x = a_y with
        # q_y = -q_x makes the pseudopotential Hessian isotropic while Q itself is not). Because A and Q commute, the
        # eigenVECTORS of any A + c Q + Q^2/2 are the shared ones for every c, so the value of _DEGENERACY_BREAKER is
        # immaterial as long as it is non-zero and not a coincidental degeneracy of the particular A, Q: only the
        # eigenvectors are read here, never the eigenvalues (they come from the monodromy).
        p = A + _DEGENERACY_BREAKER * Q + Q @ Q / 2.0
    else:
        p = A + Q @ Q / 2.0
    _, vecs = np.linalg.eigh((p + p.T) / 2.0)
    # put the axis with the largest |z| component last, keep the others ordered by their x-component
    iz = int(np.argmax(np.abs(vecs[2, :])))
    rest = [i for i in range(3) if i != iz]
    rest.sort(key=lambda i: -abs(vecs[0, i]))
    cols = [vecs[:, rest[0]], vecs[:, rest[1]], vecs[:, iz]]
    axes = np.column_stack(cols)
    for j in range(3):  # sign gauge: positive projection on the lab axis it is closest to
        k = int(np.argmax(np.abs(axes[:, j])))
        if axes[k, j] < 0.0:
            axes[:, j] = -axes[:, j]
    if np.linalg.det(axes) < 0.0:
        axes[:, 1] = -axes[:, 1]
    return axes


def mathieu_parameters(
    a: np.ndarray | tuple[float, float, float],
    q: np.ndarray | tuple[float, float, float],
    omega_rf_hz: float,
    *,
    axes: np.ndarray | None = None,
) -> MathieuParameters:
    """Exponents, secular frequencies and C0 from the static and rf curvature matrices (Sections 4.1.1, 4.1.6).

    ``a`` and ``q`` are either per-axis triples (a diagonal system; ``axes`` then names the principal
    directions, default the laboratory axes) or 3 x 3 matrices in the laboratory frame. Diagonal systems use
    the scalar monodromy per axis. Non-commuting matrices are rotated into the principal frame of the
    pseudopotential Hessian A + Q^2/2 (stored rotated, off-diagonals included), the exact exponents come from the
    6 x 6 vector monodromy matched to the axes by sqrt(P_ii), and C0 per axis is evaluated from (a_ii, sqrt((Q^2)_ii)),
    the rf strength along that axis: exact when A and Q commute, a pseudopotential-frame approximation otherwise (the
    irreducibly time-dependent modes are Landa's Floquet-Lyapunov construction, a validation option not built in M1).
    """
    if omega_rf_hz <= 0.0:
        raise ValueError("omega_rf_hz must be positive (an ordinary frequency, Section 5.6)")
    a_arr = np.asarray(a, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if a_arr.shape == (3,):
        a_arr = np.diag(a_arr)
    if q_arr.shape == (3,):
        q_arr = np.diag(q_arr)
    if a_arr.shape != (3, 3) or q_arr.shape != (3, 3):
        raise ValueError("a and q must be triples or 3 x 3 matrices")
    scale = max(float(np.max(np.abs(a_arr))), float(np.max(np.abs(q_arr))), 1e-300)
    diagonal = _off_diagonal(a_arr) <= 1e-12 * scale and _off_diagonal(q_arr) <= 1e-12 * scale
    omega_rf = TWO_PI * omega_rf_hz
    if diagonal:
        frame = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
        a_p, q_p = a_arr, q_arr
        betas = [monodromy(a_p[i, i], q_p[i, i]).beta for i in range(3)]
    else:
        if axes is not None:
            raise ValueError("axes are derived from the matrices when a and q are not diagonal")
        frame = _principal_frame(a_arr, q_arr)
        a_p = frame.T @ a_arr @ frame
        q_p = frame.T @ q_arr @ frame
        exact = vector_monodromy(a_arr, q_arr).betas
        p_frame = a_p + q_p @ q_p / 2.0
        guess = np.array([math.sqrt(max(float(p_frame[i, i]), 0.0)) for i in range(3)])
        # match each axis to the nearest exact exponent, without reuse
        remaining = list(range(3))
        betas = [0.0, 0.0, 0.0]
        for i in np.argsort(guess):
            j = min(remaining, key=lambda k: abs(exact[k] - guess[i]))
            remaining.remove(j)
            betas[int(i)] = float(exact[j])
    # the rf strength along a principal axis is sqrt((Q^2)_ii): equal to |q_ii| when Q is diagonal in this frame, and the
    # isotropic q of a two-dimensional rf null when the dc axes are rotated against the rf Hessian (Section 4.1.6)
    q2 = q_p @ q_p
    q_eff = [math.sqrt(max(float(q2[i, i]), 0.0)) for i in range(3)]
    c0 = tuple(c0_wronskian(a_p[i, i], q_eff[i]) for i in range(3))
    secular = tuple(secular_rad_s(b, omega_rf) / TWO_PI for b in betas)
    return MathieuParameters(
        a=a_p,
        q=q_p,
        beta=(float(betas[0]), float(betas[1]), float(betas[2])),
        secular_hz=(float(secular[0]), float(secular[1]), float(secular[2])),
        C0=(float(c0[0]), float(c0[1]), float(c0[2])),
        axes=frame,
        omega_rf_hz=float(omega_rf_hz),
    )


def mathieu_from_secular(
    omega_hz: tuple[float, float, float], omega_rf_hz: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Invert the exact exponents: (a_x, a_y, a_z) and (q_x, -q_x, 0) from measured secular frequencies (x, y, z).

    The explicit-frequency path of ``Trap`` (Section 3.3). The linear-trap structure is imposed: q_z = 0,
    q_y = -q_x with q_x >= 0 (Berkeland 1998), and the static coefficients obey Laplace's equation,
    a_x + a_y + a_z = 0 (Section 4.1.1), so the axial exponent fixes a_z = beta_z^2 exactly and the two
    radial exponents fix (a_x, q_x) by a root search on the monodromy beta. A radial pair with omega_x = omega_y
    therefore returns a_x = a_y = -a_z/2 (beta is even in q; the radials split only through a_x != a_y).
    """
    wx, wy, wz = omega_hz
    if min(omega_hz) <= 0.0:
        raise ValueError("secular frequencies must be positive")
    bx, by, bz = (2.0 * w / omega_rf_hz for w in (wx, wy, wz))
    if max(bx, by, bz) >= 1.0:
        raise UnstableMathieuError(
            "a secular frequency of at least half the rf frequency is outside the first stability region"
        )
    a_z = bz * bz
    q_guess = math.sqrt(max(bx * bx + by * by + a_z, 1e-12))
    ax_guess = bx * bx - q_guess * q_guess / 2.0

    def residual(p: np.ndarray) -> list[float]:
        a_x, q_x = float(p[0]), abs(float(p[1]))
        a_y = -a_z - a_x
        try:
            rx = monodromy(a_x, q_x).beta - bx
            ry = monodromy(a_y, q_x).beta - by
        except UnstableMathieuError:
            return [1.0, 1.0]
        return [rx, ry]

    sol = root(residual, np.array([ax_guess, q_guess]), method="hybr", tol=1e-13)
    if not sol.success or max(abs(r) for r in residual(sol.x)) > 1e-9:
        raise ValueError(
            f"no linear-trap Mathieu parameters reproduce {omega_hz} Hz at Omega/2pi = {omega_rf_hz} Hz"
        )
    a_x, q_x = float(sol.x[0]), abs(float(sol.x[1]))
    return (a_x, -a_z - a_x, a_z), (q_x, -q_x, 0.0)


__all__ = [
    "MATHIEU_SIGN_CONVENTION",
    "PERIOD_XI",
    "FloquetCoefficients",
    "MathieuParameters",
    "Monodromy",
    "UnstableMathieuError",
    "VectorMonodromy",
    "beta_exact",
    "beta_lowest_order",
    "beta_preprint",
    "c0_series",
    "c0_wronskian",
    "companion_matrix",
    "floquet_coefficients",
    "floquet_function_by_integration",
    "is_stable",
    "mathieu_from_secular",
    "mathieu_parameters",
    "monodromy",
    "pseudopotential_radial_rad_s",
    "secular_rad_s",
    "stability_edge_q",
    "vector_monodromy",
]
