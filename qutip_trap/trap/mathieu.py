"""The Mathieu equation of a Paul trap: exponents, the Floquet micromotion factor C0 and the coupled 3D system.

Convention (PLAN.md Section 13): d^2x/dxi^2 + [a - 2q cos 2xi] x = 0 with xi = omega_rf t/2 (Leibfried et al. 2003);
Wineland and Berkeland 1998 use +2q cos, a half-period shift of the rf phase origin. The characteristic exponent comes
from the monodromy matrix over one period, cos(pi beta) = tr M/2, stable iff |tr M| < 2. C0 is the e^{i nu t} Fourier
coefficient of the Floquet function under the Wronskian normalization sum_n c_n^2 (1 + n omega_rf/nu) = 1, so
C0 = 1 + 3q^2/16 + O(q^4) with no clock-dependent (1 + q/2)^-1 factor; the c_n solve the three-term recursion
[a - (beta + 2n)^2] c_n = q (c_{n-1} + c_{n+1}). Non-commuting static and rf curvatures are integrated as the 6 x 6
companion system u'' + [A - 2Q cos 2tau] u = 0 (House 2008 Eq. 7 with its lower-left sign corrected).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from qutip_trap.units import TWO_PI

_RTOL = 1e-12
_ATOL = 1e-14
_MARGINAL_TOL = 1e-10


class UnstableMathieuError(ValueError):
    """The (a, q) point lies outside the stability region (|tr M| >= 2)."""


@dataclass(frozen=True)
class Monodromy:
    """The monodromy matrix M of x'' + (a - 2q cos 2xi) x = 0 over one period pi."""

    a: float
    q: float
    matrix: np.ndarray

    @property
    def trace(self) -> float:
        return float(self.matrix[0, 0] + self.matrix[1, 1])

    @property
    def stable(self) -> bool:
        return abs(self.trace) < 2.0

    @property
    def beta(self) -> float:
        """cos(pi beta) = tr M/2 on the principal branch [0, 1]; raises outside the stability region.

        A marginal axis (|tr M| = 2 to round-off, e.g. a = q = 0) returns 0 or 1, so an unconfined axis reports beta = 0
        and the crystal builder, not this solver, refuses it.
        """
        if abs(self.trace) > 2.0 + _MARGINAL_TOL:
            raise UnstableMathieuError(
                f"(a, q) = ({self.a}, {self.q}) is unstable: |tr M| = {abs(self.trace):.6g} > 2"
            )
        return math.acos(max(-1.0, min(1.0, self.trace / 2.0))) / math.pi


def monodromy(a: float, q: float) -> Monodromy:
    """Integrate the two fundamental solutions over xi in [0, pi]: M = [[x1, x2], [x1', x2']](pi)."""

    def rhs(xi: float, y: np.ndarray) -> list[float]:
        return [y[1], -(a - 2.0 * q * math.cos(2.0 * xi)) * y[0]]

    cols = []
    for y0 in ([1.0, 0.0], [0.0, 1.0]):
        sol = solve_ivp(rhs, (0.0, math.pi), y0, method="DOP853", rtol=_RTOL, atol=_ATOL)
        if not sol.success:
            raise RuntimeError(f"monodromy integration failed: {sol.message}")
        cols.append(sol.y[:, -1])
    return Monodromy(a=float(a), q=float(q), matrix=np.column_stack(cols))


def beta_exact(a: float, q: float) -> float:
    """The characteristic exponent by the monodromy method; raises when (a, q) is unstable."""
    return monodromy(a, q).beta


def is_stable(a: float, q: float) -> bool:
    """|tr M| < 2 for the monodromy matrix of (a, q)."""
    return monodromy(a, q).stable


@dataclass(frozen=True)
class FloquetCoefficients:
    """u(t) = sum_n c_n e^{i(nu + n omega_rf) t}, Wronskian-normalized, c_0 > 0; ``n`` are the harmonic indices."""

    a: float
    q: float
    beta: float
    n: np.ndarray
    c: np.ndarray

    @property
    def c0(self) -> float:
        """C0, the micromotion factor on eta."""
        return float(self.c[int(np.flatnonzero(self.n == 0)[0])])

    def ratio(self, n: int) -> float:
        """c_n / c_0."""
        return float(self.c[int(np.flatnonzero(self.n == n)[0])]) / self.c0


def floquet_coefficients(a: float, q: float, *, n_max: int = 24) -> FloquetCoefficients:
    """The c_n for |n| <= n_max as the null vector of the tridiagonal recursion at the monodromy beta.

    The null vector is the right singular vector of the smallest singular value; the truncation is checked by that
    singular value and by the edge coefficients.
    """
    beta = beta_exact(a, q)
    n = np.arange(-n_max, n_max + 1)
    diag = a - (beta + 2.0 * n) ** 2
    t = np.diag(diag) - q * (np.eye(len(n), k=1) + np.eye(len(n), k=-1))
    _, s, vh = np.linalg.svd(t)
    c = vh[-1].copy()
    if s[-1] > 1e-8 * float(np.max(np.abs(diag))):
        raise RuntimeError(f"no Floquet null vector at beta = {beta}: smallest singular value {s[-1]:.3e}")
    if abs(c[0]) > 1e-10 or abs(c[-1]) > 1e-10:
        raise RuntimeError("Floquet harmonic truncation too small: edge coefficients are not negligible")
    wronskian = float(np.sum(c * c * (beta + 2.0 * n)))
    if wronskian <= 0.0:
        raise RuntimeError("the Floquet solution has a non-positive Wronskian; not the e^{+i beta xi} branch")
    c *= math.sqrt(beta / wronskian)
    if c[int(np.flatnonzero(n == 0)[0])] < 0.0:
        c = -c
    return FloquetCoefficients(a=float(a), q=float(q), beta=beta, n=n, c=c)


def c0_wronskian(a: float, q: float) -> float:
    """The exact micromotion factor C0; 1 exactly at q = 0."""
    if q == 0.0:
        return 1.0
    return floquet_coefficients(a, q).c0


def vector_betas(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The exponents beta_j (ascending) of u'' + [A - 2Q cos 2tau] u = 0 from the eigenvalues of U(pi).

    A and Q are symmetric (mass-weighted coordinates, Landa 2012); raises when a Floquet multiplier leaves the unit
    circle.
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if A.shape != Q.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and Q must be square matrices of the same shape")
    if not (np.allclose(A, A.T, atol=1e-12) and np.allclose(Q, Q.T, atol=1e-12)):
        raise ValueError("A and Q must be symmetric (Landa 2012 Eq. 1; mass-weighted coordinates)")
    n = A.shape[0]
    d = 2 * n

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        f = np.zeros((d, d))
        f[:n, n:] = np.eye(n)
        f[n:, :n] = -(A - 2.0 * Q * math.cos(2.0 * tau))
        return np.asarray(f @ y.reshape(d, d)).ravel()

    sol = solve_ivp(rhs, (0.0, math.pi), np.eye(d).ravel(), method="DOP853", rtol=1e-11, atol=1e-13)
    if not sol.success:
        raise RuntimeError(f"vector monodromy integration failed: {sol.message}")
    eigenvalues = np.linalg.eigvals(sol.y[:, -1].reshape(d, d))
    if not np.all(np.abs(np.abs(eigenvalues) - 1.0) < 1e-6):
        raise UnstableMathieuError(f"coupled Mathieu system unstable: |lambda| = {np.abs(eigenvalues)}")
    return np.asarray(np.sort(np.abs(np.angle(eigenvalues)) / math.pi)[::2])  # conjugate pairs share |arg|


@dataclass(frozen=True)
class MathieuParameters:
    """a, q, exponents, secular frequencies and C0 per principal axis.

    ``a`` and ``q`` are the 3 x 3 static and rf curvature matrices in the principal-axis frame of the pseudopotential
    (diagonal whenever the two Hessians commute); ``axes`` holds those directions as columns in the laboratory frame, so
    ``beta``, ``secular_hz`` and ``C0`` are indexed like its columns (x', y', z). ``mass_kg`` is the ion mass the
    parameters were computed for: a and q scale as 1/m at fixed voltages.
    """

    a: np.ndarray
    q: np.ndarray
    beta: tuple[float, float, float]
    secular_hz: tuple[float, float, float]
    C0: tuple[float, float, float]
    axes: np.ndarray | None = None
    omega_rf_hz: float | None = None
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        for name, m in (("a", self.a), ("q", self.q)):
            if np.asarray(m, dtype=float).shape != (3, 3):
                raise ValueError(f"MathieuParameters.{name} must be a 3 x 3 matrix, got {np.shape(m)}")
        if any(not 0.0 <= b <= 1.0 for b in self.beta):
            raise ValueError(f"beta must lie on the principal branch [0, 1], got {self.beta}")

    @property
    def diagonal_a(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in np.diag(self.a))  # type: ignore[return-value]

    @property
    def principal_axes(self) -> np.ndarray:
        return np.eye(3) if self.axes is None else np.asarray(self.axes, dtype=float)

    @property
    def q_effective(self) -> tuple[float, float, float]:
        """sqrt((Q^2)_ii): the rf modulation strength along each principal axis (|q_ii| when Q is diagonal)."""
        q2 = np.asarray(self.q, dtype=float) @ np.asarray(self.q, dtype=float)
        return tuple(math.sqrt(max(float(q2[i, i]), 0.0)) for i in range(3))  # type: ignore[return-value]

    def pseudopotential_spring(self) -> np.ndarray:
        """(Omega/2)^2 (a + q^2/2): the period-averaged spring matrix per unit mass in the principal frame, rad^2/s^2."""
        if self.omega_rf_hz is None:
            raise ValueError("the pseudopotential spring needs the rf frequency")
        a = np.asarray(self.a, dtype=float)
        q = np.asarray(self.q, dtype=float)
        return np.asarray((TWO_PI * self.omega_rf_hz / 2.0) ** 2 * (a + q @ q / 2.0))


def _off_diagonal(m: np.ndarray) -> float:
    return float(np.max(np.abs(m - np.diag(np.diag(m)))))


def _principal_frame(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Eigenvectors of the pseudopotential Hessian A + Q^2/2 as columns, the axis closest to z last, right-handed."""
    commute = float(np.max(np.abs(A @ Q - Q @ A))) <= 1e-12 * max(
        float(np.max(np.abs(A))) * float(np.max(np.abs(Q))), 1e-300
    )
    # commuting matrices share eigenvectors, so any non-zero multiple of Q breaks the degeneracies of A + Q^2/2
    # (a_x = a_y with q_y = -q_x makes it isotropic while Q is not); only the eigenvectors are read here
    p = A + (0.3712 * Q if commute else 0.0) + Q @ Q / 2.0
    _, vecs = np.linalg.eigh((p + p.T) / 2.0)
    iz = int(np.argmax(np.abs(vecs[2, :])))
    rest = sorted((i for i in range(3) if i != iz), key=lambda i: -abs(vecs[0, i]))
    axes = np.column_stack([vecs[:, rest[0]], vecs[:, rest[1]], vecs[:, iz]])
    for j in range(3):  # sign gauge: positive projection on the lab axis each column is closest to
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
    """Exponents, secular frequencies and C0 from the static and rf curvatures (triples or 3 x 3 laboratory matrices).

    Diagonal systems use the scalar monodromy per axis (``axes`` names their directions). Non-commuting matrices are
    rotated into the principal frame of A + Q^2/2, the exponents come from the 6 x 6 system matched to the axes by
    sqrt(P_ii), and C0 per axis is evaluated at (a_ii, sqrt((Q^2)_ii)): exact when A and Q commute, a pseudopotential-frame
    approximation otherwise.
    """
    if omega_rf_hz <= 0.0:
        raise ValueError("omega_rf_hz must be positive (an ordinary frequency)")
    a_arr = np.asarray(a, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if a_arr.shape == (3,):
        a_arr = np.diag(a_arr)
    if q_arr.shape == (3,):
        q_arr = np.diag(q_arr)
    if a_arr.shape != (3, 3) or q_arr.shape != (3, 3):
        raise ValueError("a and q must be triples or 3 x 3 matrices")
    scale = max(float(np.max(np.abs(a_arr))), float(np.max(np.abs(q_arr))), 1e-300)
    if _off_diagonal(a_arr) <= 1e-12 * scale and _off_diagonal(q_arr) <= 1e-12 * scale:
        frame = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
        a_p, q_p = a_arr, q_arr
        betas = [monodromy(a_p[i, i], q_p[i, i]).beta for i in range(3)]
    else:
        if axes is not None:
            raise ValueError("axes are derived from the matrices when a and q are not diagonal")
        frame = _principal_frame(a_arr, q_arr)
        a_p = frame.T @ a_arr @ frame
        q_p = frame.T @ q_arr @ frame
        exact = vector_betas(a_arr, q_arr)
        p_frame = a_p + q_p @ q_p / 2.0
        guess = np.array([math.sqrt(max(float(p_frame[i, i]), 0.0)) for i in range(3)])
        remaining = list(range(3))  # match each axis to the nearest exact exponent, without reuse
        betas = [0.0, 0.0, 0.0]
        for i in np.argsort(guess):
            j = min(remaining, key=lambda k: abs(exact[k] - guess[i]))
            remaining.remove(j)
            betas[int(i)] = float(exact[j])
    # the rf strength along a principal axis is sqrt((Q^2)_ii): |q_ii| when Q is diagonal in this frame, and the isotropic
    # q of a two-dimensional rf null when the dc axes are rotated against the rf Hessian
    q2 = q_p @ q_p
    q_eff = [math.sqrt(max(float(q2[i, i]), 0.0)) for i in range(3)]
    c0 = tuple(c0_wronskian(a_p[i, i], q_eff[i]) for i in range(3))
    omega_rf = TWO_PI * omega_rf_hz
    secular = tuple(b * omega_rf / 2.0 / TWO_PI for b in betas)
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
    """(a_x, a_y, a_z) and (q_x, -q_x, 0) that reproduce measured secular frequencies (x, y, z) with the exact exponents.

    The linear-trap structure is imposed (q_z = 0, q_y = -q_x with q_x >= 0, and Laplace a_x + a_y + a_z = 0), so beta_z
    fixes a_z = beta_z^2 and the two radial exponents fix (a_x, q_x) by a root search; equal radials give a_x = a_y.
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
        try:
            return [monodromy(a_x, q_x).beta - bx, monodromy(-a_z - a_x, q_x).beta - by]
        except UnstableMathieuError:
            return [1.0, 1.0]

    sol = root(residual, np.array([ax_guess, q_guess]), method="hybr", tol=1e-13)
    if not sol.success or max(abs(r) for r in residual(sol.x)) > 1e-9:
        raise ValueError(
            f"no linear-trap Mathieu parameters reproduce {omega_hz} Hz at Omega/2pi = {omega_rf_hz} Hz"
        )
    a_x, q_x = float(sol.x[0]), abs(float(sol.x[1]))
    return (a_x, -a_z - a_x, a_z), (q_x, -q_x, 0.0)
