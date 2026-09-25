"""Composite pulses (PLAN.md Sections 4.3.5, 6.9; Appendix E, Run 5 additions; milestone M2).

Conventions (Section 13, rows "Static control-error primitive", "Composite-pulse sequence order in time", "Gate fidelity
measure", "Composite-pulse order", "Composite-pulse durations"): R(theta, phi) = exp[-i theta rho(phi).sigma/2] with
rho(phi) = (cos phi, sin phi, 0), so a 2 pi segment is exactly -I and a carrier pi pulse is Omega t = pi; the static
error primitive is M(theta, phi; eps_a, eps_d) = exp[-i theta{(1 + eps_a) rho(phi).sigma + eps_d sigma_z}/2], exact for
a square segment under time-independent errors, with eps_a scaling the rotation ANGLE and eps_d = Delta/Omega whose
+eps_d sigma_z term does NOT reverse with the drive phase; the addressing channel has the INVERTED zero (eps_N = 0
ideal, target I, M = R(theta eps_N, phi)); a sequence is stored as (area, phase) in TIME order and folded right to left
with the target azimuth ADDED to every entry including the zeroth pulse; a negative nominal area is emitted as a
positive area at phase + pi; F_K = |Tr(U_id^dag U)|^2/4 internally with F_C = sqrt(F_K) and F_avg = (2 F_K + 1)/3 as
conversions; order n means a propagator residual O(eps^{n+1}), a distance slope n + 1 and an infidelity slope
2(n + 1); durations tau = sum_l theta_l/Omega (SK1, BB1 4 pi + theta; CORPSE 4 pi + theta - 4k; CinSK, CinBB
8 pi + theta - 4k). The library and every printed defect it repairs are those of Section 4.3.5, verified against
``validation/scripts/check_composite.py`` (mpmath); the PD6 phases are Mount et al. 2015 Table I, indexed by the target
ANGLE theta_t (the table's phi_t is a typo), with the target pulse first in time.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import binom

M2 = "milestone M2 (control/composite.py, PLAN.md Section 4.3.5)"

Family = Literal[
    "primitive",
    "SK1",
    "SKn",
    "BB1",
    "NB1",
    "PB1",
    "P2j",
    "N2j",
    "B2j",
    "CORPSE",
    "short_CORPSE",
    "SCROFULOUS",
    "B2CORPSE",
    "CinSK",
    "CinBB",
    "PDn",
    "APn",
    "ToPn",
    "BBn",
]
# typing.Literal exposes its members as __args__ at runtime; mypy does not model that attribute on a Literal alias
FAMILIES: tuple[str, ...] = Family.__args__  # type: ignore[attr-defined]
Channel = Literal["amplitude", "pulse_length", "addressing", "detuning"]

PI = math.pi
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
SUZUKI_PHASE_TOLERANCE = 1e-9
"""How far the root-found ladder phase may sit from the closed form arccos(-theta/(2 pi f_j)) before it is a defect."""

# Mount et al. 2015 Table I: PD6 phases phi_{PD6:1..6} for the target ANGLE theta_t (printed as phi_t, a typo)
MOUNT_PD6_PHASES: dict[float, tuple[float, ...]] = {
    PI: (0.38266, -2.51430, -1.75192, 0.05941, 2.67572, 0.39344),
    PI / 2.0: (0.34769, -3.06979, 1.55852, -0.70890, 3.09692, -0.62174),
}


# ---- the primitive, folding and the fidelity measures --------------------------------------------------------------------


def primitive(theta: float, phi: float, eps_a: float = 0.0, eps_d: float = 0.0) -> np.ndarray:
    """M(theta, phi; eps_a, eps_d) = exp[-i theta{(1 + eps_a) rho(phi).sigma + eps_d sigma_z}/2] (exact)."""
    nx = (1.0 + eps_a) * math.cos(phi)
    ny = (1.0 + eps_a) * math.sin(phi)
    nz = eps_d
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n == 0.0:
        return _I2.copy()
    axis = (nx * _SX + ny * _SY + nz * _SZ) / n
    return math.cos(theta * n / 2.0) * _I2 - 1j * math.sin(theta * n / 2.0) * axis


def rotation(theta: float, phi: float) -> np.ndarray:
    """R(theta, phi) = exp[-i theta sigma_phi/2] (Mount 2015 Eq. 1)."""
    return primitive(theta, phi)


def fold(segments: Sequence[tuple[float, float]], eps_a: float = 0.0, eps_d: float = 0.0) -> np.ndarray:
    """U = M_last ... M_first for TIME-ordered (area, phase) segments."""
    u = _I2.copy()
    for theta, phi in segments:
        u = primitive(theta, phi, eps_a, eps_d) @ u
    return u


def fold_addressing(segments: Sequence[tuple[float, float]], eps_n: float) -> np.ndarray:
    """The addressing model: an unaddressed spin sees R(theta eps_N, phi) per segment; its target is the identity."""
    u = _I2.copy()
    for theta, phi in segments:
        u = primitive(theta * eps_n, phi) @ u
    return u


def fidelity_c(u: np.ndarray, v: np.ndarray) -> float:
    """F_C = |Tr(U^dag V)|/2 (Cummins; Merrill-Brown's minimized-state form is the same number)."""
    return float(abs(np.trace(u.conj().T @ v))) / 2.0


def fidelity_k(u: np.ndarray, v: np.ndarray) -> float:
    """F_K = |Tr(U^dag V)|^2/4 = F_C^2, the simulator's internal measure (Kabytayev 2014)."""
    return fidelity_c(u, v) ** 2


def fidelity_avg(u: np.ndarray, v: np.ndarray) -> float:
    """F_avg = (2 F_C^2 + 1)/3."""
    return (2.0 * fidelity_k(u, v) + 1.0) / 3.0


def operator_distance(u: np.ndarray, v: np.ndarray) -> float:
    """The spectral-norm distance ||U - V||_2, global phase INCLUDED: the operator-identity measure (1e-12 rule)."""
    return float(np.linalg.norm(u - v, 2))


def operator_distance_up_to_phase(u: np.ndarray, v: np.ndarray) -> float:
    """min_g ||U - e^{ig} V||_2: the same distance with the global phase quotiented out.

    A winding folds a composite propagator to -R(theta, phi) - short-CORPSE's n = (0, 1, 0) is the one family in the
    Section 4.3.5 library that does - and a global phase is not an error: F_K, F_C and F_avg all quotient it, so the
    distance that stands in for them below the float64 infidelity floor must quotient it too. Writing
    ||U - e^{ig}V||_2 = ||V^dag U - e^{ig} 1||_2 = max_k |e^{i a_k} - e^{ig}| over the eigenphases a_k of V^dag U, the
    minimizer is the bisector of the two phases, g = arg Tr(V^dag U) (2x2); Tr = 0 is the antipodal case, where every
    g gives sqrt(2) and the unquotiented value is already correct.
    """
    m = v.conj().T @ u
    tr = complex(np.trace(m))
    g = cmath.phase(tr) if abs(tr) > 1e-300 else 0.0
    return float(np.linalg.norm(u - cmath.exp(1j * g) * v, 2))


def infidelity_from_distance(distance: float) -> float:
    """1 - F_K ~ E^2 at small spectral-norm distance E between unitaries.

    U_err = exp(-i v.sigma) has E = 2 sin(|v|/2) ~ |v|, 1 - F_C = 1 - cos|v| ~ E^2/2 and 1 - F_K = 1 - cos^2|v| ~ E^2: the
    plan's "converted as E^2/2" (Sections 4.3.5, 13) is the F_C form; the internal measure F_K takes E^2 (M2 finding,
    pinned by the primitive detuning coefficient 1 - F_K = eps_d^2 at theta = pi).
    """
    return distance**2


# ---- the verified library (Section 4.3.5 table), TIME order, target azimuth added to every entry -----------------------


SINC_FIRST_MINIMUM = -0.217233628211222
"""min_x sin(x)/x = sinc(4.493409), the floor of the SECOND branch: the least value arcsinc could ever invert."""


def arcsinc(y: float) -> float:
    """Inverse of sin(x)/x on the branch (0, pi], where sin(x)/x falls monotonically from 1 to 0; arcsinc(0) = pi.

    The branch fixes the domain at [0, 1]. A NEGATIVE y has its root on the second branch, x in (pi, 4.493409], which
    this module does not implement: SCROFULOUS calls arcsinc(2 cos(theta/2)/pi), so y < 0 means theta > pi, and the
    previous check admitted y down to the sinc minimum -0.2172 and then handed brentq a bracket with no sign change,
    which surfaced as "f(a) and f(b) must have different signs" instead of a domain error (M2 audit E19).
    """
    if y == 0.0:
        return PI
    if y >= 1.0:
        return 0.0 if y == 1.0 else _out_of_branch(y)
    if y < 0.0:
        raise ValueError(
            f"arcsinc({y:.6g}): a negative argument's root lies on the second branch x in (pi, 4.493409], where "
            f"sin(x)/x reaches {SINC_FIRST_MINIMUM:.9g}; only the branch (0, pi] is implemented (SCROFULOUS at "
            "theta_t > pi lands here)"
        )

    def f(x: float) -> float:
        return math.sin(x) / x - y

    return float(brentq(f, 1e-12, PI))


def _out_of_branch(y: float) -> float:
    raise ValueError(f"arcsinc({y:.6g}): sin(x)/x never exceeds 1, so there is no root")


def _acos_checked(x: float, what: str) -> float:
    if not -1.0 <= x <= 1.0:
        raise ValueError(
            f"{what}: arccos argument {x:.6g} outside [-1, 1]; theta is outside the family's domain"
        )
    return math.acos(x)


def phi_sk1(theta: float) -> float:
    """phi_1 = arccos(-theta/(4 pi)), shared by SK1, BB1 and NB1 (the MINUS is load-bearing)."""
    return _acos_checked(-theta / (4.0 * PI), "SK1/BB1 phase")


def seq_primitive(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    return [(theta, phi_t)]


def seq_sk1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """SK1 corrected: U_1X(-theta) M_0(theta): (theta, 0), (2 pi, +phi_1), (2 pi, -phi_1) (BHC Eq. 10 prints U_1X(+theta))."""
    p = phi_sk1(theta)
    return [(theta, phi_t), (2.0 * PI, phi_t + p), (2.0 * PI, phi_t - p)]


def seq_sk1_printed(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """NEGATIVE CONTROL: BHC Eq. 10 as printed, U_1X(+theta): first order, exactly twice the bare-pulse error."""
    p = _acos_checked(theta / (4.0 * PI), "printed SK1")
    return [(theta, phi_t), (2.0 * PI, phi_t - p), (2.0 * PI, phi_t + p)]


def seq_bb1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """BB1 = B2 = W1: (theta, 0), (pi, phi_1), (2 pi, 3 phi_1), (pi, phi_1) (Wimperis; Merrill-Brown Eq. 43; Mount Eqs. 1-2)."""
    p = phi_sk1(theta)
    return [(theta, phi_t), (PI, phi_t + p), (2.0 * PI, phi_t + 3.0 * p), (PI, phi_t + p)]


def seq_nb1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """NB1 = N2: B2 with the middle phase -phi_1 instead of 3 phi_1 (the commonest transcription bug)."""
    p = phi_sk1(theta)
    return [(theta, phi_t), (PI, phi_t + p), (2.0 * PI, phi_t - p), (PI, phi_t + p)]


def seq_pb1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """PB1 = P2 = PD2: FIVE pulses, phi_P2 = arccos(-theta/(8 pi)) (the three-pulse closure does not apply)."""
    p = _acos_checked(-theta / (8.0 * PI), "PB1 phase")
    return [
        (theta, phi_t),
        (2.0 * PI, phi_t + p),
        (2.0 * PI, phi_t - p),
        (2.0 * PI, phi_t - p),
        (2.0 * PI, phi_t + p),
    ]


def seq_pb1_three_pulse_control(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """NEGATIVE CONTROL: P2's phase forced into a three-pulse corrector (3700x worse than P2, 16x worse than nothing)."""
    p = _acos_checked(-theta / (8.0 * PI), "PB1 phase")
    return [(theta, phi_t), (2.0 * PI, phi_t + p), (2.0 * PI, phi_t - p), (2.0 * PI, phi_t + p)]


def corpse_k(theta: float) -> float:
    """k = arcsin(sin(theta/2)/2)."""
    return math.asin(math.sin(theta / 2.0) / 2.0)


def corpse_angles(theta: float, n: tuple[int, int, int] = (1, 1, 0)) -> tuple[float, float, float]:
    """(2 n_1 pi + theta/2 - k, 2 n_2 pi - 2k, 2 n_3 pi + theta/2 - k); n = (1, 1, 0) is CORPSE, (0, 1, 0) short-CORPSE."""
    # n_1 - n_2 + n_3 = 0 returns +R(theta, 0); other windings such as short-CORPSE (0, 1, 0) return -R, a global phase
    k = corpse_k(theta)
    return (2.0 * n[0] * PI + theta / 2.0 - k, 2.0 * n[1] * PI - 2.0 * k, 2.0 * n[2] * PI + theta / 2.0 - k)


def seq_corpse(
    theta: float, phi_t: float = 0.0, n: tuple[int, int, int] = (1, 1, 0)
) -> list[tuple[float, float]]:
    """CORPSE: (2 pi + theta/2 - k, 0), (2 pi - 2k, pi), (theta/2 - k, 0) (420/300/60 degrees at theta = pi)."""
    t1, t2, t3 = corpse_angles(theta, n)
    return [(t1, phi_t), (t2, phi_t + PI), (t3, phi_t)]


def seq_short_corpse(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    return seq_corpse(theta, phi_t, n=(0, 1, 0))


def seq_scrofulous(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """SCROFULOUS: theta_1 = arcsinc(2 cos(theta/2)/pi), phi_S = arccos(-pi cos theta_1/(2 theta_1 sin(theta/2))),
    middle phase phi_S - arccos(-pi/(2 theta_1)) (the MINUS is load-bearing: 180_60 180_300 180_60 at theta = pi)."""
    t1 = arcsinc(2.0 * math.cos(theta / 2.0) / PI)
    p1 = _acos_checked(-PI * math.cos(t1) / (2.0 * t1 * math.sin(theta / 2.0)), "SCROFULOUS phi_S")
    p2 = p1 - _acos_checked(-PI / (2.0 * t1), "SCROFULOUS phi_2")
    return [(t1, phi_t + p1), (PI, phi_t + p2), (t1, phi_t + p1)]


def seq_cinsk(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """Reduced CinSK (Kabytayev Table I): the CORPSE block, then (2 pi, -phi_1), (2 pi, +phi_1)."""
    t1, t2, t3 = corpse_angles(theta)
    p = phi_sk1(theta)
    return [(t1, phi_t), (t2, phi_t + PI), (t3, phi_t), (2.0 * PI, phi_t - p), (2.0 * PI, phi_t + p)]


def seq_cinbb(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """Reduced CinBB (Kabytayev Table I): the CORPSE block, then (pi, phi_1), (2 pi, 3 phi_1), (pi, phi_1)."""
    t1, t2, t3 = corpse_angles(theta)
    p = phi_sk1(theta)
    return [
        (t1, phi_t),
        (t2, phi_t + PI),
        (t3, phi_t),
        (PI, phi_t + p),
        (2.0 * PI, phi_t + 3.0 * p),
        (PI, phi_t + p),
    ]


# ---- the palindromic Trotter-Suzuki ladder P2j / N2j / B2j (Merrill-Brown Eqs. 44-50) ------------------------------------


def _seg(area: float, phase: float) -> tuple[float, float]:
    """Emit a signed area as a POSITIVE area with a phase shift of pi where needed (Section 13)."""
    return (abs(area), phase if area >= 0 else phase + PI)


def _t1(k: float, phi: float) -> list[tuple[float, float]]:
    a = 2.0 * k * PI
    return [_seg(a, -phi), _seg(a, phi)][::-1]


def _t2(k: float, phi: float, family: str) -> list[tuple[float, float]]:
    if family == "P":
        return _t1(k, phi) + _t1(k, -phi)
    if family == "N":
        return _t1(k / 2.0, -phi) + _t1(k / 2.0, phi)
    if family == "B":
        if abs(k) % 2 == 0:
            return _t1(k / 2.0, phi) + _t1(k / 2.0, -phi)
        a = k * PI  # odd k: the CORRECTED four-pulse palindromic layer (the printed one has three pulses)
        return [_seg(a, phi), _seg(a, 3.0 * phi), _seg(a, 3.0 * phi), _seg(a, phi)]
    raise ValueError(family)


def _t2j(j: int, k: float, phi: float, family: str) -> list[tuple[float, float]]:
    if j == 1:
        return _t2(k, phi, family)
    inner = _t2j(j - 1, k, phi, family)
    out: list[tuple[float, float]] = (
        inner * 2 ** (2 * j - 2) + _t2j(j - 1, -2.0 * k, phi, family) + inner * 2 ** (2 * j - 2)
    )
    return out


def suzuki_factor(j: int, f1: float, *, printed: bool = False) -> float:
    """f_j = (2^{2j-1} - 2) f_{j-1}: 4, 24, 720, 90720 (P) and 2, 12, 360, 45360 (N, B).

    ``printed=True`` returns the source's defective recursion (2^{2j-1} - 1) f_{j-1}, i.e. f_2 = 28 (P) and 14 (N, B),
    which breaks first order: the 9.15 negative control, now constructible from the package rather than only from
    ``check_composite.py`` (M2 audit E11).
    """
    f = float(f1)
    for m in range(2, j + 1):
        f *= 2 ** (2 * m - 1) - (1 if printed else 2)
    return f


def _first_order_coefficient(segments: Sequence[tuple[float, float]], family: str) -> float:
    """The leading eps coefficient the ladder phase must null, as ONE signed scalar in the phase.

    P and B correct the AMPLITUDE channel, whose first-order generator is the toggling-frame sum sum_l theta_l
    rho~^(l) (``dc_polygon``); N corrects the ADDRESSING channel, where every segment is itself O(eps_N) so there is no
    toggling and the generator is the bare sum sum_l theta_l rho(phi_l). Both lie along the target axis for the
    palindromic ladder, so the x component is a scalar with a sign change at the root (measured).
    """
    total = np.zeros(3)
    if family == "N":
        for theta, phi in segments:
            total += theta * _axis3(phi)
        return float(total[0])
    u = _I2.copy()
    for theta, phi in segments:
        lam = _adjoint(u)
        total += theta * (lam.T @ _axis3(phi))
        u = primitive(theta, phi) @ u
    return float(total[0])


def _axis3(phi: float) -> np.ndarray:
    return np.array([math.cos(phi), math.sin(phi), 0.0])


def suzuki_phase(j: int, theta: float, family: str, *, printed: bool = False) -> tuple[float, float]:
    """(phi, |closed form - root|): the ladder phase ROOT-FOUND on the leading eps coefficient.

    PLAN.md:493 requires this by name - "the module root-finds every Trotter-Suzuki phase on the leading eps
    coefficient rather than hard-coding f_j" - because the root-find is what catches a transcription defect instead of
    trusting it. The closed form arccos(-theta/(2 pi f_j)) is the initial guess and the bracket is a small interval
    around it; the returned residual is asserted against zero by the caller, so a wrong f_j shows up as a large
    residual rather than as a silently wrong phase. ``printed=True`` uses the defective f_j as the guess, and then the
    residual is LARGE, which is the point of the negative control.
    """
    if j < 1:
        raise ValueError("j >= 1")
    f1 = 4.0 if family == "P" else 2.0
    fj = suzuki_factor(j, f1, printed=printed)
    guess = _acos_checked(-theta / (2.0 * PI * fj), f"{family}{2 * j} phase")

    def objective(phi: float) -> float:
        return _first_order_coefficient([(theta, 0.0)] + _t2j(j, 1.0, phi, family), family)

    lo, hi = guess - 0.2, guess + 0.2
    if objective(lo) * objective(hi) >= 0.0:
        raise ValueError(
            f"{family}{2 * j}: the leading eps coefficient does not change sign around the closed-form phase "
            f"{guess:.9f}; the recursion f_j = {fj:g} is not the one that nulls it"
        )
    root = float(brentq(objective, lo, hi, xtol=1e-14, rtol=8.9e-16))
    return root, abs(root - guess)


def seq_ladder(j: int, theta: float, family: str, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """P2j (family P), N2j (N) or B2j (B): the target then T_2j(1, phi) with phi ROOT-FOUND (``suzuki_phase``).

    The root is used, not the closed form, and the two are asserted to agree to ``SUZUKI_PHASE_TOLERANCE``: agreement
    is what certifies the transcribed recursion, and a disagreement is a defect report rather than a silent phase.
    """
    root, residual = suzuki_phase(j, theta, family)
    if residual > SUZUKI_PHASE_TOLERANCE:
        raise ValueError(
            f"{family}{2 * j} at theta = {theta:.9g}: the root-found phase {root:.12f} disagrees with the closed form "
            f"arccos(-theta/(2 pi f_j)) by {residual:.3g}, above {SUZUKI_PHASE_TOLERANCE:g} (PLAN.md:493)"
        )
    return [(theta, phi_t)] + [(a, phi_t + p) for a, p in _t2j(j, 1.0, root, family)]


# ---- PDn / APn (Low-Yoder-Chuang 2014) and Mount's PD6 ---------------------------------------------------------------------


def certificate_phi(phases: Sequence[float], j: int) -> complex:
    """Phi_L^j(phi) = sum_{h_1 < ... < h_j} exp(-i sum_k (-1)^k phi_{h_k}), the alternating sign over the POSITION k
    in the subset (LYC Eq. 11); filled by the dynamic-programming recursion, never by enumerating subsets."""
    # DP over the phases: state (count, parity of next position); parity k starts at 1 -> sign (-1)^1 = -1
    L = len(phases)
    # dp[c] = sum over subsets of size c of exp(-i sum_k (-1)^k phi_{h_k}) with positions counted 1..c
    dp = np.zeros(L + 1, dtype=complex)
    dp[0] = 1.0
    for phi in phases:
        new = dp.copy()
        for c in range(L, 0, -1):
            sign = -1.0 if c % 2 == 1 else 1.0
            new[c] += dp[c - 1] * np.exp(-1j * sign * phi)
        dp = new
    return complex(dp[j])


def certificate_target(L: int, j: int, gamma: float) -> float:
    """f_L^j(gamma) = sum_k (-1)^k C(T, k) C(L - T, j - k) with T = (gamma + L)/2 (a generalized binomial)."""
    t = (gamma + L) / 2.0
    return float(sum((-1) ** k * binom(t, k) * binom(L - t, j - k) for k in range(j + 1)))


def certificate(
    phases: Sequence[float], gamma: float, n_max: int, rtol: float = 1e-10
) -> dict[int, tuple[complex, bool]]:
    """{j: (Phi_L^j - f_L^j, |.| < rtol L)} for 0 < j <= n_max: order n iff true for j <= n and false at n + 1 (LYC Eq. 11)."""
    L = len(phases)
    out: dict[int, tuple[complex, bool]] = {}
    for j in range(1, n_max + 1):
        dev = certificate_phi(phases, j) - certificate_target(L, j, gamma)
        out[j] = (complex(dev), abs(dev) < rtol * L)
    return out


def toggled_phases(phases: Sequence[float]) -> list[float]:
    """psi_k = -sum_{h<k} (-1)^h phi_h + sum_{h>k} (-1)^h phi_h (0-based h with the sign (-1)^{h+1}): PD_n at theta_0 = 2 pi
    to the broadband BB_n at theta_0 = pi; toggled PD2 at gamma = 1 is Wimperis's BB1 at theta = pi (Section 4.3.5)."""
    L = len(phases)
    out: list[float] = []
    for k in range(L):
        v = -sum((-1) ** (h + 1) * phases[h] for h in range(k)) + sum(
            (-1) ** (h + 1) * phases[h] for h in range(k + 1, L)
        )
        out.append(float(v))
    return out


def pd2_phases(gamma: float) -> list[float]:
    """PD2 at theta_0 = 2 pi: (q, -q, -q, q) with cos q = -gamma/4 (this is PB1's phase, gamma = theta_T/2 pi)."""
    q = _acos_checked(-gamma / 4.0, "PD2 phase")
    return [q, -q, -q, q]


def ap1_phases(gamma: float) -> list[float]:
    """AP1 at theta_0 = 2 pi: (q, -q) with cos q = -gamma/2 (this is SK1)."""
    q = _acos_checked(-gamma / 2.0, "AP1 phase")
    return [q, -q]


def seq_pd6_mount(
    theta: float, phi_t: float = 0.0, *, target_first: bool = True
) -> list[tuple[float, float]]:
    """Mount 2015 Eq. 3 with Table I: the target then twelve pi pulses at phi_t + phi_{1..6}, phi_{6..1}."""
    key = None
    for k in MOUNT_PD6_PHASES:
        if math.isclose(theta, k, rel_tol=0.0, abs_tol=1e-12):
            key = k
    if key is None:
        raise ValueError(
            "Mount's PD6 phases are tabulated for theta_t = pi and pi/2 only; the general-theta phase vector needs the "
            "numerical continuation of Low-Yoder-Chuang (Section 4.3.5, not implemented)"
        )
    ph = MOUNT_PD6_PHASES[key]
    corr = [(PI, phi_t + p) for p in ph] + [(PI, phi_t + p) for p in reversed(ph)]
    return [(theta, phi_t)] + corr if target_first else corr + [(theta, phi_t)]


def seq_pd2(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """PDn at n = 2 through LYC's closed form (2 pi segments): identical to PB1."""
    gamma = theta / (2.0 * PI)
    return [(theta, phi_t)] + [(2.0 * PI, phi_t + p) for p in pd2_phases(gamma)]


def seq_bb_toggled(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """BBn at n = 2: the toggled PD2 phases as pi pulses, which at theta = pi is Wimperis's BB1 (phi_1, 3 phi_1, 3 phi_1, phi_1)."""
    if not math.isclose(theta, PI, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "the toggling map is derived at gamma = 1 (theta = pi); other angles are not implemented"
        )
    psi = toggled_phases(pd2_phases(1.0))
    return [(theta, phi_t)] + [(PI, phi_t + p) for p in reversed(psi)]


# ---- the public record ------------------------------------------------------------------------------------------------------


_ORDER: dict[str, int] = {
    "primitive": 0,
    "SK1": 1,
    "BB1": 2,
    "NB1": 2,
    "PB1": 2,
    "CORPSE": 1,
    "short_CORPSE": 1,
    "SCROFULOUS": 1,
    "CinSK": 1,
    "CinBB": 1,
}
_CORRECTS: dict[str, frozenset[Channel]] = {
    "primitive": frozenset(),
    "SK1": frozenset({"amplitude", "addressing"}),
    "BB1": frozenset({"amplitude"}),
    "NB1": frozenset({"addressing"}),
    "PB1": frozenset({"amplitude", "addressing"}),
    "P2j": frozenset({"amplitude"}),
    "N2j": frozenset({"addressing"}),
    "B2j": frozenset({"amplitude"}),
    "CORPSE": frozenset({"detuning"}),
    "short_CORPSE": frozenset({"detuning"}),
    "SCROFULOUS": frozenset({"amplitude"}),
    "CinSK": frozenset({"amplitude", "detuning"}),
    "CinBB": frozenset({"amplitude", "detuning"}),
    "PDn": frozenset({"amplitude"}),
    "APn": frozenset({"amplitude", "addressing"}),
    "BBn": frozenset({"amplitude"}),
}
_PROVENANCE: dict[str, str] = {
    "primitive": "conv.static_error_primitive",
    "SK1": "anchor.composite.sk1_sign",
    "BB1": "anchor.composite.bb1",
    "NB1": "anchor.composite.bb1",
    "PB1": "anchor.composite.pb1_five_pulses",
    "P2j": "anchor.composite.suzuki_factor",
    "N2j": "anchor.composite.suzuki_factor",
    "B2j": "anchor.composite.suzuki_factor",
    "CORPSE": "anchor.composite.corpse",
    "short_CORPSE": "anchor.composite.corpse",
    "SCROFULOUS": "anchor.composite.scrofulous",
    "CinSK": "anchor.composite.concatenated_durations",
    "CinBB": "anchor.composite.concatenated_durations",
    "PDn": "anchor.composite.mount_pd6",
    "APn": "anchor.composite.sk1_sign",
    "BBn": "anchor.composite.toggled_pd2",
}


@dataclass(frozen=True)
class CompositePulse:
    """A static-error-compensating single-qubit sequence (Section 4.3.5)."""

    family: Family
    theta_rad: float
    phi_rad: float
    """Target axis azimuth; ADDED to every segment phase."""
    order: int
    """n; residual O(eps^{n+1}), infidelity slope 2(n+1)."""
    corrects: frozenset[Channel]
    segments: tuple[tuple[float, float], ...]
    """(area_rad, phase_rad) in TIME order, areas > 0."""
    n_rep: int = 1
    """Repetitions of the corrector block, ALREADY folded into ``segments`` (Appendix E; see ``repeat_corrector``): a
    record of how the sequence was built, never a multiplier a consumer applies a second time."""
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown composite-pulse family {self.family!r}")
        if self.order < 0 or self.n_rep < 1:
            raise ValueError("order must be >= 0 and n_rep >= 1")
        if not self.segments:
            raise ValueError("a composite pulse has at least one segment")
        if any(area <= 0.0 for area, _ in self.segments):
            raise ValueError(
                "segment areas are positive; a negative nominal area is emitted at phase + pi (Section 13)"
            )

    # ---- derived ----------------------------------------------------------------------------------------------------

    def total_rotation_rad(self) -> float:
        """Sum of the segment areas; the duration at constant amplitude is this over Omega (Section 13)."""
        return float(sum(area for area, _ in self.segments))

    def duration_s(self, omega_rabi_rad_s: float) -> float:
        return self.total_rotation_rad() / omega_rabi_rad_s

    def target(self) -> np.ndarray:
        return rotation(self.theta_rad, self.phi_rad)

    def propagator(self, eps_a: float = 0.0, eps_d: float = 0.0, eps_N: float | None = None) -> np.ndarray:
        """2x2, folded right to left; ``eps_N`` selects the addressing model (target = identity)."""
        if eps_N is not None:
            return fold_addressing(self.segments, eps_N)
        return fold(self.segments, eps_a, eps_d)

    def infidelity(
        self, eps_a: float = 0.0, eps_d: float = 0.0, measure: str = "F_K", eps_N: float | None = None
    ) -> float:
        u = self.propagator(eps_a, eps_d, eps_N)
        v = _I2 if eps_N is not None else self.target()
        if measure == "F_K":
            return 1.0 - fidelity_k(v, u)
        if measure == "F_C":
            return 1.0 - fidelity_c(v, u)
        if measure == "F_avg":
            return 1.0 - fidelity_avg(v, u)
        raise ValueError("measure is F_K, F_C or F_avg")

    def distance(self, eps_a: float = 0.0, eps_d: float = 0.0) -> float:
        """min_g ||U(eps) - e^{ig} R(theta, phi)||_2: the error size with the global phase quotiented out.

        The quotient is load-bearing, not cosmetic: short-CORPSE's (0, 1, 0) winding returns -R exactly
        (``corpse_angles``), so the unquotiented norm is the constant 2.0 at every eps and the ``order_slope`` fallback
        below the float64 infidelity floor collapsed to 0 instead of the plan's 4.000 (M2 audit E3).
        """
        return operator_distance_up_to_phase(self.propagator(eps_a, eps_d), self.target())

    def order_slope(
        self,
        channel: Literal["amplitude", "detuning", "simultaneous"],
        eps_range: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """(log-log slope of 1 - F_K, fit residual) over an order-aware window eps >> 10^(-8/(n+1)) (Section 9.17).

        Below 1 - F ~ 1e-8 the subtracted infidelity hits the float64 floor, so the window defaults to (1e-2, 1e-1) and
        the fit uses the squared operator distance where the infidelity is under 1e-8.
        """
        if eps_range is None:
            eps_range = (1e-2, 1e-1)
        lo, hi = eps_range
        if not 0.0 < lo < hi:
            raise ValueError("eps_range must be increasing and positive")
        eps = np.geomspace(lo, hi, 9)
        vals = []
        for e in eps:
            ea = e if channel in ("amplitude", "simultaneous") else 0.0
            ed = e if channel in ("detuning", "simultaneous") else 0.0
            inf = self.infidelity(ea, ed)
            if inf < 1e-8:
                inf = infidelity_from_distance(self.distance(ea, ed))
            vals.append(inf)
        x = np.log(eps)
        y = np.log(np.asarray(vals))
        slope, intercept = np.polyfit(x, y, 1)
        resid = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
        return float(slope), resid

    def dc_polygon(self, rtol: float = 1e-10) -> tuple[np.ndarray, bool]:
        """sum_l theta_l rho~^(l) in the toggling frame, shape (3,), and whether it closes (|.| < rtol sum theta_l).

        The first-order amplitude-error generator is the toggling-frame sum of the segment axes weighted by their areas;
        closure to zero is first-order amplitude compensation (Section 6.9, the dc limit of the amplitude filter function).
        """
        u = _I2.copy()
        total = np.zeros(3)
        for theta, phi in self.segments:
            axis = np.array([math.cos(phi), math.sin(phi), 0.0])
            # rotate the axis into the toggling frame of the propagator accumulated before this segment
            lam = _adjoint(u)
            total += theta * (lam.T @ axis)
            u = primitive(theta, phi) @ u
        return total, bool(np.linalg.norm(total) < rtol * self.total_rotation_rad())

    def certificate(self, rtol: float = 1e-10) -> dict[int, tuple[complex, bool]]:
        """{j: (Phi_L^j - f_L^j(gamma), |.| < rtol L)} for the equal-area corrector phases (LYC Eq. 11); raises for
        non-uniform corrector areas (SCROFULOUS) or a corrector area other than 2 pi."""
        corrector = self.segments[1:]
        if not corrector:
            raise ValueError("the primitive has no corrector")
        areas = {round(a, 12) for a, _ in corrector}
        if len(areas) != 1:
            raise ValueError("the certificate is defined for equal-area correctors only")
        theta0 = corrector[0][0]
        if not math.isclose(theta0, 2.0 * PI, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "the certificate of Section 4.3.5 is stated for 2 pi correctors (LYC theta_0 = 2 pi)"
            )
        phases = [p - self.phi_rad for _, p in corrector]
        gamma = self.theta_rad / theta0
        return certificate(phases, gamma, len(corrector), rtol)

    def filter_function_amplitude(self, omega_rad_s: np.ndarray, omega_rabi_rad_s: float) -> np.ndarray:
        """F_a(omega) = (1/4){|sum_l A_l rho~^(l)|^2 + |sum_l B_l rho~^(l)|^2} at constant Rabi frequency (Section 6.9).

        The exact A_l/B_l segment sum, no quadrature: this pulse's segments timed at ``omega_rabi_rad_s`` and handed to
        ``noise.decoupling.amplitude_filter_function``. Imported inside the call because ``noise.decoupling`` imports
        this module (the control layer must not import the noise layer at module scope).
        """
        from qutip_trap.noise.decoupling import amplitude_filter_function, composite_segments

        return amplitude_filter_function(
            composite_segments(self, omega_rabi_rad_s), np.asarray(omega_rad_s, dtype=float)
        )

    def dc_floor(self, moments: Mapping[str, float], omega_rabi_rad_s: float) -> float:
        """sum_channels c-hat_{m+1} (2m + 1)!! (<beta^2>/Omega^2)^{m+1}: the frozen-noise floor (Section 6.9).

        ``moments`` maps a channel ("amplitude" and/or "detuning") to its <beta^2> in (rad/s)^2; ``m`` is this pulse's
        order FOR THAT CHANNEL, so a family that does not correct the channel contributes at m = 0. c-hat_{m+1} is the
        leading coefficient of 1 - F_K in eps^{2(m+1)}, fitted numerically (``fitted_leading_coefficient``) and cached
        per (family, theta, order, channel); the fit reproduces the Section 4.3.5 values to better than 1e-5 relative
        (measured: SK1 22.8302557111 to 4.1e-10, BB1 9.388566343 to 1.5e-7, CORPSE 0.006500751892 to 1.6e-7).
        """
        from qutip_trap.noise.decoupling import dc_floor as _dc_floor

        if omega_rabi_rad_s <= 0.0:
            raise ValueError("the Rabi frequency must be positive")
        total = 0.0
        for channel, variance in moments.items():
            if channel not in ("amplitude", "detuning"):
                raise ValueError(
                    f"the dc floor is defined for the amplitude and detuning channels; got {channel!r}"
                )
            if variance < 0.0:
                raise ValueError("<beta^2> is a variance")
            m = self.order if channel in self.corrects else 0
            c_hat = fitted_leading_coefficient(self, channel, 2 * (m + 1))
            total += _dc_floor(c_hat, m, variance, omega_rabi_rad_s)
        return total


def _adjoint(u: np.ndarray) -> np.ndarray:
    """Lambda_ij = (1/2) Tr[U^dag sigma_i U sigma_j], the SO(3) matrix of U^dag (Section 13, toggling-frame row)."""
    s = (_SX, _SY, _SZ)
    lam = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            lam[i, j] = 0.5 * np.real(np.trace(u.conj().T @ s[i] @ u @ s[j]))
    return lam


_BUILDERS: dict[str, Callable[[float, float], list[tuple[float, float]]]] = {
    "primitive": seq_primitive,
    "SK1": seq_sk1,
    "BB1": seq_bb1,
    "NB1": seq_nb1,
    "PB1": seq_pb1,
    "CORPSE": seq_corpse,
    "short_CORPSE": seq_short_corpse,
    "SCROFULOUS": seq_scrofulous,
    "CinSK": seq_cinsk,
    "CinBB": seq_cinbb,
}


def suzuki_block_power(j: int) -> int:
    """2^{2j-2}: how many times the cached T_{2j-2} block appears in EACH outer run of T_{2j} (Merrill-Brown Eq. 47).

    This is Appendix E's ``n_rep``, "P2j/N2j/B2j: repetitions of the cached T_{2j-2} block" (PLAN.md:2800), and it is
    1 at j = 1, which is the field's declared default. It is a RECORD of how ``segments`` was built - the block is
    already expanded there - and never a multiplier a consumer applies again. The two alternative readings are both
    refuted: repeating the whole pulse contradicts ``target()``/``propagator()``, which fold the target once, and
    repeating the corrector alone destroys the compensation (measured: P2 at theta = pi/2 falls from an amplitude
    infidelity slope of 6.000 to 1.999 at n_rep = 2, because the corrector cancels the TARGET pulse's error and a
    second copy has none to cancel) - M2 audit E12.
    """
    if j < 1:
        raise ValueError("j >= 1")
    return int(2 ** (2 * j - 2))


def _check_n_rep(family: str, n_rep: int, derived: int) -> int:
    """``n_rep`` is DERIVED from the family and order (``suzuki_block_power``); an explicit value only asserts it."""
    if n_rep < 1:
        raise ValueError("n_rep >= 1")
    if n_rep not in (1, derived):
        if family in ("P2j", "N2j", "B2j"):
            what = f"{family}'s Trotter-Suzuki block power 2^(2j-2) = {derived} at this order"
        else:
            what = f"1, the only value for {family} (Appendix E scopes n_rep to P2j/N2j/B2j)"
        raise ValueError(
            f"n_rep = {n_rep} is not {what}: n_rep counts repetitions of the CACHED T_{{2j-2}} block, which "
            "``segments`` already carries, and is not a free repetition count (neither repeating the whole pulse, "
            "which would multiply the target angle, nor repeating the corrector, which drops P2's amplitude "
            "infidelity slope from 6.000 to 1.999, is order preserving)"
        )
    return derived


def composite_pulse(
    family: str, theta_rad: float, phi_rad: float = 0.0, *, order: int = 1, n_rep: int = 1
) -> CompositePulse:
    """Phases from the verified closed forms of Section 4.3.5; raises outside the arccos domain, never NaN.

    ``order`` selects j for P2j/N2j/B2j (order 2j) and n for PDn/APn/BBn (PDn: n = 2 closed form, n = 6 Mount's table
    at theta_t = pi or pi/2; APn: n = 1 = SK1; BBn: n = 2 at theta = pi, the toggled PD2).
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown composite-pulse family {family!r}; known: {FAMILIES}")
    if theta_rad <= 0.0:
        raise ValueError(
            "theta_rad is a positive target angle; a negative angle is the same area at phase + pi"
        )
    # ``family`` was validated against FAMILIES above, which IS the Literal's member tuple, so the narrowing is sound;
    # mypy cannot derive it from a runtime membership test against a tuple
    fam: Family = family  # type: ignore[assignment]
    if family in _BUILDERS:
        _check_n_rep(family, n_rep, 1)
        segs = _BUILDERS[family](theta_rad, phi_rad)
        return CompositePulse(
            fam,
            theta_rad,
            phi_rad,
            _ORDER[family],
            _CORRECTS[family],
            tuple(segs),
            n_rep,
            _PROVENANCE[family],
        )
    if family in ("P2j", "N2j", "B2j"):
        j = max(order, 1)
        n_rep = _check_n_rep(family, n_rep, suzuki_block_power(j))
        segs = seq_ladder(j, theta_rad, family[0], phi_rad)
        return CompositePulse(
            fam, theta_rad, phi_rad, 2 * j, _CORRECTS[family], tuple(segs), n_rep, _PROVENANCE[family]
        )
    if family == "PDn":
        _check_n_rep(family, n_rep, 1)
        if order == 2:
            segs = seq_pd2(theta_rad, phi_rad)
        elif order == 6:
            segs = seq_pd6_mount(theta_rad, phi_rad)
        else:
            raise NotImplementedError(
                "PDn is implemented at n = 2 (closed form) and n = 6 (Mount's table at pi and pi/2)"
            )
        return CompositePulse(
            fam, theta_rad, phi_rad, order, _CORRECTS[family], tuple(segs), n_rep, _PROVENANCE[family]
        )
    if family == "APn":
        if order != 1:
            raise NotImplementedError(
                "APn is implemented at n = 1 (SK1); AP2/AP3 closed forms were not transcribed (Section 12)"
            )
        _check_n_rep(family, n_rep, 1)
        segs = [(theta_rad, phi_rad)] + [(2.0 * PI, phi_rad + p) for p in ap1_phases(theta_rad / (2.0 * PI))]
        return CompositePulse(
            fam, theta_rad, phi_rad, 1, _CORRECTS[family], tuple(segs), n_rep, _PROVENANCE[family]
        )
    if family == "BBn":
        if order != 2:
            raise NotImplementedError("BBn is implemented at n = 2 (the toggled PD2 = BB1 at theta = pi)")
        _check_n_rep(family, n_rep, 1)
        segs = seq_bb_toggled(theta_rad, phi_rad)
        return CompositePulse(
            fam, theta_rad, phi_rad, 2, _CORRECTS[family], tuple(segs), n_rep, _PROVENANCE[family]
        )
    if family in ("SKn", "ToPn", "B2CORPSE"):
        raise NotImplementedError(
            f"{family} is not in the verified library of Section 4.3.5 (SKn needs the spectral-norm Suzuki recursion, ToPn the "
            "numerical continuation, B2CORPSE is Cummins' nesting); CinSK and CinBB are the verified concatenations"
        )
    raise NotImplementedError(f"{family} is {M2}")


_C_HAT_CACHE: dict[tuple[str, float, int, str, int], float] = {}
"""Fitted c-hat_{m+1} per (family, theta, order, channel, power): the fit is a few hundred 2x2 folds."""


def fitted_leading_coefficient(
    pulse: CompositePulse, channel: str, power: int, *, terms: int = 3, distance_floor: float = 1e-10
) -> float:
    """c-hat = lim_{eps -> 0} (1 - F_K)/eps^power, fitted rather than read off at one eps (Section 6.9's dc floor).

    ``leading_coefficient`` at a single eps carries two errors that pull in opposite directions: a truncation error in
    even powers of eps from the next term of the series, and a round-off error that grows as eps falls, because
    1 - F_K is a subtraction of numbers near 1. The fit takes the eps ladder whose operator distance is still above
    ``distance_floor`` (below which the squared-distance fallback of ``order_slope`` is itself round-off) and
    least-squares a polynomial c-hat + a eps^2 + b eps^4 in eps^2, returning the constant. Measured against the
    Section 4.3.5 values at theta = pi: SK1 4.1e-10, SCROFULOUS 9.2e-8, BB1 1.5e-7, CORPSE 1.6e-7, PB1 1.9e-7,
    CinSK 4.2e-7, CinBB 2.1e-6 relative.
    """
    key = (pulse.family, round(pulse.theta_rad, 12), pulse.order, channel, power)
    hit = _C_HAT_CACHE.get(key)
    if hit is not None:
        return hit
    eps_grid = np.geomspace(3e-2, 1e-7, 40)
    xs: list[float] = []
    ys: list[float] = []
    for e in eps_grid:
        c = leading_coefficient(pulse, channel, power, float(e))
        if math.sqrt(max(c, 0.0) * float(e) ** power) < distance_floor:
            continue  # the measure itself is round-off at this eps
        xs.append(float(e))
        ys.append(c)
    if not xs:
        raise ValueError(
            f"the {channel} infidelity of {pulse.family} never rises above the float64 floor on the eps ladder: "
            "c-hat cannot be fitted"
        )
    x = np.asarray(xs)
    y = np.asarray(ys)
    k = max(1, min(terms, x.size - 1))
    design = np.vstack([x ** (2 * i) for i in range(k)]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    out = float(coeffs[0])
    _C_HAT_CACHE[key] = out
    return out


def leading_coefficient(pulse: CompositePulse, channel: str, power: int, eps: float = 1e-3) -> float:
    """(1 - F_K)/eps^power at a small eps, the leading coefficient of Section 4.3.5 (float64: power <= 4 is reliable)."""
    ea = eps if channel == "amplitude" else 0.0
    ed = eps if channel == "detuning" else 0.0
    inf = pulse.infidelity(ea, ed)
    if inf < 1e-8:
        inf = infidelity_from_distance(pulse.distance(ea, ed))
    return inf / eps**power
