"""Composite pulses (PLAN.md Section 4.3.5).

R(theta, phi) = exp[-i theta rho(phi).sigma/2] with rho(phi) = (cos phi, sin phi, 0), so a 2 pi segment is exactly -I and a
carrier pi pulse is Omega t = pi; the static error primitive is M(theta, phi; eps_a, eps_d) =
exp[-i theta{(1 + eps_a) rho(phi).sigma + eps_d sigma_z}/2], exact for a square segment under time-independent errors,
with eps_a scaling the rotation ANGLE and eps_d = Delta/Omega, whose +eps_d sigma_z term does NOT reverse with the drive
phase; the addressing channel has the INVERTED zero (eps_N = 0 ideal, target I, M = R(theta eps_N, phi)). A sequence is
stored as (area, phase) in TIME order and folded right to left, with the target azimuth ADDED to every entry including
the zeroth pulse; a negative nominal area is emitted as a positive area at phase + pi. F_K = |Tr(U_id^dag U)|^2/4 is the
internal measure, F_C = sqrt(F_K) and F_avg = (2 F_K + 1)/3 are conversions; order n means a propagator residual
O(eps^{n+1}) and an infidelity slope 2(n + 1); durations are sum_l theta_l/Omega. The PD6 phases are Mount et al. 2015
Table I, indexed by the target ANGLE theta_t (the table's phi_t is a typo), with the target pulse first in time.
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

from qutip_trap.control.native import IDENTITY_2, PAULI_X, PAULI_Y, PAULI_Z

Family = Literal[
    "primitive",
    "SK1",
    "BB1",
    "NB1",
    "PB1",
    "P2j",
    "N2j",
    "B2j",
    "CORPSE",
    "short_CORPSE",
    "SCROFULOUS",
    "CinSK",
    "CinBB",
    "PDn",
    "APn",
    "BBn",
]
Channel = Literal["amplitude", "pulse_length", "addressing", "detuning"]

PI = math.pi
SUZUKI_PHASE_TOLERANCE = 1e-9
"""How far the root-found ladder phase may sit from the closed form arccos(-theta/(2 pi f_j)) before it is a defect."""

# Mount et al. 2015 Table I: PD6 phases phi_{PD6:1..6} for the target ANGLE theta_t (printed as phi_t, a typo)
MOUNT_PD6_PHASES: dict[float, tuple[float, ...]] = {
    PI: (0.38266, -2.51430, -1.75192, 0.05941, 2.67572, 0.39344),
    PI / 2.0: (0.34769, -3.06979, 1.55852, -0.70890, 3.09692, -0.62174),
}


# ---- the primitive, folding and the fidelity measures --------------------------------------------------------------------


def primitive(theta: float, phi: float, eps_a: float = 0.0, eps_d: float = 0.0) -> np.ndarray:
    """M(theta, phi; eps_a, eps_d) = exp[-i theta{(1 + eps_a) rho(phi).sigma + eps_d sigma_z}/2] (exact); R(theta, phi) at
    zero error."""
    nx = (1.0 + eps_a) * math.cos(phi)
    ny = (1.0 + eps_a) * math.sin(phi)
    nz = eps_d
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n == 0.0:
        return IDENTITY_2.copy()
    axis = (nx * PAULI_X + ny * PAULI_Y + nz * PAULI_Z) / n
    return math.cos(theta * n / 2.0) * IDENTITY_2 - 1j * math.sin(theta * n / 2.0) * axis


def fold(segments: Sequence[tuple[float, float]], eps_a: float = 0.0, eps_d: float = 0.0) -> np.ndarray:
    """U = M_last ... M_first for TIME-ordered (area, phase) segments."""
    u = IDENTITY_2.copy()
    for theta, phi in segments:
        u = primitive(theta, phi, eps_a, eps_d) @ u
    return u


def fold_addressing(segments: Sequence[tuple[float, float]], eps_n: float) -> np.ndarray:
    """The addressing model: an unaddressed spin sees R(theta eps_N, phi) per segment; its target is the identity."""
    u = IDENTITY_2.copy()
    for theta, phi in segments:
        u = primitive(theta * eps_n, phi) @ u
    return u


def fidelity_c(u: np.ndarray, v: np.ndarray) -> float:
    """F_C = |Tr(U^dag V)|/2 (Cummins)."""
    return float(abs(np.trace(u.conj().T @ v))) / 2.0


def fidelity_k(u: np.ndarray, v: np.ndarray) -> float:
    """F_K = |Tr(U^dag V)|^2/4 = F_C^2, the internal measure (Kabytayev 2014)."""
    return fidelity_c(u, v) ** 2


def fidelity_avg(u: np.ndarray, v: np.ndarray) -> float:
    """F_avg = (2 F_C^2 + 1)/3."""
    return (2.0 * fidelity_k(u, v) + 1.0) / 3.0


def operator_distance(u: np.ndarray, v: np.ndarray) -> float:
    """The spectral-norm distance ||U - V||_2, global phase INCLUDED."""
    return float(np.linalg.norm(u - v, 2))


def operator_distance_up_to_phase(u: np.ndarray, v: np.ndarray) -> float:
    """min_g ||U - e^{ig} V||_2, the minimizer being the bisector g = arg Tr(V^dag U) of the eigenphases of V^dag U (2x2); at
    Tr = 0 every g gives sqrt(2). A winding that folds a sequence to -R(theta, phi) (short-CORPSE) is then no error."""
    tr = complex(np.trace(v.conj().T @ u))
    g = cmath.phase(tr) if abs(tr) > 1e-300 else 0.0
    return float(np.linalg.norm(u - cmath.exp(1j * g) * v, 2))


def infidelity_from_distance(distance: float) -> float:
    """1 - F_K ~ E^2 at small spectral-norm distance E: U_err = exp(-i v.sigma) has E ~ |v| and 1 - F_K = sin^2|v|."""
    return distance**2


# ---- the verified library, TIME order, target azimuth added to every entry ------------------------------------------


def arcsinc(y: float) -> float:
    """Inverse of sin(x)/x on (0, pi], where it falls monotonically from 1 to 0; arcsinc(0) = pi. A negative y has its root
    on the second branch (pi, 4.493409], which is not implemented (SCROFULOUS at theta_t > pi)."""
    if y == 0.0:
        return PI
    if y == 1.0:
        return 0.0
    if not 0.0 < y < 1.0:
        raise ValueError(
            f"arcsinc({y:.6g}): only the branch (0, pi] of sin(x)/x, whose values are [0, 1], is implemented"
        )
    return float(brentq(lambda x: math.sin(x) / x - y, 1e-12, PI))


def _acos_checked(x: float, what: str) -> float:
    if not -1.0 <= x <= 1.0:
        raise ValueError(
            f"{what}: arccos argument {x:.6g} outside [-1, 1]; theta is outside the family's domain"
        )
    return math.acos(x)


def phi_sk1(theta: float) -> float:
    """phi_1 = arccos(-theta/(4 pi)), shared by SK1, BB1 and NB1 (the MINUS is load-bearing)."""
    return _acos_checked(-theta / (4.0 * PI), "SK1/BB1 phase")


def seq_sk1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """SK1: U_1X(-theta) M_0(theta): (theta, 0), (2 pi, +phi_1), (2 pi, -phi_1) (BHC Eq. 10 prints U_1X(+theta))."""
    p = phi_sk1(theta)
    return [(theta, phi_t), (2.0 * PI, phi_t + p), (2.0 * PI, phi_t - p)]


def seq_bb1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """BB1 = B2 = W1: (theta, 0), (pi, phi_1), (2 pi, 3 phi_1), (pi, phi_1) (Wimperis; Merrill-Brown Eq. 43; Mount Eqs. 1-2)."""
    p = phi_sk1(theta)
    return [(theta, phi_t), (PI, phi_t + p), (2.0 * PI, phi_t + 3.0 * p), (PI, phi_t + p)]


def seq_nb1(theta: float, phi_t: float = 0.0) -> list[tuple[float, float]]:
    """NB1 = N2: B2 with the middle phase -phi_1 instead of 3 phi_1."""
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
    """Emit a signed area as a POSITIVE area with a phase shift of pi where needed."""
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


def suzuki_factor(j: int, f1: float) -> float:
    """f_j = (2^{2j-1} - 2) f_{j-1}: 4, 24, 720, 90720 (P) and 2, 12, 360, 45360 (N, B); the source's (2^{2j-1} - 1)
    breaks first order."""
    f = float(f1)
    for m in range(2, j + 1):
        f *= 2 ** (2 * m - 1) - 2
    return f


def _axis3(phi: float) -> np.ndarray:
    return np.array([math.cos(phi), math.sin(phi), 0.0])


def _adjoint(u: np.ndarray) -> np.ndarray:
    """Lambda_ij = (1/2) Tr[U^dag sigma_i U sigma_j], the SO(3) matrix of U^dag (the toggling frame)."""
    s = (PAULI_X, PAULI_Y, PAULI_Z)
    lam = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            lam[i, j] = 0.5 * np.real(np.trace(u.conj().T @ s[i] @ u @ s[j]))
    return lam


def _toggled_sum(segments: Sequence[tuple[float, float]]) -> np.ndarray:
    """sum_l theta_l rho~^(l): the segment axes weighted by their areas in the toggling frame of the propagator before them,
    the first-order amplitude-error generator."""
    u = IDENTITY_2.copy()
    total = np.zeros(3)
    for theta, phi in segments:
        total += theta * (_adjoint(u).T @ _axis3(phi))
        u = primitive(theta, phi) @ u
    return total


def _first_order_coefficient(segments: Sequence[tuple[float, float]], family: str) -> float:
    """The leading eps coefficient the ladder phase must null, as one signed scalar: P and B correct the AMPLITUDE channel,
    whose first-order generator is the toggling-frame sum; N corrects the ADDRESSING channel, where every segment is itself
    O(eps_N) and the generator is the bare sum sum_l theta_l rho(phi_l). Both lie along the target axis."""
    if family == "N":
        return float(sum(theta * math.cos(phi) for theta, phi in segments))
    return float(_toggled_sum(segments)[0])


def suzuki_phase(j: int, theta: float, family: str) -> tuple[float, float]:
    """(phi, |closed form - root|): the ladder phase ROOT-FOUND on the leading eps coefficient, bracketed around the closed
    form arccos(-theta/(2 pi f_j)), so that a wrong f_j shows up as a large residual rather than a silently wrong phase."""
    if j < 1:
        raise ValueError("j >= 1")
    fj = suzuki_factor(j, 4.0 if family == "P" else 2.0)
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
    """P2j (family P), N2j (N) or B2j (B): the target then T_2j(1, phi) with phi ROOT-FOUND (``suzuki_phase``), which must
    agree with the closed form to ``SUZUKI_PHASE_TOLERANCE``."""
    root, residual = suzuki_phase(j, theta, family)
    if residual > SUZUKI_PHASE_TOLERANCE:
        raise ValueError(
            f"{family}{2 * j} at theta = {theta:.9g}: the root-found phase {root:.12f} disagrees with the closed form "
            f"arccos(-theta/(2 pi f_j)) by {residual:.3g}, above {SUZUKI_PHASE_TOLERANCE:g}"
        )
    return [(theta, phi_t)] + [(a, phi_t + p) for a, p in _t2j(j, 1.0, root, family)]


# ---- PDn / APn (Low-Yoder-Chuang 2014) and Mount's PD6 ---------------------------------------------------------------------


def certificate_phi(phases: Sequence[float], j: int) -> complex:
    """Phi_L^j(phi) = sum_{h_1 < ... < h_j} exp(-i sum_k (-1)^k phi_{h_k}), the alternating sign over the POSITION k
    in the subset (LYC Eq. 11), by dynamic programming over the phases rather than by enumerating subsets."""
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
    to the broadband BB_n at theta_0 = pi; toggled PD2 at gamma = 1 is Wimperis's BB1 at theta = pi."""
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
    key = next((k for k in MOUNT_PD6_PHASES if math.isclose(theta, k, rel_tol=0.0, abs_tol=1e-12)), None)
    if key is None:
        raise ValueError(
            "Mount's PD6 phases are tabulated for theta_t = pi and pi/2 only; the general-theta phase vector needs the "
            "numerical continuation of Low-Yoder-Chuang (not implemented)"
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


# ---- the families ---------------------------------------------------------------------------------------------------------


_Segments = list[tuple[float, float]]


def _fixed(
    sequence: Callable[[float, float], _Segments], order: int
) -> Callable[[float, float, int], tuple[_Segments, int]]:
    return lambda theta, phi, _order: (sequence(theta, phi), order)


def _ladder(family: str) -> Callable[[float, float, int], tuple[_Segments, int]]:
    """P2j/N2j/B2j at j = max(order, 1): order 2j."""

    def build(theta: float, phi: float, order: int) -> tuple[_Segments, int]:
        j = max(order, 1)
        return seq_ladder(j, theta, family, phi), 2 * j

    return build


def _pdn(theta: float, phi: float, order: int) -> tuple[_Segments, int]:
    if order == 2:
        return seq_pd2(theta, phi), 2
    if order == 6:
        return seq_pd6_mount(theta, phi), 6
    raise NotImplementedError(
        "PDn is implemented at n = 2 (closed form) and n = 6 (Mount's table at pi and pi/2)"
    )


def _apn(theta: float, phi: float, order: int) -> tuple[_Segments, int]:
    if order != 1:
        raise NotImplementedError(
            "APn is implemented at n = 1 (SK1); AP2/AP3 closed forms were not transcribed"
        )
    return [(theta, phi)] + [(2.0 * PI, phi + p) for p in ap1_phases(theta / (2.0 * PI))], 1


def _bbn(theta: float, phi: float, order: int) -> tuple[_Segments, int]:
    if order != 2:
        raise NotImplementedError("BBn is implemented at n = 2 (the toggled PD2 = BB1 at theta = pi)")
    return seq_bb_toggled(theta, phi), 2


@dataclass(frozen=True)
class _FamilySpec:
    build: Callable[[float, float, int], tuple[_Segments, int]]
    """(theta, phi, requested order) -> (TIME-ordered segments, the order achieved)."""
    corrects: frozenset[Channel]
    provenance_id: str


_AMPLITUDE: frozenset[Channel] = frozenset({"amplitude"})
_FAMILIES: dict[Family, _FamilySpec] = {
    "primitive": _FamilySpec(_fixed(lambda t, p: [(t, p)], 0), frozenset(), "conv.static_error_primitive"),
    "SK1": _FamilySpec(
        _fixed(seq_sk1, 1), frozenset({"amplitude", "addressing"}), "anchor.composite.sk1_sign"
    ),
    "BB1": _FamilySpec(_fixed(seq_bb1, 2), _AMPLITUDE, "anchor.composite.bb1"),
    "NB1": _FamilySpec(_fixed(seq_nb1, 2), frozenset({"addressing"}), "anchor.composite.bb1"),
    "PB1": _FamilySpec(
        _fixed(seq_pb1, 2), frozenset({"amplitude", "addressing"}), "anchor.composite.pb1_five_pulses"
    ),
    "P2j": _FamilySpec(_ladder("P"), _AMPLITUDE, "anchor.composite.suzuki_factor"),
    "N2j": _FamilySpec(_ladder("N"), frozenset({"addressing"}), "anchor.composite.suzuki_factor"),
    "B2j": _FamilySpec(_ladder("B"), _AMPLITUDE, "anchor.composite.suzuki_factor"),
    "CORPSE": _FamilySpec(_fixed(seq_corpse, 1), frozenset({"detuning"}), "anchor.composite.corpse"),
    "short_CORPSE": _FamilySpec(
        _fixed(seq_short_corpse, 1), frozenset({"detuning"}), "anchor.composite.corpse"
    ),
    "SCROFULOUS": _FamilySpec(_fixed(seq_scrofulous, 1), _AMPLITUDE, "anchor.composite.scrofulous"),
    "CinSK": _FamilySpec(
        _fixed(seq_cinsk, 1), frozenset({"amplitude", "detuning"}), "anchor.composite.concatenated_durations"
    ),
    "CinBB": _FamilySpec(
        _fixed(seq_cinbb, 1), frozenset({"amplitude", "detuning"}), "anchor.composite.concatenated_durations"
    ),
    "PDn": _FamilySpec(_pdn, _AMPLITUDE, "anchor.composite.mount_pd6"),
    "APn": _FamilySpec(_apn, frozenset({"amplitude", "addressing"}), "anchor.composite.sk1_sign"),
    "BBn": _FamilySpec(_bbn, _AMPLITUDE, "anchor.composite.toggled_pd2"),
}


@dataclass(frozen=True)
class CompositePulse:
    """A static-error-compensating single-qubit sequence."""

    family: Family
    theta_rad: float
    phi_rad: float
    """Target axis azimuth; ADDED to every segment phase."""
    order: int
    """n; residual O(eps^{n+1}), infidelity slope 2(n+1)."""
    corrects: frozenset[Channel]
    segments: tuple[tuple[float, float], ...]
    """(area_rad, phase_rad) in TIME order, areas > 0."""
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(f"unknown composite-pulse family {self.family!r}")
        if self.order < 0:
            raise ValueError("order must be >= 0")
        if not self.segments:
            raise ValueError("a composite pulse has at least one segment")
        if any(area <= 0.0 for area, _ in self.segments):
            raise ValueError("segment areas are positive; a negative nominal area is emitted at phase + pi")

    def total_rotation_rad(self) -> float:
        """Sum of the segment areas; the duration at constant amplitude is this over Omega."""
        return float(sum(area for area, _ in self.segments))

    def duration_s(self, omega_rabi_rad_s: float) -> float:
        return self.total_rotation_rad() / omega_rabi_rad_s

    def target(self) -> np.ndarray:
        return primitive(self.theta_rad, self.phi_rad)

    def propagator(self, eps_a: float = 0.0, eps_d: float = 0.0) -> np.ndarray:
        """2x2, folded right to left."""
        return fold(self.segments, eps_a, eps_d)

    def infidelity(self, eps_a: float = 0.0, eps_d: float = 0.0) -> float:
        """1 - F_K against the target."""
        return 1.0 - fidelity_k(self.target(), self.propagator(eps_a, eps_d))

    def distance(self, eps_a: float = 0.0, eps_d: float = 0.0) -> float:
        """min_g ||U(eps) - e^{ig} R(theta, phi)||_2: the error size with the global phase quotiented out (short-CORPSE's
        (0, 1, 0) winding returns -R exactly)."""
        return operator_distance_up_to_phase(self.propagator(eps_a, eps_d), self.target())

    def order_slope(
        self,
        channel: Literal["amplitude", "detuning", "simultaneous"],
        eps_range: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """(log-log slope of 1 - F_K, fit residual) over an order-aware window eps >> 10^(-8/(n+1)), default (1e-2, 1e-1);
        below 1 - F ~ 1e-8 the subtracted infidelity hits the float64 floor, so the fit uses the squared operator distance
        there."""
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
        """sum_l theta_l rho~^(l) in the toggling frame, shape (3,), and whether it closes (|.| < rtol sum theta_l): closure is
        first-order amplitude compensation (the dc limit of the amplitude filter function)."""
        total = _toggled_sum(self.segments)
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
            raise ValueError("the certificate is stated for 2 pi correctors (LYC theta_0 = 2 pi)")
        phases = [p - self.phi_rad for _, p in corrector]
        gamma = self.theta_rad / theta0
        return certificate(phases, gamma, len(corrector), rtol)

    def filter_function_amplitude(self, omega_rad_s: np.ndarray, omega_rabi_rad_s: float) -> np.ndarray:
        """F_a(omega) = (1/4){|sum_l A_l rho~^(l)|^2 + |sum_l B_l rho~^(l)|^2} at constant Rabi frequency: the
        exact segment sum of ``noise.decoupling.amplitude_filter_function``."""
        from qutip_trap.noise.decoupling import amplitude_filter_function, composite_segments

        return amplitude_filter_function(
            composite_segments(self, omega_rabi_rad_s), np.asarray(omega_rad_s, dtype=float)
        )

    def dc_floor(self, moments: Mapping[str, float], omega_rabi_rad_s: float) -> float:
        """sum_channels c-hat_{m+1} (2m + 1)!! (<beta^2>/Omega^2)^{m+1}: the frozen-noise floor. ``moments`` maps
        a channel ("amplitude" and/or "detuning") to its <beta^2> in (rad/s)^2; ``m`` is this pulse's order FOR THAT CHANNEL
        (0 for a channel it does not correct); c-hat is the fitted leading coefficient (``fitted_leading_coefficient``)."""
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


def composite_pulse(
    family: Family, theta_rad: float, phi_rad: float = 0.0, *, order: int = 1
) -> CompositePulse:
    """The ``family`` sequence for target angle ``theta_rad`` about azimuth ``phi_rad``, phases from the verified closed
    forms; raises outside the arccos domain, never NaN. ``order`` selects j for P2j/N2j/B2j (order 2j) and n for
    PDn/APn/BBn (PDn: n = 2 closed form, n = 6 Mount's table at theta_t = pi or pi/2; APn: n = 1 = SK1; BBn: n = 2 at
    theta = pi, the toggled PD2)."""
    if family not in _FAMILIES:
        raise ValueError(f"unknown composite-pulse family {family!r}; known: {tuple(_FAMILIES)}")
    if theta_rad <= 0.0:
        raise ValueError(
            "theta_rad is a positive target angle; a negative angle is the same area at phase + pi"
        )
    spec = _FAMILIES[family]
    segments, achieved = spec.build(theta_rad, phi_rad, order)
    return CompositePulse(
        family, theta_rad, phi_rad, achieved, spec.corrects, tuple(segments), spec.provenance_id
    )


_C_HAT_CACHE: dict[tuple[str, float, int, str, int], float] = {}
"""Fitted c-hat_{m+1} per (family, theta, order, channel, power): the fit is a few hundred 2x2 folds."""


def fitted_leading_coefficient(
    pulse: CompositePulse, channel: str, power: int, *, terms: int = 3, distance_floor: float = 1e-10
) -> float:
    """c-hat = lim_{eps -> 0} (1 - F_K)/eps^power, fitted rather than read off at one eps: the eps ladder whose operator
    distance is still above ``distance_floor`` is least-squared to c-hat + a eps^2 + b eps^4 in eps^2, balancing the
    truncation error of the next term against the round-off of 1 - F_K near 1."""
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
    """(1 - F_K)/eps^power at a small eps, the leading coefficient (float64: power <= 4 is reliable)."""
    ea = eps if channel == "amplitude" else 0.0
    ed = eps if channel == "detuning" else 0.0
    inf = pulse.infidelity(ea, ed)
    if inf < 1e-8:
        inf = infidelity_from_distance(pulse.distance(ea, ed))
    return inf / eps**power
