"""Dynamical decoupling and the filter-function description of dephasing and amplitude noise.

Conventions: H(t) = H_c(t) + beta(t) . sigma with no 1/2 on the Pauli vector; the control matrix is
R_ij(t) = (1/2) Tr[U_c^dag(t) sigma_i U_c(t) sigma_j] and R_ij(omega) = -i omega int_0^tau dt R_ij(t) e^{+i omega t},
built from the accumulated adjoints Lambda^{(l-1)} (never a scalar (-1)^l, so multi-axis sequences are exact). The
dephasing variable b is HALF the splitting fluctuation (H_deph = b sigma_z, S_b = S_delta/4); F_z = sum_i |R_zi|^2,
chi = (2/pi) int (d omega/omega^2) S_b F_z, W = e^{-chi} and 1 - F_av = chi/2 to first order. The reported infidelity
is max(first order, dc floor); chi carries d ln chi/d ln omega_min because it can diverge in the infrared.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from qutip_trap.control.composite import CompositePulse

if TYPE_CHECKING:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.experiments.result import ExperimentResult
    from qutip_trap.noise.spectra import NoiseSpectrum

M7 = "milestone M7 (noise/decoupling.py, PLAN.md Section 6.9)"

Timing = Literal["hahn", "cpmg", "udd", "xy4", "xy8", "kdd", "cdd", "custom"]
Quadrature = Literal["dephasing", "amplitude", "universal"]

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = (_SX, _SY, _SZ)


@dataclass(frozen=True)
class ControlSegment:
    """One piecewise-constant control interval: a rotation ``theta_rad`` about the equatorial axis at azimuth
    ``phi_rad`` over ``duration_s`` (0 = an instantaneous pulse)."""

    theta_rad: float
    phi_rad: float
    duration_s: float

    def __post_init__(self) -> None:
        if self.duration_s < 0.0:
            raise ValueError(
                "a control segment has a non-negative duration (zero = an instantaneous bang-bang pulse)"
            )
        if self.theta_rad < 0.0:
            raise ValueError("segment angles are non-negative; a reversed rotation is a phase of pi")
        if self.duration_s == 0.0 and self.theta_rad == 0.0:
            raise ValueError("an empty segment")

    @property
    def omega_rad_s(self) -> float:
        return math.inf if self.duration_s == 0.0 else self.theta_rad / self.duration_s

    @property
    def is_free(self) -> bool:
        return self.theta_rad == 0.0

    @property
    def is_instantaneous(self) -> bool:
        return self.duration_s == 0.0


def rotation(theta: float, phi: float) -> np.ndarray:
    """exp(-i theta sigma_phi/2), sigma_phi = cos phi X + sin phi Y."""
    s = math.cos(phi) * _SX + math.sin(phi) * _SY
    return math.cos(theta / 2.0) * np.eye(2, dtype=complex) - 1j * math.sin(theta / 2.0) * s


def adjoint(u: np.ndarray) -> np.ndarray:
    """Lambda_ij = (1/2) Tr[U^dag sigma_i U sigma_j], a real SO(3) matrix (the adjoint representation of U^dag)."""
    out = np.empty((3, 3))
    ud = u.conj().T
    for i in range(3):
        a = ud @ _PAULIS[i] @ u
        for j in range(3):
            out[i, j] = float(np.real(np.trace(a @ _PAULIS[j])) / 2.0)
    return out


def _axis(phi: float) -> np.ndarray:
    return np.array([math.cos(phi), math.sin(phi), 0.0])


def _cross_matrix(n: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])


def pulse_adjoint(theta: float, phi: float) -> np.ndarray:
    """R^P(theta) = cos theta 1 + (1 - cos theta) n n^T + sin theta [n]_x for the axis n(phi); equals ``adjoint(rotation)``."""
    n = _axis(phi)
    return (
        math.cos(theta) * np.eye(3)
        + (1.0 - math.cos(theta)) * np.outer(n, n)
        + math.sin(theta) * _cross_matrix(n)
    )


def _e_integral(x: float | np.ndarray, tau: float) -> np.ndarray:
    """E(x) = int_0^tau e^{i x s} ds = (e^{i x tau} - 1)/(i x), with the x -> 0 limit tau (removable pole)."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=complex)
    small = np.abs(x) * tau < 1e-8
    out[small] = tau
    xs = x[~small]
    out[~small] = (np.exp(1j * xs * tau) - 1.0) / (1j * xs)
    return out


def segment_frequency_integrals(
    seg: ControlSegment, t_start: float, omega: np.ndarray, *, gated: bool = False
) -> np.ndarray:
    """M_l(omega) = int_0^tau_l R^{P_l}(s) e^{i omega (t_{l-1} + s)} ds, shape (len(omega), 3, 3); ``gated=True`` turns
    the noise off during pulses (Biercuk 2009's idealization), the default keeps it on (Green 2013)."""
    w = np.asarray(omega, dtype=float)
    tau = seg.duration_s
    if seg.is_instantaneous or (gated and not seg.is_free):
        return np.zeros((w.size, 3, 3), dtype=complex)
    phase = np.exp(1j * w * t_start)
    i1 = _e_integral(w, tau)
    if seg.is_free:
        out = np.zeros((w.size, 3, 3), dtype=complex)
        for i in range(3):
            out[:, i, i] = phase * i1
        return out
    om = seg.omega_rad_s
    n = _axis(seg.phi_rad)
    ic = 0.5 * (_e_integral(w + om, tau) + _e_integral(w - om, tau))
    is_ = (_e_integral(w + om, tau) - _e_integral(w - om, tau)) / (2.0j)
    eye = np.eye(3)
    nn = np.outer(n, n)
    kx = _cross_matrix(n)
    out = (
        ic[:, None, None] * eye[None, :, :]
        + (i1 - ic)[:, None, None] * nn[None, :, :]
        + is_[:, None, None] * kx[None, :, :]
    )
    return np.asarray(phase[:, None, None] * out)


def accumulated_adjoints(segments: Sequence[ControlSegment]) -> list[np.ndarray]:
    """Lambda^{(l-1)} for l = 1..L: the adjoint of the propagator accumulated before each segment (Lambda^{(0)} = 1)."""
    out = [np.eye(3)]
    for seg in segments[:-1]:
        out.append(pulse_adjoint(seg.theta_rad, seg.phi_rad) @ out[-1])
    return out


def final_adjoint(segments: Sequence[ControlSegment]) -> np.ndarray:
    lam = np.eye(3)
    for seg in segments:
        lam = pulse_adjoint(seg.theta_rad, seg.phi_rad) @ lam
    return lam


def control_matrix(
    segments: Sequence[ControlSegment], omega: np.ndarray, *, gated: bool = False
) -> np.ndarray:
    """R_ij(omega) = -i omega sum_l M_l(omega) Lambda^{(l-1)}, shape (len(omega), 3, 3)."""
    w = np.asarray(omega, dtype=float)
    lams = accumulated_adjoints(segments)
    total = np.zeros((w.size, 3, 3), dtype=complex)
    t = 0.0
    for seg, lam in zip(segments, lams):
        total += segment_frequency_integrals(seg, t, w, gated=gated) @ lam[None, :, :]
        t += seg.duration_s
    return (-1j * w)[:, None, None] * total


def control_matrix_time(segments: Sequence[ControlSegment], t: float) -> np.ndarray:
    """R(t) by the trace formula on the actual propagator (a reference for the closed forms)."""
    u = np.eye(2, dtype=complex)
    elapsed = 0.0
    for seg in segments:
        if t <= elapsed:
            break
        if seg.is_instantaneous:
            u = rotation(seg.theta_rad, seg.phi_rad) @ u
            continue
        s = min(t - elapsed, seg.duration_s)
        if not seg.is_free:
            u = rotation(seg.omega_rad_s * s, seg.phi_rad) @ u
        elapsed += seg.duration_s
    return adjoint(u)


def dephasing_filter_function(
    segments: Sequence[ControlSegment], omega: np.ndarray, *, gated: bool = False
) -> np.ndarray:
    """F_z^{(1)}(omega) = sum_i |R_zi(omega)|^2 over all three components."""
    r = control_matrix(segments, omega, gated=gated)
    return np.asarray(np.sum(np.abs(r[:, 2, :]) ** 2, axis=1))


def amplitude_filter_function(segments: Sequence[ControlSegment], omega: np.ndarray) -> np.ndarray:
    """F_a(omega) = (1/4){|sum_l A_l rho~_l|^2 + |sum_l B_l rho~_l|^2} over the driven segments, with
    rho~_l = n(phi_l) Lambda^{(l-1)}, A_l = cos omega t_l - cos omega t_{l-1} and
    B_l = sin omega t_l - sin omega t_{l-1}."""
    w = np.asarray(omega, dtype=float)
    lams = accumulated_adjoints(segments)
    a_sum = np.zeros((w.size, 3))
    b_sum = np.zeros((w.size, 3))
    t = 0.0
    for seg, lam in zip(segments, lams):
        t1 = t + seg.duration_s
        if not seg.is_free:
            rho = _axis(seg.phi_rad) @ lam
            a_l = np.cos(w * t1) - np.cos(w * t)
            b_l = np.sin(w * t1) - np.sin(w * t)
            a_sum += a_l[:, None] * rho[None, :]
            b_sum += b_l[:, None] * rho[None, :]
        t = t1
    return np.asarray(0.25 * (np.sum(a_sum**2, axis=1) + np.sum(b_sum**2, axis=1)))


def universal_filter_function(
    segments: Sequence[ControlSegment], omega: np.ndarray, *, gated: bool = False
) -> np.ndarray:
    """sum_{ij} |R_ij(omega)|^2: the trace form for an isotropic noise vector (all three quadratures alike)."""
    r = control_matrix(segments, omega, gated=gated)
    return np.asarray(np.sum(np.abs(r) ** 2, axis=(1, 2)))


def dc_polygon(segments: Sequence[ControlSegment]) -> np.ndarray:
    """sum_l theta_l rho~_a^{(l)}: zero for an amplitude-correcting composite pulse (a closed polygon of toggled axes)."""
    lams = accumulated_adjoints(segments)
    total = np.zeros(3)
    for seg, lam in zip(segments, lams):
        if not seg.is_free:
            total += seg.theta_rad * (_axis(seg.phi_rad) @ lam)
    return total


def chi_integral(
    spectrum_two_sided: Callable[[np.ndarray], np.ndarray],
    filter_fn: Callable[[np.ndarray], np.ndarray],
    omega_min_rad_s: float,
    omega_max_rad_s: float,
    *,
    n_points: int = 20001,
) -> float:
    """chi = (2/pi) int_{omega_min}^{omega_max} (d omega/omega^2) S_b(omega) F(omega) on a dense log grid (trapezoid in ln omega)."""
    if omega_min_rad_s <= 0.0 or omega_max_rad_s <= omega_min_rad_s:
        raise ValueError("0 < omega_min < omega_max")
    u = np.linspace(math.log(omega_min_rad_s), math.log(omega_max_rad_s), int(n_points))
    w = np.exp(u)
    integrand = (
        np.asarray(spectrum_two_sided(w), dtype=float) * np.asarray(filter_fn(w), dtype=float) / w
    )  # d omega = omega du
    return float(2.0 / math.pi * np.trapezoid(integrand, u))


def double_factorial_odd(m: int) -> int:
    """(2m + 1)!! = 1 x 3 x ... x (2m + 1)."""
    out = 1
    for k in range(1, 2 * m + 2, 2):
        out *= k
    return out


def dc_floor(c_hat: float, order: int, beta_variance: float, omega_rad_s: float) -> float:
    """c^_{m+1} (2m + 1)!! (<beta^2>/Omega^2)^{m+1}, the frozen-noise floor of an mth-order sequence; ``beta_variance``
    must match c_hat's normalization (Mount's eps_d = 2 beta_d/Omega, so the detuning channel takes 4 <beta_d^2>)."""
    rel = beta_variance / omega_rad_s**2
    return c_hat * double_factorial_odd(order) * rel ** (order + 1)


def frozen_noise_infidelity(
    segs: Sequence[ControlSegment], beta_rad_s: float, quadrature: Quadrature = "dephasing"
) -> float:
    """1 - (1/4)|Tr(U_c^dag U)|^2 under a constant noise field beta, exact to all orders: H_0 = beta sigma_z for
    dephasing (beta half the splitting fluctuation), beta sigma_phi/2 on driven segments for amplitude."""
    if quadrature == "universal":
        raise ValueError(
            "the frozen-noise limit is evaluated one quadrature at a time (dephasing or amplitude)"
        )
    u = np.eye(2, dtype=complex)
    uc = np.eye(2, dtype=complex)
    for seg in segs:
        s_phi = math.cos(seg.phi_rad) * _SX + math.sin(seg.phi_rad) * _SY
        if seg.is_instantaneous:
            block_c = rotation(seg.theta_rad, seg.phi_rad)
            block = block_c
        else:
            h_c = 0.5 * seg.omega_rad_s * s_phi if not seg.is_free else np.zeros((2, 2), dtype=complex)
            h_0 = beta_rad_s * _SZ if quadrature == "dephasing" else 0.5 * beta_rad_s * s_phi
            if seg.is_free and quadrature == "amplitude":
                h_0 = np.zeros((2, 2), dtype=complex)  # no drive, no amplitude noise
            block_c = _expm_hermitian2(h_c, seg.duration_s)
            block = _expm_hermitian2(h_c + h_0, seg.duration_s)
        u = block @ u
        uc = block_c @ uc
    return float(1.0 - abs(np.trace(uc.conj().T @ u)) ** 2 / 4.0)


def frozen_noise_floor(
    segs: Sequence[ControlSegment],
    beta_variance: float,
    quadrature: Quadrature = "dephasing",
    *,
    nodes: int = 41,
) -> float:
    """<1 - F> over a zero-mean Gaussian frozen beta of variance ``beta_variance``, by Gauss-Hermite quadrature: the dc
    floor without a fitted c_hat, of which c^_{m+1} (2m + 1)!! <beta^2>^{m+1} is the leading term."""
    if beta_variance < 0.0:
        raise ValueError("a variance is non-negative")
    if beta_variance == 0.0:
        return 0.0
    x, w = np.polynomial.hermite_e.hermegauss(int(nodes))
    w = w / float(np.sum(w))
    sigma = math.sqrt(beta_variance)
    vals = np.array([frozen_noise_infidelity(segs, sigma * float(xi), quadrature) for xi in x])
    return float(np.sum(w * vals))


def _expm_hermitian2(h: np.ndarray, t: float) -> np.ndarray:
    """exp(-i t h) for a traceless Hermitian 2x2 h = v . sigma: cos(|v| t) 1 - i sin(|v| t) (v_hat . sigma)."""
    vx = float(np.real(h[0, 1]))
    vy = float(np.imag(h[1, 0]))
    vz = float(np.real(h[0, 0]))
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0.0:
        return np.eye(2, dtype=complex)
    c, s_ = math.cos(norm * t), math.sin(norm * t) / norm
    return c * np.eye(2, dtype=complex) - 1j * s_ * (vx * _SX + vy * _SY + vz * _SZ)


def cpmg_centres(n: int) -> tuple[float, ...]:
    return tuple((2.0 * j - 1.0) / (2.0 * n) for j in range(1, n + 1))


def udd_centres(n: int) -> tuple[float, ...]:
    return tuple(math.sin(math.pi * j / (2.0 * n + 2.0)) ** 2 for j in range(1, n + 1))


# mpmath is a development dependency, so these import it lazily. A filter function is an O(1) sum that cancels to high
# order in omega, so float64 loses the signal by n ~ 4; these are the float64 equations evaluated at dps digits.


def biercuk_amplitude_mp(
    deltas: Sequence[float], n_pulses: int, x: Any, tau_pi_frac: float = 0.0, dps: int = 200
) -> Any:
    """R_zz(x)/(-i) = 1 + (-1)^{n+1} e^{ix} + 2 cos(x delta_pi/2) sum_j (-1)^j e^{i d_j x} of Biercuk Eq. 2 at ``dps``
    digits, x = omega tau and delta_pi = ``tau_pi_frac``."""
    from mpmath import mp  # type: ignore[import-untyped]  # mpmath ships no py.typed marker

    with mp.workdps(int(dps)):
        xx = _as_mpf(x, dps)
        acc = mp.mpf(0)
        for j, d in enumerate(deltas, start=1):
            acc += (-1) ** j * mp.e ** (1j * _as_mpf(d, dps) * xx)
        y = 1 + (-1) ** (n_pulses + 1) * mp.e ** (1j * xx)
        y += 2 * mp.cos(xx * _as_mpf(tau_pi_frac, dps) / 2) * acc
        return +y


def chi_integral_mp(
    spectrum_two_sided: Callable[[Any], Any],
    filter_fn: Callable[[Any], Any],
    omega_min_rad_s: float,
    omega_max_rad_s: float,
    *,
    dps: int = 120,
    maxdegree: int = 9,
) -> Any:
    """The mpmath twin of :func:`chi_integral` (``mp.quad`` in u = ln omega), for an F lost to float64 cancellation."""
    from mpmath import mp

    if omega_min_rad_s <= 0.0 or omega_max_rad_s <= omega_min_rad_s:
        raise ValueError("0 < omega_min < omega_max")
    with mp.workdps(int(dps)):

        def integrand(u: Any) -> Any:
            w = mp.e**u
            return spectrum_two_sided(w) * filter_fn(w) / w

        lo = mp.log(_as_mpf(omega_min_rad_s, dps))
        hi = mp.log(_as_mpf(omega_max_rad_s, dps))
        return +(2 / mp.pi * mp.quad(integrand, [lo, hi], maxdegree=int(maxdegree)))


def _as_mpf(value: Any, dps: int) -> Any:
    """``mp.mpf`` of a float (via ``repr``, so mpmath sees the decimal the caller wrote), an int or an mpmath number."""
    from mpmath import mp

    with mp.workdps(int(dps)):
        if isinstance(value, float):
            return mp.mpf(repr(value))
        if isinstance(value, int):
            return mp.mpf(value)
        return +value


def biercuk_filter_function_mp(
    deltas: Sequence[float], n_pulses: int, x: Any, tau_pi_frac: float = 0.0, dps: int = 200
) -> Any:
    """|R_zz|^2 of Biercuk Eq. 2 at ``mp.dps = dps`` (the mpmath twin of ``biercuk_filter_function``)."""
    from mpmath import mp

    with mp.workdps(int(dps)):
        return abs(biercuk_amplitude_mp(deltas, n_pulses, x, tau_pi_frac, dps)) ** 2


def cpmg_centres_mp(n: int, dps: int = 200) -> tuple[Any, ...]:
    """delta_j = (2j - 1)/(2n) at ``dps`` digits."""
    from mpmath import mp

    with mp.workdps(int(dps)):
        return tuple(mp.mpf(2 * j - 1) / (2 * n) for j in range(1, n + 1))


def udd_centres_mp(n: int, dps: int = 200) -> tuple[Any, ...]:
    """delta_j = sin^2[pi j/(2n + 2)] at ``dps`` digits (float64 ``udd_centres`` caps the verifiable order near 4)."""
    from mpmath import mp

    with mp.workdps(int(dps)):
        return tuple(mp.sin(mp.pi * j / (2 * n + 2)) ** 2 for j in range(1, n + 1))


def biercuk_taylor_coefficients_mp(
    deltas: Sequence[Any], n_pulses: int, dps: int = 200, max_order: int = 32
) -> list[Any]:
    """The exact Taylor coefficients in x = omega tau of the bang-bang Biercuk amplitude, A_k = sum_j (-1)^j delta_j^k:
    c_0 = 1 + (-1)^{n+1} + 2 A_0 and c_k = (i^k/k!)[(-1)^{n+1} + 2 A_k] for k >= 1."""
    from mpmath import mp

    with mp.workdps(int(dps)):
        parity = mp.mpf((-1) ** (n_pulses + 1))
        out: list[Any] = []
        for k in range(int(max_order) + 1):
            a_k = mp.mpf(0)
            for j, d in enumerate(deltas, start=1):
                a_k += (-1) ** j * (mp.mpf(1) if k == 0 else d**k)
            bracket = parity + 2 * a_k
            if k == 0:
                out.append(+(1 + bracket))
            else:
                out.append(+((1j) ** k / mp.factorial(k) * bracket))
        return out


def leading_taylor_order_mp(
    deltas: Sequence[Any], n_pulses: int, dps: int = 200, max_order: int = 32
) -> tuple[int, Any]:
    """(k, c_k): the lowest Taylor order of the bang-bang Biercuk amplitude whose coefficient exceeds 10^{-dps/2}."""
    from mpmath import mp

    with mp.workdps(int(dps)):
        floor_mag = mp.mpf(10) ** (-int(dps) // 2)
        for k, c in enumerate(biercuk_taylor_coefficients_mp(deltas, n_pulses, dps, max_order)):
            if abs(c) > floor_mag:
                return k, +c
        raise ValueError(f"no nonvanishing Taylor coefficient below order {max_order} at dps = {dps}")


def _xy_axes(n: int, pattern: Sequence[float]) -> tuple[float, ...]:
    if n % len(pattern) != 0:
        raise ValueError(f"the pulse count must be a multiple of {len(pattern)} for this axis pattern")
    return tuple(pattern[j % len(pattern)] for j in range(n))


KDD_BLOCK = (math.pi / 6.0, 0.0, math.pi / 2.0, 0.0, math.pi / 6.0)
"""The five-pulse Knill block of KDD (Souza, Alvarez and Suter 2011), phases relative to the block axis: one block is a
pi rotation about the equatorial axis at -60 degrees and an XY4 cycle of four blocks (20 pulses) the identity."""


def _cdd_tokens(level: int) -> list[str]:
    """CDD_l = CDD_{l-1} X CDD_{l-1} Y CDD_{l-1} X CDD_{l-1} Y with CDD_0 = a free interval."""
    if level == 0:
        return ["f"]
    inner = _cdd_tokens(level - 1)
    return inner + ["X"] + inner + ["Y"] + inner + ["X"] + inner + ["Y"]


@dataclass(frozen=True)
class DecouplingSequence:
    """A net-identity pulse train, or a decoupled idle."""

    timing: Timing
    n_pulses: int
    tau_s: float
    """TOTAL duration, INCLUSIVE of the pi-pulse widths."""
    tau_pi_s: float
    """One pulse duration; delta_pi = tau_pi_s/tau_s."""
    deltas: tuple[float, ...]
    """Fractional pulse CENTRES in [0, 1]; len == n_pulses."""
    axes_rad: tuple[float, ...]
    """Per-pulse axis azimuth; all zero for single-axis families."""
    inner: CompositePulse | None = None
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if self.n_pulses < 0 or self.tau_s <= 0.0 or self.tau_pi_s < 0.0:
            raise ValueError("n_pulses >= 0, tau_s > 0, tau_pi_s >= 0")
        if len(self.deltas) != self.n_pulses or len(self.axes_rad) != self.n_pulses:
            raise ValueError("deltas and axes_rad have one entry per pulse")
        if any(not 0.0 <= d <= 1.0 for d in self.deltas):
            raise ValueError("pulse centres are fractions of the total duration")
        if any(b <= a for a, b in zip(self.deltas, self.deltas[1:])):
            raise ValueError("pulse centres must be strictly increasing")

    def is_single_axis(self) -> bool:
        """Whether every pulse shares one axis, the condition for Biercuk's scalar form."""
        return (
            all(math.isclose(a, self.axes_rad[0], abs_tol=1e-12) for a in self.axes_rad)
            if self.axes_rad
            else True
        )

    def feasible(self) -> bool:
        """delta_pi <= 2 sin^2[pi/(2n + 2)], every pulse inside the interval and no two overlapping."""
        if self.n_pulses == 0:
            return True
        delta_pi = self.tau_pi_s / self.tau_s
        if delta_pi > 2.0 * math.sin(math.pi / (2.0 * self.n_pulses + 2.0)) ** 2 * (1.0 + 1e-12):
            return False
        centres = list(self.deltas)
        if centres[0] - delta_pi / 2.0 < -1e-12 or centres[-1] + delta_pi / 2.0 > 1.0 + 1e-12:
            return False
        return all(b - a >= delta_pi * (1.0 - 1e-12) for a, b in zip(centres, centres[1:]))

    def moments(self, k_max: int = 2) -> tuple[float, ...]:
        """A_k = sum_j (-1)^j delta_j^k for k = 1..k_max (the low-frequency roll-off moments; j counts from 1)."""
        return tuple(
            float(sum((-1) ** j * d**k for j, d in enumerate(self.deltas, start=1)))
            for k in range(1, k_max + 1)
        )

    def segments(self) -> tuple[ControlSegment, ...]:
        """The piecewise-constant control: free intervals and pi pulses (or the inner composite pulse) at the centres."""
        out: list[ControlSegment] = []
        t = 0.0
        for delta, axis in zip(self.deltas, self.axes_rad):
            start = delta * self.tau_s - self.tau_pi_s / 2.0
            if start - t > 1e-15 * self.tau_s:
                out.append(ControlSegment(0.0, 0.0, start - t))
            if self.inner is None:
                out.append(ControlSegment(math.pi, axis, self.tau_pi_s))
            else:
                total = self.inner.total_rotation_rad()
                for area, phase in self.inner.segments:
                    out.append(ControlSegment(area, axis + phase, self.tau_pi_s * area / total))
            t = start + self.tau_pi_s
        if self.tau_s - t > 1e-15 * self.tau_s:
            out.append(ControlSegment(0.0, 0.0, self.tau_s - t))
        return tuple(out)

    def control_matrix(self, omega_rad_s: np.ndarray) -> np.ndarray:
        """(len(omega), 3, 3) complex R_ij(omega) (Green 2013 Eqs. 25-30)."""
        return control_matrix(self.segments(), np.asarray(omega_rad_s, dtype=float))

    def filter_function(
        self,
        omega_rad_s: np.ndarray,
        quadrature: Quadrature = "dephasing",
        *,
        gated: bool = False,
    ) -> np.ndarray:
        """F(omega) of one quadrature; ``gated=True`` turns the noise off during the pulses (Biercuk's idealization)."""
        segs = self.segments()
        w = np.asarray(omega_rad_s, dtype=float)
        if quadrature == "dephasing":
            return dephasing_filter_function(segs, w, gated=gated)
        if quadrature == "amplitude":
            return amplitude_filter_function(segs, w)
        return universal_filter_function(segs, w, gated=gated)

    def biercuk_filter_function(self, omega_rad_s: np.ndarray) -> np.ndarray:
        """Biercuk 2009 Eq. 2 for a single-axis pi train at both parities, x = omega tau:
        |1 + (-1)^{n+1} e^{ix} + 2 cos(x delta_pi/2) sum_j (-1)^j e^{i delta_j x}|^2."""
        if not self.is_single_axis():
            raise ValueError("Biercuk's scalar form holds for single-axis trains only (Section 6.9)")
        x = np.asarray(omega_rad_s, dtype=float) * self.tau_s
        n = self.n_pulses
        acc = np.zeros_like(x, dtype=complex)
        for j, d in enumerate(self.deltas, start=1):
            acc += (-1) ** j * np.exp(1j * d * x)
        y = 1.0 + (-1) ** (n + 1) * np.exp(1j * x) + 2.0 * np.cos(x * self.tau_pi_s / self.tau_s / 2.0) * acc
        return np.asarray(np.abs(y) ** 2)

    def rounded_to_clock(self, clock_s: float) -> tuple[DecouplingSequence, dict[str, float]]:
        """Snap pulse centres to the clock grid, then move the LAST pulse to restore A_1 exactly when the grid allows; return
        the sequence and the residual moment errors (A_1, A_2) against the ideal timings."""
        if clock_s <= 0.0:
            raise ValueError("clock step must be positive")
        centres = [round(d * self.tau_s / clock_s) * clock_s / self.tau_s for d in self.deltas]
        a1_target = self.moments(1)[0]
        n = self.n_pulses
        if n > 0:
            a1_now = sum((-1) ** j * d for j, d in enumerate(centres, start=1))
            # the last pulse carries sign (-1)^n; shift it by the whole residual, then re-snap
            shift = (a1_target - a1_now) / ((-1) ** n)
            centres[-1] = round((centres[-1] + shift) * self.tau_s / clock_s) * clock_s / self.tau_s
        new = DecouplingSequence(
            self.timing,
            n,
            self.tau_s,
            self.tau_pi_s,
            tuple(centres),
            self.axes_rad,
            self.inner,
            self.provenance_id,
        )
        a1, a2 = new.moments(2)
        b1, b2 = self.moments(2)
        return new, {"A1_error": float(a1 - b1), "A2_error": float(a2 - b2)}


def decoupling_sequence(
    timing: str,
    n_pulses: int,
    tau_s: float,
    tau_pi_s: float,
    *,
    inner: CompositePulse | None = None,
    centres: Sequence[float] | None = None,
    axes_rad: Sequence[float] | None = None,
) -> DecouplingSequence:
    """CPMG delta_j = (2j - 1)/2n (exact and tau_pi-independent under the pulse-inclusive convention), UDD delta_j =
    sin^2[pi j/(2n + 2)], Hahn (n = 1), XY4/(XY)^N (CPMG timings, alternating X and Y axes), XY8 (X Y X Y Y X Y X), KDD (each
    pulse a Knill block of five, CPMG timings, block axes cycling X Y X Y), CDD (the concatenated recursion with equal free
    intervals; ``n_pulses`` is the level), or custom centres and axes."""
    axes: tuple[float, ...]
    deltas: tuple[float, ...]
    if timing == "hahn":
        if n_pulses != 1:
            raise ValueError("a Hahn echo has one pulse")
        deltas, axes = (0.5,), (0.0,)
    elif timing == "cpmg":
        deltas, axes = cpmg_centres(n_pulses), tuple(0.0 for _ in range(n_pulses))
    elif timing == "udd":
        deltas, axes = udd_centres(n_pulses), tuple(0.0 for _ in range(n_pulses))
    elif timing == "xy4":
        deltas, axes = cpmg_centres(n_pulses), _xy_axes(n_pulses, (0.0, math.pi / 2.0))
    elif timing == "xy8":
        pattern = (0.0, math.pi / 2.0, 0.0, math.pi / 2.0, math.pi / 2.0, 0.0, math.pi / 2.0, 0.0)
        deltas, axes = cpmg_centres(n_pulses), _xy_axes(n_pulses, pattern)
    elif timing == "kdd":
        if n_pulses % 20 != 0:
            raise ValueError(
                "KDD replaces every pi pulse of an XY4 cycle by a five-pulse Knill block: n_pulses is a multiple of 20 (one block "
                "is a pi rotation about an equatorial axis, two blocks a pi rotation about z, four blocks the identity)"
            )
        blocks = n_pulses // 5
        block_axes = [0.0, math.pi / 2.0]
        ax: list[float] = []
        for b in range(blocks):
            base = block_axes[b % 2]
            ax.extend(base + p for p in KDD_BLOCK)
        deltas, axes = cpmg_centres(n_pulses), tuple(ax)
    elif timing == "cdd":
        tokens = _cdd_tokens(n_pulses)
        n = sum(1 for t in tokens if t != "f")
        n_free = sum(1 for t in tokens if t == "f")
        free = (tau_s - n * tau_pi_s) / n_free
        if free <= 0.0:
            raise ValueError("the CDD pulses do not fit the interval")
        centres_l: list[float] = []
        ax = []
        t = 0.0
        for tok in tokens:
            if tok == "f":
                t += free
            else:
                centres_l.append((t + tau_pi_s / 2.0) / tau_s)
                ax.append(0.0 if tok == "X" else math.pi / 2.0)
                t += tau_pi_s
        deltas, axes = tuple(centres_l), tuple(ax)
        n_pulses = n
    elif timing == "custom":
        if centres is None:
            raise ValueError("custom timing needs the pulse centres")
        deltas = tuple(float(c) for c in centres)
        axes = tuple(float(a) for a in axes_rad) if axes_rad is not None else tuple(0.0 for _ in deltas)
        n_pulses = len(deltas)
    else:
        raise ValueError(f"unknown decoupling timing {timing!r}")
    seq = DecouplingSequence(
        cast(Timing, timing),
        n_pulses,
        tau_s,
        tau_pi_s,
        deltas,
        axes,
        inner,
        "conv.filter_function_normalization",
    )
    if n_pulses > 0 and tau_pi_s > 0.0 and not seq.feasible():
        raise ValueError(
            "the pulses overlap or leave the interval: delta_pi <= 2 sin^2[pi/(2n + 2)] is needed"
        )
    return seq


def composite_segments(pulse: CompositePulse, rabi_rad_s: float) -> tuple[ControlSegment, ...]:
    """A composite pulse at constant Rabi frequency: durations theta_l/Omega, phases as stored (``pulse.segments``
    already carries the target azimuth and the ``n_rep`` repetitions)."""
    if rabi_rad_s <= 0.0:
        raise ValueError("the Rabi frequency must be positive")
    return tuple(ControlSegment(area, phase, area / rabi_rad_s) for area, phase in pulse.segments)


def schedule_segments(schedule: Schedule, ion: int) -> tuple[ControlSegment, ...]:
    """The square carrier pulses of one ion of a schedule as control segments (detuned or shaped pulses are refused)."""
    segs: list[ControlSegment] = []
    t = min(p.t_start_s for p in schedule.pulses) if schedule.pulses else 0.0
    for p in sorted((p for p in schedule.pulses if ion in p.drive.ions), key=lambda p: p.t_start_s):
        if len(p.drive.tones) != 1:
            raise NotImplementedError("filter functions are built for single-tone carrier pulses")
        tone = p.drive.tones[0]
        if callable(tone.envelope_hz) or isinstance(tone.envelope_hz, np.ndarray) or callable(tone.phase_rad):
            raise NotImplementedError(
                "filter functions are built for square pulses with constant phases (Section 6.9)"
            )
        if p.t_start_s - t > 1e-15:
            segs.append(ControlSegment(0.0, 0.0, p.t_start_s - t))
        omega = 2.0 * math.pi * abs(float(tone.envelope_hz))
        segs.append(ControlSegment(omega * p.duration_s, float(tone.phase_rad), p.duration_s))
        t = p.t_end_s
    return tuple(segs)


def local_slope(omega: np.ndarray, f: np.ndarray) -> float:
    """The log-log slope between the two lowest grid points where F is resolved above round-off."""
    good = np.flatnonzero(f > 1e-28)
    if good.size < 2:
        return float("nan")
    i, j = good[0], good[1]
    return float(math.log(f[j] / f[i]) / math.log(omega[j] / omega[i]))


def filter_function(
    device: Device,
    control: CompositePulse | DecouplingSequence | Schedule,
    *,
    quadrature: Quadrature = "dephasing",
    omega_rad_s: np.ndarray | None = None,
    spectrum: NoiseSpectrum | None = None,
    omega_min_rad_s: float | None = None,
    dc_floor: bool = True,
    monte_carlo_samples: int = 0,
    ion: int = 0,
    rabi_hz: float | None = None,
    seed: int = 0,
    experiment_duration_s: float | None = None,
) -> ExperimentResult:
    """F(omega), chi, W, the first-order and dc-floor infidelities and the roll-off exponent of a control.

    ``spectrum=None`` takes S_b (of half the splitting fluctuation) from the device's S_B for dephasing and
    ``device.noise.rabi_amplitude`` for amplitude. ``omega_min_rad_s`` defaults to the band's lowest frequency if
    positive, else 2 pi/``experiment_duration_s``, else three decades below the sequence; ``monte_carlo_samples``
    cross-checks 1 - F_av with sampled b(t) trajectories through the Hamiltonian builder.
    """
    from qutip_trap.experiments.result import ExperimentResult
    from qutip_trap.noise.model import GAUSS_PER_TESLA

    if isinstance(control, DecouplingSequence):
        segs = control.segments()
    elif isinstance(control, CompositePulse):
        if rabi_hz is None:
            raise ValueError("a composite pulse needs rabi_hz to become a timed control")
        segs = composite_segments(control, 2.0 * math.pi * rabi_hz)
    else:
        segs = schedule_segments(control, ion)
    tau = sum(s.duration_s for s in segs)
    omegas_ctrl = [s.omega_rad_s for s in segs if not s.is_free]
    omega_ctrl = max(omegas_ctrl) if omegas_ctrl else 0.0
    sp = device.crystal.species[ion]
    _f, d1, _d2 = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], device.field.B_gauss)
    conv = 1.0
    if spectrum is None:
        if quadrature == "amplitude":
            spectrum = device.noise.rabi_amplitude
            if spectrum is None:
                raise ValueError("the device has no rabi_amplitude spectrum")
        else:
            spectrum = device.noise.field_spectrum(device)
            if spectrum is None:
                raise ValueError("the device has no magnetic-field spectrum S_B")
            # delta nu = d1 dB (Hz), b = pi delta nu (rad/s): S_b = pi^2 d1^2 S_B (T -> G inside d1)
            conv = (math.pi * d1 * GAUSS_PER_TESLA) ** 2
    spec = spectrum

    def s_b(w: np.ndarray) -> np.ndarray:
        return conv * np.asarray(spec.value(w), dtype=float)

    def f_of(w: np.ndarray) -> np.ndarray:
        if quadrature == "dephasing":
            return dephasing_filter_function(segs, w)
        if quadrature == "amplitude":
            return amplitude_filter_function(segs, w)
        return universal_filter_function(segs, w)

    if omega_min_rad_s is not None:
        w_min = float(omega_min_rad_s)
    else:
        tab = np.abs(np.asarray(spec.omega_rad_s, dtype=float))
        positive = tab[tab > 0.0]
        if float(np.min(tab)) > 0.0 and positive.size:
            w_min = float(np.min(positive))
        elif experiment_duration_s is not None and experiment_duration_s > 0.0:
            w_min = 2.0 * math.pi / float(experiment_duration_s)
        else:
            w_min = 2.0 * math.pi / (1000.0 * tau)
    w_max = 50.0 * max(omega_ctrl, 2.0 * math.pi / tau)
    if spec.omega_max_rad_s > 0.0 and spec.white_level == 0.0:
        w_max = min(max(w_max, 2.0 * math.pi / tau), max(spec.omega_max_rad_s, w_min * 10.0))
    if omega_rad_s is None:
        omega_rad_s = np.geomspace(w_min, w_max, 1200)
    w = np.asarray(omega_rad_s, dtype=float)
    f_vals = f_of(w)
    chi = chi_integral(s_b, f_of, w_min, w_max)
    chi_hi = chi_integral(s_b, f_of, w_min * 1.1, w_max)
    dlnchi = math.log(chi_hi / chi) / math.log(1.1) if chi > 0.0 and chi_hi > 0.0 else 0.0
    a1sq = chi / 2.0
    w_coh = math.exp(-chi)
    # the tabulated band only: white noise is not frozen over the sequence (chi above does include it)
    variance = float(spec.variance()) * conv
    xi2 = tau**2 * variance
    fitted: dict[str, tuple[float, float]] = {
        "chi": (chi, 0.0),
        "W": (w_coh, 0.0),
        "infidelity_ff": (a1sq, 0.0),
        "infidelity_resummed": ((1.0 - w_coh) / 2.0, 0.0),
        "omega_min_rad_s": (w_min, 0.0),
        "dlnchi_dlnomega_min": (dlnchi, 0.0),
        "alpha": ((local_slope(w, f_vals) / 2.0) - 1.0, 0.0),
        "xi2": (xi2, 0.0),
        "tau_s": (tau, 0.0),
    }
    floor = 0.0
    if dc_floor and isinstance(control, CompositePulse) and omega_ctrl > 0.0:
        from qutip_trap.control.composite import leading_coefficient

        channel = "amplitude" if quadrature == "amplitude" else "detuning"
        # the pulse's order holds only for the channels it corrects; against any other the floor is the primitive's
        m = control.order if channel in control.corrects else 0
        c_hat = leading_coefficient(control, channel, 2 * (m + 1))
        # Mount's eps_d scales the whole splitting (eps_d = 2 beta_d/Omega), so the detuning moment is 4 <beta_d^2>
        beta_variance = variance * (4.0 if channel == "detuning" else 1.0)
        floor = dc_floor_value(c_hat, m, beta_variance, omega_ctrl)
        fitted["infidelity_dc"] = (floor, 0.0)
        fitted["dc_c_hat"] = (c_hat, 0.0)
        fitted["dc_order"] = (float(m), 0.0)
    elif dc_floor and variance > 0.0:
        # no c_hat exists for a sequence or schedule: the floor is the exact Gaussian frozen-noise average
        floor = frozen_noise_floor(segs, variance, quadrature)
        fitted["infidelity_dc"] = (floor, 0.0)
    fitted["infidelity"] = (max(a1sq, floor), 0.0)
    if monte_carlo_samples > 0:
        mean, err = _monte_carlo_dephasing(device, ion, segs, spec, conv, tau, int(monte_carlo_samples), seed)
        fitted["infidelity_mc"] = (mean, err)
        fitted["mc_agrees"] = (1.0 if abs(mean - a1sq) <= 3.0 * err + 0.2 * a1sq + xi2**2 else 0.0, 0.0)
    return ExperimentResult(
        data=np.column_stack([w, f_vals]),
        fitted=fitted,
        model=f"filter_function[{quadrature}]",
        provenance_id="conv.filter_function_normalization",
    )


def dc_floor_value(c_hat: float, order: int, beta_variance: float, omega_rad_s: float) -> float:
    return dc_floor(c_hat, order, beta_variance, omega_rad_s)


def _monte_carlo_dephasing(
    device: Device,
    ion: int,
    segs: Sequence[ControlSegment],
    spec: NoiseSpectrum,
    conv: float,
    tau: float,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    """1 - F_av = 1 - (1/4)<|Tr(U_c^dag U)|^2> over sampled b(t) trajectories propagated through the builder."""
    from qutip_trap.control.pulses import Drive, Pulse, Tone
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec, SolverOptions
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.noise.processes import synthesize, time_grid
    from qutip_trap.noise.sampling import NoiseSample, key_qubit_trajectory_hz, quiet_sample

    n_ions = device.crystal.n_ions
    n_modes = len(device.crystal.modes)
    space = HilbertSpace(tuple(2 for _ in range(n_ions)), (), None, tuple(range(n_modes)))
    pulses: list[Pulse] = []
    t = 0.0
    for k, seg in enumerate(segs):
        if not seg.is_free:
            tone = Tone(0.0, seg.phi_rad, seg.omega_rad_s / (2.0 * math.pi))
            drive = Drive("microwave", (ion,), (tone,), (), 0.0, {})
            pulses.append(Pulse(drive, t, t + seg.duration_s, f"ff[{k}]", ()))
        t += seg.duration_s
    idle = tuple(
        (sum(s.duration_s for s in segs[:k]), sum(s.duration_s for s in segs[: k + 1]))
        for k, s in enumerate(segs)
        if s.is_free
    )
    sched = Schedule(tuple(pulses), idle, (), {q: 0.0 for q in range(n_ions)})
    engine = JointExactEngine(store_per_segment=2, hardware_chain=False)
    opts = SolverOptions(atol=1e-12, rtol=1e-10)

    def propagator(sample: NoiseSample) -> np.ndarray:
        cols = []
        for level in (0, 1):
            internal = [0] * n_ions
            internal[ion] = level
            st = space.initial_state(internal)
            tr = engine.run_pulses(device, sched, st, space, sample, SeedSpec(0), opts)
            ket = tr.final.joint
            assert ket is not None
            vec = np.asarray(ket.full()).reshape([2] * n_ions)
            idx = [0] * n_ions
            col = np.array([vec[tuple(idx[:ion] + [lv] + idx[ion + 1 :])] for lv in (0, 1)])
            cols.append(col)
        return np.column_stack(cols)

    u_c = propagator(quiet_sample())
    w_max = spec.omega_max_rad_s if spec.omega_max_rad_s > 0.0 else 2.0 * math.pi / tau
    grid = time_grid(tau, w_max)
    vals = []
    for k in range(n_samples):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))
        traj = synthesize(spec, grid, rng)
        b = math.sqrt(conv) * traj.values  # rad/s, the sigma_z coefficient
        dnu = b / math.pi  # Hz: H_int = pi delta nu sigma_z = b sigma_z
        sample = NoiseSample(0, {}, {key_qubit_trajectory_hz(ion): np.vstack([grid, dnu])})
        u = propagator(sample)
        vals.append(1.0 - abs(np.trace(u_c.conj().T @ u)) ** 2 / 4.0)
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0


__all__ = [
    "ControlSegment",
    "DecouplingSequence",
    "KDD_BLOCK",
    "Quadrature",
    "Timing",
    "accumulated_adjoints",
    "adjoint",
    "amplitude_filter_function",
    "biercuk_amplitude_mp",
    "biercuk_filter_function_mp",
    "biercuk_taylor_coefficients_mp",
    "chi_integral",
    "chi_integral_mp",
    "composite_segments",
    "control_matrix",
    "control_matrix_time",
    "cpmg_centres",
    "cpmg_centres_mp",
    "dc_floor",
    "dc_polygon",
    "decoupling_sequence",
    "dephasing_filter_function",
    "double_factorial_odd",
    "filter_function",
    "final_adjoint",
    "frozen_noise_floor",
    "frozen_noise_infidelity",
    "leading_taylor_order_mp",
    "pulse_adjoint",
    "rotation",
    "schedule_segments",
    "segment_frequency_integrals",
    "udd_centres",
    "udd_centres_mp",
    "universal_filter_function",
]
