"""Dynamical decoupling and the filter-function description of dephasing and amplitude noise (PLAN.md Section 6.9; Section 13
rows "Filter-function normalization stack", "Dephasing noise variable", "Coherence and decay exponent", "Dynamical-decoupling
filter function", "dc frozen-noise coefficient", "Toggling-frame control matrix"; Appendix E ``filter_function``; M7).

The normalization stack, pinned together and asserted against ``check_composite.py`` sections 9 to 15:

- Toggling frame: H(t) = H_c(t) + beta(t) . sigma with NO 1/2 on the Pauli vector, U(t) = U_c(t) U~(t), and the control
  matrix R_ij(t) = (1/2) Tr[U_c^dag(t) sigma_i U_c(t) sigma_j], a real SO(3) element (the adjoint of U_c^dag). Under
  piecewise-constant control R(t) = R^{P_l}(t - t_{l-1}) Lambda^{(l-1)} on segment l, Lambda^{(l-1)} the same trace formula
  on the propagator accumulated BEFORE the segment, right-multiplying. For a segment rotating about the equatorial axis
  n(phi_l) = (cos phi_l, sin phi_l, 0) at Omega_l, R^{P}(s) = cos(Omega s) 1 + (1 - cos Omega s) n n^T + sin(Omega s) [n]_x,
  whose third row for phi = 0 is (0, sin Omega s, cos Omega s) as the check script requires; free evolution and a pure z
  rotation are the one case theta_l = 0 AND Omega_l = 0 (R^P = 1, the z rotation carried by Lambda).
- Frequency domain with the explicit -i omega: R_ij(omega) = -i omega int_0^tau dt R_ij(t) e^{+i omega t}, evaluated
  segment by segment in closed form (the integrals of cos, sin and 1 against e^{i omega s}, with removable poles at
  omega = +-Omega_l guarded). The dephasing filter function is F_z(omega) = sum_i |R_zi(omega)|^2 and the first-order
  infidelity 1 - F_av = (1/pi) int_0^inf (d omega/omega^2) S_b F_z (one-sided prefactor 1/pi, weight 1/omega^2 paired with
  the -i omega); for a pure dephasing S_b this reproduces Biercuk's F(omega tau) = |1 + (-1)^{n+1} e^{i omega tau} +
  2 cos(omega tau_pi/2) sum_j (-1)^j e^{i delta_j omega tau}|^2 for single-axis pi_X trains at both parities.
- b(t) is HALF the qubit-splitting fluctuation: H_deph = b sigma_z, S_b = S_delta/4 for a supplied splitting spectrum
  S_delta; chi = (2/pi) int (d omega/omega^2) S_b F = 2 <a_1^2>, W = e^{-chi}, F_av = (1 + W)/2; the module reports the
  linear 1 - F_av = <a_1^2> = chi/2 inside xi^2 << 1 and the resummed (1 - W)/2 beside it.
- Amplitude quadrature: F_a(omega) = (1/4){|sum_l A_l rho~_a^{(l)}|^2 + |sum_l B_l rho~_a^{(l)}|^2}, rho~_a^{(l)} =
  rho(phi_l) Lambda^{(l-1)}, A_l = cos omega t_l - cos omega t_{l-1}, B_l = sin omega t_l - sin omega t_{l-1}; the dc
  cancellation of a composite pulse is the closed polygon sum_l theta_l rho~_a^{(l)} = 0.
- dc floor: for an mth-order sequence under frozen noise 1 - F ~ c^_{m+1} <(beta/Omega)^{2(m+1)}>, c^ the dimensionless
  leading coefficient of ``control/composite.py`` paired ONLY with the relative error, Gaussian moments (2m + 1)!! <beta^2>^{m+1},
  <beta^2> = (1/pi) x the one-sided power; the reported estimate is max[(1 - F)_FF, (1 - F)_dc].
- omega_min is always reported with d ln chi/d ln omega_min beside chi: chi is infrared-divergent for free induction, the
  Hahn echo and every odd-n sequence on a 1/omega^4 spectrum (Section 12).

Multi-axis sequences (XY4, XY8, KDD, CDD, (XY)^N) are built by composing per-pulse blocks with the accumulated
Lambda^{(l-1)} matrices, never with the scalar (-1)^l, and are tagged **[background]**: no source validates them; the
single-axis closed forms above are what the machinery must reproduce to quadrature precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from qutip_trap.control.composite import CompositePulse

if TYPE_CHECKING:
    from qutip_trap.control.schedule import Schedule
    from qutip_trap.device.model import Device
    from qutip_trap.experiments import ExperimentResult
    from qutip_trap.noise.spectra import NoiseSpectrum

M7 = "milestone M7 (noise/decoupling.py, PLAN.md Section 6.9)"

Timing = Literal["hahn", "cpmg", "udd", "xy4", "xy8", "kdd", "cdd", "custom"]
Quadrature = Literal["dephasing", "amplitude", "universal"]

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = (_SX, _SY, _SZ)


# ---- control segments ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlSegment:
    """One piecewise-constant control interval: rotation angle theta about the equatorial axis at azimuth phi over tau."""

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
    """exp(-i theta sigma_phi/2), sigma_phi = cos phi X + sin phi Y (Section 4.3.5's primitive)."""
    s = math.cos(phi) * _SX + math.sin(phi) * _SY
    return math.cos(theta / 2.0) * np.eye(2, dtype=complex) - 1j * math.sin(theta / 2.0) * s


def adjoint(u: np.ndarray) -> np.ndarray:
    """Lambda_ij = (1/2) Tr[U^dag sigma_i U sigma_j]: the SO(3) matrix of Section 13 (the adjoint of U^dag)."""
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
    """M_l(omega) = int_0^tau_l R^{P_l}(s) e^{i omega (t_{l-1} + s)} ds, shape (len(omega), 3, 3) complex.

    ``gated``: the noise coupling is switched off while a pulse is on (Biercuk 2009's finite-pulse idealization, whose whole
    effect is the cos(omega tau_pi/2) factor), so a driven segment contributes nothing to the integral; the default keeps the
    noise on through the pulse (Green 2013's finite-width treatment, the physical case).
    """
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
    """R(t) by the trace formula on the actual propagator (the reference the closed forms are checked against)."""
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
    """F_z^{(1)}(omega) = sum_i |R_zi(omega)|^2 (all three components, Section 6.9)."""
    r = control_matrix(segments, omega, gated=gated)
    return np.asarray(np.sum(np.abs(r[:, 2, :]) ** 2, axis=1))


def amplitude_filter_function(segments: Sequence[ControlSegment], omega: np.ndarray) -> np.ndarray:
    """F_a(omega) = (1/4){|sum_l A_l rho~^{(l)}|^2 + |sum_l B_l rho~^{(l)}|^2} over the driven segments."""
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
    """c^_{m+1} (2m + 1)!! (<beta^2>/Omega^2)^{m+1}: the frozen-noise floor of an mth-order sequence (Section 6.9)."""
    rel = beta_variance / omega_rad_s**2
    return c_hat * double_factorial_odd(order) * rel ** (order + 1)


# ---- the sequences -------------------------------------------------------------------------------------------------------


def cpmg_centres(n: int) -> tuple[float, ...]:
    return tuple((2.0 * j - 1.0) / (2.0 * n) for j in range(1, n + 1))


def udd_centres(n: int) -> tuple[float, ...]:
    return tuple(math.sin(math.pi * j / (2.0 * n + 2.0)) ** 2 for j in range(1, n + 1))


def _xy_axes(n: int, pattern: Sequence[float]) -> tuple[float, ...]:
    if n % len(pattern) != 0:
        raise ValueError(f"the pulse count must be a multiple of {len(pattern)} for this axis pattern")
    return tuple(pattern[j % len(pattern)] for j in range(n))


KDD_BLOCK = (math.pi / 6.0, 0.0, math.pi / 2.0, 0.0, math.pi / 6.0)
"""Souza, Alvarez and Suter 2011: the five-pulse Knill block replacing each pi pulse, its phases relative to the block axis.
One block is a pi rotation about the equatorial axis at -60 degrees (checked by ``pulse_adjoint``), two blocks with the X and
Y bases a pi rotation about z, and the XY4 cycle of four blocks (20 pulses) the identity: ``decoupling_sequence`` needs a
multiple of 20 pulses."""


def _cdd_tokens(level: int) -> list[str]:
    """CDD_l = CDD_{l-1} X CDD_{l-1} Y CDD_{l-1} X CDD_{l-1} Y with CDD_0 = a free interval."""
    if level == 0:
        return ["f"]
    inner = _cdd_tokens(level - 1)
    return inner + ["X"] + inner + ["Y"] + inner + ["X"] + inner + ["Y"]


@dataclass(frozen=True)
class DecouplingSequence:
    """A net-identity pulse train, or a decoupled idle (Section 6.9)."""

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
        """If False the scalar (-1)^l bookkeeping is INVALID and the full toggling machinery is needed."""
        return (
            all(math.isclose(a, self.axes_rad[0], abs_tol=1e-12) for a in self.axes_rad)
            if self.axes_rad
            else True
        )

    def feasible(self) -> bool:
        """delta_pi <= 2 sin^2[pi/(2n + 2)] and no pulse overlap (Section 6.9)."""
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
        """(len(omega), 3, 3) complex R_ij(omega) from the full toggling machinery (Section 13, Green Eqs. 25-30)."""
        return control_matrix(self.segments(), np.asarray(omega_rad_s, dtype=float))

    def filter_function(
        self,
        omega_rad_s: np.ndarray,
        quadrature: Quadrature = "dephasing",
        *,
        gated: bool = False,
    ) -> np.ndarray:
        """``gated=True`` switches the noise off during the pulses (Biercuk's idealization, reproduced exactly by
        ``biercuk_filter_function`` for single-axis trains); the default keeps it on (Green's finite-width case)."""
        segs = self.segments()
        w = np.asarray(omega_rad_s, dtype=float)
        if quadrature == "dephasing":
            return dephasing_filter_function(segs, w, gated=gated)
        if quadrature == "amplitude":
            return amplitude_filter_function(segs, w)
        return universal_filter_function(segs, w, gated=gated)

    def biercuk_filter_function(self, omega_rad_s: np.ndarray) -> np.ndarray:
        """Biercuk Eq. 2 for a SINGLE-AXIS pi_X train (both parities): the closed form the machinery must reproduce."""
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


# ---- the Appendix E entry point -------------------------------------------------------------------------------------------------


def composite_segments(pulse: CompositePulse, rabi_rad_s: float) -> tuple[ControlSegment, ...]:
    """A composite pulse at constant Rabi frequency: segment durations theta_l/Omega."""
    if rabi_rad_s <= 0.0:
        raise ValueError("the Rabi frequency must be positive")
    return (
        tuple(
            ControlSegment(area, phase + pulse.phi_rad, area / rabi_rad_s) for area, phase in pulse.segments
        )
        * pulse.n_rep
    )


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
    """F(omega), 1 - F_av = (1/pi) int dw/w^2 S F, chi, W, the roll-off order, the dc floor (Section 6.9).

    ``spectrum=None`` uses the device's S_B converted through the computed Zeeman sensitivities to S_b (the PSD of the
    sigma_z coefficient, HALF the splitting fluctuation) for the dephasing quadrature and ``device.noise.rabi_amplitude`` for
    the amplitude one. ``omega_min_rad_s`` defaults to the spectrum's lowest tabulated frequency when its band starts above zero, else to 2
    pi/``experiment_duration_s`` (the total experiment, Section 12), else to three decades below the sequence; it is always reported
    with d ln chi/d ln omega_min.
    ``monte_carlo_samples > 0`` additionally propagates sampled b(t) trajectories through the Section 4.3.1 Hamiltonian (the
    builder's qubit-trajectory hook on a mode-less single-ion space) and returns the sampled 1 - F_av beside the filter-function
    estimate; the result records xi^2 and flags the comparison when it is not small.
    """
    from qutip_trap.experiments import ExperimentResult
    from qutip_trap.noise.model import GAUSS_PER_TESLA

    # ---- the control -------------------------------------------------------------------------------------------------
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
    # ---- the spectrum: S_b of the sigma_z coefficient, or the additive amplitude noise -------------------------------
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

    # the infrared cutoff (Section 12): the spectrum's own lowest tabulated frequency when the band starts above zero (the
    # device declared it), else 2 pi over the total experiment duration, else three decades below the sequence; always reported
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
    variance = float(spec.variance()) * conv + spec.white_level * conv * 0.0
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
        m = control.order
        c_hat = leading_coefficient(control, channel, 2 * (m + 1))
        floor = dc_floor_value(c_hat, m, variance, omega_ctrl)
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
    """1 - F_av = 1 - (1/4)<|Tr(U_c^dag U)|^2> over sampled b(t) trajectories propagated through the builder (Section 6.9)."""
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
    "KDD_BLOCK",
    "ControlSegment",
    "DecouplingSequence",
    "Quadrature",
    "Timing",
    "accumulated_adjoints",
    "adjoint",
    "amplitude_filter_function",
    "chi_integral",
    "composite_segments",
    "control_matrix",
    "control_matrix_time",
    "cpmg_centres",
    "dc_floor",
    "dc_polygon",
    "decoupling_sequence",
    "dephasing_filter_function",
    "double_factorial_odd",
    "filter_function",
    "final_adjoint",
    "pulse_adjoint",
    "rotation",
    "schedule_segments",
    "segment_frequency_integrals",
    "udd_centres",
    "universal_filter_function",
]
