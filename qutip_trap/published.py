"""Closed forms from the published experiments the simulator is compared with: the microwave randomized-benchmarking error
model of Harty et al. (2014) and the Molmer-Sorensen closed forms of Kirchmair et al. (2009), Roos (2008) and Ballance et
al. (2016). Frequencies are angular (rad/s), the per-tone (hbar Omega/2) convention holds and S_alpha = sum_i sigma_alpha^i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.special import jv

from qutip_trap.units import TWO_PI


def ms_alpha(eta: float, omega_rad_s: float, epsilon_rad_s: float, t_s: float) -> complex:
    """alpha(t) = (eta Omega/(2 eps))(e^{i eps t} - 1): each S_y eigenstate m moves on a circle of radius |m| eta Omega/(2 eps) (Kirchmair Eq. 4)."""
    return complex(eta * omega_rad_s / (2.0 * epsilon_rad_s) * (np.exp(1j * epsilon_rad_s * t_s) - 1.0))


def ms_lambda_rad_s(eta: float, omega_rad_s: float, epsilon_rad_s: float) -> float:
    """lambda = eta^2 Omega^2/(4 eps): the secular rate of the S_y^2 phase."""
    return eta**2 * omega_rad_s**2 / (4.0 * epsilon_rad_s)


def ms_chi(eta: float, omega_rad_s: float, epsilon_rad_s: float) -> float:
    """chi = eta^2 Omega^2/(4 eps^2): the oscillating part of the S_y^2 phase (Kirchmair's chi, not the entangling angle)."""
    return eta**2 * omega_rad_s**2 / (4.0 * epsilon_rad_s**2)


def ms_gamma(eta: float, omega_rad_s: float, epsilon_rad_s: float, t_s: float) -> float:
    """gamma(t) = lambda t - chi sin(eps t): the coefficient of S_y^2 in the exact propagator D(alpha S_y) exp[i gamma S_y^2]."""
    return ms_lambda_rad_s(eta, omega_rad_s, epsilon_rad_s) * t_s - ms_chi(
        eta, omega_rad_s, epsilon_rad_s
    ) * math.sin(epsilon_rad_s * t_s)


def kirchmair_populations(alpha_abs: float, gamma: float, nbar: float) -> tuple[float, float, float]:
    """(p_0, p_1, p_2) populations with zero, one and two ions BRIGHT from |dd> with a thermal mode (Kirchmair 2009 Eq. 14):
    p_2 = (1/8)(3 + e^{-16|a|^2(n+1/2)} + 4 cos(4 gamma) e^{-4|a|^2(n+1/2)}), p_1 = (1/4)(1 - e^{-16|a|^2(n+1/2)}).

    Bright is the fluorescing S1/2 state, the LOWER qubit level |d> of 40Ca+: p_2 = P(dd) = P_00 in the computational ordering,
    p_0 = P(uu) = P_11; at t = 0 the formula gives p_2 = 1."""
    x = alpha_abs**2 * (nbar + 0.5)
    p2 = (3.0 + math.exp(-16.0 * x) + 4.0 * math.cos(4.0 * gamma) * math.exp(-4.0 * x)) / 8.0
    p1 = (1.0 - math.exp(-16.0 * x)) / 4.0
    return 1.0 - p1 - p2, p1, p2


def roos_force_saturation(omega_tone_rad_s: float, delta_rad_s: float) -> float:
    """J_0(x) + J_2(x) at x = 2 Omega/delta for the per-tone Omega of this module and delta the tones' detuning from the
    carrier: the carrier's saturation of the spin-dependent force (Roos 2008 Eq. 17). Roos writes the argument 4 Omega_R/delta
    with his Omega_R = Omega/2 (``conv.ms_closure``, Section 4.4.1)."""
    x = 2.0 * omega_tone_rad_s / delta_rad_s
    return float(jv(0, x) + jv(2, x))


ThermalReference = Literal["mean", "n0", "minus_half"]
"""Which occupation the thermal Debye-Waller infidelity is referred to: the mean nbar, the ground state, or nbar - 1/2
(the three conventions of the two-qubit gate literature)."""


def thermal_debye_waller_infidelity(eta: float, nbar: float, reference: ThermalReference) -> float:
    """(pi^2/4) eta^4 <(n - n_ref)^2> over the thermal distribution: n_ref = nbar (Sorensen-Molmer, re-optimized duration:
    nbar^2 + nbar), 0 (Ballance, calibrated at n = 0: 2 nbar^2 + nbar), -1/2 (Zhu, referenced to eta^2 (2n + 1) = 0:
    2 nbar^2 + 2 nbar + 1/4), in units of (pi^2/4) eta^4."""
    pref = (math.pi**2 / 4.0) * eta**4
    var = nbar * (nbar + 1.0)
    if reference == "mean":
        return pref * var
    if reference == "n0":
        return pref * (var + nbar**2)
    if reference == "minus_half":
        return pref * (var + (nbar + 0.5) ** 2)
    raise ValueError("reference is 'mean', 'n0' or 'minus_half'")


def ballance_thermal_error(eta: float, nbar: float) -> float:
    """eps_nbar = (1/4) pi^2 eta^4 nbar (2 nbar + 1) = (pi^2/4) eta^4 <n^2>, calibrated at n = 0 (Ballance 2016 supplement)."""
    return 0.25 * math.pi**2 * eta**4 * nbar * (2.0 * nbar + 1.0)


@dataclass(frozen=True)
class HartyParameters:
    """The operating point of Harty et al.'s benchmarking: the pi/2 time, the dead time between pulses, the identity
    delay, and the noise levels the error-per-gate sets are simulated at."""

    t_pi2_s: float = 12.1e-6
    dead_time_s: float = 14e-6
    identity_delay_s: float = 12.1e-6
    """The paper's "delays of the same duration as the pi/2 pulses" (printed as 12 us)."""
    detuning_hz: float = 4.5
    """The microwave detuning from the bare qubit transition (+4.5 Hz deliberate)."""
    ac_zeeman_hz: float = 0.0
    """Transition shift present only while the microwaves are on (Harty: -1.0 Hz), so the detuning during a pulse is
    detuning_hz - ac_zeeman_hz = +5.5 Hz; 0 is the paper's histogram condition."""
    rabi_error: float = 5e-4
    """Constant fractional Rabi-frequency error at fixed pulse duration: theta = (pi/2)(1 + rabi_error)."""
    n_gates: int = 2000
    identity_slots: int = 2
    """Delays replacing the pulses of an identity or z Pauli: one per pi/2 pulse of the pair it stands in for."""
    dead_time_after_delay: bool = True
    """Whether each identity delay is followed by a phase-switching dead time like a pulse."""

    @property
    def rabi_rad_s(self) -> float:
        return 0.5 * math.pi / self.t_pi2_s


_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)

_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)  # |up> = index 0 here (the paper prepares |up>)

_I2 = np.eye(2, dtype=complex)


def rotation(theta: float, nx: float, ny: float, nz: float) -> np.ndarray:
    """exp[-i (theta/2) n . sigma] for a unit vector n."""
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return _I2.copy()
    a = theta * norm / 2.0
    n = (nx * _SX + ny * _SY + nz * _SZ) / norm
    return math.cos(a) * _I2 - 1j * math.sin(a) * n


def pulse_propagator(phi_rad: float, params: HartyParameters) -> np.ndarray:
    """An imperfect pi/2 pulse at azimuth phi: area (pi/2)(1 + eps_a), detuning (delta - delta_ac)/Omega on sigma_z."""
    delta = TWO_PI * (params.detuning_hz - params.ac_zeeman_hz)
    theta = 0.5 * math.pi * (1.0 + params.rabi_error)
    eps_d = -delta / params.rabi_rad_s
    return rotation(theta, math.cos(phi_rad), math.sin(phi_rad), eps_d)


def delay_propagator(tau_s: float, params: HartyParameters) -> np.ndarray:
    """A delay tau under the detuning: exp[+i (delta tau/2) sigma_z]."""
    return rotation(-TWO_PI * params.detuning_hz * tau_s, 0.0, 0.0, 1.0)


CLIFFORD_AZIMUTH = (0.0, math.pi, 0.5 * math.pi, 1.5 * math.pi)


def random_sequences(
    rng: np.random.Generator, n_sequences: int, n_gates: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pauli codes, clifford codes) of shape (n_sequences, n_gates) and the final target (0 = |up>, 1 = |down>)."""
    paulis = rng.integers(0, 8, size=(n_sequences, n_gates))
    cliffords = rng.integers(0, 4, size=(n_sequences, n_gates))
    targets = rng.integers(0, 2, size=n_sequences)
    return paulis, cliffords, targets


def _ideal_pulse(phi: float) -> np.ndarray:
    return rotation(0.5 * math.pi, math.cos(phi), math.sin(phi), 0.0)


def simulate_sequences(
    paulis: np.ndarray, cliffords: np.ndarray, targets: np.ndarray, params: HartyParameters
) -> np.ndarray:
    """The error probability of each sequence: 1 - |<target| U_actual |up>|^2 with the ideal frame tracked alongside."""
    n_seq, n_gates = paulis.shape
    dead = delay_propagator(params.dead_time_s, params)
    idle = delay_propagator(params.identity_delay_s, params)
    one_slot = dead @ idle if params.dead_time_after_delay else idle
    idle_block = _I2.copy()
    for _ in range(params.identity_slots):
        idle_block = one_slot @ idle_block
    # imperfect and ideal pulses for the four azimuths, and for azimuths shifted by the frame (0 or pi): frame offsets are
    # multiples of pi, so the physical azimuth set is the same four; precompute "pulse then dead time" for each azimuth
    actual = {k: dead @ pulse_propagator(CLIFFORD_AZIMUTH[k], params) for k in range(4)}
    ideal = {k: _ideal_pulse(CLIFFORD_AZIMUTH[k]) for k in range(4)}
    flip = {0: 1, 1: 0, 2: 3, 3: 2}  # azimuth + pi
    U = np.broadcast_to(_I2, (n_seq, 2, 2)).copy()
    V = np.broadcast_to(_I2, (n_seq, 2, 2)).copy()
    frame = np.zeros(n_seq, dtype=int)  # 0 or 1: frame rotated by pi (the +-z Paulis)

    def apply(mask: np.ndarray, op_actual: np.ndarray, op_ideal: np.ndarray | None) -> None:
        if not np.any(mask):
            return
        U[mask] = op_actual @ U[mask]
        if op_ideal is not None:
            V[mask] = op_ideal @ V[mask]

    for g in range(n_gates):
        p = paulis[:, g]
        c = cliffords[:, g]
        # Pauli: +-x, +-y as two pi/2 pulses at the (frame-shifted) azimuth
        for code in range(4):
            for fr in (0, 1):
                mask = (p == code) & (frame == fr)
                k = code if fr == 0 else flip[code]
                if np.any(mask):
                    step = actual[k] @ actual[k]
                    step_i = ideal[k] @ ideal[k]
                    apply(mask, step, step_i)
        # +-z: idle delay then frame rotation by pi (ideal: a z rotation, which is the frame change itself)
        zmask = (p == 4) | (p == 5)
        apply(zmask, idle_block, None)
        frame[zmask] ^= 1
        # +-I: idle delay
        imask = (p == 6) | (p == 7)
        apply(imask, idle_block, None)
        # Clifford: one pi/2 pulse at the frame-shifted azimuth
        for code in range(4):
            for fr in (0, 1):
                mask = (c == code) & (frame == fr)
                k = code if fr == 0 else flip[code]
                apply(mask, actual[k], ideal[k])
    # the ideal state after the sequence, up to the frame (a z rotation, irrelevant for a z-basis target choice)
    up = np.array([1.0, 0.0], dtype=complex)
    psi_ideal = np.einsum("sij,j->si", V, up)
    # final rotation into the basis: bring the ideal Bloch vector to the target pole with 0, 1 or 2 ideal pi/2 pulses
    z = np.real(np.einsum("si,ij,sj->s", np.conj(psi_ideal), _SZ, psi_ideal))
    target_sign = np.where(targets == 0, 1.0, -1.0)  # +1 for |up> (z = +1), -1 for |down>
    errors = np.zeros(n_seq)
    for s in range(n_seq):
        ops: list[int] = []
        if abs(z[s]) > 0.5:  # at a pole
            if z[s] * target_sign[s] < 0:  # flip with two pi/2 pulses about x
                ops = [0, 0]
        else:
            # rotate the equatorial state to the target pole with one pi/2 pulse: about +y takes +x to +z ... choose by trial
            best = None
            for k in range(4):
                v = ideal[k] @ psi_ideal[s]
                zz = float(np.real(np.vdot(v, _SZ @ v)))
                if zz * target_sign[s] > 0.99:
                    best = k
                    break
            if best is None:
                raise RuntimeError("no single pi/2 pulse maps the ideal state to the target pole")
            ops = [best]
        u = U[s]
        for k in ops:
            # ``best`` was chosen among PHYSICAL azimuths (V already carries the frame flips), so no further flip here
            u = actual[k] @ u
        final = u @ up
        target_state = np.array([1.0, 0.0]) if targets[s] == 0 else np.array([0.0, 1.0])
        errors[s] = 1.0 - abs(np.vdot(target_state, final)) ** 2
    return errors


def error_per_gate(errors: np.ndarray, n_gates: int) -> float:
    return float(np.mean(errors) / n_gates)


def simulate_epg_sets(
    n_sets: int,
    n_sequences: int = 32,
    params: HartyParameters | None = None,
    *,
    seed: int = 0,
    detuning_hz: float | None = None,
    rabi_error: float | None = None,
) -> np.ndarray:
    """The EPG of ``n_sets`` random sets of ``n_sequences`` sequences of ``params.n_gates`` gates (Harty's Fig. 7
    histogram); ``detuning_hz`` and ``rabi_error`` override the parameters' own."""
    p = params or HartyParameters()
    if detuning_hz is not None:
        p = replace(p, detuning_hz=detuning_hz)
    if rabi_error is not None:
        p = replace(p, rabi_error=rabi_error)
    rng = np.random.default_rng(seed)
    out = np.zeros(n_sets)
    for k in range(n_sets):
        paulis, cliffords, targets = random_sequences(rng, n_sequences, p.n_gates)
        out[k] = error_per_gate(simulate_sequences(paulis, cliffords, targets, p), p.n_gates)
    return out
