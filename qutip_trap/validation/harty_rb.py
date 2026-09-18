"""Harty et al. 2014's microwave randomized benchmarking as the paper simulated it (PLAN.md Sections 4.3.3, 9.2, 9.12).

Protocol (Harty et al., PRL 113, 220501, main text and supplement, read from arXiv:1403.1524): prepare |up>; each
computational gate is a Pauli gate followed by a Clifford gate; Clifford gates are pi/2 rotations about +-x or +-y
(one microwave pi/2 pulse); Pauli gates are pi rotations about +-x, +-y (a PAIR of pi/2 pulses), +-z (an identity
delay followed by a rotation of the logical frame) or +-I (a delay of one pulse duration); the DDS phase is switched
during the 14 us dead time between pulses; the sequence ends with additional pi/2 pulses rotating the qubit into
|down> or |up>, chosen with equal probability, and an error is recorded when the measured state disagrees. The paper's
error model multiplies the propagators of each imperfect pi/2 pulse and each dead time, averages the error over the
sequences and divides by the number of computational gates (EPG); it considers only the longest sequences (2000 gates);
500 random sets of 32 sequences with 12.1 us pulses, 14 us dead times, +4.5 Hz detuning and a constant Rabi frequency
error of 5e-4 gave mean 0.81e-6, sigma 0.14e-6 (their Fig. 7); detuning alone including the -1.0 Hz ac Zeeman shift
(delta_eff = +5.5 Hz) gave 0.7e-6, the Rabi error alone 0.3e-6; measured 1.0(3)e-6.

Propagators in the frame rotating at the DRIVE frequency: a pulse of area theta at azimuth phi under detuning
delta = omega_drive - omega_0 is exp[-i (theta/2)((1 + eps_a) sigma_phi - (delta/Omega) sigma_z)] (the primitive
M(theta, phi; eps_a, eps_d) of Section 4.3.5 with eps_d = -delta/Omega, whose sign is immaterial to the EPG by the
+-z randomization) and a delay tau is exp[+i (delta tau/2) sigma_z]; sigma_z = |up><up| - |down><down|.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class HartyParameters:
    """The operating point of Harty et al. 2014's single-qubit randomized benchmarking (PLAN.md Section 9.2): the pi/2
    time, the dead time between pulses, the identity delay, and the noise levels the error-per-gate sets are simulated at."""

    t_pi2_s: float = 12.1e-6
    dead_time_s: float = 14e-6
    identity_delay_s: float = 12.1e-6
    """The paper's "delays of the same duration as the pi/2 pulses" (printed as 12 us)."""
    detuning_hz: float = 4.5
    """The microwave detuning from the bare qubit transition (+4.5 Hz deliberate)."""
    ac_zeeman_hz: float = 0.0
    """Transition shift present only while the microwaves are on (Harty: -1.0 Hz), so the detuning during a pulse is
    detuning_hz - ac_zeeman_hz = +5.5 Hz (Section 9.2, "ac Zeeman sign"); 0 = the paper's histogram conditions."""
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


def pulse_propagator(
    phi_rad: float,
    params: HartyParameters,
    *,
    detuning_hz: float | None = None,
    rabi_error: float | None = None,
) -> np.ndarray:
    """An imperfect pi/2 pulse at azimuth phi: area (pi/2)(1 + eps_a), detuning (delta - delta_ac)/Omega on sigma_z."""
    delta = TWO_PI * ((params.detuning_hz if detuning_hz is None else detuning_hz) - params.ac_zeeman_hz)
    eps = params.rabi_error if rabi_error is None else rabi_error
    theta = 0.5 * math.pi * (1.0 + eps)
    eps_d = -delta / params.rabi_rad_s
    return rotation(theta, math.cos(phi_rad), math.sin(phi_rad), eps_d)


def delay_propagator(
    tau_s: float, params: HartyParameters, *, detuning_hz: float | None = None
) -> np.ndarray:
    delta = TWO_PI * (params.detuning_hz if detuning_hz is None else detuning_hz)
    return rotation(-delta * tau_s, 0.0, 0.0, 1.0)


# Pauli codes 0..7: +x, -x, +y, -y (pairs of pi/2 pulses), +z, -z (frame rotations), +I, -I (delays)
# Clifford codes 0..3: pi/2 about +x, -x, +y, -y (azimuths 0, pi, pi/2, 3pi/2)
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
    paulis: np.ndarray,
    cliffords: np.ndarray,
    targets: np.ndarray,
    params: HartyParameters,
    *,
    detuning_hz: float | None = None,
    rabi_error: float | None = None,
) -> np.ndarray:
    """The error probability of each sequence: 1 - |<target| U_actual |up>|^2 with the ideal frame tracked alongside."""
    n_seq, n_gates = paulis.shape
    delta = params.detuning_hz if detuning_hz is None else detuning_hz
    eps = params.rabi_error if rabi_error is None else rabi_error
    dead = delay_propagator(params.dead_time_s, params, detuning_hz=delta)
    idle = delay_propagator(params.identity_delay_s, params, detuning_hz=delta)
    one_slot = dead @ idle if params.dead_time_after_delay else idle
    idle_block = _I2.copy()
    for _ in range(params.identity_slots):
        idle_block = one_slot @ idle_block
    # imperfect and ideal pulses for the four azimuths, and for azimuths shifted by the frame (0 or pi): frame offsets are
    # multiples of pi, so the physical azimuth set is the same four; precompute "pulse then dead time" for each azimuth
    actual = {
        k: dead @ pulse_propagator(CLIFFORD_AZIMUTH[k], params, detuning_hz=delta, rabi_error=eps)
        for k in range(4)
    }
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
    """The EPG of ``n_sets`` random sets of ``n_sequences`` sequences of ``params.n_gates`` gates (Harty's Fig. 7 histogram)."""
    p = params or HartyParameters()
    rng = np.random.default_rng(seed)
    out = np.zeros(n_sets)
    for k in range(n_sets):
        paulis, cliffords, targets = random_sequences(rng, n_sequences, p.n_gates)
        errs = simulate_sequences(
            paulis, cliffords, targets, p, detuning_hz=detuning_hz, rabi_error=rabi_error
        )
        out[k] = error_per_gate(errs, p.n_gates)
    return out


__all__ = [
    "CLIFFORD_AZIMUTH",
    "HartyParameters",
    "delay_propagator",
    "error_per_gate",
    "pulse_propagator",
    "random_sequences",
    "rotation",
    "simulate_epg_sets",
    "simulate_sequences",
]
