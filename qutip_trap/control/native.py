"""Exact native gate matrices (radians); in every two-qubit matrix the gate's first qubit is the left Kronecker factor.

That factor is the most-significant index bit, so the basis reads |q_first q_second>. qiskit-ionq's ``MSGate`` puts a
gate's first qubit on the right factor instead: ms = SWAP @ M_qiskit @ SWAP. IonQ turns (1 turn = 2 pi) convert with
:func:`rad_from_turns`.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

TURN_RAD: Final[float] = 2.0 * math.pi

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def rad_from_turns(turns: float) -> float:
    """IonQ turns -> radians (1 turn = 2 pi)."""
    return TURN_RAD * turns


def turns_from_rad(rad: float) -> float:
    """Radians -> IonQ turns."""
    return rad / TURN_RAD


def gpi(phi_rad: float) -> np.ndarray:
    """GPi(phi) = [[0, e^{-i phi}], [e^{i phi}, 0]], a pi rotation about the equatorial axis at azimuth phi."""
    return np.array([[0.0, np.exp(-1j * phi_rad)], [np.exp(1j * phi_rad), 0.0]], dtype=complex)


def gpi2(phi_rad: float) -> np.ndarray:
    """GPi2(phi) = (1/sqrt 2)[[1, -i e^{-i phi}], [-i e^{i phi}, 1]] = exp[-i (pi/4)(cos phi X + sin phi Y)]."""
    return np.array(
        [[1.0, -1j * np.exp(-1j * phi_rad)], [-1j * np.exp(1j * phi_rad), 1.0]], dtype=complex
    ) / math.sqrt(2.0)


def r_phi(theta_rad: float, phi_rad: float) -> np.ndarray:
    """R_phi(theta) = exp[-i (theta/2)(cos phi X + sin phi Y)]."""
    c = math.cos(theta_rad / 2.0)
    s = math.sin(theta_rad / 2.0)
    return np.array(
        [[c, -1j * s * np.exp(-1j * phi_rad)], [-1j * s * np.exp(1j * phi_rad), c]], dtype=complex
    )


def rz(theta_rad: float) -> np.ndarray:
    """RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2}); IonQ's VirtualZ(theta_turns) = RZ(2 pi theta_turns)."""
    return np.diag([np.exp(-0.5j * theta_rad), np.exp(0.5j * theta_rad)]).astype(complex)


def xx(chi_rad: float) -> np.ndarray:
    """XX(chi) = exp(-i chi sigma_x (x) sigma_x) = MS(0, 0, 2 chi); chi = pi/4 is maximally entangling."""
    c = math.cos(chi_rad)
    s = math.sin(chi_rad)
    return np.array(
        [[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]], dtype=complex
    )


def ms(phi0_rad: float, phi1_rad: float, theta_rad: float) -> np.ndarray:
    """MS(phi0, phi1, theta) = exp[-i (theta/2) GPi(phi0) (x) GPi(phi1)]; theta = pi/2 is fully entangling."""
    g = np.kron(gpi(phi0_rad), gpi(phi1_rad))
    return math.cos(theta_rad / 2.0) * np.eye(4, dtype=complex) - 1j * math.sin(theta_rad / 2.0) * g


def zz(theta_rad: float) -> np.ndarray:
    """ZZ(theta) = exp(-i (theta/2) Z (x) Z) = diag(e^{-i theta/2}, e^{i theta/2}, e^{i theta/2}, e^{-i theta/2})."""
    h = 0.5 * theta_rad
    return np.diag([np.exp(-1j * h), np.exp(1j * h), np.exp(1j * h), np.exp(-1j * h)]).astype(complex)


def virtual_z_frame_shift(phi_rad: float, theta_rad: float) -> float:
    """The frame update of RZ(theta) on every later pulse phase, gates in time order: phi -> phi - theta (IonQ's
    documented phi -> phi + theta reads the gates in matrix order)."""
    return phi_rad - theta_rad


def equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-12) -> bool:
    """True when a = e^{i alpha} b for some real alpha, within ``atol``."""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.shape != b.shape:
        return False
    idx = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
    if abs(b[idx]) < atol:
        return bool(np.allclose(a, b, atol=atol))
    phase = a[idx] / b[idx]
    if not math.isclose(abs(phase), 1.0, rel_tol=0.0, abs_tol=atol):
        return False
    return bool(np.allclose(a, phase * b, atol=atol))


PAULI_X: Final[np.ndarray] = _X
PAULI_Y: Final[np.ndarray] = _Y
PAULI_Z: Final[np.ndarray] = _Z
IDENTITY_2: Final[np.ndarray] = _I2

__all__ = [
    "IDENTITY_2",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "TURN_RAD",
    "equal_up_to_global_phase",
    "gpi",
    "gpi2",
    "ms",
    "r_phi",
    "rad_from_turns",
    "rz",
    "turns_from_rad",
    "virtual_z_frame_shift",
    "xx",
    "zz",
]
