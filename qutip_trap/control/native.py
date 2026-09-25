"""Native gates, exactly (PLAN.md Sections 7.1, 7.6; Section 13 rows "Gate parameters", "Virtual-Z propagation").

These matrices are the compiler's definition of what a native gate is supposed to do and the validation
suite's reference; the simulator never applies them to a state (Section 3.1). Every function takes RADIANS;
IonQ's turns (1 turn = 2 pi) are converted with :func:`rad_from_turns` at the boundary, phase-type parameters
entering as e^{+-2 pi i phi} and angle-type parameters as cos(pi theta), sin(pi theta) (Section 7.6).

IonQ notation (verified against the vendor documentation, Section 7.6):

- GPi(phi)  = [[0, e^{-i phi}], [e^{i phi}, 0]]
- GPi2(phi) = (1/sqrt 2)[[1, -i e^{-i phi}], [-i e^{i phi}, 1]] = exp[-i (pi/4)(cos phi X + sin phi Y)]
- MS(phi0, phi1, theta) = exp[-i (theta/2) GPi(phi0) (x) GPi(phi1)], theta in radians = 2 pi turns, fully
  entangling at theta = pi/2 (0.25 turns)
- ZZ(theta) = exp(-i (theta/2) Z (x) Z) = diag(e^{-i theta/2}, e^{i theta/2}, e^{i theta/2}, e^{-i theta/2})
- RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2}), virtual: phi -> phi - theta on every later gate phase, gates
  read in time order (the documentation's +theta holds in matrix order).

Research notation: R_phi(theta) = exp[-i (theta/2)(cos phi sigma_x + sin phi sigma_y)], XX(chi) = exp(-i chi
sigma_x (x) sigma_x) with chi = pi/4 maximally entangling; XX(chi) = MS(0, 0, 2 chi).

Tensor order (the one statement of it; ``tests/test_docs.py`` checks that no docstring or page restates it): in every
two-qubit matrix here the gate's first qubit is the left Kronecker factor, the most-significant index bit, so
MS(phi0, phi1, theta) has GPi(phi0) on the left factor and the basis reads |q_first q_second>. Qiskit's little-endian
convention puts a gate's first qubit on the RIGHT factor, so the matrix of ``qiskit_ionq.MSGate(phi0, phi1, theta)``
(turns) is the SWAP conjugate of :func:`ms`, ms = SWAP @ M_qiskit @ SWAP (verified against qiskit-ionq 1.1.1 on
2026-09-11; ``tests/test_docs.py`` repeats the check whenever qiskit-ionq is installed).
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
    """GPi(phi): a pi rotation about the equatorial axis at azimuth phi (Section 7.1)."""
    return np.array([[0.0, np.exp(-1j * phi_rad)], [np.exp(1j * phi_rad), 0.0]], dtype=complex)


def gpi2(phi_rad: float) -> np.ndarray:
    """GPi2(phi): a pi/2 rotation about the same axis."""
    return np.array(
        [[1.0, -1j * np.exp(-1j * phi_rad)], [-1j * np.exp(1j * phi_rad), 1.0]], dtype=complex
    ) / math.sqrt(2.0)


def r_phi(theta_rad: float, phi_rad: float) -> np.ndarray:
    """R_phi(theta) = exp[-i (theta/2)(cos phi X + sin phi Y)] (research notation, Section 7.6)."""
    c = math.cos(theta_rad / 2.0)
    s = math.sin(theta_rad / 2.0)
    return np.array(
        [[c, -1j * s * np.exp(-1j * phi_rad)], [-1j * s * np.exp(1j * phi_rad), c]], dtype=complex
    )


def rz(theta_rad: float) -> np.ndarray:
    """RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2}); IonQ's VirtualZ(theta_turns) = RZ(2 pi theta_turns)."""
    return np.diag([np.exp(-0.5j * theta_rad), np.exp(0.5j * theta_rad)]).astype(complex)


def xx(chi_rad: float) -> np.ndarray:
    """XX(chi) = exp(-i chi sigma_x (x) sigma_x); chi = pi/4 is maximally entangling (Section 13)."""
    c = math.cos(chi_rad)
    s = math.sin(chi_rad)
    return np.array(
        [[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]], dtype=complex
    )


def ms(phi0_rad: float, phi1_rad: float, theta_rad: float) -> np.ndarray:
    """MS(phi0, phi1, theta) = exp[-i (theta/2) GPi(phi0) (x) GPi(phi1)]; theta = pi/2 is fully entangling.

    G = GPi(phi0) (x) GPi(phi1) is Hermitian with G^2 = 1, so the exponential is cos(theta/2) 1 - i sin(theta/2) G.
    """
    g = np.kron(gpi(phi0_rad), gpi(phi1_rad))
    return math.cos(theta_rad / 2.0) * np.eye(4, dtype=complex) - 1j * math.sin(theta_rad / 2.0) * g


def zz(theta_rad: float) -> np.ndarray:
    """ZZ(theta) = exp(-i (theta/2) Z (x) Z) = diag(e^{-i theta/2}, e^{i theta/2}, e^{i theta/2}, e^{-i theta/2})."""
    h = 0.5 * theta_rad
    return np.diag([np.exp(-1j * h), np.exp(1j * h), np.exp(1j * h), np.exp(-1j * h)]).astype(complex)


def virtual_z_frame_shift(phi_rad: float, theta_rad: float) -> float:
    """The frame update of RZ(theta) on every later pulse phase, gates in time order: phi -> phi - theta (Section 7.6)."""
    return phi_rad - theta_rad


def equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-12) -> bool:
    """True when a = e^{i alpha} b for some alpha (the compiler's verification criterion, Section 7.2)."""
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
