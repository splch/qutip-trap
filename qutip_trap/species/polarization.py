"""Polarization in the atomic frame (PLAN.md Section 4.5.3).

The quantization axis is B_hat. A laboratory Jones vector eps (unit norm, transverse to k) is decomposed as
eps = sum_q eps_q e_q with e_{+-1} = -+(x' +- i y')/sqrt 2 and e_0 = z' = B_hat, eps_q = conj(e_q) . eps; q = +1, 0, -1 is
the HELICITY label, the component eps_q driving |F m_F> -> |F' m_F + q>. Steck's operator index in <lower|d_q|upper> is
q_op = m_lower - m_upper = -q_gamma; ``AtomicStructure.dipole_element_c_m`` makes that conversion.

x' is the laboratory x axis projected perpendicular to B_hat (the laboratory y axis when B is along x), which fixes the
azimuthal phase of the sigma components: magnitudes |eps_q| and every rate are independent of that choice, the phases of
Raman couplings between different m states are not, and the drive phase absorbs them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

type Vec = Sequence[float] | np.ndarray
type CVec = Sequence[complex] | np.ndarray


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isclose(n, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{what} must be a unit vector (norm {n})")
    return v / n


def atomic_frame(b_hat: Vec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal (x', y', z') with z' = B_hat and x' the laboratory x projected out of B_hat."""
    z = _unit(np.asarray(b_hat, dtype=float), "B_hat")
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, z))) > 1.0 - 1e-9:
        seed = np.array([0.0, 1.0, 0.0])
    x = seed - np.dot(seed, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def spherical_basis(b_hat: Vec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(e_{-1}, e_0, e_{+1}) about B_hat: e_{+-1} = -+(x' +- i y')/sqrt 2, e_0 = z'."""
    x, y, z = atomic_frame(b_hat)
    e_plus = -(x + 1j * y) / math.sqrt(2.0)
    e_minus = (x - 1j * y) / math.sqrt(2.0)
    return e_minus, z.astype(complex), e_plus


def spherical_components(polarization_lab: CVec, b_hat: Vec) -> np.ndarray:
    """[eps_{-1}, eps_0, eps_{+1}] (index q + 1) of a laboratory Jones vector about B_hat; asserts sum |eps_q|^2 = 1."""
    eps = np.asarray(polarization_lab, dtype=complex)
    norm2 = float(np.vdot(eps, eps).real)
    if not math.isclose(norm2, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"polarization must be unit-normalized (|eps|^2 = {norm2})")
    comps = np.array([np.vdot(e, eps) for e in spherical_basis(b_hat)])
    total = float(np.sum(np.abs(comps) ** 2))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"|eps_-1|^2 + |eps_0|^2 + |eps_+1|^2 = {total} != 1")
    return comps


def to_atomic_frame(v_lab: CVec, b_hat: Vec) -> np.ndarray:
    """Cartesian components (x', y', z') of a laboratory vector in the B_hat frame."""
    x, y, z = atomic_frame(b_hat)
    v = np.asarray(v_lab, dtype=complex)
    return np.array([np.dot(x, v), np.dot(y, v), np.dot(z, v)])


def linear_polarization(k_hat: Vec, angle_to_b_rad: float, b_hat: Vec) -> np.ndarray:
    """A real unit polarization transverse to k_hat at an angle to B_hat's transverse projection: 0 is along it (pure pi
    when k is perpendicular to B), pi/2 perpendicular to it (equal sigma+ and sigma- components)."""
    k = _unit(np.asarray(k_hat, dtype=float), "k_hat")
    b = _unit(np.asarray(b_hat, dtype=float), "B_hat")
    b_perp = b - np.dot(b, k) * k
    if np.linalg.norm(b_perp) < 1e-12:
        raise ValueError(
            "k_hat is parallel to B_hat: a beam along B carries no pi component and the angle is undefined"
        )
    u = b_perp / np.linalg.norm(b_perp)
    w = np.cross(k, u)
    return np.asarray(math.cos(angle_to_b_rad) * u + math.sin(angle_to_b_rad) * w, dtype=float)
