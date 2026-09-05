"""Electric-quadrupole (E2) coupling for optical qubits (PLAN.md Section 4.5.7; Section 13 E2 row; M0a).

The E2 drive on |S, m> <-> |D, m'> has the dipole form with one beam's k (Section 4.3.1) and the scalar

    Omega_{mm'} = (e E_0 k/(2 hbar)) |<S||r^2 C^(2)||D>| |Lambda(m, m')| |g^(q)|,   q = m - m' = -Delta m,

with Lambda the 3j symbol (J 2 J'; -m q m'), C^(2)_q = sqrt(4 pi/5) Y_{2q} (Racah), and g^(q) = c^(q)_{ij} eps_i n_j
the contraction of the polarization and propagation direction with the rank-2 spherical basis tensors
c^(q) = (2/3) b^(q), normalized to sum_ij |c^(q)_ij|^2 = 2/3 (NOT 1) and defined by b^(q)_ij r_i r_j = r^2 C^(2)_q,
never the conjugate duals (derivation audit 2026-09-04: the duals share the norm but turn q = m - m' into
q = m' - m for circular light). The reduced element comes from the D-state lifetime,
A^(E2) = c alpha k^5 |<S||r^2 C^(2)||D>|^2 / (15 (2j' + 1)) with j' the UPPER level and k the VACUUM wavenumber;
Omega follows the plan's convention (carrier pi pulse at t = pi/Omega), twice the Boulder value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from qutip_trap.species.polarization import CVec, Vec, to_atomic_frame
from qutip_trap.species.wigner import Half, as_half_integer, wigner_3j
from qutip_trap.units import A_0_M, ALPHA_FS, C_M_PER_S, E_C, HBAR_J_S

C_ALPHA_M_PER_S: float = C_M_PER_S * ALPHA_FS
"""c alpha = e^2/(4 pi eps0 hbar) = 2.187691e6 m/s (Section 4.5.7)."""


def _b_tensors() -> dict[int, np.ndarray]:
    s38 = math.sqrt(3.0 / 8.0)
    x, y, z = np.eye(3)

    def sym(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.outer(a, b) + np.outer(b, a)

    b0 = np.diag([-0.5, -0.5, 1.0]).astype(complex)
    b_p1 = -s38 * (sym(x, z) + 1j * sym(y, z))
    b_m1 = +s38 * (sym(x, z) - 1j * sym(y, z))
    b_p2 = s38 * (np.outer(x, x) - np.outer(y, y) + 1j * sym(x, y))
    b_m2 = s38 * (np.outer(x, x) - np.outer(y, y) - 1j * sym(x, y))
    return {0: b0, 1: b_p1, -1: b_m1, 2: b_p2, -2: b_m2}


B_TENSORS: dict[int, np.ndarray] = _b_tensors()
"""b^(q)_ij with b^(q)_ij r_i r_j = r^2 C^(2)_q (James 1998 Appendix; Section 4.5.7)."""

C_TENSORS: dict[int, np.ndarray] = {q: (2.0 / 3.0) * b for q, b in B_TENSORS.items()}
"""c^(q) = (2/3) b^(q), normalized to sum_ij |c^(q)_ij|^2 = 2/3."""


def racah_c2(q: int, r: Sequence[float]) -> complex:
    """r^2 C^(2)_q(theta, phi) = sqrt(4 pi/5) r^2 Y_{2q} evaluated in Cartesian form, for the tensor identity test."""
    x, y, z = (float(v) for v in r)
    r2 = x * x + y * y + z * z
    if q == 0:
        return complex((3.0 * z * z - r2) / 2.0)
    if abs(q) == 1:
        return complex(-q * math.sqrt(3.0 / 2.0) * z * (x + 1j * q * y))
    if abs(q) == 2:
        return complex(math.sqrt(3.0 / 8.0) * (x + 1j * (q // 2) * y) ** 2)
    raise ValueError("q must lie in -2..2")


def geometric_factor(q: int, polarization_lab: CVec, k_hat_lab: Vec, b_hat: Vec) -> complex:
    """g^(q) = c^(q)_ij eps_i n_j with eps and n expressed in the B_hat frame (complex for elliptical light)."""
    eps = to_atomic_frame(polarization_lab, b_hat)
    n = to_atomic_frame(np.asarray(k_hat_lab, dtype=float), b_hat)
    return complex(eps @ C_TENSORS[q] @ n)


def geometric_factors(polarization_lab: CVec, k_hat_lab: Vec, b_hat: Vec) -> dict[int, complex]:
    return {q: geometric_factor(q, polarization_lab, k_hat_lab, b_hat) for q in (-2, -1, 0, 1, 2)}


def geometric_factor_closed_form(q: int, phi_rad: float, gamma_rad: float) -> float:
    """|g^(q)| in the frame B = z, k = (sin phi, 0, cos phi), eps = (cos gamma cos phi, sin gamma, -cos gamma sin phi) (Roos 2000; Section 4.5.7).

    g^(0) = (1/2)|cos gamma sin 2phi|; |g^(+-1)| = (1/sqrt 6)|cos gamma cos 2phi + i sin gamma cos phi|;
    |g^(+-2)| = (1/sqrt 6)|(1/2) cos gamma sin 2phi + i sin gamma sin phi|; gamma is the angle between the
    polarization and the k-B plane (arcsin|eps . y|).
    """
    cg, sg = math.cos(gamma_rad), math.sin(gamma_rad)
    if q == 0:
        return 0.5 * abs(cg * math.sin(2.0 * phi_rad))
    if abs(q) == 1:
        return abs(cg * math.cos(2.0 * phi_rad) + 1j * sg * math.cos(phi_rad)) / math.sqrt(6.0)
    if abs(q) == 2:
        return abs(0.5 * cg * math.sin(2.0 * phi_rad) + 1j * sg * math.sin(phi_rad)) / math.sqrt(6.0)
    raise ValueError("q must lie in -2..2")


def lambda_3j(J_lower: Half, m: Half, J_upper: Half, mp: Half) -> float:
    """Lambda(m, m') = (J 2 J'; -m q m') with q = m - m' (zero when |Delta m| > 2)."""
    jl, ml, ju, mu = (as_half_integer(x) for x in (J_lower, m, J_upper, mp))
    q = ml - mu
    if abs(q) > 2:
        return 0.0
    return wigner_3j(jl, 2, ju, -ml, q, mu)


def reduced_element_from_lifetime_m2(
    wavelength_vac_m: float, partial_rate_rad_s: float, J_upper: Half
) -> float:
    """|<S||r^2 C^(2)||D>| in m^2 from A = c alpha k^5 |<>|^2/(15 (2j'+1)) with the UPPER level's degeneracy."""
    k = 2.0 * math.pi / wavelength_vac_m
    gjp = 2.0 * float(as_half_integer(J_upper)) + 1.0
    return math.sqrt(15.0 * gjp * partial_rate_rad_s / (C_ALPHA_M_PER_S * k**5))


def reduced_element_a0_squared(wavelength_vac_m: float, partial_rate_rad_s: float, J_upper: Half) -> float:
    return reduced_element_from_lifetime_m2(wavelength_vac_m, partial_rate_rad_s, J_upper) / A_0_M**2


def rabi_frequency_e2_rad_s(
    e0_v_per_m: float,
    wavelength_vac_m: float,
    reduced_element_m2: float,
    J_lower: Half,
    m: Half,
    J_upper: Half,
    mp: Half,
    polarization_lab: CVec,
    k_hat_lab: Vec,
    b_hat: Vec,
) -> float:
    """Omega_{mm'} = (e E_0 k/(2 hbar)) |<S||r^2 C^(2)||D>| |Lambda(m, m')| |g^(q)|, q = m - m' (plan convention)."""
    k = 2.0 * math.pi / wavelength_vac_m
    ml, mu = as_half_integer(m), as_half_integer(mp)
    q = ml - mu
    if abs(q) > 2:
        return 0.0
    lam = lambda_3j(J_lower, ml, J_upper, mu)
    g = geometric_factor(int(q), polarization_lab, k_hat_lab, b_hat)
    return E_C * e0_v_per_m * k / (2.0 * HBAR_J_S) * reduced_element_m2 * abs(lam) * abs(g)


def stretched_closed_form_rad_s(
    e0_v_per_m: float, wavelength_vac_m: float, partial_rate_rad_s: float
) -> float:
    """Omega = (e E_0/hbar) sqrt(5 lambda_vac^3 A/(64 pi^3 c alpha)): the |Delta m| = 2 stretched line at (90, 90) degrees.

    90/(12^2 x 8) = 5/64 exactly (James Eqs. 5.11, 5.13; Roos p. 31).
    """
    return (
        E_C
        * e0_v_per_m
        / HBAR_J_S
        * math.sqrt(5.0 * wavelength_vac_m**3 * partial_rate_rad_s / (64.0 * math.pi**3 * C_ALPHA_M_PER_S))
    )


def e2_over_e1_amplitude_ratio(wavelength_vac_m: float) -> float:
    """k a_0 / 2, the E2/E1 amplitude suppression (about alpha/16 at optical wavelengths, not alpha/2)."""
    return 2.0 * math.pi / wavelength_vac_m * A_0_M / 2.0


__all__ = [
    "B_TENSORS",
    "C_ALPHA_M_PER_S",
    "C_TENSORS",
    "e2_over_e1_amplitude_ratio",
    "geometric_factor",
    "geometric_factor_closed_form",
    "geometric_factors",
    "lambda_3j",
    "rabi_frequency_e2_rad_s",
    "racah_c2",
    "reduced_element_a0_squared",
    "reduced_element_from_lifetime_m2",
    "stretched_closed_form_rad_s",
]
