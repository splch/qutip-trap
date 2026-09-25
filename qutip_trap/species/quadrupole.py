"""Electric-quadrupole (E2) coupling of optical qubits (PLAN.md Section 4.5.7).

The E2 drive on |S, m> <-> |D, m'> has the dipole form with one beam's k and the scalar

    Omega_{mm'} = (e E_0 k/(2 hbar)) |<S||r^2 C^(2)||D>| |Lambda(m, m')| |g^(q)|,   q = m - m' = -Delta m,

with Lambda the 3j symbol (J 2 J'; -m q m'), C^(2)_q = sqrt(4 pi/5) Y_{2q} (Racah), and g^(q) = c^(q)_{ij} eps_i n_j the
contraction of polarization and propagation direction with the rank-2 tensors c^(q) = (2/3) b^(q) (sum_ij |c^(q)_ij|^2 =
2/3), b^(q)_ij r_i r_j = r^2 C^(2)_q, never the conjugate duals (they share the norm but turn q = m - m' into m' - m for
circular light). The reduced element comes from the D-state lifetime, A = c alpha k^5 |<S||r^2 C^(2)||D>|^2/(15 (2j' + 1))
with j' the UPPER level and k the VACUUM wavenumber; Omega is the plan's (a carrier pi pulse at t = pi/Omega), twice the
Boulder value (James 1998).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction

import numpy as np

from qutip_trap.species.polarization import CVec, Vec, to_atomic_frame
from qutip_trap.species.wigner import Half, as_half_integer, wigner_3j
from qutip_trap.units import ALPHA_FS, C_M_PER_S, E_C, HBAR_J_S, TWO_PI

C_ALPHA_M_PER_S: float = C_M_PER_S * ALPHA_FS
"""c alpha = e^2/(4 pi eps0 hbar) = 2.187691e6 m/s."""


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
"""b^(q)_ij with b^(q)_ij r_i r_j = r^2 C^(2)_q (James 1998 Appendix)."""

C_TENSORS: dict[int, np.ndarray] = {q: (2.0 / 3.0) * b for q, b in B_TENSORS.items()}
"""c^(q) = (2/3) b^(q), normalized to sum_ij |c^(q)_ij|^2 = 2/3."""


def geometric_factor(q: int, polarization_lab: CVec, k_hat_lab: Vec, b_hat: Vec) -> complex:
    """g^(q) = c^(q)_ij eps_i n_j with eps and n expressed in the B_hat frame (complex for elliptical light)."""
    eps = to_atomic_frame(polarization_lab, b_hat)
    n = to_atomic_frame(np.asarray(k_hat_lab, dtype=float), b_hat)
    return complex(eps @ C_TENSORS[q] @ n)


def lambda_3j(J_lower: Half, m: Half, J_upper: Half, mp: Half) -> float:
    """Lambda(m, m') = (J 2 J'; -m q m') with q = m - m' (zero when |Delta m| > 2)."""
    jl, ml, ju, mu = (as_half_integer(x) for x in (J_lower, m, J_upper, mp))
    q = ml - mu
    if abs(q) > 2:
        return 0.0
    return wigner_3j(jl, 2, ju, -ml, q, mu)


def e2_stark_shift_rad_s(
    couplings_rad_s: Mapping[tuple[Fraction, Fraction], float],
    lower_energies_hz: Mapping[Fraction, float],
    upper_energies_hz: Mapping[Fraction, float],
    m_driven: Half,
    mp_driven: Half,
) -> float:
    """delta_St of the driven |S, m0>-|D, m0'> component, second order in the OTHER components' couplings (Section 4.5.7).

    With the laser on resonance with (m0, m0'), component (m, m') is detuned by the pure Zeeman offset
    Delta(m, m') = 2 pi [(E_D(m0') - E_S(m0)) - (E_D(m') - E_S(m))]. In the (hbar Omega/2) convention a component shifts
    its lower level by +|Omega|^2/(4 Delta) and its upper level by -|Omega|^2/(4 Delta) (red light lowers the lower
    level, as in ``AtomicStructure.light_shift_rad_s``), so the transition shifts by

        delta_St = -sum_{m != m0} |Omega(m, m0')|^2/(4 Delta(m, m0')) - sum_{m' != m0'} |Omega(m0, m')|^2/(4 Delta(m0, m')),

    the driven component excluded (the Rabi dynamics treats it exactly). There is no i gamma/2 in these denominators, so
    a component degenerate with the driven one raises rather than returning an infinity.
    """
    m0, mp0 = as_half_integer(m_driven), as_half_integer(mp_driven)
    if (m0, mp0) not in couplings_rad_s:
        raise KeyError(f"the driven component ({m0}, {mp0}) is not in the coupling table")
    resonance = upper_energies_hz[mp0] - lower_energies_hz[m0]
    total = 0.0
    for (ml, mu), omega in couplings_rad_s.items():
        if (ml, mu) == (m0, mp0) or omega == 0.0:
            continue
        if ml != m0 and mu != mp0:  # shares neither level with the driven pair: no second-order shift of it
            continue
        delta = TWO_PI * (resonance - (upper_energies_hz[mu] - lower_energies_hz[ml]))
        if delta == 0.0:
            raise ZeroDivisionError(
                f"E2 component ({ml}, {mu}) is degenerate with the driven ({m0}, {mp0}); the second-order sum "
                "has no i gamma/2 and is valid only away from degeneracy"
            )
        total -= abs(omega) ** 2 / (4.0 * delta)
    return total


def reduced_element_from_lifetime_m2(
    wavelength_vac_m: float, partial_rate_rad_s: float, J_upper: Half
) -> float:
    """|<S||r^2 C^(2)||D>| in m^2 from A = c alpha k^5 |<>|^2/(15 (2j'+1)) with the UPPER level's degeneracy."""
    k = 2.0 * math.pi / wavelength_vac_m
    gjp = 2.0 * float(as_half_integer(J_upper)) + 1.0
    return math.sqrt(15.0 * gjp * partial_rate_rad_s / (C_ALPHA_M_PER_S * k**5))


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
