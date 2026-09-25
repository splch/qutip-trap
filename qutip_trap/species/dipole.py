"""Electric-dipole matrix elements and saturation (PLAN.md Section 4.5.2), in Steck's normalizations.

- The reduced element <J||d||J'> (Brink-Satchler, no 1/sqrt(2j+1)) is fixed by the PARTIAL decay rate of the
  fine-structure line, Gamma_{J'->J} = omega^3 (2J+1) |<J||d||J'>|^2 / (3 pi eps0 hbar c^3 (2J'+1)).
- The J-level Wigner-Eckart element <J m|T_q|J' m'> = <J||d||J'> (-1)^{J'-1+m} sqrt(2J+1) (J' 1 J; m' q -m), nonzero for
  m = m' + q (q = m_lower - m_upper), summing to |<J||d||J'>|^2 over m', q at fixed m.
- The dipole operator on the UNCOUPLED basis |m_I, m_J> is that element times the identity on I, so elements between
  field-dressed eigenstates are the same operator sandwiched between eigenvectors.

I_sat = pi h c Gamma_partial/(3 lambda_vac^3) with the ANGULAR partial rate is the two-level value of a transition of unit
relative strength; the stretched sigma+ cycling element of a J = 1/2 -> 3/2 line is sqrt((2J+1)/(2J'+1)) = sqrt(1/2) of
the reduced element, so Omega = Gamma sqrt(I/(2 I_sat)) holds there and only there.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from qutip_trap.species.wigner import Half, as_half_integer, m_values, parity_sign, wigner_3j
from qutip_trap.units import C_M_PER_S, EPSILON_0_F_PER_M, HBAR_J_S


@lru_cache(maxsize=4096)
def reduced_element_from_partial_rate(
    partial_rate_rad_s: float, omega_rad_s: float, J_lower: Half, J_upper: Half
) -> float:
    """|<J||d||J'>| in C m from Gamma_{J'->J} = omega^3 (2J+1)|<J||d||J'>|^2/(3 pi eps0 hbar c^3 (2J'+1))."""
    if partial_rate_rad_s <= 0.0 or omega_rad_s <= 0.0:
        raise ValueError("rate and frequency must be positive")
    gj = 2.0 * float(as_half_integer(J_lower)) + 1.0
    gjp = 2.0 * float(as_half_integer(J_upper)) + 1.0
    d2 = (
        partial_rate_rad_s
        * 3.0
        * math.pi
        * EPSILON_0_F_PER_M
        * HBAR_J_S
        * C_M_PER_S**3
        / omega_rad_s**3
        * gjp
        / gj
    )
    return math.sqrt(d2)


def wigner_eckart_j(J: Half, m: Half, Jp: Half, mp: Half, q: int) -> float:
    """<J m|T_q|J' m'> / <J||d||J'> = (-1)^{J'-1+m} sqrt(2J+1) (J' 1 J; m' q -m), zero unless m = m' + q."""
    jj, mm, jp, mmp = (as_half_integer(x) for x in (J, m, Jp, mp))
    if mm != mmp + q:
        return 0.0
    return (
        parity_sign(int(jp - 1 + mm)) * math.sqrt(2.0 * float(jj) + 1.0) * wigner_3j(jp, 1, jj, mmp, q, -mm)
    )


def dipole_operator_uncoupled(nuclear_spin: Half, J_lower: Half, J_upper: Half, q: int) -> np.ndarray:
    """T_q / <J||d||J'> on the uncoupled bases: rows (m_I, m_J) of the lower level, columns (m_I', m_J') of the upper.

    Basis order matches :class:`qutip_trap.species.zeeman.HyperfineZeeman`: m_I outer, m_J inner, both ascending.
    """
    mi = m_values(nuclear_spin)
    mj = m_values(J_lower)
    mjp = m_values(J_upper)
    block = np.array([[wigner_eckart_j(J_lower, m, J_upper, mp, q) for mp in mjp] for m in mj])
    return np.kron(np.eye(len(mi)), block)


def field_amplitude_v_per_m(intensity_w_m2: float) -> float:
    """E_0 = sqrt(2 I/(eps0 c)) for a travelling wave of intensity I (Section 4.5.2)."""
    if intensity_w_m2 < 0.0:
        raise ValueError("intensity is non-negative")
    return math.sqrt(2.0 * intensity_w_m2 / (EPSILON_0_F_PER_M * C_M_PER_S))


def stretched_element_factor(J_lower: Half, J_upper: Half) -> float:
    """|<stretched upper|d_{+1}|stretched lower>| / |<J||d||J'>| = sqrt((2J+1)/(2J'+1)) (1/sqrt 2 on a 1/2 -> 3/2 line)."""
    return math.sqrt(
        (2.0 * float(as_half_integer(J_lower)) + 1.0) / (2.0 * float(as_half_integer(J_upper)) + 1.0)
    )
