"""Electric-dipole matrix elements, decay rates and saturation in Steck's normalization: the reduced element
<J||d||J'> (Brink-Satchler, no 1/sqrt(2J+1)) is fixed by the PARTIAL decay rate of the line, and q = m_lower - m_upper."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from qutip_trap.species.wigner import (
    Half,
    as_half_integer,
    clebsch_gordan,
    m_values,
    parity_sign,
    wigner_3j,
    wigner_6j,
)
from qutip_trap.units import C_M_PER_S, EPSILON_0_F_PER_M, H_J_S, HBAR_J_S


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


def partial_rate_from_reduced_element(
    d_c_m: float, omega_rad_s: float, J_lower: Half, J_upper: Half
) -> float:
    """The inverse of :func:`reduced_element_from_partial_rate` (rad/s)."""
    gj = 2.0 * float(as_half_integer(J_lower)) + 1.0
    gjp = 2.0 * float(as_half_integer(J_upper)) + 1.0
    return (
        omega_rad_s**3 * gj * d_c_m**2 / (3.0 * math.pi * EPSILON_0_F_PER_M * HBAR_J_S * C_M_PER_S**3 * gjp)
    )


def wigner_eckart_j(J: Half, m: Half, Jp: Half, mp: Half, q: int) -> float:
    """<J m|T_q|J' m'> / <J||d||J'> = (-1)^{J'-1+m} sqrt(2J+1) (J' 1 J; m' q -m), zero unless m = m' + q."""
    jj, mm, jp, mmp = (as_half_integer(x) for x in (J, m, Jp, mp))
    if mm != mmp + q:
        return 0.0
    return (
        parity_sign(int(jp - 1 + mm)) * math.sqrt(2.0 * float(jj) + 1.0) * wigner_3j(jp, 1, jj, mmp, q, -mm)
    )


def hyperfine_reduced_factor(F: Half, Fp: Half, J: Half, Jp: Half, nuclear_spin: Half) -> float:
    """<F||d||F'> / <J||d||J'> = (-1)^{F'+J+1+I} sqrt((2F'+1)(2J+1)) {J J' 1; F' F I} (Steck Rb87 Eq. 36)."""
    f, fp, j, jp, ii = (as_half_integer(x) for x in (F, Fp, J, Jp, nuclear_spin))
    phase = parity_sign(int(fp + j + 1 + ii))
    return (
        phase * math.sqrt((2.0 * float(fp) + 1.0) * (2.0 * float(j) + 1.0)) * wigner_6j(j, jp, 1, fp, f, ii)
    )


def hyperfine_element(
    F: Half, mF: Half, Fp: Half, mFp: Half, q: int, J: Half, Jp: Half, nuclear_spin: Half
) -> float:
    """<F mF|d_q|F' mF'> / <J||d||J'> in Steck's factorization, nonzero for mF = mF' + q."""
    f, mf, fp, mfp = (as_half_integer(x) for x in (F, mF, Fp, mFp))
    if mf != mfp + q:
        return 0.0
    phase = parity_sign(int(fp - 1 + mf))
    return (
        hyperfine_reduced_factor(f, fp, J, Jp, nuclear_spin)
        * phase
        * math.sqrt(2.0 * float(f) + 1.0)
        * wigner_3j(fp, 1, f, mfp, q, -mf)
    )


def absorption_strength(F: Half, Fp: Half, J: Half, Jp: Half, nuclear_spin: Half) -> float:
    """S_FF' = (2F'+1)(2J+1){J J' 1; F' F I}^2, the relative absorption strength, summing to 1 over F'."""
    f, fp, j, jp, ii = (as_half_integer(x) for x in (F, Fp, J, Jp, nuclear_spin))
    return (2.0 * float(fp) + 1.0) * (2.0 * float(j) + 1.0) * wigner_6j(j, jp, 1, fp, f, ii) ** 2


def emission_branching(Fp: Half, F: Half, J: Half, Jp: Half, nuclear_spin: Half) -> float:
    """b(F'->F) = (2F+1)(2J'+1){J J' 1; F' F I}^2, the decay branching of F' into F, summing to 1 over F."""
    f, fp, j, jp, ii = (as_half_integer(x) for x in (F, Fp, J, Jp, nuclear_spin))
    return (2.0 * float(f) + 1.0) * (2.0 * float(jp) + 1.0) * wigner_6j(j, jp, 1, fp, f, ii) ** 2


def dipole_operator_uncoupled(nuclear_spin: Half, J_lower: Half, J_upper: Half, q: int) -> np.ndarray:
    """T_q / <J||d||J'> on the uncoupled bases: rows (m_I, m_J) of the lower level, columns (m_I', m_J') of the upper.

    The identity on I; basis order as in :class:`qutip_trap.species.zeeman.HyperfineZeeman` (m_I outer, m_J inner,
    both ascending)."""
    mi = m_values(nuclear_spin)
    mj = m_values(J_lower)
    mjp = m_values(J_upper)
    block = np.array([[wigner_eckart_j(J_lower, m, J_upper, mp, q) for mp in mjp] for m in mj])
    return np.kron(np.eye(len(mi)), block)


def coupled_state_vector(nuclear_spin: Half, J: Half, F: Half, mF: Half) -> np.ndarray:
    """|F mF> on the uncoupled basis in the |(J I) F> coupling order: components <J mJ I mI|F mF>."""
    mi = m_values(nuclear_spin)
    mj = m_values(J)
    return np.array([clebsch_gordan(J, mj_, nuclear_spin, mi_, F, mF) for mi_ in mi for mj_ in mj])


def saturation_intensity_w_m2(partial_rate_rad_s: float, wavelength_vac_m: float) -> float:
    """I_sat = pi h c Gamma_partial / (3 lambda_vac^3), Gamma_partial angular: the value for unit relative strength."""
    return math.pi * H_J_S * C_M_PER_S * partial_rate_rad_s / (3.0 * wavelength_vac_m**3)


def saturation_intensity_random_orientation_w_m2(partial_rate_rad_s: float, wavelength_vac_m: float) -> float:
    """hbar omega^3 Gamma/(4 pi c^2), the widely quoted random-orientation value, exactly 3x the cycling I_sat."""
    omega = 2.0 * math.pi * C_M_PER_S / wavelength_vac_m
    return HBAR_J_S * omega**3 * partial_rate_rad_s / (4.0 * math.pi * C_M_PER_S**2)


def resonant_cross_section_m2(wavelength_vac_m: float) -> float:
    """sigma_0 = 3 lambda_vac^2/(2 pi)."""
    return 3.0 * wavelength_vac_m**2 / (2.0 * math.pi)


def field_amplitude_v_per_m(intensity_w_m2: float) -> float:
    """E_0 = sqrt(2 I/(eps0 c)) for a travelling wave of intensity I."""
    if intensity_w_m2 < 0.0:
        raise ValueError("intensity is non-negative")
    return math.sqrt(2.0 * intensity_w_m2 / (EPSILON_0_F_PER_M * C_M_PER_S))


def stretched_element_factor(J_lower: Half, J_upper: Half) -> float:
    """|<stretched upper|d_{+1}|stretched lower>| / |<J||d||J'>| = sqrt((2J+1)/(2J'+1)) (1/sqrt 2 on a 1/2 -> 3/2 line)."""
    return math.sqrt(
        (2.0 * float(as_half_integer(J_lower)) + 1.0) / (2.0 * float(as_half_integer(J_upper)) + 1.0)
    )


def rabi_frequency_two_level_rad_s(
    gamma_partial_rad_s: float, intensity_w_m2: float, wavelength_vac_m: float
) -> float:
    """Omega = Gamma sqrt(I/(2 I_sat)), valid only on a closed two-level (stretched cycling) transition."""
    return gamma_partial_rad_s * math.sqrt(
        intensity_w_m2 / (2.0 * saturation_intensity_w_m2(gamma_partial_rad_s, wavelength_vac_m))
    )
