"""Physical constants (CODATA 2022 as shipped by ``scipy.constants``) and the Lande factor (PLAN.md Section 13).

Frequencies: the public API speaks ordinary frequencies in Hz, Hamiltonians are angular (rad/s), every conversion is one
explicit 2 pi, and names carry the unit (``_hz``, ``_rad_s``). The electron g factor is Steck's g_S = +2.0023193..., the
negative of CODATA's g_e, so H_Z = mu_B (g_J J_z + g_I I_z) B with one sign for both g factors. Atomic masses are cited
per isotope in the species tables.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Final

import scipy.constants as sc

TWO_PI: Final[float] = 2.0 * math.pi

C_M_PER_S: Final[float] = sc.c
"""Speed of light in vacuum, m/s."""
H_J_S: Final[float] = sc.h
"""Planck constant, J s."""
HBAR_J_S: Final[float] = sc.hbar
"""Reduced Planck constant, J s."""
E_C: Final[float] = sc.e
"""Elementary charge, C."""
EPSILON_0_F_PER_M: Final[float] = sc.epsilon_0
"""Vacuum permittivity, F/m."""
K_B_J_PER_K: Final[float] = sc.k
"""Boltzmann constant, J/K."""
ALPHA_FS: Final[float] = sc.alpha
"""Fine-structure constant."""
ATOMIC_MASS_KG: Final[float] = sc.atomic_mass
"""Atomic mass constant u, kg."""
ELECTRON_MASS_U: Final[float] = sc.m_e / sc.atomic_mass
"""Electron mass in u: an ion's mass is the cited atomic mass minus this."""
M_E_OVER_M_P: Final[float] = sc.m_e / sc.m_p
"""Electron-to-proton mass ratio, in g_I = -(mu_I/(I mu_N))(m_e/m_p)."""
MU_B_J_PER_T: Final[float] = sc.physical_constants["Bohr magneton"][0]
"""Bohr magneton, J/T."""
MU_B_OVER_H_HZ_PER_T: Final[float] = sc.physical_constants["Bohr magneton in Hz/T"][0]
"""mu_B/h, Hz/T."""
G_S: Final[float] = -sc.physical_constants["electron g factor"][0]
"""Electron spin g factor in Steck's sign convention, +2.0023193..."""


def hz_from_wavenumber_cm(wavenumber_cm: float) -> float:
    """A level energy printed in cm^-1 (NIST ASD) as an ordinary frequency: E/h = 100 c sigma."""
    return 100.0 * C_M_PER_S * wavenumber_cm


def lande_g_j(L: float | Fraction, S: float | Fraction, J: float | Fraction) -> float:
    """The LS-coupling Lande factor with g_L = 1 and Steck's g_S (a [background] formula; tables prefer a measured g_J):
    g_J = [J(J+1) - S(S+1) + L(L+1)]/(2J(J+1)) + g_S [J(J+1) + S(S+1) - L(L+1)]/(2J(J+1))."""
    jj = float(J) * (float(J) + 1.0)
    ss = float(S) * (float(S) + 1.0)
    ll = float(L) * (float(L) + 1.0)
    return (jj - ss + ll) / (2.0 * jj) + G_S * (jj + ss - ll) / (2.0 * jj)
