"""Units, CODATA 2022 constants (via ``scipy.constants``) and the frequency convention.

Internally frequencies are angular (rad/s) and Hamiltonians are energies over hbar; the public API
speaks Hz. :data:`Hz` and :data:`RadPerS` (likewise :data:`Gauss`, :data:`Tesla`) are distinct
static types, crossed only by the converters below. g_S = +2.0023193... takes Steck's sign
(CODATA's g_e is negative), so H_Z = mu_B (g_J J_z + g_I I_z) B / hbar.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Final, NewType

import scipy
from scipy.constants import (
    alpha as _alpha,
)
from scipy.constants import (
    atomic_mass as _atomic_mass,
)
from scipy.constants import (
    c as _c,
)
from scipy.constants import (
    e as _e,
)
from scipy.constants import (
    epsilon_0 as _epsilon_0,
)
from scipy.constants import (
    h as _h,
)
from scipy.constants import (
    hbar as _hbar,
)
from scipy.constants import (
    k as _k_B,
)
from scipy.constants import (
    m_e as _m_e,
)
from scipy.constants import (
    m_p as _m_p,
)
from scipy.constants import (
    physical_constants as _pc,
)

# ---- frequency and field types --------------------------------------------------------------------

Hz = NewType("Hz", float)
"""An ordinary (cycle) frequency in Hz, the public API's unit."""

RadPerS = NewType("RadPerS", float)
"""An angular frequency or a rate in rad/s (an energy divided by hbar); internal only."""

Gauss = NewType("Gauss", float)
"""A magnetic field in gauss, the unit of the atomic-physics API."""

Tesla = NewType("Tesla", float)
"""A magnetic field in tesla (SI)."""

TWO_PI: Final[float] = 2.0 * math.pi


def rad_s_from_hz(f: Hz) -> RadPerS:
    """Hz to rad/s: 2 pi f."""
    return RadPerS(TWO_PI * f)


def hz_from_rad_s(w: RadPerS) -> Hz:
    """rad/s to Hz: w / (2 pi)."""
    return Hz(w / TWO_PI)


def tesla_from_gauss(b: Gauss) -> Tesla:
    """1 G = 1e-4 T."""
    return Tesla(b * 1e-4)


def gauss_from_tesla(b: Tesla) -> Gauss:
    """1 T = 1e4 G."""
    return Gauss(b * 1e4)


# ---- CODATA constants (SI) ------------------------------------------------------------------------

CODATA_EDITION: Final[str] = "CODATA 2022"
CODATA_SOURCE: Final[str] = (
    f"{CODATA_EDITION} recommended values as shipped by scipy.constants (SciPy {scipy.__version__}); "
    "pinned by tests/test_units.py"
)

C_M_PER_S: Final[float] = _c
"""Speed of light in vacuum, m/s (exact)."""
H_J_S: Final[float] = _h
"""Planck constant, J s (exact)."""
HBAR_J_S: Final[float] = _hbar
"""Reduced Planck constant, J s."""
E_C: Final[float] = _e
"""Elementary charge, C (exact)."""
EPSILON_0_F_PER_M: Final[float] = _epsilon_0
"""Vacuum electric permittivity, F/m."""
K_B_J_PER_K: Final[float] = _k_B
"""Boltzmann constant, J/K (exact)."""
ALPHA_FS: Final[float] = _alpha
"""Fine-structure constant."""
ATOMIC_MASS_KG: Final[float] = _atomic_mass
"""Atomic mass constant u, kg."""
M_E_KG: Final[float] = _m_e
"""Electron mass, kg."""
M_P_KG: Final[float] = _m_p
"""Proton mass, kg."""
ELECTRON_MASS_U: Final[float] = _m_e / _atomic_mass
"""Electron mass in u; an ion's mass is the atomic mass minus this."""
M_E_OVER_M_P: Final[float] = _m_e / _m_p
"""Electron-to-proton mass ratio, as in g_I = -(mu_I/(I mu_N))(m_e/m_p)."""

MU_B_J_PER_T: Final[float] = _pc["Bohr magneton"][0]
"""Bohr magneton, J/T."""
MU_N_J_PER_T: Final[float] = _pc["nuclear magneton"][0]
"""Nuclear magneton, J/T."""
MU_B_OVER_H_HZ_PER_T: Final[float] = _pc["Bohr magneton in Hz/T"][0]
"""mu_B/h in Hz/T."""
MU_B_OVER_H_MHZ_PER_G: Final[float] = MU_B_OVER_H_HZ_PER_T * 1e-6 * 1e-4
"""mu_B/h in MHz/G (1.399624...)."""
A_0_M: Final[float] = _pc["Bohr radius"][0]
"""Bohr radius, m."""
G_S: Final[float] = -_pc["electron g factor"][0]
"""Electron spin g-factor in Steck's sign convention, +2.0023193... (CODATA's g_e is negative)."""
G_L: Final[float] = 1.0
"""Orbital g-factor used by the Lande formula (the finite-mass correction 1 - m_e/M is neglected)."""


# ---- small unit helpers -----------------------------------------------------------------------------


def hz_from_wavenumber_cm(wavenumber_cm: float) -> Hz:
    """A level energy printed in cm^-1 (NIST ASD) as an ordinary frequency: E/h = 100 c sigma."""
    return Hz(100.0 * float(C_M_PER_S) * wavenumber_cm)


def wavelength_vac_m_from_hz(delta_hz: Hz) -> float:
    """Vacuum wavelength c / nu (m) of a transition of ordinary frequency ``delta_hz`` (never an air
    wavelength, which would bias k by about 274 ppm at 729 nm)."""
    if delta_hz <= 0.0:
        raise ValueError(f"transition frequency must be positive, got {delta_hz!r} Hz")
    return float(C_M_PER_S) / float(delta_hz)


def lande_g_j(
    L: float | Fraction, S: float | Fraction, J: float | Fraction, *, g_s: float = G_S, g_l: float = G_L
) -> float:
    """The LS-coupling Lande factor (Steck's positive g_S):

    g_J = g_L [J(J+1) - S(S+1) + L(L+1)] / (2J(J+1)) + g_S [J(J+1) + S(S+1) - L(L+1)] / (2J(J+1))."""
    lf, sf, jf = float(L), float(S), float(J)
    if jf <= 0.0:
        raise ValueError("lande_g_j needs J > 0")
    jj = jf * (jf + 1.0)
    ss = sf * (sf + 1.0)
    ll = lf * (lf + 1.0)
    return float(g_l * (jj - ss + ll) / (2.0 * jj) + g_s * (jj + ss - ll) / (2.0 * jj))


__all__ = [
    "A_0_M",
    "ALPHA_FS",
    "ATOMIC_MASS_KG",
    "CODATA_EDITION",
    "CODATA_SOURCE",
    "C_M_PER_S",
    "ELECTRON_MASS_U",
    "EPSILON_0_F_PER_M",
    "E_C",
    "G_L",
    "G_S",
    "HBAR_J_S",
    "H_J_S",
    "Gauss",
    "Hz",
    "K_B_J_PER_K",
    "MU_B_J_PER_T",
    "MU_B_OVER_H_HZ_PER_T",
    "MU_B_OVER_H_MHZ_PER_G",
    "MU_N_J_PER_T",
    "M_E_KG",
    "M_E_OVER_M_P",
    "M_P_KG",
    "RadPerS",
    "TWO_PI",
    "Tesla",
    "gauss_from_tesla",
    "hz_from_rad_s",
    "hz_from_wavenumber_cm",
    "lande_g_j",
    "rad_s_from_hz",
    "tesla_from_gauss",
    "wavelength_vac_m_from_hz",
]
