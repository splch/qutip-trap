"""Units, physical constants and the frequency convention (PLAN.md Sections 5.6 and 13).

Convention (Section 13, row "Frequencies"; Section 5.6): internally every frequency is ANGULAR, in
rad/s, and every energy is divided by hbar so that Hamiltonians are in rad/s; the PUBLIC API takes and
returns ORDINARY frequencies in Hz and converts at the boundary with an explicit 2 pi. Factor-of-2 pi
errors are the single most common bug in this domain, so the two are distinct static types here,
:data:`Hz` and :data:`RadPerS`: a type checker refuses an ``Hz`` where a ``RadPerS`` is expected, and
the only way across is :func:`rad_s_from_hz` / :func:`hz_from_rad_s`. Magnetic fields follow the same
pattern (:data:`Gauss` in the atomic-physics API, :data:`Tesla` where SI is needed). ``NewType`` costs
nothing at run time; the enforcement is ``mypy --strict`` in CI.

Physical constants are the CODATA 2022 recommended values as shipped by ``scipy.constants`` (SciPy
1.18.1 on the pinned toolchain); :data:`CODATA_SOURCE` records that, and ``tests/test_units.py`` pins
the values so a silent change of edition fails loudly. Atomic masses are NOT here: they are cited per
isotope in each species module (Section 5.6, "atomic masses from the AME tables with citation").

Sign convention for the electron g-factor (Section 13, row "Sign of A_hfs and the g-factors"): the
module uses Steck's g_S = +2.0023193..., the negative of CODATA's g_e, so that the Zeeman Hamiltonian
reads H_Z = mu_B (g_J J_z + g_I I_z) B / hbar with one sign for both g factors.
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
"""An ordinary (cycle) frequency. The public API speaks Hz (Section 5.6)."""

RadPerS = NewType("RadPerS", float)
"""An angular frequency or a rate in rad/s (equivalently an energy divided by hbar). Internal only."""

Gauss = NewType("Gauss", float)
"""A magnetic field in gauss, the unit of the atomic-physics sources the plan cites (Section 4.5)."""

Tesla = NewType("Tesla", float)
"""A magnetic field in tesla (SI)."""

TWO_PI: Final[float] = 2.0 * math.pi


def rad_s_from_hz(f: Hz) -> RadPerS:
    """The one conversion from the public Hz convention to the internal rad/s convention: 2 pi f."""
    return RadPerS(TWO_PI * f)


def hz_from_rad_s(w: RadPerS) -> Hz:
    """The inverse of :func:`rad_s_from_hz`: w / (2 pi)."""
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
"""Electron mass in u; an ion's mass is the cited atomic mass minus this (species tables)."""
M_E_OVER_M_P: Final[float] = _m_e / _m_p
"""Electron-to-proton mass ratio, used in g_I = -(mu_I/(I mu_N))(m_e/m_p) (Section 4.5.1)."""

MU_B_J_PER_T: Final[float] = _pc["Bohr magneton"][0]
"""Bohr magneton, J/T."""
MU_N_J_PER_T: Final[float] = _pc["nuclear magneton"][0]
"""Nuclear magneton, J/T."""
MU_B_OVER_H_HZ_PER_T: Final[float] = _pc["Bohr magneton in Hz/T"][0]
"""mu_B/h in Hz/T."""
MU_B_OVER_H_MHZ_PER_G: Final[float] = MU_B_OVER_H_HZ_PER_T * 1e-6 * 1e-4
"""mu_B/h = 1.399624... MHz/G, the number Section 4.5.1 prints for the Zeeman Hamiltonian in Hz."""
A_0_M: Final[float] = _pc["Bohr radius"][0]
"""Bohr radius, m."""
G_S: Final[float] = -_pc["electron g factor"][0]
"""Electron spin g-factor in Steck's sign convention, +2.0023193... (Section 13). CODATA's g_e is negative."""
G_L: Final[float] = 1.0
"""Orbital g-factor used by the Lande formula (the finite-mass correction 1 - m_e/M is neglected)."""


# ---- small unit helpers -----------------------------------------------------------------------------


def hz_from_wavenumber_cm(wavenumber_cm: float) -> Hz:
    """A level energy printed in cm^-1 (NIST ASD) as an ordinary frequency: E/h = 100 c sigma."""
    return Hz(100.0 * float(C_M_PER_S) * wavenumber_cm)


def wavelength_vac_m_from_hz(delta_hz: Hz) -> float:
    """Vacuum wavelength of a transition of ordinary frequency delta_hz: lambda_vac = c / nu.

    Section 13 (row "Quadrupole (E2) coupling" and Section 4.5.7): wavelengths are VACUUM wavelengths
    everywhere; an air wavelength used as vacuum biases k by about 274 ppm at 729 nm.
    """
    if delta_hz <= 0.0:
        raise ValueError(f"transition frequency must be positive, got {delta_hz!r} Hz")
    return float(C_M_PER_S) / float(delta_hz)


def lande_g_j(
    L: float | Fraction, S: float | Fraction, J: float | Fraction, *, g_s: float = G_S, g_l: float = G_L
) -> float:
    """The LS-coupling Lande factor g_J (a [background] formula, PLAN.md Section 4.5.7).

    g_J = g_L [J(J+1) - S(S+1) + L(L+1)] / (2J(J+1)) + g_S [J(J+1) + S(S+1) - L(L+1)] / (2J(J+1)),
    with Steck's positive g_S. The plan prefers a MEASURED g_J wherever a source prints one (Section
    4.5.6: "the measured g_J replaces the Lande formula"); species tables tag Lande values [background].
    """
    lf, sf, jf = float(L), float(S), float(J)
    if jf <= 0.0:
        raise ValueError("lande_g_j needs J > 0")
    jj = jf * (jf + 1.0)
    ss = sf * (sf + 1.0)
    ll = lf * (lf + 1.0)
    return float(g_l * (jj - ss + ll) / (2.0 * jj) + g_s * (jj + ss - ll) / (2.0 * jj))
