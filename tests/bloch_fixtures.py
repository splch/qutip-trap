"""Fixtures for the M3a Bloch-builder tests: closed model atoms whose closed forms the sources print.

``two_level_atom`` is an I = 0 species with a J = 0 ground level "S0/2" (one state) and a J = 1 excited level "P2/2"
(three states): pi light drives |g> <-> |e, m=0> and that pair is a CLOSED two-level system (|e, 0> decays only to
|g>), so the RMP Eq. 96 Lorentzian, Stenholm's sideband floor and Itano's Doppler limit hold exactly. ``lambda_atom``
is the mirror image, J = 1 ground ("S2/2") and J = 0 excited ("P0/2"): sigma+ and pi beams drive |g,-1> and |g,0>
to |e>, and the third ground state |g,+1> is excluded with ``leak='renormalize'`` so the two included arms share the
decay 1/2 : 1/2 (Morigi's ideal Lambda system). Level names use the plan's J = "digits/2" grammar (0 = "0/2"), the
Lande factors are entered directly (g_J of a J = 0 level is undefined) and the wavelength, linewidth and mass are the
171Yb+ 369.5 nm values so the recoil numbers are realistic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from qutip_trap.light.beams import Beam
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import ATOMIC_MASS_KG as U_KG
from qutip_trap.units import C_M_PER_S, TWO_PI

GAMMA_HZ = 19.6e6
"""gamma/2pi of the fixture's excited level."""
WAVELENGTH_M = 369.5e-9
MASS_KG = 170.93578 * U_KG
CITES = ("Steck",)


def _species(name: str, ground: str, excited: str, g_ground: float, g_excited: float) -> Species:
    e_hz = C_M_PER_S / WAVELENGTH_M
    tau = 1.0 / (TWO_PI * GAMMA_HZ)
    lower = Level(ground, 0.0, None, 0.0, 0.0, g_ground, CITES)
    upper = Level(excited, e_hz, tau, 0.0, 0.0, g_excited, CITES)
    tr = Transition(ground, excited, WAVELENGTH_M, GAMMA_HZ, 1.0, "E1", CITES)
    return Species(
        name=name,
        mass_u=MASS_KG / U_KG,
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=(lower, upper),
        transitions=(tr,),
        qubit=(f"{ground} mJ=0", f"{excited} mJ=0"),
        cycling=f"{ground}-{excited}",
        repumps=(),
        shelving=None,
    )


def two_level_atom() -> Species:
    """J = 0 -> J' = 1: pi light on |g> <-> |e, 0> is an exactly closed two-level system."""
    return _species("two-level fixture", "S0/2", "P2/2", 0.0, 1.0)


def lambda_atom() -> Species:
    """J = 1 -> J' = 0: two ground sublevels driven to one excited state form a Lambda system (third one excluded)."""
    return _species("Lambda fixture", "S2/2", "P0/2", 1.0, 0.0)


TWO_LEVEL_GROUND = "S0/2 mJ=0"
TWO_LEVEL_EXCITED = "P2/2 mJ=0"
TWO_LEVEL_EXCITED_PLUS = "P2/2 mJ=1"
LAMBDA_EXCITED = "P0/2 mJ=0"
LAMBDA_GROUND_MINUS = "S2/2 mJ=-1"
LAMBDA_GROUND_ZERO = "S2/2 mJ=0"
LAMBDA_GROUND_PLUS = "S2/2 mJ=1"


def structure(
    species: Species, b_gauss: float = 1e-6, b_hat: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> AtomicStructure:
    """The dressed structure at a negligible field (the fixtures have no hyperfine structure; the field only orients B_hat)."""
    return AtomicStructure(species, b_gauss, b_hat)


def gamma_rad_s() -> float:
    return TWO_PI * GAMMA_HZ


def power_for_rabi(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    waist_m: float,
    polarization: tuple[complex, complex, complex],
    k_hat: tuple[float, float, float],
) -> float:
    """The beam power that gives |Omega_{eg}| = omega for the pair (polarization-resolved element)."""
    probe = Beam(WAVELENGTH_M, k_hat, polarization, waist_m, 1e-3, (0.0, 0.0, 0.0))
    om = abs(st.single_photon_coupling_rad_s(st.state(lower), st.state(upper), probe))
    return 1e-3 * (omega_rad_s / om) ** 2


def pi_beam(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    detuning_rad_s: float,
    *,
    k_hat: tuple[float, float, float] = (1.0, 0.0, 0.0),
    waist_m: float = 20e-6,
) -> Beam:
    """A pi-polarized beam (polarization along B = z) propagating along ``k_hat`` with the given Rabi frequency and detuning."""
    pol = (0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j)
    power = power_for_rabi(st, lower, upper, omega_rad_s, waist_m, pol, k_hat)
    omega = TWO_PI * (st.state(upper).energy_hz - st.state(lower).energy_hz) + detuning_rad_s
    return Beam(TWO_PI * C_M_PER_S / omega, k_hat, pol, waist_m, power, (0.0, 0.0, 0.0))


def circular_beam(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    detuning_rad_s: float,
    *,
    q: int = 1,
    k_sign: int = 1,
    waist_m: float = 20e-6,
) -> Beam:
    """A sigma+ (q = +1, eps = -(x + i y)/sqrt2) or sigma- (q = -1, eps = (x - i y)/sqrt2) beam about B = z propagating
    along +-z (``k_sign``); the helicity label follows Section 13 (eps_q drives m -> m + q) and does not depend on k."""
    if q == 1:
        pol = (-1.0 / math.sqrt(2.0) + 0.0j, -1j / math.sqrt(2.0), 0.0 + 0.0j)
    elif q == -1:
        pol = (1.0 / math.sqrt(2.0) + 0.0j, -1j / math.sqrt(2.0), 0.0 + 0.0j)
    else:
        raise ValueError("q is +1 or -1")
    k_hat = (0.0, 0.0, float(k_sign))
    power = power_for_rabi(st, lower, upper, omega_rad_s, waist_m, pol, k_hat)
    omega = TWO_PI * (st.state(upper).energy_hz - st.state(lower).energy_hz) + detuning_rad_s
    return Beam(TWO_PI * C_M_PER_S / omega, k_hat, pol, waist_m, power, (0.0, 0.0, 0.0))


def sigma_plus_beam(
    st: AtomicStructure,
    lower: str,
    upper: str,
    omega_rad_s: float,
    detuning_rad_s: float,
    *,
    waist_m: float = 20e-6,
) -> Beam:
    """A sigma+ beam along +z with the given Rabi frequency on lower -> upper."""
    return circular_beam(st, lower, upper, omega_rad_s, detuning_rad_s, q=1, k_sign=1, waist_m=waist_m)


def two_level_lorentzian(omega_rad_s: float, delta_rad_s: float) -> float:
    """W(Delta) = Gamma (s/2)/(1 + s + (2 Delta/Gamma)^2), s = 2 Omega^2/Gamma^2 (RMP 2003 Eq. 96)."""
    g = gamma_rad_s()
    s = 2.0 * omega_rad_s**2 / g**2
    return g * (s / 2.0) / (1.0 + s + (2.0 * delta_rad_s / g) ** 2)


_ = (np, Fraction)
