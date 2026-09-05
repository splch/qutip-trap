"""Fixtures for the atomic-layer tests: species whose closed forms the sources print.

``be9_like`` is a TEST FIXTURE, not a species table: it carries the 9Be+ ground-state constants of the plan
(A_hfs, mu_I, I) with the excited-level hyperfine constants set to ZERO, because Ozeri's and Wineland's closed
forms neglect the P hyperfine splittings against the detuning; the P3/2 decay rate follows the P1/2 one with
the LS-coupling omega^3 scaling (one radial integral), which is the assumption behind the shared gamma of those
forms. ``spin_zero_like`` is a J = 1/2 -> P1/2, P3/2 atom without nuclear spin for the single-electron
angular-algebra anchors of Section 9.10 (Ozeri 2005). Level energies are the NIST Be II values.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from qutip_trap.api import Beam, Field
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.polarization import linear_polarization
from qutip_trap.units import C_M_PER_S, TWO_PI, hz_from_wavenumber_cm, lande_g_j

E_P12_CM = 31928.744  # Be II 2p 2P1/2 (NIST ASD 5.12)
E_P32_CM = 31935.320  # Be II 2p 2P3/2
GAMMA_HZ = 19.6e6  # Ozeri 2007 Table I gamma/2pi for 9Be+
A_S12_HZ = -625.008837048e6  # PLAN.md 9.13
MU_I_BE9 = -1.177432
G_J_BE9 = 2.00226
HALF = Fraction(1, 2)


def _levels_and_transitions(
    nuclear_spin: float,
    a_s12: float,
    gamma_p12_hz: float,
    gamma_p32_hz: float | None = None,
    g_j_s: float | None = None,
):  # type: ignore[no-untyped-def]
    e12 = float(hz_from_wavenumber_cm(E_P12_CM))
    e32 = float(hz_from_wavenumber_cm(E_P32_CM))
    if gamma_p32_hz is None:
        gamma_p32_hz = gamma_p12_hz * (e32 / e12) ** 3  # LS coupling: one radial integral, Gamma ~ omega^3
    tau12 = 1.0 / (TWO_PI * gamma_p12_hz)
    tau32 = 1.0 / (TWO_PI * gamma_p32_hz)
    cites = ("Steck",)
    s12 = Level(
        "S1/2",
        0.0,
        None,
        a_s12 if nuclear_spin else 0.0,
        0.0,
        g_j_s if g_j_s is not None else lande_g_j(0, HALF, HALF),
        cites,
    )
    p12 = Level("P1/2", e12, tau12, 0.0, 0.0, lande_g_j(1, HALF, HALF), cites)
    p32 = Level("P3/2", e32, tau32, 0.0, 0.0, lande_g_j(1, HALF, Fraction(3, 2)), cites)
    t12 = Transition("S1/2", "P1/2", C_M_PER_S / e12, gamma_p12_hz, 1.0, "E1", cites)
    t32 = Transition("S1/2", "P3/2", C_M_PER_S / e32, gamma_p32_hz, 1.0, "E1", cites)
    return (s12, p12, p32), (t12, t32)


def be9_like(*, a_scale: float = 1.0, gamma_p32_hz: float | None = None) -> Species:
    """I = 3/2 fixture with the 9Be+ ground-state hyperfine constant (times ``a_scale``) and hyperfine-free P levels."""
    levels, transitions = _levels_and_transitions(1.5, A_S12_HZ * a_scale, GAMMA_HZ, gamma_p32_hz, G_J_BE9)
    return Species(
        name="9Be+-like fixture",
        mass_u=9.012183065 - 0.00054858,
        nuclear_spin=1.5,
        mu_I_nuclear_magnetons=MU_I_BE9,
        levels=levels,
        transitions=transitions,
        qubit=("S1/2 F=2 mF=0", "S1/2 F=1 mF=0"),
        cycling="S1/2-P3/2",
        repumps=(),
        shelving=None,
    )


def spin_zero_like(*, gamma_p12_hz: float = GAMMA_HZ, gamma_p32_hz: float | None = None) -> Species:
    """I = 0 fixture: S1/2, P1/2, P3/2 with LS reduced elements (or the given unequal rates)."""
    levels, transitions = _levels_and_transitions(0.0, 0.0, gamma_p12_hz, gamma_p32_hz)
    return Species(
        name="spin-zero fixture",
        mass_u=9.0,
        nuclear_spin=0.0,
        mu_I_nuclear_magnetons=0.0,
        levels=levels,
        transitions=transitions,
        qubit=("S1/2 mJ=-1/2", "S1/2 mJ=1/2"),
        cycling="S1/2-P3/2",
        repumps=(),
        shelving=None,
    )


def field_z(b_gauss: float = 1.0) -> Field:
    return Field(B_gauss=b_gauss, direction=(0.0, 0.0, 1.0), noise=None)


def fine_structure_omega(species: Species) -> float:
    """omega_f = 2 pi (E(P3/2) - E(P1/2))."""
    return TWO_PI * (species.level("P3/2").energy_hz - species.level("P1/2").energy_hz)


def beam_at_detuning(
    species: Species,
    delta_rad_s: float,
    k_hat: tuple[float, float, float],
    polarization: tuple[complex, complex, complex],
    *,
    power_w: float = 1e-3,
    waist_m: float = 20e-6,
    ground_energy_hz: float = 0.0,
) -> Beam:
    """A beam whose frequency sits ``delta`` (angular, plan sign) from the S1/2 (ground energy) -> P1/2 transition."""
    omega_p12 = TWO_PI * (species.level("P1/2").energy_hz - ground_energy_hz)
    omega_l = omega_p12 + delta_rad_s
    return Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, polarization, waist_m, power_w, (0.0, 0.0, 0.0))


def lin_perp_lin_pair(
    species: Species,
    delta_rad_s: float,
    *,
    power_w: float = 1e-3,
    waist_m: float = 20e-6,
    ground_energy_hz: float = 0.0,
    omega_q_rad_s: float = 0.0,
) -> tuple[Beam, Beam]:
    """Two beams perpendicular to B = z with orthogonal linear polarizations perpendicular to B (b along x polarized y, r along y polarized x).

    The second beam is lowered in frequency by ``omega_q_rad_s`` (the qubit splitting) so the pair is Raman resonant.
    """
    b = beam_at_detuning(
        species,
        delta_rad_s,
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        power_w=power_w,
        waist_m=waist_m,
        ground_energy_hz=ground_energy_hz,
    )
    r = beam_at_detuning(
        species,
        delta_rad_s - omega_q_rad_s,
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        power_w=power_w,
        waist_m=waist_m,
        ground_energy_hz=ground_energy_hz,
    )
    return b, r


def sigma_plus_along_z(species: Species, delta_rad_s: float, **kw: float) -> Beam:
    """A sigma+ beam propagating along B = z: eps = -(x + i y)/sqrt2 = e_{+1}."""
    pol = (-1.0 / math.sqrt(2.0), -1j / math.sqrt(2.0), 0.0)
    return beam_at_detuning(species, delta_rad_s, (0.0, 0.0, 1.0), pol, **kw)  # type: ignore[arg-type]


def pi_beam_perp(species: Species, delta_rad_s: float, **kw: float) -> Beam:
    """A pi-polarized beam propagating along x with eps along B = z."""
    return beam_at_detuning(species, delta_rad_s, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), **kw)  # type: ignore[arg-type]


def stretched_g_half(species: Species, beam: Beam, field: Field) -> float:
    """Ozeri's g: half the plan's Rabi frequency of the stretched sigma+ cycling line at the beam's intensity."""
    from qutip_trap.species.dipole import field_amplitude_v_per_m, stretched_element_factor
    from qutip_trap.species.raman import AtomicStructure
    from qutip_trap.units import HBAR_J_S

    st = AtomicStructure(species, field.B_gauss, field.direction)
    d_red = st.reduced_element_c_m("S1/2", "P3/2")
    d_str = stretched_element_factor(HALF, Fraction(3, 2)) * d_red
    e0 = field_amplitude_v_per_m(beam.intensity_at(np.zeros(3)))
    return e0 * d_str / HBAR_J_S / 2.0


_ = linear_polarization  # re-exported convenience for tests


def toy_spin_zero(
    *,
    e_p12_hz: float = 1.0e8,
    e_p32_hz: float = 1.5e8,
    gamma_p12_s: float = 1.0e5,
    gamma_p32_s: float = 3.0e5,
) -> Species:
    """An I = 0 S1/2, P1/2, P3/2 atom with MHz-scale level spacings and s^-1 decay rates: the same angular algebra as a real ion,
    but a master equation that is not stiff. The metre-scale wavelengths are irrelevant to the check."""
    cites = ("Steck",)
    g12, g32 = gamma_p12_s / TWO_PI, gamma_p32_s / TWO_PI
    s12 = Level("S1/2", 0.0, None, 0.0, 0.0, lande_g_j(0, HALF, HALF), cites)
    p12 = Level("P1/2", e_p12_hz, 1.0 / (TWO_PI * g12), 0.0, 0.0, lande_g_j(1, HALF, HALF), cites)
    p32 = Level("P3/2", e_p32_hz, 1.0 / (TWO_PI * g32), 0.0, 0.0, lande_g_j(1, HALF, Fraction(3, 2)), cites)
    t12 = Transition("S1/2", "P1/2", C_M_PER_S / e_p12_hz, g12, 1.0, "E1", cites)
    t32 = Transition("S1/2", "P3/2", C_M_PER_S / e_p32_hz, g32, 1.0, "E1", cites)
    return Species(
        "toy spin-zero fixture",
        9.0,
        0.0,
        0.0,
        (s12, p12, p32),
        (t12, t32),
        ("S1/2 mJ=-1/2", "S1/2 mJ=1/2"),
        "S1/2-P3/2",
        (),
        None,
    )
