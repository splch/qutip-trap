"""Helpers shared by the species tables: the cited-constant -> record conversions and the gap list.

Each species module exposes ``NAME``, ``TABLE`` (a mapping of ledger-id suffix -> :class:`Cited`),
``MISSING`` (the constants the plan does not supply, each with the source to consult) and ``species()``,
which builds the Appendix E :class:`Species` or raises :class:`IncompleteSpeciesTable` when a required
constant is missing. Nothing hyperfine-resolved is typed in; the only conversions applied at ingest are
the ones below, each a documented formula on a cited input.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from qutip_trap.provenance import Cited
from qutip_trap.units import (
    C_M_PER_S,
    ELECTRON_MASS_U,
    TWO_PI,
    hz_from_wavenumber_cm,
)


@dataclass(frozen=True)
class MissingConstant:
    """A constant the species table needs and PLAN.md does not supply."""

    quantity: str
    consult: str
    """Where to get it (a source the plan names, or the database to read), so the gap is actionable."""


class IncompleteSpeciesTable(LookupError):
    """Raised by ``species()`` when a required cited constant is absent from the table."""

    def __init__(self, name: str, missing: tuple[MissingConstant, ...]) -> None:
        lines = "\n".join(f"  - {m.quantity}: consult {m.consult}" for m in missing)
        super().__init__(f"{name}: the species table is incomplete; missing cited constants:\n{lines}")
        self.species_name = name
        self.missing = missing


def ion_mass_u(atomic_mass: Cited) -> float:
    """The ion mass: the cited relative atomic mass of the neutral atom minus one electron mass.

    The binding-energy correction (the first ionization energy, a few eV, i.e. a few 1e-9 u) is below the
    precision the tables carry and is neglected; Section 9.16 row 13-6 pins 170.93578 u for 171Yb+.
    """
    if atomic_mass.unit != "u":
        raise ValueError("atomic mass must be cited in u")
    return atomic_mass.value - ELECTRON_MASS_U


def energy_hz(level_cm: Cited) -> float:
    """A NIST level energy in cm^-1 as an ordinary frequency (E/h)."""
    if level_cm.unit != "cm^-1":
        raise ValueError(f"{level_cm.ledger_id}: level energies are cited in cm^-1")
    return float(hz_from_wavenumber_cm(level_cm.value))


def wavelength_vac_m(lower_energy_hz: float, upper_energy_hz: float) -> float:
    """lambda_vac = c / (E_upper - E_lower) from two level energies."""
    if upper_energy_hz <= lower_energy_hz:
        raise ValueError("upper level must lie above lower level")
    return C_M_PER_S / (upper_energy_hz - lower_energy_hz)


def gamma_hz_from_lifetime(lifetime: Cited) -> float:
    """Gamma/2pi = 1/(2 pi tau): the TOTAL decay rate of a level as an ordinary frequency."""
    if lifetime.unit != "s":
        raise ValueError(f"{lifetime.ledger_id}: lifetimes are cited in s")
    return 1.0 / (TWO_PI * lifetime.value)


def lifetime_s_from_linewidth(linewidth: Cited) -> float:
    """tau = 1/(2 pi gamma) for a linewidth quoted as gamma/2pi in Hz (a TOTAL rate)."""
    if linewidth.unit != "Hz":
        raise ValueError(f"{linewidth.ledger_id}: linewidths are cited in Hz (gamma/2pi)")
    return 1.0 / (TWO_PI * linewidth.value)


def a_hfs_from_two_manifold_splitting(
    splitting: Cited, nuclear_spin: Fraction, J: Fraction, *, inverted: bool
) -> float:
    """The magnetic-dipole hyperfine constant A (Hz, SIGNED) from a printed zero-field splitting.

    Valid only when the level has exactly two hyperfine manifolds (I = 1/2 or J = 1/2), where the
    electric-quadrupole term vanishes and E_F = (A/2)[F(F+1) - I(I+1) - J(J+1)] gives
    Delta E = A (I + 1/2) for J = 1/2 and Delta E = A (J + 1/2) for I = 1/2 (Section 4.5.1, Breit-Rabi
    Delta E_hfs = A (I + 1/2)). ``inverted`` (the lower-F manifold above the higher-F one) makes A < 0;
    the sign is an independent input the sources rarely print (Section 4.5.6).
    """
    if splitting.unit != "Hz":
        raise ValueError(f"{splitting.ledger_id}: hyperfine splittings are cited in Hz")
    half = Fraction(1, 2)
    if J == half:
        factor = nuclear_spin + half
    elif nuclear_spin == half:
        factor = J + half
    else:
        raise ValueError(
            "the two-manifold conversion needs I = 1/2 or J = 1/2; general levels need A and B cited"
        )
    magnitude = splitting.value / float(factor)
    return -magnitude if inverted else magnitude


__all__ = [
    "IncompleteSpeciesTable",
    "MissingConstant",
    "a_hfs_from_two_manifold_splitting",
    "energy_hz",
    "gamma_hz_from_lifetime",
    "ion_mass_u",
    "lifetime_s_from_linewidth",
    "wavelength_vac_m",
]
