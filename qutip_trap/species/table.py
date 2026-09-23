"""Helpers shared by the species tables: conversions of cited constants and the list of missing ones.

Each species module exposes ``NAME``, ``TABLE`` (id -> :class:`Cited`), ``MISSING`` (the gaps, each with where to
look) and ``species()``, which builds the :class:`Species` or raises :class:`IncompleteSpeciesTable`."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Literal

from qutip_trap.units import (
    C_M_PER_S,
    ELECTRON_MASS_U,
    TWO_PI,
    hz_from_wavenumber_cm,
)

Tag = Literal["verified", "corrected", "extracted", "background", "recomputed here", "derived", "contested"]

TAGS: Final[tuple[str, ...]] = (
    "verified",
    "corrected",
    "extracted",
    "background",
    "recomputed here",
    "derived",
    "contested",
)
"""How a value was checked: verified against its primary source, corrected (an error in the extracted or printed form
fixed), extracted (quoted from a primary source, not independently checked), background (textbook physics), recomputed
here, derived (computed from cited inputs), contested (unresolved between sources)."""


@dataclass(frozen=True)
class Cited:
    """One number typed into a species table: the value as the source prints it, with its citation. ``source`` is a key
    of :data:`qutip_trap.species.sources.SOURCES`; ``key`` is the table key, ``<species>.<level>.<quantity>``."""

    value: float
    unit: str
    source: str
    key: str
    tag: Tag = "extracted"
    uncertainty: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.tag not in TAGS:
            raise ValueError(f"unknown provenance tag {self.tag!r}; allowed: {TAGS}")
        if not self.source:
            raise ValueError(f"Cited({self.key!r}) has no source: every number is cited")
        if not self.key:
            raise ValueError("Cited values must name their table key")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")


@dataclass(frozen=True)
class MissingConstant:
    """A constant a species table needs but does not have."""

    quantity: str
    consult: str
    """Where to find it (a source or a database)."""


class IncompleteSpeciesTable(LookupError):
    """Raised by ``species()`` when a required cited constant is absent from the table."""

    def __init__(self, name: str, missing: tuple[MissingConstant, ...]) -> None:
        lines = "\n".join(f"  - {m.quantity}: consult {m.consult}" for m in missing)
        super().__init__(f"{name}: the species table is incomplete; missing cited constants:\n{lines}")
        self.species_name = name
        self.missing = missing


def required_constants_missing(
    table: dict[str, Cited], required: dict[str, str], consult: dict[str, str]
) -> tuple[MissingConstant, ...]:
    """The :class:`MissingConstant` entries for the ids in ``required`` (id -> description) that ``table`` lacks;
    ``consult`` optionally overrides the default "where to get it" per id."""
    out: list[MissingConstant] = []
    for key, quantity in required.items():
        if key not in table:
            where = consult.get(key, f"no cited value for {key} is in the table yet")
            out.append(MissingConstant(f"{quantity} ({key})", where))
    return tuple(out)


def ion_mass_u(atomic_mass: Cited) -> float:
    """The ion mass in u: the cited neutral-atom mass minus one electron mass (the binding energy, ~1e-9 u, neglected)."""
    if atomic_mass.unit != "u":
        raise ValueError("atomic mass must be cited in u")
    return atomic_mass.value - ELECTRON_MASS_U


def energy_hz(level_cm: Cited) -> float:
    """A NIST level energy in cm^-1 as an ordinary frequency (E/h)."""
    if level_cm.unit != "cm^-1":
        raise ValueError(f"{level_cm.key}: level energies are cited in cm^-1")
    return float(hz_from_wavenumber_cm(level_cm.value))


def wavelength_vac_m(lower_energy_hz: float, upper_energy_hz: float) -> float:
    """lambda_vac = c / (E_upper - E_lower) from two level energies."""
    if upper_energy_hz <= lower_energy_hz:
        raise ValueError("upper level must lie above lower level")
    return C_M_PER_S / (upper_energy_hz - lower_energy_hz)


def gamma_hz_from_lifetime(lifetime: Cited) -> float:
    """Gamma/2pi = 1/(2 pi tau): the TOTAL decay rate of a level as an ordinary frequency."""
    if lifetime.unit != "s":
        raise ValueError(f"{lifetime.key}: lifetimes are cited in s")
    return 1.0 / (TWO_PI * lifetime.value)


def lifetime_s_from_linewidth(linewidth: Cited) -> float:
    """tau = 1/(2 pi gamma) for a linewidth quoted as gamma/2pi in Hz (a TOTAL rate)."""
    if linewidth.unit != "Hz":
        raise ValueError(f"{linewidth.key}: linewidths are cited in Hz (gamma/2pi)")
    return 1.0 / (TWO_PI * linewidth.value)


def a_hfs_from_two_manifold_splitting(
    splitting: Cited, nuclear_spin: Fraction, J: Fraction, *, inverted: bool
) -> float:
    """The SIGNED magnetic-dipole hyperfine constant A (Hz) from a printed zero-field splitting.

    Valid only for two hyperfine manifolds (I = 1/2 or J = 1/2), where Delta E = A (I + 1/2) for J = 1/2 and
    A (J + 1/2) for I = 1/2. ``inverted`` (lower F above higher F) makes A < 0; the sources rarely print the sign."""
    if splitting.unit != "Hz":
        raise ValueError(f"{splitting.key}: hyperfine splittings are cited in Hz")
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
