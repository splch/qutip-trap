"""The ingest kit of the species tables: the ``Cited`` constructor of a table, its gap exception, and the conversions
applied to cited constants, each a documented formula on a cited input (nothing hyperfine-resolved is typed in).

Each table module exposes ``NAME``, ``TABLE`` (table id -> :class:`Cited`) and ``species()``, which builds the
:class:`~qutip_trap.species.model.Species` or raises :class:`IncompleteSpeciesTable`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from qutip_trap.provenance import Cited, Tag
from qutip_trap.units import C_M_PER_S, ELECTRON_MASS_U, TWO_PI, hz_from_wavenumber_cm


@dataclass(frozen=True)
class CitedFactory:
    """The ``Cited`` constants of one table: ``_c(suffix, value, unit, source, ...)`` with the table's id prefix."""

    prefix: str

    def __call__(
        self,
        suffix: str,
        value: float,
        unit: str,
        source: str,
        *,
        tag: Tag = "extracted",
        uncertainty: float | None = None,
        note: str = "",
    ) -> Cited:
        return Cited(value, unit, source, self.prefix + suffix, tag, uncertainty, note)


@dataclass(frozen=True)
class MissingConstant:
    """A constant ``species()`` needs that no cited source supplies: its table id and what the literature does offer."""

    ledger_id: str
    consult: str


class IncompleteSpeciesTable(LookupError):
    """Raised by ``species()`` for a table that lacks a constant it needs."""

    def __init__(self, name: str, missing: tuple[MissingConstant, ...]) -> None:
        lines = "\n".join(f"  - {m.ledger_id}: {m.consult}" for m in missing)
        super().__init__(f"{name}: the species table is incomplete; missing cited constants:\n{lines}")
        self.missing = missing


def ion_mass_u(atomic_mass: Cited) -> float:
    """The ion mass: the cited atomic mass minus one electron mass (the eV binding energy, ~1e-9 u, is neglected)."""
    if atomic_mass.unit != "u":
        raise ValueError("atomic mass must be cited in u")
    return atomic_mass.value - ELECTRON_MASS_U


def energy_hz(level_cm: Cited) -> float:
    """A NIST level energy in cm^-1 as an ordinary frequency (E/h)."""
    if level_cm.unit != "cm^-1":
        raise ValueError(f"{level_cm.ledger_id}: level energies are cited in cm^-1")
    return hz_from_wavenumber_cm(level_cm.value)


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
    """The magnetic-dipole constant A (Hz, SIGNED) from a zero-field splitting of a level with two hyperfine manifolds.

    For I = 1/2 or J = 1/2 the quadrupole term vanishes and Delta E = A (I + 1/2) (J = 1/2) or A (J + 1/2) (I = 1/2);
    ``inverted`` (the lower-F manifold above the higher-F one) makes A < 0, a sign the sources rarely print.
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
