"""43Ca+ species table (PLAN.md Sections 4.3.3, 4.5.1, 4.5.6, 8.1, 9.13; Appendix E) - PARTIAL.

The ground-state constants the clock-point anchor of Section 9.13 needs are cited (A_hfs from Arbes et
al. via Harty 2014, mu_I from Harty's thesis value, g_J as the plan uses it); the level energies are the
Ca II NIST values (the 43Ca isotope shift, of order 1 GHz on the 397 nm line, is below the 0.01 cm^-1 the
table prints and is not resolved here). The hyperfine constants of the excited levels are not in PLAN.md,
so ``species()`` raises :class:`IncompleteSpeciesTable` until they are cited.
"""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable, MissingConstant

NAME = "43Ca+"
_c = cited_factory("ca43.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 42.95876644, "u", "NIST_AWIC", uncertainty=2.4e-7),
    _c("nuclear_spin", 3.5, "hbar", "Harty2014", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        -1.31535,
        "mu_N",
        "Harty2014",
        tag="verified",
        note="a thesis value 0.17% from the handbook number that fits f0 nine times better, effectively a fitted "
        "constant (PLAN.md 4.5.6); the sign is what makes A_hfs negative and F = 3 lie above F = 4",
    ),
    _c(
        "S12.A_hfs_hz",
        -806.4020716e6,
        "Hz",
        "Arbes1994_via_Harty2014",
        tag="verified",
        note="zero-field splitting 3225.608 MHz = |A|(I + 1/2); SIGNED negative (inverted multiplet, Section 4.5.6); "
        "with A > 0 the clock point exists nowhere between 1 and 4000 G (ingest test of the sign)",
    ),
    _c(
        "S12.g_J",
        2.00225664,
        "",
        "PLAN_9_13",
        note="the third Breit-Rabi constant Harty et al. never print (PLAN.md 4.5.6: 'g_J = 2.000 moves the "
        "field-independent point by 0.16 G and f0 by 48 Hz'); the primary source is not named in the plan",
    ),
    _c(
        "D32.energy_cm",
        13650.19,
        "cm^-1",
        "NIST_ASD_5_12",
        note="Ca II value; 43Ca isotope shift not resolved",
    ),
    _c(
        "D52.energy_cm",
        13710.88,
        "cm^-1",
        "NIST_ASD_5_12",
        note="Ca II value; 43Ca isotope shift not resolved",
    ),
    _c(
        "P12.energy_cm",
        25191.51,
        "cm^-1",
        "NIST_ASD_5_12",
        note="Ca II value; 43Ca isotope shift not resolved",
    ),
    _c(
        "P32.energy_cm",
        25414.40,
        "cm^-1",
        "NIST_ASD_5_12",
        note="Ca II value; 43Ca isotope shift not resolved",
    ),
    _c(
        "P.linewidth_ozeri_hz",
        22.5e6,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I gamma/2pi for Ca+ (PLAN.md 4.3.2 stores Ozeri's Table I constants); compare the 40Ca+ "
        "table's quoted 21.57 and 23.4 MHz",
    ),
    _c(
        "S12.hfs_splitting_ozeri_hz",
        3.23e9,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_0/2pi, rounded",
    ),
    _c(
        "fine_structure_splitting_hz",
        6.68e12,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_f/2pi; the NIST levels give 222.89 cm^-1 = 6.682 THz",
    ),
    _c(
        "wavelength_ozeri_397_m",
        396.8e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I; an AIR wavelength (the NIST vacuum value is 396.959 nm)",
    ),
    _c(
        "wavelength_ozeri_393_m",
        393.4e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I; an AIR wavelength (the NIST vacuum value is 393.478 nm)",
    ),
    _c(
        "P_to_D_branching_inverse",
        17.0,
        "",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I f^-1 = 17 for 43Ca+",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant("hyperfine A (and B where J >= 1) of P1/2, P3/2, D3/2 and D5/2", "not in PLAN.md"),
    MissingConstant(
        "P1/2 and P3/2 lifetimes (isotope independent; the 40Ca+ readings are contested)", "not in PLAN.md"
    ),
    MissingConstant("43Ca+ isotope shifts of the level energies", "not in PLAN.md"),
)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`: the excited-level hyperfine constants are not cited yet."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
