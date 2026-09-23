"""43Ca+ species table (I = 7/2; mu_I < 0 inverts every hyperfine multiplet).

Level energies are the Ca II values (the 43Ca isotope shift is below their resolution); the branchings and the
D lifetimes are the isotope-independent 40Ca+ measurements.
"""

from __future__ import annotations

from fractions import Fraction

from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.table import (
    Cited,
    IncompleteSpeciesTable,
    MissingConstant,
    energy_hz,
    gamma_hz_from_lifetime,
    ion_mass_u,
    required_constants_missing,
    wavelength_vac_m,
)
from qutip_trap.units import lande_g_j

NAME = "43Ca+"
_c = cited_factory("ca43.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 42.95876644, "u", "NIST_AWIC", uncertainty=2.4e-7),
    _c("nuclear_spin", 3.5, "hbar", "Harty2014", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        -1.31535,
        "mu_N",
        "Hanley2021",
        tag="verified",
        uncertainty=9e-6,
        note="the FREE-ION moment (uncorrected for shielding by the ion's bound electrons), -1.315350(9)(1) "
        "mu_N, which is the moment a trapped-ion Breit-Rabi Hamiltonian takes and which reproduces the value "
        "PLAN.md 4.5.6 called 'a thesis value 0.17% from the handbook number that fits f0 nine times better, "
        "effectively a fitted constant' -- it is not fitted, it is the free-ion convention. Stone's "
        "bare-nucleus -1.31733(6) mu_N is a different quantity (Section 13's mu_I row). The sign is what "
        "makes A_hfs negative and F = 3 lie above F = 4",
    ),
    _c(
        "S12.A_hfs_hz",
        -806.4020716e6,
        "Hz",
        "Arbes1994",
        tag="verified",
        uncertainty=8e-2,
        note="zero-field splitting 3225.608 MHz = |A|(I + 1/2); SIGNED negative (inverted multiplet, Section 4.5.6); "
        "with A > 0 the clock point exists nowhere between 1 and 4000 G (ingest test of the sign)",
    ),
    _c(
        "S12.g_J",
        2.00225664,
        "",
        "Tommaseo2003",
        tag="extracted",
        uncertainty=9e-8,
        note="the third Breit-Rabi constant Harty et al. never print (PLAN.md 4.5.6: 'g_J = 2.000 moves the "
        "field-independent point by 0.16 G and f0 by 48 Hz'). PLAN.md 9.13 carries the value with no source; "
        "the primary is Tommaseo et al. 2003's Penning-trap measurement (audit item E25). MEASURED ON "
        "40Ca+, NOT ON 43Ca+, and applied here as isotope-independent to the quoted 9e-8 -- which is an "
        "assumption the field states, not a measurement on 43Ca+. Downgraded from verified to extracted on "
        "2026-09-08: EPJD 25, 113 was not read directly, and the digits reach this table through Hanley et "
        "al., arXiv:2105.10352, verbatim -- 'Tommaseo et al. measured g_J = 2.002 256 64(+-0.000 000 09) "
        "using double-resonance spectroscopy of 40Ca+ ions in a Penning trap. One would expect the isotopic "
        "dependence of g_J to be smaller than the experimental measurement uncertainty, based upon similar "
        "measurements using Ba+ isotopes [Marx 1998]' -- so the tag names two papers, as the same isotope "
        "transfer does for Ba+ (ba133.S12.g_J, ba137.S12.g_J). Hanley et al. then use this g_J in their own "
        "43Ca+ Breit-Rabi fit, which is the transfer this table makes",
    ),
    _c(
        "P12.A_hfs_hz",
        -145.4e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=0.1e6,
        note="inverted, as every 43Ca+ level is (mu_I < 0)",
    ),
    _c(
        "P32.A_hfs_hz",
        -31.0e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=0.2e6,
        note="-31.0(2) MHz, NOT the -31.4 that circulates: Garcia Ruiz et al., Phys. Rev. C 91, 041304(R) "
        "(2015) Table I prints the Noertershaeuser row as -31.0(2) beside Silverans 1991's -31.9(2), and "
        "-31.43(19) MHz is the 45Ca value in the same table",
    ),
    _c(
        "P32.B_hfs_hz",
        -6.9e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=1.7e6,
    ),
    _c(
        "D32.A_hfs_hz",
        -47.3e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=0.2e6,
    ),
    _c(
        "D32.B_hfs_hz",
        -3.7e6,
        "Hz",
        "Nortershauser1998",
        tag="verified",
        uncertainty=1.9e6,
    ),
    _c(
        "D52.A_hfs_hz",
        -3.8931e6,
        "Hz",
        "Benhelm2007",
        tag="verified",
        uncertainty=0.0002e6,
        note="the SIGNS come from the erratum Phys. Rev. A 75, 049901(E); the arXiv v1/v2 print both positive",
    ),
    _c(
        "D52.B_hfs_hz",
        -4.241e6,
        "Hz",
        "Benhelm2007",
        tag="verified",
        uncertainty=0.004e6,
    ),
    _c(
        "D52.g_J",
        1.2003340,
        "",
        "Chwalla2009",
        tag="verified",
        uncertainty=3e-7,
        note="measured on 40Ca+; g_J is isotope-independent. The Lande value is 1.2004639",
    ),
    _c(
        "P12.lifetime_s",
        6.904e-9,
        "s",
        "Hettrich2015",
        tag="verified",
        uncertainty=0.026e-9,
        note="the modern single-ion value, isotope-independent; Jin and Church 1993's 7.098(20) ns is 7 sigma "
        "above it and Hettrich states it disagrees with theory by more than 11 sigma. gamma/2pi = 23.0526 MHz "
        "TOTAL, of which 21.57(8) MHz is the partial rate into S1/2 (which is what PLAN.md 9.13's 'quoted "
        "21.57 MHz' turns out to be; the ca40 table reads it that way since 2026-09-08)",
    ),
    _c(
        "P32.lifetime_s",
        6.639e-9,
        "s",
        "Meir2020",
        tag="verified",
        uncertainty=0.042e-9,
        note="modern single-ion value; Jin and Church 1993's 6.924(19) ns is 6 sigma above it",
    ),
    _c(
        "P12.branching_to_D32",
        0.06435,
        "",
        "Ramm2013",
        tag="verified",
        uncertainty=0.00007,
        note="isotope-independent (an electronic branching ratio); Hettrich et al. 2015's independent "
        "0.06428(25) agrees. The 40Ca+ table adopted the same value on 2026-09-08, in place of Section 8.1's "
        "rounder 0.06 (ledger conv.ca40_branching), so the two Ca tables now agree",
    ),
    _c(
        "P32.branching_to_D52",
        0.0587,
        "",
        "Gerritsma2008",
        tag="verified",
        uncertainty=0.0002,
        note="854 nm; isotope-independent",
    ),
    _c(
        "P32.branching_to_D32",
        0.00661,
        "",
        "Gerritsma2008",
        tag="verified",
        uncertainty=0.00004,
        note="850 nm; the three printed fractions sum to 1.00001, so the S1/2 share is taken as the remainder",
    ),
    _c(
        "D32.lifetime_s",
        1.176,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.011,
        note="measured on 40Ca+; an E2/M1 lifetime is isotope-independent far below the 1% quoted precision",
    ),
    _c(
        "D52.lifetime_s",
        1.168,
        "s",
        "Kreuter2005",
        tag="verified",
        uncertainty=0.009,
        note="measured on 40Ca+; concordant with Barton et al. 2000",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        -0.0408,
        "b",
        "Stone2019",
        tag="verified",
        uncertainty=0.0008,
        note="Q(43Ca); the sign is why every B_hfs above is negative",
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

TABLE: dict[str, Cited] = {c.key: c for c in _ENTRIES}

REQUIRED: dict[str, str] = {
    "ca43.mass_atomic_u": "relative atomic mass of the neutral atom",
    "ca43.mu_I_nuclear_magnetons": "signed free-ion nuclear moment",
    "ca43.S12.A_hfs_hz": "ground-state hyperfine A, signed",
    "ca43.S12.g_J": "measured ground-state g_J",
    "ca43.P12.A_hfs_hz": "4p 2P1/2 hyperfine A",
    "ca43.P32.A_hfs_hz": "4p 2P3/2 hyperfine A",
    "ca43.P32.B_hfs_hz": "4p 2P3/2 hyperfine B",
    "ca43.D32.A_hfs_hz": "3d 2D3/2 hyperfine A",
    "ca43.D32.B_hfs_hz": "3d 2D3/2 hyperfine B",
    "ca43.D52.A_hfs_hz": "3d 2D5/2 hyperfine A",
    "ca43.D52.B_hfs_hz": "3d 2D5/2 hyperfine B",
    "ca43.D52.g_J": "measured 3d 2D5/2 g_J",
    "ca43.P12.lifetime_s": "4p 2P1/2 lifetime",
    "ca43.P32.lifetime_s": "4p 2P3/2 lifetime",
    "ca43.P12.branching_to_D32": "P1/2 branching into D3/2 (866 nm)",
    "ca43.P32.branching_to_D52": "P3/2 branching into D5/2 (854 nm)",
    "ca43.P32.branching_to_D32": "P3/2 branching into D3/2 (850 nm)",
    "ca43.D32.lifetime_s": "3d 2D3/2 lifetime",
    "ca43.D52.lifetime_s": "3d 2D5/2 lifetime",
    "ca43.D32.energy_cm": "3d 2D3/2 level energy",
    "ca43.D52.energy_cm": "3d 2D5/2 level energy",
    "ca43.P12.energy_cm": "4p 2P1/2 level energy",
    "ca43.P32.energy_cm": "4p 2P3/2 level energy",
}
"""The ids ``species()`` needs, each with a description; :data:`MISSING` is derived from it."""

_CONSULT: dict[str, str] = {}

MISSING: tuple[MissingConstant, ...] = required_constants_missing(TABLE, REQUIRED, _CONSULT)

_OPEN: tuple[MissingConstant, ...] = (
    MissingConstant(
        "43Ca+ isotope shifts of the level energies",
        "not in PLAN.md; Mueller et al., Phys. Rev. Research 2, 043351 (2020) may hold ~100 kHz values. The "
        "shift (order 1 GHz on the 397 nm line) is below the 0.01 cm^-1 = 300 MHz these energies resolve, so "
        "the Ca II values are used",
    ),
    MissingConstant(
        "measured g_J of 4p 2P1/2 and 4p 2P3/2",
        "no measurement exists; theory gives 0.665636 and 1.333861 (Sahoo, arXiv:1710.06558) against the "
        "Lande 0.665894 and 1.334106 used here and tagged [background]",
    ),
)
"""Declared gaps that do not block the build (the record uses a substitute for each)."""


def species() -> Species:
    """The ``Species`` record for 43Ca+, built from :data:`TABLE` alone."""
    if MISSING:
        raise IncompleteSpeciesTable(NAME, MISSING)
    t = TABLE
    half = Fraction(1, 2)
    e_d32 = energy_hz(t["ca43.D32.energy_cm"])
    e_d52 = energy_hz(t["ca43.D52.energy_cm"])
    e_p12 = energy_hz(t["ca43.P12.energy_cm"])
    e_p32 = energy_hz(t["ca43.P32.energy_cm"])
    s12 = Level(
        "S1/2",
        0.0,
        None,
        t["ca43.S12.A_hfs_hz"].value,
        0.0,
        t["ca43.S12.g_J"].value,
        ("Arbes1994", "Tommaseo2003", "Hanley2021"),
    )
    d32 = Level(
        "D3/2",
        e_d32,
        t["ca43.D32.lifetime_s"].value,
        t["ca43.D32.A_hfs_hz"].value,
        t["ca43.D32.B_hfs_hz"].value,
        lande_g_j(2, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "Nortershauser1998", "Kreuter2005", "PLAN_background"),
    )
    d52 = Level(
        "D5/2",
        e_d52,
        t["ca43.D52.lifetime_s"].value,
        t["ca43.D52.A_hfs_hz"].value,
        t["ca43.D52.B_hfs_hz"].value,
        t["ca43.D52.g_J"].value,
        ("NIST_ASD_5_12", "Benhelm2007", "Chwalla2009", "Kreuter2005"),
    )
    p12 = Level(
        "P1/2",
        e_p12,
        t["ca43.P12.lifetime_s"].value,
        t["ca43.P12.A_hfs_hz"].value,
        0.0,
        lande_g_j(1, half, half),
        ("NIST_ASD_5_12", "Nortershauser1998", "Hettrich2015", "PLAN_background"),
    )
    p32 = Level(
        "P3/2",
        e_p32,
        t["ca43.P32.lifetime_s"].value,
        t["ca43.P32.A_hfs_hz"].value,
        t["ca43.P32.B_hfs_hz"].value,
        lande_g_j(1, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "Nortershauser1998", "Meir2020", "PLAN_background"),
    )

    g_p12 = gamma_hz_from_lifetime(t["ca43.P12.lifetime_s"])
    g_p32 = gamma_hz_from_lifetime(t["ca43.P32.lifetime_s"])
    b12_d32 = t["ca43.P12.branching_to_D32"].value
    b32_d52 = t["ca43.P32.branching_to_D52"].value
    b32_d32 = t["ca43.P32.branching_to_D32"].value
    c12 = ("NIST_ASD_5_12", "Hettrich2015", "Ramm2013")
    c32 = ("NIST_ASD_5_12", "Meir2020", "Gerritsma2008")
    transitions = (
        Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), g_p12, 1.0 - b12_d32, "E1", c12),
        Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), g_p12, b12_d32, "E1", c12),
        Transition("S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), g_p32, 1.0 - b32_d52 - b32_d32, "E1", c32),
        Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), g_p32, b32_d52, "E1", c32),
        Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), g_p32, b32_d32, "E1", c32),
    )
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["ca43.mass_atomic_u"]),
        nuclear_spin=float(Fraction(t["ca43.nuclear_spin"].value).limit_denominator(2)),
        mu_I_nuclear_magnetons=t["ca43.mu_I_nuclear_magnetons"].value,
        levels=(s12, d32, d52, p12, p32),
        transitions=transitions,
        # the |4,0> <-> |3,+1> clock qubit (field-independent at 146.0942 G)
        qubit=("S1/2 F=4 mF=0", "S1/2 F=3 mF=1"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-P3/2",
    )


__all__ = ["MISSING", "NAME", "TABLE", "species"]
