"""137Ba+ species table (I = 3/2).

The measured g_J of 6s 2S1/2, 5d 2D3/2 and 5d 2D5/2 come from 138Ba+ and 135Ba+ and are taken as isotope-independent;
the 6p g_J are the LS Lande values. The qubit pair is this package's choice: no published clock point pins it.
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

NAME = "137Ba+"
_c = cited_factory("ba137.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 136.90582714, "u", "NIST_AWIC", uncertainty=3.0e-7),
    _c(
        "nuclear_spin",
        1.5,
        "hbar",
        "BlattWerth1982",
        tag="verified",
        note="I = 3/2, from the F = 1, 2 hyperfine manifolds Blatt and Werth resolve; PLAN.md does not print it",
    ),
    _c(
        "mu_I_nuclear_magnetons",
        0.937365,
        "mu_N",
        "Stone2019",
        tag="verified",
        uncertainty=2e-5,
        note="+0.937365(20) mu_N (g_I = 0.6250(1)); POSITIVE, which is what puts F = I + 1/2 above "
        "F = I - 1/2 in Ba+, unlike 9Be+, 43Ca+ and 133Ba+. Quoted in Lewty et al. 2012; the IAEA/Stone "
        "compilation gives 0.9375(2)",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        0.236,
        "b",
        "Stone2019",
        tag="contested",
        uncertainty=0.003,
        note="Q(137Ba) = 0.236(3) b (Mertzimekis, Stamou, Psaltis, Nucl. Instrum. Methods A 807, 56 (2016)); "
        "Lewty et al. 2013 use 0.246(1) b, a 4% spread between compilations",
    ),
    _c(
        "S12.hfs_splitting_hz",
        8_037_741_667.69,
        "Hz",
        "BlattWerth1982",
        tag="verified",
        uncertainty=3.6e-4,
        note="W_hfs(F = 2 <-> 1)/h; the 137Ba+ microwave clock frequency",
    ),
    _c(
        "S12.A_hfs_hz",
        4018.87083385e6,
        "Hz",
        "BlattWerth1982",
        tag="verified",
        uncertainty=1.8e-4,
        note="A = W_hfs/(I + 1/2) = W_hfs/2 for I = 3/2, J = 1/2; POSITIVE because mu_I > 0. The "
        "uncertainty is (18), not the (20) that circulates",
    ),
    _c(
        "S12.g_J",
        2.00249192,
        "",
        "Marx1998",
        tag="extracted",
        uncertainty=3e-8,
        note="MEASURED in a Penning trap, but on 138Ba+ and 135Ba+, and ISOTOPE-INDEPENDENT to the quoted "
        "3e-8 -- an assumption the field states rather than a measurement on 137Ba+ (Hanley et al., "
        "arXiv:2105.10352: 'One would expect the isotopic dependence of g_J to be smaller than the "
        "experimental measurement uncertainty, based upon similar measurements using Ba+ isotopes'). "
        "Tagged extracted, not verified, because Marx et al. 1998 itself could not be read (Springer "
        "redirects to an identity provider and there is no open-access copy): the digits are read verbatim "
        "in the full text of Arnold et al., Phys. Rev. Lett. 124, 193001 (2020), which uses "
        "g_S = 2.002 491 92(3) twice. Supersedes the 2.0024906 of Hubrich et al. 1991 that still "
        "circulates, by about 40x. This is the constant whose absence made the table refuse to build "
        "until 2026-09-08",
    ),
    _c(
        "P12.A_hfs_hz",
        743.7e6,
        "Hz",
        "Villemoes1993",
        tag="verified",
        uncertainty=0.3e6,
    ),
    _c(
        "P32.A_hfs_hz",
        127.2e6,
        "Hz",
        "Villemoes1993",
        tag="verified",
        uncertainty=0.2e6,
        note="the paper was not read directly (IOP bot wall); this pair rests on one faithful secondary "
        "compilation, whose 135Ba+ Villemoes row matches an independent quotation",
    ),
    _c(
        "P32.B_hfs_hz",
        92.5e6,
        "Hz",
        "Villemoes1993",
        tag="verified",
        uncertainty=0.2e6,
    ),
    _c(
        "D32.A_hfs_hz",
        189.731494e6,
        "Hz",
        "Lewty2013",
        tag="verified",
        uncertainty=1.7e-2,
        note="Tables II and VI, read directly. The 189.7300(6) MHz that circulates matches NO source; the "
        "1980s Leuven values are 189.7296(7) (Itano 2006 Table VII, attributed to Silverans et al. 1986) and "
        "189.7288(6) (Hucul et al. 2017), and Lewty supersedes both by about 1000x",
    ),
    _c(
        "D32.B_hfs_hz",
        44.537594e6,
        "Hz",
        "Lewty2013",
        tag="verified",
        uncertainty=3.4e-2,
        note="the 44.5408(17) MHz that circulates is Silverans et al. 1986 via Itano 2006",
    ),
    _c(
        "D52.A_hfs_hz",
        -12.029234e6,
        "Hz",
        "Lewty2013",
        tag="verified",
        uncertainty=1.1e-2,
        note="NEGATIVE, unlike every other 137Ba+ level; supersedes Silverans et al. 1986's -12.028(11) MHz "
        "(which circulates misattributed to Villemoes 1993)",
    ),
    _c(
        "D52.B_hfs_hz",
        59.525520e6,
        "Hz",
        "Lewty2013",
        tag="verified",
        uncertainty=1.10e-1,
    ),
    _c(
        "D52.octupole_moment_mu_N_b",
        0.05057,
        "",
        "Lewty2013",
        tag="verified",
        uncertainty=0.00054,
        note="the nuclear magnetic octupole moment Omega in mu_N b; the Hamiltonian of Section 4.5.1 carries "
        "only A and B, so this is a declared omission rather than an input (the C constants are 29.533(86) Hz "
        "for D3/2 and -12.41(77) Hz for D5/2)",
    ),
    _c(
        "P12.lifetime_s",
        7.855e-9,
        "s",
        "Arnold2019",
        tag="verified",
        uncertainty=0.010e-9,
        note="measured on 138Ba+, isotope-independent; gamma/2pi = 20.263 MHz total",
    ),
    _c(
        "P32.lifetime_s",
        6.2615e-9,
        "s",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.0072e-9,
        note="gamma/2pi = 25.418 MHz total",
    ),
    _c(
        "P12.branching_to_D32",
        0.268177,
        "",
        "Arnold2019",
        tag="verified",
        uncertainty=0.000057,
    ),
    _c(
        "P32.branching_to_D52",
        0.230253,
        "",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.000061,
        note="0.230, NOT the 0.231 that circulates; with 0.741716(71) into S1/2 and 0.028031(23) into D3/2 "
        "the three sum to 1.0000. Ozeri's Table I f^-1 = 3 for 137Ba+ corresponds to a total P -> D branching "
        "of 1/3, against the measured 0.2583",
    ),
    _c(
        "P32.branching_to_D32",
        0.028031,
        "",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.000023,
    ),
    _c(
        "D52.lifetime_s",
        30.14,
        "s",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.40,
        note="measured on 138Ba+; Auchter et al. 2014's 31.2(9) s agrees at 1.1 sigma",
    ),
    _c(
        "D32.lifetime_s",
        79.8,
        "s",
        "Yu1997",
        tag="verified",
        uncertainty=4.6,
        note="Gurell et al. 2007 give 89(16) s",
    ),
    _c(
        "P_to_D_branching_inverse",
        3.0,
        "",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I f^-1 = 3 for 137Ba+",
    ),
    _c("D32.energy_cm", 4873.852, "cm^-1", "NIST_ASD_5_12", note="Ba II value; isotope shift not resolved"),
    _c("D52.energy_cm", 5674.807, "cm^-1", "NIST_ASD_5_12"),
    _c("P12.energy_cm", 20261.561, "cm^-1", "NIST_ASD_5_12"),
    _c("P32.energy_cm", 21952.404, "cm^-1", "NIST_ASD_5_12"),
    _c(
        "D32.g_J",
        0.7993278,
        "",
        "Knoll1996",
        tag="extracted",
        uncertainty=3e-7,
        note="MEASURED in a Penning trap on Ba+ (Knoell et al. 1996), replacing the NIST ASD Lande column's "
        "two-digit 0.79 that this table carried until 2026-09-08. Tagged extracted because PRA 54, 1199 "
        "could not be read (APS 403): the digits are read in the Chin. Phys. Lett. 42, 063101 (2025) "
        "abstract, which quotes '0.7993278(3) [Phys. Rev. A 54 1199 (1996)]' against its own MCDHF "
        "0.7993961(126). g_J is isotope-independent to this precision",
    ),
    _c(
        "D52.g_J",
        1.20036739,
        "",
        "ArnoldPRL2020",
        tag="verified",
        uncertainty=2.4e-8,
        note="MEASURED on 138Ba+ as the Zeeman-splitting ratio 0.59943681(12) times Marx's "
        "g_S = 2.00249192(3); read directly in arXiv:1906.09150v2. ISOTOPE-INDEPENDENT to this "
        "precision, like the other two Ba+ g factors. Replaces the ASD Lande column's 1.12 "
        "(kept as D52.g_J_asd), which is 6.7% away. Chin. Phys. Lett. 42, 063101 (2025) misquotes the "
        "uncertainty as (14); the PRL says (24)",
    ),
    _c(
        "D32.g_J_asd",
        0.79,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="CROSS-CHECK ONLY, not an input. The NIST ASD Lande column for Ba II was re-queried on "
        "2026-09-08 and returns literally 0.79 (5d 2D3/2), 1.12 (5d 2D5/2), 1.32 (6p 2P3/2) and BLANK for "
        "6s 2S1/2, so this repository transcribed ASD faithfully. It is 1.2% BELOW the measured "
        "0.7993278(3) -- and NOT a rounding of it, which would be 0.80 -- while the same column's D5/2 "
        "entry is 6.7% below its measured value, so the whole column reads as an independent (and "
        "inaccurate) calculation and is used as an input nowhere",
    ),
    _c(
        "D52.g_J_asd",
        1.12,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="CROSS-CHECK ONLY, and DEMONSTRABLY WRONG: 6.7% from the measured 1.20036739(24) (Arnold et "
        "al. 2020), from the earlier measured 1.200372(4)(7) (Hoffman et al. 2013), from the LS Lande "
        "1.2005 and from the isoelectronic measured 1.2003340 of Ca+ 3d 2D5/2. The M0a pass of 2026-09-08 "
        "flagged it as a suspected transcription slip for 1.20; the ASD re-query settles it -- the error is "
        "ASD's own, not this repository's",
    ),
    _c(
        "P32.g_J",
        1.32,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="CROSS-CHECK ONLY: no measured g_J of 6p 2P3/2 was located, and species() uses the LS Lande "
        "4/3 tagged [background] as ca43.py and be9.py do for their P levels. The ASD Lande column this "
        "value comes from carries a 6.7% error at 5d 2D5/2 (see D52.g_J_asd), so its 1% offset from 4/3 "
        "cannot be read as a measurement",
    ),
)

TABLE: dict[str, Cited] = {c.key: c for c in _ENTRIES}

REQUIRED: dict[str, str] = {
    "ba137.mass_atomic_u": "relative atomic mass of the neutral atom",
    "ba137.mu_I_nuclear_magnetons": "signed nuclear moment",
    "ba137.S12.A_hfs_hz": "ground-state hyperfine A, signed",
    "ba137.S12.g_J": "measured ground-state g_J",
    "ba137.P12.A_hfs_hz": "6p 2P1/2 hyperfine A",
    "ba137.P32.A_hfs_hz": "6p 2P3/2 hyperfine A",
    "ba137.P32.B_hfs_hz": "6p 2P3/2 hyperfine B",
    "ba137.D32.A_hfs_hz": "5d 2D3/2 hyperfine A",
    "ba137.D32.B_hfs_hz": "5d 2D3/2 hyperfine B",
    "ba137.D52.A_hfs_hz": "5d 2D5/2 hyperfine A",
    "ba137.D52.B_hfs_hz": "5d 2D5/2 hyperfine B",
    "ba137.D32.g_J": "measured 5d 2D3/2 g_J",
    "ba137.D52.g_J": "measured 5d 2D5/2 g_J",
    "ba137.P12.lifetime_s": "6p 2P1/2 lifetime",
    "ba137.P32.lifetime_s": "6p 2P3/2 lifetime",
    "ba137.D32.lifetime_s": "5d 2D3/2 lifetime",
    "ba137.D52.lifetime_s": "5d 2D5/2 lifetime",
    "ba137.P12.branching_to_D32": "P1/2 branching into D3/2 (649.9 nm)",
    "ba137.P32.branching_to_D52": "P3/2 branching into D5/2 (614.3 nm)",
    "ba137.P32.branching_to_D32": "P3/2 branching into D3/2 (585.5 nm)",
    "ba137.D32.energy_cm": "5d 2D3/2 level energy",
    "ba137.D52.energy_cm": "5d 2D5/2 level energy",
    "ba137.P12.energy_cm": "6p 2P1/2 level energy",
    "ba137.P32.energy_cm": "6p 2P3/2 level energy",
}
"""The ids ``species()`` needs, each with a description; :data:`MISSING` is derived from it."""

_CONSULT: dict[str, str] = {}

MISSING: tuple[MissingConstant, ...] = required_constants_missing(TABLE, REQUIRED, _CONSULT)

_OPEN: tuple[MissingConstant, ...] = (
    MissingConstant(
        "a g_J measured on 137Ba+ itself",
        "none exists: Marx, Tommaseo and Werth 1998 measured 138Ba+ and 135Ba+ (their 137Ba+ work is g_I), "
        "and the transfer to 137Ba+ is the field's stated expectation that the isotopic dependence lies "
        "below 3e-8 (Hanley et al., arXiv:2105.10352). Ledger anchor.species.ba_g_factors",
    ),
    MissingConstant(
        "measured g_J of 6p 2P1/2 and 6p 2P3/2",
        "no measurement located; the LS Lande values 2/3 and 4/3 are used and tagged [background]. Poulsen "
        "and Ramanujam, Phys. Rev. A 14, 1463 (1976) is quoted for g_J(6p 2P1/2) = 0.672(6) but only "
        "through an abstract rendering, and the NIST ASD Lande column that prints 1.32 for 6p 2P3/2 also "
        "prints a 6.7%-wrong 1.12 for 5d 2D5/2, so it is not usable as an input",
    ),
    MissingConstant(
        "the 137Ba+ isotope shifts of the level energies",
        "not in PLAN.md; the Ba II values are used. Villemoes et al., J. Phys. B 26, 4289 (1993) measured "
        "the 130-138Ba II shifts, which are of order 1 GHz on the 493 nm line and so below the 0.001 cm^-1 "
        "= 30 MHz these energies resolve",
    ),
    MissingConstant(
        "a PLAN.md Section 9.13 clock-point anchor for 137Ba+",
        "the plan names 137Ba+ as a preset (line 139) and stores Ozeri's f^-1 = 3 for it (line 457) but "
        "prints no field-independent point, frequency or curvature, so the qubit pair declared in "
        "species() is this package's choice and the anchors that exist for 43Ca+, 9Be+ and 25Mg+ have no "
        "137Ba+ counterpart to pin",
    ),
)
"""Declared gaps that do not block the build (the record uses a substitute for each)."""


def species() -> Species:
    """The ``Species`` record for 137Ba+, built from :data:`TABLE` alone."""
    if MISSING:
        raise IncompleteSpeciesTable(NAME, MISSING)
    t = TABLE
    half = Fraction(1, 2)
    e_d32 = energy_hz(t["ba137.D32.energy_cm"])
    e_d52 = energy_hz(t["ba137.D52.energy_cm"])
    e_p12 = energy_hz(t["ba137.P12.energy_cm"])
    e_p32 = energy_hz(t["ba137.P32.energy_cm"])
    s12 = Level(
        "S1/2",
        0.0,
        None,
        t["ba137.S12.A_hfs_hz"].value,
        0.0,
        t["ba137.S12.g_J"].value,
        ("BlattWerth1982", "Marx1998", "Stone2019"),
    )
    d32 = Level(
        "D3/2",
        e_d32,
        t["ba137.D32.lifetime_s"].value,
        t["ba137.D32.A_hfs_hz"].value,
        t["ba137.D32.B_hfs_hz"].value,
        t["ba137.D32.g_J"].value,
        ("NIST_ASD_5_12", "Lewty2013", "Knoll1996", "Yu1997"),
    )
    d52 = Level(
        "D5/2",
        e_d52,
        t["ba137.D52.lifetime_s"].value,
        t["ba137.D52.A_hfs_hz"].value,
        t["ba137.D52.B_hfs_hz"].value,
        t["ba137.D52.g_J"].value,
        ("NIST_ASD_5_12", "Lewty2013", "ArnoldPRL2020", "ZhangBa2020"),
    )
    p12 = Level(
        "P1/2",
        e_p12,
        t["ba137.P12.lifetime_s"].value,
        t["ba137.P12.A_hfs_hz"].value,
        0.0,
        lande_g_j(1, half, half),
        ("NIST_ASD_5_12", "Villemoes1993", "Arnold2019", "PLAN_background"),
    )
    p32 = Level(
        "P3/2",
        e_p32,
        t["ba137.P32.lifetime_s"].value,
        t["ba137.P32.A_hfs_hz"].value,
        t["ba137.P32.B_hfs_hz"].value,
        lande_g_j(1, half, Fraction(3, 2)),
        ("NIST_ASD_5_12", "Villemoes1993", "ZhangBa2020", "PLAN_background"),
    )

    g_p12 = gamma_hz_from_lifetime(t["ba137.P12.lifetime_s"])
    g_p32 = gamma_hz_from_lifetime(t["ba137.P32.lifetime_s"])
    b12_d32 = t["ba137.P12.branching_to_D32"].value
    b32_d52 = t["ba137.P32.branching_to_D52"].value
    b32_d32 = t["ba137.P32.branching_to_D32"].value
    c12 = ("NIST_ASD_5_12", "Arnold2019")
    c32 = ("NIST_ASD_5_12", "ZhangBa2020")
    transitions = (
        Transition("S1/2", "P1/2", wavelength_vac_m(0.0, e_p12), g_p12, 1.0 - b12_d32, "E1", c12),
        Transition("D3/2", "P1/2", wavelength_vac_m(e_d32, e_p12), g_p12, b12_d32, "E1", c12),
        Transition("S1/2", "P3/2", wavelength_vac_m(0.0, e_p32), g_p32, 1.0 - b32_d52 - b32_d32, "E1", c32),
        Transition("D5/2", "P3/2", wavelength_vac_m(e_d52, e_p32), g_p32, b32_d52, "E1", c32),
        Transition("D3/2", "P3/2", wavelength_vac_m(e_d32, e_p32), g_p32, b32_d32, "E1", c32),
    )
    return Species(
        name=NAME,
        mass_u=ion_mass_u(t["ba137.mass_atomic_u"]),
        nuclear_spin=float(Fraction(t["ba137.nuclear_spin"].value).limit_denominator(2)),
        mu_I_nuclear_magnetons=t["ba137.mu_I_nuclear_magnetons"].value,
        levels=(s12, d32, d52, p12, p32),
        transitions=transitions,
        # a declared choice: the mF = 0 pair, lower state first (A > 0 puts F = 1 below F = 2)
        qubit=("S1/2 F=1 mF=0", "S1/2 F=2 mF=0"),
        cycling="S1/2-P1/2",
        repumps=("D3/2-P1/2", "D5/2-P3/2"),
        shelving="S1/2-P3/2",
    )


__all__ = ["MISSING", "NAME", "TABLE", "species"]
