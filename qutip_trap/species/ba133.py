"""133Ba+ species table (PLAN.md Sections 8.1, 8.4, 9.13; Appendix E) - PARTIAL (radioactive, I = 1/2).

Two corrections from the M0a fix pass of 2026-09-08. (1) ``S12.A_hfs_hz`` was stored POSITIVE. mu_I(133Ba)
is negative, so A(6s 2S1/2) is NEGATIVE and the ground-state hyperfine structure is INVERTED: F = 0 lies
above F = 1, and the |F=0, mF=0> qubit state is the UPPER one (Hucul et al. 2017 Table I lists it negative;
Christensen et al. 2020 Fig. 1 draws F = 0 on top). Because I = J = 1/2 makes the splitting equal to |A|,
the magnitude was right and only the sign of the level ordering was wrong -- the same failure class the
``inverted`` flag guards against in ca43.py and be9.py. (2) The 12-figure value had no provenance outside
this repository; the primary is Knab, Schupp and Werth, Europhys. Lett. 4, 1361 (1987), NOT the
Knab-Knoll-Scheerer-Werth Z. Phys. D 25, 205 (1993) g_J paper.

The species-gaps pass of the same day closed two of the four remaining gaps and hardened the other two.
Closed: the measured g_J(6s 2S1/2) = 2.00249192(3) of Marx, Tommaseo and Werth 1998, and
A(6p 2P1/2) = -1840(11) MHz, which Hucul et al. 2017 Table I prints unbolded (a literature value they
relay) and which was therefore already inside a source this table cites. The two D-state g_J were read off
the NIST ASD Lande column (0.79 and 1.12); a direct ASD re-query confirms the transcription was faithful
but the 1.12 is ASD's OWN error, 6.7% from four independent determinations, so both are replaced by
measured values and kept beside them as ``*_g_J_asd`` cross-checks tagged ``contested``. STILL MISSING, and
the reason is the literature and not this repository: A(6p 2P3/2) and A(5d 2D5/2) are printed by NOBODY.
Christensen et al. 2020 measure only the SPLITTINGS 623(30) and 83(30) MHz and print neither constant nor
its sign, so ``_CONSULT`` records the negative search rather than inviting a repeat of it. Both derived
uncertainties in those notes were understated -- (100) and (67) where the splittings are (30), i.e.
-311.5(150) and +27.7(100) MHz -- and are corrected.
"""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Species
from qutip_trap.species.table import (
    IncompleteSpeciesTable,
    MissingConstant,
    required_constants_missing,
)

NAME = "133Ba+"
_c = cited_factory("ba133.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 132.9060074, "u", "NIST_AWIC", uncertainty=1.1e-6),
    _c("nuclear_spin", 0.5, "hbar", "Christensen2020", tag="verified"),
    _c(
        "S12.hfs_splitting_hz",
        9925.45355459e6,
        "Hz",
        "Knab1987",
        tag="verified",
        uncertainty=1e-4,
        note="Delta nu(F = 1 <-> 0)/h = 9 925 453 554.59(10) Hz, the qubit frequency; a POSITIVE splitting, "
        "which is the only thing a splitting can be",
    ),
    _c(
        "S12.A_hfs_hz",
        -9925.45355459e6,
        "Hz",
        "Hucul2017",
        tag="corrected",
        uncertainty=1e-4,
        note="A = -(splitting) for I = J = 1/2, and the sign is NEGATIVE because mu_I(133Ba) < 0: the "
        "multiplet is INVERTED, F = 0 lies above F = 1, and the |F=0, mF=0> qubit state is the UPPER one. "
        "Stored positive until 2026-09-08. Arithmetic cross-check: A(133)/A(137) = -2.469712 against the "
        "(mu_I/I) ratio -2.469686, agreeing to 1e-5",
    ),
    _c(
        "mu_I_nuclear_magnetons",
        -0.77167,
        "mu_N",
        "Stone2019",
        tag="verified",
        uncertainty=2e-5,
        note="-0.77167(2) mu_N, I = 1/2 so Q = 0 identically; Stone's compilation cites Knab, Schupp and "
        "Werth 1987 (1987Kn10) for it. The sign is what inverts the ground-state multiplet",
    ),
    _c(
        "S12.g_J",
        2.00249192,
        "",
        "Marx1998",
        tag="extracted",
        uncertainty=3e-8,
        note="MEASURED in a Penning trap, but on 138Ba+ and 135Ba+, and ISOTOPE-INDEPENDENT to the quoted "
        "3e-8 -- an assumption the field states rather than a measurement on 133Ba+ (Hanley et al., "
        "arXiv:2105.10352: 'One would expect the isotopic dependence of g_J to be smaller than the "
        "experimental measurement uncertainty, based upon similar measurements using Ba+ isotopes'). The "
        "SAME number is stored in ba137.py, and the note there says the same thing. Tagged extracted, not "
        "verified, because Marx et al. 1998 itself could not be read (Springer redirects to an identity "
        "provider and there is no open-access copy): the digits are read verbatim in the full text of "
        "Arnold et al., Phys. Rev. Lett. 124, 193001 (2020), which uses g_S = 2.002 491 92(3) twice. "
        "Supersedes the 2.0024906 of Hubrich et al. 1991 that still circulates, by about 40x, and it is "
        "NOT the 2.002615 of 171Yb+ that validation/scripts/check_atomic.py substitutes for want of a "
        "cited value",
    ),
    _c(
        "D32.A_hfs_hz",
        -468.5e6,
        "Hz",
        "Hucul2017",
        tag="verified",
        uncertainty=20e6,
        note="Table I: -468.5(1.5)stat +- 20sys MHz, measured there (splitting 937(3) MHz). The systematic "
        "error dominates and is what is carried",
    ),
    _c(
        "P12.A_hfs_hz",
        -1840e6,
        "Hz",
        "Hucul2017",
        tag="verified",
        uncertainty=11e6,
        note="A = -(splitting) for I = J = 1/2, NEGATIVE because mu_I(133Ba) < 0. Read directly in Hucul et "
        "al. 2017 Table I, whose A = 133 row prints -1840(11) MHz WITHOUT boldface, i.e. as a literature "
        "value they tabulate (from Hoehle et al., Phys. Lett. B 62, 390 (1976), itself unreadable behind a "
        "ScienceDirect 403) rather than as their own measurement. Hucul's own measurement of the same "
        "structure is the SPLITTING Delta_2 = 1840(2)stat +- 20sys MHz (Fig. 2b), equal to |A| here, which "
        "corroborates the magnitude independently. Arithmetic cross-check: scaling the 137Ba+ "
        "A(6p 2P1/2) = 743.7(3) MHz by (mu_I/I)(133)/(mu_I/I)(137) = -2.4697 predicts -1836.7 MHz, inside "
        "the 11 MHz bar. Until 2026-09-08 this key was a named gap whose _CONSULT said the primary was "
        "unreadable; the value was in a source this repository already cites",
    ),
    _c(
        "P32.hfs_splitting_hz",
        623e6,
        "Hz",
        "Christensen2020",
        tag="corrected",
        uncertainty=30e6,
        note="the off-resonant shelving error is detuned by this splitting (PLAN.md 8.1)",
    ),
    _c(
        "P32.branching_to_S12",
        0.74,
        "",
        "Christensen2020",
        tag="corrected",
        note="P3/2 -> S1/2 / D5/2 / D3/2 = 0.74 / 0.23 / 0.03 (PLAN.md 8.1)",
    ),
    _c("P32.branching_to_D52", 0.23, "", "Christensen2020", tag="corrected"),
    _c("P32.branching_to_D32", 0.03, "", "Christensen2020", tag="corrected"),
    _c(
        "D52.lifetime_s",
        30.0,
        "s",
        "PLAN_8_1",
        tag="corrected",
        note="'about 30 s' for Ba+ D5/2 (PLAN.md 8.1, 8.4); a rounded figure, not a measurement. The "
        "measurements are 30.14(40) s (Zhang et al. 2020) and 31.2(9) s (Auchter et al. 2014), both on "
        "138Ba+ and isotope-independent; see D52.lifetime_zhang_s",
    ),
    _c(
        "D52.lifetime_zhang_s",
        30.14,
        "s",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.40,
        note="the measured Ba+ D5/2 lifetime (138Ba+, isotope-independent); Auchter et al. 2014's 31.2(9) s "
        "agrees at 1.1 sigma. This is what the plan's rounded 30 s should become",
    ),
    _c(
        "D32.lifetime_s",
        79.8,
        "s",
        "Yu1997",
        tag="verified",
        uncertainty=4.6,
        note="measured on 138Ba+; Gurell et al., Phys. Rev. A 75, 052506 (2007) give 89(16) s",
    ),
    _c(
        "P12.lifetime_s",
        7.855e-9,
        "s",
        "Arnold2019",
        tag="verified",
        uncertainty=0.010e-9,
        note="gamma/2pi = 20.263 MHz total; measured on 138Ba+, isotope-independent",
    ),
    _c(
        "P32.lifetime_s",
        6.2615e-9,
        "s",
        "ZhangBa2020",
        tag="verified",
        uncertainty=0.0072e-9,
        note="gamma/2pi = 25.418 MHz total; Arnold et al. 2019 give 6.271(8) ns",
    ),
    _c(
        "P12.branching_to_D32",
        0.268177,
        "",
        "Arnold2019",
        tag="verified",
        uncertainty=0.000057,
        note="649.9 nm; the S1/2 share is 0.731823(57) and the two sum to 1 exactly. De Munshi et al., "
        "Phys. Rev. A 91, 040501(R) (2015) give 0.2696(4), about 3 sigma away -- do not mix the two groups' "
        "sets",
    ),
    _c("D32.energy_cm", 4873.852, "cm^-1", "NIST_ASD_5_12", note="Ba II value; isotope shift not resolved"),
    _c("D52.energy_cm", 5674.807, "cm^-1", "NIST_ASD_5_12"),
    _c("P12.energy_cm", 20261.561, "cm^-1", "NIST_ASD_5_12", note="493.55 nm vacuum"),
    _c("P32.energy_cm", 21952.404, "cm^-1", "NIST_ASD_5_12", note="455.53 nm vacuum"),
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
        "(kept as D52.g_J_asd), which the M0a pass had flagged as a suspected transcription slip for 1.20; "
        "an ASD re-query on 2026-09-08 shows the 1.12 is ASD's OWN error. Chin. Phys. Lett. 42, 063101 "
        "(2025) misquotes the uncertainty as (14); the PRL says (24)",
    ),
    _c(
        "D32.g_J_asd",
        0.79,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="CROSS-CHECK ONLY, not an input. The NIST ASD Lande column for Ba II was re-queried on "
        "2026-09-08 and returns literally 0.79 (5d 2D3/2), 1.12 (5d 2D5/2), 1.32 (6p 2P3/2) and BLANK for "
        "6s 2S1/2, so this repository transcribed ASD faithfully -- and the blank is why _CONSULT used to "
        "say 'NIST ASD lists none' for the ground state. It is 1.2% BELOW the measured 0.7993278(3) -- and "
        "NOT a rounding of it, which would be 0.80 -- while the same column's D5/2 entry is 6.7% below its "
        "measured value, so the whole column reads as an independent (and inaccurate) calculation and is "
        "used as an input nowhere",
    ),
    _c(
        "D52.g_J_asd",
        1.12,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="CROSS-CHECK ONLY, and DEMONSTRABLY WRONG: 6.7% from the measured 1.20036739(24) (Arnold et "
        "al. 2020), from the earlier measured 1.200372(4)(7) (Hoffman et al. 2013), from the LS Lande "
        "1.2005 and from the isoelectronic measured 1.2003340 of Ca+ 3d 2D5/2. NIST ASD is not a primary "
        "source for a g factor (it forwards to a reference), and the ASD re-query of 2026-09-08 settles "
        "which side the error is on -- it is ASD's own, not this repository's",
    ),
    _c(
        "P32.g_J",
        1.32,
        "",
        "NIST_ASD_5_12",
        tag="contested",
        note="CROSS-CHECK ONLY: no measured g_J of 6p 2P3/2 was located, and the LS Lande 4/3 is the honest "
        "substitute. The ASD Lande column this value comes from carries a 6.7% error at 5d 2D5/2 (see "
        "D52.g_J_asd), so its 1% offset from 4/3 cannot be read as a measurement",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

REQUIRED: dict[str, str] = {
    "ba133.mass_atomic_u": "relative atomic mass of the neutral atom",
    "ba133.mu_I_nuclear_magnetons": "signed nuclear moment",
    "ba133.S12.A_hfs_hz": "ground-state hyperfine A, signed",
    "ba133.S12.g_J": "measured ground-state g_J",
    "ba133.P12.A_hfs_hz": "6p 2P1/2 hyperfine A",
    "ba133.P32.A_hfs_hz": "6p 2P3/2 hyperfine A (no source prints it; only the 623(30) MHz splitting)",
    "ba133.D32.A_hfs_hz": "5d 2D3/2 hyperfine A",
    "ba133.D52.A_hfs_hz": "5d 2D5/2 hyperfine A (no source prints it; only the 83(30) MHz splitting)",
    "ba133.P12.lifetime_s": "6p 2P1/2 lifetime",
    "ba133.P32.lifetime_s": "6p 2P3/2 lifetime",
    "ba133.D32.lifetime_s": "5d 2D3/2 lifetime",
    "ba133.D52.lifetime_s": "5d 2D5/2 lifetime",
    "ba133.P12.branching_to_D32": "P1/2 branching into D3/2 (649.9 nm)",
    "ba133.P32.branching_to_D52": "P3/2 branching into D5/2 (614.3 nm)",
    "ba133.P32.branching_to_D32": "P3/2 branching into D3/2 (585.5 nm)",
    "ba133.D32.energy_cm": "5d 2D3/2 level energy",
    "ba133.D52.energy_cm": "5d 2D5/2 level energy",
    "ba133.P12.energy_cm": "6p 2P1/2 level energy",
    "ba133.P32.energy_cm": "6p 2P3/2 level energy",
}
"""The constants ``species()`` needs, so that :data:`MISSING` is DERIVED from :data:`TABLE` (audit item E21)."""

_CONSULT: dict[str, str] = {
    "ba133.P32.A_hfs_hz": "NO SOURCE PRINTS THIS CONSTANT -- only the splitting (re-searched 2026-09-08; do "
    "not repeat the search without a new lead). Christensen et al. 2020, the only paper that measures the "
    "6p 2P3/2 structure of 133Ba+, writes 'we find Delta_3 = 623(30) MHz' and prints no A and no sign; that "
    "splitting is stored as ba133.P32.hfs_splitting_hz. It gives A = -(623(30) MHz)/2 = -311.5(150) MHz for "
    "I = 1/2, J = 3/2, with the sign inferred from mu_I(133Ba) < 0 -- the uncertainty is (150), not the "
    "(100) this note carried until 2026-09-08, because the splitting is (30) and not (20). Corroboration, "
    "not a citation: scaling the 137Ba+ A(6p 2P3/2) = 127.2(2) MHz by (mu_I/I)(133)/(mu_I/I)(137) = "
    "-2.4697 predicts -314.1 MHz, inside the 15 MHz bar. Papers checked and negative: Hucul et al. 2017 "
    "(Table I has no P3/2 column), Arnold et al., Phys. Rev. A 100, 032503 (2019). The only plausible home "
    "for a first printed A is Hoehle, Huehnermann, Meier, Ihle, Wagner, Phys. Lett. B 62, 390 (1976), "
    "unreadable behind a ScienceDirect 403",
    "ba133.D52.A_hfs_hz": "NO SOURCE PRINTS THIS CONSTANT -- only the splitting (re-searched 2026-09-08; do "
    "not repeat the search without a new lead). Christensen et al. 2020 writes 'we find the 2D5/2 hyperfine "
    "splitting Delta_5 = 83(30) MHz' and prints no A and no sign. It gives A = +(83(30) MHz)/3 = "
    "+27.7(100) MHz for I = 1/2, J = 5/2 -- the uncertainty is (100), not the (67) this note carried until "
    "2026-09-08, because the splitting is (30) and not (20) -- and POSITIVE, unlike every other 133Ba+ "
    "level, because mu_I flips relative to 137Ba+. Corroboration, not a citation: scaling the 137Ba+ "
    "A(5d 2D5/2) = -12.029234(11) MHz by -2.4697 predicts +29.7 MHz, inside the 10 MHz bar, which is what "
    "supports the sign flip. Same negative search as the P3/2 row above",
}

MISSING: tuple[MissingConstant, ...] = required_constants_missing(TABLE, REQUIRED, _CONSULT)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
