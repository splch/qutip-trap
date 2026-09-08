"""25Mg+ species table (PLAN.md Sections 4.2.2, 4.4.5, 4.5.1, 9.13; Appendix E) - PARTIAL.

Still raising after the M0a fix pass of 2026-09-08, and the reason is a real gap rather than an
untranscribed number: **no measured g_J of the 25Mg+ ground state exists.** Itano and Wineland 1981 and
Brewer et al. 2019 both servo the field to the electronic transition, so they determine only the RATIO
g_I/g_J, which is what this table now stores; the 2.00226 the table used to carry was 9Be+'s measured
value transplanted, and the Section 9.13 212.78 G anchor depends on it at the 0.25 G level. Backing g_J
out of the ratio plus a tabulated mu_I does not work either: the two shielding conventions give 2.0045 or
fail the 9Be+ cross-check at 4e-4. The 3p hyperfine constants are likewise theory-only, and the 3p
lifetimes reach this table second-hand. Note also that mu_I here is the UNCORRECTED (shielded) moment
while 9Be+'s is the corrected one -- a per-table convention Section 13 requires be declared.
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

NAME = "25Mg+"
_c = cited_factory("mg25.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 24.985836976, "u", "NIST_AWIC", uncertainty=5.0e-8),
    _c("nuclear_spin", 2.5, "hbar", "ItanoWineland1981", tag="verified", note="I = 5/2"),
    _c(
        "mu_I_nuclear_magnetons",
        -0.85545,
        "mu_N",
        "Stone2019",
        tag="verified",
        uncertainty=8e-5,
        note="the UNCORRECTED (shielded) NMR moment, Alder and Yu, Phys. Rev. 82, 105 (1951) via Stone's "
        "compilation, which carries no diamagnetic-correction flag for this row -- no corrected 25Mg value "
        "exists. This is a DIFFERENT convention from 9Be+'s corrected -1.177432 mu_N in the same package "
        "(Section 13's mu_I row requires the convention be declared per table). Stone's corrected "
        "recommendation is -0.85533(3) mu_N; Brewer et al. 2019's g_I implies |mu_I| = 0.85471, 730 ppm below",
    ),
    _c(
        "S12.A_hfs_hz",
        -596.254376e6,
        "Hz",
        "ItanoWineland1981",
        tag="verified",
        uncertainty=5.4e-2,
        note="SIGNED negative; the value behind the 212.78 G clock-point anchor. PLAN.md 9.13 carries it "
        "without a source; the primary is Itano and Wineland 1981 (audit item E25). SUPERSEDED by Brewer et "
        "al. 2019's -596.254250949(45) MHz (2.3 sigma away, and Xu et al. 2017's -596.2542487(42) agrees with "
        "Brewer at 0.54 sigma), but kept as the value used so the plan's own anchor is reproducible",
    ),
    _c(
        "S12.A_hfs_brewer_hz",
        -596.254250949e6,
        "Hz",
        "Brewer2019",
        tag="verified",
        uncertainty=4.5e-5,
        note="Delta W/h = 1 788 762 752.85(13) Hz; the modern value, 2.3 sigma from Itano and Wineland 1981. "
        "A contested pair, not a convention choice",
    ),
    _c(
        "S12.g_I_over_g_J",
        9.299484e-5,
        "",
        "ItanoWineland1981",
        tag="verified",
        uncertainty=7.5e-9,
        note="the ONLY g-factor quantity measured for 25Mg+: the field is servoed to the electronic "
        "transition, so the absolute g_J never enters. Brewer et al. 2019's 9.299308313(60)e-5 is the modern "
        "value. g_J must NOT be back-solved from this plus a tabulated mu_I: the corrected and uncorrected "
        "conventions give 2.0045 or fail the 9Be+ cross-check at 4e-4 (audit of 2026-09-07)",
    ),
    _c(
        "S12.g_I_over_g_J_brewer",
        9.299308313e-5,
        "",
        "Brewer2019",
        tag="verified",
        uncertainty=6.0e-13,
    ),
    _c(
        "P12.A_hfs_theory_hz",
        -103.4e6,
        "Hz",
        "KaurSahoo2025",
        tag="background",
        uncertainty=0.5e6,
        note="THEORY (relativistic coupled cluster); Table X's experiment column is BLANK for both 3p rows, "
        "so no measured 25Mg+ 3p hyperfine constant exists. Sur et al. 2005 give |A| = 101.70 MHz UNSIGNED, "
        "which is where the +102.16 MHz that circulates comes from: since mu_I < 0 every 25Mg+ A is NEGATIVE",
    ),
    _c(
        "P32.A_hfs_theory_hz",
        -19.31e6,
        "Hz",
        "KaurSahoo2025",
        tag="background",
        uncertainty=0.05e6,
        note="THEORY; Sur et al. 2005's unsigned |A| = 18.89 MHz",
    ),
    _c(
        "P32.B_hfs_theory_hz",
        22.91e6,
        "Hz",
        "SurSahoo2005",
        tag="background",
        note="THEORY, printed UNSIGNED; taken positive because Q(25Mg) = +0.199(2) b, which is an inference "
        "and not a source statement",
    ),
    _c(
        "P12.lifetime_s",
        3.854e-9,
        "s",
        "Ansbacher1989",
        tag="extracted",
        uncertainty=0.030e-9,
        note="gamma/2pi = 41.30(32) MHz. NOT primary-verified: Elsevier blocks automated retrieval and the "
        "digits were read from Kaur et al. 2025 Table VII; the NIST ASD A_ki (2.57e8 s^-1, grade A+, <= 2%) "
        "agrees to about 1%",
    ),
    _c(
        "P32.lifetime_s",
        3.810e-9,
        "s",
        "Ansbacher1989",
        tag="extracted",
        uncertainty=0.040e-9,
        note="gamma/2pi = 41.77(44) MHz; same provenance caveat",
    ),
    _c(
        "nuclear_quadrupole_moment_b",
        0.199,
        "b",
        "Stone2019",
        tag="verified",
        uncertainty=0.002,
        note="Q(25Mg) > 0",
    ),
    _c(
        "P12.energy_cm",
        35669.31,
        "cm^-1",
        "NIST_ASD_5_12",
        note="280.3530 nm VACUUM (air 280.2704 nm); the table previously carried no wavelength at all",
    ),
    _c(
        "P32.energy_cm",
        35760.88,
        "cm^-1",
        "NIST_ASD_5_12",
        note="279.6352 nm VACUUM (air 279.5528 nm). The 3p fine-structure splitting measured on 24Mg+ is "
        "2 744 591.767(88) MHz (Batteiger et al. 2009); these energies give 2745.4 GHz, right to 3e-4",
    ),
    _c(
        "P_to_D_branching",
        0.0,
        "",
        "Ozeri2007",
        tag="verified",
        note="no D level below the P levels (Ozeri Table I)",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

REQUIRED: dict[str, str] = {
    "mg25.mass_atomic_u": "relative atomic mass of the neutral atom",
    "mg25.mu_I_nuclear_magnetons": "signed nuclear moment, with its shielding convention declared",
    "mg25.S12.A_hfs_hz": "ground-state hyperfine A, signed",
    "mg25.S12.g_J": "MEASURED ground-state g_J (no absolute measurement exists; only g_I/g_J is measured)",
    "mg25.P12.A_hfs_hz": "MEASURED 3p 2P1/2 hyperfine A (theory only today)",
    "mg25.P32.A_hfs_hz": "MEASURED 3p 2P3/2 hyperfine A (theory only today)",
    "mg25.P32.B_hfs_hz": "MEASURED 3p 2P3/2 hyperfine B (theory only today)",
    "mg25.P12.energy_cm": "3p 2P1/2 level energy",
    "mg25.P32.energy_cm": "3p 2P3/2 level energy",
    "mg25.P12.lifetime_s": "3p 2P1/2 lifetime",
    "mg25.P32.lifetime_s": "3p 2P3/2 lifetime",
}
"""The constants ``species()`` needs, so that :data:`MISSING` is DERIVED from :data:`TABLE` (audit item E21).

Four entries are deliberately unsatisfiable today, which is why 25Mg+ still raises: ``S12.g_J`` has no
absolute measurement anywhere in the literature (only the ratio ``S12.g_I_over_g_J``), and the three 3p
hyperfine constants are theory only (stored under ``*_theory_hz`` so that a required key is not silently
satisfied by a calculation).
"""

_CONSULT: dict[str, str] = {
    "mg25.S12.g_J": "NO SOURCE EXISTS, MEASURED OR COMPUTED -- this gap is CLOSED BY SEARCH (re-verified "
    "2026-09-08; do not repeat the search without a new lead). (1) No measurement of any Mg+ isotope: Itano "
    "and Wineland 1981 and Brewer et al. 2019 both servo the field to the electronic transition and "
    "determine only g_I/g_J, which is stored as mg25.S12.g_I_over_g_J; NIST ASD emits NO Lande column at "
    "all for Mg II (whereas the same query returns one for Ba II); the Werth-group Penning-trap g_J "
    "programme, traced through about 100 works citing Lindroth and Ynnerman 1993 and Tommaseo et al. 2003, "
    "has dedicated papers for Ba+, Ca+, Eu+ and Hg+ and none for Mg+; and Rehmert et al., arXiv:2508.15488 "
    "(quantum-logic g_J of 48Ti+) contains no occurrence of 'Mg'. (2) No theory value either: Lindroth and "
    "Ynnerman, Phys. Rev. A 47, 961 (1993) covers Li, Be+ and Ba+ ONLY; Veseth, J. Phys. B 16, 2891 (1983) "
    "and Phys. Rev. A 22, 803 (1980) cover alkali ATOMS; the Sahoo RCC group's dedicated g_j papers cover "
    "Ca+ (Phys. Rev. A 96, 012511), Cd+ (100, 042508) and Cd+/Yb+/Hg+ (102, 062824), the Lanzhou group's "
    "covers Ba+ (Chin. Phys. Lett. 42, 063101), and Kaur and Sahoo's own comprehensive Mg+ properties "
    "paper -- the one this table already cites for the 3p hyperfine constants, arXiv:2504.19515 -- computes "
    "NO g_J at all (zero occurrences of 'g_J', 'g factor' or 'g-factor' in 106 kB of extracted full text). "
    "(3) WHY the gap persists: every operational use of 25Mg+ senses the field through the hyperfine "
    "transition, which needs only A and g_I/g_J. NIST's 27Al+/25Mg+ clock (Marshall et al., "
    "arXiv:2504.13071) measures B_AC 'via hyperfine spectroscopy on the 25Mg+ ion' and no g_J appears "
    "anywhere in it. (4) Do NOT back g_J out of the ratio plus a tabulated mu_I: the two shielding "
    "conventions give 2.0045 or fail the 9Be+ cross-check at 4e-4. Closest remaining leads for a human: "
    "ref. [22] of Dickopf et al. 2024 (the calculation behind their g_s(9Be+); if its method reaches "
    "Na-like systems it is the natural home for a Mg+ value) and Werth, Sturm, Blaum, Zeeman Spectroscopy "
    "in Penning Traps, Adv. At. Mol. Opt. Phys. 67, 257 (2018), doi:10.1016/bs.aamop.2018.02.004, the "
    "group's own tabulation of every g_J they measured, which would settle the measurement question -- not "
    "fetchable (the MPG repository copy returns 403, Elsevier is not retrievable, OpenAlex carries no "
    "abstract). NOTE for scale: a theory g_J would be good enough. The one clean measured-versus-computed "
    "pair inside a single modern RCC paper is Ca+ (Sahoo and Kumar's 2.002267 against the measured "
    "2.00225664(9), 1.0e-5 apart) and the 2025 MCDHF Ba+ calculation does 7e-7, so a hypothetical Mg+ "
    "calculation would land in 1e-6 to 1e-5, i.e. about 2e-3 G on the Section 9.13 212.78 G anchor, which "
    "depends on g_J at the 0.25 G level. It does not exist. Until one does, the anchor must be evaluated "
    "against a DECLARED g_J assumption, which is what tests/test_zeeman_anchors.py does. Note that 9Be+ "
    "was thought to be in the same position and is NOT: Shiga et al. 2011's body text prints an absolute "
    "g_J, re-reduced from Wineland, Bollinger and Itano 1983's cyclotron comparison (be9.S12.g_J, ledger "
    "conv.be9_g_j_provenance)",
    "mg25.P12.A_hfs_hz": "theory only: Kaur et al. 2025 (arXiv:2504.19515) Table X gives -103.4(5) MHz with "
    "a blank experiment column; Hao et al., Phys. Rev. A 107, L020803 (2023) is the only plausible home for "
    "a first measurement and was not obtainable",
    "mg25.P32.A_hfs_hz": "theory only: Kaur et al. 2025 give -19.31(5) MHz",
    "mg25.P32.B_hfs_hz": "theory only: Sur et al. 2005 give |B| = 22.91 MHz UNSIGNED",
}

MISSING: tuple[MissingConstant, ...] = required_constants_missing(TABLE, REQUIRED, _CONSULT)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`: no measured g_J and no measured 3p hyperfine constants."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
