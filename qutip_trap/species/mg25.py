"""25Mg+ species table (PLAN.md Section 4.5.6), catalog only: I = 5/2, mu_I < 0.

``species()`` raises: no absolute g_J of any Mg+ isotope has been measured or computed (Itano and Wineland 1981 and
Brewer et al. 2019 servo the field to the electronic transition, so only g_I/g_J is known, and backing g_J out of it
with a tabulated mu_I fails the 9Be+ cross-check), and the 3p hyperfine constants are theory only. mu_I here is the
UNCORRECTED (shielded) moment, unlike 9Be+'s corrected one. The 212.78 G clock point is evaluated in the
tests at a declared g_J.
"""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species.model import Species
from qutip_trap.species.table import CitedFactory, IncompleteSpeciesTable, MissingConstant

NAME = "25Mg+"
_c = CitedFactory("mg25.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 24.985836976, "u", "NIST_AWIC", uncertainty=5.0e-8),
    _c("nuclear_spin", 2.5, "hbar", "ItanoWineland1981", tag="verified"),
    _c(
        "mu_I_nuclear_magnetons",
        -0.85545,
        "mu_N",
        "Stone2019",
        tag="verified",
        uncertainty=8e-5,
        note="the UNCORRECTED (shielded) NMR moment of Alder and Yu, Phys. Rev. 82, 105 (1951); Stone's corrected "
        "recommendation is -0.85533(3) mu_N",
    ),
    _c(
        "S12.A_hfs_hz",
        -596.254376e6,
        "Hz",
        "ItanoWineland1981",
        tag="verified",
        uncertainty=5.4e-2,
        note="SIGNED negative; the value behind the plan's 212.78 G anchor, 2.3 sigma from Brewer et al. 2019",
    ),
    _c(
        "S12.A_hfs_brewer_hz",
        -596.254250949e6,
        "Hz",
        "Brewer2019",
        tag="verified",
        uncertainty=4.5e-5,
        note="Delta W/h = 1 788 762 752.85(13) Hz, the modern value",
    ),
    _c(
        "S12.g_I_over_g_J",
        9.299484e-5,
        "",
        "ItanoWineland1981",
        tag="verified",
        uncertainty=7.5e-9,
        note="the only g-factor quantity measured for 25Mg+",
    ),
    _c("S12.g_I_over_g_J_brewer", 9.299308313e-5, "", "Brewer2019", tag="verified", uncertainty=6.0e-13),
    _c(
        "P12.A_hfs_theory_hz",
        -103.4e6,
        "Hz",
        "KaurSahoo2025",
        tag="background",
        uncertainty=0.5e6,
        note="THEORY (relativistic coupled cluster); Sur et al. 2005's unsigned |A| = 101.70 MHz",
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
        note="THEORY, printed unsigned; taken positive because Q(25Mg) > 0",
    ),
    _c(
        "P12.lifetime_s",
        3.854e-9,
        "s",
        "Ansbacher1989",
        tag="extracted",
        uncertainty=0.030e-9,
        note="gamma/2pi = 41.30(32) MHz, read in Kaur et al. 2025 Table VII; the NIST ASD A_ki agrees to 1%",
    ),
    _c(
        "P32.lifetime_s",
        3.810e-9,
        "s",
        "Ansbacher1989",
        tag="extracted",
        uncertainty=0.040e-9,
        note="gamma/2pi = 41.77(44) MHz, read in Kaur et al. 2025 Table VII",
    ),
    _c("nuclear_quadrupole_moment_b", 0.199, "b", "Stone2019", tag="verified", uncertainty=0.002),
    _c("P12.energy_cm", 35669.31, "cm^-1", "NIST_ASD_5_12", note="280.3530 nm vacuum"),
    _c(
        "P32.energy_cm",
        35760.88,
        "cm^-1",
        "NIST_ASD_5_12",
        note="279.6352 nm vacuum; the 3p fine-structure splitting comes out 3e-4 from Batteiger et al. 2009's",
    ),
    _c("P_to_D_branching", 0.0, "", "Ozeri2007", tag="verified", note="no D level below the P levels"),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`."""
    theory = "theory only (stored as the *_theory_hz entry); no measurement exists"
    raise IncompleteSpeciesTable(
        NAME,
        (
            MissingConstant(
                "mg25.S12.g_J", "no measurement or calculation exists; only S12.g_I_over_g_J is measured"
            ),
            MissingConstant("mg25.P12.A_hfs_hz", theory),
            MissingConstant("mg25.P32.A_hfs_hz", theory),
            MissingConstant("mg25.P32.B_hfs_hz", theory),
        ),
    )
