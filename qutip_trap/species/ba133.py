"""133Ba+ species table (PLAN.md Section 4.5.6), catalog only: I = 1/2 and mu_I < 0, so every A is negative except
5d 2D5/2's and the ground multiplet is inverted (F = 0 above F = 1: |F=0, mF=0> is the upper qubit state).

The isotope-independent Ba II constants (level energies, lifetimes, branchings, g_J) are the 137Ba+ table's.
``species()`` raises: A(6p 2P3/2) and A(5d 2D5/2) are printed by no source, only their splittings (Christensen et al.
2020).
"""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species.model import Species
from qutip_trap.species.table import CitedFactory, IncompleteSpeciesTable, MissingConstant

NAME = "133Ba+"
_c = CitedFactory("ba133.")

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
        note="Delta nu(F = 1 <-> 0)/h, the qubit frequency",
    ),
    _c(
        "S12.A_hfs_hz",
        -9925.45355459e6,
        "Hz",
        "Hucul2017",
        tag="corrected",
        uncertainty=1e-4,
        note="A = -(splitting) for I = J = 1/2: INVERTED since mu_I < 0; A(133)/A(137) = -2.469712 against the "
        "(mu_I/I) ratio -2.469686",
    ),
    _c(
        "mu_I_nuclear_magnetons",
        -0.77167,
        "mu_N",
        "Stone2019",
        tag="verified",
        uncertainty=2e-5,
        note="Stone's compilation cites Knab, Schupp and Werth 1987; I = 1/2, so Q = 0",
    ),
    _c(
        "D32.A_hfs_hz",
        -468.5e6,
        "Hz",
        "Hucul2017",
        tag="verified",
        uncertainty=20e6,
        note="Table I: -468.5(1.5)stat +- 20sys MHz (splitting 937(3) MHz); the systematic error is carried",
    ),
    _c(
        "P12.A_hfs_hz",
        -1840e6,
        "Hz",
        "Hucul2017",
        tag="verified",
        uncertainty=11e6,
        note="a literature value Table I relays (Hoehle et al., Phys. Lett. B 62, 390 (1976)); Hucul's own splitting "
        "1840(2)stat +- 20sys MHz corroborates the magnitude",
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
        note="P3/2 -> S1/2 / D5/2 / D3/2 = 0.74 / 0.23 / 0.03 as PLAN.md 8.1 quotes them; the measured set is 137Ba+'s",
    ),
    _c("P32.branching_to_D52", 0.23, "", "Christensen2020", tag="corrected"),
    _c("P32.branching_to_D32", 0.03, "", "Christensen2020", tag="corrected"),
    _c(
        "D52.lifetime_s",
        30.0,
        "s",
        "PLAN_8_1",
        tag="corrected",
        note="PLAN.md 8.1's rounded 'about 30 s'; the measurement is 137Ba+'s D52.lifetime_s",
    ),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`."""
    raise IncompleteSpeciesTable(
        NAME,
        (
            MissingConstant(
                "ba133.P32.A_hfs_hz",
                "no source prints it; Christensen et al. 2020 give the 623(30) MHz splitting, A = -311.5(150) MHz "
                "with the sign of mu_I",
            ),
            MissingConstant(
                "ba133.D52.A_hfs_hz",
                "no source prints it; Christensen et al. 2020 give the 83(30) MHz splitting, A = +27.7(100) MHz with "
                "the sign the 137Ba+ scaling predicts",
            ),
        ),
    )
