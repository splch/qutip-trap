"""137Ba+ species table (PLAN.md Sections 3.2, 4.3.2; Appendix E) - PARTIAL (I = 3/2).

PLAN.md names 137Ba+ as a preset and stores Ozeri's Table I P -> D branching (f^-1 = 3) for it; no
hyperfine constant, lifetime or g factor for this isotope appears in the plan.
"""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable, MissingConstant

NAME = "137Ba+"
_c = cited_factory("ba137.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 136.90582714, "u", "NIST_AWIC", uncertainty=3.0e-7),
    _c(
        "nuclear_spin",
        1.5,
        "hbar",
        "PLAN_background",
        tag="background",
        note="I = 3/2; PLAN.md does not print it",
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
    _c("D32.g_J", 0.79, "", "NIST_ASD_5_12"),
    _c("D52.g_J", 1.12, "", "NIST_ASD_5_12"),
    _c("P32.g_J", 1.32, "", "NIST_ASD_5_12"),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant("nuclear magnetic moment mu_I", "not in PLAN.md"),
    MissingConstant("hyperfine A and B of every level (S1/2, P1/2, P3/2, D3/2, D5/2)", "not in PLAN.md"),
    MissingConstant("lifetimes of P1/2, P3/2, D3/2, D5/2 and the P branchings", "not in PLAN.md"),
    MissingConstant("g_J of S1/2 and P1/2", "not in PLAN.md; NIST ASD lists none for these levels"),
)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
