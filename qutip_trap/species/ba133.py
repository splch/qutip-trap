"""133Ba+ species table (PLAN.md Sections 8.1, 8.4, 9.13; Appendix E) - PARTIAL (radioactive, I = 1/2)."""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable, MissingConstant

NAME = "133Ba+"
_c = cited_factory("ba133.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 132.9060074, "u", "NIST_AWIC", uncertainty=1.1e-6),
    _c("nuclear_spin", 0.5, "hbar", "Christensen2020", tag="verified"),
    _c(
        "S12.A_hfs_hz",
        9925.45355459e6,
        "Hz",
        "PLAN_check_atomic",
        note="= zero-field splitting for I = J = 1/2; primary source not named in PLAN.md",
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
        note="'about 30 s' for Ba+ D5/2 (PLAN.md 8.1, 8.4); a rounded figure, not a measurement",
    ),
    _c("D32.energy_cm", 4873.852, "cm^-1", "NIST_ASD_5_12", note="Ba II value; isotope shift not resolved"),
    _c("D52.energy_cm", 5674.807, "cm^-1", "NIST_ASD_5_12"),
    _c("P12.energy_cm", 20261.561, "cm^-1", "NIST_ASD_5_12", note="493.55 nm vacuum"),
    _c("P32.energy_cm", 21952.404, "cm^-1", "NIST_ASD_5_12", note="455.53 nm vacuum"),
    _c("D32.g_J", 0.79, "", "NIST_ASD_5_12"),
    _c("D52.g_J", 1.12, "", "NIST_ASD_5_12"),
    _c("P32.g_J", 1.32, "", "NIST_ASD_5_12"),
)

TABLE: dict[str, Cited] = {c.ledger_id: c for c in _ENTRIES}

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant("nuclear magnetic moment mu_I", "not in PLAN.md"),
    MissingConstant(
        "hyperfine A of P1/2, D3/2, D5/2 (P3/2's follows from the 623 MHz splitting)", "not in PLAN.md"
    ),
    MissingConstant(
        "P1/2 and P3/2 lifetimes, P1/2 branching to D3/2",
        "not in PLAN.md (Ozeri Table I f^-1 = 3 for 137Ba+)",
    ),
    MissingConstant(
        "a measured D5/2 lifetime (the plan says 'about 30 s') and the D3/2 lifetime", "not in PLAN.md"
    ),
    MissingConstant("g_J of S1/2 and P1/2", "not in PLAN.md; NIST ASD lists none for these levels"),
)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
