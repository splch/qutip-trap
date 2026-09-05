"""25Mg+ species table (PLAN.md Sections 4.2.2, 4.4.5, 4.5.1, 9.13; Appendix E) - PARTIAL."""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable, MissingConstant

NAME = "25Mg+"
_c = cited_factory("mg25.")

_ENTRIES: tuple[Cited, ...] = (
    _c("mass_atomic_u", 24.985836976, "u", "NIST_AWIC", uncertainty=5.0e-8),
    _c("nuclear_spin", 2.5, "hbar", "PLAN_check_atomic", note="I = 5/2 as check_atomic.py uses it"),
    _c(
        "mu_I_nuclear_magnetons",
        -0.85545,
        "mu_N",
        "PLAN_check_atomic",
        note="primary source not named in PLAN.md",
    ),
    _c(
        "S12.A_hfs_hz",
        -596.254376e6,
        "Hz",
        "PLAN_9_13",
        note="SIGNED negative; the value behind the 212.78 G clock-point anchor; primary source not named in PLAN.md",
    ),
    _c("S12.g_J", 2.00226, "", "PLAN_9_13", note="approximate, as for 9Be+"),
    _c("P12.energy_cm", 35669.31, "cm^-1", "NIST_ASD_5_12", note="280.35 nm vacuum"),
    _c("P32.energy_cm", 35760.88, "cm^-1", "NIST_ASD_5_12", note="279.64 nm vacuum"),
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

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant("hyperfine A of P1/2 and A, B of P3/2", "not in PLAN.md"),
    MissingConstant("P lifetimes or linewidths", "not in PLAN.md (Ozeri Table I lists Mg+ only in Table II)"),
    MissingConstant(
        "a primary citation for A_hfs(S1/2), mu_I and g_J", "PLAN.md gives the values through check_atomic.py"
    ),
)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`: the P-level constants are not cited yet."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
