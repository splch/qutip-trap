"""9Be+ species table (PLAN.md Sections 4.2.1, 4.2.2, 4.3.2, 4.5.1, 4.5.6, 9.13; Appendix E) - PARTIAL.

The ground-state constants behind the Section 9.13 clock-point anchor are those PLAN.md uses; the plan
itself says of A and the g-factors that they "are not printed there [Langer 2005] and must be cited from
elsewhere", so they carry the ``PLAN_9_13`` source until a primary citation is added. Excited-level
hyperfine constants are not in the plan, so ``species()`` raises :class:`IncompleteSpeciesTable`.
"""

from __future__ import annotations

from qutip_trap.provenance import Cited
from qutip_trap.species._partial import cited_factory
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable, MissingConstant

NAME = "9Be+"
_c = cited_factory("be9.")

_ENTRIES: tuple[Cited, ...] = (
    _c(
        "mass_atomic_u",
        9.012183065,
        "u",
        "NIST_AWIC",
        uncertainty=8.2e-8,
        note="PLAN.md 4.1.7 fixtures use 9.0121822 u for the Home 2013 Be+/Mg+ crystal; the difference is 1e-7 relative",
    ),
    _c("nuclear_spin", 1.5, "hbar", "Ozeri2007", tag="verified", note="Ozeri Table I; Langer 2005"),
    _c(
        "mu_I_nuclear_magnetons",
        -1.177432,
        "mu_N",
        "PLAN_9_13",
        note="the value check_atomic.py and Section 9.13 use; primary source not named in PLAN.md; negative, so "
        "F = I - 1/2 = 1 lies above F = 2 (Section 13 amendment)",
    ),
    _c(
        "S12.A_hfs_hz",
        -625.008837048e6,
        "Hz",
        "PLAN_9_13",
        note="SIGNED negative (inverted multiplet); Section 9.13 says A must be cited from elsewhere than Langer 2005",
    ),
    _c(
        "S12.g_J",
        2.00226,
        "",
        "PLAN_9_13",
        note="approximate; the 10 Hz residual of the clock-point anchor is the precision of the g-factors",
    ),
    _c("P12.energy_cm", 31928.744, "cm^-1", "NIST_ASD_5_12", uncertainty=0.3),
    _c("P32.energy_cm", 31935.320, "cm^-1", "NIST_ASD_5_12", uncertainty=0.3),
    _c(
        "P.linewidth_ozeri_hz",
        19.6e6,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I gamma/2pi; Monroe 1995 quotes 19.4 MHz (PLAN.md 4.2.1), Wineland 2003 19.4 MHz (9.13)",
    ),
    _c(
        "P.linewidth_monroe_hz",
        19.4e6,
        "Hz",
        "Monroe1995",
        tag="verified",
        note="Gamma/2pi of the 313 nm cycling line as PLAN.md 4.2.1 quotes it",
    ),
    _c(
        "S12.hfs_splitting_ozeri_hz",
        1.25e9,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_0/2pi, rounded; |A|(I + 1/2) = 1.250018 GHz",
    ),
    _c(
        "fine_structure_splitting_hz",
        0.198e12,
        "Hz",
        "Ozeri2007",
        tag="verified",
        note="Ozeri Table I omega_f/2pi; the NIST levels give 6.576 cm^-1 = 197.1 GHz",
    ),
    _c(
        "wavelength_ozeri_D1_m",
        313.1e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="AIR wavelength; NIST vacuum 313.197 nm",
    ),
    _c(
        "wavelength_ozeri_D2_m",
        313.0e-9,
        "m",
        "Ozeri2007",
        tag="verified",
        note="AIR wavelength; NIST vacuum 313.133 nm",
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

MISSING: tuple[MissingConstant, ...] = (
    MissingConstant("hyperfine A of P1/2 and A, B of P3/2", "not in PLAN.md"),
    MissingConstant(
        "a primary citation for A_hfs(S1/2), mu_I and g_J",
        "PLAN.md 9.13 says they must be cited from elsewhere",
    ),
    MissingConstant("P lifetimes (only linewidths quoted at 19.4 and 19.6 MHz)", "not in PLAN.md"),
)


def species() -> Species:
    """Raises :class:`IncompleteSpeciesTable`: the P-level hyperfine constants are not cited yet."""
    raise IncompleteSpeciesTable(NAME, MISSING)


__all__ = ["MISSING", "NAME", "TABLE", "species"]
