"""The species tables (PLAN.md Section 4.5.6): one module per isotope, a table of cited constants and a ``species()``
builder for its :class:`~qutip_trap.species.model.Species`. :func:`available` lists the isotopes whose tables build; the
others raise :class:`~qutip_trap.species.table.IncompleteSpeciesTable` naming what is missing.
"""

from __future__ import annotations

from typing import Protocol

from qutip_trap.provenance import Cited
from qutip_trap.species import ba133, ba137, be9, ca40, ca43, mg25, sr88, yb171
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable


class _SpeciesTable(Protocol):
    """What every table module provides."""

    NAME: str
    TABLE: dict[str, Cited]

    def species(self) -> Species: ...


_TABLES: tuple[_SpeciesTable, ...] = (yb171, ca40, ca43, ba133, ba137, be9, mg25, sr88)
MODULES: dict[str, _SpeciesTable] = {m.NAME: m for m in _TABLES}


def species(name: str) -> Species:
    """Build the ``Species`` record for an isotope by its name, for example ``"171Yb+"``."""
    try:
        module = MODULES[name]
    except KeyError:
        raise KeyError(f"no species table for {name!r}; known: {sorted(MODULES)}") from None
    return module.species()


def available() -> tuple[str, ...]:
    """The isotopes whose tables build a ``Species``."""
    ok: list[str] = []
    for name, module in MODULES.items():
        try:
            module.species()
        except IncompleteSpeciesTable:
            continue
        ok.append(name)
    return tuple(ok)
