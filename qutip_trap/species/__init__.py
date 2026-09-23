"""Species data tables: one module per isotope, each a table of cited constants and a ``species()`` builder.
``available()`` lists the isotopes whose tables are complete; the others raise ``IncompleteSpeciesTable`` naming what is
missing and where to find it."""

from __future__ import annotations

from types import ModuleType

from qutip_trap.species import ba133, ba137, be9, ca40, ca43, mg25, sr88, yb171
from qutip_trap.species.model import Species
from qutip_trap.species.table import IncompleteSpeciesTable

MODULES: dict[str, ModuleType] = {m.NAME: m for m in (yb171, ca40, ca43, ba133, ba137, be9, mg25, sr88)}


def species(name: str) -> Species:
    """Build the ``Species`` record for an isotope by its name, for example ``"171Yb+"``."""
    try:
        module = MODULES[name]
    except KeyError:
        raise KeyError(f"no species table for {name!r}; known: {sorted(MODULES)}") from None
    result: Species = module.species()
    return result


def available() -> tuple[str, ...]:
    """The isotopes whose tables build a ``Species`` today."""
    ok: list[str] = []
    for name, module in MODULES.items():
        try:
            module.species()
        except IncompleteSpeciesTable:
            continue
        ok.append(name)
    return tuple(ok)
