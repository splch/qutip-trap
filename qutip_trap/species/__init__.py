"""Species data tables (PLAN.md Sections 3.2, 3.3, 4.5.6; milestone M0).

One module per isotope, each a table of cited constants plus a ``species()`` builder for the Appendix E
:class:`~qutip_trap.species.model.Species` record; the order is the plan's ("171Yb+ and 40Ca+ first, then
43Ca+, 133Ba+/137Ba+, 9Be+, 25Mg+, 88Sr+"). :func:`available` lists the isotopes whose tables are complete
enough to build; the others raise :class:`~qutip_trap.species.table.IncompleteSpeciesTable` naming exactly
what is missing and where to get it.
"""

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
