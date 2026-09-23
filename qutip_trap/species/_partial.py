"""``cited_factory(prefix)``: a ``Cited`` constructor for one species table that prepends ``prefix`` to each id."""

from __future__ import annotations

from collections.abc import Callable

from qutip_trap.species.table import Cited, Tag


def cited_factory(prefix: str) -> Callable[..., Cited]:
    def _c(
        suffix: str,
        value: float,
        unit: str,
        source: str,
        *,
        tag: Tag = "extracted",
        uncertainty: float | None = None,
        note: str = "",
    ) -> Cited:
        return Cited(
            value=value,
            unit=unit,
            source=source,
            key=prefix + suffix,
            tag=tag,
            uncertainty=uncertainty,
            note=note,
        )

    return _c
