"""Shared helper for species tables that PLAN.md does not yet complete: a ``Cited`` factory per species."""

from __future__ import annotations

from collections.abc import Callable

from qutip_trap.provenance import Cited, Tag


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
            ledger_id=prefix + suffix,
            tag=tag,
            uncertainty=uncertainty,
            note=note,
        )

    return _c


__all__ = ["cited_factory"]
