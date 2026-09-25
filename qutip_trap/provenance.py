"""Provenance tags, the ``Cited`` record of a typed-in constant, and the ledger loader (PLAN.md Section 14.5).

The tags: ``verified`` (checked against its primary source), ``corrected`` (an error in the extracted form or the source
was found and fixed), ``extracted`` (from a primary source, not verified), ``background`` (textbook physics, no source
check), ``recomputed here`` (computed by this repository's scripts), ``derived`` (computed by the package from cited
inputs), ``contested`` (unresolved between sources). A tag records what checking was done, not that a value is final.
The ledger ``docs/provenance/ledger.yaml`` holds one record per quantity; it travels with the checkout, not the wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, get_args

Tag = Literal["verified", "corrected", "extracted", "background", "recomputed here", "derived", "contested"]
TAGS: Final[tuple[str, ...]] = get_args(Tag)


@dataclass(frozen=True)
class Cited:
    """One constant of a species table: the value as the source prints it, its unit, the ``SOURCES`` key it cites, its
    table id, tag and uncertainty, and a note on what the source prints (conditional provenance included)."""

    value: float
    unit: str
    source: str
    ledger_id: str
    tag: Tag = "extracted"
    uncertainty: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source or not self.ledger_id:
            raise ValueError(f"Cited({self.ledger_id!r}): every constant names its source and its id")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError(f"{self.ledger_id}: uncertainty must be non-negative")


@dataclass(frozen=True)
class LedgerRecord:
    """One record of the provenance ledger."""

    id: str
    symbol: str
    tag: str
    section: str
    source: str
    equation: str = ""
    corrected_form: str = ""

    def __post_init__(self) -> None:
        if self.tag not in TAGS:
            raise ValueError(f"ledger record {self.id!r} has unknown tag {self.tag!r}")


def repository_root() -> Path:
    """The checkout root (the parent of the ``qutip_trap`` package directory)."""
    return Path(__file__).resolve().parent.parent


def load_ledger(path: Path | None = None) -> dict[str, LedgerRecord]:
    """The ledger's records by id, read from ``path`` or the checkout's ``docs/provenance/ledger.yaml``; a duplicate id
    is an error."""
    import yaml

    p = repository_root() / "docs" / "provenance" / "ledger.yaml" if path is None else path
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    records: dict[str, LedgerRecord] = {}
    for item in raw["records"]:
        rec = LedgerRecord(**{key: str(value) for key, value in item.items()})
        if rec.id in records:
            raise ValueError(f"{p}: duplicate ledger id {rec.id!r}")
        records[rec.id] = rec
    return records
