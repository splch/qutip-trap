"""Provenance tags and the provenance ledger (PLAN.md Appendix D tag rules; Section 14.5 ledger).

Every physics statement in PLAN.md Part II carries a tag, and every number a species table types in
carries one too, together with its citation. The tags (Appendix D, "How the tags map"):

- ``verified``: checked against its primary source by independent verifier agents;
- ``corrected``: a verifier found and fixed an error in the extracted form or in the printed source;
- ``extracted``: pulled from a primary source with a quote but no agent verdict;
- ``background``: standard textbook physics the author added without a source check;
- ``recomputed here``: computed on this machine by a script under ``validation/scripts/``;
- ``derived``: computed by this package from cited inputs (for example a level energy from a cited
  wavenumber), so the citation is the inputs' citation plus the formula;
- ``contested``: the plan records the number as unresolved between sources.

Appendix D's own warning applies: a tag records what checking was done, not that the value is final;
the Section 9 tests are the definition of correctness.

The ledger (Section 14.5) is ``docs/provenance/ledger.yaml``, kept beside the plan, one record per
quantity with the fields ``id, symbol, tag, section, source, equation, corrected_form``; the core's
``Device.derived()`` provenance and the application's chips both consume it, so the Section 9.11
coverage test is a set difference over ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import yaml

Tag = Literal["verified", "corrected", "extracted", "background", "recomputed here", "derived", "contested"]

TAGS: Final[tuple[str, ...]] = (
    "verified",
    "corrected",
    "extracted",
    "background",
    "recomputed here",
    "derived",
    "contested",
)

LEDGER_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "symbol",
    "tag",
    "section",
    "source",
    "equation",
    "corrected_form",
)


@dataclass(frozen=True)
class Cited:
    """One number typed into a species table: the value as the source prints it, with its citation.

    ``source`` is a key of :data:`qutip_trap.species.sources.SOURCES`; ``ledger_id`` names the record in
    the provenance ledger. ``note`` carries what the source prints and any conditional provenance
    (Section 4.5.6: "two constants carry conditional provenance as a field, not a comment").
    """

    value: float
    unit: str
    source: str
    ledger_id: str
    tag: Tag = "extracted"
    uncertainty: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.tag not in TAGS:
            raise ValueError(f"unknown provenance tag {self.tag!r}; allowed: {TAGS}")
        if not self.source:
            raise ValueError(f"Cited({self.ledger_id!r}) has no source: every number is cited (M0)")
        if not self.ledger_id:
            raise ValueError("Cited values must name their ledger record")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")


@dataclass(frozen=True)
class LedgerRecord:
    """One record of the provenance ledger (Section 14.5)."""

    id: str
    symbol: str
    tag: str
    section: str
    source: str
    equation: str = ""
    corrected_form: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tag not in TAGS:
            raise ValueError(f"ledger record {self.id!r} has unknown tag {self.tag!r}")


def repository_root() -> Path:
    """The checkout root (the parent of the ``qutip_trap`` package directory)."""
    return Path(__file__).resolve().parent.parent


def ledger_path() -> Path:
    """Where the ledger lives: beside the plan, ``docs/provenance/ledger.yaml``."""
    return repository_root() / "docs" / "provenance" / "ledger.yaml"


def load_ledger(path: Path | None = None) -> dict[str, LedgerRecord]:
    """Load the ledger and return its records keyed by id; duplicate ids are an error."""
    p = ledger_path() if path is None else path
    if not p.exists():
        raise FileNotFoundError(f"provenance ledger not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "records" not in raw:
        raise ValueError(f"{p}: expected a mapping with a 'records' list")
    records: dict[str, LedgerRecord] = {}
    for item in raw["records"]:
        if not isinstance(item, dict):
            raise ValueError(f"{p}: every record must be a mapping, got {item!r}")
        missing = [k for k in ("id", "symbol", "tag", "section", "source") if k not in item]
        if missing:
            raise ValueError(f"{p}: record {item.get('id', '?')!r} lacks {missing}")
        known = {k: str(item[k]) for k in LEDGER_FIELDS if k in item}
        extra = {k: str(v) for k, v in item.items() if k not in LEDGER_FIELDS}
        rec = LedgerRecord(**known, extra=extra)
        if rec.id in records:
            raise ValueError(f"{p}: duplicate ledger id {rec.id!r}")
        records[rec.id] = rec
    return records


__all__ = [
    "LEDGER_FIELDS",
    "TAGS",
    "Cited",
    "LedgerRecord",
    "Tag",
    "ledger_path",
    "load_ledger",
    "repository_root",
]
