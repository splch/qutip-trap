"""The provenance index: the tag, source and PLAN.md sections behind every displayed quantity (PLAN.md Section 14.5).

The index is built from the ledger ``docs/provenance/ledger.yaml`` (every record) and ``PLAN.md`` (every numbered or appendix
header with its line, Part II flag, tag counts and own Markdown). A repository checkout builds it from those files on
first load; a packaged build, which carries no PLAN.md, ships it as ``provenance_index.json``, written by
``python -m qutip_trap_app.provenance`` before ``flet build``.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qutip_trap_app.core import TAGS as TAGS  # re-exported: the chips' tag order
from qutip_trap_app.core import load_ledger, repository_root

TAG_MEANING: dict[str, str] = {
    "verified": "checked against its primary source",
    "corrected": "a verifier found and fixed an error in the extracted form or in the printed source; the corrected form is shown",
    "extracted": "taken from a primary source with a quote, not independently checked",
    "background": "standard textbook physics supplied without a source check",
    "recomputed here": "computed by this repository's code and pinned by its tests",
    "derived": "computed by the simulator from cited inputs, so the citation is the inputs' citation plus the formula",
    "contested": "recorded as unresolved between sources",
}
"""The tags of PLAN.md as the chips' hover text; a tag records what checking was done, never that a value is final."""

TAG_GLYPH: dict[str, str] = {
    "verified": "✓",
    "corrected": "✎",
    "extracted": "❝",
    "background": "○",
    "recomputed here": "⌗",
    "derived": "→",
    "contested": "?",
}
"""One glyph per tag beside its word: meaning is never carried by colour alone (WCAG 1.4.1)."""

ASSET_PATH = Path(__file__).resolve().with_name("provenance_index.json")
"""The index a packaged build ships, inside the package directory that ``flet build`` bundles."""

PART_II_SECTIONS = ("4", "5", "6", "7", "8")
"""PLAN.md Part II, the module specifications."""

_HEADER = re.compile(r"^#{2,4}\s+(?:(\d+(?:\.\d+)*)\.?\s+)(.*\S)\s*$")
_TAG = re.compile(r"\*\*\[(" + "|".join(re.escape(t) for t in TAGS) + r")[^\]]*\]\*\*")
_CLAUSE_SECTION = re.compile(r"^\s*(?:Sections?\s+)?(\d+(?:\.\d+)*)(?![\w.-])")


def _parse_sections(plan_text: str) -> dict[str, dict[str, Any]]:
    """Every numbered header of the plan: number -> title, line, whether it is Part II, the tag counts of its
    own text (up to the next header of any depth) and that text."""
    lines = plan_text.splitlines()
    heads: list[tuple[str, str, int]] = []  # (number, title, line)
    for k, text in enumerate(lines, start=1):
        m = _HEADER.match(text)
        if m:
            heads.append((m.group(1), m.group(2), k))
    out: dict[str, dict[str, Any]] = {}
    for i, (number, title, line) in enumerate(heads):
        end = heads[i + 1][2] - 1 if i + 1 < len(heads) else len(lines)
        body = "\n".join(lines[line:end])
        counts = {t: 0 for t in TAGS}
        for tag in _TAG.findall(body):
            counts[tag] += 1
        entry: dict[str, Any] = {
            "title": title,
            "line": line,
            "part_ii": number.split(".")[0] in PART_II_SECTIONS,
            "tags": {t: n for t, n in counts.items() if n},
        }
        entry["text"] = body.strip("\n")
        out[number] = entry
    return out


def sections_named(section_field: str) -> tuple[str, ...]:
    """The section numbers a ledger record's ``section`` field names: the leading number of each ``;``- or ``,``-separated
    clause ("9.16 row 4.5-7; 4.5.5" -> ("9.16", "4.5.5")), so row numbers, dates and equation numbers are not sections."""
    out: list[str] = []
    for clause in re.split(r"[;,]", section_field):
        m = _CLAUSE_SECTION.match(clause)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return tuple(out)


@functools.cache
def generate_index() -> dict[str, Any]:
    """The index built from this checkout's ledger and PLAN.md (built once per process)."""
    sections = _parse_sections((repository_root() / "PLAN.md").read_text(encoding="utf-8"))
    records = {
        rid: {
            "tag": r.tag,
            "section": r.section,
            "source": r.source,
            "equation": r.equation,
            "corrected_form": r.corrected_form,
            "sections": [s for s in sections_named(r.section) if s in sections],
        }
        for rid, r in load_ledger().items()
    }
    return {"records": records, "sections": sections}


# ---- run-time lookups ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chip:
    """What a provenance chip shows: the tag with its glyph and meaning, the section, the source, the equation and the
    corrected form where the plan corrects the source (Section 14.5)."""

    id: str
    tag: str
    section: str
    source: str
    equation: str
    corrected_form: str
    sections: tuple[str, ...]

    @property
    def glyph(self) -> str:
        return TAG_GLYPH[self.tag]

    @property
    def label(self) -> str:
        return f"{self.glyph} {self.tag}"

    @property
    def meaning(self) -> str:
        return TAG_MEANING[self.tag]


@dataclass(frozen=True)
class SectionInfo:
    number: str
    title: str
    line: int
    part_ii: bool
    tags: dict[str, int]

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}"


class ProvenanceIndex:
    """The loaded index: chips by ledger id, sections by number."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._records: dict[str, dict[str, Any]] = data["records"]
        self._sections: dict[str, dict[str, Any]] = data["sections"]
        self.on_open_section: Callable[[str], None] | None = None
        """What a clicked chip does: the session sets it to open the explain drawer at the chip's section; None (tests,
        headless use) leaves chips as hover-only labels. One index is loaded per page, so the hook is per session."""

    @classmethod
    def load(cls) -> ProvenanceIndex:
        """The index of this checkout's PLAN.md and ledger, or the shipped asset in a packaged build (no PLAN.md)."""
        if (repository_root() / "PLAN.md").exists():
            return cls(generate_index())
        return cls(json.loads(ASSET_PATH.read_text(encoding="utf-8")))

    def section_for_chip(self, ledger_id: str) -> str:
        """The section a chip opens: its first Part II section, else the first section it names that has text."""
        infos = self.sections_for(ledger_id)
        for info in infos:
            if self._sections[info.number].get("text"):
                return info.number
        if infos:
            return self.nearest_section_with_text(infos[0].number)
        named = sections_named(self.chip(ledger_id).section)
        return self.nearest_section_with_text(named[0]) if named else "13"

    def chip(self, ledger_id: str) -> Chip:
        rec = self._records.get(ledger_id)
        if rec is None:
            raise KeyError(
                f"no ledger record {ledger_id!r} (a quantity without a tag is a bug, Section 14.1)"
            )
        return Chip(
            id=ledger_id,
            tag=rec["tag"],
            section=rec["section"],
            source=rec["source"],
            equation=rec["equation"],
            corrected_form=rec["corrected_form"],
            sections=tuple(rec["sections"]),
        )

    def section(self, number: str) -> SectionInfo:
        s = self._sections.get(number)
        if s is None:
            raise KeyError(f"PLAN.md has no section {number!r}")
        return SectionInfo(number, s["title"], int(s["line"]), bool(s["part_ii"]), dict(s["tags"]))

    def section_text(self, number: str) -> str:
        """The section's own Markdown (up to the next header of any depth)."""
        s = self._sections.get(number)
        if s is None:
            raise KeyError(f"PLAN.md has no section {number!r}")
        return str(s.get("text", ""))

    def subsections(self, number: str) -> tuple[SectionInfo, ...]:
        """The sections one level below ``number`` (``"4.1"`` -> 4.1.1, 4.1.2, ...), in the plan's order."""
        prefix = number + "."
        out = [
            self.section(n)
            for n in self._sections
            if n.startswith(prefix) and n.count(".") == number.count(".") + 1
        ]
        return tuple(sorted(out, key=lambda s: s.line))

    def nearest_section_with_text(self, number: str) -> str:
        """``number`` when it carries text, else the nearest enclosing section that does (a chip may name "13")."""
        parts = number.split(".")
        while parts:
            n = ".".join(parts)
            if n in self._sections and self._sections[n].get("text"):
                return n
            parts.pop()
        return number

    def sections_for(self, ledger_id: str) -> tuple[SectionInfo, ...]:
        """The plan sections a record names, Part II first (the explain panel's targets)."""
        infos = [self.section(n) for n in self.chip(ledger_id).sections]
        return tuple(sorted(infos, key=lambda s: (not s.part_ii, s.line)))

    def part_ii(self) -> tuple[SectionInfo, ...]:
        return tuple(self.section(n) for n, s in self._sections.items() if s["part_ii"])

    def missing(self, ids: Iterable[str]) -> tuple[str, ...]:
        """The ids the ledger does not carry."""
        return tuple(sorted(set(ids) - set(self._records)))


def main() -> None:
    """Write the index asset a packaged build ships."""
    index = generate_index()
    ASSET_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ASSET_PATH.name}: {len(index['records'])} records, {len(index['sections'])} sections")


if __name__ == "__main__":
    main()
