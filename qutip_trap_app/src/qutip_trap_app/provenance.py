"""The provenance index: tags, sources and Part II sections behind every displayed quantity (PLAN.md Section 14.5; M11.1).

Section 14.5: "Provenance chips on every formula and number, generated from a provenance ledger ... which the core's
``Device.derived()`` provenance and the app's chips both consume, so the Section 9.11 static test is a set difference over
ids rather than a hand-maintained index; hovering a chip shows the tag, the source and the corrected form where the plan
corrects the source." And: "Explain panel. At every level, the Part II subsection that governs what is on screen is shown
beside it."

The index is GENERATED (``python -m qutip_trap_app.provenance``) from two data files kept beside the plan, the ledger
``docs/provenance/ledger.yaml`` and ``PLAN.md`` itself, into the asset ``src/qutip_trap_app/provenance_index.json`` that the
packaged application ships and reads at run time; ``--check`` exits 1 when the asset is stale, the way the core's
``tools/docs_from_ledger.py --check`` guards the documentation. From the plan it takes every numbered section header with
its line and the count of provenance tags in its text, so a chip can name the subsection and a reader can see how much of
it was verified; from the ledger it takes every record verbatim. Nothing in the index is typed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAGS: tuple[str, ...] = (
    "verified",
    "corrected",
    "extracted",
    "background",
    "recomputed here",
    "derived",
    "contested",
)

TAG_MEANING: dict[str, str] = {
    "verified": "checked against its primary source by independent verifier agents (Appendix D)",
    "corrected": "a verifier found and fixed an error in the extracted form or in the printed source; the corrected form is shown",
    "extracted": "taken from a primary source with a quote, not independently checked",
    "background": "standard textbook physics the plan's author supplied without a source check",
    "recomputed here": "computed on this machine by a committed check script under validation/scripts/",
    "derived": "computed by the simulator from cited inputs, so the citation is the inputs' citation plus the formula",
    "contested": "recorded as unresolved between sources",
}
"""Appendix D's tag rules, as the chips' hover text states them; a tag records what checking was done, never that a value is final."""

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
"""Where the generated index lives inside the app's ``src/`` tree (shipped by ``flet build``)."""

PART_II_SECTIONS = ("4", "5", "6", "7", "8")
"""PLAN.md Part II is Sections 4 to 8: "Every numbered subsection below is the specification of one module"."""

TEXT_SECTIONS: tuple[str, ...] | None = None
"""The top-level sections whose own Markdown the index carries for the explain drawer's Specification tile (Section 14.5
"Explain panel": the subsection that governs what is on screen is shown beside it); None means every section, which is what
ships: the governing sections of the screens reach outside Part II (the seed's Section 3.4, the wall time's Section 11.1, the
provenance tags' Section 14.5), and the packaged app has no PLAN.md to fall back on."""

_HEADER = re.compile(r"^(#{2,4})\s+(?:(\d+(?:\.\d+)*)\.?\s+)(.*\S)\s*$")
_APPENDIX = re.compile(r"^##\s+(Appendix [A-Z])\.\s+(.*\S)\s*$")
_TAG = re.compile(
    r"\*\*\[(verified|corrected|extracted|background|recomputed here|contested|derived)[^\]]*\]\*\*"
)


def repository_root() -> Path:
    """The checkout root: the parent of the ``qutip_trap_app`` package directory."""
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_sections(plan_text: str) -> dict[str, dict[str, Any]]:
    """Every numbered header (and appendix header) of the plan: number -> title, line, depth, tag counts of its own text
    (up to the next header of any depth), and whether it belongs to Part II."""
    lines = plan_text.splitlines()
    heads: list[tuple[str, str, int, int]] = []  # (number, title, line, depth)
    for k, text in enumerate(lines, start=1):
        m = _HEADER.match(text)
        if m:
            heads.append((m.group(2), m.group(3), k, len(m.group(1))))
            continue
        a = _APPENDIX.match(text)
        if a:
            heads.append((a.group(1), a.group(2), k, 2))
    out: dict[str, dict[str, Any]] = {}
    for i, (number, title, line, depth) in enumerate(heads):
        end = heads[i + 1][2] - 1 if i + 1 < len(heads) else len(lines)
        body = "\n".join(lines[line:end])
        counts = {t: 0 for t in TAGS}
        for tag in _TAG.findall(body):
            counts[tag] += 1
        entry: dict[str, Any] = {
            "title": title,
            "line": line,
            "depth": depth,
            "part_ii": number.split(".")[0] in PART_II_SECTIONS,
            "tags": {t: n for t, n in counts.items() if n},
        }
        if TEXT_SECTIONS is None or number.split(".")[0] in TEXT_SECTIONS:
            entry["text"] = body.strip("\n")
        out[number] = entry
    return out


def _load_ledger(path: Path) -> dict[str, dict[str, str]]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, str]] = {}
    for item in raw["records"]:
        rid = str(item["id"])
        if rid in records:
            raise ValueError(f"duplicate ledger id {rid!r}")
        rec = {
            k: str(item.get(k, ""))
            for k in ("symbol", "tag", "section", "source", "equation", "corrected_form")
        }
        if rec["tag"] not in TAGS:
            raise ValueError(f"ledger record {rid!r} has unknown tag {rec['tag']!r}")
        records[rid] = rec
    return records


_SECTION_TOKEN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)*)(?![\w])")


def sections_named(section_field: str) -> tuple[str, ...]:
    """The section numbers a ledger record's ``section`` field names ("13 Frequencies; 5.6" -> ("13", "5.6"))."""
    seen: list[str] = []
    for tok in _SECTION_TOKEN.findall(section_field):
        if tok not in seen and not tok.startswith("0"):
            seen.append(tok)
    return tuple(seen)


def generate_index(root: Path | None = None) -> dict[str, Any]:
    """Build the index from the ledger and the plan under ``root``."""
    base = root if root is not None else repository_root()
    ledger_path = base / "docs" / "provenance" / "ledger.yaml"
    plan_path = base / "PLAN.md"
    records = _load_ledger(ledger_path)
    sections = _parse_sections(plan_path.read_text(encoding="utf-8"))
    for rec in records.values():
        named = sections_named(rec["section"])
        rec["sections"] = json.dumps([s for s in named if s in sections])
    tag_totals = {t: sum(1 for r in records.values() if r["tag"] == t) for t in TAGS}
    return {
        "format": "qutip-trap-app/provenance-index/1",
        "generated_from": {
            "ledger": "docs/provenance/ledger.yaml",
            "ledger_sha256": _sha256(ledger_path),
            "plan": "PLAN.md",
            "plan_sha256": _sha256(plan_path),
        },
        "tags": {t: TAG_MEANING[t] for t in TAGS},
        "tag_totals": tag_totals,
        "records": records,
        "sections": sections,
    }


def render_index(index: dict[str, Any]) -> str:
    return json.dumps(index, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def write_index(path: Path | None = None, root: Path | None = None) -> Path:
    target = path if path is not None else ASSET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_index(generate_index(root)), encoding="utf-8")
    return target


def index_is_current(path: Path | None = None, root: Path | None = None) -> bool:
    target = path if path is not None else ASSET_PATH
    if not target.exists():
        return False
    return target.read_text(encoding="utf-8") == render_index(generate_index(root))


# ---- run-time lookups ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chip:
    """What a provenance chip shows: the tag with its glyph and meaning, the section, the source, the equation and the
    corrected form where the plan corrects the source (Section 14.5)."""

    id: str
    tag: str
    symbol: str
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
    depth: int
    part_ii: bool
    tags: dict[str, int]

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}"


class ProvenanceIndex:
    """The loaded index: chips by ledger id, sections by number."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self._records: dict[str, dict[str, Any]] = data["records"]
        self._sections: dict[str, dict[str, Any]] = data["sections"]
        self.on_open_section: Callable[[str], None] | None = None
        """What a clicked chip does (DESIGN.md Section 10 R8): the session sets it to open the explain drawer's
        Specification tile at the chip's section; None (tests, headless use) leaves chips as hover-only labels. One index
        is loaded per page, so the hook is per session."""

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

    @classmethod
    def load(cls, path: Path | None = None) -> ProvenanceIndex:
        target = path if path is not None else ASSET_PATH
        if not target.exists():
            raise FileNotFoundError(
                f"provenance index {target} missing: run `python -m qutip_trap_app.provenance` from the repository"
            )
        return cls(json.loads(target.read_text(encoding="utf-8")))

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def has(self, ledger_id: str) -> bool:
        return ledger_id in self._records

    def chip(self, ledger_id: str) -> Chip:
        rec = self._records.get(ledger_id)
        if rec is None:
            raise KeyError(
                f"no ledger record {ledger_id!r} (a quantity without a tag is a bug, Section 14.1)"
            )
        return Chip(
            id=ledger_id,
            tag=rec["tag"],
            symbol=rec["symbol"],
            section=rec["section"],
            source=rec["source"],
            equation=rec["equation"],
            corrected_form=rec["corrected_form"],
            sections=tuple(json.loads(rec.get("sections", "[]"))),
        )

    def section(self, number: str) -> SectionInfo:
        s = self._sections.get(number)
        if s is None:
            raise KeyError(f"PLAN.md has no section {number!r}")
        return SectionInfo(
            number, s["title"], int(s["line"]), int(s["depth"]), bool(s["part_ii"]), dict(s["tags"])
        )

    def section_text(self, number: str) -> str:
        """The section's own Markdown (its text up to the next header of any depth), for the explain drawer's Specification
        tile; empty for a section outside :data:`TEXT_SECTIONS`."""
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
        """``number`` when it carries text, else the nearest enclosing section that does (a chip may name a top-level
        section such as "13")."""
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
        """The set difference of Section 9.11: ids the ledger does not carry."""
        return tuple(sorted(set(ids) - self.ids))

    @property
    def tag_totals(self) -> dict[str, int]:
        return dict(self._data["tag_totals"])

    @property
    def generated_from(self) -> dict[str, str]:
        return dict(self._data["generated_from"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate the provenance index asset from the ledger and PLAN.md"
    )
    ap.add_argument("--check", action="store_true", help="exit 1 if the asset is stale (CI)")
    ap.add_argument(
        "--root", type=Path, default=None, help="repository root (default: derived from this file)"
    )
    args = ap.parse_args(argv)
    if args.check:
        if index_is_current(root=args.root):
            print(f"{ASSET_PATH.relative_to(repository_root())} is current")
            return 0
        print(
            f"{ASSET_PATH.relative_to(repository_root())} is STALE: run `python -m qutip_trap_app.provenance`"
        )
        return 1
    target = write_index(root=args.root)
    idx = ProvenanceIndex.load(target)
    print(
        f"wrote {target.relative_to(repository_root())}: {len(idx.ids)} records, {len(idx.part_ii())} Part II sections"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ASSET_PATH",
    "PART_II_SECTIONS",
    "TAG_GLYPH",
    "TEXT_SECTIONS",
    "TAG_MEANING",
    "TAGS",
    "Chip",
    "ProvenanceIndex",
    "SectionInfo",
    "generate_index",
    "index_is_current",
    "main",
    "render_index",
    "repository_root",
    "sections_named",
    "write_index",
]
