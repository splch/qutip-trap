"""Regenerate the ledger-derived tables of the documentation (PLAN.md Section 14.5; milestone M10).

``docs/conventions.md`` and ``docs/physics_notes.md`` are hand-written prose around tables generated from the provenance
ledger ``docs/provenance/ledger.yaml``: the convention rows (``conv.*`` records: the simulator's convention, the source it
follows, the alternatives it rejects) and the numerical anchors (``anchor.*`` records grouped by family: what each milestone's
tests pinned, where, and what was corrected), each cell the ledger's text. The tables sit between the markers below and are
rewritten in place, so the documentation cannot drift from the ledger; ``--check`` exits 1 when a block is stale (CI).

    uv run python tools/docs_from_ledger.py            # rewrite in place
    uv run python tools/docs_from_ledger.py --check    # exit 1 if a block is stale
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from qutip_trap.provenance import LedgerRecord, ledger_path, load_ledger, repository_root

BEGIN = "<!-- BEGIN generated from docs/provenance/ledger.yaml by tools/docs_from_ledger.py: {name}; do not edit by hand -->"
END = "<!-- END generated: {name} -->"

FAMILY_TITLES: dict[str, str] = {
    "trap": "Trap, crystal and modes (Sections 4.1, 9.1, 9.10, 9.12, 9.13; M1)",
    "yb171": "171Yb+ atomic anchors (Sections 4.5, 9.13; M0a)",
    "ca40": "40Ca+ atomic anchors (Sections 4.5.7, 9.14; M0a)",
    "ca43": "43Ca+ atomic anchors (Section 9.13; M0a)",
    "sr88": "88Sr+ atomic anchors (Sections 4.5.7, 9.14; M0a)",
    "be9": "9Be+ atomic anchors (Sections 4.5.5, 9.13; M0a)",
    "mg25": "25Mg+ atomic anchors (Section 9.13; M0a)",
    "rb87": "87Rb reference checks of the angular algebra (Section 4.5.2; M0a)",
    "audit": "Derivation-audit regressions (Section 9.16; M0a)",
    "ionq": "IonQ formats (Section 8.6; M0)",
    "m2": "Single ion, spin-motion coupling, single-qubit gates (Section 9.2; M2)",
    "m3a": "Multi-level optical Bloch builder (Sections 4.2.8, 8.1, 9.3; M3a)",
    "m3": "Cooling and state preparation (Section 9.3; M3)",
    "m4": "Two-ion entangling gates (Section 9.4; M4)",
    "m5": "Readout (Section 9.5; M5)",
    "m6": "End-to-end circuits (Section 9.6; M6)",
    "m7": "Noise and error channels (Section 9.7; M7)",
    "m8": "Calibration emulation (Sections 7.5, 9.17; M8)",
    "m9a": "Scaling I: mode selection, frozen spectators, ENR, GATE_LOCAL (Sections 9.8, 9.9; M9a)",
    "m9b": "Scaling II: the matrix-free kernel and parallelism (Sections 9.9, 11.1; M9b)",
    "m10": "Benchmark emulation (Section 10 M10)",
    "perf": "Performance pass of 2026-09-09: the exact rotating frame, the kernel's products, the shared atomic algebra, the direct steady state (Sections 5.2, 5.3, 11.1, 11.2)",
}


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|") or "-"


def _table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def conventions_block(records: dict[str, LedgerRecord]) -> str:
    convs = sorted((r for r in records.values() if r.id.startswith("conv.")), key=lambda r: r.id)
    lines = [BEGIN.format(name="conventions"), ""]
    lines.append(
        f"{len(convs)} convention records. The **convention** column is the rule the code implements, **follows** the source it"
        " adopts, **alternatives** what other sources do and what the plan corrected; the tag is Appendix D's."
    )
    lines.append("")
    lines += _table(
        [
            "ledger id",
            "quantity",
            "plan section",
            "convention",
            "follows",
            "alternatives and corrections",
            "tag",
        ],
        ([f"`{r.id}`", r.symbol, r.section, r.equation, r.source, r.corrected_form, r.tag] for r in convs),
    )
    lines += ["", END.format(name="conventions")]
    return "\n".join(lines) + "\n"


def anchors_block(records: dict[str, LedgerRecord]) -> str:
    anchors = [r for r in records.values() if r.id.startswith("anchor.")]
    families: dict[str, list[LedgerRecord]] = {}
    for r in anchors:
        fam = r.id.split(".")[1]
        families.setdefault(fam, []).append(r)
    order = [f for f in FAMILY_TITLES if f in families] + sorted(
        f for f in families if f not in FAMILY_TITLES
    )
    lines = [BEGIN.format(name="anchors"), ""]
    lines.append(
        f"{len(anchors)} anchor records in {len(families)} families. Each row is a number or identity a test pins: **what**"
        " the check established (the ledger's equation field), **where** it lives (script, test, module), and **corrected**"
        " what the check overturned or narrowed, if anything."
    )
    for fam in order:
        recs = sorted(families[fam], key=lambda r: r.id)
        lines += ["", f"#### {FAMILY_TITLES.get(fam, fam)}", ""]
        lines += _table(
            [
                "ledger id",
                "quantity",
                "plan section",
                "what the check established",
                "where",
                "corrected",
                "tag",
            ],
            ([f"`{r.id}`", r.symbol, r.section, r.equation, r.source, r.corrected_form, r.tag] for r in recs),
        )
    lines += ["", END.format(name="anchors")]
    return "\n".join(lines) + "\n"


def render(existing: str, name: str, block: str) -> str:
    begin = BEGIN.format(name=name)
    end = END.format(name=name)
    if begin in existing and end in existing:
        head, rest = existing.split(begin, 1)
        _, tail = rest.split(end, 1)
        tail = tail.lstrip("\n")
        return head + block + ("\n" + tail if tail else "")
    raise ValueError(f"markers for the generated block {name!r} are missing")


TARGETS: dict[str, tuple[str, str]] = {
    "conventions": ("docs/conventions.md", "conventions"),
    "anchors": ("docs/physics_notes.md", "anchors"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated block is stale")
    parser.add_argument("--ledger", type=Path, default=ledger_path())
    args = parser.parse_args(argv)
    records = load_ledger(args.ledger)
    blocks = {"conventions": conventions_block(records), "anchors": anchors_block(records)}
    root = repository_root()
    stale: list[str] = []
    for key, (rel, name) in TARGETS.items():
        path = root / rel
        existing = path.read_text(encoding="utf-8")
        rendered = render(existing, name, blocks[key])
        if rendered != existing:
            if args.check:
                stale.append(rel)
            else:
                path.write_text(rendered, encoding="utf-8")
                print(f"wrote {rel}")
        elif not args.check:
            print(f"{rel}: current")
    if args.check:
        if stale:
            print(f"stale generated blocks: {stale}; run tools/docs_from_ledger.py", file=sys.stderr)
            return 1
        print("documentation tables are current with the ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
