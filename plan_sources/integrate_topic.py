"""Splice one Run 5 topic's drafted deliverables into the plan sources.

Usage: python integrate_topic.py <manifest.json>

Manifest: {"topic": "...", "dir": "<newtext/<topic>>", "blocks": [{"file": "p0x.md", "before": "<anchor>"} ...],
           "sec15_before": "<anchor in p15 or null>"}
- edits.md: FILE/OLD/NEW blocks separated by '---'; each OLD must occur exactly once (or NEW already present).
- section.md: blocks headed '## Block X ...', spliced before the manifest anchors in order.
- sec13_rows.md: rows ('| ...') under '## New rows' are appended to the Section 13 table; amendments are staged.
- sec9_rows.md, open_issues.md, appendixE.md: staged under newtext/_staged/ for one combined assembly step.
- sec15.md: entries appended to p15_sources.md (before the anchor if given).
Idempotent: every step checks for prior application.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
TOPIC, D = manifest["topic"], Path(manifest["dir"])
STAGE = D.parent / "_staged"
STAGE.mkdir(exist_ok=True)
log = []


def read(fn):
    return (HERE / fn).read_text(encoding="utf-8")


def write(fn, text):
    (HERE / fn).write_text(text, encoding="utf-8")


# ---- 1. exact-match edits ----
edits_text = (
    (D / "edits.md").read_text(encoding="utf-8") if (D / "edits.md").exists() else ""
)
n_applied = n_already = 0
for block in re.split(r"\n---+\n", edits_text):
    m = re.search(r"^FILE:\s*(\S+)\s*$", block, re.M)
    if not m:
        continue
    fn = m.group(1)
    mo = re.search(
        r"^OLD:\s*(.*?)\n\s*\nNEW:\s*(.*?)\s*$", block[m.end() :], re.S | re.M
    )
    if not mo:
        mo = re.search(
            r"^OLD:\s*(.*?)\nNEW:\s*(.*?)\s*$", block[m.end() :], re.S | re.M
        )
    if not mo:
        sys.exit(f"edits.md: cannot parse block for {fn}: {block[:200]}")
    old, new = mo.group(1).strip(), mo.group(2).strip()
    for _k in ("old", "new"):  # some drafters wrap the strings in single backticks
        _v = locals()[_k]
        if len(_v) > 2 and _v[0] == "`" and _v[-1] == "`" and _v.count("`") == 2:
            if _k == "old":
                old = _v[1:-1]
            else:
                new = _v[1:-1]
    text = read(fn)
    if new in text and old not in text:
        n_already += 1
        continue
    c = text.count(old)
    if c != 1:
        sys.exit(f"edits.md: {fn}: OLD occurs {c} times: {old[:120]}")
    write(fn, text.replace(old, new))
    n_applied += 1
log.append(f"edits: applied {n_applied}, already {n_already}")

# ---- 2. section blocks ----
sec = (D / "section.md").read_text(encoding="utf-8")
parts = re.split(r"^## Block [A-Z]", sec, flags=re.M)[1:]
blocks = []
for p in parts:
    body = p.split("\n", 1)[1] if "\n" in p else ""
    body = re.split(r"\n---+\s*\n", body)[0].strip("\n")
    blocks.append(body.strip())
if (
    not blocks
):  # a single-section deliverable: drop a leading italic insertion note, keep the rest
    lines = sec.strip("\n").split("\n")
    if lines and lines[0].startswith("*") and lines[0].rstrip().endswith("*"):
        lines = lines[1:]
    blocks = ["\n".join(lines).strip()]
anchors = manifest["blocks"]
if len(blocks) != len(anchors):
    sys.exit(
        f"section.md has {len(blocks)} blocks but the manifest lists {len(anchors)} anchors"
    )
n_ins = n_have = 0
for body, a in zip(blocks, anchors):
    text = read(a["file"])
    probe = body[:120]
    if probe in text:
        n_have += 1
        continue
    if a.get("append"):
        text = text.rstrip("\n") + "\n\n" + body + "\n"
    else:
        if text.count(a["before"]) != 1:
            sys.exit(
                f"anchor not unique in {a['file']}: {a['before']!r} ({text.count(a['before'])})"
            )
        text = text.replace(a["before"], body + "\n\n" + a["before"])
    write(a["file"], text)
    n_ins += 1
log.append(f"blocks: inserted {n_ins}, already present {n_have}")

# ---- 3. Section 13 rows ----
if (D / "sec13_rows.md").exists():
    t13 = (D / "sec13_rows.md").read_text(encoding="utf-8")
    new_part = t13.split("## Amendments")[0]
    rows = [
        ln
        for ln in new_part.splitlines()
        if ln.startswith("| ")
        and not ln.startswith("| Quantity |")
        and not ln.startswith("|---")
    ]
    text = read("p13_conventions.md")
    anchor = "\n\nSymbol collisions renamed in code"
    assert text.count(anchor) == 1
    fresh = [r for r in rows if r[:80] not in text]
    if fresh:
        text = text.replace(anchor, "\n" + "\n".join(fresh) + anchor)
        write("p13_conventions.md", text)
    log.append(f"sec13 rows: appended {len(fresh)} of {len(rows)}")
    amend = t13.split("## Amendments", 1)[1] if "## Amendments" in t13 else ""
    if amend.strip():
        (STAGE / f"sec13_amendments_{TOPIC}.md").write_text(amend, encoding="utf-8")

# ---- 4. staged material ----
for name in ("sec9_rows.md", "open_issues.md", "appendixE.md"):
    if (D / name).exists():
        (STAGE / f"{name[:-3]}_{TOPIC}.md").write_text(
            (D / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
log.append("staged sec9/open_issues/appendixE")

# ---- 5. Section 15 entries ----
if (D / "sec15.md").exists():
    t15 = (D / "sec15.md").read_text(encoding="utf-8")
    lines = [
        ln
        for ln in t15.splitlines()
        if ln.startswith("- ") or (ln.startswith("**") and ln.rstrip().endswith("**"))
    ]
    text = read("p15_sources.md")
    fresh = [ln for ln in lines if ln[:70] not in text]
    if fresh:
        addition = "\n".join(fresh) + "\n"
        anchor = manifest.get("sec15_before")
        if anchor and text.count(anchor) == 1:
            text = text.replace(anchor, addition + "\n" + anchor)
        else:
            text = text.rstrip("\n") + "\n\n" + addition
        write("p15_sources.md", text)
    log.append(f"sec15: added {len(fresh)} of {len(lines)} lines")

print(f"[{TOPIC}] " + "; ".join(log))
