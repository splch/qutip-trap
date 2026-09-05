"""Assemble the staged Run 5 material into the plan sources (idempotent).

- Section 9.15: one table from every newtext/_staged/sec9_rows_<topic>.md, appended after Section 9.14.
- Section 12: a paragraph 'Open issues named by the Run 5 briefs' from open_issues_<topic>.md, appended.
- Section 13: a paragraph of Run 5 amendments to existing rows from sec13_amendments_<topic>.md, before the
  'Symbol collisions' paragraph.
- Appendix E: the fenced Python of appendixE_<topic>.md merged into one block with the prose amendments, before
  'Three rules bind the surface'.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = Path(
    "/private/tmp/claude-501/-Users-churchill-Repositories-qutip-trap/b2f1cadd-ba53-4734-9dbe-b340274f9083/scratchpad/run5/newtext/_staged"
)
ORDER = ["comb", "composite", "constants", "transport"]
TITLES = {
    "comb": "Frequency-comb Raman drives (Section 4.3.7)",
    "composite": "Composite pulses, dynamical decoupling and filter functions (Sections 4.3.5, 6.9)",
    "constants": "Detection dark states, metastable lifetimes, laser scattering, polarization-gradient cooling (Sections 8.1, 4.5.5, 4.5.7, 4.2.4)",
    "transport": "Transport, splitting, merging and junctions (Section 4.6, milestone M12)",
}


def read(fn):
    return (HERE / fn).read_text(encoding="utf-8")


def write(fn, t):
    (HERE / fn).write_text(t, encoding="utf-8")


def rows_of(text):
    return [
        ln
        for ln in text.splitlines()
        if ln.startswith("| ")
        and not ln.startswith("| Test |")
        and not ln.startswith("|---")
    ]


def bullets_of(text):
    return [ln.strip() for ln in text.splitlines() if ln.startswith("- ")]


# ---- Section 9.15 ----
p09 = read("p09_validation.md")
if "### 9.15 " not in p09:
    parts = []
    for topic in ORDER:
        f = STAGE / f"sec9_rows_{topic}.md"
        if not f.exists():
            continue
        rows = rows_of(f.read_text(encoding="utf-8"))
        parts.append(f"| **{TITLES[topic]}** | | |")
        parts.extend(rows)
    table = (
        "### 9.15 Run 5 targets: comb drives, composite pulses and decoupling, single-source constants, transport (2026-09-04)\n\n"
        "Every number below is printed by the Run 5 check scripts (`check_comb.py`, `check_composite.py`, `check_constants.py`, `check_transport.py`, with `check_sr_lifetime_quad.py` and `check_sr_lifetime_bbr.py` for the 88Sr+ lifetime; outputs under `validation/scripts/outputs/`), and each row states its measure and tolerance where the closed form allows one; where a source's own figure does not reproduce, the row says so and the assertion is on the recomputed value. Rows are grouped by topic.\n\n"
        "| Test | Expected | Source, tag |\n|---|---|---|\n" + "\n".join(parts) + "\n"
    )
    write("p09_validation.md", p09.rstrip("\n") + "\n\n" + table)
    print(
        "9.15 written with", sum(1 for ln in parts if not ln.startswith("| **")), "rows"
    )
else:
    print("9.15 already present")

# ---- Section 12 ----
p12 = read("p12_risks.md")
if "**Open issues named by the Run 5 briefs" not in p12:
    bl = []
    for topic in ORDER:
        f = STAGE / f"open_issues_{topic}.md"
        if f.exists():
            bl.extend(bullets_of(f.read_text(encoding="utf-8")))
    para = (
        "**Open issues named by the Run 5 briefs (2026-09-04).** The four source briefs of Run 5S (Appendix D) and their consolidation named what they could not source, listed here by topic so that none is mistaken for settled; each is tagged where it arises in Sections 4.2.4, 4.3.5, 4.3.7, 4.5.5, 4.5.7, 4.6, 6.9 and 8.1.\n\n"
        + "\n".join(bl)
        + "\n"
    )
    write("p12_risks.md", p12.rstrip("\n") + "\n\n" + para)
    print("Section 12 Run 5 paragraph written with", len(bl), "bullets")
else:
    print("Section 12 Run 5 paragraph already present")

# ---- Section 13 amendments ----
p13 = read("p13_conventions.md")
if "**Run 5 amendments to the rows above.**" not in p13:
    bl = []
    for topic in ORDER:
        f = STAGE / f"sec13_amendments_{topic}.md"
        if f.exists():
            bl.extend(bullets_of(f.read_text(encoding="utf-8")))
    if bl:
        anchor = "\n\nSymbol collisions renamed in code"
        assert p13.count(anchor) == 1
        para = (
            "\n\n**Run 5 amendments to the rows above** (2026-09-04; the row named in bold is the one amended, and the check scripts of Section 9.15 pin each number).\n\n"
            + "\n".join(bl)
        )
        write("p13_conventions.md", p13.replace(anchor, para + anchor))
        print("Section 13 amendments written with", len(bl), "bullets")
else:
    print("Section 13 amendments already present")

# ---- Appendix E ----
p17 = read("p17_interfaces.md")
if "**Run 5 additions (2026-09-04).**" not in p17:
    code, prose = [], []
    for topic in ORDER:
        f = STAGE / f"appendixE_{topic}.md"
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8")
        if t.strip().lower() in ("none", "none."):
            continue
        fences = re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
        if fences:
            code.append(
                f"# ---- Run 5: {TITLES[topic]} ----\n"
                + "\n".join(fc.rstrip("\n") for fc in fences)
            )
        outside = re.sub(r"```(?:python)?\n.*?```", "", t, flags=re.S)
        for ln in outside.splitlines():
            s = ln.strip()
            if s.startswith("- ") or (
                s.startswith("**")
                and len(s) > 40
                and not s.startswith("**Field-level")
                and not s.startswith("**Run")
            ):
                prose.append(s if s.startswith("- ") else "- " + s)
    anchor = "Three rules bind the surface."
    assert p17.count(anchor) == 1
    block = (
        "**Run 5 additions (2026-09-04).** The topics folded in on 2026-09-04 add the objects below; they follow the same rules (frozen dataclasses, Hz in the public API, provenance on every derived number) and the amendments to existing declarations are listed after the code.\n\n```python\n"
        + "\n\n".join(code)
        + "\n```\n\n"
        + "\n".join(prose)
        + "\n\n"
    )
    write("p17_interfaces.md", p17.replace(anchor, block + anchor))
    print(
        "Appendix E additions written:",
        len(code),
        "code groups,",
        len(prose),
        "prose items",
    )
else:
    print("Appendix E additions already present")
