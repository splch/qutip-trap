"""Cross-reference check: every 'Section X.Y[.Z]' mentioned in the rendered plan must exist as a heading."""

import re, sys
from pathlib import Path

md = Path(sys.argv[1]).read_text(encoding="utf-8")
heads = set()
for m in re.finditer(
    r"^#{2,4}\s+(?:Appendix\s+([A-Z])|(\d+(?:\.\d+){0,2}))[\s.]", md, re.M
):
    heads.add(m.group(2) or ("Appendix " + m.group(1)))
refs = re.findall(
    r"Sections?\s+((?:\d+(?:\.\d+){0,2})(?:\s*(?:,|and|to)\s*\d+(?:\.\d+){0,2})*)", md
)
missing = {}
for r in refs:
    for num in re.findall(r"\d+(?:\.\d+){0,2}", r):
        if num not in heads:
            missing[num] = missing.get(num, 0) + 1
print("headings:", len(heads))
print(
    "distinct referenced sections:",
    len({n for r in refs for n in re.findall(r"\d+(?:\.\d+){0,2}", r)}),
)
print("missing targets:", missing if missing else "none")
# also report placeholder markers and stray tags
for marker in ("PLACEHOLDER", "Run 4 pending", "pending Run 4", "TODO"):
    print(marker, md.count(marker))
