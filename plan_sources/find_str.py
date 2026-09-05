"""Print +-W chars around each occurrence of each literal pattern across the plan sources.

Usage: python find_str.py [W] PATTERN [PATTERN ...]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
W = 160
if args and args[0].isdigit():
    W = int(args[0])
    args = args[1:]
for pat in args:
    hits = 0
    for p in sorted(HERE.glob("p*.md")):
        t = p.read_text(encoding="utf-8")
        i = t.find(pat)
        while i != -1:
            hits += 1
            line = t.count("\n", 0, i) + 1
            s = t[max(0, i - W) : i + len(pat) + W].replace("\n", "⏎")
            print(f"--- [{pat[:60]}] {p.name}:{line}\n{s}\n")
            i = t.find(pat, i + 1)
    if hits == 0:
        print(f"--- [{pat[:60]}] NOT FOUND\n")
