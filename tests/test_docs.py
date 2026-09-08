"""The documentation of milestone M10: the ledger-generated tables are current, the pages exist and the examples run."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qutip_trap.provenance import load_ledger, repository_root
from tools.docs_from_ledger import TARGETS, anchors_block, conventions_block, render

ROOT = repository_root()
DOCS = ROOT / "docs"


@pytest.mark.parametrize(
    "name", ["physics_notes.md", "conventions.md", "examples.md", "limits.md", "README.md"]
)
def test_documentation_pages_exist_and_are_not_empty(name: str) -> None:
    path = DOCS / name
    assert path.exists(), name
    assert len(path.read_text(encoding="utf-8")) > 1500, name


def test_generated_tables_are_current_with_the_ledger() -> None:
    records = load_ledger()
    blocks = {"conventions": conventions_block(records), "anchors": anchors_block(records)}
    for key, (rel, marker) in TARGETS.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert render(text, marker, blocks[key]) == text, f"run tools/docs_from_ledger.py ({rel})"
    assert "conv.rb_error_rate" in blocks["conventions"] and "anchor.m10." in blocks["anchors"]


def test_every_ledger_id_cited_in_the_documentation_resolves() -> None:
    """The M10 audit found `conv.crosstalk_ratio` cited in the prose of `docs/physics_notes.md` with no ledger record
    behind it -- one dangling reference among 136, which nothing checked. Every `conv.*` / `anchor.*` id in backticks on
    a documentation page or in the root README must be a record id (a `conv.*`-style wildcard cannot match, the pattern
    stopping at the identifier characters)."""
    records = load_ledger()
    pattern = re.compile(r"`((?:conv|anchor)\.[A-Za-z0-9_.]*[A-Za-z0-9_])`")
    dangling: dict[str, list[str]] = {}
    pages = [
        DOCS / name
        for name in ("physics_notes.md", "conventions.md", "examples.md", "limits.md", "README.md")
    ]
    for path in [*pages, ROOT / "README.md"]:
        cited = set(pattern.findall(path.read_text(encoding="utf-8")))
        missing = sorted(c for c in cited if c not in records)
        if missing:
            dangling[path.name] = missing
    assert not dangling, dangling


def _python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_examples_page_has_runnable_blocks_that_name_the_public_api() -> None:
    blocks = _python_blocks(DOCS / "examples.md")
    assert len(blocks) >= 5
    joined = "\n".join(blocks)
    for name in (
        "yb171_chain",
        "run(",
        "randomized_benchmarking",
        "ghz_fidelity",
        "quantum_volume",
        "calibrate",
    ):
        assert name in joined, name
    for block in blocks:
        compile(block, "examples.md", "exec")


@pytest.mark.slow
def test_examples_execute_in_order() -> None:
    """Every ```python block of docs/examples.md runs, in order, in one namespace (a few minutes on the reference machine)."""
    namespace: dict[str, object] = {}
    for k, block in enumerate(_python_blocks(DOCS / "examples.md")):
        exec(compile(block, f"examples.md[{k}]", "exec"), namespace)  # noqa: S102  (the documented examples)
    assert "result" in namespace and "rb" in namespace
