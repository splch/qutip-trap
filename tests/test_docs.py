"""The documentation: the examples run, the ledger ids it cites resolve, and nothing contradicts the bit order."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from qutip_trap.provenance import load_ledger, repository_root

ROOT = repository_root()
PAGES = (*sorted((ROOT / "docs").glob("*.md")), ROOT / "README.md")
EXAMPLES = ROOT / "docs" / "examples.md"


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S)


def test_every_ledger_id_cited_in_the_documentation_resolves() -> None:
    records = load_ledger()
    pattern = re.compile(r"`((?:conv|anchor)\.[A-Za-z0-9_.]*[A-Za-z0-9_])`")
    dangling = {
        page.name: missing
        for page in PAGES
        if (missing := sorted(set(pattern.findall(page.read_text(encoding="utf-8"))) - records.keys()))
    }
    assert not dangling, dangling


def test_examples_compile() -> None:
    blocks = _python_blocks(EXAMPLES)
    assert blocks
    for block in blocks:
        compile(block, "examples.md", "exec")


@pytest.mark.slow
def test_examples_execute_in_order() -> None:
    namespace: dict[str, object] = {}
    for k, block in enumerate(_python_blocks(EXAMPLES)):
        exec(compile(block, f"examples.md[{k}]", "exec"), namespace)  # noqa: S102  (the documented examples)


BIT_ORDER_CONTRADICTIONS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"qubit 0 (?:is |as |being )?(?:the )?most[- ]significant",
        r"qubit 0 (?:is |as |being )?(?:the )?leftmost",
        r"qubit 0 (?:is |as |being )?(?:the )?first character",
    )
)


def test_no_docstring_or_page_contradicts_the_bit_order() -> None:
    """Qubit 0 is the least-significant bit of every key; no docstring or page may say otherwise."""
    texts = [(page.name, page.read_text(encoding="utf-8")) for page in PAGES]
    for path in sorted((ROOT / "qutip_trap").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                texts.append((f"{path.relative_to(ROOT)}:{node.lineno}", node.value.value))
    contradictions = [
        (where, p.pattern) for where, text in texts for p in BIT_ORDER_CONTRADICTIONS if p.search(text)
    ]
    assert not contradictions, contradictions
