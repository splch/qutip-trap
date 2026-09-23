"""docs/examples.md: its blocks compile, name the public API, and run in order."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples.md"


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", EXAMPLES.read_text(encoding="utf-8"), flags=re.S)


def test_examples_page_has_runnable_blocks_that_name_the_public_api() -> None:
    blocks = _python_blocks()
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
    """Every ```python block of docs/examples.md runs, in order, in one namespace (a few minutes)."""
    namespace: dict[str, object] = {}
    for k, block in enumerate(_python_blocks()):
        exec(compile(block, f"examples.md[{k}]", "exec"), namespace)  # noqa: S102  (the documented examples)
    assert "result" in namespace and "rb" in namespace
