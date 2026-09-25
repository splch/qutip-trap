"""The provenance ledger and the import rules of the package."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from qutip_trap.provenance import TAGS, load_ledger, repository_root

PKG = repository_root() / "qutip_trap"


def test_ledger_loads_with_valid_tags_and_fields() -> None:
    ledger = load_ledger()
    assert len(ledger) > 100
    for rec in ledger.values():
        assert rec.tag in TAGS
        assert rec.section and rec.source and rec.symbol


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_control_never_imports_calibration() -> None:
    for path in (PKG / "control").glob("*.py"):
        assert not any(n.startswith("qutip_trap.calibration") for n in _imports(path)), path


def test_core_never_imports_the_application_or_flet() -> None:
    for path in PKG.rglob("*.py"):
        assert not any(n.startswith(("qutip_trap_app", "flet")) for n in _imports(path)), path


def test_every_provenance_id_the_package_stamps_exists_in_the_ledger() -> None:
    """Every `conv.*` / `anchor.*` string literal in the package names a ledger record."""
    pattern = re.compile(r"^(conv|anchor)\.[A-Za-z0-9_.]+$")
    ids = {
        node.value
        for path in PKG.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and pattern.match(node.value)
    }
    missing = sorted(ids - set(load_ledger()))
    assert not missing, f"provenance ids with no ledger record: {missing}"
