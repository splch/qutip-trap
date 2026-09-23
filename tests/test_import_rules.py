"""Import rules: ``control/`` never imports ``calibration/``, and the core never imports the application or Flet."""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "qutip_trap"


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
