"""The contract with the physics core: one module imports it, each name from the module that defines it (the root
package's own names from the package), never through a re-export facade; the view-models import no Flet."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "qutip_trap_app"

FACADES = frozenset(
    {
        "qutip_trap.api",
        "qutip_trap.physics",
        "qutip_trap.schedule",
        "qutip_trap.circuit",
        "qutip_trap.experimental",
        "qutip_trap.dynamics",
    }
)
"""The core's re-export modules, which will go: nothing may be imported through them."""


def _imports(path: Path) -> list[ast.ImportFrom | ast.Import]:
    return [
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(n, ast.Import | ast.ImportFrom)
    ]


def test_core_is_imported_in_one_module_only() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "core.py":
            continue
        for node in _imports(path):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(n == "qutip_trap" or n.startswith("qutip_trap.") for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"qutip_trap imported outside core.py: {offenders}"


def test_core_imports_every_name_from_the_module_that_defines_it() -> None:
    for node in _imports(SRC / "core.py"):
        assert isinstance(node, ast.ImportFrom) and node.module, "core.py imports names, not modules"
        if not node.module.startswith("qutip_trap"):
            continue
        assert node.module not in FACADES, f"{node.module} is a re-export facade"
        module = importlib.import_module(node.module)
        for alias in node.names:
            obj = getattr(module, alias.name)
            if node.module == "qutip_trap":
                assert alias.name in module.__all__, f"{alias.name} is not a root-package name"
            elif inspect.isclass(obj) or inspect.isfunction(obj):
                assert obj.__module__ == node.module, (
                    f"{alias.name} is defined in {obj.__module__}, not {node.module}"
                )
            else:
                source = inspect.getsource(module)
                assert f"\n{alias.name} = " in source or f"\n{alias.name}: " in source, (
                    f"{node.module} defines no {alias.name}"
                )


def test_viewmodels_do_not_import_flet() -> None:
    for path in (SRC / "viewmodel").rglob("*.py"):
        for node in _imports(path):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.startswith("flet") for n in names), path
