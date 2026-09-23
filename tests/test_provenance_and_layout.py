"""The provenance ledger, the Section 3.2 layout and the import rules of Appendix E."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from qutip_trap.provenance import TAGS, load_ledger, repository_root

ROOT = repository_root()
PKG = ROOT / "qutip_trap"

PLAN = ROOT / "PLAN.md"


def section_3_2_modules() -> list[str]:
    """The `qutip_trap/` entries of the Section 3.2 layout fence of PLAN.md, derived so the guard cannot drift."""
    text = PLAN.read_text(encoding="utf-8")
    start = text.index("### 3.2 Package layout")
    fence = text[text.index("```", start) + 3 : text.index("```", text.index("```", start) + 3)]
    modules: list[str] = []
    stack: list[tuple[int, str]] = []  # (indent, directory)
    for raw in fence.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        name = line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if name.endswith("/"):
            stack.append((indent, name))
            continue
        path = "".join(d for _, d in stack) + name
        if path.startswith("qutip_trap/") and name.endswith(".py"):
            modules.append(path[len("qutip_trap/") :])
    return modules


SECTION_3_2_MODULES = section_3_2_modules()
SECTION_3_2_PACKAGES = ["calibration", "experiments", "validation"]  # listed as directories with prose only


def test_ledger_loads_with_valid_tags_and_fields() -> None:
    ledger = load_ledger()
    assert len(ledger) > 100
    for rec in ledger.values():
        assert rec.tag in TAGS
        assert rec.section and rec.source and rec.symbol


@pytest.mark.parametrize("rel", SECTION_3_2_MODULES)
def test_section_3_2_module_exists(rel: str) -> None:
    assert (PKG / rel).exists(), rel


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
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
        names = _imports(path)
        assert not any(n.startswith(("qutip_trap_app", "flet")) for n in names), path


def test_public_api_imports_without_qutip_at_import_time() -> None:
    """The data model is importable without instantiating QuTiP objects; qutip stays a TYPE_CHECKING import in M0."""
    mod = importlib.import_module("qutip_trap.api")
    assert len(mod.__all__) > 80


def test_app_scaffold_has_the_flet_create_layout() -> None:
    app = ROOT / "qutip_trap_app"
    for rel in ("pyproject.toml", "src/main.py", "src/assets/icon.png", "tests/test_main.py", "README.md"):
        assert (app / rel).exists(), rel
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'gui = ["flet[all]>=0.86,<1.0", "flet-charts>=0.86,<1.0"]' in text


def referenced_provenance_ids() -> set[str]:
    """Every `conv.*` / `anchor.*` string literal in the package: the ids a derived number, a diagnostics line or an
    ExperimentResult can stamp on its output."""
    ids: set[str] = set()
    pattern = re.compile(r"^(conv|anchor)\.[A-Za-z0-9_.]+$")
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and pattern.match(node.value):
                ids.add(node.value)
    return ids


def test_every_provenance_id_the_package_stamps_exists_in_the_ledger() -> None:
    """Appendix E's closing rule: every derived number reachable through Device.derived(), Result.diagnostics or an
    ExperimentResult carries a provenance id FROM the ledger, so the Section 9.11 coverage test is a set difference.
    A new id stamped in code without a record is the failure this test exists for."""
    ledger = load_ledger()
    missing = sorted(referenced_provenance_ids() - set(ledger))
    assert not missing, f"provenance ids referenced from qutip_trap/ with no ledger record: {missing}"
