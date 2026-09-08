"""The provenance ledger, the Section 3.2 layout and the import rules of Appendix E."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from qutip_trap.provenance import TAGS, ledger_path, load_ledger, repository_root
from tools.ledger_from_tables import render

ROOT = repository_root()
PKG = ROOT / "qutip_trap"

SECTION_3_2_MODULES = [
    "units.py",
    "species/atomic.py",
    "trap/mathieu.py",
    "trap/pseudopotential.py",
    "trap/surface.py",
    "trap/crystal.py",
    "trap/anharmonic.py",
    "trap/micromotion.py",
    "trap/heating.py",
    "light/beams.py",
    "hilbert/space.py",
    "hilbert/operators.py",
    "hilbert/truncation.py",
    "control/native.py",
    "control/compiler.py",
    "control/pulses.py",
    "control/schedule.py",
    "control/shaping.py",
    "control/hardware.py",
    "control/table.py",
    "dynamics/hamiltonian.py",
    "dynamics/channels.py",
    "dynamics/evolve.py",
    "dynamics/frames.py",
    "dynamics/kernels.py",
    "dynamics/engine.py",
    "readout/fluorescence.py",
    "readout/detection.py",
    "readout/discriminate.py",
    "noise/spectra.py",
    "noise/sampling.py",
    "device/model.py",
    "device/presets.py",
    "run/job.py",
    "run/results.py",
    "run/levels.py",
    "io/openqasm.py",
    "calibration/__init__.py",
    "experiments/__init__.py",
    "validation/__init__.py",
]


def test_ledger_loads_with_valid_tags_and_fields() -> None:
    ledger = load_ledger()
    assert len(ledger) > 100
    for rec in ledger.values():
        assert rec.tag in TAGS
        assert rec.section and rec.source and rec.symbol


def test_species_block_of_the_ledger_is_current() -> None:
    text = ledger_path().read_text(encoding="utf-8")
    assert render(text) == text, "run tools/ledger_from_tables.py"


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
