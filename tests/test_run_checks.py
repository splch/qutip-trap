"""The Appendix D check-script runner: the comparison logic behind the first CI job."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "validation" / "scripts"


def _load_runner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("run_checks", SCRIPTS / "run_checks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_checks"] = module
    spec.loader.exec_module(module)
    return module


def test_committed_outputs_compare_equal_to_themselves() -> None:
    rc = _load_runner()
    for path in sorted((SCRIPTS / "outputs").glob("check_*.out")):
        text = path.read_text(encoding="utf-8")
        cmp = rc.compare(text, text, rtol=1e-9, atol=1e-12)
        assert cmp.ok, (path.name, cmp.mismatches[:3])
        assert cmp.lines_compared > 0, path.name


def test_comparison_detects_a_changed_number_and_tolerates_artifacts() -> None:
    rc = _load_runner()
    expected = "value = 1.2345678\nwall time 3.2 s\ndone\n"
    same = rc.compare(expected, "value = 1.2345678\nwall time 9.9 s\nexit 0\n", rtol=1e-9, atol=1e-12)
    assert not same.ok and any("wall time" in m for m in same.mismatches), (
        "timing lines are compared unless the caller strips them"
    )
    changed = rc.compare("value = 1.2345678\n", "value = 1.2345679\n", rtol=1e-9, atol=1e-12)
    assert not changed.ok
    fine = rc.compare(
        "value = 1.2345678\nInstalled 3 packages\n", "value = 1.23456780001\n", rtol=1e-9, atol=1e-12
    )
    assert fine.ok and fine.numbers_compared == 1
