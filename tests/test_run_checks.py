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


def test_residual_class_numbers_compare_to_an_order_of_magnitude() -> None:
    """A leaked population 4.06e-10 (macOS oracle) against 4.08e-10 (Linux runner) is a residual and passes; the physics
    number on the same line stays strict; a residual off by more than a factor 3 still fails."""
    rc = _load_runner()
    assert (
        rc.is_residual("4.06e-10")
        and rc.is_residual("3e-16")
        and not rc.is_residual("1.86924e-08")
        and not rc.is_residual("0.00410")
    )
    assert rc.significant_digits("4.06e-10") == 3 and rc.significant_digits("150000.000000000") == 15
    exp = "Delta/2pi = 1.00 MHz: phase = +0.00410 rad, population leaked 4.06e-10\n"
    ok = rc.compare(
        exp, "Delta/2pi = 1.00 MHz: phase = +0.00410 rad, population leaked 4.08e-10\n", rtol=1e-9, atol=1e-12
    )
    assert ok.ok and ok.residual_numbers == 1 and ok.max_rel_dev == 0.0
    bad_phase = rc.compare(
        exp, "Delta/2pi = 1.00 MHz: phase = +0.00411 rad, population leaked 4.06e-10\n", rtol=1e-9, atol=1e-12
    )
    assert not bad_phase.ok
    bad_residual = rc.compare(
        exp, "Delta/2pi = 1.00 MHz: phase = +0.00410 rad, population leaked 4.06e-09\n", rtol=1e-9, atol=1e-12
    )
    assert not bad_residual.ok


def test_residual_class_pair_at_or_across_zero_compares_to_the_bound() -> None:
    """check_calibration's fitted residual beta printed -8.62e-12 on the macOS oracle machine and +1.97e-12 on one Linux
    runner (a second runner's run of the same commit passed): a round-off difference has no order of magnitude, so a
    pair at or across zero is compared to the 1e-6 bound, while a residual grown past the bound, or by more than a
    factor 3 on the same side of zero, still fails."""
    rc = _load_runner()
    exp = "null -20.0000 +- 2.8 V/m against -20; residual beta -8.62e-12; beta before -0.0454\n"
    flipped = rc.compare(
        exp,
        "null -20.0000 +- 2.8 V/m against -20; residual beta +1.97e-12; beta before -0.0454\n",
        rtol=1e-9,
        atol=1e-12,
    )
    assert flipped.ok and flipped.residual_numbers == 1
    grown = rc.compare(
        exp,
        "null -20.0000 +- 2.8 V/m against -20; residual beta +1.97e-03; beta before -0.0454\n",
        rtol=1e-9,
        atol=1e-12,
    )
    assert not grown.ok
    same_side = rc.compare(
        exp,
        "null -20.0000 +- 2.8 V/m against -20; residual beta -8.62e-11; beta before -0.0454\n",
        rtol=1e-9,
        atol=1e-12,
    )
    assert not same_side.ok
