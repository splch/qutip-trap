"""The import budget of docs/api_implementation_plan.md item 0.5: ``import qutip_trap`` stays cheap, measured in a fresh
interpreter, because the root becomes the rung-0 namespace in 0.2.0 through a lazy ``__getattr__`` and must not start
pulling qutip in; the Appendix E surface's time is printed for the CI log, not bounded (0.4 to 0.5 s, qutip's own import)."""

from __future__ import annotations

import subprocess
import sys

import pytest

BUDGET_S = 0.2
"""The plan's budget for ``import qutip_trap`` on the CI runner (0.0003 s on the reference machine at 7a26a27)."""


def _import_in_a_fresh_interpreter(module: str) -> tuple[float, bool, bool]:
    """(seconds, qutip loaded, numpy loaded) after ``import module`` in a new process of the same interpreter."""
    code = (
        "import sys, time; t0 = time.perf_counter(); "
        f"import {module}; dt = time.perf_counter() - t0; "
        "print(dt, 'qutip' in sys.modules, 'numpy' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    seconds, qutip_loaded, numpy_loaded = proc.stdout.split()
    return float(seconds), qutip_loaded == "True", numpy_loaded == "True"


def test_import_qutip_trap_stays_under_the_budget() -> None:
    best = min(_import_in_a_fresh_interpreter("qutip_trap")[0] for _ in range(3))
    assert best < BUDGET_S, f"import qutip_trap took {best:.3f} s (budget {BUDGET_S} s)"


def test_the_root_import_loads_neither_qutip_nor_numpy() -> None:
    """The mechanism behind the budget: the root's ``__getattr__`` (1.6) must stay free of eager submodule imports."""
    _seconds, qutip_loaded, numpy_loaded = _import_in_a_fresh_interpreter("qutip_trap")
    assert not qutip_loaded and not numpy_loaded


def test_the_appendix_e_surface_imports_and_reports_its_time() -> None:
    """No budget here: ``qutip_trap.api`` imports qutip today (about 0.4 s); the number is printed for the CI log."""
    seconds, qutip_loaded, numpy_loaded = _import_in_a_fresh_interpreter("qutip_trap.api")
    print(
        f"import qutip_trap.api: {seconds:.3f} s (qutip loaded: {qutip_loaded}, numpy loaded: {numpy_loaded})"
    )


def test_the_root_names_are_the_lazy_table_and_resolve_after_import() -> None:
    """``qutip_trap.__all__`` is written out (so that a linter sees the TYPE_CHECKING re-exports) and must equal the lazy
    table plus the version; every name resolves through the module ``__getattr__`` and is then cached."""
    import qutip_trap

    assert set(qutip_trap.__all__) == {"__version__", *qutip_trap._RUNG_0}
    assert qutip_trap.Machine is qutip_trap.machine.Machine and "Machine" in vars(qutip_trap)
    assert qutip_trap.presets.yb171_chain.__module__ == "qutip_trap.presets"
    assert "Machine" in dir(qutip_trap) and "importlib" not in dir(qutip_trap)
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        qutip_trap.nothing  # noqa: B018
