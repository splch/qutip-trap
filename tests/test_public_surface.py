"""The public surface, name by name (docs/api_implementation_plan.md item 0.2, after Cirq's ``json_test_data/spec.py``, which
forces a decision per name): every ``__all__`` name of every rung module resolves, carries a docstring of its own and appears in
backticks on a documentation page, and every frozen dataclass among them round-trips through ``dataclasses.replace``.

Phase 1 added the rung modules to ``RUNG_MODULES`` as it created them; Phase 3.4 (0.4.0) gave every rung its page
(``PAGE_OF``), so every name must appear on its own rung's page and the list of not-yet-documented 0.1.0 names is gone."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from qutip_trap import api
from qutip_trap.provenance import repository_root
from tests.fixtures import (
    make_calibration_table,
    make_crystal,
    make_detector,
    make_device,
    make_diagnostics,
    make_field,
    make_hardware,
    make_noise,
    make_raman_pair,
    make_result,
    make_run_state,
    make_space,
    make_trap,
)

ROOT = repository_root()
DOCS = ROOT / "docs"

RUNG_MODULES: tuple[str, ...] = (
    "qutip_trap.api",
    "qutip_trap",
    "qutip_trap.circuit",
    "qutip_trap.schedule",
    "qutip_trap.dynamics",
    "qutip_trap.physics",
    "qutip_trap.presets",
    "qutip_trap.io",
    "qutip_trap.io.qasm2",
    "qutip_trap.io.ionq",
    "qutip_trap.experiments",
    "qutip_trap.calibration",
    "qutip_trap.benchmarks",
    "qutip_trap.experimental",
)
"""The modules whose ``__all__`` is the public surface: the Appendix E surface, the rung modules of 0.2.0
(docs/api_implementation_plan.md 1.6), the laboratory (0.3.0) and ``qutip_trap.experimental`` (3.3; 0.4.0)."""

PAGES: tuple[Path, ...] = tuple(
    DOCS / name
    for name in (
        "machine.md",
        "circuit.md",
        "schedule.md",
        "dynamics.md",
        "physics.md",
        "laboratory.md",
        "experimental.md",
        "physics_notes.md",
        "conventions.md",
        "examples.md",
        "limits.md",
        "deprecations.md",
        "README.md",
    )
)
"""The documentation pages: since 0.4.0 the ladder (one page per rung, one for the laboratory, one for the experimental
namespace), then the pages every release had (the two planning documents under docs/ are not pages)."""

PAGE_OF: dict[str, str] = {
    "qutip_trap": "machine.md",
    "qutip_trap.presets": "machine.md",
    "qutip_trap.circuit": "circuit.md",
    "qutip_trap.io": "circuit.md",
    "qutip_trap.io.qasm2": "circuit.md",
    "qutip_trap.io.ionq": "circuit.md",
    "qutip_trap.schedule": "schedule.md",
    "qutip_trap.dynamics": "dynamics.md",
    "qutip_trap.physics": "physics.md",
    "qutip_trap.experiments": "laboratory.md",
    "qutip_trap.calibration": "laboratory.md",
    "qutip_trap.benchmarks": "laboratory.md",
    "qutip_trap.experimental": "experimental.md",
}
"""Each rung module's own page (docs/api_implementation_plan.md 3.4: "the public-surface test requires every rung name on its
page"); ``qutip_trap.api``, the compatibility surface, may document a name on any page."""


def _module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _surface(module_name: str) -> list[tuple[str, object]]:
    module = _module(module_name)
    return [(name, getattr(module, name)) for name in module.__all__]


@cache
def _source_modules(module_name: str) -> dict[str, str]:
    """Exported name -> the module that defines it, from the rung module's ``from ... import`` statements (a name the rung
    module defines itself maps to the rung module)."""
    module = _module(module_name)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.ClassDef | ast.FunctionDef):
            targets = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id]
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                else [node.name]
            )
            for t in targets:
                out[t] = module_name
    return out


def _attribute_docstring(module_name: str, name: str) -> str | None:
    """The string literal that follows ``name = ...`` at the top level of the module (a type alias's or NewType's docstring)."""
    module = _module(module_name)
    assert module.__file__ is not None
    body = ast.parse(Path(module.__file__).read_text(encoding="utf-8")).body
    for node, following in zip(body, body[1:]):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if name in targets:
            if (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                return following.value.value
            return None
    return None


def _own_docstring(module_name: str, name: str, obj: object) -> str | None:
    """The docstring ``name`` itself carries: a class's own (never an inherited one, never the signature a dataclass
    generates), a function's, a module's, or the attribute docstring of a type alias or NewType in its defining module."""
    if inspect.ismodule(obj):
        return obj.__doc__
    if inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__")
        if not isinstance(doc, str):
            return None
        if dataclasses.is_dataclass(obj) and doc.startswith(obj.__name__ + "("):
            return None  # dataclasses.dataclass writes the signature when the class has no docstring
        return doc
    if inspect.isroutine(obj):
        doc = getattr(obj, "__doc__", None)
        return doc if isinstance(doc, str) else None
    source = (
        _source_modules(module_name).get(name)
        or _lazy_source(module_name, name)
        or getattr(obj, "__module__", None)
    )
    if source is None:
        return None
    return _attribute_docstring(source, name)


def _lazy_source(module_name: str, name: str) -> str | None:
    """The defining module of a name a rung module imports on first use (the root's ``_RUNG_0`` table, the dynamics rung's
    ``_LAZY`` table: name -> (module, attribute))."""
    module = _module(module_name)
    for table_name in ("_RUNG_0", "_LAZY"):
        table = getattr(module, table_name, None)
        if isinstance(table, dict) and name in table:
            return str(table[name][0])
    return None


@cache
def _code_spans(page: Path) -> tuple[str, ...]:
    """Every backticked span of one documentation page: fenced blocks and inline code."""
    text = page.read_text(encoding="utf-8")
    spans: list[str] = re.findall(r"```[A-Za-z]*\n(.*?)```", text, flags=re.S)
    # strip the fenced blocks (a fence opens with ``` and a newline; the prose "```python block" of examples.md is not one)
    spans += re.findall(r"`([^`\n]+)`", re.sub(r"```[A-Za-z]*\n.*?```", "", text, flags=re.S))
    return tuple(spans)


def _documented(name: str, pages: tuple[Path, ...] = PAGES) -> bool:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    return any(pattern.search(span) for page in pages for span in _code_spans(page))


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_public_name_resolves(module_name: str) -> None:
    module = _module(module_name)
    assert len(module.__all__) == len(set(module.__all__)), "a name is exported twice"
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.__all__ names {name!r}, which does not resolve"


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_public_name_carries_a_docstring_of_its_own(module_name: str) -> None:
    missing = [
        name
        for name, obj in _surface(module_name)
        if not (_own_docstring(module_name, name, obj) or "").strip()
    ]
    assert not missing, f"public names of {module_name} without a docstring: {missing}"


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_public_name_is_documented_on_its_page(module_name: str) -> None:
    """A name is public when it is documented (Qiskit's rule): it appears in backticks, alone or inside a code span or a
    fenced block, on its rung's page (``PAGE_OF``; any page for the Appendix E surface). The 0.1.0 names that lacked a line
    were listed in this test until 0.4.0 gave every rung its page (docs/api_implementation_plan.md 3.4); the list is gone
    and a new public name fails here until its line is written."""
    module = _module(module_name)
    pages = PAGES if module_name == "qutip_trap.api" else (DOCS / PAGE_OF[module_name],)
    undocumented = sorted(name for name in module.__all__ if not _documented(name, pages))
    where = "a documentation page" if module_name == "qutip_trap.api" else PAGE_OF[module_name]
    assert not undocumented, (
        f"public names of {module_name} with no line on {where} (write the line): {undocumented}"
    )


def test_every_page_of_the_ladder_exists() -> None:
    for page in PAGES:
        assert page.exists(), page.name
    assert set(PAGE_OF.values()) <= {p.name for p in PAGES}


MUTABLE_SERVICES: frozenset[str] = frozenset({"JointExactEngine"})
"""Dataclasses of the surface that are service objects with state (a report, a cache, a progress hook), not records: the
Appendix E immutability rule is for the data; these are excluded from the frozen check with this reason."""


def _dataclasses_of(module_name: str) -> list[tuple[str, type]]:
    return [
        (name, obj)
        for name, obj in _surface(module_name)
        if inspect.isclass(obj) and dataclasses.is_dataclass(obj)
    ]


@pytest.mark.parametrize("module_name", RUNG_MODULES)
def test_every_dataclass_is_frozen_and_rebuilt_from_its_own_fields(module_name: str) -> None:
    """``dataclasses.replace(obj)`` calls ``type(obj)(**{f.name: getattr(obj, f.name)})`` for the ``init`` fields: it
    reproduces ``obj`` when and only when the constructor takes exactly the fields (no ``init=False`` field, no ``InitVar``, no
    hand-written ``__init__``), which is what ``dataclasses.replace(machine, level=...)`` relies on from 0.2.0."""
    for name, cls in _dataclasses_of(module_name):
        if name in MUTABLE_SERVICES:
            continue
        assert cls.__dataclass_params__.frozen, f"{name} is not frozen"
        fields = dataclasses.fields(cls)
        assert all(f.init for f in fields), f"{name} has an init=False field"
        assert set(inspect.signature(cls).parameters) == {f.name for f in fields}, (
            f"{name}'s constructor does not take exactly its fields"
        )


def _equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _instances() -> list[object]:
    """Frozen dataclasses of the surface as the shared fixtures build them, plus the cheap ones built here."""
    table = make_calibration_table()
    return [
        make_device(),
        make_crystal(),
        make_trap(),
        make_field(),
        make_noise(),
        make_detector(),
        make_hardware(),
        *make_raman_pair(),
        make_run_state(),
        table,
        table.field,
        make_space(),
        make_diagnostics(),
        make_result(np.array([[0, 1], [1, 1], [0, 0]], dtype=np.uint8)),
        api.Circuit(2, (api.Operation("h", (0,), ()), api.Operation("cnot", (0, 1), ())), (0, 1)),
        api.SolverOptions(),
        api.ModeTruncation(0, 4, (0, 1), 0.1),
    ]


@pytest.mark.parametrize("instance", _instances(), ids=lambda x: type(x).__name__)
def test_fixture_instances_round_trip_through_replace(instance: object) -> None:
    copy = dataclasses.replace(instance)  # type: ignore[type-var]  (every instance is a dataclass)
    assert type(copy) is type(instance)
    assert _equal(copy, instance), type(instance).__name__
