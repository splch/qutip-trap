"""The public surface: every root name resolves and carries its own docstring, and every public dataclass of the package
is a frozen record that ``dataclasses.replace`` rebuilds from its fields."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
from pathlib import Path

import numpy as np
import pytest

import qutip_trap as trap
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.hilbert.space import ModeTruncation
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

MUTABLE_SERVICES = frozenset({"qutip_trap.dynamics.engine.JointExactEngine"})
"""Dataclasses that are service objects with state (a report, a cache, a progress hook), not records."""


def _attribute_docstring(module_name: str, name: str) -> str | None:
    """The string literal after ``name = ...`` in the defining module: the docstring of a type alias."""
    path = importlib.import_module(module_name).__file__
    assert path is not None
    body = ast.parse(Path(path).read_text(encoding="utf-8")).body
    for node, following in zip(body, body[1:]):
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == name:
            if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
                return str(following.value.value)
    return None


def _root_sources() -> dict[str, str]:
    """Root name -> the module it is imported from."""
    tree = ast.parse(Path(trap.__file__ or "").read_text(encoding="utf-8"))
    return {
        a.asname or a.name: n.module
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module
        for a in n.names
    }


def test_every_root_name_resolves_and_carries_a_docstring_of_its_own() -> None:
    assert len(trap.__all__) == len(set(trap.__all__))
    sources = _root_sources()
    for name in trap.__all__:
        obj = getattr(trap, name)
        if name == "__version__":
            continue
        if inspect.isclass(obj):
            doc = obj.__dict__.get("__doc__")
        elif inspect.isroutine(obj) or inspect.ismodule(obj):
            doc = obj.__doc__
        else:
            doc = _attribute_docstring(sources[name], name)
        if dataclasses.is_dataclass(obj) and isinstance(doc, str) and doc.startswith(obj.__name__ + "("):
            doc = None  # the signature dataclasses write when a class has no docstring
        assert (doc or "").strip(), name


def _public_dataclasses() -> list[type]:
    out: list[type] = []
    for info in pkgutil.walk_packages(trap.__path__, "qutip_trap."):
        if info.name.startswith("qutip_trap.interop"):
            continue  # needs the optional qiskit extra
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if (
                not name.startswith("_")
                and inspect.isclass(obj)
                and dataclasses.is_dataclass(obj)
                and obj.__module__ == module.__name__
            ):
                out.append(obj)
    return out


def test_every_public_dataclass_is_frozen_and_rebuilt_from_its_own_fields() -> None:
    """``dataclasses.replace(obj)`` reproduces ``obj`` exactly when the constructor takes exactly the fields (no
    ``init=False`` field, no ``InitVar``, no hand-written ``__init__``), which ``replace(machine, level=...)`` relies on."""
    classes = _public_dataclasses()
    assert len(classes) > 200
    for cls in classes:
        if f"{cls.__module__}.{cls.__name__}" in MUTABLE_SERVICES:
            continue
        name = cls.__qualname__
        assert cls.__dataclass_params__.frozen, f"{name} is not frozen"  # type: ignore[attr-defined]
        fields = dataclasses.fields(cls)
        assert all(f.init for f in fields), f"{name} has an init=False field"
        assert set(inspect.signature(cls).parameters) == {f.name for f in fields}, name


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
        Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1)),
        SolverOptions(),
        ModeTruncation(0, 4, (0, 1), 0.1),
    ]


@pytest.mark.parametrize("instance", _instances(), ids=lambda x: type(x).__name__)
def test_fixture_instances_round_trip_through_replace(instance: object) -> None:
    copy = dataclasses.replace(instance)  # type: ignore[type-var]  (every instance is a dataclass)
    assert type(copy) is type(instance)
    assert _equal(copy, instance), type(instance).__name__
