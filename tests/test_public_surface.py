"""Every public dataclass of the package is a frozen record that ``dataclasses.replace`` rebuilds from its own fields."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import qutip_trap
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.dynamics.space import ModeTruncation
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

MUTABLE_SERVICES = frozenset({"JointExactEngine"})
"""Service objects with state (a report, a cache, a progress hook), not records."""


def _public_dataclasses() -> list[type]:
    found: list[type] = []
    for info in pkgutil.walk_packages(qutip_trap.__path__, "qutip_trap."):
        if info.name.startswith("qutip_trap.interop"):  # behind the optional qiskit extra
            continue
        module = importlib.import_module(info.name)
        found += [
            obj
            for name, obj in vars(module).items()
            if inspect.isclass(obj)
            and dataclasses.is_dataclass(obj)
            and obj.__module__ == info.name
            and not name.startswith("_")
            and name not in MUTABLE_SERVICES
        ]
    return found


def test_every_public_dataclass_is_frozen_and_takes_exactly_its_fields() -> None:
    """``dataclasses.replace(obj)`` reproduces ``obj`` only when the constructor takes exactly the fields."""
    offenders = []
    for cls in _public_dataclasses():
        fields = dataclasses.fields(cls)
        if not (
            cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
            and all(f.init for f in fields)
            and set(inspect.signature(cls).parameters) == {f.name for f in fields}
        ):
            offenders.append(f"{cls.__module__}.{cls.__qualname__}")
    assert not offenders, offenders


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
