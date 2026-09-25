"""docs/api_implementation_plan.md 2.1: ``run`` is ``Machine.run`` with the machine built from its keyword arguments. Every
keyword argument of 0.1.0, passed the old way, warns once (naming the keyword, the deadline and its new home) and produces
the result the object form produces; the two forms never merge silently; the pipeline behind both is ``run.pipeline.execute``."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from qutip_trap._compat import QutipTrapDeprecationWarning
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.dynamics.hamiltonian import BuilderOptions
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Parallel, Physics, Readout, Truncation, to_run_kwargs
from qutip_trap.readout.discriminate import ThresholdDiscriminator
from qutip_trap.run.job import LEGACY_DEADLINE, LEGACY_RUN_KEYWORDS, machine_with_run_kwargs, run
from qutip_trap.run.pipeline import execute
from tests.m6_fixtures import CircuitFixture, circuit_fixture

ONE = Circuit(2, (Operation("gpi2", (0,), (0.0,)),), (0, 1))
BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = Numerics(truncation=Truncation(branch_weight_min=1e-3))
SCANS: dict[str, Any] = {"pairs": [(0, 1)], "detection_records": 300, "detection_windows_s": WINDOWS}


@pytest.fixture(scope="module")
def fx() -> CircuitFixture:
    return circuit_fixture(2)


@pytest.fixture(scope="module")
def machine(fx: CircuitFixture) -> Machine:
    return dataclasses.replace(Machine(fx.device).calibrated(**SCANS), numerics=FAST)


def same_result(a: Any, b: Any) -> None:
    assert np.array_equal(a.bitstrings, b.bitstrings) and np.array_equal(a.heralds, b.heralds)
    assert a.counts == b.counts and a.probabilities == b.probabilities and a.spam == b.spam
    assert a.machine_hash == b.machine_hash and a.qubits == b.qubits
    for f in dataclasses.fields(a.diagnostics):
        assert getattr(a.diagnostics, f.name) == getattr(b.diagnostics, f.name), f.name


Transform = Callable[[Machine, CircuitFixture], Machine]


def _physics(**fields: Any) -> Transform:
    return lambda m, _fx: dataclasses.replace(m, physics=dataclasses.replace(m.physics, **fields))


def _truncation(**fields: Any) -> Transform:
    return lambda m, _fx: dataclasses.replace(
        m,
        numerics=dataclasses.replace(
            m.numerics, truncation=dataclasses.replace(m.numerics.truncation, **fields)
        ),
    )


def _parallel(**fields: Any) -> Transform:
    return lambda m, _fx: dataclasses.replace(
        m,
        numerics=dataclasses.replace(m.numerics, parallel=dataclasses.replace(m.numerics.parallel, **fields)),
    )


def _readout(**fields: Any) -> Transform:
    return lambda m, _fx: dataclasses.replace(m, readout=dataclasses.replace(m.readout, **fields))


OPTS = SolverOptions(branch_weight_min=1e-3, atol=1e-9, rtol=1e-7)
CASES: dict[str, tuple[Callable[[CircuitFixture], dict[str, Any]], Transform, Circuit]] = {
    "t0_s": (lambda _fx: {"t0_s": 1.5e-3}, _physics(t0_s=1.5e-3), ONE),
    "shot_period_s": (lambda _fx: {"shot_period_s": 2e-3}, _physics(shot_period_s=2e-3), ONE),
    "noise": (lambda _fx: {"noise": False}, _physics(noise=False), ONE),
    "internal_levels": (lambda _fx: {"internal_levels": 3}, _physics(internal_levels=3), ONE),
    "crosstalk_suppression": (
        lambda _fx: {"crosstalk_suppression": "neighbour"},
        _physics(crosstalk_suppression="neighbour"),
        BELL,
    ),
    "stark_compensation": (
        lambda _fx: {"stark_compensation": False},
        _physics(stark_compensation=False),
        ONE,
    ),
    "entangler": (lambda _fx: {"entangler": "zz"}, _physics(entangler="zz"), BELL),
    "channels": (lambda _fx: {"channels": ()}, _physics(extra_channels=()), ONE),
    "builder_options": (
        lambda _fx: {"builder_options": BuilderOptions(lamb_dicke_order=2)},
        _physics(builder=BuilderOptions(lamb_dicke_order=2)),
        ONE,
    ),
    "options": (
        lambda _fx: {"options": OPTS},
        lambda m, _fx: dataclasses.replace(
            m, numerics=Numerics.from_solver_options(OPTS), physics=Physics.from_solver_options(OPTS)
        ),
        ONE,
    ),
    "caps": (lambda _fx: {"caps": {0: 9}}, _truncation(caps={0: 9}), ONE),
    "enr_group": (lambda _fx: {"enr_group": None}, _truncation(enr_group=None), ONE),
    "samples": (lambda _fx: {"samples": 2}, _parallel(samples=2), ONE),
    "parallel": (lambda _fx: {"parallel": False}, _parallel(addressing=False), ONE),
    "readout": (lambda _fx: {"readout": "full"}, _readout(mode="full"), ONE),
    "discriminator": (
        lambda _fx: {"discriminator": ThresholdDiscriminator(0.5, 20e-6)},
        _readout(discriminator=ThresholdDiscriminator(0.5, 20e-6)),
        ONE,
    ),
    "povm_samples": (lambda _fx: {"povm_samples": 2000}, _readout(povm_samples=2000), ONE),
    "gate_drives": (lambda fx: {"gate_drives": fx.gate_drives}, lambda m, _fx: m, ONE),
    "entangling_drives": (lambda fx: {"entangling_drives": fx.entangling_drives}, lambda m, _fx: m, ONE),
}


def test_the_case_table_covers_every_legacy_keyword() -> None:
    assert set(CASES) | {"space", "calibrate_kwargs"} == set(LEGACY_RUN_KEYWORDS)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_legacy_keyword_warns_once_and_equals_the_object_form(
    name: str, fx: CircuitFixture, machine: Machine
) -> None:
    legacy_of, transform, circuit = CASES[name]
    legacy = legacy_of(fx)
    shots = 40 if circuit is ONE else 60
    expected = transform(machine, fx).run(circuit, shots, seed=3)
    # ``options`` carries the whole SolverOptions (branch_weight_min included), so it is not passed beside numerics=
    base: dict[str, Any] = {} if name == "options" else {"numerics": FAST}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = run(circuit, fx.device, shots, table=machine.table, seed=3, **base, **legacy)
    ours = [w for w in caught if issubclass(w.category, QutipTrapDeprecationWarning)]
    assert len(ours) == 1, [str(w.message) for w in ours]
    text = str(ours[0].message)
    assert f"'{name}' argument of qutip_trap.run.job.run" in text and LEGACY_DEADLINE in text
    assert text.endswith(LEGACY_RUN_KEYWORDS[name]) and ours[0].filename == __file__
    same_result(got, expected)


def test_a_supplied_space_is_the_truncation_space(fx: CircuitFixture, machine: Machine) -> None:
    space = machine.estimate(ONE).space
    expected = _truncation(space=space)(machine, fx).run(ONE, 40, seed=3)
    with pytest.warns(QutipTrapDeprecationWarning, match="'space' argument"):
        got = run(ONE, fx.device, 40, table=machine.table, numerics=FAST, seed=3, space=space)
    same_result(got, expected)
    assert "space supplied by the caller" in got.diagnostics.approximations


def test_calibrate_kwargs_pin_the_table_the_prefix_would_have_built(fx: CircuitFixture) -> None:
    scans = {"detection_records": 300, "detection_windows_s": WINDOWS}
    pinned = Machine(fx.device, numerics=FAST).calibrated(seed=3, pairs=list(ONE.entangling_pairs()), **scans)
    expected = pinned.run(ONE, 40, seed=3)
    with pytest.warns(QutipTrapDeprecationWarning, match="'calibrate_kwargs' argument"):
        got = run(ONE, fx.device, 40, numerics=FAST, seed=3, calibrate_kwargs=scans)
    same_result(got, expected)
    assert got.diagnostics.calibration == pinned.table


def test_a_whole_solver_options_beside_an_object_is_refused_and_a_field_merges(
    fx: CircuitFixture, machine: Machine
) -> None:
    with (
        pytest.raises(TypeError, match="'options' beside numerics="),
        pytest.warns(QutipTrapDeprecationWarning),
    ):
        run(ONE, fx.device, 10, table=machine.table, numerics=FAST, options=SolverOptions())
    with (
        pytest.raises(TypeError, match="'options' beside physics="),
        pytest.warns(QutipTrapDeprecationWarning),
    ):
        run(ONE, fx.device, 10, table=machine.table, physics=Physics(), options=SolverOptions())
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run(ONE, fx.device, 10, table=machine.table, shotz=3)
    # a field-level keyword beside its object sets that field, the keyword winning
    with pytest.warns(QutipTrapDeprecationWarning, match="'noise' argument"):
        m = machine_with_run_kwargs(
            fx.device, {"noise": False}, physics=Physics(noise=True, internal_levels=3)
        )
    assert m.physics == Physics(noise=False, internal_levels=3)
    with pytest.warns(QutipTrapDeprecationWarning, match="'povm_samples' argument"):
        m = machine_with_run_kwargs(fx.device, {"povm_samples": 5}, readout=Readout(mode="full"))
    assert m.readout == Readout(mode="full", povm_samples=5)


def test_the_objects_are_accepted_as_mappings_and_a_device_becomes_a_default_machine(
    fx: CircuitFixture,
) -> None:
    m = machine_with_run_kwargs(
        fx.device,
        {},
        physics={"noise": False},
        numerics={"parallel": {"map": "serial"}},
        readout={"mode": "full"},
    )
    assert m.device is fx.device and m.physics == Physics(noise=False) and m.readout == Readout(mode="full")
    assert m.numerics.parallel == Parallel(map="serial") and m.table is None
    assert machine_with_run_kwargs(fx.device, {}).physics == Physics()


def test_machine_run_is_execute_and_the_translator_is_deprecated(
    fx: CircuitFixture, machine: Machine
) -> None:
    direct = execute(machine, ONE, 20, seed=5)
    same_result(direct, machine.run(ONE, 20, seed=5))
    assert direct.machine_hash == machine.hash()
    with pytest.warns(QutipTrapDeprecationWarning, match="to_run_kwargs is deprecated"):
        kw = to_run_kwargs(machine.physics, machine.numerics, machine.readout)
    assert set(kw) < set(LEGACY_RUN_KEYWORDS)
