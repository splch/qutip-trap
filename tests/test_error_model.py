"""docs/api_implementation_plan.md 2.6: ``Machine.error_model`` emits the vendors' phenomenology from the simulated device.
IonQ's ``r_1q`` is recomputed from the gate channels' average infidelity by the stated formula, the RB budget's ``r_channel``
is the sum the error model's per-kind infidelities give, the QDK strings match the estimator's form, and the exporters carry
the fields the proposal names."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from qutip_trap.benchmarks import ErrorModel, error_model, gate_channel, randomized_benchmarking
from qutip_trap.benchmarks.budget import kind_of
from qutip_trap.benchmarks.error_model import QDK_TIME_PATTERN, SINGLE_QUBIT_KINDS, qdk_time
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Truncation
from tests.m6_fixtures import CircuitFixture, circuit_fixture

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))


@pytest.fixture(scope="module")
def fx() -> CircuitFixture:
    return circuit_fixture(2)


@pytest.fixture(scope="module")
def machine(fx: CircuitFixture) -> Machine:
    return Machine(fx.device, numerics=Numerics(truncation=Truncation(branch_weight_min=1e-3))).calibrated(
        pairs=[(0, 1)], detection_records=300, detection_windows_s=WINDOWS
    )


@pytest.fixture(scope="module")
def model(machine: Machine) -> ErrorModel:
    return machine.error_model()


def test_ionq_r_1q_is_twice_the_mean_single_qubit_average_infidelity_of_the_channels(
    machine: Machine, model: ErrorModel
) -> None:
    kinds = [kind_of(name, (q,)) for q in (0, 1) for name in SINGLE_QUBIT_KINDS]
    assert set(model.single_qubit_kinds) == set(kinds) and model.two_qubit_kinds == ("ms[0,1]",)
    r = [gate_channel(machine, k).infidelity_on((int(k[k.index("[") + 1 : -1]),)) for k in kinds]
    assert model.to_ionq_noise()["r_1q"] == pytest.approx(2.0 * sum(r) / len(r), rel=1e-12)
    r_2q = gate_channel(machine, "ms[0,1]").infidelity_on((0, 1))
    assert model.to_ionq_noise()["r_2q"] == pytest.approx(4.0 * r_2q / 3.0, rel=1e-12)
    assert model.infidelity["ms[0,1]"] == r_2q and all(model.infidelity[k] == v for k, v in zip(kinds, r))
    assert 0.0 < model.p_1q < 1e-2 and model.p_2q is not None and 0.0 < model.p_2q < 1e-1
    assert model.machine_hash == machine.hash() and model.entangler == "ms" and model.qubits == (0, 1)
    assert model.provenance["p_1q"] == "conv.depolarizing_normalization"


def test_the_rb_budget_composes_the_same_per_kind_infidelities(machine: Machine, model: ErrorModel) -> None:
    rb = randomized_benchmarking(machine, (0,), (1, 2), n_sequences=1, shots=20, budget=True)
    assert rb.budget is not None
    composed = sum(rb.budget.counts[k] * model.infidelity[k] for k in rb.budget.channel_infidelity)
    assert rb.budget.predicted["r_channel"] == pytest.approx(composed, rel=1e-9)
    for kind, value in rb.budget.channel_infidelity.items():
        assert model.infidelity[kind] == value  # one cache, one number


def test_the_durations_spam_and_rates_come_from_the_schedule_table_recipe_and_noise_model(
    machine: Machine, model: ErrorModel
) -> None:
    assert model.durations_s["ms[0,1]"] == pytest.approx(machine.table.waveform_for((0, 1)).duration_s)  # type: ignore[union-attr]
    assert model.durations_s["gpi[0]"] == pytest.approx(2.0 * model.durations_s["gpi2[0]"], rel=1e-6)
    assert model.durations_s["measure"] == machine.device.detector.window_s and model.measurement_time_s > 0.0
    table = machine.table
    assert table is not None
    assert model.p_meas[0] == (table.detection["eps_D"].value, table.detection["eps_B"].value)
    assert 0.0 <= model.p_init[0] < 1e-3 and model.p_init.keys() == {0, 1}
    assert model.dephasing_rate_per_s == {} and model.heating_rate_per_s == {}  # the fixture is quiet
    assert model.t_1q_s == pytest.approx(0.5 * (model.durations_s["gpi[0]"] + model.durations_s["gpi2[0]"]))


def test_the_exporters_carry_the_vendors_fields_and_the_qdk_strings_match_the_estimator_form(
    model: ErrorModel,
) -> None:
    q = model.to_quantinuum_error_params()
    assert set(q) == {"p1", "p2", "p_meas", "p_init", "linear_dephasing_rate", "quadratic_dephasing_rate"}
    assert q["p1"] == model.p_1q and q["p2"] == model.p_2q and q["p_meas"] == model.p_meas_mean
    assert q["linear_dephasing_rate"] == 0.0 and q["quadratic_dephasing_rate"] == 0.0
    k = model.to_qdk_qubit_params()
    assert k["instructionSet"] == "GateBased" and k["twoQubitGateErrorRate"] == model.p_2q
    for key in ("oneQubitMeasurementTime", "oneQubitGateTime", "tGateTime", "twoQubitGateTime"):
        assert QDK_TIME_PATTERN.match(k[key]), (key, k[key])
    assert k["twoQubitGateTime"] == "100 µs" and k["oneQubitMeasurementTime"] == "22 µs"
    assert k["idleErrorRate"] == pytest.approx(
        1.0 - math.exp(-model.dephasing_rate_mean_per_s * model.t_1q_s)
    )
    assert (
        qdk_time(5e-6) == "5 µs"
        and qdk_time(2.5e-3) == "2.5 ms"
        and qdk_time(3.0) == "3 s"
        and qdk_time(4e-9) == "4 ns"
    )
    assert re.fullmatch(QDK_TIME_PATTERN.pattern, qdk_time(1.23456789e-6))
    with pytest.raises(ValueError):
        qdk_time(-1.0)


def test_a_device_is_wrapped_and_qubits_can_be_restricted(fx: CircuitFixture, machine: Machine) -> None:
    one = error_model(machine, qubits=(1,))
    assert (
        one.qubits == (1,)
        and one.two_qubit_kinds == ()
        and one.p_2q is None
        and "r_2q" not in one.to_ionq_noise()
    )
    assert "twoQubitGateTime" not in one.to_qdk_qubit_params()
    with pytest.raises(ValueError, match="distinct ions"):
        error_model(machine, qubits=(0, 0))
    bare = error_model(fx.device, qubits=(0,))
    assert bare.single_qubit_kinds == ("gpi[0]", "gpi2[0]") and bare.machine_hash != machine.hash()
