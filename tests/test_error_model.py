"""``Machine.error_model``: IonQ's ``r_1q`` and ``r_2q`` from the gate channels' average infidelities, the RB budget on the
same cached numbers, the durations, SPAM and rates from the schedule, table and noise model, and the exporters' fields."""

from __future__ import annotations

import math
import re

import pytest

from qutip_trap.benchmarks.budget import gate_channel, kind_of
from qutip_trap.benchmarks.error_model import (
    QDK_TIME_PATTERN,
    SINGLE_QUBIT_KINDS,
    ErrorModel,
    error_model,
    qdk_time,
)
from qutip_trap.benchmarks.rb import randomized_benchmarking
from qutip_trap.device.presets import yb171_chain
from qutip_trap.machine import Machine
from qutip_trap.options import Numerics, Truncation
from tests.fixtures import WINDOWS


@pytest.fixture(scope="module")
def machine() -> Machine:
    return Machine(
        yb171_chain(2).device, numerics=Numerics(truncation=Truncation(branch_weight_min=1e-3))
    ).calibrated(pairs=[(0, 1)], detection_records=300, detection_windows_s=WINDOWS)


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
    assert gate_channel(machine, kinds[0]) is gate_channel(machine, kinds[0]), (
        "the channels are cached per machine and kind"
    )


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
    table = machine.table
    assert table is not None
    assert model.durations_s["ms[0,1]"] == pytest.approx(table.waveform_for((0, 1)).duration_s)  # type: ignore[union-attr]
    assert model.durations_s["gpi[0]"] == pytest.approx(2.0 * model.durations_s["gpi2[0]"], rel=1e-6)
    assert model.durations_s["measure"] == machine.device.detector.window_s and model.measurement_time_s > 0.0
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
    assert (qdk_time(5e-6), qdk_time(2.5e-3), qdk_time(3.0), qdk_time(4e-9)) == (
        "5 µs",
        "2.5 ms",
        "3 s",
        "4 ns",
    )
    assert re.fullmatch(QDK_TIME_PATTERN.pattern, qdk_time(1.23456789e-6))
    with pytest.raises(ValueError):
        qdk_time(-1.0)


def test_qubits_can_be_restricted(machine: Machine) -> None:
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
