"""Section 8.6's per-shot noise attribution and v2 register-nested export (M6 audit items B-6 and 18).

Section 8.6: "A ``Result`` carries per shot: ... the sampled quasi-static noise parameters of that shot's dynamical sample"
and "a v2 register-nested format ... keyed by bitstrings per named register". ``Result.noise_samples`` is per SAMPLE, and the
shot -> sample map lived only in a local variable of ``run``; ``Diagnostics.shots_per_sample`` is a floor, so the map could
not be reconstructed when ``shots % samples != 0``.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from qutip_trap.control.compiler import Circuit
from qutip_trap.run.results import Result
from tests.fixtures import make_result

BITS = np.array([[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)


def test_sample_of_shot_is_the_contiguous_block_map() -> None:
    """Shots are allocated in contiguous blocks (``conv.shot_blocks_per_sample``), so sample k owns
    [sum_{j<k} M_j, sum_{j<=k} M_j) and ``noise_samples[sample_of_shot[k]]`` is shot k's parameter draw."""
    result = make_result(BITS)
    # one sample (the default fixture): every shot maps to it
    assert result.sample_of_shot.tolist() == [0, 0, 0, 0, 0]
    uneven = dataclasses.replace(
        result,
        diagnostics=dataclasses.replace(result.diagnostics, shots_per_sample_realized=(2, 2, 1)),
    )
    assert uneven.sample_of_shot.tolist() == [0, 0, 1, 1, 2]
    # the floored shots_per_sample cannot express this: 5 shots over 3 samples floors to 1
    assert uneven.diagnostics.shots_per_sample != 2
    assert sum(uneven.diagnostics.shots_per_sample_realized) == uneven.shots


def test_branches_and_trajectories_are_separate_diagnostics() -> None:
    """``trajectories`` counts Fock branches TIMES the engine's trajectory count (``conv.fock_sum_branches``); ``branches``
    reports the Fock sum alone, so a reader can tell one from the other."""
    result = make_result(BITS)
    assert result.diagnostics.branches >= 1
    assert result.diagnostics.trajectories >= result.diagnostics.branches


# ---- the IonQ character orders, pinned before the v2 exporter changes (docs/api_implementation_plan.md items 0.4 and 1.7) ---

X_ON_QUBIT_ZERO = np.array([[1, 0, 0]] * 4, dtype=np.uint8)
"""``x q[0]`` on three qubits, four shots: column j of ``Result.bitstrings`` is qubit j."""
IONQ_V1_KEY = "1"
"""The v1 decimal key of that shot: qubit 0 the least-significant bit."""
IONQ_V2_KEY = "100"
"""The v2 bitstring of that shot: wire order, ``q[0]`` the leftmost character (``qutip_trap.io.ionq`` records the source)."""


def test_x_on_qubit_zero_reads_1_in_the_v1_formats_and_001_in_this_package() -> None:
    """A Bell state cannot tell the two character orders apart; ``x q[0]`` can. The v1 exporters are correct today."""
    result = make_result(X_ON_QUBIT_ZERO)
    assert result.counts == {"001": 4}  # this package's key: qubit 0 rightmost
    assert result.to_ionq_v1_probabilities() == {IONQ_V1_KEY: 1.0}
    assert result.to_ionq_v1_histogram() == {IONQ_V1_KEY: 4}
    assert result.to_ionq_v1_shots() == [IONQ_V1_KEY] * 4
    assert IONQ_V2_KEY == "001"[::-1]  # the v2 string is this package's key reversed
    # the 0.1.0 names (to_ionq_json, to_ionq_histogram, to_ionq_shots, to_ionq_v2) were deprecated in 0.2.0 and removed in
    # 0.4.0 (docs/deprecations.md, "Removed"); a caller of one gets the plain AttributeError, not a silent alias
    for gone in ("to_ionq_json", "to_ionq_histogram", "to_ionq_shots", "to_ionq_v2"):
        assert not hasattr(result, gone), gone


def test_x_on_qubit_zero_reads_100_in_the_v2_envelope() -> None:
    """The fixture of item 0.4, now met by the exporters of 1.7: the v0.4 envelope, ``output_all`` in wire order (q[0] the
    leftmost character), the default register ``c`` beside it."""
    result = make_result(X_ON_QUBIT_ZERO)
    assert result.to_ionq_v2_probabilities() == {
        "probabilities": {"registers": {"output_all": {IONQ_V2_KEY: 1.0}, "c": {IONQ_V2_KEY: 1.0}}}
    }
    assert result.to_ionq_v2_histogram() == {
        "histogram": {"registers": {"output_all": {IONQ_V2_KEY: 4}, "c": {IONQ_V2_KEY: 4}}}
    }
    shots = result.to_ionq_v2_shots()["shots"]
    assert len(shots) == 4 and shots[0] == {"registers": {"output_all": [1, 0, 0], "c": [1, 0, 0]}}


def test_the_v2_exporters_follow_the_circuit_registers_and_the_measured_columns() -> None:
    """Columns hold the measured qubits (``qubits``); a register lists its qubits in bit order, the first bit the first
    character; ``output_all`` is every measured qubit ascending."""
    result = dataclasses.replace(
        make_result(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)),
        qubits=(0, 2),
        registers={"a": (2,), "b": (2, 0)},
    )
    v2 = result.to_ionq_v2_probabilities()["probabilities"]["registers"]
    assert v2 == {
        "output_all": {"10": pytest.approx(2 / 3), "01": pytest.approx(1 / 3)},
        "a": {"0": pytest.approx(2 / 3), "1": pytest.approx(1 / 3)},
        "b": {"01": pytest.approx(2 / 3), "10": pytest.approx(1 / 3)},
    }
    with pytest.raises(ValueError, match="did not measure"):
        dataclasses.replace(result, registers={"r": (1,)}).to_ionq_v2_probabilities()
    with pytest.raises(ValueError, match="names each column"):
        dataclasses.replace(result, qubits=(0, 0))


def test_reversed_bits_puts_qubit_zero_leftmost_and_is_its_own_inverse() -> None:
    result = make_result(X_ON_QUBIT_ZERO)
    flipped = result.reversed_bits()
    assert (
        flipped.bit_order == "qubit0_msb"
        and flipped.counts == {"100": 4}
        and flipped.probabilities == {"100": 1.0}
    )
    assert flipped.qubits == (2, 1, 0) and np.array_equal(flipped.bitstrings, result.bitstrings[:, ::-1])
    back = flipped.reversed_bits()
    assert (
        back.bit_order == "qubit0_lsb"
        and back.counts == result.counts
        and np.array_equal(back.bitstrings, result.bitstrings)
    )
    with pytest.raises(ValueError, match="bit order"):
        dataclasses.replace(result, bit_order="little")  # type: ignore[arg-type]


def test_from_ionq_v1_shots_closes_the_round_trip() -> None:
    shots = ["6", "1", "0", "7"]
    result = Result.from_ionq_v1_shots(shots, 3)
    assert result.to_ionq_v1_shots() == shots and set(result.counts) == {"110", "001", "000", "111"}
    assert result.shots == 4 and result.spam == {} and result.diagnostics.samples == 0
    assert result.diagnostics.calibration.uncalibrated() == ("field",)
    assert any("no simulation behind these shots" in a for a in result.diagnostics.approximations)
    assert result.error_bars["110"] == pytest.approx(math.sqrt(0.25 * 0.75 / 4))
    assert Result.from_ionq_v1_shots([6, 1], 3).counts == {"110": 1, "001": 1}


def test_the_envelope_round_trips_through_json() -> None:
    result = dataclasses.replace(
        make_result(BITS),
        spam={"q0": (1e-3, 2e-3), "q1": (3e-4, 4e-4)},
        heralds=np.array([0, 1, 2, 4, 3], dtype=np.uint8),
        machine_hash="m" * 64,
        created_at="2026-09-11T12:00:00+00:00",
        duration_s=8.5,
    )
    d = result.to_dict()
    assert json.loads(json.dumps(d)) == d, "plain JSON values only"
    assert d["schema_version"] == 1 and d["device_hash"] == "fixture" and d["machine_hash"] == "m" * 64
    assert d["heralds"] == {"collision": 2, "dark_or_lost": 2, "count_anomaly": 1}
    assert (
        d["diagnostics"]["level"] == "JOINT_EXACT"
        and d["diagnostics"]["calibration"]["entries"]["field"]["value"] == 5.0
    )
    back = Result.from_dict(d)
    assert (
        back.counts == result.counts
        and back.probabilities == result.probabilities
        and back.spam == result.spam
    )
    assert back.error_bars == result.error_bars and back.registers is None and back.qubits == (0, 1)
    assert (
        back.machine_hash == result.machine_hash
        and back.created_at == result.created_at
        and back.duration_s == 8.5
    )
    assert (
        back.diagnostics.space == result.diagnostics.space
        and back.diagnostics.mode_class == result.diagnostics.mode_class
    )
    assert back.diagnostics.calibration.entries()["field"] == result.diagnostics.calibration.field
    assert back.diagnostics.calibration.ms == {} and back.diagnostics.gate_local is None
    assert (
        back.shots == result.shots and int(back.heralds.sum()) == 0
    )  # rebuilt from the counts: no per-shot flags
    full = result.to_dict(per_shot=True)
    again = Result.from_dict(full)
    assert np.array_equal(again.bitstrings, result.bitstrings) and np.array_equal(
        again.heralds, result.heralds
    )
    with pytest.raises(ValueError, match="schema version 1"):
        Result.from_dict({**d, "schema_version": 2})


def test_dump_job_writes_the_v0_4_body_and_load_job_reads_v0_3_and_v0_4() -> None:
    from qutip_trap.control.native import rad_from_turns
    from qutip_trap.io.ionq import JOB_KEYS, IonQJob, dump_job, dumps, load_job, loads

    circuit = Circuit(2).gpi2(0, rad_from_turns(0.25)).ms(0, 1, 0.0, 0.0, rad_from_turns(0.25))
    body = dump_job(
        circuit,
        backend="simulator",
        shots=2000,
        noise={"model": "aria-1", "seed": 7},
        settings={"error_mitigation": {"debiasing": False}},
        name="bell",
        metadata={"run": "1"},
    )
    # the v0.4 CircuitJobCreationPayload: required type, backend, input; additionalProperties false (IonQ OpenAPI v0.4,
    # spec dated 2026-09-10, read through ionq-core 0.1.1's openapi.json)
    assert {"type", "backend", "input"} <= set(body) <= JOB_KEYS
    assert body["type"] == "ionq.circuit.v1" and body["backend"] == "simulator" and body["shots"] == 2000
    assert set(body["input"]) == {"gateset", "qubits", "circuit"} and body["noise"] == {
        "model": "aria-1",
        "seed": 7,
    }
    assert json.loads(json.dumps(body)) == body
    job = load_job(body)
    assert job == IonQJob(
        circuit,
        "simulator",
        2000,
        "bell",
        {"run": "1"},
        {"model": "aria-1", "seed": 7},
        {"error_mitigation": {"debiasing": False}},
        None,
    )
    assert (
        load_job({"target": "qpu.aria-1", "input": body["input"]}).backend == "qpu.aria-1"
    )  # the v0.3 spelling
    assert load_job(json.dumps(body)).circuit == circuit
    assert loads(dumps(circuit)) == circuit and loads(body) == circuit
    with pytest.raises(ValueError, match=r"no key \['target'\]"):
        dump_job(circuit, backend="simulator", noise={"target": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="'model'"):
        dump_job(circuit, backend="simulator", noise={"seed": 1})
    with pytest.raises(ValueError, match="settings.error_mitigation has no key"):
        dump_job(circuit, backend="simulator", settings={"error_mitigation": {"debias": True}})
    with pytest.raises(ValueError, match="ionq.qasm3.v1"):
        load_job({"type": "ionq.qasm3.v1", "input": {"data": "OPENQASM 3.0;"}})
    with pytest.raises(ValueError, match="compile first"):
        dump_job(Circuit(1).h(0), backend="simulator")
