"""Native gates exactly (PLAN.md Section 7.6), the IonQ circuit and job formats, and the Result's exports: the v1 decimal
keys, the v2 register envelope in wire order, the reversed-bit view and the versioned record."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qutip_trap.control import native
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.io.ionq import (
    JOB_KEYS,
    IonQJob,
    dump_ionq_json,
    dump_job,
    load_ionq_json,
    load_job,
)
from qutip_trap.run.results import (
    RESULT_SCHEMA_VERSION,
    CarrierScales,
    EntanglingScales,
    IntrinsicBudget,
    Result,
    ScatteringScales,
    aggregate,
    bits_from_decimal,
    bitstring_key,
    decimal_key,
)
from tests.fixtures import make_result


def _unitary(u: np.ndarray) -> bool:
    return bool(np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-12))


def test_gpi_is_cos_x_plus_sin_y() -> None:
    for phi in (0.0, 0.3, 2.0, -1.1):
        assert np.allclose(native.gpi(phi), math.cos(phi) * native.PAULI_X + math.sin(phi) * native.PAULI_Y)


PHASE_SWEEP_TURNS = tuple(float(x) for x in np.linspace(0.0, 1.0, 9)[:-1])
"""Eight phases over a full turn."""
OPERATOR_ATOL = 1e-12
"""The operator-identity tolerance (the measured deviations are 0 to 7.1e-16)."""


@pytest.mark.parametrize("turns", PHASE_SWEEP_TURNS)
def test_the_six_native_gate_identities(turns: float) -> None:
    """GPi(phi)^2 = I, GPi2(phi)^2 = -i GPi(phi), GPi2(phi + 0.5) = GPi2(phi)^dag, GPi2(phi) = R(pi/2, 2 pi phi),
    GPi(phi) = i R(pi, 2 pi phi) and VZ(t) G(p) VZ(t)^dag = G(p + t) to 1e-12 over a phase sweep (phi in turns)."""
    p = native.rad_from_turns(turns)
    eye = np.eye(2, dtype=complex)
    assert np.allclose(native.gpi(p) @ native.gpi(p), eye, atol=OPERATOR_ATOL)
    assert np.allclose(native.gpi2(p) @ native.gpi2(p), -1j * native.gpi(p), atol=OPERATOR_ATOL)
    assert np.allclose(
        native.gpi2(native.rad_from_turns(turns + 0.5)), native.gpi2(p).conj().T, atol=OPERATOR_ATOL
    )
    assert np.allclose(native.gpi2(p), native.r_phi(math.pi / 2, p), atol=OPERATOR_ATOL)
    assert np.allclose(native.gpi(p), 1j * native.r_phi(math.pi, p), atol=OPERATOR_ATOL)
    for t in (0.1, -0.7, 2.3):
        for gate in (native.gpi, native.gpi2):
            conjugated = native.rz(t) @ gate(p) @ native.rz(t).conj().T
            assert np.allclose(conjugated, gate(p + t), atol=OPERATOR_ATOL), (gate.__name__, turns, t)
    # the identities are sharp: GPi2(phi)^2 is NOT +i GPi(phi), and GPi2(phi + 0.5) is not GPi2(phi) itself
    assert not np.allclose(native.gpi2(p) @ native.gpi2(p), 1j * native.gpi(p), atol=1e-3)
    assert not np.allclose(native.gpi2(native.rad_from_turns(turns + 0.5)), native.gpi2(p), atol=1e-3)


def test_gpi2_half_turn_is_hilbert_schmidt_orthogonal_to_rx_plus_pi_over_two() -> None:
    """GPi2(0.5) = RX(-pi/2) with a Hilbert-Schmidt overlap below 1e-12 against RX(+pi/2), GPi(0) and GPi(0.25) are
    orthogonal, and the self-overlap is 2."""
    half = native.gpi2(native.rad_from_turns(0.5))
    assert np.allclose(half, native.r_phi(-math.pi / 2, 0.0), atol=OPERATOR_ATOL)
    overlap = complex(np.trace(half.conj().T @ native.r_phi(math.pi / 2, 0.0)))
    assert abs(overlap) < OPERATOR_ATOL, overlap
    assert (
        abs(complex(np.trace(native.gpi(0.0).conj().T @ native.gpi(native.rad_from_turns(0.25)))))
        < OPERATOR_ATOL
    )
    assert abs(complex(np.trace(half.conj().T @ half))) == pytest.approx(2.0, rel=1e-12)


def test_ms_fully_entangling_matrix_and_xx_conversion() -> None:
    phi0, phi1 = 0.4, -1.3
    m = native.ms(phi0, phi1, native.rad_from_turns(0.25))
    s = 1.0 / math.sqrt(2.0)
    expected = s * np.array(
        [
            [1, 0, 0, -1j * np.exp(-1j * (phi0 + phi1))],
            [0, 1, -1j * np.exp(-1j * (phi0 - phi1)), 0],
            [0, -1j * np.exp(1j * (phi0 - phi1)), 1, 0],
            [-1j * np.exp(1j * (phi0 + phi1)), 0, 0, 1],
        ]
    )
    assert np.allclose(m, expected)
    # XX(chi) = MS(0, 0, theta) with chi = pi theta_turns, i.e. chi = theta_rad/2
    for turns in (0.05, 0.1, 0.25):
        assert np.allclose(native.ms(0.0, 0.0, native.rad_from_turns(turns)), native.xx(math.pi * turns))
    assert _unitary(m)


def test_zz_and_rz_diagonals_in_turns() -> None:
    theta = 0.13
    assert np.allclose(
        native.zz(native.rad_from_turns(theta)),
        np.diag(np.exp(1j * math.pi * theta * np.array([-1, 1, 1, -1]))),
    )
    assert np.allclose(
        native.rz(native.rad_from_turns(theta)), np.diag(np.exp(1j * math.pi * theta * np.array([-1, 1])))
    )


def test_virtual_z_rule_in_time_order() -> None:
    """RZ(theta), GPi(0.5), GPi2(0) equals GPi(0.5 - theta), GPi2(-theta) followed by RZ(theta) (Section 7.6); the
    +theta reading does not."""
    theta = native.rad_from_turns(0.17)
    half = native.rad_from_turns(0.5)
    left = native.gpi2(0.0) @ native.gpi(half) @ native.rz(theta)  # time order right to left
    right = (
        native.rz(theta)
        @ native.gpi2(native.virtual_z_frame_shift(0.0, theta))
        @ native.gpi(native.virtual_z_frame_shift(half, theta))
    )
    assert np.allclose(left, right)
    wrong = native.rz(theta) @ native.gpi2(theta) @ native.gpi(half + theta)
    assert not native.equal_up_to_global_phase(left, wrong)


def test_ionq_circuit_json_round_trip() -> None:
    obj = {
        "gateset": "native",
        "qubits": 2,
        "circuit": [
            {"gate": "ms", "targets": [0, 1], "phases": [0, 0.25], "angle": 0.25},
            {"gate": "gpi2", "target": 0, "phase": 0.75},
            {"gate": "zz", "targets": [0, 1], "angle": 0.1},
        ],
    }
    circ = load_ionq_json(obj)
    ms_op, gpi2_op, zz_op = circ.ops
    assert ms_op.name == "ms" and ms_op.params == pytest.approx((0.0, math.pi / 2, math.pi / 2))
    assert gpi2_op.name == "gpi2" and gpi2_op.params == pytest.approx((1.5 * math.pi,))
    assert zz_op.params == pytest.approx((0.2 * math.pi,))
    assert circ.measure == (0, 1)
    out = dump_ionq_json(circ)
    assert out["circuit"] == obj["circuit"]
    assert out["qubits"] == 2 and out["gateset"] == "native"


def test_ionq_zz_carries_angle_only_and_qis_rotations_are_radians() -> None:
    with pytest.raises(ValueError, match="angle only"):
        load_ionq_json(
            {"qubits": 2, "circuit": [{"gate": "zz", "targets": [0, 1], "angle": 0.1, "phases": [0, 0]}]}
        )
    circ = load_ionq_json(
        {
            "input": {
                "gateset": "qis",
                "qubits": 2,
                "circuit": [
                    {"gate": "rx", "target": 1, "rotation": 1.57},
                    {"gate": "cnot", "control": 0, "target": 1},
                ],
            }
        }
    )
    assert circ.ops[0] == Operation("rx", (1,), (1.57,))
    assert circ.ops[1] == Operation("cnot", (0, 1), ())
    with pytest.raises(ValueError, match="compile first"):
        dump_ionq_json(circ)


def test_circuit_ir_validation() -> None:
    with pytest.raises(ValueError):
        Operation("ms", (0, 1), (0.0,))  # ms takes three parameters
    with pytest.raises(ValueError):
        Operation("gpi", (0, 0), (0.0,))
    with pytest.raises(ValueError):
        Circuit(1, (Operation("gpi", (1,), (0.0,)),), (0,))
    c = Circuit(2, (Operation("measure", (0,), ()), Operation("gpi2", (1,), (0.0,))), (0, 1))
    assert c.is_native


def test_result_bit_order_examples() -> None:
    # "5" and "7" on three qubits are 101 and 111 with qubit 0 least significant
    rows = np.array([[1, 0, 1], [1, 1, 1]] * 2, dtype=np.uint8)
    res = make_result(rows)
    assert res.to_ionq_v1_probabilities() == {"5": 0.5, "7": 0.5}
    assert res.to_ionq_v1_histogram() == {"5": 2, "7": 2}
    assert set(res.counts) == {"101", "111"}
    shots = np.array([bits_from_decimal(k, 3) for k in ("6", "1", "0", "7")])
    assert [bitstring_key(r) for r in shots] == ["110", "001", "000", "111"]
    assert make_result(shots).to_ionq_v1_shots() == ["6", "1", "0", "7"]


@given(
    st.lists(st.lists(st.integers(0, 1), min_size=1, max_size=6), min_size=1, max_size=20).filter(
        lambda rows: len({len(r) for r in rows}) == 1
    )
)
def test_decimal_key_round_trip(rows: list[list[int]]) -> None:
    arr = np.array(rows, dtype=np.uint8)
    n = arr.shape[1]
    for row in arr:
        back = bits_from_decimal(decimal_key(row), n)
        assert np.array_equal(back, row)
        assert int(bitstring_key(row), 2) == int(decimal_key(row))
    counts, probs = aggregate(arr)
    assert sum(counts.values()) == len(rows)
    assert sum(probs.values()) == pytest.approx(1.0)


# ---- the Result's exports -----------------------------------------------------------------------------------------------------

BITS = np.array([[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
X_ON_QUBIT_ZERO = np.array([[1, 0, 0]] * 4, dtype=np.uint8)
"""``x q[0]`` on three qubits, four shots: a Bell state cannot tell the two character orders apart, this can."""


def test_sample_of_shot_names_a_dynamical_sample_per_row() -> None:
    """``Result.sample_of_shot`` holds one index per row below ``Diagnostics.samples`` (the run's own map is tested
    through discards in test_run_noise.py); a result with no sample behind its rows carries None."""
    result = make_result(BITS)
    assert result.sample_of_shot is None and Result.from_ionq_v1_shots(["0", "3"], 2).sample_of_shot is None
    three = dataclasses.replace(result.diagnostics, samples=3)
    mapped = dataclasses.replace(result, diagnostics=three, sample_of_shot=[0, 0, 2, 2, 1])
    assert mapped.sample_of_shot is not None and mapped.sample_of_shot.dtype == np.int64
    assert mapped.sample_of_shot.tolist() == [0, 0, 2, 2, 1]
    with pytest.raises(ValueError, match="one sample index per shot"):
        dataclasses.replace(mapped, sample_of_shot=[0, 1])
    with pytest.raises(ValueError, match="dynamical samples"):
        dataclasses.replace(mapped, sample_of_shot=[0, 0, 3, 2, 1])


def test_x_on_qubit_zero_reads_1_in_the_v1_formats_and_100_in_the_v2_envelope() -> None:
    """``x q[0]`` on three qubits reads "1" in the v1 formats (qubit 0 the least-significant bit) and "100" in the v2
    envelope (wire order, q[0] leftmost), in ``output_all`` and the default register ``c``."""
    result = make_result(X_ON_QUBIT_ZERO)
    assert result.counts == {"001": 4}  # this package's key: qubit 0 rightmost
    assert result.to_ionq_v1_probabilities() == {"1": 1.0}
    assert result.to_ionq_v1_histogram() == {"1": 4}
    assert result.to_ionq_v1_shots() == ["1"] * 4
    assert result.to_ionq_v2_probabilities() == {
        "probabilities": {"registers": {"output_all": {"100": 1.0}, "c": {"100": 1.0}}}
    }
    assert result.to_ionq_v2_histogram() == {
        "histogram": {"registers": {"output_all": {"100": 4}, "c": {"100": 4}}}
    }
    shots = result.to_ionq_v2_shots()["shots"]
    assert len(shots) == 4 and shots[0] == {"registers": {"output_all": [1, 0, 0], "c": [1, 0, 0]}}


def test_the_v2_exporters_follow_the_circuit_registers_and_the_measured_columns() -> None:
    """A v2 register lists its qubits in bit order, first bit first, and ``output_all`` every measured qubit ascending; an
    unmeasured register qubit and a repeated column are refused."""
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
        dataclasses.replace(result, bit_order="little")


def test_from_ionq_v1_shots_closes_the_round_trip() -> None:
    shots = ["6", "1", "0", "7"]
    result = Result.from_ionq_v1_shots(shots, 3)
    assert result.to_ionq_v1_shots() == shots and set(result.counts) == {"110", "001", "000", "111"}
    assert result.shots == 4 and result.spam == {} and result.diagnostics.samples == 0
    assert result.diagnostics.calibration.uncalibrated() == ("field",)
    assert any("no simulation behind these shots" in a for a in result.diagnostics.approximations)
    assert result.error_bars["110"] == pytest.approx(math.sqrt(0.25 * 0.75 / 4))
    assert Result.from_ionq_v1_shots([6, 1], 3).counts == {"110": 1, "001": 1}


BUDGET = IntrinsicBudget(
    entangling=(
        EntanglingScales(
            gate_id="ms[2]",
            residual_displacement=1e-6,
            debye_waller=4e-6,
            carrier_scale=1.4e-3,
            carrier_steps=1.3e-4,
            bessel_saturation=2e-5,
            frozen_angle=0.0,
            sideband_lamb_dicke_deficit=2.1e-2,
            frozen_angle_rad=3e-5,
        ),
    ),
    carriers=(CarrierScales("gpi2[0]", 3e-4, 1.8e-5), CarrierScales("gpi2[0]", 3e-4, 1.8e-5)),
    scattering=(ScatteringScales("ms[2]/ion0", 0, 5e-6, 5e-6, 2e-8, 1e-11),),
)
"""Two carrier records under one gate id (a pulse train on one ion shares it): both are kept and summed."""


def test_the_envelope_round_trips() -> None:
    result = dataclasses.replace(
        make_result(BITS),
        spam={"q0": (1e-3, 2e-3), "q1": (3e-4, 4e-4)},
        heralds=np.array([0, 1, 2, 4, 3], dtype=np.uint8),
        machine_hash="m" * 64,
        created_at="2026-09-11T12:00:00+00:00",
        duration_s=8.5,
        diagnostics=dataclasses.replace(make_result(BITS).diagnostics, intrinsic_budget=BUDGET),
    )
    d = result.to_dict()
    assert json.loads(json.dumps(d)) == d, "plain JSON values only"
    assert d["schema_version"] == RESULT_SCHEMA_VERSION == 3
    assert d["diagnostics"]["intrinsic_budget"]["total"] == BUDGET.total
    assert d["device_hash"] == "fixture" and d["machine_hash"] == "m" * 64
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
    assert back.diagnostics.intrinsic_budget == BUDGET
    assert (
        back.shots == result.shots and int(back.heralds.sum()) == 0
    )  # rebuilt from the counts: no per-shot flags
    full = result.to_dict(per_shot=True)
    again = Result.from_dict(full)
    assert np.array_equal(again.bitstrings, result.bitstrings) and np.array_equal(
        again.heralds, result.heralds
    )
    assert full["per_shot"]["sample_of_shot"] is None and again.sample_of_shot is None
    with pytest.raises(ValueError, match="schema version 3"):
        Result.from_dict({**d, "schema_version": 2})


def test_dump_job_writes_the_v0_4_body_and_load_job_reads_v0_3_and_v0_4() -> None:
    """``dump_job`` writes IonQ's v0.4 ``CircuitJobCreationPayload`` with only the spec's keys, ``load_job`` reads it and
    v0.3's ``target`` back, and malformed noise, settings and job types are refused."""
    circuit = Circuit(2).gpi2(0, native.rad_from_turns(0.25)).ms(0, 1, 0.0, 0.0, native.rad_from_turns(0.25))
    body = dump_job(
        circuit,
        backend="simulator",
        shots=2000,
        noise={"model": "aria-1", "seed": 7},
        settings={"error_mitigation": {"debiasing": False}},
        name="bell",
        metadata={"run": "1"},
    )
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
    assert load_job({"target": "qpu.aria-1", "input": body["input"]}).backend == "qpu.aria-1"
    assert load_job(json.dumps(body)).circuit == circuit
    assert load_ionq_json(dump_ionq_json(circuit)) == circuit and load_ionq_json(body) == circuit
    with pytest.raises(ValueError, match=r"no key \['target'\]"):
        dump_job(circuit, backend="simulator", noise={"target": 1})
    with pytest.raises(ValueError, match="'model'"):
        dump_job(circuit, backend="simulator", noise={"seed": 1})
    with pytest.raises(ValueError, match="settings.error_mitigation has no key"):
        dump_job(circuit, backend="simulator", settings={"error_mitigation": {"debias": True}})
    with pytest.raises(ValueError, match="ionq.qasm3.v1"):
        load_job({"type": "ionq.qasm3.v1", "input": {"data": "OPENQASM 3.0;"}})
    with pytest.raises(ValueError, match="compile first"):
        dump_job(Circuit(1).h(0), backend="simulator")
