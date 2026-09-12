"""Native gates exactly (PLAN.md Section 7.6) and the IonQ formats (Sections 8.6, 9.13)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qutip_trap.api import Circuit, Operation, dump_ionq_json, load_ionq_json
from qutip_trap.control import native
from qutip_trap.run.results import aggregate, bits_from_decimal, bitstring_key, decimal_key
from tests.fixtures import make_result


def _unitary(u: np.ndarray) -> bool:
    return bool(np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-12))


def test_gpi2_special_cases_of_section_7_6() -> None:
    assert np.allclose(native.gpi2(0.0), native.r_phi(math.pi / 2, 0.0))
    assert np.allclose(native.gpi2(native.rad_from_turns(0.25)), native.r_phi(math.pi / 2, math.pi / 2))
    # phi = 0.5 turns gives R_x(-pi/2); the documentation's prose (+pi/2) contradicts its own matrix [corrected]
    assert np.allclose(native.gpi2(native.rad_from_turns(0.5)), native.r_phi(-math.pi / 2, 0.0))
    assert not np.allclose(native.gpi2(native.rad_from_turns(0.5)), native.r_phi(math.pi / 2, 0.0))


def test_gpi_is_cos_x_plus_sin_y() -> None:
    for phi in (0.0, 0.3, 2.0, -1.1):
        assert np.allclose(native.gpi(phi), math.cos(phi) * native.PAULI_X + math.sin(phi) * native.PAULI_Y)


PHASE_SWEEP_TURNS = tuple(float(x) for x in np.linspace(0.0, 1.0, 9)[:-1])
"""Eight phases over a full turn: the sweep the Section 9.10 identity row is checked over."""
OPERATOR_ATOL = 1e-12
"""Section 9's operator-identity tolerance (the measured deviations are 0 to 7.1e-16)."""


@pytest.mark.parametrize("turns", PHASE_SWEEP_TURNS)
def test_the_six_native_gate_identities_of_section_9_10(turns: float) -> None:
    """Section 9.10 row "Native-gate identities" in full, over a phase sweep at the 1e-12 operator tolerance:
    GPi(phi)^2 = I; GPi2(phi)^2 = -i GPi(phi); GPi2(phi + 0.5) = GPi2(phi)^dag; GPi2(phi) = R(pi/2, 2 pi phi);
    GPi(phi) = i R(pi, 2 pi phi); VZ(t) G(p) VZ(t)^dag = G(p + t) for both natives (phi in TURNS, p in radians)."""
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
    """Section 9.10 row "Native-gate axis mapping": GPi2(0.5) = RX(-pi/2), and its Hilbert-Schmidt overlap with RX(+pi/2)
    is EXACTLY 0 (the two half-turn rotations about opposite senses of x differ by Z, which is traceless against I)."""
    half = native.gpi2(native.rad_from_turns(0.5))
    assert np.allclose(half, native.r_phi(-math.pi / 2, 0.0), atol=OPERATOR_ATOL)
    overlap = complex(np.trace(half.conj().T @ native.r_phi(math.pi / 2, 0.0)))
    assert abs(overlap) < OPERATOR_ATOL, overlap
    # the same statement for GPi: GPi(0) and GPi(0.25) are orthogonal (X against Y)
    assert (
        abs(complex(np.trace(native.gpi(0.0).conj().T @ native.gpi(native.rad_from_turns(0.25)))))
        < OPERATOR_ATOL
    )
    # a negative control: the overlap of GPi2(0.5) with itself is the full 2
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
    """Section 7.6: RZ(theta), GPi(0.5), GPi2(0) equals GPi(0.5 - theta), GPi2(-theta) followed by RZ(theta)."""
    theta = native.rad_from_turns(0.17)
    half = native.rad_from_turns(0.5)
    left = native.gpi2(0.0) @ native.gpi(half) @ native.rz(theta)  # time order right to left
    right = (
        native.rz(theta)
        @ native.gpi2(native.virtual_z_frame_shift(0.0, theta))
        @ native.gpi(native.virtual_z_frame_shift(half, theta))
    )
    assert np.allclose(left, right)
    # the +theta rule of the documentation, read in time order, is wrong
    wrong = native.rz(theta) @ native.gpi2(theta) @ native.gpi(half + theta)
    assert not native.equal_up_to_global_phase(left, wrong)


def test_ionq_circuit_json_round_trip_of_section_9_13() -> None:
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


def test_result_bit_order_examples_of_section_9_13() -> None:
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
