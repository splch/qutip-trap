"""The circuit builder's model (DESIGN.md Sections 5 and 11, R16): the palette is the compiler's vocabulary, the layout
packs left without crossing a connector, every edit is a pure function, and the OpenQASM 2 text round-trips, with the
worked example a fixed point; the session's edit path keeps an undo history and the parse path reads both formats."""

from __future__ import annotations

import json
import math

import pytest

from qutip_trap_app.core import Circuit, Operation
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.viewmodel import builder
from qutip_trap_app.viewmodel.builder import GATES, PALETTE
from qutip_trap_app.views.state import BELL_QASM, Session, Store


def _bell() -> Circuit:
    return builder.parse_circuit_text(BELL_QASM, "openqasm2")


def test_every_palette_gate_is_a_compiler_gate_that_round_trips_through_the_text() -> None:
    n = 2
    circuit = builder.empty_circuit(n)
    for spec in PALETTE:
        qubits = builder.qubits_for_new(spec, 0, n)
        circuit = builder.append_gate(circuit, spec, qubits)
    assert len(circuit.ops) == len(PALETTE)
    text = builder.to_openqasm2(circuit)
    back = builder.parse_circuit_text(text, "openqasm2")
    assert back.n_qubits == n and back.measure == (0, 1)
    assert [op.name for op in back.ops] == [spec.name for spec in PALETTE]
    for a, b in zip(circuit.ops, back.ops):
        assert a.qubits == b.qubits
        assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(a.params, b.params))
    families = builder.families()
    assert families == ("fixed", "rotation", "pair", "native")
    assert sum(len(builder.palette_group(f)) for f in families) == len(PALETTE)
    assert GATES["cx"] is GATES["cnot"] and GATES["cnot"].qasm_name == "cx"


def test_the_worked_example_is_a_fixed_point_of_the_round_trip() -> None:
    assert builder.to_openqasm2(_bell()) == BELL_QASM, "opening the builder must never rewrite the Bell text"


def test_layout_packs_left_and_a_connector_blocks_the_wires_it_crosses() -> None:
    lay = builder.layout(_bell())
    assert [(p.op.name, p.column) for p in lay.placed] == [("h", 0), ("cnot", 1)] and lay.n_columns == 2
    c = Circuit(
        3,
        (
            Operation("h", (0,), ()),
            Operation("x", (1,), ()),
            Operation("cz", (0, 2), ()),  # spans wire 1: nothing on wire 1 may share its column
            Operation("x", (1,), ()),
            Operation("h", (2,), ()),
        ),
        (0, 1, 2),
    )
    lay = builder.layout(c)
    cols = [p.column for p in lay.placed]
    assert cols == [0, 0, 1, 2, 2] and lay.n_columns == 3
    assert lay.placed[2].spans(1) and not lay.placed[2].spans(3)
    assert [p.index for p in lay.on_wire(1)] == [1, 3]
    assert builder.layout(builder.empty_circuit(2)).n_columns == 0


def test_drop_geometry() -> None:
    c = _bell()
    lay = builder.layout(c)
    assert builder.insert_index(lay, 0, 0) == 0, "before the H"
    assert builder.insert_index(lay, 0, 1) == 1, "before the CNOT"
    assert builder.insert_index(lay, 1, 0) == 1, "wire 1 is first touched by the CNOT"
    assert builder.insert_index(lay, 1, 5) == 2, "past everything: the end"
    assert builder.qubits_for_new(GATES["h"], 1, 2) == (1,)
    assert builder.qubits_for_new(GATES["cnot"], 0, 3) == (0, 1)
    assert builder.qubits_for_new(GATES["cnot"], 2, 3) == (2, 1), "the last wire takes the one above"
    with pytest.raises(ValueError, match="two qubits"):
        builder.qubits_for_new(GATES["cnot"], 0, 1)
    cnot = Operation("cnot", (0, 1), ())
    assert builder.qubits_for_move(cnot, 1, 3) == (1, 2)
    assert builder.qubits_for_move(cnot, 2, 3) == (1, 2), "shifted back inside the register"
    assert builder.qubits_for_move(Operation("cnot", (2, 0), ()), 1, 3) == (2, 0), "the shape is kept"
    assert builder.qubits_for_move(Operation("x", (2,), ()), 0, 3) == (0,)


def test_edits_are_pure_and_keep_the_circuit_valid() -> None:
    c = _bell()
    c2 = builder.insert_gate(c, 1, GATES["rx"], (1,))
    assert [op.name for op in c2.ops] == ["h", "rx", "cnot"] and c2.ops[1].params == (math.pi / 2,)
    assert [op.name for op in c.ops] == ["h", "cnot"], "the input is untouched"
    c3 = builder.remove_gate(c2, 0)
    assert [op.name for op in c3.ops] == ["rx", "cnot"]
    c4 = builder.set_param(c3, 0, 0, math.pi)
    assert c4.ops[0].params == (math.pi,)
    c5 = builder.set_qubits(c4, 1, (1, 1))
    assert c5.ops[1].qubits == (1, 0), "a coinciding choice swaps the slots instead of making an invalid gate"
    c6 = builder.set_qubits(c5, 1, (0, 0))
    assert c6.ops[1].qubits == (0, 1)
    # moving past the nearest gate on a shared wire, and not past an unrelated one
    c7 = Circuit(
        3,
        (Operation("h", (0,), ()), Operation("x", (2,), ()), Operation("cnot", (0, 1), ())),
        (0, 1, 2),
    )
    assert builder.neighbours(c7, 2) == (0, None), "the X on wire 2 is not in the CNOT's way"
    moved, k = builder.move_gate(c7, 2, -1)
    assert k == 0 and [op.name for op in moved.ops] == ["cnot", "h", "x"]
    same, k2 = builder.move_gate(c7, 1, +1)
    assert same is c7 and k2 == 1, "nothing to pass: unchanged"
    later, k3 = builder.move_gate(moved, 0, +1)
    assert k3 == 1 and [op.name for op in later.ops] == ["h", "cnot", "x"]
    # a drag move: index before removal, then the qubits of the drop
    dragged, k4 = builder.move_to(c7, 0, 3, (1,))
    assert k4 == 2 and [op.name for op in dragged.ops] == ["x", "cnot", "h"] and dragged.ops[2].qubits == (1,)
    dragged2, k5 = builder.move_to(c7, 2, 0, (2, 1))
    assert k5 == 0 and dragged2.ops[0].qubits == (2, 1)


def test_qubits_are_added_and_removed_within_bounds() -> None:
    c = _bell()
    c3 = builder.add_qubit(c)
    assert c3.n_qubits == 3 and c3.measure == (0, 1, 2) and len(c3.ops) == 2
    c4 = builder.add_qubit(c3)
    with pytest.raises(ValueError, match="at most"):
        builder.add_qubit(c4)
    back = builder.remove_qubit(builder.remove_qubit(c4))
    assert back.n_qubits == 2 and len(back.ops) == 2
    one = builder.remove_qubit(back)
    assert one.n_qubits == 1 and [op.name for op in one.ops] == ["h"], "the CNOT touched the removed wire"
    with pytest.raises(ValueError, match="at least"):
        builder.remove_qubit(one)
    assert builder.clear(c).ops == () and builder.clear(c).n_qubits == 2
    assert builder.summary(builder.layout(builder.clear(c))) == "no gates yet"
    assert builder.summary(builder.layout(c)) == "2 gates in 2 columns"
    assert builder.summary(builder.layout(one)) == "1 gate in 1 column"


def test_angles_format_parse_and_serialise() -> None:
    assert builder.format_angle(0.0) == "0" and builder.qasm_angle(0.0) == "0"
    assert builder.format_angle(math.pi) == "π" and builder.qasm_angle(math.pi) == "pi"
    assert builder.format_angle(-math.pi) == "-π" and builder.qasm_angle(-math.pi) == "-pi"
    assert builder.format_angle(math.pi / 2) == "π/2" and builder.qasm_angle(math.pi / 2) == "pi/2"
    assert (
        builder.format_angle(-3 * math.pi / 4) == "-3π/4"
        and builder.qasm_angle(-3 * math.pi / 4) == "-3*pi/4"
    )
    assert builder.format_angle(2 * math.pi) == "2π" and builder.qasm_angle(2 * math.pi) == "2*pi"
    assert builder.format_angle(0.35) == "0.35" and builder.qasm_angle(0.35) == "0.35"
    odd = 0.123456789012345
    assert builder.qasm_angle(odd) == repr(odd) and builder.format_angle(odd) == "0.123"
    assert builder.pi_fraction(math.pi / 16) == (1, 16) and builder.pi_fraction(math.pi / 32) is None
    assert builder.pi_fraction(float("nan")) is None
    for text, value in (("pi/2", math.pi / 2), ("π/2", math.pi / 2), ("-3*pi/4", -3 * math.pi / 4)):
        assert math.isclose(builder.parse_angle(text), value)
    assert math.isclose(builder.parse_angle(" 0.35 "), 0.35)
    assert math.isclose(builder.parse_angle("sqrt(2)"), math.sqrt(2))
    assert math.isclose(builder.parse_angle("2^-1"), 0.5)
    for bad in ("", "pi/", "abc", "1) q[0]; x q[0]; (", "1/0", "1e400"):
        with pytest.raises(ValueError):
            builder.parse_angle(bad)
    # what the text carries is what the parser reads back, for a fraction and for an arbitrary float
    c = builder.append_gate(builder.empty_circuit(1), GATES["rz"], (0,), (odd,))
    c = builder.append_gate(c, GATES["u3"], (0,), (math.pi / 2, 0.0, -3 * math.pi / 4))
    text = builder.to_openqasm2(c)
    assert f"rz({odd!r}) q[0];" in text and "u3(pi/2, 0, -3*pi/4) q[0];" in text
    back = builder.parse_circuit_text(text)
    assert back.ops[0].params == (odd,)
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(back.ops[1].params, c.ops[1].params))


def test_text_in_both_formats_and_the_other_operations() -> None:
    ionq = json.dumps(
        {
            "input": {
                "gateset": "native",
                "qubits": 2,
                "circuit": [
                    {"gate": "gpi2", "target": 0, "phase": 0.25},
                    {"gate": "ms", "targets": [0, 1], "phases": [0, 0], "angle": 0.25},
                ],
            }
        }
    )
    assert builder.detect_format(ionq) == "ionq_json" and builder.detect_format(BELL_QASM) == "openqasm2"
    c = builder.parse_circuit_text(ionq)
    assert [op.name for op in c.ops] == ["gpi2", "ms"] and math.isclose(c.ops[1].params[2], math.pi / 2)
    assert builder.describe(c.ops[0]) == "GPi2(π/2) on q0"
    assert builder.describe(c.ops[1]) == "MS(0, 0, π/2) on q0 and q1"
    assert builder.describe(Operation("cnot", (1, 0), ())) == "CNOT: control q1, target q0"
    assert builder.describe(Operation("h", (0,), ())) == "H on q0"
    assert builder.qubits_text((0, 1, 2)) == "q0, q1 and q2"
    # a mid-circuit measure and a reset from imported code are drawn as plain tiles and survive the round trip
    text = "qreg q[2]; creg c[2]; h q[0]; measure q[0] -> c[0]; reset q[0]; x q[0]; cx q[0],q[1]; measure q -> c;"
    imported = builder.parse_circuit_text(text)
    names = [op.name for op in imported.ops]
    assert names == ["h", "measure", "reset", "x", "cnot"]
    other = builder.spec_of(imported.ops[1])
    assert other.family == "other" and other.label == "M" and other.arity == 1
    again = builder.parse_circuit_text(builder.to_openqasm2(imported))
    assert [op.name for op in again.ops] == names
    with pytest.raises(ValueError, match="recool"):
        builder.to_openqasm2(Circuit(1, (Operation("recool", (0,), ()),), (0,)))


def test_the_session_edits_through_one_path_and_undoes() -> None:
    session = Session(Store(), ProvenanceIndex.load())
    store = session.store
    assert store.circuit_undo == ()
    edited = builder.to_openqasm2(builder.append_gate(_bell(), GATES["x"], (1,)))
    store.active_preset = "bell_state"
    store.error = "stale"
    session.edit_circuit(edited)
    assert store.circuit_text == edited and store.circuit_format == "openqasm2"
    assert store.active_preset is None and store.error == ""
    assert store.circuit_undo == ((BELL_QASM, "openqasm2"),)
    assert store.prediction_pending(), "an edited circuit is a new run: predict again"
    ionq = json.dumps({"qubits": 1, "circuit": [{"gate": "x", "target": 0}]})
    session.edit_circuit(ionq, "ionq_json")
    assert store.circuit_format == "ionq_json" and len(session.parse_circuit().ops) == 1
    assert session.undo_circuit()
    assert store.circuit_text == edited and store.circuit_format == "openqasm2"
    assert session.undo_circuit() and store.circuit_text == BELL_QASM
    assert not session.undo_circuit(), "nothing left to undo"
    # loading the worked example or a preset is undoable too, and the history is bounded
    session.edit_circuit(edited)
    session.load_bell_example()
    assert store.circuit_text == BELL_QASM and session.undo_circuit() and store.circuit_text == edited
    for k in range(60):
        session.edit_circuit(edited + f"// {k}\n")
    assert len(store.circuit_undo) == builder_undo_depth()


def builder_undo_depth() -> int:
    from qutip_trap_app.views.state import UNDO_DEPTH

    return UNDO_DEPTH
