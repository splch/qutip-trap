"""OpenQASM 2 both ways (PLAN.md Section 1.4): the importer subset, the registers and the exporter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.compiler import (
    Circuit,
    Operation,
    circuit_unitary,
    compile_report,
    ideal_probabilities,
)
from qutip_trap.control.native import gpi, gpi2
from qutip_trap.io.openqasm import (
    NATIVE_DECLARATIONS,
    OpenQASMError,
    dump_openqasm2,
    evaluate,
    load_openqasm2,
    tokenize,
)

BELL = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


def test_bell_circuit_imports_with_terminal_measurements() -> None:
    circ = load_openqasm2(BELL)
    assert circ.n_qubits == 2 and circ.measure == (0, 1)
    assert circ.ops == (Operation("h", (0,), ()), Operation("cnot", (0, 1), ()))
    assert ideal_probabilities(circ) == pytest.approx({"00": 0.5, "11": 0.5})


def test_expressions_registers_and_broadcasting() -> None:
    text = """
    OPENQASM 2.0;
    qreg a[2];
    qreg b[1];
    creg c[3];
    rx(pi/2) a;              // broadcast over the register
    ry(-pi/4 + 2*pi/8) b[0]; // zero angle
    u3(pi, 0, pi) b[0];      // an X up to a phase
    cp(pi^2/pi/2) a[0], b[0];
    u1(sin(pi/2)*0.25) a[1];
    U(0.3, 0.1, -0.2) a[1];
    cu1(0.5) a[1], a[0];
    swap a[0], b[0];
    rxx(0.3) a[0], a[1];
    rzz(0.4) a[1], b[0];
    measure a[1] -> c[1];
    measure b -> c[2];
    """
    circ = load_openqasm2(text)
    assert circ.n_qubits == 3 and circ.measure == (1, 2)
    names = [op.name for op in circ.ops]
    assert names[:2] == ["rx", "rx"] and circ.ops[0].qubits == (0,) and circ.ops[1].qubits == (1,)
    assert circ.ops[2] == Operation("ry", (2,), (pytest.approx(0.0),))  # type: ignore[arg-type]
    assert circ.ops[3].name == "u3" and circ.ops[3].qubits == (2,)
    assert circ.ops[4] == Operation("cp", (0, 2), (pytest.approx(math.pi / 2.0),))  # type: ignore[arg-type]
    assert circ.ops[5] == Operation("rz", (1,), (pytest.approx(0.25),))  # type: ignore[arg-type]
    assert circ.ops[7] == Operation("cp", (1, 0), (0.5,))
    assert circ.ops[8].name == "swap" and circ.ops[9].name == "rxx" and circ.ops[10].name == "rzz"
    rep = compile_report(circ)
    assert rep.circuit_residual is not None and rep.circuit_residual < 1e-9


def test_custom_gate_definitions_are_inlined_including_the_sdk_style_native_declarations() -> None:
    """Custom gate definitions, the SDK-style gpi, gpi2, ms and zz declarations among them, are inlined to the native
    gates' unitary up to a global phase (1e-9)."""
    text = """
    OPENQASM 2.0;
    include "qelib1.inc";
    gate gpi(phi) a { u(pi, phi, pi - phi) a; }
    gate gpi2(phi) a { u(pi/2, phi - pi/2, pi/2 - phi) a; }
    gate ms(phi0, phi1, theta) a, b { rz(-phi0) a; rz(-phi1) b; rxx(theta) a, b; rz(phi0) a; rz(phi1) b; }
    gate zz(theta) a, b { rzz(theta) a, b; }
    gate mybell q0, q1 { h q0; cx q0, q1; }
    qreg q[2];
    creg c[2];
    gpi2(0.3) q[0];
    gpi(1.1) q[1];
    ms(0.2, -0.4, 0.9) q[0], q[1];
    zz(0.5) q[1], q[0];
    mybell q[0], q[1];
    measure q -> c;
    """
    circ = load_openqasm2(text)
    got = circuit_unitary(circ)
    ref = Circuit(
        2,
        (
            Operation("gpi2", (0,), (0.3,)),
            Operation("gpi", (1,), (1.1,)),
            Operation("ms", (0, 1), (0.2, -0.4, 0.9)),
            Operation("zz", (1, 0), (0.5,)),
            Operation("h", (0,), ()),
            Operation("cnot", (0, 1), ()),
        ),
        (0, 1),
    )
    want = circuit_unitary(ref)
    idx = np.unravel_index(int(np.argmax(np.abs(want))), want.shape)
    r = got[idx] / want[idx]
    assert abs(abs(r) - 1.0) < 1e-9 and np.allclose(got, r * want, atol=1e-9)


def test_mid_circuit_measure_and_reset_stay_in_ops() -> None:
    text = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[0] -> c[0]; x q[0]; reset q[1]; measure q[1] -> c[1];"
    circ = load_openqasm2(text)
    assert [op.name for op in circ.ops] == ["h", "measure", "x", "reset"]
    assert circ.measure == (1,)


def test_refusals_and_expression_errors() -> None:
    with pytest.raises(OpenQASMError, match="if"):
        load_openqasm2("OPENQASM 2.0; qreg q[1]; creg c[1]; if (c==1) x q[0];")
    with pytest.raises(OpenQASMError, match="unknown gate"):
        load_openqasm2("OPENQASM 2.0; qreg q[1]; foo q[0];")
    with pytest.raises(OpenQASMError, match="no qreg"):
        load_openqasm2("OPENQASM 2.0;")
    with pytest.raises(OpenQASMError, match="out of range"):
        load_openqasm2("OPENQASM 2.0; qreg q[1]; x q[1];")
    with pytest.raises(OpenQASMError, match="3.0"):
        load_openqasm2("OPENQASM 3.0; qreg q[1];")
    with pytest.raises(OpenQASMError):
        evaluate(tokenize("1/0")[:-1], {})
    assert evaluate(tokenize("-pi/2 + 3*(1 - 0.5)^2")[:-1], {}) == pytest.approx(-math.pi / 2.0 + 0.75)
    assert evaluate(tokenize("cos(theta)")[:-1], {"theta": 0.0}) == 1.0
    with pytest.raises(OpenQASMError, match="unknown identifier"):
        evaluate(tokenize("lambda")[:-1], {})


# ---- registers and the exporter -----------------------------------------------------------------------------------------------


def test_registers_survive_an_openqasm_round_trip() -> None:
    text = (
        "OPENQASM 2.0; qreg q[3]; creg a[1]; creg b[2]; h q[0]; cx q[0], q[2]; "
        "measure q[2] -> a[0]; measure q[0] -> b[1]; measure q[1] -> b[0];"
    )
    c = Circuit.from_openqasm(text)
    assert c.registers == {"a": (2,), "b": (1, 0)} and c.measure == (0, 1, 2)
    assert load_openqasm2(dump_openqasm2(c)) == c
    assert Circuit.from_openqasm(BELL).registers == {"c": (0, 1)}
    # a broadcast measure writes bit k of the creg from qubit k of the qreg
    assert load_openqasm2("OPENQASM 2.0; qreg q[2]; creg r[2]; measure q -> r;").registers == {"r": (0, 1)}
    # no creg, or a creg nothing writes into: the default register over the terminal targets, which a program without a
    # measure statement leaves empty (a run then measures every ion)
    for text in ("OPENQASM 2.0; qreg q[2]; h q[0];", "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0];"):
        assert load_openqasm2(text).measure == () and load_openqasm2(text).registers == {"c": ()}
    assert load_openqasm2("OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[1] -> c[0];").registers == {
        "c": (1,)
    }
    with pytest.raises(OpenQASMError, match="out of range"):
        load_openqasm2("OPENQASM 2.0; qreg q[1]; creg c[1]; measure q[0] -> c[1];")


def test_dumps_declares_the_native_gates_and_the_bare_form_round_trips_exactly() -> None:
    """``dumps`` declares the natives in the SDK forms, the native matrices up to a global phase (1e-9) one by one and
    together, and its bare form round-trips the native operations exactly."""
    native = Circuit(2).gpi2(0, 0.3).gpi(1, 1.1).ms(0, 1, 0.2, -0.4, 0.9).zz(1, 0, 0.5)
    text = dump_openqasm2(native)
    assert text.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";\n') and NATIVE_DECLARATIONS["ms"] in text
    assert "creg c[2];" in text and text.rstrip().endswith("measure q[1] -> c[1];")
    got = circuit_unitary(load_openqasm2(text))
    want = circuit_unitary(native)
    idx = np.unravel_index(int(np.argmax(np.abs(want))), want.shape)
    r = got[idx] / want[idx]
    assert abs(abs(r) - 1.0) < 1e-9 and np.allclose(got, r * want, atol=1e-9), (
        "the declared forms are the native matrices"
    )
    bare = dump_openqasm2(native, declare_native=False)
    assert "gate " not in bare and load_openqasm2(bare) == native
    assert native.to_openqasm() == text and Circuit.from_openqasm(bare) == native
    # each declaration alone, against control.native's matrix
    for name, matrix in (("gpi", gpi(0.7)), ("gpi2", gpi2(-0.4))):
        one = load_openqasm2(
            f"OPENQASM 2.0; {NATIVE_DECLARATIONS[name]} qreg q[1]; {name}("
            + ("0.7" if name == "gpi" else "-0.4")
            + ") q[0];"
        )
        u = circuit_unitary(one)
        idx = np.unravel_index(int(np.argmax(np.abs(matrix))), matrix.shape)
        r = u[idx] / matrix[idx]
        assert abs(abs(r) - 1.0) < 1e-9 and np.allclose(u, r * matrix, atol=1e-9), name
    two = load_openqasm2(
        f"OPENQASM 2.0; {NATIVE_DECLARATIONS['ms']} {NATIVE_DECLARATIONS['zz']} qreg q[2]; "
        "ms(0.2, -0.4, 0.9) q[0], q[1]; zz(0.5) q[0], q[1];"
    )
    u2 = circuit_unitary(two)
    want2 = circuit_unitary(Circuit(2).ms(0, 1, 0.2, -0.4, 0.9).zz(0, 1, 0.5))
    assert np.allclose(
        circuit_unitary(Circuit(2).ms(0, 1, 0.2, -0.4, 0.9)),
        circuit_unitary(Circuit(2, (Operation("ms", (0, 1), (0.2, -0.4, 0.9)),))),
    )
    idx = np.unravel_index(int(np.argmax(np.abs(want2))), want2.shape)
    r = u2[idx] / want2[idx]
    assert abs(abs(r) - 1.0) < 1e-9 and np.allclose(u2, r * want2, atol=1e-9)


def test_dumps_writes_standard_gates_mid_circuit_operations_and_refuses_recool() -> None:
    c = (
        Circuit(3)
        .h(0)
        .cnot(0, 1)
        .cz(1, 2)
        .rx(2, 0.25)
        .u3(0, 0.1, 0.2, 0.3)
        .cp(0, 2, 1.5)
        .swap(1, 2)
        .measured(0, 2)
    )
    text = dump_openqasm2(c)
    assert "cx q[0], q[1];" in text and "u3(0.1, 0.2, 0.3) q[0];" in text and "measure q[1]" not in text
    assert load_openqasm2(text) == c
    mid = load_openqasm2(
        "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[0] -> c[0]; x q[0]; reset q[1]; measure q[1] -> c[1];"
    )
    assert load_openqasm2(dump_openqasm2(mid)) == mid and "reset q[1];" in dump_openqasm2(mid)
    with pytest.raises(ValueError, match="recool"):
        dump_openqasm2(Circuit(1, (Operation("recool", (0,), ()),)))
    with pytest.raises(ValueError, match="mid-circuit measure of qubit 1"):
        dump_openqasm2(
            Circuit(2, (Operation("measure", (1,), ()), Operation("x", (1,), ())), (0,), {"c": (0,)})
        )
