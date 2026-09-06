"""The OpenQASM 2 importer subset (PLAN.md Sections 1.4, 7.2, 7.6; milestone M6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import (
    Circuit,
    Operation,
    circuit_unitary,
    compile_with_report,
    ideal_probabilities,
    load_openqasm2,
)
from qutip_trap.io.openqasm import OpenQASMError, evaluate, tokenize

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
    rep = compile_with_report(circ)
    assert rep.circuit_residual is not None and rep.circuit_residual < 1e-9


def test_custom_gate_definitions_are_inlined_including_the_sdk_style_native_declarations() -> None:
    """Section 7.6: the client SDKs' OpenQASM 2 export declares gpi, gpi2, ms and zz as custom gates built from u, rz, rxx and rzz;
    the importer expands them through their own bodies, and the result compiles to the same unitary as the native gates."""
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
    from qutip_trap.control.native import gpi, gpi2, ms, zz

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
    # the SDK-style bodies really are the native matrices, one gate at a time
    for stmt, mat in (
        ("gpi(0.7) q[0];", gpi(0.7)),
        ("gpi2(-0.4) q[0];", gpi2(-0.4)),
    ):
        one = load_openqasm2(
            "OPENQASM 2.0; gate gpi(phi) a { u(pi, phi, pi - phi) a; } "
            "gate gpi2(phi) a { u(pi/2, phi - pi/2, pi/2 - phi) a; } qreg q[1]; " + stmt
        )
        u = circuit_unitary(one)
        idx = np.unravel_index(int(np.argmax(np.abs(mat))), mat.shape)
        r = u[idx] / mat[idx]
        assert abs(abs(r) - 1.0) < 1e-9 and np.allclose(u, r * mat, atol=1e-9)
    two = load_openqasm2(
        "OPENQASM 2.0; gate ms(p0, p1, t) a, b { rz(-p0) a; rz(-p1) b; rxx(t) a, b; rz(p0) a; rz(p1) b; } "
        "gate zz(t) a, b { rzz(t) a, b; } qreg q[2]; ms(0.2, -0.4, 0.9) q[0], q[1]; zz(0.5) q[0], q[1];"
    )
    want = circuit_unitary(
        Circuit(2, (Operation("ms", (0, 1), (0.2, -0.4, 0.9)), Operation("zz", (0, 1), (0.5,))), (0, 1))
    )
    u2 = circuit_unitary(two)
    idx = np.unravel_index(int(np.argmax(np.abs(want))), want.shape)
    r = u2[idx] / want[idx]
    assert abs(abs(r) - 1.0) < 1e-9 and np.allclose(u2, r * want, atol=1e-9)
    _ = (ms, zz)


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
