"""The compiler of PLAN.md Section 7.2 and its templates of Section 7.7 (Section 9.6 rows "Compiled CNOT and CP", "Native-gate
identities" (virtual-Z propagation), "IonQ JSON round trip"; Section 13 rows "Operator order in templates", "Virtual-Z
propagation")."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import (
    Circuit,
    Operation,
    circuit_unitary,
    compile_to_native,
    compile_with_report,
    dump_ionq_json,
    ideal_probabilities,
    load_ionq_json,
)
from qutip_trap.control import native
from qutip_trap.control.compiler import (
    CNOT_MATRIX,
    CompileError,
    cnot_global_phase,
    cnot_template,
    cp_matrix,
    cp_template,
    cp_template_local_defect_rad,
    cp_template_overlap,
    debnath_cp_template,
    decompose_single_qubit,
    embed,
    frame_unitary,
    gate_matrix,
    propagate_frames,
    zyz_angles,
)


def _phase(a: np.ndarray, b: np.ndarray) -> float | None:
    idx = np.unravel_index(int(np.argmax(np.abs(b))), b.shape)
    r = a[idx] / b[idx]
    return float(np.angle(r)) if np.allclose(a, r * b, atol=1e-9) and abs(abs(r) - 1.0) < 1e-9 else None


def _random_su2(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    u = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    return np.asarray(u / np.sqrt(np.linalg.det(u)))


def test_zxzxz_decomposition_is_complete_on_twenty_random_su2_targets() -> None:
    """Section 7.2 item 1: every U in SU(2) is RZ GPi2(0) RZ GPi2(0) RZ up to a global phase, to round-off (the plan's 20 targets)."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        u = _random_su2(rng)
        ops = decompose_single_qubit(u, 0)
        assert sum(op.name == "gpi2" for op in ops) <= 2 and all(op.name in ("gpi2", "rz") for op in ops)
        got = circuit_unitary(Circuit(1, tuple(ops), (0,)))
        ph = _phase(got, u)
        assert ph is not None
        worst = max(worst, float(np.max(np.abs(got - np.exp(1j * ph) * u))))
    assert worst < 1e-12
    a, b, c, _ = zyz_angles(native.r_phi(0.7, 0.3))
    assert 0.0 <= b <= math.pi
    with pytest.raises(CompileError):
        zyz_angles(np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_one_pulse_and_zero_pulse_special_cases() -> None:
    """Pure Z rotations need no pulse; pi and pi/2 equatorial rotations need one GPi or GPi2 (plus a virtual rz)."""
    for name in ("z", "s", "sdg", "t", "tdg"):
        ops = decompose_single_qubit(gate_matrix(Operation(name, (0,), ())), 0)
        assert [op.name for op in ops] == ["rz"]
    for name in ("x", "y"):
        assert [op.name for op in decompose_single_qubit(gate_matrix(Operation(name, (0,), ())), 0)] == [
            "gpi"
        ]
    h_ops = decompose_single_qubit(gate_matrix(Operation("h", (0,), ())), 0)
    assert [op.name for op in h_ops] == ["rz", "gpi2"]
    assert h_ops[1].params[0] == pytest.approx(math.pi / 2.0), (
        "H = RY(pi/2) Z: GPi2(pi/2) after a virtual Z(pi)"
    )
    assert [op.name for op in decompose_single_qubit(gate_matrix(Operation("sx", (0,), ())), 0)] == ["gpi2"]
    assert decompose_single_qubit(np.eye(2, dtype=complex), 0) == []
    assert decompose_single_qubit(gate_matrix(Operation("rz", (0,), (2e-11,))), 0) == []


@pytest.mark.parametrize("s", [1, -1])
@pytest.mark.parametrize("v", [1, -1])
def test_maslov_cnot_template_for_all_four_signs_with_its_global_phase(s: int, v: int) -> None:
    """Section 7.7: RY(v pi/2)_c, XX(s pi/4), RX(-s pi/2)_c RX(-v s pi/2)_t, RY(-v pi/2)_c equals e^{i pi v s/4} CNOT for every (s, v)."""
    ops = cnot_template(0, 1, s=s, v=v)
    assert sum(op.name == "ms" for op in ops) == 1 and sum(op.name == "gpi2" for op in ops) == 4
    got = circuit_unitary(Circuit(2, tuple(ops), (0, 1)))
    ph = _phase(got, embed(CNOT_MATRIX, (0, 1), 2))
    assert ph is not None
    assert np.exp(1j * ph) == pytest.approx(np.exp(1j * cnot_global_phase(s, v)), abs=1e-9)
    # the exported angle stays in [0, pi/2]: a negative XX is a pi on the second phase
    ms_op = next(op for op in ops if op.name == "ms")
    assert 0.0 <= ms_op.params[2] <= math.pi / 2.0 + 1e-12
    assert ms_op.params[1] == pytest.approx(0.0 if s > 0 else math.pi)


def test_trout_template_as_printed_is_not_a_cnot() -> None:
    """Section 7.7's negative control: Trout et al.'s Fig. 8 rotation signs give (Z (x) I) CNOT for s = +1, (Z (x) X) CNOT for s = -1."""
    for s in (1, -1):
        # the printed variant flips the sign of the control's post-XX RX relative to Maslov's template
        ops = (
            [Operation("gpi2", (0,), (math.pi / 2.0,))]
            + [Operation("ms", (0, 1), (0.0, 0.0 if s > 0 else math.pi, math.pi / 2.0))]
            + [
                Operation("gpi2", (0,), (0.0 if s > 0 else math.pi,)),  # RX(+s pi/2) instead of RX(-s pi/2)
                Operation("gpi2", (1,), (math.pi if s > 0 else 0.0,)),
                Operation("gpi2", (0,), (-math.pi / 2.0,)),
            ]
        )
        got = circuit_unitary(Circuit(2, tuple(ops), (0, 1)))
        assert _phase(got, embed(CNOT_MATRIX, (0, 1), 2)) is None, "a wrong-sign template must not verify"


@pytest.mark.parametrize("theta", [math.pi, math.pi / 2.0, math.pi / 4.0, -0.7, 2.5])
@pytest.mark.parametrize("entangler", ["ms", "zz"])
def test_cp_template_is_exact_and_debnaths_is_cp_only_up_to_local_phases(
    theta: float, entangler: str
) -> None:
    """Section 9.6 "Compiled CNOT and CP": the compiler's CP equals its target up to a global phase; Debnath's template as drawn
    (Section 7.7) is CP only up to RZ((sgn theta pi - theta)/2) on both qubits, overlap 0.854 at pi/2 and 0.691 at pi/4, and
    appending RZ((theta - sgn theta pi)/2) on both ions makes it exact."""
    ops = cp_template(theta, (0, 1), entangler)  # type: ignore[arg-type]
    got = circuit_unitary(Circuit(2, tuple(ops), (0, 1)))
    target = embed(cp_matrix(theta), (0, 1), 2)
    assert _phase(got, target) is not None
    drawn = debnath_cp_template(theta, (0, 1), entangler)  # type: ignore[arg-type]
    t_drawn = circuit_unitary(Circuit(2, tuple(drawn), (0, 1)))
    x = cp_template_local_defect_rad(theta)
    if abs(x) > 1e-12:
        assert _phase(t_drawn, target) is None, "as drawn it is not CP"
    # diagonal with the right conditional phase: the defect is the local RZ(x) pair
    assert np.allclose(np.abs(t_drawn - np.diag(np.diag(t_drawn))), 0.0, atol=1e-9)
    defect = embed(np.kron(native.rz(x), native.rz(x)), (0, 1), 2)
    assert _phase(t_drawn, target @ defect) is not None
    overlap = abs(np.trace(target.conj().T @ t_drawn)) / 4.0
    assert overlap == pytest.approx(cp_template_overlap(theta), abs=1e-9)
    if theta == math.pi / 2.0:
        assert overlap == pytest.approx(0.854, abs=1e-3)
    if theta == math.pi / 4.0:
        assert overlap == pytest.approx(0.691, abs=1e-3)
    fixed = drawn + [Operation("rz", (0,), (-x,)), Operation("rz", (1,), (-x,))]
    assert _phase(circuit_unitary(Circuit(2, tuple(fixed), (0, 1))), target) is not None


def test_standard_two_qubit_gates_and_u3_compile_and_verify() -> None:
    for op in (
        Operation("cz", (0, 1), ()),
        Operation("swap", (0, 1), ()),
        Operation("rxx", (0, 1), (0.9,)),
        Operation("rzz", (1, 0), (-1.3,)),
        Operation("cnot", (1, 0), ()),
    ):
        circ = Circuit(2, (op,), (0, 1))
        for ent in ("ms", "zz"):
            rep = compile_with_report(circ, entangler=ent)  # type: ignore[arg-type]
            assert rep.circuit.is_exported_native
            assert rep.circuit_residual is not None and rep.circuit_residual < 1e-9
            assert all(r < 1e-9 for r in rep.block_residuals)
    rep = compile_with_report(Circuit(1, (Operation("u3", (0,), (0.3, 1.1, -2.0)),), (0,)))
    assert rep.circuit_residual is not None and rep.circuit_residual < 1e-12 and rep.n_pulses == 2
    with pytest.raises(CompileError):
        compile_with_report(Circuit(1, (Operation("rx", (0,), (float("nan"),)),), (0,)))


def test_frame_propagation_absorbs_every_rz_and_the_measurement_discards_the_frame() -> None:
    """Sections 7.6 and 9.6: rz shifts every later gpi/gpi2/ms phase by -theta (time order), zz commutes; the compiled circuit
    equals the target up to the residual frame, whose Z rotations the computational-basis measurement cannot see."""
    ops = [
        Operation("rz", (0,), (0.1,)),
        Operation("gpi2", (0,), (0.0,)),
        Operation("rz", (1,), (0.4,)),
        Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2.0)),
        Operation("zz", (0, 1), (0.3,)),
        Operation("rz", (0,), (0.25,)),
        Operation("gpi", (1,), (1.0,)),
    ]
    out, frame = propagate_frames(ops, 2)
    assert [op.name for op in out] == ["gpi2", "ms", "zz", "gpi"]
    assert out[0].params[0] == pytest.approx(-0.1), "RZ(0.1) then GPi2(0) is GPi2(-0.1) (Section 9.6)"
    assert out[1].params[:2] == pytest.approx((-0.1, -0.4))
    assert out[3].params[0] == pytest.approx(1.0 - 0.4)
    assert frame == {0: pytest.approx(0.35), 1: pytest.approx(0.4)}
    circ = Circuit(2, tuple(ops), (0, 1))
    target = circuit_unitary(circ)
    got = frame_unitary(frame, 2) @ circuit_unitary(Circuit(2, tuple(out), (0, 1)))
    assert _phase(got, target) is not None
    # the Section 9.6 concrete sequence as matrices: RZ(0.1) then GPi2(0) on |0> equals GPi2(-0.1) followed by RZ(0.1)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    assert np.allclose(native.gpi2(0.0) @ native.rz(0.1) @ ket0, native.rz(0.1) @ native.gpi2(-0.1) @ ket0)
    # the sign is sharp: GPi2(+0.1) differs from GPi2(-0.1) by e^{-0.2i} on the lower component, so the two states are not
    # equal even up to a global phase (the magnitudes DO agree, which is why the comparison has to be the phase test)
    assert (
        _phase(
            (native.gpi2(0.0) @ native.rz(0.1) @ ket0).reshape(2, 1),
            (native.rz(0.1) @ native.gpi2(+0.1) @ ket0).reshape(2, 1),
        )
        is None
    )
    probs = ideal_probabilities(circ)
    assert probs == pytest.approx(ideal_probabilities(Circuit(2, tuple(out), (0, 1))), abs=1e-12)


def test_bell_and_ghz_compile_to_the_expected_pulse_counts_and_distributions() -> None:
    bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    rep = compile_with_report(bell)
    assert rep.n_entangling == 1 and rep.n_pulses == 6 and rep.circuit.is_exported_native
    assert ideal_probabilities(bell) == pytest.approx({"00": 0.5, "11": 0.5})
    ghz = Circuit(
        3, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ()), Operation("cnot", (1, 2), ())), (0, 1, 2)
    )
    assert ideal_probabilities(ghz) == pytest.approx({"000": 0.5, "111": 0.5})
    assert compile_with_report(ghz).n_entangling == 2
    # bit order: X on qubit 0 alone reads "01" (qubit 0 rightmost), and a measured subset drops the other qubits
    x0 = Circuit(2, (Operation("x", (0,), ()),), (0, 1))
    assert ideal_probabilities(x0) == pytest.approx({"01": 1.0})
    assert ideal_probabilities(Circuit(2, (Operation("x", (0,), ()),), (1,))) == pytest.approx({"0": 1.0})


def test_ionq_json_round_trip_of_a_compiled_circuit_and_native_passthrough() -> None:
    """Section 9.6: a compiled circuit exports to IonQ JSON and imports back identically; a native circuit compiles to itself
    (its rz absorbed), so the round trip through the compiler is the identity on the exported set."""
    bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    compiled = compile_to_native(bell)
    obj = dump_ionq_json(compiled)
    back = load_ionq_json(obj)
    assert dump_ionq_json(back) == obj
    for a, b in zip(back.ops, compiled.ops):
        assert a.name == b.name and a.qubits == b.qubits and a.params == pytest.approx(b.params, abs=1e-9)
    assert compile_to_native(back).ops == back.ops
    nat = Circuit(
        2, (Operation("rz", (1,), (0.4,)), Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2.0))), (0, 1)
    )
    rep = compile_with_report(nat)
    assert rep.circuit.ops == (Operation("ms", (0, 1), (0.0, pytest.approx(-0.4), math.pi / 2.0)),)  # type: ignore[arg-type]
    assert rep.final_frame_rad == {0: 0.0, 1: pytest.approx(0.4)}


def test_mid_circuit_operations_pass_through_and_skip_the_whole_circuit_check() -> None:
    circ = Circuit(
        2, (Operation("h", (0,), ()), Operation("measure", (0,), ()), Operation("x", (1,), ())), (0, 1)
    )
    rep = compile_with_report(circ)
    assert rep.circuit_residual is None and any("mid-circuit" in n for n in rep.notes)
    assert [op.name for op in rep.circuit.ops] == ["gpi2", "measure", "gpi"]
    with pytest.raises(ValueError, match="non-unitary"):
        circuit_unitary(circ)


# ---- the Circuit builder and its defaults (docs/api_implementation_plan.md 1.5; 0.2.0) -----------------------------------------


def test_the_builder_equals_explicit_construction_and_measures_every_qubit_by_default() -> None:
    import inspect

    from qutip_trap.control.compiler import GATE_PARAMETERS, NATIVE_GATES, STANDARD_GATES

    built = Circuit(2).h(0).cnot(0, 1)
    assert built == Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    assert built.measure == (0, 1) and built.registers == {"c": (0, 1)}
    native = (
        Circuit(2)
        .gpi2(0, phase=0.0)
        .ms(0, 1, phi0=0.0, phi1=0.0, theta=math.pi / 2)
        .zz(1, 0, 0.3)
        .rz(0, theta=0.1)
    )
    assert native.is_native and native.ops[1] == Operation("ms", (0, 1), (0.0, 0.0, math.pi / 2))
    assert native.ops[2] == Operation("zz", (1, 0), (0.3,)) and native.ops[3] == Operation("rz", (0,), (0.1,))
    assert Circuit(2).u3(1, 0.1, 0.2, 0.3).ops[0].params == (0.1, 0.2, 0.3)
    assert Circuit(1).rx(0, theta=0.5) == Circuit(1).rx(0, 0.5)
    # a builder never mutates: the original stays what it was
    base = Circuit(2)
    assert base.h(0) != base and base.ops == () and base.measure == (0, 1)
    # every gate of the two tables is a method whose signature is the table's arity plus its parameters, by name
    for name, (arity, n_params) in {**NATIVE_GATES, **STANDARD_GATES}.items():
        sig = inspect.signature(getattr(Circuit, name))
        assert len(sig.parameters) == 1 + arity + n_params, name
        assert tuple(sig.parameters)[1 + arity :] == GATE_PARAMETERS.get(name, ()), name
        assert (getattr(Circuit, name).__doc__ or "").startswith(f"Append ``{name}``")
    with pytest.raises(ValueError, match="outside range"):
        Circuit(2).h(2)
    with pytest.raises(TypeError):
        Circuit(2).rx(0)  # type: ignore[call-arg]  (the parameter is required)


def test_measured_registers_and_the_third_positional_argument() -> None:
    narrowed = Circuit(3).x(0).measured(0, 1)
    assert narrowed.measure == (0, 1) and narrowed.registers == {"c": (0, 1)}
    split = Circuit(3).x(0).measured(2, 0, registers={"a": (2,), "b": (0,)})
    assert split.measure == (2, 0) and split.registers == {"a": (2,), "b": (0,)}
    # the 0.1.0 call form, unchanged: the third positional argument narrows the measurement
    assert Circuit(2, (), (1,)).measure == (1,) and Circuit(2, (), (1,)).registers == {"c": (1,)}
    assert Circuit(2, (), ()).measure == () and Circuit(2, (), ()).registers == {"c": ()}
    with pytest.raises(ValueError, match="register 'r'"):
        Circuit(2, registers={"r": (0, 5)})
    with pytest.raises(ValueError, match="register 'r'"):
        Circuit(2, registers={"r": (0, 0)})
    # the compiler keeps the registers on the native circuit
    c = Circuit(2, registers={"a": (0,), "b": (1,)}).h(0).cnot(0, 1)
    assert compile_with_report(c).circuit.registers == {"a": (0,), "b": (1,)}
    assert compile_to_native(c).measure == (0, 1)


def test_from_and_to_ionq_on_the_builder() -> None:
    turn = native.rad_from_turns
    native_circuit = Circuit(2).gpi2(0, turn(0.75)).ms(0, 1, 0.0, turn(0.25), turn(0.25)).zz(0, 1, turn(0.1))
    body = native_circuit.to_ionq()
    assert body["gateset"] == "native" and body["circuit"][0] == {"gate": "gpi2", "target": 0, "phase": 0.75}
    assert Circuit.from_ionq(body) == native_circuit  # exact: the turns above are exact binary fractions
    assert Circuit.from_ionq({"input": body, "backend": "simulator"}) == native_circuit
    with pytest.raises(ValueError, match="compile first"):
        Circuit(2).h(0).to_ionq()
