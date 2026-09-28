"""A request the machine cannot run is refused at the door, before anything is calibrated or integrated, with a message that
says what to do: the wrong kind of circuit, a circuit wider than the crystal, an entangling gate on a device without an
entangling drive, and shots or a seed that are not whole numbers."""

from __future__ import annotations

import pytest

from qutip_trap import Circuit, FidelityLevel, presets
from qutip_trap.machine import Machine
from tests.fixtures import make_device

BELL = Circuit(2).h(0).cnot(0, 1)


def test_a_builder_call_with_its_arguments_out_of_order_shows_the_call_and_the_order() -> None:
    with pytest.raises(
        TypeError, match=r"rx\(0\.5, 0\): the qubits come first, as whole numbers, then the angles"
    ):
        Circuit(1).rx(0.5, 0)
    with pytest.raises(TypeError, match=r"rx\(0, 'pi/2'\): angles are numbers in radians"):
        Circuit(1).rx(0, "pi/2")  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def machine() -> Machine:
    return Machine(make_device())


@pytest.mark.parametrize(
    ("circuit", "hint"),
    [
        ('OPENQASM 2.0; include "qelib1.inc"; qreg q[1];', "Circuit.from_openqasm"),
        ({"qubits": 1, "circuit": []}, "Circuit.from_ionq"),
        ([BELL, BELL], "run a batch in a loop"),
        (None, "Circuit(n_qubits)"),
    ],
)
def test_anything_but_a_circuit_names_the_door_it_belongs_to(
    machine: Machine, circuit: object, hint: str
) -> None:
    for call in (machine.run, machine.submit):
        with pytest.raises(TypeError, match="expected a qutip_trap Circuit") as err:
            call(circuit, 10)  # type: ignore[arg-type]
        assert hint in str(err.value)
    for method in (machine.compile, machine.schedule, machine.estimate):
        with pytest.raises(TypeError, match="expected a qutip_trap Circuit"):
            method(circuit)  # type: ignore[arg-type]


def test_a_qiskit_circuit_is_pointed_at_the_qiskit_backend(machine: Machine) -> None:
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(2)
    with pytest.raises(TypeError, match="QutipTrapBackend"):
        machine.run(qc, 10)


def test_a_circuit_wider_than_the_crystal_is_refused_before_it_runs(machine: Machine) -> None:
    for wide in (Circuit(3).h(0).x(2), Circuit(3).x(0)):  # the second leaves the extra qubit idle
        for call in (
            lambda c: machine.run(c, 10),
            lambda c: machine.submit(c, 10),
            machine.estimate,
            machine.schedule,
        ):
            with pytest.raises(ValueError, match="3 qubits and the machine 2 ion") as err:
                call(wide)
            assert "yb171_chain(3)" in str(err.value)


def test_an_entangling_gate_without_an_entangling_drive_is_refused_before_the_calibration() -> None:
    optical = presets.ca40_optical(2)
    for call in (lambda c: optical.run(c, 10), optical.estimate, optical.schedule):
        with pytest.raises(ValueError, match="no entangling drive"):
            call(BELL)


@pytest.mark.parametrize(
    ("shots", "seed", "error", "match"),
    [
        (10.5, 0, TypeError, "shots is a whole number"),
        (True, 0, TypeError, "shots is a whole number"),
        ("100", 0, TypeError, "shots is a whole number"),
        (0, 0, ValueError, "shots is at least 1"),
        (10, -1, ValueError, "seed is at least 0"),
        (10, 1.5, TypeError, "seed is a whole number"),
    ],
)
def test_shots_and_seed_are_whole_numbers(
    machine: Machine, shots: object, seed: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        machine.run(Circuit(1).x(0), shots, seed=seed)  # type: ignore[arg-type]
    with pytest.raises(error, match=match):
        machine.spec(Circuit(1).x(0), shots, seed=seed)  # type: ignore[arg-type]


def test_a_level_is_named_in_any_case_and_a_wrong_name_lists_the_levels() -> None:
    assert FidelityLevel("joint_exact") is FidelityLevel.JOINT_EXACT
    assert FidelityLevel("AUTO") is FidelityLevel.AUTO
    with pytest.raises(ValueError, match="the levels are"):
        FidelityLevel("joint")


def test_calibrate_names_the_scan_settings_it_takes(machine: Machine) -> None:
    """The settings ``calibrate`` passes through are exactly the builders' keywords the machine does not supply, so an
    unknown one is refused by ``calibrate`` rather than deep inside a builder."""
    import inspect

    from qutip_trap.calibration import SCANS, calibrate
    from qutip_trap.calibration.experiments import full_calibration
    from qutip_trap.calibration.surrogate import surrogate_table

    supplied = {"device", "seed", "t0_s", "gate_drives", "entangling_drives", "pairs", "options", "physics"}
    own = {n for n, p in inspect.signature(surrogate_table).parameters.items() if p.kind is p.KEYWORD_ONLY}
    assert SCANS["closed_form"] == own - supplied
    full = {n for n, p in inspect.signature(full_calibration).parameters.items() if p.kind is p.KEYWORD_ONLY}
    assert SCANS["experiments"] == (full - supplied - {"experiments"}) | SCANS["closed_form"]
    with pytest.raises(TypeError, match=r"unknown scan setting\(s\) \['not_a_scan'\]"):
        calibrate(machine, not_a_scan=1)
    with pytest.raises(ValueError, match="method is 'closed_form' or 'experiments'"):
        calibrate(machine, method="bogus")  # type: ignore[arg-type]


def test_an_experiment_on_an_ion_the_crystal_does_not_have_says_which_ions_it_has(machine: Machine) -> None:
    from qutip_trap.experiments.single_ion import rabi_scan

    with pytest.raises(ValueError, match="ion 5 is not an ion of the 2-ion device"):
        rabi_scan(machine, 5, (0.0, 1e-6))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0] cx q[0],q[1];',
            r"line 4, column 8.*a missing ';'",
        ),
        ("OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nc = measure q;", "OpenQASM 3.0 is not supported"),
        ("bell.qasm", "looks like a file path"),
        ('OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; ccx q[0],q[1],q[2];', "qiskit.transpile"),
        ("OPENQASM 2.0;\nqreg q[1];\nh r[0];", "unknown qreg 'r' at line 3, column 3"),
    ],
)
def test_the_openqasm_importer_says_where_and_what_to_do(text: str, match: str) -> None:
    from qutip_trap.io.openqasm import OpenQASMError

    with pytest.raises(OpenQASMError, match=match):
        Circuit.from_openqasm(text)


def test_qelib1s_phase_gate_is_a_virtual_rz() -> None:
    circuit = Circuit.from_openqasm('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; p(0.25) q[0];')
    assert [(op.name, op.params) for op in circuit.ops] == [("rz", (0.25,))]


@pytest.mark.parametrize(
    ("obj", "error", "match"),
    [
        ('{"qubits": 1, "circuit": []}', TypeError, r"json.loads\(text\)"),
        ({"circuit": []}, ValueError, r"missing \['qubits'\]"),
        ({"qubits": 1, "circuit": [{"gate": "gpi", "target": 0}]}, ValueError, "needs its 'phase'"),
    ],
)
def test_the_ionq_importer_names_what_is_missing(obj: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        Circuit.from_ionq(obj)  # type: ignore[arg-type]


def test_a_result_record_is_refused_by_what_it_lacks() -> None:
    from qutip_trap import Result

    with pytest.raises(ValueError, match="no 'schema_version'"):
        Result.from_dict({})
    with pytest.raises(TypeError, match=r"json.loads\(text\)"):
        Result.from_dict("{}")  # type: ignore[arg-type]
