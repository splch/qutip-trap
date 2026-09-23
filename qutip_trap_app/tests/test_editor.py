"""The circuit editor: OpenQASM 2 and IonQ JSON texts parse through the core's importers, a malformed text or too many
qubits is refused with a message, the presets load, and a run is built from the text or not submitted at all."""

from __future__ import annotations

import json
import math

import pytest

from qutip_trap_app.viewmodel.editor import BELL_QASM, GHZ_QASM, MAX_QUBITS, MAX_SHOTS, PRESETS, parse_circuit
from qutip_trap_app.views.state import FAST, Session, Store
from qutip_trap_app.workers import Ticket


def test_the_presets_parse() -> None:
    bell = parse_circuit(BELL_QASM)
    assert bell.n_qubits == 2 and [op.name for op in bell.ops] == ["h", "cnot"] and bell.measure == (0, 1)
    ghz = parse_circuit(GHZ_QASM)
    assert ghz.n_qubits == 3 and [op.qubits for op in ghz.ops if op.name == "cnot"] == [(0, 1), (1, 2)]
    assert {p.text for p in PRESETS.values()} == {BELL_QASM, GHZ_QASM}


def test_ionq_json_is_read_when_the_text_is_an_object() -> None:
    text = json.dumps(
        {
            "qubits": 2,
            "circuit": [
                {"gate": "gpi2", "target": 0, "phase": 0.25},
                {"gate": "ms", "targets": [0, 1], "phases": [0, 0], "angle": 0.25},
            ],
        }
    )
    circuit = parse_circuit("\n  " + text)
    assert [op.name for op in circuit.ops] == ["gpi2", "ms"] and math.isclose(
        circuit.ops[1].params[2], math.pi / 2
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "qreg q[2]; h q[5];",
        "OPENQASM 3.0; qubit q;",
        "{not json",
        '{"qubits": 1, "circuit": [{"gate": "nope"}]}',
    ],
)
def test_a_malformed_text_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_circuit(text)


def test_too_many_qubits_are_refused() -> None:
    with pytest.raises(ValueError, match=f"at most {MAX_QUBITS}"):
        parse_circuit(f"qreg q[{MAX_QUBITS + 1}]; h q[0];")


class _Recorder:
    alive = True

    def __init__(self) -> None:
        self.submitted: list[tuple[str, dict[str, object]]] = []

    def submit(self, request: str, **payload: object) -> Ticket:
        self.submitted.append((request, payload))
        return Ticket(f"t{len(self.submitted)}", request)


def test_the_session_builds_a_run_from_the_text() -> None:
    session = Session(Store())
    worker = _Recorder()
    session.worker = worker
    store = session.store
    session.load_preset("GHZ state, three ions")
    assert store.circuit_text == GHZ_QASM and store.device_kwargs == {"address_waist_m": 2.0e-6}
    status = session.submit_run()
    assert status is not None and store.running_of("run") is status
    job = worker.submitted[0][1]["job"]
    assert job.n_ions == 3 and job.device_kwargs == {"address_waist_m": 2.0e-6}
    assert job.spec.numerics == FAST and job.spec.keep_final_state and job.spec.shots == store.shots
    store.circuit_text = "qreg q[2]; h q[7];"
    assert session.submit_run() is None and "could not be read" in store.error and len(worker.submitted) == 1
    session.load_preset("Bell state")
    assert store.error == "" and store.device_kwargs == {}
    assert session.build_job().n_ions == 2


def test_shots_are_a_bounded_whole_number() -> None:
    session = Session(Store())
    session.set_shots("50")
    assert session.store.shots == 50 and session.store.error == ""
    session.set_shots("many")
    assert session.store.shots == 50 and "whole number" in session.store.error
    session.set_shots(str(10 * MAX_SHOTS))
    assert session.store.shots == MAX_SHOTS and "capped" in session.store.error
    session.set_shots("0")
    assert session.store.shots == 1
