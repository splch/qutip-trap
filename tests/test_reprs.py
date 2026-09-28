"""What a record prints: one informative line (the option records as the constructor call that rebuilds them, a circuit as
its builder chain), a circuit's text diagram, a progress report as a log line, and a result's summary."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from qutip_trap import Circuit, FidelityLevel, Machine, Numerics, Physics, Readout, presets
from qutip_trap.control.compiler import Operation
from qutip_trap.control.diagram import angle_text
from qutip_trap.control.table import CalibrationTable
from qutip_trap.run.results import Progress
from tests.fixtures import make_calibration_table, make_device, make_result


def test_a_circuit_reads_as_the_builder_chain_that_makes_it() -> None:
    bell = Circuit(2).h(0).cnot(0, 1)
    narrowed = Circuit(3).rx(1, math.pi / 2).cp(0, 2, 0.3).measure(2, 0)
    two_registers = Circuit.from_openqasm(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg a[1]; creg b[1]; h q[0]; cx q[0],q[1];'
        " measure q[0] -> a[0]; measure q[1] -> b[0];"
    )
    mid_circuit = Circuit(2, (Operation("h", (0,), ()), Operation("reset", (1,), ())))
    assert repr(bell) == "Circuit(2).h(0).cnot(0, 1)"
    assert repr(narrowed) == "Circuit(3).rx(1, 1.5707963267948966).cp(0, 2, 0.3).measure(2, 0)"
    assert repr(two_registers) == "Circuit(2).h(0).cnot(0, 1).measure(0, 1, registers={'a': (0,), 'b': (1,)})"
    assert repr(mid_circuit).startswith("Circuit(2, ops=(Operation(name='h'")
    for c in (bell, narrowed, two_registers, mid_circuit):
        assert eval(repr(c)) == c
    long = Circuit(1)
    for _ in range(40):
        long = long.x(0)
    assert repr(long).endswith(".x(0)... (40 operations)") and repr(long).count(".x(0)") == 16


def test_a_circuit_prints_as_its_diagram() -> None:
    assert str(Circuit(2).h(0).cnot(0, 1)) == "q0: ─H──●──M─\n        │\nq1: ────X──M─"
    crossing = str(Circuit(3).rzz(0, 2, math.pi / 2).measure(1))
    assert crossing.splitlines()[2] == "q1: ────┼──────M─" and crossing.splitlines()[0].startswith(
        "q0: ─RZZ(π/2)─"
    )
    folded = Circuit(1)
    for _ in range(60):
        folded = folded.h(0)
    blocks = str(folded).split("\n\n")
    assert len(blocks) > 1 and all(len(line) <= 100 for block in blocks for line in block.splitlines())
    assert sum(block.count("H") for block in blocks) == 60
    assert [angle_text(x) for x in (math.pi / 2, -3 * math.pi / 4, 2 * math.pi, 0.0, 0.3)] == [
        "π/2",
        "-3π/4",
        "2π",
        "0",
        "0.3",
    ]


def test_the_option_records_print_what_differs_from_their_defaults() -> None:
    assert (repr(Numerics()), repr(Physics()), repr(Readout())) == ("Numerics()", "Physics()", "Readout()")
    tuned = Numerics(atol=1e-9, caps={0: 12}, ntraj=128)
    assert repr(tuned) == "Numerics(atol=1e-09, caps={0: 12}, ntraj=128)" and eval(repr(tuned)) == tuned
    assert repr(Physics(noise=False, entangler="zz")) == "Physics(noise=False, entangler='zz')"


def test_a_machine_prints_its_device_and_what_differs_from_the_defaults() -> None:
    device = make_device()
    assert repr(device) == f"<Device {device.hash()[:12]}: 2 x 171Yb+, 6 modes, 2 beams, B = 5 G>"
    machine = Machine(device)
    assert repr(machine) == f"Machine(device={device!r})"
    variant = dataclasses.replace(machine, physics=Physics(noise=False), level="GATE_LOCAL", name="bench")
    assert repr(variant) == (
        f"Machine(device={device!r}, physics=Physics(noise=False), level=FidelityLevel.GATE_LOCAL, name='bench')"
    )
    assert variant.level is FidelityLevel.GATE_LOCAL
    assert repr(presets.yb171_chain(2)).startswith("Machine(device=<Device ")


def test_a_calibration_table_prints_its_identity_and_entry_counts() -> None:
    table = make_calibration_table()
    assert repr(table) == (
        "<CalibrationTable for device fixture: surrogate, seed 0, 1 entries (none uncalibrated), waveforms for pairs "
        "none>"
    )
    assert repr(CalibrationTable.empty()) == "<CalibrationTable: empty>"


def test_a_progress_report_prints_as_one_log_line() -> None:
    assert str(Progress("pulse", 3, 12, 0.84)) == "pulse 3/12 (0.8 s)"


def test_a_result_prints_its_counts_and_summarises_without_a_record() -> None:
    bits = np.array([[1, 1], [0, 0], [0, 0], [1, 0]], dtype=np.uint8)
    res = make_result(bits)
    assert list(res.counts) == ["00", "01", "11"], "counts in key order; bit 0 is the rightmost character"
    assert repr(res) == "<Result: 4 shots on 2 qubits at JOINT_EXACT, counts {'00': 2, '01': 1, '11': 1}>"
    lines = res.summary().splitlines()
    assert lines[1].split() == ["outcome", "counts", "probability"], (
        "no record: no compiled circuit, no ideal"
    )
    assert not any("total variation" in line for line in lines)
    assert lines[-1] == "0 approximations in diagnostics.approximations"


def test_an_experiment_result_prints_its_fits_with_their_uncertainties() -> None:
    from qutip_trap.experiments.result import RabiScan

    scan = RabiScan(
        data=np.zeros((5, 2)),
        fitted={"f_rabi_hz": (160208.9, 17298.7), "contrast": (0.98, 0.02)},
        model="thermal_rabi",
        provenance_id="conv.rabi_frequency",
        subject={"ion": 0, "beam": 2},
    )
    assert repr(scan) == (
        "<RabiScan of ion 0, beam 2: f_rabi_hz = 1.602e+05 +- 1.7e+04, contrast = 0.98 +- 0.02; 5 points, quality exact>"
    )
