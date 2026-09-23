"""The Qiskit door (``qutip_trap.interop.qiskit``): transpiled Qiskit circuits run on the two-ion example device and come
back as a Qiskit ``Result`` in Qiskit's bit order, with the qutip-trap ``Result`` behind each experiment."""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit, transpile  # noqa: E402

from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.interop.qiskit import QutipTrapBackend, QutipTrapProvider  # noqa: E402
from qutip_trap.machine import Machine  # noqa: E402
from qutip_trap.options import Numerics, Readout  # noqa: E402

FAST = Numerics(truncation={"branch_weight_min": 1e-3})  # type: ignore[arg-type]  (a mapping is accepted)


def test_bell_and_x_on_qubit_zero_through_qiskit() -> None:
    backend = QutipTrapProvider().get_backend("yb171_chain", n_ions=2)
    bell = QuantumCircuit(2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    flip = QuantumCircuit(2, name="x0")  # asymmetric: it pins the bit order, which a Bell state cannot
    flip.x(0)
    flip.measure_all()
    assert (
        isinstance(backend.machine, Machine) and backend.machine.device.hash() == yb171_chain(2).device.hash()
    )
    job = backend.run([transpile(c, backend) for c in (bell, flip)], shots=400, seed=3, numerics=FAST)
    result = job.result()
    assert result.success and job.status().name == "DONE"
    counts = result.get_counts("bell")
    assert set(counts) <= {"00", "11", "01", "10"}
    assert (counts.get("00", 0) + counts.get("11", 0)) / 400 > 0.95
    x0 = result.get_counts("x0")
    # qubit 0 set reads "01": Qiskit's rightmost bit and qutip-trap's are the same qubit
    assert x0.get("01", 0) > 380, x0
    assert job.results[0].diagnostics.level == "JOINT_EXACT"
    assert job.results[1].probabilities["01"] == x0["01"] / 400


def test_the_option_objects_flow_through_the_backend() -> None:
    """A run through the option objects (``readout="full"`` as a ``Readout``); an unknown option is refused."""
    machine = yb171_chain(2).machine()
    backend = QutipTrapBackend(machine, numerics=FAST)
    flip = QuantumCircuit(2, name="x0")
    flip.x(0)
    flip.measure_all()
    circuit = transpile(flip, backend)
    full = backend.run(circuit, shots=50, seed=1, readout=Readout(mode="full")).results[0]
    assert full.photon_records is not None and full.photon_records.shape == (50, 2)
    assert full.diagnostics.calibration is backend.machine.table or backend.machine.table is None
    with pytest.raises(TypeError, match="unexpected options"):
        backend.run(circuit, shots=5, options=SolverOptions())
