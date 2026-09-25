"""The register spans every ion while a circuit may address, and measure, fewer qubits (Sections 3.4, 8.6): the fidelity
against the ideal state and the run record's readout outcome keep their shapes straight in those cases. Found by the
application, whose device never has fewer than two ions: a one-qubit circuit on the two-ion chain raised ``incompatible
dimensions [2, 2] and [2]`` in ``register_fidelity``, and the record's readout outcome held the last (sample, branch) batch
of shots instead of the run's."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.options import Numerics, Physics
from qutip_trap.run.job import ideal_register_state, last_record, register_fidelity
from tests.fixtures import run
from tests.m6_fixtures import circuit_fixture

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
FAST = SolverOptions(branch_weight_min=1e-3)


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    sur = surrogate_table(fx.device, pairs=[(0, 1)], detection_records=500, detection_windows_s=WINDOWS)
    return fx, sur


def _run(two_ion, circuit: Circuit, shots: int = 40):  # type: ignore[no-untyped-def]
    fx, sur = two_ion
    return run(
        circuit,
        fx.device,
        shots,
        table=sur.table,
        keep_final_state=True,
        physics=Physics.from_solver_options(FAST),
        numerics=Numerics.from_solver_options(FAST),
    )


def test_register_fidelity_of_a_circuit_on_fewer_qubits_than_ions(two_ion) -> None:  # type: ignore[no-untyped-def]
    """A one-qubit circuit on the two-ion chain: the register state is 4 x 4, the ideal ket is |+> on ion 0 and |0> on the
    idle ion 1, and the fidelity is that of the played pi/2 pulse."""
    res = _run(two_ion, Circuit(1, (Operation("h", (0,), ()),), (0,)))
    assert res.final_state is not None and res.final_state.dims[0] == [2, 2]
    assert res.n_qubits == 1 and res.bitstrings.shape == (40, 1)
    fid = register_fidelity(res)
    assert 0.99 < fid <= 1.0 + 1e-12
    # the same number by hand: the compiled circuit's ket on ion 0 (its frame absorbed) with |0> on the idle ion 1
    rho = np.asarray(res.final_state.full())
    ideal = np.kron(ideal_register_state(res), np.array([1.0, 0.0])).astype(complex)
    assert fid == pytest.approx(float(np.real(ideal.conj() @ rho @ ideal)), abs=1e-12)
    # an explicit target on the circuit's qubits is embedded the same way
    plus = np.array([1.0, 1.0]) / math.sqrt(2.0)
    plus0 = np.kron(plus, np.array([1.0, 0.0])).astype(complex)
    assert register_fidelity(res, plus) == pytest.approx(
        float(np.real(plus0.conj() @ rho @ plus0)), abs=1e-12
    )
    with pytest.raises(ValueError, match="spans 3 qubits"):
        register_fidelity(res, np.ones(8) / math.sqrt(8.0))


def test_register_fidelity_when_the_circuit_measures_a_subset(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The Bell circuit measuring qubit 0 only: one histogram column, the fidelity against the full two-qubit Bell ket."""
    bell_q0 = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0,))
    res = _run(two_ion, bell_q0)
    assert res.n_qubits == 1 and res.qubits == (0,) and set(res.counts) <= {"0", "1"}
    fid = register_fidelity(res)
    assert 0.98 < fid <= 1.0 + 1e-12


def test_run_record_outcome_covers_every_kept_shot(two_ion) -> None:  # type: ignore[no-untyped-def]
    """The readout runs once per (sample, branch) batch; the RunRecord's outcome is the concatenation over the kept shots in
    the Result's row order, over every ion, so a per-shot reader lines up with ``Result.bitstrings``."""
    bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    res = _run(two_ion, bell, shots=60)
    rec = last_record(res)
    assert len(rec.branches) > 1, (
        "the fixture's initial mixture has several branches, so several readout batches"
    )
    out = rec.outcome
    assert out.bits.shape == (res.shots, 2) and out.levels.shape == (res.shots, 2)
    assert out.time_used_s.shape == (res.shots, 2)
    assert np.array_equal(out.bits, res.bitstrings), "every ion measured: the declared bits are the Result's"
    assert out.mode == "fast" and out.records is None
