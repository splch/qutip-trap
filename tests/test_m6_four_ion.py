"""A four-ion circuit through ``run`` (M6 audit item P1-6; Section 10's M6 bullet 2, Section 9.6 row 2b).

The M6 bullet asks for "2- to 4-ion circuits (Bell, GHZ, small algorithms) with all physics on". Before this milestone no
``run()`` call anywhere used four ions: ``validation/scripts/check_circuits.py`` section 3b builds the four-ion space and the
row's dimension 2304, but calls ``JointExactEngine.run_pulses`` directly rather than the pipeline.

Section 9.6 row 2b's literal fixture is "four ions with two resolved modes at d_m = 12 (dimension 2304)". The surrogate's
adjacent-pair AM waveforms resolve THREE of the four x modes per pair (measured: pair (0,1) resolves 5, 6, 7 and freezes 4),
so the cap rule of Section 5.5 puts the pipeline's own space at dimension 19200, above the Section 11.5 guard, and ``run``
routes to GATE_LOCAL. The row's 2304 is reached by naming the space explicitly, which is what this test does.
"""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.dynamics.engine import SolverOptions
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.options import Numerics
from qutip_trap.run.job import register_fidelity
from tests.fixtures import run
from tests.m6_fixtures import circuit_fixture

WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
X_MODES = (4, 5, 6, 7)
"""The transverse_1 family of a four-ion crystal: 2.2013, 2.5681, 2.8284 and 3.0000 MHz."""
RESOLVED = (6, 7)
"""Row 2b's two resolved modes (the x-COM at 3.0 MHz and the tilt at 2.8284 MHz)."""
D_M = 12
"""Row 2b's cap; 2^4 x 12 x 12 = 2304."""


@pytest.fixture(scope="module")
def four_ion():  # type: ignore[no-untyped-def]
    fixture = circuit_fixture(4, address_waist_m=2.0e-6)
    table = surrogate_table(
        fixture.device, pairs=[(0, 1)], detection_records=800, detection_windows_s=WINDOWS
    )
    return fixture, table


def test_the_row_2b_space_is_dimension_2304() -> None:
    """The arithmetic the row states, as a guard on the fixture: four qubits and two modes at d_m = 12."""
    space = HilbertSpace(
        (2, 2, 2, 2),
        tuple(ModeTruncation(m, D_M, (0, 4), 0.15) for m in RESOLVED),
        None,
        tuple(m for m in range(12) if m not in RESOLVED),
    )
    assert space.dimension == 2304 and space.dims == [2, 2, 2, 2, D_M, D_M]
    three = HilbertSpace(
        (2, 2, 2),
        tuple(ModeTruncation(m, D_M, (0, 4), 0.15) for m in (4, 5)),
        None,
        tuple(range(9))[:4] + (6, 7, 8),
    )
    assert three.dimension == 1152, "row 2a's three-ion dimension"


@pytest.mark.slow
def test_a_four_ion_circuit_runs_through_the_pipeline_at_the_row_2b_dimension(four_ion) -> None:  # type: ignore[no-untyped-def]
    """A Bell pair on ions 0 and 1 of a four-ion crystal, all physics on, on row 2b's explicit space: the pipeline completes
    at JOINT_EXACT dimension 2304, the two named modes are carried at d_m = 12, the histogram is the Bell one on the gate
    pair with the two spectators dark, and the truncation monitor stays inside its threshold."""
    fixture, surrogate = four_ion
    space = HilbertSpace(
        (2, 2, 2, 2),
        tuple(ModeTruncation(m, D_M, (0, 4), 0.15) for m in RESOLVED),
        None,
        tuple(m for m in range(12) if m not in RESOLVED),
    )
    assert space.dimension == 2304
    circuit = Circuit(4, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1, 2, 3))
    result = run(
        circuit,
        fixture.device,
        200,
        table=surrogate.table,
        keep_final_state=True,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=3e-3), space=space),
    )
    diagnostics = result.diagnostics
    assert diagnostics.level == "JOINT_EXACT"
    assert diagnostics.space.dimension == 2304 and diagnostics.space.dims == [2, 2, 2, 2, D_M, D_M]
    assert tuple(t.mode for t in diagnostics.space.resolved) == RESOLVED
    assert all(t.d == D_M for t in diagnostics.space.resolved)
    assert all(v < 1e-6 for v in diagnostics.boundary_population.values())
    assert diagnostics.cap_growth == {}
    # ions 2 and 3 were never addressed, so every shot reads them dark: the histogram lives on 00xx and 11xx
    probabilities = result.probabilities
    bell = probabilities.get("0000", 0.0) + probabilities.get("0011", 0.0)
    assert bell > 0.9, probabilities
    assert 1.0 - register_fidelity(result) < diagnostics.intrinsic_budget["total"]
    assert result.bitstrings.shape == (200, 4)


@pytest.mark.slow
def test_the_pipeline_s_own_four_ion_space_exceeds_the_guard_and_routes_to_gate_local(four_ion) -> None:  # type: ignore[no-untyped-def]
    """Why row 2b's dimension has to be named: the adjacent-pair waveform resolves three x modes, whose cap rule puts the
    joint space above the Section 11.5 guard, so ``run`` takes the GATE_LOCAL level and says so (Section 5.4)."""
    fixture, surrogate = four_ion
    circuit = Circuit(4, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1, 2, 3))
    result = run(
        circuit,
        fixture.device,
        50,
        table=surrogate.table,
        numerics=Numerics.from_solver_options(SolverOptions(branch_weight_min=1e-2)),
    )
    diagnostics = result.diagnostics
    assert diagnostics.level == "GATE_LOCAL" and diagnostics.gate_local is not None
    assert any("GATE_LOCAL" in note for note in diagnostics.approximations)
    resolved = sorted(t.mode for t in diagnostics.space.resolved)
    assert len(resolved) == 3 and set(resolved) <= set(X_MODES), resolved
    assert diagnostics.space.dimension > 4096
