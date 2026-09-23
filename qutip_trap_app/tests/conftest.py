"""Session fixtures: one two-ion Bell run on the example 171Yb+ chain (about 20 s), the same job routed to GATE_LOCAL and
an under-truncated one, and the Bell run with its entangling step re-simulated, tomographed and re-checked."""

from __future__ import annotations

import pytest
from fixtures import BellResimulated, job, ms_step, truncation

from qutip_trap_app import resim
from qutip_trap_app.record import LiveRun, Record, execute


@pytest.fixture(scope="session")
def bell() -> tuple[Record, LiveRun]:
    return execute(job())


@pytest.fixture(scope="session")
def bell_gate_local() -> tuple[Record, LiveRun]:
    """The Bell job routed to GATE_LOCAL by a joint-dimension guard far below its dimension: the record stores the walk's
    register after every step and no trace, the case a larger circuit meets (about 4 s)."""
    return execute(job(numerics=truncation(joint_dimension_max=8)))


@pytest.fixture(scope="session")
def under_truncated() -> tuple[Record, LiveRun]:
    """The Bell job with the two gate modes capped far below what the pulse populates and the monitor told not to trip."""
    return execute(
        job(numerics=truncation(boundary_population_max=0.5, margin_check=False, caps={2: 4, 3: 4}))
    )


@pytest.fixture(scope="session")
def bell_resimulated(bell: tuple[Record, LiveRun]) -> BellResimulated:
    record, live = bell
    step = ms_step(record)
    record, z, _cached = resim.zoom(record, live, step)
    record, ham = resim.hamiltonian_record(record, live, step)
    record, pm = resim.process_matrix(record, live, step)
    record, tol = resim.tolerance_recheck(record, live, step)
    record, cap = resim.truncation_recheck(record, live, step)
    return BellResimulated(record, live, step, z, ham, pm, tol, cap)
