"""Fixtures for the application's view-model tests (PLAN.md Section 9.11: they run in the app's CI, not the core's).

One two-ion Bell job on the public 171Yb+ preset is executed once per session (about 20 s: the surrogate calibration and
the run), and every test reads its record. The Flet integration tests (``flet_app`` fixture, ``flet test``) need the Flutter
client and are skipped unless ``QUTIP_TRAP_APP_UI_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest
from fixtures import BELL, FAST, SEED, SHOTS

from qutip_trap_app.core import SolverOptions
from qutip_trap_app.record import LiveRun, Record, calibrate_for, execute, job_for_preset
from qutip_trap_app.replay import ChannelLibrary, replay
from qutip_trap_app.replay_record import build_replay_record


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("QUTIP_TRAP_APP_UI_TESTS"):
        return
    skip = pytest.mark.skip(
        reason="Flet UI tests need the Flutter client: set QUTIP_TRAP_APP_UI_TESTS=1 and run `flet test`"
    )
    for item in items:
        if "flet_app" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def bell() -> tuple[Record, LiveRun]:
    job, preset = job_for_preset(
        "yb171_chain", 2, BELL, SHOTS, seed=SEED, options=FAST, detection_records=500
    )
    return execute(job, preset)


@pytest.fixture(scope="session")
def bell_gate_local() -> tuple[Record, LiveRun]:
    """The same job routed to GATE_LOCAL by a joint-dimension guard far below the Bell run's dimension (Section 11.5): the
    record stores step channels and no trace, the case a three-qubit circuit meets on the full engine (about 4 s)."""
    options = SolverOptions(branch_weight_min=1e-3, joint_dimension_max=8)
    job, preset = job_for_preset(
        "yb171_chain", 2, BELL, SHOTS, seed=SEED, options=options, detection_records=500
    )
    return execute(job, preset)


@pytest.fixture(scope="session")
def under_truncated() -> tuple[Record, LiveRun]:
    """The same job with the two gate modes capped far below what the pulse populates, and the monitor told not to trip
    (Section 9.11 row "Convergence badge": an under-truncated run turns the badge red)."""
    options = SolverOptions(branch_weight_min=1e-3, boundary_population_max=0.5, margin_check=False)
    job, preset = job_for_preset(
        "yb171_chain", 2, BELL, SHOTS, seed=SEED, options=options, detection_records=500, caps={2: 4, 3: 4}
    )
    return execute(job, preset)


@pytest.fixture(scope="session")
def bell_replay(bell: tuple[Record, LiveRun]) -> tuple[Record, ChannelLibrary]:
    """The same Bell job by the app-side channel replay: the library (gpi2 on each ion, the ms pair, each with its frame
    covariance measured) and the replay record (about 45 s: the ms tomography twice)."""
    record, _live = bell
    job = record.job
    preset = job.device.build()
    table = calibrate_for(job, preset)
    library = ChannelLibrary.for_job(job, preset.device, table)
    outcome = replay(job, preset.device, table, library)
    return build_replay_record(job, preset.device, table, outcome, library), library
