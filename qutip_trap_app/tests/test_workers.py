"""The simulation worker: a job runs in the worker process with progress events, and a zoom reuses its live state."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures import SHOTS, job, ms_step

from qutip_trap_app.record import Record, ZoomTrace
from qutip_trap_app.resim import DEFAULT_ZOOM_POINTS, zoom_key
from qutip_trap_app.workers import Event, Resimulated, SimulationWorker, WorkerError


@pytest.fixture(scope="module")
def worker() -> Iterator[SimulationWorker]:
    w = SimulationWorker()
    w.start()
    yield w
    w.stop()


def test_ping_and_errors(worker: SimulationWorker) -> None:
    assert worker.wait(worker.submit("ping"), timeout_s=60.0) == "pong"
    with pytest.raises(WorkerError, match="unknown request"):
        worker.wait(worker.submit("nonsense"), timeout_s=60.0)


def test_a_run_then_a_zoom_in_the_worker(worker: SimulationWorker) -> None:
    seen: list[Event] = []
    record = worker.wait(worker.submit("run", job=job()), timeout_s=600.0, on_progress=seen.append)
    assert isinstance(record, Record) and record.results.bitstrings.shape == (SHOTS, 2)
    stages = [e.stage for e in seen]
    assert stages[0] == "calibrating" and any(s.startswith("running: ") for s in stages)
    step = ms_step(record)
    out = worker.wait(worker.submit("zoom", key=record.key, step=step), timeout_s=600.0)
    assert isinstance(out, Resimulated)
    merged = record.with_cached(*out.cached)
    z = merged.cached(zoom_key(step, 0, 0, DEFAULT_ZOOM_POINTS), ZoomTrace)
    assert z is not None and z.step_index == step and z.trace.times_s.size > 200
    again = worker.wait(worker.submit("zoom", key=record.key, step=step), timeout_s=600.0)
    assert isinstance(again, Resimulated) and again.cached == (), "served from the worker's cache"
    with pytest.raises(WorkerError, match="no live run"):
        worker.wait(worker.submit("zoom", key="nope", step=0), timeout_s=60.0)
