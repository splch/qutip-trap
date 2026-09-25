"""The simulation worker (Section 14.6): a job runs in the worker process with progress events, a zoom reuses its live state."""

from __future__ import annotations

import pytest
from fixtures import BELL, FAST, SEED, SHOTS

from qutip_trap_app.record import Record, job_for_preset
from qutip_trap_app.workers import Event, SimulationWorker, WorkerError


@pytest.fixture(scope="module")
def worker():
    w = SimulationWorker()
    w.start()
    yield w
    w.stop()


def test_an_unknown_request_is_refused_at_submit(worker: SimulationWorker) -> None:
    with pytest.raises(WorkerError, match="unknown request"):
        worker.submit("nonsense")


def test_run_job_then_zoom_in_the_worker(worker: SimulationWorker) -> None:
    job, _preset = job_for_preset(
        "yb171_chain", 2, BELL, SHOTS, seed=SEED, options=FAST, detection_records=500
    )
    seen: list[Event] = []
    record = worker.wait(worker.submit("run_job", job=job), timeout_s=600.0, on_progress=seen.append)
    assert isinstance(record, Record) and record.results.bitstrings.shape == (SHOTS, 2)
    stages = [e.stage for e in seen]
    assert "calibrating" in stages and "running" in stages and stages[-1] == "recording"
    step = next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))
    out = worker.wait(worker.submit("zoom", key=record.key(), step=step, n_store=51), timeout_s=600.0)
    z = out["zoom"]
    assert z.step_index == step and z.trace.times_s.size > 50 and out["stats"].engine_calls == 1
    merged = record.with_boundaries(out["boundaries"]).with_zoom(z)
    assert merged.zoom(z.key) is z and merged.boundary(step, 0, 0) is not None
    again = worker.wait(worker.submit("zoom", key=record.key(), step=step, n_store=51), timeout_s=600.0)
    assert again["stats"].cached
    with pytest.raises(WorkerError, match="no live run"):
        worker.wait(worker.submit("zoom", key="nope", step=0), timeout_s=60.0)
