"""The application state: a finished run becomes the current record and drops the previous one's selections, the progress
rows keep moving between worker events, a failed job shows its reason, and cancel marks every running job."""

from __future__ import annotations

import pytest

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.views.state import JobStatus, Session, Store
from qutip_trap_app.workers import Event


class _Page:
    def __init__(self) -> None:
        self.routes: list[str] = []

    def navigate(self, route: str) -> None:
        self.routes.append(route)


def test_a_finished_run_becomes_the_current_record(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    page = _Page()
    session = Session(Store(), page)
    store = session.store
    store.selected_bar, store.selected_shot, store.branch = "01", 3, 2
    store.jobs = {"t1": JobStatus("t1", "run")}
    session.apply_events(
        [Event("progress", "t1", "run", stage="running: pulse", fraction=0.5, message="pulse 2 of 4")]
    )
    assert store.jobs["t1"].fraction == 0.5 and not store.jobs["t1"].done
    session.apply_events([Event("result", "t1", "run", payload=record)])
    assert store.jobs["t1"].done and store.current == record.key and store.records[record.key] is record
    assert (store.selected_bar, store.selected_shot, store.branch) == (None, None, 0)
    assert page.routes == [f"/job/{record.key}"]


def test_a_failed_job_shows_its_reason() -> None:
    session = Session(Store())
    store = session.store
    store.jobs = {"t1": JobStatus("t1", "zoom")}
    session.apply_events(
        [Event("error", "t1", "zoom", message="Traceback ...\nResimError: no joint state\n")]
    )
    assert store.jobs["t1"].stage == "failed" and store.error == "ResimError: no joint state"


def test_progress_rows_keep_moving_between_worker_events() -> None:
    session = Session(Store())
    store = session.store
    t0 = 1000.0
    session._last_beat = t0
    assert not session.heartbeat(t0 + 5.0), "nothing runs: nothing to re-render"
    store.jobs = {"t1": JobStatus("t1", "run")}
    tick = store.tick
    assert not session.heartbeat(t0 + 0.5) and store.tick == tick
    assert session.heartbeat(t0 + 1.0) and store.tick == tick + 1
    assert not session.heartbeat(t0 + 1.5), "once per second"


def test_cancel_marks_every_running_job(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Session(Store())
    calls: list[str] = []
    monkeypatch.setattr(session.worker, "cancel", lambda: calls.append("cancel"))
    finished = JobStatus("t0", "run", done=True, stage="done")
    running = JobStatus("t1", "zoom")
    session.store.jobs = {"t0": finished, "t1": running}
    session.cancel_all()
    assert calls == ["cancel"] and running.done and running.stage == "cancelled" and finished.stage == "done"
