"""The application state behind the screens (DESIGN.md Sections 3 and 4): the learner persists as a document, a run scores
the prediction made for it and a new prediction is asked only for a new run, the progress rows keep moving between worker
events, cancel marks every running job, and the zoom bar's parent route needs no hidden global."""

from __future__ import annotations

import json

import pytest

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.learn import Attempt, MasteryLog, review_gap_days
from qutip_trap_app.views.shell import parent_route
from qutip_trap_app.views.state import (
    LEARNER_KEY,
    JobStatus,
    Learner,
    Session,
    Store,
    learner_document,
    learner_from_document,
)
from qutip_trap_app.workers import Event


@pytest.fixture(scope="module")
def index() -> ProvenanceIndex:
    return ProvenanceIndex.load()


def test_learner_round_trips_through_its_document() -> None:
    log = MasteryLog(retention_days=30.0)
    log.record(Attempt("histogram", "histogram.q1", 1.5, True, unaided=False, score=0.42))
    log.record(Attempt("histogram", "histogram.q2", 2.5, None, unaided=False))
    learner = Learner(
        knowledge="circuits",
        retention_days=30.0,
        asked=True,
        depth_override="picture",
        explain_open=False,
        log=log,
    )
    back = learner_from_document(json.loads(json.dumps(learner_document(learner))))
    assert (back.knowledge, back.retention_days, back.asked, back.depth_override, back.explain_open) == (
        "circuits",
        30.0,
        True,
        "picture",
        False,
    )
    assert back.log.attempts == log.attempts and back.log.retention_days == 30.0
    assert back.log.attempts[0].score == 0.42 and back.log.attempts[1].score is None
    assert LEARNER_KEY.startswith("qutip_trap_app.")


def test_learner_document_reports_unknown_values_and_accepts_an_older_document() -> None:
    with pytest.raises(ValueError, match="prior knowledge"):
        learner_from_document({"knowledge": "wizard"})
    with pytest.raises(ValueError, match="depth"):
        learner_from_document({"depth_override": "video"})
    with pytest.raises(ValueError, match="retention"):
        learner_from_document({"retention_days": 0.0})
    fresh = learner_from_document({})
    assert fresh.knowledge == "unknown" and not fresh.asked and fresh.log.attempts == []


def test_set_learner_keeps_the_log_in_step_and_a_skip_counts_as_an_exposure(index: ProvenanceIndex) -> None:
    session = Session(Store(), index)
    session.set_learner(retention_days=30.0)
    assert session.store.learner.log.retention_days == 30.0
    _lo, hi = review_gap_days(30.0)
    session.record_attempt(Attempt("shot", "shot.q1", 0.0, None, unaided=False))
    assert session.store.learner.log.due(now_days=hi + 0.1) == ("shot",)
    assert session.store.learner.log.session_accuracy("shot", now_days=0.1) is None, "a skip is never wrong"


def test_a_run_scores_its_own_prediction_and_a_new_run_is_predicted_again(
    bell: tuple[Record, LiveRun], index: ProvenanceIndex
) -> None:
    record, _live = bell
    session = Session(Store(), index)
    store = session.store
    assert store.prediction_pending(), "before the first run the learner is asked"
    store.prediction = "ideal"
    store.last_run_text = store.circuit_text  # what submit_run records
    assert not store.prediction_pending(), "not asked again while that run is in flight"
    store.jobs = {"t1": JobStatus("t1", "run_job", engine="full")}
    session.apply_events([Event("result", "t1", "run_job", payload=record)])
    assert store.current == record.key() and store.jobs["t1"].done
    assert store.scored_prediction == "ideal" and store.prediction is None
    assert not store.prediction_pending(), (
        "the same circuit has been run: its prediction is scored, not re-asked"
    )
    store.circuit_text = store.circuit_text + "\n"
    assert store.prediction_pending(), "an edited circuit is a new run: predict again"


def test_progress_rows_keep_moving_between_worker_events(index: ProvenanceIndex) -> None:
    session = Session(Store(), index)
    store = session.store
    t0 = 1000.0
    session._last_beat = t0
    assert not session.heartbeat(t0 + 5.0), "nothing runs: nothing to re-render"
    store.jobs = {"t1": JobStatus("t1", "replay")}
    tick = store.tick
    assert not session.heartbeat(t0 + 0.5) and store.tick == tick
    assert session.heartbeat(t0 + 1.0) and store.tick == tick + 1
    assert not session.heartbeat(t0 + 1.5), "once per second"


def test_cancel_marks_every_running_job(index: ProvenanceIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    session = Session(Store(), index)
    calls: list[str] = []
    monkeypatch.setattr(session.worker, "cancel", lambda: calls.append("cancel"))
    finished = JobStatus("t0", "replay", done=True, stage="done")
    running = JobStatus("t1", "verify")
    session.store.jobs = {"t0": finished, "t1": running}
    session.cancel_all()
    assert calls == ["cancel"]
    assert running.done and running.stage == "cancelled" and running.error
    assert finished.stage == "done"


def test_parent_route_reads_the_record_from_the_store(bell: tuple[Record, LiveRun]) -> None:
    store = Store()
    assert parent_route(store, "/job/k") == "/"
    assert parent_route(store, "/job/k/circuit/ms[2]") == "/job/k"
    assert parent_route(store, "/job/k/schedule/3") == "/job/k/circuit/0", (
        "an unknown record falls back to gate 0"
    )
    assert parent_route(store, "/job/k/dynamics/3/0") == "/job/k/schedule/3"
    assert parent_route(store, "/learn") is None
    record, _live = bell
    key = record.key()
    store.records = {key: record}
    step = next(s for s in record.schedule.steps if s.gate_id.startswith("ms"))
    up = parent_route(store, f"/job/{key}/schedule/{step.pulse_indices[0]}")
    assert up is not None and up.startswith(f"/job/{key}/circuit/ms")
