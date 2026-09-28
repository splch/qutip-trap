"""The application state behind the screens: the learner persists as a document, a run scores the prediction made for it and
a new prediction is asked only for a new run, a job keeps the circuit's named registers and gets the readout picked for
it, the progress rows keep moving between worker events, cancel marks every running job, and the zoom bar's parent route
needs no hidden global."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import CircuitRecord, JobSpec, LiveRun, Record, complete_job, execute
from qutip_trap_app.viewmodel.learn import Attempt, MasteryLog, review_gap_days
from qutip_trap_app.viewmodel.machine import shot
from qutip_trap_app.views import routes, state
from qutip_trap_app.views.shell import parent_route
from qutip_trap_app.views.state import (
    FULL,
    LEARNER_KEY,
    REPLAY,
    JobStatus,
    Landing,
    Learner,
    Session,
    Store,
    learner_document,
    learner_from_document,
)
from qutip_trap_app.workers import Event, Ticket


@pytest.fixture(scope="module")
def index() -> ProvenanceIndex:
    return ProvenanceIndex.load()


class _Preferences:
    """A storage service whose read answers when the test says so."""

    def __init__(self, saved: str | None) -> None:
        self.saved = saved
        self.answer = asyncio.Event()
        self.writes: list[str] = []

    async def get(self, key: str) -> str | None:
        assert key == LEARNER_KEY
        await self.answer.wait()
        return self.saved

    async def set(self, key: str, value: str) -> bool:
        self.writes.append(value)
        return True


class _Page:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[Any]] = []

    def run_task(self, fn: Any, *args: Any) -> None:
        self.tasks.append(asyncio.ensure_future(fn(*args)))


def _saved_learner() -> str:
    log = MasteryLog()
    log.record(Attempt("histogram", "histogram.q1", 1.0, True, unaided=False))
    return json.dumps(learner_document(Learner(knowledge="physicist", asked=True, log=log)))


async def test_a_saved_learner_read_in_time_is_the_learner(index: ProvenanceIndex) -> None:
    prefs, page = _Preferences(_saved_learner()), _Page()
    session = Session(Store(), index, page, preferences=prefs)
    prefs.answer.set()
    await session.restore_learner()
    learner = session.store.learner
    assert session.store.learner_loaded and learner.knowledge == "physicist" and learner.asked
    assert len(learner.log.attempts) == 1 and prefs.writes == [], "read, not written back"


async def test_a_slow_read_asks_and_then_lands_without_losing_the_log(
    index: ProvenanceIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the bound the question is asked; the read goes on, and when it lands it keeps the answer and the attempts
    made meanwhile. Nothing is saved before, so the saved log is never replaced by the session's empty one."""
    monkeypatch.setattr(state, "RESTORE_TIMEOUT_S", 0.01)
    prefs, page = _Preferences(_saved_learner()), _Page()
    session = Session(Store(), index, page, preferences=prefs)
    restoring = asyncio.ensure_future(session.restore_learner())
    await asyncio.sleep(0.05)
    assert session.store.learner_loaded and not session.store.learner.asked, "the question is asked"
    session.set_learner(knowledge="circuits", asked=True)
    session.record_attempt(Attempt("shot", "shot.q1", 2.0, False, unaided=True))
    assert page.tasks == [] and prefs.writes == [], "nothing saved while the read is out"
    prefs.answer.set()
    await restoring
    await asyncio.gather(*page.tasks)
    learner = session.store.learner
    assert learner.knowledge == "circuits" and learner.asked, "the answer given meanwhile stands"
    assert [a.prompt_id for a in learner.log.attempts] == ["histogram.q1", "shot.q1"], (
        "the saved log and the new"
    )
    assert [json.loads(w)["attempts"] for w in prefs.writes] == [
        [dataclasses.asdict(a) for a in learner.log.attempts]
    ], "saved once, merged"


async def test_an_unreadable_saved_learner_is_reported_and_replaced(index: ProvenanceIndex) -> None:
    prefs, page = _Preferences("{not json"), _Page()
    session = Session(Store(), index, page, preferences=prefs)
    prefs.answer.set()
    await session.restore_learner()
    assert session.store.learner_loaded and "could not be read" in session.store.error
    session.set_learner(theme="dark")
    await asyncio.gather(*page.tasks)
    assert len(prefs.writes) == 1 and json.loads(prefs.writes[0])["theme"] == "dark"


def test_learner_round_trips_through_its_document() -> None:
    log = MasteryLog(retention_days=30.0)
    log.record(Attempt("histogram", "histogram.q1", 1.5, True, unaided=False))
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


def test_learner_document_refuses_unknown_values_and_missing_keys() -> None:
    doc = learner_document(Learner())
    assert learner_from_document(doc) == Learner()
    for key, value, message in (
        ("knowledge", "wizard", "prior knowledge"),
        ("depth_override", "video", "depth"),
        ("theme", "sepia", "theme"),
        ("retention_days", 0.0, "retention"),
    ):
        with pytest.raises(ValueError, match=message):
            learner_from_document({**doc, key: value})
    with pytest.raises(KeyError, match="theme"):
        learner_from_document({k: v for k, v in doc.items() if k != "theme"})


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
    store.jobs = {"t1": JobStatus("t1", "run_job", landing=Landing(None, routes.job))}
    session.apply_events([Event("result", "t1", "run_job", payload=record)])
    assert store.current == record.key() and store.jobs["t1"].done, (
        "headless, the finished run becomes current"
    )
    assert store.scored_prediction == "ideal" and store.prediction is None
    assert not store.prediction_pending(), (
        "the same circuit has been run: its prediction is scored, not re-asked"
    )
    store.circuit_text = store.circuit_text + "\n"
    assert store.prediction_pending(), "an edited circuit is a new run: predict again"


def test_the_job_keeps_the_circuits_named_registers(index: ProvenanceIndex) -> None:
    """A program measuring into named classical registers crosses the job boundary whole: the record's circuit turns back
    into the circuit the text parses to, registers included, and back into the same record."""
    session = Session(Store(), index)
    session.edit_circuit(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg a[1];\ncreg b[1];\nh q[0];\ncx q[0],q[1];\n'
        "measure q[1] -> a[0];\nmeasure q[0] -> b[0];\n"
    )
    circuit = session.parse_circuit()
    assert dict(circuit.registers) == {"a": (1,), "b": (0,)}
    job = session.build_job()
    assert job.circuit.to_core() == circuit
    assert CircuitRecord.from_core(job.circuit.to_core()) == job.circuit


def test_the_readout_picked_for_the_full_simulation_reaches_the_shots(
    index: ProvenanceIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The readout picked on the full simulation is the job's, and a shot of that job opens its photon counts; the channel
    replay reads its shots out through the calibration table's errors, so its jobs read out fast whatever was picked."""
    session = Session(Store(), index)
    store = session.store
    store.shots = 20
    submitted: list[tuple[str, JobSpec]] = []

    def submit(request: str, **payload: Any) -> Ticket:
        submitted.append((request, payload["job"]))
        return Ticket(f"t{len(submitted)}", request)

    monkeypatch.setattr(session.worker, "submit", submit)
    store.engine, store.readout = FULL, "full"
    store.engine = REPLAY
    session.submit_run()
    store.engine = FULL
    session.submit_run()
    assert [(request, job.readout) for request, job in submitted] == [("replay", "fast"), ("run_job", "full")]
    # the worker's path: build the device, complete the job with it, run
    job = submitted[-1][1]
    preset = job.device.build(check=False)
    record, _live = execute(complete_job(job, preset), preset)
    assert shot(record, 0).photon_counts is not None


def test_progress_rows_keep_moving_between_worker_events(index: ProvenanceIndex) -> None:
    session = Session(Store(), index)
    store = session.store
    t0 = 1000.0
    session._last_beat = t0
    assert not session.heartbeat(t0 + 5.0), "nothing runs: nothing to re-render"
    store.jobs = {"t1": JobStatus("t1", "replay")}
    tick = store.activity.tick
    assert not session.heartbeat(t0 + 0.5) and store.activity.tick == tick
    assert session.heartbeat(t0 + 1.0) and store.activity.tick == tick + 1
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
