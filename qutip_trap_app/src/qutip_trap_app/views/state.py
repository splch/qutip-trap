"""The application state: records, jobs in flight, the learner's settings, and the worker bridge (DESIGN.md Sections 3, 4).

An observable dataclass (Flet re-renders every component that read it through ``use_state``). Records are immutable and
replaced whole; the worker's events are applied by :meth:`Session.apply_events`, which the shell polls from an asyncio task
(``page.run_task``) so that no solver ever runs on the UI loop (Section 14.6). The learner's settings and mastery log stay
on the learner's device through Flet's ``SharedPreferences`` (browser storage in the served mode, a preferences file in
the desktop window): the spacing rule of DESIGN.md Section 3 needs them to outlive the session, and nothing is uploaded.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import flet as ft

from qutip_trap_app.core import Circuit, Operation, SolverOptions
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import JobSpec, Record, job_for_preset
from qutip_trap_app.verify import VerifyReport
from qutip_trap_app.viewmodel.learn import (
    DEFAULT_RETENTION_DAYS,
    Attempt,
    Depth,
    MasteryLog,
    PriorKnowledge,
    plan_for,
)
from qutip_trap_app.workers import Event, SimulationWorker

Engine = Literal["replay", "full"]

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
)
"""The worked example (DESIGN.md Section 3): the two-ion Bell state."""

FAST_OPTIONS = SolverOptions(branch_weight_min=1e-3)
"""The options every job the app submits uses by default (the Section 9.6 fixture rule's fast setting)."""

LEARNER_KEY = "qutip_trap_app.learner.v1"
"""The SharedPreferences key under which the learner's settings and mastery log are kept on the device."""

KNOWLEDGE_VALUES: tuple[PriorKnowledge, ...] = ("newcomer", "circuits", "physicist", "unknown")
DEPTH_VALUES: tuple[Depth, ...] = ("sentence", "picture", "equation")


@dataclass
class JobStatus:
    ticket: str
    request: str
    stage: str = "queued"
    fraction: float | None = None
    message: str = ""
    started: float = field(default_factory=time.monotonic)
    error: str | None = None
    done: bool = False
    job: JobSpec | None = None
    engine: Engine = "full"

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started


@dataclass
class Learner:
    knowledge: PriorKnowledge = "unknown"
    retention_days: float = DEFAULT_RETENTION_DAYS
    asked: bool = False
    """Whether the first-launch prior-knowledge question was answered or dismissed."""
    depth_override: Depth | None = None
    explain_open: bool | None = None
    log: MasteryLog = field(default_factory=MasteryLog)

    def plan(self, level: int) -> Any:
        return plan_for(self.knowledge, level)

    def depth(self, level: int) -> Depth:
        return self.depth_override or self.plan(level).explain_depth

    def explain_is_open(self, level: int) -> bool:
        return self.plan(level).explain_open if self.explain_open is None else self.explain_open


def learner_document(learner: Learner) -> dict[str, Any]:
    """The learner as a JSON-ready document: what is kept on the device under :data:`LEARNER_KEY`."""
    return {
        "knowledge": learner.knowledge,
        "retention_days": float(learner.retention_days),
        "asked": bool(learner.asked),
        "depth_override": learner.depth_override,
        "explain_open": learner.explain_open,
        "attempts": [dataclasses.asdict(a) for a in learner.log.attempts],
    }


def learner_from_document(doc: Mapping[str, Any]) -> Learner:
    """The inverse of :func:`learner_document`. A key that an older document lacks takes the fresh learner's value; a
    value outside its vocabulary raises, so a corrupt document is reported rather than half-loaded."""
    knowledge = doc.get("knowledge", "unknown")
    if knowledge not in KNOWLEDGE_VALUES:
        raise ValueError(f"unknown prior knowledge {knowledge!r}; expected one of {KNOWLEDGE_VALUES}")
    depth = doc.get("depth_override")
    if depth is not None and depth not in DEPTH_VALUES:
        raise ValueError(f"unknown explanation depth {depth!r}; expected one of {DEPTH_VALUES} or null")
    retention = float(doc.get("retention_days", DEFAULT_RETENTION_DAYS))
    if not retention > 0.0:
        raise ValueError(f"the retention target must be positive, not {retention}")
    explain_open = doc.get("explain_open")
    log = MasteryLog(retention_days=retention)
    for a in doc.get("attempts", ()):
        correct = a["correct"]
        score = a.get("score")
        log.record(
            Attempt(
                str(a["concept_id"]),
                str(a["prompt_id"]),
                float(a["t_days"]),
                None if correct is None else bool(correct),
                bool(a["unaided"]),
                None if score is None else float(score),
            )
        )
    return Learner(
        knowledge=cast(PriorKnowledge, knowledge),
        retention_days=retention,
        asked=bool(doc.get("asked", False)),
        depth_override=cast("Depth | None", depth),
        explain_open=None if explain_open is None else bool(explain_open),
        log=log,
    )


@ft.observable
@dataclass
class Store:
    """Everything the screens read. Assigning a field notifies the components that read it."""

    records: dict[str, Record] = field(default_factory=dict)
    current: str | None = None
    jobs: dict[str, JobStatus] = field(default_factory=dict)
    verify_reports: dict[str, VerifyReport] = field(default_factory=dict)
    zoom_cache: dict[str, Any] = field(default_factory=dict)
    learner: Learner = field(default_factory=Learner)
    learner_loaded: bool = False
    """True once the saved learner was read (or found absent), so the first-launch question is asked once per device."""
    circuit_text: str = BELL_QASM
    circuit_format: Literal["openqasm2", "ionq_json"] = "openqasm2"
    shots: int = 200
    engine: Engine = "replay"
    prediction: str | None = None
    """The learner's histogram pick for the NEXT run (a sketch id), "skipped" when declined, None when not yet made."""
    scored_prediction: str | None = None
    """The pick that applied to the current record, scored beside its histogram (DESIGN.md Section 3)."""
    last_run_text: str | None = None
    """The circuit text of the last submitted run; a prediction is asked again once the text differs from it."""
    numerics_open: bool | None = None
    """The learner's own toggle of the numerics strip; None means the level plan's default."""
    selected_bar: str | None = None
    selected_shot: int | None = None
    error: str = ""
    worker_alive: bool = False
    tick: int = 0
    """Bumped by the poll loop when any job progressed (and once a second while one runs), so progress rows re-render."""

    def record(self) -> Record | None:
        return self.records.get(self.current) if self.current else None

    def running(self) -> list[JobStatus]:
        return [j for j in self.jobs.values() if not j.done]

    def prediction_pending(self) -> bool:
        """Whether the next Run is a run the learner has not predicted: nothing submitted yet, or the circuit text changed
        since the last submission. Predict-then-reveal then happens before every reveal, not only the first."""
        return self.circuit_text != self.last_run_text


class Session:
    """The worker, the polling loop and the learner's persistence behind a Store (not observable: the store is)."""

    def __init__(
        self, store: Store, provenance: ProvenanceIndex, page: Any = None, preferences: Any = None
    ) -> None:
        self.store = store
        self.provenance = provenance
        self.worker = SimulationWorker()
        self.page = page
        self.preferences = preferences
        """A ``flet.SharedPreferences`` service; None in tests and headless use, where the learner lives in memory only."""
        self._last_beat = time.monotonic()

    def start(self) -> None:
        self.worker.start()
        self.store.worker_alive = self.worker.alive

    def stop(self) -> None:
        self.worker.stop()
        self.store.worker_alive = False

    # -- the learner (DESIGN.md Sections 1 and 3) --

    def set_learner(self, **changes: Any) -> None:
        """Replace the learner so the store notifies its readers, keep the log's retention target in step, and save."""
        learner = dataclasses.replace(self.store.learner, **changes)
        learner.log.retention_days = learner.retention_days
        self.store.learner = learner
        self.persist_learner()

    def record_attempt(self, attempt: Attempt) -> None:
        """Log one prompt attempt and save. A skip is an attempt with ``correct=None``: it counts as an exposure, so the
        prompt comes due in the review tray, and never as wrong."""
        self.store.learner.log.record(attempt)
        self.set_learner()

    def persist_learner(self) -> None:
        if self.preferences is None or self.page is None:
            return
        self.page.run_task(self._save_learner, json.dumps(learner_document(self.store.learner)))

    async def _save_learner(self, document: str) -> None:
        try:
            await self.preferences.set(LEARNER_KEY, document)
        except (
            Exception
        ) as exc:  # the settings stay in memory for this session; the failure is shown, not swallowed
            self.store.error = f"the learner settings could not be saved on this device: {exc}"

    async def restore_learner(self, timeout_s: float = 3.0) -> None:
        """Read the saved learner at start-up. The first-launch question waits for ``learner_loaded``, so it is asked
        once per device rather than once per launch; the read is bounded so a slow or not-yet-ready storage service can
        never leave the question permanently gated off (a new device is then asked, and the settings still save)."""
        try:
            if self.preferences is not None:
                raw = await asyncio.wait_for(self.preferences.get(LEARNER_KEY), timeout=timeout_s)
                if isinstance(raw, str) and raw:
                    self.store.learner = learner_from_document(json.loads(raw))
        except TimeoutError:
            self.store.error = (
                "the saved settings did not load in time; starting fresh (new settings will still be saved)"
            )
        except (
            Exception
        ) as exc:  # a malformed or unreadable document: the defaults are used and the reason is shown
            self.store.error = (
                f"the saved learner settings could not be read, so the defaults are used: {exc}"
            )
        finally:
            self.store.learner_loaded = True

    # -- submitting work --

    def parse_circuit(self) -> Circuit:
        from qutip_trap_app.core import load_ionq_json, load_openqasm2

        text = self.store.circuit_text
        if self.store.circuit_format == "ionq_json":
            return load_ionq_json(json.loads(text))
        return load_openqasm2(text)

    def build_job(self, *, n_ions: int | None = None, seed: int = 0) -> JobSpec:
        circuit = self.parse_circuit()
        n = n_ions if n_ions is not None else max(2, circuit.n_qubits)
        job, _preset = job_for_preset(
            "yb171_chain",
            n,
            circuit,
            int(self.store.shots),
            seed=seed,
            options=FAST_OPTIONS,
            detection_records=500,
        )
        return job

    def submit_run(self) -> JobStatus | None:
        try:
            job = self.build_job()
        except Exception as exc:  # a syntax error in the circuit text: shown beside the editor, never a crash
            self.store.error = f"the circuit could not be read: {exc}"
            return None
        self.store.error = ""
        self.store.last_run_text = self.store.circuit_text
        request = "replay" if self.store.engine == "replay" else "run_job"
        ticket = self.worker.submit(request, job=job)
        status = JobStatus(ticket.id, request, job=job, engine=self.store.engine)
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_verify(self, key: str, shots: int | None = None) -> JobStatus:
        record = self.store.records[key]
        ticket = self.worker.submit("verify", key=key, record=record, shots=shots)
        status = JobStatus(ticket.id, "verify", job=record.job)
        status.message = key
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_zoom(self, key: str, step: int, n_store: int = 201) -> JobStatus:
        ticket = self.worker.submit("zoom", key=key, step=step, n_store=n_store)
        status = JobStatus(ticket.id, "zoom")
        status.message = f"{key}:{step}"
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def cancel_all(self) -> None:
        """Hard cancel of every running job (DESIGN.md Section 4: over ten seconds means progress and cancel). The worker
        process is terminated and restarted, which loses its live handles and channel library but not the records the
        screens hold; the jobs are marked cancelled rather than failed."""
        self.worker.cancel()
        jobs = dict(self.store.jobs)
        for status in jobs.values():
            if not status.done:
                status.done, status.stage, status.error = True, "cancelled", "cancelled by the user"
        self.store.jobs = jobs
        self.store.worker_alive = self.worker.alive
        self.store.tick = self.store.tick + 1

    # -- applying events --

    def apply_events(self, events: list[Event]) -> None:
        if not events:
            return
        jobs = dict(self.store.jobs)
        for ev in events:
            status = jobs.get(ev.ticket)
            if status is None:
                continue
            if ev.kind == "progress":
                status.stage, status.fraction, status.message = ev.stage, ev.fraction, ev.message
            elif ev.kind == "error":
                status.error, status.done, status.stage = ev.message, True, "failed"
                self.store.error = ev.message.strip().splitlines()[-1] if ev.message else "the worker failed"
            else:
                status.done, status.stage, status.fraction = True, "done", 1.0
                self._apply_result(status, ev.payload)
        self.store.jobs = jobs
        self.store.tick = self.store.tick + 1

    def _apply_result(self, status: JobStatus, payload: Any) -> None:
        if status.request in ("run_job", "replay") and isinstance(payload, Record):
            key = payload.key()
            self.store.records = {**self.store.records, key: payload}
            self.store.current = key
            self.store.selected_bar = None
            self.store.selected_shot = None
            # the pick made before this run is scored beside its histogram; the next run gets its own prompt
            self.store.scored_prediction = self.store.prediction
            self.store.prediction = None
            if self.page is not None:
                self.page.navigate(f"/job/{key}")
        elif status.request == "verify" and isinstance(payload, dict):
            report: VerifyReport = payload["report"]
            key = status.message
            self.store.verify_reports = {**self.store.verify_reports, key: report}
            deep = payload.get("deep")
            if isinstance(deep, Record):
                self.store.records = {**self.store.records, deep.key(): deep}
        elif status.request == "zoom" and isinstance(payload, dict):
            key, _step = status.message.split(":")
            record = self.store.records.get(key)
            if record is not None:
                merged = record.with_boundaries(payload["boundaries"]).with_zoom(payload["zoom"])
                self.store.records = {**self.store.records, key: merged}

    def heartbeat(self, now: float, period_s: float = 1.0) -> bool:
        """Re-render the progress rows once per ``period_s`` while a job runs, so the elapsed time keeps moving between
        the worker's events (a long solver step emits none). Returns whether a re-render was requested."""
        if not self.store.running() or now - self._last_beat < period_s:
            return False
        self._last_beat = now
        self.store.tick = self.store.tick + 1
        return True

    async def poll_forever(self, interval_s: float = 0.2) -> None:
        while True:
            try:
                self.apply_events(self.worker.poll())
                self.heartbeat(time.monotonic())
            except (
                Exception
            ) as exc:  # the loop must survive a worker hiccup; the error is shown, not swallowed
                self.store.error = f"worker: {exc}"
            alive = self.worker.alive
            if alive != self.store.worker_alive:
                self.store.worker_alive = alive
            await asyncio.sleep(interval_s)


def bell_circuit() -> Circuit:
    return Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))


__all__ = [
    "BELL_QASM",
    "DEPTH_VALUES",
    "FAST_OPTIONS",
    "KNOWLEDGE_VALUES",
    "LEARNER_KEY",
    "Engine",
    "JobStatus",
    "Learner",
    "Session",
    "Store",
    "bell_circuit",
    "learner_document",
    "learner_from_document",
]
