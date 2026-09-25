"""The application state: records, jobs in flight, the learner's settings, and the worker bridge (DESIGN.md Sections 3, 4).

An observable dataclass (Flet re-renders every component that read it through ``use_state``). Records are immutable and
replaced whole; the worker's events are applied by :meth:`Session.apply_events`, which the shell polls from an asyncio task
so that no solver ever runs on the UI loop. The learner's settings and mastery log stay on the learner's device through
Flet's ``SharedPreferences`` (browser storage in the served mode, a preferences file in the desktop window).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, assert_never, get_args

import flet as ft

from qutip_trap_app.core import Circuit, SolverOptions
from qutip_trap_app.device_layer import DeviceLayer
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import ConvergenceRecord, DeviceRef, JobSpec, Record, TableRecord, job_for_preset
from qutip_trap_app.requests import GateRequest
from qutip_trap_app.resim import TruncationCheck
from qutip_trap_app.verify import VerifyReport
from qutip_trap_app.viewmodel.builder import CircuitFormat, parse_circuit_text
from qutip_trap_app.viewmodel.learn import (
    DEFAULT_RETENTION_DAYS,
    DEPTHS,
    Attempt,
    Depth,
    LevelPlan,
    MasteryLog,
    PriorKnowledge,
    plan_for,
)
from qutip_trap_app.viewmodel.presets import PRESETS, PresetResult, PresetSpec
from qutip_trap_app.workers import Event, SimulationWorker

Engine = Literal["replay", "full"]
ThemeChoice = Literal["system", "light", "dark"]
Request = Literal[
    "run_job",
    "replay",
    "request_run",
    "verify",
    "zoom",
    "tomography",
    "recheck",
    "preset",
    "derive",
    "recalibrate",
]
"""The worker requests the screens submit."""

REQUEST_LABELS: dict[Request, str] = {
    "replay": "channel replay",
    "run_job": "full simulation",
    "request_run": "the requested run",
    "verify": "verify deeper",
    "zoom": "re-simulation",
    "tomography": "process tomography",
    "recheck": "convergence re-check",
    "preset": "published experiment",
    "derive": "deriving the device",
    "recalibrate": "recalibration",
}

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
)
"""The worked example: the two-ion Bell state."""

DEVICE_PRESET = "yb171_chain"
"""The public preset every job runs on (the Level 4 knobs edit it)."""

FAST_OPTIONS = SolverOptions(branch_weight_min=1e-3)
"""The options every job the app submits uses (the Section 9.6 fixture rule's fast setting)."""

UNDO_DEPTH = 30
"""How many circuit edits the builder's Undo can take back."""

MAX_SHOTS = 100_000
"""The most shots one run takes: both engines read the register out shot by shot in Python, so a run of this many takes
tens of seconds, and an unbounded count typed by hand would run for hours or exhaust memory."""

LEARNER_KEY = "qutip_trap_app.learner.v1"
"""The SharedPreferences key under which the learner's settings and mastery log are kept on the device."""


@dataclass
class JobStatus:
    ticket: str
    request: Request
    stage: str = "queued"
    fraction: float | None = None
    message: str = ""
    started: float = field(default_factory=time.monotonic)
    error: str | None = None
    done: bool = False
    job: JobSpec | None = None
    target: dict[str, Any] = field(default_factory=dict)
    """What the request was about (record key, step, sample, branch, device cache key), read back when its result lands;
    ``message`` is display text that every progress event overwrites."""

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
    theme: ThemeChoice = "system"
    """The colour scheme: the platform's, or light or dark regardless of it."""
    log: MasteryLog = field(default_factory=MasteryLog)

    def plan(self, level: int) -> LevelPlan:
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
        "theme": learner.theme,
        "attempts": [dataclasses.asdict(a) for a in learner.log.attempts],
    }


def learner_from_document(doc: Mapping[str, Any]) -> Learner:
    """The inverse of :func:`learner_document`. A missing key or a value outside its vocabulary raises, so a corrupt
    document is reported rather than half-loaded."""
    knowledge = doc["knowledge"]
    if knowledge not in get_args(PriorKnowledge):
        raise ValueError(f"unknown prior knowledge {knowledge!r}; expected one of {get_args(PriorKnowledge)}")
    depth = doc["depth_override"]
    if depth is not None and depth not in DEPTHS:
        raise ValueError(f"unknown explanation depth {depth!r}; expected one of {DEPTHS} or null")
    theme_choice = doc["theme"]
    if theme_choice not in get_args(ThemeChoice):
        raise ValueError(f"unknown theme {theme_choice!r}; expected one of {get_args(ThemeChoice)}")
    retention = float(doc["retention_days"])
    if not retention > 0.0:
        raise ValueError(f"the retention target must be positive, not {retention}")
    log = MasteryLog(retention_days=retention)
    for a in doc["attempts"]:
        correct = a["correct"]
        log.record(
            Attempt(
                str(a["concept_id"]),
                str(a["prompt_id"]),
                float(a["t_days"]),
                None if correct is None else bool(correct),
                bool(a["unaided"]),
            )
        )
    explain_open = doc["explain_open"]
    return Learner(
        knowledge=knowledge,
        retention_days=retention,
        asked=bool(doc["asked"]),
        depth_override=depth,
        explain_open=None if explain_open is None else bool(explain_open),
        theme=theme_choice,
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
    learner: Learner = field(default_factory=Learner)
    learner_loaded: bool = False
    """True once the saved learner was read (or found absent), so the first-launch question is asked once per device."""
    circuit_text: str = BELL_QASM
    circuit_format: CircuitFormat = "openqasm2"
    circuit_undo: tuple[tuple[str, CircuitFormat], ...] = ()
    """The circuit texts (with their formats) before each edit made in the builder, oldest first."""
    shots: int = 200
    engine: Engine = "replay"
    prediction: str | None = None
    """The learner's histogram pick for the next run (a sketch id), "skipped" when declined, None when not yet made."""
    scored_prediction: str | None = None
    """The pick that applied to the current record, scored beside its histogram."""
    last_run_text: str | None = None
    """The circuit text of the last submitted run; a prediction is asked again once the text differs from it."""
    numerics_open: bool | None = None
    """The learner's own toggle of the numerics strip; None means the level plan's default."""
    selected_bar: str | None = None
    selected_shot: int | None = None
    error: str = ""
    tick: int = 0
    """Bumped by the poll loop when any job progressed (and once a second while one runs), so progress rows re-render."""
    # the device model: the Level 4 knobs over the preset, the derived layers and the recalibrated tables
    preset_kwargs: dict[str, Any] = field(default_factory=dict)
    device_overrides: dict[str, float] = field(default_factory=dict)
    """The Level 4 knobs as set (``knobs`` ids); empty is the preset as published."""
    layers: dict[str, DeviceLayer] = field(default_factory=dict)
    """Derived device layers per ``DeviceRef.cache_key()``."""
    tables: dict[str, TableRecord] = field(default_factory=dict)
    """Recalibrated tables per device hash."""
    device_page: str = "hamiltonian"
    # Level 3 selections
    branch: int = 0
    """The initial-mixture branch Level 3 shows (the sample comes from the route)."""
    closure_predictions: dict[str, str] = field(default_factory=dict)
    """Per ``key:step``, the learner's closure pick before the loops were revealed ("skipped" when declined)."""
    rechecks: dict[str, tuple[ConvergenceRecord, TruncationCheck]] = field(default_factory=dict)
    """Per ``key:step:sample:branch``, the (tolerance, truncation) re-check pair a Level 3 request produced."""
    hamiltonian_target: tuple[str, int, int, int] | None = None
    """(record key, step, sample, branch) the Hamiltonian page shows; None = the current record's first entangling step."""
    selected_channel: str | None = None
    """A collapse channel opened from a jump marker on Level 3 (highlighted on the Hamiltonian page)."""
    selected_term: int | None = None
    # requests, presets, the explain drawer's selection and the Learn activities
    preset_results: dict[str, PresetResult] = field(default_factory=dict)
    active_preset: str | None = None
    """The circuit preset the current run was made from (its reference numbers show beside the histogram), if any."""
    last_request: GateRequest | None = None
    """The most recent request made at Level 1 or 2 (accepted or refused), shown where it was made."""
    explain_concept: dict[int, str] = field(default_factory=dict)
    """Per level, the concept the explain drawer shows; absent = the level's first."""
    spec_section: str | None = None
    """The Part II section the drawer's Specification tile shows, when a chip or a why-button chose one."""
    details_open: dict[str, bool] = field(default_factory=dict)
    """Per Details tile id, whether the learner opened it (absent = the plan's default)."""
    revealed_stops: set[int] = field(default_factory=set)
    """Stops of the faded GHZ exercise whose annotation the learner asked for."""
    drill_answers: dict[str, str] = field(default_factory=dict)
    """Per drill id, the answer given."""

    def record(self) -> Record | None:
        return self.records.get(self.current) if self.current else None

    def running(self) -> list[JobStatus]:
        return [j for j in self.jobs.values() if not j.done]

    def running_of(self, request: Request, **target: Any) -> JobStatus | None:
        """The running job of one request kind whose target carries the given items, if any."""
        return next(
            (
                j
                for j in self.jobs.values()
                if not j.done
                and j.request == request
                and all(j.target.get(k) == v for k, v in target.items())
            ),
            None,
        )

    def device_ref(self, n_ions: int | None = None) -> DeviceRef:
        """The current device as a reference: the preset, its arguments, the overrides, and the hash when a derived layer
        already knows it (the UI never builds a device; the worker does)."""
        rec = self.record()
        n = n_ions if n_ions is not None else (rec.device_card.n_ions if rec is not None else 2)
        ref = DeviceRef("", DEVICE_PRESET, int(n), dict(self.preset_kwargs), dict(self.device_overrides))
        layer = self.layers.get(ref.cache_key())
        return ref if layer is None else dataclasses.replace(ref, hash=layer.device_hash)

    def layer(self) -> DeviceLayer | None:
        return self.layers.get(self.device_ref().cache_key())

    def table_for(self, device_hash: str) -> TableRecord | None:
        """The calibration table of a device hash: a recalibrated one, else the current record's when it was made on that
        device."""
        t = self.tables.get(device_hash)
        if t is not None:
            return t
        rec = self.record()
        return rec.table if rec is not None and rec.device_hash == device_hash else None

    def prediction_pending(self) -> bool:
        """Whether the next Run is one the learner has not predicted: nothing submitted yet, or the circuit text changed
        since the last submission, so predict-then-reveal happens before every reveal."""
        return self.circuit_text != self.last_run_text


RESTORE_TIMEOUT_S = 8.0
HEARTBEAT_S = 1.0
POLL_S = 0.2


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
        # a clicked provenance chip opens the drawer's Specification tile at its section
        provenance.on_open_section = self.open_specification

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    # ---- the learner ----

    def set_learner(self, **changes: Any) -> None:
        """Replace the learner so the store notifies its readers, keep the log's retention target in step, and save."""
        learner = dataclasses.replace(self.store.learner, **changes)
        learner.log.retention_days = learner.retention_days
        self.store.learner = learner
        if self.preferences is not None and self.page is not None:
            self.page.run_task(self._save_learner, json.dumps(learner_document(learner)))

    def record_attempt(self, attempt: Attempt) -> None:
        """Log one prompt attempt and save. A skip (``correct=None``) counts as an exposure, so the prompt comes due in the
        review tray, and never as wrong."""
        self.store.learner.log.record(attempt)
        self.set_learner()

    async def _save_learner(self, document: str) -> None:
        try:
            await self.preferences.set(LEARNER_KEY, document)
        except (
            Exception
        ) as exc:  # the settings stay in memory for this session; the failure is shown, not swallowed
            self.store.error = f"the learner settings could not be saved on this device: {exc}"

    async def restore_learner(self) -> None:
        """Read the saved learner at start-up. The first-launch question waits for ``learner_loaded``, so it is asked once
        per device; the read is bounded so a slow storage service never leaves the question gated off."""
        try:
            if self.preferences is not None:
                raw = await asyncio.wait_for(self.preferences.get(LEARNER_KEY), timeout=RESTORE_TIMEOUT_S)
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

    # ---- the circuit (every change to the text goes through here) ----

    def parse_circuit(self) -> Circuit:
        return parse_circuit_text(self.store.circuit_text, self.store.circuit_format)

    def _set_circuit(self, text: str, fmt: CircuitFormat) -> None:
        self.store.circuit_text = text
        self.store.circuit_format = fmt
        self.store.active_preset = None

    def edit_circuit(self, text: str, fmt: CircuitFormat = "openqasm2") -> None:
        """Replace the circuit text (the builder's serialisation, or an imported program), remembering the old one for
        Undo. An edited circuit is nobody's preset, and the last error about the old text no longer applies."""
        self._remember_circuit()
        self._set_circuit(text, fmt)
        self.store.error = ""

    def undo_circuit(self) -> bool:
        """The circuit before the last edit, back in the builder; False when there is nothing to take back."""
        if not self.store.circuit_undo:
            return False
        *rest, (text, fmt) = self.store.circuit_undo
        self.store.circuit_undo = tuple(rest)
        self._set_circuit(text, fmt)
        self.store.error = ""
        return True

    def _remember_circuit(self) -> None:
        history = (*self.store.circuit_undo, (self.store.circuit_text, self.store.circuit_format))
        self.store.circuit_undo = history[-UNDO_DEPTH:]

    def load_circuit_preset(self, preset_id: str) -> PresetSpec:
        """Put a circuit preset (the Bell state, the three-ion GHZ) into the editor with its shots, device arguments and
        the full engine; the Results card then shows the reference numbers beside the run's."""
        spec = PRESETS[preset_id]
        if spec.kind != "circuit":
            raise ValueError(f"{preset_id} is an experiment preset; it runs from the Learn view")
        self._remember_circuit()
        self._set_circuit(spec.circuit_text, "openqasm2")
        self.store.shots = int(spec.shots)
        self.store.engine = "full"
        self.store.preset_kwargs = dict(spec.preset_kwargs)
        self.store.active_preset = preset_id
        self.store.prediction = None
        return spec

    def load_bell_example(self) -> None:
        """The worked example back in the builder, on the published preset."""
        self._remember_circuit()
        self._set_circuit(BELL_QASM, "openqasm2")
        self.store.preset_kwargs = {}

    # ---- submitting work ----

    def _track(self, status: JobStatus) -> JobStatus:
        self.store.jobs = {**self.store.jobs, status.ticket: status}
        return status

    def build_job(self, *, label: str = "") -> JobSpec:
        """The job for the current circuit on the current device (the Level 4 overrides included). Nothing is built here:
        the device hash is left for the worker to fill unless a derived layer already knows it."""
        circuit = self.parse_circuit()
        n = max(2, circuit.n_qubits)
        job, _preset = job_for_preset(
            DEVICE_PRESET,
            n,
            circuit,
            int(self.store.shots),
            seed=0,
            options=FAST_OPTIONS,
            detection_records=500,
            preset_kwargs=self.store.preset_kwargs,
            overrides=self.store.device_overrides,
            build=False,
            label=label,
        )
        ref = self.store.device_ref(n)
        return (
            dataclasses.replace(job, device=dataclasses.replace(job.device, hash=ref.hash))
            if ref.hash
            else job
        )

    def submit_run(self) -> JobStatus | None:
        try:
            job = self.build_job(
                label=f"preset: {self.store.active_preset}" if self.store.active_preset else ""
            )
        except Exception as exc:  # a syntax error in the circuit text: shown beside the editor, never a crash
            self.store.error = f"the circuit could not be read: {exc}"
            return None
        self.store.error = ""
        self.store.last_run_text = self.store.circuit_text
        request: Request = "replay" if self.store.engine == "replay" else "run_job"
        return self._track(JobStatus(self.worker.submit(request, job=job).id, request, job=job))

    def submit_preset(self, preset_id: str) -> JobStatus | None:
        """Run a published-experiment preset in the worker; a repeat while one runs is skipped."""
        spec = PRESETS[preset_id]
        if spec.kind != "experiment":
            raise ValueError(f"{preset_id} is a circuit preset: load it into the editor and Run")
        if self.store.running_of("preset", preset_id=preset_id) is not None:
            return None
        ticket = self.worker.submit("preset", preset_id=preset_id)
        return self._track(
            JobStatus(ticket.id, "preset", message=spec.title, target={"preset_id": preset_id})
        )

    def submit_request(self, request: GateRequest) -> JobStatus | None:
        """Run an accepted Level 1 or 2 request as its own job at the full engine, the requested gate's process matrix
        computed when the run finishes. A refused request is kept for display and not run."""
        self.store.last_request = request
        if request.job is None:
            return None
        self.store.error = ""
        self.store.active_preset = None
        ticket = self.worker.submit("request_run", job=request.job, step=request.step_index)
        target = {"gate_id": request.gate_id, "kind": request.kind}
        return self._track(
            JobStatus(ticket.id, "request_run", message=request.job.label, job=request.job, target=target)
        )

    def submit_verify(self, key: str, shots: int | None = None) -> JobStatus:
        record = self.store.records[key]
        ticket = self.worker.submit("verify", key=key, record=record, shots=shots)
        return self._track(JobStatus(ticket.id, "verify", message=key, job=record.job, target={"key": key}))

    def _submit_step(
        self, request: Request, key: str, step: int, sample: int, branch: int, **extra: Any
    ) -> JobStatus:
        """A request about one gate step of a record (zoom, tomography, re-check)."""
        target = {"key": key, "step": step, "sample": sample, "branch": branch}
        ticket = self.worker.submit(request, **target, **extra)
        return self._track(JobStatus(ticket.id, request, target=target))

    def submit_zoom(self, key: str, step: int, sample: int = 0, branch: int = 0) -> JobStatus:
        status = self._submit_step("zoom", key, step, sample, branch, n_store=201)
        status.message = f"{key}:{step}"
        return status

    def submit_tomography(self, key: str, step: int, sample: int = 0, branch: int = 0) -> JobStatus:
        return self._submit_step("tomography", key, step, sample, branch)

    def submit_recheck(self, key: str, step: int, sample: int = 0, branch: int = 0) -> JobStatus:
        return self._submit_step("recheck", key, step, sample, branch)

    def toggle_details(self, tile_id: str, open_: bool) -> None:
        self.store.details_open = {**self.store.details_open, tile_id: open_}

    def select_concept(self, level: int, concept_id: str) -> None:
        """Open the explain drawer at one concept of the level (one concept at a time)."""
        self.store.explain_concept = {**self.store.explain_concept, level: concept_id}
        self.set_learner(explain_open=True)

    def open_specification(self, section: str) -> None:
        """A chip or a why-button asked for a section's text: open the drawer with its Specification tile at that section."""
        self.store.spec_section = section
        self.set_learner(explain_open=True)

    # ---- the device model ----

    def set_knob(self, knob_id: str, value: float | None) -> None:
        """Set (or, with None, reset) one Level 4 knob and re-derive the analytic layer for the new device."""
        ov = dict(self.store.device_overrides)
        if value is None:
            ov.pop(knob_id, None)
        else:
            ov[knob_id] = float(value)
        self.store.device_overrides = ov
        self.submit_derive()

    def reset_knobs(self) -> None:
        self.store.device_overrides = {}
        self.submit_derive()

    def submit_derive(self) -> JobStatus | None:
        """Derive the device layer of the current device in the worker; a layer the store holds or a derive already running
        for it is not requested again."""
        ref = self.store.device_ref()
        ck = ref.cache_key()
        if ck in self.store.layers or self.store.running_of("derive", cache_key=ck) is not None:
            return None
        rec = self.store.record()
        ticket = self.worker.submit("derive", device=ref, table_record=rec.table if rec is not None else None)
        return self._track(JobStatus(ticket.id, "derive", target={"cache_key": ck}))

    def submit_recalibrate(self) -> JobStatus | None:
        """The user-initiated recalibration: the surrogate table for the current device (edited knobs included) and the
        current circuit's pairs, so that the next Run finds it in the worker's cache."""
        try:
            job = self.build_job()
        except Exception as exc:  # a syntax error in the circuit text: shown, never a crash
            self.store.error = f"the circuit could not be read: {exc}"
            return None
        ck = job.device.cache_key()
        if self.store.running_of("recalibrate", cache_key=ck) is not None:
            return None
        ticket = self.worker.submit("recalibrate", job=job)
        return self._track(JobStatus(ticket.id, "recalibrate", job=job, target={"cache_key": ck}))

    def cancel_all(self) -> None:
        """Hard cancel of every running job: the worker process is terminated and restarted, which loses its live handles
        and channel library but not the records the screens hold; the jobs are marked cancelled rather than failed."""
        self.worker.cancel()
        self._finish_running("cancelled", "cancelled by the user")

    def _finish_running(self, stage: str, error: str) -> None:
        jobs = dict(self.store.jobs)
        for status in jobs.values():
            if not status.done:
                status.done, status.stage, status.error = True, stage, error
        self.store.jobs = jobs
        self.store.tick = self.store.tick + 1

    # ---- applying events ----

    def apply_events(self, events: list[Event]) -> None:
        if not events:
            return
        jobs = dict(self.store.jobs)
        # published before the handlers run: a result handler may submit a follow-up job (a finished recalibration
        # re-derives the layer), and a write-back after the loop would drop that job's status
        self.store.jobs = jobs
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
        self.store.tick = self.store.tick + 1

    def _merge(self, status: JobStatus, change: Callable[[Record], Record]) -> None:
        """A step result lands on the record its ticket was about (the record may have left the session since)."""
        key = status.target["key"]
        record = self.store.records.get(key)
        if record is not None:
            self.store.records = {**self.store.records, key: change(record)}

    def _apply_result(self, status: JobStatus, payload: Any) -> None:
        store = self.store
        match status.request:
            case "run_job" | "replay" | "request_run":
                assert isinstance(payload, Record)
                key = payload.key()
                store.records = {**store.records, key: payload}
                store.current = key
                self._forget_selections()
                route = f"/job/{key}"
                if status.request == "request_run":
                    store.scored_prediction = None
                    route = f"/job/{key}/circuit/{status.target['gate_id']}"
                else:
                    # the last request was made against an earlier record (gate ids repeat across runs); the pick made
                    # before this run is scored beside its histogram, and the next run gets its own prompt
                    store.last_request = None
                    store.scored_prediction = store.prediction
                    store.prediction = None
                if self.page is not None:
                    self.page.navigate(route)
            case "preset":
                assert isinstance(payload, PresetResult)
                store.preset_results = {**store.preset_results, payload.preset_id: payload}
            case "verify":
                store.verify_reports = {**store.verify_reports, status.target["key"]: payload["report"]}
                deep = payload["deep"]
                if isinstance(deep, Record):
                    store.records = {**store.records, deep.key(): deep}
            case "zoom":
                ham = payload["hamiltonian"]

                def with_zoom(r: Record) -> Record:
                    merged = r.with_boundaries(payload["boundaries"]).with_zoom(payload["zoom"])
                    return merged if ham is None else merged.with_hamiltonian(ham)

                self._merge(status, with_zoom)
            case "tomography":
                self._merge(
                    status,
                    lambda r: r.with_boundaries(payload["boundaries"]).with_process_matrix(
                        payload["process_matrix"]
                    ),
                )
            case "recheck":

                def with_zooms(r: Record) -> Record:
                    merged = r.with_boundaries(payload["boundaries"])
                    for z in payload["zooms"]:
                        merged = merged.with_zoom(z)
                    return merged

                self._merge(status, with_zooms)
                t = status.target
                store.rechecks = {
                    **store.rechecks,
                    f"{t['key']}:{t['step']}:{t['sample']}:{t['branch']}": (
                        payload["tolerance"],
                        payload["truncation"],
                    ),
                }
            case "derive":
                assert isinstance(payload, DeviceLayer)
                store.layers = {**store.layers, status.target["cache_key"]: payload}
            case "recalibrate":
                store.tables = {**store.tables, str(payload["device_hash"]): payload["table"]}
                # the layer for this device compared itself with an older table: re-derive so the stale badge clears
                store.layers = {k: v for k, v in store.layers.items() if k != status.target["cache_key"]}
                self.submit_derive()
            case _:
                assert_never(status.request)

    def _forget_selections(self) -> None:
        """A new record is current: the selections that index into a record (a bar, a shot, a branch, a Hamiltonian term or
        channel, the step the Hamiltonian page shows) belonged to the previous one and are dropped. The per-record caches
        keyed by record key (re-checks, closure picks) stay."""
        self.store.selected_bar = None
        self.store.selected_shot = None
        self.store.branch = 0
        self.store.selected_term = None
        self.store.selected_channel = None
        self.store.hamiltonian_target = None

    def heartbeat(self, now: float) -> bool:
        """Re-render the progress rows once per ``HEARTBEAT_S`` while a job runs, so the elapsed time keeps moving between
        the worker's events. Returns whether a re-render was requested."""
        if not self.store.running() or now - self._last_beat < HEARTBEAT_S:
            return False
        self._last_beat = now
        self.store.tick = self.store.tick + 1
        return True

    def worker_died(self) -> bool:
        """The worker process is gone while jobs are still running (no error event will come): those jobs are marked failed
        with the reason and the worker is restarted for the next request. True when there was something to do."""
        if self.worker.alive or not self.worker.started or not self.store.running():
            return False
        self._finish_running("failed", "the worker process died")
        self.store.error = (
            "the worker process died; its live state is lost (run the job again to re-simulate)"
        )
        self.worker.start()
        return True

    async def poll_forever(self) -> None:
        while True:
            try:
                self.apply_events(self.worker.poll())
                self.heartbeat(time.monotonic())
                self.worker_died()
            except (
                Exception
            ) as exc:  # the loop must survive a worker hiccup; the error is shown, not swallowed
                self.store.error = f"worker: {exc}"
            await asyncio.sleep(POLL_S)
