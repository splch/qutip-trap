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
from qutip_trap_app.device_layer import DeviceLayer
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import DeviceRef, JobSpec, Record, TableRecord, job_for_preset
from qutip_trap_app.requests import GateRequest
from qutip_trap_app.verify import VerifyReport
from qutip_trap_app.viewmodel.builder import CircuitFormat, parse_circuit_text
from qutip_trap_app.viewmodel.learn import (
    DEFAULT_RETENTION_DAYS,
    Attempt,
    Depth,
    MasteryLog,
    PriorKnowledge,
    plan_for,
)
from qutip_trap_app.viewmodel.presets import PRESETS, PresetResult, PresetSpec
from qutip_trap_app.workers import Event, SimulationWorker

Engine = Literal["replay", "full"]
ThemeChoice = Literal["system", "light", "dark"]

BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
)
"""The worked example (DESIGN.md Section 3): the two-ion Bell state."""

FAST_OPTIONS = SolverOptions(branch_weight_min=1e-3)
"""The options every job the app submits uses by default (the Section 9.6 fixture rule's fast setting)."""

UNDO_DEPTH = 30
"""How many circuit edits the builder's Undo can take back."""

LEARNER_KEY = "qutip_trap_app.learner.v1"
"""The SharedPreferences key under which the learner's settings and mastery log are kept on the device."""

KNOWLEDGE_VALUES: tuple[PriorKnowledge, ...] = ("newcomer", "circuits", "physicist", "unknown")
DEPTH_VALUES: tuple[Depth, ...] = ("sentence", "picture", "equation")
THEME_VALUES: tuple[ThemeChoice, ...] = ("system", "light", "dark")


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
    target: dict[str, Any] = field(default_factory=dict)
    """What the request was about (record key, step, sample, branch, device cache key), read back when its result lands."""

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
    """The colour scheme the learner chose: the platform's, or light or dark regardless of it (DESIGN.md Section 4)."""
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
        "theme": learner.theme,
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
    theme_choice = doc.get("theme", "system")
    if theme_choice not in THEME_VALUES:
        raise ValueError(f"unknown theme {theme_choice!r}; expected one of {THEME_VALUES}")
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
        theme=cast(ThemeChoice, theme_choice),
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
    circuit_format: CircuitFormat = "openqasm2"
    circuit_undo: tuple[tuple[str, CircuitFormat], ...] = ()
    """The circuit texts (with their formats) before each edit made in the builder, oldest first (DESIGN.md R16)."""
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
    # ---- Level 4: the device model as the single source of truth (Section 14.4; M11.3) ----
    preset_name: str = "yb171_chain"
    preset_kwargs: dict[str, Any] = field(default_factory=dict)
    device_overrides: dict[str, float] = field(default_factory=dict)
    """The Level 4 knobs as set (``knobs`` ids); empty is the preset as published."""
    layers: dict[str, DeviceLayer] = field(default_factory=dict)
    """Derived device layers per ``DeviceRef.cache_key()``."""
    tables: dict[str, TableRecord] = field(default_factory=dict)
    """Recalibrated tables per device hash (the user-initiated job of Section 14.4)."""
    device_page: str = "hamiltonian"
    # ---- Level 3 selections ----
    branch: int = 0
    """The initial-mixture branch Level 3 shows (the sample comes from the route)."""
    closure_predictions: dict[str, str] = field(default_factory=dict)
    """Per ``key:step``, the learner's closure pick before the loops were revealed ("skipped" when declined)."""
    rechecks: dict[str, Any] = field(default_factory=dict)
    """Per ``key:step:sample:branch``, the (tolerance, truncation) re-check pair a Level 3 request produced."""
    hamiltonian_target: tuple[str, int, int, int] | None = None
    """(record key, step, sample, branch) the Hamiltonian page shows; None = the current record's first entangling step."""
    selected_channel: str | None = None
    """A collapse channel opened from a jump marker on Level 3 (highlighted on the Hamiltonian page)."""
    selected_term: int | None = None
    # ---- M11.4: requests, presets, the explain drawer's selection and the Learn activities ----
    preset_results: dict[str, PresetResult] = field(default_factory=dict)
    """Finished published-experiment presets by id (Section 14.5)."""
    active_preset: str | None = None
    """The circuit preset the current run was made from (its check-script numbers show beside the histogram), if any."""
    last_request: GateRequest | None = None
    """The most recent request made at Level 1 or 2 (accepted or refused), shown where it was made."""
    explain_concept: dict[int, str] = field(default_factory=dict)
    """Per level, the concept the explain drawer shows (DESIGN.md Section 10 R7); absent = the level's first."""
    spec_section: str | None = None
    """The Part II section the drawer's Specification tile shows, when a chip or a why-button chose one."""
    details_open: dict[str, bool] = field(default_factory=dict)
    """Per Details tile id, whether the learner opened it (None = the plan's default)."""
    revealed_stops: set[int] = field(default_factory=set)
    """Stops of the faded GHZ exercise whose annotation the learner asked for."""
    drill_answers: dict[str, str] = field(default_factory=dict)
    """Per drill id, the answer given (scored against the record, DESIGN.md Section 3)."""

    def record(self) -> Record | None:
        return self.records.get(self.current) if self.current else None

    def running(self) -> list[JobStatus]:
        return [j for j in self.jobs.values() if not j.done]

    def running_of(self, request: str, **target: Any) -> JobStatus | None:
        """The running job of one request kind whose target carries the given items, if any."""
        for j in self.jobs.values():
            if j.done or j.request != request:
                continue
            if all(j.target.get(k) == v for k, v in target.items()):
                return j
        return None

    def device_ref(self, n_ions: int | None = None) -> DeviceRef:
        """The current device as a reference: the preset, its arguments, the overrides, and the hash when a derived layer
        already knows it (the UI never builds a device; the worker does, Section 14.6)."""
        rec = self.record()
        n = n_ions if n_ions is not None else (rec.device_card.n_ions if rec is not None else 2)
        ref = DeviceRef("", self.preset_name, int(n), dict(self.preset_kwargs), dict(self.device_overrides))
        layer = self.layers.get(ref.cache_key())
        if layer is not None:
            ref = dataclasses.replace(ref, hash=layer.device_hash)
        return ref

    def layer(self, n_ions: int | None = None) -> DeviceLayer | None:
        return self.layers.get(self.device_ref(n_ions).cache_key())

    def table_for(self, device_hash: str) -> TableRecord | None:
        """The calibration table that belongs to a device hash: a recalibrated one, else the current record's when it
        was made on that device."""
        t = self.tables.get(device_hash)
        if t is not None:
            return t
        rec = self.record()
        if rec is not None and rec.device_hash == device_hash:
            return rec.table
        return None

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
        # a clicked provenance chip opens the drawer's Specification tile at its section (DESIGN.md Section 10 R8)
        provenance.on_open_section = self.open_specification

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

    async def restore_learner(self, timeout_s: float = 8.0) -> None:
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
        return parse_circuit_text(self.store.circuit_text, self.store.circuit_format)

    # -- editing the circuit (the builder of DESIGN.md R16: every change to the text goes through here) --

    def edit_circuit(self, text: str, fmt: CircuitFormat = "openqasm2") -> None:
        """Replace the circuit text (the builder's serialisation, or an imported program), remembering the old one for
        Undo. An edited circuit is nobody's preset, and the last error about the old text no longer applies."""
        self._remember_circuit()
        self.store.circuit_text = text
        self.store.circuit_format = fmt
        self.store.active_preset = None
        self.store.error = ""

    def undo_circuit(self) -> bool:
        """The circuit before the last edit, back in the builder; False when there is nothing to take back."""
        if not self.store.circuit_undo:
            return False
        *rest, (text, fmt) = self.store.circuit_undo
        self.store.circuit_undo = tuple(rest)
        self.store.circuit_text = text
        self.store.circuit_format = fmt
        self.store.active_preset = None
        self.store.error = ""
        return True

    def _remember_circuit(self) -> None:
        history = (*self.store.circuit_undo, (self.store.circuit_text, self.store.circuit_format))
        self.store.circuit_undo = history[-UNDO_DEPTH:]

    def build_job(self, *, n_ions: int | None = None, seed: int = 0, label: str = "") -> JobSpec:
        """The job for the current circuit on the current device (the Level 4 overrides included). Nothing is built here:
        the device hash is left for the worker to fill unless a derived layer already knows it."""
        circuit = self.parse_circuit()
        n = n_ions if n_ions is not None else max(2, circuit.n_qubits)
        job, _preset = job_for_preset(
            self.store.preset_name,
            n,
            circuit,
            int(self.store.shots),
            seed=seed,
            options=FAST_OPTIONS,
            detection_records=500,
            preset_kwargs=self.store.preset_kwargs,
            overrides=self.store.device_overrides,
            build=False,
            label=label,
        )
        ref = self.store.device_ref(n)
        if ref.hash:
            job = dataclasses.replace(job, device=dataclasses.replace(job.device, hash=ref.hash))
        return job

    def submit_run(self, *, label: str = "") -> JobStatus | None:
        try:
            job = self.build_job(
                label=label or (f"preset: {self.store.active_preset}" if self.store.active_preset else "")
            )
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

    def load_circuit_preset(self, preset_id: str) -> PresetSpec:
        """Put a circuit preset of Section 14.5 (the Bell state, the three-ion GHZ) into the editor: its circuit, shots,
        device arguments and the full engine; the Results card then shows the check script's numbers beside the run's."""
        spec = PRESETS[preset_id]
        if spec.kind != "circuit":
            raise ValueError(f"{preset_id} is an experiment preset; it runs from the Learn view")
        self._remember_circuit()
        self.store.circuit_text = spec.circuit_text
        self.store.circuit_format = "openqasm2"
        self.store.shots = int(spec.shots)
        self.store.engine = "full"
        self.store.preset_kwargs = dict(spec.preset_kwargs)
        self.store.active_preset = preset_id
        self.store.prediction = None
        return spec

    def load_bell_example(self) -> None:
        """The worked example back in the builder, on the published preset (no preset arguments, no active circuit preset)."""
        self._remember_circuit()
        self.store.circuit_text = BELL_QASM
        self.store.circuit_format = "openqasm2"
        self.store.preset_kwargs = {}
        self.store.active_preset = None

    def submit_preset(self, preset_id: str) -> JobStatus | None:
        """Run a published-experiment preset in the worker (Section 14.5); a repeat while one runs is skipped."""
        spec = PRESETS[preset_id]
        if spec.kind != "experiment":
            raise ValueError(f"{preset_id} is a circuit preset: load it into the editor and Run")
        if self.store.running_of("preset", preset_id=preset_id) is not None:
            return None
        ticket = self.worker.submit("preset", preset_id=preset_id)
        status = JobStatus(ticket.id, "preset", target={"preset_id": preset_id})
        status.message = spec.title
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_request(self, request: GateRequest) -> JobStatus | None:
        """Run an accepted request of Section 14.4 as its own job at the full engine, with the requested gate's process matrix
        computed when the run finishes (the worker's ``request_run``). A refused request is kept for display and not run."""
        self.store.last_request = request
        if request.job is None:
            return None
        self.store.error = ""
        self.store.active_preset = None
        ticket = self.worker.submit("request_run", job=request.job, step=request.step_index)
        status = JobStatus(
            ticket.id,
            "request_run",
            job=request.job,
            engine="full",
            target={"gate_id": request.gate_id, "kind": request.kind},
        )
        status.message = request.job.label
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def toggle_details(self, tile_id: str, open_: bool) -> None:
        self.store.details_open = {**self.store.details_open, tile_id: open_}

    def select_concept(self, level: int, concept_id: str | None) -> None:
        """Open the explain drawer at one concept of the level (DESIGN.md Section 10 R7: one concept at a time)."""
        chosen = dict(self.store.explain_concept)
        if concept_id is None:
            chosen.pop(level, None)
        else:
            chosen[level] = concept_id
        self.store.explain_concept = chosen
        self.set_learner(explain_open=True)

    def open_specification(self, section: str) -> None:
        """A chip or a why-button asked for a section's text: open the drawer with its Specification tile at that section."""
        self.store.spec_section = section
        self.set_learner(explain_open=True)

    def submit_verify(self, key: str, shots: int | None = None) -> JobStatus:
        record = self.store.records[key]
        ticket = self.worker.submit("verify", key=key, record=record, shots=shots)
        status = JobStatus(ticket.id, "verify", job=record.job)
        status.message = key
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_zoom(
        self, key: str, step: int, sample: int = 0, branch: int = 0, n_store: int = 201
    ) -> JobStatus:
        ticket = self.worker.submit("zoom", key=key, step=step, sample=sample, branch=branch, n_store=n_store)
        status = JobStatus(
            ticket.id, "zoom", target={"key": key, "step": step, "sample": sample, "branch": branch}
        )
        status.message = f"{key}:{step}"
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_fock_movie(
        self, key: str, step: int, sample: int = 0, branch: int = 0, n_frames: int = 8
    ) -> JobStatus:
        ticket = self.worker.submit(
            "fock_movie", key=key, step=step, sample=sample, branch=branch, n_frames=n_frames
        )
        status = JobStatus(
            ticket.id, "fock_movie", target={"key": key, "step": step, "sample": sample, "branch": branch}
        )
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_tomography(self, key: str, step: int, sample: int = 0, branch: int = 0) -> JobStatus:
        ticket = self.worker.submit("tomography", key=key, step=step, sample=sample, branch=branch)
        status = JobStatus(
            ticket.id, "tomography", target={"key": key, "step": step, "sample": sample, "branch": branch}
        )
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_recheck(self, key: str, step: int, sample: int = 0, branch: int = 0) -> JobStatus:
        ticket = self.worker.submit("recheck", key=key, step=step, sample=sample, branch=branch)
        status = JobStatus(
            ticket.id, "recheck", target={"key": key, "step": step, "sample": sample, "branch": branch}
        )
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    # -- the device model (Section 14.4) --

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

    def submit_derive(self, n_ions: int | None = None) -> JobStatus | None:
        """Derive the device layer of the current device (immediate tier in the worker); a repeat for a layer the store
        already holds is skipped, and so is a duplicate of a running request."""
        ref = self.store.device_ref(n_ions)
        ck = ref.cache_key()
        if ck in self.store.layers or self.store.running_of("derive", cache_key=ck) is not None:
            return None
        rec = self.store.record()
        table_record = rec.table if rec is not None else None
        ticket = self.worker.submit("derive", device=ref, table_record=table_record)
        status = JobStatus(ticket.id, "derive", target={"cache_key": ck})
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def submit_recalibrate(self) -> JobStatus | None:
        """The user-initiated recalibration of Section 14.4: the surrogate table for the current device (edited knobs
        included) and the current circuit's pairs, so that the next Run finds it in the worker's cache."""
        try:
            job = self.build_job()
        except Exception as exc:  # a syntax error in the circuit text: shown, never a crash
            self.store.error = f"the circuit could not be read: {exc}"
            return None
        ck = job.device.cache_key()
        if self.store.running_of("recalibrate", cache_key=ck) is not None:
            return None
        ticket = self.worker.submit("recalibrate", job=job)
        status = JobStatus(ticket.id, "recalibrate", job=job, target={"cache_key": ck})
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
        # published BEFORE the handlers run: a result handler may submit a follow-up job (a finished recalibration re-derives
        # the layer), and a write-back after the loop would silently drop that job's status from the store
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
        elif status.request == "request_run" and isinstance(payload, Record):
            key = payload.key()
            self.store.records = {**self.store.records, key: payload}
            self.store.current = key
            self.store.selected_bar = None
            self.store.selected_shot = None
            self.store.scored_prediction = None
            if self.page is not None:
                gate = str(status.target.get("gate_id", ""))
                self.page.navigate(f"/job/{key}/circuit/{gate}" if gate else f"/job/{key}")
        elif status.request == "preset" and isinstance(payload, PresetResult):
            self.store.preset_results = {**self.store.preset_results, payload.preset_id: payload}
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
                ham = payload.get("hamiltonian")
                if ham is not None:
                    merged = merged.with_hamiltonian(ham)
                self.store.records = {**self.store.records, key: merged}
        elif status.request == "fock_movie" and isinstance(payload, dict):
            key = str(status.target.get("key"))
            record = self.store.records.get(key)
            if record is not None:
                merged = record.with_boundaries(payload["boundaries"]).with_fock_movie(payload["movie"])
                self.store.records = {**self.store.records, key: merged}
        elif status.request == "tomography" and isinstance(payload, dict):
            key = str(status.target.get("key"))
            record = self.store.records.get(key)
            if record is not None:
                merged = record.with_boundaries(payload["boundaries"]).with_process_matrix(
                    payload["process_matrix"]
                )
                self.store.records = {**self.store.records, key: merged}
        elif status.request == "recheck" and isinstance(payload, dict):
            key = str(status.target.get("key"))
            record = self.store.records.get(key)
            if record is not None:
                merged = record.with_boundaries(payload["boundaries"])
                for z in payload["zooms"]:
                    merged = merged.with_zoom(z)
                self.store.records = {**self.store.records, key: merged}
            t = status.target
            self.store.rechecks = {
                **self.store.rechecks,
                f"{key}:{t.get('step')}:{t.get('sample')}:{t.get('branch')}": (
                    payload["tolerance"],
                    payload["truncation"],
                ),
            }
        elif status.request == "derive" and isinstance(payload, DeviceLayer):
            ck = str(status.target.get("cache_key"))
            self.store.layers = {**self.store.layers, ck: payload}
        elif status.request == "recalibrate" and isinstance(payload, dict):
            table: TableRecord = payload["table"]
            self.store.tables = {**self.store.tables, str(payload["device_hash"]): table}
            # the layer for this device compared itself with an older table: re-derive so the stale badge clears
            ck = str(status.target.get("cache_key"))
            layers = dict(self.store.layers)
            layers.pop(ck, None)
            self.store.layers = layers
            self.submit_derive()

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
    "PRESETS",
    "THEME_VALUES",
    "UNDO_DEPTH",
    "Engine",
    "JobStatus",
    "Learner",
    "Session",
    "Store",
    "ThemeChoice",
    "bell_circuit",
    "learner_document",
    "learner_from_document",
]
