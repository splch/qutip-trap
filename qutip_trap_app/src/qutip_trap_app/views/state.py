"""The application state: the observable ``Store`` the shell renders from (records are replaced whole, never mutated),
and the ``Session`` that submits work to the worker and applies its events."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import flet as ft

from qutip_trap_app import core
from qutip_trap_app.record import Job, Record
from qutip_trap_app.viewmodel.editor import BELL_QASM, MAX_SHOTS, PRESETS, parse_circuit
from qutip_trap_app.workers import Event, Resimulated, SimulationWorker

FAST = core.Numerics(truncation=core.Truncation(branch_weight_min=1e-3))
"""The numerics of every run the app submits: initial-mixture branches below 1e-3 are dropped and reported."""

THEME_NEXT = {
    ft.ThemeMode.SYSTEM: ft.ThemeMode.LIGHT,
    ft.ThemeMode.LIGHT: ft.ThemeMode.DARK,
    ft.ThemeMode.DARK: ft.ThemeMode.SYSTEM,
}


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
    target: dict[str, Any] = field(default_factory=dict)
    """What the request was about (record key, step, sample, branch), read back when its result lands."""

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started


@ft.observable
@dataclass
class Store:
    """Everything the screens read."""

    records: dict[str, Record] = field(default_factory=dict)
    current: str | None = None
    jobs: dict[str, JobStatus] = field(default_factory=dict)
    circuit_text: str = BELL_QASM
    shots: int = 200
    device_kwargs: dict[str, float] = field(default_factory=dict)
    """The example device's keyword arguments for the loaded preset."""
    selected_bar: str | None = None
    selected_shot: int | None = None
    branch: int = 0
    """The branch of the initial mixture Level 3 shows."""
    expanded: dict[str, bool] = field(default_factory=dict)
    """Which disclosures are open, by id."""
    error: str = ""
    width: float = 1200.0
    """The window's width: the lanes and drawings size themselves from it."""
    tick: int = 0
    """Bumped when a job progressed and once a second while one runs, so the progress rows re-render."""

    def record(self) -> Record | None:
        return self.records.get(self.current) if self.current else None

    def running(self) -> list[JobStatus]:
        return [j for j in self.jobs.values() if not j.done]

    def running_of(self, request: str, **target: Any) -> JobStatus | None:
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


class Session:
    """The worker, the polling loop and the actions behind a Store."""

    def __init__(self, store: Store, page: ft.Page | None = None) -> None:
        self.store = store
        self.page = page
        self.worker = SimulationWorker()
        self._last_beat = time.monotonic()

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def navigate(self, route: str) -> None:
        if self.page is not None:
            self.page.navigate(route)

    def toggle_theme(self) -> None:
        if self.page is not None:
            self.page.theme_mode = THEME_NEXT[self.page.theme_mode or ft.ThemeMode.SYSTEM]
            self.page.update()

    # -- the circuit and the run --

    def load_preset(self, name: str) -> None:
        preset = PRESETS[name]
        self.store.circuit_text = preset.text
        self.store.shots = preset.shots
        self.store.device_kwargs = dict(preset.device_kwargs)
        self.store.error = ""

    def set_shots(self, text: str) -> None:
        try:
            shots = int(text)
        except ValueError:
            self.store.error = "shots must be a positive whole number"
            return
        self.store.shots = min(max(1, shots), MAX_SHOTS)
        self.store.error = f"shots are capped at {MAX_SHOTS}" if shots > MAX_SHOTS else ""

    def build_job(self) -> Job:
        """The job for the editor's circuit on the example chain (at least two ions)."""
        circuit = parse_circuit(self.store.circuit_text)
        spec = core.RunSpec(circuit, int(self.store.shots), numerics=FAST, keep_final_state=True)
        return Job(spec, max(2, circuit.n_qubits), dict(self.store.device_kwargs))

    def submit_run(self) -> JobStatus | None:
        try:
            job = self.build_job()
        except (ValueError, KeyError, TypeError) as exc:  # the importers' errors on a malformed circuit text
            self.store.error = f"the circuit could not be read: {exc}"
            return None
        self.store.error = ""
        return self._submit("run", {}, job=job)

    def submit_resim(self, request: str, key: str, step: int, sample: int = 0, branch: int = 0) -> JobStatus:
        """A zoom, a tomography or a re-check of one step of a record, run in the worker."""
        target = {"key": key, "step": step, "sample": sample, "branch": branch}
        return self._submit(request, target, **target)

    def _submit(self, request: str, target: dict[str, Any], **payload: Any) -> JobStatus:
        ticket = self.worker.submit(request, **payload)
        status = JobStatus(ticket.id, request, target=target)
        self.store.jobs = {**self.store.jobs, ticket.id: status}
        return status

    def cancel_all(self) -> None:
        """Terminate the worker, which restarts empty, and mark every running job cancelled."""
        self.worker.cancel()
        for status in self.store.running():
            status.done, status.stage, status.error = True, "cancelled", "cancelled"
        self.store.jobs = dict(self.store.jobs)
        self.store.tick += 1

    # -- the worker's events --

    def apply_events(self, events: list[Event]) -> None:
        if not events:
            return
        for ev in events:
            status = self.store.jobs.get(ev.ticket)
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
        self.store.jobs = dict(self.store.jobs)
        self.store.tick += 1

    def _apply_result(self, status: JobStatus, payload: Any) -> None:
        if isinstance(payload, Record):
            self.store.records = {**self.store.records, payload.key: payload}
            self.store.current = payload.key
            # the selections index into a record: they belonged to the previous one
            self.store.selected_bar = None
            self.store.selected_shot = None
            self.store.branch = 0
            self.navigate(f"/job/{payload.key}")
        elif isinstance(payload, Resimulated):
            key = str(status.target.get("key"))
            record = self.store.records.get(key)
            if record is not None:
                self.store.records = {**self.store.records, key: record.with_cached(*payload.cached)}

    def heartbeat(self, now: float, period_s: float = 1.0) -> bool:
        """Re-render the progress rows once per ``period_s`` while a job runs, so the elapsed time keeps moving."""
        if not self.store.running() or now - self._last_beat < period_s:
            return False
        self._last_beat = now
        self.store.tick += 1
        return True

    def worker_died(self) -> bool:
        """Fail the running jobs of a worker that is gone (no error event will come) and start a new one."""
        if self.worker.alive or not self.worker.started:
            return False
        running = self.store.running()
        if not running:
            return False
        for status in running:
            status.done, status.stage, status.error = True, "failed", "the worker process died"
        self.store.jobs = dict(self.store.jobs)
        self.store.error = (
            "the worker process died and its live runs are lost: run the job again to re-simulate"
        )
        self.worker.start()
        self.store.tick += 1
        return True

    async def poll_forever(self, interval_s: float = 0.2) -> None:
        while True:
            try:
                self.apply_events(self.worker.poll())
                self.heartbeat(time.monotonic())
                self.worker_died()
            except (
                Exception
            ) as exc:  # the loop must outlive a worker hiccup: the error is shown, not swallowed
                self.store.error = f"worker: {exc}"
            await asyncio.sleep(interval_s)
