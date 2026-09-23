"""The simulation worker: one process that keeps every run's live state for re-simulation, runs the requests and streams
their progress and results back; under WebAssembly, where there are no processes, it runs in the page's thread."""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from qutip_trap_app import core, resim
from qutip_trap_app.record import Cached, Job, LiveRun, Record, execute

EventKind = Literal["progress", "result", "error"]
Progress = Callable[[str, float | None, str], None]

IN_PROCESS = sys.platform == "emscripten"
"""WebAssembly has no processes: requests run in the page's own thread."""


@dataclass(frozen=True)
class Ticket:
    id: str
    kind: str


@dataclass(frozen=True)
class Event:
    kind: EventKind
    ticket: str
    request: str
    stage: str = ""
    fraction: float | None = None
    message: str = ""
    payload: Any = None
    wall_time_s: float = 0.0


@dataclass(frozen=True)
class Resimulated:
    """What a re-simulation request adds to its record's cache."""

    cached: tuple[Cached, ...]


class WorkerError(RuntimeError):
    """A request failed in the worker; the message carries the worker's traceback."""


# ---- the worker process ---------------------------------------------------------------------------------------------------------


@dataclass
class _LiveState:
    records: dict[str, Record] = field(default_factory=dict)
    lives: dict[str, LiveRun] = field(default_factory=dict)


def _run_job(state: _LiveState, job: Job, progress: Progress) -> Record:
    progress("calibrating", None, "the closed-form calibration table, cached per device and seed")

    def forward(p: core.Progress) -> None:
        progress(f"running: {p.stage}", p.fraction, f"{p.stage} {p.done} of {p.total}")

    record, live = execute(job, forward)
    state.records[record.key] = record
    state.lives[record.key] = live
    return record


def _resimulate(state: _LiveState, request: str, payload: dict[str, Any], progress: Progress) -> Resimulated:
    key = str(payload["key"])
    found, live = state.records.get(key), state.lives.get(key)
    if found is None or live is None:
        raise WorkerError(f"no live run for record {key}: run the job again to re-simulate")
    at = (int(payload["step"]), int(payload.get("sample", 0)), int(payload.get("branch", 0)))
    record = found
    if request == "zoom":
        progress(
            "zooming", None, f"re-simulating step {at[0]} at {resim.DEFAULT_ZOOM_POINTS} points per segment"
        )
        record, _z, _cached = resim.zoom(record, live, *at)
        progress("building", 0.9, "the Hamiltonian terms and collapse operators of the step")
        record, _h = resim.hamiltonian_record(record, live, *at)
    elif request == "tomography":
        progress("tomography", None, f"the channel of step {at[0]} on every product input")
        record, _pm = resim.process_matrix(record, live, *at)
    else:
        progress("rechecking", 0.0, "tolerances tightened by ten")
        record, _tol = resim.tolerance_recheck(record, live, *at)
        progress("rechecking", 0.5, f"every cap raised by {resim.CAP_RAISE}, chained from the initial state")
        record, _cap = resim.truncation_recheck(record, live, *at)
    state.records[key] = record
    return Resimulated(tuple(v for k, v in record.cache.items() if found.cache.get(k) is not v))


def _handle(state: _LiveState, request: str, payload: dict[str, Any], progress: Progress) -> Any:
    if request == "ping":
        return "pong"
    if request == "run":
        return _run_job(state, payload["job"], progress)
    if request in ("zoom", "tomography", "recheck"):
        return _resimulate(state, request, payload, progress)
    raise WorkerError(f"unknown request {request!r}")


def _run(state: _LiveState, results: Any, item: tuple[str, str, dict[str, Any]]) -> None:
    """One request: its progress events, then its result or its error, on ``results``."""
    ticket, request, payload = item
    t0 = time.perf_counter()

    def progress(stage: str, fraction: float | None, message: str) -> None:
        results.put(
            Event(
                "progress",
                ticket,
                request,
                stage=stage,
                fraction=fraction,
                message=message,
                wall_time_s=time.perf_counter() - t0,
            )
        )

    try:
        value = _handle(state, request, payload, progress)
        results.put(Event("result", ticket, request, payload=value, wall_time_s=time.perf_counter() - t0))
    except Exception:  # every failure goes back to the UI as an error event carrying its traceback
        results.put(
            Event(
                "error", ticket, request, message=traceback.format_exc(), wall_time_s=time.perf_counter() - t0
            )
        )


def _worker_main(requests: Any, results: Any) -> None:
    state = _LiveState()
    while True:
        item = requests.get()
        if item is None:
            return
        _run(state, results, item)


# ---- the UI-side handle -------------------------------------------------------------------------------------------------------------


class SimulationWorker:
    """One worker process (in-process under WebAssembly): submit requests, poll their events, or wait for one ticket."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._requests: Any = None
        self._results: Any = None
        self._process: Any = None
        self._buffer: list[Event] = []
        self._state: _LiveState | None = None

    @property
    def alive(self) -> bool:
        if IN_PROCESS:
            return self._state is not None
        return self._process is not None and bool(self._process.is_alive())

    @property
    def started(self) -> bool:
        """Whether a worker was started and not deliberately stopped; with ``alive`` False this means it died."""
        return self._process is not None or self._state is not None

    def start(self) -> None:
        if self.alive:
            return
        if IN_PROCESS:
            self._state, self._results = _LiveState(), queue.Queue()
            return
        self._requests = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._requests, self._results),
            daemon=False,  # the core's own parallel maps start processes inside it
            name="qutip-trap-app-worker",
        )
        self._process.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._state = None
        if self._process is None:
            self._results = None
            return
        try:
            if self.alive and self._requests is not None:
                self._requests.put(None)
            self._process.join(timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
        finally:
            self._process = None
            self._requests = None
            self._results = None

    def cancel(self) -> None:
        """Terminate the process and start a fresh one."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        self._process = None
        self._state = None
        self._buffer.clear()
        self.start()

    def submit(self, request: str, **payload: Any) -> Ticket:
        if not self.alive:
            self.start()
        ticket = Ticket(uuid.uuid4().hex[:12], request)
        if IN_PROCESS:
            assert self._state is not None
            _run(self._state, self._results, (ticket.id, request, payload))
            return ticket
        self._requests.put((ticket.id, request, payload))
        return ticket

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        """Every event that has arrived (after one blocking wait of ``timeout_s`` when nothing is buffered)."""
        out = list(self._buffer)
        self._buffer.clear()
        if self._results is None:
            return out
        block = timeout_s > 0.0 and not out
        while True:
            try:
                ev: Event = self._results.get(timeout=timeout_s) if block else self._results.get_nowait()
            except queue.Empty:
                break
            block = False
            out.append(ev)
        return out

    def wait(
        self, ticket: Ticket, timeout_s: float = 600.0, on_progress: Callable[[Event], None] | None = None
    ) -> Any:
        """Block until ``ticket`` finishes; other tickets' events are kept for a later ``poll``."""
        deadline = time.monotonic() + timeout_s
        others: list[Event] = []
        try:
            while time.monotonic() < deadline:
                for ev in self.poll(timeout_s=0.2):
                    if ev.ticket != ticket.id:
                        others.append(ev)
                    elif ev.kind == "progress":
                        if on_progress is not None:
                            on_progress(ev)
                    elif ev.kind == "result":
                        return ev.payload
                    else:
                        raise WorkerError(ev.message)
                if not self.alive:
                    raise WorkerError("the worker process died")
            raise WorkerError(f"request {ticket.kind} timed out after {timeout_s:g} s")
        finally:
            self._buffer.extend(others)
