"""The simulation worker: core calls in a separate process, progress and results streamed back (PLAN.md Section 14.6).

One long-lived worker holds the live state a record cannot carry (the device, the calibration table, the core's run record,
the channel library) keyed by the record's run key, so that a zoom or a verify-deeper request needs no re-run. Requests and
results are picklable values; progress events name a stage, a fraction where one is known and a message. A cancel terminates
the process and restarts it: the caches live in the records the UI keeps, so only the live handles are lost. The process is
not daemonic, so the core's own parallel maps may spawn inside it; under WebAssembly (no processes) a request runs in the
page's thread when it is submitted.
"""

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
from qutip_trap_app import device_layer as layer_mod
from qutip_trap_app import presets as presets_mod
from qutip_trap_app.record import (
    DeviceRef,
    JobSpec,
    LiveRun,
    Progress,
    Record,
    TableRecord,
    calibrate_for,
    complete_job,
    execute,
    options_digest,
    table_record,
)
from qutip_trap_app.replay import ChannelLibrary, replay
from qutip_trap_app.verify import verify_deeper

EventKind = Literal["progress", "result", "error"]


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


class WorkerError(RuntimeError):
    """A request failed in the worker; the message carries the worker's traceback."""


IN_PROCESS = sys.platform == "emscripten"
"""WebAssembly has no processes: a request runs in the page's own thread at ``submit`` time, its events read afterwards."""


# ---- the requests ---------------------------------------------------------------------------------------------------------------


@dataclass
class _LiveState:
    records: dict[str, Record] = field(default_factory=dict)
    lives: dict[str, LiveRun] = field(default_factory=dict)
    libraries: dict[str, ChannelLibrary] = field(default_factory=dict)
    """Channel libraries per (device hash, table seed, options digest)."""
    presets: dict[str, core.DevicePreset] = field(default_factory=dict)
    """Built presets per ``DeviceRef.cache_key()`` (a build with the recipe re-derived takes seconds)."""
    tables: dict[str, core.CalibrationTable] = field(default_factory=dict)
    """The latest calibration table per device hash (the device layer compares its solutions with it)."""


def _preset_for(state: _LiveState, ref: DeviceRef, progress: Progress) -> core.DevicePreset:
    """The built preset of a device reference, cached by its arguments and overrides; the hash is checked when the
    reference carries one (an empty hash is a request from the UI, which never builds a device itself)."""
    key = ref.cache_key()
    preset = state.presets.get(key)
    if preset is None:
        if ref.overrides:
            progress(
                "building device", None, "applying the Level 4 knobs and re-deriving the preparation recipe"
            )
        preset = ref.build(check=bool(ref.hash))
        state.presets[key] = preset
    elif ref.hash and preset.device.hash() != ref.hash:
        raise WorkerError(
            f"the device reference's hash {ref.hash[:12]} does not match the built device {preset.device.hash()[:12]}"
        )
    return preset


def _calibrated(
    state: _LiveState, job: JobSpec, progress: Progress, message: str
) -> tuple[JobSpec, core.DevicePreset, core.CalibrationTable]:
    """The job completed from its built device, with the job's calibration table (kept as the device's latest)."""
    preset = _preset_for(state, job.device, progress)
    job = complete_job(job, preset)
    progress("calibrating", None, message)
    table = calibrate_for(job, preset.device)
    state.tables[preset.device.hash()] = table
    return job, preset, table


def _core_progress(progress: Progress) -> Callable[[core.Progress], None]:
    """The core's ``Progress`` of ``Machine.run`` (per pulse, branch, sample and readout) as worker progress events."""

    def forward(p: core.Progress) -> None:
        progress(f"running: {p.stage}", p.fraction, f"{p.stage} {p.done} of {p.total}")

    return forward


def _live(state: _LiveState, key: str) -> tuple[Record, LiveRun]:
    record, live = state.records.get(key), state.lives.get(key)
    if record is None or live is None:
        raise WorkerError(f"no live run for record key {key}: run the job again to re-simulate")
    return record, live


def _run_job(state: _LiveState, progress: Progress, *, job: JobSpec) -> Record:
    job, preset, _table = _calibrated(
        state, job, progress, "the surrogate calibration table (cached per device and seed)"
    )
    progress("running", None, f"{job.level} run of {job.shots} shots on {preset.device.crystal.n_ions} ions")
    record, live = execute(job, preset, progress=_core_progress(progress))
    state.records[record.key()] = record
    state.lives[record.key()] = live
    progress("recording", 1.0, "record built")
    return record


def _replay(state: _LiveState, progress: Progress, *, job: JobSpec) -> Record:
    job, preset, table = _calibrated(
        state, job, progress, "the surrogate calibration table (cached per device and seed)"
    )
    library = state.libraries.setdefault(
        f"{preset.device.hash()}/{table.seed}/{options_digest(job.options)}", ChannelLibrary()
    )
    record = replay(job, preset.device, table, library, progress=progress)
    state.records[record.key()] = record
    progress("recording", 1.0, "record built")
    return record


def _zoom(
    state: _LiveState,
    progress: Progress,
    *,
    key: str,
    step: int,
    sample: int = 0,
    branch: int = 0,
    n_store: int = resim.DEFAULT_ZOOM_POINTS,
) -> dict[str, Any]:
    record, live = _live(state, key)
    progress("zooming", None, f"re-simulating step {step} with {n_store} stored points per segment")
    record, z, stats = resim.zoom(record, live, step, sample, branch, n_store=n_store)
    progress("building", 0.9, "listing the Hamiltonian terms and collapse operators of the step")
    record, ham = resim.hamiltonian_record(record, live, step, sample, branch)
    state.records[key] = record
    return {"zoom": z, "boundaries": record.boundaries, "stats": stats, "hamiltonian": ham}


def _tomography(
    state: _LiveState, progress: Progress, *, key: str, step: int, sample: int = 0, branch: int = 0
) -> dict[str, Any]:
    record, live = _live(state, key)
    record, pm = resim.process_matrix(record, live, step, sample, branch, progress=progress)
    state.records[key] = record
    return {"process_matrix": pm, "boundaries": record.boundaries}


def _recheck(
    state: _LiveState, progress: Progress, *, key: str, step: int, sample: int = 0, branch: int = 0
) -> dict[str, Any]:
    record, live = _live(state, key)
    progress("rechecking", 0.0, "tolerances tightened by ten on the zoomed step")
    record, tol = resim.tolerance_recheck(record, live, step, sample, branch)
    progress("rechecking", 0.5, "every resolved cap raised by two, re-chained from the initial state")
    record, trunc = resim.truncation_recheck(record, live, step, sample, branch)
    state.records[key] = record
    return {"tolerance": tol, "truncation": trunc, "zooms": record.zooms, "boundaries": record.boundaries}


def _derive(
    state: _LiveState,
    progress: Progress,
    *,
    device: DeviceRef,
    table_record: TableRecord | None = None,
    sweeps: bool = True,
) -> layer_mod.DeviceLayer:
    preset = _preset_for(state, device, progress)
    return layer_mod.derive_device_layer(
        preset,
        overrides=device.overrides,
        table=state.tables.get(preset.device.hash()),
        table_record=table_record,
        sweeps=sweeps,
        progress=progress,
    )


def _recalibrate(state: _LiveState, progress: Progress, *, job: JobSpec) -> dict[str, Any]:
    _job, preset, table = _calibrated(
        state,
        job,
        progress,
        "the surrogate table for the edited device: closed forms, exact spot checks, detection records",
    )
    progress("recording", 1.0, "table built")
    return {"table": table_record(table), "device_hash": preset.device.hash()}


def _verify(
    state: _LiveState, progress: Progress, *, key: str, record: Record | None = None, shots: int | None = None
) -> dict[str, Any]:
    shallow = record if record is not None else state.records.get(key)
    if shallow is None:
        raise WorkerError(f"no record for key {key}")
    report, deep, deep_live = verify_deeper(
        shallow, live=state.lives.get(key), shots=shots, progress=progress
    )
    if deep is not None:
        state.records[deep.key()] = deep
        if deep_live is not None:
            state.lives[deep.key()] = deep_live
        if report.deep_level is None:
            state.records[key] = deep
    return {"report": report, "deep": deep}


def _request_run(state: _LiveState, progress: Progress, *, job: JobSpec, step: int = -1) -> Record:
    """A request made at Level 1 or 2 runs as its own job at the full engine (Section 14.4), and the requested gate's
    process matrix is computed at once, so Level 1 can show the actual unitary beside the requested one."""
    job, preset, _table = _calibrated(
        state,
        job,
        progress,
        "the surrogate calibration table, with the hand-set detunings applied where requested",
    )
    progress("running", None, f"{job.level} run of {job.shots} shots: {job.label or 'the requested job'}")
    record, live = execute(job, preset, progress=_core_progress(progress))
    key = record.key()
    state.lives[key] = live
    if step >= 0 and record.traces:
        progress("tomography", 0.8, "the process matrix of the requested gate's step")
        record, _pm = resim.process_matrix(record, live, step, 0, 0, progress=progress)
    state.records[key] = record
    progress("recording", 1.0, "record built")
    return record


def _preset(_state: _LiveState, progress: Progress, *, preset_id: str) -> Any:
    return presets_mod.run_preset(preset_id, progress)


HANDLERS: dict[str, Callable[..., Any]] = {
    "run_job": _run_job,
    "replay": _replay,
    "zoom": _zoom,
    "tomography": _tomography,
    "recheck": _recheck,
    "derive": _derive,
    "recalibrate": _recalibrate,
    "verify": _verify,
    "request_run": _request_run,
    "preset": _preset,
}
"""The requests the UI names, each resolved to its handler once, at submit time."""


# ---- running a request ---------------------------------------------------------------------------------------------------------

_Item = tuple[str, str, Callable[..., Any], dict[str, Any]]
"""(ticket id, request name, handler, payload)."""


def _run(state: _LiveState, results: Any, item: _Item) -> None:
    """One request: its progress events, then its result or its error, on ``results``."""
    ticket, request, handler, payload = item
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
        value = handler(state, progress, **payload)
        results.put(Event("result", ticket, request, payload=value, wall_time_s=time.perf_counter() - t0))
    except Exception:
        results.put(
            Event(
                "error", ticket, request, message=traceback.format_exc(), wall_time_s=time.perf_counter() - t0
            )
        )


def _worker_main(requests: Any, results: Any) -> None:
    state = _LiveState()
    while (item := requests.get()) is not None:
        _run(state, results, item)


class _ProcessTransport:
    """The worker in a spawned, non-daemonic process."""

    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        self.requests: Any = ctx.Queue()
        self.results: Any = ctx.Queue()
        self.process = ctx.Process(
            target=_worker_main,
            args=(self.requests, self.results),
            daemon=False,
            name="qutip-trap-app-worker",
        )
        self.process.start()

    @property
    def alive(self) -> bool:
        return bool(self.process.is_alive())

    def put(self, item: _Item) -> None:
        self.requests.put(item)

    def stop(self, timeout_s: float) -> None:
        if self.alive:
            self.requests.put(None)
        self.process.join(timeout_s)
        self.kill()

    def kill(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2.0)


class _InProcessTransport:
    """The worker in the page's own thread: a request runs when it is submitted, its events wait in a queue."""

    alive = True

    def __init__(self) -> None:
        self.state = _LiveState()
        self.results: Any = queue.Queue()

    def put(self, item: _Item) -> None:
        _run(self.state, self.results, item)

    def stop(self, timeout_s: float) -> None:
        return None

    def kill(self) -> None:
        return None


class SimulationWorker:
    """One worker; submit requests, poll events, or wait for one ticket."""

    def __init__(self) -> None:
        self._transport: _ProcessTransport | _InProcessTransport | None = None
        self._buffer: list[Event] = []

    @property
    def alive(self) -> bool:
        return self._transport is not None and self._transport.alive

    @property
    def started(self) -> bool:
        """Whether a worker was started and not deliberately stopped; with ``alive`` False this means it died."""
        return self._transport is not None

    def start(self) -> None:
        if not self.alive:
            self._transport = _InProcessTransport() if IN_PROCESS else _ProcessTransport()

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._transport is not None:
            self._transport.stop(timeout_s)
        self._transport = None

    def cancel(self) -> None:
        """Hard cancel: terminate the worker and start a fresh one (the live handles are lost; the records' caches are not)."""
        if self._transport is not None:
            self._transport.kill()
        self._transport = None
        self._buffer.clear()
        self.start()

    def submit(self, request: str, **payload: Any) -> Ticket:
        handler = HANDLERS.get(request)
        if handler is None:
            raise WorkerError(f"unknown request {request!r}; known: {sorted(HANDLERS)}")
        self.start()
        assert self._transport is not None
        ticket = Ticket(uuid.uuid4().hex[:12], request)
        self._transport.put((ticket.id, request, handler, payload))
        return ticket

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        """Every event that has arrived (a first blocking wait of ``timeout_s`` when nothing is buffered)."""
        out: list[Event] = list(self._buffer)
        self._buffer.clear()
        if self._transport is None:
            return out
        block = timeout_s > 0.0 and not out
        while True:
            try:
                ev: Event = (
                    self._transport.results.get(timeout=timeout_s)
                    if block
                    else self._transport.results.get_nowait()
                )
            except queue.Empty:
                break
            block = False
            out.append(ev)
        return out

    def wait(self, ticket: Ticket, timeout_s: float = 600.0, on_progress: Any = None) -> Any:
        """Block until ``ticket`` finishes; other tickets' events are buffered for a later ``poll``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for ev in self.poll(timeout_s=0.2):
                if ev.ticket != ticket.id:
                    self._buffer.append(ev)
                    continue
                if ev.kind == "progress":
                    if on_progress is not None:
                        on_progress(ev)
                elif ev.kind == "result":
                    return ev.payload
                else:
                    raise WorkerError(ev.message)
            if not self.alive:
                raise WorkerError("the worker process died")
        raise WorkerError(f"request {ticket.kind} timed out after {timeout_s:g} s")
