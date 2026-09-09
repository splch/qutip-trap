"""The simulation worker: core calls in a separate process, progress and results streamed back (PLAN.md Section 14.6
"Layering": "workers.py runs core calls in a ProcessPoolExecutor, streaming progress and partial results through
page.pubsub; view updates are scheduled with page.run_task, so the UI never blocks on a solver"; milestone M11.2).

One long-lived worker process holds the live state a record cannot carry (the device, the calibration table, the core's
run record, the channel library) keyed by the record's run key, so that a zoom or a verify-deeper request needs no re-run.
Requests and results are plain picklable values (the frozen records of ``record.py``); progress events name a stage, a
fraction where one is known and a message. The UI side drains the event queue from an asyncio task (``page.run_task``) and
applies results to its own copies of the records. A cancel terminates the process and restarts it: the caches live in the
records, which the UI keeps, so only the live handles are lost.

The worker is not daemonic, so the core's own parallel maps (Section 11.3 item 9) may spawn their processes inside it.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from qutip_trap_app.record import DeviceRef, JobSpec, LiveRun, Record

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


# ---- the worker process ---------------------------------------------------------------------------------------------------------


@dataclass
class _LiveState:
    records: dict[str, Record] = field(default_factory=dict)
    lives: dict[str, LiveRun] = field(default_factory=dict)
    libraries: dict[str, Any] = field(default_factory=dict)
    """Channel libraries per (device hash, table seed, options digest)."""
    presets: dict[str, Any] = field(default_factory=dict)
    """Built presets per ``DeviceRef.cache_key()`` (a build with the recipe re-derived takes seconds; M11.3)."""
    tables: dict[str, Any] = field(default_factory=dict)
    """Recalibrated ``CalibrationTable`` per device hash (Section 14.4's user-initiated job)."""
    stability: Any = None


def _preset_for(state: _LiveState, ref: DeviceRef, progress: Any) -> Any:
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


def complete_job(job: JobSpec, preset: Any) -> JobSpec:
    """A job the UI submitted without building a device (empty hash, empty drive maps) completed from the built preset."""
    import dataclasses

    from qutip_trap_app.record import DriveRef

    out = job
    if not out.device.hash:
        out = dataclasses.replace(out, device=dataclasses.replace(out.device, hash=preset.device.hash()))
    if not out.gate_drives and not out.entangling_drives:
        out = dataclasses.replace(
            out,
            gate_drives={int(i): DriveRef.from_core(d) for i, d in preset.gate_drives.items()},
            entangling_drives={int(i): DriveRef.from_core(d) for i, d in preset.entangling_drives.items()},
        )
    return out


def _handle(state: _LiveState, request: str, payload: dict[str, Any], progress: Any) -> Any:
    from qutip_trap_app import device_layer as layer_mod
    from qutip_trap_app import record as rec_mod
    from qutip_trap_app import replay as replay_mod
    from qutip_trap_app import resim
    from qutip_trap_app.replay_record import build_replay_record
    from qutip_trap_app.verify import verify_deeper

    if request == "run_job":
        job: JobSpec = payload["job"]
        preset = _preset_for(state, job.device, progress)
        job = complete_job(job, preset)
        progress("calibrating", None, "the surrogate calibration table (cached per device and seed)")
        table = rec_mod.calibrate_for(job, preset)
        state.tables[preset.device.hash()] = table
        progress(
            "running", None, f"{job.level} run of {job.shots} shots on {preset.device.crystal.n_ions} ions"
        )
        record, live = rec_mod.execute(job, preset)
        key = record.key()
        state.records[key] = record
        state.lives[key] = live
        progress("recording", 1.0, "record built")
        return record
    if request == "replay":
        job = payload["job"]
        preset = _preset_for(state, job.device, progress)
        job = complete_job(job, preset)
        progress("calibrating", None, "the surrogate calibration table (cached per device and seed)")
        table = rec_mod.calibrate_for(job, preset)
        lib_key = f"{preset.device.hash()}/{table.seed}/{replay_mod.options_digest(job.solver_options())}"
        library = state.libraries.get(lib_key)
        if library is None:
            library = replay_mod.ChannelLibrary.for_job(job, preset.device, table)
            state.libraries[lib_key] = library
        outcome = replay_mod.replay(job, preset.device, table, library, progress=progress)
        state.tables[preset.device.hash()] = table
        record = build_replay_record(job, preset.device, table, outcome, library)
        state.records[record.key()] = record
        progress("recording", 1.0, "record built")
        return record
    if request == "zoom":
        key = payload["key"]
        live_opt = state.lives.get(key)
        record_opt = state.records.get(key)
        if live_opt is None or record_opt is None:
            raise WorkerError(f"no live run for record key {key}: run the job again to re-simulate")
        live, record = live_opt, record_opt
        step_i, sample_i, branch_i = (
            int(payload["step"]),
            int(payload.get("sample", 0)),
            int(payload.get("branch", 0)),
        )
        progress(
            "zooming",
            None,
            f"re-simulating step {step_i} with {payload.get('n_store', resim.DEFAULT_ZOOM_POINTS)} stored points per segment",
        )
        record, z, stats = resim.zoom(
            record,
            live,
            step_i,
            sample_i,
            branch_i,
            n_store=int(payload.get("n_store", resim.DEFAULT_ZOOM_POINTS)),
        )
        progress("building", 0.9, "listing the Hamiltonian terms and collapse operators of the step")
        record, ham = resim.hamiltonian_record(record, live, step_i, sample_i, branch_i)
        state.records[key] = record
        return {"zoom": z, "boundaries": record.boundaries, "stats": stats, "hamiltonian": ham}
    if request == "fock_movie":
        key = payload["key"]
        live_opt = state.lives.get(key)
        record_opt = state.records.get(key)
        if live_opt is None or record_opt is None:
            raise WorkerError(f"no live run for record key {key}: run the job again to re-simulate")
        record, movie = resim.fock_movie(
            record_opt,
            live_opt,
            int(payload["step"]),
            int(payload.get("sample", 0)),
            int(payload.get("branch", 0)),
            n_frames=int(payload.get("n_frames", resim.DEFAULT_FOCK_FRAMES)),
            progress=progress,
        )
        state.records[key] = record
        return {"movie": movie, "boundaries": record.boundaries}
    if request == "tomography":
        key = payload["key"]
        live_opt = state.lives.get(key)
        record_opt = state.records.get(key)
        if live_opt is None or record_opt is None:
            raise WorkerError(f"no live run for record key {key}: run the job again to re-simulate")
        record, pm = resim.process_matrix(
            record_opt,
            live_opt,
            int(payload["step"]),
            int(payload.get("sample", 0)),
            int(payload.get("branch", 0)),
            progress=progress,
        )
        state.records[key] = record
        return {"process_matrix": pm, "boundaries": record.boundaries}
    if request == "recheck":
        key = payload["key"]
        live_opt = state.lives.get(key)
        record_opt = state.records.get(key)
        if live_opt is None or record_opt is None:
            raise WorkerError(f"no live run for record key {key}: run the job again to re-simulate")
        step_i, sample_i, branch_i = (
            int(payload["step"]),
            int(payload.get("sample", 0)),
            int(payload.get("branch", 0)),
        )
        progress("rechecking", 0.0, "tolerances tightened by ten on the zoomed step")
        record, tol = resim.tolerance_recheck(record_opt, live_opt, step_i, sample_i, branch_i)
        progress("rechecking", 0.5, "every resolved cap raised by two, re-chained from the initial state")
        record, trunc = resim.truncation_recheck(record, live_opt, step_i, sample_i, branch_i)
        state.records[key] = record
        return {"tolerance": tol, "truncation": trunc, "zooms": record.zooms, "boundaries": record.boundaries}
    if request == "derive":
        ref: DeviceRef = payload["device"]
        preset = _preset_for(state, ref, progress)
        if state.stability is None:
            progress("deriving", 0.1, "the Mathieu stability boundary (once per session)")
            state.stability = layer_mod.stability_map()
        table_opt = state.tables.get(preset.device.hash())
        layer = layer_mod.derive_device_layer(
            preset,
            preset_name=str(ref.preset),
            kwargs=ref.kwargs,
            overrides=ref.overrides,
            table=table_opt,
            table_record=payload.get("table_record"),
            stability=state.stability,
            sweeps=bool(payload.get("sweeps", True)),
            progress=progress,
        )
        return layer
    if request == "recalibrate":
        job = payload["job"]
        preset = _preset_for(state, job.device, progress)
        job = complete_job(job, preset)
        progress(
            "calibrating",
            None,
            "the surrogate table for the edited device: closed forms, exact spot checks, detection records",
        )
        table = rec_mod.calibrate_for(job, preset)
        state.tables[preset.device.hash()] = table
        progress("recording", 1.0, "table built")
        return {"table": rec_mod.table_record(table), "device_hash": preset.device.hash(), "job": job}
    if request == "verify":
        key = payload["key"]
        record_opt = payload.get("record") or state.records.get(key)
        if record_opt is None:
            raise WorkerError(f"no record for key {key}")
        record = record_opt
        report, deep, deep_live = verify_deeper(
            record, live=state.lives.get(key), shots=payload.get("shots"), progress=progress
        )
        if deep is not None:
            state.records[deep.key()] = deep
            if deep_live is not None:
                state.lives[deep.key()] = deep_live
        if report.deep_level is None and deep is not None:
            state.records[key] = deep
        return {"report": report, "deep": deep}
    if request == "ping":
        return "pong"
    raise WorkerError(f"unknown request {request!r}")


def _worker_main(requests: Any, results: Any) -> None:
    state = _LiveState()
    while True:
        item = requests.get()
        if item is None:
            return
        ticket, request, payload = item
        t0 = time.perf_counter()

        def progress(
            stage: str,
            fraction: float | None,
            message: str,
            _ticket: str = ticket,
            _request: str = request,
            _t0: float = t0,
        ) -> None:
            results.put(
                Event(
                    "progress",
                    _ticket,
                    _request,
                    stage=stage,
                    fraction=fraction,
                    message=message,
                    wall_time_s=time.perf_counter() - _t0,
                )
            )

        try:
            value = _handle(state, request, payload, progress)
            results.put(Event("result", ticket, request, payload=value, wall_time_s=time.perf_counter() - t0))
        except Exception:
            results.put(
                Event(
                    "error",
                    ticket,
                    request,
                    message=traceback.format_exc(),
                    wall_time_s=time.perf_counter() - t0,
                )
            )


# ---- the UI-side handle -----------------------------------------------------------------------------------------------------------


class SimulationWorker:
    """One worker process; submit requests, poll events, or wait for one ticket."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._requests: Any = None
        self._results: Any = None
        self._process: Any = None
        self._buffer: list[Event] = []

    @property
    def alive(self) -> bool:
        return self._process is not None and bool(self._process.is_alive())

    def start(self) -> None:
        if self.alive:
            return
        self._requests = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._requests, self._results),
            daemon=False,
            name="qutip-trap-app-worker",
        )
        self._process.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._process is None:
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
        """Hard cancel: terminate the process and start a fresh one (the live handles are lost; the records' caches are not)."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        self._process = None
        self._buffer.clear()
        self.start()

    def submit(self, request: str, **payload: Any) -> Ticket:
        if not self.alive:
            self.start()
        ticket = Ticket(uuid.uuid4().hex[:12], request)
        assert self._requests is not None
        self._requests.put((ticket.id, request, payload))
        return ticket

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        """Every event that has arrived (a first blocking wait of ``timeout_s`` when nothing is buffered)."""
        out: list[Event] = list(self._buffer)
        self._buffer.clear()
        if self._results is None:
            return out
        block = timeout_s > 0.0 and not out
        while True:
            try:
                ev: Event = (
                    self._results.get(timeout=timeout_s if block else 0.0)
                    if block
                    else self._results.get_nowait()
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


__all__ = ["Event", "EventKind", "SimulationWorker", "Ticket", "WorkerError", "complete_job"]
