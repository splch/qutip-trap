# qutip-trap-app

The zoomable application of PLAN.md Section 14 (milestone M11): one simulated trapped-ion quantum computer at five levels of
abstraction, from a cloud customer's histogram down to the Hamiltonian terms the solver integrates, built as a learning
tool. The design is [`DESIGN.md`](DESIGN.md). This package consumes the physics core `qutip_trap` through its public API
only (`src/qutip_trap_app/core.py` is the one module that imports it, with the gaps it records); nothing in the core
imports this package.

## Status

**M11.1 and M11.2 done.** The record layer and the first three levels of the ladder exist and run in a native window or a
browser served from the host. What exists and is tested:

- `record.py` - the run record of Section 14.3 (job, compiled circuit, schedule, space, preparation, noise samples,
  branches, per-(sample, branch) traces, readout records, results with the target beside them, diagnostics, device card,
  calibration table, core gaps) and its storage policy; `execute(job)` runs a job and records it.
- `codec.py`, `storage.py` - export to one zip of JSON plus `.npy` arrays with a digest; bitwise re-import.
- `resim.py` - the engine as `run()` built it; boundary states by chaining over gate steps; zoom into one pulse with a fine
  store, cached by key; tolerance and cap re-checks for the convergence badge.
- `replay.py`, `replay_record.py` (M11.2) - the app-side channel replay of Section 5.4: every gate as the Section 6.8
  channel the core extracts by GATE_LOCAL tomography, conjugated to the played phase (the frame covariance is measured,
  not assumed), the register as a density matrix, the table's readout errors; the channel-derivation residual it reports
  bounds its distance from JOINT_EXACT (Section 9.11, tested cold and, as a slow test, hot).
- `verify.py` (M11.2) - verify deeper: the same job at the next engine, the register populations compared against the
  residual or Section 9.8's bound; a JOINT_EXACT record runs the Section 5.5 re-checks instead.
- `workers.py` (M11.2) - one worker process that holds the live state and streams progress; the UI never blocks.
- `provenance.py` - the index generated from `docs/provenance/ledger.yaml` and `PLAN.md` into
  `src/assets/provenance_index.json`; chips and Part II sections at run time.
- `viewmodel/` - `catalogue` (every displayed quantity with its ledger id), `machine` (Level 0), `circuit` (Level 1),
  `schedule` (Level 2), `dynamics` (Level 3), `numerics` (the panel and badge), `learn` (concepts, prompts, prior-knowledge
  plans, spacing, mastery log, the six-click tour).
- `views/` (M11.2) - the Flet screens: the shell with the navigation rail, the breadcrumb zoom bar with keyboard zoom, the
  explain drawer and the numerics strip; Level 0 (device card, circuit editor with OpenQASM 2 and IonQ JSON, engine choice,
  Run, predict-then-reveal, histogram with the target beside it, shots behind a bar, verify deeper); Level 1 (timeline,
  gate card with target unitary and calibrated parameters, register after the gate, phase register, compile report);
  Level 2 (time axis, tones against the mode spectrum, waveform segments, closure indicators, crosstalk); the Learn page;
  placeholders for Level 3 and the Level 4 pages (M11.3). Every level page scrolls; the explain cards and the expandable
  tiles keep their open state across re-renders; a run shows a live elapsed time and a Cancel button; the prediction is
  asked before every run of a changed circuit and scored beside that run's histogram; the learner's settings and mastery
  log are kept on the device through Flet's `SharedPreferences`, so the first-launch question is asked once per device
  and the review tray can come due across launches.
- `tests/` - the Section 9.11 rows M11.1 and M11.2 own: coarse-graining identity, record round trip, re-simulation cache,
  convergence badge, provenance coverage, channel derivation (cold; the hotter half is `-m slow`), plus verify deeper, the
  worker, the learning layer's integrity, and the application state (`test_state.py`: the learner's persistence document,
  one prediction per run, the progress heartbeat, cancel, the zoom bar's parent route).

## Commands

From the repository root:

    uv sync --all-packages --extra gui --group dev                 # installs the core, the app and Flet
    uv run --package qutip-trap-app pytest qutip_trap_app/tests -m 'not slow'   # the view-model tests (about five minutes)
    uv run --package qutip-trap-app pytest qutip_trap_app/tests               # with the hotter-state channel test (about ten)
    uv run python -m qutip_trap_app.provenance                     # regenerate src/assets/provenance_index.json
    uv run python -m qutip_trap_app.provenance --check             # CI: the asset is current with the ledger and the plan
    (cd qutip_trap_app && uv run mypy)                              # strict types for src/qutip_trap_app
    uv run ruff check qutip_trap_app && uv run ruff format --check qutip_trap_app

The `flet test` navigation and run-flow tests are milestone M11.4 (the `flet create` counter test that shipped with the
template was removed: it asserted a screen the app no longer has). `tests/conftest.py` already skips any `flet_app` test
unless `QUTIP_TRAP_APP_UI_TESTS=1` is set, so adding them needs no plumbing:

    cd qutip_trap_app && QUTIP_TRAP_APP_UI_TESTS=1 uv run flet test

Run the app (the first Run derives the channel library for the device, about a minute; a full simulation of the Bell
circuit takes about 20 s):

    cd qutip_trap_app && uv run --project .. flet run src/main.py            # native window
    cd qutip_trap_app && uv run --project .. flet run --web -p 8550 src/main.py   # browser served from the host
    cd qutip_trap_app && FLET_FORCE_WEB_SERVER=true FLET_SERVER_PORT=8550 uv run --project .. python src/main.py   # server only

The static Pyodide web build is not a target (PLAN.md Section 1.5).

## Using the record from Python

```python
from qutip_trap_app.core import Circuit, Operation, SolverOptions
from qutip_trap_app.record import execute, job_for_preset
from qutip_trap_app.resim import zoom
from qutip_trap_app.storage import export_record, import_record
from qutip_trap_app.viewmodel.circuit import register_after, timeline
from qutip_trap_app.viewmodel.machine import histogram

bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
job, preset = job_for_preset("yb171_chain", 2, bell, 200, seed=7, options=SolverOptions(branch_weight_min=1e-3))
record, live = execute(job, preset)                 # calibrate, run, record (about 20 s)
h = histogram(record)                               # bars with counts, probabilities, error bars, target beside them
ms = next(g for g in timeline(record) if g.name.value == "ms")
after = register_after(record, ms.index)            # Bloch vectors vanish, purity stays near one
record, z, stats = zoom(record, live, ms.step_index)  # the MS pulse at 201 points per segment, cached in the record
record_id = export_record(record, "bell.qtrec.zip")
assert import_record("bell.qtrec.zip").digest() == record_id
```

Every displayed value is a `Shown(quantity, value)` whose quantity names a ledger record; `provenance.ProvenanceIndex`
turns it into a chip.
