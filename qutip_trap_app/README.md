# qutip-trap-app

The zoomable application of PLAN.md Section 14 (milestone M11): one simulated trapped-ion quantum computer at five levels of
abstraction, from a cloud customer's histogram down to the Hamiltonian terms the solver integrates, built as a learning
tool. The design is [`DESIGN.md`](DESIGN.md). This package consumes the physics core `qutip_trap` through its public API
only (`src/qutip_trap_app/core.py` is the one module that imports it, with the gaps it records); nothing in the core
imports this package.

## Status

**M11.1 to M11.3 done.** The record layer and all five levels of the ladder exist and run in a native window or a browser
served from the host. What exists and is tested:

- `record.py` - the run record of Section 14.3 (job, compiled circuit, schedule, space, preparation, noise samples,
  branches, per-(sample, branch) traces, readout records, results with the target beside them, diagnostics, device card,
  calibration table, core gaps) and its storage policy; `execute(job)` runs a job and records it.
- `codec.py`, `storage.py` - export to one zip of JSON plus `.npy` arrays with a digest; bitwise re-import.
- `resim.py` - the engine as `run()` built it; boundary states by chaining over gate steps; zoom into one pulse with a fine
  store, cached by key; tolerance and cap re-checks for the convergence badge; (M11.3) the Fock movie by truncated
  re-simulation, the Hamiltonian record of a step (terms, matrix elements, collapse operators) and the process matrix of a
  step by state-based tomography from its recorded motional state.
- `knobs.py`, `device_layer.py` (M11.3) - the Level 4 knobs of Section 14.4 and the rebuild of a preset with overrides
  (crystal re-solved, recipe re-derived); the device layer every physics page reads, derived in the worker from the public
  API alone: species, trap with the Mathieu stability boundary, crystal, light with the scattering sweep, noise, cooling,
  readout with exact count distributions and the threshold scan, the pulse-solver solutions per pair, and the device card of
  an edited device.
- `replay.py`, `replay_record.py` (M11.2) - the app-side channel replay of Section 5.4: every gate as the Section 6.8
  channel the core extracts by GATE_LOCAL tomography, conjugated to the played phase (the frame covariance is measured,
  not assumed), the register as a density matrix, the table's readout errors; the channel-derivation residual it reports
  bounds its distance from JOINT_EXACT (Section 9.11, tested cold and, as a slow test, hot).
- `verify.py` (M11.2) - verify deeper: the same job at the next engine, the register populations compared against the
  residual or Section 9.8's bound; a JOINT_EXACT record runs the Section 5.5 re-checks instead.
- `workers.py` (M11.2) - one worker process that holds the live state and streams progress; the UI never blocks.
- `provenance.py` - the index generated from `docs/provenance/ledger.yaml` and `PLAN.md` into
  `src/assets/provenance_index.json`; chips and Part II sections at run time.
- `viewmodel/` - `catalogue` (every displayed quantity with its ledger id, 191 of them), `machine` (Level 0), `circuit`
  (Level 1), `schedule` (Level 2), `dynamics` (Level 3: the recorded trace, the fine trace, Fock heatmaps, the process
  view, the closure table), `physics` (Level 4: one view per page over the device layer, the Hamiltonian view, the knob rows,
  the edited device's card), `numerics` (the panel and badge), `learn` (29 concepts, prompts, prior-knowledge plans, spacing,
  mastery log, the six-click tour, the per-page concept sets).
- `views/` (M11.2) - the Flet screens: the shell with the navigation rail, the breadcrumb zoom bar with keyboard zoom, the
  explain drawer and the numerics strip; Level 0 (device card, circuit editor with OpenQASM 2 and IonQ JSON, engine choice,
  Run, predict-then-reveal, histogram with the target beside it, shots behind a bar, verify deeper); Level 1 (timeline,
  gate card with target unitary and calibrated parameters, register after the gate, phase register, compile report);
  Level 2 (time axis, tones against the mode spectrum, waveform segments, closure indicators, crosstalk); the Learn page;
  (M11.3) Level 3 (sample and branch selectors, the recorded trace at once and Re-simulate, the closure prediction before
  the loops are drawn, P1 and coherences, concurrence and Pauli correlators, <n_m>(t), the spin-branch loops of the played waveform, Fock bars and
  the P(n, t) heatmap, jumps linking to their collapse operator, the process matrix, the convergence re-checks) and the
  eight Level 4 pages with their knob panels, stale badges and Recalibrate, drawn on Flet's canvas and a pure-Python PNG
  heatmap (`views/drawing.py`; matplotlib is not a dependency). Every level page scrolls; the explain cards and the expandable
  tiles keep their open state across re-renders; a run shows a live elapsed time and a Cancel button; the prediction is
  asked before every run of a changed circuit and scored beside that run's histogram; the learner's settings and mastery
  log are kept on the device through Flet's `SharedPreferences`, so the first-launch question is asked once per device
  and the review tray can come due across launches.
- `tests/` - the Section 9.11 rows M11.1 to M11.3 own: coarse-graining identity, record round trip, re-simulation cache,
  convergence badge, provenance coverage (Levels 0 to 4), channel derivation (cold; the hotter half is `-m slow`), downward
  propagation (`test_knobs.py`: the rf amplitude scales q at fixed a with beta, nu and eta following Section 4.1, the
  recalibrated table re-solves the pair's waveform to the layer's own closed-form solution up to the exact spot check's
  amplitude factor, the device card updates; the layer agrees
  with the record it describes), plus verify deeper, the worker, the learning layer's integrity, the application state
  (`test_state.py`, `test_device_state.py`: one derive per knob change, the layer and the recalibration landing in the
  store, the worker building an edited device once) and Level 3 on demand (`test_level3.py`: the Hamiltonian record's
  matrix elements against QuTiP's displacement operator, the Fock movie's frames against the boundary states and the fine
  zoom, the process matrix's normalizations, the recorded trace, the closure scorer).

## Commands

From the repository root:

    uv sync --all-packages --extra gui --group dev                 # installs the core, the app and Flet
    uv run --package qutip-trap-app pytest qutip_trap_app/tests -m 'not slow'   # the view-model tests (about eight minutes)
    uv run --package qutip-trap-app pytest qutip_trap_app/tests               # with the hotter-state channel test (about thirteen)
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

The Level 4 layer from Python (M11.3): a device edited by its knobs, derived into the records the physics pages read, and
the pulse re-simulated on demand for the Level 3 views:

```python
from qutip_trap_app import device_layer, knobs, resim
from qutip_trap_app.viewmodel.dynamics import fock_heatmaps, process_view
from qutip_trap_app.viewmodel.physics import trap_view

edited = record.job.device.with_overrides({"trap.rf_frequency_hz": 40e6, "trap.rf_amplitude_scale": 1.1})
layer = device_layer.derive_device_layer(edited.build(), preset_name="yb171_chain", overrides=edited.overrides,
                                         table_record=record.table)      # about 4 s; layer.stale is True
print(trap_view(layer).mathieu)                                          # a, q, beta, nu, C0 per axis, each a Shown
record, ham = resim.hamiltonian_record(record, live, ms.step_index)      # the terms and collapse operators of the step
record, movie = resim.fock_movie(record, live, ms.step_index, n_frames=4)   # P(n, t) by truncated re-simulation
record, pm = resim.process_matrix(record, live, ms.step_index)          # the step as a channel (16 inputs, about 15 s)
print(fock_heatmaps(movie)[0].values.shape, process_view(pm).infidelity)
print(knobs.knobs_for(edited.build().device)["trap.rf_amplitude_scale"].doc)
```
