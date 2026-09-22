# qutip-trap-app

The zoomable application of PLAN.md Section 14 (milestone M11): one simulated trapped-ion quantum computer at five levels of
abstraction, from a cloud customer's histogram down to the Hamiltonian terms the solver integrates, built as a learning
tool. The design is [`DESIGN.md`](DESIGN.md). This package consumes the physics core `qutip_trap` through its public API
only (`src/qutip_trap_app/core.py` is the one module that imports it, with the gaps it records); nothing in the core
imports this package.

## Status

**M11.1 to M11.4 done.** The record layer and all five levels of the ladder exist and run in a native window or a browser
served from the host; M11.4 rebuilt every screen as a learning tool (DESIGN.md Section 10) and added the presets, the
request semantics, the explain drawer's specification text and the navigation tests. What exists and is tested:

- `record.py` - the run record of Section 14.3 (job, compiled circuit, schedule, space, preparation, noise samples,
  branches, per-(sample, branch) traces, readout records, results with the target beside them, diagnostics, device card,
  calibration table, core gaps) and its storage policy; `execute(job)` runs a job and records it; (M11.4) the job's
  `waveform_overrides` (a beat-note detuning set by hand, applied to the table as written), `requests` and `label`; record
  format `qutip-trap-app/record/3`.
- `codec.py`, `storage.py` - export to one zip of JSON plus `.npy` arrays with a digest; bitwise re-import.
- `resim.py` - the engine as `run()` built it; boundary states by chaining over gate steps; zoom into one pulse with a fine
  store, cached by key; tolerance and cap re-checks for the convergence badge; the Fock movie by truncated re-simulation, the
  Hamiltonian record of a step and the process matrix of a step by state-based tomography.
- `requests.py` (M11.4) - the request semantics of Section 14.4: XX(chi) requested at Level 1 rebuilds the job with that
  angle (the scheduler rescales the calibrated waveform by sqrt(|chi|/|chi_cal|), the s-squared law) and a detuning set by
  hand at Level 2 shifts the pair's waveform as written; requests the device cannot satisfy are refused with the reason; the
  MS angle is read back from a process matrix by a fit over the ideal family, so "the simulated unitary is XX(0.3)" is a number.
- `presets.py`, `viewmodel/presets.py` (M11.4) - the published-experiment presets of Section 14.5: Harty 2014 microwave RB
  (Section 9.2), James 1998 positions and axial modes (9.1), Monroe 1995 Doppler cooling (9.3), Roos 2000 Lamb-Dicke
  parameters (9.3), Kirchmair 2009 thermal MS populations (9.4), Myerson 2008 and Crain 2019 readout (9.5), plus the Bell
  state and the three-ion GHZ of Section 9.6 as circuit presets that run through the machine; every published number carries a
  `published.*` ledger record with the Section 9 row's tag, every simulated one the `anchor.*` of the check that recomputed
  it, and the comparison says when a number is not a first-principles prediction and why (Kirchmair's fidelity, Myerson's
  optimum).
- `knobs.py`, `device_layer.py` - the Level 4 knobs of Section 14.4 and the rebuild of a preset with overrides; the device
  layer every physics page reads, derived in the worker from the public API alone.
- `replay.py`, `replay_record.py` - the app-side channel replay of Section 5.4 with its measured frame covariance and
  derivation residual; `verify.py` - verify deeper; `workers.py` - one worker process that holds the live state and streams
  progress (M11.4: the `request_run` and `preset` requests).
- `provenance.py` - the index generated from `docs/provenance/ledger.yaml` and `PLAN.md` into
  `src/qutip_trap_app/provenance_index.json`; (M11.4) every section's own Markdown, so the explain drawer's
  Specification tile shows the text that governs the screen and a clicked chip opens the section it cites.
- `viewmodel/` - `catalogue` (every displayed quantity with its ledger id, 221 of them), `machine`, `circuit`, `schedule`,
  `dynamics`, `physics`, `numerics`, `learn` (30 concepts, the six-click tour, the faded GHZ exercise, the free exercise, the
  Learn routes, the spacing rule, the mastery log), `drills` (M11.4: four discriminations generated from the current record and
  interleaved), `presets`.
- `views/` - the Flet screens on the information hierarchy of DESIGN.md Section 10: one focal picture per level, numbers as
  stat tiles, tables behind Details, a why button per card opening its concept in the explain drawer (one concept at a time,
  the Specification tile beneath), chips that are buttons, a badge-only numerics strip; the request controls on Levels 1
  and 2 with the actual-versus-requested tiles; Level 1's Bloch discs; the Learn ladder (tour, three ions, your own, drills,
  review, published experiments, progress) as routes; control keys for the `flet test` flows. Level 0's circuit is built,
  not typed (DESIGN.md R16): `views/builder.py` draws the qubits as wires and the gates as tiles over the pure
  `viewmodel/builder.py` (the palette, the greedy layout, the edits, the OpenQASM 2 round trip), with the text and an
  OpenQASM 2 or IonQ JSON import behind a Code disclosure.
- `views/theme.py` - the visual system of DESIGN.md Section 11: the light and dark colour schemes (every text/background
  pair checked at or above 4.5:1, outlines at 3:1), the type scale, the 8 pt grid and shapes, the component themes, the
  status colours of the pills and the categorical chart palette (validated for colour-vision deficiency in both modes).
  `views/common.py` builds every shared piece on it (cards, tiles, chips, pills, the tab strip, the width-aware two-column
  layout, the explain drawer); the shell adds the rail with the app mark and a system/light/dark toggle that persists with
  the learner.
- `tests/` - the Section 9.11 rows M11.1 to M11.4 own: coarse-graining identity, record round trip, re-simulation cache,
  convergence badge, provenance coverage (Levels 0 to 4, the presets, the drills), channel derivation (cold; the hotter half is
  `-m slow`), downward propagation, request semantics (`test_requests.py`: XX(0.3) read back within 5e-3 rad of the request
  by tomography; a 5 kHz hand-set detuning opens the loops and moves the played gate's channel away from the requested unitary
  while the target stays the requested one), presets (`test_presets.py`), navigation (`test_navigation.py` over the routing
  model; `test_main.py` under `flet test`), the text budget (`test_text_budget.py`, an AST walk of every screen's `ft.Text` and `status_line` literals), the
  explain index (`test_explain_index.py`), plus verify deeper, the worker, the learning layer, the application state and
  Level 3 on demand.

## Commands

From the repository root:

    uv sync --all-packages --extra gui --group dev                 # installs the core, the app and Flet
    uv run --package qutip-trap-app pytest qutip_trap_app/tests -m 'not slow'   # the view-model tests (about twelve minutes)
    uv run --package qutip-trap-app pytest qutip_trap_app/tests               # with the hotter-state channel test
    uv run python -m qutip_trap_app.provenance                     # regenerate src/qutip_trap_app/provenance_index.json
    uv run python -m qutip_trap_app.provenance --check             # CI: the asset is current with the ledger and the plan
    (cd qutip_trap_app && uv run mypy)                              # strict types for src/qutip_trap_app
    uv run ruff check qutip_trap_app && uv run ruff format --check qutip_trap_app

The `flet test` navigation and run-flow tests (`tests/test_main.py`) drive the rendered controls by their keys through
Flet's `flet_app` fixture. They need the Flutter test host: `flet test` provisions it with the Flutter SDK it installs on
demand (under `~/flutter`), and on macOS the host is a desktop build that needs the full Xcode, which the Command Line Tools
do not provide. `tests/conftest.py` skips them unless `QUTIP_TRAP_APP_UI_TESTS=1`:

    cd qutip_trap_app && QUTIP_TRAP_APP_UI_TESTS=1 uv run flet test

On the 2026-09-09 development machine (Command Line Tools only, no Xcode, no CocoaPods) `flet build macos` stopped at
`flutter doctor` with "Xcode installation is incomplete; a full installation is necessary" and the UI tests were not run;
`test_navigation.py` covers the same six-click path over the routing model without a Flutter client. The packaging
configuration is in `pyproject.toml` (`[tool.flet]`); with Xcode present the bundles are

    cd qutip_trap_app && uv run --project .. flet build macos      # also windows, linux on those hosts
    cd qutip_trap_app && uv run --project .. flet pack src/main.py   # the PyInstaller fallback, no Flutter SDK or Xcode needed

Run the app (the first Run derives the channel library for the device, about a minute; a full simulation of the Bell
circuit takes about 20 s):

    cd qutip_trap_app && uv run --project .. flet run src/main.py            # native window
    cd qutip_trap_app && uv run --project .. flet run --web -p 8550 src/main.py   # browser served from the host
    cd qutip_trap_app && FLET_FORCE_WEB_SERVER=true FLET_SERVER_PORT=8550 uv run --project .. python src/main.py   # server only

The static Pyodide build is the GitHub Pages site, https://splch.github.io/qutip-trap/, deployed by `.github/workflows/pages.yml` on every push to `main`: QuTiP 5.3.1 compiled for Pyodide 314 with `pyodide build` (there is no WebAssembly wheel on PyPI), the core's wheel, then `flet build web --python-version 3.14 --base-url /qutip-trap/ --route-url-strategy hash`, both wheels offered to Flet's pip through `PIP_FIND_LINKS`. In the browser the worker runs in-process (`workers.IN_PROCESS`): the page waits for a request instead of streaming its progress.

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
