# qutip-trap-app

The zoomable application of PLAN.md Section 14: one simulated trapped-ion quantum computer at five levels of abstraction,
from a cloud customer's histogram down to the Hamiltonian terms the solver integrates, built as a learning tool. The design
is [`DESIGN.md`](DESIGN.md). The app uses the physics core `qutip_trap` through `src/qutip_trap_app/core.py`, the one module
that imports it; nothing in the core imports the app.

## Layout

- `record.py` - the run record every screen reads (Section 14.3) and `execute(job)`, which calibrates, runs and records a job;
  `codec.py` - the record's canonical JSON-plus-arrays encoding, whose digest is the record's key.
- `resim.py` - re-simulation of one gate step on demand: boundary states by chaining the engine, the fine zoom (cached by key),
  the tolerance and cap re-checks behind the convergence badge, the Hamiltonian listing and the process matrix.
- `replay.py` - the app-side channel replay, Level 0's default engine, with its derivation residual; `verify.py` - verify
  deeper; `requests.py` - the Level 1 and 2 requests of Section 14.4.
- `knobs.py`, `device_layer.py` - the Level 4 knobs and the device layer the physics pages read.
- `presets.py` - the published-experiment presets of Section 14.5.
- `workers.py` - the worker process that holds the live state and streams progress (in-process under WebAssembly).
- `provenance.py` - the index behind the provenance chips and the explain drawer, built from `docs/provenance/ledger.yaml`
  and `PLAN.md`.
- `viewmodel/` - pure view-models over the record (no Flet import); `views/` - the Flet screens.

## Commands

From the repository root:

    uv sync --all-packages --extra gui --group dev
    uv run --package qutip-trap-app pytest qutip_trap_app/tests -m 'not slow'   # without -m: the hotter-state replay test too
    (cd qutip_trap_app && uv run mypy)
    uv run ruff check qutip_trap_app && uv run ruff format --check qutip_trap_app

Run the app (the first Run derives the channel library for the device, about a minute):

    cd qutip_trap_app && uv run --project .. flet run src/main.py                  # native window
    cd qutip_trap_app && uv run --project .. flet run --web -p 8550 src/main.py   # browser served from the host

A checkout builds the provenance index from PLAN.md and the ledger when it is first loaded. A packaged build has no PLAN.md,
so the index is written into the package before building:

    uv run --package qutip-trap-app python -m qutip_trap_app.provenance   # writes src/qutip_trap_app/provenance_index.json
    cd qutip_trap_app && uv run --project .. flet build macos              # or windows, linux, web

The GitHub Pages site, https://splch.github.io/qutip-trap/, is the static Pyodide build deployed by
`.github/workflows/pages.yml`: QuTiP 5.3.1 compiled for Pyodide 314 (there is no WebAssembly wheel on PyPI), the core's
wheel, then `flet build web --python-version 3.14 --base-url /qutip-trap/ --route-url-strategy hash`. In the browser the
worker runs in-process (`workers.IN_PROCESS`): the page waits for a request instead of streaming its progress.

The `flet test` flows (`tests/test_main.py`) drive the rendered controls by their keys. They need the Flutter test host (on
macOS the full Xcode) and are skipped unless `QUTIP_TRAP_APP_UI_TESTS=1`:

    cd qutip_trap_app && QUTIP_TRAP_APP_UI_TESTS=1 uv run flet test

## Using the record from Python

```python
from qutip_trap_app.core import Circuit, Operation
from qutip_trap_app.record import execute, job_for_preset
from qutip_trap_app.resim import zoom
from qutip_trap_app.viewmodel.circuit import register_after, timeline
from qutip_trap_app.viewmodel.machine import histogram

bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
job, preset = job_for_preset("yb171_chain", 2, bell, 200, seed=7)
record, live = execute(job, preset)                   # calibrate, run, record
h = histogram(record)                                 # bars with counts and error bars, the target beside them
ms = next(g for g in timeline(record) if g.name.value == "ms")
after = register_after(record, ms.index)              # the Bloch vectors vanish, the purity stays near one
record, z, stats = zoom(record, live, ms.step_index)  # the MS pulse at 201 points per segment, cached on the record
```

Every displayed value is a `Shown(quantity, value)` whose quantity names a ledger record; `provenance.ProvenanceIndex`
turns it into a chip.
