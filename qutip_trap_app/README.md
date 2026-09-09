# qutip-trap-app

The zoomable application of PLAN.md Section 14 (milestone M11): one simulated trapped-ion quantum computer at five levels of
abstraction, from a cloud customer's histogram down to the Hamiltonian terms the solver integrates, built as a learning
tool. The design is [`DESIGN.md`](DESIGN.md). This package consumes the physics core `qutip_trap` through its public API
only (`src/qutip_trap_app/core.py` is the one module that imports it, with the gaps it records); nothing in the core
imports this package.

## Status

**M11.1 done: the run record and the view-models.** No screens yet (they are M11.2 to M11.4); `src/main.py` is still the
`flet create` placeholder. What exists and is tested:

- `record.py` - the run record of Section 14.3 (job, compiled circuit, schedule, space, preparation, noise samples,
  branches, per-(sample, branch) traces, readout records, results with the target beside them, diagnostics, device card,
  calibration table, core gaps) and its storage policy; `execute(job)` runs a job and records it.
- `codec.py`, `storage.py` - export to one zip of JSON plus `.npy` arrays with a digest; bitwise re-import.
- `resim.py` - the engine as `run()` built it; boundary states by chaining over gate steps; zoom into one pulse with a fine
  store, cached by key; tolerance and cap re-checks for the convergence badge.
- `provenance.py` - the index generated from `docs/provenance/ledger.yaml` and `PLAN.md` into
  `src/assets/provenance_index.json`; chips and Part II sections at run time.
- `viewmodel/` - `catalogue` (every displayed quantity with its ledger id), `machine` (Level 0), `circuit` (Level 1),
  `schedule` (Level 2), `dynamics` (Level 3), `numerics` (the panel and badge), `learn` (concepts, prompts, prior-knowledge
  plans, spacing, mastery log, the six-click tour).
- `tests/` - the Section 9.11 rows M11.1 owns: coarse-graining identity, record round trip, re-simulation cache,
  convergence badge, provenance coverage; plus the learning layer's integrity.

## Commands

From the repository root:

    uv sync --all-packages --extra gui --group dev                 # installs the core, the app and Flet
    uv run --package qutip-trap-app pytest qutip_trap_app/tests    # the view-model tests (about three minutes: one Bell run)
    uv run python -m qutip_trap_app.provenance                     # regenerate src/assets/provenance_index.json
    uv run python -m qutip_trap_app.provenance --check             # CI: the asset is current with the ledger and the plan
    (cd qutip_trap_app && uv run mypy)                              # strict types for src/qutip_trap_app
    uv run ruff check qutip_trap_app && uv run ruff format --check qutip_trap_app

The Flet integration tests (`flet test`, the `flet_app` fixture) need the Flutter client; they are skipped unless
`QUTIP_TRAP_APP_UI_TESTS=1` is set:

    cd qutip_trap_app && QUTIP_TRAP_APP_UI_TESTS=1 uv run flet test

Run the placeholder app:

    uv run --package qutip-trap-app flet run qutip_trap_app          # native window
    uv run --package qutip-trap-app flet run --web qutip_trap_app    # browser served from the host

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
