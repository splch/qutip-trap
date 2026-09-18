# qutip-trap documentation

The specification is [`PLAN.md`](../PLAN.md) at the repository root (Part II is the physics, Section 9 the validation suite,
Section 13 the conventions, Appendix E the public API). Since 0.4.0 these pages are ordered as the ladder of
[api_proposal.md](api_proposal.md): one page per rung, each listing every public name of its module, so that a reader
climbs down from the machine to the physics one page at a time.

- [examples.md](examples.md): the two-minute quickstart on the machine, then runnable examples down the ladder (a device,
  a calibrated run, a job in the background, an experiment, the benchmarks, the error model, a second species), executed
  in order by the test suite.
- [machine.md](machine.md): rung 0, `qutip_trap` itself. The `Machine`, the option objects and the level, the `Circuit`
  in and the `Result` out, what a run leaves behind, the jobs (`Machine.submit`, `Job`, `RunSpec`) and the presets.
- [circuit.md](circuit.md): rung 1, `qutip_trap.circuit` and `qutip_trap.io`. The IR, the compiler, the native gate
  matrices and the two wire formats (OpenQASM 2, IonQ JSON).
- [schedule.md](schedule.md): rung 2, `qutip_trap.schedule`. Pulses, drives and tones, the beam roles, the calibration
  table, the electronics, the waveforms and their closure, the composite pulses and the comb.
- [dynamics.md](dynamics.md): rung 3, `qutip_trap.dynamics`. The engine and its traces, the Hamiltonian builder, the
  spaces, the channels and the noise sample, the solvers, the multi-level builder, the gate channels and the readout
  models.
- [physics.md](physics.md): rung 4, `qutip_trap.physics`. The species, the trap and the crystal, the light and the field,
  the noise, detection and preparation, the published models, and the `Device` with its derived numbers.
- [laboratory.md](laboratory.md): `qutip_trap.experiments`, `qutip_trap.calibration` and `qutip_trap.benchmarks` on the
  machine, with the typed results, the calibration methods and the error model.
- [experimental.md](experimental.md): `qutip_trap.experimental`, the names outside the stability guarantee (the M12
  transport records, the tomography internals, two oracles).
- [conventions.md](conventions.md): the units, the bit order, the tensor order and the seed contract first, then the
  vocabulary and the one convention per quantity the code enforces (the table generated from the ledger).
- [physics_notes.md](physics_notes.md): the equations the simulator integrates, each with the plan section that specifies
  it, the module that implements it and the provenance record behind it; the appendix lists every numerical anchor the
  tests pin, by milestone (generated from the ledger).
- [limits.md](limits.md): the scope of the release, the approximations a run can make and how each is reported, what the
  physics leaves as inputs, and the compute cost that bounds exact simulation.
- [deprecations.md](deprecations.md): the deprecation policy, every entry "deprecated in / removed in", and the removed
  names.
- [provenance/ledger.yaml](provenance/ledger.yaml): the provenance ledger of Section 14.5, one record per quantity, with the
  fields `id, symbol, tag, section, source, equation, corrected_form`; the species block is generated from the species
  tables by `tools/ledger_from_tables.py`.
- [schemas/result.schema.json](schemas/result.schema.json), [schemas/device.schema.json](schemas/device.schema.json) and
  [schemas/runspec.schema.json](schemas/runspec.schema.json): the JSON schemas of `Result.to_dict()`, `Device.to_dict()`
  and `RunSpec.to_dict()`, written by `tools/schemas.py` and checked in CI.
- [api_proposal.md](api_proposal.md) and [api_implementation_plan.md](api_implementation_plan.md): the survey of the field's
  SDK shapes and the phased plan the ladder followed (0.2.0 the executor, 0.3.0 the options and the laboratory, 0.4.0 the
  jobs, the record and these pages).

## How the documentation is kept true

- The two generated tables are rewritten by `uv run python tools/docs_from_ledger.py` from the ledger, and CI runs it
  with `--check`, so a convention or anchor added to the ledger without regenerating the pages fails the build.
- The examples are executed in order by `tests/test_docs.py` (a slow test, about three minutes), so a signature change
  that breaks a documented call fails the build.
- Every number quoted in the prose of these pages comes from a committed check script under `validation/scripts/`
  (`check_benchmarks.py` for the benchmark numbers, `check_circuits.py` for the Bell and GHZ circuits, the timing
  benchmarks for the cost table), whose outputs CI re-runs. What CI *compares* is every numeric token of every line
  except the ones a script prefixes `MC:`, which `run_checks.py` skips because their last digits depend on the platform's
  libm: a Monte-Carlo number — a fitted p or r, a GHZ bound, a heavy-output probability — is therefore reproduced but not
  compared digit for digit. Each headline Monte-Carlo number of `check_benchmarks.py` also gets a compared `pinned` line
  that states only the band the number must stay inside; the verdict word is part of that line, so a number leaving its
  band no longer matches the committed output and CI fails.
- The public surface is enumerated by `tests/test_public_surface.py`: every name a rung module exports must resolve, carry
  a docstring of its own and appear in backticks on its rung's page (the Appendix E surface, `qutip_trap.api`, on any
  page); since 0.4.0 no name is exempt. `tests/test_docs.py` also checks that the bit-order sentence is stated exactly once
  (on the conventions page) and never contradicted by a docstring, and that the tensor order of the native gate matrices
  is stated once, in `control/native.py`.
- The README at the repository root is the front page; [OLD_README.md](OLD_README.md) keeps the milestone-by-milestone
  account of what was built and what each milestone's tests established.
