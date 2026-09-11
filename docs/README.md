# qutip-trap documentation

The specification is [`PLAN.md`](../PLAN.md) at the repository root (Part II is the physics, Section 9 the validation suite,
Section 13 the conventions, Appendix E the public API). These pages are the release documentation of milestone M10:

- [physics_notes.md](physics_notes.md): the equations the simulator integrates, each with the plan section that specifies
  it, the module that implements it and the provenance record behind it; the appendix lists every numerical anchor the
  tests pin, by milestone (generated from the ledger).
- [conventions.md](conventions.md): the one convention per quantity the code enforces, and the alternatives it rejects
  (the table generated from the ledger).
- [examples.md](examples.md): runnable examples, from a device to a circuit's histogram, the calibration and experiment
  APIs, and the benchmarks with their error budget.
- [limits.md](limits.md): the scope of the first release, the approximations a run can make and how each is reported, what
  the physics leaves as inputs, and the compute cost that bounds exact simulation.
- [provenance/ledger.yaml](provenance/ledger.yaml): the provenance ledger of Section 14.5, one record per quantity, with the
  fields `id, symbol, tag, section, source, equation, corrected_form`; the species block is generated from the species
  tables by `tools/ledger_from_tables.py`.

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
- The public surface is enumerated by `tests/test_public_surface.py`: every name a rung module exports (today
  `qutip_trap.api`) must resolve, carry a docstring of its own and appear in backticks on one of these pages; the names of
  the 0.1.0 surface that still lack a line are listed in the test as a set that may only shrink. `tests/test_docs.py` also
  checks that the bit-order sentence is stated exactly once (on the conventions page) and never contradicted by a
  docstring, and that the tensor order of the native gate matrices is stated once, in `control/native.py`.
- The README at the repository root carries the milestone-by-milestone account of what was built and what each milestone's
  tests established, including the plan inconsistencies each milestone surfaced.
