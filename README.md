# qutip-trap

A first-principles trapped-ion quantum computer simulator built on QuTiP. The specification is
[`PLAN.md`](PLAN.md); this repository implements it milestone by milestone (Section 10 of the plan).

**Status: milestones M0 (scaffolding and public interfaces) and M0a (atomic structure layer).** No trap or
gate is simulated yet. What exists is the `uv`-managed project, the units module with the Hz/rad·s convention
enforced by types, the public API of Appendix E frozen as Python dataclasses and protocols (every unimplemented
method raises `NotImplementedError` naming its milestone), the species data tables, the provenance ledger, the
committed check scripts behind every `[recomputed here]` number, the CI definition, and the atomic layer of
Section 4.5: exact 3j/6j algebra, the hyperfine-Zeeman diagonalization with adiabatic labels and clock-point
finder, Wigner-Eckart dipole elements on the uncoupled basis, the Gamma to reduced element to I_sat chain, the
laboratory-to-atomic-frame polarization decomposition, Raman couplings, light shifts and Kramers-Heisenberg
scattering (Raman, Rayleigh, leakage, differential-Rayleigh dephasing) from explicit intermediate-state sums,
and the electric-quadrupole coupling of optical qubits. Its acceptance tests are the anchors of Sections 9.10,
9.13, 9.14 and 9.16 (43Ca+ 146.0942 G, 9Be+ 119.446 G, 25Mg+ 212.78 G, 171Yb+ 310.87 Hz/G^2, the 1/3 : 2/3
branching, I_sat 50.83 mW/cm^2, the Ozeri, Wineland and Uys closed forms, a full master-equation check).

## Layout

| Path | What it is |
|---|---|
| `PLAN.md`, `plan_sources/` | the specification and the section files it is rendered from |
| `qutip_trap/` | the core package (Section 3.2 layout); `qutip_trap/api.py` re-exports the Appendix E surface |
| `qutip_trap/units.py` | CODATA constants and the `Hz` / `RadPerS` and `Gauss` / `Tesla` types (Sections 5.6, 13) |
| `qutip_trap/species/` | one table of cited constants per isotope, the `Species` builder, and the atomic layer (`wigner`, `zeeman`, `dipole`, `polarization`, `raman`, `quadrupole`; `atomic.py` is the facade) |
| `docs/provenance/ledger.yaml` | the provenance ledger of Section 14.5 (one record per quantity) |
| `validation/scripts/` | the check and benchmark scripts of Appendix D with their committed outputs; `run_checks.py` re-runs and compares them |
| `tests/` | pytest suite (API freeze against Appendix E, units, species tables, hashing, seeds, IonQ formats, the Section 9.13/9.14/9.16 atomic anchors) |
| `qutip_trap_app/` | the separate Flet application package of Section 14 (scaffold only until M11) |
| `.github/workflows/ci.yml` | CI: validation scripts first, then lint, type-check, tests, `flet doctor`, convergence-report artifact |

## Commands

    uv sync --all-extras                       # Python 3.13 (see .python-version), pinned in uv.lock
    uv run ruff check . && uv run ruff format --check .
    uv run mypy
    uv run pytest
    uv run python validation/scripts/run_checks.py --report    # re-run the Appendix D check scripts, write validation/report/
    uv run flet doctor                          # the gui extra (Flet) is installed

Ruff 0.16 formats fenced Python blocks inside Markdown, which would rewrite Appendix E of `PLAN.md`; `*.md` is
excluded in `pyproject.toml` so the plan stays byte-identical to its render from `plan_sources/`.

## Conventions that matter from the first line of code

Internal frequencies are angular (rad/s); the public API is in Hz and converts at the boundary with an
explicit 2π (`qutip_trap.units`). Wavelengths are vacuum wavelengths. `Transition.gamma_hz` is the upper
level's total decay rate over 2π, and I_sat uses the angular partial rate. `A_hfs` is signed. Section 13 of
the plan is the full convention table; each module restates the rows it touches in its docstring.

## Species tables

`171Yb+`, `40Ca+` and `88Sr+` build a `Species`; `43Ca+`, `133Ba+`, `137Ba+`, `9Be+` and `25Mg+` carry every
constant the plan or NIST cites and raise `IncompleteSpeciesTable` listing what is still missing. Every
number is a `Cited` value with its source and provenance tag; hyperfine-resolved quantities are never typed in
(they are derived in milestone M0a).
