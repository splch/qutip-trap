# qutip-trap

A first-principles trapped-ion quantum computer simulator built on QuTiP. The specification is
[`PLAN.md`](PLAN.md); this repository implements it milestone by milestone (Section 10 of the plan).

**Status: milestones M0 (scaffolding and public interfaces), M0a (atomic structure layer) and M1 (trap and
crystal).** No gate is simulated yet. What exists is the `uv`-managed project, the units module with the Hz/rad·s
convention enforced by types, the public API of Appendix E frozen as Python dataclasses and protocols (every
unimplemented method raises `NotImplementedError` naming its milestone), the species data tables, the provenance
ledger, the committed check scripts behind every `[recomputed here]` number, the CI definition, the atomic layer of
Section 4.5 (exact 3j/6j algebra, hyperfine-Zeeman diagonalization with adiabatic labels and clock-point finder,
Wigner-Eckart dipole elements, the Gamma to reduced element to I_sat chain, polarization decomposition, Raman
couplings, light shifts, Kramers-Heisenberg scattering, electric-quadrupole coupling), and the trap layer of
Section 4.1: the Mathieu equation by the monodromy method (scalar and the coupled 3 x 3 system) with the Floquet
function under the Wronskian normalization that yields the micromotion factor C0, the gapless-plane
surface-electrode model (strip complex potentials and House's rectangle potential, rf null, escape-point saddle,
depth, principal axes, the map to Mathieu parameters, House's Eq. 30 minimum-norm dc solution), Berkeland's rod-trap
map, the ion crystal (James equilibrium by Newton iteration, the mass-weighted eigenproblem for mixed species, three
mode families in the canonical Appendix E order, the zigzag refusal, per-ion per-mode Lamb-Dicke parameters with C0
applied exactly once), excess micromotion indices, heating rates from single-sided S_E with the mode-projected
correlation model and the micromotion-sideband sum, and the cubic and quartic Coulomb mode couplings with a
three-mode resonance checker. Acceptance tests are the anchors of Sections 9.1, 9.10, 9.12, 9.13, 9.14, 9.16 and 9.17
(among them C0 = 1.001890/1.007741/1.018161, the House five-wire fixture, Nizamani's q_N versus q, James's spectrum
and the zigzag ratios, Home's Be+/Mg+ table, the 171Yb+ Lamb-Dicke triple 0.1103/0.0493/0.0268, the 40 quanta/s heating
round trip, D_222 = -1.1225).

Three plan inconsistencies surfaced by M1 are recorded in the ledger (`anchor.trap.house_five_wire`,
`anchor.trap.be9_infinite_chain`, `anchor.trap.marquet_selection_rules`): the "4.077 MHz Floquet" of Section 9.13
is the preprint closed form, not the exact exponent (4.080 MHz); the 7.806 MHz 9Be+ zigzag example uses m = 9 u;
and the leading sign of the cubic Coulomb term printed in Section 4.1.4 does not match its own D_222 = -1.1225.

## Layout

| Path | What it is |
|---|---|
| `PLAN.md`, `plan_sources/` | the specification and the section files it is rendered from |
| `qutip_trap/` | the core package (Section 3.2 layout); `qutip_trap/api.py` re-exports the Appendix E surface |
| `qutip_trap/units.py` | CODATA constants and the `Hz` / `RadPerS` and `Gauss` / `Tesla` types (Sections 5.6, 13) |
| `qutip_trap/species/` | one table of cited constants per isotope, the `Species` builder, and the atomic layer (`wigner`, `zeeman`, `dipole`, `polarization`, `raman`, `quadrupole`; `atomic.py` is the facade) |
| `qutip_trap/trap/` | the trap layer of Section 4.1: `mathieu` (monodromy, Floquet function, C0), `pseudopotential` (rf drive, geometry-free maps), `surface` (gapless-plane electrodes), `crystal` (equilibrium, mass-weighted modes, Lamb-Dicke), `micromotion`, `heating`, `anharmonic`; `model.py` is the `Trap` record |
| `docs/provenance/ledger.yaml` | the provenance ledger of Section 14.5 (one record per quantity) |
| `validation/scripts/` | the check and benchmark scripts of Appendix D with their committed outputs; `run_checks.py` re-runs and compares them |
| `tests/` | pytest suite (API freeze against Appendix E, units, species tables, hashing, seeds, IonQ formats, the atomic anchors of Sections 9.13/9.14/9.16, the trap and crystal anchors of Sections 9.1/9.10/9.12/9.13/9.17) |
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
