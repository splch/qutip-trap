# qutip-trap

A first-principles trapped-ion quantum computer simulator built on QuTiP. The specification is
[`PLAN.md`](PLAN.md); this repository implements it milestone by milestone (Section 10 of the plan).

**Status: milestones M0 (scaffolding and public interfaces), M0a (atomic structure layer), M1 (trap and
crystal) and M2 (single ion, spin-motion coupling, single-qubit gates).** The simulator now evolves one ion with its
motional modes through the one Hamiltonian builder of Section 4.3: exact displacement operators by matrix exponential
asserted against the analytic Laguerre elements over the populated range (Section 5.1.1), cached operators and
marginals (ENR included), the boundary monitor with cap-raising retries (Section 5.5), Raman, single-photon optical (E1
and E2) and microwave drives whose Rabi frequencies, Stark shifts and scattering rates are derived from the beams and
the atomic layer rather than entered (`light/`), frames (Schroedinger-motion default, the sideband-decomposed
interaction picture with `k_max` and `rwa` as declared approximations), the `dop853` -> `vern9` ladder with the
step-density record (Section 5.3), a JOINT_EXACT pulse engine (`JointExactEngine.run_pulses`) that segments a schedule
at pulse boundaries, applies frozen-spectator Debye-Waller factors drawn per shot from the keyed seeds, micromotion
J_0 factors or the rf-locked modulation, crosstalk terms, Stark and anharmonic terms and Cetina's beam-curvature
coupling, single-qubit native gates (GPi, GPi2) as pulses with virtual-RZ frame tracking through the scheduler subset
of Section 7.3, the composite-pulse library of Section 4.3.5 (SK1, BB1, NB1, PB1, the P2j/N2j/B2j ladder, CORPSE,
SCROFULOUS, CinSK, CinBB, Mount's PD6, the Low-Yoder-Chuang certificate), the frequency-comb tone set of Section
4.3.7, Harty's microwave randomized-benchmarking model, and the Rabi, Ramsey, Ramsey-frequency and
sideband-spectroscopy experiments on the engine. Everything before M2 is as described below. Acceptance tests are the
anchors of Sections 9.2, 9.6, 9.10, 9.12, 9.13, 9.15 and 9.16 (among them the Section 5.1.1 displacement table, the
exact Rabi matrix elements to rtol 1e-10, the Debye-Waller identity 0.9851119396031, picture equivalence to 1.5e-8, the
RZ(0.1)-then-GPi2(0) sequence, Harty's 0.81(14)e-6 reproduced as 0.77(11)e-6, Mount's PD6 3.713e-11, every Section
9.15 comb number, Cetina's -2.6424e12 m^-2 and 17.19 nm). Two-ion entangling gates (M4), cooling (M3) and readout
(M5) are not simulated yet.

Plan inconsistencies surfaced by M1 and M2 are recorded in the ledger rather than absorbed (`anchor.trap.*`,
`anchor.m2.*`): the "4.077 MHz Floquet" of Section 9.13 is the preprint closed form, not the exact exponent (4.080 MHz);
the 7.806 MHz 9Be+ zigzag example uses m = 9 u; the leading sign of the cubic Coulomb term printed in Section 4.1.4
does not match its own D_222 = -1.1225; the distance-to-infidelity conversion "E^2/2" of Sections 4.3.5 and 13 is the
1 - F_C form (1 - F_K needs E^2); Section 9.12's Debye-Waller "n = 1 start gives 0.4662967" belongs to its second case
(eta^2 = 0.09, nbar = 1.7) and the thermal mean is exactly exp[-eta^2(nbar + 1/2)]; Section 9.10's target-first versus
target-last composite-pulse numbers under detuning (4.33e-3 vs 1.84e-3) are not reproduced; Harty's identity-gate
timing is ambiguous in the source and the one-delay-per-replaced-pulse reading is the one that reproduces the budget.

## Layout

| Path | What it is |
|---|---|
| `PLAN.md`, `plan_sources/` | the specification and the section files it is rendered from |
| `qutip_trap/` | the core package (Section 3.2 layout); `qutip_trap/api.py` re-exports the Appendix E surface |
| `qutip_trap/units.py` | CODATA constants and the `Hz` / `RadPerS` and `Gauss` / `Tesla` types (Sections 5.6, 13) |
| `qutip_trap/species/` | one table of cited constants per isotope, the `Species` builder, and the atomic layer (`wigner`, `zeeman`, `dipole`, `polarization`, `raman`, `quadrupole`; `atomic.py` is the facade) |
| `qutip_trap/trap/` | the trap layer of Section 4.1: `mathieu` (monodromy, Floquet function, C0), `pseudopotential` (rf drive, geometry-free maps), `surface` (gapless-plane electrodes), `crystal` (equilibrium, mass-weighted modes, Lamb-Dicke), `micromotion`, `heating`, `anharmonic`; `model.py` is the `Trap` record |
| `qutip_trap/hilbert/` | the composite space of Section 5.1: `operators` (analytic Laguerre elements, expm displacement, the Section 5.1.1 tolerance and margin fixture, Debye-Waller factors, qubit operators in the computational ordering), `space` (cached operators, marginals with the ENR index sums, state constructors), `truncation` (boundary monitor, cap growth, the tolerance-tightening test) |
| `qutip_trap/dynamics/` | the ONE builder `hamiltonian.build_hamiltonian` (H_mot + H_int + drives with exact D + Stark + anharmonic + curvature, frames, micromotion, crosstalk, frozen Debye-Waller), `frames` (virtual-Z `PhaseFrame`, the sideband decomposition), `evolve` (sesolve/mesolve with the dop853 -> vern9 ladder), `engine` (`JointExactEngine`, the Appendix E protocol), `channels` (heating, dephasing, Rayleigh collapse operators) |
| `qutip_trap/light/` | drives derived from beams: `raman` (two-photon Rabi frequency, Delta k, eta per mode, Stark shift, scattering budget, crosstalk ratios), `microwave` (magnetic-dipole Rabi frequency, ac Zeeman shift), `stark`, `scattering`, `comb` (Section 4.3.7 tone set, comb factor, guards), `beams` |
| `qutip_trap/control/` | native gates, the circuit IR, `pulses` (Tone/Drive/Pulse), `schedule` (single-qubit gates as pulses with virtual-RZ tracking, M2 subset), `composite` (Section 4.3.5 library) |
| `qutip_trap/experiments/` | `rabi_scan`, `ramsey`, `ramsey_frequency`, `sideband_spectroscopy` on the engine (M2); the rest is M8 |
| `qutip_trap/validation/` | closed forms used as test oracles (`atomic_closed_forms`, `spin_motion_closed_forms`, Harty's RB model `harty_rb`) |
| `docs/provenance/ledger.yaml` | the provenance ledger of Section 14.5 (one record per quantity) |
| `validation/scripts/` | the check and benchmark scripts of Appendix D with their committed outputs; `run_checks.py` re-runs and compares them |
| `tests/` | pytest suite (API freeze against Appendix E, units, species tables, hashing, seeds, IonQ formats, the atomic anchors of Sections 9.13/9.14/9.16, the trap and crystal anchors of Sections 9.1/9.10/9.12/9.13/9.17, the M2 spin-motion, composite-pulse, comb, native-pulse, Harty RB and experiment tests) |
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
