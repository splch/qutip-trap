# qutip-trap

A first-principles trapped-ion quantum computer simulator built on QuTiP. The specification is
[`PLAN.md`](PLAN.md); this repository implements it milestone by milestone (Section 10 of the plan).

**Status: milestones M0 (scaffolding and public interfaces), M0a (atomic structure layer), M1 (trap and
crystal), M2 (single ion, spin-motion coupling, single-qubit gates) and M3a (multi-level optical-Bloch builder).** The
simulator now evolves one ion with its motional modes through the one Hamiltonian builder of Section 4.3: exact displacement
operators by matrix exponential asserted against the analytic Laguerre elements over the populated range (Section 5.1.1),
cached operators and marginals (ENR included), the boundary monitor with cap-raising retries (Section 5.5), Raman,
single-photon optical (E1 and E2) and microwave drives whose Rabi frequencies, Stark shifts and scattering rates are derived
from the beams and the atomic layer rather than entered (`light/`), frames (Schroedinger-motion default, the
sideband-decomposed interaction picture with `k_max` and `rwa` as declared approximations), the `dop853` -> `vern9` ladder
with the step-density record (Section 5.3), a JOINT_EXACT pulse engine (`JointExactEngine.run_pulses`) that segments a
schedule at pulse boundaries, applies frozen-spectator Debye-Waller factors drawn per shot from the keyed seeds, micromotion
J_0 factors or the rf-locked modulation, crosstalk terms, Stark and anharmonic terms and Cetina's beam-curvature coupling,
single-qubit native gates (GPi, GPi2) as pulses with virtual-RZ frame tracking through the scheduler subset of Section 7.3,
the composite-pulse library of Section 4.3.5, the frequency-comb tone set of Section 4.3.7, Harty's microwave
randomized-benchmarking model, and the Rabi, Ramsey, Ramsey-frequency and sideband-spectroscopy experiments on the engine.

M3a adds the shared multi-level optical-Bloch builder of Section 4.2.8 (`dynamics/multilevel.py`, re-exported by the one
builder module): every fine-structure level's dressed sublevels from the atomic layer, one frame frequency per manifold
assigned along the beams with the inconsistency detector (two tones on one transition, closed beam loops) and the
time-periodic fallback, one collapse operator per decay line and emitted polarization from the dipole operator itself
(the sum rule sum C^dag C = Gamma P_e holds to 1e-16), the recoil kernel of `light/recoil.py` (the vector form over a
direction quadrature that DERIVES every angular factor, its per-q marginal reduction and the second-moment-exact minimal
quadrature), and the leak policies include / sink / renormalize for decay targets outside the included manifold. On top of
it `light/bloch.py` is the single scattering-rate object of Section 13: `steadystate` (direct solve, or the fixed point of
the one-period propagator with period averaging when the frame graph is inconsistent or a polarization is modulated), the
photon rate per line asserted against the equal-population ceiling n_e/(n_e + n_g), the slow-manifold coarse graining that
yields R_o in the conditional bright state with the pumping rates R_d and R_b, dark-state counts from the coupling matrix,
optical pumping by the exact exponential of the Liouvillian, W(Delta) as a function of detuning, the level-A/B coefficients
A_+- = W(Delta -+ nu) + (eta~^2/eta^2) W(Delta) with a weak-drive guard, and the semi-analytic A_+- = 2 Re[S(-+ nu) + D]
from the dipole-force spectrum. `prep/level_c.py` is the one-mode level-C cooling solve (steady state, relaxation rate
from the Liouvillian's slowest eigenvalue, `mesolve` traces) and the level-B Fock rate equation. Acceptance tests
(`tests/test_bloch.py`, `test_recoil.py`, `test_level_c.py`): the RMP Eq. 96 Lorentzian to 1e-9; the 171Yb+ detection
rate (Gamma/18) s_o/[1 + (2/9) s_o] recovered to 0.5 % at its optimum field with the Gamma/4 ceiling; the leakage
prefactors from the angular algebra settling the Noek-Crain factor in Noek's favour, R_d = (2/27)(Gamma/2) s_o
(Gamma/2 Delta_HFP)^2, and R_b/R_d = 3/49; Berkeland's dark-state counts with the sigma+ unit test of Section 13; three
photons per optical pump into |0>; the Floquet fixed point against the secular static model; the 40Ca+ S-P-D dark
resonance; Stenholm's sideband floor (Gamma/2 nu)^2 [alpha/cos^2 theta_L + 1/4] from level C with all three recoil
discretizations (alpha -> 0 leaving (Gamma/4 nu)^2), the mixed pi + sigma channels needing one alpha per channel, the
Doppler limit (Gamma/4 nu)(1 + alpha) at Delta = -Gamma/2, the phonon rate equation to 1e-12, and Morigi's EIT figure.
Two-ion entangling gates (M4), the cooling stages themselves (M3) and readout (M5) are not simulated yet.

Plan inconsistencies surfaced by M3a (recorded in the ledger as `anchor.m3a.*`): the four-level 171Yb+ steady state
reproduces (Gamma/18, 2/9) as a CEILING to 0.15-0.5 % at the best destabilizing field, not "to nine digits"; the
W(Delta -+ nu) closed form must be fed the unsaturated scattering rate (at Omega = Gamma/2 it is off by exactly 1 + s), which
Section 4.2.8 (vii) states and the code now guards; the "Morigi Fig. 3" fixture of Sections 4.2.8 and 13 (nu = 2.0068,
Omega_1 = Omega_2 = 17 MHz, Delta = 70 MHz) is not the caption of Morigi, Eschner and Keitel 2000, which reads
Omega_r = gamma, Omega_g = gamma/20, nu = gamma/10, eta = 0.145, Delta = 2.5 gamma with 99 % ground-state occupation
(reproduced); the 935 nm repump upper level (3D[3/2]1/2) has no tabulated lifetime or branching (Olmschenk 2007 cites
Bell et al. 1991 for it), so the 171Yb+ detection model folds the 0.501 % D3/2 branch back or routes it to a sink until
that constant is cited.

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
| `qutip_trap/dynamics/` | the ONE builder `hamiltonian.build_hamiltonian` (H_mot + H_int + drives with exact D + Stark + anharmonic + curvature, frames, micromotion, crosstalk, frozen Debye-Waller), `frames` (virtual-Z `PhaseFrame`, the sideband decomposition), `evolve` (sesolve/mesolve with the dop853 -> vern9 ladder), `engine` (`JointExactEngine`, the Appendix E protocol), `channels` (heating, dephasing, Rayleigh collapse operators), `multilevel` (the multi-level mode of Section 4.2.8: manifold frames with the inconsistency detector, per-polarization collapse operators, recoil kernels, leak policies; M3a) |
| `qutip_trap/light/` | drives derived from beams: `raman` (two-photon Rabi frequency, Delta k, eta per mode, Stark shift, scattering budget, crosstalk ratios), `microwave` (magnetic-dipole Rabi frequency, ac Zeeman shift), `stark`, `scattering`, `comb` (Section 4.3.7 tone set, comb factor, guards), `beams`, `bloch` (the scattering-rate object of Section 13: steady state, Floquet fixed point, slow-manifold rates, dark states, pumping evolution, A_+- suppliers; M3a), `recoil` (the emission kernel of Section 4.2.8; M3a) |
| `qutip_trap/prep/` | `level_c` (the one-mode level-C cooling solve and the level-B Fock rate equation; M3a); the cooling stages are M3 |
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
