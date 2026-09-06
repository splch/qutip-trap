# qutip-trap

A first-principles trapped-ion quantum computer simulator built on QuTiP. The specification is
[`PLAN.md`](PLAN.md); this repository implements it milestone by milestone (Section 10 of the plan).

**Status: milestones M0 (scaffolding and public interfaces), M0a (atomic structure layer), M1 (trap and
crystal), M2 (single ion, spin-motion coupling, single-qubit gates), M3a (multi-level optical-Bloch builder), M3 (cooling
and state preparation) and M4 (two-ion entangling gates).** The
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
M3 adds the cooling and preparation stages of Section 4.2 on top of the M3a builder (`prep/`). `light/recoil.py` gained the
per-ion participation and the joint multi-mode kick: the emitted photon's Lamb-Dicke parameter eta_em,{i,m} = k |c_{i,m}| x0_{i,m}
with the ion's own mass, the product of displacements over the modes for one sampled direction, and the recoil-energy identity
(the modes of one axis family share exactly alpha_axis (hbar k)^2/(2 m_i), to 1e-16, for single and mixed crystals). `prep/rates.py`
is the level-A engine every stage shares: per illuminated ion and mode the phonon heating and cooling rates c^2 [S(+-nu) + 2D]
from the dipole-force spectrum of the ion's internal steady state under all its beams (exact in the saturation) with the
emission-recoil diffusion per channel, the participation folded into the zero-point length so the rates carry W_k = sum_i c^2
and nbar is participation independent, and the guard that raises for a mode no cooling beam addresses. `prep/doppler.py`
produces nbar_D per mode at any nu/Gamma, optimizes one detuning per beam group over the participation-weighted mode set and
reports the RMP force model only where nu/Gamma < 0.1. `prep/sideband.py` holds the resolved-sideband and Raman closed forms
(Stenholm's floor with Marzoli's effective two-level parameters, the saturating rate), the exact pulsed transfer matrices with
higher sidebands and the three schedule optimizations, the Laguerre nodes where single-order cooling strands population, the
repump recoil kernel, and the thermometry of Section 4.2.7 (the exact sideband ratio, Rasmusson's tail sums, blue-sideband
flopping inversion, the double-thermal fit). `prep/eit.py` is Section 4.2.3 in the plan's sign with the Zeeman-resolved level-C
model; `prep/polarization_gradient.py` carries Joshi's analytic j = 1/2 <-> 1/2 model (with the static-gradient raise) and builds
the Lindblad layer through the M3a builder (`PolGradientBeams.xi/limits/moving_gradient_ok` now work); `prep/pumping.py`
evolves the pump from the scrambled manifold and returns the preparation error, duration, photon count and per-mode recoil
heating; `prep/sequence.py` enforces the Doppler -> sideband/EIT -> final-pump order and hands off the Appendix E `State`.
Acceptance tests (`tests/test_recoil_modes.py`, `test_cooling_closed_forms.py`, `test_doppler.py`, `test_sideband_cooling.py`,
`test_eit.py`, `test_polarization_gradient.py`, `test_pumping.py`, `test_prep_sequence.py`): the Doppler fork 0.480/0.336 mK,
the force model's 349.5 against 350, Stenholm's 7/12 and 13/20, the Morigi-Walther steady-state list, Marzoli's 0.55/5.5 and
-0.058, Cirac's 0.554 and 2.66, Che's 0.031117 repump recoil, the Laguerre nodes 39.79/71.77/112.29/202.01, Che's accumulation
centres and Rasmusson's fixed-pulse prediction, the Turchette ratio to 1e-13, the EIT fixture 0.005102 with its negative controls
and the RMP tuning 0.00361702, the PGC minima 1/2 and 0.8693 with alpha = 1/3, the 171Yb+ pump's three photons and 8.5e-3 quanta
of recoil, and the level-C checks of the Doppler limit, the PGC limit (0.756 against 0.75) and the multi-ion participation
weights.

M4 adds the two-qubit entangling gates of Section 4.4 on the same builder: nothing is a separate gate implementation, an MS
pulse is one `Drive` per ion carrying the red and blue tones (the coefficient already summed tones), and the exact dynamics
(carrier, every sideband, Debye-Waller factors, spectator modes) come with it. `control/shaping.py` holds the Section 4.4.3
integrals with the symmetrized two-body kernel in two forms that are never mixed (the exact first-order Lamb-Dicke kernel
cos(mu t - phi_m) e^{i omega t}, Choi's sine form being its phi_m = pi/2 case, and the slow-envelope kernel), analytic for
segmented pulses (F_k, the triangle T_k and Im F_l F_k^*, checked against scipy quadrature to 1e-9) and by Simpson's rule for
sampled ones, and the three solvers in the plan's pi/4 convention: Zhu/Choi's segmented AM (2N + 1 segments close N modes,
the power-optimal null-space direction), Leung's vertex-parameterized FM with the time-averaged-trajectory robustness cost,
and Blumel's Fourier-sine AM with K-th-order frequency-derivative stabilization and the extended-null-space relaxation.
`Waveform.symmetric` is the square bichromatic pulse (eta Omega/eps = 1/(2 sqrt K) on one mode, spectators included in chi);
`Waveform.chi_m` is signed; the scheduler plays `ms` and `zz` from the table (MS(phi_0, phi_1, theta) sets the spin phases
phi_i - pi/2 because the force acts about phi_s + pi/2, a positive kernel sign is a pi on the second ion, partial angles
rescale by the s^2 law; ZZ on an MS waveform is the inferred GPi2-wrapper construction, on a light-shift waveform the
spin-echo pair of Section 4.4.4). The `light_shift` drive kind derives Zhu's sigma_z force from the atomic layer (the
two-photon self-couplings of the two qubit levels, weights differing by exactly 2, the spin-independent force and the
far-off-resonant Raman coupling kept). `calibration/entangling.py` is Section 7.5's spot check: the closed-form waveform
played through the JOINT_EXACT engine, chi read from P_11 = sin^2 chi (or in the x basis through the echo pair for a sigma_z
force), the residual displacement read exactly as the final mode energy (sum_j |alpha_j|^2), the amplitude corrected by
sqrt(chi_target/chi) until |chi| = pi/4, and the thermal robustness curve; `ms_scan` and `parity_scan` are the laboratory's
population and parity scans on the engine. Acceptance tests (`tests/test_shaping.py`, `test_two_qubit_gates.py`,
`test_entangling_schedule.py`, `test_light_shift_gate.py`, `test_gate_calibration.py`; `validation/scripts/check_two_qubit.py`):
the check_ms_closure.py anchors through the builder (concurrence 0.9999 with populations (0.5074, 0, 0, 0.4926) at eta Omega/eps
= 1/2, 0.3825 at 1/4), the exact propagator D(alpha S_y) exp[i gamma S_y^2] and Kirchmair's thermal envelopes at first order in
eta, the Debye-Waller law and the three thermal references (n0 recovered, the re-optimized one excluded), the symmetrized kernel
against block-diagonal exact integration (the printed factor 2 wrong by tens of percent for non-proportional envelopes), the
residual-displacement conversions (eps_ent = final mode energy, 1 - F_ent, Landsman's 4/5), the two-mode 171Yb+ gate (the
symmetric pulse's open rocking-mode loop against sum |alpha|^2, the five-segment AM closure, exact chi 0.772 against the
surrogate 0.785 corrected in two checks to 0.785398 with Bell fidelity 0.99977), Ballance's alpha_K = 11/16, 19/64, 35/256 from
block Liouvillians, Choi's 11 and 21 segments for five ions with one and two transverse families, Leung's FM robustness and
Blumel's stabilization, Baldwin's echo diag(1, i, i, 1), the 40Ca+ optical-qubit light-shift ZZ gate from first principles
(fidelity above 0.99 after calibration), and the native MS(phi_0, phi_1, theta), partial angles, virtual-Z frames and the
IonQ JSON path through `schedule`. Readout (M5) is not simulated yet; `run.prepare` waits for M6, where the Device carries the
preparation recipe; the microwave-gradient drive of Section 4.4.5 exists as closed forms (Srinivas's J_2 factor and the
IDD ratio 0.6012) but not as a device-level drive.

Plan inconsistencies surfaced by M4 (recorded in the ledger as `conv.motion_phase_default` and `anchor.m4.*`): Section
4.4.3's sine beat-note convention (tone phases differing by pi, force zero at t = 0) is a convention for the closed forms only:
played literally, the carrier's frame rotation acquires the mean 2 Omega/mu and tilts the entangling axis by Roos 2008's
psi = (4 Omega_Roos/delta) sin zeta = 0.202 rad on the check_ms_closure.py fixture, leaking 3.8% into |du>, |ud> (concurrence
0.9597 against 0.9999 with equal tone phases), so the played waveforms default to equal tone phases; Zhu's printed thermal
infidelity sum beta (|alpha_j|^2 + |alpha_n|^2)/4 is one quarter of what direct integration gives for the |dd> input with
Choi's alpha (the plan's "a factor 1/4 below eps_ent for the same displacements" needs Zhu's alpha to be twice Choi's);
Hughes's printed theta_g ~ int (Omega_g^2 + alpha_dot^2)/delta dt is twice the angle of the displayed Hamiltonian
(int Omega_g^2/(2 delta) dt = pi K (Omega_g/delta)^2 for the square pulse); Baldwin's diag(1, i, i, 1) is one detuning side
(the other gives its conjugate); Choi's 190 us / 9 segments for five ions closes four modes, not five; Ballance's 0.686 is
alpha_1 at t_g/tau = 1e-3, not the coefficient 0.6875; and Ballance's thermal error (pi^2/4) eta^4 nbar (2 nbar + 1) is the
entangling-angle part only: the exact n-dependent sideband coupling also leaves residual spin-motion entanglement of the same order
in eta^4 and linear in n (the Fock-state loss is close to (pi^2/4) eta^4 n(n + 1)), so the exact thermal curve of an n = 0-calibrated
gate lies 1.05 to 1.5 times above the printed form at nbar <= 2, the excess shrinking with nbar.

Plan inconsistencies surfaced by M3 (recorded in the ledger as `anchor.m3.*`): Monroe 1995's 'theoretical 0.484' is the
semiclassical force model with isotropic emission at Delta = -30 MHz, outside that model's nu << Gamma regime, and one oblique
beam does not reproduce his measured triple (0.47, 0.30, 0.18) at any saturation, so the Section 4.2.1 acceptance test needs
the D1-D3 beam geometry; the 40Ca+ 'two-level estimate fails because of the multi-level structure' is not reproduced (the
eight-state model is within 10 % of the S-P model at the plan's parameters); Section 9.13 row 123 prints half the A_+- of
Section 4.2.3's own formula for the EIT fixture (nbar unaffected, W halved); Section 9.12's Lechner rate ratio is inverted
(the higher mode cools faster, 3.2 against the measured 3.4); Rasmusson's 0.06 after 50 optimized pulses is not reached
(0.11-0.12 with independent durations) and his 'about 0.3 quanta stranded above n = 112' is 0.1 for the thermal tail; Joshi's
twelve-operator kernel is nine in the Wigner-Eckart form (one operator per emitted polarization and recoil class, the two pi
decays summed) with identical second moments; the analytic polarization-gradient W at Joshi's operating point is 9.9e4 s^-1
against the quoted 6.6e4 (the window W < delta < omega_z holds either way).


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
| `qutip_trap/prep/` | the cooling and preparation stages of Section 4.2 (M3): `closed_forms` (the level-A oracles), `rates` (per-ion, per-mode level-A rates with participation), `doppler`, `sideband` (continuous and pulsed schedules, thermometry), `eit`, `polarization_gradient` (Joshi's analytic model and the Lindblad layer), `pumping`, `sequence` (stage order and the `State` hand-off), `level_c` (the one-mode level-C solve and the level-B Fock rate equation; M3a) |
| `qutip_trap/control/` | native gates, the circuit IR, `pulses` (Tone/Drive/Pulse, the `light_shift` kind and its couplings), `schedule` (single-qubit gates as pulses with virtual-RZ tracking; `ms`/`zz` from the table's waveforms with the wrapper and echo constructions, M4), `shaping` (the Section 4.4.3 integrals and the AM/FM/Fourier solvers, M4), `table` (`Waveform.symmetric`, segments with callable amplitudes and detunings), `composite` (Section 4.3.5 library) |
| `qutip_trap/calibration/` | `entangling` (the exact spot-check calibration of the entangling angle, the thermal robustness curve, the light-shift echo schedule; M4); `calibrate()` itself is M8 |
| `qutip_trap/experiments/` | `rabi_scan`, `ramsey`, `ramsey_frequency`, `sideband_spectroscopy` (M2), `ms_scan`, `parity_scan` (M4) on the engine; the rest is M8 |
| `qutip_trap/validation/` | closed forms used as test oracles (`atomic_closed_forms`, `spin_motion_closed_forms`, `two_qubit_closed_forms`, Harty's RB model `harty_rb`) |
| `docs/provenance/ledger.yaml` | the provenance ledger of Section 14.5 (one record per quantity) |
| `validation/scripts/` | the check and benchmark scripts of Appendix D with their committed outputs; `run_checks.py` re-runs and compares them |
| `tests/` | pytest suite (API freeze against Appendix E, units, species tables, hashing, seeds, IonQ formats, the atomic anchors of Sections 9.13/9.14/9.16, the trap and crystal anchors of Sections 9.1/9.10/9.12/9.13/9.17, the M2 spin-motion, composite-pulse, comb, native-pulse, Harty RB and experiment tests, the M3a Bloch and recoil tests, the M3 cooling and preparation tests, the M4 shaping, two-qubit gate, scheduler, light-shift and calibration tests) |
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
