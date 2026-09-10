# Limits: what the simulator does not do, and what it approximates

qutip-trap integrates the physics of a trapped-ion quantum computer from the device parameters up, and it says on every
result which approximations it made (`Result.diagnostics.approximations`, the mode classes, the truncation record, the
fidelity level). This page collects the boundaries in one place: the scope the first release fixes ([`PLAN.md`](../PLAN.md)
Section 1.3), the approximations a run can make and how each is reported (Sections 5.2 to 5.5), what the physics layers leave
as user inputs (Section 12), and the compute cost that bounds what exact simulation can reach (Section 11).

## Scope of the first release

- **One crystal in one zone.** Shuttling, splitting, merging and junctions are specified (Section 4.6, milestone M12) and the
  interfaces exist (`Transport`, `Zone`, `VoltageWaveform`, the closed-form transport budget of `transport/`), but no dynamics
  through a transport is simulated; a per-transport quanta budget is the only transport input `run` accepts.
- **No photonic interconnects, no error-correction logic.** A QEC stack is a client of `run`, not part of it.
- **The provenance ledger travels with the repository, not with the wheel.** Section 14.5 specifies
  `docs/provenance/ledger.yaml` as "a YAML file kept beside the plan", and the wheel ships only `qutip_trap/**`, so
  `qutip_trap.provenance.load_ledger()` raises `FileNotFoundError` from an installed wheel (`repository_root()` resolves
  into site-packages). Nothing in the public API depends on it — `Device.derived()` stores record *ids*, never the records
  — but a client that wants the records themselves, the milestone-M11 application included, must locate the YAML file
  itself and pass its path to `load_ledger` (`anchor.m10.release_checks`).
- **Two of Section 7.9's four randomized-benchmarking variants.** Clifford RB (one qubit, several at once as simultaneous
  RB, and a pair from the 11520-element group) and the Knill-style variant fitted as B p^L + ½ run through `run()` as
  protocols; direct RB with random XX layers at probability p_2Q and its two-instance (r_1Q, r_2Q) extraction do not
  exist, so Section 7.9's corrected direct-RB right-hand side r = (4ⁿ − 1)(1 − p)/4ⁿ is implemented as a conversion
  (`noise/summary.py`) with no protocol behind it (`anchor.m10.knill_style_rb`).
- **Compilation is correct, not optimal.** Standard gates compile to the native set through verified templates (Section 7.7;
  an arbitrary SU(4) through the KAK decomposition at three entangling gates); no gate-count optimization beyond the
  standard identities and the merging of adjacent single-qubit unitaries in the benchmark circuits.
- **Measurement is terminal.** `measure`, `reset` and `recool` are IR operations and `Schedule` carries their positions, but
  the scheduler refuses a mid-circuit measurement or reset with a clear error until the detection back-action physics of
  Section 8.5 is implemented.
- **Mixed species at the crystal level only.** Equilibrium positions, normal modes and per-species Lamb-Dicke parameters are
  computed for mixed crystals (Section 4.1.7); sympathetic cooling is modelled at the rate level (Section 4.2.5) and no gate
  on a mixed crystal is scheduled or validated.
- **Collisions are events, not molecular dynamics.** Background-gas collisions are Langevin-rate events with an outcome
  distribution (heating kick, reorder, loss, dark ion), heralded per shot; the remaining ions' dynamics stay on the nominal
  crystal after a loss or dark-ion event, which the diagnostics state.

## Approximations a run can make, and how each is reported

- **Fock truncation.** Every resolved mode has a cap sized from the pulse's closed-form coherent excursion at the 10⁻⁶ tail
  plus the Section 5.1.1 margin; the boundary population is monitored after every pulse and a cap that loses its margin is
  raised and the run repeated (`Diagnostics.boundary_population`, `margin_reached`, `cap_growth`). The displacement
  operators are exact matrix exponentials asserted against the analytic Laguerre elements over the populated range.
- **Mode classes.** A mode is *resolved* (in the joint space), *frozen* (removed from the space, entering through per-shot
  Debye-Waller factors, with its off-resonant excitation bound and its entangling-angle loss χ_m reported) or *dropped*
  (contribution below 10⁻⁶ in |α|²(2n̄ + 1) and 10⁻⁴ rad in |χ|, the summed dropped contribution reported), by the
  closed-form integrals of the played waveform and never by η alone (`Diagnostics.mode_class`, `frozen_contribution`,
  `frozen_excitation_bound`, `dropped_contribution`). The ENR option carries a group of cold modes as one
  excitation-number-restricted factor.
- **The initial mixture as branches.** The pumped internal state and the thermal modes form a mixture diagonal in the
  computational and Fock bases, evolved as weighted pure branches; branches below `SolverOptions.branch_weight_min` (default
  10⁻⁶) are dropped and their weight reported (`Diagnostics.dropped_branch_weight`).
- **Solvers.** `sesolve` on pure branches, `mesolve` with collapse operators up to `mesolve_dimension_max` (default 128),
  keyed quantum-jump trajectories above it (`ntraj` per pure initial state); the `dop853` → `vern9` escalation ladder at
  atol 10⁻¹⁰, rtol 10⁻⁸; constant-Hamiltonian segments by their exact propagator. A segment whose integration fails its
  tolerance escalates and reports it (`Diagnostics.integrator`, `tolerances`).
- **GATE_LOCAL** (`run(level="auto")` above a joint dimension of 4096 or 2 × 10⁷ drive-operator non-zeros). Each gate step
  runs exactly on its own space (the addressed ions, the crosstalk neighbours above `crosstalk_threshold`, the resolved
  modes at the tracked occupations) and its channel is extracted by state-based process tomography and applied to the
  register; spin-motion and mode-mode correlations left after a step are traced out rather than carried to the next. The
  approximation is measured, not hidden: every step reports the residual displacement per spin eigenstate, its bound
  Σ|α_m|²(2n̄_m + 1), the purity deficit of the reduced motional state, the frozen excitation bound and the dropped
  crosstalk, and Section 9.8 holds a JOINT_EXACT comparison to that bound (`Diagnostics.gate_local`). How the channel is
  extracted is not an approximation (performance pass 2026-09-09, `TomographyRecord.route`): a unitary step propagates the
  Π d_i internal basis kets per motional branch and reads the channel off the Stinespring isometry they form (a carrier step
  on an internal-state-only space takes the cached segment propagator itself, one engine call per branch); a dissipative step
  propagates every one of the Π d_i² input states and fits the Choi matrix by least squares, the reference route that
  `SolverOptions(tomography_isometry=False)` forces everywhere. The routes agree to the solver tolerance. One thing does
  change: the truncation monitor of Section 5.5 runs on the propagated basis kets, so a superposition input's boundary
  population is bounded by Π d_i times the maximum it reports (Cauchy-Schwarz), which the step's notes say.
- **Scattering at d = 2.** A run on two-level register factors reports the photon-scattering probabilities of every pulse as
  estimates in the intrinsic budget (Raman, leakage, Rayleigh); `internal_levels > 2` and `SolverOptions.scattering_channels`
  simulate them as collapse operators with recoil and read the leaked levels out by their manifold's class.
- **Micromotion.** Default runs use the pseudopotential-frame Hamiltonian with C₀ on every Lamb-Dicke parameter and the
  J₀(β) carrier factor of excess micromotion; explicit rf (Floquet) dynamics is a validation option because it multiplies
  the step budget by Ω_rf/ω_m.
- **Calibration.** The default table is the closed-form surrogate with exact spot checks (Section 7.5); the full calibration by
  simulated experiments (`calibrate(surrogate=False)`) is an opt-in audit that fits the compute budget up to two ions. Drift
  enters through the dynamical samples at the shot clock and an optional servo; recalibration in the loop is not modelled.
- **The fast readout path.** The product POVM is exact at zero readout crosstalk; with camera or neighbour crosstalk the
  fast path carries the register-wide confusion tensor (dense to twelve ions) and the full photon-record path is the
  reference (`readout="full"`).

## What the physics leaves as inputs

- **Anomalous heating** has no first-principles model: S_E(ω) is a device input (constant, power law or tabulated) with its
  provenance, and every heating number is apparatus-local.
- **Electric-field, magnetic-field, laser-phase and intensity noise** enter as spectra with declared white parts; the
  sources supply exponents and single numbers, not spectra, so the absolute infidelity of the filter-function layer is
  conditional on the device's own spectra.
- **Apparatus quantities** of the readout (detection efficiency, background rate, a D-manifold branching fraction the
  sources do not tabulate for 171Yb+) are presets, never species constants.
- **Empirical species inputs** (lifetimes, branching ratios, hyperfine constants, g-factors) are cited numbers; everything
  hyperfine-resolved is derived from them and never typed in.
- **Open conventions** the plan records rather than resolves: the composite-pulse detuning-error normalization (closed by
  the Kabytayev/Cummins identification), the direct-RB error-rate prefactor, the factor 2 between two internally consistent
  comb Rabi-frequency chains.
- **One deliverable the plan names but never specifies.** Sections 6.8 and 4.4.7 list Bermudez et al.'s three-variant
  mapping of ε onto Pauli channels as provided, but the plan transcribes no formulas for the three variants (one mention of
  them in the whole plan, with no equations), so this release omits it and says so (`conv.bermudez_pauli_mapping`);
  implementing it needs the source, not the plan. What Section 6.8 does specify, `noise/summary.py` provides in full: the
  entanglement and average gate infidelity of a simulated channel, its Pauli twirl (itself a mapping of a channel onto a
  Pauli channel), and the depolarizing rate of `conv.depolarizing_normalization` with its p/3 and p/15 Kraus form and the
  Qiskit λ = p 4ⁿ/(4ⁿ − 1) conversion.

## Cost and reach

The exact level is bounded by the drive operator: a joint dimension of about 2 × 10³ to 4 × 10³, that is two or three ions
with two or three dynamically resolved modes, or more ions with one or two resolved modes and frozen spectators (Section
5.4), with the guards of Section 11.5 routing anything larger to GATE_LOCAL under `level="auto"` and refusing an explicit
`level="JOINT_EXACT"` above them (raise `joint_dimension_max` or `nnz_max` deliberately to build such a space; a cap that
would grow past the guard during a run is refused the same way). Measured on the reference machine (Apple
Accelerate BLAS, 18 CPUs, QuTiP 5.3.1) for a two-ion 100 µs Mølmer-Sørensen pulse; the last column is the default since the
performance pass of 2026-09-09, which integrates every ket segment in the exact rotating frame ψ = e^{−iH₀t} φ of
`dynamics/rotating.py` (the same drive operators, tolerances and integrator, 3 to 8 times fewer right-hand-side evaluations,
results equal to the solver tolerance; ledger `anchor.perf.rotating_frame_exactness`, `validation/scripts/bench_rotating.py`):

| joint dimension | merged dense, Schrödinger picture (Section 11.1) | factorized kernel, Schrödinger picture | factorized kernel, rotating frame |
|---|---|---|---|
| 48 (one mode, d_m = 12) | 0.15 s | 1.6 s | 0.7 s |
| 256 (two modes, d_m = 8) | 0.6 s | 2.5 s | 0.8 s |
| 864 (three modes, d_m = 6) | 6 s | 4.9 s | 1.2 s |
| 2048 (three modes, d_m = 8) | 2.1 min | 12 s | 1.9 s |

(the two right-hand columns are five times the measured 20 µs integrations of `bench_rotating.py`; the engine's `auto` kernel
still assembles the small spaces, where the assembled operator is the cheaper one per evaluation). The two-ion 100 µs
single-loop pulse of the M4 fixtures on the space [2, 2, 9, 15] the cap rule selects takes 5.1 s in the Schrödinger picture
and 1.0 s in the rotating frame, 43 against 7 integrator steps per period of the 3 MHz mode, with the register populations
equal to 3 × 10⁻⁸. `SolverOptions(rotating_frame=False)` restores the Schrödinger-picture integration, and
`SegmentReport.frame` names the picture each segment used.

Parallel maps fork their workers (QuTiP's `parallel` map uses the `fork` start method), so every worker begins as a
copy-on-write image of the calling process and can grow toward its size as it touches objects. `SolverOptions.workers = None`
therefore no longer means every CPU: `dynamics/parallel.worker_count` caps the count so that the workers' worst-case total
(each as large as the parent's peak resident size) stays inside half of physical memory (a 3 GB parent on a 48 GB machine
gets 8 workers, a fresh process every CPU), an explicit request is capped the same way, and `Diagnostics.workers` reports the
count actually used (ledger `conv.worker_memory_cap`). A long session that has built many Hamiltonians is the case this
guards; two kernel watchdog panics on 2026-09-08 came from 18 workers forked out of multi-gigabyte pytest processes.

and for the two-ion example device of `device/presets.py` (2026-09-09, after the performance pass): the surrogate
calibration about 6 s (15 s before it), a Bell circuit with 2000 shots about 3.5 s (8 s before; five carrier pulses and one
entangling gate on the 572-dimensional space [2, 2, 11, 13]), a 384-Clifford
single-qubit RB sequence 2 s (the carrier pulses run on the internal space through the propagator cache), a two-qubit
Clifford about 10 s (1.5 entangling gates on average), the Bell circuit through `level="GATE_LOCAL"` about 5 s (its
entangling step's channel from 40 propagated basis columns over 10 motional branches on the space [2, 2, 11, 13]; 17 s on the
sixteen-input reference route `SolverOptions(tomography_isometry=False)`, 60 s before the performance pass), a
two-qubit quantum-volume circuit (two SU(4) layers, six entangling gates) about one minute. Calibration by simulated
experiments of the two-ion device takes about twelve minutes; the three-ion GHZ circuit at dimension 1152 several minutes
per run. Randomized benchmarking to the 10⁻⁵ level therefore needs sequences of hundreds of single-qubit Cliffords and
thousands of shots, which the machine affords; two-qubit RB and quantum volume are run at tens of Cliffords and a handful of
circuits, and the results say so (`QVResult.protocol_circuit_count_met`).
