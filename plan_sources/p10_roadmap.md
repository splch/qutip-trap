## 10. Roadmap and milestones

Each milestone ends with tests that pass in CI. Nothing in a later milestone is started until the earlier milestone's validation cases pass in JOINT_EXACT mode. The ordering follows the physical chain, so at every stage a runnable, physically meaningful subset exists.

### M0. Scaffolding and public interfaces (1 to 2 weeks)
- `uv`-managed project, Python 3.13, dependencies pinned (qutip 5.3.x, numpy, scipy; pytest, hypothesis; matplotlib optional).
- GUI dependency declared as the optional extra `gui`: `flet[all]>=0.86,<1.0` and `flet-charts>=0.86,<1.0` (Flet 0.86.5 of 2026-08-01, Apache-2.0, Python 3.10 or later; the 0.86 line is the announced last release before 1.0, so the pin is reviewed when 1.0 ships). Flet is the Flutter-based Python UI framework that will carry the separate application package of Section 14: device configuration, live experiment scans and run monitoring in a native window (`flet run`) or a browser served from the host (`flet run --web`). Nothing in the physics core imports it. M0 only installs it, scaffolds `app/main.py` with the `flet create` layout (`pyproject.toml`, `src/main.py`, `src/assets/icon.png`), and checks that `flet doctor` passes in CI; the application itself is milestone M11 (Section 14).
- `units.py` with constants and the Hz/rad·s convention enforced by types.
- Appendix E frozen as the public API: the dataclasses, the `PulseEngine` protocol, `run`, the experiment API and the `Result` and `Diagnostics` types, so that M11's contract can be checked against the core from the start; frozen only once the fields the 2026-09-04 experimentalist critique added are in place (Trap stray field, shims and axis angle; CalibrationTable qubit frequencies, waveforms and micromotion entries with timestamps; NoiseModel rf noise, mains and collisions; Result heralds).
- The provenance ledger (Section 14.5) started from Part II, and the check scripts of Appendix D committed under `validation/scripts/` and re-run on the pinned toolchain as the first CI job.
- Species data tables for 171Yb+ and 40Ca+ first, then 43Ca+, 133Ba+/137Ba+, 9Be+, 25Mg+, 88Sr+; every number cited; nothing hyperfine-resolved typed in.
- CI: lint, type-check, tests, a convergence-report artifact.

### M0a. Atomic structure layer (3 to 4 weeks)
- `species/atomic.py` (Section 4.5): hyperfine-Zeeman diagonalization with the Breit-Rabi check, Wigner-Eckart dipole elements, Γ to reduced element to I_sat, polarization decomposition, clock-point finder; acceptance tests are the anchors of Section 9.13 (43Ca+ 146.0942 G, 9Be+ 119.446 G, 25Mg+ 212.78 G, 171Yb+ 310.87 ± 0.02 Hz/G² with the adopted g_J = 2.002615, the 171Yb+ 1/3 : 2/3 branching, I_sat = 50.83 mW/cm² from the partial 19.62 MHz rate; the second revision's 310.85 and 50.77 were the retired readings, and a CI check greps the tree for retired constants).
- Raman couplings, light shifts and scattering amplitudes from the level structure (Sections 4.5.4 and 4.5.5): the explicit intermediate-state sums, the Ozeri closed forms as tests, the differential-Rayleigh rate, leakage branching; the electric-quadrupole coupling of Section 4.5.7 with the Section 9.14 tests.
- State labelling by adiabatic connection through avoided crossings, the clock-point finder, and the polarization decomposition of Section 4.5.3.

### M1. Trap and crystal (2 to 3 weeks)
- Mathieu module with explicit convention; pseudopotential; stability check.
- Equilibrium positions for N ions; Hessian; normal modes on all three axes; ordering checks (axial COM lowest, transverse COM highest).
- Lamb-Dicke parameters per ion per mode for a given Δk, from the mass-weighted eigenproblem of Section 4.1.7 so that mixed-species crystals are a parameter change; two-ion closed forms as tests.
- Surface-electrode module (Section 4.1.6): gapless-plane electrode potentials, rf pseudopotential, rf null, principal axes, the map to Mathieu parameters; tests: Laplace residual, boundary values, strip limit, the five-wire height h = √(a(a + 2b))/2 for full widths a (centre) and b (rails), equivalently √(x₁x₂) for rail edges at x₁ and x₂ (the first version wrote √(a(a + b)), the same formula with a as a half-width, unstated; caught by the 2026-09-04 critique), the House and Nizamani closed forms.
- Heating rates from S_E(ω) models.
- Tests: Sections 9.1 and 9.13.

### M2. Single ion, spin-motion coupling, single-qubit gates (3 weeks)
- HilbertSpace for 1 ion × 1 to 3 modes; cached operators; exact displacement operators built by matrix exponential and asserted against the analytic Laguerre elements over the populated range (Section 5.1.1); the boundary monitor.
- Hamiltonian builder for Raman, single-photon optical (electric-dipole and electric-quadrupole, Section 4.5.7) and microwave drives, with Rabi frequencies, Stark shifts and scattering rates computed from Section 4.5 rather than entered; frames (Schrödinger-motion default, interaction-picture option); coefficient handling; the `dop853` default and step-density budget of Section 5.3.
- Rabi flopping, sideband flopping, Ramsey; carrier and sideband Rabi frequencies versus n; Debye-Waller.
- Single-qubit native gates (GPi, GPi2) as pulses; virtual RZ phase tracking.
- Tests: Section 9.2.

### M3a. Multi-level optical Bloch builder (2 weeks; runs before M3, whose Doppler stage needs its W(Δ))
- The shared builder of Section 4.2.8: rotating-frame assignment manifold by manifold with detection of inconsistent frame graphs and the Floquet fallback, `steadystate` as the single scattering-rate object of Section 13, time evolution for pumping, the level-C cooling solve with one mode.
- Tests: the level-A and level-B closed forms recovered from level C in their stated regimes; the 171Yb+ detection rate and leakage prefactors of Section 8.1 recovered from the angular algebra (this milestone owns that rate object; M5 consumes it).

### M3. Cooling and state preparation (3 weeks)
- Spontaneous-emission recoil operators of Section 4.2.8 with the per-ion participation and the recoil-energy unit test; optical pumping from first principles on the M3a builder, which precedes this milestone.
- Doppler cooling rate model producing n̄_D per mode with the configured angular factor; master-equation check on one mode.
- Resolved-sideband and Raman sideband cooling rate model with final n̄, including pulsed schedules with exact matrix elements and higher sidebands; EIT cooling model; master-equation checks.
- Optical pumping model with preparation error.
- Tests: Section 9.3.

### M4. Two-ion entangling gates (4 weeks)
- Bichromatic MS drive; loop closure; χ = π/4 calibration; thermal robustness curves.
- Light-shift (σzσz) gate.
- Multi-mode: 3 to 5 ions, AM/FM/PM pulse solvers for closure over all modes.
- Exit criteria: JOINT_EXACT at 2 to 3 ions with the resolved modes of Section 5.4, plus closed-form closure checks (the Section 4.4.3 integrals) at 4 to 5 ions; the exact multi-mode comparison at 4 to 5 ions is deferred to M9a, where mode selection, frozen spectators and the matrix-free kernel exist (the second revision required it here, where the plan's own rule made it unvalidatable; 2026-09-04 architect critique).
- Tests: Section 9.4.

### M5. Readout (2 weeks)
- Fluorescence record model consuming M3a's rate object (R_∘, R_d, R_b from the Bloch solve); shelving; photon-count sampling; threshold and time-resolved discrimination; crosstalk; the product POVM at zero crosstalk and the register-wide confusion tensor otherwise (Section 5.7).
- Tests: Section 9.5.

### M6. End-to-end circuits, JOINT_EXACT (2 weeks)
- Compiler (standard gates → native), scheduler, `run`, results, IonQ JSON and OpenQASM 2 import.
- 2- to 4-ion circuits (Bell, GHZ, small algorithms) with all physics on, compared against ideal statevector results and against the analytic error expectations.
- Tests: Section 9.6.

### M7. Noise and error channels (3 to 4 weeks)
- Heating, motional dephasing, qubit dephasing (quasi-static + Lindblad), laser phase and amplitude noise, scattering with leakage (d > 2) and differential-Rayleigh dephasing from the amplitudes of Section 4.5.5, recoil on scattering, addressing crosstalk, spectator modes (resolved and frozen), idle-time evolution between pulses, the control-hardware chain of Section 7.10.
- Error budgets reproduced (Section 9.7).

### M8. Calibration emulation (3 weeks)
- Simulated experiments: Rabi frequency scans, sideband spectroscopy for mode frequencies, MS amplitude/detuning/phase scans, parity scans, heating-rate measurement, detection-threshold calibration.
- Calibration tables cached per device; scheduler consumes them. Tests: calibrated parameters recover the device's true derived values within stated uncertainty, and gates built from calibrations reach the fidelity predicted from the noise model.

### M9a. Scaling I: resolved-mode selection, frozen spectators, ENR option, GATE_LOCAL (3 weeks)
- The contribution criterion of Section 11.3, frozen spectators with χ_m and α_m reporting, the ENR option for cold undriven groups with its sum-generator operator, the adaptive cap and margin policy of Sections 5.1.1 and 5.5.
- GATE_LOCAL with state-based process tomography, Choi reconstruction and CP projection, motional-model tracking and residual-displacement reporting; validation against JOINT_EXACT on 3- and 4-ion circuits (Section 9.8).

### M9b. Scaling II: matrix-free kernel and parallelism (3 weeks)
- The mode-factorized drive kernel of Section 11.3 as a matrix-free right-hand side (scipy `solve_ivp` on a `LinearOperator` or a custom QuTiP `Data` type), with the Section 11.1 table as its acceptance test; the integrator escalation ladder of Section 5.3.
- Trajectory parallelism with keyed seeds and the 1-versus-18-worker reproducibility test; caching of internal-space propagators.

### M10. Benchmark emulation and release (2 weeks)
- Randomized benchmarking, GHZ fidelity, and a quantum-volume style run on the simulated device, with the simulator's own error budget reported alongside.
- Documentation: physics notes with equation provenance, conventions, examples, limits.

### M11. The zoomable application (6 to 8 weeks, Section 14; separate package, no core changes)
- **Separation.** The app is `qutip_trap_app`, a package of its own that consumes the released core through its public API. M0 to M10 do not depend on it, core CI does not run it, and no core change is scheduled for it; needs discovered while building it become core feature requests judged on the core's own merits.
- **M11.1 Run record and view-models (2 weeks).** Record schema and storage policy; on-demand re-simulation of a zoomed pulse with caching; export and bitwise re-import; the provenance index generated from Part II; view-model tests for the coarse-graining identities (Section 9.11).
- **M11.2 Levels 0 to 2 (2 weeks).** Device card, circuit editor and importers, compiled timeline with the phase register, schedule view with the mode spectrum and closure indicators; channel-replay (app-side) and GATE_LOCAL runs in workers with progress; the verify-deeper action.
- **M11.3 Levels 3 and 4 (2 weeks).** Dynamics view with re-simulation, Fock heatmaps and phase-space loops; the Hamiltonian builder page; the species, trap, crystal, light, noise, cooling and readout pages; downward propagation of parameter changes through the calibration emulation with stale badges.
- **M11.4 Presets, packaging and polish (1 to 2 weeks).** Validation-suite presets with published numbers beside simulated ones; explain panels; `flet test` navigation and run-flow tests; desktop bundles with `flet build`; documentation.
- **Exit criteria.** Every displayed quantity resolves to a provenance chip; the Section 9.11 tests pass; a Bell-state job can be followed from a histogram bar to a matrix element in at most six clicks; exported records re-import bitwise; a parameter change at Level 4 propagates to the device card without manual steps.

### M12. Transport, splitting and junctions (specification only in this plan; estimate 8 to 10 weeks when scheduled)

- **Status.** Not scheduled for the first release. Section 4.6 is the physics specification; this entry is the work breakdown, so that the estimate is on the record and so that the interfaces M0 freezes (Appendix E's `Transport`, `Waveform` and `Zone` records and the `Trap` voltage-schedule field) need not change when it is picked up. Nothing in M0 to M11 depends on it; it depends on M1 (crystal solver), M7 (noise channels) and M8 (calibration emulation).
- **M12.1 Waveform and hardware chain (2 weeks).** One interface returning (q₀(t), ω(t)) for transport and (d(t), α(t), β(t), γ(t)) for splitting. Transport shapes: sine, error-function with its width parameter, Blackman-like smooth ramps, the Bézier/Bernstein family s(τ) = Σ_j s_j C(N,j) τ^j (1 − τ)^{N−j} with s₀ = s₁ = 0, s_N = s_{N−1} = 1, s_j + s_{N−j} = 1 and N = 2n_t + 3, and a bare linear ramp, which is *not* in that family and is the shape that actually reached sub-quantum excitation in Sterk's axial-frequency-only run. Splitting shapes: the sine-squared distance ramp and the sign-corrected quintic as the default. Round trips are forward, hold(t_hold + h) with h quantized to the DAC step, then exact time reversal. The delivered-waveform chain is a fixed transfer function applied to every candidate before x_well(t) and ω(t) are extracted: zero-order hold at the configured DAC step, the digital anti-alias filter, then the analog low-pass (a sixth-order Butterworth at 1.3 MHz contributes 0.473 µs of group delay, about 8% of a 6 µs move, against only 2% amplitude loss at 1 MHz). The DAC rate is a first-class knob, because the J R_DAC = f_z resonance is reproduced by unitary evolution of the sampled staircase, not by a rate.
- **M12.2 Analytic fast path (1 to 2 weeks).** The Ermakov integrator ρ̈ + ω²(t)ρ = 1/ρ³ with ρ²μ̇ = 1 from ρ(−t₀) = 1/√ω₀, ρ̇(−t₀) = 0, cached per frequency waveform; u_p from the Green function; Ξ = u̇_p + iωu_p; γ = m|Ξ|²/(2ħω₀); the Husimi-Kerner P_mn. This is the waveform-design path, the truncation sizer and the exact oracle the QuTiP solve must reproduce. Exit test: the whole transport block of Section 9.15.
- **M12.3 Transport module (2 weeks).** The co-moving Hamiltonian as a `QobjEvo` with array coefficients sampled at at least twenty points per period of the largest |ω| and fine enough to resolve the staircase; `sesolve` unitary and `mesolve` with the Section 4.1.5 channels; α read out as ⟨a⟩ in the co-moving frame, reporting the worst case over hold offsets and never the mean. Truncation sized on the largest instantaneous |α|² in the chosen frame and re-checked at the highest ω on the trajectory. Segment chaining carries the coherent amplitude *with* its unimodular phase, the Ermakov state (ρ, ρ̇, μ) and the accumulated squeezing (δ, θ).
- **M12.4 Splitting and merging module (2 weeks).** The classical two-ion solve with the factor e restored and the signs written out; the voltage-to-coefficient map with the f_n multiplicities and the antisymmetric half-amplitude tilt; the piecewise static sets with their α domains; the design loop over d_CP, ω_CP, χ and δE′ minimizing n̄_coh(T) + Δn̄_th(T); the hand-off to the quantum layer as a displaced thermal state, with a uniform coherent-phase average for shot-averaged comparisons; a hard failure when |γ| ≥ γ̃; and a record of which output qubit received the larger kick. Merging is the plan's own construction (time reversal plus an explicit relative-phase parameter) tagged **[background]**, because no source quantifies it.
- **M12.5 Junction module (1 to 2 weeks).** φ_ps in two or three dimensions from the Section 4.1.6 electrode basis; ω_i² = ω̃_i² + ω_rf,i² with Σ_i ω̃_i² = 0 asserted everywhere and the 2ω_rf² sum rule asserted in the straight channels and *disabled* inside the junction; the frequency schedule across a traversal with the apex cap the voltage limit imposes; the transverse degeneracy lifted by a shim or absorbed by a mode-mixing dwell; axial micromotion with an explicit peak-or-rms declaration; the rf-sideband heating term with its documented 1.4 scaling factor.
- **M12.6 Optional closed loop (1 week).** The pseudo-energy objective with the second weight inside the square, the two exponential penalties, Nelder-Mead over the frequency and trajectory control points, and a regenerated loss-to-quanta calibration for both the thermal and the coherent family, never inverted above its fold.
- **Exit criteria.** Every transport row of Section 9.15 passes. A designed adiabatic junction waveform returns essentially zero coherent excitation, since the measured 0.053 to 0.18 quanta per round trip are noise-limited rather than adiabaticity-limited. The dwell-time sweep of an out-and-back pair reproduces the 1/ν_ax periodicity of its minima. The corrected quintic ramp terminates at d_f with vanishing first and second derivatives at both ends. And every phonon number in the results object carries the frequency it is referenced to.

Total: roughly 36 to 44 weeks of focused work for one implementer through M10, with M0a, M4, M7 and M9 the long poles (the first version of this plan estimated 22 to 26 weeks; the 2026-09-04 critique judged that not credible for the atomic layer, the optical Bloch builder and the scaling work, and this baseline follows its re-scoping), plus 6 to 8 weeks for the application in M11. Milestone M12 (transport, splitting and junctions) is specified but not scheduled and is not in either figure; its own estimate is 8 to 10 weeks.
