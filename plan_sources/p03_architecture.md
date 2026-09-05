## 3. Architecture

### 3.1 Design principles

- **Physics objects, not gate matrices.** The only place a unitary matrix appears is in the compiler's definition of what a native gate is supposed to do and in the validation suite that compares simulated pulses to it. The simulator never applies that matrix to the state.
- **One Hamiltonian builder.** Every pulse, whether a calibration scan, a single-qubit gate, a Mølmer-Sørensen gate or a cooling pulse, is expressed as a set of drive terms on a shared Hilbert space and assembled by the same builder. Approximations are options of the builder, defaulting to none.
- **Device parameters in, everything else derived.** The user specifies physical parameters (species, trap frequencies or voltages, beam wavelengths, geometry and powers, magnetic field, noise spectral densities, detector efficiency). Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering rates, heating rates, mode structure and gate pulse parameters are derived, the last of these by the simulator's own calibration routines.
- **Every run reports its own approximation status.** Truncation-boundary population, mode drop decisions, frame choices, and which fidelity level (Section 5) was used are attached to every result.
- **Deterministic by seed.** All stochastic elements (quantum-jump trajectories, quasi-static noise sampling, photon-count sampling) draw from keyed substreams of one root seed, spawned by (sample, trajectory, ion, channel) so that reproducibility survives parallel execution and truncation retries (Section 3.4).

### 3.2 Package layout

```
qutip_trap/
  __init__.py
  units.py            # unit system, constants (CODATA values with sources), conversions
  species/            # atomic data per ion: levels, transitions, linewidths, branching ratios,
                      #   hyperfine constants, g-factors, masses; one module per species
    atomic.py         # angular-momentum algebra shared by all species: Wigner-Eckart hyperfine dipole
                      #   elements (3j, 6j), Gamma -> reduced element -> I_sat, hyperfine + Zeeman
                      #   diagonalization at any field (Breit-Rabi as the J = 1/2 check), polarization
                      #   decomposition about B, clock-point finder (Section 4.5)
  trap/
    mathieu.py        # a, q parameters (convention explicit), beta, secular frequencies, stability check
    pseudopotential.py# static pseudopotential from voltages/geometry, or direct frequency input
    surface.py        # gapless-plane analytic electrode potentials, rf null, depth, principal axes (Section 4.1.6)
    crystal.py        # equilibrium positions, mass-weighted Hessian (mixed species), normal modes
                      #   (axial + two transverse), ordering, per-species eta (Sections 4.1.3, 4.1.7)
    anharmonic.py     # cubic/quartic mode couplings (optional correction terms)
    micromotion.py    # intrinsic/excess micromotion: C0 factor, rf sidebands, J0 carrier suppression
    heating.py        # S_E(omega) models -> heating rates per mode; multi-ion generalization
  light/
    beams.py          # beam geometry: k-vectors, polarizations, waists, powers, per-ion intensities
    raman.py          # effective two-photon Rabi frequency, effective Delta k, ac Stark shifts,
    bloch.py          # multi-level matrix elements and optical-Bloch steady state: W(Δ) = Γρ_ee in s⁻¹, R_∘, pumping rates, dark-state kernel check (Section 5.3); one supplier for cooling, readout, scattering. Level-C time evolution (cooling with one mode, pumping) runs through dynamics/hamiltonian.py's multi-level mode, owned by prep/
                      #   adiabatic-elimination validity
    microwave.py      # direct hyperfine drive, near-field gradient drive
    scattering.py     # Raman/Rayleigh scattering rates and amplitudes, differential-Rayleigh dephasing,
                      #   leakage branching from the angular algebra of species/atomic.py
    recoil.py         # spontaneous-emission recoil kernel: dipole-pattern average, per-mode kicks (Section 4.2.8)
    stark.py          # differential light shifts for light-shift gates and error terms
  hilbert/
    space.py          # composite space: per-ion qudit (2 + leakage levels) x modes; ENR support
    operators.py      # cached sigma_i, projectors, a_m, displacement D_i = prod_m D(i eta_im)
    truncation.py     # n_max / excitation-cap policy, boundary-population monitor, adaptive growth
  control/
    native.py         # native gate definitions (GPi, GPi2, MS, ZZ, RZ virtual) and their target unitaries
    compiler.py       # standard gates -> native gates; phase tracking (virtual Z); circuit IR
    pulses.py         # pulse objects: envelope, detuning schedule, phase schedule, tones, targets
    schedule.py       # gate -> pulses using calibration tables; timing, parallel single-qubit gates
    shaping.py        # AM/FM/PM pulse solvers for multi-mode MS closure (Section 4.4.3)
    hardware.py       # DDS/AOM/amplifier chain: quantization, phase continuity, envelope response, dead times (Section 7.10)
    table.py          # CalibrationTable: a plain data object of pulse parameters with provenance and status,
                      #   consumed by schedule.py; no behaviour, so control/ never imports calibration/
  dynamics/
    hamiltonian.py    # the ONE Hamiltonian builder: H(t) = H0 + drive terms from pulses and device, plus the multi-level mode of Section 4.2.8 (manifold-by-manifold frames, Floquet fallback) that cooling, pumping and detection share; frames
    channels.py       # collapse operators and stochastic processes from noise parameters
    evolve.py         # solver selection (sesolve/mesolve/mcsolve), step control, checkpoints
    frames.py         # interaction pictures; phase bookkeeping between pulses
    kernels.py        # mode-factorized application of sigma_+ (x) prod_m D_m for matrix-free propagation (Section 11.3)
    engine.py         # PulseEngine protocol: run_pulses(device, schedule, state, space, sample, seeds, options) -> Traces, Appendix E's signature verbatim; the one entry point
                      #   that run/, calibration/ and experiments/ share (Appendix E)
  prep/
    doppler.py        # Doppler cooling: Section 4.2.2 rate coefficients with W from light/bloch.py, one (Delta, s, k_hat) per beam optimized over all modes -> n_bar per mode; force model and two-level closed form as guarded cross-checks only (Section 4.2.1)
    sideband.py       # resolved-sideband / Raman sideband cooling model (rate equations, optional ME)
    eit.py            # EIT cooling model
    pumping.py        # optical pumping -> initial internal state with preparation error
  readout/
    fluorescence.py   # scattering-rate model, bright/dark pumping rates, shelving
    detection.py      # photon-count sampling (window, efficiency, background), crosstalk
    discriminate.py   # threshold and time-resolved discriminators; calibration of threshold
  noise/
    spectra.py        # S_E(omega), S_B(omega), laser phase/intensity noise models
    sampling.py       # quasi-static parameter sampling per shot; OU processes for fast noise
  device/
    model.py          # Device: species + trap + beams + field + noise + detector; derived quantities
    presets.py        # example devices (e.g. 171Yb+ chain, 40Ca+ optical, 43Ca+ clock, 137Ba+)
  run/
    job.py            # run(circuit, shots, level): orchestrates prep -> schedule -> evolve -> readout
    results.py        # Result: bitstrings, counts, probabilities, raw counts, diagnostics
    levels.py         # fidelity levels: JOINT_EXACT, GATE_LOCAL, and their reporting
  io/
    openqasm.py       # OpenQASM 2 importer (subset), IonQ circuit JSON importer/exporter
  calibration/        # ABOVE control/ and dynamics/: simulated experiments -> CalibrationTable (Rabi, modes,
                      #   MS amplitude/detuning/phase, thresholds, heating); bootstrap from derived values as tagged
                      #   initial guesses; scan ranges, fit models, convergence criteria, failure modes (Section 7.5)
  experiments/        # user-facing simulated experiments built on the PulseEngine protocol; calibration/ uses them
  validation/         # analytic reference formulas used by tests
    scripts/          # the check and benchmark scripts behind every [recomputed here] number (Appendix D)
tests/                # pytest suite mirroring the validation plan (Section 9)

qutip_trap_app/       # SEPARATE package for the zoomable application (Section 14, M11). Depends on the released
                      # qutip-trap and uses only its public API; never imported by the core; own pyproject and tests
  pyproject.toml      # depends on qutip-trap, flet[all], flet-charts
  main.py             # ft.run(main); ft.Router routes mirroring the zoom ladder
  record.py           # run record built from core Results: job -> circuit -> schedule -> traces -> readout records -> results
  replay.py           # channel-replay engine built from the core's Section 6.8 channel summaries (app-side, labelled derived)
  viewmodel/          # pure Python, no Flet import: level views over records, coarse-graining, provenance lookups
  views/              # Flet controls per level: machine, circuit, schedule, dynamics, physics pages, numerics panel
  workers.py          # ProcessPoolExecutor bridge around core calls; progress via page.pubsub
  plots.py            # flet-charts and RawImage adapters: time series, phase-space loops, histograms, heatmaps
  provenance.py       # tag and source index generated from Part II of this document
  presets.py          # validation-suite experiments as loadable device-plus-experiment presets
docs/                 # this plan, physics notes, conventions
```

### 3.3 Core data model

- `Species`: immutable atomic data for one isotope: mass; nuclear spin I and magnetic moment; level list (n, L, S, J) with energies, lifetimes, hyperfine constants A and B and Landé g_J; transitions with wavelength, natural linewidth Γ and the branching ratios that the angular algebra does not fix (fine-structure branching to D levels); the designated qubit pair; the cycling transition, repump transitions, shelving transition where applicable. Everything hyperfine-resolved (Zeeman energies at the operating field, dipole matrix elements, polarization-resolved Rabi frequencies, Raman and Rayleigh amplitudes, leakage branching) is derived by `species/atomic.py` (Section 4.5), never typed in. Species modules ship with citations for every number.
- `Field`: the static magnetic field vector at the ions; it defines the quantization axis in which beam polarizations are decomposed into σ⁺, π and σ⁻ components, and its magnitude and noise spectrum feed the Zeeman shifts of every level.
- `Trap`: either explicit secular frequencies (ω_x, ω_y, ω_z) with a radial principal-axis rotation angle, or (V_rf, Ω_rf, U_dc, geometry factors) converted through the Mathieu module with the convention documented in Section 13, plus the stray field and the shim voltages that compensate it, from which the residual excess-micromotion β per beam direction follows (Section 4.1.1). Provides the potential's harmonic coefficients, the mode directions and optional anharmonic terms; milestone M12 (Section 4.6) adds an optional dc voltage schedule V_n(t) and a zone list, from which the well trajectory and the instantaneous curvature are derived after the Section 7.10 hardware chain rather than from the commanded staircase. The axis angle exists because the explicit-frequency path would otherwise pin the modes to the laboratory frame, and a mode with no projection on any cooling beam is uncoolable; the stray field and shims exist because compensation is something a laboratory measures and re-nulls, not an assumption (both added after the 2026-09-04 experimentalist critique).
- `Crystal`: derived from `Trap` and `Species` and N: equilibrium positions, the 3N normal modes as (frequency, eigenvector) pairs grouped by axis, per-ion per-mode Lamb-Dicke parameters for a given Δk.
- `Beam`, `BeamPair`: wavelength(s), k-vectors, polarization, waist, power, pointing; per-ion intensity via the beam profile (this is where addressing crosstalk originates).
- `Drive`: a physical drive on a set of ions: type (Raman, single-photon optical, microwave), effective Rabi frequency envelope Ω(t), detuning schedule δ(t) (per tone), phase schedule φ(t), effective Δk, Stark shift; derived from beams by `light/`.
- `Pulse`: a `Drive` with start and end times and metadata linking it to the gate it implements.
- `HilbertSpace`: the composite space with ion qudit dimension d (2 by default, 3 or more when leakage is modelled), mode list with truncation policy, and cached operators.
- `NoiseModel`: spectral densities and stochastic-process parameters with the derived rates.
- `Device`: the aggregate; exposes `derived()` with all computed quantities and their provenance.
- `Circuit`: a list of operations on qubit indices in a small IR (native gates plus standard gates before compilation), with classical measurement targets.
- `Schedule`: ordered pulses with timings, plus the per-qubit phase frame at each point.
- `Result`: per-shot records and aggregates, plus a `diagnostics` block.

### 3.4 Control flow of `run`

```
Device + Circuit
  -> compile: Circuit -> native gates (phase-tracked)                    [control.compiler]
  -> calibrate (cached per Device): CalibrationTable                    [calibration/, through dynamics.engine]
  -> schedule: native gates -> Pulses with absolute times               [control.schedule]
  -> prepare: motional state per mode + internal state per ion          [prep.*]
  -> evolve: for each pulse, build H(t), c_ops; advance state           [dynamics.*]
       (between pulses: free evolution with heating/dephasing for idle time)
  -> readout: fluorescence model -> photon counts -> bits; measure, reset and recool are IR
       operations, so the loop is prep -> [gates | measure | reset | recool]* -> results (7.2) [readout.*, prep.*]
  -> Result
```

Shots are produced by the ensemble structure of the physics, with one vocabulary throughout the plan: a **sample** is one draw of every quasi-static parameter (Section 6.1); a **trajectory** is one quantum-jump unravelling within a sample, or the single deterministic density-matrix evolution when `mesolve` is used; a **shot** is one measurement record. `run(circuit, shots, samples=None, level=...)` distributes shots round-robin over samples (default samples = min(shots, 64)), chooses the trajectory count per sample in two phases, an estimate from `mcsolve`'s `target_tol` on the final populations (which needs `e_ops`, Section 5.4) followed by a replay of a fixed seed list of that length, draws shots from the weighted trajectory mixture (Section 5.3), and reports the realized (samples, trajectories, shots per sample) triple in the diagnostics together with the effective sample size from which the histogram's error bars are derived, since shots drawn from one sample are not independent draws from the ensemble. Estimators are chosen per quantity: deterministic `mesolve` where the joint dimension allows it, first-order perturbative channel evaluation for error rates at the 10⁻⁵ to 10⁻⁶ level that no trajectory count could resolve, trajectories with `target_tol` otherwise, and Gauss-Hermite quadrature rather than Monte Carlo over quasi-static parameters when a closed form is being reproduced. Randomness comes from one root `SeedSequence` per run, spawned deterministically by (sample, trajectory, shot, ion, channel), the shot slot added because photon-count sampling is per shot per ion and two shots on one (sample, trajectory) would otherwise share a record and the error bars would lose the within-trajectory detection variance, so that every variate's key is independent of execution order and of truncation retries; the per-trajectory children are passed to `mcsolve` as an explicit seed list, and the reproducibility test is a tolerance test (agreement to 10⁻¹² over 1 and 18 workers plus per-trajectory identity under the same seed) rather than a bitwise one, because `MultiTrajResult` accumulates running sums in completion order and floating-point addition is not associative (4.4 × 10⁻¹⁶ measured on 5.3.1 with identical seeds), and because `target_tol` stops on a check whose firing point depends on scheduling (200 trajectories serial against 5 parallel in one measured call; 2026-09-04 numerics critique) **[corrected: critique, 2026-09-04]**.
