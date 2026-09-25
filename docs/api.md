# API

`import qutip_trap as trap` exports `Circuit`, `Machine`, `Physics`, `Numerics`, `Readout`, `FidelityLevel`, `Result`,
`presets` and `__version__`. Every other name is imported from the module that defines it; this page lists them by what a
user does, one line each, and the docstrings carry the detail. Public numbers are in Hz, seconds and radians (IonQ's turns
only at the IonQ boundary), and every bitstring key has qubit 0 as the least-significant bit, the rightmost character
(PLAN.md Section 13). [examples.md](examples.md) runs the main path.

## Run a circuit on a machine

**`qutip_trap.presets`**: the example machines; the numbers are a realizable laboratory configuration, not a published apparatus.
- `yb171_chain(n_ions=2, **knobs)`: a 171Yb+ chain: a global 355 nm Raman pair for the entangling gates, one addressing pair per ion, the oblique 369.5 nm detection beam, Crain's SNSPD detector.
- `ca40_optical(n_ions=1, **knobs)`: a 40Ca+ optical-qubit chain on the 729 nm quadrupole line, Myerson's PMT chain, no entangling drive.

**`qutip_trap.machine`**
- `Machine(device, table=None, physics=Physics(), numerics=Numerics(), readout=Readout(), level=FidelityLevel.AUTO, name="")`: a trapped-ion computer as a client sees it, frozen; variants are `dataclasses.replace(machine, ...)`; a table of None is the closed-form surrogate, cached per device.
  - `run(circuit, shots, *, seed=0, keep_final_state=False, progress=None)`: compile, calibrate, schedule, prepare, evolve, read out: the `Result`.
  - `submit(circuit, shots, *, seed=0, keep_final_state=False, label="")`: the same run in a worker process, a `Job`; `spec(...)`: its `RunSpec`.
  - `compile(circuit)`: the `CompileReport`; `schedule(circuit, *, seed=0)`: the `Schedule`, nothing integrated.
  - `calibrated(method="closed_form", *, seed=0, **scans)`: this machine with the table of `calibrate` pinned.
  - `estimate(circuit, *, seed=0)`: the `Estimate` of a run before anything is integrated.
  - `engine`: the `JointExactEngine` configured from the machine's physics and table.
  - `error_model(*, qubits=None)`: the `ErrorModel`; `specs()`: a text report; `hash()`: the identity a record stores.
- `Estimate`: the level and why, the declared space and every mode's class, the dimension and drive non-zeros, the pulse counts, the schedule's length and a wall-time guess.

**`qutip_trap.run.spec`**
- `RunSpec`: a run request as a frozen record a JSON document can carry (`of`, `to_dict`, `from_dict`).
- `Job`: a run in a worker process: `status()`, `progress`, `result(timeout_s=None)`, `record()`, `cancel(terminate_after_s=None)`.
- `JobStatus`: queued, running, done, failed or cancelled; `JobCancelled` and `JobError` are what `Job.result()` raises.

**`qutip_trap.run.results`**: `Progress`, one step of a run handed to the `progress` callback (`stage`, `done`, `total`, `elapsed_s`, `fraction`).

## The option objects

**`qutip_trap.options`**: three frozen records, one per concern.
- `Physics`: which effects are simulated: `noise`, `internal_levels`, `scattering` (`"estimate"` or `"channels"`), `scattering_recoil`, `intensity_noise_channels`, `hardware_chain`, `stark_compensation`, `crosstalk_suppression`, `entangler` (`"ms"` or `"zz"`), `extra_channels`, `builder`, `t0_s`, `shot_period_s`.
- `Numerics`: how the integration is done, one flat record (`Numerics(atol=1e-9, ntraj=128, caps={0: 12})`): the tolerances and integrator ladder, the Section 11.5 guards, the Fock caps and mode-class tolerances, the branch cutoff, an ENR group or a declared space, the Lindblad method and trajectory count, the GATE_LOCAL map accuracy and tomography, the parallel map, workers and samples, parallel addressing and `convergence_check`. The engine and every lower-level routine take it.
- `Readout`: `mode` (`"fast"`, the POVM on the joint outcome, or `"full"`, every photon record generated and discriminated), `discriminator`, `povm_samples`.

**`qutip_trap.run.levels`**
- `FidelityLevel`: `AUTO` (JOINT_EXACT inside the Section 11.5 guards, GATE_LOCAL above them), `JOINT_EXACT`, `GATE_LOCAL`.
- `decide_level(budget, options, policy)`: the `LevelDecision` (level, dimension, non-zeros and the `reason` in words) a run acts on, `options` a `Numerics`; `within_budget(space, options)`: the guards on a declared space, a `Budget`.

## Circuits and wire formats

**`qutip_trap.control.compiler`**
- `Circuit(n_qubits, ops=(), measure=None, registers=None)`: the program and a persistent builder: one method per gate (`gpi`, `gpi2`, `ms`, `zz`, `rz`, `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `sx`, `rx`, `ry`, `cnot`, `cx`, `cz`, `swap`, `cp`, `rxx`, `rzz`, `u3`, `id`), `measured(*qubits, registers=None)`, and `from_openqasm`, `to_openqasm`, `from_ionq`, `to_ionq`.
- `Operation(name, qubits, params)`: one operation, parameters in radians.
- `NATIVE_GATES`, `STANDARD_GATES`, `NON_UNITARY`, `EXPORTED_NATIVE`: the gate sets with their arities.
- `compile_report(circuit, *, entangler="ms")`: a `CompileReport` (the native circuit, the pulse and entangling counts, every block and the whole circuit verified); `compile_to_native(circuit)`; `CompileError`.
- `ideal_probabilities(circuit)`, `circuit_unitary(circuit)`, `gate_matrix(op)`: what the circuit should compute.

**`qutip_trap.control.two_qubit`**: `kak_decomposition(u)` (a `KAK`), `decompose_two_qubit_unitary(u, pair)`, `haar_random_unitary(rng, dim)`.

**`qutip_trap.control.native`**: the exact native matrices in radians, `gpi`, `gpi2`, `ms`, `zz`, `rz`, `r_phi`, `xx`, with `rad_from_turns`, `turns_from_rad` and `equal_up_to_global_phase`.

**`qutip_trap.io.openqasm`**: `loads(text)` and `dumps(circuit, *, declare_native=True)` for OpenQASM 2; `NATIVE_DECLARATIONS`, `QELIB_NAMES`, `OpenQASMError`.

**`qutip_trap.io.ionq`**: `loads(obj)` and `dumps(circuit)` for IonQ's circuit JSON; `load_job(obj)` (an `IonQJob`) and `dump_job(circuit, *, backend, shots=100, ...)` for a v0.3 or v0.4 job body; `JOB_TYPE`, `JOB_KEYS`, `NOISE_KEYS`, `SETTINGS_KEYS`.

**`qutip_trap.interop.qiskit`** (the `qiskit` extra): `QutipTrapProvider` (`get_backend(name="yb171_chain", **knobs)`, `backends()`), `QutipTrapBackend(machine)`, a Qiskit `BackendV2`, and its `QutipTrapJob`.

## Devices and presets

**`qutip_trap.device.model`**
- `Device(crystal, trap, field, beams, noise, detector, hardware, preparation, gradient, roles)`: the apparatus: `derived()`, `specs()`, `to_dict()`, `from_dict(data)`, `hash()`.
- `DerivedQuantities`: every derived number (`values`) with the ledger id of its provenance (`provenance`) and `notes`.
- `Field`: the static magnetic field, the quantization axis; `GradientField`: a near-field microwave gradient.
- `BeamRoles`: which beams play the single-qubit, entangling and detection parts; `resolve(device)` gives `ResolvedRoles`.

**`qutip_trap.device.presets`**: `yb171_chain` and `ca40_optical` as `DevicePreset` records (`device`, `machine()`) and their parts: `secular_trap`, `raman_pair_along_x`, `oblique_detection_beam`, `ideal_hardware`, `crain_snspd_detector`, `myerson_ca40_pmt_detector`, `ca40_optical_recipe`.

**`qutip_trap.species`**: `species(name)` builds the `Species` of an isotope (`"171Yb+"`, `"40Ca+"`, ...) and `available()` lists the tables that build.
- `species.model`: `Species`, `Level`, `Transition`; `species.table`: `IncompleteSpeciesTable`, what a table lacking a constant raises.
- `species.raman`: `AtomicStructure`, the dressed sublevels and couplings at a field, and `structure_at(species, b_gauss, b_hat)`.
- `species.zeeman`: `ZeemanSpectrum`, `ClockPoint`, `clock_points`; `species.metastable`: `MetastableChannels`.

**`qutip_trap.trap`**
- `model`: `Trap`, from secular frequencies or from voltages and geometry; `pseudopotential`: `RfDrive`, `DcElectrodes`; `surface`: `Electrodes`, `GaplessPlaneTrap`.
- `mathieu`: `monodromy(a, q)`, `is_stable(a, q)`, `mathieu_parameters`, `MathieuParameters`, `UnstableMathieuError`.
- `crystal`: `solve_crystal(trap, species)`, `Crystal`, `Mode`, `ZigzagError`, `equilibrium_dimensionless`, `axial_modes_dimensionless`.
- `heating`: `heating_rate_quanta_per_s`, `s_e_from_heating_rate`; `anharmonic`: `AnharmonicTerms`, `anharmonic_estimate`; `micromotion`: `MicromotionIndex`.

**`qutip_trap.light`**
- `beams`: `Beam`, `PolarizationModulation`, `PolGradientBeams`; `comb`: `CombSpec`, a mode-locked Raman drive.
- `raman`: `derive_raman_drive`, `derive_optical_drive`, `derive_light_shift_drive`, their `DerivedDrive`, `ScatteringBudget`.
- `microwave`: `derive_gradient_drive`, `GradientDrive`; `bloch`: `BlochModel`, `SteadyStateReport`, `DetectionRates`.

**`qutip_trap.noise`**
- `model`: `NoiseModel`, noise as spectra, drifts and event rates (`NoiseModel()` is quiet; `summary(device)` lists its channels).
- `spectra`: `NoiseSpectrum` with `white_spectrum`, `ou_spectrum`, `gaussian_spectrum` and `power_law_spectrum`, `Drift`, `Mains`, `Collisions`.
- `sampling`: `NoiseSample`, one draw of every quasi-static parameter, and `quiet_sample`; `collisions`: `CollisionEvent`, `collision_rate_per_ion`.

**`qutip_trap.readout`**
- `detection`: `Detector`, `CameraGeometry`, `RecordModel`, `PhotonRecord`; `presets`: `ApparatusPreset`, `MYERSON_CA40_PMT`, `CRAIN_YB171_SNSPD`.
- `fluorescence`: `ReadoutScheme`, `FluorescenceRates`, `detection_rates_for_ion`.
- `discriminate`: `ThresholdDiscriminator`, `TimeResolvedML`, `AdaptiveML`, `FirstPhoton`, `optimize_threshold`, `POVM`, `povm_for`, `ReadoutOutcome`.

**`qutip_trap.prep`**
- `recipe`: `PreparationRecipe`, `SidebandCoolingSpec`, `standard_recipe(device)`, `run_preparation(device, recipe)` (a `PreparationRun`).
- `doppler`: `doppler_cooling`, `DopplerResult`, `UncooledModeError`; `pumping`: `optical_pumping`, `PumpingResult`; `eit`: `EitClosedForm`.
- `sideband`: `transfer_matrix`, `apply_pulses`, `thermal_distribution`, `mean_occupation`; `closed_forms`: `x0_m`, `lamb_dicke_parameter`, `doppler_force_nbar`, `stenholm_coefficients`.

**`qutip_trap.control.hardware`**: `HardwareChain`, the DDS, modulator and amplifier of Section 7.10.

## The laboratory

**`qutip_trap.experiments`**: every experiment takes the machine first and returns an `ExperimentResult` (`experiments.result`): `value(key)`, `uncertainty(key)`, `converged`, `chi2`, `quality`, the scan `requested` and `realized` (`ScanParameters`), the `subject`, and `table_updates` for `CalibrationTable.updated_with`. Beside its own parameters each takes `table`, `options`, `builder_options`, `sample`, `shots`, `readout`, `seed`, `stream`, `nbar` and `qubit_shifts_hz`.
- `single_ion`: `rabi_scan` (a `RabiScan`), `ramsey` and `ramsey_frequency` (`RamseyFringe`), `sideband_spectroscopy` (`SidebandSpectrum`).
- `motion`: `thermometry` (`ThermometryResult`), `mode_spectroscopy` (`SidebandSpectrum`), `heating_rate` (`HeatingRateFit`).
- `entangling`: `ms_scan`, `ms_phase_scan`, `parity_scan`; `readout`: `detection_histogram`; `imaging`: `crystal_image`.
- `light`: `stark_scan`, `crosstalk_scan`, `field_scan`; `micromotion`: `micromotion_scan`, `device_with_compensation`.
- `result`: the typed results above and `ParityScan`, `DetectionHistogram`, `StarkScan`, `CrosstalkScan`, `FieldScan`, `MicromotionScan`, `CrystalImage`.
- `fitting`: `weighted_fit` (a `FitResult`), `thermal_rabi_model`, `lineshape_model`, `fit_lineshape`, `ReadoutErrors`, `readout_errors_for`, `Observation`.

**`qutip_trap.calibration`**
- `calibrate(machine, *, method="closed_form", experiments=("all",), seed=0, t0_s=None, cache=DEFAULT_CACHE, **scans)`: the `CalibrationReport` whose `table` the scheduler reads; `CalibrationMethod`, `CalibrationCache`.
- `surrogate`: `surrogate_table`, the closed-form table with exact spot checks, and its `SurrogateReport`.
- `experiments`: `full_calibration`, `CalibrationScans`, `CalibrationReport`, `CalibrationError`, the dependency order `ORDER`, `UPSTREAM`, `PRODUCES`, `ALIASES`.
- `entangling`: `exact_gate_check` (a `GateCheck`), `calibrate_entangling_angle`, `thermal_robustness`; `readout`: `calibrate_detection`, `DetectionCalibration`.

**`qutip_trap.control.table`**: `CalibrationTable` (`with_params`, `updated_with`, `entries`, `kind_of`, `uncalibrated`, `to_dict`, `from_dict`), `CalEntry`, `Waveform`, `Segment`, `EntryKind`, `ENTRY_KINDS`.

**`qutip_trap.benchmarks`**
- `rb`: `randomized_benchmarking(machine, qubits, lengths, *, n_sequences=4, shots=200, ...)`, single, simultaneous, two-qubit or Knill-style: an `RBResult` of `RBSequence` records.
- `ghz`: `ghz_fidelity(machine, qubits, ...)` (a `GHZResult`), `ghz_circuit`, `parity_circuit`; `volume`: `quantum_volume` (a `QVResult` of `QVCircuit`), `random_square_circuit`.
- `clifford`: `SINGLE_QUBIT_CLIFFORDS`, `TwoQubitClifford`, `random_two_qubit_clifford`, `decompose_two_qubit_clifford`, `CLASS_SIZES`, `TWO_QUBIT_GROUP_ORDER`.
- `budget`: `gate_channel(machine, kind)`, a native gate kind's `GateChannel` of `StepChannel` steps, and `BenchmarkBudget`, the budget beside every benchmark.
- `error_model`: `error_model(machine, *, qubits=None)`, an `ErrorModel` with `to_ionq_noise`, `to_quantinuum_error_params` and `to_qdk_qubit_params`.

**`qutip_trap.published`**: the published closed forms the app compares with: `HartyParameters`, `simulate_epg_sets`, `ms_alpha`, `ms_gamma`, `kirchmair_populations`, `roos_force_saturation`, `thermal_debye_waller_infidelity`, `ThermalReference`, `ballance_thermal_error`.

**`qutip_trap.noise.summary`**: `average_gate_infidelity`, `entanglement_infidelity`, `pauli_twirl`, `depolarizing_rate`, `choi_from_unitary`, `depolarizing_choi`, `qiskit_depolarizing_lambda`, `rb_error_per_clifford`.

## Lower levels: schedule, engine, Hamiltonian builder

**`qutip_trap.run.pipeline`**: `compile_calibrate_schedule(machine, circuit, *, seed=0)`, the prefix `run`, `schedule` and `estimate` share (a `Prefix`); `execute`, the whole run.

**`qutip_trap.control.schedule`**: `schedule(circuit, device, table, ...)`, native gates to pulses with absolute times; `Schedule`, `ScheduledEvent`, `PlayedGate`, `GateTarget`, `PhaseFrame`, `GateDrive`, `resolve_drives`, `ScheduleError`.

**`qutip_trap.control.pulses`**: `Pulse`, `Drive`, `Tone`, `LightShiftCouplings`.

**`qutip_trap.control.hardware`**: `physical_schedule(device, schedule, table)`, the drives as the ions see them, and `apply_hardware_chain(schedule, hardware)`.

**`qutip_trap.control.shaping`**: `gate_modes(device, ions, beams)` (`GateModes`), `solve_amplitude_modulation`, `solve_fourier_amplitude_modulation`, `solve_frequency_modulation`, `solve_phase_modulation`, `symmetric_pulse`, `ShapedPulse`, `envelope_of`, `SegmentedEnvelope`, `SampledEnvelope`, `integrals`, `GateIntegrals`, `trajectory_sampled`, `closure_rabi_rad_s`, `closure_duration_s`, `CHI_MAXIMAL_RAD`, `ClosureError`.

**`qutip_trap.control.composite`**: `composite_pulse(family, theta_rad, phi_rad)`, a `CompositePulse` (BB1, SK1, CORPSE, ...); **`qutip_trap.noise.decoupling`**: `decoupling_sequence`, `DecouplingSequence`, `filter_function`, `ControlSegment`.

**`qutip_trap.dynamics.engine`**: `JointExactEngine`, the JOINT_EXACT engine (`run_pulses(device, schedule, state, space, sample, seeds, options)` with `options` a `Numerics`, `tomography`, `process_tomography`, `last_report`); `State`, `MotionalModel`, `SeedSpec`, `Traces`, `ChannelSummary`, `EngineReport`, `SegmentReport`, `TruncationLimit`.

**`qutip_trap.dynamics.space`**: `HilbertSpace`, a declared composite space whose operators are built lazily and cached, `ModeTruncation`, `enr_dimension`; **`qutip_trap.dynamics.truncation`**: `TruncationWarning`.

**`qutip_trap.dynamics.hamiltonian`**: `build_hamiltonian(device, pulses, space, ...)`, the one builder: a `BuiltHamiltonian` with a `DriveRecord` per pulse and ion, written under `BuilderOptions`.

**`qutip_trap.dynamics`**
- `evolve`: `evolve`, the integrator ladder, and `convergence_check` (a `ConvergenceReport`); `channels`: `CollapseOp`.
- `tomography`: `TomographyRecord`, `choi_least_squares`, `project_cptp`, `kraus_operators`; `parallel`: `map_tasks`, `worker_count`.
- `multilevel`: `build_multilevel`, `MultiLevelOptions`, `ModeSpec`, the optical-Bloch builder of Section 4.2.8.
- `operators`: `displacement_operator`, `rabi_matrix_element`, `rabi_table`, `debye_waller_factor`.

**`qutip_trap.noise.scattering`**: `scattering_channels`, `scattering_estimates`, `InternalLevels`, `internal_levels`.

**`qutip_trap.run`**
- `job`: `prepare(device, space, table, ...)`, the initial state of a run; `space`: `select_space`, the Section 5.2 mode classes as a `SpaceSelection`.
- `gate_local`: `gate_steps`, `GateStep`, `GateLocalReport`, `GateLocalStep`, the walk of Section 5.4.

## Results and records

**`qutip_trap.run.results`**
- `Result`: the per-shot `bitstrings` and their `counts`, `probabilities` and `error_bars`, the photon records and posteriors of a full readout, the noise samples, `heralds`, `spam`, `final_state` (with `keep_final_state=True`), `diagnostics`, `machine_hash`, `created_at`, `duration_s` and `record`.
  - `to_ionq_v1_probabilities`, `to_ionq_v1_histogram`, `to_ionq_v1_shots`, `from_ionq_v1_shots`: IonQ's v1 formats, decimal keys.
  - `to_ionq_v2_probabilities`, `to_ionq_v2_histogram`, `to_ionq_v2_shots`: IonQ's v0.4 envelope, bitstrings in its wire order, q[0] first.
  - `reversed_bits()`: every key reversed, for the SDKs that write qubit 0's bit first; `sample_of_shot`: each shot's dynamical sample.
  - `to_dict(per_shot=False)` and `from_dict(d)`: the versioned record (schema version 1).
- `Diagnostics`: what the run did and approximated: the level and why, the space and mode classes, the boundary populations and margins, the integrator and tolerances, samples, trajectories and branches, the seeds, the approximations and the intrinsic error budget.
- `RunState`: the machine state a run threads through its shots (ion order, dark and lost ions, events).
- `bitstring_key`, `decimal_key`, `aggregate`: the key conventions.

**`qutip_trap.run.job`**
- `RunRecord`: everything a run produced besides the `Result` (compile report, schedule, space selection, preparation, branches, traces, readout stage and outcome, table, GATE_LOCAL report), carried as `Result.record`; `last_record(result)` reads it.
- `register_fidelity(result, target=None)`, `ideal_register_state(result_or_circuit)`, `to_register_order`: the register state against its ideal.

**`qutip_trap.provenance`**: `load_ledger()`, the records of `docs/provenance/ledger.yaml` by id (`LedgerRecord`); `Cited`, a cited constant of a species table; `TAGS`.
