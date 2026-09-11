# How qutip-trap should be used: a proposal from a survey of quantum SDK APIs

*Status: proposal, 2026-09-11, against commit 7a26a27 (v0.1.0); the phased work plan is [`api_implementation_plan.md`](api_implementation_plan.md). Nothing here is implemented except the Qiskit adapter of Section 4.12. Where a claim about another SDK was verified against a primary source it is stated as fact with the version; where it was not, it says so.*

## 1. The proposal in one screen

qutip-trap already has the hard part that every other SDK lacks: a device described by physics, with everything else derived and every approximation reported. What it lacks is the usage shape the field converged on. The proposal is one ladder, five rungs, one executor:

1. **Add a `Machine`** (device + beam roles + calibration + policy) and make `machine.run(circuit, shots)` the front door. Every surveyed SDK puts configuration on the executor and keeps the program pure (Qiskit `Sampler(backend)`, Cirq `Simulator(noise=, seed=)`, Braket `device.run`, pytket `Backend`, Pulser `EmulatorBackend(config)`). Today the same job needs `**preset.run_kwargs()` threaded through `run`, `calibrate` and every benchmark (20 call sites in the docs, tests and app).
2. **Put beam roles on the `Device`.** The root cause of the threading is that a `Device` does not know which of its beams play which gates; `control/schedule.py` infers a drive only for the trivial cases and otherwise says "pass gate_drives". Fixing the root collapses `DevicePreset` into `Device`.
3. **Replace the 25 keyword arguments of `run` and the 35 fields of `SolverOptions` with three frozen option objects** grouped by what they govern: `Physics` (which effects are simulated), `Numerics` (how the integration is done), `Readout` (how the record is read). Qiskit routed the same fix through an RFC (nested options with dot access, dicts accepted); Pulser's `EmulationConfig` and PennyLane's `ExecutionConfig` are the frozen-dataclass precedents; Qiskit's 23-argument `transpile()` and Bloqade's 13-argument `run()` are the cautionary tales.
4. **Make the namespace the zoom ladder.** `import qutip_trap` gives the cloud customer's dozen names; `qutip_trap.circuit`, `.schedule`, `.dynamics`, `.physics` are the rungs of PLAN.md Section 14.2; `qutip_trap.api` stays exactly as it is, the frozen Appendix E surface. No surveyed SDK ships a separate flat re-export module; the three flat namespaces in the field (cirq 588 names, pennylane 380, qutip 278) put the names on the package root.
5. **Give `Circuit` a builder and defaults.** `Circuit(2).h(0).cnot(0, 1)` instead of `Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))`. Braket, Qiskit and Cirq all build a Bell circuit in one or two lines.
6. **Make `Result` a versioned, serialisable record** with `to_dict()/from_dict()`, a schema, the bit-order sentence on the accessor, and IonQ exporters named by the convention they emit. The v2 exporter is wrong today (verified: it omits the `probabilities/registers` envelope and names the register `c` where IonQ's v0.4 spec fixes `output_all`).
7. **Keep `run` synchronous but observable**: a `progress` callback, and a `Job` handle for the app, following QuTiP's progress option and Braket's local task handle.
8. **Publish the inverse direction.** Every vendor emulator surveyed takes phenomenological numbers (IonQ `r_1q`/`r_2q`, Quantinuum `p1`/`p2`/`p_meas`, AQT two constants, QDK Pauli probabilities) and none derives them; QDK has physical parameters only in the estimator that never simulates. A `machine.error_model()` that emits those numbers from the simulated device is the niche nobody occupies.
9. **Write the conventions and the deprecation policy down now**, while the surface is small: a typed deprecation warning with a removal version (PennyLane), a `deprecated(deadline, fix)` helper with call rewriting (Cirq), "public means documented" (Qiskit), a docs test that the bit-order sentence appears once (CUDA-Q's docs contradict themselves on it).

Sections 2 and 3 are the evidence and the audit; Section 4 is the design with code; Section 5 is the migration in three additive phases; Section 6 lists the decisions left to the author.

## 2. What the field does, and what it regretted

### 2.1 Method

Six research passes on 2026-09-11 read the current releases from source and primary documentation: Qiskit 2.5.2 with Aer 0.17.2, IBM Runtime 0.49.0, Dynamics 0.6.0 and Experiments 0.14.2; Cirq 1.7.0; CUDA-Q 0.15.1; PennyLane 0.45.1; QuTiP 5.3.1 and qutip-qip 0.4.2; Pulser 1.9.1; Bloqade analog 0.16.9 and circuit 0.15.1; Amazon Braket 1.127.0; IonQ REST v0.4 (spec dated 2026-09-10) with qiskit-ionq 1.1.1, cirq-ionq 1.7.0, pennylane-ionq 0.45.0 and ionq-core 0.1.1; Quantinuum pytket 2.18.1, pytket-quantinuum 0.59.2, guppylang 1.0.4, selene-sim 0.3.2; AQT qiskit-aqt-provider 1.15.0; Microsoft qdk 1.32.3; plus IonSim.jl 0.5.2 and oqd-trical as academic peers. Nine SDKs were installed and introspected for namespace sizes, import times and bit order. A seventh pass over the physics-level toolkits (C3, scqubits, Qibolab and Qibocal, Strawberry Fields, Perceval, True-Q, Mitiq, Stim, BQSKit, dynamiqs, Boulder Opal, OpenPulse) is summarised in Section 2.5.

### 2.2 The abstraction ladders

Every mature SDK exposes the same five rungs under different names. The table names the rung, the object at that rung, and the one call that moves the user down or back up.

| Rung | Qiskit 2.5 | Cirq 1.7 | PennyLane 0.45 | Pulser 1.9 | qutip-qip 0.4 | Braket 1.127 | Bloqade analog 0.16 |
|---|---|---|---|---|---|---|---|
| one call | `StatevectorSampler().run([qc], shots=)` | `cirq.sample(circuit, repetitions=)` | QNode call | `QutipBackendV2(seq).run()` | `qc.run(state)` | `device.run(circuit, shots=)` | `program.bloqade.python().run(shots)` |
| executor | primitives over a `BackendV2`; options on the primitive | `Sampler`; `Simulator(dtype, noise, seed)` | `qml.device(name)`; frozen `ExecutionConfig` | `EmulatorBackend` + `EmulationConfig` | `Processor.run_state(...)` | `LocalSimulator("braket_dm", noise_model=)` | backend route `.bloqade.python()` |
| program | `QuantumCircuit`; PUBs `(circuit, bindings, shots)` | `Circuit` of `Moment`s | tape | `Sequence(register, device)` | `QubitCircuit` | `Circuit().h(0).cnot(0, 1)` | builder chain -> `Routine` |
| device description | `Target` (instruction -> `InstructionProperties(duration, error)`) | `DeviceMetadata`, `NoiseProperties` | plugin entry point | frozen `Device` of hardware limits with derived properties | `Model(params)` | pydantic `DeviceCapabilities` with schema header | `QuEraCapabilities`, all `Decimal` |
| noise | Aer `NoiseModel.from_backend`; Runtime resilience options | `NoiseModel`, `NoiseProperties.build_noise_models()` | `qml.NoiseModel({condition: fn})` | frozen `NoiseModel`; noise types derived from the parameters given | `Noise.get_noisy_pulses(pulses)` | `NoiseModel.add_noise(noise, criteria)` | none in analog; dialect statements in circuit |
| pulse or dynamics | removed in 2.0; Dynamics archived | none | `qml.pulse` (`ParametrizedEvolution`) | `sampler.sample(seq)`, `get_hamiltonian(t)` | `Processor.get_qobjevo()` | `PulseSequence`, `defcal`; AHS | `.hamiltonian().evolve()` |
| escape down | stage-editable pass manager, `transpile(callback=)` | `simulate_moment_steps`, `SimulationState` | `device_options` mapping | `run_from_sequence_samples` | `set_tlist/set_coeffs` | verbatim box, `gate_definitions` | `.parse_*()` on every node |
| escape up | `Target.add_instruction` | `optimize_for_target_gateset` | transforms | `Device.to_virtual()` | subclass `Model` + `GateCompiler` | `Gate.PulseGate(pulse_sequence)` | `register.apply(sequence)` lifts IR back into the chain |

Six regularities:

- **The hello world is three concepts.** A program, an executor, a shot count. Nobody asks the newcomer to calibrate first or to name a beam.
- **The executor owns configuration; the program stays pure.** Options live on the primitive, simulator, backend or config object and are copied on construction (Qiskit: "changing the original dictionary or options instance doesn't affect the options that the primitive owns"). Per-call arguments are the program and the shots, occasionally a seed.
- **Device description is data, execution is behaviour.** pytket replaced a behaviour-carrying `Device` with the data record `BackendInfo` in 0.12; Qiskit's `BackendV2` splits immutable facts (`target`, `dt`, `num_qubits`) from mutable `options` and states the rule: "if a property is immutable to the user ... that should be a configuration of the backend class itself instead of the options". Pulser's frozen `Device` carries derived physics as properties (`interaction_coeff` from `rydberg_level`).
- **Noise is an object attached at the executor, never a boolean.** The two vendors that use a boolean (AQT `with_noise_model`, pytket-quantinuum `noisy_simulation` plus an opaque `options["error-params"]` dict) are the two whose noise is least inspectable. Pulser's move in 0.20 is the model: the explicit `noise_types` tuple was removed so that "only the relevant noise parameters should be provided and the associated noise types are activated accordingly".
- **Escape hatches are named on the object being escaped.** Qiskit Dynamics' `DynamicsBackend` exposed the live solver as `backend.options.solver`, a `solve()` that bypasses measurement sampling, and a swappable `experiment_result_function` with the default exported; Braket lets a circuit stay at its rung while only the `defcal` changes (`gate_definitions=`); Bloqade's `.apply(sequence)` lifts a hand-built IR back into the fluent chain. The ladder is bidirectional.
- **Kwarg explosions get refactored, always after the fact.** Qiskit's `transpile()` has 23 keyword parameters; the primitives got nested option dataclasses through an RFC. Bloqade's `run()` has 13 mixing truncation, approximation, implementation strategy, parallelism and tolerances, and its `atol`/`rtol` defaults are declared three times with two of them swapped. Quantinuum moved `n_qubits` and `max_cost` off the config onto the job because they are "per-submission, not per-device".

### 2.3 What the field regretted

The deprecation notes are the most useful design evidence, because they record a cost that was paid.

- **Pulse-level rungs die unless the physics is the product.** Qiskit deprecated `qiskit.pulse` in 1.3 (2024-11) promising it "will be moved to the Qiskit Dynamics repository", removed it in 2.0 (2025-03) with the note that the migration "has been put on hold due to Qiskit Dynamics development priorities", and `qiskit.compiler.schedule` went "with no proposed alternative". qiskit-dynamics 0.6.0 is archived, described as "DEPRECATED", and pins `qiskit<=1.3`. qiskit-experiments deleted its whole calibration layer (`Calibrations`, `Rabi`, `QubitSpectroscopy`) in PR #1511 with the admission that it was "better to do a clean removal of all pulse features than to adhere strictly to the deprecation policy". IBM's hardware notice removed pulse access on 2025-02-03 to "focus more of our attention and resources on higher-level services". qutip-trap should own its pulse, dynamics and calibration types outright and touch the gate-model SDKs only as OpenQASM text or a thin adapter.
- **Two conventions for one quantity is a permanent tax.** IonQ's v1 results key on decimal integers with qubit 0 as the least significant bit; v2 keys on zero-padded bitstrings; `x q[0]` on three qubits is `"1"` in one and, by the surveying pass's reading of the docs, `"100"` in the other. IonQ's one circuit document carries `rotation` in radians for the `qis` gate set and `phase` in turns for the native one. Aer reads gate lengths in seconds on the `Target` path and nanoseconds on the `BackendProperties` path, and `thermal_relaxation_error(t1, t2, time)` documents no unit at all. CUDA-Q's docs call `get_state` big-endian on one page and little-endian on another. pytket deprecated `get_distribution()` because it "silently returned either exact or sampled probabilities".
- **Derived quantities written back into the input record rot.** qutip-qip's `SCQubitsModel._compute_params()` stores `zz_coeff` and `wq_dressed_cavity` into the same flat `params` dict that holds the user's `wq` and `g`; `ZZCrossTalk` then pulls seven derived keys out by string, which is how PR #270 ("use wq instead of wr for zz_coeff") became possible. qutip-qip's own issue #217, open since 2023-09, asks for exactly the composition (model, transpiler, compiler as parts; presets as thin factories; "functionality to be aggregated rather than inherited") that never shipped.
- **Seeds are an afterthought almost everywhere.** Braket's default simulator has no seed parameter (an unseeded `np.random.default_rng()`; the AHS path uses the legacy global RNG); Pulser and qutip-qip draw from the global `np.random`; bloqade-analog breaks reproducibility under `multiprocessing=True`; PennyLane's default is `seed="global"`. QuTiP's `mcsolve` spawns a `SeedSequence` per trajectory, records them in `Result.seeds`, and documents `seeds=prev_result.seeds`. Stim states three disclaimers (results change across versions, SIMD widths and shot counts). qutip-trap's keyed substreams by `(sample, trajectory, shot, ion, channel)` are ahead of every SDK surveyed; the gap is documentation of exactly what is invariant.
- **Vendors expose phenomenology, not physics.** IonQ's noise model is a depolarizing channel after each ideal gate with two scalars per machine (aria-1: `r_1q = 5e-4`, `r_2q = 1.33e-2`; note `F_avg = 1 - r/2` for one qubit and `1 - 3r/4` for two, so `r_2q = 0.0133` is 0.998 % infidelity, not 1.33 %), and the Forte models were replaced in 2025-09 by something still undocumented. Quantinuum's `UserErrorParams` has 28 fields of which exactly two are dimensional rates, with `None` meaning "inherit the machine's calibration" so that disabling a mechanism needs an explicit `0`. AQT's noisy simulator is `depolarizing_error(0.003)` on `R` and `0.01` on `RXX`. QDK's `PauliNoise` was introduced "for education purposes" while its Resource Estimator carries real gate times and error rates for a device it never simulates. Braket's emulator derives depolarizing rates from fidelities with stated formulas (`p_1q = (1 - F) * 3/2`, `p_2q = (1 - F) * 5/4`). Nobody goes the other way.
- **The public API is what is documented.** Qiskit: "An object is publicly documented if and only if it appears in the hosted API documentation ... The presence of a docstring in the Python source is not sufficient", and "the only public-API import location for a given object is the location it is documented at". Cirq: "only code reachable through the `cirq` Python module is covered", excluding `contrib` and anything named `experimental`. Cirq's `_compat.deprecated(*, deadline, fix)` validates the deadline against `^v\d+\.\d+$`, requires `fix` to be a full sentence, and `deprecated_parameter(match=, rewrite=)` rewrites old calls silently.

### 2.4 Conventions, measured

| Convention | Field | qutip-trap today |
|---|---|---|
| root namespace size | qiskit 27, pytket 9, pulser 18 (including device instances as constants), qsharp 23; cirq 588, pennylane 380, qutip 278 flat | `qutip_trap` exports `__version__` only; `qutip_trap.api` 237 names (128 classes, 104 functions) |
| cold import | pytket 0.11 s, qiskit 0.28 s, qutip 0.60 s, cirq 0.95 s, pennylane 1.59 s; CUDA-Q lazy `__getattr__` | `import qutip_trap` 0.05 s; `import qutip_trap.api` 0.49 s (qutip alone 0.37 s on this machine) |
| bit order for X on qubit 0 of 2 | Qiskit `'01'` and IonQ v1 `"1"` (qubit 0 least significant); Cirq int 2, PennyLane `'10'`, Braket `'10'` (qubit 0 leftmost); pytket int tuples plus a `BasisOrder` toggle | qubit 0 least significant, stated as `Result.bit_order == "qubit0_lsb"` |
| frequency | Pulser rad/us per field; Braket AHS rad/s; PennyLane pulse "MHz ... converted internally to angular frequency"; QuTiP and CUDA-Q dimensionless with the user writing `2*np.pi`; scqubits a global `set_units` switch | Hz public, rad/s internal, one conversion in `units.py` |
| angles | radians in Qiskit, Cirq, PennyLane, Braket; half-turns in pytket and Guppy; turns on the IonQ wire and in `cirq_ionq.GPIGate`; units of pi on the AQT wire | radians, turns only in `io/ionq.py` |
| result serialisation | qiskit legacy `Result.to_dict/from_dict`; V2 primitives none; Braket schema envelope with `id`, `shots`, `deviceId`, timestamps even for local tasks; Pulser JSON schemas with `version` and `pulser_version` | four IonQ exporters, no general envelope |
| seeds | QuTiP per trajectory and returned; qsharp split classical from quantum; Braket unseedable | one root seed keyed by `(sample, trajectory, shot, ion, channel)` |
| options | Qiskit nested dataclasses with dot access and `update()`; PennyLane frozen `ExecutionConfig` with a `device_options` escape; Pulser `EmulationConfig` with per-backend `config_type`; QuTiP a dict that rejects unknown keys | `SolverOptions` (35 fields) plus 25 keyword arguments on `run` |
| execution | pytket `ResultHandle` and Braket `LocalQuantumTask` give local runs the remote surface; QuTiP `progress_bar` plus `timeout`/`target_tol`; qsharp per-shot `on_result` | synchronous `run`, no progress, no handle |
| docs | Cirq's navigation is the pipeline (Build, Simulate, Transform, Hardware, Noise, Experiments) with `style.md` and `nomenclature.md`; Pulser has a Conventions page | four pages by topic, examples executed by a test |

### 2.5 The physics-level toolkits

The toolkits that model a device from physical parameters, calibrate it, or benchmark it are the nearest neighbours to qutip-trap's rungs 2 to 4 and its laboratory. Versions: C3-toolset 1.4 (last release 2021-12, dormant), scqubits 4.3.1, Qibo 0.3.4 with Qibolab 0.2.16 and Qibocal 0.2.6, Strawberry Fields 0.23.0 (archived), Perceval 1.2.4, True-Q 2.14.5 (docs no longer public; read from a 2025-12 snapshot), Mitiq 1.1.0, Stim 1.16.0, BQSKit 1.2.1, dynamiqs 0.3.4 on PyPI (0.3.6 documented but unpublished), Boulder Opal 6.0.0 and Fire Opal 12.2.0 (signatures read from the wheels; the docs returned HTTP 429), OpenQASM 3.1 with OpenPulse.

| Toolkit | What it gets right | What it gets wrong |
|---|---|---|
| Qibolab / Qibocal | the device split four ways: channel ids (topology), `configs` (setpoints the hardware plays), `natives` (gate recipes), and `calibration.json` (measured facts as `(value, error)` pairs); `Protocol(acquisition, fit, report, update)` types the calibration flow `Parameters -> Data -> Results -> Platform`, `acquire` and `fit` rerun separately against stored data, and the update is a proposal gated by an acceptance test (`if chi2 > 2: raise`) before anything is written; `Platform.execute(sequences, sweepers, **options)` validates the options into one model where `None` means "the platform default"; `estimate_duration()` tells the cost before running | the protocol's types are recovered by `inspect.signature` with a comment admitting unease; no changelog for the 0.1 to 0.2 break; six doc pages carry a "partially updated" banner |
| C3-toolset | the control electronics as a DAG of devices (`LO`, `AWG`, `DigitalToAnalog`, `Response`, `Mixer`, `VoltsToHertz`) validated for arity and matching resolution, the direct ancestor of qutip-trap's `HardwareChain`; one hjson file with a `c3type` discriminator for the whole model; calibration by a callable so hardware and simulator share the code | `Quantity` infers the 2 pi by substring-matching the unit label (`if "2pi" in unit: pref = 2*np.pi`); no propagator convergence metric; "API breaking changes between minor releases" |
| scqubits | physical energies as named constructor arguments (`Transmon(EJ, EC, ng, ncut, truncated_dim)`); physical coherence-time methods gated by `supported_noise_channels()`; named sweep axes that survive slicing | `ncut` (basis cutoff) against `truncated_dim` (levels kept) named `cutoff` on another class; "These cutoffs must be varied by the user to ensure convergence" with no checker; a global `set_units` that warns after the fact |
| Strawberry Fields | the Fock cutoff is one backend option (`cutoff_dim`); `repr` says what a result holds (`<Result: shots=1, num_modes=3, contains state=True>`); `prog.compile_info` records the device and compiler used | the only truncation check is the user remembering `state.trace()`; a module-level `sf.hbar = 2`; archived |
| Perceval | `NoiseModel` fields are bounded descriptors whose defaults all mean "no noise", so `NoiseModel()` is a perfect simulation and serialisation emits only the non-defaults; `Processor(backend, experiment, noise)` separates physics from engine; the QPU's published calibration converts into the same `NoiseModel` a user would write | the same physics has two names either side of `Source.from_noise_model` (`brightness` against `emission_probability`, `transmittance` against `losses = 1 - transmittance`) |
| True-Q | a fit result is directly a noise source (`Simulator().add_knr_noise(knr_circuits)`), so characterisation output and simulator input are one object; estimates are `(name, val, std)` tuples; a bias warning in the docstring where the estimator is biased | angles in degrees; pickle persistence; pinned to Qiskit < 1 and QuTiP < 5 |
| Mitiq | stability gated by import path: `mitiq.experimental` is "not covered by mitiq's semantic versioning guarantees"; the executor callable is the whole device abstraction | the executor's return type is read from its annotation by reflection |
| Stim | noise is program text, so 44 public names cover 100 gates and 13 channels; `Circuit.generated(...)` documents the exact instruction each parameter emits; the seed docstring states what is not stable | Pauli noise only, by design |
| BQSKit | `MachineModel(num_qudits, coupling_graph, gate_set)` separate from the circuit and free of physics; an accuracy guard (`error_threshold`) paired with a knob for the guard's own accuracy (`error_sim_size`) that logs rather than raises | none of note |
| dynamiqs | tolerances live on the method object (`Tsit5(rtol, atol, max_steps)`) while conveniences were flattened into keyword arguments in 0.3.6 "to align with common practices in scientific python"; `result.infos` is a real convergence report (`11 steps (11 accepted, 0 rejected)`) | the documented version is not the installable one |
| Boulder Opal | `obtain_ion_chain_properties(atomic_mass, ion_count, center_of_mass_frequencies, wavevector, laser_detuning)` returns Lamb-Dicke parameters, relative detunings and eigenvectors, and `ms_simulate`/`ms_optimize` take exactly six physics arguments; `infidelities` is returned only when a `target_phases` was given, so no fidelity is reported without a target; validation constraints travel with the parameter as `Annotated` types | inside that one module amplitudes are rad/s, detunings Hz, and two of the six inputs carry no documented unit; `metadata` comes with "No guarantees are made about the contents" |
| OpenQASM 3 / OpenPulse | `duration` is a type with mandatory unit suffixes (`300ns`, `800dt`); `defcal` resolution is most-specific-match over (gate, parameter values, physical qubits), stated with an example; `frame` is a clock and a carrier, which is what makes a virtual Z a bookkeeping operation | frequency on `newframe` is a bare float; the spec admits a `defcal` may touch qubits not in its signature (crosstalk) and defers the fix |

What this adds to Sections 2.2 to 2.4: the calibration loop has a settled shape (acquire, fit, report, update, with the update a proposal); truncation adequacy must be checked by the code, not by a documented habit; a "no noise" default should be the empty constructor; a fit should be constructible into a noise model; guards should log with a reason and carry a knob for their own accuracy; and the units lesson holds even inside the one module in the survey built specifically for trapped-ion physics.

## 3. Where qutip-trap stands

### 3.1 Measured

| Quantity | Value |
|---|---|
| public names in `qutip_trap.api.__all__` | 237 (128 classes, 104 functions, 5 other) |
| parameters of `run` | 28, of which 25 keyword-only |
| fields of `SolverOptions` | 35 |
| fields of `Diagnostics` / `Result` / `Device` / `NoiseModel` | 32 / 16 / 10 / 21 |
| `**preset.run_kwargs()` call sites in docs, tests and app | 20 |
| `gate_drives=` occurrences in package and tests | 94 |
| gaps the app records against the core (`qutip_trap_app/src/qutip_trap_app/core.py`, `CORE_GAPS`) | 12 |
| names the app imports from outside `qutip_trap.api` | 40, from eleven internal modules |

### 3.2 What is already better than the field

- A `Device` that is physics, frozen, hashed, with `derived()` naming a provenance record for every number. No surveyed SDK has this; the nearest are Pulser's device with derived properties and IonSim.jl's `Chamber`.
- A `Result` whose `Diagnostics` states the level that ran, the space, the mode classes, the boundary populations, the dropped weights, the integrators and every approximation. Qiskit Dynamics' docs admit that setting qubit frequencies on its `target` "does not impact the behaviour of the `DynamicsBackend` itself. It is purely a data field"; qutip-trap has no such fiction.
- Keyed seeds by `(sample, trajectory, shot, ion, channel)`, reproducible across worker counts.
- One unit discipline: Hz at the boundary, rad/s inside, the 2 pi in one function, turns only at the IonQ boundary. The field's failures here (Section 2.3) are all failures to do this.
- An experiment API that runs the laboratory's scans through the same engine and fits them with the laboratory's uncertainties, and a calibration that follows the dependency graph and refuses a fit whose upstream entry is uncalibrated. qiskit-experiments deleted the equivalent; Cirq's lives in `cirq.experiments` with one result type per experiment.
- A frozen public surface with a test (`tests/test_api_freeze.py`) that checks every Appendix E declaration.
- IonQ circuit JSON both ways, and the native MS matrix in IonQ's own tensor order (verified by the vendor pass: `qutip_trap.control.native.ms` matches `ionq_core.gates.ms_matrix`; `qiskit_ionq.MSGate` is its SWAP conjugate).

### 3.3 Where it hurts

1. **There is no executor.** A `DevicePreset` is a `Device` plus two drive maps and a detection-beam index, and `run`, `calibrate`, `randomized_benchmarking`, `ghz_fidelity`, `quantum_volume` and `gate_channel` each take `gate_drives=` and `entangling_drives=` because the `Device` does not record which beams address which ion. `light/roles.py` infers roles by wavelength (resonant against far-detuned) and `default_gate_drives` covers only "none, one, or one Raman pair".
2. **`run` mixes four concerns in 25 keyword arguments**: physics switches (`noise`, `internal_levels`, `stark_compensation`, `crosstalk_suppression`, `entangler`, `channels`, `builder_options`), numerics (`options`, `space`, `caps`, `enr_group`, `samples`, `parallel`), readout (`readout`, `discriminator`, `povm_samples`), and bookkeeping (`table`, `t0_s`, `shot_period_s`, `seed`, `keep_final_state`, `calibrate_kwargs`, `level`). `SolverOptions` repeats the mixture: `scattering_channels`, `intensity_noise_channels` and `hardware_chain` are statements about which physics is simulated, not about the solver.
3. **Building a circuit is verbose and the importers are functions.** `Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))` for a Bell pair; `load_openqasm2` and `load_ionq_json` rather than constructors or a stdlib-shaped `io` module; the terminal `measure` is a required positional tuple; classical registers are flattened on import.
4. **The namespace has no rungs.** The app's `core.py` shows the consequence: 80 names from `qutip_trap.api` and 40 more from eleven internal modules, with twelve recorded gaps (no concrete engine exported, no public Hamiltonian builder, no per-time Fock marginals in `Traces`, no per-gate register state in a GATE_LOCAL run, closed forms and readout presets not public, noise-sample key builders not public). Those gaps are precisely the escape hatches of rungs 2 to 4.
5. **`Result` is IonQ-shaped and one-way.** Four exporters, no `to_dict`/`from_dict`, no schema, no version or timestamp in the envelope. The v2 exporter emits `{register: {bitstring: p}}` with the register named `"c"`; the v0.4 spec's `ionq.result.probabilities.json.v2` is `{"probabilities": {"registers": {"output_all": {...}}}}` (verified against `https://api.ionq.co/v0.4/api-docs`). The vendor pass also reports that v2 strings put `q[0]` leftmost, opposite to the `bitstring_key` the exporter uses; the spec text does not state the character order and the Bell example is symmetric, so that half is unverified here. `dump_ionq_json` emits only the `input` object, not a job body.
6. **The app had to build a job model.** `JobSpec` (circuit, shots, seed, device reference, drive references, calibration reference, options mapping, level, samples, readout, noise, `qubit_to_ion`, waveform overrides, requests, label), `execute(job)`, a worker process with progress, and its own re-simulation cache. Every one of those is a consumer need the core does not meet.

## 4. The design

### 4.1 The ladder

The five levels of PLAN.md Section 14.2 are the five rungs of the API. Each rung is a module with a small interface, each object on a rung has a named way down and a named way back up, and the app's routes, the docs' chapters and the package's modules share the vocabulary.

| Rung | Module | The user constructs | The user calls | The user gets | Down |
|---|---|---|---|---|---|
| 0 machine | `qutip_trap` | `Machine` (from a preset or from a `Device`), `Circuit` | `machine.run(circuit, shots)` | `Result` | `result.diagnostics`, `machine.compile` |
| 1 circuit | `qutip_trap.circuit` | `Circuit`, gates | `machine.compile(circuit)`, `ideal_probabilities` | `CompileReport`, native `Circuit`, `GateChannel` | `machine.schedule` |
| 2 schedule | `qutip_trap.schedule` | `Waveform`, `Pulse`, `GateDrive`, `HardwareChain` | `machine.schedule(circuit)`, the AM/FM solvers | `Schedule` with played gates | `machine.engine` |
| 3 dynamics | `qutip_trap.dynamics` | `HilbertSpace`, `State`, `Numerics` | `prepare`, `engine.run_pulses`, `build_hamiltonian`, `gate_channel` | `Traces`, `ChannelSummary`, the built Hamiltonian | `qutip` objects |
| 4 physics | `qutip_trap.physics` | `Species`, `Trap`, `Beam`, `Field`, `NoiseModel`, `Detector`, `Device` | `solve_crystal`, `device.derived()`, `device.specs()` | `Crystal`, `DerivedQuantities` with provenance | the ledger |
| the laboratory | `qutip_trap.experiments`, `.calibration`, `.benchmarks` (unchanged packages) | scan ranges | `rabi_scan(machine, ...)`, `calibrate(machine)`, `randomized_benchmarking(machine, ...)` | per-experiment results, `CalibrationTable`, budgets | `Result` of every shot they took |

`qutip_trap.api` is untouched: the Appendix E names, the freeze test, the app's contract. The new modules re-export the same objects under rung names, the way `api.py` already re-exports them under one name. Adding names never breaks the freeze test; only renaming or removing would, and this proposal does neither.

### 4.2 Rung 0: the machine

Three designs were sketched before choosing.

- **A. `Machine` object.** Device, beam roles, calibration table, option objects and the level policy in one frozen record with `run`, `compile`, `schedule` and `calibrated` methods. This is the executor pattern of every cloud SDK; it kills the kwarg threading; it is the thing the app records and hashes. The risk is a shallow "manager" class, avoided only if the machine genuinely owns drive-role resolution, calibration caching, level selection and option defaults.
- **B. Roles on the `Device`, functions unchanged.** Add `roles` to `Device`; `run(circuit, device, shots)` needs no drive maps; split the options. Smallest change and the freeze test is trivially safe, but the caller still threads a table and three option objects through every call, and nothing owns the configuration a run record needs.
- **C. Rung modules only.** Fixes the namespace and the documentation without touching the call shape.

The recommendation is A on top of B's root-cause fix, with C as the namespace. B alone leaves the app's `JobSpec` with nowhere to go; A alone would keep the drive maps as machine state that ought to be device facts.

```python
# proposed
import qutip_trap as trap

machine = trap.presets.yb171_chain(2)                 # a Machine, not a DevicePreset
result = machine.run(trap.Circuit(2).h(0).cnot(0, 1), shots=2000)
print(result.counts, result.error_bars, result.diagnostics.level)
```

against today's `docs/examples.md`:

```python
# today
from qutip_trap.api import Circuit, Operation, SolverOptions, calibrate, run, yb171_chain

preset = yb171_chain(2)
device = preset.device
windows = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
table = calibrate(device, pairs=[(0, 1)], detection_records=2000, detection_windows_s=windows, **preset.run_kwargs())
bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
result = run(bell, device, 2000, table=table, options=SolverOptions(branch_weight_min=1e-3), **preset.run_kwargs())
```

The `Machine` record:

```python
@dataclass(frozen=True)
class Machine:
    """A trapped-ion computer as a client sees it: the physical device, which beams play which gates, the calibration
    it runs on and the policy that turns a circuit into a Result. Immutable; derive variants with dataclasses.replace."""

    device: Device                      # rung 4, unchanged; now carries BeamRoles (Section 4.6)
    table: CalibrationTable | None = None   # None: the closed-form surrogate, cached by device hash (today's behaviour)
    physics: Physics = Physics()        # which effects are simulated (Section 4.7)
    numerics: Numerics = Numerics()     # how the integration is done
    readout: Readout = Readout()        # how the record is read
    level: Level = Level.AUTO           # JOINT_EXACT inside the guards, GATE_LOCAL above them
    name: str = ""

    def run(self, circuit: Circuit, shots: int, *, seed: int = 0, keep_final_state: bool = False,
            progress: Callable[[Progress], None] | None = None) -> Result: ...
    def submit(self, circuit: Circuit, shots: int, *, seed: int = 0) -> Job: ...     # Section 4.9
    def compile(self, circuit: Circuit) -> CompileReport: ...                          # rung 1
    def schedule(self, circuit: Circuit) -> Schedule: ...                              # rung 2, compile + calibrate + schedule
    def calibrated(self, method: CalibrationMethod = "closed_form", **scans) -> Machine: ...  # pins a table
    def estimate(self, circuit: Circuit) -> Estimate: ...                              # level, space, cost before running
    @property
    def engine(self) -> JointExactEngine: ...                                          # rung 3
    def error_model(self) -> ErrorModel: ...                                           # Section 4.8
    def hash(self) -> str: ...                                                         # device hash + table + policy
    def specs(self) -> str: ...                                                        # Pulser's print_specs
```

`estimate` answers "what will this cost" before anything is integrated: the level `auto` would resolve to and why (the joint dimension and the drive-operator non-zeros against the Section 11.5 guards), the space, the mode classes, and a wall-time guess from the Section 11.2 cost model. Qibolab's `Platform.estimate_duration` is the precedent, and the app's interactive budgets (PLAN.md Section 14.7) need exactly this number to decide between running at once and starting a background job.

What the machine hides: drive-role resolution (from `device.roles`), calibration caching (today's `calibration.cache` keyed by device hash, seed and scan settings), level selection and its reason, the compile-calibrate-schedule-prepare-evolve-read pipeline, the option defaults, and the shot clock. What it does not hide: the device, the table and the three option objects are plain attributes, because the app and the physicist both need to read them. The existing function `run(circuit, device, shots, ...)` stays in `qutip_trap.api` and becomes `Machine(device, ...).run(circuit, shots)` with the old keyword arguments rewritten into the option objects by the deprecation helper of Section 4.11.

Naming: `Machine` is the word PLAN.md Section 14.2 already uses for Level 0 and the word the app's device card uses. `Emulator` (Pulser, Quantinuum) and `Backend` (Qiskit, pytket) are the alternatives; `Device` is taken by the physical record and should stay so, exactly as pytket separated `BackendInfo` (data) from `Backend` (behaviour).

Presets return machines: `yb171_chain(n) -> Machine`, `ca40_optical(n) -> Machine`, following Cirq's `create_default_noisy_quantum_virtual_machine(processor_id) -> engine` and Pulser's module-level device instances. `DevicePreset` and `run_kwargs()` remain for one release as deprecated aliases.

### 4.3 Rung 1: the circuit

`Circuit` stays a frozen dataclass and gains a persistent builder, defaults, registers and constructors:

```python
c = trap.Circuit(2).h(0).cnot(0, 1)                 # measure defaults to every qubit; each call returns a new Circuit
c = trap.Circuit(2).gpi2(0, phase=0.0).ms(0, 1, phases=(0.0, 0.0), angle=0.25)   # native gates, radians in, turns at the wire
c = trap.Circuit.from_openqasm(text)                # was load_openqasm2; keeps creg names in c.registers
c = trap.Circuit.from_ionq(job_or_input)            # was load_ionq_json; accepts a v0.3 or v0.4 job body or a bare input
c.to_openqasm(); c.to_ionq()                        # exporters live with the type; io.qasm2 / io.ionq are the module forms
c.measure(0)                                        # restricts the terminal measurement
c.registers                                         # {"c": (0, 1)} by default; the IonQ exporter maps the default to output_all
```

Three details from the evidence. The default measurement defines an error out of existence (a circuit with no `measure` is measured in full, as Braket and IonQ do), so the constructor's third positional argument stops being required. `registers` becomes a field because the vendor pass found both `qiskit-ionq` and `cirq-ionq` smuggling register structure through the job `metadata` string until IonQ added named registers in v2; `load_openqasm2` flattens `creg` names today and `Result.to_ionq_v2` asks the caller to reconstruct them. And gate methods are the one place the IonQ turns convention meets the radians convention: the builder takes radians like the rest of the package, and only `to_ionq()` converts, with the conversion named (`turns_from_rad`).

`machine.compile(circuit) -> CompileReport` carries the native circuit, the per-block residuals and the whole-circuit residual (all of which exist today), and `ideal_probabilities`/`circuit_unitary` stay as functions. A `dry_run` in IonQ's sense is `machine.schedule(circuit)`: compile, calibrate and schedule without integrating.

### 4.4 Rung 2: the schedule

`machine.schedule(circuit) -> Schedule` is the compile-calibrate-schedule prefix of `run`, with `played` gates and the measure event, exactly what the app's Level 2 draws. `Waveform`, `Segment`, `Pulse`, `Drive`, `Tone`, `GateDrive`, `HardwareChain`, `physical_schedule`, the composite-pulse library, the comb specification and the three closure solvers are re-exported under `qutip_trap.schedule`. The two additions:

- `Waveform.trajectory(...)` and `Waveform.integrals(...)` as public methods, closing the app gap that imports `envelope_of`, `integrals_segmented` and `trajectory_sampled` from `control.shaping`.
- `CalibrationTable.with_params(**overrides) -> CalibrationTable`, after Cirq's `NoiseProperties.with_params(gate_times_ns=37)`: the app's `waveform_overrides` ("a detuning set by hand at Level 2, played as written") is this operation, done on the app side today.

### 4.5 Rung 3: the dynamics

`qutip_trap.dynamics` already exists as the engine package. It becomes the public rung by exporting what the app imports privately today: `JointExactEngine` (the concrete `PulseEngine`), `build_hamiltonian` with `BuilderOptions`, `BuiltHamiltonian` and `DriveRecord`, `prepare`, `State`, `Traces`, `HilbertSpace`, `gate_channel` and the tomography records. Two changes to `Traces`, both from the app's gap list: an optional per-time `mode_marginal` store, and the wall time per pulse. The GATE_LOCAL report gains the register after every step (or the idle channels), which the app composes itself today.

The precedent for the shape is `DynamicsBackend`: `machine.engine` is the live engine (its `backend.options.solver`), `engine.run_pulses(...)` returns traces without any readout (its `solve()`), and `Readout` (Section 4.7) accepts a callable so the whole fluorescence-and-discriminator stage is swappable with the default exported (its `experiment_result_function`).

### 4.6 Rung 4: the physics

`Device` keeps every field and gains `roles`:

```python
@dataclass(frozen=True)
class BeamRoles:
    """Which beams play which part. None on a field means 'infer from the wavelengths' (today's light/roles.py rule)."""
    gate: Mapping[int, GateDrive] | None = None          # per ion: the single-qubit drive and its beams
    entangling: Mapping[int, GateDrive] | None = None    # per ion: the entangling drive (the global pair)
    detection: int | None = None                         # the detection beam
```

`Device.roles: BeamRoles = BeamRoles()`; the scheduler's `default_gate_drives` becomes the resolution rule for the `None` fields; the 94 `gate_drives=` occurrences disappear from the package and tests over the migration. This is the information-hiding fix: the fact "beams 2 and 3 address ion 0" is a fact about the apparatus and belongs with the beams, not with every caller.

`NoiseModel` gets a quiet default. Today a quiet machine needs the hand-written `quiet_noise_model()` in `device/presets.py`, which builds a five-point zero spectrum and eight zero-rms drifts; Perceval's `NoiseModel()` is a perfect simulation because every field's default means "no noise", and Pulser derives the active noise types from which physical parameters were given. `NoiseModel()` should mean quiet, every spectrum and drift should be optional, and `NoiseModel.summary()` should list the channels that follow from what was set, with units, after Pulser's `get_noise_table()`. `NoiseModel.from_experiments(results)` closes True-Q's loop (a fit result is directly a noise source): a `heating_rate` measurement becomes the `S_E` level it implies.

Two additions from Pulser's `Device`: `Device.specs()` (a human-readable derived-parameter report: modes, Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering rates, each with its provenance id), and `Device.to_dict()/from_dict()` with a JSON schema file, so a published simulator configuration is citable and diffable outside Python. `Device.hash()` already exists and is the identity the schema carries.

`qutip_trap.physics` re-exports `Species`, `species()`, `available()`, `Level`, `Transition`, `Trap`, `Crystal`, `solve_crystal`, `Mode`, `Beam`, `Field`, `NoiseModel`, the spectra constructors, `Drift`, `Collisions`, `Detector`, `HardwareChain`, `Device`, `DerivedQuantities`, plus the closed forms the app imports from `validation/`, `prep/` and `trap/` (the Mathieu stability functions, the Lamb-Dicke and zero-point closed forms, the sideband transfer, the readout apparatus presets). Making them public is a decision to document them, not a new implementation.

### 4.7 The option objects

Today's 25 keyword arguments and 35 fields regroup by what they govern. Each object is a frozen dataclass with two-line field docstrings, validates in `__post_init__`, accepts a dict at the machine boundary (Qiskit keeps dict input as an alias), and is copied on construction so a caller mutating their own object cannot change a run in flight.

```python
@dataclass(frozen=True)
class Physics:
    """Which physical effects a run simulates. Every default is 'on' where the device provides the input."""
    noise: NoiseModel | None | bool = True          # True: the device's model; False: the quiet nominal sample; a model: override
    internal_levels: int = 2
    scattering: Literal["estimate", "channels"] = "estimate"       # was SolverOptions.scattering_channels
    scattering_recoil: RecoilOption = "minimal"
    intensity_noise_channels: bool = True
    hardware_chain: bool = True
    stark_compensation: bool = True
    crosstalk_suppression: CrosstalkSuppression = "none"
    entangler: Literal["ms", "zz"] = "ms"
    extra_channels: tuple[CollapseOp, ...] = ()
    builder: BuilderOptions = BuilderOptions()
    t0_s: float = 0.0
    shot_period_s: float | None = None

@dataclass(frozen=True)
class Numerics:
    """How the integration is done. Nested so that a physicist reads the truncation policy without the tomography knobs."""
    integration: Integration = Integration()        # atol, rtol, nsteps, integrators, rotating_frame
    truncation: Truncation = Truncation()           # joint_dimension_max, nnz_max, mode_dimension_max, boundary_population_max,
                                                    #   freeze_*, branch_weight_min, margin_*, caps, enr_group, space
    trajectories: Trajectories = Trajectories()     # lindblad_method, mesolve_dimension_max, ntraj, improved_sampling, target_tol
    gate_local: GateLocal = GateLocal()             # map_accuracy, crosstalk_threshold, register_*, tomography_*
    parallel: Parallel = Parallel()                 # map, workers, propagator_cache, samples
    convergence_check: bool = False

@dataclass(frozen=True)
class Readout:
    """How the photon record is read. A callable replaces the whole stage (the DynamicsBackend precedent)."""
    mode: Literal["fast", "full"] | Callable[..., ReadoutOutcome] = "fast"
    discriminator: Discriminator | None = None
    povm_samples: int = 20_000
```

The split follows the cut dynamiqs made in 0.3.6: conveniences that change per call (`seed`, `keep_final_state`, `progress`) are keyword arguments; anything that describes how the physics or the integration is done lives on an object. The truncation report becomes loud as well as complete: a run whose boundary population exceeds `boundary_population_max` after the cap-raising retries, or whose cap was clamped by `mode_dimension_max`, emits a `warnings.warn` naming the mode and the numbers in addition to the `Diagnostics` entry, so that silence means converged (Strawberry Fields leaves the check to the user's memory of `state.trace()`; scqubits has no checker at all). The `auto` level guard already carries a knob for its own accuracy (`map_accuracy`, the tomography tolerance keyed to it) in the BQSKit sense; `Diagnostics.level_reason` states the numbers it compared.

Where each of today's `run` keyword arguments goes:

| today | proposed home |
|---|---|
| `table`, `calibrate_kwargs` | `Machine.table`, `Machine.calibrated(...)` |
| `gate_drives`, `entangling_drives` | `Device.roles` |
| `level` | `Machine.level` (a `Level` enum; `Diagnostics.level_reason` states the dimension and the guard) |
| `noise`, `internal_levels`, `crosstalk_suppression`, `stark_compensation`, `entangler`, `channels`, `builder_options`, `t0_s`, `shot_period_s` | `Physics` |
| `options`, `space`, `caps`, `enr_group`, `samples`, `parallel` | `Numerics` |
| `readout`, `discriminator`, `povm_samples` | `Readout` |
| `seed`, `keep_final_state` | stay on the call |

`SolverOptions` remains importable from `qutip_trap.api` (the freeze test) as an alias that builds a `Numerics` plus the three physics fields it carried, with a deprecation warning naming the new home.

`Level` is an enum, not a string: CUDA-Q replaced a `store_intermediate_results: bool` with an enum while still accepting the bool, and PennyLane's `add_noise(..., level=)` shows that "level" already means "pipeline insertion point" to a PennyLane user, so the enum's docstring should say what qutip-trap means by it.

### 4.8 The laboratory

The experiments, the calibration and the benchmarks keep their packages and their names and take a `Machine` where they take a `Device` today, which removes their `**run_kwargs` and `gate_drive=` arguments. Four changes:

- **One result type per experiment**, subclassing `ExperimentResult`: `RabiScan`, `RamseyFringe`, `SidebandSpectrum`, `HeatingRateFit`, `ParityScan`, `DetectionHistogram`, each with the fitted parameters as typed attributes, `.plot()`, and the realized scan parameters beside the requested ones. Cirq ships `T1DecayResult`, `SingleQubitReadoutCalibrationResult` and `TwoQubitXEBResult`; qiskit-experiments records the achieved delay, not the requested one, as the fit's x value. `value(key)`, `uncertainty(key)` and `converged` stay.
- **`calibrate(machine, *, method="closed_form" | "experiments", ...)`** replaces `surrogate=True/False`, whose `**kwargs` today route to two different functions depending on the boolean; it returns the `CalibrationReport` (which carries the table) on every call, retiring the `calibrate`/`calibrate_with_report` and `compile_to_native`/`compile_with_report` doubling. The report keeps every `ExperimentResult` that set an entry, and `CalEntry` already carries `fitted_at_s`, `sample_id`, `experiment` and `provenance_id`, which is the write-back provenance qiskit-experiments lost when it deleted `ParameterValue(value, date_time, valid, exp_id, group)`. Two Qibocal properties are worth adding: `fit` reruns against stored scan data without re-acquiring, and an update is a proposal, `table.updated_with(result)`, that a caller can gate on the fit's quality before adopting, rather than a write that happens inside the scan. The table's entries should also be tagged by kind, setpoint (what the scheduler programs: `rabi`, `stark`, `qubit_freq`, the `ms` waveforms, the micromotion shims, the detection threshold and window) against characterisation (what was measured about the device: `heating`, `nbar`, `crosstalk`), the partition Qibolab drew between `parameters.json` and `calibration.json`; the lookup precedence for an entry (per pair, per ion, per beam) should be stated the way OpenQASM 3 states `defcal` resolution: most-specific match over (gate, parameters, ions).
- **The inverse direction.** `machine.error_model() -> ErrorModel`: the phenomenological summary of the simulated device, computed from `gate_channel` per native gate kind, the intrinsic budget and the SPAM errors, with exporters to the vendors' vocabularies and the conversions stated in the docstrings: `to_ionq_noise()` (`r_1q`, `r_2q` with `F_avg = 1 - r/2` and `1 - 3r/4`), `to_quantinuum_error_params()` (`p1`, `p2`, `p_meas`, `p_init`, the two dephasing rates), `to_qdk_qubit_params()` (gate times and error rates as the estimator's unit-suffixed strings). Naming follows the one convention that survived three vendor API generations: `p_*` a probability, `*_rate` per second, `*_ratio` a fraction of another probability, `*_scale` a multiplier. Braket's emulator does the forward half of this (`p_1q = (1 - F) * 3/2`); nobody does the half qutip-trap can do.
- **Benchmarks** keep their protocols and their budgets; `randomized_benchmarking(machine, qubits, lengths, ...)` loses `**run_kwargs`.

### 4.9 Execution: synchronous, observable, and a handle for the app

`machine.run(...)` stays synchronous and gains `progress: Callable[[Progress], None]`, fired per pulse, per branch and per sample, after QuTiP's `progress_bar`/`progress_kwargs` option and qsharp's per-shot `on_result`. The app's `workers.py` streams exactly this through `page.pubsub` today.

`machine.submit(circuit, shots, seed=0) -> Job` returns a handle with `status()`, `result()`, `cancel()` and `progress`, implemented in-process by a worker, after Braket's `LocalQuantumTask` and pytket's `ResultHandle`, which give local runs the remote surface so one code path serves both. The unit of submission is a frozen, JSON-serialisable `RunSpec` (circuit, shots, seed, machine hash, the three option objects, level, label), which is the app's `JobSpec` moved to the core; `RunRecord` and `last_record` already exist and become `Job.record()`.

### 4.10 `Result`

- `Result.to_dict()/from_dict()` with a versioned envelope: `schema_version`, `qutip_trap_version`, `device_hash`, `machine_hash`, `shots`, `root_seed`, `created_at`, `duration_s`, then counts, probabilities, error bars, SPAM, heralds and the diagnostics summary; a generated JSON schema file, after Pulser's `results-schema.json` and Braket's `braketSchemaHeader`. Per-shot arrays are opt-in in the serialised form.
- The bit-order sentence on the accessors that need it ("qubit 0 is the least-significant bit, the rightmost character; column j of `bitstrings` is qubit j"), a `Result.reversed_bits()` converter for the SDKs that put qubit 0 leftmost, and a docs test that the sentence appears once and is never contradicted.
- IonQ exporters named by the convention they emit: `to_ionq_v1_probabilities()`, `to_ionq_v1_histogram()`, `to_ionq_v1_shots()` (today's three, correct), `to_ionq_v2_probabilities()`, `to_ionq_v2_histogram()`, `to_ionq_v2_shots()` with the `probabilities/registers` envelope, `output_all` as the whole-circuit register and the circuit's `registers` beside it, and the v2 character order settled against IonQ's documentation before release (Section 3.3 item 5). `Result.from_ionq_v1_shots()` closes the round trip. `dump_ionq_job(circuit, *, backend, shots, noise, settings, name, metadata)` emits the whole v0.4 body (`type`, `backend`, `input`, `noise`, `settings.error_mitigation.debiasing`, `dry_run`); v0.4 sets `additionalProperties: false`, so a v0.3 `target` is rejected by the service, and the importer should keep accepting both.
- Seeds: `Diagnostics.root_seed` already answers "how do I rerun this". The docs should state exactly what is invariant: the same `(shots, samples, seed, machine hash)` reproduces every shot; the same `seed` with a different `shots` does not, because shots are allocated to dynamical samples in contiguous blocks whose size depends on `shots` (`conv.shot_blocks_per_sample`). Stim's disclaimer about version boundaries applies and should be copied: integrator and cap-rule changes may move sampled outcomes between minor versions.

### 4.11 Namespace, conventions, policy, docs

**Namespace.** `qutip_trap/__init__.py` exports the rung-0 dozen (`Machine`, `Circuit`, `Result`, `Level`, `Physics`, `Numerics`, `Readout`, `presets`, `circuit`, `schedule`, `dynamics`, `physics`, `__version__`) through a lazy module `__getattr__`, so `import qutip_trap` stays at 0.05 s for the app and `import qutip_trap.api` keeps its 0.49 s. The rung modules are the documented import locations; `qutip_trap.api` is documented as the Appendix E compatibility surface. `qutip_trap.io` gets stdlib-shaped `qasm2.loads/dumps` and `ionq.loads/dumps/load_job/dump_job`, mirroring `qiskit.qasm2` and `json`. `qutip_trap.interop` holds the adapters to other SDKs, each importing its SDK only inside its own module and installed by an extra (`qiskit` today). A `qutip_trap.experimental` module gates stability by import path, as Mitiq's does ("not covered by semantic versioning guarantees"): the M12 transport records whose methods raise `NotImplementedError` today, the tomography internals and the closed-form oracles belong there until they have a second consumer. Extension points stay explicit protocols (`PulseEngine` already is one); nothing is recovered by reflection on signatures, the habit both Mitiq and Qibocal adopted and both annotate with discomfort.

**Conventions**, written into `docs/conventions.md`'s first section rather than a new page: units at the point of use (every public float suffixed `_hz`, `_s`, `_m`, `_w`, `_gauss`, `_rad`; the 2 pi lives in `rad_s_from_hz` and nowhere else; turns exist only in `io.ionq`), the bit order, the tensor order of the native gate matrices (documented in `control/native.py` with a test that `ms(...) == SWAP @ qiskit_ionq_form @ SWAP`), the seed invariants, and the `p_*`/`_rate`/`_ratio`/`_scale` vocabulary for the error model.

**Policy.** Adopt Qiskit's boundary ("public means documented"), PennyLane's shape (`QutipTrapDeprecationWarning`, every entry "deprecated in / removed in", a `docs/deprecations.md`), and Cirq's mechanism (`qutip_trap._compat.deprecated(deadline="v0.4", fix="Use Machine.run.")` with the deadline validated, plus `deprecated_parameter(match, rewrite)` so the 25 `run` keyword arguments are rewritten into the option objects for two minor releases before removal). At 0.x, read 0.y as the minor: deprecate in one, remove no earlier than two later, always with a warning-free path. A `spec`-style test (Cirq's `json_test_data/spec.py`) enumerates every public name of every rung module and asserts it is documented and, where it is a frozen dataclass, that it round-trips.

**Docs.** Reorder `docs/` as the ladder (Cirq's navigation is its pipeline): a two-minute quickstart at the top of `examples.md` (Qiskit's "build a circuit in under two minutes"), then one section per rung and one for the laboratory, with `conventions.md`, `limits.md` and `physics_notes.md` unchanged. The executed-examples test already keeps it honest.

### 4.12 Worked examples at every rung

Everything below is the proposed surface; none of it runs today except the last block. Field names of existing types (`Schedule`, `PlayedGate`, `Pulse`, `Drive`, `Traces`, `BuiltHamiltonian`, `CompileReport`, `RBResult`, `CalibrationReport`) are the ones in the tree; the numbers in comments are the ones `docs/examples.md` and the M6 tests report for the two-ion 171Yb+ device.

**Rung 0, the machine.** The cloud customer's view: a circuit in, a histogram out, the approximation status beside it.

```python
import dataclasses
import qutip_trap as trap

machine = trap.presets.yb171_chain(2)          # a Machine: the device, its beam roles, the closed-form calibration, the defaults
bell = trap.Circuit(2).h(0).cnot(0, 1)         # measurement of every qubit is the default
result = machine.run(bell, shots=2000)         # compile, calibrate (cached by device hash), schedule, prepare, evolve, read out

result.counts                                  # {'00': 1003, '11': 993, '01': 3, '10': 1}
result.probabilities["11"], result.error_bars["11"]
result.spam                                    # {'0': (6.1e-4, 3.5e-4), '1': (...)}: per qubit (eps_B, eps_D)
result.diagnostics.level                       # Level.JOINT_EXACT
result.diagnostics.level_reason                # 'joint dimension 572 <= 4096 and 2.9e5 drive non-zeros <= 2e7'
result.diagnostics.space.dims                  # [2, 2, 11, 13]: two ions, the two x modes resolved, four modes dropped
result.diagnostics.intrinsic_budget["total"]   # the closed-form error budget the register infidelity must sit inside
trap.circuit.ideal_probabilities(bell)         # the target, shown beside the simulated histogram, never in its place

deeper = dataclasses.replace(machine, level=trap.Level.GATE_LOCAL)        # the app's "verify deeper"
quiet = dataclasses.replace(machine, physics=trap.Physics(noise=False))   # the nominal sample, no channels
est = machine.estimate(trap.Circuit(3).h(0).cnot(0, 1).cnot(1, 2))        # before running: level, space, cost
est.level, est.space.dims, est.wall_time_s                                # Level.JOINT_EXACT, [2, 2, 2, 12, 12], about 180

def show(p: trap.Progress) -> None:            # fired per pulse, branch and sample; what the app streams today
    print(f"{p.stage:9s} {p.done}/{p.total}")

result = machine.run(bell, shots=2000, seed=7, keep_final_state=True, progress=show)
job = machine.submit(bell, shots=20_000, seed=7)   # a Job: status(), result(), cancel(), progress; the app's worker, in the core
job.spec.to_dict()                             # RunSpec: circuit, shots, seed, machine hash, the option objects, level
```

Interop at the same rung:

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = trap.Circuit.from_openqasm(qasm)     # registers survive: circuit.registers == {'c': (0, 1)}
circuit = trap.Circuit.from_ionq(job_body)     # a v0.3 or v0.4 job body or a bare input; native or qis gate set; turns -> radians

result = machine.run(circuit, shots=2000)
result.to_ionq_v1_probabilities()              # {'0': 0.5015, '3': 0.4965, '1': 0.0013, '2': 0.0008}: qubit 0 is the 2^0 bit
result.to_ionq_v2_probabilities()              # {'probabilities': {'registers': {'output_all': {...}, 'c': {...}}}}
result.reversed_bits().counts                  # for Cirq, Braket and PennyLane comparisons, which put qubit 0 leftmost
result.to_dict()["schema_version"], result.to_dict()["device_hash"]

trap.io.ionq.dump_job(circuit, backend="simulator", shots=2000, noise={"model": "forte-1", "seed": 7})
# {'type': 'ionq.circuit.v1', 'backend': 'simulator', 'shots': 2000, 'noise': {...},
#  'input': {'gateset': 'native', 'qubits': 2, 'circuit': [{'gate': 'gpi2', 'target': 0, 'phase': 0.0}, ...]}}
trap.io.qasm2.dumps(machine.compile(circuit).circuit)   # the native circuit as OpenQASM 2 with the gpi/gpi2/ms declarations
```

**Rung 1, the circuit.** The compiler engineer's view: what the native gates are, that they are verified, and what each one does to the register.

```python
import math
from qutip_trap import circuit as circ

report = machine.compile(bell)                 # CompileReport: the native circuit and its verification
[op.name for op in report.circuit.ops]         # ['gpi2', 'gpi2', 'ms', 'gpi2', 'gpi2', 'gpi']: rz absorbed into later phases
report.n_pulses, report.n_entangling           # 5, 1
report.circuit_residual                        # 1.2e-15: max |U_compiled - e^{i a} U_target| over the whole circuit
report.final_frame_rad                         # the virtual-Z frame per ion that the measurement discards

native = trap.Circuit(2).gpi2(0, phase=0.0).ms(0, 1, phases=(0.0, 0.0), angle=math.pi / 4)   # radians in; turns only at the wire
native.is_native                               # True: the scheduler plays it without the compiler
circ.circuit_unitary(native)                   # the 4 x 4 target in IonQ's tensor order (ion 0 the first factor)

ch = circ.gate_channel(machine, "ms[0,1]")     # one native gate kind, played once from the prepared motional state
ch.average_gate_infidelity(), ch.infidelity_on((0, 1)), ch.depolarizing_rate()
ch.steps[0].summary.pauli_twirled["II"]        # the twirl the app's channel replay composes
```

**Rung 2, the schedule.** The control engineer's view: which light hit which ion, when, with which tones, and what the calibrated waveform is.

```python
from qutip_trap import schedule as sched

s = machine.schedule(bell)                     # compile + calibrate + schedule; nothing integrated (IonQ's dry_run)
s.duration_s(), len(s.pulses), s.measurement().t_start_s
for g in s.gates:                              # PlayedGate: every entangling gate as played
    print(g.kind, g.pair, g.beams, f"{(g.t_end_s - g.t_start_s) * 1e6:.0f} us", f"{g.waveform.chi_total_rad():+.4f} rad")
ms = next(p for p in s.pulses if p.gate_id and p.gate_id.startswith("ms"))
ms.drive.kind, ms.drive.ions, ms.closes_modes  # 'raman', (0, 1), (0, 1)
[t.detuning_hz for t in ms.drive.tones]        # the red and blue legs, -(omega_m + delta) and +(omega_m + delta) from the carrier
ms.drive.crosstalk                             # the neighbours' share of the light, from the beam waists

table = machine.calibrated().table             # pin the surrogate table explicitly (about 6 s); it travels with the record
wf = table.waveform_for((0, 1))
wf.duration_s, wf.chi_m, wf.alpha_m            # 1e-4, {0: 0.785, 1: -0.003}, the residual displacement per mode at closure
table.rabi[(0, 2)].value, table.rabi[(0, 2)].status   # 147e3, 'seed'

# a miscalibration, played as written: the scheduler believes a Rabi frequency 2 % high (the app's Level 2 request)
entry = table.rabi[(0, 2)]
wrong = table.with_params(rabi={(0, 2): dataclasses.replace(entry, value=1.02 * entry.value, status="calibrated")})
dataclasses.replace(machine, table=wrong).run(bell, shots=2000).probabilities

# a five-segment closure over both x modes instead of the symmetric square pulse
modes = sched.gate_modes(machine.device, ions=(0, 1), beams=(0, 1))
shaped = sched.solve_amplitude_modulation(modes, mu_hz=3.02e6, duration_s=120e-6, n_segments=5)   # a ShapedPulse
shaped_machine = dataclasses.replace(machine, table=table.with_params(ms={(0, 1): shaped.waveform}))
```

**Rung 3, the dynamics.** The physicist's view inside one pulse: the space, the prepared state, the traces, the Hamiltonian terms.

```python
from qutip_trap import dynamics as dyn

s = machine.schedule(bell)
space = dyn.HilbertSpace.for_(machine.device, s, machine.numerics)     # [2, 2, 11, 13]
state = dyn.prepare(machine.device, space, machine.table, dyn.quiet_sample(), dyn.SeedSpec(0))   # Doppler -> sideband -> pump
state.motional.nbar, state.provenance          # {0: 0.017, 1: 0.019, ...}, ('doppler', 'sideband', 'pump')

engine = machine.engine                        # the JointExactEngine that run() uses
traces = engine.run_pulses(machine.device, s, state, space, dyn.quiet_sample(), dyn.SeedSpec(0), machine.numerics)
traces.times_s[-1], traces.expectations["P1[0]"][-1]
{m: n[-1] for m, n in traces.mode_occupations.items()}      # <n_m>(t) at the end of the schedule
{m: abs(a[-1]) for m, a in traces.alpha_m.items()}          # the spin-dependent loops: closed when near zero
traces.boundary_population, traces.jumps                     # the truncation monitor per mode; (time, channel) of every jump
traces.mode_marginal[0][-1]                                  # new: the Fock distribution of mode 0 at the last stored time

H = dyn.build_hamiltonian(machine.device, [ms], space)       # the terms the solver integrates on that segment
H.n_drive_terms, H.frame, H.approximations, H.kernel
for r in H.records:                                          # one DriveRecord per (pulse, ion): etas, Debye-Waller, crosstalk
    print(r.ion, r.etas, f"{r.debye_waller:.4f}", f"{abs(r.crosstalk):.3f}")

def my_readout(register, machine, seeds) -> trap.ReadoutOutcome: ...   # the whole fluorescence-and-discriminator stage
dataclasses.replace(machine, readout=trap.Readout(mode=my_readout)).run(bell, shots=200)
```

**Rung 4, the physics.** The device physicist's view: a device from species, trap, beams, field, noise and detector, with the roles on it.

```python
from qutip_trap import physics as phys

yb = phys.species("171Yb+")                    # every hyperfine-resolved number derived, every constant cited
trap_ = phys.Trap(omega_hz=(3.0e6, 2.9e6, 1.0e6), axis_angle_rad=0.0, rf=None, dc=None, geometry=None,
                  stray_field_v_per_m=(0.0, 0.0, 0.0), shim_voltages_v={})
crystal = phys.solve_crystal(trap_, (yb, yb))  # equilibrium positions, the six modes, eta per ion and mode
[round(m.omega_hz / 1e6, 4) for m in crystal.modes]

global_pair = phys.raman_pair_along_x(power_w=0.3, waist_m=60e-6, pointing_m=(0.0, 0.0, 0.0))
address = [phys.raman_pair_along_x(power_w=0.3 * (2.5 / 60) ** 2, waist_m=2.5e-6, pointing_m=tuple(crystal.positions_m[i]))
           for i in range(2)]                  # each helper is two eight-field Beams: wavelength, k_hat, polarization, waist, power, pointing, ...
detection = phys.oblique_detection_beam(s_o=2.45)

device = phys.Device(
    crystal=crystal, trap=trap_, field=phys.Field(5.0, (1.0, 0.0, 0.0), None),
    beams=(*global_pair, *address[0], *address[1], detection),
    noise=phys.NoiseModel(),                   # quiet by default; set S_E, a Drift, Collisions to turn channels on
    detector=phys.crain_snspd_detector(window_s=22e-6),
    hardware=phys.ideal_hardware(),
    roles=phys.BeamRoles(gate={0: phys.GateDrive("raman", (2, 3)), 1: phys.GateDrive("raman", (4, 5))},
                         entangling={0: phys.GateDrive("raman", (0, 1)), 1: phys.GateDrive("raman", (0, 1))},
                         detection=6),
)
d = device.derived()
d.values["rabi_hz[(0, 2)]"], d.provenance["rabi_hz[(0, 2)]"]   # 147e3 and the ledger record that defines it
print(device.specs())                          # modes, eta, Rabi frequencies, Stark shifts, scattering rates, each with its record
phys.Device.from_dict(device.to_dict()) == device               # the versioned JSON round trip; device.hash() is the identity

noisy = dataclasses.replace(device, noise=phys.NoiseModel(
    S_E=phys.power_law_spectrum(level_at_ref=1e-12, omega_ref_rad_s=2 * math.pi * 1e6, alpha=1.0,
                                unit="(V/m)^2/(rad/s)", omega_min_rad_s=2 * math.pi * 1e3, omega_max_rad_s=2 * math.pi * 1e7),
    collisions=phys.Collisions(pressure_pa=1e-9, gas={"H2": 1.0},
                               outcome_probabilities={"heating_kick": 0.9, "reorder": 0.05, "loss": 0.04, "dark_ion": 0.01}),
))
noisy.noise.summary()                          # the channels that follow from what was set, with units (Pulser's noise table)
lab_machine = trap.Machine(noisy, name="lab 2026-09")
```

**The laboratory.** Experiments, calibration and benchmarks on the machine, and the inverse direction.

```python
import numpy as np
from qutip_trap import benchmarks as bench, calibration as cal, experiments as lab

scan = lab.rabi_scan(machine, ion=0, durations_s=np.linspace(0.0, 30e-6, 13), shots=200)   # a RabiScan
scan.f_rabi_hz, scan.uncertainty("f_rabi_hz"), scan.converged      # (146.9e3, 0.4e3, True)
scan.requested.durations_s[3], scan.realized.durations_s[3]        # the pulse the electronics could play (dead time, DDS words)
scan.plot()

spectrum = lab.sideband_spectroscopy(machine, ion=0, detunings_hz=np.linspace(-3.2e6, 3.2e6, 161))
heat = lab.heating_rate(machine, mode=4, delays_s=(0.0, 1e-3, 2e-3, 5e-3))
hist = lab.detection_histogram(machine, ion=0, n_records=2000)
hist.threshold, hist.window_s, hist.errors                          # (n_c, t_b) and (eps_B, eps_D) at the optimum

report = cal.calibrate(machine, method="experiments", experiments=("modes", "rabi", "stark", "ms", "detection"))   # about 12 min
report.table.rabi[(0, 2)]                      # CalEntry(value, uncertainty, status='calibrated', experiment='rabi_scan', fitted_at_s, sample_id)
report.surrogate_error()                       # how far the closed forms were from the fitted values
report.refused                                 # {'crosstalk': 'upstream entry rabi[(1, 4)] uncalibrated'}
proposal = report.table.updated_with(scan)     # an update is a proposal ...
if scan.converged and scan.chi2 < 2:           # ... adopted only past an acceptance test (Qibocal's rule)
    machine = dataclasses.replace(machine, table=proposal)

rb = bench.randomized_benchmarking(machine, qubits=(0,), lengths=(1, 128, 512), n_sequences=4, shots=2000)
rb.fidelity_form(), rb.error_per_clifford      # 'A p^m + B', (2.1e-5, 0.3e-5)
rb.budget.channel_infidelity, rb.budget.predicted["r_channel"]     # per native gate kind; the composed prediction
ghz = bench.ghz_fidelity(machine, qubits=(0, 1), shots=400)
ghz.fidelity_bound, ghz.register_fidelity_max_phase               # (0.9975, 0.0112), 0.99808
qv = bench.quantum_volume(machine, qubits=(0, 1), n_circuits=4, shots=200)
qv.heavy_output_probability.mean(), qv.threshold_cleared, qv.passed

em = machine.error_model()                     # physics in, phenomenology out
em.p_1q, em.p_2q, em.p_meas, em.durations_s    # from gate_channel per kind, the SPAM errors, the schedule
em.to_ionq_noise()                             # {'r_1q': ..., 'r_2q': ...}: F_avg = 1 - r/2 and 1 - 3r/4, stated in the docstring
em.to_quantinuum_error_params()                # {'p1': ..., 'p2': ..., 'p_meas': (p01, p10), 'p_init': ..., 'linear_dephasing_rate': ...}
em.to_qdk_qubit_params()                       # {'oneQubitGateTime': '5 µs', 'twoQubitGateTime': '100 µs', 'twoQubitGateErrorRate': ...}
```

**The option objects.** Numerics nested by concern, physics switches separate, readout separate; dot access, dicts accepted, copied on construction.

```python
from qutip_trap import dynamics as dyn
from qutip_trap.readout import TimeResolvedML

careful = trap.Numerics(
    integration=dyn.Integration(atol=1e-11, rtol=1e-9, rotating_frame=True),
    truncation=dyn.Truncation(boundary_population_max=1e-8, branch_weight_min=1e-6, mode_dimension_max=96),
    trajectories=dyn.Trajectories(lindblad_method="mcsolve", ntraj=256),
    gate_local=dyn.GateLocal(map_accuracy=1e-4, crosstalk_threshold=1e-3),
    parallel=dyn.Parallel(workers=8),
    convergence_check=True,                    # repeat at ten times tighter tolerances and report the change
)
physics = trap.Physics(noise=True, internal_levels=3, scattering="channels", crosstalk_suppression="echo", entangler="zz")
readout = trap.Readout(mode="full", discriminator=TimeResolvedML(n_bins=8))
m = dataclasses.replace(machine, numerics=careful, physics=physics, readout=readout)
m.numerics.truncation.mode_dimension_max       # 96; m.numerics.asdict() for the record
```

**Through Qiskit.** This block runs today: `qutip_trap/interop/qiskit.py` (added with this proposal, 2026-09-11) is a Qiskit 2 `BackendV2` over `run`, under 200 lines including docstrings. A circuit crosses as OpenQASM 2 text (`qiskit.qasm2.dumps` into `load_openqasm2`), the `Target` advertises exactly the gates the importer and the compiler accept so `transpile` rewrites anything else into them, run options beyond `shots`, `seed`, `level` and `table` pass to `run` unchanged (Aer's convention), and the job comes back finished with the qutip-trap `Result` of every circuit behind it. No bit-order conversion is needed: Qiskit and qutip-trap both put qubit 0 rightmost. Install with `uv sync --extra qiskit` (it is also in the dev group so `tests/test_interop_qiskit.py` runs in CI).

```python
from qiskit import QuantumCircuit, transpile
from qutip_trap.api import SolverOptions
from qutip_trap.interop.qiskit import QutipTrapProvider

backend = QutipTrapProvider().get_backend("yb171_chain", n_ions=2)   # a DevicePreset as a BackendV2
sorted(backend.operation_names)
# ['cp', 'cx', 'cz', 'h', 'id', 'measure', 'rx', 'rxx', 'ry', 'rz', 'rzz', 's', 'sdg', 'swap', 'sx', 't', 'tdg', 'u3', 'x', 'y', 'z']

bell = QuantumCircuit(2, name="bell"); bell.h(0); bell.cx(0, 1); bell.measure_all()
job = backend.run(transpile(bell, backend), shots=2000, seed=7, options=SolverOptions(branch_weight_min=1e-3))
job.status().name, job.result().get_counts()          # 'DONE', {'00': 982, '11': 1015, '01': 3}   (5.3 s on the reference machine)

res = job.results[0]                                   # the qutip-trap Result behind the Qiskit one
res.diagnostics.level, res.diagnostics.space.dims      # 'JOINT_EXACT', [2, 2, 11, 13]
res.diagnostics.intrinsic_budget["total"], res.spam    # 8.9e-3, {'q0': (5.4e-4, 4.2e-4), 'q1': (...)}
res.to_ionq_json()                                     # {'0': 0.491, '3': 0.5075, '1': 0.0015}

QutipTrapBackend(my_preset, table=table, readout="full")   # your own preset; defaults for every run
```

Under the proposal the constructor takes a `Machine` instead of a `DevicePreset` and the option objects replace the pass-through keyword arguments; nothing else in the adapter changes. A Cirq `Sampler` and a PennyLane device would be the same fifty lines each around the same OpenQASM 2 door, which is why the proposal keeps them out of scope until a consumer asks.

**Compatibility.** The frozen Appendix E surface keeps working unchanged; the bridge to the new shape is one method.

```python
from qutip_trap.api import Circuit, Operation, calibrate, run, yb171_chain   # unchanged; today's code runs as before

preset = yb171_chain(2)                        # still a DevicePreset here
table = calibrate(preset.device, pairs=[(0, 1)], **preset.run_kwargs())           # QutipTrapDeprecationWarning names Machine
bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
result = run(bell, preset.device, 2000, table=table, **preset.run_kwargs())        # the keyword arguments are rewritten into the option objects
machine = preset.machine()                     # the bridge: the same device, roles and table as a Machine
```

## 5. Migration: three additive releases

Nothing in Appendix E is renamed or removed. Each phase adds, deprecates with a rewrite, and keeps the freeze test green.

**0.2.0 (additive).** `BeamRoles` on `Device` with inference for the `None` fields; `Machine` with `run`, `compile`, `schedule`, `calibrated`, `engine`, `hash`, `specs`; presets return machines, `DevicePreset.run_kwargs()` warns; `Circuit` builder, defaults, `registers`, `from_openqasm`, `from_ionq`, `to_openqasm`, `to_ionq`; `Level` enum and `Diagnostics.level_reason`; `Result.to_dict/from_dict` and the schema; the IonQ exporters renamed with the v2 fix and `dump_ionq_job`; `progress=` on `run`; the rung modules and the lazy root; `docs/deprecations.md` and the `_compat` helper. Amend Appendix E with the additions.

**0.3.0.** `Physics`, `Numerics`, `Readout` with `deprecated_parameter` rewrites of the `run` keyword arguments and of `SolverOptions`; the laboratory takes machines; per-experiment result types; `calibrate(method=)`; `CalibrationTable.with_params`; `Device.specs`, `to_dict`, `from_dict`; `machine.error_model()` with the three exporters; the app deletes its `run_kwargs` and its overrides code.

**0.4.0.** `RunSpec`, `Job`, `machine.submit`; `Traces.mode_marginal` and per-pulse wall time; the GATE_LOCAL per-step register; the closed forms and presets under `qutip_trap.physics`; docs reordered as the ladder; the spec test; removal of the deprecated keyword arguments after their two-release window. The app's `CORE_GAPS` list should be empty at this point.

## 6. Decisions left to the author

1. `Machine` against `Emulator` or `Backend`, and `import qutip_trap as trap` against no recommended alias.
2. Whether the machine pins its table (immutable, explicit, exportable, as proposed) or caches it lazily on the instance (convenient, but the only mutable state in the package).
3. Whether experiments stay named functions with typed results (proposed; the physicist's vocabulary, Cirq's shape) or become parameter sweeps over one coercible input (Qiskit's PUBs), which would unify `rabi_scan`, `ms_scan` and `parity_scan` at the cost of that vocabulary.
4. Whether `qutip_trap.api` is retired at 1.0 or kept forever as the Appendix E alias. The survey found no precedent for a flat re-export module, and every precedent for a small root plus documented submodules.
5. The v2 IonQ character order, to be settled against IonQ's documentation (not stated in the v0.4 spec text) before the exporter is renamed.
6. Whether the adapters live in the package or in separate distributions. The Qiskit `BackendV2` now exists as `qutip_trap.interop.qiskit` behind an optional extra (Section 4.12); a Cirq `Sampler` and a PennyLane device would be the same shape around the same OpenQASM 2 door, and three real consumers would justify splitting them out the way `cirq-ionq` and `pennylane-ionq` are split.
7. Whether `Diagnostics` stays fully typed (32 fields, every one a promise) or splits into a stable typed core (`level`, `approximations`, `boundary_population`, `intrinsic_budget`, `root_seed`, `calibration`) plus a `metadata` mapping carried "for interpreting this run" without a stability guarantee, Boulder Opal's shape. PLAN.md Section 14.1 rule 4 wants every result to carry its convergence report, which argues for the typed core being generous.

## Appendix: sources

Primary sources read by the research passes, by cluster. Qiskit: `DEPRECATION.md`, the bit-ordering guide, the primitives and `Target` source at 2.5.2, Aer's `noise/device` module, the Dynamics 0.6.0 release notes and archived repository, the qiskit-experiments PR #1511. Cirq: the 1.7.0 tag (`examples/hello_qubit.py`, `cirq-core/cirq/experiments`, `cirq_google/engine/virtual_engine_factory.py`), `docs/dev/style.md`, `docs/dev/nomenclature.md`, the versions page. CUDA-Q: the 0.15.0 tag (`python/cudaq/__init__.py`, `runtime/sample.py`, `py_NoiseModel.cpp`), the backends and simulators pages. PennyLane: the 0.45.1 tag, the deprecations page, `qml.pulse.rydberg_drive`. QuTiP and qutip-qip: the 5.3.1 and 0.4.2 source trees, qutip-qip issue #217. Pulser: the 1.9.1 tag (`devices`, `noise_model.py`, `backend`), the conventions page, the JSON schemas. Bloqade: `bloqade-analog` and `bloqade-circuit` at their tags, the manifesto and the 2025 blog posts, issues #53 and #410. Braket: the 1.127.0 SDK, schemas 1.32.1 and default-simulator 1.40.1 source. IonQ: the v0.4 and v0.3 OpenAPI documents, the direct-API-submission guide, `qiskit-ionq` 1.1.1, `cirq-ionq` 1.7.0, `pennylane-ionq` 0.45.0, `ionq-core` 0.1.1, the live `/backends` and `/characterizations` responses on 2026-09-11. Quantinuum: pytket 2.18.1, pytket-quantinuum 0.59.2 (changelog), `quantinuum-schemas` 7.8.2 (`UserErrorParams`, `HeliosErrorParams`), guppylang 1.0.4, selene-sim 0.3.2. AQT: qiskit-aqt-provider 1.15.0 and the Arnica OpenAPI document. Microsoft: qdk 1.32.3 (`physical_qubit.rs`, `NoiseConfig`), azure-quantum 3.12.0, the noise-model docs page. Physics toolkits: C3 `dev` (`c3objs.py`, `generator/devices.py`), scqubits 4.3.1 docs and source, qibolab 0.2.16 (`platform.py`, `parameters`), qibocal 0.2.6 (`protocol.py`, `update.py`, `calibration_scripts/rx_calibration.py`), Strawberry Fields 0.23.0 (`backends/states.py`), Perceval 1.2.4 (`noise_model.py`, `source.py`), True-Q 2.14.5 docs snapshot, Mitiq 1.1.0, Stim 1.16.0 stubs, BQSKit 1.2.1, dynamiqs 0.3.4 and the 0.3.6 changelog, boulder-opal 6.0.0 and fire-opal 12.2.0 wheels, the OpenQASM `pulses.rst`. Items the passes could not verify are marked in the text.
