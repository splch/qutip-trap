# Design

qutip-trap turns a quantum circuit into measurement counts the way a trapped-ion processor does: by simulating the physics that produces them. No gate matrix is ever applied to the simulated state; ideal unitaries appear only in the compiler's definition of the native gates and in the tests that compare simulated pulses with them. Every `Result` states which fidelity level ran, which Hilbert space was integrated and every approximation that was made.

The equations and conventions are in [docs/physics.md](docs/physics.md); runnable examples are in [docs/examples.md](docs/examples.md).

## What "exact" means

- **Model.** Each ion is a few electronic levels (the qubit pair, plus leakage levels when asked for), each motional mode a quantum harmonic oscillator. The optical rotating-wave approximation is taken everywhere. Nothing else is dropped by default: the Hamiltonian is built from the exact displacement operator exp(i eta (a + a^dag)), so the carrier, every sideband, the Debye-Waller factors and off-resonant terms are all present. Raman drives adiabatically eliminate the excited manifold and report the residual excited population as scattering. The Lamb-Dicke and sideband rotating-wave approximations exist as builder options, never as defaults.
- **Truncation.** Motional Fock spaces are truncated by a policy with a boundary-population monitor, so a result is either converged or flagged.
- **Size.** The joint spin-motion space grows as 2^N x prod_m d_m. Up to a few thousand dimensions a run is JOINT_EXACT. Above that it switches to GATE_LOCAL, where each gate is exact on the ions it addresses and the modes they couple to, and the approximation that introduces is measured and reported.

Out of scope: shuttling and transport dynamics, photonic interconnects, error-correction logic, gate-count optimization, crystal melting dynamics, gates on mixed-species crystals (their crystals and modes are computed), and mid-circuit measurement (the scheduler refuses it).

## The run

`Machine.run(circuit, shots, seed=...)` (`run/pipeline.py`) walks one pipeline:

1. **Compile** (`control/compiler.py`): standard gates to the native set GPi, GPi2, MS, ZZ and virtual RZ, with phase tracking; every block and the whole circuit are verified against the target unitary up to a global phase.
2. **Calibrate** (`calibration/`): the scheduler reads only a `CalibrationTable`. By default it is the closed-form surrogate with exact spot checks of the entangling gate, cached per device and seed. `Machine.calibrated(method="experiments")` fits the table from simulated laboratory experiments instead.
3. **Schedule** (`control/schedule.py`): native gates become pulses with absolute times, from the table and the device's beam roles. Virtual Z rotations become phase shifts of later pulses, light shifts are compensated, and the control hardware chain (DDS quantization, modulator response, amplifier bandwidth, beat-note phase) shapes what is played.
4. **Prepare** (`prep/`): Doppler cooling, pulsed Raman sideband cooling and optical pumping produce per-mode thermal states and a pumped internal state with its preparation error.
5. **Select the space** (`run/space.py`): each mode is resolved, frozen or dropped by the closed-form integrals of the played waveform. Resolved modes get Fock caps from the pulse's coherent excursion plus a margin. `decide_level` picks JOINT_EXACT or GATE_LOCAL against the guards.
6. **Evolve** (`dynamics/`, `run/gate_local.py`): the initial mixture is split into weighted pure branches, and every pulse and idle interval is integrated under the one Hamiltonian builder with the noise channels the device implies.
7. **Read out** (`readout/`): the joint internal outcome is sampled from the final state, so entanglement survives, then turned into bits by the fluorescence model and a discriminator.

The `Result` carries counts, probabilities, error bars from the effective sample size, SPAM estimates, and `Diagnostics`: the level and why, the space and mode classes, boundary populations, integrators and tolerances, seeds, the calibration used, a closed-form intrinsic error budget and the list of approximations. `last_record(result)` returns the `RunRecord` behind it (compile report, schedule, space, traces, readout records). `Machine.submit` runs the same pipeline in a worker process behind a `Job`, with the `RunSpec` a JSON document can carry.

## Package layout

```
qutip_trap/
  __init__.py      Machine, Circuit, Result, the option objects, FidelityLevel, Device, presets
  machine.py       Machine: a device, its table, Physics/Numerics/Readout and the level policy
  options.py       Physics (which effects), Numerics (how to integrate), Readout (how to read)
  units.py         Hz and rad/s types, CODATA constants; hashing.py: canonical digests
  species/         atomic data per isotope, hyperfine/Zeeman structure, dipole and quadrupole couplings
  trap/            Mathieu stability, pseudopotential, surface electrodes, crystal and normal modes, heating
  light/           beams, Raman and optical drives, the optical Bloch steady state, recoil
  prep/            Doppler, sideband, EIT and polarization-gradient cooling, optical pumping, the recipe
  noise/           spectra, quasi-static sampling, stochastic processes, scattering channels, collisions,
                   dynamical decoupling and filter functions, channel summaries
  readout/         fluorescence rates, photon records, discriminators, POVMs and budgets
  device/          Device and its derived quantities, the example presets, the JSON record
  hilbert/         the composite space, cached operators, truncation policy
  dynamics/        the Hamiltonian builder, the JOINT_EXACT engine, solvers, kernels, tomography
  control/         native gates, compiler, pulses, scheduler, pulse shaping, composite pulses, hardware chain
  calibration/     the surrogate table, the entangling spot check, calibration by experiments
  experiments/     simulated laboratory scans with typed, fitted results
  benchmarks/      randomized benchmarking, GHZ fidelity, quantum volume, error model and exporters
  run/             the pipeline, space selection, fidelity levels, GATE_LOCAL, results, jobs
  validation/      closed forms from published papers that the tests compare against
  io/              OpenQASM 2 and IonQ JSON; interop/: a Qiskit BackendV2
qutip_trap_app/    the Flet application (a separate package in the uv workspace)
tests/
```

Dependencies point downward: `control/` never imports `calibration/`, and the core never imports the application or Flet (both are tested).

## Data model

- `Species`: immutable atomic data for one isotope with a citation per constant; everything hyperfine-resolved (Zeeman energies, dipole elements, Raman and Rayleigh amplitudes, leakage branching) is derived, never typed in.
- `Trap`: explicit secular frequencies or rf/dc voltages through the Mathieu equation, with stray field and shims. `Crystal`: equilibrium positions, the 3N normal modes, Lamb-Dicke parameters.
- `Beam`, `Field`, `NoiseModel`, `Detector`, `HardwareChain`: the rest of the apparatus. `NoiseModel()` is quiet; every channel is opt-in.
- `Device`: the aggregate, with `BeamRoles` naming which beams drive which gates; `Device.derived()` returns every computed number and `Device.hash()` is the canonical identity (the calibration cache key).
- `Machine`: a frozen record of a device, its table (None: the surrogate, built on demand), `Physics`, `Numerics`, `Readout` and a `FidelityLevel`; variants come from `dataclasses.replace`.
- `Circuit` and `Operation`: a small IR of standard and native gates with measured qubits and registers.
- `CalibrationTable`: plain data of pulse parameters with provenance and status; `Schedule`, `Pulse`, `Drive`, `Tone`: what is played.
- `Result`, `Diagnostics`, `RunRecord`, `RunSpec`, `Job`: what comes back.

## Numerics

- **One master equation.** d rho/dt = -i[H(t), rho] + sum_k D[L_k] rho on the ions' levels and the resolved modes. Every drive, whether a calibration scan, a gate or a cooling pulse, is a set of terms assembled by the one builder (`dynamics/hamiltonian.py`).
- **Solvers.** `sesolve` on pure branches; `mesolve` with collapse operators up to dimension 128; keyed `mcsolve` trajectories above it. The integrators escalate `dop853` then `vern9` (atol 1e-10, rtol 1e-8). Idle intervals with a diagonal Hamiltonian are applied in closed form. Branches of the initial mixture below 1e-6 weight are dropped, and the dropped weight is reported.
- **Truncation.** After every pulse the monitor checks the top Fock levels of each resolved mode (threshold 1e-6) and the margin above the populated range, and raises the cap and repeats the pulse when either fails. A cap the policy wanted beyond `mode_dimension_max`, or a boundary left above threshold after the retries, raises a `TruncationWarning`.
- **Levels.** AUTO runs JOINT_EXACT up to a joint dimension of 4096 and 2e7 drive-operator non-zeros, and GATE_LOCAL above that. Each GATE_LOCAL step (a gate or an idle interval) runs exactly on the addressed ions, their crosstalk neighbours and the modes they couple to. Its channel is extracted by process tomography: from the Stinespring isometry of propagated basis kets for unitary steps, or by a least-squares Choi fit projected onto CPTP maps for dissipative ones. The channel is applied to the register, and the motional state is carried by a tracked model. Each step reports its residual displacement, purity deficit and bounds; the tests compare GATE_LOCAL with JOINT_EXACT where both fit.
- **Speed.** Ket segments are integrated in the exact rotating frame of the diagonal H_0. The drive operator is applied mode by mode when that is cheaper than the assembled sparse matrix (`dynamics/kernels.py`). Internal-state-only segments reuse cached propagators. Trajectories, branches and samples run in parallel maps whose worker count is capped by memory. None of these changes a result beyond the solver tolerance.
- **Seeds.** One root `SeedSequence` per run is spawned by (sample, trajectory, shot, ion, channel), so every variate is independent of execution order, worker count and truncation retries; the same shots, seed and machine hash reproduce every shot.

## Noise

Every noise source enters as physics, never as a phenomenological channel. A white spectral part becomes a Lindblad operator (heating, motional and qubit dephasing, intensity noise, photon scattering with recoil). A slow part becomes a quasi-static parameter drawn per dynamical sample at the shot clock (field drift, mode-frequency and Rabi offsets, frozen spectator Fock states). A band becomes a sampled time series that the drive coefficients see. Rare events such as background-gas collisions become heralded shots. Circuit-level summaries of a simulated channel (entanglement and average gate infidelity, Pauli twirl, depolarizing rate) are exported, never integrated.

## Control and calibration

- Native gates (`control/native.py`): GPi(phi), GPi2(phi), MS(phi0, phi1, theta), ZZ(theta) and RZ in radians, IonQ turns at the boundary. The compiler uses ZXZXZ for any SU(2), Maslov's one-XX CNOT, an exact controlled phase, and KAK for any SU(4) with at most three entangling gates.
- The scheduler reads only the table and the beam roles. The MS gate is a bichromatic pulse whose amplitude-modulated segments close every resolved mode's phase-space loop (`control/shaping.py`). A light-shift ZZ gate is available as `Physics(entangler="zz")`.
- The surrogate table takes derived values as seeds and corrects the entangling waveform by exact spot checks on the gate's own space. Calibration by experiments runs the laboratory's dependency order (field, micromotion, modes, Rabi, Stark, qubit frequency, crosstalk, entangling scans, detection, heating), with shot noise and readout errors, and records every entry's uncertainty.

## Readout

Four layers, each replaceable. (1) Rates: the photon rate of the bright state and the off-resonant pumping out of each manifold, from the detection beams' optical Bloch steady state. (2) Photon records: a bright/dark/shelf continuous-time Markov chain with Poisson counts, dead time, afterpulsing and neighbour crosstalk. (3) Discriminators: a threshold (the default), time-resolved maximum likelihood, adaptive and first-photon protocols, camera decoding. (4) The measurement operator: the product POVM of the record-plus-discriminator chain, applied directly to the sampled outcome on the fast path (`Readout(mode="fast")`, exact at zero readout crosstalk), or the full record path (`mode="full"`). Readout error is never applied twice.

## The laboratory

- `experiments/`: Rabi, Ramsey, sideband spectroscopy, thermometry, heating rate, Stark, crosstalk, field, micromotion, detection, MS and parity scans on a machine through the same engine, each fitted into a typed result with uncertainties. A result can propose a table update.
- `benchmarks/`: single- and two-qubit, simultaneous and Knill-style randomized benchmarking, GHZ fidelity with its parity bound, and quantum volume. Each reports the simulator's own budget beside the measured number: closed-form scales, the GATE_LOCAL channel of every native gate kind, and SPAM. `Machine.error_model()` derives per-gate infidelities, durations and SPAM, and exports them in IonQ's, Quantinuum's and the QDK estimator's vocabularies.
- `io/` and `interop/`: OpenQASM 2 in and out, IonQ circuit JSON and job bodies, IonQ v1 and v0.4 result formats, and a Qiskit `BackendV2`.

## The application

`qutip_trap_app` shows one run of the example 171Yb+ chain at five zoom levels:
- **Machine**: the histogram beside the ideal distribution, and the shots behind a bar.
- **Circuit**: the native gates and the register after each.
- **Schedule**: the pulses, their tones against the modes, and loop closure.
- **Dynamics**: one pulse re-simulated at fine resolution, with phase-space loops, populations, the Fock heatmap, jumps, the process matrix and convergence re-checks.
- **Equation**: the terms of H(t) and the collapse operators of that pulse.

A job is the core's `RunSpec` plus the ion count and the preset's keyword arguments. `record.execute` calibrates and runs the job's `Machine`, then copies what the screens read into a `Record` of plain values and arrays, next to a `LiveRun` of the core objects.

The UI never calls the solver. `workers.py` keeps one process that runs jobs, holds each run's `LiveRun` and answers zoom, tomography and re-check requests; under Pyodide it runs in the page's thread. `resim.py` re-simulates a gate step by chaining the run's own engine over the steps before it, and caches every result on the record.

The view-models (`viewmodel/`, no Flet) turn a record into what a screen shows. The views (`views/`) are plain functions of the store and the route, so every screen can be built and checked without a Flutter client. Every import of the core goes through `core.py`. `.github/workflows/pages.yml` deploys the app as a static Pyodide site.

## Testing

`uv run pytest -n 4 --dist loadscope -m "not slow"` is the fast tier that CI runs on every push. The nightly run adds the `slow` tests (level-C solves, full calibrations, long scans) and runs the `heavy` ones alone (multi-gigabyte factorizations, timing measurements). The tests pin published closed forms and numbers (atomic structure, crystal modes, cooling limits, Rabi and MS dynamics, readout optima, RB and GHZ relations), check convergence in tolerance and truncation, compare GATE_LOCAL with JOINT_EXACT, run end-to-end circuits on both example machines, and execute docs/examples.md.
