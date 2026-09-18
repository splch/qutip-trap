# Rung 3: the dynamics

`qutip_trap.dynamics` is the one Hamiltonian builder and what it builds, the channels, the solvers, the matrix-free kernel,
the parallel maps and the pulse engine (PLAN.md Sections 4.3.1, 5 and 11). `Machine.engine` is the way here from
[schedule.md](schedule.md); the records a device is built from are [physics.md](physics.md).

## The engine

`JointExactEngine` is the JOINT_EXACT pulse engine of Section 5.4 and the one dataclass of the surface that is not frozen
(a service object with a report, a propagator cache and a progress hook). `run_pulses(device, schedule, state, space, sample, seeds, options)` evolves a `State` through every pulse of a schedule on a `HilbertSpace` and returns `Traces`
without any readout; `process_tomography(...)` and `tomography(...)` extract the channel of a pulse group as a
`ChannelSummary` (the Choi matrix, the CP and TP residuals, the average gate infidelity, the Pauli twirl and the
depolarizing rate of Section 6.8) or the full `TomographyRecord` the GATE_LOCAL walk tracks; `last_report` is the
`EngineReport` of the most recent run, with one `SegmentReport` per integration segment (the integrator and tolerance, the
right-hand-side evaluations, the boundary populations, the channels, the kernel, the frame and, since 0.4.0, the wall
seconds). `PulseEngine` is the protocol every engine implements, the one entry point `run`, the calibration and the
experiments share.

`Traces` is what `run_pulses` returns: the stored times, the expectation values (`P1[i]`, `n[m]`), the reduced register
state at every stored time, the mode occupations and the spin-averaged displacements per carried mode, the quantum jumps
of the trajectory path, the final `State` and the boundary populations. Since 0.4.0 it also carries, when
`Numerics.integration.store_marginals` (or `SolverOptions.store_marginals`) asks for them, `mode_marginal`: per carried
mode the (T, d_m) Fock populations at every stored time, the same branch and trajectory weighting as the expectation
values, so that `sum_n n P(n, t)` equals the occupation at every time; and `wall_time_s`, the integration wall seconds per
pulse by its gate id. A `State` is the internal state, the `MotionalModel` (the reduced density matrix of every tracked
mode, the mean occupation of the others, the frozen set) and, when it exists, the joint state;
`prepare(device, space, table, sample, seeds)` builds the initial one from the preparation recipe, whose physics is `run_preparation(device, recipe)`, a `PreparationRun` with the mean occupations after Doppler cooling, sideband cooling and pumping.

## The Hamiltonian

`build_hamiltonian(device, pulses, space, sample=, options=, ...)` is the one builder of Section 4.3.1: the terms a segment
integrates as a `BuiltHamiltonian` (the QobjEvo, one `DriveRecord` per pulse and ion with its Lamb-Dicke parameters, the
Debye-Waller factor, the crosstalk and the Stark shift, the approximations made), written under `BuilderOptions` (the
frame, the Lamb-Dicke expansion order or the exact displacement, the rotating-wave approximation and its sideband cutoff,
the micromotion and phase modes, the Stark, anharmonic, crosstalk and Debye-Waller terms, the beam-curvature couplings,
the gradient-drive form and the kernel). `Physics.builder` sets it on a machine.

## Spaces and truncation

A `HilbertSpace` (Sections 5.1 and 5.4) is one qudit factor per ion, one truncated oscillator per resolved mode as a
`ModeTruncation` (the mode, its Fock dimension, the expected range, the largest Lamb-Dicke parameter and the element
tolerance the margin was derived for), an optional ENR group as one factor, and the frozen spectators; it builds and caches
its operators once (`CachedOperators`). The matrix-free kernel of Section 11.3 item 4 holds the drive operators factorized
(`FactorizedOperator`, `factorized_qobj`, `is_factorized`) and applies them mode by mode (`apply_drive_kernel`).

## The channels and the noise sample

A `CollapseOp` is one collapse operator with its rate, its channel name and the ion or mode it acts on; the device's own
channels come from the `NoiseModel` of [physics.md](physics.md), the photon-scattering channels of every pulse from
`scattering_channels` (Sections 4.5.5 and 6.5, under `ScatteringOptions`: which branches, the recoil option) with
`scattering_estimates` and `scattering_rate` the closed-form rates a run reports instead when `Physics.scattering` is
"estimate", and `CollisionEvent` and `collision_rate_per_ion` the background-gas collisions of Section 6.7. A `NoiseSample`
is one draw of the quasi-static and sampled noise values a run evolves under, keyed by name: `KEY_BRANCH_WEIGHT` (the
weight of the initial-mixture branch), `key_frozen_n(mode)` (the Fock state of a frozen spectator), `key_qubit_offset_hz(ion)`
and `key_mode_offset_hz(mode)` (the quasi-static frequency offsets); `SeedSpec` is the root seed with its keyed children by
(sample, trajectory, shot, ion, channel).

## Solvers and their options

`SolverOptions` is the numerical policy in one record (the tolerances and the integrator ladder, the Section 11.5 guards,
the caps and monitors, the trajectory method and count, the parallel map, the GATE_LOCAL tomography knobs, the physics
switches a run still reads from it); `Numerics.to_solver_options(physics)` builds it from the option objects and
`Numerics.from_solver_options` reads one back. The `Numerics` groups are exported here too: `Integration`, `Truncation`,
`Trajectories`, `GateLocal` and `Parallel`. `convergence_check` repeats an evolution at ten times tighter tolerances and
reports the change as a `ConvergenceReport` (Section 5.5). `map_tasks` and `worker_count` are the parallel maps of Section
11.3 item 9 (QuTiP's serial, multiprocessing and loky maps, capped by the memory rule of `dynamics/parallel.py`).

## The multi-level builder and the trajectories

Beyond the two-level register the multi-level optical Bloch builder of Section 4.2.8 integrates the full atomic structure:
`InternalLevels` names the register levels of an ion (the qubit pair and the leakage levels), `internal_levels` builds
them, `MultiLevelOptions` selects the levels, the light and the truncation of a `BlochModel` (the density matrix of one
ion under its beams), a `ModeSpec` a motional mode it couples to, `SteadyStateReport` the direct steady state of Section
5.3 and `DarkStateReport` the coherent dark states of Section 8.1 the detection light must avoid. A `Trajectory` is one
quantum-jump trajectory of the mcsolve path with its jumps, and a `ControlSegment` one piece of a control sequence the
decoupling and composite-pulse layers integrate.

## Channels of gates

`gate_channel(machine, kind)` is the Section 6.8 summary of one native gate kind on its own: a `GateChannel` with the
steps of the gate played once from the prepared motional state on the exact gate-local space, cached per machine hash and
kind; `average_gate_infidelity`, `entanglement_infidelity` and `depolarizing_rate` convert a Choi matrix into the three
numbers of docs/conventions.md ("Fidelity measures"), `pauli_twirl` gives the twirled Pauli channel, `choi_from_unitary`
and `depolarizing_choi` the Choi matrices of a unitary and of a depolarizing channel, `input_states` the linearly
independent inputs of state tomography and `kraus_operators` the Kraus form of a Choi matrix (what the GATE_LOCAL walk
applies to the register). The least-squares reconstruction and the CPTP projection moved to
[experimental.md](experimental.md) in 0.4.0.

## Reading the record

The readout of Section 8 turns the joint outcome into photon records and back into bits. A `POVM` is the confusion of a
detection scheme, `povm_for(device, ion, table)` the one a run applies on the fast path; a `PhotonRecord` is the counts per
ion (and per sub-bin when time resolved), a `ReadoutOutcome` the discriminated bits with the posteriors and heralds, and a
`ReadoutScheme` names the strategy over the record: `ThresholdDiscriminator` (a count threshold in a window), `TimeResolvedML`
(the maximum-likelihood decision over the sub-bins), `AdaptiveML` (the adaptive stopping rule) and `FirstPhoton` (Noek's
first-photon protocol). `FluorescenceRates` and `DetectionRates` are the bright and dark count rates of an ion under its
detection beam with the pumping rates (`detection_rates_for_ion`), `optimize_threshold` the threshold that minimizes the
mean error, and `RecordModel` the count-distribution model a record is scored against. `ReadoutBudget` with its
`BudgetLine` entries is the error budget of a readout (the pumping, the dark counts, the crosstalk, each with its provenance
id); `DetectionCalibration` and `calibrate_detection` fit the threshold and the window from simulated histograms.

The Section 5.4 walk above the joint-dimension guard is `run.gate_local`: its `GateLocalReport`, `GateLocalStep`, `GateStep`
and the register they carry are described with the `RunRecord` on [machine.md](machine.md).
