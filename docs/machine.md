# Rung 0: the machine

The front door of qutip-trap is the root namespace, `import qutip_trap as trap` (docs/conventions.md, "Vocabulary"). It
holds the executor and what goes in and out of it, and nothing else: every name below is imported on first use, so
`import qutip_trap` stays free of numpy and qutip (`tests/test_import_time.py`). The pages below this one walk down the
ladder, [circuit.md](circuit.md), [schedule.md](schedule.md), [dynamics.md](dynamics.md) and [physics.md](physics.md), with
[laboratory.md](laboratory.md) for the experiments, the calibration and the benchmarks and
[experimental.md](experimental.md) for the names outside the stability guarantee; [examples.md](examples.md) runs the whole
ladder on the example device.

## The machine

A `Machine` is a trapped-ion computer as a client sees it: the physical `Device` (rung 4) with the roles its beams play
(`BeamRoles`: which beams drive the single-qubit gates of each ion, which the entangling gates, which one detects), the
calibration table it runs on (None means the closed-form surrogate, cached per device), the three option objects and the
level policy, in one frozen record. Variants are `dataclasses.replace(machine, ...)`; `as_machine(device)` wraps a bare
device in a default machine (the laboratory's first argument; passing the device itself warns since 0.4.0).

| Method | What it does |
|---|---|
| `Machine.run(circuit, shots, seed=, keep_final_state=, progress=)` | compile, calibrate, schedule, prepare, evolve and read out: the `Result` |
| `Machine.submit(circuit, shots, seed=, label=)` | the same run in a worker process behind a `Job` (0.4.0) |
| `Machine.spec(circuit, shots, seed=)` | the `RunSpec` of that call, the record of the request (0.4.0) |
| `Machine.compile(circuit)` | rung 1: the `CompileReport` of the native circuit |
| `Machine.schedule(circuit, seed=)` | rung 2: the pulses with absolute times, nothing integrated |
| `Machine.engine` | rung 3: the `JointExactEngine` a run of this machine builds |
| `Machine.calibrated(method=, seed=, **scans)` | the same machine with its table pinned |
| `Machine.estimate(circuit)` | an `Estimate`: the level and why, the space, the pulse counts and a wall-time guess before anything is integrated |
| `Machine.error_model(qubits=)` | the inverse direction: the `ErrorModel` of [laboratory.md](laboratory.md) |
| `Machine.specs()` | the derived quantities with their provenance ids, then the roles, the table and the level |
| `Machine.hash()` | the identity a record stores: device digest, roles, table digest and policy; `name` is not part of it |

The example machines come from `presets`: `yb171_chain(n)` (a 171Yb+ chain with a global 355 nm Raman pair for the
entangling gates, one addressing pair per ion, the oblique 369.5 nm detection beam, Crain's SNSPD detector) and
`ca40_optical(n)` (the 40Ca+ optical qubit on the 729 nm quadrupole line, Myerson's PMT chain, no entangling drive), each
returning a `Machine`; `DevicePreset` is the 0.1.0 record of a device with its drive maps, and `DevicePreset.machine()`
bridges code that holds one.

## The policy: three option objects and a level

The knobs of a run are grouped by what they govern (docs/api_proposal.md Section 4.7), each a frozen dataclass that also
accepts a plain mapping of its fields:

- `Physics`: which effects are simulated. `noise`, `internal_levels`, `scattering` ("estimate" or "channels"),
  `scattering_recoil`, `intensity_noise_channels`, `hardware_chain`, `stark_compensation`, `crosstalk_suppression`,
  `entangler` ("ms" or "zz"), `extra_channels`, `builder` (the Hamiltonian builder's options), `t0_s` and `shot_period_s`.
  Every default is what a run did in 0.1.0: the device's noise on, two register levels per ion, scattering as an
  estimate, the hardware chain applied.
- `Numerics`: how the integration is done, nested so that a physicist reads the truncation policy without the tomography
  knobs: `integration` (tolerances, the integrator ladder, the rotating frame, the propagator cache and, since 0.4.0,
  `store_marginals` for the per-time Fock populations of [dynamics.md](dynamics.md)), `truncation` (the Section 11.5
  guards, the caps, the branch cutoff, an explicit space or ENR group), `trajectories`, `gate_local` and `parallel`, plus
  the `convergence_check` of Section 5.5.
- `Readout`: how the photon record is read, `mode` "fast" (the POVM on the joint outcome) or "full" (every photon record
  generated and discriminated), the `discriminator` and the `povm_samples`.
- `FidelityLevel`: `AUTO` picks `JOINT_EXACT` inside the Section 11.5 guards and `GATE_LOCAL` above them; either can be
  forced. `decide_level(device, circuit, options)` returns the `LevelDecision` (the joint dimension and the drive-operator
  non-zeros against the guards, and the reason in words) that `AUTO` acts on; `Diagnostics.level_reason` repeats it on the
  result.

## What goes in

A `Circuit` is the program: `n_qubits`, the `Operation` tuple (a gate name, its qubits, its parameters in radians), the
terminal `measure` targets (every qubit unless narrowed) and the classical `registers`. It is also a persistent builder,
`trap.Circuit(2).h(0).cnot(0, 1)`, with the wire formats `from_openqasm`, `from_ionq`, `to_openqasm` and `to_ionq`; the
whole rung is [circuit.md](circuit.md).

## What comes out

A `Result` carries the per-shot bitstrings and their aggregation into `counts`, `probabilities` and `error_bars` (qubit 0 the
least-significant bit of every key; docs/conventions.md), the photon records and posteriors when the record was read in
full, the noise sample of every dynamical sample, the herald flags, the SPAM errors per qubit, the recombined register
state when kept, and the `Diagnostics`: the level that ran and why, the `HilbertSpace` and the class of every mode, the
boundary populations, the branch cutoff, the integrators, the seeds, every approximation made, the intrinsic error budget
and the GATE_LOCAL summary. `Result.to_dict()` and `Result.from_dict()` are the versioned record of
`schemas/result.schema.json`; `to_ionq_v1_probabilities`, `to_ionq_v1_histogram`, `to_ionq_v1_shots` (decimal keys) and
`to_ionq_v2_probabilities`, `to_ionq_v2_histogram`, `to_ionq_v2_shots` (IonQ's v0.4 envelope, q[0] the leftmost character)
are the IonQ formats, `from_ionq_v1_shots` closes the round trip, and `reversed_bits()` reverses every key for comparisons
with the SDKs that put qubit 0 first. `Result.machine_hash`, `created_at` and `duration_s` identify the run;
`trap.__version__` is the release the record carries as `qutip_trap_version`.

`run(..., progress=callback)` and `Machine.run` call back with a `Progress` per pulse, per (sample, branch) run, per sample
and per readout: the `stage`, how many of `total` are `done`, the seconds elapsed, and `fraction`.

What a run produced besides the `Result` is its `RunRecord` (the compile report, the schedule, the space selection, the
preparation, the branches, the traces, the readout stage, the table it ran on, the GATE_LOCAL report), reachable through
`last_record(result)` for a result of this process and through `Job.record()` for a submitted one; `RunState` is the
persistent machine state a run threads through its shots (the ion order, the dark and lost flags, the events).
`SpaceSelection` and `select_space` are the Section 5.2 selection of the joint space (resolved, frozen, dropped and ENR
modes) a JOINT_EXACT run reports, `resolve_level` the 0.1.0 form of the level decision, and the GATE_LOCAL walk of Section
5.4 leaves a `GateLocalReport` whose `GateLocalStep` entries record, per `GateStep` of the schedule (`gate_steps`, on the
gate-local `step_space`), the channel extracted, the residual displacement, the frozen-spectator excitation and, since
0.4.0, the register after the step and the channels applied to it ([dynamics.md](dynamics.md)).

## Jobs: the same run in a worker process (0.4.0)

`Machine.submit(circuit, shots, seed=)` returns a `Job`: the run of `Machine.run` in a spawned interpreter (so the run's own
parallel maps fork inside it as they do in-process), its `Progress` streamed back, cancellable between pulses, with the
`Result` and its `RunRecord` handed back when it is done. Braket's `LocalQuantumTask` and pytket's `ResultHandle` are the
precedents: a local run gets the remote surface so one code path serves both.

```python
job = fast.submit(bell, 200, seed=7, label="bell in the background")
job.status()                   # "running", then "done" | "failed" | "cancelled" (JobStatus)
job.progress                   # the latest Progress, or None before the first report
job.result()                   # blocks: the same Result machine.run(bell, 200, seed=7) returns, shot for shot
job.record()                   # the RunRecord behind it; last_record(job.result()) finds the same one
job.cancel()                   # stops within one pulse when the engines run in-process; cancel(terminate_after_s=t) kills
job.spec.to_dict()             # the RunSpec: circuit, shots, seed, the machine's hash, the option objects, the level, the label
```

`Job.result()` raises `JobCancelled` after a cancel and `JobError` with the worker's traceback after a failure; a
`timeout_s` turns a long wait into a `TimeoutError`. The unit of submission is the `RunSpec`, a frozen record that a JSON
document can carry: `RunSpec.of(machine, circuit, shots)` or `Machine.spec(...)` builds it, `to_dict()` and `from_dict()`
are its exact JSON form under `schemas/runspec.schema.json` (tuples become lists, integer keys strings), and `to_dict`
refuses by name the three option values a record cannot carry (explicit collapse operators, a declared `HilbertSpace`, a
discriminator object) rather than dropping them. A script that submits jobs runs under `if __name__ == "__main__":` like
every user of multiprocessing's spawn start method; jobs still running when the interpreter exits are terminated.

## The rungs below, and the doors

`circuit`, `schedule`, `dynamics` and `physics` are the rung modules (rungs 1 to 4), `io` the wire formats (OpenQASM 2 and
IonQ JSON, on [circuit.md](circuit.md)), `interop` the adapters to other SDKs (`qutip_trap.interop.qiskit`, a Qiskit 2
`BackendV2` over a machine, behind the `qiskit` extra) and `experimental` the names outside the stability guarantee
([experimental.md](experimental.md)). `qutip_trap.api` is the frozen Appendix E surface of PLAN.md: every object of the
rungs under its 0.1.0 name, unchanged.
