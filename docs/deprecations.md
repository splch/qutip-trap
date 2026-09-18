# Deprecations

The policy (docs/api_implementation_plan.md, Section 1; `qutip_trap/_compat.py`): names are added first, deprecated second and
removed third. A name is deprecated only once its replacement has shipped in a release; it is removed no earlier than two
minor releases after the warning, always with a warning-free path. Every deprecated call raises a
`QutipTrapDeprecationWarning` (a `DeprecationWarning`, so Python shows it from scripts and hides it inside libraries; pytest
shows it, and this package's own test session turns it into an error) whose message names the deadline and the fix. While
the version is 0.x the minor number counts the releases: deprecated in 0.2.0 means removable in 0.4.0.

## Deprecated in 0.2.0, removed in 0.4.0 at the earliest

| Name | Replacement | Why |
|---|---|---|
| `DevicePreset.run_kwargs()` (returns `{}` and warns) | nothing: the preset's device carries its drive maps as `Device.roles`; `DevicePreset.machine()` gives the `Machine` | the drive maps were threaded through every `run`, `calibrate` and benchmark call because the device did not know which beams played which gates |
| `Result.to_ionq_json()` | `Result.to_ionq_v1_probabilities()` | the exporters are named by the convention they emit (v1: decimal keys, qubit 0 the 2^0 bit) |
| `Result.to_ionq_histogram()` | `Result.to_ionq_v1_histogram()` | as above |
| `Result.to_ionq_shots()` | `Result.to_ionq_v1_shots()` | as above; `Result.from_ionq_v1_shots()` closes the round trip |
| `Result.to_ionq_v2(registers)` (unchanged behaviour) | `Result.to_ionq_v2_probabilities()`, `to_ionq_v2_histogram()`, `to_ionq_v2_shots()` | the old method's strings put qubit 0 rightmost and carried no envelope; IonQ's v0.4 format is `{"probabilities": {"registers": {"output_all": ...}}}` with q[0] the leftmost character (`io/ionq.py` records the source) |

## Deprecated in 0.3.0, removed in 0.5.0 at the earliest

| Name | Replacement | Why |
|---|---|---|
| the 0.1.0 keyword arguments of `run` (`t0_s`, `shot_period_s`, `noise`, `internal_levels`, `crosstalk_suppression`, `stark_compensation`, `entangler`, `channels`, `builder_options`, `options`, `space`, `caps`, `enr_group`, `samples`, `parallel`, `readout` as a string, `discriminator`, `povm_samples`, `gate_drives`, `entangling_drives`, `calibrate_kwargs`; `run.job.LEGACY_RUN_KEYWORDS`) | `physics=Physics(...)`, `numerics=Numerics(...)`, `readout=Readout(...)` on `run`, or a `Machine` with them; the drive maps on `Device.roles`; `Machine.calibrated(**scans)` for the calibration settings | `run` mixed four concerns in twenty-five keywords; each is rewritten onto the machine with a warning naming its new home, and the result is the same |
| `options.to_run_kwargs` | pass the objects to `run`, or `Machine.run` | the translation ran in the direction 0.3.0 reversed |
| `gate_drive=`, `gate_drives=`, `entangling_drives=` on the experiments and on `calibrate` (`machine.DRIVE_KEYWORDS`) | `Device.roles` | the device knows which beams play which gates |
| `calibrate(..., surrogate=True / False)` | `calibrate(machine, method="closed_form" / "experiments")` | one keyword routed to two different functions by a boolean |
| `calibrate_with_report` | `calibrate(machine, method=...)`, which returns the `CalibrationReport` on every call | the two entry points doubled the calibration |
| `compile_with_report` | `Machine.compile`, the same `CompileReport` | the report belongs to the executor |
| `table=`, `options=`, `level=`, `noise=`, ... on `randomized_benchmarking`, `ghz_fidelity`, `quantum_volume`, `gate_channel`, `channels_for` (`run.job.LEGACY_CALL_KEYWORDS` and the run keywords) | a `Machine` as the first argument (`dataclasses.replace(machine, table=..., level=...)`) | the benchmarks forwarded `run`'s keywords; the machine carries them |
| `quiet_noise_model()` | `NoiseModel()`, whose every default means off | the quiet model is the default model |

The scheduler's own `schedule(circuit, device, table, gate_drives=, entangling_drives=)` keeps its explicit drive maps: they are
rung 2's plumbing (the `Prefix` of Appendix E carries the resolved maps), and the roles are their default. `run(circuit, device, shots)` keeps taking a device: it is the 0.1.0 entry point and builds the machine itself.

## Deprecated in 0.4.0, removed in 0.6.0 at the earliest

| Name | Replacement | Why |
|---|---|---|
| a bare `Device` as the first argument of an experiment, of `calibrate` or of a benchmark (`machine.warn_bare_device`; `calibrate(device)` still returns the table) | `Machine(device)` or `as_machine(device)`, and `.table` on the report `calibrate` returns for a machine | the machine supplies the table and the option objects; two argument types with two return types was the 0.3.0 bridge, not the destination |
| `qutip_trap.dynamics.choi_least_squares`, `qutip_trap.dynamics.project_cptp` (the rung's export; `qutip_trap.api` keeps them) | `qutip_trap.experimental` | the tomography internals are outside the stability guarantee until they have a second consumer (docs/api_implementation_plan.md 3.3) |

## Coming in 0.5.0

The names deprecated in 0.3.0 become removable: the 0.1.0 keyword arguments of `run` and of the benchmarks,
`options.to_run_kwargs`, the drive keywords of the laboratory, `calibrate(surrogate=)`, `calibrate_with_report`,
`compile_with_report` and `quiet_noise_model`. Each removal will move its row into the table below.

## Removed

| Name | Removed in | Deprecated in | Use instead |
|---|---|---|---|
| `DevicePreset.run_kwargs()` | 0.4.0 | 0.2.0 | nothing: `Device.roles` carries the drive maps, `DevicePreset.machine()` gives the `Machine` |
| `Result.to_ionq_json()` | 0.4.0 | 0.2.0 | `Result.to_ionq_v1_probabilities()` |
| `Result.to_ionq_histogram()` | 0.4.0 | 0.2.0 | `Result.to_ionq_v1_histogram()` |
| `Result.to_ionq_shots()` | 0.4.0 | 0.2.0 | `Result.to_ionq_v1_shots()` |
| `Result.to_ionq_v2(registers)` | 0.4.0 | 0.2.0 | `Result.to_ionq_v2_probabilities()`, `to_ionq_v2_histogram()`, `to_ionq_v2_shots()` (IonQ's v0.4 envelope, q[0] leftmost) |

A removed name raises the plain `AttributeError`; its replacement has been in place since the release named in the third
column.
