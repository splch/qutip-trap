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

## Coming in 0.3.0

`run`'s keyword arguments other than `table`, `level`, `seed`, `keep_final_state` and `progress` move to the option objects
(`Physics`, `Numerics`, `Readout`) and are rewritten with a warning; `SolverOptions` becomes a deprecated constructor of
`Numerics`; `gate_drives=` and `entangling_drives=` on `run`, `calibrate` and `schedule` are deprecated in favour of
`Device.roles`; `calibrate_with_report` and `compile_with_report` become aliases of `calibrate` and `Machine.compile`;
`quiet_noise_model()` becomes `NoiseModel()`.

## Removed

Nothing yet.
