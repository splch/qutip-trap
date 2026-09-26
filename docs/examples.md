# Examples

A short tour, runnable in order: `tests/test_docs.py` executes every Python block below in one namespace. Frequencies
are in Hz, angles in radians and times in seconds; a bitstring key has qubit 0 rightmost, the least-significant bit
(PLAN.md Section 13, "Result bit order"). Every name outside the root namespace is imported from the module that defines
it, and [api.md](api.md) lists them.

## Quickstart

```python
import dataclasses

import qutip_trap as trap

machine = trap.presets.yb171_chain(2)             # the example two-ion 171Yb+ chain
bell = trap.Circuit(2).h(0).cnot(0, 1)            # every qubit is measured unless .measured(...) narrows it
result = machine.run(bell, shots=2000)            # compile, calibrate, schedule, prepare, evolve, read out
print(result.counts, result.diagnostics.level, "|", result.diagnostics.level_reason)
assert result.probabilities["00"] + result.probabilities["11"] > 0.98

estimate = machine.estimate(bell)                 # the level, the space and a wall-time guess, nothing integrated
assert estimate.level == result.diagnostics.level and estimate.space == result.diagnostics.space
assert trap.Result.from_dict(result.to_dict()).counts == result.counts

quiet = dataclasses.replace(machine, physics=trap.Physics(noise=False))   # a variant: the same device, no noise
assert quiet.hash() != machine.hash()
```

A `Machine` is frozen, and its variants are `dataclasses.replace(machine, ...)`: `numerics=trap.Numerics(branch_weight_min=1e-3)`
drops the lighter branches of the initial mixture, `level=trap.FidelityLevel.GATE_LOCAL` walks the circuit gate by gate.
`machine.run(bell, 200, progress=print)` reports every pulse, branch, sample and readout, and `machine.submit(bell, 200)`
runs it in a worker process behind a `Job`.

## A device

A device is physical parameters; everything else is derived, each number with the id of its provenance record in
`docs/provenance/ledger.yaml`.

```python
from qutip_trap.device.model import Device

device = machine.device
derived = device.derived()
print("modes (MHz):", [round(m.omega_hz / 1e6, 4) for m in device.crystal.modes])
print("qubit frequency of ion 0 (Hz):", round(derived.values["qubit_freq_hz[0]"]))
assert all(derived.provenance[k] for k in derived.values)
assert Device.from_dict(device.to_dict()).hash() == device.hash()
```

`device.specs()` renders the derived numbers as a report and `machine.specs()` adds the roles the beams play, the table and
the level policy.

## A calibrated run

`Machine.calibrated` pins a calibration table on the machine, so that every later call reads the same one; without it a run
builds the closed-form surrogate, cached per device.

```python
import numpy as np

from qutip_trap.control.compiler import ideal_probabilities
from qutip_trap.run.job import register_fidelity

windows = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
pinned = machine.calibrated(pairs=[(0, 1)], detection_records=2000, detection_windows_s=windows)
result = pinned.run(bell, 2000, keep_final_state=True)
print("ideal:", ideal_probabilities(bell), "| simulated:", result.probabilities)
infidelity = 1 - register_fidelity(result)
assert infidelity < result.diagnostics.intrinsic_budget.total        # inside the closed-form error budget
assert result.record is not None and result.record.schedule.pulses  # the RunRecord travels on the Result
print("SPAM per qubit (eps_B, eps_D):", {k: v for k, v in result.spam.items() if "." not in k})
```

## OpenQASM 2 and IonQ JSON

```python
from qutip_trap.io.ionq import dump_job, load_job

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = trap.Circuit.from_openqasm(qasm)
report = machine.compile(circuit)                 # native gates, every block verified against its target
native = report.circuit
print([op.name for op in native.ops], "| whole-circuit residual:", f"{report.circuit_residual:.1e}")
assert trap.Circuit.from_ionq(native.to_ionq()).ops == native.ops
assert load_job(dump_job(native, backend="simulator", shots=1000)).circuit.ops == native.ops
print(result.to_ionq_v1_probabilities(), result.to_ionq_v2_probabilities())
```

IonQ's v1 keys are decimal integers with qubit 0 the least-significant bit; the v0.4 envelope writes bitstrings in IonQ's
wire order, q[0] first. `result.reversed_bits()` reverses every key for comparisons with Cirq, Braket and PennyLane.

## An experiment

The laboratory's scans run on the machine through the same engine and are fitted the way the laboratory fits them.

```python
from qutip_trap.experiments.single_ion import rabi_scan

durations = tuple(float(t) for t in np.linspace(0.0, 30e-6, 13))
scan = rabi_scan(pinned, 0, durations, shots=200)
print("carrier Rabi frequency (Hz):", round(scan.value("f_rabi_hz")), "+-", round(scan.uncertainty("f_rabi_hz")))
assert scan.f_rabi_hz == scan.value("f_rabi_hz") and scan.requested.durations_s == scan.realized.durations_s
proposal = pinned.table.updated_with(scan)        # a proposal: adopt it only when scan.quality is "good"
assert proposal.rabi[(0, 2)].experiment == "rabi_scan" and proposal.kind_of("rabi[(0, 2)]") == "setpoint"
```

## Benchmarks

Every benchmark runs through `Machine.run` and reports, beside the measured number, what the simulator's own physics
accounts for.

```python
from qutip_trap.benchmarks.ghz import ghz_fidelity
from qutip_trap.benchmarks.rb import randomized_benchmarking
from qutip_trap.benchmarks.volume import quantum_volume

rb = randomized_benchmarking(pinned, (0,), (1, 128, 512), n_sequences=1, shots=2000, fix_offset=True)
print(rb.fidelity_form(), "| r per Clifford:", rb.error_per_clifford, "| predicted:", rb.budget.predicted["r_channel"])

ghz = ghz_fidelity(pinned, (0, 1), shots=400, analysis_phases_rad=np.linspace(0.0, np.pi, 4, endpoint=False))
print("bound (P0 + P1 + C)/2:", ghz.fidelity_bound, "| exact:", ghz.register_fidelity_max_phase, ghz.register_fidelity)
assert ghz.register_fidelity_max_phase >= ghz.register_fidelity - 1e-12

qv = quantum_volume(pinned, (0, 1), n_circuits=1, shots=200)
print("heavy-output probability:", qv.heavy_output_probability, "| ideal:", qv.ideal_heavy_probability)
assert not qv.protocol_circuit_count_met and not qv.passed   # the protocol asks for at least 100 circuits
```

## The error model

`Machine.error_model()` derives the phenomenological numbers a vendor emulator takes from the simulated device: one
GATE_LOCAL channel per native gate kind and qubit, the SPAM errors and the durations, with exporters to IonQ's,
Quantinuum's and the QDK estimator's vocabularies.

```python
from qutip_trap.benchmarks.budget import gate_channel

em = pinned.error_model()
print({k: f"{v:.1e}" for k, v in em.infidelity.items()}, em.to_ionq_noise(), em.to_qdk_qubit_params())
kinds = em.single_qubit_kinds
assert em.to_ionq_noise()["r_1q"] == 2.0 * sum(em.infidelity[k] for k in kinds) / len(kinds)   # F_avg = 1 - r/2

channel = gate_channel(pinned, "gpi2[0]")         # one kind's channel, cached with the error model's
twirl = channel.steps[0].summary.pauli_twirled
print("reduced to qubit 0:", f"{channel.infidelity_on((0,)):.1e}", "| twirl p_II:", twirl["II"])
```
