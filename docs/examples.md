# Examples

Every ```python block below builds on the ones before it; `tests/test_docs.py` runs them in order in one namespace (a few minutes). Frequencies are in Hz.

## A histogram

A `Machine` is a device with the roles its beams play, the calibration it runs on, the option objects `Physics` (which effects are simulated), `Numerics` (how the integration is done) and `Readout` (how the record is read), and a `FidelityLevel`. The example machines come from `trap.presets`; a circuit is built with one method per gate.

```python
import dataclasses

import qutip_trap as trap

machine = trap.presets.yb171_chain(2)                # the example two-ion 171Yb+ device
bell = trap.Circuit(2).h(0).cnot(0, 1)               # every qubit is measured unless .measured(...) narrows it
quick = dataclasses.replace(machine, numerics=trap.Numerics(truncation={"branch_weight_min": 1e-3}))
result = quick.run(bell, shots=2000)                 # compile, calibrate (cached per device), schedule, prepare, evolve, read out
print("histogram:", {k: round(v, 4) for k, v in sorted(result.probabilities.items())})
print("level:", result.diagnostics.level, "|", result.diagnostics.level_reason)
assert result.probabilities["00"] + result.probabilities["11"] > 0.98
estimate = quick.estimate(bell)                      # the level, the space and a wall-time guess before anything runs
assert estimate.level == result.diagnostics.level and estimate.space == result.diagnostics.space
deeper = dataclasses.replace(quick, level=trap.FidelityLevel.GATE_LOCAL)
quiet = dataclasses.replace(quick, physics=trap.Physics(noise=False))
record = result.to_dict()                            # the versioned JSON record
assert trap.Result.from_dict(record).counts == result.counts
```

`quick.run(..., progress=print)` reports a `Progress` per pulse, branch, sample and readout. `result.to_ionq_v2_probabilities()` is IonQ's v0.4 envelope (bitstrings in wire order, q[0] first) and `result.to_ionq_v1_probabilities()` the v1 decimal keys; `result.reversed_bits()` reverses every key for comparison with SDKs that put qubit 0 first.

## A device

A device is physical parameters: species, trap, beams, magnetic field, noise model, detector and control electronics. Mode structure, Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering and heating rates are derived.

```python
from qutip_trap.device.presets import yb171_chain

preset = yb171_chain(2)                 # a DevicePreset; preset.machine() is the Machine above
device = preset.device
derived = device.derived()
modes_mhz = [round(m.omega_hz / 1e6, 4) for m in device.crystal.modes]
print(f"{device.crystal.n_ions} ions; modes (MHz):", modes_mhz)
print("qubit frequency of ion 0 (Hz):", round(derived.values["qubit_freq_hz[0]"]))
rabi_keys = sorted(k for k in derived.values if k.startswith("rabi_hz[(0,"))
print("carrier Rabi frequencies ion 0 sees (Hz):", {k: round(derived.values[k]) for k in rabi_keys})
assert all(derived.provenance[k] for k in derived.values)  # every derived number carries its provenance id
```

`device.specs()` renders the derived quantities as a report and `device.to_dict()` is its JSON record. `Device.roles` names which beams play which gates, so the machine, the calibration and the benchmarks need no drive maps.

## Calibrate, run, read the result

`Machine.calibrated` pins the calibration table on the machine; `calibration.calibrate(machine)` returns the whole `CalibrationReport`.

```python
import numpy as np

from qutip_trap.control.compiler import Circuit, Operation, ideal_probabilities
from qutip_trap.run.job import register_fidelity

windows = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
pinned = machine.calibrated(pairs=[(0, 1)], detection_records=2000, detection_windows_s=windows)
table = pinned.table
fast = dataclasses.replace(pinned, numerics=trap.Numerics(truncation={"branch_weight_min": 1e-3}))
bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
result = fast.run(bell, 2000, keep_final_state=True)
print("histogram:", {k: round(v, 4) for k, v in sorted(result.probabilities.items())}, "ideal:", ideal_probabilities(bell))
print("register infidelity:", f"{1 - register_fidelity(result):.2e}",
      "inside the closed-form budget", f"{result.diagnostics.intrinsic_budget['total']:.1e}")
print("level:", result.diagnostics.level, "| space dims:", result.diagnostics.space.dims,
      "| mode classes:", result.diagnostics.mode_class)
print("SPAM per qubit (eps_B, eps_D):", {k: tuple(round(x, 5) for x in v) for k, v in result.spam.items() if "." not in k})
assert result.probabilities["00"] + result.probabilities["11"] > 0.98
```

The same run in a worker process is a `Job`: `status()`, the latest `Progress` as `progress`, `result()` for the same `Result` shot for shot, `record()` for the `RunRecord` behind it, and `cancel()`; its `spec` is the JSON-serialisable `RunSpec`.

```python
job = fast.submit(bell, 200, seed=7, label="bell in the background")
print("submitted:", job.status(), "|", job.spec.to_dict()["shots"], "shots, machine", job.spec.machine_hash[:12])
background = job.result(timeout_s=600.0)
assert job.status() == "done" and background.counts == fast.run(bell, 200, seed=7).counts
assert job.record().schedule.pulses, "the RunRecord travelled back with the result"
```

## OpenQASM 2 and IonQ JSON

```python
from qutip_trap.control.compiler import compile_to_native
from qutip_trap.io.ionq import dump_ionq_json, load_ionq_json
from qutip_trap.io.openqasm import load_openqasm2

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = load_openqasm2(qasm)
native = compile_to_native(circuit)
print("native gates:", [op.name for op in native.ops])
assert load_ionq_json(dump_ionq_json(native)).ops == native.ops, "the IonQ JSON round trip is exact"
report = machine.compile(circuit)
print("pulses:", report.n_pulses, "| entangling:", report.n_entangling,
      "| whole-circuit residual:", f"{report.circuit_residual:.1e}")
```

## An experiment

Experiments run the laboratory's scans on the machine through the same engine and fit them with shot noise and readout errors; the result is typed (`RabiScan`), with the scan as requested beside the scan the electronics played. A result proposes a table update; adopt it only past the quality test.

```python
from qutip_trap.experiments.single_ion import rabi_scan

durations = tuple(float(t) for t in np.linspace(0.0, 30e-6, 13))
scan = rabi_scan(pinned, 0, durations, shots=200)
print("fitted carrier Rabi frequency (Hz):", round(scan.value("f_rabi_hz")), "+-", round(scan.uncertainty("f_rabi_hz")),
      "| converged:", scan.converged, "| quality:", scan.quality)
assert scan.f_rabi_hz == scan.value("f_rabi_hz") and scan.requested.durations_s == scan.realized.durations_s
proposal = table.updated_with(scan)
assert proposal.rabi[(0, 2)].experiment == "rabi_scan" and proposal.kind_of("rabi[(0, 2)]") == "setpoint"
```

## Benchmarks with the simulator's own budget

Each benchmark reports, beside the measured number, what the simulator's physics accounts for: closed-form error scales, the GATE_LOCAL channel of every native gate kind, and SPAM. Single-qubit randomized benchmarking fits the survival to A p^m + B, with r = (1 - p)/2 per Clifford; `pair=False` on two ions runs simultaneous RB (each ion's own marginal decay), and `variant="knill"` Knill-style sequences.

```python
from qutip_trap.benchmarks.rb import randomized_benchmarking

rb = randomized_benchmarking(fast, (0,), (1, 128, 512), n_sequences=1, shots=2000, fix_offset=True)
print(rb.fidelity_form(), "| r per Clifford:", f"{rb.error_per_clifford[0]:.1e} +- {rb.error_per_clifford[1]:.1e}")
budget = rb.budget
print("channel infidelity per native kind (reduced to qubit 0):", {k: f"{v:.1e}" for k, v in budget.channel_infidelity.items()})
print("predicted r from the channels:", f"{budget.predicted['r_channel']:.1e}",
      "| closed-form scales per Clifford:", f"{budget.predicted['r_intrinsic']:.1e}",
      "| F(0) from SPAM:", f"{budget.predicted['F0_spam']:.4f}")

sim = randomized_benchmarking(fast, (0, 1), (1, 64), n_sequences=1, shots=400, pair=False, fix_offset=True, budget=False)
print("simultaneous RB, per-ion marginal r_q:", [f"{v:.1e}" for v, _ in sim.marginal_error_per_clifford],
      "| the joint decay per layer:", f"{sim.joint_error_per_layer[0]:.1e}")
knill = randomized_benchmarking(fast, (0,), (1, 64), n_sequences=1, shots=400, variant="knill", budget=False)
print("Knill-style RB:", knill.fidelity_form(), "| r per computational gate:", f"{knill.error_per_clifford[0]:.1e}")
```

GHZ fidelity reports the laboratory's bound (P_0 + P_1 + C)/2, which equals the phase-optimised fidelity max_theta <GHZ_theta|rho|GHZ_theta> and so sits above the fixed-phase fidelity; the simulator reports both exact numbers beside it. Quantum volume applies the Cross et al. 2019 criterion, mean - 2 sigma > 2/3 over at least 100 circuits.

```python
from qutip_trap.benchmarks.ghz import ghz_fidelity
from qutip_trap.benchmarks.volume import quantum_volume

ghz = ghz_fidelity(fast, (0, 1), shots=400, analysis_phases_rad=np.linspace(0.0, np.pi, 4, endpoint=False))
print("P00, P11:", {k: round(v[0], 4) for k, v in ghz.populations.items()},
      "| parity contrast:", f"{ghz.fit['contrast'][0]:.4f}",
      "| bound (P0 + P1 + C)/2:", f"{ghz.fidelity_bound[0]:.4f} +- {ghz.fidelity_bound[1]:.4f}",
      "| exact max-phase:", f"{ghz.register_fidelity_max_phase:.5f}",
      "| exact fixed-phase:", f"{ghz.register_fidelity:.5f}")
assert ghz.register_fidelity_max_phase >= ghz.register_fidelity - 1e-12
qv = quantum_volume(fast, (0, 1), n_circuits=1, shots=200)
print("heavy-output probability:", qv.heavy_output_probability.round(3), "ideal:", qv.ideal_heavy_probability.round(3),
      "| clears 2/3 by two sigma:", qv.threshold_cleared, "| passes the protocol:", qv.passed)
```

## A gate's channel and the error model

`gate_channel` gives the summary of one native gate kind: the gate played once on its exact gate-local space, its Choi matrix, average gate infidelity, depolarizing rate and Pauli twirl. `Machine.error_model()` derives per-gate infidelities, durations and SPAM from those channels and exports them in IonQ's, Quantinuum's and the QDK estimator's vocabularies (IonQ's r is the maximally-mixed weight, so `r_1q = 2 r` for the average gate infidelity r on one qubit).

```python
from qutip_trap.benchmarks.budget import gate_channel

ch = gate_channel(fast, "gpi2[0]")
step = ch.steps[0]
print("step ions:", step.ions, "| full-step average infidelity:", f"{step.summary.average_gate_infidelity:.1e}",
      "| reduced to qubit 0:", f"{ch.infidelity_on((0,)):.1e}", "| twirl p_II:", f"{step.summary.pauli_twirled['II']:.5f}")
em = fast.error_model()
print("average gate infidelity per kind:", {k: f"{v:.1e}" for k, v in em.infidelity.items()})
print("IonQ:", {k: f"{v:.2e}" for k, v in em.to_ionq_noise().items()},
      "| QDK:", em.to_qdk_qubit_params()["oneQubitGateTime"], em.to_qdk_qubit_params()["twoQubitGateTime"])
assert em.to_ionq_noise()["r_1q"] == 2.0 * sum(em.infidelity[k] for k in em.single_qubit_kinds) / len(em.single_qubit_kinds)
```

## A second species: the 40Ca+ optical qubit

`ca40_optical` drives the 729 nm S1/2-D5/2 quadrupole line, prepares with 397 nm Doppler cooling (866 nm repumper) and optical pumping, and reads out by shelving through a PMT. It has no entangling drive, so single-qubit circuits run and a two-qubit gate is refused.

```python
from qutip_trap.device.presets import ca40_optical

ca = ca40_optical(1)
ca_derived = ca.device.derived()
print("729 nm carrier Rabi frequency (Hz):",
      {k: round(ca_derived.values[k]) for k in sorted(ca_derived.values) if k.startswith("rabi_hz[")})
ca_result = ca.machine().run(Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,)), 200)
print("one GPi2 on the optical qubit:", {k: round(v, 3) for k, v in sorted(ca_result.probabilities.items())})
```
