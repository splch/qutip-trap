# Examples

Runnable, in order: every ```python block below builds on the ones before it, and `tests/test_docs.py` executes them in
one namespace (about three minutes on the reference machine, an 18-core laptop with Apple Accelerate BLAS, whose wall times
are the ones quoted). Since 0.2.0 the front door is the `Machine` of the root namespace (`import qutip_trap as trap`); the
frozen Appendix E surface stays `qutip_trap.api`, and everything below the first section is written on it. Frequencies are
in Hz on both.

## Two minutes to a histogram

A `Machine` is a device with the roles its beams play (`BeamRoles`), the calibration it runs on and the policy that turns a
circuit into a `Result`: `Physics` (which effects are simulated), `Numerics` (how the integration is done), `Readout` (how the
record is read) and the `FidelityLevel`. The example machines come from `trap.presets`; a circuit is built with one method per
gate.

```python
import dataclasses

import qutip_trap as trap

machine = trap.presets.yb171_chain(2)                # a Machine on the example two-ion 171Yb+ device
bell = trap.Circuit(2).h(0).cnot(0, 1)               # every qubit is measured unless .measured(...) narrows it
quick = dataclasses.replace(machine, numerics=trap.Numerics(truncation={"branch_weight_min": 1e-3}))
result = quick.run(bell, shots=2000)                 # compile, calibrate (cached by device hash), schedule, prepare, evolve, read
print("histogram:", {k: round(v, 4) for k, v in sorted(result.probabilities.items())})
print("level:", result.diagnostics.level, "|", result.diagnostics.level_reason)
print("machine:", result.machine_hash[:12], "| took", f"{result.duration_s:.1f} s")
assert result.probabilities["00"] + result.probabilities["11"] > 0.98
estimate = quick.estimate(bell)                      # the level, the space and a wall-time guess before anything is integrated
assert estimate.level == result.diagnostics.level and estimate.space == result.diagnostics.space
deeper = dataclasses.replace(quick, level=trap.FidelityLevel.GATE_LOCAL)   # the app's "verify deeper"
quiet = dataclasses.replace(quick, physics=trap.Physics(noise=False))      # the nominal sample, no channels
record = result.to_dict()                            # the versioned record of docs/schemas/result.schema.json
assert trap.Result.from_dict(record).counts == result.counts
```

`quick.run(..., progress=print)` reports a `Progress` per pulse, branch, sample and readout. `result.to_ionq_v2_probabilities()`
is IonQ's v0.4 envelope (bitstrings in wire order, q[0] first) and `result.to_ionq_v1_probabilities()` the v1 decimal keys;
`result.reversed_bits()` reverses every key (its first character is then qubit 0's bit) for comparisons with Cirq, Braket
and PennyLane, which report that way. `trap.__version__` is the release the record carries.

## The ladder

The five levels of PLAN.md Section 14.2 are five modules, each a rung of the same machine (docs/conventions.md, "Vocabulary"),
and since 0.4.0 each has its own page listing every public name: [machine.md](machine.md) for `qutip_trap` (rung 0: the
`Machine`, the option objects, `Circuit`, `Result`, the jobs and the presets), [circuit.md](circuit.md) for `qutip_trap.circuit`
and the wire formats of `qutip_trap.io`, [schedule.md](schedule.md) for `qutip_trap.schedule`, [dynamics.md](dynamics.md) for
`qutip_trap.dynamics`, [physics.md](physics.md) for `qutip_trap.physics`, [laboratory.md](laboratory.md) for the experiments,
the calibration and the benchmarks, and [experimental.md](experimental.md) for `qutip_trap.experimental`. The examples below
walk the rungs in that order on the example device; every name they use is on those pages, and the same objects are what
`qutip_trap.api` exports under the Appendix E names.

## A device

A device is physical parameters: species, trap frequencies, beams (wavelength, direction, polarization, waist, power,
pointing), the magnetic field, the noise model, the detector and the control electronics. Everything else (mode structure,
Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering rates, heating rates, pulse parameters) is derived. The
package ships the two-ion 171Yb+ example device the validation suite runs on: a global 355 nm Raman pair for the entangling
gates, one addressing pair per ion (2.2 % Rabi crosstalk on the neighbour from the 2.5 µm waist), the 369.5 nm cooling and
detection light, Crain's SNSPD detector, a quiet noise model and near-ideal electronics.

```python
from qutip_trap.device.presets import yb171_chain

preset = yb171_chain(2)                 # the DevicePreset of the 0.1.0 surface; preset.machine() is the Machine above
device = preset.device
derived = device.derived()
modes_mhz = [round(m.omega_hz / 1e6, 4) for m in device.crystal.modes]
print(f"{device.crystal.n_ions} ions; modes (MHz):", modes_mhz)
print("qubit frequency of ion 0 (Hz):", round(derived.values["qubit_freq_hz[0]"]))
rabi_keys = sorted(k for k in derived.values if k.startswith("rabi_hz[(0,"))
print("carrier Rabi frequencies ion 0 sees (Hz):", {k: round(derived.values[k]) for k in rabi_keys})
assert all(derived.provenance[k] for k in derived.values)  # every derived number names its ledger record
```

Every derived number carries the id of its provenance record in `docs/provenance/ledger.yaml`; `device.specs()` renders
them as a report and `device.to_dict()` is the JSON record of `docs/schemas/device.schema.json`. The device carries which
beams play which gates as `Device.roles` (the addressing pair of each ion for the single-qubit gates, the global pair for
the entangling gates, the detection beam), so the machine, the calibration and the benchmarks need no drive maps; a device
with one Raman pair infers its roles from the wavelengths.

## Calibrate, run a circuit, read the result

`Machine.run` compiles the circuit to native gates (verified against the target unitary), calibrates the device (the
closed-form surrogate with exact spot checks, cached per device), schedules pulses from the calibration table, prepares the
ions (Doppler cooling, pulsed sideband cooling, optical pumping), integrates every pulse on the joint space of the ions and
the resolved motional modes, and reads out through the fluorescence model. `Machine.calibrated` pins the table on the
machine so that every later call reads the same one; `calibrate(machine)` returns the whole `CalibrationReport`.

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

The same run submitted to a worker process is a `Job` (0.4.0): `status()`, the latest `Progress` as `progress`,
`result()` blocking for the same `Result` shot for shot, `record()` for the `RunRecord` behind it, and `cancel()`; its
`spec` is the `RunSpec` a JSON document can carry.

```python
job = fast.submit(bell, 200, seed=7, label="bell in the background")
print("submitted:", job.status(), "|", job.spec.to_dict()["shots"], "shots, machine", job.spec.machine_hash[:12])
background = job.result(timeout_s=600.0)
assert job.status() == "done" and background.counts == fast.run(bell, 200, seed=7).counts
assert job.record().schedule.pulses, "the RunRecord travelled back with the result"
print("the job's histogram:", {k: round(v, 3) for k, v in sorted(background.probabilities.items())})
```

On the reference machine the surrogate takes about 15 s and the Bell run about 8 s: five carrier pulses and one 100 µs
Mølmer-Sørensen pulse on the 572-dimensional space [2, 2, 11, 13] (two ions, the two x modes resolved with 11 and 13 Fock
levels, the other four modes dropped by the contribution criterion), giving a histogram within the readout errors of
0.5/0.5 and a register infidelity of about 2 × 10⁻³ against the compiled circuit's state. `result.diagnostics` records the
level that ran, the space, the mode classes, the boundary populations, the branch cutoff, the integrators, the seeds and
every approximation made; `result.to_ionq_v1_probabilities()` gives IonQ's v1 probability format (decimal keys, qubit 0
the least-significant bit) and `result.to_ionq_v2_probabilities()` the v0.4 envelope (bitstrings in wire order, q[0] first);
`result.to_dict()` is the versioned record of `docs/schemas/result.schema.json`.

## A circuit from OpenQASM 2 or IonQ JSON

```python
from qutip_trap.control.compiler import compile_to_native
from qutip_trap.io.ionq import dump_ionq_json, load_ionq_json
from qutip_trap.io.openqasm import load_openqasm2

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = load_openqasm2(qasm)
native = compile_to_native(circuit)
print("native gates:", [op.name for op in native.ops])
as_json = dump_ionq_json(native)
assert load_ionq_json(as_json).ops == native.ops, "the IonQ JSON round trip is exact"
report = machine.compile(circuit)
print("pulses:", report.n_pulses, "| entangling:", report.n_entangling,
      "| whole-circuit residual:", f"{report.circuit_residual:.1e}")
```

## An experiment the calibration runs

The experiment API runs the laboratory's scans on the machine through the same engine and fits them the way the
laboratory does, with shot noise and readout errors declared; the machine supplies its table and options, the device's
roles the drive, and the result is typed (`RabiScan`) with the scan as requested beside the scan the electronics played:

```python
from qutip_trap.experiments.single_ion import rabi_scan

durations = tuple(float(t) for t in np.linspace(0.0, 30e-6, 13))
scan = rabi_scan(pinned, 0, durations, shots=200)
print("fitted carrier Rabi frequency (Hz):", round(scan.value("f_rabi_hz")), "+-", round(scan.uncertainty("f_rabi_hz")),
      "| converged:", scan.converged, "| quality:", scan.quality)
assert scan.f_rabi_hz == scan.value("f_rabi_hz") and scan.requested.durations_s == scan.realized.durations_s
proposal = table.updated_with(scan)              # an update is a proposal: adopt it only past the quality test
assert proposal.rabi[(0, 2)].experiment == "rabi_scan" and proposal.kind_of("rabi[(0, 2)]") == "setpoint"
```

## Benchmarks with the simulator's own budget

The benchmarks run the standard protocols on the machine through `Machine.run` and report, beside the measured number, what
the simulator's own physics accounts for: the closed-form error scales of every pulse, the channel of every native gate kind by
process tomography, and the SPAM errors.

Single-qubit randomized benchmarking: random Cliffords and the inverse of their product, the survival fitted to A p^m + B,
r = (1 − p)/2 the error per Clifford (Section 13 of the plan).

```python
from qutip_trap.benchmarks.rb import randomized_benchmarking

rb = randomized_benchmarking(fast, (0,), (1, 128, 512), n_sequences=1, shots=2000, fix_offset=True)
print(rb.fidelity_form(), "| r per Clifford:", f"{rb.error_per_clifford[0]:.1e} +- {rb.error_per_clifford[1]:.1e}")
budget = rb.budget
print("channel infidelity per native kind (reduced to qubit 0):", {k: f"{v:.1e}" for k, v in budget.channel_infidelity.items()})
print("predicted r from the channels:", f"{budget.predicted['r_channel']:.1e}",
      "| Section 9.6 scales per Clifford (a bound over a wider scope):", f"{budget.predicted['r_intrinsic']:.1e}",
      "| F(0) from SPAM:", f"{budget.predicted['F0_spam']:.4f}")

sim = randomized_benchmarking(fast, (0, 1), (1, 64), n_sequences=1, shots=400, pair=False, fix_offset=True, budget=False)
print("simultaneous RB, per-ion marginal r_q:", [f"{v:.1e}" for v, _ in sim.marginal_error_per_clifford],
      "| mean:", f"{sim.error_per_clifford[0]:.1e}",
      "| the joint decay per layer of two Cliffords:", f"{sim.joint_error_per_layer[0]:.1e}")
knill = randomized_benchmarking(fast, (0,), (1, 64), n_sequences=1, shots=400, variant="knill", budget=False)
print("Knill-style RB (Section 7.9):", knill.fidelity_form(),
      "| r per computational gate:", f"{knill.error_per_clifford[0]:.1e}")
```

On the example device the survival falls from 0.9997 at one Clifford to about 0.96 at 2048, an error per Clifford near
2 × 10⁻⁵ against 2.5 × 10⁻⁵ predicted by composing the gpi2 and gpi channels (coherent errors within a Clifford partly
cancel, which the first-order composition does not know). Simultaneous RB on several ions at once (`pair=False`) exposes
the addressing crosstalk that single-ion RB cannot see. Each ion's own marginal survival is fitted (Gambetta et al. 2012):
on the two-ion device the marginals give r_q = 1.3 × 10⁻⁴ and 3.9 × 10⁻⁴, a mean thirteen times the 2 × 10⁻⁵ the same ion
measures alone, and the composition of each gate kind's channel *reduced to that one qubit* predicts that mean to 9 %
(2.86 × 10⁻⁴ against 2.59 × 10⁻⁴). `error_per_clifford` is the mean of the marginals, in the same unit as the budget's
`r_channel`; `joint_error_per_layer` carries the joint P(0…0) decay — 6.9 × 10⁻⁴ per layer of one Clifford per ion,
thirty-five times the isolated r, and within 1 % of `r_channel_joint_layer` — as the correlation diagnostic. Two-qubit RB
(`qubits=(0, 1)`) draws from the 11520-element group at 1.5 entangling gates per Clifford, and `variant="knill"` runs
Section 7.9's Knill-style sequences instead (π/2 pulses with an interleaved π Pauli or an identity and one final π/2,
fitted as B p^L + ½; the Pauli randomization twirls the coherent errors, so its r per computational gate, 8.5 × 10⁻⁶, sits
below the Clifford r despite costing 1.56 pulses per gate against 0.83). GHZ fidelity and a quantum-volume style run
follow the same pattern:

```python
from qutip_trap.benchmarks.ghz import ghz_fidelity
from qutip_trap.benchmarks.volume import quantum_volume

ghz = ghz_fidelity(fast, (0, 1), shots=400, analysis_phases_rad=np.linspace(0.0, np.pi, 4, endpoint=False))
print("P00, P11:", {k: round(v[0], 4) for k, v in ghz.populations.items()},
      "| parity contrast:", f"{ghz.fit['contrast'][0]:.4f}",
      "| bound (P0 + P1 + C)/2:", f"{ghz.fidelity_bound[0]:.4f} +- {ghz.fidelity_bound[1]:.4f}",
      "| exact max-phase (what the bound estimates):", f"{ghz.register_fidelity_max_phase:.5f}",
      "| exact fixed-phase (which it exceeds):", f"{ghz.register_fidelity:.5f}")
assert ghz.register_fidelity_max_phase >= ghz.register_fidelity - 1e-12
qv = quantum_volume(fast, (0, 1), n_circuits=1, shots=200)
print("heavy-output probability:", qv.heavy_output_probability.round(3), "ideal:", qv.ideal_heavy_probability.round(3),
      "| exact register fidelity:", qv.register_fidelity.round(4),
      "| Eq. (32) sigma:", f"{qv.sigma:.3f}", "| clears 2/3 by two sigma:", qv.threshold_cleared,
      "| passes the protocol:", qv.passed)
print(qv.notes[0])
```

A two-ion GHZ run (populations plus a parity scan, five runs) takes about 40 s and returns a bound of 0.9975 ± 0.0112 at
1000 shots in the check script. That number is exactly max_θ ⟨GHZ_θ|ρ|GHZ_θ⟩, which the simulator reads off the same
register state as 0.99808 — they agree to 0.0006 — and it therefore sits *above* the fixed-phase ⟨GHZ|ρ|GHZ⟩ of 0.99762
rather than below it; on three ions the gap is larger (bound 0.9939 against max_θ 0.99227 and fixed-phase 0.98508). One
quantum-volume circuit (two Haar-random SU(4) layers, six entangling gates) takes about 45 s, with a heavy-output
probability within shot noise of its ideal value and an exact register fidelity of 0.99; four circuits give mean 0.7163
and Eq. (32)'s σ = 0.2254, so `threshold_cleared` is False, and even a run that cleared it would report `passed = False`
below the protocol's hundred circuits.

## The error budget's channels

`gate_channel` gives the Section 6.8 summary of one native gate kind on its own: the gate played once from the prepared
motional state on the exact gate-local space, its Choi matrix, average gate infidelity, depolarizing rate and Pauli twirl.

```python
from qutip_trap.benchmarks.budget import gate_channel

ch = gate_channel(fast, "gpi2[0]")
step = ch.steps[0]
print("step ions:", step.ions, "| full-step average infidelity:", f"{step.summary.average_gate_infidelity:.1e}",
      "| reduced to qubit 0:", f"{ch.infidelity_on((0,)):.1e}", "| twirl p_II:", f"{step.summary.pauli_twirled['II']:.5f}")
```

The full step's infidelity counts the crosstalk rotation of the neighbour (about 3 × 10⁻⁴ from a 2.2 % Rabi ratio on a
π/2 pulse); reduced to the addressed ion it drops to the off-resonant sideband scale of a few 10⁻⁵.

## The inverse direction: the error model

Every vendor emulator takes phenomenological numbers and none derives them. `Machine.error_model()` emits them from the
simulated device: one GATE_LOCAL tomography per native gate kind and qubit (cached with the channels above), the SPAM from
the detection calibration and the preparation recipe, the durations from the schedule, with exporters to IonQ's, Quantinuum's
and the QDK estimator's vocabularies and the conversion stated on each (IonQ's `r` is the maximally-mixed weight,
`F_avg = 1 - r/2` on one qubit and `1 - 3r/4` on two, so `r_1q = 2 r` for the average gate infidelity `r`).

```python
em = fast.error_model()
print("average gate infidelity per kind:", {k: f"{v:.1e}" for k, v in em.infidelity.items()})
print("IonQ:", {k: f"{v:.2e}" for k, v in em.to_ionq_noise().items()},
      "| QDK:", em.to_qdk_qubit_params()["oneQubitGateTime"], em.to_qdk_qubit_params()["twoQubitGateTime"])
assert em.to_ionq_noise()["r_1q"] == 2.0 * sum(em.infidelity[k] for k in em.single_qubit_kinds) / len(em.single_qubit_kinds)
```

## A second species: the 40Ca+ optical qubit

`ca40_optical` drives the 729 nm S1/2-D5/2 quadrupole line, so the qubit is optical, the preparation is
`ca40_optical_recipe` (397 nm Doppler cooling with the 866 nm repumper, then optical pumping) and the readout is shelving
detection through Myerson's PMT chain. It carries no entangling drive, so single-qubit circuits run and a two-qubit gate is
refused by the scheduler.

```python
from qutip_trap.device.presets import ca40_optical

ca = ca40_optical(1)
ca_derived = ca.device.derived()
print("729 nm carrier Rabi frequency (Hz):",
      {k: round(ca_derived.values[k]) for k in sorted(ca_derived.values) if k.startswith("rabi_hz[")})
ca_result = ca.machine().run(Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,)), 200)
print("one GPi2 on the optical qubit:", {k: round(v, 3) for k, v in sorted(ca_result.probabilities.items())})
```

The derived 729 nm Rabi frequency is 34.7 kHz at 5 mW in a 200 um waist with B = 5 G along x (a beam along y with x
polarization would derive exactly zero E2 coupling, which `control/played.py` refuses rather than plays), and the GPi2 on
`|S1/2, mJ = -1/2>` lands near 1/2 with the 40Ca+ P1/2 rate read as Hettrich's partial-rate convention
(`conv.ca40_linewidth_reading`).
