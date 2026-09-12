# Examples

Runnable, in order: every ```python block below builds on the ones before it, and `tests/test_docs.py` executes them in
one namespace (about three minutes on the reference machine, an 18-core laptop with Apple Accelerate BLAS, whose wall times
are the ones quoted). The public surface is `qutip_trap.api` (Appendix E of the plan); frequencies are in Hz there.

## A device

A device is physical parameters: species, trap frequencies, beams (wavelength, direction, polarization, waist, power,
pointing), the magnetic field, the noise model, the detector and the control electronics. Everything else (mode structure,
Lamb-Dicke parameters, Rabi frequencies, Stark shifts, scattering rates, heating rates, pulse parameters) is derived. The
package ships the two-ion 171Yb+ example device the validation suite runs on: a global 355 nm Raman pair for the entangling
gates, one addressing pair per ion (2.2 % Rabi crosstalk on the neighbour from the 2.5 µm waist), the 369.5 nm cooling and
detection light, Crain's SNSPD detector, a quiet noise model and near-ideal electronics.

```python
from qutip_trap.api import yb171_chain

preset = yb171_chain(2)
device = preset.device
derived = device.derived()
modes_mhz = [round(m.omega_hz / 1e6, 4) for m in device.crystal.modes]
print(f"{device.crystal.n_ions} ions; modes (MHz):", modes_mhz)
print("qubit frequency of ion 0 (Hz):", round(derived.values["qubit_freq_hz[0]"]))
rabi_keys = sorted(k for k in derived.values if k.startswith("rabi_hz[(0,"))
print("carrier Rabi frequencies ion 0 sees (Hz):", {k: round(derived.values[k]) for k in rabi_keys})
assert all(derived.provenance[k] for k in derived.values)  # every derived number names its ledger record
```

Every derived number carries the id of its provenance record in `docs/provenance/ledger.yaml`. The device carries which
beams play which gates as `Device.roles` (the addressing pair of each ion for the single-qubit gates, the global pair for
the entangling gates, the detection beam), so `run`, `calibrate` and the benchmarks need no drive maps; a device with one
Raman pair infers its roles from the wavelengths.

## Calibrate, run a circuit, read the result

`run` compiles the circuit to native gates (verified against the target unitary), calibrates the device (the closed-form
surrogate with exact spot checks, cached per device), schedules pulses from the calibration table, prepares the ions
(Doppler cooling, pulsed sideband cooling, optical pumping), integrates every pulse on the joint space of the ions and the
resolved motional modes, and reads out through the fluorescence model.

```python
import numpy as np

from qutip_trap.api import Circuit, Operation, SolverOptions, calibrate, ideal_probabilities, register_fidelity, run

windows = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
table = calibrate(device, pairs=[(0, 1)], detection_records=2000, detection_windows_s=windows)
options = SolverOptions(branch_weight_min=1e-3)
bell = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
result = run(bell, device, 2000, table=table, keep_final_state=True, options=options)
print("histogram:", {k: round(v, 4) for k, v in sorted(result.probabilities.items())}, "ideal:", ideal_probabilities(bell))
print("register infidelity:", f"{1 - register_fidelity(result):.2e}",
      "inside the closed-form budget", f"{result.diagnostics.intrinsic_budget['total']:.1e}")
print("level:", result.diagnostics.level, "| space dims:", result.diagnostics.space.dims,
      "| mode classes:", result.diagnostics.mode_class)
print("SPAM per qubit (eps_B, eps_D):", {k: tuple(round(x, 5) for x in v) for k, v in result.spam.items() if "." not in k})
assert result.probabilities["00"] + result.probabilities["11"] > 0.98
```

On the reference machine the surrogate takes about 15 s and the Bell run about 8 s: five carrier pulses and one 100 µs
Mølmer-Sørensen pulse on the 572-dimensional space [2, 2, 11, 13] (two ions, the two x modes resolved with 11 and 13 Fock
levels, the other four modes dropped by the contribution criterion), giving a histogram within the readout errors of
0.5/0.5 and a register infidelity of about 2 × 10⁻³ against the compiled circuit's state. `result.diagnostics` records the
level that ran, the space, the mode classes, the boundary populations, the branch cutoff, the integrators, the seeds and
every approximation made; `result.to_ionq_json()` gives the IonQ probability format (decimal keys, qubit 0 least
significant).

## A circuit from OpenQASM 2 or IonQ JSON

```python
from qutip_trap.api import compile_to_native, compile_with_report, dump_ionq_json, load_ionq_json, load_openqasm2

qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = load_openqasm2(qasm)
native = compile_to_native(circuit)
print("native gates:", [op.name for op in native.ops])
as_json = dump_ionq_json(native)
assert load_ionq_json(as_json).ops == native.ops, "the IonQ JSON round trip is exact"
report = compile_with_report(circuit)
print("pulses:", report.n_pulses, "| entangling:", report.n_entangling,
      "| whole-circuit residual:", f"{report.circuit_residual:.1e}")
```

## An experiment the calibration runs

The experiment API runs the laboratory's scans on the simulated device through the same engine and fits them the way the
laboratory does, with shot noise and readout errors declared:

```python
from qutip_trap.api import rabi_scan

durations = tuple(float(t) for t in np.linspace(0.0, 30e-6, 13))
scan = rabi_scan(device, 0, durations, gate_drive=preset.gate_drives[0], shots=200, table=table)
print("fitted carrier Rabi frequency (Hz):", round(scan.value("f_rabi_hz")), "+-", round(scan.uncertainty("f_rabi_hz")),
      "| converged:", scan.converged)
```

## Benchmarks with the simulator's own budget

The benchmarks run the standard protocols on the device through `run` and report, beside the measured number, what the
simulator's own physics accounts for: the closed-form error scales of every pulse, the channel of every native gate kind by
process tomography, and the SPAM errors.

Single-qubit randomized benchmarking: random Cliffords and the inverse of their product, the survival fitted to A p^m + B,
r = (1 − p)/2 the error per Clifford (Section 13 of the plan).

```python
from qutip_trap.api import randomized_benchmarking

rb = randomized_benchmarking(device, (0,), (1, 128, 512), n_sequences=1, shots=2000, fix_offset=True,
                             table=table, options=options)
print(rb.fidelity_form(), "| r per Clifford:", f"{rb.error_per_clifford[0]:.1e} +- {rb.error_per_clifford[1]:.1e}")
budget = rb.budget
print("channel infidelity per native kind (reduced to qubit 0):", {k: f"{v:.1e}" for k, v in budget.channel_infidelity.items()})
print("predicted r from the channels:", f"{budget.predicted['r_channel']:.1e}",
      "| Section 9.6 scales per Clifford (a bound over a wider scope):", f"{budget.predicted['r_intrinsic']:.1e}",
      "| F(0) from SPAM:", f"{budget.predicted['F0_spam']:.4f}")

sim = randomized_benchmarking(device, (0, 1), (1, 64), n_sequences=1, shots=400, pair=False, fix_offset=True,
                              budget=False, table=table, options=options)
print("simultaneous RB, per-ion marginal r_q:", [f"{v:.1e}" for v, _ in sim.marginal_error_per_clifford],
      "| mean:", f"{sim.error_per_clifford[0]:.1e}",
      "| the joint decay per layer of two Cliffords:", f"{sim.joint_error_per_layer[0]:.1e}")
knill = randomized_benchmarking(device, (0,), (1, 64), n_sequences=1, shots=400, variant="knill",
                                budget=False, table=table, options=options)
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
from qutip_trap.api import ghz_fidelity, quantum_volume

ghz = ghz_fidelity(device, (0, 1), shots=400, analysis_phases_rad=np.linspace(0.0, np.pi, 4, endpoint=False),
                   table=table, options=options)
print("P00, P11:", {k: round(v[0], 4) for k, v in ghz.populations.items()},
      "| parity contrast:", f"{ghz.fit['contrast'][0]:.4f}",
      "| bound (P0 + P1 + C)/2:", f"{ghz.fidelity_bound[0]:.4f} +- {ghz.fidelity_bound[1]:.4f}",
      "| exact max-phase (what the bound estimates):", f"{ghz.register_fidelity_max_phase:.5f}",
      "| exact fixed-phase (which it exceeds):", f"{ghz.register_fidelity:.5f}")
assert ghz.register_fidelity_max_phase >= ghz.register_fidelity - 1e-12
qv = quantum_volume(device, (0, 1), n_circuits=1, shots=200, table=table, options=options)
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
from qutip_trap.api import gate_channel

ch = gate_channel(device, "gpi2[0]", table=table, options=options)
step = ch.steps[0]
print("step ions:", step.ions, "| full-step average infidelity:", f"{step.summary.average_gate_infidelity:.1e}",
      "| reduced to qubit 0:", f"{ch.infidelity_on((0,)):.1e}", "| twirl p_II:", f"{step.summary.pauli_twirled['II']:.5f}")
```

The full step's infidelity counts the crosstalk rotation of the neighbour (about 3 × 10⁻⁴ from a 2.2 % Rabi ratio on a
π/2 pulse); reduced to the addressed ion it drops to the off-resonant sideband scale of a few 10⁻⁵.

## A second species: the 40Ca+ optical qubit

`ca40_optical` drives the 729 nm S1/2-D5/2 quadrupole line, so the qubit is optical, the preparation is
`ca40_optical_recipe` (397 nm Doppler cooling with the 866 nm repumper, then optical pumping) and the readout is shelving
detection through Myerson's PMT chain. It carries no entangling drive, so single-qubit circuits run and a two-qubit gate is
refused by the scheduler.

```python
from qutip_trap.api import ca40_optical

ca = ca40_optical(1)
ca_derived = ca.device.derived()
print("729 nm carrier Rabi frequency (Hz):",
      {k: round(ca_derived.values[k]) for k in sorted(ca_derived.values) if k.startswith("rabi_hz[")})
ca_result = run(Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,)), ca.device, 200)
print("one GPi2 on the optical qubit:", {k: round(v, 3) for k, v in sorted(ca_result.probabilities.items())})
```

The derived 729 nm Rabi frequency is 34.7 kHz at 5 mW in a 200 um waist with B = 5 G along x (a beam along y with x
polarization would derive exactly zero E2 coupling, which `control/played.py` refuses rather than plays), and the GPi2 on
`|S1/2, mJ = -1/2>` lands near 1/2 with the 40Ca+ P1/2 rate read as Hettrich's partial-rate convention
(`conv.ca40_linewidth_reading`).
