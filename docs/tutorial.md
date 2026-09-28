# Tutorial: a trapped-ion quantum computer in Python

qutip-trap simulates a trapped-ion quantum computer from its physics: the trap, the ions' motion, laser cooling, the laser
pulses that play every gate, the noise and the fluorescence readout. You give it a circuit and it returns counts the way a
trapped-ion QPU does, with a record of everything it did and approximated. This tutorial takes you from a first run to the
pulse engine in one sitting of about an hour.

It is written for readers who know Python and basic quantum computing (circuits, measurement, density matrices). No
trapped-ion background is assumed: where the physics changes what you should do, the tutorial says so in a sentence.

**What you will be able to do by the end**

1. Run a circuit on a simulated trapped-ion machine and read its counts in the right bit order.
2. Say from a result what the run did and approximated, and compare its register state with the ideal one correctly.
3. Predict and control what a run costs: calibration, fidelity levels, branches and noise ensembles.
4. Add physical noise in the right units, build your own device, and run laboratory experiments and benchmarks on it.
5. Bring circuits in from OpenQASM, IonQ JSON and Qiskit, run jobs in the background, and drop to the pulse schedule and
   the engine.

**How to use it.** Run the code blocks in order in one Python session or notebook; later blocks use names defined by
earlier ones. The test suite executes them exactly as printed (`tests/test_docs.py`), so they run. Every section ends
with *Check yourself*: answer the questions before you open the answers. Section 14 mixes questions from every section,
Section 15 is a final task to do without looking back, and Section 16 says when to come back.

Frequencies are in Hz, times in seconds and angles in radians throughout, except IonQ's turns at the IonQ boundary.

**Contents**

1. [Setup](#1-setup)
2. [A first run](#2-a-first-run)
3. [Circuits and the conventions that bite](#3-circuits-and-the-conventions-that-bite)
4. [Reading a result](#4-reading-a-result)
5. [Cost, calibration and fidelity levels](#5-cost-calibration-and-fidelity-levels)
6. [Variants: the option objects](#6-variants-the-option-objects)
7. [Adding noise](#7-adding-noise)
8. [Devices](#8-devices)
9. [The laboratory](#9-the-laboratory)
10. [Benchmarks and the error model](#10-benchmarks-and-the-error-model)
11. [Other front doors: OpenQASM, IonQ, Qiskit, background jobs](#11-other-front-doors-openqasm-ionq-qiskit-background-jobs)
12. [Going lower: the schedule and the engine](#12-going-lower-the-schedule-and-the-engine)
13. [Troubleshooting](#13-troubleshooting)
14. [Mixed practice](#14-mixed-practice)
15. [Final task](#15-final-task)
16. [Review plan and where to go next](#16-review-plan-and-where-to-go-next)

## 1. Setup

qutip-trap needs Python 3.13 and [uv](https://docs.astral.sh/uv/). It is not on PyPI yet:

```sh
uv add git+https://github.com/splch/qutip-trap                 # the core
uv add "qutip-trap[qiskit] @ git+https://github.com/splch/qutip-trap"   # with the Qiskit backend
```

From a checkout, `uv sync --group dev` installs everything this tutorial uses, Qiskit included.

The package exports nine names at the top level. Everything else is imported from the module that defines it, and
[api.md](api.md) lists every public name by module.

```python
import dataclasses
import math
import pickle

import numpy as np

import qutip_trap as trap

print(trap.__version__, trap.__all__)
```

## 2. A first run

A **machine** is a trapped-ion computer as a client sees it. The package ships two example machines as presets. Take a
two-ion 171Yb+ chain and run a Bell circuit on it:

```python
machine = trap.presets.yb171_chain(2)      # two 171Yb+ ions, 355 nm Raman gates, fluorescence readout
bell = trap.Circuit(2).h(0).cnot(0, 1)     # every qubit is measured at the end unless .measure(...) narrows it
print(bell)                                # the circuit as a text diagram; repr(bell) is the builder chain

result = machine.run(bell, shots=2000)
print(result.counts)                       # about 1000 each of '00' and '11', and a few '01' and '10'
print(result.summary())
```

`machine.run` did six things:

1. **Compiled** the circuit to the trapped-ion native gates: GPi and GPi2 (π and π/2 rotations about an axis in the
   equator of the Bloch sphere), the Mølmer-Sørensen (MS) entangling gate, and virtual Z rotations, which cost no pulse.
2. **Calibrated** the machine. Nobody told it the gate pulses, so it derived a calibration table from the device's
   physics (the closed-form *surrogate*, corrected by exact spot checks). This takes a few seconds the first time and is
   cached for the rest of the Python session.
3. **Scheduled** laser pulses from that table, with absolute times.
4. **Prepared** the ions: Doppler cooling, sideband cooling of the modes the gate uses, optical pumping into |0⟩.
5. **Evolved** the joint quantum state of the ions *and their motion* through every pulse. The ions' shared motion is
   the bus the MS gate uses; the motional modes are harmonic oscillators truncated to a few Fock levels.
6. **Read out** each ion by state-dependent fluorescence: a bright ion scatters photons and a dark one almost none.

The machine itself is a frozen record: a `Device` (the physics), a calibration table (here `None`, meaning "build the
surrogate"), three option objects (`Physics`, `Numerics`, `Readout`) and a fidelity level. You make variants of it with
`dataclasses.replace`, which Section 6 covers.

The preset's noise model is quiet, yet a few shots read `01` or `10`. The summary shows where they come from: the gate
physics itself (the entangling gate's motional loops, the ions' thermal motion, the addressing beam's light on the
neighbouring ion) and the state preparation and measurement (SPAM) line.

```python
print(result.diagnostics.level)                    # which fidelity level integrated the run
print(result.spam["q0"])                           # (eps_B, eps_D): bright read dark, dark read bright
print(result.diagnostics.intrinsic_budget.total)   # the closed-form error budget of the gates
```

### Check yourself

1. Name the six stages inside `machine.run`, in order.
2. The preset's noise model is quiet. Name two sources of the few `01` and `10` counts.
3. You call `machine.run(bell, shots=2000)` again. Is the second call faster, and why?

<details><summary>Answers</summary>

1. Compile, calibrate, schedule, prepare, evolve, read out.
2. Any two of: the entangling gate's residual spin-motion coupling and the motion's thermal occupation, off-resonant
   excitation during the pulses, addressing crosstalk (the neighbour's share of an addressing beam), and SPAM (readout
   and preparation errors).
3. Yes: the calibration built on the first call is cached in the Python session, so only the evolution and readout run.

</details>

## 3. Circuits and the conventions that bite

`Circuit(n)` is an immutable builder: every gate method returns a new circuit, qubits first and then angles in radians.
The standard gates are `x y z h s sdg t tdg sx id rx ry u3 cnot cx cz swap cp rxx rzz`; the native gates are
`gpi(q, phi)`, `gpi2(q, phi)`, `ms(q0, q1, phi0, phi1, theta)`, `zz(q0, q1, theta)` and the virtual `rz(q, theta)`.

Two native conventions matter. `ms(q0, q1, 0.0, 0.0, math.pi / 2)` is maximally entangling (IonQ writes the same angle
as 0.25 turns). A virtual RZ changes no state; it shifts the phase of every later pulse on that qubit.

```python
from qutip_trap.control.compiler import ideal_probabilities

report = machine.compile(bell)
print(report)                              # native gates, entangling count, whole-circuit residual
print(repr(report.circuit))                # the native circuit the scheduler plays
native_bell = trap.Circuit(2).ms(0, 1, 0.0, 0.0, math.pi / 2)   # (|00> - i|11>)/sqrt 2 from |00>
for circuit in (native_bell, bell):
    ideal = ideal_probabilities(circuit)                          # |<b|U|00>|^2, qubit 0 rightmost
    assert ideal.keys() == {"00", "11"} and all(math.isclose(p, 0.5) for p in ideal.values())
```

### Bit order

Every bitstring key puts **qubit 0 rightmost**, as the least-significant bit. Other tools order bits differently, so
learn this one first. Run an X on qubit 0 of a three-qubit circuit:

```python
machine3 = trap.presets.yb171_chain(3)
flip0 = machine3.run(trap.Circuit(3).x(0), 200)
print(flip0.counts)


def top(counts):
    """The most frequent outcome."""
    return max(counts, key=counts.get)


assert top(flip0.counts) == "001"                                   # qubit 0 is the rightmost character
assert top(flip0.to_ionq_v1_probabilities()) == "1"                 # IonQ v1: decimal keys, qubit 0 the 2^0 bit
v2 = flip0.to_ionq_v2_probabilities()["probabilities"]["registers"]["output_all"]
assert top(v2) == "100"                                             # IonQ v2 writes q[0] first
assert top(flip0.reversed_bits().counts) == "100"                   # the order Cirq, Braket and PennyLane print
```

`reversed_bits()` returns the same result with every key reversed, for comparing with SDKs that print qubit 0's bit
first.

### Measuring fewer qubits, and fewer qubits than ions

`.measure(...)` names the qubits measured at the end. The keys then cover only those, still in ascending order with the
lowest rightmost, and `result.qubits` names each column of `result.bitstrings`:

```python
partial = machine3.run(trap.Circuit(3).x(2).measure(2, 0), 200)
assert top(partial.counts) == "10"            # qubit 2 (left) read 1, qubit 0 (right) read 0
assert partial.qubits == (0, 2)
narrow = machine3.run(trap.Circuit(1).x(0), 100)
assert top(narrow.counts) == "1"              # qubit i runs on ion i; the idle ions are simulated in |0>
```

A circuit wider than the machine is refused, and so is a mid-circuit measurement, whose physics is not modelled. Every
refusal says what to do instead:

```python
try:
    machine3.run(trap.Circuit(4).x(3), 100)
except ValueError as err:
    print(err)
```

### Check yourself

1. What key does `trap.Circuit(3).x(2)` produce in `counts`?
2. `trap.Circuit(4).x(1).measure(3, 1)`: how many characters do its keys have, and what is the most likely key?
3. What `theta` makes `ms` maximally entangling, in radians and in IonQ turns?

<details><summary>Answers</summary>

1. `'100'`: qubit 2 is the leftmost of three characters.
2. Two characters, qubit 3 on the left and qubit 1 on the right: `'01'`.
3. π/2 radians, which IonQ writes as 0.25 turns.

</details>

## 4. Reading a result

A `Result` is plain data:

- `counts`, `probabilities` and `error_bars`, keyed as above.
- `bitstrings`: one row per shot, one column per entry of `qubits`.
- `spam`: per qubit `q{i}` the readout errors (ε_B, ε_D), and `q{i}.state_preparation`.
- `diagnostics`: what the run did and approximated.
- `record`: everything else the run produced (the compile report, the schedule, the traces).
- `summary()`: all of it as a readable report.

```python
d = result.diagnostics
print(d.level_reason)                 # why this fidelity level
print(d.space.dims, d.mode_class)     # the ion and motional factors integrated; each mode resolved, frozen or dropped
print(d.boundary_population)          # population at each resolved mode's truncation edge (small is good)
print(len(d.approximations), "approximations:", d.approximations[0])
print(result.record.schedule)         # the pulses that were played
record = result.to_dict()             # the versioned JSON record; trap.Result.from_dict reads it back
assert trap.Result.from_dict(record).counts == result.counts
```

A mode is **resolved** when the gates move it and it is carried as a truncated oscillator. It is **frozen** when it
only dims the drive through its thermal motion, and **dropped** when nothing couples to it.

### The register state, and the frame it is in

`keep_final_state=True` keeps the register's density matrix, with the motion traced out. Two traps hide in it. Ion 0 is
the *first* tensor factor, the opposite of the counts keys. And the physical state still carries each qubit's
**residual Z frame**: the virtual Z rotations the compiler never played, plus the light-shift phase the scheduler
compensated in software. Both are harmless for measurement in the computational basis, but they rotate the state.
Compare against the textbook Bell state and you get nonsense:

```python
from qutip_trap.run.job import ideal_register_state, register_fidelity

kept = machine.run(bell, 500, keep_final_state=True)
phi_plus = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
naive = float(np.real(phi_plus.conj() @ kept.final_state.full() @ phi_plus))
print("naive <Phi+|rho|Phi+>:", round(naive, 4))              # near 0.01: the residual frames rotate the state
print("register_fidelity:", round(register_fidelity(kept), 4))  # near 0.998: the ideal state in the same frame
print("frames (rad):", kept.record.schedule.phase_frame)
print(np.round(ideal_register_state(kept), 3))                # the ideal ket register_fidelity compares with
```

Use `register_fidelity(result)`, or build your own target from `ideal_register_state(result)`.
`qutip_trap.run.job.to_register_order` converts a ket from the compiler's bit order to the register's tensor order.

### Check yourself

1. Why does ⟨Φ⁺|ρ|Φ⁺⟩ come out near 0.01 for a good Bell pair, and what do you use instead?
2. In `final_state`, which ion is the first tensor factor? In a `counts` key, which qubit is the rightmost character?
3. Where in a result do you check that no motional mode was truncated too tightly?

<details><summary>Answers</summary>

1. The physical state carries each qubit's residual Z frame (the compiler's unplayed virtual Z rotations and the
   compensated light shifts). Use `register_fidelity(result)`, whose target `ideal_register_state(result)` is in that
   frame.
2. Ion 0 is the first tensor factor of `final_state`; qubit 0 is the rightmost character of a key.
3. `result.diagnostics.boundary_population` (also the "modes" line of `summary()`). A run whose boundary population stays
   too high after the automatic cap increases also raises a `TruncationWarning`.

</details>

## 5. Cost, calibration and fidelity levels

### Ask before you run

`machine.estimate(circuit)` compiles, calibrates and schedules, then reports what the run would integrate without
integrating anything:

```python
estimate = machine.estimate(bell)
print(estimate)          # the level, the joint space and its dimension, the pulse counts, the time of one pass
print(estimate.reason)
```

Its `wall_time_s` is **one pass** through the schedule. A run makes one pass per *branch* of the initial mixture (the
thermal motion's Fock states and the pumped internal levels above `Numerics.branch_weight_min`), per noise *sample*, per
*trajectory*. The branches run in parallel on worker processes. The number of shots barely matters on a quiet machine,
because the shots are sampled from the final register state after the evolution.

### Pin the calibration

The surrogate cache lives in the Python session. Its key is the device, the beam roles, the run's `seed`, the
`Numerics`, the `Physics` and the pairs the circuit entangles. So a machine without a table **recalibrates whenever
you change the seed or the numerics**, several seconds each time on two ions. Pin a table once before any sweep:

```python
pinned = machine.calibrated(pairs=[(0, 1)])   # reuses the cached surrogate of the first run
print(pinned.table)
repeats = [pinned.run(bell, 200, seed=s) for s in range(3)]    # no recalibration between seeds
assert len({tuple(map(tuple, r.bitstrings)) for r in repeats}) == 3   # different seeds, different shots
assert np.array_equal(repeats[0].bitstrings, pinned.run(bell, 200, seed=0).bitstrings)  # one seed, one result
```

Without `pairs=`, `calibrated()` calibrates every pair of the crystal. To keep a calibration between Python sessions,
pickle the machine. `CalibrationTable.to_dict()` writes a JSON record without the entangling waveforms (they may hold
functions), and a table read back from it cannot play a two-qubit gate.

```python
restored = pickle.loads(pickle.dumps(pinned))
assert restored.hash() == pinned.hash() and restored.table.waveform_for((0, 1)) is not None
```

### Branches and the branch cut

`Numerics.branch_weight_min` (default 10⁻⁶) is the smallest weight of the initial mixture that gets its own
evolution. The rest is dropped, renormalized and reported. A coarser cut is the cheapest speed-up when you do not need
the last 10⁻⁴:

```python
lean = dataclasses.replace(pinned, numerics=trap.Numerics(branch_weight_min=1e-3))
lean_result = lean.run(bell, 2000)
print(lean_result.diagnostics.branches, "branches; dropped weight", lean_result.diagnostics.dropped_branch_weight)
```

### Fidelity levels

`FidelityLevel.AUTO`, the default, runs **JOINT_EXACT** when the joint space of the ions and the resolved modes stays
within the size guards (`Numerics.joint_dimension_max = 4096` and `nnz_max = 2 × 10⁷`). There every branch evolves
exactly on the whole space. Above the guards it runs **GATE_LOCAL**: each gate is exact on its own ions and the modes
they couple to, and the motion is carried between gates by a tracked model whose error is reported. Two ions run
JOINT_EXACT; a four-ion GHZ circuit already goes GATE_LOCAL:

```python
ghz4 = trap.Circuit(4).h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)
estimate4 = trap.presets.yb171_chain(4).estimate(ghz4)
print(estimate4.level, estimate4.dimension, f"about {estimate4.wall_time_s:.3g} s per pass")
```

### Check yourself

1. You will run one circuit at 20 seeds. What do you do first, and why?
2. A quiet run of 2,000 shots takes 2 s. About how long does the same run take at 200,000 shots?
3. What decides, under `AUTO`, between JOINT_EXACT and GATE_LOCAL?

<details><summary>Answers</summary>

1. Pin the calibration with `machine.calibrated(pairs=...)`. The surrogate cache is keyed on the seed, so an unpinned
   machine would rebuild it for every new seed.
2. Barely longer: the shots are sampled from the final register state, so the evolution does not repeat.
3. The declared joint space's dimension and drive-operator non-zeros against `Numerics.joint_dimension_max` and
   `Numerics.nnz_max`. `estimate(...).reason` prints the comparison.

</details>

## 6. Variants: the option objects

A machine holds three option records, one per concern. Change one with `dataclasses.replace`; each variant has its own
`hash()`.

| Record | Decides | Examples |
|---|---|---|
| `Physics` | which effects are simulated | `noise=False`, `internal_levels=3` (leakage levels), `entangler="zz"`, `crosstalk_suppression="local"`, `hardware_chain=False`, `builder=BuilderOptions(...)` |
| `Numerics` | how the integration is done | `branch_weight_min`, `atol`/`rtol`, `caps={mode: d}`, `samples`, `ntraj`, `workers`, `map="serial"`, `convergence_check=True` |
| `Readout` | how the photon record is read | `mode="fast"` (a POVM on the joint outcome) or `"full"` (every photon record generated and discriminated), `discriminator=` |

The fidelity level is a field of the machine too: `level=trap.FidelityLevel.GATE_LOCAL`.

```python
full = dataclasses.replace(pinned, readout=trap.Readout(mode="full"))
photons = full.run(bell, 300)
print(photons.photon_records[:4], photons.bitstrings[:4])   # detected photons per ion per shot, and the bits

forced = dataclasses.replace(pinned, level=trap.FidelityLevel.GATE_LOCAL)
walked = forced.run(bell, 2000)
print(walked.diagnostics.level, "|", walked.diagnostics.level_reason)
```

A pinned table was fitted under the `Physics` it was calibrated with, because the spot checks play the gate under it.
After you change `Physics`, calibrate the variant again:

```python
zz_machine = dataclasses.replace(machine, physics=trap.Physics(entangler="zz")).calibrated(pairs=[(0, 1)])
print(zz_machine.compile(bell))       # the two-qubit gate now compiles to the native ZZ
print(zz_machine.run(bell, 500).counts)
```

### Check yourself

Which record, and which field, do you change to:

1. get the detected photon counts of every shot,
2. switch off the device's noise for a comparison run,
3. evolve fewer branches of the initial mixture,
4. compile two-qubit gates to ZZ instead of MS?

<details><summary>Answers</summary>

1. `Readout(mode="full")`.
2. `Physics(noise=False)`.
3. `Numerics(branch_weight_min=...)`, set higher.
4. `Physics(entangler="zz")`, and recalibrate the variant.

</details>

## 7. Adding noise

The presets are quiet. Noise is part of the `Device`, as a `NoiseModel` of spectra, drifts and event rates, never as
error rates. How a noise source enters a run depends on its correlation time:

- **White** noise (the flat `white_level` of a `NoiseSpectrum`) becomes a Lindblad collapse operator: heating from
  electric-field noise, qubit dephasing from magnetic-field noise, laser-intensity noise.
- A **tabulated band** of a `NoiseSpectrum` is synthesized as a time series in each dynamical sample.
- A **`Drift`** is a slow parameter drawn once per *sample* at the shot clock, as an Ornstein-Uhlenbeck chain.
- **`Collisions`** with background gas are events drawn per shot, with heralds and discarded shots.

Every `NoiseSpectrum` is **two-sided in angular frequency**, with its unit in the name. To reproduce a measured heating
rate, convert it with `s_e_from_heating_rate`, which returns the single-sided density, and halve that for the spectrum:

```python
from qutip_trap.noise.model import NoiseModel
from qutip_trap.noise.spectra import Drift, white_spectrum
from qutip_trap.trap.heating import s_e_from_heating_rate

x_com = machine.device.crystal.modes[3]                      # the 3.0 MHz centre-of-mass mode of the gate
s_e = s_e_from_heating_rate(100.0, float(machine.device.crystal.masses_kg[0]), x_com.omega_rad_s)
noise = NoiseModel(
    S_E=white_spectrum(s_e / 2.0, "(V/m)^2/(rad/s)"),       # 100 quanta/s at that mode; uncorrelated between ions
    correlation_length_m=0.0,
    rabi_drift=Drift(rms=0.02, tau_s=1.0, servo_bandwidth_hz=None),   # 2 % rms Rabi-frequency drift, correlated over 1 s
)
noisy = trap.presets.yb171_chain(2, noise=noise)
print(noisy.device.noise.summary(noisy.device))              # the rates that follow: heating per mode, drift rms
```

Noise multiplies the cost. A run draws `Numerics.samples` dynamical samples (by default `min(shots, 64)` when anything
is sampled). Above dimension 128 its collapse operators run as `Numerics.ntraj` quantum trajectories (64 by default)
per branch. Size the ensemble deliberately:

```python
noisy = dataclasses.replace(noisy, numerics=trap.Numerics(samples=4, ntraj=8, branch_weight_min=1e-3))
noisy = noisy.calibrated(pairs=[(0, 1)])
noisy_result = noisy.run(bell, 400, keep_final_state=True)
dn = noisy_result.diagnostics
print(dn.samples, "samples,", dn.trajectories, "evolutions; effective sample size", round(dn.effective_sample_size))
print("F noisy:", round(register_fidelity(noisy_result), 4), "| F quiet:", round(register_fidelity(kept), 4))
```

Shots drawn from one sample share its drift, so they are not independent draws from the ensemble. The **effective sample
size** measures that, and the error bars use it, not the shot count. `Physics(noise=False)` runs the same device
quietly.

### Check yourself

1. How does each of a white level, a tabulated band and a `Drift` enter a run?
2. Your run got thirty times slower when you added noise. Which three counts multiply, and which `Numerics` fields set
   them?
3. `effective_sample_size` is 166 for 400 shots. What does that tell you about the error bars?

<details><summary>Answers</summary>

1. A white level is a Lindblad collapse operator; a band is a time series synthesized per dynamical sample; a `Drift` is
   a parameter drawn once per sample at the shot clock.
2. Branches × samples × trajectories, set by `branch_weight_min`, `samples` and `ntraj`.
3. The shots are correlated through the samples' drifts, so the error bars are about √(400/166) ≈ 1.55 times wider than
   400 independent shots would give; the run's `error_bars` already include this.

</details>

## 8. Devices

A `Device` is physical parameters only: the crystal and its trap, the magnetic field, the beams, the noise model, the
detector, the control electronics, the preparation recipe and the roles the beams play. Everything else is derived:
Lamb-Dicke parameters, Rabi frequencies, light shifts, crosstalk, mode frequencies, detection rates. Each derived number
carries the id of its provenance record in `docs/provenance/ledger.yaml`.

```python
device = machine.device
print(device.specs())                      # every derived number, grouped, with its provenance id
derived = device.derived().values
print(derived["rabi_hz[(0, 2)]"], derived["eta[(0, 3)]"], derived["crosstalk[(0, 1)]"])
```

The keys follow the calibration table's names: `rabi_hz[(ion, first beam)]`, `eta[(ion, mode)]`,
`crosstalk[(ion, neighbour)]`, `mode_hz[mode]`. A mode index is a position in `device.crystal.modes`, ordered axial,
then the two transverse families, each ascending in frequency.

### The presets and their knobs

`trap.presets.yb171_chain(n, ...)` takes `noise=`, `hardware=` (a `control.hardware.HardwareChain`: DDS bit depths,
modulator rise time, amplifier bandwidth, dead time), `detector=`, `recipe=`, `omega_hz=` (the secular frequencies
x, y, z), `s_o=` (the detection beam's saturation), `address_waist_m=` and `phase_continuous=`. At its default trap
frequencies a chain of more than five ions buckles out of line, and the crystal solver refuses it; lower the axial
frequency for a longer chain:

```python
from qutip_trap.trap.crystal import ZigzagError

try:
    trap.presets.yb171_chain(6)
except ZigzagError as err:
    print("six ions at the default trap:", str(err)[:80], "...")
six = trap.presets.yb171_chain(6, omega_hz=(3.0e6, 2.9e6, 0.6e6))
print(six.device)
```

`trap.presets.ca40_optical(n)` is a 40Ca+ optical qubit on the 729 nm quadrupole line with a photomultiplier readout.
Use it with one ion. It has no entangling drive, so it refuses two-qubit gates, and its one 729 nm beam is global, so on
a chain a single-qubit gate drives every ion alike. It also has no sideband cooling. Its radial modes stay Doppler-hot,
and every carrier pulse is averaged over their thermal Fock states as branches. One ion takes a few thousand branches.
Two ions take about 2 × 10⁵ at the default cut and still drop over a third of the mixture, which
`diagnostics.dropped_branch_weight` reports.

### Your own device

Build a device from its parts. This one is the example chain in a stiffer trap with more laser power. The
`device.presets` helpers give the preset's beams, detector and electronics:

```python
from qutip_trap.control.schedule import GateDrive
from qutip_trap.device.model import BeamRoles, Device, Field
from qutip_trap.device.presets import (
    crain_snspd_detector,
    ideal_hardware,
    oblique_detection_beam,
    raman_pair_along_x,
    secular_trap,
)
from qutip_trap.prep.recipe import standard_recipe
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal

trap_model = secular_trap((3.2e6, 3.1e6, 1.1e6))              # secular frequencies x, y, z (Hz)
yb = species("171Yb+")
crystal = solve_crystal(trap_model, (yb, yb))                  # equilibrium positions and normal modes
beams = list(raman_pair_along_x(0.5, 60e-6, (0.0, 0.0, 0.0)))  # the global Raman pair: beams 0 and 1
gate_drives = {}
for ion in range(2):                                           # one tightly focused addressing pair per ion
    first = len(beams)
    at = tuple(float(v) for v in crystal.positions_m[ion])
    beams += raman_pair_along_x(0.5 * (2.5e-6 / 60e-6) ** 2, 2.5e-6, at)
    gate_drives[ion] = GateDrive("raman", (first, first + 1))
beams.append(oblique_detection_beam())                         # 369.5 nm cooling, pumping and detection light
my_device = Device(
    crystal=crystal,
    trap=trap_model,
    field=Field(5.0, (1.0, 0.0, 0.0)),                         # 5 G along x: the quantization axis
    beams=tuple(beams),
    noise=NoiseModel(),
    detector=crain_snspd_detector(),
    hardware=ideal_hardware(),
    roles=BeamRoles(
        gate=gate_drives,
        entangling={0: GateDrive("raman", (0, 1)), 1: GateDrive("raman", (0, 1))},
        detection=len(beams) - 1,
    ),
)
my_device = dataclasses.replace(my_device, preparation=standard_recipe(my_device, raman_pair=(0, 1)))
my_machine = trap.Machine(my_device, name="stiffer two-ion chain")
print(my_machine.run(bell, 1000).counts)
assert Device.from_dict(my_device.to_dict()).hash() == my_device.hash()   # a device round-trips through JSON exactly
```

A calibration table belongs to the device it was fitted for (`table.is_current_for(device.hash())`). Changing any device
parameter makes a new device, and a pinned table from the old one is played as given with a note saying so. Calibrate
the new device instead.

### Check yourself

1. Why does `yb171_chain(6)` fail at the default trap, and which knob fixes it?
2. Which field of `BeamRoles` names the beams that play the entangling gates?
3. You change a device's magnetic field. Does a table pinned before the change still describe it?

<details><summary>Answers</summary>

1. At the default transverse-to-axial frequency ratio six ions buckle into a zigzag, which the crystal solver refuses
   (`ZigzagError`). Lower the axial frequency with `omega_hz=`.
2. `entangling`, a map from ion to the `GateDrive` (kind and beam indices) that plays its two-qubit gates.
3. No. The device's hash changed, the table is no longer current for it, and a run on it plays the old beliefs with a note
   saying so. Calibrate the new device.

</details>

## 9. The laboratory

`qutip_trap.experiments` runs the experiments a laboratory calibrates with: Rabi flopping, Ramsey fringes, sideband
spectroscopy, thermometry, heating rate, light-shift and crosstalk scans, micromotion, detection histograms, and MS and
parity scans. They run on a machine through the same engine and are fitted the way a laboratory fits them.

By default an experiment is idealized: exact populations (`shots=None`), no readout error (`readout=False`), the motion
in its ground state (`nbar={}`), no noise sample. Pass the laboratory's conditions to see what a laboratory sees:

```python
from qutip_trap.experiments.single_ion import rabi_scan

durations = tuple(float(t) for t in np.linspace(0.0, 30e-6, 13))
ideal_scan = rabi_scan(pinned, 0, durations)
nbar = {m: entry.value for m, entry in pinned.table.nbar.items()}
lab_scan = rabi_scan(pinned, 0, durations, shots=200, readout=True, nbar=nbar, seed=1)
print(ideal_scan)
print(lab_scan)
print("fitted:", round(lab_scan.value("f_rabi_hz")), "+-", round(lab_scan.uncertainty("f_rabi_hz")), "Hz")
```

A result proposes calibration-table entries; you adopt them only after checking the fit:

```python
proposal = pinned.table.updated_with(lab_scan)
if lab_scan.quality == "good":
    recalibrated = dataclasses.replace(pinned, table=proposal)
    print(recalibrated.table.rabi[(0, 2)])
```

`qutip_trap.calibration.calibrate(machine, method="experiments")` runs the whole calibration by simulated experiments in
dependency order. That takes minutes rather than seconds, and it cross-checks the closed-form surrogate.

### Check yourself

1. An experiment run with default keywords looks cleaner than your laboratory's data. Name three reasons.
2. What do you check before adopting `table.updated_with(scan)`?

<details><summary>Answers</summary>

1. Any three of: exact populations with no shot noise, no readout error, the motion in its ground state, no noise sample.
2. `scan.quality` (`"good"`, not `"poor"` or `"failed"`): the fit converged and its reduced chi-square is consistent with
   its error bars.

</details>

## 10. Benchmarks and the error model

The benchmarks run circuits through `Machine.run` and report, beside each measured number, what the simulator's own
error budget predicts:

```python
from qutip_trap.benchmarks.ghz import ghz_fidelity
from qutip_trap.benchmarks.rb import randomized_benchmarking

rb = randomized_benchmarking(pinned, (0,), (1, 256, 1024), n_sequences=1, shots=1000, fix_offset=True)
r, r_sigma = rb.error_per_clifford
print(rb.fidelity_form())
print(f"r per Clifford {r:.1e} +- {r_sigma:.1e}; the gates' channels predict {rb.budget.predicted.r_channel:.1e}")

ghz = ghz_fidelity(pinned, (0, 1), shots=300, analysis_phases_rad=np.linspace(0.0, np.pi, 4, endpoint=False))
bound, bound_sigma = ghz.fidelity_bound
print(f"parity bound {bound:.4f} +- {bound_sigma:.4f}; exact register fidelity {ghz.register_fidelity:.4f}")
```

Single-qubit gates on the quiet preset err at the 10⁻⁵ level, so randomized benchmarking needs sequences of a thousand
Cliffords before the decay rises above the shot noise; a shorter sequence returns an error per Clifford consistent with
zero. The prediction beside it tells you which case you are in.

The GHZ parity bound, (P₀₀ + P₁₁ + C)/2 with C the parity contrast, is the laboratory's estimate: it comes from shots,
carries their uncertainty and can land above 1. The register fidelity comes from the run's kept state and has no shot
noise.

`benchmarks.volume.quantum_volume` runs the quantum-volume protocol; a pass needs at least 100 circuits, and the result
says when it has fewer.

`machine.error_model()` condenses the machine into the phenomenological numbers vendor emulators take: the average
infidelity of each native gate kind (from its simulated channel), the gate durations, and the SPAM errors. It exports
them in IonQ's, Quantinuum's and the QDK resource estimator's vocabularies:

```python
em = pinned.error_model()
print({kind: f"{r:.1e}" for kind, r in em.infidelity.items()})
print(em.to_ionq_noise())
print(em.to_quantinuum_error_params())
```

### Check yourself

1. What does `rb.budget.predicted` add to `rb.error_per_clifford`?
2. Where do the error model's gate infidelities come from?

<details><summary>Answers</summary>

1. The error per Clifford the simulator's own gate channels predict: a check on the benchmark from the same physics, so a
   disagreement points at something the budget does not compose (coherent errors, SPAM, too few sequences).
2. From each native gate kind's simulated channel on its own qubits (GATE_LOCAL process tomography of the played
   pulses), not from assumed numbers.

</details>

## 11. Other front doors: OpenQASM, IonQ, Qiskit, background jobs

**OpenQASM 2 and IonQ JSON.** `Circuit.from_openqasm` reads OpenQASM 2 with `qelib1.inc` and inlined gate declarations.
IonQ's circuit JSON carries native gates only, so compile first:

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
from_qasm = trap.Circuit.from_openqasm(qasm)
native = machine.compile(from_qasm).circuit
ionq_input = native.to_ionq()                              # the IonQ "input" object, phases in turns
assert trap.Circuit.from_ionq(ionq_input).ops == native.ops
print(ionq_input["circuit"][:2])
```

**Qiskit.** `QutipTrapBackend(machine)` is a Qiskit `BackendV2`; `QutipTrapProvider().get_backend("yb171_chain",
n_ions=2)` builds one on a preset. Qiskit keys its counts by *classical* bit, so where a qubit's bit appears depends on
the clbit you measured it into:

```python
from qiskit import QuantumCircuit, transpile

from qutip_trap.interop.qiskit import QutipTrapBackend

backend = QutipTrapBackend(pinned)
qc = QuantumCircuit(2, 2)
qc.x(0)
qc.measure(0, 1)                                           # qubit 0 into clbit 1
qc.measure(1, 0)
qiskit_job = backend.run(transpile(qc, backend), shots=200)
assert top(qiskit_job.result().get_counts()) == "10"       # Qiskit: clbit 1 (qubit 0) on the left
assert top(qiskit_job.results[0].counts) == "01"           # the qutip-trap Result behind it: qubit 0 rightmost
```

**Background jobs.** `machine.submit(...)` runs the same run in a separate, *spawned* Python process and returns a `Job`
with `status()`, `progress`, `result()`, `record()` and `cancel()`. The shots are identical to `machine.run` at the same
seed. A spawned process re-imports your script, so **in a script, call `submit` under `if __name__ == "__main__":`**. A
notebook or REPL needs no guard. Submit a pinned machine: an unpinned one recalibrates inside the worker.

```python
job = pinned.submit(bell, 300, seed=3, label="bell")
background = job.result(timeout_s=600)
assert job.status() == "done"
assert np.array_equal(background.bitstrings, pinned.run(bell, 300, seed=3).bitstrings)
```

In the foreground, `machine.run(circuit, shots, progress=print)` prints a line per step of the run: the calibration,
the branches as they finish (a chunk at a time when they run on worker processes), each sample and the readout.

### Check yourself

1. Why does `bell.to_ionq()` fail while `machine.compile(bell).circuit.to_ionq()` works?
2. What guard does `submit` need in a script, and why?
3. A Qiskit circuit measures qubit 0 into clbit 1. Where is qubit 0's bit in `get_counts()` keys, and where in the keys of
   `job.results[0].counts`?

<details><summary>Answers</summary>

1. IonQ's JSON carries the native gates only; `bell` holds `h` and `cnot`, the compiled circuit `gpi2` and `ms`.
2. `if __name__ == "__main__":`. The worker is a spawned Python process that imports your script's module; unguarded, the
   import would submit again.
3. Second from the right in `get_counts()`, because Qiskit orders keys by clbit and clbit 1 is second from the right;
   rightmost in the qutip-trap `Result`, because it orders keys by qubit.

</details>

## 12. Going lower: the schedule and the engine

The methods of a machine walk down the levels of the simulation. `run` is the machine, `compile` the circuit,
`schedule` the pulses, `engine` the dynamics, and `device` the physics.

```python
schedule = pinned.schedule(bell)
print(schedule)                                   # the pulses with absolute times, then the measurement
print([p.gate_id for p in schedule.pulses][:6])
print(schedule.gates[0].pair, schedule.gates[0].waveform.duration_s)   # the entangling gate as played
```

`machine.engine` is the JOINT_EXACT engine configured from the machine. Replay a recorded schedule through it from one
pure branch, the ground state of motion, and watch the populations and the phonon numbers:

```python
from qutip_trap.dynamics.engine import JointExactEngine, SeedSpec
from qutip_trap.noise.sampling import quiet_sample

space = result.diagnostics.space
start = space.initial_state([0, 0], fock={t.mode: 0 for t in space.resolved})
engine = pinned.engine
traces = engine.run_pulses(
    pinned.device, result.record.schedule, start, space, quiet_sample(0), SeedSpec(0), pinned.numerics
)
print(sorted(traces.expectations))                # P1 per ion and the phonon number of every carried mode
print("final P1:", [round(float(np.real(traces.expectations[f"P1[{i}]"][-1])), 3) for i in (0, 1)])
print(engine.last_report.method, engine.last_report.kernel, len(engine.last_report.segments), "segments")
```

Or build a pulse yourself. Here ion 0's addressing pair drives the carrier for two π times, one full Rabi cycle, on a
space you declare: the two ions and the gate's 3 MHz mode with 12 Fock levels, every other mode frozen. A `HilbertSpace`
allocates nothing when you declare it; its operators are built lazily.

```python
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.raman import derive_raman_drive, square_drive

drive = derive_raman_drive(pinned.device, 0, pinned.device.roles.gate[0].beams, scattering=False)
t_pi = 0.5 / drive.carrier_rabi_hz                                   # a pi pulse: Omega t = pi
pulse = Pulse(square_drive(drive, include_stark=False), 0.0, 2.0 * t_pi, "flop", ())
flop_space = HilbertSpace((2, 2), (ModeTruncation(3, 12, (0, 3), 0.1),), None, (0, 1, 2, 4, 5))
flop = JointExactEngine(store_per_segment=41).run_pulses(
    pinned.device,
    Schedule((pulse,), (), (), {0: 0.0, 1: 0.0}),
    flop_space.initial_state([0, 0], fock={3: 0}),
    flop_space,
    quiet_sample(),
    SeedSpec(0),
    trap.Numerics(),
)
p1 = np.real(flop.expectations["P1[0]"])
print(f"Rabi frequency {drive.carrier_rabi_hz:.0f} Hz; P1 at t_pi {np.interp(t_pi, flop.times_s, p1):.4f}")
```

`dynamics.hamiltonian.BuilderOptions` (passed as `Physics(builder=...)` or to the engine) holds the builder's optional
approximations: the Lamb-Dicke expansion (`lamb_dicke_order`), the interaction frame with the sideband rotating-wave
approximation (`frame="interaction", rwa=True`: the textbook model) and the individual terms. None of them is on by
default.

### Check yourself

1. Which machine method gives you the pulses of a circuit without integrating anything?
2. What does declaring a `HilbertSpace` allocate?
3. What does `traces.expectations` hold after `run_pulses`?

<details><summary>Answers</summary>

1. `machine.schedule(circuit)` (and `machine.estimate(circuit)` for the space and cost it would take).
2. Nothing: operators are built lazily and cached, so the size guards are checked before anything joint exists.
3. `P1[i]`, the population of |1⟩ of every ion, and `n[m]`, the phonon number of every carried mode, at the stored times
   `traces.times_s`.

</details>

## 13. Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| `ValueError: the circuit has N qubits and the machine M ion(s)` | one ion per qubit | a preset with more ions, `yb171_chain(N)` |
| `ScheduleError: ... no entangling drive on this device` | `ca40_optical` has no two-qubit gate drive | an entangling device such as `yb171_chain` |
| `ScheduleError: no entangling waveform in the calibration table ...` | the table lacks that pair, or was read back from JSON | `calibrated(pairs=[...])` with the pair, or pickle the machine to keep a table |
| `ScheduleError: ... mid-circuit measure, reset and recool are refused` | only a terminal measurement is modelled | measure at the end |
| `ZigzagError` | the chain buckles at these trap frequencies | lower the axial frequency in `omega_hz` |
| `RunError: ... branch_weight_min ... keeps no branch` | the cut is above the likeliest branch of a hot mixture | lower `Numerics.branch_weight_min` below the weight the message quotes |
| `RunError: level='JOINT_EXACT' asks for a joint space of dimension ...` | a forced level above the size guards | let `level=AUTO` route it, or raise the guards deliberately |
| `TruncationWarning` | a mode's cap was clamped or its boundary population stayed high | check `diagnostics.boundary_population`; set `Numerics(caps=...)` or `mode_dimension_max`; cool the mode |
| a fidelity against your ideal state is tiny | the residual Z frames | `register_fidelity(result)` |
| counts look reversed next to Cirq or PennyLane | the bit orders differ | `result.reversed_bits()` |
| every run with a new seed takes seconds longer | the surrogate is keyed on the seed | pin with `calibrated()` |
| a noisy run is very slow | branches × samples × trajectories | `Numerics(samples=..., ntraj=..., branch_weight_min=...)`; `estimate()` first |

## 14. Mixed practice

These questions mix the sections on purpose: the conventions are easiest to confuse when they come one after another.
Answer all of them before you open the answers.

1. `Circuit(3).x(1)` on a three-ion machine: the key in `result.counts`, in IonQ v1 and in IonQ v2?
2. You want 50 seeds of a noisy three-ion GHZ circuit. Name, in order, the three things you do before the loop.
3. `result.final_state` of a two-ion run: is |01⟩ in its basis ion 0 in |0⟩ and ion 1 in |1⟩, or the reverse?
4. Which of `Physics`, `Numerics` and `Readout` must be followed by a recalibration when you change it on a pinned
   machine?
5. A `NoiseSpectrum` for electric-field noise: one-sided or two-sided, per Hz or per rad/s?
6. `estimate(...).wall_time_s` is 2 s. The run takes 30 s on a quiet machine. Give the likeliest reason.
7. `ms(0, 1, 0.0, 0.0, theta)`: which `theta` makes a Bell pair from |00⟩, and is that radians or turns?

<details><summary>Answers</summary>

1. `'010'` in `counts`, `'2'` in IonQ v1 (decimal, qubit 0 the 2⁰ bit), `'010'` in IonQ v2 (q[0] first, so the middle
   character is still qubit 1).
2. Build the noise into the device (`yb171_chain(3, noise=...)`), size the ensemble with `Numerics(samples=...,
   ntraj=..., branch_weight_min=...)`, and pin the calibration with `calibrated(pairs=[(0, 1), (1, 2)])`. Then check
   `estimate()` before the loop.
3. Ion 0 in |0⟩ and ion 1 in |1⟩: ion 0 is the first tensor factor of `final_state`. In a `counts` key the same
   outcome reads `'10'`.
4. `Physics`, because the calibration's spot checks play the gate under it. `Readout` never touches the gate. `Numerics`
   only changes how the spot checks were integrated, so recalibrate after a large change of its tolerances or caps.
5. Two-sided, per rad/s. A measured heating rate converts through `s_e_from_heating_rate` (single-sided), halved.
6. More branches of the initial mixture than worker processes: `estimate` is one pass, and the run makes a pass per
   branch (`result.diagnostics.branches` says how many). On an unpinned machine, a run at another seed or with other
   numerics than the estimate also recalibrates first.
7. θ = π/2, in radians (IonQ writes 0.25 turns).

</details>

## 15. Final task

Do this without looking back at the tutorial, the next day if you can. Build a three-ion 171Yb+ machine whose 3 MHz
centre-of-mass mode heats at 500 quanta/s, with the field noise uncorrelated between ions. Pin its calibration for the
pairs a GHZ circuit on qubits 0, 1 and 2 uses, and estimate the run. Then
run it with an ensemble that finishes in under a minute, and report the register fidelity next to the intrinsic budget.
Finish by explaining, in your own words, one entry of `diagnostics.approximations`.

A correct solution:

- converts 500 quanta/s with `s_e_from_heating_rate` at that mode's angular frequency and halves it into
  `white_spectrum(..., "(V/m)^2/(rad/s)")`, with `correlation_length_m=0.0`;
- passes the noise model to `trap.presets.yb171_chain(3, noise=...)`;
- sets `Numerics(samples=..., ntraj=..., branch_weight_min=...)` before calibrating, and pins with
  `calibrated(pairs=[(0, 1), (1, 2)])`;
- checks `estimate(...)` (level JOINT_EXACT, the space, one pass's time) before running;
- runs with `keep_final_state=True` and compares `register_fidelity(result)` with
  `result.diagnostics.intrinsic_budget.total`, expecting the noisy infidelity to exceed the quiet budget;
- names the approximation and the physics it leaves out or bounds.

## 16. Review plan and where to go next

Memory for conventions fades quickly, and retrieving them again at spaced intervals is what makes them last. To keep
this for about three months, answer Section 14 again without the tutorial about **two weeks** from now, then roughly
**every three weeks** (around days 22, 45 and 67). If you have to guess an answer, reread only that section.

Where to go next:

- [api.md](api.md): every public name, module by module, one line each.
- [examples.md](examples.md): the terse tour of the main path, also executed by the tests.
- [PLAN.md](../PLAN.md): the specification. Section 1.3 lists what the simulator does not do, Section 5 the numerics,
  Section 13 one convention per quantity, and Section 11 the measured performance.
- [`qutip_trap_app`](../qutip_trap_app/README.md): the same runs at five zoom levels, from the histogram down to the
  Hamiltonian terms, in a desktop window or the browser.
