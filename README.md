# qutip-trap

![qutip-trap](docs/qutip-trap-logo-light.svg)

A first-principles trapped-ion quantum computer simulator built on [QuTiP](https://qutip.org). It takes a circuit and returns counts the way a trapped-ion QPU does: not by multiplying gate matrices and bolting on an error model, but by simulating the physics that produces them.

A run holds the ions in a trap and solves for their crystal and motional modes, cools and optically pumps them, compiles the circuit to the native gates (GPi, GPi2, Mølmer-Sørensen, ZZ, virtual RZ), schedules laser pulses from a calibration performed on the simulated device, integrates every pulse on the joint qubit-plus-motion state under the spin-motion Hamiltonian with noise channels whose rates follow from the device parameters, and reads the ions out by state-dependent fluorescence. Every `Result` states which fidelity level ran, which Hilbert space was integrated and every approximation that was made.

The Flet application runs in the browser at https://splch.github.io/qutip-trap/ (Python and QuTiP compiled to WebAssembly; the first load fetches about 60 MB).

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/). The package is not on PyPI yet.

```sh
uv add git+https://github.com/splch/qutip-trap
```

Extras: `qiskit` (a Qiskit `BackendV2`), `gui` (the Flet application).

## Quickstart

```python
import qutip_trap as trap

machine = trap.presets.yb171_chain(2)        # a two-ion 171Yb+ chain with 355 nm Raman gates
bell = trap.Circuit(2).h(0).cnot(0, 1)       # every qubit is measured unless .measured(...) narrows it
result = machine.run(bell, shots=2000)       # compile, calibrate, schedule, prepare, evolve, read out

print(result.counts)                         # about 1000 each of '00' and '11', a few of '01' and '10'
print(result.diagnostics.level)              # JOINT_EXACT
print(result.diagnostics.approximations)     # what the run approximated, in words
```

A two-ion Bell circuit takes a few seconds on a laptop, calibration included. A `Machine` is a frozen record of a `Device`, its calibration table, the option objects `Physics`, `Numerics` and `Readout`, and a `FidelityLevel`; variants come from `dataclasses.replace(machine, physics=trap.Physics(noise=False))`. Small crystals run `JOINT_EXACT` on the full joint space, and larger ones switch to `GATE_LOCAL`, where each gate is exact on the ions it addresses and the modes they couple to.

### From Qiskit

```python
from qiskit import QuantumCircuit, transpile
from qutip_trap.interop.qiskit import QutipTrapProvider

backend = QutipTrapProvider().get_backend("yb171_chain", n_ions=2)
bell = QuantumCircuit(2); bell.h(0); bell.cx(0, 1); bell.measure_all()
counts = backend.run(transpile(bell, backend), shots=2000).result().get_counts()
```

Circuits also load from OpenQASM 2 and IonQ JSON (`trap.Circuit.from_openqasm`, `trap.Circuit.from_ionq`), and a `Result` exports to IonQ's v1 and v0.4 formats.

## What is in the box

- **The machine and the levels below it.** `Machine.run` takes a circuit to counts; `Machine.compile`, `Machine.schedule`, `Machine.engine` and `Machine.device` open the verified compiler, the pulse schedule with its calibration table, the Hamiltonian builder with the JOINT_EXACT engine, and the physical records (species, trap and crystal, beams, noise, detector, preparation), each importable on its own.
- **The laboratory.** `qutip_trap.experiments` runs Rabi, Ramsey, sideband, thermometry, heating-rate, Stark, crosstalk, field, micromotion, detection, MS and parity scans on the machine through the same engine; `qutip_trap.calibration` fits them into a table or builds the closed-form surrogate; `qutip_trap.benchmarks` runs randomized benchmarking, GHZ fidelity and quantum volume with the simulator's own error budget beside each number.
- **The error model.** `machine.error_model()` derives per-gate infidelities, durations and SPAM errors from the simulation and exports them in IonQ's, Quantinuum's and the QDK estimator's vocabularies.
- **Devices.** Two example machines, `yb171_chain` (hyperfine qubit, Raman gates) and `ca40_optical` (optical qubit on the 729 nm line), with atomic data for 171Yb+, 40Ca+, 43Ca+, 137Ba+, 9Be+ and 88Sr+. The example numbers are a realizable laboratory configuration, not a published apparatus.
- **The app.** [`qutip_trap_app`](qutip_trap_app/README.md) shows one run at five zoom levels, from the histogram down to the Hamiltonian terms being integrated.

## Documentation

- [PLAN.md](PLAN.md) is the specification: the scope and what the simulator does not do (Section 1.3), the physics (Part II), the numerics, noise, control and readout with the governing equations in one place (Part III, Section 5.7), the validation targets (Section 9), the open physics (Section 12) and one convention per quantity (Section 13).
- [docs/api.md](docs/api.md): the public API, module by module.
- [docs/examples.md](docs/examples.md): a runnable tour from a device to the error model, executed by the test suite.
- [docs/provenance/ledger.yaml](docs/provenance/ledger.yaml): the provenance record every derived number and every provenance chip names.

## Development

```sh
git clone https://github.com/splch/qutip-trap && cd qutip-trap
uv sync --group dev
uv run pytest -n 4 --dist loadscope -m "not slow and not heavy"
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

CI runs the fast tier on every push and pull request, and the whole suite, the slow tier and the heavy tests included, on a nightly schedule and on manual dispatch. The app has its own commands in its README.
