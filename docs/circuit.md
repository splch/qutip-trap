# Rung 1: the circuit

`qutip_trap.circuit` is what a program is, what the compiler does to it and what it should compute; `qutip_trap.io` holds
the two wire formats. `Machine.compile(circuit)` is the way here from rung 0 and `Machine.schedule` the way down to
[schedule.md](schedule.md).

## The IR

A `Circuit` (PLAN.md Section 3.3) is `n_qubits` labels 0 to n - 1, the `Operation` tuple in time order (a gate name, the
qubits in the gate's own order, the parameters in radians), the terminal `measure` targets and the classical `registers`
(name to qubits in bit order; the OpenQASM 2 importer fills them from the `creg` declarations and the IonQ v2 result
exporter reports `output_all` beside them). It is also a persistent builder: one method per gate name, each returning a new
circuit with the operation appended, `measured(*qubits, registers=)` for the terminal targets. The gate tables:

- `NATIVE_GATES`: `gpi`, `gpi2`, `ms`, `zz`, `rz` (a virtual frame rotation) and their parameter counts, the set the
  scheduler plays;
- `STANDARD_GATES`: `h`, `x`, `y`, `z`, `s`, `t`, `sx`, `rx`, `ry`, `cnot`, `cz`, `swap`, `u3`, `rxx`, `rzz`, `cp` and
  the rest of the qelib1 vocabulary the compiler rewrites;
- `NON_UNITARY`: `measure`, `reset` and `recool`, which pass through the compiler untouched;
- `EXPORTED_NATIVE`: the native names IonQ's JSON accepts, and `GATE_PARAMETERS`: the builder's parameter names per gate.

## The compiler

`Machine.compile(circuit)` (Section 7.2) rewrites the standard gates into native gates with phase tracking (every `rz` is a
frame shift applied to the phases of the later pulses) and verifies every block and the whole circuit against the target
unitary; the `CompileReport` carries the native `circuit`, the pulse and entangling counts, the per-gate and whole-circuit
residuals and the target unitaries the schedule and the GATE_LOCAL summaries are compared with. `compile_to_native` is the
plain function, `compile_with_report` its deprecated 0.1.0 report form, and `CompileError` what an unsupported gate or an
unverifiable block raises. `ideal_probabilities(circuit)` and `circuit_unitary(circuit)` are what the circuit should
compute, `gate_matrix(op)` one operation's matrix. Two-qubit synthesis is `kak_decomposition` (the `KAK` record of a
two-qubit unitary: the local unitaries and the three interaction angles), `decompose_two_qubit_unitary` (the native
gates that realize it) and `haar_random_unitary` (the random SU(4) layers of the quantum-volume protocol).

## The native gate matrices

The exact matrices of `control/native.py`, in radians: `gpi`, `gpi2`, `ms` (θ = π/2 fully entangling), `zz`, `rz`, `r_phi`
(R(θ, φ) with θ/2 in the exponent), `xx` (χ with no ½, maximally entangling at π/4), `equal_up_to_global_phase` for the
comparisons, and `rad_from_turns` and `turns_from_rad` at the IonQ boundary (1 turn = 2π). The tensor order is stated
once, in that module's docstring, with the check that `ms` is the SWAP conjugate of qiskit-ionq's matrix.

## The wire formats: `qutip_trap.io`

`qasm2`: `loads(text)` reads OpenQASM 2 (the qelib1 gates, the native gates declared as gate definitions, `measure` into the
named registers) and `dumps(circuit, declare_native=)` writes it, with `NATIVE_DECLARATIONS` the qelib1.inc-style
definitions of the native gates and `QELIB_NAMES` the map from this package's names to qelib1's (`cnot` to `cx`).

`ionq`: `loads(obj)` reads an IonQ circuit input (native or qis gate set, turns converted to radians) and `dumps(circuit)`
writes the native input; `load_job(obj)` reads a whole v0.3 or v0.4 job body into an `IonQJob` (the circuit with the
backend, shots, name, metadata, noise and settings) and `dump_job(circuit, backend=, shots=, noise=, settings=, name=, metadata=, dry_run=)` writes the v0.4 body of type `JOB_TYPE`, whose allowed keys are `JOB_KEYS`, `NOISE_KEYS` and
`SETTINGS_KEYS` (the spec sets `additionalProperties: false`). `load_ionq_json`, `dump_ionq_json` and `load_openqasm2` are
the 0.1.0 names of the same three readers and writers, on `qutip_trap.io` and on the Appendix E surface; the same four
operations are `Circuit.from_openqasm`, `from_ionq`, `to_openqasm` and `to_ionq` on the type.

```python
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
circuit = trap.io.qasm2.loads(qasm)                     # a Circuit with registers {"c": (0, 1)}
body = trap.io.ionq.dump_job(circuit, backend="simulator", shots=2000)   # the v0.4 job body
assert trap.io.ionq.load_job(body).circuit.ops == trap.io.ionq.loads(body["input"]).ops
```
