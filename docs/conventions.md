# Conventions the simulator fixes

Every physical quantity qutip-trap handles has at least two conventions in the primary literature, and the research runs
behind [`PLAN.md`](../PLAN.md) found published factor-of-2 (and 2π, and √N) errors that trace to mixing them. Section 13 of the
plan adopts one convention per quantity, names the source it follows, and lists the alternatives; every module that touches
a quantity restates the row it implements in its docstring, and the tests pin the conversions. The `conv.*` records of
[`docs/provenance/ledger.yaml`](provenance/ledger.yaml) hold the rows the implementation encodes.

## The rules that reach the public API

The units, the bit order, the tensor order and the seed invariants come first because every other page assumes them.

- **Frequencies.** Every public number is an ordinary frequency in Hz (`qutip_trap.units.Hz`); every internal frequency is
  angular in rad/s (`RadPerS`); the conversion is one explicit 2π at the boundary (`rad_s_from_hz`, `hz_from_rad_s`). A
  linewidth `Transition.gamma_hz` is the upper level's total decay rate divided by 2π; the saturation intensity uses the
  angular partial rate of the cycling branch.
- **Rabi frequency.** H = (ħΩ/2) σ₊ e^{…} + h.c., so a carrier π pulse takes Ωt = π and a π/2 pulse Ωt = π/2. Sources that
  write P = sin²(Ωt) quote half this Ω; the ingest maps in `light/` double them.
- **Detuning.** δ = ω_drive − ω_transition, red negative. The beat-note detuning of a bichromatic gate is μ, measured from the
  carrier, and δ_{i,m} = μ_i − ω_m is always two-indexed.
- **Lamb-Dicke parameter.** η_{i,m} = (Δk · ê_m) c_{i,m} √(ħ/(2 m_i ω_m)) with the ion's own mass and the mass-weighted
  unit-norm eigenvector component; the micromotion factor C₀ = 1 + 3q²/16 + O(q⁴) is applied once, in `Crystal.lamb_dicke`.
- **Gates.** Parameters are radians internally and IonQ turns at the API boundary (1 turn = 2π). GPi(φ), GPi2(φ), MS(φ₀, φ₁,
  θ) with θ = π/2 fully entangling and ZZ(θ) are the matrices of `control/native.py`; R(θ, φ) = exp[−iθ σ_φ/2] carries θ/2 in
  the exponent and XX(χ) = exp(−iχ σ_x σ_x) carries χ with no ½, maximally entangling at χ = π/4. A virtual RZ(θ) shifts
  every later pulse phase by φ → φ − θ, gates read in time order.
- **Result bit order.** In every bitstring key of a `Result`, qubit 0 is the least-significant bit, the rightmost character:
  "101" on three qubits is qubit 0 = 1, qubit 1 = 0, qubit 2 = 1, and the IonQ v1 decimal key "5". This sentence is stated
  once, here; `tests/test_docs.py` checks that no other page restates it and that no docstring contradicts it. IonQ's v2
  result strings run the other way, `q[0]` first (`io/ionq.py` records the source). `Result.final_state` is in QuTiP's tensor
  order (ion 0 the first factor); `run.job.to_register_order` converts.
- **Tensor order of the native gate matrices.** Stated once, in the module docstring of `control/native.py`, with the check
  that `ms` is the SWAP conjugate of qiskit-ionq's matrix (`tests/test_docs.py`, when qiskit-ionq is installed).
- **Fidelity measures, three of them, kept apart.** The compiler and the benchmarks compare unitaries up to a global phase.
  (i) A channel's *average gate infidelity* is 1 − F_avg = d/(d + 1) (1 − F_e) with F_e the entanglement fidelity, and the
  *depolarizing rate* ε of Section 6.8 equals the entanglement infidelity in Chen et al. 2023's normalization
  Λ_ε(ρ) = (1 − ε)ρ + ε/(4ⁿ − 1) Σ_{P≠I} PρP — the same channel as Trout 2018's uniform p/3 and p/15 Kraus weights, with
  Qiskit's λ = p 4ⁿ/(4ⁿ − 1) as a conversion and never as the same number (`conv.depolarizing_normalization`).
  (ii) Randomized benchmarking reports r = (1 − p)(2ⁿ − 1)/2ⁿ, the average error per Clifford (r = 1 − F_avg of the Clifford
  channel), beside the entanglement infidelity (4ⁿ − 1)(1 − p)/4ⁿ of the depolarizing channel with the same p, which is kept
  under that name and never called the RB error rate; simultaneous RB fits each qubit's own marginal decay rather than
  applying the n-qubit formula to the joint one (`conv.simultaneous_rb_units`).
  (iii) For a single-qubit *unitary* error the composite-pulse and filter-function layers use F_K = ¼|Tr(U_id†U)|², with F_C
  and F_avg as conversions (`conv.gate_fidelity_measure`). A decay base, a channel rate and a unitary overlap are three
  different quantities and the plan's Section 13 asks for three rows; so does this page.
- **Readout figure of merit.** ε = ½(ε_B + ε_D), both reported: ε_B the probability that a bright ion reads dark, ε_D that a
  dark ion reads bright. For 171Yb+ the qubit |0⟩ (F = 0) is dark and |1⟩ (F = 1) bright.
- **Noise spectra.** Every spectrum the noise layer stores is two-sided in angular frequency with the e^{−iωt} kernel; the
  single-sided S_E of the heating literature (Γ_h = e² S_E/(4mħω)) is converted at ingest. Qubit dephasing enters as
  L = √(γ_φ/2) σ_z, under which the coherence decays at γ_φ, so γ_φ = 1/T₂ for the white component.
- **Seeds.** One root `SeedSequence` per run, spawned by (sample, trajectory, shot, ion, channel), so every random variate is
  independent of execution order, of the worker count and of truncation retries; the 1-versus-18-worker agreement is a
  tolerance test at 10⁻¹² (Section 3.4). The contract, exactly as `run/pipeline.py` has it: the same
  (`shots`, `samples`, `seed`, machine hash) reproduces every shot bit for bit, whether the run went through `Machine.run`,
  `run` or a `Job` in a worker process, because the root seed and the machine fix every keyed stream and the shots are dealt
  to the dynamical samples in contiguous blocks (`conv.shot_blocks_per_sample`: sample k owns the shots from Σ_{j<k} M_j to
  Σ_{j≤k} M_j, with M_j = shots // n_samples plus one for the first shots mod n_samples samples; `Result.sample_of_shot`
  reads the blocks back). A different `shots` does NOT reproduce the shots the two runs share: with `samples=None` the
  sample count is min(shots, 64) and the block sizes move with it, so a shot lands in a different sample with different
  quasi-static values, and the per-shot draws (the photon records, the collisions) are keyed by the sample id and the shot
  index inside its block. Integrator and cap-rule changes may move sampled outcomes between minor releases, because the
  trajectory and readout draws follow the state they are drawn from; a record is therefore compared across releases by its
  histogram within its error bars and by its diagnostics, never shot list against shot list.
- **Identity.** `Device.hash()` is a canonical serialization (declaration-order fields, floats rounded to 12 significant
  digits, dicts by sorted key, `Qobj` fields excluded), the key of the calibration cache and of every cached tomography.

## Vocabulary

Settled on 2026-09-11 for the API work of [`api_implementation_plan.md`](api_implementation_plan.md) (Phase 0, item 0.6). Every
later phase uses these names; a name here changes only through a deprecation cycle.

- **Machine.** The executor that 0.2.0 adds: a `Device`, the roles its beams play, the calibration table and the policy that
  turns a circuit into a `Result`, with `run`, `compile`, `schedule` and `calibrated`. PLAN.md Section 14.2 already names
  Level 0 "Machine" and the app's device card uses the word. Rejected: `Emulator` (Pulser, Quantinuum), because the object is
  the machine as the physics describes it, not an imitation of one; `Backend` (Qiskit, pytket), because a backend is an
  execution service, and this is a record with behaviour and no service behind it. `Device` stays the physical record, the
  split pytket made between `BackendInfo` (data) and `Backend` (behaviour).
- **`trap`.** The recommended import alias is `import qutip_trap as trap`; the documentation's examples use it from 0.2.0 on.
  `qutip_trap.api` stays the spelled-out Appendix E import and is never aliased.
- **Rungs.** The five levels of PLAN.md Section 14.2 are five modules: `qutip_trap` (rung 0, the machine), `qutip_trap.circuit`,
  `qutip_trap.schedule`, `qutip_trap.dynamics` and `qutip_trap.physics`; `qutip_trap.interop` holds the adapters to other SDKs
  (`qutip_trap.interop.qiskit` today) and `qutip_trap.experimental` the names outside the stability guarantee. The experiments,
  calibration and benchmarks packages are "the laboratory". In API prose a step of the ladder is a **rung**, so that "level"
  keeps its two other meanings; the app's "Level 0" to "Level 4" labels are the same five rungs.
- **Fidelity levels.** `JOINT_EXACT` and `GATE_LOCAL` are the two levels a run can integrate at (Section 5.4), and `"auto"` the
  policy that picks one against the Section 11.5 guards. The enum that replaces the strings in 0.2.0 is named `FidelityLevel`,
  with members `AUTO = "auto"`, `JOINT_EXACT` and `GATE_LOCAL` equal to today's strings (so `Diagnostics.level == "GATE_LOCAL"`
  stays true); it widens the `Literal` alias of that name in `run/levels.py`. The proposal's `Level` is rejected because the
  word is taken twice: `qutip_trap.api.Level` is an atomic level (Appendix E, frozen), and "Level n" is a rung in the app.
- **Error-model names** (`error_model()`, 0.3.0): `p_*` a probability per operation, `*_rate` per second, `*_ratio` a fraction
  of another probability, `*_scale` a dimensionless multiplier. They join the unit suffixes every public float carries: `_hz`,
  `_s`, `_m`, `_w`, `_gauss`, `_rad`, `_v`, `_pa`, `_cps`, and `_turns` only at the IonQ boundary.
