# The laboratory: experiments, calibration, benchmarks

Three packages run the laboratory's protocols on a machine through the same engine as a circuit: `qutip_trap.experiments`
(the scans with their fits), `qutip_trap.calibration` (the surrogate and the simulated-experiment calibration) and
`qutip_trap.benchmarks` (the standard protocols with the simulator's own error budget). Since 0.3.0 every function takes a
`Machine` first and reads its table and option objects; a bare `Device` is wrapped in a default machine and warns since
0.4.0 (docs/deprecations.md).

## Experiments

Each experiment returns its typed result, an `ExperimentResult` with the fitted parameters as attributes, `value(key)` and
`uncertainty(key)`, `converged`, the fit's `chi2`, a `quality` verdict ("good", "poor", "failed" or "exact"), the scan as
`requested` beside the scan as `realized` (`ScanParameters`, attribute access by name; `realized_drive` reads the tone
words the hardware chain plays), the `subject` as the table keys it, and `table_updates` for the proposal a
`CalibrationTable.updated_with` adopts; `plot(ax=)` needs matplotlib. `RESULT_TYPES` maps each experiment to its type:

| Experiment | Result | What it fits |
|---|---|---|
| `rabi_scan(machine, ion, durations_s)` | `RabiScan` | the carrier Rabi frequency, the mean occupation, the contrast |
| `ramsey(machine, ion, delays_s)` | `RamseyFringe` | the fringe frequency and contrast, the phase offset |
| `ramsey_frequency(machine, ion, delays_s)` | `RamseyFringe` | the qubit frequency by two probes of opposite sign |
| `sideband_spectroscopy(machine, ion, detunings_hz)` | `SidebandSpectrum` | the carrier and sideband frequencies, the mode frequency, η and the occupation |
| `thermometry(machine, ion, mode)` | `ThermometryResult` | the mean occupation by the sideband ratio |
| `mode_spectroscopy(machine, ion, mode)` | `SidebandSpectrum` | a mode frequency by a coarse then a fine sideband scan |
| `heating_rate(machine, mode, delays_s)` | `HeatingRateFit` | the heating rate and the initial occupation |
| `ms_scan(machine, pair, amplitudes, detunings_hz)` | `MSScan` | the closure offset and scale, the entangling angle per unit amplitude |
| `ms_phase_scan(machine, pair, phases_rad)` | `MSScan` | the phase of the entangling gate against the analysis pulse |
| `parity_scan(machine, pair, analysis_phases_rad)` | `ParityScan` | the parity contrast and the Bell-fidelity bound |
| `detection_histogram(machine, ion, n_records)` | `DetectionHistogram` | the threshold, the window and the readout errors |
| `stark_scan(machine, ion, delays_s)` | `StarkScan` | the differential Stark shift per beam |
| `crosstalk_scan(machine, ion, durations_s)` | `CrosstalkScan` | the addressing crosstalk on the neighbours |
| `field_scan(machine, ion, delays_s)` | `FieldScan` | the magnetic field from the qubit frequency |
| `micromotion_scan(machine, ion, beam, shim_ranges_v)` | `MicromotionScan` | the compensation shims by rf-photon correlation or the sideband ratio |
| `crystal_image(machine)` | `CrystalImage` | the ions seen bright, dark and lost |

The fitting layer is public: `weighted_fit` (the least-squares fit with the declared shot-noise weights, a `FitResult`
with the parameters, their uncertainties and the reduced chi-square), the models `thermal_rabi_model`,
`thermal_rabi_model_fixed_nbar` and `half_rabi_lineshape` (a Rabi flop averaged over a thermal distribution), the
lineshapes `lineshape_model`, `sideband_lineshape` and `fit_lineshape` of the spectroscopy scans, `correlation_signal` and
`signed_beta` of the rf-photon-correlation micromotion measurement, and `periodic_scattering` (the scattering rate under a
micromotion-modulated detuning). An `Observation` is one scan point with its shots and the readout applied,
`ReadoutErrors` the readout errors an experiment declares and `readout_errors_for(machine, ion)` the ones the table implies;
`device_with_compensation(device, shims)` is the device with a micromotion scan's shims programmed.

## Calibration

`calibrate(machine, method="closed_form" | "experiments", experiments=, seed=, t0_s=, cache=, refresh=, **scans)` returns
the `CalibrationReport` whose `table` the scheduler reads (`CalibrationMethod` is the two words): the Section 7.5 surrogate
(the closed forms with exact spot checks; `surrogate_table` and its `SurrogateReport` are the function and record behind
it) or the M8 simulated experiments in the dependency order (`full_calibration`, a `CalibrationRun` per experiment with
its `CalibrationScans` settings, `EXPERIMENTS` the names, `ORDER` the dependency order, `UPSTREAM` what each depends on,
`ALIASES` the accepted spellings, `upstream_status` the readiness of an entry's inputs). `Machine.calibrated(method=, **scans)`
pins the result on the machine. Reports are cached per (device hash, seed, arguments) in a `CalibrationCache`
(`DEFAULT_CACHE` is the process-wide one); a device whose hash changed never hits a cached table. `CalibrationError` is
what a scan that cannot be fitted raises; `calibrate_entangling_angle` and `exact_gate_check` (a `GateCheck` against the
exact gate on the `gate_space`) are the spot checks of the surrogate, `frame_rotated` the ideal target as the physical
state carries it after the scheduler absorbed a frame offset per ion (Section 7.6), `thermal_robustness` the check of a
waveform over the thermal distribution, and `calibrate_with_report` the deprecated 0.1.0 name of the report on a device.

```python
report = calibrate(machine, method="experiments", experiments=("detection_histogram", "rabi_scan"), pairs=[(0, 1)])
report.table.rabi[(0, 2)].status                # "calibrated": set by the simulated scan, not the closed form
```

## Benchmarks

`randomized_benchmarking(machine, qubits, lengths, n_sequences=, shots=, variant=, pair=, budget=)` runs Clifford RB:
single-qubit, simultaneous on several ions with `pair=False`, two-qubit on a pair, or Knill-style with `variant="knill"`.
It returns an `RBResult`: the survival per length, the fit, `error_per_clifford`, the per-ion `marginal_error_per_clifford`
and `joint_error_per_layer` of simultaneous RB, the `RBSequence` records played and the `BenchmarkBudget` when asked (the
channel of every native kind by process tomography, the Section 9.6 scales and the SPAM, composed into the predicted r).
`ghz_fidelity(machine, qubits, shots=, analysis_phases_rad=)` returns a `GHZResult` (the populations, the parity contrast,
the fidelity bound and the exact register fidelities). `quantum_volume(machine, qubits, n_circuits=, shots=)` returns a
`QVResult` over `QVCircuit` records (the heavy-output probability against its ideal, Eq. (32)'s σ, whether the protocol
passed). The circuits are public: `ghz_circuit`, `parity_circuit`, `random_square_circuit`; the Clifford group is
`SINGLE_QUBIT_CLIFFORDS`, `TwoQubitClifford` with `two_qubit_clifford_group`, `random_two_qubit_clifford` and
`decompose_two_qubit_clifford`, and `CLASS_SIZES` and `TWO_QUBIT_GROUP_ORDER` (11520) its bookkeeping.

`gate_channel(machine, kind)` is the Section 6.8 channel of one native gate kind ([dynamics.md](dynamics.md)), a
`GateChannel` whose `StepChannel` entries are the steps of the gate played once; the channels are cached per machine hash
and kind, and `clear_budget_cache()` empties the cache.

The inverse direction is `Machine.error_model()`, the module function `error_model(machine, qubits=)`: an `ErrorModel` with
the average gate infidelity and the duration per native gate kind, the depolarizing weights `p_1q` and `p_2q`, `p_meas` and
`p_init` per qubit, the dephasing and heating rates from the noise model, and the exporters `to_ionq_noise` (`r_1q`, `r_2q`
as the maximally-mixed weights), `to_quantinuum_error_params` and `to_qdk_qubit_params`, each stating its conversion.
