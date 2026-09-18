# Rung 2: the schedule

`qutip_trap.schedule` is the pulses a circuit becomes on the time axis and everything they are made of.
`Machine.schedule(circuit)` is the way here from [circuit.md](circuit.md) (the compile-calibrate-schedule prefix of a run,
IonQ's dry run: nothing integrated) and `Machine.engine` the way down to [dynamics.md](dynamics.md).

## The prefix of a run

`compile_calibrate_schedule(circuit, device, table=, seed=, t0_s=, options=, ...)` is the prefix `Machine.run`,
`Machine.schedule` and `Machine.estimate` share: it returns a `Prefix` with the `CompileReport`, the native circuit, the
`CalibrationTable` used (the cached closed-form surrogate when none was given), the device with the calibrated micromotion
shims programmed, the `Schedule`, the resolved drive maps, the solver options and the notes.

## Pulses, drives, tones

A `Schedule` is the pulses with absolute times, the idle intervals, the `ScheduledEvent` tuple (the measurement event) and
the `PlayedGate` records that tie each pulse group back to a `GateTarget` (the compiled gate, its qubits and its ideal
unitary). A `Pulse` plays one `Drive` (the beams and their geometry: a Raman pair, an optical E1 or E2 beam, a microwave or
gradient drive, a light-shift pair) for a duration with `Tone` entries, each a frequency, an envelope and a phase; the
scheduler `schedule(circuit, device, table, ...)` builds them from the native gates and raises `ScheduleError` where a
gate has no drive or no calibrated waveform. `LightShiftCouplings` is the record `derive_light_shift_drive` fills for the
Section 4.4.4 light-shift force, and `derive_raman_drive` derives the Raman drive of a beam pair (the two-photon Rabi
frequency, the differential Stark shift, the crosstalk on the neighbours) from the device.

## The beam roles

`GateDrive` names which beams play a gate on an ion; `Device.roles` (a `BeamRoles`, [physics.md](physics.md)) carries the
single-qubit, entangling and detection roles, and the scheduler reads them through `resolve_drives(device, gate_drives, entangling_drives)`, which lets an explicit map override the roles. `default_gate_drives(device)` and
`infer_gate_drives(device)` are the inference for a device that declares no roles: one Raman pair per ion, or the one pair
of the device, from the wavelengths (the "none, one, or one Raman pair" rule of `light/roles.py`).

## The calibration table

A `CalibrationTable` is what the scheduler reads: the `CalEntry` records (a value, its uncertainty, a status, the
experiment that set it and its provenance id) for the Rabi frequencies, the qubit frequency, the Stark shifts, the
sideband frequencies, the crosstalk ratios, the detection threshold and the field, and the calibrated `Waveform` of every
entangling pair, a `Segment` sequence with its closure angles. Since 0.3.0 an edit is a proposal: `with_params` merges new
values as fresh entries, `updated_with(result)` adopts an experiment result's `table_updates`, `kind_of(key)` and
`CalEntry.kind` tell a setpoint (a value the electronics are programmed with) from a characterisation (a measured fact),
with `EntryKind` and `ENTRY_KINDS` the two words. The table's `device_hash` ties it to the device it was made for.

## The electronics

The `HardwareChain` of Section 7.10 (the DDS phase and amplitude words, the modulator rise time, the amplifier bandwidth,
the dead time) is applied by `apply_hardware_chain(schedule, device)` before a schedule is integrated;
`physical_schedule(schedule, device, table)` is the played chain of Section 7.3 that converts every requested Rabi
frequency, believed Stark shift and believed crosstalk into what the ions see through the device's derived values.

## Waveforms and their closure

The three closure solvers of Section 4.4.3 design an entangling waveform for the `GateModes` of a pair (`gate_modes(device, pair, drive)`: the coupled modes with their Lamb-Dicke parameters and frequencies): `solve_amplitude_modulation`,
`solve_fourier_amplitude_modulation` and `solve_frequency_modulation`, each returning a `ShapedPulse` (the segments, the
closure residuals, the geometric phase). Since 0.4.0 the closed-form trajectory behind a played waveform is public too:
`envelope_of(waveform, ions)` reads the waveform as an `Envelope`, a `SegmentedEnvelope` (piecewise-constant amplitudes and
detunings) or a `SampledEnvelope` (the hardware chain's sampled amplitude and phase), `integrals_segmented(envelope, modes)`
gives the `GateIntegrals` (the spin-branch displacement α per ion and mode, the geometric phase χ per pair), and
`trajectory_sampled(envelope, modes, times)` the loop α(t) the Level 3 view draws; `closure_rabi_rad_s` and
`closure_duration_s` are the closure algebra of the symmetric square pulse (the Rabi frequency and the duration that close
one loop at a detuning) and `CHI_MAXIMAL_RAD` the maximally entangling angle π/4.

## Composite pulses, decoupling, comb drives

`CompositePulse` and `composite_pulse(name, ...)` are the library of Section 4.3.5 (BB1, SK1, CORPSE and the rest, each a
sequence of rotations that cancels an amplitude or detuning error to a stated order); a `DecouplingSequence` from
`decoupling_sequence(name, n, duration_s)` is a dynamical-decoupling sequence (CPMG, XY8, UDD) as pulses on the time axis,
whose filter function is `filter_function` on [experimental.md](experimental.md). A `CombSpec` specifies a frequency-comb
Raman drive (Section 4.3.7): the repetition rate, the pulse duration and its convention, the beat-note order.
