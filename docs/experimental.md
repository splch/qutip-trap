# The experimental namespace

`qutip_trap.experimental` gathers the names outside the stability guarantee (docs/api_proposal.md Section 4.11): anything
importable from here may change or disappear in a minor release without the deprecation cycle of
[deprecations.md](deprecations.md), Mitiq's rule for its own experimental namespace ("not covered by semantic versioning
guarantees"). The stability is gated by the import path, not by the object: the same objects stay importable from
`qutip_trap.api`, the frozen Appendix E surface, which never changes under them. Three groups live here until each has a
second consumer or a settled design.

## Transport (milestone M12, specification only)

PLAN.md Section 4.6's transport, splitting, merging and junctions are specified so that the frozen interfaces need not
change when the milestone is picked up; nothing in the released milestones depends on them and their methods raise
`NotImplementedError` naming M12. A `Zone` is a trap region with its electrodes, a `VoltageWaveform` the electrode voltages
against time with the `FilterStage` records of the low-pass chain that shapes them, a `Transport` one move of an ion between
zones, and a `TransportBudget` the motional excitation and the time it costs; `design_waveform` designs the waveform of a
move, `split_feasible` says whether a split is within the electrodes' reach, and `transport_budget` accounts for a whole
sequence of moves.

## Tomography internals

`choi_least_squares(inputs, outputs)` reconstructs a Choi matrix from the input and output states of state-based process
tomography by least squares, and `project_cptp(choi)` projects it onto the completely positive, trace-preserving set by
Dykstra's alternating projection (Section 5.4). The GATE_LOCAL walk uses both behind `JointExactEngine.tomography`; the
application uses neither. `qutip_trap.dynamics` exported them in 0.2.0 and 0.3.0 and warns for them since 0.4.0.

## Two oracles

`filter_function(sequence, omega_rad_s)` is the filter-function formalism of Section 6.9 for a `DecouplingSequence`
([schedule.md](schedule.md)): the spectral weight a dephasing spectrum is integrated against. `frozen_excitation_bounds(device, pulses, frozen_modes, nbar)` is the Section 5.2 bound on the off-resonant excitation of the frozen spectator modes of a
step, which the GATE_LOCAL report sums into its discrepancy bound; both are numbers a run reports against rather than
integrates.
