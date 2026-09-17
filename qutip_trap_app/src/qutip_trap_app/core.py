"""The core contract: every import of ``qutip_trap`` the application makes, in one module (PLAN.md Section 14.6).

Section 14.6: "The app calls only public core API that exists for the core's own reasons ... If a zoom view needs data the
core does not expose, the app records the gap and shows the view as unavailable rather than patching the core." The public
surface is ``qutip_trap.api`` (Appendix E). Everything the application uses from the core is re-exported from here, so
that a reader can audit the contract in one place and ``tests/test_provenance_coverage.py`` can assert that no other
module of the application imports ``qutip_trap`` directly.

Two names below come from outside ``qutip_trap.api``. Each is a core feature request (a re-export, nothing more) that the
application works around on its own side, exactly as Section 14.1 asks; they are listed in :data:`CORE_GAPS` and the run
record carries that list, so a reader of an exported record knows which core facilities the app leaned on.
"""

from __future__ import annotations

# ---- the public surface (Appendix E) -------------------------------------------------------------------------------------------
from qutip_trap import FidelityLevel, Machine, Numerics, Physics, Readout, as_machine
from qutip_trap import __version__ as core_version
from qutip_trap.api import (
    Beam,
    CalEntry,
    CalibrationTable,
    Circuit,
    CollapseOp,
    CompileReport,
    Crystal,
    Detector,
    Device,
    DevicePreset,
    Diagnostics,
    Drift,
    Field,
    GateDrive,
    GateModes,
    GateStep,
    GateTarget,
    HilbertSpace,
    MathieuParameters,
    ModeTruncation,
    MotionalModel,
    NoiseModel,
    NoiseSample,
    NoiseSpectrum,
    Operation,
    PlayedGate,
    PreparationRecipe,
    Pulse,
    RecordModel,
    Result,
    RfDrive,
    RunRecord,
    ScatteringOptions,
    Schedule,
    SeedSpec,
    Segment,
    ShapedPulse,
    SolverOptions,
    Species,
    State,
    ThresholdDiscriminator,
    Traces,
    Trap,
    Waveform,
    average_gate_infidelity,
    ca40_optical,
    calibrate,
    choi_from_unitary,
    circuit_unitary,
    collision_rate_per_ion,
    compile_with_report,
    depolarizing_rate,
    derive_raman_drive,
    detection_rates_for_ion,
    entanglement_infidelity,
    gate_modes,
    gate_steps,
    ideal_probabilities,
    input_states,
    kraus_operators,
    last_record,
    load_ionq_json,
    load_openqasm2,
    optimize_threshold,
    pauli_twirl,
    prepare,
    register_fidelity,
    run,
    run_preparation,
    scattering_channels,
    schedule,
    select_space,
    solve_amplitude_modulation,
    solve_crystal,
    species_by_name,
    standard_recipe,
    white_spectrum,
    yb171_chain,
)

# ---- names the core does not re-export through qutip_trap.api (each a filed core feature request, Section 14.1) -----------
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    SampledEnvelope,
    SegmentedEnvelope,
    closure_duration_s,
    closure_rabi_rad_s,
    envelope_of,
    integrals_segmented,
    trajectory_sampled,
)
from qutip_trap.dynamics.engine import JointExactEngine
from qutip_trap.dynamics.hamiltonian import BuilderOptions, BuiltHamiltonian, DriveRecord, build_hamiltonian
from qutip_trap.hilbert.operators import debye_waller_factor, rabi_matrix_element, rabi_table
from qutip_trap.noise.sampling import KEY_BRANCH_WEIGHT, key_frozen_n, key_mode_offset_hz, key_qubit_offset_hz
from qutip_trap.prep.closed_forms import (
    doppler_force_nbar,
    lamb_dicke_parameter,
    stenholm_coefficients,
    x0_m,
)
from qutip_trap.prep.sideband import apply_pulses, mean_occupation, thermal_distribution
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.trap.crystal import axial_modes_dimensionless, equilibrium_dimensionless
from qutip_trap.trap.mathieu import is_stable, monodromy
from qutip_trap.units import ATOMIC_MASS_KG
from qutip_trap.validation.harty_rb import HartyParameters, simulate_epg_sets
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_thermal_error,
    kirchmair_populations,
    ms_alpha,
    ms_gamma,
    thermal_debye_waller_infidelity,
)

CORE_GAPS: tuple[str, ...] = (
    "qutip_trap.api exports the PulseEngine protocol but no concrete engine; the per-pulse evolution entry point Section 14.6 "
    "names (JointExactEngine.run_pulses) is imported from qutip_trap.dynamics.engine [core feature request: re-export "
    "JointExactEngine in qutip_trap.api]",
    "the noise-sample keys run() stamps on every branch (branch_weight, frozen_n[m]) are imported from qutip_trap.noise.sampling "
    "so that a re-simulation rebuilds the same NoiseSample [core feature request: re-export KEY_BRANCH_WEIGHT and key_frozen_n]",
    "Traces carries <n_m>(t) and the final reduced motional states only: a per-time Fock distribution inside a pulse is not "
    "exposed, so the Level 3 Fock heatmap has values at pulse boundaries and re-simulated sub-steps only [core feature request: "
    "an optional per-time mode_marginal store in Traces]",
    "run() returns no wall time per pulse and no per-gate register state for GATE_LOCAL runs; the app measures the run's wall "
    "time itself and derives the GATE_LOCAL register after each gate by composing the recorded step channels in time order "
    "(viewmodel.circuit.gate_local_register_after); the idle intervals' one-qubit channels are not recorded, so they are "
    "taken as the identity there [core feature request: the register after every step, or the idle channels, in the "
    "GATE_LOCAL report]",
    "the Hamiltonian builder of Section 5.7 (build_hamiltonian, BuilderOptions, BuiltHamiltonian, DriveRecord) is not "
    "re-exported: the Level 4 Hamiltonian page imports it from qutip_trap.dynamics.hamiltonian to list the terms the engine "
    "integrates for a zoomed pulse [core feature request: a public per-segment builder entry point]",
    "the analytic matrix elements of Section 4.3.1 (rabi_table, rabi_matrix_element, debye_waller_factor) are not re-exported: "
    "the Omega_{n',n} table of the Hamiltonian page imports them from qutip_trap.hilbert.operators [core feature request]",
    "the Mathieu stability functions of Section 4.1.1 (monodromy, is_stable) are not re-exported: the trap page's stability "
    "diagram imports them from qutip_trap.trap.mathieu [core feature request]",
    "the pulsed sideband-cooling transfer (apply_pulses, mean_occupation, thermal_distribution) is not re-exported: the cooling "
    "page's nbar-after-each-pulse staircase imports it from qutip_trap.prep.sideband; the repump recoil kernel the recipe "
    "applies between pulses is not reachable at all, so the staircase is drawn without it and says so [core feature request]",
    "the quasi-static noise-sample keys (qubit_offset_hz[i], mode_offset_hz[m]) are imported from qutip_trap.noise.sampling so "
    "that the Hamiltonian page can name what a sample's values shift [core feature request: re-export the key builders]",
    "Traces carries no per-time Fock distribution inside a pulse: the Level 3 heatmap re-simulates truncated copies of the pulse "
    "(causality makes the truncated pulse's final state the full pulse's state at that time) as a background job [core "
    "feature request: an optional per-time mode_marginal store in Traces would make it free]",
    "the closed-form gate integrals of Section 4.4.3 behind a played waveform (envelope_of, integrals_segmented, "
    "trajectory_sampled and the two envelope types) are not re-exported: the Level 3 spin-branch loops alpha_im(t) and the "
    "gate page's closed-form angle at the played amplitude import them from qutip_trap.control.shaping [core feature "
    "request: a public trajectory entry point on Waveform]",
    "the validation-suite closed forms the published-experiment presets of Section 14.5 run (Harty's randomized-benchmarking "
    "model of Section 9.2, Kirchmair's thermal populations of Section 9.4, the Doppler force and rate coefficients of "
    "Section 9.3, James's dimensionless equilibrium and axial modes of Section 9.1, the zero-point length and Lamb-Dicke "
    "closed form, the closure algebra constants) and the published readout apparatus presets of Section 8.4 (Myerson, "
    "Crain) and the CODATA atomic mass unit are not re-exported by qutip_trap.api: the app imports them from qutip_trap.validation, "
    "qutip_trap.prep, qutip_trap.trap, qutip_trap.readout and qutip_trap.units [core feature request: a public validation namespace]",
)
"""What the core does not expose (or does not re-export) that the application needs; Section 14.6's record of the gaps."""

__all__ = [
    "FidelityLevel",
    "Machine",
    "Numerics",
    "Physics",
    "Readout",
    "as_machine",
    "Beam",
    "BuilderOptions",
    "BuiltHamiltonian",
    "CHI_MAXIMAL_RAD",
    "ATOMIC_MASS_KG",
    "CORE_GAPS",
    "CRAIN_YB171_SNSPD",
    "CalEntry",
    "CalibrationTable",
    "Circuit",
    "CollapseOp",
    "CompileReport",
    "Crystal",
    "Detector",
    "Device",
    "DevicePreset",
    "Diagnostics",
    "Drift",
    "DriveRecord",
    "Field",
    "GateDrive",
    "GateModes",
    "GateStep",
    "GateTarget",
    "HartyParameters",
    "HilbertSpace",
    "JointExactEngine",
    "KEY_BRANCH_WEIGHT",
    "MYERSON_CA40_PMT",
    "MathieuParameters",
    "ModeTruncation",
    "MotionalModel",
    "NoiseModel",
    "NoiseSample",
    "NoiseSpectrum",
    "Operation",
    "PlayedGate",
    "PreparationRecipe",
    "Pulse",
    "RecordModel",
    "Result",
    "RfDrive",
    "RunRecord",
    "SampledEnvelope",
    "ScatteringOptions",
    "Schedule",
    "SeedSpec",
    "Segment",
    "SegmentedEnvelope",
    "ShapedPulse",
    "SolverOptions",
    "Species",
    "State",
    "ThresholdDiscriminator",
    "Traces",
    "Trap",
    "Waveform",
    "apply_pulses",
    "average_gate_infidelity",
    "axial_modes_dimensionless",
    "ballance_thermal_error",
    "build_hamiltonian",
    "ca40_optical",
    "calibrate",
    "choi_from_unitary",
    "circuit_unitary",
    "closure_duration_s",
    "closure_rabi_rad_s",
    "collision_rate_per_ion",
    "compile_with_report",
    "core_version",
    "debye_waller_factor",
    "depolarizing_rate",
    "derive_raman_drive",
    "detection_rates_for_ion",
    "doppler_force_nbar",
    "entanglement_infidelity",
    "envelope_of",
    "equilibrium_dimensionless",
    "gate_modes",
    "gate_steps",
    "ideal_probabilities",
    "input_states",
    "integrals_segmented",
    "is_stable",
    "key_frozen_n",
    "key_mode_offset_hz",
    "key_qubit_offset_hz",
    "kirchmair_populations",
    "kraus_operators",
    "lamb_dicke_parameter",
    "last_record",
    "load_ionq_json",
    "load_openqasm2",
    "mean_occupation",
    "monodromy",
    "ms_alpha",
    "ms_gamma",
    "optimize_threshold",
    "pauli_twirl",
    "prepare",
    "rabi_matrix_element",
    "rabi_table",
    "register_fidelity",
    "run",
    "run_preparation",
    "scattering_channels",
    "schedule",
    "select_space",
    "simulate_epg_sets",
    "solve_amplitude_modulation",
    "solve_crystal",
    "species_by_name",
    "standard_recipe",
    "stenholm_coefficients",
    "thermal_debye_waller_infidelity",
    "thermal_distribution",
    "trajectory_sampled",
    "white_spectrum",
    "x0_m",
    "yb171_chain",
]
