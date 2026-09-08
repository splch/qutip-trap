"""The public surface of qutip-trap: Appendix E of PLAN.md, re-exported from one place.

The application of Section 14 imports nothing outside this module (Appendix E, closing rules). Frequencies in
this API are in Hz and are converted to rad/s at the boundary (Section 5.6); every object is immutable after
construction; ``Device.hash()`` is the canonical identity. Methods and functions whose physics arrives in a
later milestone raise ``NotImplementedError`` naming that milestone.
"""

from __future__ import annotations

from qutip_trap.benchmarks import (
    BenchmarkBudget,
    GateChannel,
    GHZResult,
    QVCircuit,
    QVResult,
    RBResult,
    RBSequence,
    StepChannel,
    TwoQubitClifford,
    clear_budget_cache,
    decompose_two_qubit_clifford,
    gate_channel,
    ghz_circuit,
    ghz_fidelity,
    parity_circuit,
    quantum_volume,
    random_square_circuit,
    random_two_qubit_clifford,
    randomized_benchmarking,
)
from qutip_trap.calibration import (
    CalibrationCache,
    CalibrationError,
    CalibrationReport,
    CalibrationScans,
    calibrate,
    calibrate_with_report,
    full_calibration,
)
from qutip_trap.calibration.entangling import (
    CalibrationRun,
    GateCheck,
    calibrate_entangling_angle,
    exact_gate_check,
    frame_rotated,
    gate_space,
    thermal_robustness,
)
from qutip_trap.calibration.readout import DetectionCalibration, calibrate_detection
from qutip_trap.calibration.surrogate import SurrogateReport, surrogate_table
from qutip_trap.control.compiler import (
    Circuit,
    CompileReport,
    Operation,
    circuit_unitary,
    compile_to_native,
    compile_with_report,
    ideal_probabilities,
)
from qutip_trap.control.composite import CompositePulse, composite_pulse
from qutip_trap.control.hardware import HardwareChain, apply_hardware_chain
from qutip_trap.control.played import physical_schedule
from qutip_trap.control.pulses import Drive, LightShiftCouplings, Pulse, Tone
from qutip_trap.control.schedule import GateDrive, GateTarget, PlayedGate, Schedule, ScheduledEvent, schedule
from qutip_trap.control.shaping import (
    GateModes,
    ShapedPulse,
    gate_modes,
    solve_amplitude_modulation,
    solve_fourier_amplitude_modulation,
    solve_frequency_modulation,
)
from qutip_trap.control.table import CalEntry, CalibrationTable, Segment, Waveform
from qutip_trap.control.two_qubit import (
    KAK,
    decompose_two_qubit_unitary,
    haar_random_unitary,
    kak_decomposition,
)
from qutip_trap.device.model import DerivedQuantities, Device, Field
from qutip_trap.device.presets import DevicePreset, ca40_optical, ca40_optical_recipe, yb171_chain
from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.engine import (  # noqa: I001
    ChannelSummary,
    MotionalModel,
    PulseEngine,
    SeedSpec,
    SolverOptions,
    State,
    Traces,
)
from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions
from qutip_trap.dynamics.tomography import (
    TomographyRecord,
    choi_least_squares,
    input_states,
    kraus_operators,
    project_cptp,
)
from qutip_trap.experiments import (
    ExperimentResult,
    Observation,
    ReadoutErrors,
    crosstalk_scan,
    crystal_image,
    detection_histogram,
    field_scan,
    heating_rate,
    micromotion_scan,
    mode_spectroscopy,
    ms_phase_scan,
    ms_scan,
    parity_scan,
    rabi_scan,
    ramsey,
    ramsey_frequency,
    sideband_lineshape,
    sideband_spectroscopy,
    stark_scan,
    thermometry,
)
from qutip_trap.hilbert.space import CachedOperators, HilbertSpace, ModeTruncation
from qutip_trap.io.ionq import dump_ionq_json, load_ionq_json
from qutip_trap.io.openqasm import load_openqasm2
from qutip_trap.light.beams import Beam, PolarizationModulation, PolGradientBeams
from qutip_trap.light.bloch import BlochModel, DetectionRates, SteadyStateReport
from qutip_trap.light.comb import CombSpec
from qutip_trap.light.raman import derive_light_shift_drive, derive_raman_drive
from qutip_trap.noise.collisions import CollisionEvent, collision_rate_per_ion
from qutip_trap.noise.decoupling import (
    ControlSegment,
    DecouplingSequence,
    decoupling_sequence,
    filter_function,
)
from qutip_trap.noise.levels import InternalLevels, internal_levels
from qutip_trap.noise.model import NoiseModel
from qutip_trap.noise.processes import Trajectory
from qutip_trap.noise.sampling import NoiseSample
from qutip_trap.noise.scattering import ScatteringOptions, scattering_channels, scattering_estimates
from qutip_trap.noise.spectra import (
    Collisions,
    Drift,
    Mains,
    NoiseSpectrum,
    gaussian_spectrum,
    ou_spectrum,
    power_law_spectrum,
    white_spectrum,
)
from qutip_trap.noise.summary import (
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_choi,
    depolarizing_rate,
    entanglement_infidelity,
    pauli_twirl,
)
from qutip_trap.prep.recipe import (
    PreparationRecipe,
    PreparationRun,
    SidebandCoolingSpec,
    run_preparation,
    standard_recipe,
)
from qutip_trap.readout.detection import CameraGeometry, Detector, PhotonRecord, RecordModel
from qutip_trap.readout.discriminate import (
    POVM,
    AdaptiveML,
    BudgetLine,
    FirstPhoton,
    ReadoutBudget,
    ReadoutOutcome,
    ThresholdDiscriminator,
    TimeResolvedML,
    measure,
    optimize_threshold,
    povm_for,
)
from qutip_trap.readout.fluorescence import (
    DarkStateReport,
    FluorescenceRates,
    ReadoutScheme,
    detection_rates_for_ion,
    scattering_rate,
)
from qutip_trap.readout.presets import ApparatusPreset
from qutip_trap.run.gate_local import GateLocalReport, GateLocalStep, GateStep, gate_steps, step_space
from qutip_trap.run.job import RunRecord, last_record, prepare, register_fidelity, run
from qutip_trap.run.levels import FidelityLevel, resolve_level
from qutip_trap.run.results import Diagnostics, Result, RunState
from qutip_trap.run.space import SpaceSelection, frozen_excitation_bounds, select_space
from qutip_trap.species import IncompleteSpeciesTable, available
from qutip_trap.species import species as species_by_name
from qutip_trap.species.metastable import MetastableChannels
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.zeeman import ClockPoint, ZeemanSpectrum
from qutip_trap.transport.budget import (
    Transport,
    TransportBudget,
    design_waveform,
    split_feasible,
    transport_budget,
)
from qutip_trap.transport.waveforms import FilterStage, VoltageWaveform
from qutip_trap.transport.zones import Zone
from qutip_trap.trap.anharmonic import AnharmonicTerms
from qutip_trap.trap.crystal import Crystal, Mode, ZigzagError, solve_crystal
from qutip_trap.trap.mathieu import MathieuParameters
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes
from qutip_trap.units import Gauss, Hz, RadPerS, Tesla, hz_from_rad_s, rad_s_from_hz

__all__ = [
    "BenchmarkBudget",
    "DevicePreset",
    "GHZResult",
    "GateChannel",
    "KAK",
    "QVCircuit",
    "QVResult",
    "RBResult",
    "RBSequence",
    "StepChannel",
    "TwoQubitClifford",
    "clear_budget_cache",
    "decompose_two_qubit_clifford",
    "decompose_two_qubit_unitary",
    "gate_channel",
    "ghz_circuit",
    "ghz_fidelity",
    "haar_random_unitary",
    "kak_decomposition",
    "parity_circuit",
    "quantum_volume",
    "random_square_circuit",
    "random_two_qubit_clifford",
    "randomized_benchmarking",
    "ca40_optical",
    "ca40_optical_recipe",
    "yb171_chain",
    "GateLocalReport",
    "GateLocalStep",
    "GateStep",
    "GateTarget",
    "TomographyRecord",
    "choi_least_squares",
    "frozen_excitation_bounds",
    "gate_steps",
    "input_states",
    "kraus_operators",
    "project_cptp",
    "step_space",
    "CalibrationCache",
    "CalibrationError",
    "CalibrationReport",
    "CalibrationScans",
    "Observation",
    "ReadoutErrors",
    "calibrate_with_report",
    "crosstalk_scan",
    "crystal_image",
    "field_scan",
    "frame_rotated",
    "full_calibration",
    "mode_spectroscopy",
    "ms_phase_scan",
    "physical_schedule",
    "sideband_lineshape",
    "stark_scan",
    "thermometry",
    "CollisionEvent",
    "ControlSegment",
    "InternalLevels",
    "ScatteringOptions",
    "Trajectory",
    "apply_hardware_chain",
    "average_gate_infidelity",
    "choi_from_unitary",
    "collision_rate_per_ion",
    "depolarizing_choi",
    "depolarizing_rate",
    "entanglement_infidelity",
    "gaussian_spectrum",
    "internal_levels",
    "ou_spectrum",
    "pauli_twirl",
    "power_law_spectrum",
    "scattering_channels",
    "scattering_estimates",
    "white_spectrum",
    "POVM",
    "AdaptiveML",
    "ApparatusPreset",
    "BudgetLine",
    "CameraGeometry",
    "FirstPhoton",
    "FluorescenceRates",
    "PhotonRecord",
    "ReadoutBudget",
    "ReadoutOutcome",
    "ReadoutScheme",
    "RecordModel",
    "ThresholdDiscriminator",
    "TimeResolvedML",
    "detection_rates_for_ion",
    "measure",
    "optimize_threshold",
    "povm_for",
    "scattering_rate",
    "BlochModel",
    "DetectionRates",
    "ModeSpec",
    "MultiLevelOptions",
    "SteadyStateReport",
    "AnharmonicTerms",
    "AtomicStructure",
    "Beam",
    "CachedOperators",
    "CalEntry",
    "CalibrationRun",
    "CalibrationTable",
    "ChannelSummary",
    "CompileReport",
    "PlayedGate",
    "PreparationRecipe",
    "PreparationRun",
    "RunRecord",
    "SidebandCoolingSpec",
    "SpaceSelection",
    "SurrogateReport",
    "circuit_unitary",
    "compile_with_report",
    "ideal_probabilities",
    "last_record",
    "register_fidelity",
    "run_preparation",
    "select_space",
    "standard_recipe",
    "surrogate_table",
    "ClockPoint",
    "Circuit",
    "CollapseOp",
    "Collisions",
    "CombSpec",
    "CompositePulse",
    "Crystal",
    "DarkStateReport",
    "DcElectrodes",
    "DecouplingSequence",
    "DerivedQuantities",
    "Detector",
    "Device",
    "Diagnostics",
    "Drift",
    "Drive",
    "Electrodes",
    "ExperimentResult",
    "FidelityLevel",
    "Field",
    "FilterStage",
    "GateCheck",
    "GateDrive",
    "GateModes",
    "Gauss",
    "HardwareChain",
    "HilbertSpace",
    "Hz",
    "IncompleteSpeciesTable",
    "Level",
    "LightShiftCouplings",
    "Mains",
    "MathieuParameters",
    "MetastableChannels",
    "MicromotionIndex",
    "Mode",
    "ModeTruncation",
    "MotionalModel",
    "NoiseModel",
    "NoiseSample",
    "NoiseSpectrum",
    "Operation",
    "PolGradientBeams",
    "PolarizationModulation",
    "Pulse",
    "PulseEngine",
    "RadPerS",
    "Result",
    "RfDrive",
    "RunState",
    "Schedule",
    "ScheduledEvent",
    "SeedSpec",
    "Segment",
    "ShapedPulse",
    "SolverOptions",
    "Species",
    "State",
    "Tesla",
    "Tone",
    "Traces",
    "Transition",
    "Transport",
    "TransportBudget",
    "Trap",
    "VoltageWaveform",
    "Waveform",
    "ZeemanSpectrum",
    "ZigzagError",
    "Zone",
    "available",
    "calibrate",
    "calibrate_detection",
    "DetectionCalibration",
    "calibrate_entangling_angle",
    "compile_to_native",
    "composite_pulse",
    "decoupling_sequence",
    "derive_light_shift_drive",
    "derive_raman_drive",
    "design_waveform",
    "detection_histogram",
    "exact_gate_check",
    "dump_ionq_json",
    "filter_function",
    "gate_modes",
    "gate_space",
    "heating_rate",
    "hz_from_rad_s",
    "load_ionq_json",
    "load_openqasm2",
    "micromotion_scan",
    "ms_scan",
    "parity_scan",
    "prepare",
    "rabi_scan",
    "rad_s_from_hz",
    "ramsey",
    "ramsey_frequency",
    "resolve_level",
    "run",
    "schedule",
    "sideband_spectroscopy",
    "solve_amplitude_modulation",
    "solve_crystal",
    "solve_fourier_amplitude_modulation",
    "solve_frequency_modulation",
    "species_by_name",
    "split_feasible",
    "thermal_robustness",
    "transport_budget",
]
