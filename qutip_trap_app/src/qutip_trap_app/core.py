"""Every import of ``qutip_trap`` the application makes, in one module, so the contract can be audited in one place."""

from __future__ import annotations

from qutip_trap import __version__ as core_version
from qutip_trap.calibration import calibrate
from qutip_trap.control.compiler import (
    Circuit,
    CompileReport,
    Operation,
    circuit_unitary,
    compile_with_report,
    ideal_probabilities,
)
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import GateDrive, GateTarget, PlayedGate, Schedule, schedule

# ---- rung 2: the schedule and the closed-form trajectory behind a played waveform -----------------------------------------------
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    GateModes,
    SampledEnvelope,
    SegmentedEnvelope,
    ShapedPulse,
    closure_duration_s,
    closure_rabi_rad_s,
    envelope_of,
    gate_modes,
    integrals_segmented,
    solve_amplitude_modulation,
    trajectory_sampled,
)
from qutip_trap.control.table import CalEntry, CalibrationTable, Segment, Waveform
from qutip_trap.device.model import Device, Field
from qutip_trap.device.presets import DevicePreset, ca40_optical, yb171_chain
from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel, SeedSpec, SolverOptions, State, Traces
from qutip_trap.dynamics.hamiltonian import BuilderOptions, BuiltHamiltonian, DriveRecord, build_hamiltonian
from qutip_trap.dynamics.tomography import input_states, kraus_operators
from qutip_trap.hilbert.operators import debye_waller_factor, rabi_matrix_element, rabi_table
from qutip_trap.hilbert.space import HilbertSpace, ModeTruncation
from qutip_trap.io.ionq import load_ionq_json
from qutip_trap.io.openqasm import load_openqasm2

# ---- the Appendix E compatibility surface ------------------------------------------------------------------------------------
from qutip_trap.light.beams import Beam
from qutip_trap.light.raman import derive_raman_drive
from qutip_trap.machine import Machine, as_machine
from qutip_trap.noise.collisions import collision_rate_per_ion
from qutip_trap.noise.model import NoiseModel

# ---- rung 3: the dynamics ------------------------------------------------------------------------------------------------------
from qutip_trap.noise.sampling import (
    KEY_BRANCH_WEIGHT,
    NoiseSample,
    key_frozen_n,
    key_mode_offset_hz,
    key_qubit_offset_hz,
)
from qutip_trap.noise.scattering import ScatteringOptions, scattering_channels
from qutip_trap.noise.spectra import Drift, NoiseSpectrum, white_spectrum
from qutip_trap.noise.summary import (
    average_gate_infidelity,
    choi_from_unitary,
    depolarizing_rate,
    entanglement_infidelity,
    pauli_twirl,
)
from qutip_trap.options import Numerics, Physics, Readout
from qutip_trap.prep.closed_forms import doppler_force_nbar, lamb_dicke_parameter, stenholm_coefficients, x0_m
from qutip_trap.prep.recipe import PreparationRecipe, run_preparation, standard_recipe
from qutip_trap.prep.sideband import apply_pulses, mean_occupation, thermal_distribution
from qutip_trap.readout.detection import Detector, RecordModel
from qutip_trap.readout.discriminate import ThresholdDiscriminator, optimize_threshold
from qutip_trap.readout.fluorescence import detection_rates_for_ion
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.run.gate_local import GateStep, gate_steps
from qutip_trap.run.job import RunRecord, last_record, prepare, register_fidelity, run

# ---- rung 0: the machine, the jobs and the option objects ------------------------------------------------------------------
from qutip_trap.run.levels import FidelityLevel
from qutip_trap.run.results import Diagnostics, Result
from qutip_trap.run.space import select_space
from qutip_trap.run.spec import Job, JobCancelled, RunSpec
from qutip_trap.species import species as species_by_name
from qutip_trap.species.model import Species
from qutip_trap.trap.crystal import (
    Crystal,
    axial_modes_dimensionless,
    equilibrium_dimensionless,
    solve_crystal,
)
from qutip_trap.trap.mathieu import MathieuParameters, is_stable, monodromy
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import RfDrive

# ---- rung 4: the physics, its closed forms and the published models ----------------------------------------------------------
from qutip_trap.units import ATOMIC_MASS_KG
from qutip_trap.validation.harty_rb import HartyParameters, simulate_epg_sets
from qutip_trap.validation.two_qubit_closed_forms import (
    ballance_thermal_error,
    kirchmair_populations,
    ms_alpha,
    ms_gamma,
    thermal_debye_waller_infidelity,
)

CORE_GAPS: tuple[str, ...] = ()
"""What the core does not expose (or does not re-export) that the application needs: Section 14.6's record of the gaps.
Empty since 0.4.0 (docs/api_implementation_plan.md 3.3): every gap of 0.1.0 to 0.3.0 is closed by a documented rung
module; the field stays on the run record so that an exported record still says which core facilities the app leaned
on (none)."""

__all__ = [
    "FidelityLevel",
    "Job",
    "JobCancelled",
    "Machine",
    "Numerics",
    "Physics",
    "Readout",
    "RunSpec",
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
