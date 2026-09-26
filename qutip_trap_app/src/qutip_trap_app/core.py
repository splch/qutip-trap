"""Every ``qutip_trap`` name the application uses: the one module that imports the core."""

from __future__ import annotations

from qutip_trap.calibration import calibrate
from qutip_trap.calibration.surrogate import MU_ABOVE_TOP_FRACTION, surrogate_waveform
from qutip_trap.control.compiler import Circuit, CompileReport, Operation, embed, ideal_probabilities
from qutip_trap.control.native import ms, rz
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import GateDrive, Schedule, schedule
from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    GateModes,
    SegmentedEnvelope,
    closure_rabi_rad_s,
    envelope_of,
    gate_modes,
    integrals_segmented,
    trajectory_sampled,
)
from qutip_trap.control.table import CalEntry, CalibrationTable, Segment, Waveform
from qutip_trap.device.model import Device
from qutip_trap.device.presets import (
    DevicePreset,
    crain_snspd_detector,
    myerson_ca40_pmt_detector,
    yb171_chain,
)
from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.engine import (
    ChannelSummary,
    JointExactEngine,
    MotionalModel,
    SeedSpec,
    State,
    Traces,
)
from qutip_trap.dynamics.hamiltonian import BuiltHamiltonian, build_hamiltonian
from qutip_trap.dynamics.operators import debye_waller_factor, rabi_table
from qutip_trap.dynamics.space import HilbertSpace
from qutip_trap.dynamics.tomography import kraus_operators
from qutip_trap.io.ionq import load_ionq_json
from qutip_trap.io.openqasm import load_openqasm2
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive
from qutip_trap.light.roles import detection_beams
from qutip_trap.machine import Machine
from qutip_trap.noise.collisions import collision_rate_per_ion
from qutip_trap.noise.sampling import (
    KEY_BRANCH_WEIGHT,
    NoiseSample,
    key_frozen_n,
    key_mode_offset_hz,
    key_qubit_offset_hz,
)
from qutip_trap.noise.scattering import scattering_channels
from qutip_trap.noise.spectra import NoiseSpectrum, white_spectrum
from qutip_trap.noise.summary import choi_from_unitary, entanglement_infidelity
from qutip_trap.options import Numerics, Physics, Readout, ReadoutMode
from qutip_trap.prep.closed_forms import doppler_force_nbar, lamb_dicke_parameter, stenholm_coefficients, x0_m
from qutip_trap.prep.recipe import PreparationRun, run_preparation, standard_recipe
from qutip_trap.prep.sideband import apply_pulses, mean_occupation, thermal_distribution
from qutip_trap.provenance import TAGS, load_ledger, repository_root
from qutip_trap.published import (
    HartyParameters,
    ballance_thermal_error,
    kirchmair_populations,
    ms_alpha,
    ms_gamma,
    simulate_epg_sets,
    thermal_debye_waller_infidelity,
)
from qutip_trap.readout.detection import RecordModel
from qutip_trap.readout.discriminate import ThresholdDiscriminator, optimize_threshold
from qutip_trap.readout.fluorescence import detection_rates_for_ion
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT
from qutip_trap.run.gate_local import GateLocalReport, GateStep, gate_steps
from qutip_trap.run.job import RunRecord, last_record, prepare, register_fidelity
from qutip_trap.run.levels import FidelityLevel
from qutip_trap.run.results import Diagnostics, Progress, Result, aggregate, binomial_error_bars
from qutip_trap.run.space import SpaceSelection, select_space
from qutip_trap.species import species
from qutip_trap.species.model import Species
from qutip_trap.trap.crystal import axial_modes_dimensionless, equilibrium_dimensionless, solve_crystal
from qutip_trap.trap.mathieu import is_stable, monodromy
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import RfDrive
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, E_C, EPSILON_0_F_PER_M, TWO_PI

__all__ = [
    "ATOMIC_MASS_KG",
    "BuiltHamiltonian",
    "C_M_PER_S",
    "CHI_MAXIMAL_RAD",
    "CRAIN_YB171_SNSPD",
    "CalEntry",
    "CalibrationTable",
    "ChannelSummary",
    "Circuit",
    "CollapseOp",
    "CompileReport",
    "Device",
    "DevicePreset",
    "Diagnostics",
    "E_C",
    "EPSILON_0_F_PER_M",
    "FidelityLevel",
    "GateDrive",
    "GateLocalReport",
    "GateModes",
    "GateStep",
    "HartyParameters",
    "HilbertSpace",
    "JointExactEngine",
    "KEY_BRANCH_WEIGHT",
    "MU_ABOVE_TOP_FRACTION",
    "MYERSON_CA40_PMT",
    "Machine",
    "MotionalModel",
    "NoiseSample",
    "NoiseSpectrum",
    "Numerics",
    "Operation",
    "Physics",
    "PreparationRun",
    "Progress",
    "Pulse",
    "Readout",
    "ReadoutMode",
    "RecordModel",
    "Result",
    "RfDrive",
    "RunRecord",
    "Schedule",
    "SeedSpec",
    "Segment",
    "SegmentedEnvelope",
    "SpaceSelection",
    "Species",
    "State",
    "TAGS",
    "TWO_PI",
    "ThresholdDiscriminator",
    "Traces",
    "Trap",
    "Waveform",
    "aggregate",
    "apply_pulses",
    "axial_modes_dimensionless",
    "ballance_thermal_error",
    "binomial_error_bars",
    "build_hamiltonian",
    "calibrate",
    "choi_from_unitary",
    "closure_rabi_rad_s",
    "collision_rate_per_ion",
    "crain_snspd_detector",
    "crosstalk_ratios",
    "debye_waller_factor",
    "derive_raman_drive",
    "detection_beams",
    "detection_rates_for_ion",
    "doppler_force_nbar",
    "embed",
    "entanglement_infidelity",
    "envelope_of",
    "equilibrium_dimensionless",
    "gate_modes",
    "gate_steps",
    "ideal_probabilities",
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
    "load_ledger",
    "load_openqasm2",
    "mean_occupation",
    "monodromy",
    "ms",
    "ms_alpha",
    "ms_gamma",
    "myerson_ca40_pmt_detector",
    "optimize_threshold",
    "prepare",
    "rabi_table",
    "register_fidelity",
    "repository_root",
    "run_preparation",
    "rz",
    "scattering_channels",
    "schedule",
    "select_space",
    "simulate_epg_sets",
    "solve_crystal",
    "species",
    "standard_recipe",
    "stenholm_coefficients",
    "surrogate_waveform",
    "thermal_debye_waller_infidelity",
    "thermal_distribution",
    "trajectory_sampled",
    "white_spectrum",
    "x0_m",
    "yb171_chain",
]
