"""Rung 4 of the ladder, the physics (docs/api_proposal.md Section 4.6; docs/api_implementation_plan.md 1.6; 0.2.0): the
records a device is built from and what the package derives from them. The species with its cited constants (``species``,
``Species``, ``Level``, ``Transition``, the Zeeman structure), the trap and the crystal it holds (``Trap``, ``Crystal``,
``Mode``, ``solve_crystal``), the light (``Beam``), the field (``Field``), the noise as spectra and drifts (``NoiseModel`` and
the spectrum constructors), the detector and the control electronics, the ``Device`` that aggregates them with the roles its
beams play (``BeamRoles``), the preparation recipe, the example-device helpers of ``device.presets`` and the two unit types
of Section 5.6 (Hz public, rad/s internal). ``Device.derived()`` returns every computed number with its provenance id, and
``Machine(device)`` is the way back up the ladder. Since 0.4.0 (docs/api_implementation_plan.md 3.3) the rung also exports
the closed forms and the published models the derived numbers are checked against: the Mathieu stability functions
(``monodromy``, ``is_stable``), James's dimensionless crystal (``equilibrium_dimensionless``, ``axial_modes_dimensionless``),
the zero-point length and Lamb-Dicke closed form (``x0_m``, ``lamb_dicke_parameter``), the analytic matrix elements of
Section 4.3.1 (``rabi_matrix_element``, ``rabi_table``, ``debye_waller_factor``), the Doppler force and Stenholm
coefficients (``doppler_force_nbar``, ``stenholm_coefficients``), the pulsed sideband-cooling transfer (``apply_pulses``,
``mean_occupation``, ``thermal_distribution``), Harty's randomized-benchmarking model (``HartyParameters``,
``simulate_epg_sets``), Kirchmair's and Ballance's two-qubit closed forms (``ms_alpha``, ``ms_gamma``,
``kirchmair_populations``, ``thermal_debye_waller_infidelity``, ``ballance_thermal_error``, ``ThermalReference``), the
published readout apparatus presets (``MYERSON_CA40_PMT``, ``CRAIN_YB171_SNSPD``) and the CODATA atomic mass unit
(``ATOMIC_MASS_KG``).
"""

from __future__ import annotations

from qutip_trap.control.hardware import HardwareChain
from qutip_trap.device.model import BeamRoles, DerivedQuantities, Device, Field, GradientField, ResolvedRoles
from qutip_trap.device.presets import (
    crain_snspd_detector,
    ideal_hardware,
    myerson_ca40_pmt_detector,
    oblique_detection_beam,
    quiet_noise_model,
    raman_pair_along_x,
    secular_trap,
)
from qutip_trap.hilbert.operators import debye_waller_factor, rabi_matrix_element, rabi_table
from qutip_trap.light.beams import Beam, PolarizationModulation, PolGradientBeams
from qutip_trap.noise.collisions import CollisionEvent, collision_rate_per_ion
from qutip_trap.noise.model import NoiseModel
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
from qutip_trap.prep.closed_forms import doppler_force_nbar, lamb_dicke_parameter, stenholm_coefficients, x0_m
from qutip_trap.prep.recipe import PreparationRecipe, SidebandCoolingSpec, standard_recipe
from qutip_trap.prep.sideband import apply_pulses, mean_occupation, thermal_distribution
from qutip_trap.readout.detection import CameraGeometry, Detector
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT, ApparatusPreset
from qutip_trap.species import IncompleteSpeciesTable, available, species
from qutip_trap.species.metastable import MetastableChannels
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.zeeman import ClockPoint, ZeemanSpectrum
from qutip_trap.trap.anharmonic import AnharmonicTerms
from qutip_trap.trap.crystal import (
    Crystal,
    Mode,
    ZigzagError,
    axial_modes_dimensionless,
    equilibrium_dimensionless,
    solve_crystal,
)
from qutip_trap.trap.mathieu import MathieuParameters, Monodromy, is_stable, monodromy
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes
from qutip_trap.units import ATOMIC_MASS_KG, Gauss, Hz, RadPerS, Tesla, hz_from_rad_s, rad_s_from_hz
from qutip_trap.validation.harty_rb import HartyParameters, simulate_epg_sets
from qutip_trap.validation.two_qubit_closed_forms import (
    ThermalReference,
    ballance_thermal_error,
    kirchmair_populations,
    ms_alpha,
    ms_gamma,
    thermal_debye_waller_infidelity,
)

__all__ = [
    "ATOMIC_MASS_KG",
    "CRAIN_YB171_SNSPD",
    "MYERSON_CA40_PMT",
    "AnharmonicTerms",
    "ApparatusPreset",
    "AtomicStructure",
    "Beam",
    "BeamRoles",
    "CameraGeometry",
    "ClockPoint",
    "CollisionEvent",
    "Collisions",
    "Crystal",
    "DcElectrodes",
    "DerivedQuantities",
    "Detector",
    "Device",
    "Drift",
    "Electrodes",
    "Field",
    "Gauss",
    "GradientField",
    "HardwareChain",
    "HartyParameters",
    "Hz",
    "IncompleteSpeciesTable",
    "Level",
    "Mains",
    "MathieuParameters",
    "MetastableChannels",
    "MicromotionIndex",
    "Mode",
    "Monodromy",
    "NoiseModel",
    "NoiseSpectrum",
    "PolGradientBeams",
    "PolarizationModulation",
    "PreparationRecipe",
    "RadPerS",
    "ResolvedRoles",
    "RfDrive",
    "SidebandCoolingSpec",
    "Species",
    "Tesla",
    "ThermalReference",
    "Transition",
    "Trap",
    "ZeemanSpectrum",
    "ZigzagError",
    "apply_pulses",
    "available",
    "axial_modes_dimensionless",
    "ballance_thermal_error",
    "collision_rate_per_ion",
    "crain_snspd_detector",
    "debye_waller_factor",
    "doppler_force_nbar",
    "equilibrium_dimensionless",
    "gaussian_spectrum",
    "hz_from_rad_s",
    "ideal_hardware",
    "is_stable",
    "kirchmair_populations",
    "lamb_dicke_parameter",
    "mean_occupation",
    "monodromy",
    "ms_alpha",
    "ms_gamma",
    "myerson_ca40_pmt_detector",
    "oblique_detection_beam",
    "ou_spectrum",
    "power_law_spectrum",
    "quiet_noise_model",
    "rabi_matrix_element",
    "rabi_table",
    "rad_s_from_hz",
    "raman_pair_along_x",
    "secular_trap",
    "simulate_epg_sets",
    "solve_crystal",
    "species",
    "standard_recipe",
    "stenholm_coefficients",
    "thermal_debye_waller_infidelity",
    "thermal_distribution",
    "white_spectrum",
    "x0_m",
]
