"""Rung 4 of the ladder, the physics (docs/api_proposal.md Section 4.6; docs/api_implementation_plan.md 1.6; 0.2.0): the
records a device is built from and what the package derives from them. The species with its cited constants (``species``,
``Species``, ``Level``, ``Transition``, the Zeeman structure), the trap and the crystal it holds (``Trap``, ``Crystal``,
``Mode``, ``solve_crystal``), the light (``Beam``), the field (``Field``), the noise as spectra and drifts (``NoiseModel`` and
the spectrum constructors), the detector and the control electronics, the ``Device`` that aggregates them with the roles its
beams play (``BeamRoles``), the preparation recipe, the example-device helpers of ``device.presets`` and the two unit types
of Section 5.6 (Hz public, rad/s internal). ``Device.derived()`` returns every computed number with its provenance id, and
``Machine(device)`` is the way back up the ladder. Phase 3.3 of the plan adds the closed forms.
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
from qutip_trap.prep.recipe import PreparationRecipe, SidebandCoolingSpec, standard_recipe
from qutip_trap.readout.detection import CameraGeometry, Detector
from qutip_trap.readout.presets import ApparatusPreset
from qutip_trap.species import IncompleteSpeciesTable, available, species
from qutip_trap.species.metastable import MetastableChannels
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.zeeman import ClockPoint, ZeemanSpectrum
from qutip_trap.trap.anharmonic import AnharmonicTerms
from qutip_trap.trap.crystal import Crystal, Mode, ZigzagError, solve_crystal
from qutip_trap.trap.mathieu import MathieuParameters
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes
from qutip_trap.units import Gauss, Hz, RadPerS, Tesla, hz_from_rad_s, rad_s_from_hz

__all__ = [
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
    "Hz",
    "IncompleteSpeciesTable",
    "Level",
    "Mains",
    "MathieuParameters",
    "MetastableChannels",
    "MicromotionIndex",
    "Mode",
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
    "Transition",
    "Trap",
    "ZeemanSpectrum",
    "ZigzagError",
    "available",
    "collision_rate_per_ion",
    "crain_snspd_detector",
    "gaussian_spectrum",
    "hz_from_rad_s",
    "ideal_hardware",
    "myerson_ca40_pmt_detector",
    "oblique_detection_beam",
    "ou_spectrum",
    "power_law_spectrum",
    "quiet_noise_model",
    "rad_s_from_hz",
    "raman_pair_along_x",
    "secular_trap",
    "solve_crystal",
    "species",
    "standard_recipe",
    "white_spectrum",
]
