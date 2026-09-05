"""The public surface of qutip-trap: Appendix E of PLAN.md, re-exported from one place.

The application of Section 14 imports nothing outside this module (Appendix E, closing rules). Frequencies in
this API are in Hz and are converted to rad/s at the boundary (Section 5.6); every object is immutable after
construction; ``Device.hash()`` is the canonical identity. Methods and functions whose physics arrives in a
later milestone raise ``NotImplementedError`` naming that milestone.
"""

from __future__ import annotations

from qutip_trap.calibration import calibrate
from qutip_trap.control.compiler import Circuit, Operation, compile_to_native
from qutip_trap.control.composite import CompositePulse, composite_pulse
from qutip_trap.control.hardware import HardwareChain
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule, ScheduledEvent, schedule
from qutip_trap.control.table import CalEntry, CalibrationTable, Segment, Waveform
from qutip_trap.device.model import DerivedQuantities, Device, Field
from qutip_trap.dynamics.channels import CollapseOp
from qutip_trap.dynamics.engine import (
    ChannelSummary,
    MotionalModel,
    PulseEngine,
    SeedSpec,
    SolverOptions,
    State,
    Traces,
)
from qutip_trap.experiments import (
    ExperimentResult,
    detection_histogram,
    heating_rate,
    micromotion_scan,
    ms_scan,
    parity_scan,
    rabi_scan,
    ramsey,
    ramsey_frequency,
    sideband_spectroscopy,
)
from qutip_trap.hilbert.space import CachedOperators, HilbertSpace, ModeTruncation
from qutip_trap.io.ionq import dump_ionq_json, load_ionq_json
from qutip_trap.io.openqasm import load_openqasm2
from qutip_trap.light.beams import Beam, PolarizationModulation, PolGradientBeams
from qutip_trap.light.comb import CombSpec
from qutip_trap.noise.decoupling import DecouplingSequence, decoupling_sequence, filter_function
from qutip_trap.noise.model import NoiseModel
from qutip_trap.noise.sampling import NoiseSample
from qutip_trap.noise.spectra import Collisions, Drift, Mains, NoiseSpectrum
from qutip_trap.readout.detection import Detector
from qutip_trap.readout.discriminate import POVM
from qutip_trap.readout.fluorescence import DarkStateReport
from qutip_trap.run.job import prepare, run
from qutip_trap.run.levels import FidelityLevel, resolve_level
from qutip_trap.run.results import Diagnostics, Result, RunState
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
from qutip_trap.trap.crystal import Crystal, Mode
from qutip_trap.trap.mathieu import MathieuParameters
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes
from qutip_trap.units import Gauss, Hz, RadPerS, Tesla, hz_from_rad_s, rad_s_from_hz

__all__ = [
    "POVM",
    "AnharmonicTerms",
    "AtomicStructure",
    "Beam",
    "CachedOperators",
    "CalEntry",
    "CalibrationTable",
    "ChannelSummary",
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
    "Gauss",
    "HardwareChain",
    "HilbertSpace",
    "Hz",
    "IncompleteSpeciesTable",
    "Level",
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
    "Zone",
    "available",
    "calibrate",
    "compile_to_native",
    "composite_pulse",
    "decoupling_sequence",
    "design_waveform",
    "detection_histogram",
    "dump_ionq_json",
    "filter_function",
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
    "species_by_name",
    "split_feasible",
    "transport_budget",
]
