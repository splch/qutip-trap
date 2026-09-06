"""Noise and error models (PLAN.md Section 6; milestone M7)."""

from __future__ import annotations

from qutip_trap.noise.collisions import CollisionEvent, collision_rate_per_ion, sample_collisions
from qutip_trap.noise.decoupling import (
    ControlSegment,
    DecouplingSequence,
    decoupling_sequence,
    filter_function,
)
from qutip_trap.noise.levels import SINK, InternalLevels, internal_levels
from qutip_trap.noise.model import NoiseModel
from qutip_trap.noise.processes import Trajectory, ou_process, synthesize, time_grid
from qutip_trap.noise.sampling import NoiseSample, quiet_sample
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

__all__ = [
    "SINK",
    "CollisionEvent",
    "Collisions",
    "ControlSegment",
    "DecouplingSequence",
    "Drift",
    "InternalLevels",
    "Mains",
    "NoiseModel",
    "NoiseSample",
    "NoiseSpectrum",
    "ScatteringOptions",
    "Trajectory",
    "collision_rate_per_ion",
    "decoupling_sequence",
    "filter_function",
    "gaussian_spectrum",
    "internal_levels",
    "ou_process",
    "ou_spectrum",
    "power_law_spectrum",
    "quiet_sample",
    "sample_collisions",
    "scattering_channels",
    "scattering_estimates",
    "synthesize",
    "time_grid",
    "white_spectrum",
]
