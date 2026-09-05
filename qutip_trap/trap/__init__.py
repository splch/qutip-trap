"""Trap potential, ion crystal, motional modes, micromotion and heating (PLAN.md Section 4.1; milestone M1)."""

from __future__ import annotations

from qutip_trap.trap.anharmonic import (
    AnharmonicTerms,
    Resonance,
    coulomb_anharmonic_terms,
    three_mode_resonances,
)
from qutip_trap.trap.crystal import Crystal, LambDicke, Mode, ZigzagError, build_crystal, solve_crystal
from qutip_trap.trap.heating import heating_rate_quanta_per_s, heating_rates_per_mode
from qutip_trap.trap.mathieu import FloquetCoefficients, MathieuParameters, UnstableMathieuError
from qutip_trap.trap.micromotion import MicromotionIndex
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import DcElectrodes, RfDrive
from qutip_trap.trap.surface import Electrodes, GaplessPlaneTrap

__all__ = [
    "AnharmonicTerms",
    "Crystal",
    "DcElectrodes",
    "Electrodes",
    "FloquetCoefficients",
    "GaplessPlaneTrap",
    "LambDicke",
    "MathieuParameters",
    "MicromotionIndex",
    "Mode",
    "Resonance",
    "RfDrive",
    "Trap",
    "UnstableMathieuError",
    "ZigzagError",
    "build_crystal",
    "coulomb_anharmonic_terms",
    "heating_rate_quanta_per_s",
    "heating_rates_per_mode",
    "solve_crystal",
    "three_mode_resonances",
]
