"""Rung 2 of the ladder, the schedule (docs/api_proposal.md Section 4.4; docs/api_implementation_plan.md 1.6; 0.2.0): the
pulses a circuit becomes on the time axis and everything they are made of. ``Machine.schedule(circuit)`` is the way here
from rung 1 (``compile_calibrate_schedule``, the prefix of ``run``); ``Schedule``, ``Pulse``, ``Drive``, ``Tone``, the
calibrated ``Waveform`` and ``CalibrationTable`` the scheduler reads, the beam roles as ``GateDrive`` maps, the hardware
chain of Section 7.10, the composite-pulse library, the comb specification and the three closure solvers of Section 4.4.3
are the objects; ``Machine.engine`` is the way down to rung 3.
"""

from __future__ import annotations

from qutip_trap.control.composite import CompositePulse, composite_pulse
from qutip_trap.control.hardware import HardwareChain, apply_hardware_chain
from qutip_trap.control.played import physical_schedule
from qutip_trap.control.pulses import Drive, LightShiftCouplings, Pulse, Tone
from qutip_trap.control.schedule import (
    GateDrive,
    GateTarget,
    PlayedGate,
    Schedule,
    ScheduledEvent,
    ScheduleError,
    default_gate_drives,
    infer_gate_drives,
    resolve_drives,
    schedule,
)
from qutip_trap.control.shaping import (
    GateModes,
    ShapedPulse,
    gate_modes,
    solve_amplitude_modulation,
    solve_fourier_amplitude_modulation,
    solve_frequency_modulation,
)
from qutip_trap.control.table import (
    ENTRY_KINDS,
    CalEntry,
    CalibrationTable,
    EntryKind,
    Segment,
    Waveform,
)
from qutip_trap.light.comb import CombSpec
from qutip_trap.light.raman import derive_light_shift_drive, derive_raman_drive
from qutip_trap.run.pipeline import Prefix, compile_calibrate_schedule

__all__ = [
    "apply_hardware_chain",
    "CalEntry",
    "CalibrationTable",
    "CombSpec",
    "compile_calibrate_schedule",
    "composite_pulse",
    "CompositePulse",
    "default_gate_drives",
    "derive_light_shift_drive",
    "derive_raman_drive",
    "Drive",
    "ENTRY_KINDS",
    "EntryKind",
    "gate_modes",
    "GateDrive",
    "GateModes",
    "GateTarget",
    "HardwareChain",
    "infer_gate_drives",
    "LightShiftCouplings",
    "physical_schedule",
    "PlayedGate",
    "Prefix",
    "Pulse",
    "resolve_drives",
    "Schedule",
    "schedule",
    "ScheduledEvent",
    "ScheduleError",
    "Segment",
    "ShapedPulse",
    "solve_amplitude_modulation",
    "solve_fourier_amplitude_modulation",
    "solve_frequency_modulation",
    "Tone",
    "Waveform",
]
