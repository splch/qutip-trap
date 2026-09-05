"""Control stack: native gates, circuit IR, pulses, schedules, calibration tables (PLAN.md Section 7).

Layering rule (Appendix E): ``control`` never imports ``calibration``, which sits above it and communicates
through ``CalibrationTable`` (a plain data object, ``control/table.py``).
"""

from __future__ import annotations

from qutip_trap.control.compiler import Circuit, Operation, compile_to_native
from qutip_trap.control.composite import CompositePulse, composite_pulse
from qutip_trap.control.hardware import HardwareChain
from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule, ScheduledEvent, schedule
from qutip_trap.control.table import CalEntry, CalibrationTable, Segment, Waveform

__all__ = [
    "CalEntry",
    "CalibrationTable",
    "Circuit",
    "CompositePulse",
    "Drive",
    "HardwareChain",
    "Operation",
    "Pulse",
    "Schedule",
    "ScheduledEvent",
    "Segment",
    "Tone",
    "Waveform",
    "compile_to_native",
    "composite_pulse",
    "schedule",
]
