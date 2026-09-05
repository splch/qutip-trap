"""Schedules (PLAN.md Sections 3.3, 7.2, 7.3, 7.6; Appendix E; milestone M6 for the scheduler).

Virtual-Z rule (Section 13, "Virtual-Z propagation"): RZ(theta) shifts every later pulse phase phi -> phi - theta
with gates read in time order; the frame at the end of the schedule is ``phase_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from qutip_trap.control.pulses import Pulse
from qutip_trap.transport.budget import Transport

if TYPE_CHECKING:
    from qutip_trap.control.compiler import Circuit
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device

M6 = "milestone M6 (control/schedule.py, PLAN.md Sections 7.2, 7.3)"


@dataclass(frozen=True)
class ScheduledEvent:
    """measure / reset / recool with absolute times (Section 7.2)."""

    kind: Literal["measure", "reset", "recool"]
    ions: tuple[int, ...]
    t_start_s: float
    t_end_s: float

    def __post_init__(self) -> None:
        if self.t_end_s < self.t_start_s:
            raise ValueError("an event cannot end before it starts")


@dataclass(frozen=True)
class Schedule:
    pulses: tuple[Pulse, ...]
    idle: tuple[tuple[float, float], ...]
    """Idle intervals (start, end) during which heating and dephasing act (Section 3.4)."""
    events: tuple[ScheduledEvent, ...]
    phase_frame: dict[int, float]
    """Per-qubit virtual-Z frame at the end; the rule is phi -> phi - theta (Section 7.6)."""
    transports: tuple[Transport, ...] = ()
    """M12: interleaved with ``pulses`` by absolute time."""

    def __post_init__(self) -> None:
        for a, b in self.idle:
            if b < a:
                raise ValueError("idle intervals must be ordered (start, end)")
        # pulses that share an ion must not overlap in time (parallel pulses on distinct ions are allowed)
        ordered = sorted(self.pulses, key=lambda p: p.t_start_s)
        for i, p in enumerate(ordered):
            for q in ordered[i + 1 :]:
                if q.t_start_s >= p.t_end_s:
                    break
                if set(p.drive.ions) & set(q.drive.ions):
                    raise ValueError(f"pulses {p.gate_id!r} and {q.gate_id!r} overlap on a shared ion")

    @property
    def duration_s(self) -> float:
        ends = [p.t_end_s for p in self.pulses] + [e.t_end_s for e in self.events] + [b for _, b in self.idle]
        return max(ends) if ends else 0.0


def schedule(circuit: Circuit, device: Device, table: CalibrationTable) -> Schedule:
    """Native gates -> pulses with absolute times, from the calibration table (Section 7.3)."""
    raise NotImplementedError(f"schedule is {M6}")


__all__ = ["Schedule", "ScheduledEvent", "schedule"]
