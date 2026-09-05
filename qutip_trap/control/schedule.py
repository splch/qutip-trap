"""Schedules and the single-qubit scheduler subset (PLAN.md Sections 3.3, 7.1, 7.2, 7.3, 7.6; Appendix E; M2 for the
single-qubit native gates, M6 for entangling gates, events and the full scheduler).

Virtual-Z rule (Section 13, "Virtual-Z propagation"): RZ(theta) shifts every later pulse phase phi -> phi - theta
with gates read in time order; the frame at the end of the schedule is ``phase_frame``. A native GPi(phi) or GPi2(phi)
is a resonant carrier pulse of area pi or pi/2 at phase phi relative to the ion's frame (Section 4.3.5); its duration
is area/(2 pi f_Rabi) with f_Rabi the CALIBRATION TABLE's carrier Rabi frequency for (ion, drive), never the device's
hidden true value (Section 7.3), and an entry the calibration could not establish refuses to schedule. Between pulses
the scheduler inserts the hardware's dead time as an idle interval during which the engine applies free evolution
(Sections 3.4, 7.10; Harty's 14 us). Single-qubit gates on distinct ions run sequentially by default and in parallel
on request (Section 7.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.control.compiler import NATIVE_GATES, Circuit
from qutip_trap.control.pulses import Drive, DriveKind, Pulse, Tone
from qutip_trap.dynamics.frames import PhaseFrame
from qutip_trap.transport.budget import Transport
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device

M4 = "milestone M4 (control/schedule.py, entangling gates, PLAN.md Section 4.4)"
M6 = "milestone M6 (control/schedule.py, events and mid-circuit operations, PLAN.md Sections 7.2, 7.3)"

MICROWAVE_BEAM_KEY = -1
"""The beam index that keys a microwave drive's carrier Rabi frequency in ``CalibrationTable.rabi`` (no beams)."""


class ScheduleError(ValueError):
    """The scheduler refused: an uncalibrated entry, an ambiguous drive or an unsupported operation."""


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


@dataclass(frozen=True)
class GateDrive:
    """How single-qubit gates on one ion are driven: the drive kind and the beams that implement it (Section 7.3)."""

    kind: DriveKind
    beams: tuple[int, ...]

    @property
    def table_key_beam(self) -> int:
        """The beam index that keys the ion's carrier Rabi frequency in the table (the first beam; -1 for microwaves)."""
        return self.beams[0] if self.beams else MICROWAVE_BEAM_KEY


def default_gate_drives(device: Device) -> dict[int, GateDrive]:
    """Infer the single-qubit gate drive from the device's beams: none = microwave, one = optical, a pair of equal
    wavelength = Raman; anything else is ambiguous and must be passed explicitly."""
    beams = device.beams
    n = device.crystal.n_ions
    if len(beams) == 0:
        spec = GateDrive("microwave", ())
    elif len(beams) == 1:
        species = device.crystal.species[0]
        lo, up = (lab.split()[0] for lab in species.qubit)
        e2 = any(t.multipole == "E2" and {t.lower, t.upper} == {lo, up} for t in species.transitions)
        spec = GateDrive("optical_E2" if e2 else "optical_E1", (0,))
    elif (
        len(beams) == 2 and abs(beams[0].wavelength_m - beams[1].wavelength_m) <= 1e-3 * beams[0].wavelength_m
    ):
        spec = GateDrive("raman", (0, 1))
    else:
        raise ScheduleError(
            "the device's beams do not identify a single-qubit gate drive (need none, one, or one Raman pair); pass gate_drives"
        )
    return {i: spec for i in range(n)}


def carrier_rabi_hz(table: CalibrationTable, ion: int, gate_drive: GateDrive) -> float:
    """The table's carrier Rabi frequency for (ion, drive); refuses an absent or ``uncalibrated`` entry (Section 7.3)."""
    entry = table.rabi.get((ion, gate_drive.table_key_beam))
    if entry is None:
        raise ScheduleError(
            f"no carrier Rabi frequency in the calibration table for ion {ion} under drive {gate_drive} "
            f"(key {(ion, gate_drive.table_key_beam)}); the scheduler never reads the device's true value (Section 7.3)"
        )
    if entry.status == "uncalibrated":
        raise ScheduleError(
            f"the Rabi entry for ion {ion} is uncalibrated and refuses to schedule (Section 7.3)"
        )
    if entry.value <= 0.0:
        raise ScheduleError(f"the Rabi entry for ion {ion} is not positive")
    return float(entry.value)


def single_qubit_pulse(
    ion: int,
    area_rad: float,
    phase_rad: float,
    gate_drive: GateDrive,
    rabi_hz: float,
    t_start_s: float,
    *,
    stark_shift_hz: float = 0.0,
    detuning_hz: float = 0.0,
    gate_id: str | None = None,
    crosstalk: dict[int, complex] | None = None,
) -> Pulse:
    """A square carrier pulse of ``area_rad`` (Omega t = area in the plan's convention) at ``phase_rad`` in the ion's frame."""
    if area_rad <= 0.0:
        raise ValueError(
            "a pulse area is positive; a negative nominal area is played at phase + pi (Section 13)"
        )
    if rabi_hz <= 0.0:
        raise ValueError("rabi_hz must be positive")
    duration = area_rad / (TWO_PI * rabi_hz)
    tone = Tone(detuning_hz=float(detuning_hz), phase_rad=float(phase_rad), envelope_hz=float(rabi_hz))
    drive = Drive(
        kind=gate_drive.kind,
        ions=(ion,),
        tones=(tone,),
        beams=gate_drive.beams,
        stark_shift_hz=float(stark_shift_hz),
        crosstalk=dict(crosstalk or {}),
    )
    return Pulse(drive, t_start_s, t_start_s + duration, gate_id, ())


NATIVE_AREAS: dict[str, float] = {"gpi": float(np.pi), "gpi2": float(np.pi / 2.0)}


def schedule(
    circuit: Circuit,
    device: Device,
    table: CalibrationTable,
    *,
    gate_drives: dict[int, GateDrive] | None = None,
    t0_s: float = 0.0,
    parallel: bool = False,
) -> Schedule:
    """Native gates -> pulses with absolute times from the calibration table (Section 7.3); M2: the single-qubit subset.

    gpi and gpi2 become carrier pulses of area pi and pi/2 at the frame-shifted phase; rz is a frame update (no pulse,
    no time beyond the dead time the hardware inserts between pulses, Section 7.6); a terminal ``measure`` is not an
    event of the schedule (run() performs it); ms and zz are milestone M4, mid-circuit measure/reset/recool M6.
    """
    if not circuit.is_native:
        raise ScheduleError("schedule() takes a native circuit; compile_to_native first (Section 7.2)")
    drives = gate_drives or default_gate_drives(device)
    dead = float(device.hardware.dead_time_s)
    frame = PhaseFrame()
    pulses: list[Pulse] = []
    idle: list[tuple[float, float]] = []
    clock: dict[int, float] = {q: t0_s for q in range(circuit.n_qubits)}
    global_clock = t0_s
    for k, op in enumerate(circuit.ops):
        if op.name == "rz":
            frame = frame.rz(op.qubits[0], op.params[0])
            continue
        if op.name in ("ms", "zz"):
            raise NotImplementedError(f"{op.name} pulses are {M4}")
        if op.is_non_unitary:
            if op.name == "measure" and k == len(circuit.ops) - 1:
                continue  # the terminal measurement is run()'s readout stage, not a scheduled event
            raise NotImplementedError(f"mid-circuit {op.name} is {M6} (Section 7.2 item 4)")
        if op.name not in NATIVE_GATES:
            raise ScheduleError(f"unknown native operation {op.name!r}")
        q = op.qubits[0]
        spec = drives[q]
        rabi = carrier_rabi_hz(table, q, spec)
        stark_entry = table.stark.get((q, spec.table_key_beam))
        stark = (
            0.0 if stark_entry is None or stark_entry.status == "uncalibrated" else float(stark_entry.value)
        )
        xt: dict[int, complex] = {}
        for (i, j), entry in table.crosstalk.items():
            if i == q and entry.status != "uncalibrated" and entry.value != 0.0:
                xt[j] = complex(entry.value)
        start = clock[q] if parallel else global_clock
        if pulses and start > t0_s:
            pass
        pulse = single_qubit_pulse(
            q,
            NATIVE_AREAS[op.name],
            frame.pulse_phase(q, op.params[0]),
            spec,
            rabi,
            start,
            stark_shift_hz=stark,
            gate_id=f"{op.name}[{k}]",
            crosstalk=xt,
        )
        pulses.append(pulse)
        end = pulse.t_end_s
        if dead > 0.0:
            idle.append((end, end + dead))
        if parallel:
            clock[q] = end + dead
        else:
            global_clock = end + dead
    return Schedule(tuple(pulses), tuple(idle), (), frame.as_dict(circuit.n_qubits))


__all__ = [
    "MICROWAVE_BEAM_KEY",
    "NATIVE_AREAS",
    "GateDrive",
    "Schedule",
    "ScheduleError",
    "ScheduledEvent",
    "carrier_rabi_hz",
    "default_gate_drives",
    "schedule",
    "single_qubit_pulse",
]
