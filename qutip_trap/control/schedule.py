"""Schedules and the scheduler (PLAN.md Sections 3.3, 4.4, 7.1, 7.2, 7.3, 7.6; Appendix E; M2 for the single-qubit native
gates, M4 for the entangling gates, M6 for events and mid-circuit operations).

Entangling gates (M4, Section 4.4): ``ms`` and ``zz`` read the pair's calibrated ``Waveform`` from the table and play it
as one ``Pulse`` per (ion, segment), each a Drive with the red and blue tones (Section 4.4.1), the tone phases being the
segment's phi_s -/+ phi_m offsets plus the gate phase in the ion's virtual-Z frame. The bichromatic force on ion i acts
about the azimuth phi_s,i + pi/2 in the same-Delta-k geometry (Section 4.3.4; derivation audit 2026-09-04), so MS(phi_0,
phi_1, theta) = exp[-i (theta/2) sigma_phi0 (x) sigma_phi1] sets phi_s,i = phi_i - pi/2; the pulse applies exp(+i chi sigma
sigma) with chi the waveform's signed angle, so a POSITIVE chi (tones inside the sidebands) is played with a pi on the
second ion's tones (GPi(phi + pi) = -GPi(phi)), and a partial angle rescales every amplitude by sqrt((theta/2)/|chi|)
(the s^2 law of Section 4.4.7 (7), closure re-checked by the calibration). ZZ(theta) = exp(-i (theta/2) Z Z) on an MS
waveform is the inferred wrapper construction of Sections 7.6 and 12 (GPi2(3 pi/2) on both ions, MS(0, 0, theta), GPi2(pi/2)
on both: R_y(pi/2) X R_y(pi/2)^dag = -Z), labelled as such in the gate ids; on a light-shift waveform it is the direct
sigma_z sigma_z force in Ballance's and Baldwin's spin-echo form (two half-angle pulses around a pi pulse on both ions,
Section 4.4.4), whose sign can only be chosen through the detuning side.

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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.control.compiler import NATIVE_GATES, Circuit
from qutip_trap.control.pulses import Drive, DriveKind, LightShiftCouplings, Pulse, Tone
from qutip_trap.control.table import Waveform
from qutip_trap.dynamics.frames import PhaseFrame
from qutip_trap.transport.budget import Transport
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from collections.abc import Callable

    from qutip_trap.control.table import CalibrationTable, Segment
    from qutip_trap.device.model import Device

M6 = "milestone M6 (control/schedule.py, events and mid-circuit operations, PLAN.md Sections 7.2, 7.3)"
FORCE_AXIS_OFFSET_RAD = math.pi / 2.0
"""The bichromatic force acts about phi_s + pi/2 in the same-Delta-k geometry (Section 4.3.4): the i of i eta (a + a^dag)."""

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
    """How gates on one ion are driven: the drive kind and the beams that implement it (Section 7.3); for a light-shift
    entangling drive the level couplings the beams produce (``light_shift``, from ``derive_light_shift_drive``)."""

    kind: DriveKind
    beams: tuple[int, ...]
    light_shift: LightShiftCouplings | None = None

    def __post_init__(self) -> None:
        if (self.kind == "light_shift") != (self.light_shift is not None):
            raise ValueError(
                "a light_shift gate drive carries its LightShiftCouplings and no other kind does"
            )

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


def _stark_for_segment(
    table: CalibrationTable, ion: int, gate_drive: GateDrive, segment: Segment
) -> float | Callable[[float], float]:
    """The table's Stark entry scaled with the played intensity: delta_cal sum_legs (Omega_leg/Omega_cal)^2 (Section 4.3.2), or 0
    when the table has no calibrated Stark or Rabi entry for (ion, beam)."""
    key = (ion, gate_drive.table_key_beam)
    stark = table.stark.get(key)
    rabi = table.rabi.get(key)
    if stark is None or rabi is None or stark.status == "uncalibrated" or rabi.status == "uncalibrated":
        return 0.0
    if rabi.value <= 0.0:
        return 0.0
    amps = [segment.amplitude_hz[(ion, leg)] for leg in segment.legs]
    constants = [a for a in amps if not callable(a)]
    if len(constants) == len(amps):
        return float(stark.value) * float(sum((float(a) / rabi.value) ** 2 for a in constants))

    def shift(tau: float) -> float:
        total = 0.0
        for a in amps:
            val = float(a(tau)) if callable(a) else float(a)
            total += (val / rabi.value) ** 2
        return float(stark.value) * total

    return shift


def entangling_pulses(
    waveform: Waveform,
    gate_drives: dict[int, GateDrive],
    *,
    spin_phases_rad: dict[int, float],
    t_start_s: float,
    table: CalibrationTable,
    gate_id: str,
    crosstalk: dict[int, dict[int, complex]] | None = None,
) -> list[Pulse]:
    """One Pulse per (segment, ion) from a calibrated Waveform: tones per leg with the segment's phase offsets plus the ion's spin
    phase (already in its frame), the segment's amplitudes and detunings, the Stark shift scaled with the played intensity."""
    if waveform.segments is None:
        raise ScheduleError("the scheduler plays segmented waveforms (the solvers emit segments)")
    pulses: list[Pulse] = []
    t = t_start_s
    for k, seg in enumerate(waveform.segments):
        for ion in seg.ions:
            spec = gate_drives[ion]
            if waveform.kind == "light_shift" and spec.kind != "light_shift":
                raise ScheduleError(
                    f"ion {ion}: a light-shift waveform needs a light_shift gate drive (Section 4.4.4)"
                )
            if waveform.kind == "ms" and spec.kind not in ("raman", "optical_E1", "optical_E2", "microwave"):
                raise ScheduleError(f"ion {ion}: an MS waveform needs a spin-flip drive, not {spec.kind}")
            tones = tuple(
                Tone(
                    detuning_hz=seg.detuning_hz[leg],
                    phase_rad=float(seg.phase_rad[(ion, leg)]) + float(spin_phases_rad.get(ion, 0.0)),
                    envelope_hz=seg.amplitude_hz[(ion, leg)],
                )
                for leg in seg.legs
            )
            drive = Drive(
                kind=spec.kind,
                ions=(ion,),
                tones=tones,
                beams=spec.beams,
                stark_shift_hz=_stark_for_segment(table, ion, spec, seg),
                crosstalk=dict((crosstalk or {}).get(ion, {})),
                light_shift=spec.light_shift,
            )
            pulses.append(
                Pulse(
                    drive,
                    t,
                    t + seg.duration_s,
                    f"{gate_id}/seg{k}/ion{ion}" if len(waveform.segments) > 1 else f"{gate_id}/ion{ion}",
                    tuple(waveform.chi_m),
                )
            )
        t += seg.duration_s
    return pulses


def ms_spin_phases(
    waveform: Waveform, pair: tuple[int, int], phases_rad: tuple[float, float], frame: PhaseFrame
) -> tuple[dict[int, float], float]:
    """(spin phase per ion in its frame, amplitude factor) for MS(phi_0, phi_1, theta) on ``waveform`` (Section 4.4.2):
    phi_s,i = frame(phi_i) - pi/2 (the force axis sits at phi_s + pi/2), plus pi on the second ion when the waveform's chi is positive,
    because the pulse applies exp(+i chi sigma sigma) and the native gate is exp(-i (theta/2) sigma sigma)."""
    a, b = pair
    chi = waveform.chi_total_rad
    if chi == 0.0:
        raise ScheduleError("the waveform carries no entangling angle")
    phases = {
        a: frame.pulse_phase(a, phases_rad[0]) - FORCE_AXIS_OFFSET_RAD,
        b: frame.pulse_phase(b, phases_rad[1]) - FORCE_AXIS_OFFSET_RAD + (math.pi if chi > 0.0 else 0.0),
    }
    return phases, abs(chi)


def _single_qubit(
    q: int,
    area: float,
    phase: float,
    drives: dict[int, GateDrive],
    table: CalibrationTable,
    start: float,
    gate_id: str,
) -> Pulse:
    spec = drives[q]
    rabi = carrier_rabi_hz(table, q, spec)
    stark_entry = table.stark.get((q, spec.table_key_beam))
    stark = 0.0 if stark_entry is None or stark_entry.status == "uncalibrated" else float(stark_entry.value)
    xt: dict[int, complex] = {}
    for (i, j), entry in table.crosstalk.items():
        if i == q and entry.status != "uncalibrated" and entry.value != 0.0:
            xt[j] = complex(entry.value)
    return single_qubit_pulse(
        q, area, phase, spec, rabi, start, stark_shift_hz=stark, gate_id=gate_id, crosstalk=xt
    )


def schedule(
    circuit: Circuit,
    device: Device,
    table: CalibrationTable,
    *,
    gate_drives: dict[int, GateDrive] | None = None,
    entangling_drives: dict[int, GateDrive] | None = None,
    t0_s: float = 0.0,
    parallel: bool = False,
) -> Schedule:
    """Native gates -> pulses with absolute times from the calibration table (Section 7.3).

    gpi and gpi2 become carrier pulses of area pi and pi/2 at the frame-shifted phase; rz is a frame update (no pulse,
    no time beyond the dead time the hardware inserts between pulses, Section 7.6); ms and zz play the pair's calibrated
    Waveform (M4, see the module docstring; ``entangling_drives`` default to ``gate_drives``, the same beam pairs); a
    terminal ``measure`` is not an event of the schedule (run() performs it); mid-circuit measure/reset/recool are M6.
    """
    if not circuit.is_native:
        raise ScheduleError("schedule() takes a native circuit; compile_to_native first (Section 7.2)")
    drives = gate_drives or default_gate_drives(device)
    ent_drives = entangling_drives or drives
    dead = float(device.hardware.dead_time_s)
    frame = PhaseFrame()
    pulses: list[Pulse] = []
    idle: list[tuple[float, float]] = []
    clock: dict[int, float] = {q: t0_s for q in range(circuit.n_qubits)}
    global_clock = t0_s

    def advance(qubits: tuple[int, ...], end: float) -> None:
        nonlocal global_clock
        if dead > 0.0:
            idle.append((end, end + dead))
        if parallel:
            for q in qubits:
                clock[q] = end + dead
        else:
            global_clock = end + dead

    for k, op in enumerate(circuit.ops):
        if op.name == "rz":
            frame = frame.rz(op.qubits[0], op.params[0])
            continue
        if op.name in ("ms", "zz"):
            a, b = op.qubits
            wf = table.waveform_for((a, b))
            if wf is None:
                raise ScheduleError(
                    f"no entangling waveform in the calibration table for ions {(a, b)} (CalibrationTable.ms; Section 7.5)"
                )
            start = max(clock[a], clock[b]) if parallel else global_clock
            if op.name == "ms":
                phi0, phi1, theta = op.params
                if theta < 0.0:
                    phi1, theta = phi1 + math.pi, -theta
                if wf.kind != "ms":
                    raise ScheduleError(
                        "MS(phi_0, phi_1, theta) needs an MS (spin-flip) waveform; zz plays a light-shift one"
                    )
                spins, chi_abs = ms_spin_phases(wf, (a, b), (phi0, phi1), frame)
                play = _rescaled(wf, 0.5 * theta, chi_abs)
                new = entangling_pulses(
                    play, ent_drives, spin_phases_rad=spins, t_start_s=start, table=table, gate_id=f"ms[{k}]"
                )
                pulses.extend(new)
                advance((a, b), start + play.duration_s)
                continue
            (theta,) = op.params
            if wf.kind == "ms":
                # inferred construction (Sections 7.6, 12): GPi2(3 pi/2) both, MS(0, 0, theta), GPi2(pi/2) both
                for q in (a, b):
                    p = _single_qubit(
                        q,
                        NATIVE_AREAS["gpi2"],
                        frame.pulse_phase(q, 1.5 * math.pi),
                        drives,
                        table,
                        start,
                        f"zz[{k}]/wrap_in/ion{q}",
                    )
                    pulses.append(p)
                    start = max(start, p.t_end_s) if not parallel else start
                if not parallel:
                    start = max(p.t_end_s for p in pulses[-2:]) + dead
                    idle.append((start - dead, start))
                else:
                    start = max(p.t_end_s for p in pulses[-2:]) + dead
                    idle.append((start - dead, start))
                spins, chi_abs = ms_spin_phases(wf, (a, b), (0.0, 0.0), frame)
                play = _rescaled(wf, 0.5 * abs(theta), chi_abs)
                if theta < 0.0:
                    spins[b] += math.pi
                pulses.extend(
                    entangling_pulses(
                        play,
                        ent_drives,
                        spin_phases_rad=spins,
                        t_start_s=start,
                        table=table,
                        gate_id=f"zz[{k}]/ms",
                    )
                )
                start = start + play.duration_s + dead
                idle.append((start - dead, start))
                ends = []
                for q in (a, b):
                    p = _single_qubit(
                        q,
                        NATIVE_AREAS["gpi2"],
                        frame.pulse_phase(q, 0.5 * math.pi),
                        drives,
                        table,
                        start,
                        f"zz[{k}]/wrap_out/ion{q}",
                    )
                    pulses.append(p)
                    ends.append(p.t_end_s)
                advance((a, b), max(ends))
                continue
            # light-shift sigma_z sigma_z gate in the spin-echo form of Section 4.4.4: two half-angle loops around a pi on both ions
            chi = wf.chi_total_rad
            if chi * theta > 0.0:
                raise ScheduleError(
                    f"the light-shift waveform applies exp(+i {chi:.4g} sigma_z sigma_z) per pulse; ZZ({theta:.4g}) needs the "
                    "opposite sign, which a sigma_z force can only take from the other detuning side (Section 4.4.4)"
                )
            half = _rescaled(wf, 0.25 * abs(theta), abs(chi))
            spins = {a: 0.0, b: 0.0}
            pulses.extend(
                entangling_pulses(
                    half,
                    ent_drives,
                    spin_phases_rad=spins,
                    t_start_s=start,
                    table=table,
                    gate_id=f"zz[{k}]/loop1",
                )
            )
            start += half.duration_s + dead
            idle.append((start - dead, start))
            ends = []
            for q in (a, b):
                p = _single_qubit(
                    q,
                    NATIVE_AREAS["gpi"],
                    frame.pulse_phase(q, 0.0),
                    drives,
                    table,
                    start,
                    f"zz[{k}]/echo/ion{q}",
                )
                pulses.append(p)
                ends.append(p.t_end_s)
            start = max(ends) + dead
            idle.append((start - dead, start))
            pulses.extend(
                entangling_pulses(
                    half,
                    ent_drives,
                    spin_phases_rad=spins,
                    t_start_s=start,
                    table=table,
                    gate_id=f"zz[{k}]/loop2",
                )
            )
            start += half.duration_s + dead
            idle.append((start - dead, start))
            ends = []
            for q in (a, b):
                p = _single_qubit(
                    q,
                    NATIVE_AREAS["gpi"],
                    frame.pulse_phase(q, math.pi),
                    drives,
                    table,
                    start,
                    f"zz[{k}]/unecho/ion{q}",
                )
                pulses.append(p)
                ends.append(p.t_end_s)
            advance((a, b), max(ends))
            continue
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
        advance((q,), pulse.t_end_s)
    return Schedule(tuple(pulses), tuple(idle), (), frame.as_dict(circuit.n_qubits))


def _rescaled(waveform: Waveform, chi_target_abs: float, chi_abs: float) -> Waveform:
    """The waveform scaled to |chi| = chi_target_abs by the s^2 law (Section 4.4.7 (7)); the identity at the calibrated angle."""
    from qutip_trap.control.shaping import scaled

    if chi_target_abs <= 0.0:
        raise ScheduleError("an entangling gate needs a positive angle; a zero-angle gate is no pulse")
    factor = math.sqrt(chi_target_abs / chi_abs)
    return waveform if abs(factor - 1.0) < 1e-12 else scaled(waveform, factor)


__all__ = [
    "FORCE_AXIS_OFFSET_RAD",
    "MICROWAVE_BEAM_KEY",
    "NATIVE_AREAS",
    "GateDrive",
    "Schedule",
    "ScheduleError",
    "ScheduledEvent",
    "carrier_rabi_hz",
    "default_gate_drives",
    "entangling_pulses",
    "ms_spin_phases",
    "schedule",
    "single_qubit_pulse",
]
