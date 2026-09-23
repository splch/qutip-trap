"""Schedules and the scheduler: native gates -> pulses with absolute times, from the calibration table's beliefs.

GPi and GPi2 are resonant carrier pulses of area pi and pi/2 at the frame-shifted phase, lasting area/(2 pi f_Rabi) with
f_Rabi from the table (never the device's true value); rz shifts every later pulse phase phi -> phi - theta; ms and zz
play the pair's calibrated ``Waveform``. Pulses carry the table's values (``Drive.programmed``, made physical by
``control.played``) with the believed Stark shift compensated and its phase absorbed into the frame, so the ideal target
is the ideal gates followed by RZ(phase_frame) per ion. Dead time is idle between pulses; a measure, reset or recool
before a later gate raises ``ScheduleError``.
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
from qutip_trap.light.roles import gate_beams
from qutip_trap.transport.budget import Transport
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from qutip_trap.control.table import CalibrationTable, Segment
    from qutip_trap.device.model import Device

M6_MID_CIRCUIT = (
    "mid-circuit measure, reset and recool are refused in the first release (PLAN.md Section 7.2 item 4): the IR and the "
    "Schedule carry their positions, the physics of Section 8.5 (detection recoil, neighbour Stark shift, depumping, recooling "
    "and re-preparation of the measured ion) is the stage still to be implemented"
)
FORCE_AXIS_OFFSET_RAD = math.pi / 2.0
"""The bichromatic force acts about phi_s + pi/2 in the same-Delta-k geometry: the i of i eta (a + a^dag)."""

MICROWAVE_BEAM_KEY = -1
"""The beam index that keys a microwave drive's carrier Rabi frequency in ``CalibrationTable.rabi`` (no beams)."""


class ScheduleError(ValueError):
    """The scheduler refused: an uncalibrated entry, an ambiguous drive or an unsupported operation."""


@dataclass(frozen=True)
class ScheduledEvent:
    """measure / reset / recool with absolute times."""

    kind: Literal["measure", "reset", "recool"]
    ions: tuple[int, ...]
    t_start_s: float
    t_end_s: float

    def __post_init__(self) -> None:
        if self.t_end_s < self.t_start_s:
            raise ValueError("an event cannot end before it starts")


@dataclass(frozen=True)
class PlayedGate:
    """One entangling gate as the scheduler played it."""

    gate_id: str
    kind: Literal["ms", "zz"]
    pair: tuple[int, int]
    waveform: Waveform
    """The waveform AS PLAYED (rescaled to the gate's angle), with its signed chi_m and alpha_m per mode."""
    beams: tuple[int, ...]
    """The entangling drive's beams (the Raman pair whose Delta k sets the Lamb-Dicke parameters)."""
    t_start_s: float
    t_end_s: float


@dataclass(frozen=True)
class GateTarget:
    """The ideal physical unitary F G of one played gate piece, the target of a channel extracted from its pulses: G the
    matrix of ``native`` (phases already frame-applied), F = (x)_q RZ(-theta_q) with theta_q the ``stark_frame_rad``
    absorbed for the compensated light shift, on ``ions`` in matrix order (first ion = first factor)."""

    gate_id: str
    ions: tuple[int, ...]
    native: tuple[str, tuple[float, ...]]
    stark_frame_rad: dict[int, float]
    pulse_ids: tuple[str, ...]
    t_start_s: float
    t_end_s: float

    def unitary(self) -> np.ndarray:
        from qutip_trap.control import native as _native
        from qutip_trap.control.compiler import Operation, gate_matrix

        name, params = self.native
        g = gate_matrix(Operation(name, tuple(range(len(self.ions))), tuple(float(p) for p in params)))
        f = np.array([[1.0 + 0.0j]])
        for q in self.ions:
            f = np.kron(f, _native.rz(-float(self.stark_frame_rad.get(q, 0.0))))
        return np.asarray(f @ g)


@dataclass(frozen=True)
class Schedule:
    """A circuit as pulses on the time axis, with its idle intervals, events and final virtual-Z frame."""

    pulses: tuple[Pulse, ...]
    idle: tuple[tuple[float, float], ...]
    """Idle intervals (start, end) during which heating and dephasing act."""
    events: tuple[ScheduledEvent, ...]
    phase_frame: dict[int, float]
    """Per-qubit virtual-Z frame at the end (radians)."""
    transports: tuple[Transport, ...] = ()
    """Ion transports, interleaved with ``pulses`` by absolute time."""
    gates: tuple[PlayedGate, ...] = ()
    """The entangling gates as played, in time order."""
    targets: tuple[GateTarget, ...] = ()
    t0_s: float | None = None
    """The time the state handed to the engine is given at; None = min(0, the first pulse or idle start)."""

    def __post_init__(self) -> None:
        for a, b in self.idle:
            if b < a:
                raise ValueError("idle intervals must be ordered (start, end)")
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

    @property
    def pulses_end_s(self) -> float:
        """The end of the last pulse or idle interval (the measurement event starts here)."""
        ends = [p.t_end_s for p in self.pulses] + [b for _, b in self.idle]
        return max(ends) if ends else 0.0

    @property
    def measurement(self) -> ScheduledEvent | None:
        """The terminal measurement event, if the circuit measures."""
        for e in self.events:
            if e.kind == "measure":
                return e
        return None


@dataclass(frozen=True)
class GateDrive:
    """How gates on one ion are driven: the drive kind, its beams and, for a light-shift drive, its level couplings."""

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


def infer_gate_drives(device: Device) -> dict[int, GateDrive]:
    """Infer the single-qubit gate drive from the device's far-detuned beams: none = microwave, one = optical, a pair of
    equal wavelength = Raman; anything else raises ``ScheduleError`` and must be declared (``Device.roles``)."""
    n = device.crystal.n_ions
    idx = gate_beams(device)  # cooling, detection and repump light drives no gate
    beams = [device.beams[k] for k in idx]
    if len(beams) == 0:
        spec = GateDrive("microwave", ())
    elif len(beams) == 1:
        species = device.crystal.species[0]
        lo, up = (lab.split()[0] for lab in species.qubit)
        e2 = any(t.multipole == "E2" and {t.lower, t.upper} == {lo, up} for t in species.transitions)
        spec = GateDrive("optical_E2" if e2 else "optical_E1", (idx[0],))
    elif (
        len(beams) == 2 and abs(beams[0].wavelength_m - beams[1].wavelength_m) <= 1e-3 * beams[0].wavelength_m
    ):
        spec = GateDrive("raman", (idx[0], idx[1]))
    else:
        raise ScheduleError(
            "the device's far-detuned beams do not identify a single-qubit gate drive (need none, one, or one Raman pair); "
            "declare them in Device.roles (or pass gate_drives)"
        )
    return {i: spec for i in range(n)}


def default_gate_drives(device: Device) -> dict[int, GateDrive]:
    """The single-qubit gate drives of ``device``: its declared ``roles.gate``, else :func:`infer_gate_drives`."""
    return device.roles.resolve(device).gate


def resolve_drives(
    device: Device,
    gate_drives: Mapping[int, GateDrive] | None = None,
    entangling_drives: Mapping[int, GateDrive] | None = None,
) -> tuple[dict[int, GateDrive], dict[int, GateDrive]]:
    """(single-qubit drives, entangling drives): explicit arguments over the device's roles over the inference."""
    resolved = device.roles.resolve(device, gate_drives=gate_drives, entangling_drives=entangling_drives)
    return resolved.gate, resolved.entangling


def carrier_rabi_hz(table: CalibrationTable, ion: int, gate_drive: GateDrive) -> float:
    """The table's carrier Rabi frequency for (ion, drive); ScheduleError if absent, uncalibrated or not positive."""
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
    stark_compensation: bool = True,
    programmed: bool = True,
) -> Pulse:
    """A square carrier pulse of ``area_rad`` (2 pi f_Rabi t = area) at ``phase_rad`` in the ion's frame; with
    ``stark_compensation`` the tone is detuned by the believed ``stark_shift_hz`` to stay resonant with the shifted qubit."""
    if area_rad <= 0.0:
        raise ValueError(
            "a pulse area is positive; a negative nominal area is played at phase + pi (Section 13)"
        )
    if rabi_hz <= 0.0:
        raise ValueError("rabi_hz must be positive")
    duration = area_rad / (TWO_PI * rabi_hz)
    compensation = float(stark_shift_hz) if stark_compensation else 0.0
    tone = Tone(
        detuning_hz=float(detuning_hz) + compensation,
        phase_rad=float(phase_rad) + compensation_phase_rad(compensation, t_start_s),
        envelope_hz=float(rabi_hz),
    )
    drive = Drive(
        kind=gate_drive.kind,
        ions=(ion,),
        tones=(tone,),
        beams=gate_drive.beams,
        stark_shift_hz=float(stark_shift_hz),
        crosstalk=dict(crosstalk or {}),
        programmed=programmed,
    )
    return Pulse(drive, t_start_s, t_start_s + duration, gate_id, ())


NATIVE_AREAS: dict[str, float] = {"gpi": float(np.pi), "gpi2": float(np.pi / 2.0)}


def stark_scaling_power(kind: DriveKind) -> int:
    """The power p in delta_Stark ~ Omega^p: 1 for a two-photon (Raman, light-shift) drive, whose Omega and shift are both
    linear in intensity; 2 for a single-photon optical or a microwave drive."""
    return 1 if kind in ("raman", "light_shift") else 2


def _stark_for_segment(
    table: CalibrationTable, ion: int, gate_drive: GateDrive, segment: Segment
) -> float | Callable[[float], float]:
    """delta_cal sum_legs (Omega_leg/Omega_cal)^p, or 0 without usable Stark and Rabi entries for (ion, beam)."""
    key = (ion, gate_drive.table_key_beam)
    stark = table.stark.get(key)
    rabi = table.rabi.get(key)
    if stark is None or rabi is None or stark.status == "uncalibrated" or rabi.status == "uncalibrated":
        return 0.0
    if rabi.value <= 0.0:
        return 0.0
    power = stark_scaling_power(gate_drive.kind)
    amps = [segment.amplitude_hz[(ion, leg)] for leg in segment.legs]
    constants = [a for a in amps if not callable(a)]
    if len(constants) == len(amps):
        return float(stark.value) * float(sum((float(a) / rabi.value) ** power for a in constants))

    def shift(tau: float) -> float:
        total = 0.0
        for a in amps:
            val = float(a(tau)) if callable(a) else float(a)
            total += (val / rabi.value) ** power
        return float(stark.value) * total

    return shift


def _compensated_detuning(
    detuning_hz: float | Callable[[float], float], shift_hz: float | Callable[[float], float]
) -> float | Callable[[float], float]:
    """The leg's detuning plus the believed Stark shift, composing callables (an FM leg, a shaped amplitude) when needed."""
    if not callable(detuning_hz) and not callable(shift_hz):
        return float(detuning_hz) + float(shift_hz)
    if not callable(shift_hz) and float(shift_hz) == 0.0:
        return detuning_hz

    def value(tau: float) -> float:
        d = float(detuning_hz(tau)) if callable(detuning_hz) else float(detuning_hz)
        sft = float(shift_hz(tau)) if callable(shift_hz) else float(shift_hz)
        return d + sft

    return value


def crosstalk_beliefs(table: CalibrationTable, ion: int) -> dict[int, complex]:
    """The table's crosstalk of ``ion`` onto its neighbours as complex ratios: |epsilon_ij| from ``crosstalk`` and arg from
    ``crosstalk_phase`` (0 when absent); uncalibrated or zero entries are skipped."""
    xt: dict[int, complex] = {}
    for (i, j), entry in table.crosstalk.items():
        if i == ion and entry.status != "uncalibrated" and entry.value != 0.0:
            phase = table.crosstalk_phase.get((i, j))
            arg = float(phase.value) if phase is not None and phase.status != "uncalibrated" else 0.0
            xt[j] = complex(entry.value) * complex(math.cos(arg), math.sin(arg))
    return xt


def stark_phase_rad(pulse: Pulse) -> float:
    """2 pi int delta_St,believed dt over the pulse, the virtual-Z offset of the compensated light shift: the shift rotates
    the state by RZ(-2 pi delta t), what a virtual RZ(+2 pi delta t) leaves, so later pulses carry phi - 2 pi delta t."""
    shift = pulse.drive.stark_shift_hz
    if callable(shift):
        grid = np.linspace(0.0, pulse.duration_s, 201)
        integral = float(np.trapezoid([float(shift(x)) for x in grid], grid))
    else:
        integral = float(shift) * pulse.duration_s
    return float(TWO_PI * integral)


def frame_after(pulses: Sequence[Pulse], frame: PhaseFrame, *, stark_compensation: bool = True) -> PhaseFrame:
    """The frame after ``pulses`` with their believed Stark phases absorbed (per ion, the primary ion of each drive)."""
    if not stark_compensation:
        return frame
    for p in pulses:
        theta = stark_phase_rad(p)
        if theta != 0.0:
            frame = frame.rz(p.drive.ions[0], theta)
    return frame


def compensation_phase_rad(shift_hz: float | Callable[[float], float], t_start_s: float) -> float:
    """2 pi delta_s(0) t_s mod 2 pi: the tone phase that references a compensation detuning delta_s to the qubit frame at the
    pulse start t_s, since the builder plays the tone in absolute time but the qubit is shifted only from t_s on."""
    d0 = float(shift_hz(0.0)) if callable(shift_hz) else float(shift_hz)
    if d0 == 0.0:
        return 0.0
    return (TWO_PI * d0 * float(t_start_s)) % TWO_PI


def beat_phase_offset_rad(detuning_hz: float | Callable[[float], float], t_gate_start_s: float) -> float:
    """2 pi mu t_g mod 2 pi: the tone phase that restarts a running beat note at zero phase at the gate start t_g; the red
    and blue legs shift oppositely, leaving the spin phase alone. 0 for a frequency-modulated (callable) leg."""
    if callable(detuning_hz):
        return 0.0
    return (TWO_PI * float(detuning_hz) * float(t_gate_start_s)) % TWO_PI


def response_phase_rad(detuning_hz: float | Callable[[float], float], delay_s: float) -> float:
    """arctan(2 pi |mu| tau_r) with the sign of mu: the beat-note lag of a first-order modulator response (the phase of
    H(mu) = 1/(1 - i 2 pi mu tau_r)), to compensate; a frequency-modulated leg uses its initial detuning."""
    if delay_s == 0.0:
        return 0.0
    mu = float(detuning_hz(0.0)) if callable(detuning_hz) else float(detuning_hz)
    return math.copysign(math.atan(TWO_PI * abs(mu) * delay_s), mu)


def entangling_pulses(
    waveform: Waveform,
    gate_drives: dict[int, GateDrive],
    *,
    spin_phases_rad: dict[int, float],
    t_start_s: float,
    table: CalibrationTable,
    gate_id: str,
    crosstalk: dict[int, dict[int, complex]] | None = None,
    beat_phase_reset: bool = False,
    response_delay_s: float = 0.0,
    stark_compensation: bool = True,
) -> list[Pulse]:
    """One Pulse per (segment, ion) of a segmented Waveform: per leg a tone at the segment's phase offset plus the ion's spin
    phase (in its frame), with the segment's amplitude and detuning and the believed Stark shift at that amplitude.
    ``stark_compensation`` detunes both legs by that shift and references each segment's spin phase to its own start, less
    the frame accumulated in the gate's earlier segments; ``beat_phase_reset`` (``HardwareChain.phase_continuous = False``)
    starts the beat note at zero phase at the gate start, as when the waveform was calibrated."""
    if waveform.segments is None:
        raise ScheduleError("the scheduler plays segmented waveforms (the solvers emit segments)")
    pulses: list[Pulse] = []
    in_gate_frame: dict[int, float] = {}
    t = t_start_s
    for k, seg in enumerate(waveform.segments):
        for ion in seg.ions:
            spec = gate_drives[ion]
            if waveform.kind in ("light_shift", "gradient") and spec.kind != waveform.kind:
                raise ScheduleError(
                    f"ion {ion}: a {waveform.kind} waveform needs a {waveform.kind} gate drive "
                    "(Sections 4.4.4, 4.4.5)"
                )
            if waveform.kind == "ms" and spec.kind not in ("raman", "optical_E1", "optical_E2", "microwave"):
                raise ScheduleError(f"ion {ion}: an MS waveform needs a spin-flip drive, not {spec.kind}")
            believed_shift = _stark_for_segment(table, ion, spec, seg)
            compensation = believed_shift if stark_compensation else 0.0
            spin_reference = compensation_phase_rad(compensation, t) - in_gate_frame.get(ion, 0.0)
            tones = tuple(
                Tone(
                    detuning_hz=_compensated_detuning(seg.detuning_hz[leg], compensation),
                    phase_rad=float(seg.phase_rad[(ion, leg)])
                    + float(spin_phases_rad.get(ion, 0.0))
                    + spin_reference
                    + (beat_phase_offset_rad(seg.detuning_hz[leg], t_start_s) if beat_phase_reset else 0.0)
                    + response_phase_rad(seg.detuning_hz[leg], response_delay_s),
                    envelope_hz=seg.amplitude_hz[(ion, leg)],
                )
                for leg in seg.legs
            )
            drive = Drive(
                kind=spec.kind,
                ions=(ion,),
                tones=tones,
                beams=spec.beams,
                stark_shift_hz=believed_shift,
                crosstalk=dict((crosstalk or {}).get(ion, {})),
                light_shift=spec.light_shift,
                programmed=True,
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
            if stark_compensation:
                in_gate_frame[ion] = in_gate_frame.get(ion, 0.0) + stark_phase_rad(pulses[-1])
        t += seg.duration_s
    return pulses


def ms_spin_phases(
    waveform: Waveform, pair: tuple[int, int], phases_rad: tuple[float, float], frame: PhaseFrame
) -> tuple[dict[int, float], float]:
    """(spin phase per ion in its frame, |chi| of the waveform) for MS(phi_0, phi_1, theta): phi_s,i = frame(phi_i) - pi/2,
    plus pi on the second ion when chi > 0 (the pulse applies exp(+i chi sigma sigma), MS exp(-i (theta/2) sigma sigma))."""
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
    stark_compensation: bool = True,
) -> Pulse:
    spec = drives[q]
    rabi = carrier_rabi_hz(table, q, spec)
    stark_entry = table.stark.get((q, spec.table_key_beam))
    stark = 0.0 if stark_entry is None or stark_entry.status == "uncalibrated" else float(stark_entry.value)
    return single_qubit_pulse(
        q,
        area,
        phase,
        spec,
        rabi,
        start,
        stark_shift_hz=stark,
        gate_id=gate_id,
        crosstalk=crosstalk_beliefs(table, q),
        stark_compensation=stark_compensation,
    )


CrosstalkSuppression = Literal["none", "neighbour", "local"]


def schedule(
    circuit: Circuit,
    device: Device,
    table: CalibrationTable,
    *,
    gate_drives: dict[int, GateDrive] | None = None,
    entangling_drives: dict[int, GateDrive] | None = None,
    t0_s: float = 0.0,
    parallel: bool | None = None,
    crosstalk_suppression: CrosstalkSuppression = "none",
    response_delay: bool = True,
    stark_compensation: bool = True,
) -> Schedule:
    """Native gates -> pulses with absolute times from the calibration table (see the module docstring).

    ``crosstalk_suppression`` splits every MS gate into two half-angle plays around a physical echo (Fang et al. 2022):
    Y(pi) on both targets between the halves and after the second (``local``), or Z(pi) on every crosstalk spectator
    between the halves (``neighbour``). ``parallel`` defaults to ``HardwareChain.parallel_addressing`` and needs it;
    entangling gates are serialized per crystal either way."""
    if not circuit.is_native:
        raise ScheduleError("schedule() takes a native circuit; compile_to_native first (Section 7.2)")
    allows_parallel = bool(device.hardware.parallel_addressing)
    if parallel is None:
        parallel = allows_parallel
    elif parallel and not allows_parallel:
        raise ScheduleError(
            "parallel=True needs HardwareChain.parallel_addressing = True: Section 7.3 parallelizes single-qubit gates only "
            "if the device model allows parallel addressing"
        )
    if crosstalk_suppression != "none" and parallel:
        raise ScheduleError("crosstalk suppression is scheduled on the serial path (Section 6.6)")
    drives, ent_drives = resolve_drives(device, gate_drives, entangling_drives)
    dead = float(device.hardware.dead_time_s)
    reset = not bool(device.hardware.phase_continuous)
    delay = float(device.hardware.aom_rise_s) if response_delay else 0.0
    frame = PhaseFrame()
    pulses: list[Pulse] = []
    idle: list[tuple[float, float]] = []
    gates: list[PlayedGate] = []
    targets: list[GateTarget] = []

    def record_single(p: Pulse, name: str, phase_played: float) -> None:
        """Record one single-qubit pulse's target: the native gate at its frame-applied phase, then its Stark rotation."""
        q = p.drive.ions[0]
        gid = p.gate_id or ""
        targets.append(
            GateTarget(
                gid,
                (q,),
                (name, (float(phase_played),)),
                {q: stark_phase_rad(p) if stark_compensation else 0.0},
                (gid,),
                p.t_start_s,
                p.t_end_s,
            )
        )

    def record_pair(
        gate_id: str,
        pair: tuple[int, int],
        native_op: tuple[str, tuple[float, ...]],
        before: PhaseFrame,
        after: PhaseFrame,
        new_pulses: Sequence[Pulse],
    ) -> None:
        targets.append(
            GateTarget(
                gate_id,
                pair,
                native_op,
                {q: after.offset(q) - before.offset(q) for q in pair},
                tuple(p.gate_id or "" for p in new_pulses),
                min(p.t_start_s for p in new_pulses),
                max(p.t_end_s for p in new_pulses),
            )
        )

    clock: dict[int, float] = {q: t0_s for q in range(circuit.n_qubits)}
    global_clock = t0_s
    last_unitary = max([k for k, op in enumerate(circuit.ops) if not op.is_non_unitary], default=-1)
    measured: list[int] = list(circuit.measure)
    for k, op in enumerate(circuit.ops):
        if op.is_non_unitary and (op.name != "measure" or k < last_unitary):
            raise ScheduleError(f"{op.name!r} at position {k}: {M6_MID_CIRCUIT}")
        if op.name == "measure":
            measured.extend(q for q in op.qubits if q not in measured)

    def advance(qubits: tuple[int, ...], end: float, *, serialized: bool = False) -> None:
        """Move the clocks past a gate that ended at ``end``, leaving the dead time idle; under ``parallel`` a ``serialized``
        (entangling) gate also advances ``global_clock``, which the next entangling gate waits on."""
        nonlocal global_clock
        if dead > 0.0:
            idle.append((end, end + dead))
        if parallel:
            for q in qubits:
                clock[q] = end + dead
            if serialized:
                global_clock = end + dead
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
            # entangling gates are serialized per crystal: they wait on global_clock even under parallel addressing
            start = max(global_clock, clock[a], clock[b]) if parallel else global_clock
            frame_op = frame
            if op.name == "ms":
                phi0, phi1, theta = op.params
                if theta < 0.0:
                    phi1, theta = phi1 + math.pi, -theta
                if wf.kind != "ms":
                    raise ScheduleError(
                        "MS(phi_0, phi_1, theta) needs an MS (spin-flip) waveform; zz plays a light-shift or "
                        "microwave-gradient one"
                    )
                spins, chi_abs = ms_spin_phases(wf, (a, b), (phi0, phi1), frame)
                ms_native = ("ms", (frame_op.pulse_phase(a, phi0), frame_op.pulse_phase(b, phi1), theta))
                if crosstalk_suppression == "none":
                    play = _rescaled(wf, 0.5 * theta, chi_abs)
                    new = entangling_pulses(
                        play,
                        ent_drives,
                        spin_phases_rad=spins,
                        t_start_s=start,
                        table=table,
                        gate_id=f"ms[{k}]",
                        beat_phase_reset=reset,
                        response_delay_s=delay,
                        stark_compensation=stark_compensation,
                    )
                    pulses.extend(new)
                    frame = frame_after(new, frame, stark_compensation=stark_compensation)
                    record_pair(f"ms[{k}]", (a, b), ms_native, frame_op, frame, new)
                    gates.append(
                        PlayedGate(
                            f"ms[{k}]",
                            "ms",
                            (a, b),
                            play,
                            ent_drives[a].beams,
                            start,
                            start + play.duration_s,
                        )
                    )
                    advance((a, b), start + play.duration_s, serialized=True)
                    continue
                half = _rescaled(wf, 0.25 * theta, chi_abs)
                spectators = sorted(
                    {
                        j
                        for (i, j), e in table.crosstalk.items()
                        if i in (a, b) and j not in (a, b) and e.status != "uncalibrated" and e.value != 0.0
                    }
                )
                echo_ions = [a, b] if crosstalk_suppression == "local" else spectators
                echo_pulses: list[tuple[float, float]] = (
                    [(NATIVE_AREAS["gpi"], 0.5 * math.pi)]
                    if crosstalk_suppression == "local"
                    else [(NATIVE_AREAS["gpi"], 0.0), (NATIVE_AREAS["gpi"], 0.5 * math.pi)]
                )

                def play_half(
                    t_half: float,
                    tag: str,
                    *,
                    _half: Waveform = half,
                    _spins: dict[int, float] = spins,
                    _k: int = k,
                    _ab: tuple[int, int] = (a, b),
                    _native: tuple[str, tuple[float, ...]] = (
                        "ms",
                        (ms_native[1][0], ms_native[1][1], 0.5 * theta),
                    ),
                ) -> float:
                    nonlocal frame
                    frame_half = frame
                    new_half = entangling_pulses(
                        _half,
                        ent_drives,
                        spin_phases_rad=_spins,
                        t_start_s=t_half,
                        table=table,
                        gate_id=f"ms[{_k}]/{tag}",
                        beat_phase_reset=reset,
                        response_delay_s=delay,
                        stark_compensation=stark_compensation,
                    )
                    pulses.extend(new_half)
                    frame = frame_after(new_half, frame, stark_compensation=stark_compensation)
                    record_pair(f"ms[{_k}]/{tag}", _ab, _native, frame_half, frame, new_half)
                    gates.append(
                        PlayedGate(
                            f"ms[{_k}]/{tag}",
                            "ms",
                            _ab,
                            _half,
                            ent_drives[_ab[0]].beams,
                            t_half,
                            t_half + _half.duration_s,
                        )
                    )
                    return t_half + _half.duration_s

                def play_echo(
                    t_echo: float,
                    tag: str,
                    *,
                    _ions: list[int] = echo_ions,
                    _echo: list[tuple[float, float]] = echo_pulses,
                    _k: int = k,
                ) -> float:
                    nonlocal frame
                    _frame = frame
                    end = t_echo
                    for q in _ions:
                        t_q = t_echo
                        for area, phase in _echo:
                            ph = _frame.pulse_phase(q, phase)
                            p = _single_qubit(
                                q,
                                area,
                                ph,
                                drives,
                                table,
                                t_q,
                                f"ms[{_k}]/{tag}/ion{q}",
                                stark_compensation=stark_compensation,
                            )
                            pulses.append(p)
                            record_single(p, "gpi" if area == NATIVE_AREAS["gpi"] else "gpi2", ph)
                            _frame = frame_after([p], _frame, stark_compensation=stark_compensation)
                            t_q = p.t_end_s + dead
                        end = max(end, t_q - dead)
                    frame = _frame
                    return end

                t = play_half(start, "half1")
                idle.append((t, t + dead))
                t = t + dead
                if echo_ions:
                    t = play_echo(t, "echo")
                    idle.append((t, t + dead))
                    t = t + dead
                t = play_half(t, "half2")
                if crosstalk_suppression == "local":
                    idle.append((t, t + dead))
                    t = t + dead
                    t = play_echo(t, "unecho")
                else:
                    # a physical Z(pi) is a frame operation on every later pulse of the spectator: absorb it
                    for j in echo_ions:
                        frame = frame.rz(j, math.pi)
                advance((a, b), t, serialized=True)
                continue
            (theta,) = op.params
            if wf.kind == "ms":
                # ZZ on an MS waveform: GPi2(3 pi/2) on both, MS(0, 0, theta), GPi2(pi/2) on both
                for q in (a, b):
                    ph = frame.pulse_phase(q, 1.5 * math.pi)
                    p = _single_qubit(
                        q,
                        NATIVE_AREAS["gpi2"],
                        ph,
                        drives,
                        table,
                        start,
                        f"zz[{k}]/wrap_in/ion{q}",
                        stark_compensation=stark_compensation,
                    )
                    pulses.append(p)
                    record_single(p, "gpi2", ph)
                    frame = frame_after([p], frame, stark_compensation=stark_compensation)
                    start = max(start, p.t_end_s) if not parallel else start
                # the wrapper pulses on both ions play together; the MS waits for both
                start = max(p.t_end_s for p in pulses[-2:]) + dead
                idle.append((start - dead, start))
                spins, chi_abs = ms_spin_phases(wf, (a, b), (0.0, 0.0), frame)
                frame_ms = frame
                play = _rescaled(wf, 0.5 * abs(theta), chi_abs)
                if theta < 0.0:
                    spins[b] += math.pi
                new_ms = entangling_pulses(
                    play,
                    ent_drives,
                    spin_phases_rad=spins,
                    t_start_s=start,
                    table=table,
                    gate_id=f"zz[{k}]/ms",
                    beat_phase_reset=reset,
                    response_delay_s=delay,
                    stark_compensation=stark_compensation,
                )
                pulses.extend(new_ms)
                frame = frame_after(new_ms, frame, stark_compensation=stark_compensation)
                record_pair(
                    f"zz[{k}]/ms",
                    (a, b),
                    ("ms", (frame_ms.pulse_phase(a, 0.0), frame_ms.pulse_phase(b, 0.0), theta)),
                    frame_ms,
                    frame,
                    new_ms,
                )
                gates.append(
                    PlayedGate(
                        f"zz[{k}]/ms", "zz", (a, b), play, ent_drives[a].beams, start, start + play.duration_s
                    )
                )
                start = start + play.duration_s + dead
                idle.append((start - dead, start))
                ends = []
                for q in (a, b):
                    ph = frame.pulse_phase(q, 0.5 * math.pi)
                    p = _single_qubit(
                        q,
                        NATIVE_AREAS["gpi2"],
                        ph,
                        drives,
                        table,
                        start,
                        f"zz[{k}]/wrap_out/ion{q}",
                        stark_compensation=stark_compensation,
                    )
                    pulses.append(p)
                    record_single(p, "gpi2", ph)
                    frame = frame_after([p], frame, stark_compensation=stark_compensation)
                    ends.append(p.t_end_s)
                advance((a, b), max(ends), serialized=True)
                continue
            # light-shift or gradient waveform: the sigma_z spin-echo form, two half-angle loops around a pi on both ions
            chi = wf.chi_total_rad
            if chi * theta > 0.0:
                raise ScheduleError(
                    f"the light-shift waveform applies exp(+i {chi:.4g} sigma_z sigma_z) per pulse; ZZ({theta:.4g}) needs the "
                    "opposite sign, which a sigma_z force can only take from the other detuning side (Section 4.4.4)"
                )
            half = _rescaled(wf, 0.25 * abs(theta), abs(chi))
            spins = {a: 0.0, b: 0.0}
            # each loop applies exp(+i chi_half sigma_z sigma_z) with chi_half = -sign(theta) |theta|/4 = zz(theta/2)
            loop_native: tuple[str, tuple[float, ...]] = ("zz", (0.5 * theta,))
            frame_loop = frame
            new_loop1 = entangling_pulses(
                half,
                ent_drives,
                spin_phases_rad=spins,
                t_start_s=start,
                table=table,
                gate_id=f"zz[{k}]/loop1",
                beat_phase_reset=reset,
                response_delay_s=delay,
                stark_compensation=stark_compensation,
            )
            pulses.extend(new_loop1)
            frame = frame_after(new_loop1, frame, stark_compensation=stark_compensation)
            record_pair(f"zz[{k}]/loop1", (a, b), loop_native, frame_loop, frame, new_loop1)
            gates.append(
                PlayedGate(
                    f"zz[{k}]/loop1", "zz", (a, b), half, ent_drives[a].beams, start, start + half.duration_s
                )
            )
            start += half.duration_s + dead
            idle.append((start - dead, start))
            ends = []
            for q in (a, b):
                ph = frame.pulse_phase(q, 0.0)
                p = _single_qubit(
                    q,
                    NATIVE_AREAS["gpi"],
                    ph,
                    drives,
                    table,
                    start,
                    f"zz[{k}]/echo/ion{q}",
                    stark_compensation=stark_compensation,
                )
                pulses.append(p)
                record_single(p, "gpi", ph)
                frame = frame_after([p], frame, stark_compensation=stark_compensation)
                ends.append(p.t_end_s)
            start = max(ends) + dead
            idle.append((start - dead, start))
            frame_loop = frame
            new_loop2 = entangling_pulses(
                half,
                ent_drives,
                spin_phases_rad=spins,
                t_start_s=start,
                table=table,
                gate_id=f"zz[{k}]/loop2",
                beat_phase_reset=reset,
                response_delay_s=delay,
                stark_compensation=stark_compensation,
            )
            pulses.extend(new_loop2)
            frame = frame_after(new_loop2, frame, stark_compensation=stark_compensation)
            record_pair(f"zz[{k}]/loop2", (a, b), loop_native, frame_loop, frame, new_loop2)
            gates.append(
                PlayedGate(
                    f"zz[{k}]/loop2", "zz", (a, b), half, ent_drives[a].beams, start, start + half.duration_s
                )
            )
            start += half.duration_s + dead
            idle.append((start - dead, start))
            ends = []
            for q in (a, b):
                ph = frame.pulse_phase(q, math.pi)
                p = _single_qubit(
                    q,
                    NATIVE_AREAS["gpi"],
                    ph,
                    drives,
                    table,
                    start,
                    f"zz[{k}]/unecho/ion{q}",
                    stark_compensation=stark_compensation,
                )
                pulses.append(p)
                record_single(p, "gpi", ph)
                frame = frame_after([p], frame, stark_compensation=stark_compensation)
                ends.append(p.t_end_s)
            advance((a, b), max(ends), serialized=True)
            continue
        if op.is_non_unitary:
            continue  # a trailing measure: recorded above, scheduled as the terminal event below
        if op.name not in NATIVE_GATES:
            raise ScheduleError(f"unknown native operation {op.name!r}")
        q = op.qubits[0]
        spec = drives[q]
        rabi = carrier_rabi_hz(table, q, spec)
        stark_entry = table.stark.get((q, spec.table_key_beam))
        stark = (
            0.0 if stark_entry is None or stark_entry.status == "uncalibrated" else float(stark_entry.value)
        )
        start = clock[q] if parallel else global_clock
        ph = frame.pulse_phase(q, op.params[0])
        pulse = single_qubit_pulse(
            q,
            NATIVE_AREAS[op.name],
            ph,
            spec,
            rabi,
            start,
            stark_shift_hz=stark,
            gate_id=f"{op.name}[{k}]",
            crosstalk=crosstalk_beliefs(table, q),
            stark_compensation=stark_compensation,
        )
        pulses.append(pulse)
        record_single(pulse, op.name, ph)
        frame = frame_after([pulse], frame, stark_compensation=stark_compensation)
        advance((q,), pulse.t_end_s)
    events: list[ScheduledEvent] = []
    if measured:
        window_entry = table.detection.get("window_s")
        window = (
            float(window_entry.value)
            if window_entry is not None and window_entry.status != "uncalibrated" and window_entry.value > 0.0
            else float(device.detector.window_s)
        )
        ends = [p.t_end_s for p in pulses] + [b for _, b in idle]
        t_meas = max(ends) if ends else t0_s
        events.append(ScheduledEvent("measure", tuple(sorted(measured)), t_meas, t_meas + window))
    return Schedule(
        tuple(pulses),
        tuple(idle),
        tuple(events),
        frame.as_dict(circuit.n_qubits),
        gates=tuple(gates),
        targets=tuple(targets),
    )


def _rescaled(waveform: Waveform, chi_target_abs: float, chi_abs: float) -> Waveform:
    """The waveform with amplitudes scaled by sqrt(chi_target_abs/chi_abs), since chi goes as amplitude squared."""
    from qutip_trap.control.shaping import scaled

    if chi_target_abs <= 0.0:
        raise ScheduleError("an entangling gate needs a positive angle; a zero-angle gate is no pulse")
    factor = math.sqrt(chi_target_abs / chi_abs)
    return waveform if abs(factor - 1.0) < 1e-12 else scaled(waveform, factor)
