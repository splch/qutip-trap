"""Schedules and the scheduler: a native circuit and a ``CalibrationTable`` -> pulses with absolute times (PLAN.md
Section 7.3).

Single-qubit gates: GPi(phi) and GPi2(phi) are resonant carrier pulses of area pi and pi/2 at phase phi in the ion's frame,
of duration area/(2 pi f_Rabi) with f_Rabi the TABLE's carrier Rabi frequency, never the device's true value; an entry the
calibration could not establish refuses to schedule. RZ(theta) is a frame update, phi -> phi - theta on every later pulse
of the ion (time order); the frame at the end is ``Schedule.phase_frame``. The hardware's dead time follows every pulse as
an idle interval. Single-qubit gates on distinct ions run sequentially, or in parallel when the chain allows it, and so do
the pulses one gate plays on several ions (the ZZ wrappers and echoes, the crosstalk echoes).

Entangling gates play the pair's calibrated ``Waveform`` as one ``Pulse`` per (ion, segment) with the red and blue tones
at the segment's phi_s -/+ phi_m plus the gate phase in the ion's frame. The force acts about phi_s + pi/2, so
MS(phi_0, phi_1, theta) sets phi_s,i = phi_i - pi/2; the pulse applies exp(+i chi sigma sigma), so a positive chi is played
with a pi on the second ion's tones, and a partial angle rescales every amplitude by sqrt((theta/2)/|chi|) (the s^2 law).
ZZ(theta) on an MS waveform is the wrapper GPi2(3 pi/2) on both ions, MS(0, 0, theta), GPi2(pi/2) on both
(R_y(pi/2) X R_y(pi/2)^dag = -Z); on a light-shift or gradient waveform it is the sigma_z sigma_z spin echo (two half-angle
loops around a pi pulse on both ions), whose sign only the detuning side can choose. Entangling gates are
serialized one at a time per crystal (they share the global beam pair).

Beliefs, not truth: every pulse carries the table's values and is marked ``Drive.programmed`` (the requested Rabi
frequency, the believed Stark shift and crosstalk); ``control.played`` restores what the ions see. The believed Stark shift
is compensated by detuning every spin-flip tone by it (a sigma_z force's tone is the motion's beat note and keeps its
detuning), and the phase 2 pi int delta dt the shifted qubit gains is absorbed into the ion's virtual-Z frame, so the ideal
target of a schedule is its gates followed by RZ(phase_frame) per ion.

The terminal measurement is one ``ScheduledEvent`` after the last pulse and dead time, of the table's detection window;
a measure, reset or recool before a later gate is refused. Every entangling gate played is a ``PlayedGate`` and every gate
piece's ideal unitary a ``GateTarget``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.control import native
from qutip_trap.control.compiler import Circuit, Operation, gate_matrix
from qutip_trap.control.pulses import Drive, DriveKind, LightShiftCouplings, Pulse, Tone
from qutip_trap.control.table import Waveform
from qutip_trap.light.roles import gate_beams
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qutip_trap.control.table import CalibrationTable, Segment
    from qutip_trap.device.model import Device
from dataclasses import field


@dataclass(frozen=True)
class PhaseFrame:
    """The per-qubit virtual-Z frame: an offset theta_i (rad) that every later pulse phase on qubit i subtracts."""

    offsets_rad: dict[int, float] = field(default_factory=dict)

    def offset(self, qubit: int) -> float:
        return float(self.offsets_rad.get(qubit, 0.0))

    def rz(self, qubit: int, theta_rad: float) -> PhaseFrame:
        """RZ(theta) on ``qubit``: later pulses on it carry phi -> phi - theta."""
        new = dict(self.offsets_rad)
        new[qubit] = new.get(qubit, 0.0) + float(theta_rad)
        return PhaseFrame(new)

    def pulse_phase(self, qubit: int, phi_program_rad: float) -> float:
        """The phase the hardware plays for a gate programmed at phi on ``qubit``."""
        return float(phi_program_rad) - self.offset(qubit)

    def as_dict(self, n_qubits: int) -> dict[int, float]:
        return {q: self.offset(q) for q in range(n_qubits)}


MID_CIRCUIT_REFUSAL = (
    "mid-circuit measure, reset and recool are refused: their physics (detection recoil, neighbour Stark shift, "
    "depumping, recooling and re-preparation of the measured ion) is not implemented"
)
FORCE_AXIS_OFFSET_RAD = math.pi / 2.0
"""The bichromatic force acts about phi_s + pi/2 in the same-Delta-k geometry: the i of i eta (a + a^dag)."""

MICROWAVE_BEAM_KEY = -1
"""The beam index that keys a microwave drive's carrier Rabi frequency in ``CalibrationTable.rabi`` (no beams)."""

NATIVE_AREAS: dict[str, float] = {"gpi": float(np.pi), "gpi2": float(np.pi / 2.0)}
"""The rotation area of each native single-qubit pulse (radians)."""


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
    """One entangling gate as the scheduler played it: what the mode selection and the diagnostics read."""

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
    """The ideal PHYSICAL unitary of one played gate piece, native or wrapper: what its pulses are meant to do in the frame
    they were programmed in, the target of a channel extracted from them (GATE_LOCAL tomography).

    ``native`` is (operation name, parameters) with every phase already frame-applied (what the pulses carry);
    ``stark_frame_rad`` is the virtual-Z increment the scheduler absorbed for this gate's compensated light shift, which the
    physical state carries as RZ(-theta). The unitary is F G with F = (x)_q RZ(-theta_q) and G the native matrix, on
    ``ions`` in matrix order (first ion = first factor)."""

    gate_id: str
    ions: tuple[int, ...]
    native: tuple[str, tuple[float, ...]]
    stark_frame_rad: dict[int, float]
    pulse_ids: tuple[str, ...]
    t_start_s: float
    t_end_s: float

    def unitary(self) -> np.ndarray:
        name, params = self.native
        g = gate_matrix(Operation(name, tuple(range(len(self.ions))), tuple(float(p) for p in params)))
        f = np.array([[1.0 + 0.0j]])
        for q in self.ions:
            f = np.kron(f, native.rz(-float(self.stark_frame_rad.get(q, 0.0))))
        return np.asarray(f @ g)


@dataclass(frozen=True)
class Schedule:
    """A circuit as pulses on the time axis: the pulses, the idle intervals (s) during which heating and dephasing act, the
    events (the terminal measurement), the per-qubit virtual-Z frame at the end (radians), the entangling gates as played
    and the ideal target of every played gate piece; ``t0_s`` is the time the incoming state is given at."""

    pulses: tuple[Pulse, ...]
    idle: tuple[tuple[float, float], ...]
    """Idle intervals (start, end) during which heating and dephasing act."""
    events: tuple[ScheduledEvent, ...]
    phase_frame: dict[int, float]
    """Per-qubit virtual-Z frame at the end; the rule is phi -> phi - theta."""
    gates: tuple[PlayedGate, ...] = ()
    """The entangling gates as played, in time order."""
    targets: tuple[GateTarget, ...] = ()
    """The ideal physical unitary of every played gate piece, in time order."""
    t0_s: float | None = None
    """The time the state handed to the engine is given at (a GATE_LOCAL step starts where the register and the motional
    model stand); None = min(0, the first pulse or idle start)."""

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
    """How gates on one ion are driven: the drive kind and the beams that implement it; for a light-shift entangling drive
    the level couplings the beams produce (``light_shift``, from ``derive_light_shift_drive``)."""

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
    equal wavelength = Raman; anything else is ambiguous and must be declared in ``Device.roles``."""
    n = device.crystal.n_ions
    idx = gate_beams(device)  # resonant cooling, detection and repump light plays no gate
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
    """(single-qubit drives, entangling drives): explicit maps over the device's roles, the roles over the inference."""
    resolved = device.roles.resolve(device, gate_drives=gate_drives, entangling_drives=entangling_drives)
    return resolved.gate, resolved.entangling


def carrier_rabi_hz(table: CalibrationTable, ion: int, gate_drive: GateDrive) -> float:
    """The table's carrier Rabi frequency for (ion, drive); refuses an absent or ``uncalibrated`` entry."""
    entry = table.rabi.get((ion, gate_drive.table_key_beam))
    if entry is None:
        raise ScheduleError(
            f"no carrier Rabi frequency in the calibration table for ion {ion} under drive {gate_drive} "
            f"(key {(ion, gate_drive.table_key_beam)}); the scheduler never reads the device's true value"
        )
    if entry.status == "uncalibrated":
        raise ScheduleError(f"the Rabi entry for ion {ion} is uncalibrated and refuses to schedule")
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
    gate_id: str | None = None,
    crosstalk: dict[int, complex] | None = None,
    stark_compensation: bool = True,
    programmed: bool = True,
) -> Pulse:
    """A square carrier pulse of ``area_rad`` (Omega t = area) at ``phase_rad`` in the ion's frame; with
    ``stark_compensation`` the tone is detuned by the believed light shift ``stark_shift_hz`` so that the pulse stays
    resonant with the shifted qubit. ``programmed`` marks the drive as table-driven for ``control.played``."""
    if area_rad <= 0.0:
        raise ValueError("a pulse area is positive; a negative nominal area is played at phase + pi")
    if rabi_hz <= 0.0:
        raise ValueError("rabi_hz must be positive")
    duration = area_rad / (TWO_PI * rabi_hz)
    compensation = float(stark_shift_hz) if stark_compensation else 0.0
    tone = Tone(
        detuning_hz=compensation,
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


def stark_scaling_power(kind: DriveKind) -> int:
    """How the differential Stark shift scales with the played Rabi frequency: a two-photon Rabi frequency and its light
    shift are both proportional to the intensity, so a Raman or light-shift drive scales it LINEARLY (power 1); a
    single-photon optical drive (Omega ~ field, shift ~ intensity) and a microwave drive (ac Zeeman ~ B_1^2) as Omega^2."""
    return 1 if kind in ("raman", "light_shift") else 2


@dataclass(frozen=True, eq=False)
class _BelievedStarkFn:
    """tau -> delta_cal sum_legs (Omega_leg(tau)/Omega_cal)^p (picklable when the amplitudes are)."""

    stark_hz: float
    rabi_hz: float
    power: int
    amplitudes: tuple[float | Callable[[float], float], ...]

    def __call__(self, tau: float) -> float:
        total = 0.0
        for a in self.amplitudes:
            val = float(a(tau)) if callable(a) else float(a)
            total += (val / self.rabi_hz) ** self.power
        return self.stark_hz * total


@dataclass(frozen=True, eq=False)
class _SumFn:
    """tau -> first(tau) + second(tau), each a constant or a callable (picklable when they are)."""

    first: float | Callable[[float], float]
    second: float | Callable[[float], float]

    def __call__(self, tau: float) -> float:
        a = float(self.first(tau)) if callable(self.first) else float(self.first)
        b = float(self.second(tau)) if callable(self.second) else float(self.second)
        return a + b


def _stark_for_segment(
    table: CalibrationTable, ion: int, gate_drive: GateDrive, segment: Segment
) -> float | Callable[[float], float]:
    """The table's Stark entry scaled with the played amplitude, delta_cal sum_legs (Omega_leg/Omega_cal)^p with p the
    drive kind's ``stark_scaling_power``, or 0 when the table has no calibrated Stark or Rabi entry for (ion, beam)."""
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
    return _BelievedStarkFn(float(stark.value), float(rabi.value), power, tuple(amps))


def _compensated_detuning(
    detuning_hz: float | Callable[[float], float], shift_hz: float | Callable[[float], float]
) -> float | Callable[[float], float]:
    """The leg's detuning plus the believed Stark shift, composing callables (an FM leg, a shaped amplitude) when needed."""
    if not callable(detuning_hz) and not callable(shift_hz):
        return float(detuning_hz) + float(shift_hz)
    if not callable(shift_hz) and float(shift_hz) == 0.0:
        return detuning_hz
    return _SumFn(detuning_hz, shift_hz)


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
    """2 pi int delta_St,believed dt over the pulse: the virtual-Z offset the compensated light shift costs the pulse's ion.
    The builder applies (2 pi delta/2) sigma_z with sigma_z = |1><1| - |0><0|, i.e. RZ(-2 pi delta t) in the native
    convention; a physical RZ(-theta) is what a virtual RZ(+theta) leaves behind, so the frame offset is +2 pi delta t and
    every later pulse on the ion carries phi - 2 pi delta t."""
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
    """The tone phase that references a phase-continuous compensation detuning to the qubit frame at the PULSE start: the
    builder plays e^{-i(2 pi (mu + delta_s) t - phi)} in absolute time while the qubit is shifted only from t_s on, so
    phi_prog + 2 pi delta_s(0) t_s puts the axis at phi_prog at t_s, after which the shifted qubit co-rotates with the tone
    and the frame absorbs 2 pi int delta_s dt (``frame_after``)."""
    d0 = float(shift_hz(0.0)) if callable(shift_hz) else float(shift_hz)
    if d0 == 0.0:
        return 0.0
    return (TWO_PI * d0 * float(t_start_s)) % TWO_PI


def beat_phase_offset_rad(detuning_hz: float | Callable[[float], float], t_gate_start_s: float) -> float:
    """The tone phase that resets a continuously running beat note to zero at the GATE start (hardware that programs each
    gate from its own start): 2 pi mu(0) t_g, the phase the builder's absolute-time beat has reached at the gate start for
    a constant detuning (mu t) and a detuning schedule (int mu dtau from mu(0) t_start) alike, so phi_prog + 2 pi mu(0) t_g
    plays every segment of the gate as it plays from t_g = 0; the red and blue legs shift oppositely, so the spin phase
    (their half-sum) is untouched."""
    mu0 = float(detuning_hz(0.0)) if callable(detuning_hz) else float(detuning_hz)
    return (TWO_PI * mu0 * float(t_gate_start_s)) % TWO_PI


def response_phase_rad(detuning_hz: float | Callable[[float], float], delay_s: float) -> float:
    """The per-leg phase that keeps the bichromatic beat note where the calibration put it when the modulator's first-order
    response delays the envelope: arctan(2 pi |mu| tau_r) with the sign of the leg's detuning mu, the phase of the
    switch-on transient's spectral weight H(mu) = 1/(1 - i 2 pi mu tau_r) at the beat frequency (without it Roos's
    spin-axis tilt returns at the gate start). A frequency-modulated leg uses its initial detuning."""
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
    beat_phase_reset: bool = False,
    response_delay_s: float = 0.0,
    stark_compensation: bool = True,
) -> list[Pulse]:
    """One Pulse per (segment, ion) from a calibrated Waveform: tones per leg with the segment's phase offsets plus the ion's
    spin phase (already in its frame), the segment's amplitudes and detunings, the Stark shift scaled with the played
    amplitude.

    ``stark_compensation`` detunes BOTH legs of every segment of an MS waveform by the believed shift for the segment's
    amplitude (which moves the beat-note centre with the shifted qubit and leaves the motion phase alone); the spin phase is
    referenced per segment, ``compensation_phase_rad`` at the segment's own start minus the frame the ion accumulated in the
    gate's earlier segments, so that every segment's force axis is where the calibration put it. A sigma_z force (a
    light-shift or gradient waveform) keeps its calibrated tones: its beat note drives the motion itself, so a detuning
    would open the loops, and the shift its drive carries commutes with the force (the frame absorbs it, ``frame_after``).
    ``beat_phase_reset`` offsets every leg by ``beat_phase_offset_rad`` (hardware that programs each gate's tones from its
    own start); with phase-continuous tones the beat phase at the start is 2 pi mu(0) t_g and Roos's spin-axis tilt is part
    of the gate."""
    pulses: list[Pulse] = []
    in_gate_frame: dict[int, float] = {}
    compensated = stark_compensation and waveform.kind == "ms"
    t = t_start_s
    for k, seg in enumerate(waveform.segments):
        for ion in seg.ions:
            spec = gate_drives[ion]
            if waveform.kind in ("light_shift", "gradient") and spec.kind != waveform.kind:
                raise ScheduleError(
                    f"ion {ion}: a {waveform.kind} waveform needs a {waveform.kind} gate drive"
                )
            if waveform.kind == "ms" and spec.kind not in ("raman", "optical_E1", "optical_E2", "microwave"):
                raise ScheduleError(f"ion {ion}: an MS waveform needs a spin-flip drive, not {spec.kind}")
            believed_shift = _stark_for_segment(table, ion, spec, seg)
            compensation = believed_shift if compensated else 0.0
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
                crosstalk={},
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
            if compensated:
                in_gate_frame[ion] = in_gate_frame.get(ion, 0.0) + stark_phase_rad(pulses[-1])
        t += seg.duration_s
    return pulses


def ms_spin_phases(
    waveform: Waveform, pair: tuple[int, int], phases_rad: tuple[float, float], frame: PhaseFrame
) -> tuple[dict[int, float], float]:
    """(spin phase per ion in its frame, |chi|) for MS(phi_0, phi_1, theta) on ``waveform``: phi_s,i = frame(phi_i) - pi/2
    (the force axis sits at phi_s + pi/2), plus pi on the second ion when the waveform's chi is positive, because the pulse
    applies exp(+i chi sigma sigma) and the native gate is exp(-i (theta/2) sigma sigma)."""
    a, b = pair
    chi = waveform.chi_total_rad
    if chi == 0.0:
        raise ScheduleError("the waveform carries no entangling angle")
    phases = {
        a: frame.pulse_phase(a, phases_rad[0]) - FORCE_AXIS_OFFSET_RAD,
        b: frame.pulse_phase(b, phases_rad[1]) - FORCE_AXIS_OFFSET_RAD + (math.pi if chi > 0.0 else 0.0),
    }
    return phases, abs(chi)


CrosstalkSuppression = Literal["none", "neighbour", "local"]


class _Timeline(ABC):
    """When the scheduler's pulses start (module docstring): the hardware's dead time follows every pulse as an idle
    interval, and the addressing rule decides whether pulses on distinct ions wait for each other. ``schedule`` builds one
    per schedule from ``parallel``."""

    def __init__(self, dead_time_s: float) -> None:
        self.dead_time_s = float(dead_time_s)
        self.idle: list[tuple[float, float]] = []

    def rest(self, end_s: float) -> float:
        """The dead time after a pulse ending at ``end_s``, recorded as an idle interval (none for a zero dead time); returns
        the earliest start of the next pulse."""
        if self.dead_time_s > 0.0:
            self.idle.append((end_s, end_s + self.dead_time_s))
        return end_s + self.dead_time_s

    @abstractmethod
    def start(self, qubits: tuple[int, ...], *, entangling: bool) -> float:
        """The earliest start of a gate on ``qubits``; an entangling gate also waits for the previous one."""

    @abstractmethod
    def advance(self, qubits: tuple[int, ...], end_s: float, *, entangling: bool) -> None:
        """Move the clocks past a gate on ``qubits`` that ended at ``end_s``, leaving the dead time idle."""

    @abstractmethod
    def on_each(self, ions: Sequence[int], start_s: float, play: Callable[[int, float], float]) -> float:
        """``play(ion, t) -> end`` on every one of ``ions`` (at least one) from ``start_s``; returns the last end."""


class _SerialTimeline(_Timeline):
    """One pulse at a time, on a chain without parallel addressing: one clock every gate waits on, and the pulses a gate
    plays on several ions one after another with the dead time between them."""

    def __init__(self, t0_s: float, dead_time_s: float) -> None:
        super().__init__(dead_time_s)
        self.clock_s = float(t0_s)

    def start(self, qubits: tuple[int, ...], *, entangling: bool) -> float:
        return self.clock_s

    def advance(self, qubits: tuple[int, ...], end_s: float, *, entangling: bool) -> None:
        self.clock_s = self.rest(end_s)

    def on_each(self, ions: Sequence[int], start_s: float, play: Callable[[int, float], float]) -> float:
        end = play(ions[0], start_s)
        for q in ions[1:]:
            end = play(q, self.rest(end))
        return end


class _ParallelTimeline(_Timeline):
    """A clock per ion, on a chain with parallel addressing: single-qubit pulses on distinct ions overlap and the pulses a gate
    plays on several ions share one window; an entangling gate still waits for the previous one on any pair (the entangling
    gates share the global beam pair)."""

    def __init__(self, qubits: Sequence[int], t0_s: float, dead_time_s: float) -> None:
        super().__init__(dead_time_s)
        self.clock_s = {q: float(t0_s) for q in qubits}
        self.entangling_clock_s = float(t0_s)

    def start(self, qubits: tuple[int, ...], *, entangling: bool) -> float:
        t = max(self.clock_s[q] for q in qubits)
        return max(t, self.entangling_clock_s) if entangling else t

    def advance(self, qubits: tuple[int, ...], end_s: float, *, entangling: bool) -> None:
        t = self.rest(end_s)
        for q in qubits:
            self.clock_s[q] = t
        if entangling:
            self.entangling_clock_s = t

    def on_each(self, ions: Sequence[int], start_s: float, play: Callable[[int, float], float]) -> float:
        return max(play(q, start_s) for q in ions)


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
    stark_compensation: bool = True,
) -> Schedule:
    """Native gates -> pulses with absolute times from the calibration table (module docstring).

    ``gate_drives``/``entangling_drives`` override the device's roles. ``stark_compensation``: every spin-flip tone (the
    carrier pulses' and an MS waveform's) is detuned by the believed Stark shift for the played amplitude, and the frame
    absorbs the shift of every pulse. ``crosstalk_suppression`` (Fang et al. 2022): every MS gate is split
    into two half-angle plays around a physical echo, exact to first order in the leaked drives; ``local`` plays Y(pi) (a
    GPi(pi/2)) on both targets between the halves and again after the second (Y X Y = -X flips the leaked terms while
    Y (x) Y commutes with XX), ``neighbour`` a physical Z(pi) = GPi(0) GPi(pi/2) on every crosstalk spectator of the pair
    between the halves, absorbed afterwards into its frame. ``parallel`` defaults to the chain's ``parallel_addressing``
    and is refused on a chain without it."""
    if not circuit.is_native:
        raise ScheduleError("schedule() takes a native circuit; compile_to_native first")
    allows_parallel = bool(device.hardware.parallel_addressing)
    if parallel is None:
        parallel = allows_parallel
    elif parallel and not allows_parallel:
        raise ScheduleError(
            "parallel=True needs HardwareChain.parallel_addressing = True: single-qubit gates run in parallel only if "
            "the device model allows parallel addressing"
        )
    if crosstalk_suppression != "none" and parallel:
        raise ScheduleError("crosstalk suppression is scheduled on the serial path")
    drives, ent_drives = resolve_drives(device, gate_drives, entangling_drives)
    dead = float(device.hardware.dead_time_s)
    timeline: _Timeline = (
        _ParallelTimeline(range(circuit.n_qubits), t0_s, dead) if parallel else _SerialTimeline(t0_s, dead)
    )
    reset = not bool(device.hardware.phase_continuous)
    delay = float(device.hardware.aom_rise_s)
    frame = PhaseFrame()
    pulses: list[Pulse] = []
    gates: list[PlayedGate] = []
    targets: list[GateTarget] = []

    def entangle(
        pair: tuple[int, int],
        wf: Waveform,
        spins: dict[int, float],
        start: float,
        gate_id: str,
        kind: Literal["ms", "zz"],
        native_op: tuple[str, tuple[float, ...]],
    ) -> float:
        """Play ``wf`` on ``pair`` from ``start`` and record its target and its PlayedGate; returns its end."""
        nonlocal frame
        before = frame
        new = entangling_pulses(
            wf,
            ent_drives,
            spin_phases_rad=spins,
            t_start_s=start,
            table=table,
            gate_id=gate_id,
            beat_phase_reset=reset,
            response_delay_s=delay,
            stark_compensation=stark_compensation,
        )
        pulses.extend(new)
        frame = frame_after(new, frame, stark_compensation=stark_compensation)
        targets.append(
            GateTarget(
                gate_id,
                pair,
                native_op,
                {q: frame.offset(q) - before.offset(q) for q in pair},
                tuple(p.gate_id or "" for p in new),
                min(p.t_start_s for p in new),
                max(p.t_end_s for p in new),
            )
        )
        gates.append(
            PlayedGate(gate_id, kind, pair, wf, ent_drives[pair[0]].beams, start, start + wf.duration_s)
        )
        return start + wf.duration_s

    def carrier(q: int, area: float, phase: float, start: float, gate_id: str) -> Pulse:
        """A GPi (area pi) or GPi2 (area pi/2) at ``phase`` in the ion's frame from the table's Rabi frequency, Stark shift
        and crosstalk, with its target and its absorbed Stark phase."""
        nonlocal frame
        spec = drives[q]
        stark_entry = table.stark.get((q, spec.table_key_beam))
        ph = frame.pulse_phase(q, phase)
        p = single_qubit_pulse(
            q,
            area,
            ph,
            spec,
            carrier_rabi_hz(table, q, spec),
            start,
            stark_shift_hz=0.0
            if stark_entry is None or stark_entry.status == "uncalibrated"
            else stark_entry.value,
            gate_id=gate_id,
            crosstalk=crosstalk_beliefs(table, q),
            stark_compensation=stark_compensation,
        )
        pulses.append(p)
        targets.append(
            GateTarget(
                gate_id,
                (q,),
                ("gpi" if area == NATIVE_AREAS["gpi"] else "gpi2", (float(ph),)),
                {q: stark_phase_rad(p) if stark_compensation else 0.0},
                (gate_id,),
                p.t_start_s,
                p.t_end_s,
            )
        )
        frame = frame_after([p], frame, stark_compensation=stark_compensation)
        return p

    def carriers(
        ions: Sequence[int], sequence: Sequence[tuple[float, float]], start: float, tag: str
    ) -> float:
        """The (area, phase) carrier pulses of ``sequence`` on every one of ``ions`` from ``start`` under the addressing
        rule, one after another on each ion with the dead time between them; the last end."""

        def train(q: int, t: float) -> float:
            (area, phase), *later = sequence
            end = carrier(q, area, phase, t, f"{tag}/ion{q}").t_end_s
            for area, phase in later:
                end = carrier(q, area, phase, timeline.rest(end), f"{tag}/ion{q}").t_end_s
            return end

        return timeline.on_each(ions, start, train)

    last_unitary = max([k for k, op in enumerate(circuit.ops) if not op.is_non_unitary], default=-1)
    measured: list[int] = list(circuit.measure)
    for k, op in enumerate(circuit.ops):
        if op.is_non_unitary and (op.name != "measure" or k < last_unitary):
            raise ScheduleError(f"{op.name!r} at position {k}: {MID_CIRCUIT_REFUSAL}")
        if op.name == "measure":
            measured.extend(q for q in op.qubits if q not in measured)

    for k, op in enumerate(circuit.ops):
        if op.name == "rz":
            frame = frame.rz(op.qubits[0], op.params[0])
            continue
        if op.is_non_unitary:
            continue  # a trailing measure: recorded above, scheduled as the terminal event below
        if op.name in ("gpi", "gpi2"):
            q = op.qubits[0]
            start = timeline.start((q,), entangling=False)
            p = carrier(q, NATIVE_AREAS[op.name], op.params[0], start, f"{op.name}[{k}]")
            timeline.advance((q,), p.t_end_s, entangling=False)
            continue
        a, b = op.qubits
        wf = table.waveform_for((a, b))
        if wf is None:
            raise ScheduleError(
                f"no entangling waveform in the calibration table for ions {(a, b)} (CalibrationTable.ms)"
            )
        start = timeline.start((a, b), entangling=True)
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
            phases = (frame.pulse_phase(a, phi0), frame.pulse_phase(b, phi1))
            if crosstalk_suppression == "none":
                play = _rescaled(wf, 0.5 * theta, chi_abs)
                end = entangle((a, b), play, spins, start, f"ms[{k}]", "ms", ("ms", (*phases, theta)))
                timeline.advance((a, b), end, entangling=True)
                continue
            half = _rescaled(wf, 0.25 * theta, chi_abs)
            half_native = ("ms", (*phases, 0.5 * theta))
            if crosstalk_suppression == "local":
                echo_ions = [a, b]
                sequence = [(NATIVE_AREAS["gpi"], 0.5 * math.pi)]
            else:
                echo_ions = sorted(
                    {
                        j
                        for (i, j), e in table.crosstalk.items()
                        if i in (a, b) and j not in (a, b) and e.status != "uncalibrated" and e.value != 0.0
                    }
                )
                sequence = [(NATIVE_AREAS["gpi"], 0.0), (NATIVE_AREAS["gpi"], 0.5 * math.pi)]
            t = timeline.rest(entangle((a, b), half, spins, start, f"ms[{k}]/half1", "ms", half_native))
            if echo_ions:
                t = timeline.rest(carriers(echo_ions, sequence, t, f"ms[{k}]/echo"))
            t = entangle((a, b), half, spins, t, f"ms[{k}]/half2", "ms", half_native)
            if crosstalk_suppression == "local":
                t = carriers(echo_ions, sequence, timeline.rest(t), f"ms[{k}]/unecho")
            else:
                # a physical Z(pi) is a frame operation on every later pulse of the spectator: absorb it
                for j in echo_ions:
                    frame = frame.rz(j, math.pi)
            timeline.advance((a, b), t, entangling=True)
            continue
        (theta,) = op.params
        if wf.kind == "ms":
            # GPi2(3 pi/2) on both, MS(0, 0, theta), GPi2(pi/2) on both; the MS waits for both wrapper pulses
            wrap_in = [(NATIVE_AREAS["gpi2"], 1.5 * math.pi)]
            t = timeline.rest(carriers((a, b), wrap_in, start, f"zz[{k}]/wrap_in"))
            spins, chi_abs = ms_spin_phases(wf, (a, b), (0.0, 0.0), frame)
            play = _rescaled(wf, 0.5 * abs(theta), chi_abs)
            if theta < 0.0:
                spins[b] += math.pi
            ms_native = ("ms", (frame.pulse_phase(a, 0.0), frame.pulse_phase(b, 0.0), theta))
            t = timeline.rest(entangle((a, b), play, spins, t, f"zz[{k}]/ms", "zz", ms_native))
            end = carriers((a, b), [(NATIVE_AREAS["gpi2"], 0.5 * math.pi)], t, f"zz[{k}]/wrap_out")
            timeline.advance((a, b), end, entangling=True)
            continue
        # the sigma_z sigma_z spin echo of a light-shift or microwave-gradient waveform:
        # two half-angle loops, each exp(+i chi_half sigma_z sigma_z) = zz(theta/2), around a pi on both ions
        chi = wf.chi_total_rad
        if chi * theta > 0.0:
            raise ScheduleError(
                f"the light-shift waveform applies exp(+i {chi:.4g} sigma_z sigma_z) per pulse; ZZ({theta:.4g}) needs the "
                "opposite sign, which a sigma_z force can only take from the other detuning side"
            )
        half = _rescaled(wf, 0.25 * abs(theta), abs(chi))
        spins = {a: 0.0, b: 0.0}
        loop_native: tuple[str, tuple[float, ...]] = ("zz", (0.5 * theta,))
        t = timeline.rest(entangle((a, b), half, spins, start, f"zz[{k}]/loop1", "zz", loop_native))
        t = timeline.rest(carriers((a, b), [(NATIVE_AREAS["gpi"], 0.0)], t, f"zz[{k}]/echo"))
        t = timeline.rest(entangle((a, b), half, spins, t, f"zz[{k}]/loop2", "zz", loop_native))
        end = carriers((a, b), [(NATIVE_AREAS["gpi"], math.pi)], t, f"zz[{k}]/unecho")
        timeline.advance((a, b), end, entangling=True)
    events: list[ScheduledEvent] = []
    if measured:
        window_entry = table.detection.get("window_s")
        window = (
            float(window_entry.value)
            if window_entry is not None and window_entry.status != "uncalibrated" and window_entry.value > 0.0
            else float(device.detector.window_s)
        )
        ends = [p.t_end_s for p in pulses] + [b for _, b in timeline.idle]
        t_meas = max(ends) if ends else t0_s
        events.append(ScheduledEvent("measure", tuple(sorted(measured)), t_meas, t_meas + window))
    return Schedule(
        tuple(pulses),
        tuple(timeline.idle),
        tuple(events),
        frame.as_dict(circuit.n_qubits),
        gates=tuple(gates),
        targets=tuple(targets),
    )


def _rescaled(waveform: Waveform, chi_target_abs: float, chi_abs: float) -> Waveform:
    """The waveform scaled to |chi| = chi_target_abs by the s^2 law; the identity at the calibrated angle."""
    from qutip_trap.control.shaping import scaled

    if chi_target_abs <= 0.0:
        raise ScheduleError("an entangling gate needs a positive angle; a zero-angle gate is no pulse")
    factor = math.sqrt(chi_target_abs / chi_abs)
    return waveform if abs(factor - 1.0) < 1e-12 else scaled(waveform, factor)
