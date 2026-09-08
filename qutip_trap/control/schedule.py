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
Section 4.4.4), whose sign can only be chosen through the detuning side; a ``gradient`` waveform (the two microwave tones
at -/+ delta of a near-field magnetic-gradient drive, Section 4.4.5) plays through that same sigma_z echo path.

Measurement (Section 7.2 item 4, M6): the terminal ``measure`` (the circuit's ``measure`` targets and any trailing measure
operations) becomes one ``ScheduledEvent`` after the last pulse and dead time, of the calibration table's detection window
(the detector's when the table has none); a measure, reset or recool that precedes a later gate is refused with a
``ScheduleError`` until the mid-circuit physics of Section 8.5 (recoil, neighbour Stark shift, recooling and
re-preparation of the measured ion) is implemented, which is a stage and not a pipeline change. Every entangling gate the
scheduler plays is recorded as a ``PlayedGate`` (the pair, the beams and the waveform AS PLAYED, rescaled), which is what the
resolved-mode selection of Section 5.2 reads (``qutip_trap.run.space``).

Beliefs and truth (Section 7.3; M8): every pulse the scheduler emits carries the TABLE's values, marked ``Drive.programmed``:
the requested Rabi frequency (the table's carrier entry, or the waveform's amplitude in the table's units), the believed
differential Stark shift, the believed crosstalk ratios and phases. The believed Stark shift is COMPENSATED the way a
laboratory does it, by detuning every tone of the pulse by the shift the table predicts for the played amplitude
(``stark_compensation``), so that a correct belief keeps the pulse resonant with the shifted qubit and a wrong one leaves a
reproducible detuning error; the z-rotation 2 pi int delta dt the shifted qubit accumulates relative to the frame during the
pulse is absorbed into the ion's virtual-Z frame afterwards (``frame_after``: the differential light shift 'sets the MS phase',
Section 7.5 item 7), so the ideal target of a schedule is its ideal gates followed by RZ(phase_frame) per ion. What the ions actually see (the physical Rabi frequency through the rf-power-to-Omega map the Rabi
scan calibrated, the physical light shift, the physical crosstalk) is restored by ``control.played.physical_schedule`` inside
the engine from the device's derived values, never by the scheduler.

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
from qutip_trap.light.roles import gate_beams
from qutip_trap.transport.budget import Transport
from qutip_trap.units import TWO_PI

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qutip_trap.control.table import CalibrationTable, Segment
    from qutip_trap.device.model import Device

M6_MID_CIRCUIT = (
    "mid-circuit measure, reset and recool are refused in the first release (PLAN.md Section 7.2 item 4): the IR and the "
    "Schedule carry their positions, the physics of Section 8.5 (detection recoil, neighbour Stark shift, depumping, recooling "
    "and re-preparation of the measured ion) is the stage still to be implemented"
)
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
class PlayedGate:
    """One entangling gate as the scheduler played it: what the mode selection of Section 5.2 and the diagnostics read."""

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
    """The ideal PHYSICAL unitary of one played gate, native or wrapper piece (M9a, Section 5.4): what its pulses are meant to do
    in the frame they were programmed in, so that a channel extracted from them (GATE_LOCAL tomography, Section 6.8) has a target.

    ``native`` is (operation name, parameters) with every phase already FRAME-APPLIED (phi - theta_frame, what the pulses carry);
    ``stark_frame_rad`` is the virtual-Z increment the scheduler absorbed for this gate's compensated light shift (Section 7.5
    item 7), a rotation the physical state carries as RZ(-theta) (Section 7.6, ``frame_rotated``). The unitary is therefore
    F(stark) G with F = (x)_q RZ(-theta_q) and G the native matrix, on ``ions`` in matrix order (first ion = first factor)."""

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
    pulses: tuple[Pulse, ...]
    idle: tuple[tuple[float, float], ...]
    """Idle intervals (start, end) during which heating and dephasing act (Section 3.4)."""
    events: tuple[ScheduledEvent, ...]
    phase_frame: dict[int, float]
    """Per-qubit virtual-Z frame at the end; the rule is phi -> phi - theta (Section 7.6)."""
    transports: tuple[Transport, ...] = ()
    """M12: interleaved with ``pulses`` by absolute time."""
    gates: tuple[PlayedGate, ...] = ()
    """The entangling gates as played (M6), in time order."""
    targets: tuple[GateTarget, ...] = ()
    """The ideal physical unitary of every played gate piece (M9a), in time order."""
    t0_s: float | None = None
    """The time the state handed to the engine is given at (M9a: a GATE_LOCAL step starts where the register and the motional
    model stand); None = min(0, the first pulse or idle start), the whole-schedule convention of M2 to M8."""

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
    n = device.crystal.n_ions
    idx = gate_beams(
        device
    )  # the far-detuned beams; resonant cooling, detection and repump light plays no gate
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
            "pass gate_drives"
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
    stark_compensation: bool = True,
    programmed: bool = True,
) -> Pulse:
    """A square carrier pulse of ``area_rad`` (Omega t = area in the plan's convention) at ``phase_rad`` in the ion's frame.

    ``stark_shift_hz`` is the differential light shift the table predicts for this pulse; with ``stark_compensation`` the tone is
    detuned by it so that the pulse stays resonant with the shifted qubit (Section 7.5 item 7). ``programmed`` marks the drive
    as table-driven for the played chain of ``control.played`` (M8)."""
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
    """How the differential Stark shift scales with the played Rabi frequency (Section 4.3.2): both a two-photon Rabi frequency
    and the ac Stark shift are proportional to the intensity, so a Raman or light-shift drive scales its shift LINEARLY with
    Omega (power 1); a single-photon optical drive (Omega proportional to the field, the shift to the intensity) and a microwave
    drive (the ac Zeeman shift proportional to B_1^2) scale it as Omega^2 (power 2)."""
    return 1 if kind in ("raman", "light_shift") else 2


def _stark_for_segment(
    table: CalibrationTable, ion: int, gate_drive: GateDrive, segment: Segment
) -> float | Callable[[float], float]:
    """The table's Stark entry scaled with the played amplitude: delta_cal sum_legs (Omega_leg/Omega_cal)^p with p = 1 for a
    two-photon drive and 2 for a single-photon or microwave one (``stark_scaling_power``, Section 4.3.2), or 0 when the table
    has no calibrated Stark or Rabi entry for (ion, beam)."""
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
    ``crosstalk_phase`` (0 when absent); uncalibrated or zero entries are skipped (Section 7.5 item 8)."""
    xt: dict[int, complex] = {}
    for (i, j), entry in table.crosstalk.items():
        if i == ion and entry.status != "uncalibrated" and entry.value != 0.0:
            phase = table.crosstalk_phase.get((i, j))
            arg = float(phase.value) if phase is not None and phase.status != "uncalibrated" else 0.0
            xt[j] = complex(entry.value) * complex(math.cos(arg), math.sin(arg))
    return xt


def stark_phase_rad(pulse: Pulse) -> float:
    """2 pi int delta_St,believed dt over the pulse: the virtual-Z offset the compensated light shift costs the pulse's ion. The
    builder applies (2 pi delta/2) sigma_z with sigma_z = |1><1| - |0><0| (the upper level up by delta/2), i.e. the rotation
    e^{-i (2 pi delta t/2) sigma_z} = RZ(-2 pi delta t) in the native convention RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2}); a
    physical RZ(-theta) on the state is what a virtual RZ(+theta) leaves behind (Section 7.6), so the frame offset is +2 pi delta t
    and every later pulse on the ion carries phi - 2 pi delta t (Section 7.5 item 7: 'the differential light shift that sets the
    MS phase')."""
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
    """The tone phase that references a phase-continuous compensation detuning to the qubit frame at the PULSE start (M8): the
    builder plays e^{-i(2 pi (mu + delta_s) t - phi)} in absolute time, while the qubit is light-shifted only from t_s on, so a
    compensated tone leads the qubit's frame by 2 pi delta_s t_s when the pulse starts; phi_prog + 2 pi delta_s(0) t_s makes the
    axis phi_prog at t_s, after which the shifted qubit co-rotates with the tone and the frame absorbs 2 pi int delta_s dt
    (``frame_after``). A laboratory's control software does the same bookkeeping for a compensation detuning it programs into a
    phase-continuous synthesizer; the tone's gate detuning mu is left as ``beat_phase_offset_rad`` and the hardware's
    ``phase_continuous`` flag decide (Section 7.10). Verified on the two-ion fixture: an uncorrected compensated GPi2 one
    millisecond into a schedule lost 1.5e-2 in fidelity, (2 pi delta_s t_s)^2/4 for delta_s = -38 Hz."""
    d0 = float(shift_hz(0.0)) if callable(shift_hz) else float(shift_hz)
    if d0 == 0.0:
        return 0.0
    return (TWO_PI * d0 * float(t_start_s)) % TWO_PI


def beat_phase_offset_rad(detuning_hz: float | Callable[[float], float], t_gate_start_s: float) -> float:
    """The tone phase that resets a continuously running beat note to zero at the GATE start (Section 7.10, ``phase="reset"``
    per gate): the builder plays e^{-i(2 pi mu t - phi)}, so phi_prog + 2 pi mu t_g makes it e^{-i(2 pi mu (t - t_g) - phi_prog)};
    the red and blue legs shift oppositely, so the spin phase (their half-sum) is untouched and the virtual-Z frame holds. A
    frequency-modulated leg (a callable detuning) is integrated from the pulse start by the builder already and needs none."""
    if callable(detuning_hz):
        return 0.0
    return (TWO_PI * float(detuning_hz) * float(t_gate_start_s)) % TWO_PI


def response_phase_rad(detuning_hz: float | Callable[[float], float], delay_s: float) -> float:
    """The per-leg phase that keeps the bichromatic beat note where the calibration put it when the modulator's first-order
    response delays the envelope (Section 7.10; M7): arctan(2 pi |mu| tau_r) with the sign of the leg's detuning mu.

    The field the ion sees is the programmed envelope through the response h(t) = e^{-t/tau_r}/tau_r; the switch-on
    transient's spectral weight at the beat frequency is H(mu) = 1/(1 - i 2 pi mu tau_r), whose phase arctan(2 pi mu tau_r)
    is the beat phase the envelope's arrival lags by (the linear mu tau_r for mu tau_r << 1). Without it Roos's spin-axis
    tilt returns at the gate start even after the per-gate reset (M7 finding on the two-ion fixture: mu tau_r = 0.96 rad,
    leakage 2.9e-3 uncompensated, 2.2e-4 with the linear phase, 3.3e-5 with the arctan against 4.5e-5 for an ideal
    modulator). A laboratory calibrates its tone phases against the delivered field; this is that calibration for a
    modelled chain, and the residual is what the exact check reports. A frequency-modulated leg uses its initial detuning.
    """
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
    """One Pulse per (segment, ion) from a calibrated Waveform: tones per leg with the segment's phase offsets plus the ion's spin
    phase (already in its frame), the segment's amplitudes and detunings, the Stark shift scaled with the played amplitude.

    ``stark_compensation`` detunes BOTH legs of every segment by the table's believed shift for the segment's amplitude (the
    same shift on both legs moves the beat-note centre with the shifted qubit and leaves the motion phase alone), which is how
    a laboratory keeps the bichromatic drive centred on the light-shifted transition (Section 7.5 item 7). The spin phase is
    referenced per segment: ``compensation_phase_rad`` at the segment's own start (phase-continuous tones), minus the frame
    2 pi int delta_s dt the ion accumulated in the gate's earlier segments (the virtual-Z rule inside the gate; an amplitude-
    modulated waveform shifts each segment differently), so that every segment's force axis is where the calibration put it.

    ``beat_phase_reset`` (hardware that programs each gate's tones from its own start, ``HardwareChain.phase_continuous =
    False``) offsets every leg's phase by ``beat_phase_offset_rad`` so that the bichromatic beat note starts at zero phase at the
    gate start, as it did when the waveform was calibrated at t = 0; with phase-continuous tones the beat phase at the start
    is 2 pi mu t_g and Roos's spin-axis tilt psi = (4 Omega/mu) sin(2 pi mu t_g) is part of the gate (M6 finding: 1.1 %
    leakage at the worst phase on the two-ion fixture, reported in the intrinsic budget).
    """
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
    """Native gates -> pulses with absolute times from the calibration table (Section 7.3).

    ``stark_compensation`` (M8, Section 7.5 item 7): every pulse's tones are detuned by the differential Stark shift the table
    believes for the played amplitude, the laboratory's compensation of the light shift; the physical shift the ions see is
    restored by the engine's played chain (``control.played``) from the device.

    ``crosstalk_suppression`` (Section 6.6, Fang et al. 2022; M7): every MS gate is split into two half-angle plays with a
    physical echo between them, exact to first order in the leaked drives. ``local``: Y(pi) (a GPi(pi/2) pulse) on both
    targets between the halves and again after the second, since Y X Y = -X flips the leaked X^(1) sigma^(j) terms while
    Y (x) Y commutes with XX; ``neighbour``: a physical Z(pi) = GPi(0) then GPi(pi/2) on every crosstalk spectator of the pair
    between the halves (Z sigma_phi Z = -sigma_phi), absorbed afterwards into the spectator's virtual frame (rz(pi)).

    gpi and gpi2 become carrier pulses of area pi and pi/2 at the frame-shifted phase; rz is a frame update (no pulse,
    no time beyond the dead time the hardware inserts between pulses, Section 7.6); ms and zz play the pair's calibrated
    Waveform (M4, see the module docstring; ``entangling_drives`` default to ``gate_drives``, the same beam pairs); the
    terminal ``measure`` (the circuit's targets plus trailing measure operations) is the schedule's one event, of the table's
    detection window; a measure, reset or recool before a later gate is refused (Section 7.2 item 4).

    ``parallel`` (Section 7.3: single-qubit gates run "in parallel if the device model allows parallel addressing") defaults
    to the device's ``HardwareChain.parallel_addressing``; ``parallel=True`` on a chain that declares no parallel addressing
    is refused. Entangling gates are serialized one at a time per crystal either way (they share the global beam pair).
    """
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
    drives = gate_drives or default_gate_drives(device)
    ent_drives = entangling_drives or drives
    dead = float(device.hardware.dead_time_s)
    reset = not bool(device.hardware.phase_continuous)
    delay = float(device.hardware.aom_rise_s) if response_delay else 0.0
    frame = PhaseFrame()
    pulses: list[Pulse] = []
    idle: list[tuple[float, float]] = []
    gates: list[PlayedGate] = []
    targets: list[GateTarget] = []

    def record_single(p: Pulse, name: str, phase_played: float) -> None:
        """The ideal physical unitary of one single-qubit pulse: the native gate at the frame-applied phase it was programmed
        with, followed by the Stark rotation the frame absorbs for it (M9a)."""
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
        """Move the clocks past a gate that ended at ``end``, leaving the hardware's dead time idle.

        ``global_clock`` is the crystal-wide clock every gate waits on when ``parallel`` is false. Under ``parallel`` the
        per-ion clocks carry single-qubit gates, and an entangling gate (``serialized=True``) advances BOTH its ions' clocks
        and ``global_clock``: Section 7.3 serializes MS gates one at a time per crystal in the first release, so the next
        entangling gate waits on the previous one even on a disjoint pair (both would draw on the same global beam pair)."""
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
            # Section 7.3: MS gates are serialized one at a time per crystal, so an entangling gate waits on global_clock even
            # under parallel addressing (which only parallelizes the single-qubit beams)
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
                # Section 6.6 echo schemes: two half-angle plays around a physical echo (M7)
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
                    # a physical Z(pi) is a frame operation on every later pulse of the spectator: absorb it (Section 7.6)
                    for j in echo_ions:
                        frame = frame.rz(j, math.pi)
                advance((a, b), t, serialized=True)
                continue
            (theta,) = op.params
            if wf.kind == "ms":
                # inferred construction (Sections 7.6, 12): GPi2(3 pi/2) both, MS(0, 0, theta), GPi2(pi/2) both
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
                # the wrapper pulses on both ions play together either way (distinct ions, distinct beams); the MS that
                # follows them waits for both
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
            # sigma_z sigma_z gate in the spin-echo form of Section 4.4.4 (a light-shift waveform) or 4.4.5 (a
            # near-field microwave-gradient waveform, whose dressed force is the same sigma_z force with the
            # J_2(4 Omega_mu/delta) weight): two half-angle loops around a pi on both ions
            chi = wf.chi_total_rad
            if chi * theta > 0.0:
                raise ScheduleError(
                    f"the light-shift waveform applies exp(+i {chi:.4g} sigma_z sigma_z) per pulse; ZZ({theta:.4g}) needs the "
                    "opposite sign, which a sigma_z force can only take from the other detuning side (Section 4.4.4)"
                )
            half = _rescaled(wf, 0.25 * abs(theta), abs(chi))
            spins = {a: 0.0, b: 0.0}
            # each loop applies exp(+i chi_half sigma_z sigma_z) with chi_half = -sign(theta) |theta|/4 = zz(theta/2) (Section 4.4.4)
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
    """The waveform scaled to |chi| = chi_target_abs by the s^2 law (Section 4.4.7 (7)); the identity at the calibrated angle."""
    from qutip_trap.control.shaping import scaled

    if chi_target_abs <= 0.0:
        raise ScheduleError("an entangling gate needs a positive angle; a zero-angle gate is no pulse")
    factor = math.sqrt(chi_target_abs / chi_abs)
    return waveform if abs(factor - 1.0) < 1e-12 else scaled(waveform, factor)


__all__ = [
    "CrosstalkSuppression",
    "FORCE_AXIS_OFFSET_RAD",
    "M6_MID_CIRCUIT",
    "MICROWAVE_BEAM_KEY",
    "NATIVE_AREAS",
    "GateDrive",
    "GateTarget",
    "PlayedGate",
    "Schedule",
    "ScheduleError",
    "ScheduledEvent",
    "beat_phase_offset_rad",
    "compensation_phase_rad",
    "carrier_rabi_hz",
    "crosstalk_beliefs",
    "default_gate_drives",
    "entangling_pulses",
    "frame_after",
    "ms_spin_phases",
    "response_phase_rad",
    "schedule",
    "single_qubit_pulse",
    "stark_phase_rad",
    "stark_scaling_power",
]
