"""What the ions see: the requested drives rescaled to the true device (``physical_schedule``), then the control hardware
chain of DDS, AOM and amplifier (PLAN.md Section 7.10).

The pulse a ``Schedule`` specifies is not the field the ion sees. ``apply_hardware_chain`` turns a schedule into the one
the hardware plays, every effect entering through the drive coefficient Omega(t) e^{-i(mu t - phi(t))} so that no new
Hamiltonian term is needed:

- Quantization: a constant tone detuning is rounded to the DDS frequency grid f_clk/2^bits (when ``dds_clock_hz`` and
  ``dds_frequency_bits`` are set), a constant phase to 2 pi/2^phase_bits, an amplitude to full_scale/2^amplitude_bits
  with full scale ``amplitude_full_scale_hz`` (or the pulse train's own peak when None); the rounded values are what the
  simulator plays, so the residual detuning is physics, not tolerance.
- Amplifier saturation: Omega = Omega_sat tanh(Omega_prog/Omega_sat) when ``amplifier_saturation_hz`` is set.
- Modulator response: the programmed envelope of every tone is low-pass filtered by a first-order response of time constant
  ``aom_rise_s`` (the beam diameter over the acoustic velocity) for laser light and 1/(2 pi B) of ``amplifier_bandwidth_hz``
  for a microwave or gradient drive (no modulator in that path; an infinite bandwidth is an ideal envelope), continuously
  across the contiguous segments of one pulse train (an entangling gate's segments, a composite pulse) and with an
  exponential tail of 8 time constants after the train (e^-8 dropped), truncated at the next pulse on a shared ion when the
  dead time is shorter; a filtered envelope is a uniformly sampled array the builder splines, an unfiltered one keeps its
  form (a callable is never resampled). The drive's Stark shift follows the played light: delta(t) = delta_ref mean_tones
  (|Omega_played(t)|/Omega_ref)^p with p the ``stark_scaling_power`` of the drive kind and the reference read at the
  pulse's programmed peak, so the tail carries the decaying shift of the decaying light; a programmed shift is read as
  homogeneous of degree p in the tone amplitudes, exact when the tones of a drive share one envelope shape.
- Timing jitter: each pulse train is shifted rigidly by a normal deviate of ``timing_jitter_s`` drawn from the keyed seeds
  (a train shares one trigger), clamped so that no two pulses on one ion overlap.
- Phase coherence and dead time are the scheduler's (``phase_continuous`` selects the per-gate beat-note reset; the dead
  time is idle evolution with the noise channels active).

A chain with zero rise time, infinite amplifier bandwidth, zero jitter, no frequency word and no saturation is ideal
electronics, which ``describe`` says.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.control.pulses import ConstantFn, as_time_function

if TYPE_CHECKING:
    from qutip_trap.control.pulses import Pulse
    from qutip_trap.control.schedule import Schedule
from dataclasses import replace
from typing import TYPE_CHECKING

from qutip_trap.control.pulses import Drive, Pulse, ScaledFn
from qutip_trap.control.schedule import MICROWAVE_BEAM_KEY, Schedule, stark_scaling_power
from qutip_trap.control.table import CalibrationTable, usable

if TYPE_CHECKING:
    from qutip_trap.device.model import Device


Envelope = Callable[[float], float] | np.ndarray | float


UNLISTED_CROSSTALK_REPORT = 1e-3


"""A derived crosstalk ratio above this on a neighbour the table does not list is reported in the chain's notes."""


def _scale_envelope(env: Envelope, factor: float) -> Envelope:
    if factor == 1.0:
        return env
    if callable(env):
        return ScaledFn(env, factor)
    if isinstance(env, np.ndarray):
        return np.asarray(env, dtype=float) * factor
    return float(env) * factor


@dataclass(frozen=True, eq=False)
class _PhysicalStarkFn:
    """tau -> delta_derived sum_tones (|Omega_tone(tau)|/Omega_derived)^p, an array envelope read at its nearest sample
    (picklable when the envelopes are)."""

    stark_hz: float
    omega_hz: float
    power: int
    envelopes: tuple[Envelope, ...]
    duration_s: float

    def __call__(self, tau: float) -> float:
        total = 0.0
        for e in self.envelopes:
            if callable(e):
                val = abs(float(e(tau)))
            elif isinstance(e, np.ndarray):
                k = min(int(round(tau / max(self.duration_s, 1e-300) * (len(e) - 1))), len(e) - 1)
                val = abs(float(e[max(k, 0)]))
            else:
                val = abs(float(e))
            total += (val / self.omega_hz) ** self.power
        return self.stark_hz * total


class _Truth:
    """Derived values per (ion, beams, kind), computed once per schedule."""

    def __init__(self, device: Device) -> None:
        self.device = device
        self._rabi: dict[tuple[int, tuple[int, ...], str], tuple[float, float]] = {}
        self._crosstalk: dict[tuple[int, tuple[int, ...], str], dict[int, complex] | None] = {}

    def rabi_and_stark(self, ion: int, beams: tuple[int, ...], kind: str) -> tuple[float, float]:
        """(carrier Rabi frequency, differential Stark shift) in Hz the device derives for the drive."""
        key = (ion, beams, kind)
        if key not in self._rabi:
            from qutip_trap.light.raman import derive_optical_drive, derive_raman_drive

            if kind == "raman":
                dd = derive_raman_drive(self.device, ion, (beams[0], beams[1]), scattering=False)
            else:
                dd = derive_optical_drive(self.device, ion, beams[0], scattering=False)
            self._rabi[key] = (float(dd.carrier_rabi_hz), float(dd.stark_shift_hz))
        return self._rabi[key]

    def crosstalk(self, ion: int, beams: tuple[int, ...], kind: str) -> dict[int, complex] | None:
        """The derived ratios, or None when the device derives no coupling for the addressed ion (a beam geometry whose
        quadrupole coupling vanishes): the table's beliefs are then kept."""
        key = (ion, beams, kind)
        if key not in self._crosstalk:
            from qutip_trap.light.raman import crosstalk_ratios

            try:
                self._crosstalk[key] = dict(crosstalk_ratios(self.device, ion, beams, kind=kind))  # type: ignore[arg-type]
            except ZeroDivisionError:
                self._crosstalk[key] = None
        return self._crosstalk[key]


def physical_drive(
    device: Device,
    drive: Drive,
    duration_s: float,
    table: CalibrationTable,
    truth: _Truth,
    notes: list[str],
    tag: str,
) -> Drive:
    """The drive as the ions see it: physical envelopes, light shift and crosstalk (module docstring)."""
    if not drive.programmed:
        return drive
    if drive.kind not in ("raman", "optical_E1", "optical_E2"):
        # a microwave, gradient or light-shift drive has no derivable rf-power-to-Omega map here: the request is played as is
        return replace(drive, programmed=False)
    ion = drive.ions[0]
    key = (ion, drive.beams[0] if drive.beams else MICROWAVE_BEAM_KEY)
    omega_true, stark_true = truth.rabi_and_stark(ion, drive.beams, drive.kind)
    belief = table.rabi.get(key)
    if usable(belief) and belief is not None and belief.value > 0.0 and omega_true == 0.0:
        # the table can drive the pulse and the device derives zero coupling for this beam geometry: refuse rather than
        # play a pulse the beams cannot produce
        raise ValueError(
            f"{tag}: the device derives no {drive.kind} coupling for beams {drive.beams} on ion {ion} (Omega = 0), while "
            f"the table carries a usable Rabi entry of {belief.value:.6g} Hz for {key}: the requested Rabi frequency "
            "cannot be played. Fix the beam geometry (polarization, k_hat and B direction) or the drive assignment."
        )
    if usable(belief) and belief is not None and belief.value > 0.0 and omega_true > 0.0:
        ratio = omega_true / float(belief.value)
    else:
        ratio = 1.0
        notes.append(
            f"{tag}: no usable Rabi entry for {key} in the table; the requested Rabi frequency is played as the physical one"
        )
    tones = tuple(replace(t, envelope_hz=_scale_envelope(t.envelope_hz, ratio)) for t in drive.tones)
    # the physical light shift at the played intensity
    power = stark_scaling_power(drive.kind)
    stark: float | Callable[[float], float] = 0.0
    if omega_true > 0.0 and stark_true != 0.0:
        envs = [t.envelope_hz for t in tones]
        constants = [float(e) for e in envs if isinstance(e, (int, float))]
        if len(constants) == len(envs):
            stark = stark_true * float(sum((abs(e) / omega_true) ** power for e in constants))
        else:
            stark = _PhysicalStarkFn(stark_true, omega_true, power, tuple(envs), duration_s)
    # crosstalk: the derived ratios of the neighbours the table lists
    derived = truth.crosstalk(ion, drive.beams, drive.kind)
    if derived is None:
        notes.append(
            f"{tag}: the device derives no coupling for ion {ion} under beams {drive.beams}; the table's crosstalk kept"
        )
        derived = {j: complex(v) for j, v in drive.crosstalk.items()}
    xt: dict[int, complex] = {}
    for j, believed in drive.crosstalk.items():
        if j in derived:
            xt[j] = complex(derived[j])
        else:
            xt[j] = complex(believed)
            notes.append(
                f"{tag}: the device derives no light on ion {j} under beams {drive.beams}; the table's ratio kept"
            )
    unlisted = {j: abs(v) for j, v in derived.items() if j not in xt and j not in drive.ions}
    if unlisted:
        worst = max(unlisted, key=lambda j: unlisted[j])
        if unlisted[worst] >= UNLISTED_CROSSTALK_REPORT:
            notes.append(
                f"{tag}: derived crosstalk {unlisted[worst]:.3g} on ion {worst} is not listed in the table and is not simulated"
            )
    return replace(drive, tones=tones, stark_shift_hz=stark, crosstalk=xt, programmed=False)


def physical_schedule(
    device: Device, schedule: Schedule, table: CalibrationTable
) -> tuple[Schedule, tuple[str, ...]]:
    """The schedule as the ions see it: every programmed drive converted through the device's derived values."""
    truth = _Truth(device)
    notes: list[str] = []
    pulses: list[Pulse] = []
    changed = False
    for p in schedule.pulses:
        if not p.drive.programmed:
            pulses.append(p)
            continue
        changed = True
        local: list[str] = []
        d = physical_drive(
            device, p.drive, p.duration_s, table, truth, local, p.gate_id or f"pulse@{p.t_start_s:.9g}"
        )
        for note in local:
            if note not in notes:
                notes.append(note)
        pulses.append(Pulse(d, p.t_start_s, p.t_end_s, p.gate_id, p.closes_modes))
    if not changed:
        return schedule, ()
    return replace(schedule, pulses=tuple(pulses), phase_frame=dict(schedule.phase_frame)), tuple(notes)


TAIL_TIME_CONSTANTS = 8.0
"""The exponential tail kept after a train: e^-8 = 3.4e-4 of the last amplitude is dropped."""
SAMPLES_PER_TIME_CONSTANT = 20
MIN_SAMPLES = 64
MAX_SAMPLES = 200_001


@dataclass(frozen=True)
class HardwareChain:
    """DDS/AOM/amplifier response: quantization, rise time, bandwidth, dead time, phase continuity."""

    dds_phase_bits: int
    dds_amplitude_bits: int
    aom_rise_s: float
    amplifier_bandwidth_hz: float
    dead_time_s: float
    phase_continuous: bool
    dds_clock_hz: float | None = None
    """DDS system clock; with ``dds_frequency_bits`` it sets the frequency grid f_clk/2^bits (0.23 Hz at 1 GHz, 32 bits)."""
    dds_frequency_bits: int | None = None
    amplitude_full_scale_hz: float | None = None
    """The Rabi frequency at the full-scale amplitude word; None quantizes relative to each train's own peak."""
    amplifier_saturation_hz: float | None = None
    """Omega_sat of the static nonlinearity Omega = Omega_sat tanh(V/V_sat); None = linear."""
    timing_jitter_s: float = 0.0
    """rms per-train start-time jitter (sequencer latency), drawn per sample from the keyed seeds."""
    parallel_addressing: bool = False
    """Whether the chain can play single-qubit pulses on distinct ions at the same time (one AWG channel per addressing
    beam; a switched single channel cannot). Entangling gates are serialized one at a time per crystal whatever it says:
    they share the global beam pair."""

    def __post_init__(self) -> None:
        if self.dds_phase_bits <= 0 or self.dds_amplitude_bits <= 0:
            raise ValueError("DDS bit depths must be positive")
        if self.aom_rise_s < 0.0 or self.amplifier_bandwidth_hz <= 0.0 or self.dead_time_s < 0.0:
            raise ValueError("rise time and dead time are non-negative, bandwidth positive")
        if (self.dds_clock_hz is None) != (self.dds_frequency_bits is None):
            raise ValueError("dds_clock_hz and dds_frequency_bits come together")
        if self.dds_clock_hz is not None and (
            self.dds_clock_hz <= 0.0 or (self.dds_frequency_bits or 0) <= 0
        ):
            raise ValueError("a positive DDS clock and a positive frequency word")
        if self.timing_jitter_s < 0.0:
            raise ValueError("timing jitter is non-negative")
        if self.amplifier_saturation_hz is not None and self.amplifier_saturation_hz <= 0.0:
            raise ValueError("the saturation Rabi frequency is positive")

    @property
    def frequency_resolution_hz(self) -> float | None:
        if self.dds_clock_hz is None or self.dds_frequency_bits is None:
            return None
        return float(self.dds_clock_hz / 2**self.dds_frequency_bits)

    @property
    def phase_resolution_rad(self) -> float:
        return float(2.0 * math.pi / 2**self.dds_phase_bits)

    def amplitude_step_hz(self, full_scale_hz: float) -> float:
        return float(full_scale_hz / 2**self.dds_amplitude_bits)

    def response_time_s(self, kind: str) -> float:
        """The first-order time constant a drive's envelope sees: the AOM rise time for laser light, 1/(2 pi B) of the amplifier
        for a microwave or gradient drive (no modulator in that path)."""
        if kind in ("microwave", "gradient"):
            return 1.0 / (2.0 * math.pi * self.amplifier_bandwidth_hz)
        return float(self.aom_rise_s)

    @property
    def ideal(self) -> bool:
        return (
            self.aom_rise_s == 0.0
            and math.isinf(self.amplifier_bandwidth_hz)
            and self.timing_jitter_s == 0.0
            and self.frequency_resolution_hz is None
            and self.amplifier_saturation_hz is None
        )

    def describe(self) -> tuple[str, ...]:
        """What the chain applies, for ``Device.derived()`` and the run's approximations."""
        out = [
            f"hardware: phase word {self.dds_phase_bits} bits ({self.phase_resolution_rad:.2e} rad), amplitude word "
            f"{self.dds_amplitude_bits} bits, dead time {self.dead_time_s:.3g} s, "
            f"{'phase-continuous tones' if self.phase_continuous else 'beat note reset per gate'}"
        ]
        if self.frequency_resolution_hz is None:
            out.append("hardware: no DDS frequency word configured (ideal frequency resolution)")
        else:
            out.append(f"hardware: DDS frequency grid {self.frequency_resolution_hz:.3g} Hz")
        out.append(
            f"hardware: AOM first-order response tau = {self.aom_rise_s:.3g} s"
            if self.aom_rise_s > 0.0
            else "hardware: ideal modulator (zero rise time)"
        )
        out.append(
            f"hardware: amplifier bandwidth {self.amplifier_bandwidth_hz:.3g} Hz (microwave envelope time constant "
            f"{self.response_time_s('microwave'):.3g} s)"
            if math.isfinite(self.amplifier_bandwidth_hz)
            else "hardware: infinite amplifier bandwidth (ideal microwave envelope)"
        )
        out.append(
            f"hardware: amplifier saturation Omega_sat/2pi = {self.amplifier_saturation_hz:.3g} Hz"
            if self.amplifier_saturation_hz is not None
            else "hardware: linear amplifier"
        )
        out.append(
            f"hardware: timing jitter {self.timing_jitter_s:.3g} s rms per train"
            if self.timing_jitter_s > 0.0
            else "hardware: no timing jitter"
        )
        return tuple(out)


# ---- applying the chain to a schedule -------------------------------------------------------------------------------------------


def _round_to(value: float, step: float) -> float:
    return float(round(value / step) * step) if step > 0.0 else value


def _sample_envelope(value: Envelope, duration: float, n: int) -> np.ndarray:
    """The programmed envelope on n uniform samples over [0, duration]."""
    grid = np.linspace(0.0, duration, n)
    if callable(value):
        return np.array([float(value(x)) for x in grid])
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        src = np.linspace(0.0, duration, arr.size)
        return np.asarray(np.interp(grid, src, arr))
    return np.full(n, float(value))


def _shape_array(x: np.ndarray, amp_step: float, saturation: float | None) -> np.ndarray:
    """The amplitude word (rounding to ``amp_step``) and the amplifier's static nonlinearity (a tanh saturation)."""
    y = np.round(x / amp_step) * amp_step if amp_step > 0.0 else x
    if saturation is not None:
        y = saturation * np.tanh(y / saturation)
    return np.asarray(y, dtype=float)


@dataclass(frozen=True, eq=False)
class ShapedFn:
    """A programmed callable envelope through the amplitude word and the amplifier saturation, sample by sample (picklable
    when the programmed callable is)."""

    fn: Callable[[float], float]
    amp_step: float
    saturation: float | None

    def __call__(self, tau: float) -> float:
        return float(
            _shape_array(np.asarray(float(self.fn(tau)), dtype=float), self.amp_step, self.saturation)
        )


def _shaped(value: Envelope, amp_step: float, saturation: float | None) -> Envelope:
    """The amplitude word and the amplifier's static nonlinearity applied to a programmed envelope without resampling it: a
    constant stays a constant, a sampled array keeps its grid, a callable is wrapped."""
    if amp_step <= 0.0 and saturation is None:
        return value
    if callable(value):
        return ShapedFn(value, amp_step, saturation)
    if isinstance(value, np.ndarray):
        return _shape_array(np.asarray(value, dtype=float), amp_step, saturation)
    return float(_shape_array(np.asarray(float(value), dtype=float), amp_step, saturation))


def _constant(value: Envelope) -> float | None:
    """The value of a constant envelope, None for a callable or a sampled one."""
    if callable(value) or isinstance(value, np.ndarray):
        return None
    return float(value)


def _stark_reference(pulse: Pulse, power: int) -> tuple[float, tuple[float, ...]]:
    """The programmed Stark shift and tone amplitudes read at the pulse's programmed peak intensity: for a shift homogeneous of
    degree ``power`` in the amplitudes any time with light would do, and the peak keeps the reference away from zero."""
    n = MIN_SAMPLES + 1
    grid = np.linspace(0.0, pulse.duration_s, n)
    sampled = [np.abs(_sample_envelope(tone.envelope_hz, pulse.duration_s, n)) for tone in pulse.drive.tones]
    total = sum((arr**power for arr in sampled), np.zeros(n))
    k = int(np.argmax(total))
    st = pulse.drive.stark_shift_hz
    st_ref = float(st(float(grid[k]))) if callable(st) else float(st)
    return st_ref, tuple(float(arr[k]) for arr in sampled)


@dataclass(frozen=True, eq=False)
class PlayedStarkFn:
    """delta_ref x mean_tones (|Omega_played(tau)|/Omega_ref)^p: the Stark shift that follows the played light (picklable)."""

    st_ref: float
    fns: tuple[Callable[[float], float], ...]
    refs: tuple[float, ...]
    power: int

    def __call__(self, tau: float) -> float:
        return self.st_ref * float(
            np.mean([(abs(fn(tau)) / ref) ** self.power for fn, ref in zip(self.fns, self.refs)])
        )


def _stark_played(
    programmed: Callable[[float], float] | float,
    reference: tuple[float, tuple[float, ...]],
    played: Sequence[Envelope],
    power: int,
    duration: float,
) -> Callable[[float], float] | float:
    """The Stark shift of the played light, delta_ref mean_tones (|Omega_played(t)|/Omega_ref)^p over the tones with a non-zero
    reference amplitude; the programmed shift itself when no tone carries light (there is nothing to follow)."""
    st_ref, refs = reference
    pairs = [(env, ref) for env, ref in zip(played, refs) if ref > 0.0]
    if not pairs:
        return programmed
    if st_ref == 0.0:
        return 0.0
    constants = [_constant(env) for env, _ in pairs]
    if all(c is not None for c in constants):
        return st_ref * float(
            np.mean([(abs(c) / ref) ** power for c, (_, ref) in zip(constants, pairs) if c is not None])
        )
    fns = [as_time_function(env, duration) for env, _ in pairs]
    scales = [ref for _, ref in pairs]
    return PlayedStarkFn(st_ref, tuple(fns), tuple(scales), int(power))


def _low_pass(samples: np.ndarray, dt: float, tau: float, y0: float) -> np.ndarray:
    """y' = (x - y)/tau with the exponential integrator (exact for a constant input, second order otherwise)."""
    out = np.empty_like(samples)
    y = y0
    decay = math.exp(-dt / tau)
    gain = 1.0 - decay
    out[0] = y
    for k in range(1, samples.size):
        y = y * decay + gain * 0.5 * (samples[k - 1] + samples[k])
        out[k] = y
    return out


def _trains(pulses: Sequence[Pulse]) -> list[list[Pulse]]:
    """Contiguous pulses on the same ions with the same drive kind and beams (the segments of one gate or composite pulse)."""
    ordered = sorted(pulses, key=lambda p: (p.t_start_s, tuple(p.drive.ions)))
    trains: list[list[Pulse]] = []
    for p in ordered:
        for tr in trains:
            last = tr[-1]
            same = (
                tuple(last.drive.ions) == tuple(p.drive.ions)
                and last.drive.kind == p.drive.kind
                and tuple(last.drive.beams) == tuple(p.drive.beams)
                and len(last.drive.tones) == len(p.drive.tones)
            )
            if same and abs(last.t_end_s - p.t_start_s) <= 1e-12 * max(1.0, abs(p.t_start_s)):
                tr.append(p)
                break
        else:
            trains.append([p])
    return trains


def apply_hardware_chain(
    schedule: Schedule,
    hardware: HardwareChain,
    *,
    rng: np.random.Generator | None = None,
    quantize: bool = True,
    response: bool = True,
    jitter: bool = True,
) -> tuple[Schedule, tuple[str, ...]]:
    """The schedule the hardware plays and the notes: quantized tone words, saturated and low-pass-filtered envelopes with
    their tails, jittered train starts."""
    from qutip_trap.control.pulses import Pulse, Tone
    from qutip_trap.control.schedule import Schedule, stark_scaling_power

    notes: list[str] = []
    if hardware.ideal and not quantize:
        return schedule, ("hardware: ideal electronics",)
    trains = _trains(schedule.pulses)
    new_pulses: list[Pulse] = []
    f_res = hardware.frequency_resolution_hz
    phase_step = hardware.phase_resolution_rad
    # rigid per-train jitter, clamped against neighbours on shared ions
    shifts: list[float] = [0.0] * len(trains)
    if jitter and hardware.timing_jitter_s > 0.0:
        if rng is None:
            raise ValueError("timing jitter needs the keyed random generator")
        for k in range(len(trains)):
            shifts[k] = float(rng.normal(0.0, hardware.timing_jitter_s))
        notes.append(
            f"hardware: {len(trains)} pulse trains jittered with rms {hardware.timing_jitter_s:.3g} s"
        )
    unquantized_callables = 0
    for k, train in enumerate(trains):
        ions = set(train[0].drive.ions)
        tau_r = hardware.response_time_s(train[0].drive.kind) if response else 0.0
        tail_s = TAIL_TIME_CONSTANTS * tau_r
        power = stark_scaling_power(train[0].drive.kind)
        # the earliest later pulse start on a shared ion bounds the tail and the shift
        later = [
            p.t_start_s
            for other in trains[k + 1 :]
            for p in other[:1]
            if set(p.drive.ions) & ions and p.t_start_s >= train[-1].t_end_s - 1e-15
        ]
        next_start = min(later) if later else math.inf
        shift = shifts[k]
        if shift > 0.0 and math.isfinite(next_start):
            shift = min(shift, max(next_start - train[-1].t_end_s - tail_s, 0.0))
        if shift < 0.0 and k > 0:
            prev_ends = [
                q.t_end_s
                for other in trains[:k]
                for q in other[-1:]
                if set(q.drive.ions) & ions and q.t_end_s <= train[0].t_start_s + 1e-15
            ]
            if prev_ends:
                shift = max(
                    shift,
                    min(train[0].t_start_s - max(prev_ends), 0.0) - (train[0].t_start_s - max(prev_ends)),
                )
        # full scale for the amplitude word: the train's programmed peak, probed on MIN_SAMPLES + 1 points per pulse
        peaks: list[float] = []
        for p in train:
            for tone in p.drive.tones:
                env = _sample_envelope(tone.envelope_hz, p.duration_s, MIN_SAMPLES + 1)
                peaks.append(float(np.max(np.abs(env))))
        full_scale = (
            hardware.amplitude_full_scale_hz
            if hardware.amplitude_full_scale_hz is not None
            else (max(peaks) if peaks else 1.0)
        )
        amp_step = hardware.amplitude_step_hz(full_scale) if quantize else 0.0
        saturation = hardware.amplifier_saturation_hz
        needs_array = response and tau_r > 0.0
        reshaped = needs_array or amp_step > 0.0 or saturation is not None
        # per-tone filter state across the train; the Stark reference of the pulse being played
        n_tones = len(train[0].drive.tones)
        states = [0.0] * n_tones
        reference: tuple[float, tuple[float, ...]] = (0.0, ())
        for p in train:
            if reshaped:
                reference = _stark_reference(p, power)
            tones: list[Tone] = []
            played: list[Envelope] = []
            for j, tone in enumerate(p.drive.tones):
                det = tone.detuning_hz
                if quantize and f_res is not None and not callable(det):
                    det = _round_to(float(det), f_res)
                ph = tone.phase_rad
                if quantize and not callable(ph):
                    ph = _round_to(float(ph), phase_step)
                elif callable(ph):
                    unquantized_callables += 1
                env_val: Envelope = tone.envelope_hz
                if needs_array:
                    n = min(
                        max(MIN_SAMPLES, int(math.ceil(p.duration_s / tau_r * SAMPLES_PER_TIME_CONSTANT))),
                        MAX_SAMPLES,
                    )
                    arr = _sample_envelope(tone.envelope_hz, p.duration_s, n)
                    arr = np.asarray(_shaped(arr, amp_step, saturation), dtype=float)
                    arr = _low_pass(arr, p.duration_s / (n - 1), tau_r, states[j])
                    states[j] = float(arr[-1])
                    env_val = arr
                elif reshaped:
                    env_val = _shaped(tone.envelope_hz, amp_step, saturation)
                played.append(env_val)
                tones.append(dataclasses.replace(tone, detuning_hz=det, phase_rad=ph, envelope_hz=env_val))
            stark = (
                _stark_played(p.drive.stark_shift_hz, reference, played, power, p.duration_s)
                if reshaped
                else p.drive.stark_shift_hz
            )
            drive = dataclasses.replace(p.drive, tones=tuple(tones), stark_shift_hz=stark)
            new_pulses.append(Pulse(drive, p.t_start_s + shift, p.t_end_s + shift, p.gate_id, p.closes_modes))
        if tail_s > 0.0 and any(abs(s) > 0.0 for s in states):
            last = train[-1]
            room = next_start - last.t_end_s - shift if math.isfinite(next_start) else math.inf
            t_tail = min(tail_s, max(room, 0.0))
            if t_tail <= 0.0:
                notes.append(
                    f"hardware: no room for the modulator tail after {last.gate_id!r} (dead time shorter than the response)"
                )
            else:
                if t_tail < tail_s:
                    notes.append(
                        f"hardware: modulator tail after {last.gate_id!r} truncated to {t_tail:.3g} s by the next pulse on a shared ion"
                    )
                n = max(MIN_SAMPLES, int(math.ceil(t_tail / tau_r * SAMPLES_PER_TIME_CONSTANT)))
                grid = np.linspace(0.0, t_tail, n)
                tones = []
                tails: list[Envelope] = []
                for j, tone in enumerate(last.drive.tones):
                    arr = states[j] * np.exp(-grid / tau_r)
                    ph = tone.phase_rad
                    if callable(ph):
                        ph = ConstantFn(
                            float(ph(last.duration_s))
                        )  # the phase holds its final value through the tail
                    det = tone.detuning_hz
                    if callable(det):
                        det = ConstantFn(float(det(last.duration_s)))
                    if quantize and not callable(det) and f_res is not None:
                        det = _round_to(float(det), f_res)
                    if quantize and not callable(ph):
                        ph = _round_to(float(ph), phase_step)
                    tails.append(arr)
                    tones.append(dataclasses.replace(tone, detuning_hz=det, phase_rad=ph, envelope_hz=arr))
                # the decaying light carries the decaying Stark shift, referenced to the last pulse's programmed peak
                stark = _stark_played(last.drive.stark_shift_hz, reference, tails, power, t_tail)
                drive = dataclasses.replace(last.drive, tones=tuple(tones), stark_shift_hz=stark)
                new_pulses.append(
                    Pulse(
                        drive,
                        last.t_end_s + shift,
                        last.t_end_s + shift + t_tail,
                        f"{last.gate_id}/tail",
                        last.closes_modes,
                    )
                )
    if unquantized_callables:
        notes.append(f"hardware: {unquantized_callables} callable phase schedules left unquantized")
    if response and hardware.aom_rise_s > 0.0:
        notes.append(
            f"hardware: envelopes low-pass filtered with tau = {hardware.aom_rise_s:.3g} s and {TAIL_TIME_CONSTANTS:g} tau tails"
        )
    if quantize:
        notes.append(
            f"hardware: phases rounded to {phase_step:.2e} rad, amplitudes to {hardware.dds_amplitude_bits}-bit words"
            + (f", frequencies to {f_res:.3g} Hz" if f_res is not None else "")
        )
    new = Schedule(
        tuple(sorted(new_pulses, key=lambda p: (p.t_start_s, tuple(p.drive.ions)))),
        schedule.idle,
        schedule.events,
        schedule.phase_frame,
        gates=schedule.gates,
        targets=schedule.targets,
        t0_s=schedule.t0_s,
    )
    return new, tuple(notes)
