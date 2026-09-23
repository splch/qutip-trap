"""The control hardware chain (DDS, AOM, amplifier): the schedule the hardware plays rather than the one programmed.

Every effect enters through the tone coefficients Omega(t) e^{-i(mu t - phi(t))}: constant detunings and phases and the
amplitudes are rounded to the DDS words; the amplifier saturates as Omega_sat tanh(Omega/Omega_sat); envelopes are
low-pass filtered (first order) continuously across a pulse train, with an exponential tail truncated at the next pulse
on a shared ion; the Stark shift follows the played light; each train is shifted rigidly by the timing jitter.
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

TAIL_TIME_CONSTANTS = 8.0
"""The exponential tail kept after a train, in time constants: e^-8 = 3.4e-4 of the last amplitude is dropped."""
SAMPLES_PER_TIME_CONSTANT = 20
MIN_SAMPLES = 64
MAX_SAMPLES = 200_001


@dataclass(frozen=True)
class HardwareChain:
    """DDS/AOM/amplifier parameters: quantization, rise time, bandwidth, dead time, phase continuity."""

    dds_phase_bits: int
    dds_amplitude_bits: int
    aom_rise_s: float
    amplifier_bandwidth_hz: float
    dead_time_s: float
    phase_continuous: bool
    dds_clock_hz: float | None = None
    """DDS system clock; with ``dds_frequency_bits`` it sets the frequency grid f_clk/2^bits."""
    dds_frequency_bits: int | None = None
    amplitude_full_scale_hz: float | None = None
    """The Rabi frequency at the full-scale amplitude word; None quantizes relative to each train's own peak."""
    amplifier_saturation_hz: float | None = None
    """Omega_sat of the static nonlinearity Omega = Omega_sat tanh(V/V_sat); None = linear."""
    timing_jitter_s: float = 0.0
    """rms per-train start-time jitter, drawn from the keyed seeds."""
    parallel_addressing: bool = False
    """Whether single-qubit pulses on distinct ions may play at the same time (the default of ``parallel=None``);
    entangling gates are serialized per crystal regardless."""

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
        """First-order envelope time constant: the AOM rise time for light, 1/(2 pi B) for microwave and gradient drives."""
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
        """One line of text per effect the chain applies."""
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


def _held(fn: Callable[[float], float], at: float) -> Callable[[float], float]:
    return ConstantFn(float(fn(at)))


def _round_to(value: float, step: float) -> float:
    return float(round(value / step) * step) if step > 0.0 else value


Envelope = Callable[[float], float] | np.ndarray | float


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
    """A callable envelope through the amplitude word and the amplifier saturation; picklable when ``fn`` is."""

    fn: Callable[[float], float]
    amp_step: float
    saturation: float | None

    def __call__(self, tau: float) -> float:
        return float(
            _shape_array(np.asarray(float(self.fn(tau)), dtype=float), self.amp_step, self.saturation)
        )


def _shaped(value: Envelope, amp_step: float, saturation: float | None) -> Envelope:
    """``_shape_array`` applied to an envelope without resampling it (a callable is wrapped in ``ShapedFn``)."""
    if amp_step <= 0.0 and saturation is None:
        return value
    if callable(value):
        return ShapedFn(value, amp_step, saturation)
    if isinstance(value, np.ndarray):
        return _shape_array(np.asarray(value, dtype=float), amp_step, saturation)
    return float(_shape_array(np.asarray(float(value), dtype=float), amp_step, saturation))


def _constant(value: Envelope) -> float | None:
    if callable(value) or isinstance(value, np.ndarray):
        return None
    return float(value)


def _as_function(value: Envelope, duration: float) -> Callable[[float], float]:
    return as_time_function(value, duration)


def _stark_reference(pulse: Pulse, power: int) -> tuple[float, tuple[float, ...]]:
    """The programmed Stark shift and tone amplitudes at the pulse's programmed peak intensity (the shift is taken as
    homogeneous of degree ``power`` in the amplitudes, exact when the tones share one envelope shape)."""
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
    """The Stark shift of the played light over the tones with light at the reference; the programmed one if none has."""
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
    fns = [_as_function(env, duration) for env, _ in pairs]
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
    """Contiguous pulses on the same ions with the same drive kind, beams and tone count (one gate's segments)."""
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
    """The schedule the hardware plays and the chain's notes; ``quantize``, ``response`` and ``jitter`` switch those
    effects off, and jitter needs ``rng``."""
    from qutip_trap.control.pulses import Pulse, Tone
    from qutip_trap.control.schedule import Schedule, stark_scaling_power

    notes: list[str] = []
    if hardware.ideal and not quantize:
        return schedule, ("hardware: ideal electronics",)
    trains = _trains(schedule.pulses)
    new_pulses: list[Pulse] = []
    f_res = hardware.frequency_resolution_hz
    phase_step = hardware.phase_resolution_rad
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
                        ph = _held(ph, last.duration_s)
                    det = tone.detuning_hz
                    if callable(det):
                        det = _held(det, last.duration_s)
                    if quantize and not callable(det) and f_res is not None:
                        det = _round_to(float(det), f_res)
                    if quantize and not callable(ph):
                        ph = _round_to(float(ph), phase_step)
                    tails.append(arr)
                    tones.append(dataclasses.replace(tone, detuning_hz=det, phase_rad=ph, envelope_hz=arr))
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
        transports=schedule.transports,
        gates=schedule.gates,
        targets=schedule.targets,
        t0_s=schedule.t0_s,
    )
    return new, tuple(notes)
