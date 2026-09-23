"""Level 2, the schedule: the pulses of each ion on a time axis, one pulse's tones against the motional modes, and each
entangling gate's loop closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import PulseRecord, Record
from qutip_trap_app.viewmodel.shown import Shown

CLOSURE_THRESHOLD = 1e-6
"""|alpha_m|^2 (2 nbar_m + 1) below which a mode's loop counts as closed."""
SIDEBAND_TOLERANCE_HZ = 2e5
"""How close a tone must sit to a sideband to be named after it."""


def step_of_pulse(record: Record, pulse_index: int) -> int:
    return next((st.index for st in record.schedule.steps if pulse_index in st.pulse_indices), -1)


@dataclass(frozen=True)
class Span:
    pulse_index: int
    gate_id: str | None
    kind: str
    t_start_s: float
    t_end_s: float


@dataclass(frozen=True)
class TimeAxis:
    lanes: tuple[tuple[int, tuple[Span, ...]], ...]
    """Per ion, the pulses that address it."""
    measure: tuple[float, float] | None
    t0_s: float
    t1_s: float


def time_axis(record: Record) -> TimeAxis:
    sched = record.schedule
    lanes = tuple(
        (
            ion,
            tuple(
                Span(p.index, p.gate_id, p.kind, p.t_start_s, p.t_end_s)
                for p in sched.pulses
                if ion in p.ions
            ),
        )
        for ion in range(record.n_ions)
    )
    return TimeAxis(lanes, sched.measure, sched.t0_s, max(sched.duration_s, sched.t0_s + 1e-9))


@dataclass(frozen=True)
class Sideband:
    mode: int
    mode_frequency_hz: float
    detuning_hz: float
    """From the sideband on the tone's side of the carrier."""
    role: str
    """``red``, ``blue``, ``carrier`` or ``far``."""


@dataclass(frozen=True)
class ToneView:
    index: int
    detuning: Shown
    envelope_peak: Shown
    times_s: np.ndarray
    envelope_hz: np.ndarray
    sidebands: tuple[Sideband, ...]

    @property
    def role(self) -> str:
        """The sideband this tone drives, or ``carrier`` or ``far``."""
        near = next((sb.role for sb in self.sidebands if sb.role in ("red", "blue")), None)
        if near is not None:
            return near
        return "carrier" if abs(float(self.detuning.value or 0.0)) <= SIDEBAND_TOLERANCE_HZ else "far"


@dataclass(frozen=True)
class PulseView:
    pulse: PulseRecord
    tones: tuple[ToneView, ...]
    modes: tuple[tuple[int, str, float], ...]
    """Every mode of the crystal: (index, family and number, frequency in Hz)."""
    stark_shift: Shown
    crosstalk: tuple[Shown, ...]
    step_index: int


def _role(mu_hz: float, omega_hz: float) -> str:
    if abs(mu_hz) <= SIDEBAND_TOLERANCE_HZ:
        return "carrier"
    if abs(mu_hz - omega_hz) <= SIDEBAND_TOLERANCE_HZ:
        return "blue"
    if abs(mu_hz + omega_hz) <= SIDEBAND_TOLERANCE_HZ:
        return "red"
    return "far"


def pulse_view(record: Record, pulse_index: int) -> PulseView:
    p = record.schedule.pulses[pulse_index]
    tones = []
    for k, t in enumerate(p.tones):
        mu = t.detuning_hz
        tones.append(
            ToneView(
                index=k,
                detuning=Shown(
                    f"Tone {k}: detuning from the carrier", mu, "Hz", "+ blue of the carrier, - red"
                ),
                envelope_peak=Shown(
                    f"Tone {k}: peak Rabi frequency", float(np.max(np.abs(t.envelope_hz))), "Hz", "Omega/2 pi"
                ),
                times_s=np.linspace(p.t_start_s, p.t_end_s, t.envelope_hz.size),
                envelope_hz=t.envelope_hz,
                sidebands=tuple(
                    Sideband(
                        m.index,
                        m.omega_hz,
                        mu - m.omega_hz if mu >= 0 else mu + m.omega_hz,
                        _role(mu, m.omega_hz),
                    )
                    for m in record.modes
                ),
            )
        )
    return PulseView(
        pulse=p,
        tones=tuple(tones),
        modes=tuple((m.index, f"{m.family} {m.family_index}", m.omega_hz) for m in record.modes),
        stark_shift=Shown("Light shift of its own beams", p.stark_shift_hz, "Hz", "at the pulse start"),
        crosstalk=tuple(
            Shown(f"Crosstalk onto ion {j}", abs(e), detail="Rabi-frequency ratio")
            for j, e in sorted(p.crosstalk.items())
        ),
        step_index=step_of_pulse(record, pulse_index),
    )


@dataclass(frozen=True)
class ModeClosure:
    mode: int
    mode_class: str
    chi_m: Shown
    alpha: Shown
    residual: float
    """|alpha_m|^2 (2 nbar_m + 1)."""
    closed: bool


@dataclass(frozen=True)
class ClosureView:
    gate_id: str
    pair: tuple[int, int]
    modes: tuple[ModeClosure, ...]
    chi_total: Shown
    duration: Shown


def closure(record: Record, gate_id: str) -> ClosureView:
    gate = next((g for g in record.schedule.gates if g.gate_id == gate_id), None)
    if gate is None:
        raise KeyError(f"{gate_id!r} is not an entangling gate of this schedule")
    modes = []
    for m in sorted(set(gate.chi_m) | set(gate.alpha_m)):
        a = gate.alpha_m.get(m, 0.0)
        residual = abs(a) ** 2 * (2.0 * record.space.nbar.get(m, 0.0) + 1.0)
        modes.append(
            ModeClosure(
                mode=m,
                mode_class=record.space.mode_class.get(m, "?"),
                chi_m=Shown(f"Entangling angle, mode {m}", gate.chi_m.get(m, 0.0), "rad"),
                alpha=Shown(f"Leftover displacement, mode {m}", abs(a), detail="|alpha_m| at the gate's end"),
                residual=residual,
                closed=residual < CLOSURE_THRESHOLD,
            )
        )
    return ClosureView(
        gate_id=gate_id,
        pair=gate.pair,
        modes=tuple(modes),
        chi_total=Shown("Entangling angle", gate.chi_total_rad, "rad", "signed sum over the modes"),
        duration=Shown("Gate duration", gate.duration_s, "s"),
    )
