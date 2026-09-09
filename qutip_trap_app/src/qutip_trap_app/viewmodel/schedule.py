"""Level 2, the schedule: pulses on a time axis per ion, tones against the mode spectrum, waveform segment tables,
loop-closure indicators and crosstalk (PLAN.md Section 14.2, row 2).

Every quantity is computed from the record: the tones are the scheduler's own (sampled where they were functions), the
mode spectrum is the crystal's, the two-indexed detuning delta_{i,m} = mu_i - omega_m follows Section 13
(``conv.detuning_symbols``), and a mode's closure indicator is the played waveform's residual displacement alpha_m weighted
by (2 nbar_m + 1) against the drop threshold of Section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import PulseRecord, Record, SampledFn, WaveformRecord
from qutip_trap_app.viewmodel.catalogue import Shown

CLOSURE_THRESHOLD = 1e-6
"""Section 5.2's drop criterion |alpha_m(tau)|^2 (2 nbar_m + 1) < 1e-6: below it a loop counts as closed."""


@dataclass(frozen=True)
class Span:
    pulse_index: int
    gate_id: str | None
    kind: str
    t_start_s: float
    t_end_s: float
    step_index: int


@dataclass(frozen=True)
class Lane:
    ion: int
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class TimeAxis:
    lanes: tuple[Lane, ...]
    idle: tuple[Shown, ...]
    measurement: Shown | None
    t0_s: float
    pulses_end_s: float
    duration_s: float


def _step_of_pulse(record: Record, pulse_index: int) -> int:
    for st in record.schedule.steps:
        if pulse_index in st.pulse_indices:
            return st.index
    return -1


def time_axis(record: Record) -> TimeAxis:
    n_ions = record.device_card.n_ions
    lanes: list[Lane] = []
    for ion in range(n_ions):
        spans = tuple(
            Span(p.index, p.gate_id, p.kind, p.t_start_s, p.t_end_s, _step_of_pulse(record, p.index))
            for p in record.schedule.pulses
            if ion in p.ions
        )
        lanes.append(Lane(ion, spans))
    meas = None
    for e in record.schedule.events:
        if e.kind == "measure":
            meas = Shown("measurement_event", e.t_start_s, f"until {e.t_end_s:.6g} s on ions {e.ions}")
    return TimeAxis(
        lanes=tuple(lanes),
        idle=tuple(Shown("idle", b - a, f"{a:.6g} to {b:.6g} s") for a, b in record.schedule.idle),
        measurement=meas,
        t0_s=record.schedule.t0_s,
        pulses_end_s=record.schedule.pulses_end_s,
        duration_s=record.schedule.duration_s,
    )


@dataclass(frozen=True)
class SidebandView:
    mode: int
    mode_frequency: Shown
    detuning: Shown
    """delta_{i,m} = mu - omega_m (Hz), the two-indexed detuning (conv.detuning_symbols)."""
    role: str
    """``red`` (mu ~ -omega_m), ``blue`` (mu ~ +omega_m), ``carrier`` (mu ~ 0) or ``far``."""


@dataclass(frozen=True)
class ToneView:
    index: int
    detuning: Shown
    envelope_peak: Shown
    phase_start: Shown
    envelope_samples: np.ndarray
    detuning_samples: np.ndarray
    times_s: np.ndarray
    sidebands: tuple[SidebandView, ...]


@dataclass(frozen=True)
class PulseView:
    pulse: PulseRecord
    ions: tuple[int, ...]
    beams: tuple[Shown, ...]
    tones: tuple[ToneView, ...]
    mode_spectrum: tuple[Shown, ...]
    crosstalk: tuple[Shown, ...]
    stark_shift: Shown
    step_index: int
    closes_modes: tuple[int, ...]


def _grid(fn: SampledFn, duration_s: float, n: int) -> np.ndarray:
    tau = np.linspace(0.0, duration_s, n)
    return fn.at(tau, duration_s)


def _role(delta_hz: float, mu_hz: float, omega_hz: float, tolerance_hz: float) -> str:
    if abs(mu_hz) <= tolerance_hz:
        return "carrier"
    if abs(mu_hz - omega_hz) <= tolerance_hz:
        return "blue"
    if abs(mu_hz + omega_hz) <= tolerance_hz:
        return "red"
    return "far"


def pulse_view(
    record: Record, pulse_index: int, *, n_points: int = 201, sideband_tolerance_hz: float = 2e5
) -> PulseView:
    p = record.schedule.pulses[pulse_index]
    card = record.device_card
    tau = np.linspace(0.0, p.duration_s, n_points)
    tones: list[ToneView] = []
    for k, t in enumerate(p.tones):
        env = _grid(t.envelope_hz, p.duration_s, n_points)
        det = _grid(t.detuning_hz, p.duration_s, n_points)
        ph = _grid(t.phase_rad, p.duration_s, n_points)
        mu0 = float(det[0])
        sidebands = tuple(
            SidebandView(
                mode=m.index,
                mode_frequency=Shown("mode_frequency", m.omega_hz, f"{m.family} {m.family_index}"),
                detuning=Shown(
                    "sideband_detuning",
                    mu0 - m.omega_hz if mu0 >= 0 else mu0 + m.omega_hz,
                    f"tone {k}, mode {m.index}",
                ),
                role=_role(mu0 - m.omega_hz, mu0, m.omega_hz, sideband_tolerance_hz),
            )
            for m in card.modes
        )
        tones.append(
            ToneView(
                index=k,
                detuning=Shown("tone_detuning", mu0, "from the carrier; sign: + blue, - red"),
                envelope_peak=Shown("tone_envelope", float(np.max(np.abs(env))), "peak Rabi frequency"),
                phase_start=Shown("tone_phase", float(ph[0])),
                envelope_samples=env,
                detuning_samples=det,
                times_s=tau + p.t_start_s,
                sidebands=sidebands,
            )
        )
    beams = tuple(
        Shown(
            "beam_direction",
            f"({b.k_hat[0]:+.2f}, {b.k_hat[1]:+.2f}, {b.k_hat[2]:+.2f})",
            f"beam {b.index}, {b.wavelength_m * 1e9:.1f} nm",
        )
        for b in card.beams
        if b.index in p.beams
    )
    stark = _grid(p.stark_shift_hz, p.duration_s, 3)
    return PulseView(
        pulse=p,
        ions=p.ions,
        beams=beams,
        tones=tuple(tones),
        mode_spectrum=tuple(
            Shown("mode_frequency", m.omega_hz, f"mode {m.index}: {m.family} {m.family_index}")
            for m in card.modes
        ),
        crosstalk=tuple(Shown("crosstalk", abs(e), f"onto ion {j}") for j, e in sorted(p.crosstalk.items())),
        stark_shift=Shown("stark_shift", float(stark[0])),
        step_index=_step_of_pulse(record, pulse_index),
        closes_modes=p.closes_modes,
    )


@dataclass(frozen=True)
class ModeClosure:
    mode: int
    chi_m: Shown
    alpha_m: Shown
    residual: float
    """|alpha_m|^2 (2 nbar_m + 1)."""
    closed: bool
    mode_class: str


@dataclass(frozen=True)
class ClosureView:
    gate_id: str
    pair: tuple[int, int]
    modes: tuple[ModeClosure, ...]
    chi_total: Shown
    duration: Shown
    segments: tuple[tuple[Shown, ...], ...]
    """One row per segment: duration, then Omega per (ion, leg), then the leg detunings."""
    beat_phase: Shown | None


def closure(record: Record, gate_id: str) -> ClosureView:
    gate = next((g for g in record.schedule.gates if g.gate_id == gate_id), None)
    if gate is None:
        raise KeyError(f"{gate_id!r} is not an entangling gate of this schedule")
    wf: WaveformRecord = gate.waveform
    nbar = record.space.nbar
    classes = record.space.mode_class
    modes: list[ModeClosure] = []
    for m in sorted(set(wf.chi_m) | set(wf.alpha_m)):
        a = wf.alpha_m.get(m, 0.0)
        res = abs(a) ** 2 * (2.0 * float(nbar.get(m, 0.0)) + 1.0)
        modes.append(
            ModeClosure(
                mode=m,
                chi_m=Shown("chi_m", wf.chi_m.get(m, 0.0)),
                alpha_m=Shown("closure_alpha", abs(a), f"|alpha_{m}| at closure"),
                residual=res,
                closed=res < CLOSURE_THRESHOLD,
                mode_class=classes.get(m, "?"),
            )
        )
    rows: list[tuple[Shown, ...]] = []
    for s in wf.segments or ():
        row = [Shown("waveform_segment", s.duration_s, "duration")]
        for key, fn in sorted(s.amplitude_hz.items()):
            peak = (
                fn.value
                if fn.kind == "constant" and fn.value is not None
                else float(np.max(np.abs(fn.samples)))
                if fn.samples is not None
                else 0.0
            )
            row.append(Shown("tone_envelope", peak, f"Omega ion,leg {key}"))
        for leg, fn in sorted(s.detuning_hz.items()):
            val = (
                fn.value
                if fn.kind == "constant" and fn.value is not None
                else float(fn.samples[0])
                if fn.samples is not None
                else 0.0
            )
            row.append(Shown("tone_detuning", val, f"{leg} leg"))
        rows.append(tuple(row))
    return ClosureView(
        gate_id=gate_id,
        pair=gate.pair,
        modes=tuple(modes),
        chi_total=Shown("entangling_angle", wf.chi_total_rad, "signed sum over modes"),
        duration=Shown("gate_duration", wf.duration_s),
        segments=tuple(rows),
        beat_phase=Shown("beat_phase", wf.phi_m.value, wf.phi_m.status),
    )


def lamb_dicke_table(record: Record) -> tuple[Shown, ...]:
    """eta per (ion, mode) where the calibration table carries it (M8's sideband extraction); empty otherwise."""
    out = []
    for key, e in sorted(record.table.entries.items()):
        if key.startswith("lamb_dicke["):
            out.append(Shown("lamb_dicke", e.value, key))
    return tuple(out)


__all__ = [
    "CLOSURE_THRESHOLD",
    "ClosureView",
    "Lane",
    "ModeClosure",
    "PulseView",
    "SidebandView",
    "Span",
    "TimeAxis",
    "ToneView",
    "closure",
    "lamb_dicke_table",
    "pulse_view",
    "time_axis",
]
