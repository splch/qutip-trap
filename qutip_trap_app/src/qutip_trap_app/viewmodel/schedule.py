"""Level 2, the schedule: pulses on a time axis per ion, tones against the mode spectrum, waveform segment tables and
loop-closure indicators (PLAN.md Section 14.2).

The two-indexed detuning delta_{i,m} = mu_i - omega_m follows ``conv.detuning_symbols``; a mode's closure indicator is the
played waveform's residual displacement alpha_m weighted by (2 nbar_m + 1) against the drop threshold of Section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qutip_trap_app.record import PulseRecord, Record, SampledFn
from qutip_trap_app.viewmodel.catalogue import Shown, vector_text

CLOSURE_THRESHOLD = 1e-6
"""Section 5.2's drop criterion |alpha_m(tau)|^2 (2 nbar_m + 1) < 1e-6: below it a loop counts as closed."""

N_POINTS = 201
"""Samples of a tone's envelope over its pulse."""

SIDEBAND_TOLERANCE_HZ = 2e5
"""A tone within this of the carrier, or of a mode's sideband, drives it."""

ToneRole = Literal["carrier", "blue", "red", "far"]


@dataclass(frozen=True)
class Span:
    pulse_index: int
    gate_id: str | None
    kind: str
    t_start_s: float
    t_end_s: float


@dataclass(frozen=True)
class Lane:
    ion: int
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class TimeAxis:
    lanes: tuple[Lane, ...]
    measurement: Shown | None
    t0_s: float
    duration_s: float


def time_axis(record: Record) -> TimeAxis:
    lanes = tuple(
        Lane(
            ion,
            tuple(
                Span(p.index, p.gate_id, p.kind, p.t_start_s, p.t_end_s)
                for p in record.schedule.pulses
                if ion in p.ions
            ),
        )
        for ion in range(record.device_card.n_ions)
    )
    meas = None
    for e in record.schedule.events:
        if e.kind == "measure":
            meas = Shown("measurement_event", e.t_start_s, f"until {e.t_end_s:.6g} s on ions {e.ions}")
    return TimeAxis(
        lanes=lanes, measurement=meas, t0_s=record.schedule.t0_s, duration_s=record.schedule.duration_s
    )


@dataclass(frozen=True)
class SidebandView:
    mode: int
    mode_frequency: Shown
    detuning: Shown
    """The tone's distance to the sideband on its side of the carrier (conv.detuning_symbols)."""
    role: ToneRole


@dataclass(frozen=True)
class ToneView:
    index: int
    detuning: Shown
    envelope_peak: Shown
    phase_start: Shown
    envelope_samples: np.ndarray
    times_s: np.ndarray
    sidebands: tuple[SidebandView, ...]
    role: ToneRole
    """The sideband the tone drives (red, blue), else the carrier or far."""


@dataclass(frozen=True)
class PulseView:
    pulse: PulseRecord
    ions: tuple[int, ...]
    beams: tuple[Shown, ...]
    tones: tuple[ToneView, ...]
    mode_spectrum: tuple[Shown, ...]
    crosstalk: tuple[Shown, ...]
    stark_shift: Shown


def _role(mu_hz: float, omega_hz: float) -> ToneRole:
    if abs(mu_hz) <= SIDEBAND_TOLERANCE_HZ:
        return "carrier"
    if abs(mu_hz - omega_hz) <= SIDEBAND_TOLERANCE_HZ:
        return "blue"
    if abs(mu_hz + omega_hz) <= SIDEBAND_TOLERANCE_HZ:
        return "red"
    return "far"


def pulse_view(record: Record, pulse_index: int) -> PulseView:
    p = record.schedule.pulses[pulse_index]
    card = record.device_card
    tau = np.linspace(0.0, p.duration_s, N_POINTS)
    tones: list[ToneView] = []
    for k, t in enumerate(p.tones):
        env = t.envelope_hz.at(tau, p.duration_s)
        mu0 = float(t.detuning_hz.at(tau[:1], p.duration_s)[0])
        side = "blue" if mu0 >= 0 else "red"
        # the distance to the sideband on the tone's side of the carrier, so the two legs of a bichromatic pulse read as the
        # mirror pair they are
        sidebands = tuple(
            SidebandView(
                mode=m.index,
                mode_frequency=Shown("mode_frequency", m.omega_hz, f"{m.family} {m.family_index}"),
                detuning=Shown(
                    "sideband_detuning",
                    mu0 - m.omega_hz if mu0 >= 0 else mu0 + m.omega_hz,
                    f"tone {k}, mode {m.index}: from the {side} sideband",
                ),
                role=_role(mu0, m.omega_hz),
            )
            for m in card.modes
        )
        role: ToneRole = next(
            (sb.role for sb in sidebands if sb.role in ("red", "blue")),
            "carrier" if abs(mu0) <= SIDEBAND_TOLERANCE_HZ else "far",
        )
        tones.append(
            ToneView(
                index=k,
                detuning=Shown("tone_detuning", mu0, "from the carrier; sign: + blue, - red"),
                envelope_peak=Shown("tone_envelope", float(np.max(np.abs(env))), "peak Rabi frequency"),
                phase_start=Shown("tone_phase", float(t.phase_rad.at(tau[:1], p.duration_s)[0])),
                envelope_samples=env,
                times_s=tau + p.t_start_s,
                sidebands=sidebands,
                role=role,
            )
        )
    beams = tuple(
        Shown("beam_direction", vector_text(b.k_hat), f"beam {b.index}, {b.wavelength_m * 1e9:.1f} nm")
        for b in card.beams
        if b.index in p.beams
    )
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
        stark_shift=Shown("stark_shift", float(p.stark_shift_hz.at(tau[:1], p.duration_s)[0])),
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
    modes: tuple[ModeClosure, ...]
    chi_total: Shown
    duration: Shown
    segments: tuple[tuple[Shown, ...], ...]
    """One row per segment: duration, then Omega per (ion, leg), then the leg detunings."""
    beat_phase: Shown


def _values(fn: SampledFn) -> np.ndarray:
    """A tone function's own values: the constant, or its samples."""
    return fn.at(np.linspace(0.0, 1.0, 1 if fn.samples is None else fn.samples.size), 1.0)


def closure(record: Record, gate_id: str) -> ClosureView:
    gate = next((g for g in record.schedule.gates if g.gate_id == gate_id), None)
    if gate is None:
        raise KeyError(f"{gate_id!r} is not an entangling gate of this schedule")
    wf = gate.waveform
    nbar = record.space.nbar
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
                mode_class=record.space.mode_class.get(m, "?"),
            )
        )
    rows: list[tuple[Shown, ...]] = []
    for s in wf.segments or ():
        row = [Shown("waveform_segment", s.duration_s, "duration")]
        for key, fn in sorted(s.amplitude_hz.items()):
            row.append(Shown("tone_envelope", float(np.max(np.abs(_values(fn)))), f"Omega ion,leg {key}"))
        for leg, fn in sorted(s.detuning_hz.items()):
            row.append(Shown("tone_detuning", float(_values(fn)[0]), f"{leg} leg"))
        rows.append(tuple(row))
    return ClosureView(
        modes=tuple(modes),
        chi_total=Shown("entangling_angle", wf.chi_total_rad, "signed sum over modes"),
        duration=Shown("gate_duration", wf.duration_s),
        segments=tuple(rows),
        beat_phase=Shown("beat_phase", wf.phi_m.value, wf.phi_m.status),
    )
