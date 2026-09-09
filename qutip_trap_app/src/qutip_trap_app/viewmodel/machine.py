"""Level 0, the machine: the device card, the results histogram with the target beside it, and the shots behind a bar
(PLAN.md Section 14.2, row 0; Section 14.5 "Target beside simulated").

Every bar is the count of recorded shots (Section 14.1 rule 1), every number is a :class:`Shown` with its catalogue id,
and the compiler's target distribution is shown beside the simulated one and never in its place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.catalogue import Shown

HERALD_NAMES: tuple[tuple[int, str], ...] = (
    (1, "collision"),
    (2, "dark or lost ion"),
    (4, "count anomaly"),
)
"""``Result.heralds`` bit meanings (Section 8.6)."""


@dataclass(frozen=True)
class Bar:
    key: str
    """Bitstring, qubit 0 rightmost (conv.result_bit_order)."""
    count: Shown
    probability: Shown
    error_bar: Shown
    target_probability: Shown
    shots: tuple[int, ...]
    """Indices of the recorded shots behind this bar (the zoom-in target of Section 14.2)."""

    @property
    def deviation_in_error_bars(self) -> float | None:
        """(simulated - target)/error bar; None when the bar has no statistical width."""
        eb = float(self.error_bar.value or 0.0)
        if eb <= 0.0:
            return None
        return (float(self.probability.value or 0.0) - float(self.target_probability.value or 0.0)) / eb


@dataclass(frozen=True)
class Histogram:
    bars: tuple[Bar, ...]
    shots: Shown
    n_qubits: int
    total_variation_to_target: float
    """(1/2) sum_x |p_x - t_x|: how far the simulated distribution sits from the compiler's target."""
    largest_deviation_in_error_bars: float
    target_distance: Shown
    largest_deviation: Shown
    discarded_shots: int
    bit_order_note: str
    level_note: str


def histogram(record: Record) -> Histogram:
    res = record.results
    keys = sorted(set(res.probabilities) | set(res.target_probabilities))
    bits = np.asarray(res.bitstrings)
    n = record.n_qubits
    bars: list[Bar] = []
    tv = 0.0
    worst = 0.0
    for key in keys:
        p = float(res.probabilities.get(key, 0.0))
        t = float(res.target_probabilities.get(key, 0.0))
        tv += 0.5 * abs(p - t)
        row = np.array([int(ch) for ch in key[::-1]], dtype=np.uint8)  # qubit 0 is the rightmost character
        idx = tuple(int(i) for i in np.flatnonzero(np.all(bits == row, axis=1))) if bits.size else ()
        bar = Bar(
            key=key,
            count=Shown("count", int(res.counts.get(key, 0))),
            probability=Shown("probability", p),
            error_bar=Shown("error_bar", float(res.error_bars.get(key, 0.0))),
            target_probability=Shown("target_probability", t),
            shots=idx,
        )
        dev = bar.deviation_in_error_bars
        if dev is not None:
            worst = max(worst, abs(dev))
        bars.append(bar)
    return Histogram(
        bars=tuple(bars),
        shots=Shown("shots", int(bits.shape[0])),
        n_qubits=n,
        total_variation_to_target=tv,
        largest_deviation_in_error_bars=worst,
        target_distance=Shown("target_distance", tv),
        largest_deviation=Shown(
            "largest_deviation", worst, "the bar farthest from its target, in its own error bars"
        ),
        discarded_shots=res.discarded_shots,
        bit_order_note="keys read qubit 0 rightmost (the IonQ decimal key is the same integer)",
        level_note=f"simulated at fidelity level {record.diagnostics.level}",
    )


@dataclass(frozen=True)
class ShotView:
    """One repetition opened from its bar: the bits, the sampled levels behind them, the flags, and the photon records
    when the run kept them (the ``readout="full"`` path)."""

    index: int
    bitstring: Shown
    levels: tuple[int, ...]
    heralds: tuple[str, ...]
    sample_index: int
    photon_counts: tuple[Shown, ...] | None
    posteriors: tuple[float, ...] | None
    time_used_s: tuple[float, ...]
    """Detection time the discriminator used, per ion (the adaptive discriminators stop early)."""
    detection_window: Shown
    threshold: Shown | None
    note: str


def shot(record: Record, index: int) -> ShotView:
    res = record.results
    ro = record.readout
    bits = np.asarray(res.bitstrings)
    if not 0 <= index < bits.shape[0]:
        raise IndexError(f"shot {index} outside 0..{bits.shape[0] - 1}")
    row = bits[index]
    key = "".join(str(int(b)) for b in row[::-1])
    flags = int(res.heralds[index])
    heralds = tuple(name for bit, name in HERALD_NAMES if flags & bit)
    counts = None
    if ro.photon_records is not None:
        counts = tuple(
            Shown("photon_count", int(c), f"ion {i}") for i, c in enumerate(ro.photon_records[index])
        )
    post = None if ro.posteriors is None else tuple(float(x) for x in ro.posteriors[index])
    note = (
        "photon records kept: this shot's counts are the record the discriminator read"
        if counts is not None
        else "fast readout path: the bits were drawn from the POVM (Section 5.7); run with readout='full' to keep photon records"
    )
    return ShotView(
        index=index,
        bitstring=Shown("bitstring", key),
        levels=tuple(int(x) for x in ro.levels[index]),
        heralds=heralds,
        sample_index=int(res.sample_of_shot[index]),
        photon_counts=counts,
        posteriors=post,
        time_used_s=tuple(float(x) for x in np.atleast_1d(ro.time_used_s[index])),
        detection_window=Shown("detection_window", ro.window_s),
        threshold=None if ro.threshold is None else Shown("threshold", ro.threshold),
        note=note,
    )


@dataclass(frozen=True)
class CardRow:
    label: str
    value: Shown
    status: str
    """``measured`` (from the run), ``calibrated`` or ``seed`` (from the table), ``estimate`` (closed form), ``derived``."""


@dataclass(frozen=True)
class CardView:
    """The device card of Level 0: what a cloud customer sees, every row with its status and chip."""

    species: tuple[str, ...]
    n_ions: int
    native_gates: Shown
    rows: tuple[CardRow, ...]
    modes: tuple[Shown, ...]
    spam: tuple[CardRow, ...]
    gate_errors: tuple[CardRow, ...]
    device_hash: Shown
    seed: Shown
    fidelity_level: Shown
    notes: tuple[str, ...]


def device_card_view(record: Record) -> CardView:
    card = record.device_card
    rows: list[CardRow] = []
    if card.trap_omega_hz is not None:
        for axis, w in zip(("x", "y", "z"), card.trap_omega_hz):
            rows.append(CardRow(f"trap frequency {axis}", Shown("secular_frequency", w), "device parameter"))
    rows.append(CardRow("magnetic field", Shown("field", card.field_gauss), "device parameter"))
    for key, value in sorted(card.derived.values.items()):
        if key.startswith("qubit_freq_hz"):
            rows.append(
                CardRow(
                    f"qubit frequency {key[len('qubit_freq_hz') :]}",
                    Shown("qubit_frequency", value),
                    "derived",
                )
            )
    for m, rate in sorted(card.heating_quanta_per_s.items()):
        rows.append(CardRow(f"heating rate, mode {m}", Shown("heating_rate", rate), "derived"))
    rows.append(
        CardRow(
            "detector efficiency", Shown("detector_efficiency", card.detector.efficiency), "device parameter"
        )
    )
    rows.append(
        CardRow("detection window", Shown("detection_window", card.detector.window_s), "device parameter")
    )
    modes = tuple(Shown("mode_frequency", m.omega_hz, f"{m.family} {m.family_index}") for m in card.modes)
    spam: list[CardRow] = []
    spam_status = (
        "calibrated (table)"
        if record.diagnostics.level == "CHANNEL_REPLAY"
        else "measured (this run's readout model)"
    )
    for key in sorted(card.spam):
        eps_b, eps_d = card.spam[key]
        if key.endswith(".state_preparation"):
            spam.append(
                CardRow(
                    f"{key.split('.')[0]} preparation error", Shown("prep_error", eps_b), "derived (recipe)"
                )
            )
        else:
            spam.append(CardRow(f"{key} bright read as dark", Shown("spam_eps_b", eps_b), spam_status))
            spam.append(CardRow(f"{key} dark read as bright", Shown("spam_eps_d", eps_d), spam_status))
    errors: list[CardRow] = []
    for gate, value in sorted(card.gate_error_tomography.items()):
        errors.append(CardRow(gate, Shown("gate_error_tomography", value), "calibrated"))
    for gate, value in sorted(card.gate_error_estimates.items()):
        errors.append(CardRow(gate, Shown("gate_error_estimate", value), "estimate"))
    notes = tuple(card.derived.notes) + (
        "estimate = a closed-form error scale of Section 9.6; calibrated = process tomography of the simulated pulse (Section 6.8)",
    )
    return CardView(
        species=card.species,
        n_ions=card.n_ions,
        native_gates=Shown("native_gate_set", ", ".join(card.native_gates)),
        rows=tuple(rows),
        modes=modes,
        spam=tuple(spam),
        gate_errors=tuple(errors),
        device_hash=Shown("device_hash", record.device_hash),
        seed=Shown("seed", record.job.seed),
        fidelity_level=Shown("fidelity_level", record.diagnostics.level),
        notes=notes,
    )


__all__ = [
    "HERALD_NAMES",
    "Bar",
    "CardRow",
    "CardView",
    "Histogram",
    "ShotView",
    "device_card_view",
    "histogram",
    "shot",
]
