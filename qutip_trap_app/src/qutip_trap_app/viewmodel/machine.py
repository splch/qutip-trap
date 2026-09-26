"""Level 0, the machine: the results histogram with the target beside it, the shots behind a bar and the device card
(PLAN.md Section 14.2). Every bar is the count of recorded shots, and the compiler's target distribution is shown beside
the simulated one, never in its place."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap_app.core import Circuit, ideal_probabilities
from qutip_trap_app.record import DeviceCard, Record
from qutip_trap_app.viewmodel.catalogue import Row, Shown

HERALD_NAMES: tuple[tuple[int, str], ...] = ((1, "collision"), (2, "dark or lost ion"), (4, "count anomaly"))
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
    """Indices of the recorded shots behind this bar."""

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
    target_distance: Shown
    largest_deviation: Shown
    bit_order_note: str
    level_note: str


def histogram(record: Record) -> Histogram:
    res = record.results
    keys = sorted(set(res.probabilities) | set(res.target_probabilities))
    bits = np.asarray(res.bitstrings)
    # the keys are as long as the measured set: a subset of the circuit's qubits when the circuit says so
    n = int(bits.shape[1]) if bits.ndim == 2 else record.n_qubits
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
        target_distance=Shown("target_distance", tv),
        largest_deviation=Shown(
            "largest_deviation", worst, "the bar farthest from its target, in its own error bars"
        ),
        bit_order_note="keys read qubit 0 rightmost (the IonQ decimal key is the same integer)",
        level_note=f"simulated at fidelity level {record.diagnostics.level}",
    )


def ideal_outcomes(circuit: Circuit) -> str:
    """The ideal machine's four most likely outcomes, "00: 0.50, 11: 0.50", as the prediction prompt names them."""
    target = ideal_probabilities(circuit)
    return ", ".join(
        f"{k}: {float(v):.2f}" for k, v in sorted(target.items(), key=lambda kv: -float(kv[1]))[:4]
    )


@dataclass(frozen=True)
class ShotView:
    """One repetition opened from its bar: the bits, the sampled levels behind them, the flags, and the photon records
    when the run kept them."""

    bitstring: Shown
    levels: tuple[int, ...]
    heralds: tuple[str, ...]
    sample_index: int
    photon_counts: tuple[Shown, ...] | None
    """Every ion's photon count, ion 0 first, when the run kept its records (the full readout)."""
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
    key = "".join(str(int(b)) for b in bits[index][::-1])
    flags = int(res.heralds[index])
    counts = None
    if ro.photon_records is not None:
        counts = tuple(
            Shown("photon_count", int(c), f"ion {i}") for i, c in enumerate(ro.photon_records[index])
        )
    note = (
        "photon records kept: this shot's counts are the record the discriminator read"
        if counts is not None
        else "fast readout, no photon counts: pick the full simulation and full readout"
    )
    return ShotView(
        bitstring=Shown("bitstring", key),
        levels=tuple(int(x) for x in ro.levels[index]),
        heralds=tuple(name for bit, name in HERALD_NAMES if flags & bit),
        sample_index=int(res.sample_of_shot[index]),
        photon_counts=counts,
        time_used_s=tuple(float(x) for x in np.atleast_1d(ro.time_used_s[index])),
        detection_window=Shown("detection_window", ro.window_s),
        threshold=None if ro.threshold is None else Shown("threshold", ro.threshold),
        note=note,
    )


# ---- the device card ---------------------------------------------------------------------------------------------------


def axis_rows(
    label: str, quantity: str, values: Sequence[float] | None, status: str, detail: str = ""
) -> list[Row]:
    """One row per axis x, y, z (none when the values are absent); ``{ax}`` in ``detail`` names the axis."""
    if values is None:
        return []
    return [
        Row(f"{label} {ax}", Shown(quantity, v, detail.format(ax=ax)), status) for ax, v in zip("xyz", values)
    ]


def card_rows(card: DeviceCard) -> tuple[Row, ...]:
    """The device card's parameter rows: trap frequencies, field, qubit frequencies, heating rates, detector."""
    rows = axis_rows("trap frequency", "secular_frequency", card.trap_omega_hz, "device parameter")
    rows.append(Row("magnetic field", Shown("field", card.field_gauss), "device parameter"))
    for key, value in sorted(card.derived.values.items()):
        if key.startswith("qubit_freq_hz"):
            label = f"qubit frequency {key[len('qubit_freq_hz') :]}"
            rows.append(Row(label, Shown("qubit_frequency", value), "derived"))
    for m, rate in sorted(card.heating_quanta_per_s.items()):
        rows.append(Row(f"heating rate, mode {m}", Shown("heating_rate", rate), "derived"))
    rows.append(
        Row("detector efficiency", Shown("detector_efficiency", card.detector.efficiency), "device parameter")
    )
    rows.append(
        Row("detection window", Shown("detection_window", card.detector.window_s), "device parameter")
    )
    return tuple(rows)


def card_modes(card: DeviceCard) -> tuple[Shown, ...]:
    return tuple(Shown("mode_frequency", m.omega_hz, f"{m.family} {m.family_index}") for m in card.modes)


def spam_rows(card: DeviceCard, status: str) -> list[Row]:
    """Per qubit the two readout errors (with ``status``) and the preparation error."""
    rows: list[Row] = []
    for key in sorted(card.spam):
        eps_b, eps_d = card.spam[key]
        if key.endswith(".state_preparation"):
            label = f"{key.split('.')[0]} preparation error"
            rows.append(Row(label, Shown("prep_error", eps_b), "derived (recipe)"))
        else:
            rows.append(Row(f"{key} bright read as dark", Shown("spam_eps_b", eps_b), status))
            rows.append(Row(f"{key} dark read as bright", Shown("spam_eps_d", eps_d), status))
    return rows


def estimate_rows(card: DeviceCard) -> tuple[Row, ...]:
    return tuple(
        Row(gate, Shown("gate_error_estimate", v), "estimate")
        for gate, v in sorted(card.gate_error_estimates.items())
    )


@dataclass(frozen=True)
class CardView:
    """The device card of Level 0: what a cloud customer sees, every row with its status and chip."""

    species: tuple[str, ...]
    n_ions: int
    native_gates: Shown
    rows: tuple[Row, ...]
    modes: tuple[Shown, ...]
    spam: tuple[Row, ...]
    gate_errors: tuple[Row, ...]
    device_hash: Shown
    seed: Shown
    fidelity_level: Shown
    notes: tuple[str, ...]


def device_card_view(record: Record) -> CardView:
    card = record.device_card
    replay = record.diagnostics.level == "CHANNEL_REPLAY"
    spam_status = "calibrated (table)" if replay else "measured (this run's readout model)"
    tomography = tuple(
        Row(gate, Shown("gate_error_tomography", v), "calibrated")
        for gate, v in sorted(card.gate_error_tomography.items())
    )
    return CardView(
        species=card.species,
        n_ions=card.n_ions,
        native_gates=Shown("native_gate_set", ", ".join(card.native_gates)),
        rows=card_rows(card),
        modes=card_modes(card),
        spam=tuple(spam_rows(card, spam_status)),
        gate_errors=tomography + estimate_rows(card),
        device_hash=Shown("device_hash", record.device_hash),
        seed=Shown("seed", record.job.seed),
        fidelity_level=Shown("fidelity_level", record.diagnostics.level),
        notes=tuple(card.derived.notes),
    )
