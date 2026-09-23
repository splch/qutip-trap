"""Level 0, the machine: the histogram of the recorded shots with the ideal target beside it, and the shots behind a bar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.shown import Shown

HERALD_NAMES: tuple[tuple[int, str], ...] = ((1, "collision"), (2, "dark or lost ion"), (4, "count anomaly"))
"""The meaning of each bit of a shot's herald flags."""


@dataclass(frozen=True)
class Bar:
    key: str
    """The outcome, qubit 0 rightmost."""
    count: int
    probability: float
    error_bar: float
    target: float
    shots: tuple[int, ...]
    """The recorded shots behind the bar."""

    @property
    def deviation(self) -> float | None:
        """(simulated - target) in error bars; None when the bar has no statistical width."""
        return (self.probability - self.target) / self.error_bar if self.error_bar > 0.0 else None


@dataclass(frozen=True)
class Histogram:
    bars: tuple[Bar, ...]
    n_qubits: int
    shots: Shown
    target_distance: Shown
    largest_deviation: Shown
    discarded_shots: int


def histogram(record: Record) -> Histogram:
    res = record.results
    bits = np.asarray(res.bitstrings)
    # the keys are as long as the measured set: a circuit may measure a subset of its qubits
    n = int(bits.shape[1]) if bits.ndim == 2 else record.n_qubits
    bars: list[Bar] = []
    for key in sorted(set(res.probabilities) | set(res.target_probabilities)):
        row = np.array([int(ch) for ch in key[::-1]], dtype=np.uint8)
        shots = tuple(int(i) for i in np.flatnonzero(np.all(bits == row, axis=1))) if bits.size else ()
        bars.append(
            Bar(
                key=key,
                count=int(res.counts.get(key, 0)),
                probability=float(res.probabilities.get(key, 0.0)),
                error_bar=float(res.error_bars.get(key, 0.0)),
                target=float(res.target_probabilities.get(key, 0.0)),
                shots=shots,
            )
        )
    distance = 0.5 * sum(abs(b.probability - b.target) for b in bars)
    worst = max((abs(d) for b in bars if (d := b.deviation) is not None), default=0.0)
    return Histogram(
        bars=tuple(bars),
        n_qubits=n,
        shots=Shown("Shots", int(bits.shape[0])),
        target_distance=Shown(
            "Distance from the ideal", distance, detail="total variation: half the summed |simulated - ideal|"
        ),
        largest_deviation=Shown(
            "Largest deviation",
            worst,
            "error bars",
            "the bar farthest from its ideal value, in its own error bars",
        ),
        discarded_shots=int(res.discarded_shots),
    )


@dataclass(frozen=True)
class ShotView:
    """One repetition: its bits, the sampled levels behind them, the machine's flags and its dynamical sample."""

    index: int
    bitstring: str
    levels: tuple[int, ...]
    heralds: tuple[str, ...]
    sample_index: int


def shot(record: Record, index: int) -> ShotView:
    res = record.results
    bits = np.asarray(res.bitstrings)
    if not 0 <= index < bits.shape[0]:
        raise IndexError(f"shot {index} outside 0..{bits.shape[0] - 1}")
    flags = int(res.heralds[index])
    return ShotView(
        index=index,
        bitstring="".join(str(int(b)) for b in bits[index][::-1]),
        levels=tuple(int(x) for x in res.levels[index]),
        heralds=tuple(name for bit, name in HERALD_NAMES if flags & bit),
        sample_index=int(res.sample_of_shot[index]),
    )
