"""Results, diagnostics and the persistent machine state (PLAN.md Sections 3.4, 6.7, 8.6; Appendix E).

Bit order (Section 13, row "Result bit order"): qubit 0 is the LEAST-significant bit of the decimal histogram
key; ``Result.bit_order`` says so explicitly and the exporter converts. A bitstring key such as "101" is read
right to left, qubit 0 rightmost, so "101" on three qubits is qubit0 = 1, qubit1 = 0, qubit2 = 1 and the IonQ
key "5". PennyLane reverses to big-endian on its side; Qiskit does not (Section 8.6).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from qutip import Qobj

    from qutip_trap.control.table import CalibrationTable
    from qutip_trap.device.model import Device
    from qutip_trap.hilbert.space import HilbertSpace
    from qutip_trap.noise.sampling import NoiseSample

M6 = "milestone M6 (run/, PLAN.md Section 3.4)"


def bitstring_key(bits: np.ndarray) -> str:
    """The histogram key of one shot: qubit 0 rightmost (least significant)."""
    b = np.asarray(bits).astype(int)
    return "".join(str(int(x)) for x in b[::-1])


def decimal_key(bits: np.ndarray) -> str:
    """The IonQ decimal key of one shot: the integer sum_j b_j 2^j with qubit 0 least significant."""
    b = np.asarray(bits).astype(int)
    return str(int(sum(int(x) << j for j, x in enumerate(b))))


def bits_from_decimal(key: str, n_qubits: int) -> np.ndarray:
    """Inverse of :func:`decimal_key`: qubit j is bit j of the integer."""
    value = int(key)
    if value < 0 or value >= 2**n_qubits:
        raise ValueError(f"decimal key {key!r} does not fit {n_qubits} qubits")
    return np.array([(value >> j) & 1 for j in range(n_qubits)], dtype=np.uint8)


def aggregate(bitstrings: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    """Counts and probabilities keyed by :func:`bitstring_key`."""
    arr = np.asarray(bitstrings)
    if arr.ndim != 2:
        raise ValueError("bitstrings must be a (shots, n_qubits) array")
    counts = Counter(bitstring_key(row) for row in arr)
    n = arr.shape[0]
    probabilities = {k: v / n for k, v in counts.items()} if n else {}
    return dict(counts), probabilities


@dataclass(frozen=True)
class RunState:
    """Machine state that PERSISTS across shots (Sections 6.7, 8.5)."""

    order: tuple[int, ...]
    """Physical position -> qubit label; a reorder permutes it for every later shot."""
    dark: frozenset[int]
    lost: frozenset[int]
    events: tuple[tuple[int, str], ...]
    """(shot index, event) log."""

    def recrystallize(self) -> RunState:
        raise NotImplementedError(f"RunState.recrystallize is {M6}")

    def reload(self, device: Device) -> RunState:
        raise NotImplementedError(f"RunState.reload is {M6}")


@dataclass(frozen=True)
class Diagnostics:
    level: Literal["JOINT_EXACT", "GATE_LOCAL"]
    """The level actually run."""
    space: HilbertSpace
    mode_class: dict[int, Literal["resolved", "frozen", "dropped"]]
    run_state: RunState
    wall_clock_span_s: float
    """t0 to the last shot: shot k is evaluated at t0 + k T_rep (Section 7.5)."""
    boundary_population: dict[int, float]
    margin_levels: dict[int, int]
    dropped_modes: tuple[int, ...]
    frozen_contribution: dict[int, tuple[float, float]]
    """(|alpha|^2 (2n + 1), chi) per frozen mode."""
    integrator: str
    tolerances: tuple[float, float]
    samples: int
    trajectories: int
    shots_per_sample: int
    effective_sample_size: float
    root_seed: int
    calibration: CalibrationTable
    approximations: tuple[str, ...]


@dataclass(frozen=True)
class Result:
    bitstrings: np.ndarray
    """(shots, n_qubits) array of 0/1, column j = qubit j."""
    bit_order: Literal["qubit0_lsb"]
    counts: dict[str, int]
    probabilities: dict[str, float]
    error_bars: dict[str, float]
    photon_records: np.ndarray | None
    posteriors: np.ndarray | None
    noise_samples: tuple[NoiseSample, ...]
    heralds: np.ndarray
    """Per-shot flags (collision, all-dark, count anomaly); Section 6.7."""
    discarded_shots: int
    run_state: RunState
    spam: dict[str, tuple[float, float]]
    """Per qubit (eps_B, eps_D) with the definition used (Section 13, "Readout figure of merit")."""
    final_state: Qobj | None
    diagnostics: Diagnostics

    def __post_init__(self) -> None:
        arr = np.asarray(self.bitstrings)
        if arr.ndim != 2:
            raise ValueError("bitstrings must be a (shots, n_qubits) array")
        if self.bit_order != "qubit0_lsb":
            raise ValueError("the only supported bit order is qubit0_lsb (Section 13)")
        counts, probabilities = aggregate(arr)
        if counts != self.counts:
            raise ValueError("counts must aggregate the bitstrings with qubit 0 as the least-significant bit")
        if set(probabilities) != set(self.probabilities) or any(
            abs(probabilities[k] - self.probabilities[k]) > 1e-12 for k in probabilities
        ):
            raise ValueError("probabilities must be counts / shots")
        if len(self.heralds) != arr.shape[0]:
            raise ValueError("one herald flag per shot")

    @property
    def n_qubits(self) -> int:
        return int(np.asarray(self.bitstrings).shape[1])

    @property
    def shots(self) -> int:
        return int(np.asarray(self.bitstrings).shape[0])

    def to_ionq_json(self) -> dict[str, float]:
        """The legacy IonQ probability result: decimal-integer keys, qubit 0 least significant (Section 8.6)."""
        return {str(int(key, 2)): p for key, p in self.probabilities.items()}

    def to_ionq_histogram(self) -> dict[str, int]:
        """The histogram variant: shot counts with the same decimal keys."""
        return {str(int(key, 2)): c for key, c in self.counts.items()}

    def to_ionq_shots(self) -> list[str]:
        """The per-shot format: an ordered list of decimal strings ("6", "1", "0", "7" is 110, 001, 000, 111 on three qubits)."""
        return [decimal_key(row) for row in np.asarray(self.bitstrings)]


__all__ = [
    "Diagnostics",
    "Result",
    "RunState",
    "aggregate",
    "bits_from_decimal",
    "bitstring_key",
    "decimal_key",
]
