"""The calibration table and its entries (PLAN.md Sections 3.2, 7.5; Appendix E).

``CalibrationTable`` is plain data: ``control.schedule`` reads it and never writes it, and ``control`` never
imports ``calibration`` (Appendix E). Every entry carries its status, the experiment that produced it, its
provenance id, its age and the noise sample it was fitted under (Section 7.5).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qutip_trap.control.shaping import GateModes, Kernel

Leg = Literal["red", "blue"]
WaveformKind = Literal["ms", "light_shift"]
"""``ms``: bichromatic red/blue legs on the spin flip (Section 4.4.1); ``light_shift``: one beat note ("blue") on the
state-dependent light shift, sigma_z sigma_z (Section 4.4.4)."""


@dataclass(frozen=True)
class CalEntry:
    value: float
    uncertainty: float
    status: Literal["seed", "calibrated", "uncalibrated"]
    experiment: str
    provenance_id: str
    fitted_at_s: float
    """Calibration age: the laboratory time the fit was made at (Section 7.5)."""
    sample_id: int
    """The noise sample the entry was fitted under."""

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")


@dataclass(frozen=True)
class Segment:
    """One segment of a calibrated entangling pulse (Appendix E, ``Waveform.segments``).

    Amplitudes and detunings are constants (the segmented AM family) or callables of the time since the SEGMENT start
    (a Fourier-sine amplitude, an FM detuning schedule), the same two forms ``Tone`` accepts; the scheduler plays each
    segment as one ``Pulse`` per ion so that step discontinuities fall on integration boundaries (Section 7.4).
    """

    duration_s: float
    amplitude_hz: dict[tuple[int, Leg], float | Callable[[float], float]]
    """Omega per (ion, leg), leg in {red, blue}; per-ion imbalance is a CalibrationTable entry."""
    phase_rad: dict[tuple[int, Leg], float]
    detuning_hz: dict[Leg, float | Callable[[float], float]]

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("segment duration must be positive")
        if set(self.amplitude_hz) != set(self.phase_rad):
            raise ValueError("amplitude_hz and phase_rad must be indexed by the same (ion, leg) pairs")
        legs = {leg for _ion, leg in self.amplitude_hz}
        if not legs <= set(self.detuning_hz):
            raise ValueError("every leg with an amplitude needs a detuning")
        for key, v in self.amplitude_hz.items():
            if not callable(v) and float(v) < 0.0:
                raise ValueError(f"amplitude of {key} is a magnitude; a reversed force is a phase of pi")

    @property
    def ions(self) -> tuple[int, ...]:
        return tuple(sorted({ion for ion, _leg in self.amplitude_hz}))

    @property
    def legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in ("red", "blue") if any(k[1] == leg for k in self.amplitude_hz))


@dataclass(frozen=True)
class Waveform:
    """A calibrated entangling pulse: what the scheduler plays (Sections 4.4.3, 7.4)."""

    segments: tuple[Segment, ...] | None
    fourier: tuple[complex, ...] | None
    """Or Fourier coefficients of the modulation (Section 4.4.3)."""
    duration_s: float
    phi_s: CalEntry
    phi_m: CalEntry
    chi_m: dict[int, float]
    """Per-mode entangling angle at closure, SIGNED: the pulse applies exp(+i sum_m chi_m sigma sigma) (Section 13)."""
    alpha_m: dict[int, complex]
    """Per-mode residual displacement at closure (the worst gate ion's)."""
    kind: WaveformKind = "ms"

    def __post_init__(self) -> None:
        if (self.segments is None) == (self.fourier is None):
            raise ValueError("a Waveform is either segmented or Fourier-parameterized, not both or neither")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.segments is not None:
            total = sum(s.duration_s for s in self.segments)
            if not math.isclose(total, self.duration_s, rel_tol=1e-9, abs_tol=1e-15):
                raise ValueError(f"segment durations sum to {total}, not duration_s = {self.duration_s}")
            ions = {s.ions for s in self.segments}
            if len(ions) != 1:
                raise ValueError("every segment addresses the same ions")
            for s in self.segments:
                if self.kind == "light_shift" and s.legs != ("blue",):
                    raise ValueError("a light-shift waveform has one leg, 'blue' (the beat note itself)")
                if self.kind == "ms" and s.legs != ("red", "blue"):
                    raise ValueError("an MS waveform has a red and a blue leg on every ion")

    @property
    def ions(self) -> tuple[int, ...]:
        if self.segments is None:
            raise ValueError("a Fourier-parameterized waveform names no ions")
        return self.segments[0].ions

    @property
    def chi_total_rad(self) -> float:
        """The signed two-body angle: exp(+i chi sigma sigma) = XX(-chi) (Section 13)."""
        return float(sum(self.chi_m.values()))

    @classmethod
    def symmetric(
        cls,
        modes: GateModes,
        *,
        gate_mode: int,
        loops: int = 1,
        epsilon_hz: float | None = None,
        duration_s: float | None = None,
        chi_target_rad: float = math.pi / 4.0,
        kernel: Kernel = "choi",
        pair: tuple[int, int] | None = None,
        phi_s_rad: float = 0.0,
        phi_m_rad: float = 0.0,
        imbalance: float = 0.0,
        detuning_side: Literal["inside", "outside"] = "inside",
        all_modes: bool = True,
        kind: WaveformKind = "ms",
    ) -> Waveform:
        """The equal-envelope symmetric-detuning shortcut (Appendix E): one square segment, tones at -/+ (omega_g -/+ eps) closing
        ``gate_mode`` after ``loops`` loops (tau = 2 pi K/eps), the amplitude from |chi| = chi_target over the pair's modes; for
        one mode eta Omega/eps = 1/(2 sqrt K) exactly (Section 4.4.1). Equals the general solver at equal envelopes (Section 9.17)."""
        from qutip_trap.control.shaping import symmetric_pulse

        return symmetric_pulse(
            modes,
            gate_mode=gate_mode,
            loops=loops,
            epsilon_hz=epsilon_hz,
            duration_s=duration_s,
            chi_target_rad=chi_target_rad,
            kernel=kernel,
            pair=pair,
            phi_s_rad=phi_s_rad,
            phi_m_rad=phi_m_rad,
            imbalance=imbalance,
            detuning_side=detuning_side,
            all_modes=all_modes,
            kind=kind,
        ).waveform


@dataclass(frozen=True)
class CalibrationTable:
    device_hash: str
    seed: int
    surrogate: bool
    """Closed-form surrogate with spot checks, or full simulated experiments (Section 7.5)."""
    qubit_freq: dict[int, CalEntry]
    """From the Ramsey-frequency experiment; the scheduler never reads the true value."""
    rabi: dict[tuple[int, int], CalEntry]
    """(ion, beam)."""
    stark: dict[tuple[int, int], CalEntry]
    crosstalk: dict[tuple[int, int], CalEntry]
    modes: dict[int, CalEntry]
    nbar: dict[int, CalEntry]
    ms: dict[tuple[int, int], Waveform]
    """Entangling waveforms per pair; looked up as (a, b) or (b, a) (``waveform_for``)."""
    field: CalEntry
    micromotion: dict[str, CalEntry]
    """Shim voltages and residual beta per beam direction."""
    detection: dict[str, CalEntry]
    heating: dict[int, CalEntry]

    def waveform_for(self, pair: Sequence[int]) -> Waveform | None:
        """The pair's entangling waveform under either key order, None when uncalibrated."""
        a, b = int(pair[0]), int(pair[1])
        return self.ms.get((a, b)) or self.ms.get((b, a))


__all__ = ["CalEntry", "CalibrationTable", "Leg", "Segment", "Waveform", "WaveformKind"]
