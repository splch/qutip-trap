"""The control hardware chain: DDS, AOM, amplifier (PLAN.md Section 7.10; Appendix E; milestone M7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareChain:
    """DDS/AOM/amplifier response: quantization, rise time, bandwidth, dead time, phase continuity (Section 7.10)."""

    dds_phase_bits: int
    dds_amplitude_bits: int
    aom_rise_s: float
    amplifier_bandwidth_hz: float
    dead_time_s: float
    phase_continuous: bool

    def __post_init__(self) -> None:
        if self.dds_phase_bits <= 0 or self.dds_amplitude_bits <= 0:
            raise ValueError("DDS bit depths must be positive")
        if self.aom_rise_s < 0.0 or self.amplifier_bandwidth_hz <= 0.0 or self.dead_time_s < 0.0:
            raise ValueError("rise time and dead time are non-negative, bandwidth positive")


__all__ = ["HardwareChain"]
