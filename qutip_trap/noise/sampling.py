"""Quasi-static parameter sampling per shot and OU processes for fast noise (PLAN.md Sections 5.5, 6.1; M7).

The ``NoiseSample`` record exists from M0; M2 fixes the KEYS the Hamiltonian builder and the engine read from
``values`` (Section 5.7, "Stochastic elements, sampled once per dynamical sample"), so that M7's sampler and the
builder agree on names. A missing key means the nominal value (no offset, unit scale, n = drawn from the seeds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEY_RABI_SCALE = "rabi_scale"
"""Multiplies every drive's Rabi frequency (a per-sample scale factor, Section 6.4)."""
KEY_RF_PHASE = "rf_phase_rad"
"""The trap rf phase at t = 0 for an UNLOCKED drive's micromotion modulation (Section 4.3.6); absent = J_0 average."""


def key_qubit_offset_hz(ion: int) -> str:
    """The true transition of ``ion`` minus the frame frequency, in Hz (H_int of Section 5.7)."""
    return f"qubit_offset_hz[{ion}]"


def key_mode_offset_hz(mode: int) -> str:
    """Mode-frequency offset of ``mode`` in Hz (Section 6.2 quasi-static part)."""
    return f"mode_offset_hz[{mode}]"


def key_frozen_n(mode: int) -> str:
    """The Fock state of a frozen spectator for this shot (Section 5.2)."""
    return f"frozen_n[{mode}]"


def key_beam_phase_rad(beam: int) -> str:
    return f"beam_phase_rad[{beam}]"


@dataclass(frozen=True)
class NoiseSample:
    """One draw of every quasi-static parameter (Section 6.1)."""

    sample_id: int
    values: dict[str, float]
    """Field offset, mode offsets, Rabi scale, beam phases, ..."""
    ou_grids: dict[str, np.ndarray]
    """Fixed-grid realizations of the fast processes (Section 5.5)."""

    def get(self, key: str, default: float) -> float:
        return float(self.values.get(key, default))


def quiet_sample(sample_id: int = 0) -> NoiseSample:
    """The nominal sample: no offsets, unit scales, nothing sampled (what M2's tests run under)."""
    return NoiseSample(sample_id=sample_id, values={}, ou_grids={})


__all__ = [
    "KEY_RABI_SCALE",
    "KEY_RF_PHASE",
    "NoiseSample",
    "key_beam_phase_rad",
    "key_frozen_n",
    "key_mode_offset_hz",
    "key_qubit_offset_hz",
    "quiet_sample",
]
