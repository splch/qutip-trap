"""Tones, drives and pulses (PLAN.md Sections 3.3, 4.3, 5.2, 7.4; Appendix E).

Conventions (Section 13): a tone's detuning mu is measured from the carrier and delta_{i,m} = mu_i - omega_m is
always two-indexed; the drive term is (hbar/2) sum_tones Omega(t) e^{-i(mu t - phi(t))} sigma_+ (x) prod_m
D_m(i eta) + h.c. (Section 5.7); the effective wavevector is Delta k = k_1 - k_2 with beam 1 the higher-frequency
beam absorbed from the lower qubit level (Wineland 2003 Eq. 2.3, Section 13; the row's 'coupling to the upper level'
wording is ambiguous), |Delta k| = 2k sin(theta_cross/2) for a Raman pair, k for one beam and 0 for a microwave.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.light.beams import Beam

if TYPE_CHECKING:
    from qutip_trap.light.comb import CombSpec

DriveKind = Literal["raman", "optical_E1", "optical_E2", "microwave", "gradient"]

_BEAMS_PER_KIND: dict[str, int] = {
    "raman": 2,
    "optical_E1": 1,
    "optical_E2": 1,
    "microwave": 0,
    "gradient": 0,
}


@dataclass(frozen=True)
class Tone:
    detuning_hz: Callable[[float], float] | float
    """mu(t) from the carrier (ordinary Hz in the public API)."""
    phase_rad: Callable[[float], float] | float
    """phi_tone(t) in the ion frame (Section 5.2)."""
    envelope_hz: Callable[[float], float] | np.ndarray | float
    """Omega(t), the Rabi frequency in the (hbar Omega/2) convention, as an ordinary frequency: a callable of the
    time since the pulse start, an array uniformly sampled over the pulse (cubic spline), or a constant (square)."""
    theta_bessel_rad: float | None = None
    """Kick backend: the Bessel argument of exp[i Theta_B sin(Delta k x + phi) sigma_x]; a perfect
    spin-dependent kick sits at sum_k Theta_{B,k} = pi (Appendix E, Run 5 amendment)."""


@dataclass(frozen=True)
class Drive:
    """A physical drive on a set of ions (Section 3.3)."""

    kind: DriveKind
    ions: tuple[int, ...]
    tones: tuple[Tone, ...]
    beams: tuple[int, ...]
    """Indices into Device.beams: one for optical_E1/E2, two for raman (k_1 - k_2), none for microwave."""
    stark_shift_hz: Callable[[float], float] | float
    crosstalk: dict[int, complex]
    """epsilon_ij onto neighbours: a Rabi (amplitude) ratio, not an intensity ratio (Section 13)."""
    rf_locked: bool = False
    rf_phase_rad: float | None = None
    """Pulse start relative to the trap rf (Section 4.3.6); unlocked = averaged."""
    comb: CombSpec | None = None
    """Set for a mode-locked Raman drive; then ``tones`` comes from comb.tones() and stark_shift_hz from comb.stark4_hz()."""

    def __post_init__(self) -> None:
        if not self.ions:
            raise ValueError("a Drive addresses at least one ion")
        expected = _BEAMS_PER_KIND[self.kind]
        if len(self.beams) != expected:
            raise ValueError(f"a {self.kind} drive references {expected} beam(s), got {len(self.beams)}")
        if not self.rf_locked and self.rf_phase_rad is not None:
            raise ValueError("rf_phase_rad is only meaningful for an rf-locked pulse (unlocked = averaged)")
        if self.comb is not None and self.kind != "raman":
            raise ValueError("a frequency comb generates a Raman drive")

    def delta_k(self, beams: Sequence[Beam]) -> np.ndarray:
        """The effective wavevector, DERIVED from the beams' wavelengths and directions, never a free field.

        Raman: k_1 - k_2 with beam 1 the higher-frequency beam absorbed from the lower qubit level (Section 13, "Effective wavevector");
        single-photon optical: k k_hat; microwave and gradient: 0. Appendix E declares this as a property; it
        takes the device's beam list here because a ``Drive`` stores indices into ``Device.beams``.
        """
        if self.kind == "raman":
            k1 = beams[self.beams[0]].k_vector()
            k2 = beams[self.beams[1]].k_vector()
            return np.asarray(k1 - k2, dtype=float)
        if self.kind in ("optical_E1", "optical_E2"):
            return np.asarray(beams[self.beams[0]].k_vector(), dtype=float)
        return np.zeros(3)


@dataclass(frozen=True)
class Pulse:
    """A ``Drive`` with start and end times and metadata linking it to the gate it implements."""

    drive: Drive
    t_start_s: float
    t_end_s: float
    gate_id: str | None
    closes_modes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.t_end_s <= self.t_start_s:
            raise ValueError("a pulse must end after it starts")

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s


__all__ = ["Drive", "DriveKind", "Pulse", "Tone"]
