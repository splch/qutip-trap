"""Beam geometry (PLAN.md Sections 3.3, 4.2.4, 4.5.3; Appendix E; milestone M2 for the physics).

A ``Beam`` carries wavelength, propagation direction, polarization, waist, power and pointing; the per-ion
intensity through the Gaussian profile is where addressing crosstalk originates (Section 3.3). Polarization
is decomposed about B into the spherical basis of Section 13 (row "Polarization components") by the atomic
layer of M0a, never stored per sublevel here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

M2 = "milestone M2 (light/, PLAN.md Section 4.3)"
M3 = "milestone M3 (prep/, PLAN.md Section 4.2.4)"


def _unit(v: tuple[float, float, float], what: str) -> None:
    norm = math.sqrt(sum(x * x for x in v))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{what} must be a unit vector, got norm {norm}")


@dataclass(frozen=True)
class PolarizationModulation:
    """AOM, PEM or EOM polarization modulation for dark-state destabilization (Section 8.1).

    Its presence makes ``steadystate`` illegal and forces the propagate-then-period-average path
    (Section 13, row "Polarization-modulation mappings").
    """

    kind: Literal["aom", "pem", "eom"]
    frequency_hz: float
    depth_rad: float

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("modulation frequency must be positive")


@dataclass(frozen=True)
class Beam:
    wavelength_m: float
    """VACUUM wavelength."""
    k_hat: tuple[float, float, float]
    polarization: tuple[complex, complex, complex]
    """Laboratory-frame Jones vector, unit norm, orthogonal to k_hat."""
    waist_m: float
    """1/e^2 intensity radius w0; the peak on-axis intensity is 2P/(pi w0^2) (Section 13, comb row)."""
    power_w: float
    pointing_m: tuple[float, float, float]
    """A point on the beam axis."""
    polarization_amplitudes: tuple[complex, complex, complex] | None = None
    """(sigma-, pi, sigma+) spherical components about B, unit-normalized (Appendix E, Run 5 amendment)."""
    modulation: PolarizationModulation | None = None

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0.0 or self.waist_m <= 0.0 or self.power_w < 0.0:
            raise ValueError("wavelength and waist must be positive, power non-negative")
        _unit(self.k_hat, "Beam.k_hat")
        pol = np.asarray(self.polarization, dtype=complex)
        if not math.isclose(float(np.vdot(pol, pol).real), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Beam.polarization must be unit-normalized")
        k = np.asarray(self.k_hat, dtype=float)
        if abs(complex(np.dot(k, pol))) > 1e-9:
            raise ValueError("Beam.polarization must be transverse to k_hat")
        if self.polarization_amplitudes is not None:
            amp = np.asarray(self.polarization_amplitudes, dtype=complex)
            if not math.isclose(float(np.vdot(amp, amp).real), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "polarization_amplitudes must be unit-normalized (the Wineland reductions require it)"
                )

    @property
    def k_rad_per_m(self) -> float:
        """|k| = 2 pi / lambda_vac."""
        return 2.0 * math.pi / self.wavelength_m

    def k_vector(self) -> np.ndarray:
        return self.k_rad_per_m * np.asarray(self.k_hat, dtype=float)

    def intensity_at(self, position_m: np.ndarray) -> float:
        """Gaussian-beam intensity I = (2P/(pi w0^2)) exp(-2 r_perp^2/w0^2) at the beam waist plane (W/m^2).

        The Rayleigh-range variation of the waist along the beam is neglected (ions sit near the focus);
        M2 may refine this with the configured focal position.
        """
        r = np.asarray(position_m, dtype=float) - np.asarray(self.pointing_m, dtype=float)
        k = np.asarray(self.k_hat, dtype=float)
        r_perp = r - np.dot(r, k) * k
        peak = 2.0 * self.power_w / (math.pi * self.waist_m**2)
        return float(peak * math.exp(-2.0 * float(np.dot(r_perp, r_perp)) / self.waist_m**2))


@dataclass(frozen=True)
class PolGradientBeams:
    """A lin-perp-lin polarization-gradient pair, not two independent Beams (Section 4.2.4)."""

    beam_a: Beam
    beam_b: Beam
    detuning_hz: float
    """Must be > 0 (blue) on an inverted j_e <= j_g line; asserted, not warned."""
    beat_hz: float = 0.0
    phase_rad: float = 0.0
    level_scheme: Literal["jg12_je12", "F1_to_F0"] = "jg12_je12"

    def __post_init__(self) -> None:
        if self.detuning_hz <= 0.0:
            raise ValueError(
                "polarization-gradient cooling needs blue detuning, detuning_hz > 0 (Section 4.2.4)"
            )
        ka = np.asarray(self.beam_a.k_hat)
        kb = np.asarray(self.beam_b.k_hat)
        if not np.allclose(ka, -kb, atol=1e-9):
            raise ValueError("the lin-perp-lin pair must counter-propagate")
        pa = np.asarray(self.beam_a.polarization, dtype=complex)
        pb = np.asarray(self.beam_b.polarization, dtype=complex)
        if abs(complex(np.vdot(pa, pb))) > 1e-9:
            raise ValueError("the lin-perp-lin pair must have orthogonal polarizations")

    def xi(self, mode_freq_hz: float, s_single_beam: float) -> float:
        """Delta*s/(3*omega), angular: the light-shift modulation AMPLITUDE over the mode frequency (Section 13)."""
        raise NotImplementedError(f"PolGradientBeams.xi is {M3}")

    def limits(self, mode_freq_hz: float, s_single_beam: float) -> tuple[float, float]:
        """(fixed-phase <n_0>, phase-averaged <n>); raises for level_scheme != "jg12_je12"."""
        raise NotImplementedError(f"PolGradientBeams.limits is {M3}")

    def moving_gradient_ok(self, cooling_rate_hz: float, mode_freq_hz: float) -> bool:
        """W < delta < omega."""
        raise NotImplementedError(f"PolGradientBeams.moving_gradient_ok is {M3}")


__all__ = ["Beam", "PolGradientBeams", "PolarizationModulation"]
