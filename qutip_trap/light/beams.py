"""Laser beam geometry: wavelength, direction, polarization, waist, power and pointing.

The per-ion intensity through the Gaussian profile is where addressing crosstalk originates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

M2 = "milestone M2 (light/, PLAN.md Section 4.3)"


def _unit(v: tuple[float, float, float], what: str) -> None:
    norm = math.sqrt(sum(x * x for x in v))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{what} must be a unit vector, got norm {norm}")


@dataclass(frozen=True)
class PolarizationModulation:
    """AOM, PEM or EOM polarization modulation for dark-state destabilization (it makes the Liouvillian time periodic)."""

    kind: Literal["aom", "pem", "eom"]
    frequency_hz: float
    depth_rad: float

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("modulation frequency must be positive")


@dataclass(frozen=True)
class Beam:
    """One laser beam at the ions: vacuum wavelength, unit propagation direction, Jones vector, waist, power, pointing."""

    wavelength_m: float
    """Vacuum wavelength."""
    k_hat: tuple[float, float, float]
    polarization: tuple[complex, complex, complex]
    """Laboratory-frame Jones vector, unit norm, orthogonal to k_hat."""
    waist_m: float
    """1/e^2 intensity radius w0."""
    power_w: float
    pointing_m: tuple[float, float, float]
    """A point on the beam axis."""
    polarization_amplitudes: tuple[complex, complex, complex] | None = None
    """(sigma-, pi, sigma+) components about B, unit-normalized; when set, used instead of decomposing ``polarization``."""
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
        """Gaussian-beam intensity I = (2P/(pi w0^2)) exp(-2 r_perp^2/w0^2) in W/m^2, neglecting the Rayleigh range."""
        r = np.asarray(position_m, dtype=float) - np.asarray(self.pointing_m, dtype=float)
        k = np.asarray(self.k_hat, dtype=float)
        r_perp = r - np.dot(r, k) * k
        peak = 2.0 * self.power_w / (math.pi * self.waist_m**2)
        return float(peak * math.exp(-2.0 * float(np.dot(r_perp, r_perp)) / self.waist_m**2))


@dataclass(frozen=True)
class PolGradientBeams:
    """A counter-propagating lin-perp-lin polarization-gradient pair, not two independent Beams."""

    beam_a: Beam
    beam_b: Beam
    detuning_hz: float
    """Must be > 0 (blue) on an inverted j_e <= j_g line."""
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
        """xi = Delta s / (3 omega): the light-shift modulation amplitude over the mode frequency, with ``s`` the
        single-beam saturation parameter on the S1/2-P3/2 stretched transition (Joshi et al. 2020)."""
        from qutip_trap.prep.polarization_gradient import xi_depth

        return xi_depth(2.0 * math.pi * self.detuning_hz, s_single_beam, 2.0 * math.pi * mode_freq_hz)

    def limits(self, mode_freq_hz: float, s_single_beam: float) -> tuple[float, float]:
        """(fixed-phase <n_0> at this pair's phase, phase-averaged <n>) of the analytic model; raises for any level scheme
        but "jg12_je12" and at a node of the gradient."""
        if self.level_scheme != "jg12_je12":
            raise ValueError(
                "the analytic polarization-gradient limits hold for j_g = 1/2 <-> j_e = 1/2 only; no source prints them for "
                f"{self.level_scheme} (Section 4.2.4)"
            )
        from qutip_trap.prep.polarization_gradient import fixed_phase_nbar, phase_averaged_nbar

        xi = self.xi(mode_freq_hz, s_single_beam)
        return fixed_phase_nbar(xi, self.phase_rad), phase_averaged_nbar(xi)

    def moving_gradient_ok(self, cooling_rate_hz: float, mode_freq_hz: float) -> bool:
        """W < delta < omega: the beat must outrun the cooling and stay below the trap frequency."""
        from qutip_trap.prep.polarization_gradient import moving_gradient_window

        return moving_gradient_window(
            2.0 * math.pi * cooling_rate_hz, 2.0 * math.pi * self.beat_hz, 2.0 * math.pi * mode_freq_hz
        )


__all__ = ["Beam", "PolGradientBeams", "PolarizationModulation"]
