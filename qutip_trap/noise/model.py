"""The ``NoiseModel`` record (PLAN.md Section 6; Appendix E; milestone M7 for the methods)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qutip_trap.noise.sampling import NoiseSample
from qutip_trap.noise.spectra import Collisions, Drift, Mains, NoiseSpectrum

if TYPE_CHECKING:
    from qutip_trap.device.model import Device
    from qutip_trap.dynamics.channels import CollapseOp
    from qutip_trap.hilbert.space import HilbertSpace

M7 = "milestone M7 (noise/, PLAN.md Section 6)"


@dataclass(frozen=True)
class NoiseModel:
    S_E: NoiseSpectrum
    """Electric-field noise (heating, Section 6.2); stored two-sided like every NoiseSpectrum."""
    correlation_length_m: float | None
    S_B: NoiseSpectrum | None
    mains: Mains | None
    laser_phase: NoiseSpectrum | None
    laser_intensity: NoiseSpectrum | None
    rf_amplitude_noise: NoiseSpectrum | None
    rf_phase_noise: NoiseSpectrum | None
    rf_amplitude_drift: Drift
    """Common-mode fractional drift of every rf-derived mode of a family."""
    mode_drift_differential: Drift
    rabi_drift: Drift
    beam_phase_drift: Drift
    field_drift: Drift
    stray_field_drift: Drift
    pointing_drift: Drift
    """Beam pointing, which moves crosstalk and Rabi rate together (Section 6.6)."""
    rabi_amplitude: NoiseSpectrum | None
    """Two-sided S_a(omega) of the ADDITIVE amplitude noise in rad/s (Section 6.9); Omega is taken from the pulse
    at use and never folded as Omega^2 into a device PSD."""
    collisions: Collisions | None

    def channels(self, device: Device, space: HilbertSpace) -> tuple[CollapseOp, ...]:
        """The collapse operators of Section 5.7 from the spectra and the space (Section 13 normalizations)."""
        raise NotImplementedError(f"NoiseModel.channels is {M7}")

    def sample(self, rng: np.random.Generator) -> NoiseSample:
        raise NotImplementedError(f"NoiseModel.sample is {M7}")


__all__ = ["NoiseModel"]
