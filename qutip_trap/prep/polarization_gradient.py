"""Polarization-gradient (Sisyphus) cooling: Joshi et al. 2020's analytic j_g = 1/2 <-> j_e = 1/2 lin-perp-lin model
(PLAN.md Section 4.2.4), two counter-propagating beams along the quantization axis.

With s the single-beam saturation parameter (Omega^2/2)/(Gamma^2/4 + Delta^2) of the S1/2-P3/2 stretched transition,
Delta > 0 and phi the gradient phase at the trap centre, the depth is xi = Delta s/(3 omega) ("depth equals the trap
frequency" is xi = 1/2). Linearized in kz, W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi and
H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi), steady state H/W - 1/2: at fixed phi = 0,
xi + 1/(4 xi) - 1/2 (minimum 1/2 at xi = 1/2); phase-averaged (a gradient moving fast against W inside W < delta < omega),
(3/4) xi + 5/(8 xi) - 1/2 (minimum sqrt(15/8) - 1/2 at xi = sqrt(5/6)). W vanishes at cos 2phi = 0, which raises.
"""

from __future__ import annotations

import math

from qutip_trap.light.bloch import CoolingError

COS2_THRESHOLD = 1e-2
"""|cos 2phi|^2 below which a static-gradient ion is not cooled (W is proportional to cos^2 2phi)."""


class UncooledPhaseError(CoolingError):
    """An ion sits at a node of the polarization gradient (cos 2phi = 0) and is not cooled."""


def xi_depth(delta_rad_s: float, s_joshi: float, omega_mode_rad_s: float) -> float:
    """xi = Delta s/(3 omega): the light-shift modulation amplitude over the mode frequency; Delta > 0 required."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning")
    if omega_mode_rad_s <= 0.0:
        raise ValueError("the mode frequency is positive")
    return delta_rad_s * s_joshi / (3.0 * omega_mode_rad_s)


def fixed_phase_nbar(xi: float, phi_rad: float = 0.0, *, threshold: float = COS2_THRESHOLD) -> float:
    """H(phi)/W(phi) - 1/2 at one gradient phase (xi + 1/(4 xi) - 1/2 at phi = 0); raises at a node of the gradient."""
    c2 = math.cos(2.0 * phi_rad) ** 2
    if c2 < threshold:
        raise UncooledPhaseError(
            f"|cos 2phi|^2 = {c2:.2e} < {threshold:.0e}: the ion sits at a node of the polarization gradient and is "
            "not cooled"
        )
    # eta^2 Gamma s cancels between H and W
    w = 16.0 / 9.0 * xi * c2
    h = 2.0 / 9.0 * (8.0 * xi**2 * c2 * c2 + 2.0 + (1.0 - c2))
    return h / w - 0.5


def phase_averaged_nbar(xi: float) -> float:
    """<H>_phi/<W>_phi - 1/2 = (3/4) xi + 5/(8 xi) - 1/2, valid for a gradient moving fast against W inside
    W < delta < omega_z, never for a static gradient sampled by many ions."""
    if xi <= 0.0:
        raise CoolingError("xi > 0 (blue detuning) is required")
    return 0.75 * xi + 5.0 / (8.0 * xi) - 0.5


def moving_gradient_window(cooling_rate_rad_s: float, beat_rad_s: float, omega_mode_rad_s: float) -> bool:
    """W < delta < omega_z: the inter-beam frequency difference must outrun the cooling and stay below the trap
    frequency."""
    return cooling_rate_rad_s < beat_rad_s < omega_mode_rad_s
