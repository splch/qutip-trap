"""Closed-form cooling rates: the Doppler force model and Stenholm's resolved-sideband coefficients.

Conventions: Delta = omega_drive - omega_transition (red negative), Gamma the full (angular) width, Omega in the
(hbar Omega/2) convention, s = 2 Omega^2/Gamma^2, alpha the emission recoil angular factor and cos^2 theta_L the beam's
projection on the mode axis.

- Doppler force model (RMP 2003 Eqs. 96-107; Itano and Wineland 1982), valid for nu << Gamma: the oscillator energy
  E = (hbar Gamma/8)(1 + alpha/cos^2 theta_L)[(1 + s) Gamma/(2|Delta|) + 2|Delta|/Gamma].
- Stenholm 1986 Eqs. 5.49-5.54: A_+- = W(Delta -+ nu) + (eta~/eta)^2 W(Delta) with the weak-drive Lorentzian
  W(Delta) = Omega^2 Gamma/(Gamma^2 + 4 Delta^2), eta^2 outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qutip_trap.light.bloch import CoolingError
from qutip_trap.prep.validity import assert_weak_drive
from qutip_trap.units import HBAR_J_S


def x0_m(mass_kg: float, omega_rad_s: float) -> float:
    """sqrt(hbar/(2 m omega))."""
    if mass_kg <= 0.0 or omega_rad_s <= 0.0:
        raise ValueError("mass and frequency are positive")
    return math.sqrt(HBAR_J_S / (2.0 * mass_kg * omega_rad_s))


def lamb_dicke_parameter(k_rad_per_m: float, mass_kg: float, omega_rad_s: float) -> float:
    """eta = k x0 for a single ion with the full wavevector along the mode (Roos p. 28: 40Ca+ 729 nm at 1 MHz, 0.0969)."""
    return k_rad_per_m * x0_m(mass_kg, omega_rad_s)


@dataclass(frozen=True)
class RateCoefficients:
    """Bare A_+- in s^-1, the drive's eta^2 kept outside."""

    A_plus_per_s: float
    A_minus_per_s: float
    carrier_weight: float
    """eta~^2/eta^2, the emitted photon's recoil weight on the carrier term."""

    @property
    def cooling_rate_bare_per_s(self) -> float:
        """A_- - A_+: times eta^2 the phonon relaxation rate W."""
        return self.A_minus_per_s - self.A_plus_per_s

    @property
    def nbar(self) -> float:
        """A_+/(A_- - A_+); raises CoolingError when the configuration heats."""
        if self.A_minus_per_s <= self.A_plus_per_s:
            raise CoolingError(
                f"A_- = {self.A_minus_per_s:.4g} <= A_+ = {self.A_plus_per_s:.4g} s^-1: no cooling steady state"
            )
        return self.A_plus_per_s / (self.A_minus_per_s - self.A_plus_per_s)

    def cooling_rate_per_s(self, eta: float) -> float:
        return eta**2 * self.cooling_rate_bare_per_s


def doppler_force_energy_j(gamma_rad_s: float, delta_rad_s: float, s: float, alpha_over_cos2: float) -> float:
    """The steady oscillator energy of the force model: friction against the absorption recoil (weight 1 along the beam)
    plus the emission recoil (alpha, projected on the mode); k_B T = E. Raises for Delta >= 0 (the force heats)."""
    if delta_rad_s >= 0.0:
        raise CoolingError("the Doppler force cools only for Delta < 0 (RMP Eq. 101)")
    if s < 0.0 or alpha_over_cos2 < 0.0:
        raise ValueError("s and alpha/cos^2 are non-negative")
    a = abs(delta_rad_s)
    return (
        HBAR_J_S
        * gamma_rad_s
        / 8.0
        * (1.0 + alpha_over_cos2)
        * ((1.0 + s) * gamma_rad_s / (2.0 * a) + 2.0 * a / gamma_rad_s)
    )


def doppler_force_nbar(
    gamma_rad_s: float, delta_rad_s: float, s: float, nu_rad_s: float, alpha_over_cos2: float
) -> float:
    """E/(hbar nu) - 1/2 of the force model (349.5 at Gamma/nu = 1e3, alpha = 2/5, the 1/2 being the zero point)."""
    return doppler_force_energy_j(gamma_rad_s, delta_rad_s, s, alpha_over_cos2) / (HBAR_J_S * nu_rad_s) - 0.5


def lorentzian_scattering_rate(
    omega_rad_s: float, gamma_rad_s: float, delta_rad_s: float, *, allow_saturation: bool = False
) -> float:
    """W(Delta) = Omega^2 Gamma/(Gamma^2 + 4 Delta^2), the unsaturated two-level rate (Eschner Eq. 6); it is high by
    1 + s, so Omega/Gamma above ``validity.SMALL`` is refused unless ``allow_saturation``."""
    assert_weak_drive(
        omega_rad_s,
        gamma_rad_s,
        allow_saturation=allow_saturation,
        what="the unsaturated Lorentzian W(Delta)",
    )
    return omega_rad_s**2 * gamma_rad_s / (gamma_rad_s**2 + 4.0 * delta_rad_s**2)


def stenholm_coefficients(
    omega_rad_s: float,
    gamma_rad_s: float,
    nu_rad_s: float,
    delta_rad_s: float,
    carrier_weight: float,
    *,
    allow_saturation: bool = False,
) -> RateCoefficients:
    """Bare A_+- = W(Delta -+ nu) + (eta~/eta)^2 W(Delta) with the weak-drive Lorentzian; the carrier weight is alpha for
    one beam along the mode axis and alpha (k_em/k_L)^2/cos^2 theta_L in general."""
    if nu_rad_s <= 0.0:
        raise ValueError("the mode frequency is positive")
    assert_weak_drive(
        omega_rad_s, gamma_rad_s, allow_saturation=allow_saturation, what="Stenholm's bare A_+-"
    )
    w0 = lorentzian_scattering_rate(omega_rad_s, gamma_rad_s, delta_rad_s, allow_saturation=True)
    a_plus = (
        lorentzian_scattering_rate(omega_rad_s, gamma_rad_s, delta_rad_s - nu_rad_s, allow_saturation=True)
        + carrier_weight * w0
    )
    a_minus = (
        lorentzian_scattering_rate(omega_rad_s, gamma_rad_s, delta_rad_s + nu_rad_s, allow_saturation=True)
        + carrier_weight * w0
    )
    return RateCoefficients(a_plus, a_minus, carrier_weight)
