"""EIT cooling closed forms (PLAN.md Section 4.2.3; Morigi, Eschner and Keitel 2000; Morigi 2003; Roos 2000).

In the plan's sign, Delta = omega_drive - omega_transition, cooling needs Delta > 0 (blue) and the tuning rule reads
Omega_r^2 = 4 nu (nu + Delta) (Morigi's Delta is the negative of the plan's). With Delta_r = Delta_g = Delta the bare
rate coefficients are

    A_+- = (Omega_g^2/gamma) gamma^2 nu^2 / { gamma^2 nu^2 + 4 [Omega_r^2/4 - nu (nu -+ Delta)]^2 },

eta^2 outside with eta the two-photon Lamb-Dicke parameter, and the steady state

    <n>_S = A_+/(A_- - A_+) = [gamma^2 nu^2 + 4 (Omega_r^2/4 - nu (nu + Delta))^2] / [4 Delta nu (Omega_r^2 - 4 nu^2)],

equal to (gamma/4 Delta)^2 exactly at delta = nu, with poles at Delta = 0 and Omega_r = 2 nu. The light shift of the
narrow, ground-like dressed state is delta = (sqrt(Delta^2 + Omega_r^2) - |Delta|)/2; several coupling Rabi frequencies
compose as the quadrature sum. No carrier recoil term appears: W(Delta) = 0 at the dark resonance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qutip_trap.light.bloch import CoolingError
from qutip_trap.prep.closed_forms import RateCoefficients
from qutip_trap.prep.validity import assert_weak_drive


def light_shift_rad_s(delta_rad_s: float, omega_r_rad_s: float) -> float:
    """delta = (sqrt(Delta^2 + Omega_r^2) - |Delta|)/2 ~ Omega_r^2/(4 Delta): the shift of the narrow dressed state
    (Morigi 2000 Eq. 1)."""
    return 0.5 * (math.sqrt(delta_rad_s**2 + omega_r_rad_s**2) - abs(delta_rad_s))


def coupling_for_target_rad_s(nu_rad_s: float, delta_rad_s: float) -> float:
    """Omega_r = 2 sqrt(nu (nu + Delta)): the coupling that puts the light shift exactly on the mode, delta = nu."""
    if delta_rad_s <= 0.0:
        raise CoolingError("EIT cooling needs a blue-detuned coupling beam, Delta > 0 in the plan's sign")
    return 2.0 * math.sqrt(nu_rad_s * (nu_rad_s + delta_rad_s))


def dressed_linewidth_rad_s(gamma_rad_s: float, delta_rad_s: float, omega_r_rad_s: float) -> float:
    """Gamma' ~ Gamma (2 delta/Omega_r)^2 ~ Gamma nu/Delta: the narrow dressed state's width, the cooling bandwidth."""
    return gamma_rad_s * (2.0 * light_shift_rad_s(delta_rad_s, omega_r_rad_s) / omega_r_rad_s) ** 2


def eit_rate_coefficients(
    omega_g_rad_s: float,
    omega_r_rad_s: float,
    nu_rad_s: float,
    delta_rad_s: float,
    gamma_rad_s: float,
    *,
    allow_saturation: bool = False,
) -> RateCoefficients:
    """The bare A_+- in s^-1 (eta^2 outside), carrier weight zero; the perturbative form needs a weak probe,
    Omega_g << gamma, and refuses a saturated one unless ``allow_saturation``."""
    if nu_rad_s <= 0.0 or gamma_rad_s <= 0.0:
        raise ValueError("nu and gamma are positive")
    assert_weak_drive(
        omega_g_rad_s,
        gamma_rad_s,
        allow_saturation=allow_saturation,
        what="the EIT bare A_+- (Omega_g << gamma)",
    )

    def a(sign: float) -> float:
        bracket = omega_r_rad_s**2 / 4.0 - nu_rad_s * (nu_rad_s - sign * delta_rad_s)
        return (
            (omega_g_rad_s**2 / gamma_rad_s)
            * gamma_rad_s**2
            * nu_rad_s**2
            / (gamma_rad_s**2 * nu_rad_s**2 + 4.0 * bracket**2)
        )

    return RateCoefficients(a(+1.0), a(-1.0), 0.0)


def eit_steady_state_nbar(
    omega_r_rad_s: float, nu_rad_s: float, delta_rad_s: float, gamma_rad_s: float
) -> float:
    """<n>_S, independent of Omega_g; raises at the poles Delta = 0 and Omega_r = 2 nu and whenever A_- <= A_+
    (Delta < 0, or Omega_r < 2 nu with Delta > 0)."""
    denominator = 4.0 * delta_rad_s * nu_rad_s * (omega_r_rad_s**2 - 4.0 * nu_rad_s**2)
    if denominator <= 0.0:
        raise CoolingError(
            "EIT closed form: A_- <= A_+ (stability needs Delta (Omega_r^2 - 4 nu^2) > 0; the poles Delta = 0 and "
            "Omega_r = 2 nu are where cooling vanishes)"
        )
    numerator = (
        gamma_rad_s**2 * nu_rad_s**2
        + 4.0 * (omega_r_rad_s**2 / 4.0 - nu_rad_s * (nu_rad_s + delta_rad_s)) ** 2
    )
    return numerator / denominator


@dataclass(frozen=True)
class EitClosedForm:
    """The EIT description of one mode: coefficients, steady state, light shift and bandwidth."""

    nu_rad_s: float
    delta_rad_s: float
    omega_g_rad_s: float
    omega_r_rad_s: float
    gamma_rad_s: float
    eta: float

    @property
    def coefficients(self) -> RateCoefficients:
        """The bare A_+-, evaluated outside the weak-probe regime too (``weak_probe`` reports it)."""
        return eit_rate_coefficients(
            self.omega_g_rad_s,
            self.omega_r_rad_s,
            self.nu_rad_s,
            self.delta_rad_s,
            self.gamma_rad_s,
            allow_saturation=not self.weak_probe(),
        )

    @property
    def nbar(self) -> float:
        return eit_steady_state_nbar(self.omega_r_rad_s, self.nu_rad_s, self.delta_rad_s, self.gamma_rad_s)

    @property
    def cooling_rate_per_s(self) -> float:
        return self.coefficients.cooling_rate_per_s(self.eta)

    @property
    def light_shift_rad_s(self) -> float:
        return light_shift_rad_s(self.delta_rad_s, self.omega_r_rad_s)

    @property
    def bandwidth_rad_s(self) -> float:
        return dressed_linewidth_rad_s(self.gamma_rad_s, self.delta_rad_s, self.omega_r_rad_s)

    def weak_probe(self, ratio: float = 0.1) -> bool:
        """Omega_g << Omega_r, the regime the closed form assumes."""
        return self.omega_g_rad_s < ratio * self.omega_r_rad_s
