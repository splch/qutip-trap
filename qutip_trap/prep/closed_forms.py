"""Closed-form cooling results the sources print, kept as cross-checks: none of them feeds a solve.

Delta = omega_drive - omega_transition (red negative), Gamma the full angular linewidth, Omega in the (hbar Omega/2)
convention, s = 2 Omega^2/Gamma^2; alpha and cos^2 theta_L are written explicitly, never also folded into eta.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qutip_trap.light.bloch import CoolingError, RateCoefficients
from qutip_trap.prep.validity import (
    assert_resolved,
    assert_resolved_linewidth,
    assert_three_level_valid,
    assert_weak_drive,
)
from qutip_trap.units import HBAR_J_S, K_B_J_PER_K


def x0_m(mass_kg: float, omega_rad_s: float) -> float:
    """x0 = sqrt(hbar/(2 m omega))."""
    if mass_kg <= 0.0 or omega_rad_s <= 0.0:
        raise ValueError("mass and frequency are positive")
    return math.sqrt(HBAR_J_S / (2.0 * mass_kg * omega_rad_s))


def lamb_dicke_parameter(k_rad_per_m: float, mass_kg: float, omega_rad_s: float) -> float:
    """eta = k x0 for a single ion with the full wavevector along the mode."""
    return k_rad_per_m * x0_m(mass_kg, omega_rad_s)


def bose_occupation(temperature_k: float, omega_rad_s: float) -> float:
    """nbar = 1/(exp(hbar omega/k_B T) - 1), the occupation of a mode at temperature T."""
    if temperature_k <= 0.0:
        return 0.0
    return 1.0 / math.expm1(HBAR_J_S * omega_rad_s / (K_B_J_PER_K * temperature_k))


def two_level_excited_population(gamma_rad_s: float, omega_rad_s: float, delta_rad_s: float) -> float:
    """rho_ee = (s/2)/[1 + s + (2 Delta/Gamma)^2], s = 2 Omega^2/Gamma^2 (Leibfried et al. 2003 Eq. 96)."""
    s = 2.0 * omega_rad_s**2 / gamma_rad_s**2
    return (s / 2.0) / (1.0 + s + (2.0 * delta_rad_s / gamma_rad_s) ** 2)


def doppler_force_n(gamma_rad_s: float, k_rad_per_m: float, s: float, delta_rad_s: float) -> float:
    """F_0 = hbar k Gamma (s/2)/[1 + s + (2 Delta/Gamma)^2], the push of one beam in N (Leibfried et al. 2003 Eq. 97)."""
    return (
        HBAR_J_S * k_rad_per_m * gamma_rad_s * (s / 2.0) / (1.0 + s + (2.0 * delta_rad_s / gamma_rad_s) ** 2)
    )


def doppler_friction_kg_per_s(gamma_rad_s: float, k_rad_per_m: float, s: float, delta_rad_s: float) -> float:
    """dF/dv at v = 0 for F = hbar k Gamma rho_ee(Delta - k v), negative (cooling) for Delta < 0."""
    x = 2.0 * delta_rad_s / gamma_rad_s
    return HBAR_J_S * k_rad_per_m**2 * (s / 2.0) * 8.0 * delta_rad_s / gamma_rad_s / (1.0 + s + x * x) ** 2


def doppler_force_energy_j(gamma_rad_s: float, delta_rad_s: float, s: float, alpha_over_cos2: float) -> float:
    """E = (hbar Gamma/8)(1 + alpha/cos^2 theta_L)[(1 + s) Gamma/(2|Delta|) + 2|Delta|/Gamma]: the Doppler force model's
    oscillator energy for nu << Gamma (Itano and Wineland 1982); raises CoolingError for Delta >= 0."""
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
    """E/(hbar nu) - 1/2 of the force model (the 1/2 is the zero point)."""
    return doppler_force_energy_j(gamma_rad_s, delta_rad_s, s, alpha_over_cos2) / (HBAR_J_S * nu_rad_s) - 0.5


def doppler_force_optimum_detuning(gamma_rad_s: float, s: float) -> float:
    """Delta_opt = -(Gamma/2) sqrt(1 + s) (Leibfried et al. 2003 Eq. 106)."""
    return -0.5 * gamma_rad_s * math.sqrt(1.0 + s)


def doppler_force_minimum_energy_j(gamma_rad_s: float, s: float, alpha_over_cos2: float) -> float:
    """E_min = (hbar Gamma/4)(1 + alpha/cos^2 theta_L) sqrt(1 + s), the force model's minimum energy."""
    return HBAR_J_S * gamma_rad_s / 4.0 * (1.0 + alpha_over_cos2) * math.sqrt(1.0 + s)


def doppler_temperature_k(energy_j: float) -> float:
    """k_B T = E for one degree of freedom of a harmonic oscillator (equipartition of kinetic and potential energy)."""
    return energy_j / K_B_J_PER_K


def three_beam_optimum(f_s: Sequence[float]) -> tuple[float, ...]:
    """gamma_si/gamma_tot = sqrt(f_si)/sum_j sqrt(f_sj): the best split of a fixed total scattering rate over orthogonal
    Doppler beams, f_si each axis's recoil share (Itano and Wineland 1982 Eqs. 20-26)."""
    root = [math.sqrt(float(f)) for f in f_s]
    if len(root) < 2 or any(f <= 0.0 for f in f_s):
        raise ValueError("three_beam_optimum needs at least two positive recoil shares f_si")
    total = sum(root)
    return tuple(r / total for r in root)


def three_beam_energy_j(
    gamma_rad_s: float, f_s: Sequence[float], shares: Sequence[float]
) -> tuple[float, ...]:
    """E_i = (hbar Gamma/4)(1 + f_si/share_i) per mode for orthogonal beams at Delta = -Gamma/2, s -> 0, carrying the
    fractions ``shares`` (not necessarily optimal) of the total scattering rate; k_B T_i = E_i."""
    if len(shares) != len(f_s):
        raise ValueError("one share per axis")
    if abs(sum(shares) - 1.0) > 1e-9 or any(x <= 0.0 for x in shares):
        raise ValueError("the beams' shares of the total scattering rate are positive and sum to one")
    return tuple(HBAR_J_S * gamma_rad_s / 4.0 * (1.0 + float(f) / float(x)) for f, x in zip(f_s, shares))


def three_beam_minimum_energy_j(gamma_rad_s: float, f_s: Sequence[float]) -> tuple[float, ...]:
    """E_i = (hbar Gamma/4)(1 + sqrt(f_si) sum_j sqrt(f_sj)) at the optimum split."""
    return three_beam_energy_j(gamma_rad_s, f_s, three_beam_optimum(f_s))


def unaddressed_axis_heating_per_s(two_d_per_s: float) -> float:
    """d<n>/dt = 2D on an axis no beam projects on: it heats linearly, with no steady state."""
    if two_d_per_s < 0.0:
        raise ValueError("the diffusion rate is non-negative")
    return two_d_per_s


def lorentzian_scattering_rate(
    omega_rad_s: float, gamma_rad_s: float, delta_rad_s: float, *, allow_saturation: bool = False
) -> float:
    """W(Delta) = Omega^2 Gamma/(Gamma^2 + 4 Delta^2), the unsaturated two-level rate (Eschner Eq. 6); refuses
    Omega/Gamma > 0.1 unless ``allow_saturation``."""
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
    """Bare A_+- = W(Delta -+ nu) + (eta~/eta)^2 W(Delta) with the weak-drive Lorentzian (Stenholm 1986 Eqs.
    5.49-5.54); refuses Omega/Gamma > 0.1 unless ``allow_saturation``."""
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


def sideband_floor(
    gamma_rad_s: float, nu_rad_s: float, carrier_weight: float, *, allow_unresolved: bool = False
) -> float:
    """nbar_SB = (Gamma/2 nu)^2 [(eta~/eta)^2 + 1/4] at Delta = -nu, Gamma << nu (Leibfried et al. 2003 Eq. 116);
    refuses Gamma/nu > 0.1 unless ``allow_unresolved``."""
    assert_resolved_linewidth(gamma_rad_s, nu_rad_s, allow_unresolved=allow_unresolved)
    return (gamma_rad_s / (2.0 * nu_rad_s)) ** 2 * (carrier_weight + 0.25)


def doppler_limit_nbar(gamma_rad_s: float, nu_rad_s: float, alpha_over_cos2: float) -> float:
    """(Gamma/4 nu)(1 + alpha/cos^2 theta_L): the rate-equation coefficients at Delta = -Gamma/2 in the Doppler regime, the
    zero-point 1/2 not subtracted."""
    return gamma_rad_s / (4.0 * nu_rad_s) * (1.0 + alpha_over_cos2)


def sideband_cooling_rate_n(n: int, eta: float, omega_rad_s: float, gamma_eff_rad_s: float) -> float:
    """R_n = Gamma~ (eta sqrt n Omega)^2/[2 (eta sqrt n Omega)^2 + Gamma~^2]: the rate out of |n> on the red sideband,
    saturating at Gamma~/2 (Leibfried et al. 2003 Eq. 110)."""
    if n < 0:
        raise ValueError("n is non-negative")
    x2 = (eta * math.sqrt(n) * omega_rad_s) ** 2
    return gamma_eff_rad_s * x2 / (2.0 * x2 + gamma_eff_rad_s**2)


Configuration = Literal["Xi", "V"]


@dataclass(frozen=True)
class EffectiveTwoLevel:
    """Gamma', gamma' and the cooling roles after adiabatic elimination of the fast level (Marzoli et al. 1994)."""

    configuration: Configuration
    excitation_fraction: float
    """Pi = (Omega_aux/2)^2/[delta_aux^2 + ((Gamma_10 + Gamma_12)/2)^2] = s_aux/2."""
    gamma_prime_rad_s: float
    """The effective population decay rate of the driven narrow transition."""
    gamma_coherence_rad_s: float
    """gamma' = Pi (Gamma_10 + Gamma_12)/2, the effective half width."""
    light_shift_rad_s: float
    """-delta_aux Pi: the ac Stark shift the auxiliary beam puts on the dressed level."""
    cooling_detuning_sign: int
    """-1 for Xi (cool at delta' = -nu), +1 for V (cool at delta_20 = +nu)."""

    @property
    def ratio(self) -> float:
        """gamma'/Gamma' = (Gamma_10 + Gamma_12)/(2 Gamma_10) (Xi) or /(2 Gamma_12) (V), never 1/2."""
        return self.gamma_coherence_rad_s / self.gamma_prime_rad_s

    def optimum_detuning_rad_s(self, nu_rad_s: float) -> float:
        """The cooling detuning with the light shift compensated: -nu - shift (Xi), +nu - shift (V)."""
        return self.cooling_detuning_sign * nu_rad_s - self.light_shift_rad_s

    def resolved(self, nu_rad_s: float) -> bool:
        """nu >> Gamma' and nu >> gamma' separately, taken as both below nu/5."""
        return max(self.gamma_prime_rad_s, self.gamma_coherence_rad_s) < 0.2 * nu_rad_s

    def assert_resolved(self, nu_rad_s: float, *, allow_unresolved: bool = False) -> None:
        """The raising form of :meth:`resolved`."""
        assert_resolved(
            self.gamma_prime_rad_s,
            self.gamma_coherence_rad_s,
            nu_rad_s,
            allow_unresolved=allow_unresolved,
            what=f"the effective {self.configuration} two-level system",
        )


def effective_two_level(
    gamma_10_rad_s: float,
    gamma_12_rad_s: float,
    omega_aux_rad_s: float,
    delta_aux_rad_s: float,
    configuration: Configuration,
    *,
    allow_invalid: bool = False,
) -> EffectiveTwoLevel:
    """The effective two-level system of a fast level |1> decaying to |0> (Gamma_10) and |2> (Gamma_12) (Marzoli et al.
    1994 Eqs. 12, 15): Xi drives |0> -> |2> narrow and |2> -> |1> auxiliary, V drives |0> -> |1> auxiliary and
    |0> -> |2> narrow; outside s_aux << 1, |delta_aux| << Gamma_10 + Gamma_12 refused unless ``allow_invalid``."""
    assert_three_level_valid(
        gamma_10_rad_s, gamma_12_rad_s, omega_aux_rad_s, delta_aux_rad_s, allow_invalid=allow_invalid
    )
    total = gamma_10_rad_s + gamma_12_rad_s
    pi = (omega_aux_rad_s / 2.0) ** 2 / (delta_aux_rad_s**2 + (total / 2.0) ** 2)
    if configuration == "Xi":
        gamma_prime = pi * gamma_10_rad_s
        sign = -1
    elif configuration == "V":
        gamma_prime = pi * gamma_12_rad_s
        sign = +1
    else:
        raise ValueError("configuration is 'Xi' or 'V'")
    return EffectiveTwoLevel(configuration, pi, gamma_prime, pi * total / 2.0, -delta_aux_rad_s * pi, sign)


def effective_two_level_valid(
    gamma_10_rad_s: float, gamma_12_rad_s: float, omega_aux_rad_s: float, delta_aux_rad_s: float
) -> bool:
    """s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12, taken as below 0.1 and 0.1 of the total."""
    total = gamma_10_rad_s + gamma_12_rad_s
    s_aux = (omega_aux_rad_s**2 / 2.0) / (delta_aux_rad_s**2 + (total / 2.0) ** 2)
    return s_aux < 0.1 and abs(delta_aux_rad_s) < 0.1 * total


def effective_two_level_rate(omega_rad_s: float, gamma_coherence_rad_s: float, delta_rad_s: float) -> float:
    """W(Delta) = (Omega^2/2) gamma'/(gamma'^2 + Delta^2): the weak-drive excitation rate of a line with coherence decay
    gamma' (Omega^2 Gamma/(Gamma^2 + 4 Delta^2) at gamma' = Gamma/2)."""
    return 0.5 * omega_rad_s**2 * gamma_coherence_rad_s / (gamma_coherence_rad_s**2 + delta_rad_s**2)


def v_system_diffusion(
    s_10: float,
    rho_00: float,
    gamma_12_rad_s: float,
    gamma_10_rad_s: float,
    eta_10: float,
    eta_12: float,
    alpha: float,
) -> float:
    """D = (s_10/4) rho_00 [Gamma_12 (eta_10^2 + alpha eta_12^2) + Gamma_10 eta_10^2 (1 + alpha)], alpha on the
    emitted photons only (Marzoli et al. 1994 Eq. B4)."""
    return (
        s_10
        / 4.0
        * rho_00
        * (gamma_12_rad_s * (eta_10**2 + alpha * eta_12**2) + gamma_10_rad_s * eta_10**2 * (1.0 + alpha))
    )


def half_rate_emission_diffusion(terms: list[tuple[float, float, float, float]]) -> float:
    """D_em = (1/2) sum_i rho_ii Gamma_ij alpha_ij eta_ij^2 over (population, rate, alpha, eta) channels; the phonon
    heating rate is 2 D."""
    return 0.5 * sum(p * g * a * e * e for p, g, a, e in terms)


def absorption_diffusion(excitation_rate_per_s: float, eta_abs: float) -> float:
    """D_abs = (1/2) R_exc eta_abs^2: the absorbed photon's recoil, never alpha-weighted."""
    return 0.5 * excitation_rate_per_s * eta_abs**2


def lambda_repumper_nbar(beta: float, nu_rad_s: float, delta_rad_s: float, omega_phi_rad_s: float) -> float:
    """<n> = (1/2)[Omega_phi^2/(4 Delta nu) + 4 Delta nu/(beta Omega_phi^2)] - 1/2 for a Lambda system with a repumper,
    inside R << nu << Gamma << Delta (Cirac et al. 1992 Eq. 58)."""
    if not 0.0 < beta <= 1.0:
        raise ValueError("the branching fraction beta lies in (0, 1]")
    x = omega_phi_rad_s**2 / (4.0 * delta_rad_s * nu_rad_s)
    return 0.5 * (x + 1.0 / (beta * x)) - 0.5


def lambda_repumper_minimum(beta: float, nu_rad_s: float, delta_rad_s: float) -> tuple[float, float]:
    """(Omega_phi at the minimum, <n>_min) = (sqrt(4 Delta nu/sqrt(beta)), 1/sqrt(beta) - 1/2)."""
    omega = math.sqrt(4.0 * delta_rad_s * nu_rad_s / math.sqrt(beta))
    return omega, 1.0 / math.sqrt(beta) - 0.5


def mixed_two_ion_axial_in_phase(mu: float) -> tuple[float, float]:
    """(b_1, b_2) of the lower (in-phase) axial mode of a two-ion mixed crystal, mass-weighted and normalized, with
    mu = m_2/m_1 and ion 1 the cooling ion (Wubbena et al. 2012 Eqs. 12-14)."""
    from qutip_trap.trap.crystal import two_ion_mixed_axial_squared

    lam, _upper = two_ion_mixed_axial_squared(mu)
    ratio = math.sqrt(mu) * (2.0 - lam)
    b1 = 1.0 / math.sqrt(1.0 + ratio * ratio)
    return b1, ratio * b1


def uniform_field_couplings(mu: float) -> tuple[float, float]:
    """(sum_j c_j/sqrt(m_j)) of the in-phase and out-of-phase axial modes in units of 1/sqrt(m_1): b_1 + b_2/sqrt(mu)
    and b_2 - b_1/sqrt(mu) (Wubbena et al. 2012 Eqs. 30-31)."""
    b1, b2 = mixed_two_ion_axial_in_phase(mu)
    root = math.sqrt(mu)
    return b1 + b2 / root, b2 - b1 / root


def sympathetic_cost(mu: float, *, swapped: bool = True) -> float:
    """F(mu) = b_2^2 (b_1 + b_2/sqrt mu)^2/b_1^2 + b_1^2 (b_2 - b_1/sqrt mu)^2/b_2^2: the clock ion's heating from
    uniform field noise per single-mode rate (Wubbena et al. 2012 Eqs. 34-36); ``swapped=False`` is a negative control."""
    b1, b2 = mixed_two_ion_axial_in_phase(mu)
    e_in, e_out = uniform_field_couplings(mu)
    w_in, w_out = (b2**2, b1**2) if swapped else (b1**2, b2**2)
    return w_in * e_in**2 / b1**2 + w_out * e_out**2 / b2**2


SYMPATHETIC_OPTIMUM_MU = 8.0 / 11.0
"""mu* = 8/11: the mass ratio minimizing F (Wubbena et al. 2012 Eq. 36)."""

SYMPATHETIC_OPTIMUM_COST = 23.0 / 16.0
"""F_min = 23/16 at mu* = 8/11 (Wubbena et al. 2012 Eq. 36)."""


def sympathetic_optimum(bounds: tuple[float, float] = (0.05, 20.0)) -> tuple[float, float]:
    """(mu*, F_min) by bounded minimization of :func:`sympathetic_cost`; (8/11, 23/16) exactly."""
    from scipy.optimize import minimize_scalar

    sol = minimize_scalar(sympathetic_cost, bounds=bounds, method="bounded", options={"xatol": 1e-12})
    return float(sol.x), float(sol.fun)


def offresonant_scattering_rate_per_s(s: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """R_sc = s Gamma^3/(8 Delta^2): the far-detuned scattering rate of the repump light (Che et al. 2017)."""
    return s * gamma_rad_s**3 / (8.0 * delta_rad_s**2)


def repump_recoil_quanta(
    t_repump_s: float, s: float, gamma_rad_s: float, eta: float, delta_hf_rad_s: float
) -> float:
    """Delta nbar = t s Gamma^3 eta^2/(4 Delta_HF^2) = R_sc t (2 eta^2), Gamma and Delta_HF angular: the repump recoil
    per cooling cycle (Che et al. 2017 Eq. 6)."""
    return t_repump_s * s * gamma_rad_s**3 * eta**2 / (4.0 * delta_hf_rad_s**2)
