"""Closed-form cooling oracles of PLAN.md Section 4.2 (milestone M3): the level-A forms the sources print.

Every form here is a stated-regime approximation that the level-B/C engines of ``prep`` are tested against
(Section 9.3); none is fed back into a solve. Conventions (Section 13): Delta = omega_drive - omega_transition
(red negative), Gamma the full width (angular), Omega in the (hbar Omega/2) convention so a resonant pi pulse is
Omega t = pi, s = 2 Omega^2/Gamma^2 the saturation parameter of a two-level line, alpha the recoil angular factor of
Section 4.2.8 and cos^2 theta_L the beam's projection on the mode axis, the two never folded into eta while also
written explicitly (Section 13, "Where eta^2 sits").

- Doppler force model (RMP 2003 Eqs. 96-107; Itano and Wineland 1982): valid for nu << Gamma only (Section 4.2.1);
  balancing the linearized friction against the absorption-plus-emission recoil gives the oscillator energy
  E = (hbar Gamma/8)(1 + alpha/cos^2 theta_L)[(1 + s) Gamma/(2|Delta|) + 2|Delta|/Gamma], minimized at
  Delta = -(Gamma/2) sqrt(1 + s) to E_min = (hbar Gamma/4)(1 + alpha/cos^2 theta_L) sqrt(1 + s); k_B T = E for one
  degree of freedom, the kinetic part E_K = E/2 is Itano's (1 + f_s) hbar Gamma/8, and alpha = 1 (Berkeland's
  hbar Gamma/2) against alpha = 2/5 (the RMP's 0.35 hbar Gamma) is the 1.43 fork of Section 13.
- Stenholm's sideband-cooling coefficients (Stenholm 1986 Eqs. 5.49-5.54; RMP Eqs. 117-121): the bare
  A_+- = W(Delta -+ nu) + (eta~^2/eta^2) W(Delta) with the weak-drive Lorentzian W(Delta) = Omega^2 Gamma/(Gamma^2 + 4 Delta^2),
  the floor (Gamma/2 nu)^2 [(eta~/eta)^2 + 1/4] and the Doppler limit (Gamma/4 nu)(1 + alpha/cos^2 theta_L) at Delta = -Gamma/2.
- Marzoli et al. 1994 (Eqs. 12, 15; Roos Eq. 3.21): the effective two-level parameters of a quenched or repumped
  fast level, Pi = (Omega_aux/2)^2/[delta_aux^2 + ((Gamma_10 + Gamma_12)/2)^2] = s_aux/2, Gamma' = Pi Gamma_10 (Xi) or
  Pi Gamma_12 (V), gamma' = Pi (Gamma_10 + Gamma_12)/2, so gamma'/Gamma' is (Gamma_10 + Gamma_12)/(2 Gamma_10) or
  /(2 Gamma_12) and never 1/2; the V configuration cools at delta_20 = +nu (Section 13, "Effective two-level roles").
- Marzoli Eq. B4: the V-system diffusion D = (s_10/4) rho_00 [Gamma_12 (eta_10^2 + alpha eta_12^2) + Gamma_10 eta_10^2 (1 + alpha)],
  the half-rate emission diffusion plus the absorption recoil, alpha on the emitted photon only (Section 13,
  "Absorption versus emission recoil").
- Cirac, Blatt, Zoller and Phillips 1992 Eq. 58: the Lambda system with a repumper, <n>_min = 1/sqrt(beta) - 1/2 at
  Omega_phi^2 = 4 Delta nu/sqrt(beta) inside R << nu << Gamma << Delta.
- Che et al. 2017 Eq. 6: the repump recoil per cooling cycle Delta nbar = t s Gamma^3 eta^2/(4 Delta_HF^2), which equals the
  off-resonant scattering rate s Gamma^3/(8 Delta^2) times t times 2 eta^2 (Section 9.15).
- Roos thesis p. 21: the saturating cooling rate R_n = Gamma (eta sqrt n Omega)^2/[2 (eta sqrt n Omega)^2 + Gamma^2] -> Gamma/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from qutip_trap.light.bloch import CoolingError, RateCoefficients
from qutip_trap.units import HBAR_J_S, K_B_J_PER_K


def x0_m(mass_kg: float, omega_rad_s: float) -> float:
    """sqrt(hbar/(2 m omega)) (Section 13, "Ladder operators")."""
    if mass_kg <= 0.0 or omega_rad_s <= 0.0:
        raise ValueError("mass and frequency are positive")
    return math.sqrt(HBAR_J_S / (2.0 * mass_kg * omega_rad_s))


def lamb_dicke_parameter(k_rad_per_m: float, mass_kg: float, omega_rad_s: float) -> float:
    """eta = k x0 for a single ion with the full wavevector along the mode (Roos p. 28: 40Ca+ 729 nm at 1 MHz, 0.0969)."""
    return k_rad_per_m * x0_m(mass_kg, omega_rad_s)


def bose_occupation(temperature_k: float, omega_rad_s: float) -> float:
    """nbar = 1/(exp(hbar omega/k_B T) - 1), the occupation of a mode at temperature T."""
    if temperature_k <= 0.0:
        return 0.0
    return 1.0 / math.expm1(HBAR_J_S * omega_rad_s / (K_B_J_PER_K * temperature_k))


# ---- Doppler force model (Section 4.2.1; nu << Gamma only) ------------------------------------------------------------------


def two_level_excited_population(gamma_rad_s: float, omega_rad_s: float, delta_rad_s: float) -> float:
    """rho_ee = (s/2)/[1 + s + (2 Delta/Gamma)^2], s = 2 Omega^2/Gamma^2 (RMP 2003 Eq. 96)."""
    s = 2.0 * omega_rad_s**2 / gamma_rad_s**2
    return (s / 2.0) / (1.0 + s + (2.0 * delta_rad_s / gamma_rad_s) ** 2)


def doppler_force_n(gamma_rad_s: float, k_rad_per_m: float, s: float, delta_rad_s: float) -> float:
    """F_0 = hbar k Gamma (s/2)/[1 + s + (2 Delta/Gamma)^2]: the constant push of one beam (RMP Eq. 97), in newtons."""
    return (
        HBAR_J_S * k_rad_per_m * gamma_rad_s * (s / 2.0) / (1.0 + s + (2.0 * delta_rad_s / gamma_rad_s) ** 2)
    )


def doppler_friction_kg_per_s(gamma_rad_s: float, k_rad_per_m: float, s: float, delta_rad_s: float) -> float:
    """dF/dv at v = 0 for F = hbar k Gamma rho_ee(Delta - k v): negative (cooling) for Delta < 0 (RMP Eqs. 98-101)."""
    x = 2.0 * delta_rad_s / gamma_rad_s
    return HBAR_J_S * k_rad_per_m**2 * (s / 2.0) * 8.0 * delta_rad_s / gamma_rad_s / (1.0 + s + x * x) ** 2


def doppler_force_energy_j(gamma_rad_s: float, delta_rad_s: float, s: float, alpha_over_cos2: float) -> float:
    """The steady oscillator energy of the force model, friction against absorption-plus-emission recoil (Section 4.2.1).

    E = (hbar Gamma/8)(1 + alpha/cos^2 theta_L)[(1 + s) Gamma/(2|Delta|) + 2|Delta|/Gamma]; the absorption recoil enters
    with weight 1 along the beam and the emission recoil with alpha, projected on the mode (Section 13, "Absorption versus
    emission recoil"); k_B T = E for one degree of freedom. Raises for Delta >= 0 (the force heats).
    """
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
    """E/(hbar nu) - 1/2 of the force model (Section 9.3: 349.5 against 350 at Gamma/nu = 1e3, the 1/2 being the zero point)."""
    return doppler_force_energy_j(gamma_rad_s, delta_rad_s, s, alpha_over_cos2) / (HBAR_J_S * nu_rad_s) - 0.5


def doppler_force_optimum_detuning(gamma_rad_s: float, s: float) -> float:
    """Delta_opt = -(Gamma/2) sqrt(1 + s) (RMP Eq. 106, the radical over 1 + s alone; Section 4.2.1)."""
    return -0.5 * gamma_rad_s * math.sqrt(1.0 + s)


def doppler_force_minimum_energy_j(gamma_rad_s: float, s: float, alpha_over_cos2: float) -> float:
    """E_min = (hbar Gamma/4)(1 + alpha/cos^2 theta_L) sqrt(1 + s): hbar Gamma/2 for alpha = 1 (Berkeland Eq. 13), 0.35 hbar Gamma
    for alpha = 2/5 (RMP Eq. 106), the 1.43 fork of Section 13 "Doppler limit"; Itano's E_K = E/2 = (1 + f_s) hbar Gamma/8."""
    return HBAR_J_S * gamma_rad_s / 4.0 * (1.0 + alpha_over_cos2) * math.sqrt(1.0 + s)


def doppler_temperature_k(energy_j: float) -> float:
    """k_B T = E for one degree of freedom of a harmonic oscillator (equipartition of kinetic and potential energy)."""
    return energy_j / K_B_J_PER_K


# ---- Stenholm / RMP coefficients (Sections 4.2.2, 4.2.8 iv) ----------------------------------------------------------------


def lorentzian_scattering_rate(omega_rad_s: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """W(Delta) = Omega^2 Gamma/(Gamma^2 + 4 Delta^2) = (Omega^2/Gamma) L(Delta): the unsaturated two-level rate (Eschner Eq. 6)."""
    return omega_rad_s**2 * gamma_rad_s / (gamma_rad_s**2 + 4.0 * delta_rad_s**2)


def stenholm_coefficients(
    omega_rad_s: float, gamma_rad_s: float, nu_rad_s: float, delta_rad_s: float, carrier_weight: float
) -> RateCoefficients:
    """Bare A_+- = W(Delta -+ nu) + (eta~/eta)^2 W(Delta) with the weak-drive Lorentzian (Stenholm Eqs. 5.49-5.54; Section 13
    "Cooling coefficients A_+-"); the weight is alpha for one resonant beam along the mode axis and alpha (k_em/k_L)^2/cos^2 theta_L
    in general (Section 4.2.2)."""
    if nu_rad_s <= 0.0:
        raise ValueError("the mode frequency is positive")
    w0 = lorentzian_scattering_rate(omega_rad_s, gamma_rad_s, delta_rad_s)
    a_plus = (
        lorentzian_scattering_rate(omega_rad_s, gamma_rad_s, delta_rad_s - nu_rad_s) + carrier_weight * w0
    )
    a_minus = (
        lorentzian_scattering_rate(omega_rad_s, gamma_rad_s, delta_rad_s + nu_rad_s) + carrier_weight * w0
    )
    return RateCoefficients(a_plus, a_minus, carrier_weight)


def sideband_floor(gamma_rad_s: float, nu_rad_s: float, carrier_weight: float) -> float:
    """nbar_SB = (Gamma/2 nu)^2 [(eta~/eta)^2 + 1/4] at Delta = -nu, Gamma << nu (RMP Eq. 116; Roos Eq. 3.20; Eschner Eq. 6).

    Stenholm's (gamma/nu)^2 (alpha + 1/4) with the half width gamma = Gamma/2: 7/12 and 13/20 in units of (gamma/nu)^2 for
    alpha = 1/3 and 2/5; alpha -> 0 leaves (Gamma/4 nu)^2 from off-resonant blue-sideband excitation, never zero.
    """
    return (gamma_rad_s / (2.0 * nu_rad_s)) ** 2 * (carrier_weight + 0.25)


def doppler_limit_nbar(gamma_rad_s: float, nu_rad_s: float, alpha_over_cos2: float) -> float:
    """(Gamma/4 nu)(1 + alpha/cos^2 theta_L): the Section 4.2.2 coefficients at Delta = -Gamma/2 in the Doppler regime, the zero-point
    1/2 not subtracted (Section 9.3 "Doppler limit with recoil")."""
    return gamma_rad_s / (4.0 * nu_rad_s) * (1.0 + alpha_over_cos2)


def sideband_cooling_rate_n(n: int, eta: float, omega_rad_s: float, gamma_eff_rad_s: float) -> float:
    """R_n = Gamma~ (eta sqrt n Omega)^2/[2 (eta sqrt n Omega)^2 + Gamma~^2]: the rate out of |n> on the red sideband, saturating at
    Gamma~/2 (RMP Eq. 110; Roos p. 21); (eta Omega)^2/Gamma~ at n = 1 in the weak-drive limit. An unbounded rate is a bug."""
    if n < 0:
        raise ValueError("n is non-negative")
    x2 = (eta * math.sqrt(n) * omega_rad_s) ** 2
    return gamma_eff_rad_s * x2 / (2.0 * x2 + gamma_eff_rad_s**2)


# ---- effective two-level parameters of a quenched or repumped fast level (Marzoli 1994; Roos Eq. 3.21) ---------------------


Configuration = Literal["Xi", "V"]


@dataclass(frozen=True)
class EffectiveTwoLevel:
    """Gamma', gamma' and the roles of Section 13 "Effective two-level roles" after adiabatic elimination of the fast level."""

    configuration: Configuration
    excitation_fraction: float
    """Pi = (Omega_aux/2)^2/[delta_aux^2 + ((Gamma_10 + Gamma_12)/2)^2] = s_aux/2."""
    gamma_prime_rad_s: float
    """The effective population decay rate of the driven narrow transition."""
    gamma_coherence_rad_s: float
    """gamma' = Pi (Gamma_10 + Gamma_12)/2, the effective half width."""
    light_shift_rad_s: float
    """-delta_aux Pi: the ac Stark shift the auxiliary beam puts on the dressed level (Marzoli Fig. 3)."""
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
        """nu >> Gamma' and nu >> gamma' separately (Section 4.2.8 v); here both below nu/5."""
        return max(self.gamma_prime_rad_s, self.gamma_coherence_rad_s) < 0.2 * nu_rad_s


def effective_two_level(
    gamma_10_rad_s: float,
    gamma_12_rad_s: float,
    omega_aux_rad_s: float,
    delta_aux_rad_s: float,
    configuration: Configuration,
) -> EffectiveTwoLevel:
    """Marzoli Eqs. 12 and 15: the fast level |1> decays to |0> (Gamma_10) and |2> (Gamma_12) and the auxiliary beam
    (Omega_aux, delta_aux) connects it to the metastable level; Xi drives |0> -> |2> narrow with |2> -> |1| auxiliary,
    V drives |0> -> |1> auxiliary and |0> -> |2> narrow (Section 4.2.8 v). Requires s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12
    for the reduction (asserted by ``valid``)."""
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
    """s_aux << 1 and |delta_aux| << Gamma_10 + Gamma_12 (Section 4.2.8 vii), taken as below 0.1 and 0.1 of the total."""
    total = gamma_10_rad_s + gamma_12_rad_s
    s_aux = (omega_aux_rad_s**2 / 2.0) / (delta_aux_rad_s**2 + (total / 2.0) ** 2)
    return s_aux < 0.1 and abs(delta_aux_rad_s) < 0.1 * total


def effective_two_level_rate(omega_rad_s: float, gamma_coherence_rad_s: float, delta_rad_s: float) -> float:
    """W(Delta) = (Omega^2/2) gamma'/(gamma'^2 + Delta^2): the weak-drive excitation rate of a line with coherence decay gamma'
    (reduces to Omega^2 Gamma/(Gamma^2 + 4 Delta^2) at gamma' = Gamma/2)."""
    return 0.5 * omega_rad_s**2 * gamma_coherence_rad_s / (gamma_coherence_rad_s**2 + delta_rad_s**2)


# ---- V-system diffusion (Marzoli Eq. B4) ---------------------------------------------------------------------------------


def v_system_diffusion(
    s_10: float,
    rho_00: float,
    gamma_12_rad_s: float,
    gamma_10_rad_s: float,
    eta_10: float,
    eta_12: float,
    alpha: float,
) -> float:
    """D = (s_10/4) rho_00 [Gamma_12 (eta_10^2 + alpha eta_12^2) + Gamma_10 eta_10^2 (1 + alpha)] (Marzoli Eq. B4; Section 9.3).

    The half-rate emission diffusion (alpha/2) sum rho_ii Gamma_ij eta_ij^2 plus the absorption recoil eta_10^2 per excitation,
    with rho_11 = (s_10/2) rho_00 for the weakly driven fast level; alpha multiplies the emitted photons only.
    """
    return (
        s_10
        / 4.0
        * rho_00
        * (gamma_12_rad_s * (eta_10**2 + alpha * eta_12**2) + gamma_10_rad_s * eta_10**2 * (1.0 + alpha))
    )


def half_rate_emission_diffusion(terms: list[tuple[float, float, float, float]]) -> float:
    """D_em = (1/2) sum_i rho_ii Gamma_ij alpha_ij eta_ij^2 over (population, rate, alpha, eta) channels (Section 4.2.8 ii);
    it multiplies Cirac's half-rate dissipator, so the phonon heating rate is 2 D."""
    return 0.5 * sum(p * g * a * e * e for p, g, a, e in terms)


def absorption_diffusion(excitation_rate_per_s: float, eta_abs: float) -> float:
    """D_abs = (1/2) R_exc eta_abs^2: the absorbed photon's recoil, entering through the Hamiltonian's e^{i k x}, never alpha-weighted."""
    return 0.5 * excitation_rate_per_s * eta_abs**2


# ---- Lambda system with a repumper (Cirac 1992 Eq. 58) -------------------------------------------------------------------


def lambda_repumper_nbar(beta: float, nu_rad_s: float, delta_rad_s: float, omega_phi_rad_s: float) -> float:
    """<n> = (1/2)[Omega_phi^2/(4 Delta nu) + 4 Delta nu/(beta Omega_phi^2)] - 1/2 inside R << nu << Gamma << Delta (Cirac Eq. 58)."""
    if not 0.0 < beta <= 1.0:
        raise ValueError("the branching fraction beta lies in (0, 1]")
    x = omega_phi_rad_s**2 / (4.0 * delta_rad_s * nu_rad_s)
    return 0.5 * (x + 1.0 / (beta * x)) - 0.5


def lambda_repumper_minimum(beta: float, nu_rad_s: float, delta_rad_s: float) -> tuple[float, float]:
    """(Omega_phi at the minimum, <n>_min) = (sqrt(4 Delta nu/sqrt(beta)), 1/sqrt(beta) - 1/2): 0.554 at beta = 0.9 (Omega_phi = 2.05)
    and 2.66 at beta = 0.1 (3.56) for nu = 0.1, Delta = 10 (Section 9.3 "Lambda system with repumper")."""
    omega = math.sqrt(4.0 * delta_rad_s * nu_rad_s / math.sqrt(beta))
    return omega, 1.0 / math.sqrt(beta) - 0.5


# ---- repump recoil per cycle (Che 2017 Eq. 6; Section 9.15) --------------------------------------------------------------


def offresonant_scattering_rate_per_s(s: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """R_sc = s Gamma^3/(8 Delta^2): Che's far-detuned scattering rate of the repump light (1.7287e4 s^-1 for 25Mg+)."""
    return s * gamma_rad_s**3 / (8.0 * delta_rad_s**2)


def repump_recoil_quanta(
    t_repump_s: float, s: float, gamma_rad_s: float, eta: float, delta_hf_rad_s: float
) -> float:
    """Delta nbar = t s Gamma^3 eta^2/(4 Delta_HF^2) = R_sc t (2 eta^2): 0.031117 for Che's 25Mg+ (t = 10 us, s = 1, Gamma = 2 pi x 41.3 MHz,
    eta = 0.3, Delta_HF = 2 pi x 1.789 GHz); ordinary Hz fed in return 0.0049524, low by exactly 2 pi (Section 9.15)."""
    return t_repump_s * s * gamma_rad_s**3 * eta**2 / (4.0 * delta_hf_rad_s**2)


__all__ = [
    "Configuration",
    "EffectiveTwoLevel",
    "absorption_diffusion",
    "bose_occupation",
    "doppler_force_energy_j",
    "doppler_force_minimum_energy_j",
    "doppler_force_n",
    "doppler_force_nbar",
    "doppler_force_optimum_detuning",
    "doppler_friction_kg_per_s",
    "doppler_limit_nbar",
    "doppler_temperature_k",
    "effective_two_level",
    "effective_two_level_rate",
    "effective_two_level_valid",
    "half_rate_emission_diffusion",
    "lamb_dicke_parameter",
    "lambda_repumper_minimum",
    "lambda_repumper_nbar",
    "lorentzian_scattering_rate",
    "offresonant_scattering_rate_per_s",
    "repump_recoil_quanta",
    "sideband_cooling_rate_n",
    "sideband_floor",
    "stenholm_coefficients",
    "two_level_excited_population",
    "v_system_diffusion",
    "x0_m",
]
