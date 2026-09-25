"""Polarization-gradient (Sisyphus) cooling (PLAN.md Section 4.2.4; Joshi et al. 2020; Ejtemaee and Haljan 2017).

Two models with no shared coefficients:

1. Joshi's analytic j_g = 1/2 <-> j_e = 1/2 lin-perp-lin model, two counter-propagating beams along the quantization axis.
   With s the single-beam saturation parameter (Omega^2/2)/(Gamma^2/4 + Delta^2) of the S1/2-P3/2 stretched transition,
   Delta > 0 and phi the gradient phase at the trap centre: U_+- = U_trap + (hbar/3) Delta s -+ (hbar/3) Delta s
   sin(2kz + 2phi); pumping rates Gamma_{+- -> -+} = (1/9) Gamma s (1 -+ sin(2kz + 2phi)), fastest out of a state at its
   own maximum; the depth xi = Delta s/(3 omega) ("depth equals the trap frequency" is xi = 1/2); recoil heating with the
   isotropic alpha = 1/3. Linearized in kz, W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi and
   H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi), steady state H/W - 1/2: at fixed phi = 0,
   xi + 1/(4 xi) - 1/2 (minimum 1/2 at xi = 1/2); phase-averaged (a gradient moving fast against W inside
   W < delta < omega), (3/4) xi + 5/(8 xi) - 1/2 (minimum sqrt(15/8) - 1/2 at xi = sqrt(5/6)). A static gradient sampled
   by ions at distinct phi gives each ion its own H/W - 1/2, and W vanishes at cos 2phi = 0, which raises.
2. The Lindblad layer: the S1/2 and P1/2 Zeeman manifolds times one mode, the two beams along B decomposed into the
   sigma+- standing waves by the multi-level builder, k z = eta (a + a^dagger) kept exact and the three-class recoil
   kernel per decay channel (``polarization_gradient_model``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qutip_trap.dynamics.multilevel import ModeSpec, MultiLevelOptions
from qutip_trap.light.beams import Beam, PolGradientBeams
from qutip_trap.light.bloch import BlochModel, CoolingError
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.units import TWO_PI

ISOTROPIC_ALPHA = 1.0 / 3.0
"""The recoil angular factor of the analytic model's heating terms: the isotropic <cos^2 theta>."""

COS2_THRESHOLD = 1e-2
"""|cos 2phi|^2 below which a static-gradient ion is not cooled (W is proportional to cos^2 2phi)."""


class UncooledPhaseError(CoolingError):
    """An ion sits at a node of the polarization gradient (cos 2phi = 0) and is not cooled."""


def saturation_bridge(s0_on_resonance: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """s_Joshi = (3/2) s0 (Gamma/2)^2/((Gamma/2)^2 + Delta^2) from Ejtemaee's on-resonance s0 = I/I_sat [background]."""
    g2 = (gamma_rad_s / 2.0) ** 2
    return 1.5 * s0_on_resonance * g2 / (g2 + delta_rad_s**2)


def xi_depth(delta_rad_s: float, s_joshi: float, omega_mode_rad_s: float) -> float:
    """xi = Delta s/(3 omega): the light-shift modulation amplitude over the mode frequency; Delta > 0 required."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning")
    if omega_mode_rad_s <= 0.0:
        raise ValueError("the mode frequency is positive")
    return delta_rad_s * s_joshi / (3.0 * omega_mode_rad_s)


def xi_from_d1_sigma_rabi(
    omega_d1_sigma_rad_s: float, delta_rad_s: float, gamma_rad_s: float, omega_mode_rad_s: float
) -> float:
    """xi from the D1 sigma Rabi frequency one beam's full amplitude drives: the ac Stark amplitude
    Omega_1^2 Delta/(4 (Delta^2 + Gamma^2/4)) equals (1/3) Delta s with Omega_1 = sqrt(2/3) Omega_stretched."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning")
    shift = omega_d1_sigma_rad_s**2 * delta_rad_s / (4.0 * (delta_rad_s**2 + gamma_rad_s**2 / 4.0))
    return shift / omega_mode_rad_s


def potentials_rad_s(
    z_m: float, k_rad_per_m: float, phi_rad: float, delta_rad_s: float, s: float
) -> tuple[float, float]:
    """(U_+, U_-)/hbar without the trap term: (1/3) Delta s -+ (1/3) Delta s sin(2kz + 2phi)."""
    base = delta_rad_s * s / 3.0
    mod = base * math.sin(2.0 * k_rad_per_m * z_m + 2.0 * phi_rad)
    return base - mod, base + mod


def pumping_rates_per_s(
    z_m: float, k_rad_per_m: float, phi_rad: float, gamma_rad_s: float, s: float
) -> tuple[float, float]:
    """(Gamma_{+ -> -}, Gamma_{- -> +}) = (1/9) Gamma s (1 -+ sin(2kz + 2phi))."""
    base = gamma_rad_s * s / 9.0
    mod = math.sin(2.0 * k_rad_per_m * z_m + 2.0 * phi_rad)
    return base * (1.0 - mod), base * (1.0 + mod)


def detailed_balance_populations(z_m: float, k_rad_per_m: float, phi_rad: float) -> tuple[float, float]:
    """p_+- = (1/2)(1 +- sin(2kz + 2phi)), the stationary internal populations of the z-substituted rates."""
    mod = math.sin(2.0 * k_rad_per_m * z_m + 2.0 * phi_rad)
    return 0.5 * (1.0 + mod), 0.5 * (1.0 - mod)


def recoil_heating_terms(
    eta: float, gamma_rad_s: float, s: float, phi_rad: float, alpha: float = ISOTROPIC_ALPHA
) -> tuple[float, float]:
    """(H_carr, H_sb) = ((alpha/3) eta^2 Gamma s (1 - sin^2 2phi), (1/3) eta^2 Gamma s (1 + sin^2 2phi)); with
    alpha = 1/3 their sum is the non-xi^2 part of H(phi)."""
    s2 = math.sin(2.0 * phi_rad) ** 2
    return alpha / 3.0 * eta**2 * gamma_rad_s * s * (1.0 - s2), eta**2 * gamma_rad_s * s * (1.0 + s2) / 3.0


def cooling_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi (Joshi Eqs. 7-10), proportional to s^2 through xi."""
    return 16.0 / 9.0 * eta**2 * gamma_rad_s * s * xi * math.cos(2.0 * phi_rad) ** 2


def heating_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi)."""
    c = math.cos(2.0 * phi_rad)
    return 2.0 / 9.0 * eta**2 * gamma_rad_s * s * (8.0 * xi**2 * c**4 + 2.0 + math.sin(2.0 * phi_rad) ** 2)


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


def static_gradient_nbar(
    xi: float, phases_rad: Sequence[float], *, threshold: float = COS2_THRESHOLD
) -> tuple[float, ...]:
    """Each ion's own steady state H(phi_i)/W(phi_i) - 1/2 under a static gradient; raises UncooledPhaseError for an ion
    near a node."""
    return tuple(fixed_phase_nbar(xi, float(p), threshold=threshold) for p in phases_rad)


def static_gradient_mean_nbar(
    xi: float, phases_rad: Sequence[float], *, threshold: float = COS2_THRESHOLD
) -> float:
    """The chain's mean occupation under a static gradient: the average of the per-ion occupations (each ion reaches its
    own steady state), not the phase-averaged rate ratio of a moving gradient."""
    values = static_gradient_nbar(xi, phases_rad, threshold=threshold)
    if not values:
        raise ValueError("a static-gradient average needs at least one ion phase")
    return float(sum(values) / len(values))


def moving_gradient_window(cooling_rate_rad_s: float, beat_rad_s: float, omega_mode_rad_s: float) -> bool:
    """W < delta < omega_z: the inter-beam frequency difference must outrun the cooling and stay below the trap
    frequency."""
    return cooling_rate_rad_s < beat_rad_s < omega_mode_rad_s


@dataclass(frozen=True)
class PolarizationGradientLevelC:
    """The Lindblad model as a Bloch build, with the parameters the analytic model takes."""

    model: BlochModel
    omega_d1_sigma_rad_s: float
    """The D1 sigma Rabi frequency one beam's full amplitude drives (sqrt 2 times the builder's coupling of one linear
    beam)."""
    delta_rad_s: float
    gamma_rad_s: float
    eta: float
    """k x0 of one beam on the mode."""
    xi: float
    phi_rad: float


def polarization_gradient_model(
    structure: AtomicStructure,
    pair: PolGradientBeams,
    mode: ModeSpec,
    *,
    lower: str,
    upper: str,
    levels: Sequence[str] = ("S1/2", "P1/2"),
    leak: str = "renormalize",
) -> PolarizationGradientLevelC:
    """The lin-perp-lin pair along B as a level-C model: beam phases (2 phi, 0) put the gradient phase phi at the origin;
    the S1/2 and P1/2 manifolds only (the D branch renormalized) and the three-class recoil kernel per decay channel."""
    k_hat = np.asarray(pair.beam_a.k_hat, dtype=float)
    if not np.allclose(np.abs(np.dot(k_hat, structure.b_hat)), 1.0, atol=1e-9):
        raise ValueError(
            "the analytic and Lindblad models both need the beam pair along the quantization axis"
        )
    if not np.allclose(np.abs(np.dot(k_hat, np.asarray(mode.axis))), 1.0, atol=1e-9):
        raise ValueError("the mode must lie along the beam pair (a 1D model)")
    options = MultiLevelOptions(leak=leak, recoil="minimal", beam_phases_rad=(2.0 * pair.phase_rad, 0.0))  # type: ignore[arg-type]
    model = BlochModel(structure, [pair.beam_a, pair.beam_b], levels=levels, mode=mode, options=options)
    b = model.build
    coupling = next(
        (c for c in b.couplings if c.beam == 0 and c.lower == lower and c.upper == upper),
        None,
    )
    if coupling is None:
        raise ValueError(f"beam a does not couple {lower} -> {upper}; check the polarizations against B")
    gamma = b.level_rates_rad_s[b.level_of(upper)]
    omega_1 = math.sqrt(2.0) * abs(coupling.omega_rad_s)
    delta = TWO_PI * pair.detuning_hz
    return PolarizationGradientLevelC(
        model=model,
        omega_d1_sigma_rad_s=omega_1,
        delta_rad_s=delta,
        gamma_rad_s=gamma,
        eta=abs(mode.eta(pair.beam_a.k_vector())),
        xi=xi_from_d1_sigma_rabi(omega_1, delta, gamma, mode.omega_rad_s),
        phi_rad=pair.phase_rad,
    )


def lin_perp_lin_pair(
    wavelength_m: float,
    power_w: float,
    waist_m: float,
    detuning_hz: float,
    *,
    phase_rad: float = 0.0,
    beat_hz: float = 0.0,
) -> PolGradientBeams:
    """A counter-propagating pair along z polarized along x and y; ``beat_hz`` offsets beam b for a moving gradient."""
    a = Beam(wavelength_m, (0.0, 0.0, 1.0), (1.0 + 0j, 0j, 0j), waist_m, power_w, (0.0, 0.0, 0.0))
    b = Beam(wavelength_m, (0.0, 0.0, -1.0), (0j, 1.0 + 0j, 0j), waist_m, power_w, (0.0, 0.0, 0.0))
    return PolGradientBeams(a, b, detuning_hz, beat_hz, phase_rad, "jg12_je12")
