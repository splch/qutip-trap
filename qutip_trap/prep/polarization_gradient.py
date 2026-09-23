"""Polarization-gradient (Sisyphus) cooling: the analytic lin-perp-lin model of Joshi et al. 2020 and a Lindblad model.

The analytic prefactors (1/3, 1/9, 16/9, 2/9) hold for j_g = 1/2 <-> j_e = 1/2 only; cooling needs blue detuning.
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
"""cos^2 2phi below which a static-gradient ion counts as not cooled (W is proportional to cos^2 2phi)."""


class UncooledPhaseError(CoolingError):
    """An ion sits at a node of the polarization gradient (cos 2phi = 0) and is not cooled."""


def joshi_saturation(omega_stretched_rad_s: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """s = (Omega^2/2)/(Gamma^2/4 + Delta^2) per beam, Omega referenced to the S1/2-P3/2 stretched transition."""
    return (omega_stretched_rad_s**2 / 2.0) / (gamma_rad_s**2 / 4.0 + delta_rad_s**2)


def saturation_bridge(s0_on_resonance: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """s_Joshi = (3/2) s0 (Gamma/2)^2/((Gamma/2)^2 + Delta^2) from the on-resonance s0 = I/I_sat of Ejtemaee and Haljan
    2017 (derived here; neither paper prints it)."""
    g2 = (gamma_rad_s / 2.0) ** 2
    return 1.5 * s0_on_resonance * g2 / (g2 + delta_rad_s**2)


def xi_depth(delta_rad_s: float, s_joshi: float, omega_mode_rad_s: float) -> float:
    """xi = Delta s/(3 omega): the light-shift modulation amplitude over the mode frequency (a depth equal to the trap
    frequency is xi = 1/2); Delta > 0 required."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning (Section 4.2.4)")
    if omega_mode_rad_s <= 0.0:
        raise ValueError("the mode frequency is positive")
    return delta_rad_s * s_joshi / (3.0 * omega_mode_rad_s)


def saturation_for_xi(xi: float, delta_rad_s: float, omega_mode_rad_s: float) -> float:
    """The inverse of :func:`xi_depth`: s = 3 omega xi/Delta."""
    return 3.0 * omega_mode_rad_s * xi / delta_rad_s


def xi_from_d1_sigma_rabi(
    omega_d1_sigma_rad_s: float, delta_rad_s: float, gamma_rad_s: float, omega_mode_rad_s: float
) -> float:
    """xi from the D1 sigma Rabi frequency of one beam's full amplitude, whose light shift
    Omega_1^2 Delta/(4 (Delta^2 + Gamma^2/4)) is (1/3) Delta s."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning (Section 4.2.4)")
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
    """(H_carr, H_sb) = ((alpha/3) eta^2 Gamma s (1 - sin^2 2phi), (1/3) eta^2 Gamma s (1 + sin^2 2phi))."""
    s2 = math.sin(2.0 * phi_rad) ** 2
    return alpha / 3.0 * eta**2 * gamma_rad_s * s * (1.0 - s2), eta**2 * gamma_rad_s * s * (1.0 + s2) / 3.0


def cooling_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi (Joshi et al. 2020 Eqs. 7-10), quadratic in s through xi."""
    return 16.0 / 9.0 * eta**2 * gamma_rad_s * s * xi * math.cos(2.0 * phi_rad) ** 2


def heating_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi)."""
    c = math.cos(2.0 * phi_rad)
    return 2.0 / 9.0 * eta**2 * gamma_rad_s * s * (8.0 * xi**2 * c**4 + 2.0 + math.sin(2.0 * phi_rad) ** 2)


def fixed_phase_nbar(xi: float, phi_rad: float = 0.0, *, threshold: float = COS2_THRESHOLD) -> float:
    """H(phi)/W(phi) - 1/2 at one gradient phase (xi + 1/(4 xi) - 1/2 at phi = 0, minimum 1/2 at xi = 1/2); raises
    UncooledPhaseError at a node of the gradient."""
    c2 = math.cos(2.0 * phi_rad) ** 2
    if c2 < threshold:
        raise UncooledPhaseError(
            f"|cos 2phi|^2 = {c2:.2e} < {threshold:.0e}: the ion sits at a node of the polarization gradient and is not cooled"
        )
    # eta^2 Gamma s cancels between H and W
    w = 16.0 / 9.0 * xi * c2
    h = 2.0 / 9.0 * (8.0 * xi**2 * c2 * c2 + 2.0 + (1.0 - c2))
    return h / w - 0.5


def fixed_phase_minimum() -> tuple[float, float]:
    """(xi, <n_0>) at the fixed-phase minimum: (1/2, 1/2)."""
    return 0.5, 0.5


def phase_averaged_nbar(xi: float) -> float:
    """<H>_phi/<W>_phi - 1/2 = (3/4) xi + 5/(8 xi) - 1/2: valid for a gradient moving fast against W
    (W < delta < omega_z), never for a static gradient."""
    if xi <= 0.0:
        raise CoolingError("xi > 0 (blue detuning) is required")
    return 0.75 * xi + 5.0 / (8.0 * xi) - 0.5


def phase_averaged_minimum() -> tuple[float, float]:
    """(xi, <n>) at the phase-averaged minimum: (sqrt(5/6), sqrt(15/8) - 1/2)."""
    return math.sqrt(5.0 / 6.0), math.sqrt(15.0 / 8.0) - 0.5


def multi_ion_fit_offset() -> float:
    """n_0 = sqrt(15/8) = min <n> + 1/2 of the multi-ion fitting model (Joshi et al. 2020 Eq. 13)."""
    return math.sqrt(15.0 / 8.0)


def static_gradient_nbar(
    xi: float, phases_rad: Sequence[float], *, threshold: float = COS2_THRESHOLD
) -> tuple[float, ...]:
    """Each ion's own steady state H(phi_i)/W(phi_i) - 1/2 under a static gradient (raises near a node)."""
    return tuple(fixed_phase_nbar(xi, float(p), threshold=threshold) for p in phases_rad)


def static_gradient_mean_nbar(
    xi: float, phases_rad: Sequence[float], *, threshold: float = COS2_THRESHOLD
) -> float:
    """The mean of the ions' own occupations H(phi_i)/W(phi_i) - 1/2 under a static gradient (occupations average, not
    rates); raises UncooledPhaseError when any ion sits at a node."""
    values = static_gradient_nbar(xi, phases_rad, threshold=threshold)
    if not values:
        raise ValueError("a static-gradient average needs at least one ion phase")
    return float(sum(values) / len(values))


def moving_gradient_window(cooling_rate_rad_s: float, beat_rad_s: float, omega_mode_rad_s: float) -> bool:
    """W < delta < omega_z: the inter-beam frequency difference must outrun the cooling and stay below the trap frequency."""
    return cooling_rate_rad_s < beat_rad_s < omega_mode_rad_s


def three_axis_lamb_dicke(
    k_rad_per_m: float,
    mass_kg: float,
    mode_freqs_hz: Sequence[float],
    projections: Sequence[float] = (0.5, 0.5, 1.0 / math.sqrt(2.0)),
) -> tuple[float, ...]:
    """eta_i = projection_i k sqrt(hbar/(2 m omega_i)); the projections (default: the 171Yb+ beam of Ejtemaee and Haljan
    2017) must square-sum to one."""
    from qutip_trap.units import HBAR_J_S

    if not math.isclose(sum(p * p for p in projections), 1.0, rel_tol=1e-9):
        raise ValueError("the beam's direction cosines must square to one")
    return tuple(
        float(p) * k_rad_per_m * math.sqrt(HBAR_J_S / (2.0 * mass_kg * TWO_PI * float(f)))
        for p, f in zip(projections, mode_freqs_hz)
    )


@dataclass(frozen=True)
class PolarizationGradientLevelC:
    """The four-level-plus-mode Lindblad model as a Bloch build, with the parameters the analytic model takes."""

    model: BlochModel
    omega_d1_sigma_rad_s: float
    """The D1 sigma Rabi frequency of one beam's full amplitude (sqrt 2 times the builder's coupling of the beam)."""
    delta_rad_s: float
    gamma_rad_s: float
    eta: float
    """k x0 of one beam on the mode."""
    xi: float
    phi_rad: float

    @property
    def n_collapse_operators(self) -> int:
        return len(self.model.build.c_ops)


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
    """The lin-perp-lin pair along B as a level-C model on S1/2 and P1/2 with the minimal recoil kernel; beam phases
    (2 phi, 0) put the gradient phase phi at the origin."""
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
    """A counter-propagating pair along z with polarizations x and y (the Lindblad build ignores ``beat_hz``)."""
    a = Beam(wavelength_m, (0.0, 0.0, 1.0), (1.0 + 0j, 0j, 0j), waist_m, power_w, (0.0, 0.0, 0.0))
    b = Beam(wavelength_m, (0.0, 0.0, -1.0), (0j, 1.0 + 0j, 0j), waist_m, power_w, (0.0, 0.0, 0.0))
    return PolGradientBeams(a, b, detuning_hz, beat_hz, phase_rad, "jg12_je12")


__all__ = [
    "COS2_THRESHOLD",
    "ISOTROPIC_ALPHA",
    "PolarizationGradientLevelC",
    "UncooledPhaseError",
    "cooling_rate_per_s",
    "detailed_balance_populations",
    "fixed_phase_minimum",
    "fixed_phase_nbar",
    "heating_rate_per_s",
    "joshi_saturation",
    "lin_perp_lin_pair",
    "moving_gradient_window",
    "multi_ion_fit_offset",
    "phase_averaged_minimum",
    "phase_averaged_nbar",
    "polarization_gradient_model",
    "potentials_rad_s",
    "pumping_rates_per_s",
    "recoil_heating_terms",
    "saturation_bridge",
    "saturation_for_xi",
    "static_gradient_mean_nbar",
    "static_gradient_nbar",
    "three_axis_lamb_dicke",
    "xi_depth",
    "xi_from_d1_sigma_rabi",
]
