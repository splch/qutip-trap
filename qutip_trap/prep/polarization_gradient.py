"""Polarization-gradient (Sisyphus) cooling (PLAN.md Section 4.2.4; Joshi et al. 2020; Ejtemaee and Haljan 2017; M3).

Two models with no shared coefficients (Section 4.2.4, "why a second model"):

1. The analytic j_g = 1/2 <-> j_e = 1/2 lin-perp-lin model of Joshi et al. 2020, two counter-propagating beams ALONG the
   quantization axis so the ion sees sigma light of constant intensity with a rotating polarization. With s the
   single-beam saturation parameter (Omega^2/2)/(Gamma^2/4 + Delta^2) referenced to the S1/2-P3/2 stretched transition
   (Section 13 "Two saturation parameters"), Delta > 0 and phi the gradient phase at the trap centre:
   U_+- = U_trap + (hbar/3) Delta s -+ (hbar/3) Delta s sin(2kz + 2phi); pumping rates Gamma_{+- -> -+}(z) = (1/9) Gamma s (1 -+ sin(2kz + 2phi))
   (the same sign as the potential, fastest out of a state at its own maximum; the substitution 2phi -> 2kz + 2phi belongs
   to the RATES only); detailed balance p_+- = (1/2)(1 +- sin(2kz + 2phi)); the depth parameter xi = Delta s/(3 omega)
   (the modulation AMPLITUDE over the mode frequency, "depth equals the trap frequency" is xi = 1/2); recoil heating
   H_carr = (alpha/3) eta^2 Gamma s (1 - sin^2 2phi), H_sb = (1/3) eta^2 Gamma s (1 + sin^2 2phi) with the ISOTROPIC alpha = 1/3
   (2/5 fails internal consistency by 5 %); linearized in kz, W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi and
   H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi); d<n>/dt = -W(<n> + 1/2) + H; steady state H/W - 1/2:
   at fixed phi = 0, xi + 1/(4 xi) - 1/2 with minimum exactly 1/2 at xi = 1/2; phase-averaged (a gradient moving fast
   against W inside W < delta < omega), (3/4) xi + 5/(8 xi) - 1/2 with minimum sqrt(15/8) - 1/2 = 0.8693 at xi = sqrt(5/6);
   for a static gradient sampled by ions at distinct phi each ion has its own H/W - 1/2 and W vanishes at |cos 2phi| = 0,
   so the module averages per ion and raises when any |cos 2phi| is below the cooling threshold. Blue detuning is required:
   W ∝ xi ∝ Delta, and on an inverted j_e <= j_g line the most strongly coupled sublevel is the pumping source.
   The prefactors 1/3, 1/9, 16/9, 2/9 and 8 xi^2 are specific to j_g = 1/2 <-> j_e = 1/2 and carry to no other scheme.

2. The Lindblad layer: the four-level atom (S1/2 and P1/2 Zeeman manifolds, no D branch) times one mode, the two circular
   standing waves offset by -+ pi/4 about the common phase (relative phase pi/2, a lambda/8 displacement), coupling
   amplitude sqrt(1/3) = sqrt(2/3) x 1/sqrt 2 (Clebsch-Gordan ratio times each linear beam's projection on one circular
   component), k z = eta (a + a^dagger) kept as the full exponential, and the recoil-resolved dissipator of TWELVE operators
   J_mq = p_mq C_m sqrt(Gamma) e^{-i k_q z} sigma_m^- over four decay channels and three recoil classes k_q in {-k, 0, +k}
   entered INCOHERENTLY with sum_q p_mq^2 = 1 and second moments 2/5 (sigma) and 1/5 (pi) (Section 13 "Recoil kernel
   discretization"; the printed coherent q-sum inflates the decay rate by 2.7856 and 2.3314). The M3a builder produces
   exactly this: two counter-propagating linearly polarized beams along B decompose into the sigma+- standing waves,
   ``recoil='minimal'`` is the three-class kernel per channel, and the beam phases set the gradient phase
   (``polarization_gradient_model``). The 171Yb+ F = 1 -> F' = 0 scheme of Ejtemaee and Haljan (eta = (0.052, 0.053, 0.090),
   rate ∝ s0^2, nbar ~ 1.5-2) shares NO coefficient with the analytic model; only its beam projections and the s0 bridge
   s_Joshi = (3/2) s0 (Gamma/2)^2/((Gamma/2)^2 + Delta^2) [background] are computed here.
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
"""The recoil angular factor of the analytic model's heating terms: the isotropic <cos^2 theta> (Section 4.2.4)."""

COS2_THRESHOLD = 1e-2
"""|cos 2phi| below which a static-gradient ion is not cooled (W ∝ cos^2 2phi): the raise of Section 4.2.4."""


class UncooledPhaseError(CoolingError):
    """An ion sits at a node of the polarization gradient (cos 2phi = 0) and is not cooled (Section 4.2.4)."""


# ---- saturation parameters and the depth parameter ---------------------------------------------------------------------------


def joshi_saturation(omega_stretched_rad_s: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """s = (Omega^2/2)/(Gamma^2/4 + Delta^2): per beam, detuning-suppressed, Omega referenced to the S1/2-P3/2 stretched transition."""
    return (omega_stretched_rad_s**2 / 2.0) / (gamma_rad_s**2 / 4.0 + delta_rad_s**2)


def saturation_bridge(s0_on_resonance: float, gamma_rad_s: float, delta_rad_s: float) -> float:
    """s_Joshi = (3/2) s0 (Gamma/2)^2/((Gamma/2)^2 + Delta^2) from Ejtemaee's on-resonance s0 = I/I_sat [background, derived by the
    Run 5 consolidation and printed by neither paper]; used only to check s0 = 11-15 at 310 MHz lands inside Joshi's s < 0.07."""
    g2 = (gamma_rad_s / 2.0) ** 2
    return 1.5 * s0_on_resonance * g2 / (g2 + delta_rad_s**2)


def xi_depth(delta_rad_s: float, s_joshi: float, omega_mode_rad_s: float) -> float:
    """xi = Delta s/(3 omega): the light-shift modulation amplitude over the mode frequency (Section 13 "Polarization-gradient depth
    parameter"); "depth equals the trap frequency" is xi = 1/2; Delta > 0 required."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning (Section 4.2.4)")
    if omega_mode_rad_s <= 0.0:
        raise ValueError("the mode frequency is positive")
    return delta_rad_s * s_joshi / (3.0 * omega_mode_rad_s)


def saturation_for_xi(xi: float, delta_rad_s: float, omega_mode_rad_s: float) -> float:
    """The inverse: s = 3 omega xi/Delta (Joshi's operating point xi = 1.35 at 210 MHz and 1088 kHz returns s = 0.02098)."""
    return 3.0 * omega_mode_rad_s * xi / delta_rad_s


def xi_from_d1_sigma_rabi(
    omega_d1_sigma_rad_s: float, delta_rad_s: float, gamma_rad_s: float, omega_mode_rad_s: float
) -> float:
    """xi from the actual D1 sigma Rabi frequency one beam's FULL amplitude would drive: the light-shift amplitude is the ac Stark
    shift Omega_1^2 Delta/(4 (Delta^2 + Gamma^2/4)), which equals (1/3) Delta s with s referenced to the D2 stretched line
    (Omega_1 = sqrt(2/3) Omega_stretched); this is what a Lindblad build with only S1/2 and P1/2 can be compared through."""
    if delta_rad_s <= 0.0:
        raise CoolingError("polarization-gradient cooling needs blue detuning (Section 4.2.4)")
    shift = omega_d1_sigma_rad_s**2 * delta_rad_s / (4.0 * (delta_rad_s**2 + gamma_rad_s**2 / 4.0))
    return shift / omega_mode_rad_s


# ---- the analytic model ------------------------------------------------------------------------------------------------------


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
    """(Gamma_{+ -> -}, Gamma_{- -> +}) = (1/9) Gamma s (1 -+ sin(2kz + 2phi)): fastest out of a state at its own potential maximum."""
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
    """(H_carr, H_sb) = ((alpha/3) eta^2 Gamma s (1 - sin^2 2phi), (1/3) eta^2 Gamma s (1 + sin^2 2phi)); with alpha = 1/3 their sum
    is the non-xi^2 part (2/9) eta^2 Gamma s (2 + sin^2 2phi) of H(phi) exactly (Section 9.15)."""
    s2 = math.sin(2.0 * phi_rad) ** 2
    return alpha / 3.0 * eta**2 * gamma_rad_s * s * (1.0 - s2), eta**2 * gamma_rad_s * s * (1.0 + s2) / 3.0


def cooling_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """W(phi) = (16/9) eta^2 Gamma s xi cos^2 2phi (Joshi Eqs. 7-10); ∝ s^2 through xi, which is the measured s0^2 law."""
    return 16.0 / 9.0 * eta**2 * gamma_rad_s * s * xi * math.cos(2.0 * phi_rad) ** 2


def heating_rate_per_s(eta: float, gamma_rad_s: float, s: float, xi: float, phi_rad: float) -> float:
    """H(phi) = (2/9) eta^2 Gamma s (8 xi^2 cos^4 2phi + 2 + sin^2 2phi)."""
    c = math.cos(2.0 * phi_rad)
    return 2.0 / 9.0 * eta**2 * gamma_rad_s * s * (8.0 * xi**2 * c**4 + 2.0 + math.sin(2.0 * phi_rad) ** 2)


def fixed_phase_nbar(xi: float, phi_rad: float = 0.0, *, threshold: float = COS2_THRESHOLD) -> float:
    """H(phi)/W(phi) - 1/2 at one gradient phase; at phi = 0 it is xi + 1/(4 xi) - 1/2 with minimum 1/2 at xi = 1/2 (the phi = 0 VALUE,
    which minimizes the occupation only for xi <~ 0.612 but always maximizes the rate). Raises at a node of the gradient."""
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
    """<H>_phi/<W>_phi - 1/2 = (3/4) xi + 5/(8 xi) - 1/2 (phase averages <cos^2 2phi> = 1/2, <cos^4 2phi> = 3/8, <sin^2 2phi> = 1/2),
    valid for a gradient moving fast against W inside W < delta < omega_z, never for a static gradient sampled by many ions."""
    if xi <= 0.0:
        raise CoolingError("xi > 0 (blue detuning) is required")
    return 0.75 * xi + 5.0 / (8.0 * xi) - 0.5


def phase_averaged_minimum() -> tuple[float, float]:
    """(xi, <n>) at the phase-averaged minimum: (sqrt(5/6), sqrt(15/8) - 1/2) = (0.9128709, 0.8693064)."""
    return math.sqrt(5.0 / 6.0), math.sqrt(15.0 / 8.0) - 0.5


def multi_ion_fit_offset() -> float:
    """n_0 = (3/2) sqrt(5/6) = sqrt(15/8) = 1.369306 = min <n> + 1/2 (Joshi Eq. 13, the multi-ion fitting model; Section 9.15)."""
    return math.sqrt(15.0 / 8.0)


def static_gradient_nbar(
    xi: float, phases_rad: Sequence[float], *, threshold: float = COS2_THRESHOLD
) -> tuple[float, ...]:
    """Each ion's own steady state H(phi_i)/W(phi_i) - 1/2 for a static gradient; raises UncooledPhaseError for an ion near a node
    rather than reporting the moving-gradient 0.87 (Section 4.2.4 [corrected: critique, 2026-09-04])."""
    return tuple(fixed_phase_nbar(xi, float(p), threshold=threshold) for p in phases_rad)


def static_gradient_mean_nbar(
    xi: float, phases_rad: Sequence[float], *, threshold: float = COS2_THRESHOLD
) -> float:
    """The chain's mean occupation under a STATIC gradient: the average of H(phi_i)/W(phi_i) - 1/2 over the ions' actual
    gradient phases (Section 4.2.4, "the module averages H/W - 1/2 over the ions' actual phi").

    This is the per-ion average, not the phase-averaged rate ratio <H>/<W> of the moving gradient
    (:func:`phase_averaged_nbar`): each ion reaches its OWN steady state under a static gradient, so the occupations
    average and not the rates. Raises UncooledPhaseError when any ion sits at a node.
    """
    values = static_gradient_nbar(xi, phases_rad, threshold=threshold)
    if not values:
        raise ValueError("a static-gradient average needs at least one ion phase")
    return float(sum(values) / len(values))


def moving_gradient_window(cooling_rate_rad_s: float, beat_rad_s: float, omega_mode_rad_s: float) -> bool:
    """W < delta < omega_z: the inter-beam frequency difference must outrun the cooling and stay below the trap frequency."""
    return cooling_rate_rad_s < beat_rad_s < omega_mode_rad_s


# ---- geometry of the 171Yb+ scheme (Ejtemaee and Haljan 2017; Section 9.15 "Three-axis Lamb-Dicke triple") -----------------------


def three_axis_lamb_dicke(
    k_rad_per_m: float,
    mass_kg: float,
    mode_freqs_hz: Sequence[float],
    projections: Sequence[float] = (0.5, 0.5, 1.0 / math.sqrt(2.0)),
) -> tuple[float, ...]:
    """eta_i = projection_i k sqrt(hbar/(2 m omega_i)): {1/2, 1/2, 1/sqrt 2} x 2 pi/369.5 nm at (0.790, 0.766, 0.525) MHz gives
    (0.0520, 0.0528, 0.0902) for 171Yb+; the projections square to one."""
    from qutip_trap.units import HBAR_J_S

    if not math.isclose(sum(p * p for p in projections), 1.0, rel_tol=1e-9):
        raise ValueError("the beam's direction cosines must square to one")
    return tuple(
        float(p) * k_rad_per_m * math.sqrt(HBAR_J_S / (2.0 * mass_kg * TWO_PI * float(f)))
        for p, f in zip(projections, mode_freqs_hz)
    )


# ---- the Lindblad layer through the M3a builder ------------------------------------------------------------------------------


@dataclass(frozen=True)
class PolarizationGradientLevelC:
    """The four-level-plus-mode Lindblad model of Section 4.2.4 as a Bloch build, with the parameters the analytic model wants."""

    model: BlochModel
    omega_d1_sigma_rad_s: float
    """The D1 sigma Rabi frequency one beam's full amplitude would drive (sqrt 2 times the builder's coupling of one linear beam)."""
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
    """Build the lin-perp-lin pair along B as a level-C model: two beams with phases (2 phi, 0) so the gradient phase at the origin is
    phi, the S1/2 and P1/2 manifolds only (the D branch renormalized as an instantaneous repump), the minimal (three-class) recoil
    kernel per decay channel, giving 4 channels x 3 classes = 12 collapse operators for a j = 1/2 -> 1/2 line."""
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
    """A counter-propagating pair along z with polarizations x and y (the geometry of both models); the frequency offset of beam b
    by ``beat_hz`` drives the moving gradient (the pair carries it as a parameter; the Lindblad build takes the static case)."""
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
