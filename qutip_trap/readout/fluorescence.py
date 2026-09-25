"""The atomic-rate layer of readout: scattering rate, pumping rates, detected rate, shelving (PLAN.md Sections 8.1, 8.8; M5).

Three conventions are pinned before anything else (Section 8): bright/dark polarity is a property of the readout
SCHEME (:class:`ReadoutScheme`), never of the qubit, because direct fluorescence and shelving invert it within the same
papers; Gamma is an angular population decay rate in s^-1 (the species tables assert Gamma/2 pi against the quoted
linewidth); and the system detection efficiency epsilon_sys enters exactly once, at the conversion from scattered to
detected rate (:meth:`FluorescenceRates.detected`), with the background rate kept separate.

The rates themselves come from the M3a scattering-rate object: :func:`scattering_rate` builds the multi-level
optical-Bloch model of the detection beams (Section 4.2.8) and reads R_o (the photon rate of the CONDITIONAL bright
state), R_d (bright -> dark pumping) and R_b (dark -> bright) from its slow-manifold coarse graining, asserting the
photon rate against the equal-population ceiling n_e/(n_e + n_g) of the closed manifold (Gamma/4 for the 171Yb+
F = 1 -> F' = 0 detection cycle, Section 13). The closed forms of the sources (Noek's and Crain's (Gamma/18, 2/9) form,
Acton's lambda_0, alpha_1, alpha_2 with the angular factors M_1, M_2pi, M_2-, the 4/9 clock-state ceiling) are kept
here as the oracles the exact solve is tested against, never as the rates the simulator runs on; apparatus numbers
(detected rates, efficiencies, backgrounds) enter through :func:`rates_from_detected` and the presets of
:mod:`qutip_trap.readout.presets`, tagged as apparatus data and never as species constants (Section 8.8).

Symbol collisions (Section 13): R_o here is a photon rate in s^-1; eta is the DETECTION efficiency in Acton's
alpha/eta (never the Lamb-Dicke parameter); alpha_1, alpha_2 are leak probabilities per emitted photon (never the recoil
angular factor); tau_D is the detection window in Acton and the shelf lifetime in Myerson, so the code names them
``window_s`` and ``shelf_lifetime_s``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0, j1

from qutip_trap.units import C_M_PER_S, H_J_S, HBAR_J_S

if TYPE_CHECKING:
    from qutip_trap.dynamics.multilevel import MultiLevelOptions
    from qutip_trap.light.beams import Beam
    from qutip_trap.light.bloch import BlochModel
    from qutip_trap.readout.detection import Detector
    from qutip_trap.species.model import Species
    from qutip_trap.species.raman import AtomicStructure

M5 = "milestone M5 (readout/fluorescence.py, PLAN.md Section 8.1)"

ReadoutClass = Literal["bright", "dark", "shelf"]
CLASSES: tuple[ReadoutClass, ...] = ("bright", "dark", "shelf")
"""The three classes of the readout continuous-time Markov chain (Section 8.2): the fluorescing manifold, the dark
ground manifold reached by off-resonant pumping, and the metastable shelf that decays back to bright."""

ALL_LINES = "all"
"""``line=ALL_LINES`` sums the photon rate over every non-sink decay line of the build: the total scattering rate
W(Delta) of Section 4.2.2, a DIAGNOSTIC and not a detected rate, because epsilon_sys carries exactly one
epsilon_filter and the filter passes one wavelength (Section 8.1, "Detected rate and background")."""


def saturation_ceiling(n_ground: int, n_excited: int) -> float:
    """P_f^max = n_e/(n_e + n_g), the equal-population ceiling of a closed manifold (Section 13; Berkeland Eqs. 13-14, 21)."""
    if n_ground <= 0 or n_excited <= 0:
        raise ValueError("a closed manifold has at least one ground and one excited state")
    return n_excited / (n_excited + n_ground)


@dataclass(frozen=True)
class DarkStateReport:
    """Returned by the CPT solve so the ceiling is never assumed (Section 8.1)."""

    n_ground: int
    n_excited: int
    ceiling: float
    """n_e/(n_e + n_g); the photon rate must satisfy Gamma*P_f <= Gamma*ceiling."""
    dark_dimension: int
    dark_basis: np.ndarray
    """(dark_dimension, n_ground) complex amplitudes c_m, STRAIGHT pairing (Section 13)."""
    delta_over_omega: float
    theta_be_deg: float
    """Angle between the linear polarization and B; optimum 54.7356 degrees."""
    raman_zero_margin_hz: float | None

    def __post_init__(self) -> None:
        expected = saturation_ceiling(self.n_ground, self.n_excited)
        if abs(self.ceiling - expected) > 1e-12:
            raise ValueError(f"ceiling must be n_e/(n_e + n_g) = {expected}, got {self.ceiling}")


# ---- closed forms of Section 8.1 (oracles for the exact solve) --------------------------------------------------------------


def saturation_intensity_w_m2(gamma_partial_rad_s: float, wavelength_m: float) -> float:
    """I_sat = pi h c Gamma/(3 lambda^3) with the ANGULAR partial rate (Section 13, "Saturation intensity").

    87Rb D2 (Gamma = 2 pi x 6.0666 MHz, 780.241 nm) gives 1.669 mW/cm^2 (Steck); 171Yb+ 369.5 nm with the partial rate
    2 pi x 19.62 MHz gives 50.83 mW/cm^2 (Section 9.12). Acton's printed form carries hbar for h and is 2 pi too small.
    """
    if gamma_partial_rad_s <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("Gamma and the wavelength are positive")
    return math.pi * H_J_S * C_M_PER_S * gamma_partial_rad_s / (3.0 * wavelength_m**3)


def yb171_bright_rate_closed(s_o: float, gamma_rad_s: float, detuning_rad_s: float = 0.0) -> float:
    """R_o = (Gamma/18) s_o/[1 + (2/9) s_o + (2 Delta/Gamma)^2] for the 171Yb+ F = 1 -> F' = 0 cycle (Section 8.1).

    ``s_o = I/I_sat = 2 Omega^2/Gamma^2`` with the FULL-LINE Rabi frequency and the two-level I_sat of the partial
    rate; saturates at Gamma/4 (three ground plus one excited state), not Gamma/2, with half maximum at s_o = 4.5.
    The prefactor Gamma/18 = Gamma x 1/2 x 1/3 x 1/3 carries the hyperfine reduced element |<F=1||d||F'=0>|^2 =
    |<J||d||J'>|^2/3 and the magic-angle polarization share |eps_q|^2 = 1/3 of each ground sublevel (Section 4.5.2;
    derivation audit 2026-09-04). The exact four-level steady state sits 0.15 to 0.5 % BELOW this form at its optimum
    field (M3a finding, ``check_bloch.py``), so this is the CEILING of the exact rate, used as an oracle only.
    """
    if s_o < 0.0 or gamma_rad_s <= 0.0:
        raise ValueError("s_o is non-negative and Gamma positive")
    return (gamma_rad_s / 18.0) * s_o / (1.0 + (2.0 / 9.0) * s_o + (2.0 * detuning_rad_s / gamma_rad_s) ** 2)


def noek_saturation_from_s_o(s_o: float) -> float:
    """Noek's s = 2 Omega^2/Gamma^2 in his (Gamma/6, 2/3) form is s_o/3 (Section 8.8: two saturation parameters, never aliased)."""
    return s_o / 3.0


def crain_saturation_from_s_o(s_o: float) -> float:
    """Crain's s~ = I/(229 mW/cm^2) in the (Gamma/4, 1) form is (2/9) s_o (Section 8.1)."""
    return 2.0 * s_o / 9.0


def yb171_bright_rate_noek_form(s: float, gamma_rad_s: float, detuning_rad_s: float = 0.0) -> float:
    """Noek 2013 Eq. 1 as printed, (Gamma/6) s/[1 + (2/3) s + (2 Delta/Gamma)^2]: the same result written for s = s_o/3."""
    return (gamma_rad_s / 6.0) * s / (1.0 + (2.0 / 3.0) * s + (2.0 * detuning_rad_s / gamma_rad_s) ** 2)


def yb171_bright_rate_crain_form(s_tilde: float, gamma_rad_s: float, detuning_rad_s: float = 0.0) -> float:
    """The (Gamma/4) s~/[1 + s~ + (2 Delta/Gamma)^2] form with s~ = (2/9) s_o, saturating explicitly at Gamma/4."""
    return (gamma_rad_s / 4.0) * s_tilde / (1.0 + s_tilde + (2.0 * detuning_rad_s / gamma_rad_s) ** 2)


def yb171_leakage_rates_closed(
    s_o: float, gamma_rad_s: float, delta_hfp_rad_s: float, delta_hfs_rad_s: float
) -> tuple[float, float]:
    """(R_d, R_b) of Section 8.1 with the prefactors the angular algebra settles in Noek's favour (M3a finding).

    R_d = (2/3)(1/3)(Gamma/2) s (Gamma/(2 Delta_HFP))^2 is the bright state's off-resonant F = 1 -> F' = 1 channel with
    its 1/3 branching into F = 0; R_b = (2/3)(Gamma/2) s (Gamma/(2(Delta_HFP + Delta_HFS)))^2 the dark state's only
    dipole-allowed channel; both with s = s_o/3, linear in intensity with no saturation denominator (Section 8.1).
    """
    s = noek_saturation_from_s_o(s_o)
    r_d = (2.0 / 3.0) * (1.0 / 3.0) * (gamma_rad_s / 2.0) * s * (gamma_rad_s / (2.0 * delta_hfp_rad_s)) ** 2
    r_b = (
        (2.0 / 3.0)
        * (gamma_rad_s / 2.0)
        * s
        * (gamma_rad_s / (2.0 * (delta_hfp_rad_s + delta_hfs_rad_s))) ** 2
    )
    return r_d, r_b


def crain_dark_pumping_form(s_o: float, gamma_rad_s: float, delta_hfp_rad_s: float) -> float:
    """Crain 2019 Eq. 7 as printed, (1/3)(Gamma/2) s (Gamma/2 Delta_HFP)^2 with s = s_o/3: the prefactor pair the exact
    solve refutes (Section 8.8; the negative control of the leakage test)."""
    s = noek_saturation_from_s_o(s_o)
    return (1.0 / 3.0) * (gamma_rad_s / 2.0) * s * (gamma_rad_s / (2.0 * delta_hfp_rad_s)) ** 2


def yb171_leakage_ratio(delta_hfp_rad_s: float, delta_hfs_rad_s: float) -> float:
    """R_b/R_d = 3 (Delta_HFP/(Delta_HFP + Delta_HFS))^2 = 3/49 at the 171Yb+ splittings, against 16.4/341 = 0.048 measured."""
    return 3.0 * (delta_hfp_rad_s / (delta_hfp_rad_s + delta_hfs_rad_s)) ** 2


@dataclass(frozen=True)
class ActonAngularFactors:
    """Acton 2006 Eqs. 9, 12, 13 normalized so the cycling line has unit strength (Section 8.8 [corrected])."""

    M1: float
    """Dark (|I-1/2, I-1/2>) -> bright leak through P3/2 |I+1/2, I+1/2>: 4I(3 + 2I)/(9(1 + 2I)^2)."""
    M2pi: float
    """Bright -> dark through the pi impurity: 4I/(9 + 18I)."""
    M2minus: float
    """Bright -> dark through the sigma- impurity: 16I/(9(1 + 2I)^3)."""


def acton_angular_factors(nuclear_spin: float) -> ActonAngularFactors:
    """The closed-form Clebsch-Gordan combinations of the sigma+ stretch-state scheme; 2/9, 1/9, 1/9 at I = 1/2.

    The paper's printed normalization gives M_1 = 8/9 unless C -> C/2; the code pins the normalization by reproducing
    (2/9, 1/9, 1/9) at I = 1/2 (Section 8.1 [corrected]).
    """
    i = float(nuclear_spin)
    if i <= 0.0:
        raise ValueError("the stretch-state scheme needs a nonzero nuclear spin")
    return ActonAngularFactors(
        M1=4.0 * i * (3.0 + 2.0 * i) / (9.0 * (1.0 + 2.0 * i) ** 2),
        M2pi=4.0 * i / (9.0 + 18.0 * i),
        M2minus=16.0 * i / (9.0 * (1.0 + 2.0 * i) ** 3),
    )


ACTON_P12_FACTORS: tuple[float, float] = (2.0 / 9.0, 2.0 / 9.0)
"""M_1' = M_2' = 1/3 x (1/3 + 1/3) = 2/9 for the I = 1/2 clock-state scheme through P1/2 (Acton Eqs. 15-16), where
alpha_2' > alpha_1' always because Delta_1' = omega_HFS + omega_HFP' exceeds Delta_2' = omega_HFP'."""


def acton_mean_count(
    window_s: float, efficiency: float, s: float, gamma_rad_s: float, detuning_rad_s: float = 0.0
) -> float:
    """lambda_0 = tau_D eta (s gamma/2)/(1 + s + (2 delta/gamma)^2), Acton Eq. 7: the mean detected count of a bright ion
    on a closed two-level cycling line (s = I/I_sat exact there, Section 4.5.2)."""
    return (
        window_s
        * efficiency
        * (s * gamma_rad_s / 2.0)
        / (1.0 + s + (2.0 * detuning_rad_s / gamma_rad_s) ** 2)
    )


def acton_leak_dark_to_bright(
    m1: float, s: float, gamma_rad_s: float, detuning_rad_s: float, delta1_rad_s: float
) -> float:
    """alpha_1 = M_1 (1 + s + (2 delta/gamma)^2)(gamma/(2 Delta_1))^2, the dark -> bright leak probability per emitted
    photon (Acton Eq. 8), Delta_1 = omega_HFS - omega_HFP for the P3/2 scheme, omega_HFS + omega_HFP' through P1/2."""
    return (
        m1 * (1.0 + s + (2.0 * detuning_rad_s / gamma_rad_s) ** 2) * (gamma_rad_s / (2.0 * delta1_rad_s)) ** 2
    )


def acton_leak_bright_to_dark(
    factors: ActonAngularFactors,
    p_pi: float,
    p_minus: float,
    s: float,
    gamma_rad_s: float,
    detuning_rad_s: float,
    delta2_rad_s: float,
) -> float:
    """alpha_2 = (1 + s + (2 delta/gamma)^2)(gamma/(2 Delta_2))^2 (M_2pi P_pi + M_2- P_-)/(1 - P_pi - P_-), Acton Eq. 11:
    the bright -> dark leak through the polarization impurities P_pi, P_- of the sigma+ beam, Delta_2 = omega_HFP."""
    if not 0.0 <= p_pi + p_minus < 1.0:
        raise ValueError("the impurity fractions P_pi + P_- lie in [0, 1)")
    impurity = (factors.M2pi * p_pi + factors.M2minus * p_minus) / (1.0 - p_pi - p_minus)
    return (
        (1.0 + s + (2.0 * detuning_rad_s / gamma_rad_s) ** 2)
        * (gamma_rad_s / (2.0 * delta2_rad_s)) ** 2
        * impurity
    )


def acton_clock_state_ceiling(gamma_rad_s: float, omega_hfp_rad_s: float) -> float:
    """F_max = 1 - (4/9)(gamma/(2 omega_HFP))^2 for direct clock-state readout through P3/2 (Acton Eq. 20 [corrected]):
    the 4/9 is 2 x (1/3) x (2/3) of squared Clebsch-Gordan factors and event multiplicity; 99.9375 % for 111Cd+."""
    return 1.0 - (4.0 / 9.0) * (gamma_rad_s / (2.0 * omega_hfp_rad_s)) ** 2


def acton_optimal_light_level(alpha1_over_eta: float) -> float:
    """lambda_0 ~ ln(eta/alpha_1): the light level at which the zero-threshold bright and dark errors are equal (Acton Eq. 18)."""
    if not 0.0 < alpha1_over_eta < 1.0:
        raise ValueError("alpha_1/eta is a leak probability per detected photon in (0, 1)")
    return -math.log(alpha1_over_eta)


def neighbour_intensity_ratio(wavelength_m: float, spacing_m: float) -> float:
    """I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2): the resonant light of a saturated bright neighbour at distance x relative to
    I_sat = pi h c Gamma/(3 lambda^3) (Section 8.5 [corrected]: Acton's hbar for h makes the printed 3 lambda^2/(4 pi x^2)
    and its 7e-4 estimate 2 pi too large; 1.1e-4 for 111Cd+ at 4 um)."""
    return 3.0 * wavelength_m**2 / (8.0 * math.pi**2 * spacing_m**2)


def neighbour_pumping_rates(
    rates: FluorescenceRates,
    s_beam: float,
    wavelength_m: float,
    spacing_m: float,
    *,
    polarization_purity: float = 1.0,
) -> tuple[float, float]:
    """(Delta R_d, Delta R_b) that ONE saturated bright neighbour at ``spacing_m`` adds to an ion's chain: the DEPUMPING
    half of Wineland's readout-crosstalk mechanism (Section 8.5).

    Section 8.5: the mechanism "is a degradation of state-discrimination efficiency by a neighbour's scattered light of
    different polarization, not a change in the instrumental eta_d ... modelled as added counts on a dark neighbour PLUS
    DEPUMPING-REDUCED N", with the radiative pumping "bounded by I_ion/I_sat = 3 lambda^2/(8 pi^2 x^2)". Section 8.1
    makes R_d and R_b linear in intensity with no saturation denominator, so the leaked light drives the same two channels
    at the intensity ratio s_neighbour/s_beam of its own saturation parameter to the detection beam's. The neighbour's
    fluorescence is unpolarized in the atomic frame, so ``polarization_purity`` = 1 is the BOUND the plan states, and a
    smaller value is the share of the leaked light that reaches the pumping polarization at the ion.
    """
    if s_beam <= 0.0:
        raise ValueError("the detection beam's saturation parameter is positive")
    if not 0.0 <= polarization_purity <= 1.0:
        raise ValueError("the polarization share lies in [0, 1]")
    if spacing_m <= 0.0 or wavelength_m <= 0.0:
        raise ValueError("the spacing and the wavelength are positive")
    scale = polarization_purity * neighbour_intensity_ratio(wavelength_m, spacing_m) / s_beam
    return rates.R_dark_pumping_per_s * scale, rates.R_bright_pumping_per_s * scale


# ---- the detection-efficiency chain (Section 8.1 "Detected rate and background") --------------------------------------------


def geometric_efficiency(numerical_aperture: float) -> float:
    """(1 - cos(arcsin NA))/2: the solid-angle fraction of an objective of numerical aperture NA (10 % at NA 0.6)."""
    if not 0.0 < numerical_aperture <= 1.0:
        raise ValueError("NA lies in (0, 1]")
    return 0.5 * (1.0 - math.cos(math.asin(numerical_aperture)))


def emccd_effective_quantum_efficiency(
    quantum_efficiency: float, excess_noise_factor: float = math.sqrt(2.0)
) -> float:
    """QE/F^2: an EM register's stochastic gain is an excess noise factor F = sqrt 2 at high gain, equivalent to halving
    the quantum efficiency for photon counting (Burrell 2010; Section 8.1)."""
    if not 0.0 < quantum_efficiency <= 1.0 or excess_noise_factor < 1.0:
        raise ValueError("QE lies in (0, 1] and the excess noise factor is at least 1")
    return quantum_efficiency / excess_noise_factor**2


def system_efficiency(*factors: float) -> float:
    """epsilon_sys = epsilon_geom epsilon_optics epsilon_filter epsilon_det, the product entered ONCE (Section 13)."""
    out = 1.0
    for f in factors:
        if not 0.0 < f <= 1.0:
            raise ValueError("every efficiency factor lies in (0, 1]")
        out *= f
    return out


def camera_snr(mean_counts: float, gain: float, read_noise: float, n_readouts: int) -> float:
    """SNR = g lambda_0/sqrt(g^2 lambda_0 + (k r)^2): shot noise plus read noise r per pixel readout over k readouts, the
    read-noise term (kr)^2 linear in the number of readouts (Acton Eq. 21 [corrected]); sqrt(lambda_0) when read-noise free."""
    if mean_counts < 0.0 or gain <= 0.0 or read_noise < 0.0 or n_readouts < 0:
        raise ValueError("counts and read noise are non-negative, gain positive")
    return gain * mean_counts / math.sqrt(gain**2 * mean_counts + (n_readouts * read_noise) ** 2)


def airy_first_zero_radius_m(wavelength_m: float, numerical_aperture: float) -> float:
    """0.61 lambda/NA, the first zero of the Airy pattern (diameter 1.22 lambda/NA). Burrell prints 1.22 lambda/tan alpha
    [contested]; the code carries NA = sin alpha, so the two coincide only for small apertures."""
    return 0.61 * wavelength_m / numerical_aperture


# ---- micromotion and Doppler factors on the detection line (Sections 8.8, 4.1.1) --------------------------------------------


def micromotion_detection_rate(
    rate_at_detuning: Callable[[float], float],
    beta: float,
    omega_rf_rad_s: float,
    detuning_rad_s: float = 0.0,
) -> float:
    """J_0(beta)^2 R(Delta) + J_1(beta)^2 [R(Delta - Omega_rf) + R(Delta + Omega_rf)]: the carrier and first-sideband channels
    of a detection beam under excess micromotion of modulation index beta, like every other drive (Section 8.8)."""
    if beta < 0.0:
        raise ValueError("the modulation index is non-negative")
    carrier = float(j0(beta)) ** 2 * rate_at_detuning(detuning_rad_s)
    side = float(j1(beta)) ** 2 * (
        rate_at_detuning(detuning_rad_s - omega_rf_rad_s) + rate_at_detuning(detuning_rad_s + omega_rf_rad_s)
    )
    return carrier + side


def rms_velocity_m_per_s(
    mass_kg: float,
    omegas_rad_s: Sequence[float],
    nbars: Sequence[float],
    projections: Sequence[float] | None = None,
) -> float:
    """v_rms = sqrt(sum_m hbar omega_m (2 nbar_m + 1) (k^ . e_m)^2/(2 m)) along a beam direction (Section 8.8 [background]):
    the thermal plus zero-point velocity variance of the modes the detection k-vector projects onto."""
    if len(omegas_rad_s) != len(nbars):
        raise ValueError("one nbar per mode")
    proj = np.ones(len(omegas_rad_s)) if projections is None else np.asarray(projections, dtype=float)
    if proj.shape != (len(omegas_rad_s),):
        raise ValueError("one projection per mode")
    var = sum(
        HBAR_J_S * w * (2.0 * n + 1.0) * p**2 / (2.0 * mass_kg) for w, n, p in zip(omegas_rad_s, nbars, proj)
    )
    return math.sqrt(var)


def doppler_width_hz(v_rms_m_per_s: float, wavelength_m: float) -> float:
    """k v_rms/2 pi = v_rms/lambda: the first-order Doppler width of the detection or shelving line (Section 8.8)."""
    return v_rms_m_per_s / wavelength_m


def thermal_transfer_probability(
    rabi_rad_s: float, duration_s: float, k_rad_per_m: float, v_rms_m_per_s: float, *, nodes: int = 48
) -> float:
    """A resonant pulse's transfer probability averaged over a Maxwell-Boltzmann velocity along k (Section 8.8,
    "motion-to-readout coupling": the thermal Doppler profile of the shelving transition).

    P(delta) = Omega^2/(Omega^2 + delta^2) sin^2(sqrt(Omega^2 + delta^2) t/2) with delta = k v, Gauss-Hermite averaged over
    v ~ N(0, v_rms^2). Derived here from first principles, tagged [background]: no source gives a functional form and the
    coupling is flagged UNTESTED against data (Harty et al. observe the degradation without an equation).
    """
    if rabi_rad_s <= 0.0 or duration_s <= 0.0 or nodes < 4:
        raise ValueError("Rabi frequency and duration are positive; at least four quadrature nodes")
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    total = 0.0
    for xi, wi in zip(x, w):
        delta = k_rad_per_m * v_rms_m_per_s * float(xi)
        w_eff = math.sqrt(rabi_rad_s**2 + delta**2)
        total += float(wi) * (rabi_rad_s / w_eff) ** 2 * math.sin(0.5 * w_eff * duration_s) ** 2
    return total / math.sqrt(2.0 * math.pi)


# ---- shelving (Section 8.1 "Shelving") ------------------------------------------------------------------------------------


def shelf_decay_error(window_s: float, shelf_lifetime_s: float) -> float:
    """epsilon_decay = 1 - exp(-t_det/tau_D): the shelf decays during the window (Christensen et al. print the exponent
    with the wrong sign [corrected]); 1.5e-4 for 133Ba+ at 4.5 ms and 30 s."""
    if window_s < 0.0 or shelf_lifetime_s <= 0.0:
        raise ValueError("window non-negative, lifetime positive")
    return 1.0 - math.exp(-window_s / shelf_lifetime_s)


@dataclass(frozen=True)
class ShelvingBranching:
    """Shelving by optical pumping through an excited level (Section 8.1, 133Ba+ through P3/2).

    Each excitation ends in the shelf (``to_shelf``), back in the bright ground manifold (``to_bright``, re-excited),
    in another metastable level (``to_other_metastable``, re-excited only when ``repump_other``) or in the dark ground
    state (``to_dark_ground``, stranded: the off-resonant path through the wrong hyperfine level). The end state of the
    absorbing chain is the shelving fidelity; 0.74/0.23/0.03 gives 0.885 with the D3/2 population stranded (Christensen's
    "F = 0.88") and 1 with it repumped, before the off-resonant leak.
    """

    to_shelf: float
    to_bright: float
    to_other_metastable: float = 0.0
    to_dark_ground: float = 0.0
    repump_other: bool = False

    def __post_init__(self) -> None:
        total = self.to_shelf + self.to_bright + self.to_other_metastable + self.to_dark_ground
        if min(self.to_shelf, self.to_bright, self.to_other_metastable, self.to_dark_ground) < 0.0:
            raise ValueError("branching fractions are non-negative")
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"branching fractions must sum to 1, got {total}")
        if self.to_shelf <= 0.0:
            raise ValueError("a shelving scheme needs a nonzero branch into the shelf")

    @property
    def stranded(self) -> float:
        """The probability per excitation of ending somewhere other than the shelf or the re-excited manifold."""
        return self.to_dark_ground + (0.0 if self.repump_other else self.to_other_metastable)

    def shelving_probability(self) -> float:
        """P(shelf) = b_s/(b_s + stranded): the absorbing-chain end state."""
        return self.to_shelf / (self.to_shelf + self.stranded)

    def expected_excitations(self) -> float:
        """Mean number of excitations before absorption, 1/(1 - p_recycle)."""
        recycle = self.to_bright + (self.to_other_metastable if self.repump_other else 0.0)
        return 1.0 / (1.0 - recycle)


def mean_count_curve(
    tau_s: np.ndarray | float,
    detected_bright_per_s: float,
    dark_pumping_per_s: float,
    bright_pumping_per_s: float,
) -> np.ndarray | float:
    """n̄(τ) = εR_o [(R_b/k) τ + (R_d/k^2)(1 - e^{-kτ})], k = R_b + R_d: the mean detected count of a bright-prepared ion
    from the two-state rate equation ṗ_1 = R_b p_0 - R_d p_1 (Noek Eq. 4; Section 8.2), the calibration layer's fit model."""
    tau = np.asarray(tau_s, dtype=float)
    k = dark_pumping_per_s + bright_pumping_per_s
    if k <= 0.0:
        out = detected_bright_per_s * tau
    else:
        out = detected_bright_per_s * (
            bright_pumping_per_s / k * tau + dark_pumping_per_s / k**2 * (1.0 - np.exp(-k * tau))
        )
    return float(out) if np.ndim(tau_s) == 0 else out


def fit_mean_count_curve(
    tau_s: Sequence[float], mean_counts: Sequence[float], *, guess: tuple[float, float, float] | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fit (εR_o, R_d, R_b) to n̄(τ) samples (Section 8.2, the calibration fit); returns values and 1-sigma uncertainties."""
    tau = np.asarray(tau_s, dtype=float)
    n = np.asarray(mean_counts, dtype=float)
    if tau.shape != n.shape or tau.size < 4:
        raise ValueError("at least four (tau, nbar) samples")
    if guess is None:
        slope = float(n[1] / tau[1]) if tau[1] > 0 else float(n[-1] / tau[-1])
        guess = (slope, 0.01 / max(tau[-1], 1e-12), 0.001 / max(tau[-1], 1e-12))
    scale = np.array(guess, dtype=float)
    scale[scale == 0.0] = 1.0

    def residual(p: np.ndarray) -> np.ndarray:
        r0, rd, rb = p * scale
        model = np.asarray(mean_count_curve(tau, r0, abs(rd), abs(rb)))
        return np.asarray(model - n)

    sol = least_squares(residual, np.ones(3), method="lm")
    values = sol.x * scale
    dof = max(tau.size - 3, 1)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * float(np.sum(sol.fun**2)) / dof
    sigma = np.sqrt(np.abs(np.diag(cov))) * scale
    return (float(values[0]), abs(float(values[1])), abs(float(values[2]))), tuple(float(s) for s in sigma)  # type: ignore[return-value]


# ---- the readout scheme: polarity as an enum ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutScheme:
    """Which internal level fluoresces, and how each level starts the photon record (Sections 8.1, 8.4, 13).

    ``classes[l]`` is the ideal readout class of internal level l of the ion's qudit (levels 0 and 1 the qubit, the rest
    leakage levels); ``kind`` names the scheme. For shelving schemes the transfer is imperfect and only partially
    state-selective (43Ca+ at 146 G: non-stretch ground states shelve with up to about 30 % probability), so
    ``transfer`` carries the START distribution over classes per level and the dark-outcome operator
    Pi_dark = sum_l p_l |l><l| is not a rank-one projector (Section 8.1); the post-measurement state is discarded because
    shelving readout is not QND (Section 8.4).
    """

    kind: Literal["direct", "shelving"]
    classes: tuple[ReadoutClass, ...]
    transfer: tuple[tuple[tuple[ReadoutClass, float], ...], ...] | None = None
    """Per level, the (class, probability) pairs of the start distribution; None means the ideal ``classes``."""

    def __post_init__(self) -> None:
        if len(self.classes) < 2:
            raise ValueError("a scheme classifies at least the two qubit levels")
        if any(c not in CLASSES for c in self.classes):
            raise ValueError(f"classes are among {CLASSES}")
        if "bright" not in self.classes:
            raise ValueError("no level fluoresces: the scheme cannot discriminate anything")
        if self.kind == "direct" and "shelf" in self.classes:
            raise ValueError("a direct-fluorescence scheme has no shelf class")
        if self.transfer is not None:
            if len(self.transfer) != len(self.classes):
                raise ValueError("one start distribution per level")
            for dist in self.transfer:
                total = sum(p for _c, p in dist)
                if any(p < 0.0 for _c, p in dist) or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("each start distribution has non-negative weights summing to 1")

    @property
    def n_levels(self) -> int:
        return len(self.classes)

    @property
    def bright_level(self) -> int:
        """The computational level (0 or 1) whose ideal class is bright; the polarity of Section 13."""
        for lev in (0, 1):
            if self.classes[lev] == "bright":
                return lev
        raise ValueError("neither qubit level is bright in this scheme")

    @property
    def polarity(self) -> Literal["bright_is_0", "bright_is_1"]:
        return "bright_is_1" if self.bright_level == 1 else "bright_is_0"

    def start_distribution(self, level: int) -> dict[ReadoutClass, float]:
        """P(start class | internal level), the transfer-imperfect assignment (ideal when ``transfer`` is None)."""
        if self.transfer is None:
            return {self.classes[level]: 1.0}
        return {c: p for c, p in self.transfer[level] if p > 0.0}

    @property
    def dark_class(self) -> ReadoutClass:
        """The non-bright class the other qubit level starts in: "shelf" for a shelving scheme, "dark" for direct
        fluorescence (the class whose 1/tau the time-resolved discriminators of Section 8.3 need; with an imperfect
        transfer it is the MOST LIKELY non-bright start, the one that carries the decay term)."""
        dist = {c: p for c, p in self.start_distribution(1 - self.bright_level).items() if c != "bright"}
        if not dist:
            return "dark" if self.kind == "direct" else "shelf"
        return max(dist, key=dist.__getitem__)

    def dark_weights(self) -> tuple[float, ...]:
        """The weights p_l of Pi_dark = sum_l p_l |l><l| (Section 8.1): the probability that level l starts NON-bright."""
        return tuple(1.0 - self.start_distribution(lev).get("bright", 0.0) for lev in range(self.n_levels))

    def bit_of_class(self, cls: ReadoutClass) -> int:
        """The bit a perfectly discriminated class is reported as: bright -> the bright level, anything else -> the other."""
        return self.bright_level if cls == "bright" else 1 - self.bright_level

    @classmethod
    def direct(cls, bright_level: int, *, leak_classes: Sequence[ReadoutClass] = ()) -> ReadoutScheme:
        """Direct hyperfine fluorescence: 171Yb+ has |1> = F = 1 bright, |0> = F = 0 dark (``bright_level = 1``)."""
        if bright_level not in (0, 1):
            raise ValueError("bright_level is a qubit level, 0 or 1")
        base: list[ReadoutClass] = ["dark", "dark"]
        base[bright_level] = "bright"
        return cls("direct", tuple(base) + tuple(leak_classes))

    @classmethod
    def shelving(
        cls,
        shelved_level: int,
        *,
        transfer_probability: float = 1.0,
        off_resonant_shelving: float = 0.0,
        leak_classes: Sequence[ReadoutClass] = (),
        leak_shelving: Sequence[float] = (),
    ) -> ReadoutScheme:
        """Shelving readout: ``shelved_level`` is transferred to the metastable shelf with ``transfer_probability`` and the
        other qubit level is shelved off-resonantly with ``off_resonant_shelving`` (Harty's 1.7e-4 and Christensen's rows).

        For an optical qubit whose upper level IS the shelf (40Ca+ S1/2-D5/2) the transfer probability is 1 and the
        polarity inverts: |0> = S1/2 is bright.
        """
        if shelved_level not in (0, 1):
            raise ValueError("shelved_level is a qubit level, 0 or 1")
        if not 0.0 <= transfer_probability <= 1.0 or not 0.0 <= off_resonant_shelving <= 1.0:
            raise ValueError("probabilities lie in [0, 1]")
        if len(leak_shelving) not in (0, len(leak_classes)):
            raise ValueError("one shelving probability per leak level, or none")
        classes: list[ReadoutClass] = ["bright", "bright"]
        classes[shelved_level] = "shelf"
        classes.extend(leak_classes)
        transfer: list[tuple[tuple[ReadoutClass, float], ...]] = []
        for lev in (0, 1):
            p = transfer_probability if lev == shelved_level else off_resonant_shelving
            transfer.append((("shelf", p), ("bright", 1.0 - p)))
        for k, c in enumerate(leak_classes):
            if leak_shelving:
                p = float(leak_shelving[k])
                transfer.append((("shelf", p), (c if c != "shelf" else "bright", 1.0 - p)))
            else:
                transfer.append(((c, 1.0),))
        ideal = (
            all(
                (lev == shelved_level and transfer_probability == 1.0)
                or (lev != shelved_level and off_resonant_shelving == 0.0)
                for lev in (0, 1)
            )
            and not leak_shelving
        )
        return cls("shelving", tuple(classes), None if ideal else tuple(transfer))

    @classmethod
    def for_species(
        cls, species: Species, bright_labels: Sequence[str], *, labels: Sequence[str] | None = None
    ) -> ReadoutScheme:
        """The ideal scheme of a species: a label in the bright manifold is bright, one in a metastable D level is the shelf
        (optical qubits), any other is dark (hyperfine qubits under direct fluorescence). ``labels`` extends the classes to
        every level of a register factor with d > 2 (``noise/levels.py``, M7): a leaked F = 1 sublevel of 171Yb+ is bright,
        the SINK is read as dark (a D-level population is repumped during detection; the scheme records the approximation
        through its class, not its dynamics)."""
        from qutip_trap.noise.levels import SINK
        from qutip_trap.species.model import parse_state_label

        wanted = tuple(species.qubit) if labels is None else tuple(labels)
        classes: list[ReadoutClass] = []
        for lab in wanted:
            if lab == SINK:
                classes.append("dark")
                continue
            level, _ = parse_state_label(lab)
            if lab in bright_labels:
                classes.append("bright")
            elif level.startswith("D"):
                classes.append("shelf")
            else:
                classes.append("dark")
        kind: Literal["direct", "shelving"] = "shelving" if "shelf" in classes else "direct"
        if kind == "direct":
            return cls(kind, tuple(classes))
        # a shelving scheme with leaked D-sublevels: those are shelf too, and the direct-scheme check on 'shelf' does not apply
        return cls(kind, tuple(classes))


# ---- the rate object ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FluorescenceRates:
    """R_o, R_d, R_b and the shelf rates of one ion under its detection beams (Section 8.1), all in s^-1, SCATTERED photons.

    ``ceiling`` is the equal-population ceiling the photon rate was asserted against (None when the rates were ingested
    from a measurement rather than solved); ``provenance`` names the ledger records and the source of each number.
    """

    R_bright_per_s: float
    """R_o: photon scattering rate of the conditional bright state."""
    R_dark_pumping_per_s: float
    """R_d: bright -> dark off-resonant pumping."""
    R_bright_pumping_per_s: float
    """R_b: dark -> bright off-resonant pumping."""
    shelf_decay_per_s: float = 0.0
    """1/tau_D: the shelf decays back into the bright manifold (Myerson's 1168 ms, Christensen's 30 s)."""
    shelf_pumping_per_s: float = 0.0
    """Bright -> shelf during detection (Myerson measures < 1e-3 s^-1 for 40Ca+; the detection light's D5/2 channel)."""
    ceiling: float | None = None
    excited_population: float | None = None
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "R_bright_per_s",
            "R_dark_pumping_per_s",
            "R_bright_pumping_per_s",
            "shelf_decay_per_s",
            "shelf_pumping_per_s",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} is a rate and non-negative")
        if self.R_bright_per_s <= 0.0:
            raise ValueError("a bright state that scatters nothing cannot be read out")

    def transition_rates(self) -> dict[tuple[ReadoutClass, ReadoutClass], float]:
        """The continuous-time Markov chain of Section 8.2: (from, to) -> rate, zero rates omitted."""
        pairs: dict[tuple[ReadoutClass, ReadoutClass], float] = {
            ("bright", "dark"): self.R_dark_pumping_per_s,
            ("dark", "bright"): self.R_bright_pumping_per_s,
            ("shelf", "bright"): self.shelf_decay_per_s,
            ("bright", "shelf"): self.shelf_pumping_per_s,
        }
        return {k: v for k, v in pairs.items() if v > 0.0}

    def detected(self, detector: Detector) -> tuple[float, float]:
        """(epsilon_sys R_o, R_bg): the detected bright rate with the efficiency applied ONCE and the background beside it."""
        return detector.efficiency * self.R_bright_per_s, detector.background_cps

    def with_shelf(self, lifetime_s: float, *, pumping_per_s: float = 0.0) -> FluorescenceRates:
        from dataclasses import replace

        if lifetime_s <= 0.0:
            raise ValueError("the shelf lifetime is positive")
        return replace(self, shelf_decay_per_s=1.0 / lifetime_s, shelf_pumping_per_s=pumping_per_s)


def rates_from_detected(
    detected_bright_per_s: float,
    efficiency: float,
    *,
    dark_pumping_per_s: float = 0.0,
    bright_pumping_per_s: float = 0.0,
    shelf_lifetime_s: float | None = None,
    shelf_pumping_per_s: float = 0.0,
    provenance: tuple[str, ...] = (),
) -> FluorescenceRates:
    """Ingest an apparatus's MEASURED detected rate (Myerson's R_B, Crain's 472 kcps) by dividing the efficiency out once, so
    that the record layer applies it once again and the scattered rate can be compared with the Bloch solve (Section 8.8)."""
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency lies in (0, 1]")
    return FluorescenceRates(
        R_bright_per_s=detected_bright_per_s / efficiency,
        R_dark_pumping_per_s=dark_pumping_per_s,
        R_bright_pumping_per_s=bright_pumping_per_s,
        shelf_decay_per_s=0.0 if shelf_lifetime_s is None else 1.0 / shelf_lifetime_s,
        shelf_pumping_per_s=shelf_pumping_per_s,
        ceiling=None,
        excited_population=None,
        provenance=tuple(provenance) + ("apparatus: detected rate ingested, not solved",),
    )


def detected_line(model: BlochModel) -> str:
    """The one decay line whose photons the detector counts, when the model carries only one (Section 8.1).

    R_det = epsilon_sys R_o with epsilon_filter INSIDE epsilon_sys: the interference filter passes one wavelength, so
    photons emitted on a repump line (866 nm beside 40Ca+'s detected 397 nm, 935 nm beside 171Yb+'s 369.5 nm) must not be
    counted in R_o. When the build carries several non-sink lines the choice is a device property and cannot be derived
    here, so the caller must name it (or ask for :data:`ALL_LINES` explicitly, as a diagnostic).
    """
    from qutip_trap.dynamics.multilevel import SINK

    lines = []
    for ch in model.build.channels:
        key = f"{ch.lower}<-{ch.upper}"
        if not key.startswith(SINK) and key not in lines:
            lines.append(key)
    if not lines:
        raise ValueError("the build has no decay line: nothing is detected")
    if len(lines) > 1:
        raise ValueError(
            f"the detection model carries {len(lines)} decay lines {lines}: name the detected one "
            f"(R_det = eps_sys R_o counts one wavelength, Section 8.1) or pass line={ALL_LINES!r} for the "
            "all-line diagnostic sum"
        )
    return lines[0]


def rates_from_bloch(
    model: BlochModel,
    bright_labels: Sequence[str],
    dark_labels: Sequence[str],
    *,
    line: str | None = None,
    shelf_lifetime_s: float | None = None,
    shelf_pumping_per_s: float = 0.0,
    provenance: tuple[str, ...] = ("conv.scattering_rate_object",),
) -> FluorescenceRates:
    """R_o, R_d, R_b from one Liouvillian (the M3a slow-manifold analysis), the photon rate asserted against the ceiling.

    With no dark labels (a shelving scheme whose bright manifold is everything the beams drive, 40Ca+ under 397 + 866 nm)
    the steady state is the bright state and the pumping rates vanish.

    ``line`` names the DETECTED line "lower<-upper" (level names); None derives it with :func:`detected_line` and refuses
    a model with several lines; :data:`ALL_LINES` sums every non-sink line, which is the total scattering rate of Section
    4.2.2 and a diagnostic, NOT a detected rate (the single epsilon_sys of Section 8.1 has one epsilon_filter in it).
    """
    if line == ALL_LINES:
        chosen: str | None = None
    else:
        chosen = line if line is not None else detected_line(model)
    if dark_labels:
        dr = model.detection_rates(bright_labels, dark_labels, line=chosen)
        r_o, r_d, r_b = dr.R_bright_per_s, dr.R_dark_pumping_per_s, dr.R_bright_pumping_per_s
        ceiling, p_e = dr.ceiling.ceiling, dr.ceiling.excited_population
    else:
        ss = model.steadystate()
        rates = ss.photon_rates_per_s
        r_o = float(rates[chosen]) if chosen is not None else ss.total_photon_rate_per_s
        r_d, r_b = 0.0, 0.0
        ceiling, p_e = ss.ceiling.ceiling, ss.ceiling.excited_population
    return FluorescenceRates(
        R_bright_per_s=r_o,
        R_dark_pumping_per_s=r_d,
        R_bright_pumping_per_s=r_b,
        shelf_decay_per_s=0.0 if shelf_lifetime_s is None else 1.0 / shelf_lifetime_s,
        shelf_pumping_per_s=shelf_pumping_per_s,
        ceiling=ceiling,
        excited_population=p_e,
        provenance=tuple(provenance),
    )


def ground_manifold_labels(
    structure: AtomicStructure, level: str, hyperfine_f: float | None = None
) -> tuple[str, ...]:
    """The dressed sublevel labels of ``level`` (optionally one hyperfine F only), the bright and dark manifolds of Section 8.1."""
    from qutip_trap.species.zeeman import parse_quantum_numbers

    out: list[str] = []
    for st in structure.states_of(level):
        if hyperfine_f is not None:
            qn = parse_quantum_numbers(st.label)
            if "F" not in qn or float(qn["F"]) != float(hyperfine_f):
                continue
        out.append(f"{level} {st.label}")
    return tuple(out)


def scattering_rate(
    structure: AtomicStructure,
    beams: Sequence[Beam],
    *,
    bright_labels: Sequence[str] | None = None,
    dark_labels: Sequence[str] | None = None,
    line: str | None = None,
    levels: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    position_m: Sequence[float] | None = None,
    shelf_lifetime_s: float | None = None,
) -> tuple[FluorescenceRates, BlochModel]:
    """The Section 8.1 rate object from first principles: the multi-level Bloch model of ``beams`` on ``structure``.

    The bright manifold defaults to the ground states the beams drive resonantly (``BlochModel.resonant_manifold``) and
    the dark manifold to the remaining sublevels of the same levels (171Yb+: F = 1 bright, F = 0 dark). ``line`` names the
    DETECTED decay line; None derives it when the model carries one line and refuses to guess when it carries several
    (:func:`detected_line`), and :data:`ALL_LINES` opts into the all-line diagnostic sum. Returns the rates and the model
    so callers can read the steady state, the dark states and the ceiling report from the same object.
    """
    from qutip_trap.dynamics.multilevel import SINK, MultiLevelOptions
    from qutip_trap.light.bloch import BlochModel

    opts = options or MultiLevelOptions(leak="renormalize")
    model = BlochModel(structure, beams, levels=levels, position_m=position_m, options=opts)
    ground, _excited = model.resonant_manifold()
    bright = tuple(bright_labels) if bright_labels is not None else tuple(ground)
    if not bright:
        raise ValueError("the detection beams drive no ground state resonantly: nothing fluoresces")
    if dark_labels is None:
        levels_of_bright = {model.build.level_of(lab) for lab in bright}
        dark = tuple(
            lab
            for lab in model.build.labels
            if lab != SINK and lab not in bright and model.build.level_of(lab) in levels_of_bright
        )
    else:
        dark = tuple(dark_labels)
    rates = rates_from_bloch(model, bright, dark, line=line, shelf_lifetime_s=shelf_lifetime_s)
    return rates, model


def detection_rates_for_ion(
    species: Species,
    b_gauss: float,
    b_hat: tuple[float, float, float],
    beams: Sequence[Beam],
    *,
    position_m: Sequence[float] | None = None,
    levels: Sequence[str] | None = None,
    options: MultiLevelOptions | None = None,
    scheme: ReadoutScheme | None = None,
    micromotion_beta: float = 0.0,
    omega_rf_rad_s: float = 0.0,
) -> tuple[FluorescenceRates, ReadoutScheme, BlochModel]:
    """Rates and scheme for one ion of a device: the species' cycling and repump lines under the detection beams, the shelf
    lifetime from the species' metastable level when the scheme shelves (the D level of an optical qubit).

    R_o counts photons on the species' CYCLING line only, because epsilon_sys carries one epsilon_filter and the filter
    passes one wavelength (Section 8.1): counting the repump line as well makes R_o 6.9 % high for 40Ca+ under 397 + 866 nm
    (it was 6.4 % until 2026-09-08, when the 866 nm branching became Ramm et al. 2013's 0.06435 in place of Section 8.1's
    0.06; ledger conv.ca40_branching).
    ``micromotion_beta`` > 0 applies Section 8.8's "R_o carries J_0(beta)^2 and a first-sideband channel J_1(beta)^2, like
    every other drive": the detection drive as a whole is phase-modulated at ``omega_rf_rad_s``, so every rate the Bloch
    solve returns becomes J_0^2 R(Delta) + J_1^2 [R(Delta - Omega_rf) + R(Delta + Omega_rf)] over three solves.
    """
    from qutip_trap.light.bloch import shifted_beam
    from qutip_trap.light.roles import RESONANT_WINDOW
    from qutip_trap.species.model import parse_state_label, parse_transition_label
    from qutip_trap.species.raman import structure_at

    st = structure_at(species, b_gauss, b_hat)
    lower, upper = parse_transition_label(species.cycling)
    if levels is None:
        lv = [lower, upper]
        for rep in species.repumps:
            tr = next((t for t in species.transitions if t.label == rep and t.multipole == "E1"), None)
            if tr is None:
                continue
            # a repump's manifold enters the detection model only when this device actually shines that light on the ion.
            # Without the beam the metastable level is a dead end whose own lifetime (52.7 ms for 171Yb+ D3/2) becomes the
            # slowest Liouvillian mode, so the bright/dark coarse graining of Section 8.1 has no two-manifold slow mode at
            # all; Section 8.1's "a mis-set or failed repump is a simulable fault" is a property of the BEAMS, not of the
            # species table (M5 fix, 2026-09-07: tabulating the 935.2 nm line made every 171Yb+ readout raise).
            if not any(
                abs(b.wavelength_m - tr.wavelength_vac_m) < RESONANT_WINDOW * tr.wavelength_vac_m
                for b in beams
            ):
                continue
            for end in parse_transition_label(rep):
                if end not in lv:
                    lv.append(end)
        levels = tuple(lv)
    cycling_line = f"{lower}<-{upper}"
    rates, model = scattering_rate(
        st, beams, levels=levels, options=options, position_m=position_m, line=cycling_line
    )
    if micromotion_beta > 0.0:
        if omega_rf_rad_s <= 0.0:
            raise ValueError(
                "a micromotion modulation index needs the trap's rf frequency Omega_rf for the sideband channels"
            )
        solved: dict[float, FluorescenceRates] = {0.0: rates}

        def at(offset: float) -> FluorescenceRates:
            if offset not in solved:
                solved[offset], _ = scattering_rate(
                    st,
                    [shifted_beam(b, offset) for b in beams],
                    levels=levels,
                    options=options,
                    position_m=position_m,
                    line=cycling_line,
                )
            return solved[offset]

        from dataclasses import replace

        rates = replace(
            rates,
            R_bright_per_s=micromotion_detection_rate(
                lambda d: at(d).R_bright_per_s, micromotion_beta, omega_rf_rad_s
            ),
            R_dark_pumping_per_s=micromotion_detection_rate(
                lambda d: at(d).R_dark_pumping_per_s, micromotion_beta, omega_rf_rad_s
            ),
            R_bright_pumping_per_s=micromotion_detection_rate(
                lambda d: at(d).R_bright_pumping_per_s, micromotion_beta, omega_rf_rad_s
            ),
            provenance=rates.provenance
            + (
                f"micromotion on the detection beam (Section 8.8): beta = {micromotion_beta:.4g}, "
                f"Omega_rf/2pi = {omega_rf_rad_s / (2.0 * math.pi):.4g} Hz, "
                f"J_0^2 = {float(j0(micromotion_beta)) ** 2:.6f} on the carrier and "
                f"J_1^2 = {float(j1(micromotion_beta)) ** 2:.6f} on each first sideband",
            ),
        )
    ground, _ = model.resonant_manifold()
    chosen = scheme if scheme is not None else ReadoutScheme.for_species(species, ground)
    if "shelf" in chosen.classes and rates.shelf_decay_per_s == 0.0:
        shelf_level = None
        for lev_index, cls_name in enumerate(chosen.classes):
            if cls_name == "shelf" and lev_index < len(species.qubit):
                shelf_level, _ = parse_state_label(species.qubit[lev_index])
        if shelf_level is None and species.shelving is not None:
            _, shelf_level = parse_transition_label(species.shelving)
        if shelf_level is not None:
            tau = species.level(shelf_level).lifetime_s
            if tau is not None:
                rates = rates.with_shelf(tau)
    return rates, chosen, model
