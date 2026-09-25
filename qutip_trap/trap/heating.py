"""Electric-field noise spectra to heating rates (PLAN.md Section 4.1.5; Section 13; milestone M1).

Conventions (Section 13, rows "Electric-field noise density", "Heating-rate meaning", "Heating master equation"):

- S_E is SINGLE-sided, S_E(omega) = 2 int dtau <dE(tau) dE(0)> e^{-i omega tau} in (V/m)^2/Hz, and
  Gamma_h = e^2 S_E(omega)/(4 m hbar omega) (Brownnutt 2015 Eqs. 11-12; the published Turchette 2000 agrees, only
  its arXiv preprint prints an inconsistent Eq. 3). A two-sided density S^(2) = S^(1)/2 gives e^2 S^(2)/(2 m hbar omega),
  which is how ``NoiseSpectrum`` (ALWAYS two-sided, angular frequency, e^{-i omega t} kernel) converts at the boundary.
- A quoted heating rate is d<n>/dt at n = 0, i.e. Gamma N_bar, not the thermal diffusion rate Gamma(2 N_bar + 1);
  the master equation carries Gamma(N_bar + 1) on a and Gamma N_bar on a^dagger, both -> Gamma_h for N_bar >> 1
  (RMP 2003's high-temperature amplitude damping), and the coherence of (|n0> + |m0>)/sqrt 2 decays at
  Gamma_h (n0 + m0 + 1) in that limit (2.01, 4.03, 10.09 Gamma_h for (0, 1), (1, 2), (3, 6); Section 4.1.5).
- Multi-ion (Brownnutt Eqs. 19-23; Kielpinski Eq. 20): mode k heats at e^2/(4 hbar omega_k) sum_ij c_i c_j S_E^{ij}(omega_k)/sqrt(m_i m_j)
  with the mass-weighted eigenvector c and the cross-spectral density S_E^{ij}; spatially uniform noise
  (correlation length -> infinity) heats only the centre-of-mass modes of an EQUAL-mass chain, at N times the
  single-ion rate (Lechner 2016: 9 x 7.2 = 65 quanta/s), while every mode of a mixed crystal with a non-zero
  sum_j c_j/sqrt(m_j) heats; uncorrelated noise (correlation length 0) heats every mode of an equal-mass chain at the
  single-ion rate. The simulator never defaults to the uniform limit: the correlation length is a required argument.
- With rf drive present the rate sums over micromotion sidebands, Gamma_h = (e^2/(4 m hbar omega)) sum_j |C_2j|^2 S_E(|omega + j Omega|)
  with the Wronskian-normalized Floquet weights |c_j/c_0|^2 of ``trap/mathieu.py``; Turchette's leading correction
  (omega^2/(2 Omega^2)) S_E(Omega +- omega) is the same j = +-1 weight q^2/16 at lowest order and must never be added to it.
- Empirical S_E ~ omega^-alpha d^-beta T^gamma (Brownnutt Eq. 30) with no a-priori justification; Johnson noise
  S_E = 4 k_B T R/d^2; the fluctuating-patch model S_E = 3 C S_V r_p^2/(4 a^4). Named models with literature parameters,
  never a prediction from materials (Section 12).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np

from qutip_trap.trap.crystal import Crystal
from qutip_trap.trap.mathieu import floquet_coefficients
from qutip_trap.units import E_C, HBAR_J_S, K_B_J_PER_K

if TYPE_CHECKING:
    from qutip_trap.noise.spectra import NoiseSpectrum

SpectralDensity = Callable[[float], float]
"""A single-sided S_E(omega) in (V/m)^2/Hz as a function of ANGULAR frequency omega in rad/s."""


def heating_rate_quanta_per_s(s_e_single_sided_v2_m2_hz: float, mass_kg: float, omega_rad_s: float) -> float:
    """n_dot = e^2 S_E(omega)/(4 m hbar omega) for a single ion (Brownnutt Eqs. 11-12; Section 9.15 round trip).

    ``omega_rad_s`` is ANGULAR. Section 9.15: S_E = 2.2250e-13 (V/m)^2/Hz for 9Be+ at omega_z/2pi = 3.6 MHz
    gives 40 quanta/s, the round-trip test.
    """
    if omega_rad_s <= 0.0 or mass_kg <= 0.0:
        raise ValueError("mass and angular frequency must be positive")
    if s_e_single_sided_v2_m2_hz < 0.0:
        raise ValueError("a spectral density is non-negative")
    return E_C * E_C * s_e_single_sided_v2_m2_hz / (4.0 * mass_kg * HBAR_J_S * omega_rad_s)


def s_e_from_heating_rate(n_dot_quanta_per_s: float, mass_kg: float, omega_rad_s: float) -> float:
    """The inverse: S_E = 4 m hbar omega n_dot/e^2, single-sided, what a measured rate implies for the device."""
    return 4.0 * mass_kg * HBAR_J_S * omega_rad_s * n_dot_quanta_per_s / (E_C * E_C)


def single_sided_from_two_sided(s_two_sided: float | np.ndarray) -> float | np.ndarray:
    """S^(1) = 2 S^(2): the boundary conversion from a ``NoiseSpectrum`` (two-sided) to Brownnutt's single-sided S_E."""
    return (
        2.0 * np.asarray(s_two_sided, dtype=float)
        if isinstance(s_two_sided, np.ndarray)
        else 2.0 * s_two_sided
    )


def two_sided_from_single_sided(s_single_sided: float) -> float:
    return s_single_sided / 2.0


def single_sided_from_spectrum(spectrum: NoiseSpectrum) -> SpectralDensity:
    """S_E(omega) = 2 S^(2)(|omega|) from a two-sided ``NoiseSpectrum``, its band edges and its white level.

    The evaluation is the spectrum's OWN ``value`` (Section 6.1): the tabulated part folded onto |omega| and ZERO above
    the tabulated band, plus ``white_level`` everywhere - which is the split the two fields declare, the tabulated band
    being the sampled-trajectory route and the white level the Lindblad route. The caller must therefore not add the
    white level a second time. Taking the record rather than its arrays is what keeps the folding in one place: a
    symmetric tabulation running from -omega_max to +omega_max has a non-monotonic ``|omega_rad_s|``, on which a bare
    ``np.interp`` silently returns garbage, and ``np.interp`` clamps at the top of the band where the physics is the
    white level alone (the micromotion-sideband sum of ``micromotion_sideband_heating_rate`` evaluates S_E at tens of MHz).
    """

    def s_e(omega: float) -> float:
        return 2.0 * float(spectrum.value(abs(omega)))

    return s_e


def thermal_collapse_rates(
    heating_rate_quanta_per_s: float, n_bar_bath: float | None = None
) -> tuple[float, float]:
    """(rate on a, rate on a^dagger) = (Gamma (N_bar + 1), Gamma N_bar) with Gamma N_bar = the quoted heating rate.

    ``None`` is the N_bar -> infinity limit of electric-field noise (Section 4.1.5): both rates equal Gamma_h, the
    collapse operators sqrt(Gamma_h) a and sqrt(Gamma_h) a^dagger. A finite N_bar must not be imported from a transport
    source's equal-rate pair (Section 13): it adds unphysical damping on microsecond timescales.
    """
    if heating_rate_quanta_per_s < 0.0:
        raise ValueError("heating rate is non-negative")
    if n_bar_bath is None:
        return heating_rate_quanta_per_s, heating_rate_quanta_per_s
    if n_bar_bath <= 0.0:
        raise ValueError("the bath occupation must be positive (or None for the high-temperature limit)")
    gamma = heating_rate_quanta_per_s / n_bar_bath
    return gamma * (n_bar_bath + 1.0), gamma * n_bar_bath


def coherence_decay_rate(
    heating_rate_quanta_per_s: float, n0: int, m0: int, n_bar_bath: float | None = None
) -> float:
    """Short-time decay rate of the coherence of (|n0> + |m0>)/sqrt 2: (Gamma/2)[(2 N_bar + 1)(n0 + m0) + 2 N_bar]
    -> Gamma_h (n0 + m0 + 1) for N_bar >> 1 (Section 4.1.5, with the +1 the 2026-09-04 critique restored)."""
    if n_bar_bath is None:
        return heating_rate_quanta_per_s * (n0 + m0 + 1)
    gamma = heating_rate_quanta_per_s / n_bar_bath
    return gamma / 2.0 * ((2.0 * n_bar_bath + 1.0) * (n0 + m0) + 2.0 * n_bar_bath)


# ---- multi-ion generalization (Section 4.1.5; Kielpinski Eq. 20; Brownnutt Eqs. 19-23) ---------------------------------


def correlation_matrix(positions_m: np.ndarray, correlation_length_m: float) -> np.ndarray:
    """g_ij = exp(-|r_i - r_j|/l_c): 1 everywhere for l_c = inf (uniform), the identity for l_c = 0 (uncorrelated)."""
    pos = np.asarray(positions_m, dtype=float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    if correlation_length_m < 0.0:
        raise ValueError("the correlation length is non-negative (0 uncorrelated, inf uniform)")
    if math.isinf(correlation_length_m):
        return np.ones_like(d)
    if correlation_length_m == 0.0:
        return np.eye(len(pos))
    return np.asarray(np.exp(-d / correlation_length_m))


def heating_rates_per_mode(
    crystal: Crystal,
    s_e_single_sided: SpectralDensity | float,
    correlation_length_m: float,
    *,
    modes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """n_dot_k = e^2/(4 hbar omega_k) sum_a sum_ij c_{ia} c_{ja} g_ij S_E(omega_k)/sqrt(m_i m_j), quanta/s per mode (Section 4.1.5).

    ``s_e_single_sided`` is a single-sided density in (V/m)^2/Hz, either a constant or a callable of angular frequency,
    the same along every axis (isotropic, uncorrelated field components); ``correlation_length_m`` is REQUIRED
    (inf: uniform, 0: uncorrelated). For an equal-mass chain in uniform noise only the three centre-of-mass modes heat,
    at N times the single-ion rate; for a mixed crystal every mode with a non-zero mass-weighted sum heats (Section 4.1.7).
    """
    masses = crystal.masses_kg
    g = correlation_matrix(crystal.positions_m, correlation_length_m)
    idx = range(len(crystal.modes)) if modes is None else modes
    out = []
    for k in idx:
        mode = crystal.modes[k]
        pattern = mode.displacement_pattern() / np.sqrt(masses)[:, None]  # c_{ia}/sqrt(m_i), shape (N, 3)
        w = mode.omega_rad_s
        s_e = s_e_single_sided(w) if callable(s_e_single_sided) else float(s_e_single_sided)
        coupling = float(np.einsum("ia,ij,ja->", pattern, g, pattern))  # sum_a (..)
        out.append(E_C * E_C * s_e * coupling / (4.0 * HBAR_J_S * w))
    return np.asarray(out)


def micromotion_sideband_heating_rate(
    s_e_single_sided: SpectralDensity,
    mass_kg: float,
    a: float,
    q: float,
    omega_rf_rad_s: float,
    *,
    n_max: int = 6,
) -> float:
    """Gamma_h = (e^2/(4 m hbar omega)) sum_j |c_j/c_0|^2 S_E(|omega + j Omega|) with the exact Floquet weights (Brownnutt; Section 4.1.5).

    Reduces to the single-frequency formula for q = 0. The j = +-1 weight is q^2/16 at lowest order, Turchette's
    omega^2/(2 Omega^2) at a = 0 (they agree to 0.4% at q = 0.1); never add the two.
    """
    if q == 0.0:
        if a <= 0.0:
            raise ValueError(
                f"a = {a} with q = 0 is not a confined axis: omega = sqrt(a) Omega/2 has no heating rate "
                "(the axis is unstable; check the sign of the dc curvature)"
            )
        omega = math.sqrt(a) * omega_rf_rad_s / 2.0
        return heating_rate_quanta_per_s(s_e_single_sided(omega), mass_kg, omega)
    fc = floquet_coefficients(a, q)
    omega = fc.beta * omega_rf_rad_s / 2.0
    total = 0.0
    for j, weight in fc.sideband_weights().items():
        if abs(j) > n_max:
            continue
        total += weight * s_e_single_sided(abs(omega + j * omega_rf_rad_s))
    return E_C * E_C * total / (4.0 * mass_kg * HBAR_J_S * omega)


def turchette_micromotion_correction(
    s_e_single_sided: SpectralDensity, mass_kg: float, omega_rad_s: float, omega_rf_rad_s: float
) -> float:
    """n_dot = (e^2/(4 m hbar omega))[S_E(omega) + (omega^2/(2 Omega^2))(S_E(Omega - omega) + S_E(Omega + omega))] (Turchette Eq. 4),
    the leading micromotion-sideband correction, absent for axial motion; a CHECK on the Floquet sum, not an addend."""
    w = omega_rad_s
    s = s_e_single_sided(w) + (w * w / (2.0 * omega_rf_rad_s**2)) * (
        s_e_single_sided(omega_rf_rad_s - w) + s_e_single_sided(omega_rf_rad_s + w)
    )
    return E_C * E_C * s / (4.0 * mass_kg * HBAR_J_S * w)


# ---- empirical models of S_E (Section 4.1.5; Brownnutt Eq. 30 and Sec. IV) --------------------------------------------------


def power_law_s_e(
    s0_v2_m2_hz: float,
    *,
    omega0_rad_s: float,
    alpha: float,
    d0_m: float | None = None,
    beta: float = 0.0,
    t0_k: float | None = None,
    gamma: float = 0.0,
) -> Callable[..., float]:
    """S_E(omega, d, T) = S0 (omega/omega0)^-alpha (d/d0)^-beta (T/T0)^gamma, Brownnutt's empirical form with its ranges
    (alpha 0.57 to 1.5 and unstable under surface treatment, beta 2.6 to 4.0; a quoted exponent may refer to S_E or to
    n_dot ~ omega^-(alpha+1)). Returns f(omega, d=d0, T=T0)."""
    if s0_v2_m2_hz < 0.0 or omega0_rad_s <= 0.0:
        raise ValueError("S0 non-negative, omega0 positive")

    def s_e(omega_rad_s: float, d_m: float | None = None, t_k: float | None = None) -> float:
        value = s0_v2_m2_hz * (abs(omega_rad_s) / omega0_rad_s) ** (-alpha)
        if beta != 0.0:
            if d0_m is None or d_m is None:
                raise ValueError("a distance scaling needs d0 and d")
            value *= (d_m / d0_m) ** (-beta)
        if gamma != 0.0:
            if t0_k is None or t_k is None:
                raise ValueError("a temperature scaling needs T0 and T")
            value *= (t_k / t0_k) ** gamma
        return float(value)

    return s_e


def johnson_noise_s_e(temperature_k: float, resistance_ohm: float, distance_m: float) -> float:
    """S_E = 4 k_B T R/d^2 (single-sided), the Johnson-noise floor of an electrode of resistance R at distance d."""
    return 4.0 * K_B_J_PER_K * temperature_k * resistance_ohm / distance_m**2


def patch_potential_s_e(
    patch_density_per_m2: float, s_v_v2_hz: float, patch_radius_m: float, distance_m: float
) -> float:
    """S_E = 3 C S_V r_p^2/(4 a^4): the fluctuating-patch model (patch areal density C, voltage noise S_V, radius r_p,
    distance a), a d^-4 scaling with the single-axis projection <cos^2 theta> = 1/3 folded into the prefactor."""
    return 3.0 * patch_density_per_m2 * s_v_v2_hz * patch_radius_m**2 / (4.0 * distance_m**4)


BROWNNUTT_MEDIANS_V2_M2_HZ: dict[Literal["room_temperature", "cryogenic_6k"], float] = {
    "room_temperature": 40e-12,
    "cryogenic_6k": 0.2e-12,
}
"""Brownnutt 2015's median S_E for 30-230 um traps at 300 K and 6 K (Section 4.1.5), device-preset seeds with their conditions."""


def n_dot_exponent_from_s_e_exponent(alpha: float) -> float:
    """n_dot ~ omega^-(alpha+1) when S_E ~ omega^-alpha (Section 13, "Heating-rate meaning")."""
    return alpha + 1.0
