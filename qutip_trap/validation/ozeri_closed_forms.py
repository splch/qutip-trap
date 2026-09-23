"""Closed forms of Ozeri 2005 (the clock-qubit differential Stark shift, Eq. 3), Ozeri 2007 (the Table II power
regression of Eq. 17) and Uys et al. 2010 (the 9Be+ 4.5 T anchor), in the conventions of ``atomic_closed_forms``;
``delta_f`` and ``delta_hf`` are the angular fine-structure and ground-state hyperfine splittings. Ozeri's ``a^(J)_{ss'}``
are signed real amplitudes with a^(1/2) + a^(3/2) the same for both clock states (the LS sum rule), so the leading
1/Delta term of Eq. 3 cancels and the shift falls as delta_hf/Delta^2."""

from __future__ import annotations

from qutip_trap.units import C_M_PER_S, HBAR_J_S, TWO_PI
from qutip_trap.validation.atomic_closed_forms import ozeri_epsilon_s_from_power

MU_B_HZ_PER_TESLA: float = 13.996e9
"""mu_B/h in Hz/T to four figures, enough for the Uys order-of-magnitude check."""

# ---- Ozeri 2005 Eq. 3: the clock-qubit differential Stark shift -----------------------------------------


def ozeri_2005_clock_stark_shift(
    g_squared: float,
    a_half_uu: float,
    a_three_half_uu: float,
    a_half_dd: float,
    a_three_half_dd: float,
    delta: float,
    delta_hf: float,
    delta_f: float,
) -> float:
    """Delta_St = g^2[a^(1/2)_uu/(D+D_hf) + a^(3/2)_uu/(D+D_hf-D_f) - a^(1/2)_dd/D - a^(3/2)_dd/(D-D_f)] (Ozeri 2005
    Eq. 3), the upper clock state delta_hf above the lower; with a^(1/2) = 1/3, a^(3/2) = 2/3 it reduces to first order in
    delta_hf/D onto Wineland 2003 Eq. 2.17."""
    for name, denominator in (
        ("D", delta),
        ("D + D_hf", delta + delta_hf),
        ("D - D_f", delta - delta_f),
        ("D + D_hf - D_f", delta + delta_hf - delta_f),
    ):
        if denominator == 0.0:
            raise ZeroDivisionError(
                f"Ozeri 2005 Eq. 3 has no i gamma/2 in its denominators (Section 4.5.6): {name} is zero"
            )
    return g_squared * (
        a_half_uu / (delta + delta_hf)
        + a_three_half_uu / (delta + delta_hf - delta_f)
        - a_half_dd / delta
        - a_three_half_dd / (delta - delta_f)
    )


def ozeri_2005_scattering_rate(
    g_squared: float, gamma: float, a_half: float, a_three_half: float, delta: float, delta_f: float
) -> float:
    """Gamma_{i,f} = g^2 gamma |a^(1/2)/D + a^(3/2)/(D - D_f)|^2 (Ozeri 2005), a coherent sum of signed amplitudes:
    opposite signs (Raman) fall as 1/Delta^4, equal signs (elastic) as 1/Delta^2."""
    if delta == 0.0 or delta == delta_f:
        raise ZeroDivisionError("Ozeri 2005's scattering amplitudes are valid only away from resonance")
    return g_squared * gamma * abs(a_half / delta + a_three_half / (delta - delta_f)) ** 2


# ---- Ozeri 2007 Eq. 17: the Table II power regression ---------------------------------------------------


def ozeri_eq_17_constant_w(omega_r_half: float, omega_32: float, w0_m: float) -> float:
    """eps_S x P = 2 pi |Omega_R| hbar omega_{3/2}^3 w_0^2/(3 c^2) in W, the invariant of Ozeri 2007 Eq. 17 (angular
    frequencies in rad/s)."""
    return TWO_PI * abs(omega_r_half) * HBAR_J_S * omega_32**3 * w0_m**2 / (3.0 * C_M_PER_S**2)


def ozeri_table_ii_power_w(
    epsilon_s: float, omega_r_half: float, wavelength_32_vac_m: float, w0_m: float
) -> float:
    """P_0 = 2 pi |Omega_R| hbar omega_{3/2}^3 w_0^2/(3 c^2 eps_S) in W, Ozeri 2007 Eq. 17 inverted with omega_{3/2} =
    2 pi c/lambda (the printed lambda_{3/2} = c/omega_{3/2} read as a typo). Table II's absolute powers do not follow:
    for 9Be+ (w_0 = 20 um, Omega_R/2pi = 0.25 MHz, 313.0 nm, eps_S = 0.337e-4) this gives 9.986 mW, not the printed 3.37 mW."""
    if epsilon_s <= 0.0:
        raise ValueError("a scattering error per pulse is positive")
    if wavelength_32_vac_m <= 0.0:
        raise ValueError("the P3/2 wavelength must be positive")
    omega_32 = TWO_PI * C_M_PER_S / wavelength_32_vac_m
    return ozeri_eq_17_constant_w(omega_r_half, omega_32, w0_m) / epsilon_s


def ozeri_table_ii_power_printed_lambda_w(
    epsilon_s: float, omega_r_half: float, wavelength_32_vac_m: float, w0_m: float
) -> float:
    """Negative control: the same regression with Ozeri's printed lambda_{3/2} = c/omega_{3/2} taken literally, which
    makes P_0 (2 pi)^3 = 248.05 times smaller."""
    if epsilon_s <= 0.0:
        raise ValueError("a scattering error per pulse is positive")
    omega_32 = C_M_PER_S / wavelength_32_vac_m  # the printed relation, read literally
    return ozeri_eq_17_constant_w(omega_r_half, omega_32, w0_m) / epsilon_s


def ozeri_epsilon_s_at_power(
    power_w: float, omega_r_half: float, wavelength_32_vac_m: float, w0_m: float
) -> float:
    """eps_S at a given power, Ozeri 2007 Eq. 17 forward with omega_{3/2} = 2 pi c/lambda (``ozeri_epsilon_s_from_power``)."""
    omega_32 = TWO_PI * C_M_PER_S / wavelength_32_vac_m
    return ozeri_epsilon_s_from_power(omega_r_half, omega_32, w0_m, power_w)


# ---- Uys 2010: the 9Be+ 4.5 T anchor --------------------------------------------------------------------


class Uys2010Be9Anchor:
    """The constants of Uys et al. 2010's 9Be+ electron-spin-qubit anchor."""

    ZEEMAN_SPLITTING_HZ: float = 124.1e9
    """Omega_z/2pi at 4.5 T, as printed."""
    FIELD_T: float = 4.5
    WAVELENGTH_NM: float = 313.0
    P32_RESONANCE_LOW_HZ: float = -79.4e9
    """The lower P3/2 resonance, measured from the cycling transition."""
    P32_RESONANCE_HIGH_HZ: float = -37.7e9
    EQUAL_ELASTIC_RATE_DETUNING_HZ: float = -56.0e9
    """Where the two qubit states' elastic rates coincide (the rate-difference estimator vanishes, Gamma_el is largest),
    between the two P3/2 resonances."""


def free_electron_zeeman_splitting_hz(field_t: float) -> float:
    """2 (mu_B/h) B in Hz, the free-electron spin splitting: 125.96 GHz at 4.5 T against Uys's 124.1 GHz, an
    order-of-magnitude check."""
    return 2.0 * MU_B_HZ_PER_TESLA * field_t


def ozeri_rate_difference_estimate(gamma_uu: float, gamma_dd: float) -> float:
    """(Gamma_uu - Gamma_dd)^2/(Gamma_uu + Gamma_dd), Ozeri 2007's eps_delta estimator of Rayleigh dephasing: exactly
    zero where the two elastic rates coincide, where Uys's Gamma_el (a squared difference of amplitude sums) is largest."""
    total = gamma_uu + gamma_dd
    if total <= 0.0:
        raise ValueError("at least one elastic rate must be positive")
    return (gamma_uu - gamma_dd) ** 2 / total


def uys_elastic_rates_two_path(
    a_half: float, a_three_half: float, delta_from_p12: float, delta_f: float, gamma: float, g_squared: float
) -> float:
    """One qubit state's elastic rate through the two P pathways, the coherent two-amplitude sum of
    ``ozeri_2005_scattering_rate``."""
    return ozeri_2005_scattering_rate(g_squared, gamma, a_half, a_three_half, delta_from_p12, delta_f)


def equal_elastic_rate_detuning(
    a_up: tuple[float, float], a_down: tuple[float, float], delta_f: float
) -> float:
    """D = u D_f/(u + v), where two states' elastic amplitude sums A_s(D) = a^(1/2)_s/D + a^(3/2)_s/(D - D_f) satisfy
    A_up = -A_down, with u and v the two states' summed a^(1/2) and a^(3/2)."""
    u = a_up[0] + a_down[0]
    v = a_up[1] + a_down[1]
    if u + v == 0.0:
        raise ValueError("the two amplitude sums never cross: u + v = 0")
    return u * delta_f / (u + v)


__all__ = [
    "MU_B_HZ_PER_TESLA",
    "Uys2010Be9Anchor",
    "equal_elastic_rate_detuning",
    "free_electron_zeeman_splitting_hz",
    "ozeri_2005_clock_stark_shift",
    "ozeri_2005_scattering_rate",
    "ozeri_eq_17_constant_w",
    "ozeri_epsilon_s_at_power",
    "ozeri_rate_difference_estimate",
    "ozeri_table_ii_power_printed_lambda_w",
    "ozeri_table_ii_power_w",
    "uys_elastic_rates_two_path",
]
