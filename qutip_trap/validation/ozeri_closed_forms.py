"""Ozeri 2005 / 2007 and Uys 2010 closed forms that PLAN.md names but Section 12 left unimplemented
(PLAN.md 447, 457, 683, 1182, 1410, 1411; Sections 9.10, 9.13; the tests of milestone M0a).

Kept apart from :mod:`qutip_trap.validation.atomic_closed_forms` (Ozeri 2007's power-series forms and
Wineland 2003's clock forms) because the three items here are the ones the audit of 2026-09-07 found
*named and required but absent*:

- **Ozeri 2005 Eq. 3**, the clock-qubit differential Stark shift as four detuning-weighted diagonal
  amplitudes. ``light/stark.py``'s docstring claimed it already lived in ``atomic_closed_forms``; it did not.
- the **Table II power regression** of Ozeri 2007 Eq. 17, which makes ``ozeri_epsilon_s_from_power`` live code
  instead of the dead function the audit found. Read :func:`ozeri_table_ii_power_w` before trusting the
  absolute numbers: PLAN.md's own (eps_S, P_0) pair for 9Be+ does **not** satisfy its own Eq. 17.
- **Uys's 9Be+ 4.5 T anchor**, whose point is that Ozeri's rate-difference estimator is exactly zero where
  Gamma_el is largest, so the "factor of five" of the plan's second revision is a ratio to zero.

The Gamma_total/Delta_St saturation (0.9579 gamma/Delta_hf, PLAN.md 447 and the 9.10 row) lives in
``atomic_closed_forms`` as ``ozeri_photons_per_stark_radian`` beside the exact Wineland identity it corrects;
it is not duplicated here.

Conventions follow ``atomic_closed_forms``: ``gamma`` is the ANGULAR P-level decay rate, ``delta`` the plan's
detuning omega_L - omega(P1/2) from the LOWER ground state (positive blue), ``delta_f`` the angular
fine-structure splitting and ``delta_hf`` the angular ground-state hyperfine splitting; Ozeri's ``g`` is the
half-convention stretched-state coupling. Ozeri's ``a^(J)_{ss'}`` are signed real amplitudes normalized so
that a^(1/2) + a^(3/2) is the same for both clock states (the LS sum rule), which is why the leading 1/Delta
term of Eq. 3 cancels and the shift falls as delta_hf/Delta^2.
"""

from __future__ import annotations

from qutip_trap.units import C_M_PER_S, HBAR_J_S, TWO_PI
from qutip_trap.validation.atomic_closed_forms import ozeri_epsilon_s_from_power

MU_B_HZ_PER_TESLA: float = 13.996e9
"""mu_B/h in Hz/T to the four figures PLAN.md 9.16 uses for the Uys order-of-magnitude sanity check."""

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
    """Delta_St = g^2[a^(1/2)_uu/(D+D_hf) + a^(3/2)_uu/(D+D_hf-D_f) - a^(1/2)_dd/D - a^(3/2)_dd/(D-D_f)] (Eq. 3).

    The four denominators are the three-index detunings of Section 4.5.6 read off the two ground states: the
    upper clock state sits delta_hf above the lower one, so its two pathways are detuned by D + D_hf and
    D + D_hf - D_f while the lower state's are D and D - D_f. With the LS-symmetric amplitudes
    a^(1/2) = 1/3, a^(3/2) = 2/3 for both states this collapses, to first order in delta_hf/D, onto
    Wineland 2003 Eq. 2.17, -(g^2)(delta_hf/3)[1/D^2 + 2/(D - D_f)^2] -- the cross-check that the four
    amplitudes were read as diagonal line strengths and not as branching ratios.
    """
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
    """Gamma_{i,f} = g^2 gamma |a^(1/2)/D + a^(3/2)/(D - D_f)|^2 (PLAN.md 1182, Section 9.10).

    The two amplitudes are SIGNED and the sum is coherent: for a Raman (spin-flip) channel they have
    opposite signs (-sqrt2/3 and +sqrt2/3 on the 171Yb+-like clock line), which is the origin of the
    1/Delta^4 fall-off; for the elastic channels they share a sign and the rate falls as 1/Delta^2.
    """
    if delta == 0.0 or delta == delta_f:
        raise ZeroDivisionError("Ozeri 2005's scattering amplitudes are valid only away from resonance")
    return g_squared * gamma * abs(a_half / delta + a_three_half / (delta - delta_f)) ** 2


# ---- Ozeri 2007 Eq. 17: the Table II power regression ---------------------------------------------------


def ozeri_eq_17_constant_w(omega_r_half: float, omega_32: float, w0_m: float) -> float:
    """eps_S x P = 2 pi |Omega_R| hbar omega_{3/2}^3 w_0^2/(3 c^2), the invariant of Eq. 17 in watts.

    Derived rather than transcribed: eps_S = P_Raman = 2 pi |Omega_R| (gamma/g^2) with
    gamma/g^2 = 4 hbar omega^3/(3 pi eps0 c^3 E^2) and the Gaussian peak E^2 = 4P/(pi eps0 c w_0^2), which
    reproduces Eq. 17 exactly. A (eps_S, P_0) pair from Table II must lie on this hyperbola.
    """
    return TWO_PI * abs(omega_r_half) * HBAR_J_S * omega_32**3 * w0_m**2 / (3.0 * C_M_PER_S**2)


def ozeri_table_ii_power_w(
    epsilon_s: float, omega_r_half: float, wavelength_32_vac_m: float, w0_m: float
) -> float:
    """P_0 = 2 pi |Omega_R| hbar omega_{3/2}^3 w_0^2/(3 c^2 eps_S) with omega_{3/2} = 2 pi c/lambda (Eq. 17 inverted).

    The plan (Section 9.13 "Ozeri scattering optimum and power regression") reads Ozeri's printed
    "lambda_{3/2} = c/omega_{3/2}" as a typo for 2 pi c/omega, which is the reading taken here; the
    alternative reading is :func:`ozeri_table_ii_power_printed_lambda_w` and differs by exactly (2 pi)^3.

    **The absolute Table II numbers are NOT reproducible from PLAN.md's own inputs.** For 9Be+ at
    w_0 = 20 um, Omega_R/2pi = 0.25 MHz and lambda_{3/2} = 313.0 nm (the plan's Table I value), Eq. 17 as
    printed gives P_0 = 9.986 mW at eps_S = 0.337e-4, not the 3.37 mW the plan prints; the plan's own pair
    violates its own Eq. 17 by a factor 2.9633 (it would need the prefactor 9 rather than 3). The plan's
    "133 mW" negative control matches neither lambda reading either: 133/3.37 = 39.47 = (2 pi)^2, while the
    two readings differ by (2 pi)^3 = 248.05. Ledger: ``anchor.m0a.ozeri_table_ii``. The Yb+ row is excluded
    as the plan instructs (its printed 0.2e-4 / 2 mW against 0.29e-4 / 2.9 mW recomputed).
    """
    if epsilon_s <= 0.0:
        raise ValueError("a scattering error per pulse is positive")
    if wavelength_32_vac_m <= 0.0:
        raise ValueError("the P3/2 wavelength must be positive")
    omega_32 = TWO_PI * C_M_PER_S / wavelength_32_vac_m
    return ozeri_eq_17_constant_w(omega_r_half, omega_32, w0_m) / epsilon_s


def ozeri_table_ii_power_printed_lambda_w(
    epsilon_s: float, omega_r_half: float, wavelength_32_vac_m: float, w0_m: float
) -> float:
    """The NEGATIVE CONTROL: the same regression with Ozeri's printed lambda_{3/2} = c/omega_{3/2}.

    Taking the printed relation literally makes omega_{3/2} = c/lambda, a factor 2 pi small, so
    P_0 comes out (2 pi)^3 = 248.05 times smaller. The plan flags the printed relation as a typo
    ([verified flag], PLAN.md 457); this function exists so the flag is a test rather than a comment.
    """
    if epsilon_s <= 0.0:
        raise ValueError("a scattering error per pulse is positive")
    omega_32 = C_M_PER_S / wavelength_32_vac_m  # the printed relation, read literally
    return ozeri_eq_17_constant_w(omega_r_half, omega_32, w0_m) / epsilon_s


def ozeri_epsilon_s_at_power(
    power_w: float, omega_r_half: float, wavelength_32_vac_m: float, w0_m: float
) -> float:
    """eps_S at a given power per beam, the forward direction of Eq. 17 (a thin wrapper that fixes lambda -> omega).

    Delegates to :func:`~qutip_trap.validation.atomic_closed_forms.ozeri_epsilon_s_from_power`, which the
    audit found to be dead code; this is its live caller.
    """
    omega_32 = TWO_PI * C_M_PER_S / wavelength_32_vac_m
    return ozeri_epsilon_s_from_power(omega_r_half, omega_32, w0_m, power_w)


# ---- Uys 2010: the 9Be+ 4.5 T anchor --------------------------------------------------------------------


class Uys2010Be9Anchor:
    """The named constants of Uys et al. 2010's 9Be+ electron-spin-qubit anchor (PLAN.md 683, 1410, 1529)."""

    ZEEMAN_SPLITTING_HZ: float = 124.1e9
    """Omega_z/2pi at 4.5 T, as printed."""
    FIELD_T: float = 4.5
    WAVELENGTH_NM: float = 313.0
    P32_RESONANCE_LOW_HZ: float = -79.4e9
    """The lower P3/2 resonance, measured from the cycling transition."""
    P32_RESONANCE_HIGH_HZ: float = -37.7e9
    EQUAL_ELASTIC_RATE_DETUNING_HZ: float = -56.0e9
    """Where the two qubit states' elastic rates coincide, i.e. where the rate-difference estimator vanishes
    and Gamma_el is largest. It lies between the two P3/2 resonances, as it must."""


def free_electron_zeeman_splitting_hz(field_t: float) -> float:
    """2 (mu_B/h) B, the free-electron spin splitting: PLAN.md 9.16's order-of-magnitude check on Omega_z.

    2 x 13.996 GHz/T x 4.5 T = 125.96 GHz against Uys's printed 124.1 GHz (1.5% high, which is the
    hyperfine and g-factor structure the free-electron estimate omits). "Order-of-magnitude sanity only"
    is the plan's own qualification.
    """
    return 2.0 * MU_B_HZ_PER_TESLA * field_t


def ozeri_rate_difference_estimate(gamma_uu: float, gamma_dd: float) -> float:
    """(Gamma_uu - Gamma_dd)^2/(Gamma_uu + Gamma_dd), Ozeri 2007's eps_delta estimator of Rayleigh dephasing.

    It is EXACTLY ZERO where the two elastic rates coincide, while Uys's Gamma_el -- a squared difference of
    amplitude SUMS -- is largest there. So the "about 5x" of PLAN.md 1410 is a ratio to zero (the plan's own
    [corrected] reading), and no finite factor may be pinned: what is testable is that the estimator
    vanishes identically at that detuning while the amplitude form does not.
    """
    total = gamma_uu + gamma_dd
    if total <= 0.0:
        raise ValueError("at least one elastic rate must be positive")
    return (gamma_uu - gamma_dd) ** 2 / total


def uys_elastic_rates_two_path(
    a_half: float, a_three_half: float, delta_from_p12: float, delta_f: float, gamma: float, g_squared: float
) -> float:
    """One qubit state's elastic rate through the two P pathways: the same coherent two-amplitude sum as Eq. 3.

    A convenience for the 4.5 T anchor's construction: with SIGNED amplitudes the two paths can add or
    cancel, so two states with different amplitude pairs cross in rate at some detuning between the
    resonances -- Uys's -56 GHz -- while their amplitude DIFFERENCE (and hence Gamma_el) stays finite.
    """
    return ozeri_2005_scattering_rate(g_squared, gamma, a_half, a_three_half, delta_from_p12, delta_f)


def equal_elastic_rate_detuning(
    a_up: tuple[float, float], a_down: tuple[float, float], delta_f: float
) -> float:
    """The detuning at which two states' elastic amplitude sums have equal MAGNITUDE, solved in closed form.

    With A_s(D) = a^(1/2)_s/D + a^(3/2)_s/(D - D_f), |A_up| = |A_down| when A_up = -A_down (the roots of
    A_up = +A_down being at infinity when the LS sum rule makes the leading 1/D coefficients equal). Writing
    u = a^(1/2)_up + a^(1/2)_dn and v = a^(3/2)_up + a^(3/2)_dn, A_up + A_down = 0 gives
    u(D - D_f) + v D = 0, i.e. D = u D_f/(u + v).
    """
    u = a_up[0] + a_down[0]
    v = a_up[1] + a_down[1]
    if u + v == 0.0:
        raise ValueError("the two amplitude sums never cross: u + v = 0")
    return u * delta_f / (u + v)
