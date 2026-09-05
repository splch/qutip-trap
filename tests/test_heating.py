"""Section 9.1 (Heating conversion, Heating dynamics, Multi-ion heating), 9.13 row 49 and 9.15 heating targets."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.species import species
from qutip_trap.trap.crystal import build_crystal, solve_crystal
from qutip_trap.trap.heating import (
    BROWNNUTT_MEDIANS_V2_M2_HZ,
    coherence_decay_rate,
    correlation_matrix,
    heating_rate_quanta_per_s,
    heating_rates_per_mode,
    johnson_noise_s_e,
    micromotion_sideband_heating_rate,
    n_dot_exponent_from_s_e_exponent,
    patch_potential_s_e,
    power_law_s_e,
    s_e_from_heating_rate,
    single_sided_from_spectrum,
    single_sided_from_two_sided,
    thermal_collapse_rates,
    turchette_micromotion_correction,
    two_sided_from_single_sided,
)
from qutip_trap.trap.model import Trap
from qutip_trap.units import ATOMIC_MASS_KG, E_C, HBAR_J_S, K_B_J_PER_K, TWO_PI
from tests.test_crystal import _explicit, _Mass


def test_heating_round_trip_of_section_9_15_and_sidedness() -> None:
    """S_E = 2.2250e-13 (V/m)^2/Hz for 9Be+ at 3.6 MHz gives 40 quanta/s; two-sided input needs the factor 2."""
    m = 9.0121822 * ATOMIC_MASS_KG
    w = TWO_PI * 3.6e6
    assert heating_rate_quanta_per_s(2.2250e-13, m, w) == pytest.approx(40.0, rel=1e-3)
    assert s_e_from_heating_rate(40.0, m, w) == pytest.approx(2.2250e-13, rel=1e-3)
    assert heating_rate_quanta_per_s(
        single_sided_from_two_sided(two_sided_from_single_sided(2.2250e-13)), m, w
    ) == pytest.approx(40.0, rel=1e-3)
    assert heating_rate_quanta_per_s(2 * 1.1125e-13, m, w) == pytest.approx(
        2 * heating_rate_quanta_per_s(1.1125e-13, m, w)
    )
    s_e = single_sided_from_spectrum(np.array([0.0, 1e8]), np.array([1e-13, 1e-13]))
    assert s_e(w) == pytest.approx(2e-13) and s_e(-w) == s_e(w)
    with pytest.raises(ValueError):
        heating_rate_quanta_per_s(1e-13, m, 0.0)


def test_collapse_rates_and_coherence_decay_conventions() -> None:
    assert thermal_collapse_rates(50.0) == (50.0, 50.0)
    down, up = thermal_collapse_rates(50.0, n_bar_bath=1e6)
    assert up == pytest.approx(50.0) and down == pytest.approx(50.0 * (1 + 1e-6))
    assert (
        coherence_decay_rate(1.0, 0, 1) == 2.0
        and coherence_decay_rate(1.0, 1, 2) == 4.0
        and coherence_decay_rate(1.0, 3, 6) == 10.0
    )
    assert coherence_decay_rate(1.0, 0, 1, n_bar_bath=1e6) == pytest.approx(2.0, rel=1e-5)


def test_mesolve_heating_is_linear_and_coherences_decay_at_n_plus_m_plus_one() -> None:
    """n_bar(t) = n_bar_0 + Gamma_h t under sqrt(Gamma_h) a and sqrt(Gamma_h) a^dag; the coherence of (|n0> + |m0>)/sqrt 2 decays
    at Gamma_h (n0 + m0 + 1) initially (2.01, 4.03, 10.09 Gamma_h over a short mesolve window; Section 4.1.5)."""
    gamma_h = 100.0
    d = 40
    a = qt.destroy(d)
    c_ops = [math.sqrt(gamma_h) * a, math.sqrt(gamma_h) * a.dag()]
    times = np.linspace(0.0, 10e-3, 21)
    res = qt.mesolve(0 * a, qt.basis(d, 0), times, c_ops, e_ops=[a.dag() * a])
    slope = np.polyfit(times, res.expect[0], 1)[0]
    assert slope == pytest.approx(gamma_h, rel=2e-3)
    assert res.expect[0][-1] == pytest.approx(gamma_h * times[-1], rel=2e-3)
    # the exact initial rate is Gamma_h (n0 + m0 + 1) = 2, 4, 10 Gamma_h (the plan's 2.01, 4.03, 10.09 are its finite-window fit);
    # beyond the first instants the decay is not exponential because the jumps repopulate rho_{n+-1, m+-1}
    for (n0, m0), expected in (((0, 1), 2.0), ((1, 2), 4.0), ((3, 6), 10.0)):
        psi = (qt.basis(d, n0) + qt.basis(d, m0)).unit()
        proj = qt.basis(d, n0) * qt.basis(d, m0).dag()
        t_short = np.linspace(0.0, 1e-5, 11)
        out = qt.mesolve(0 * a, psi, t_short, c_ops, e_ops=[proj])
        coh = np.abs(out.expect[0])
        rate = -np.polyfit(t_short, np.log(coh), 1)[0] / gamma_h
        assert rate == pytest.approx(expected, rel=3e-3)
        assert rate == pytest.approx(coherence_decay_rate(1.0, n0, m0), rel=3e-3)
        # the analytic initial slope from the Liouvillian: d rho_nm/dt = -Gamma_h (n + m + 1) rho_nm at t = 0
        lind = qt.liouvillian(0 * a, c_ops)
        drho = qt.vector_to_operator(lind * qt.operator_to_vector(psi * psi.dag()))
        assert (drho[n0, m0] / (psi * psi.dag())[n0, m0]).real == pytest.approx(
            -gamma_h * (n0 + m0 + 1), rel=1e-12
        )


def test_uniform_noise_heats_only_the_com_at_n_times_the_single_ion_rate() -> None:
    """Lechner 2016: 9 x 7.2 = 65 quanta/s on the 9-ion COM at 2.74 MHz; uncorrelated noise heats every mode at the single-ion rate."""
    yb = species("171Yb+")
    m = yb.mass_u * ATOMIC_MASS_KG
    w_com = TWO_PI * 2.74e6
    s_e = s_e_from_heating_rate(7.2, m, w_com)
    cr = solve_crystal(_explicit((2.74e6, 2.9e6, 0.4e6)), (yb,) * 9)
    uniform = heating_rates_per_mode(cr, s_e, math.inf)
    com = cr.mode_index("transverse_1", 8)
    assert uniform[com] == pytest.approx(65.0, abs=0.3) and uniform[com] == pytest.approx(9 * 7.2, rel=1e-9)
    for k in range(8):
        assert uniform[cr.mode_index("transverse_1", k)] < 1e-12 * uniform[com]
    assert uniform[cr.mode_index("axial", 0)] == pytest.approx(
        9 * heating_rate_quanta_per_s(s_e, m, cr.modes[0].omega_rad_s), rel=1e-9
    )
    uncorrelated = heating_rates_per_mode(cr, s_e, 0.0)
    for k, mode in enumerate(cr.modes):
        assert uncorrelated[k] == pytest.approx(heating_rate_quanta_per_s(s_e, m, mode.omega_rad_s), rel=1e-9)
    partial = heating_rates_per_mode(cr, s_e, 5e-6)
    assert uniform[com] > partial[com] > uncorrelated[com]
    assert correlation_matrix(cr.positions_m, math.inf).min() == 1.0 and np.allclose(
        correlation_matrix(cr.positions_m, 0.0), np.eye(9)
    )


def test_mixed_crystal_every_mode_heats_and_the_two_ion_sum_rule_holds() -> None:
    """A mixed pair has no mode orthogonal to a uniform field; sum_k hbar omega_k n_dot_k = (e^2 S_E/4)(1/m1 + 1/m2) for every mu (Wubbena Eqs. 30-31)."""
    for mu in (0.5, 2.6614, 4.6):
        w1 = TWO_PI * 1e6
        w = np.array(
            [[8 * w1, 8 * w1, w1], [8 * w1 / math.sqrt(mu), 8 * w1 / math.sqrt(mu), w1 / math.sqrt(mu)]]
        )
        cr = build_crystal((_Mass(25.0), _Mass(25.0 * mu)), w)  # type: ignore[arg-type]
        s_e = 1e-13
        rates = heating_rates_per_mode(cr, s_e, math.inf)
        axial = [cr.mode_index("axial", k) for k in (0, 1)]
        assert all(rates[k] > 0.05 * rates[axial[0]] for k in axial)
        energy_rate = sum(HBAR_J_S * cr.modes[k].omega_rad_s * rates[k] for k in axial)
        expected = E_C**2 * s_e / 4.0 * np.sum(1.0 / cr.masses_kg)
        assert energy_rate == pytest.approx(expected, rel=1e-9)
    equal = build_crystal((_Mass(25.0), _Mass(25.0)), np.array([[8 * w1, 8 * w1, w1]] * 2))  # type: ignore[arg-type]
    rates_eq = heating_rates_per_mode(equal, 1e-13, math.inf)
    assert rates_eq[equal.mode_index("axial", 1)] < 1e-12 * rates_eq[equal.mode_index("axial", 0)], (
        "breathing weight identically zero at mu = 1"
    )


def test_parity_rule_for_a_reflection_symmetric_mixed_chain() -> None:
    """Be-Mg-Be: exactly (N - 1)/2 = 1 axial mode has an antisymmetric pattern (centre ion at rest) and zero uniform-field heating."""
    w1 = TWO_PI * 1e6
    mu = 23.985042 / 9.0121822
    w_be = np.array([8 * w1, 8 * w1, w1])
    cr = build_crystal(
        (_Mass(9.0121822), _Mass(23.985042), _Mass(9.0121822)), np.vstack([w_be, w_be / math.sqrt(mu), w_be])
    )  # type: ignore[arg-type]
    rates = heating_rates_per_mode(cr, 1e-13, math.inf)
    axial_rates = [rates[cr.mode_index("axial", k)] for k in range(3)]
    exempt = [k for k, r in enumerate(axial_rates) if r < 1e-15 * max(axial_rates)]
    assert exempt == [1]
    m = cr.family("axial")[1]
    assert abs(m.eigenvector[1]) < 1e-10 and m.eigenvector[0] == pytest.approx(-m.eigenvector[2])
    weights = m.eigenvector / np.sqrt(cr.masses_kg)
    assert abs(np.sum(weights)) < 1e-12 * np.sum(np.abs(weights)), (
        "sum_i c_i/sqrt(m_i) = 0 for the even-parity mode"
    )
    assert cr.family("axial")[0].eigenvector[0] == pytest.approx(cr.family("axial")[0].eigenvector[2]), (
        "the in-phase lowest mode is symmetric, hence odd under Morigi-Walther's parity"
    )


def test_micromotion_sideband_sum_matches_turchette_at_lowest_order_and_never_adds() -> None:
    m = 40.0 * ATOMIC_MASS_KG
    omega_rf = TWO_PI * 30e6
    q = 0.1
    s_e = power_law_s_e(1e-12, omega0_rad_s=TWO_PI * 1e6, alpha=1.0)
    floquet = micromotion_sideband_heating_rate(s_e, m, 0.0, q, omega_rf)
    from qutip_trap.trap.mathieu import beta_exact

    w = beta_exact(0.0, q) * omega_rf / 2
    base = heating_rate_quanta_per_s(s_e(w), m, w)
    turchette = turchette_micromotion_correction(s_e, m, w, omega_rf)
    assert (floquet - base) == pytest.approx(turchette - base, rel=0.03), (
        "the j = +-1 weight is q^2/16 = Turchette's omega^2/(2 Omega^2)"
    )
    assert floquet == pytest.approx(turchette, rel=1e-3)
    assert micromotion_sideband_heating_rate(s_e, m, 0.01, 0.0, omega_rf) == pytest.approx(
        heating_rate_quanta_per_s(s_e(0.1 * omega_rf / 2), m, 0.1 * omega_rf / 2)
    )


def test_empirical_models_and_exponent_bookkeeping() -> None:
    s_e = power_law_s_e(
        40e-12, omega0_rad_s=TWO_PI * 1e6, alpha=1.0, d0_m=100e-6, beta=4.0, t0_k=300.0, gamma=1.5
    )
    assert s_e(TWO_PI * 2e6, 100e-6, 300.0) == pytest.approx(20e-12)
    assert s_e(TWO_PI * 1e6, 200e-6, 300.0) == pytest.approx(40e-12 / 16)
    assert s_e(TWO_PI * 1e6, 100e-6, 1200.0) == pytest.approx(40e-12 * 8)
    with pytest.raises(ValueError):
        s_e(TWO_PI * 1e6)
    assert johnson_noise_s_e(300.0, 10.0, 100e-6) == pytest.approx(4 * K_B_J_PER_K * 300 * 10 / 1e-8)
    assert patch_potential_s_e(1e10, 1e-18, 1e-7, 100e-6) == pytest.approx(
        3 * 1e10 * 1e-18 * 1e-14 / (4 * 1e-16)
    )
    assert n_dot_exponent_from_s_e_exponent(1.0) == 2.0
    assert BROWNNUTT_MEDIANS_V2_M2_HZ["room_temperature"] / BROWNNUTT_MEDIANS_V2_M2_HZ[
        "cryogenic_6k"
    ] == pytest.approx(200.0)


def test_heating_rates_need_an_explicit_correlation_length() -> None:
    yb = species("171Yb+")
    cr = solve_crystal(_explicit((3e6, 3e6, 1e6)), (yb, yb))
    with pytest.raises(TypeError):
        heating_rates_per_mode(cr, 1e-13)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        heating_rates_per_mode(cr, 1e-13, -1.0)
    assert isinstance(Trap, type)
