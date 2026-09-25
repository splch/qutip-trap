"""Electric-field noise to heating rates: the single-ion conversion, the S_E adapter of ``NoiseSpectrum`` and the
multi-ion correlation and mass structure."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.device.presets import secular_trap
from qutip_trap.noise.spectra import NoiseSpectrum, power_law_spectrum
from qutip_trap.species import species
from qutip_trap.trap.crystal import build_crystal, solve_crystal
from qutip_trap.trap.heating import (
    correlation_matrix,
    heating_rate_quanta_per_s,
    heating_rates_per_mode,
    s_e_from_heating_rate,
    single_sided_from_spectrum,
    thermal_collapse_rates,
)
from qutip_trap.units import ATOMIC_MASS_KG, E_C, HBAR_J_S, TWO_PI
from tests.fixtures import MassOnly, chain_device


def test_heating_round_trip_and_sidedness() -> None:
    """S_E = 2.2250e-13 (V/m)^2/Hz for 9Be+ at 3.6 MHz gives 40 quanta/s and back (to 1e-3), a two-sided density entering
    with the factor 2."""
    m = 9.0121822 * ATOMIC_MASS_KG
    w = TWO_PI * 3.6e6
    assert heating_rate_quanta_per_s(2.2250e-13, m, w) == pytest.approx(40.0, rel=1e-3)
    assert s_e_from_heating_rate(40.0, m, w) == pytest.approx(2.2250e-13, rel=1e-3)
    s_e = single_sided_from_spectrum(
        NoiseSpectrum(np.array([0.0, 1e8]), np.array([1e-13, 1e-13]), "(V/m)^2/(rad/s)")
    )
    assert s_e(w) == pytest.approx(2e-13) and s_e(-w) == s_e(w)
    with pytest.raises(ValueError):
        heating_rate_quanta_per_s(1e-13, m, 0.0)


def test_collapse_rates_carry_n_plus_one_and_n() -> None:
    assert thermal_collapse_rates(50.0) == (50.0, 50.0)
    down, up = thermal_collapse_rates(50.0, n_bar_bath=1e6)
    assert up == pytest.approx(50.0) and down == pytest.approx(50.0 * (1 + 1e-6))


def test_a_symmetric_two_sided_tabulation_keeps_its_shape() -> None:
    """S_E = 2 x NoiseSpectrum.tabulated, even in omega, at DC, mid-band and the edge of a symmetric table, to 1e-12."""
    spectrum = NoiseSpectrum(
        np.array([-3e7, -1e7, 0.0, 1e7, 3e7]),
        np.array([1e-14, 5e-14, 1e-13, 5e-14, 1e-14]),
        "(V/m)^2/(rad/s)",
    )
    s_e = single_sided_from_spectrum(spectrum)
    for omega, expected in ((0.0, 2e-13), (1e7, 1e-13), (2e7, 6e-14), (3e7, 2e-14)):
        assert s_e(omega) == pytest.approx(expected, rel=1e-12)
        assert s_e(omega) == pytest.approx(2.0 * float(spectrum.tabulated(omega)), rel=1e-12)
        assert s_e(-omega) == pytest.approx(s_e(omega), rel=1e-12)
    assert s_e(0.0) / s_e(3e7) == pytest.approx(10.0, rel=1e-12)


def test_above_the_tabulated_band_the_density_is_the_white_level_alone() -> None:
    """Above the tabulated band S_E is twice the white level alone (1e-12), and every two-ion mode there heats at that rate
    (1e-9)."""
    white_two_sided = 3e-15
    spectrum = power_law_spectrum(
        1e-13,
        2.0 * np.pi * 1e6,
        1.0,
        "(V/m)^2/(rad/s)",
        omega_min_rad_s=2.0 * np.pi * 1e3,
        omega_max_rad_s=2.0 * np.pi * 0.5e6,
        n=201,
    )
    spectrum = dataclasses.replace(spectrum, white_level=white_two_sided)
    s_e = single_sided_from_spectrum(spectrum)
    top = spectrum.omega_max_rad_s
    assert s_e(0.5 * top) > 2.0 * white_two_sided, "the tabulated band dominates inside it"
    for omega in (1.5 * top, 10.0 * top, 100.0 * top):
        assert s_e(omega) == pytest.approx(2.0 * white_two_sided, rel=1e-12)
    assert s_e(top) == pytest.approx(2.0 * (float(spectrum.tabulated(top)) + white_two_sided), rel=1e-12)
    dev = chain_device(2)
    dev = dataclasses.replace(
        dev, noise=dataclasses.replace(dev.noise, S_E=spectrum, correlation_length_m=0.0)
    )
    mass = dev.crystal.species[0].mass_u * ATOMIC_MASS_KG
    rates = dev.noise.heating_rates_quanta_per_s(dev)
    assert all(m.omega_rad_s > top for m in dev.crystal.modes), "every fixture mode is above the band"
    for k, mode in enumerate(dev.crystal.modes):
        assert rates[k] == pytest.approx(
            heating_rate_quanta_per_s(2.0 * white_two_sided, mass, mode.omega_rad_s), rel=1e-9
        )


def test_uniform_noise_heats_only_the_com_at_n_times_the_single_ion_rate() -> None:
    """Lechner 2016: uniform noise heats only the 9-ion COM modes, the 2.74 MHz one at 9 x 7.2 = 65 quanta/s (to 0.3), and
    uncorrelated noise heats every mode at the single-ion rate (1e-9)."""
    yb = species("171Yb+")
    m = yb.mass_u * ATOMIC_MASS_KG
    s_e = s_e_from_heating_rate(7.2, m, TWO_PI * 2.74e6)
    cr = solve_crystal(secular_trap((2.74e6, 2.9e6, 0.4e6)), (yb,) * 9)
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
    assert uniform[com] > heating_rates_per_mode(cr, s_e, 5e-6)[com] > uncorrelated[com]
    assert correlation_matrix(cr.positions_m, math.inf).min() == 1.0
    assert np.allclose(correlation_matrix(cr.positions_m, 0.0), np.eye(9))


def test_mixed_crystal_every_mode_heats_and_the_two_ion_sum_rule_holds() -> None:
    """A uniform field heats both axial modes of a mixed pair with sum_k hbar omega_k n_dot_k = (e^2 S_E/4)(1/m1 + 1/m2) to
    1e-9 (Wubbena 2012 Eqs. 30-31), and not the stretch of an equal-mass pair."""
    w1 = TWO_PI * 1e6
    for mu in (0.5, 2.6614, 4.6):
        w = np.array(
            [[8 * w1, 8 * w1, w1], [8 * w1 / math.sqrt(mu), 8 * w1 / math.sqrt(mu), w1 / math.sqrt(mu)]]
        )
        cr = build_crystal((MassOnly(25.0), MassOnly(25.0 * mu)), w)  # type: ignore[arg-type]
        rates = heating_rates_per_mode(cr, 1e-13, math.inf)
        axial = [cr.mode_index("axial", k) for k in (0, 1)]
        assert all(rates[k] > 0.05 * rates[axial[0]] for k in axial)
        energy_rate = sum(HBAR_J_S * cr.modes[k].omega_rad_s * rates[k] for k in axial)
        assert energy_rate == pytest.approx(E_C**2 * 1e-13 / 4.0 * np.sum(1.0 / cr.masses_kg), rel=1e-9)
    equal = build_crystal((MassOnly(25.0), MassOnly(25.0)), np.array([[8 * w1, 8 * w1, w1]] * 2))  # type: ignore[arg-type]
    rates_eq = heating_rates_per_mode(equal, 1e-13, math.inf)
    assert rates_eq[equal.mode_index("axial", 1)] < 1e-12 * rates_eq[equal.mode_index("axial", 0)]


def test_parity_rule_for_a_reflection_symmetric_mixed_chain() -> None:
    """Be-Mg-Be: exactly one axial mode has an antisymmetric pattern (centre ion at rest) and zero uniform-field heating;
    the in-phase lowest mode is symmetric."""
    w1 = TWO_PI * 1e6
    w_be = np.array([8 * w1, 8 * w1, w1])
    mu = 23.985042 / 9.0121822
    cr = build_crystal(
        (MassOnly(9.0121822), MassOnly(23.985042), MassOnly(9.0121822)),
        np.vstack([w_be, w_be / math.sqrt(mu), w_be]),
    )  # type: ignore[arg-type]
    rates = heating_rates_per_mode(cr, 1e-13, math.inf)
    axial_rates = [rates[cr.mode_index("axial", k)] for k in range(3)]
    assert [k for k, r in enumerate(axial_rates) if r < 1e-15 * max(axial_rates)] == [1]
    m = cr.family("axial")[1]
    assert abs(m.eigenvector[1]) < 1e-10 and m.eigenvector[0] == pytest.approx(-m.eigenvector[2])
    weights = m.eigenvector / np.sqrt(cr.masses_kg)
    assert abs(np.sum(weights)) < 1e-12 * np.sum(np.abs(weights))
    assert cr.family("axial")[0].eigenvector[0] == pytest.approx(cr.family("axial")[0].eigenvector[2])


def test_heating_rates_need_an_explicit_correlation_length() -> None:
    yb = species("171Yb+")
    cr = solve_crystal(secular_trap((3e6, 3e6, 1e6)), (yb, yb))
    with pytest.raises(TypeError):
        heating_rates_per_mode(cr, 1e-13)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        heating_rates_per_mode(cr, 1e-13, -1.0)
