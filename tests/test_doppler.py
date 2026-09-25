"""Doppler cooling at level A: per-mode nbar_D from the rate framework, the uncooled-mode guard, multi-ion
participation, the force-model cross-check, the validity conditions and the Monroe 1995 and 40Ca+ anchors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.device.presets import OBLIQUE, secular_trap
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import CoolingError, beam_for_transition
from qutip_trap.light.recoil import angular_factor
from qutip_trap.prep.closed_forms import doppler_force_nbar, stenholm_coefficients
from qutip_trap.prep.doppler import (
    UncooledModeError,
    doppler_cooling,
    ion_mode_rates,
    mode_rates,
    models_per_ion,
    optimize_detuning,
    with_detuning_offset,
)
from qutip_trap.prep.validity import (
    LAMB_DICKE_MAX,
    SMALL,
    ValidityError,
    assert_adiabatic,
    assert_doppler_recoil_limit,
    assert_lamb_dicke,
    recoil_frequency_rad_s,
)
from qutip_trap.species import species
from qutip_trap.species.polarization import spherical_basis
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, ELECTRON_MASS_U, TWO_PI
from tests.fixtures import (
    MASS_KG,
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    WAVELENGTH_M,
    gamma_rad_s,
    power_for_rabi,
    sigma_plus_beam,
    structure,
    two_level_atom,
)


def sigma_plus_along(
    st: AtomicStructure, k_hat: tuple[float, float, float], omega: float, delta: float, waist_m: float = 20e-6
) -> Beam:
    """A sigma+ beam propagating along B = k_hat (pure helicity about the field) with Rabi frequency ``omega`` on the closed line."""
    _e_minus, _e_zero, e_plus = spherical_basis(k_hat)
    pol = tuple(complex(x) for x in e_plus)
    power = power_for_rabi(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega, waist_m, pol, k_hat)  # type: ignore[arg-type]
    omega_l = (
        TWO_PI * (st.state(TWO_LEVEL_EXCITED_PLUS).energy_hz - st.state(TWO_LEVEL_GROUND).energy_hz) + delta
    )
    return Beam(TWO_PI * C_M_PER_S / omega_l, k_hat, pol, waist_m, power, (0.0, 0.0, 0.0))  # type: ignore[arg-type]


def test_doppler_stage_reproduces_the_rate_framework_limit_on_every_mode_and_the_force_model_cross_check() -> (
    None
):
    """One sigma+ beam along B = (1,1,1)/sqrt3 cools every mode to (Gamma/4 nu)(1 + alpha/cos^2) - 1/2 with cos^2 = 1/3
    (3 %), the nu = Gamma/20 axial mode to 9.5 (6e-3) where the force model gives 9.5 to 1e-5."""
    sp = two_level_atom()
    g = gamma_rad_s()
    nu = 0.05 * g
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, nu / TWO_PI)), (sp,))
    st = structure(sp, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, 0.05 * g, -0.5 * g)
    res = doppler_cooling(st, [beam], crystal)
    assert res.illuminated == (0,)
    for m in res.modes:
        alpha = angular_factor(1, float(np.dot(crystal.modes[m.mode].e_hat, OBLIQUE)))
        assert m.projection_cos2_max == pytest.approx(1.0 / 3.0)
        closed = g / (4.0 * m.omega_rad_s) * (1.0 + alpha / m.projection_cos2_max) - 0.5
        assert m.nbar == pytest.approx(closed, rel=0.03)
        assert m.participation_weight == pytest.approx(1.0)
    axial = res.mode(0)
    assert axial.nbar == pytest.approx(9.5, rel=6e-3)
    assert res.force_model_nbar[0] == pytest.approx(9.5, abs=1e-5)
    assert 1 not in res.force_model_nbar  # nu/Gamma = 0.128 > 0.1: the force model is not reported there
    assert set(res.lamb_dicke_guard()) == {0, 1, 2}
    assert res.weighted_nbar == pytest.approx(float(np.mean([m.nbar for m in res.modes])))


def test_a_mode_no_cooling_beam_addresses_raises_instead_of_returning_a_steady_state() -> None:
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
    st = structure(sp)
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * g, -0.5 * g)  # along z = B
    with pytest.raises(UncooledModeError):
        doppler_cooling(st, [beam], crystal)
    # the z mode alone is fine
    res = doppler_cooling(st, [beam], crystal, modes=[0])
    assert [m.mode for m in res.modes] == [0] and res.mode(0).nbar == pytest.approx(6.54, abs=0.05)


def test_one_beam_along_one_axis_leaves_the_transverse_occupations_growing_linearly() -> None:
    """A beam along z gives the transverse modes no projection and heating and cooling rates both equal to the recoil 2D
    (1e-15), so W = 0 and their nbar is refused, while the axial mode cools."""
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
    st = structure(sp)
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * g, -0.5 * g)  # along z = B
    models = models_per_ion(st, [beam], crystal, (0,))
    driven = crystal.mode_index("axial", 0)
    for m in range(len(crystal.modes)):
        r = ion_mode_rates(models[0], crystal, 0, m)
        assert r is not None
        if m == driven:
            assert r.cooling_per_s > r.heating_per_s
            continue
        assert r.projection_cos2[0] < 1e-12  # the beam does not project on this axis at all
        assert r.heating_per_s == pytest.approx(r.two_d_per_s, rel=1e-15)
        assert r.cooling_per_s == pytest.approx(r.two_d_per_s, rel=1e-15)
        assert r.two_d_per_s > 0.0
        rates = mode_rates(models, crystal, m, projection_threshold=0.0)
        assert rates.rate_per_s == pytest.approx(0.0, abs=1e-12 * r.two_d_per_s)
        with pytest.raises(CoolingError):
            _ = rates.nbar


def test_optimum_detuning_of_a_low_frequency_mode_is_minus_half_the_linewidth() -> None:
    """With all the weight on the nu = Gamma/20 mode the optimum detuning is -(Gamma/2) sqrt(1 + s) (RMP Eq. 106) to
    0.02 Gamma, and a heating configuration inside the bounds is skipped."""
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
    st = structure(sp, b_hat=OBLIQUE)
    beams = [sigma_plus_along(st, OBLIQUE, 0.05 * g, -0.5 * g)]

    def evaluate(offset: float):  # type: ignore[no-untyped-def]
        return doppler_cooling(st, with_detuning_offset(beams, [0], offset), crystal, weights={0: 1.0})

    offset, best = optimize_detuning(evaluate, (-0.8 * g, 0.4 * g), tolerance_rad_s=1e-3 * g)
    s = 2.0 * 0.05**2
    assert (-0.5 * g + offset) / g == pytest.approx(-0.5 * math.sqrt(1.0 + s), abs=0.02)
    assert best.mode(0).nbar <= doppler_cooling(st, beams, crystal, weights={0: 1.0}).mode(0).nbar + 1e-9
    # a heating configuration inside the bounds is skipped, not fatal
    offset2, _ = optimize_detuning(evaluate, (-0.8 * g, 0.9 * g), tolerance_rad_s=2e-3 * g)
    assert offset2 < 0.0


def test_two_ion_chain_participation_weights_scale_the_rates_and_leave_nbar_unchanged() -> None:
    """Illuminating one ion of a two-ion chain halves every mode's rate (participation 0.5 against 1, to 1e-6) and
    leaves nbar unchanged, and the COM matches the single ion to 1e-5."""
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp, sp))
    st = structure(sp, b_hat=OBLIQUE)
    beam = sigma_plus_along(
        st, OBLIQUE, 0.05 * g, -0.5 * g, waist_m=2e-3
    )  # wide: both ions see one intensity
    both = doppler_cooling(st, [beam], crystal)
    one = doppler_cooling(st, [beam], crystal, illuminated=[0])
    assert both.illuminated == (0, 1) and one.illuminated == (0,)
    for m_both, m_one in zip(both.modes, one.modes):
        assert m_both.participation_weight == pytest.approx(1.0, rel=1e-9)
        assert m_one.participation_weight == pytest.approx(0.5, rel=1e-9)
        assert m_both.rate_per_s == pytest.approx(2.0 * m_one.rate_per_s, rel=1e-6)
        assert m_both.nbar == pytest.approx(m_one.nbar, rel=1e-6)
    com = crystal.mode_index("axial", 0)
    single = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
    solo = doppler_cooling(st, [beam], single)
    assert both.mode(com).nbar == pytest.approx(solo.mode(0).nbar, rel=1e-5)
    assert both.mode(com).rate_per_s == pytest.approx(solo.mode(0).rate_per_s, rel=1e-5)  # 2.7 um off axis
    models = models_per_ion(st, [beam], crystal, (0,))
    r = ion_mode_rates(models[0], crystal, 0, com)
    assert r is not None and r.participation == pytest.approx(1.0 / math.sqrt(2.0))


# ---- Monroe 1995 9Be+ ---------------------------------------------------------------------------------------------------


def test_monroe_1995_theory_value_is_the_force_model_with_isotropic_emission_at_minus_30_mhz() -> None:
    """Monroe 1995's theoretical 0.484 for the 11.2 MHz mode at Delta = -30 MHz is the alpha = 1/3 force model (3e-3);
    the rate framework gives 0.573 (alpha = 0.4) and 0.533 (alpha = 1/3), and 0.211 and 0.065 on the 18.2 and 29.8 MHz
    modes."""
    g = TWO_PI * 19.4e6
    nu = TWO_PI * 11.2e6
    assert doppler_force_nbar(g, -TWO_PI * 30e6, 0.0, nu, 1.0 / 3.0) == pytest.approx(0.484, abs=0.003)
    weak = stenholm_coefficients(0.01 * g, g, nu, -TWO_PI * 30e6, 0.4).nbar
    assert weak == pytest.approx(0.573, abs=0.002)
    assert stenholm_coefficients(0.01 * g, g, nu, -TWO_PI * 30e6, 1.0 / 3.0).nbar == pytest.approx(
        0.533, abs=0.002
    )
    # the other two modes, each under a beam along it (measured 0.30 and 0.18)
    assert stenholm_coefficients(0.01 * g, g, TWO_PI * 18.2e6, -TWO_PI * 30e6, 0.4).nbar == pytest.approx(
        0.211, abs=0.002
    )
    assert stenholm_coefficients(0.01 * g, g, TWO_PI * 29.8e6, -TWO_PI * 30e6, 0.4).nbar == pytest.approx(
        0.065, abs=0.002
    )


def test_monroe_three_modes_from_one_beam_are_not_reproduced_by_a_single_oblique_beam() -> None:
    """One oblique sigma+ beam at Delta = -30 MHz and s = 0.5 cools Monroe's 11.2, 18.2 and 29.8 MHz modes to 1.02, 0.41
    and 0.13 (3e-2, 2e-2, 1e-2), in the measured order but not to the measured (0.47, 0.30, 0.18)."""
    mass = (9.0121831 - ELECTRON_MASS_U) * ATOMIC_MASS_KG
    be = two_level_atom(gamma_hz=19.4e6, wavelength_m=313e-9, mass_kg=mass)
    g = TWO_PI * 19.4e6
    crystal = solve_crystal(secular_trap((18.2e6, 29.8e6, 11.2e6)), (be,))
    st = structure(be, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, math.sqrt(0.25) * g, -TWO_PI * 30e6)  # s = 2 Omega^2/Gamma^2 = 0.5
    res = doppler_cooling(st, [beam], crystal)
    by_freq = {round(crystal.modes[m.mode].omega_hz / 1e6, 1): m.nbar for m in res.modes}
    assert by_freq[11.2] == pytest.approx(1.02, abs=0.03)
    assert by_freq[18.2] == pytest.approx(0.41, abs=0.02)
    assert by_freq[29.8] == pytest.approx(0.13, abs=0.01)
    assert by_freq[11.2] > by_freq[18.2] > by_freq[29.8]


# ---- 40Ca+ multi-level Doppler cooling (Roos 2000) ----------------------------------------------------------------------


def test_ca40_multilevel_doppler_limit_against_the_two_level_estimate() -> None:
    """40Ca+ S-P-D under the 397 nm beam at -20 MHz and the 866 nm repump cools Roos's 3.3 MHz axial and 1.6 MHz radial
    modes to 4.02 and 9.41 (0.1, 0.2), within 6 % of the renormalized two-level 3.90 and 9.08 and below the measured
    6.5(1.0) and 16(2)."""
    ca = species("40Ca+")
    st = AtomicStructure(ca, 4.0, (0.0, 0.0, 1.0))
    l397, l866 = ca.transition("S1/2-P1/2"), ca.transition("D3/2-P1/2")
    crystal = solve_crystal(secular_trap((1.7e6, 1.6e6, 3.3e6)), (ca,))
    waist = 20e-6
    pol397 = (1.0 / math.sqrt(2.0) + 0j, -1.0 / math.sqrt(2.0) + 0j, 0j)
    k866 = (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0)
    pol866 = (1.0 / math.sqrt(2.0) + 0j, 1.0 / math.sqrt(2.0) + 0j, 0j)
    p397 = 1.0 * l397.i_sat_w_m2 * math.pi * waist**2 / 2.0
    p866 = 3.0 * l866.i_sat_w_m2 * math.pi * waist**2 / 2.0
    b397 = beam_for_transition(
        st, "S1/2 mJ=-1/2", "P1/2 mJ=-1/2", -TWO_PI * 20e6, OBLIQUE, pol397, power_w=p397, waist_m=waist
    )
    b866 = beam_for_transition(
        st, "D3/2 mJ=-1/2", "P1/2 mJ=-1/2", 0.0, k866, pol866, power_w=p866, waist_m=waist
    )
    full = doppler_cooling(st, [b397, b866], crystal, levels=("S1/2", "P1/2", "D3/2"), cooling_beams=[0])
    two = doppler_cooling(
        st, [b397], crystal, levels=("S1/2", "P1/2"), options=MultiLevelOptions(leak="renormalize")
    )
    z = crystal.mode_index("axial", 0)
    y = crystal.mode_index("transverse_2", 0)
    assert full.mode(z).nbar == pytest.approx(4.02, abs=0.1)
    assert full.mode(y).nbar == pytest.approx(9.41, abs=0.2)
    assert two.mode(z).nbar == pytest.approx(3.90, abs=0.1)
    assert two.mode(y).nbar == pytest.approx(9.08, abs=0.2)
    assert full.mode(z).nbar == pytest.approx(two.mode(z).nbar, rel=0.06)
    assert full.mode(y).nbar == pytest.approx(two.mode(y).nbar, rel=0.06)
    assert full.mode(z).nbar < 6.5 - 1.0 and full.mode(y).nbar < 16.0 - 2.0


# ---- validity conditions ----------------------------------------------------------------------------------------------


def test_the_lamb_dicke_expansion_parameter_is_asserted_on_every_level_a_mode() -> None:
    """The Lamb-Dicke guard eta sqrt(2 nbar + 1) < 1/2 passes 0.09023 sqrt(21) and a nu = 0.15 Gamma mode, and refuses a
    nu = 0.01 Gamma mode at the Doppler limit unless strong coupling is allowed."""
    assert_lamb_dicke(0.09023, 10.0)
    with pytest.raises(ValidityError, match=r"eta\^2"):
        assert_lamb_dicke(0.3, 20.0)
    assert_lamb_dicke(0.3, 20.0, allow_strong_coupling=True)
    with pytest.raises(ValidityError, match="non-negative"):
        assert_lamb_dicke(0.1, -1.0)
    g = gamma_rad_s()
    sp = two_level_atom()
    st = structure(sp, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, 0.05 * g, -0.5 * g)
    inside = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.15 * g / TWO_PI)), (sp,))
    res = doppler_cooling(st, [beam], inside, modes=[inside.mode_index("axial", 0)])
    assert res.lamb_dicke_guard()[inside.mode_index("axial", 0)] ** 2 < LAMB_DICKE_MAX
    outside = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.01 * g / TWO_PI)), (sp,))
    axial = outside.mode_index("axial", 0)
    with pytest.raises(ValidityError, match=r"eta\^2"):
        doppler_cooling(st, [beam], outside, modes=[axial])
    loose = doppler_cooling(st, [beam], outside, modes=[axial], allow_strong_coupling=True)
    assert loose.lamb_dicke_guard()[axial] ** 2 > LAMB_DICKE_MAX


def test_the_adiabatic_elimination_conditions_are_asserted_on_the_rate_and_the_internal_rates() -> None:
    """assert_adiabatic enforces W << nu and W << every internal rate, and a real level-A stage has W below SMALL times
    both."""
    assert_adiabatic(1.0, 1e3, {"P": 1e4})
    with pytest.raises(ValidityError, match="W/nu"):
        assert_adiabatic(200.0, 1e3, {"P": 1e9})
    with pytest.raises(ValidityError, match="W/Gamma_min"):
        assert_adiabatic(1.0, 1e6, {"P": 5.0})
    assert_adiabatic(1.0, 1e3, {})
    with pytest.raises(ValidityError, match="mode frequency is positive"):
        assert_adiabatic(1.0, 0.0, {})
    # on a real level-A stage the Doppler W is 1e-5 of nu
    g = gamma_rad_s()
    sp = two_level_atom()
    st = structure(sp, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, 0.05 * g, -0.5 * g)
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, 0.15 * g / TWO_PI)), (sp,))
    models = models_per_ion(st, [beam], crystal, (0,))
    rates = mode_rates(models, crystal, 0)
    assert rates.rate_per_s < SMALL * rates.omega_rad_s
    assert rates.rate_per_s < SMALL * min(models[0].build.level_rates_rad_s.values())
    with pytest.raises(ValidityError, match="W/nu"):
        assert_adiabatic(rates.omega_rad_s, rates.omega_rad_s, models[0].build.level_rates_rad_s)


def test_doppler_cooling_refuses_a_recoil_limited_line() -> None:
    """171Yb+ at 369.5 nm has omega_R/2pi = 8.5 kHz (2 %), so a 1 kHz line on the same mass and wavelength is refused as
    recoil-limited and, with the escapes, cools nothing (nbar above 1e6)."""
    assert recoil_frequency_rad_s(TWO_PI / WAVELENGTH_M, MASS_KG) / TWO_PI == pytest.approx(8.5e3, rel=0.02)
    assert_doppler_recoil_limit(gamma_rad_s(), TWO_PI / WAVELENGTH_M, MASS_KG)
    with pytest.raises(ValidityError, match="Gamma/omega_R"):
        assert_doppler_recoil_limit(TWO_PI * 1e3, TWO_PI / WAVELENGTH_M, MASS_KG)
    narrow = two_level_atom(gamma_hz=1e3)
    st = structure(narrow, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, 0.05 * TWO_PI * 1e3, -0.5 * TWO_PI * 1e3)
    crystal = solve_crystal(secular_trap((2.5e6, 2.6e6, 1.0e6)), (narrow,))
    with pytest.raises(ValidityError, match="Gamma/omega_R"):
        doppler_cooling(st, [beam], crystal, modes=[crystal.mode_index("axial", 0)])
    # with the escapes the stage evaluates, and Gamma << nu at Delta = -Gamma/2 leaves no cooling at all
    ok = doppler_cooling(
        st,
        [beam],
        crystal,
        modes=[crystal.mode_index("axial", 0)],
        allow_recoil_limited=True,
        allow_strong_coupling=True,
    )
    assert ok.modes[0].nbar > 1e6
