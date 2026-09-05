"""The Doppler stage of PLAN.md Section 4.2.1 at level A (M3): per-mode nbar_D from the rate framework with the configured angular
factor, one detuning per beam over the whole mode set, the uncooled-mode guard, the multi-ion participation weights and the
force-model cross-check; Section 9.3 rows "Doppler anchors" and "Doppler limit with recoil", Section 9.17 "Doppler detuning per beam"."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import Beam, Trap
from qutip_trap.dynamics.multilevel import MultiLevelOptions
from qutip_trap.light.recoil import angular_factor
from qutip_trap.prep.closed_forms import doppler_force_nbar, doppler_limit_nbar, stenholm_coefficients
from qutip_trap.prep.doppler import doppler_cooling, optimize_detuning, with_detuning_offset
from qutip_trap.prep.rates import UncooledModeError, ion_mode_rates, models_per_ion
from qutip_trap.species import species
from qutip_trap.species.polarization import spherical_basis
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, ELECTRON_MASS_U, TWO_PI
from tests.bloch_fixtures import (
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    gamma_rad_s,
    power_for_rabi,
    sigma_plus_beam,
    structure,
    two_level_atom,
)

OBLIQUE = (1.0 / math.sqrt(3.0),) * 3


def trap_for(freqs_hz: tuple[float, float, float]) -> Trap:
    return Trap(
        omega_hz=freqs_hz,
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
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
    """One sigma+ beam along B = (1,1,1)/sqrt3 sees all three modes at cos^2 = 1/3 with alpha(chi) = 1/3: nbar_D = (Gamma/4 nu)(1 + alpha/cos^2) - 1/2
    to 0.5 % at nu = Gamma/20 (the force model, valid there, agrees to 1e-6), and the closed-form method agrees with the spectrum path
    for the weak drive."""
    sp = two_level_atom()
    g = gamma_rad_s()
    nu = 0.05 * g
    crystal = solve_crystal(trap_for((2.5e6, 2.6e6, nu / TWO_PI)), (sp,))
    st = structure(sp, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, 0.05 * g, -0.5 * g)
    res = doppler_cooling(st, [beam], crystal)
    assert res.illuminated == (0,) and res.method == "spectrum"
    for m in res.modes:
        alpha = angular_factor(1, float(np.dot(crystal.modes[m.mode].e_hat, OBLIQUE)))
        assert m.projection_cos2_max == pytest.approx(1.0 / 3.0)
        closed = doppler_limit_nbar(g, m.omega_rad_s, alpha / m.projection_cos2_max) - 0.5
        assert m.nbar == pytest.approx(closed, rel=0.03)
        assert m.participation_weight == pytest.approx(1.0)
    axial = res.mode(0)
    assert axial.nbar == pytest.approx(9.5, rel=6e-3)
    assert res.force_model_nbar[0] == pytest.approx(9.5, abs=1e-5)
    assert 1 not in res.force_model_nbar  # nu/Gamma = 0.128 > 0.1: the force model is not reported there
    closed_form = doppler_cooling(st, [beam], crystal, method="closed_form")
    for a, b in zip(res.modes, closed_form.modes):
        assert b.nbar == pytest.approx(a.nbar, rel=5e-3)
    assert any("closed form" in a for a in closed_form.approximations)
    assert set(res.lamb_dicke_guard()) == {0, 1, 2}
    assert res.weighted_nbar == pytest.approx(float(np.mean([m.nbar for m in res.modes])))


def test_a_mode_no_cooling_beam_addresses_raises_instead_of_returning_a_steady_state() -> None:
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(trap_for((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
    st = structure(sp)
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.05 * g, -0.5 * g)  # along z = B
    with pytest.raises(UncooledModeError):
        doppler_cooling(st, [beam], crystal)
    # the z mode alone is fine
    res = doppler_cooling(st, [beam], crystal, modes=[0])
    assert [m.mode for m in res.modes] == [0] and res.mode(0).nbar == pytest.approx(6.54, abs=0.05)


def test_optimum_detuning_of_a_low_frequency_mode_is_minus_half_the_linewidth() -> None:
    """Section 4.2.1: the beam detuning is chosen once for the mode set; with all the weight on the nu = Gamma/20 mode the optimum sits at
    -(Gamma/2) sqrt(1 + s) (RMP Eq. 106) within the search tolerance."""
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(trap_for((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
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
    """W_k = sum_{i illuminated} c_{i,k}^2 (...) (Section 4.2): illuminating one ion of a two-ion chain halves every COM rate against
    illuminating both, while nbar is identical (the participation cancels between eta~ and eta, Section 4.2.2)."""
    sp = two_level_atom()
    g = gamma_rad_s()
    crystal = solve_crystal(trap_for((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp, sp))
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
    single = solve_crystal(trap_for((2.5e6, 2.6e6, 0.05 * g / TWO_PI)), (sp,))
    solo = doppler_cooling(st, [beam], single)
    assert both.mode(com).nbar == pytest.approx(solo.mode(0).nbar, rel=1e-5)
    assert both.mode(com).rate_per_s == pytest.approx(solo.mode(0).rate_per_s, rel=1e-5)  # 2.7 um off axis
    models = models_per_ion(st, [beam], crystal, (0,))
    r = ion_mode_rates(models[0], crystal, 0, com)
    assert r is not None and r.participation == pytest.approx(1.0 / math.sqrt(2.0))


# ---- Monroe 1995 9Be+ (Section 9.3 "Doppler anchors") --------------------------------------------------------------------------


def test_monroe_1995_theory_value_is_the_force_model_with_isotropic_emission_at_minus_30_mhz() -> None:
    """Monroe's 'theoretical 0.484' for the 11.2 MHz mode at Gamma/2pi = 19.4 MHz and Delta = -30 MHz is the semiclassical force model with alpha = 1/3
    and cos^2 = 1 minus the zero point (0.486, M3 finding), although nu/Gamma = 0.58 is outside that model's regime; the rate framework
    valid there gives 0.573 (sigma pattern along B) or 0.533 (alpha = 1/3) for a weak beam along the mode, against the measured 0.47(5)."""
    g = TWO_PI * 19.4e6
    nu = TWO_PI * 11.2e6
    assert doppler_force_nbar(g, -TWO_PI * 30e6, 0.0, nu, 1.0 / 3.0) == pytest.approx(0.484, abs=0.003)
    weak = stenholm_coefficients(0.01 * g, g, nu, -TWO_PI * 30e6, 0.4).nbar
    assert weak == pytest.approx(0.573, abs=0.002)
    assert stenholm_coefficients(0.01 * g, g, nu, -TWO_PI * 30e6, 1.0 / 3.0).nbar == pytest.approx(
        0.533, abs=0.002
    )
    # the other two modes for a beam along each: 0.211 and 0.065 against the measured 0.30 and 0.18
    assert stenholm_coefficients(0.01 * g, g, TWO_PI * 18.2e6, -TWO_PI * 30e6, 0.4).nbar == pytest.approx(
        0.211, abs=0.002
    )
    assert stenholm_coefficients(0.01 * g, g, TWO_PI * 29.8e6, -TWO_PI * 30e6, 0.4).nbar == pytest.approx(
        0.065, abs=0.002
    )


def test_monroe_three_modes_from_one_beam_are_not_reproduced_by_a_single_oblique_beam() -> None:
    """The Section 4.2.1 acceptance test as designed: one sigma+ beam at Delta = -30 MHz along (1,1,1)/sqrt3 (equal projections) on modes at
    11.2, 18.2, 29.8 MHz. The rate framework at s = 0.5 gives about (1.02, 0.41, 0.13) against the measured (0.47, 0.30, 0.18): the
    ordering is right, the x mode is 2x too hot and the z mode 30 % too cold, so the measured triple needs the actual D1-D3 beam geometry
    (recorded as a plan finding; the numbers here are the regression)."""
    mass = (9.0121831 - ELECTRON_MASS_U) * ATOMIC_MASS_KG
    be = two_level_atom(gamma_hz=19.4e6, wavelength_m=313e-9, mass_kg=mass)
    g = TWO_PI * 19.4e6
    crystal = solve_crystal(trap_for((18.2e6, 29.8e6, 11.2e6)), (be,))
    st = structure(be, b_hat=OBLIQUE)
    beam = sigma_plus_along(st, OBLIQUE, math.sqrt(0.25) * g, -TWO_PI * 30e6)  # s = 2 Omega^2/Gamma^2 = 0.5
    res = doppler_cooling(st, [beam], crystal)
    by_freq = {round(crystal.modes[m.mode].omega_hz / 1e6, 1): m.nbar for m in res.modes}
    assert by_freq[11.2] == pytest.approx(1.02, abs=0.03)
    assert by_freq[18.2] == pytest.approx(0.41, abs=0.02)
    assert by_freq[29.8] == pytest.approx(0.13, abs=0.01)
    assert by_freq[11.2] > by_freq[18.2] > by_freq[29.8]


# ---- 40Ca+ multi-level Doppler cooling (Section 9.3 "Doppler anchors", Roos 2000) ----------------------------------------------


def test_ca40_multilevel_doppler_limit_against_the_two_level_estimate() -> None:
    """S1/2-P1/2-D3/2 with the 397 nm beam at -20 MHz along (1,1,1)/sqrt3 and the 866 nm repumper, B = 4 G along z, on Roos's 3.3 MHz axial and
    1.6 MHz radial modes: the eight-state model gives nbar_z = 3.7 and nbar_y = 8.7 at s_397 = 1, s_866 = 3, within 5 % of the S-P two-level
    model with the D branch renormalized (3.65, 8.5); the measured 6.5(1.0) and 16(2) are not a level-structure effect at these parameters
    (M3 finding: the plan's 'two-level estimate n_z ~ 3, n_y ~ 6 fails because of the multi-level structure' is not reproduced)."""
    ca = species("40Ca+")
    st = AtomicStructure(ca, 4.0, (0.0, 0.0, 1.0))
    l397, l866 = ca.transition("S1/2-P1/2"), ca.transition("D3/2-P1/2")
    crystal = solve_crystal(trap_for((1.7e6, 1.6e6, 3.3e6)), (ca,))
    waist = 20e-6
    from qutip_trap.light.bloch import beam_for_transition

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
    assert full.mode(z).nbar == pytest.approx(3.74, abs=0.1)
    assert full.mode(y).nbar == pytest.approx(8.74, abs=0.2)
    assert full.mode(z).nbar == pytest.approx(two.mode(z).nbar, rel=0.06)
    assert full.mode(y).nbar == pytest.approx(two.mode(y).nbar, rel=0.06)
    assert full.mode(z).nbar < 6.5 - 1.0 and full.mode(y).nbar < 16.0 - 2.0
