"""The Section 4.2.8 (vii) validity conditions RAISE rather than report (PLAN.md Section 4.2.8; milestone M3).

Section 4.2.8 (vii): "Validity conditions the module asserts at run time rather than documents: eta_ij^2(2n + 1) << 1
per transition including the emitted photon, Omega << Gamma for every closed form, cooling rate W << nu and << every
internal rate for the adiabatic elimination, Gamma', gamma' << nu for resolved sidebands and Gamma > omega_R for
Doppler cooling, s_aux << 1 and |delta| << Gamma_10 + Gamma_12 for the three-level reduction."

One negative control per condition, plus the escape hatch that lets a caller probe the boundary deliberately (a
source's own published fixture outside its own regime, an algebraic normalization, a saturation study) and the
booleans the report functions still return.
"""

from __future__ import annotations

import math

import pytest

from qutip_trap.light.beams import Beam
from qutip_trap.light.bloch import WEAK_DRIVE_MAX, BlochModel, rate_coefficients_from_model
from qutip_trap.prep.closed_forms import (
    effective_two_level,
    effective_two_level_valid,
    lorentzian_scattering_rate,
    sideband_floor,
    stenholm_coefficients,
)
from qutip_trap.prep.doppler import doppler_cooling
from qutip_trap.prep.eit import eit_rate_coefficients
from qutip_trap.prep.rates import ion_mode_rates, mode_rates, models_per_ion
from qutip_trap.prep.validity import (
    ADIABATIC_MAX,
    LAMB_DICKE_MAX,
    RESOLVED_MAX,
    SMALL,
    ValidityError,
    assert_adiabatic,
    assert_doppler_recoil_limit,
    assert_lamb_dicke,
    assert_resolved,
    assert_resolved_linewidth,
    assert_three_level_valid,
    assert_weak_drive,
    recoil_frequency_rad_s,
)
from qutip_trap.species.polarization import spherical_basis
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.model import Trap
from qutip_trap.units import ATOMIC_MASS_KG, C_M_PER_S, TWO_PI
from tests.bloch_fixtures import (
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

OBLIQUE = (1.0 / math.sqrt(3.0),) * 3


def _trap(freqs_hz: tuple[float, float, float]) -> Trap:
    return Trap(
        omega_hz=freqs_hz,
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


def _oblique_beam(st, omega, delta, waist_m=20e-6):  # type: ignore[no-untyped-def]
    _m, _z, e_plus = spherical_basis(OBLIQUE)
    pol = tuple(complex(x) for x in e_plus)
    power = power_for_rabi(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega, waist_m, pol, OBLIQUE)  # type: ignore[arg-type]
    omega_l = (
        TWO_PI * (st.state(TWO_LEVEL_EXCITED_PLUS).energy_hz - st.state(TWO_LEVEL_GROUND).energy_hz) + delta
    )
    return Beam(TWO_PI * C_M_PER_S / omega_l, OBLIQUE, pol, waist_m, power, (0.0, 0.0, 0.0))  # type: ignore[arg-type]


# ---- Omega << Gamma on EVERY closed form -------------------------------------------------------------------------------------


def test_every_closed_form_refuses_a_saturated_drive_unless_the_caller_says_so() -> None:
    """Before this guard only ``rate_coefficients_from_model`` and ``ion_mode_rates(method='closed_form')`` checked
    Omega << Gamma; Stenholm's A_+-, the unsaturated Lorentzian, the sideband floor and the EIT coefficients accepted
    any Omega silently (the M3 finding). The threshold is one tenth, the same number as ``bloch.WEAK_DRIVE_MAX``."""
    assert SMALL == WEAK_DRIVE_MAX == 0.1
    # exactly at the boundary passes; above it raises
    assert lorentzian_scattering_rate(0.1, 1.0, 0.0) == pytest.approx(0.01)
    for bad in (0.11, 0.5, 1.0):
        with pytest.raises(ValidityError, match="Omega/Gamma"):
            lorentzian_scattering_rate(bad, 1.0, 0.0)
        with pytest.raises(ValidityError, match="Omega/Gamma"):
            stenholm_coefficients(bad, 1.0, 50.0, -50.0, 0.4)
        with pytest.raises(ValidityError, match="Omega/Gamma"):
            eit_rate_coefficients(bad, 24.0, 2.0, 70.0, 1.0)
    # and the escape hatch evaluates them anyway, with the same numbers a saturated caller would have got silently
    assert lorentzian_scattering_rate(0.5, 1.0, 0.0, allow_saturation=True) == pytest.approx(0.25)
    assert stenholm_coefficients(0.5, 1.0, 50.0, -50.0, 0.4, allow_saturation=True).nbar > 0.0
    assert eit_rate_coefficients(17.0, 24.0416, 2.0068, 70.0, 20.0, allow_saturation=True).A_minus_per_s > 0.0
    with pytest.raises(ValidityError):
        assert_weak_drive(1.0, 1.0)
    assert_weak_drive(1.0, 1.0, allow_saturation=True)
    with pytest.raises(ValidityError, match="linewidth is positive"):
        assert_weak_drive(0.1, 0.0)


def test_the_sideband_floor_refuses_an_unresolved_linewidth() -> None:
    """Gamma << nu is the floor's own regime (RMP Eq. 116); the 7/12 and 13/20 normalizations sit at Gamma/nu = 2 and
    must pass the escape explicitly."""
    assert sideband_floor(0.01, 1.0, 0.4) > 0.0
    with pytest.raises(ValidityError, match="Gamma/nu"):
        sideband_floor(2.0, 1.0, 1.0 / 3.0)
    assert sideband_floor(2.0, 1.0, 1.0 / 3.0, allow_unresolved=True) == pytest.approx(7.0 / 12.0)
    with pytest.raises(ValidityError, match="mode frequency is positive"):
        assert_resolved_linewidth(1.0, 0.0)


# ---- eta^2 (2 nbar + 1) << 1 -------------------------------------------------------------------------------------------------


def test_the_lamb_dicke_expansion_parameter_is_asserted_on_every_level_a_mode() -> None:
    """Section 4.2.8 (vii): the bound is eta sqrt(2 nbar + 1) < 1/2, i.e. eta^2 (2 nbar + 1) < 1/4, which the plan's own
    anchors satisfy (0.09023 sqrt(21) = 0.413 for the 171Yb+ polarization-gradient triple, 0.242 and 0.35 for the
    nu = Gamma/20 Doppler fixtures). ``StageRates.lamb_dicke_guard`` still REPORTS the same quantity."""
    assert LAMB_DICKE_MAX == 0.25
    assert 0.09023**2 * 21.0 == pytest.approx(0.17097, abs=1e-5)
    assert_lamb_dicke(0.09023, 10.0)  # the plan's 0.413 guard value: inside
    with pytest.raises(ValidityError, match=r"eta\^2"):
        assert_lamb_dicke(0.3, 20.0)
    assert_lamb_dicke(0.3, 20.0, allow_strong_coupling=True)
    with pytest.raises(ValidityError, match="non-negative"):
        assert_lamb_dicke(0.1, -1.0)
    # on a real level-A stage: a low-frequency mode at the Doppler limit is inside, a much lower one is refused
    g = gamma_rad_s()
    sp = two_level_atom()
    st = structure(sp, b_hat=OBLIQUE)
    beam = _oblique_beam(st, 0.05 * g, -0.5 * g)
    inside = solve_crystal(_trap((2.5e6, 2.6e6, 0.15 * g / TWO_PI)), (sp,))
    res = doppler_cooling(st, [beam], inside, modes=[inside.mode_index("axial", 0)])
    guard = res.lamb_dicke_guard()[inside.mode_index("axial", 0)]
    assert guard**2 < LAMB_DICKE_MAX
    outside = solve_crystal(_trap((2.5e6, 2.6e6, 0.01 * g / TWO_PI)), (sp,))
    axial = outside.mode_index("axial", 0)
    with pytest.raises(ValidityError, match=r"eta\^2"):
        doppler_cooling(st, [beam], outside, modes=[axial])
    loose = doppler_cooling(st, [beam], outside, modes=[axial], allow_strong_coupling=True)
    assert loose.lamb_dicke_guard()[axial] ** 2 > LAMB_DICKE_MAX


# ---- W << nu and W << every internal rate ------------------------------------------------------------------------------------


def test_the_adiabatic_elimination_conditions_are_asserted_on_the_rate_and_the_internal_rates() -> None:
    """W << nu (the motion is eliminated) and W << every internal rate (the internal state is), Section 4.2.8 (vii).
    Before this guard ``level_c.level_c_relaxation_rate`` only documented them in a docstring (the M3 finding)."""
    assert ADIABATIC_MAX == 0.1
    assert_adiabatic(1.0, 1e3, [1e4])
    with pytest.raises(ValidityError, match="W/nu"):
        assert_adiabatic(200.0, 1e3, [1e9])
    with pytest.raises(ValidityError, match="W/Gamma_min"):
        assert_adiabatic(1.0, 1e6, [5.0])
    assert_adiabatic(200.0, 1e3, [5.0], allow_fast_cooling=True)
    assert_adiabatic(1.0, 1e3, {})  # no internal rates: W << nu alone
    assert_adiabatic(1.0, 1e3, {"P": 1e5})  # a mapping, as the build's level_rates_rad_s is
    with pytest.raises(ValidityError, match="mode frequency is positive"):
        assert_adiabatic(1.0, 0.0, [])
    # on a real level-A stage: the Doppler stage's W is 1e-5 of nu, so it passes, and the guard is reachable
    g = gamma_rad_s()
    sp = two_level_atom()
    st = structure(sp, b_hat=OBLIQUE)
    beam = _oblique_beam(st, 0.05 * g, -0.5 * g)
    crystal = solve_crystal(_trap((2.5e6, 2.6e6, 0.15 * g / TWO_PI)), (sp,))
    models = models_per_ion(st, [beam], crystal, (0,))
    rates = mode_rates(models, crystal, 0)
    assert rates.rate_per_s < ADIABATIC_MAX * rates.omega_rad_s
    assert rates.rate_per_s < ADIABATIC_MAX * min(models[0].build.level_rates_rad_s.values())
    with pytest.raises(ValidityError, match="W/nu"):
        assert_adiabatic(
            rates.omega_rad_s, rates.omega_rad_s, models[0].build.level_rates_rad_s
        )  # a W as fast as the mode


# ---- Gamma', gamma' << nu and the three-level reduction ----------------------------------------------------------------------


def test_the_resolved_sideband_and_three_level_reduction_conditions_raise() -> None:
    """``EffectiveTwoLevel.resolved`` and ``effective_two_level_valid`` kept their booleans and gained raising forms
    (Section 4.2.8 v, vii). Marzoli's own Fig. 3 fixture (delta_aux = -Gamma_10) violates the reduction's
    |delta_aux| << Gamma_10 + Gamma_12, so ``effective_two_level`` refuses it without ``allow_invalid``."""
    assert RESOLVED_MAX == 0.2
    xi = effective_two_level(1.0, 0.1, 0.2, -1.0, "Xi", allow_invalid=True)
    assert xi.resolved(0.05) and not xi.resolved(0.005)
    xi.assert_resolved(0.05)
    with pytest.raises(ValidityError, match=r"max\(Gamma', gamma'\)/nu"):
        xi.assert_resolved(0.005)
    xi.assert_resolved(0.005, allow_unresolved=True)
    with pytest.raises(ValidityError, match="mode frequency is positive"):
        assert_resolved(1.0, 1.0, 0.0)
    # the reduction's own conditions
    assert effective_two_level_valid(1.0, 0.1, 0.2, -0.05)
    assert not effective_two_level_valid(1.0, 0.1, 2.0, -0.05)
    effective_two_level(1.0, 0.1, 0.2, -0.05, "Xi")  # inside: no escape needed
    with pytest.raises(ValidityError, match="s_aux"):
        effective_two_level(1.0, 0.1, 0.2, -1.0, "Xi")
    with pytest.raises(ValidityError, match="s_aux"):
        effective_two_level(1.0, 0.1, 2.0, -0.05, "V")
    assert_three_level_valid(1.0, 0.1, 2.0, -0.05, allow_invalid=True)
    with pytest.raises(ValidityError, match="decay rates are positive"):
        assert_three_level_valid(0.0, 0.0, 0.1, 0.0)


# ---- Gamma > omega_R for Doppler cooling -------------------------------------------------------------------------------------


def test_doppler_cooling_refuses_a_recoil_limited_line() -> None:
    """Section 4.2.8 (vii): below Gamma = omega_R = hbar k^2/2m the single-photon recoil exceeds the linewidth and the
    Doppler limit is replaced by the recoil limit. 171Yb+ at 369.5 nm has omega_R/2 pi = 8.5 kHz against
    Gamma/2 pi = 19.6 MHz (a ratio of 2300), so a 1 kHz line on the same mass and wavelength is the negative control."""
    assert recoil_frequency_rad_s(TWO_PI / WAVELENGTH_M, MASS_KG) / TWO_PI == pytest.approx(8.5e3, rel=0.02)
    assert_doppler_recoil_limit(gamma_rad_s(), TWO_PI / WAVELENGTH_M, MASS_KG)
    with pytest.raises(ValidityError, match="Gamma/omega_R"):
        assert_doppler_recoil_limit(TWO_PI * 1e3, TWO_PI / WAVELENGTH_M, MASS_KG)
    assert_doppler_recoil_limit(TWO_PI * 1e3, TWO_PI / WAVELENGTH_M, MASS_KG, allow_recoil_limited=True)
    # and through the stage: a 1 kHz line on the fixture's mass and wavelength
    narrow = two_level_atom(gamma_hz=1e3)
    st = structure(narrow, b_hat=OBLIQUE)
    beam = _oblique_beam(st, 0.05 * TWO_PI * 1e3, -0.5 * TWO_PI * 1e3)
    crystal = solve_crystal(_trap((2.5e6, 2.6e6, 1.0e6)), (narrow,))
    with pytest.raises(ValidityError, match="Gamma/omega_R"):
        doppler_cooling(st, [beam], crystal, modes=[crystal.mode_index("axial", 0)])
    # with the escape the stage evaluates, and its own nbar shows why the guard exists: Gamma << nu at Delta = -Gamma/2
    # leaves no cooling at all, so the Lamb-Dicke escape is needed too
    ok = doppler_cooling(
        st,
        [beam],
        crystal,
        modes=[crystal.mode_index("axial", 0)],
        allow_recoil_limited=True,
        allow_strong_coupling=True,
    )
    assert ok.modes[0].nbar > 1e6


# ---- the guards that existed keep working ------------------------------------------------------------------------------------


def test_the_pre_existing_weak_drive_guards_still_raise_and_still_escape() -> None:
    g = gamma_rad_s()
    sp = two_level_atom()
    st = structure(sp)
    strong = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.5 * g, -0.5 * g)
    model = BlochModel(st, [strong])
    with pytest.raises(ValueError, match="Omega/Gamma"):
        rate_coefficients_from_model(model, 0, 0.05 * g, 0.4)
    assert rate_coefficients_from_model(model, 0, 0.05 * g, 0.4, allow_saturation=True).nbar > 0.0
    crystal = solve_crystal(_trap((2.5e6, 2.6e6, 0.15 * g / TWO_PI)), (sp,))
    with pytest.raises(ValueError, match="Omega/Gamma"):
        ion_mode_rates(model, crystal, 0, 0, method="closed_form")
    assert ion_mode_rates(model, crystal, 0, 0, method="closed_form", allow_saturation=True) is not None
    assert ATOMIC_MASS_KG > 0.0  # (imported for the mass bookkeeping above)
