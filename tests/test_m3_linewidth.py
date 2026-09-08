"""The laser linewidth on the optical coherences (PLAN.md Section 8.1; milestone M3a).

Section 8.1 states of the coherent-population-trapping Liouvillian that "-gamma on excited-excited elements and
-(gamma/2 + delta omega_L/2) on optical coherences must be added or the map is not trace-preserving". The Lindblad
form supplies -gamma and -gamma/2 by itself; the laser's own contribution was missing entirely until
``MultiLevelOptions.laser_linewidth_rad_s`` (the M3a finding), and Section 12's grant covers only "the laser
linewidth AS A SPECTRUM", which is weaker than the constant Lorentzian rate Section 8.1 asks for.

The construction is one phase-diffusion collapse operator per beam, C_b = sqrt(delta omega_L/4)(P_upper - P_lower)
on the manifolds that beam connects (Berkeland and Boshier, Phys. Rev. A 65, 033413 (2002) Eq. 12: a Lorentzian
laser spectrum of full width delta omega_L is a Wiener phase whose Lindblad generator damps the driven transition's
coherence at delta omega_L/2 and touches no population). The observable consequence is that the excitation profile
convolves to the full width Gamma + delta omega_L at constant area.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from scipy.optimize import brentq

from qutip_trap.dynamics.multilevel import MultiLevelOptions, decay_sum_rule_residual
from qutip_trap.light.bloch import BlochModel
from tests.bloch_fixtures import (
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    gamma_rad_s,
    sigma_plus_beam,
    structure,
    two_level_atom,
)

SP = two_level_atom()
ST = structure(SP)
G = gamma_rad_s()


def _beam(detuning_rad_s: float, omega_rad_s: float = 0.01 * G):  # type: ignore[no-untyped-def]
    return sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, omega_rad_s, detuning_rad_s)


def _profile_fwhm_over_gamma(width_rad_s: float) -> tuple[float, float]:
    """(peak W, full width at half maximum in units of Gamma) of the weak-drive excitation profile.

    The profile is symmetric and single peaked, so the half width is found by root-finding rather than on a grid
    (the M3 lesson of the Doppler argmin: a grid resolves a width only to its own spacing).
    """
    options = MultiLevelOptions(laser_linewidth_rad_s=(width_rad_s,))

    def rate(detuning: float) -> float:
        return BlochModel(ST, [_beam(detuning)], options=options).scattering_rate_per_s()

    peak = rate(0.0)
    half = brentq(lambda d: rate(d) - 0.5 * peak, 0.0, 20.0 * G, xtol=1e-4 * G)
    return peak, 2.0 * half / G


def test_the_excitation_profile_broadens_to_gamma_plus_the_laser_linewidth() -> None:
    """W(Delta) is a Lorentzian of full width Gamma with an ideal laser and of full width Gamma + delta omega_L with a
    finite one, at constant area: the peak falls as Gamma/(Gamma + delta omega_L) (Section 8.1)."""
    peak0, fwhm0 = _profile_fwhm_over_gamma(0.0)
    assert fwhm0 == pytest.approx(1.0, abs=0.02)
    for factor in (0.5, 1.0, 2.0):
        peak, fwhm = _profile_fwhm_over_gamma(factor * G)
        assert fwhm == pytest.approx(1.0 + factor, abs=0.02)
        assert peak / peak0 == pytest.approx(1.0 / (1.0 + factor), rel=2e-3)
    # the profile is still a Lorentzian, now of half width (Gamma + delta omega_L)/2
    width = 0.7 * G
    options = MultiLevelOptions(laser_linewidth_rad_s=(width,))
    on_resonance = BlochModel(ST, [_beam(0.0)], options=options).scattering_rate_per_s()
    for offset in (0.3, 0.9, 2.0):
        delta = offset * (G + width)
        got = BlochModel(ST, [_beam(delta)], options=options).scattering_rate_per_s()
        assert got / on_resonance == pytest.approx(1.0 / (1.0 + (2.0 * offset) ** 2), rel=3e-3)


def test_a_zero_linewidth_changes_no_rate_and_adds_no_operator() -> None:
    """R_o at delta omega_L = 0 is unchanged to 1e-12 (in fact bit for bit), and no collapse operator is added."""
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.3 * G, -0.5 * G)
    plain = BlochModel(ST, [beam])
    zero = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(0.0,)))
    assert zero.scattering_rate_per_s() == pytest.approx(plain.scattering_rate_per_s(), rel=1e-12)
    assert zero.build.dephasing_slice == (len(zero.build.c_ops), len(zero.build.c_ops))
    assert len(zero.build.c_ops) == len(plain.build.c_ops)
    assert not any("linewidth" in a for a in zero.build.approximations)
    finite = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(0.3 * G,)))
    start, stop = finite.build.dephasing_slice
    assert stop - start == 1 and stop == len(finite.build.c_ops)
    assert any("laser linewidth" in a for a in finite.build.approximations)
    assert finite.scattering_rate_per_s() < plain.scattering_rate_per_s()


def test_the_phase_diffusion_operator_touches_no_population_and_leaves_the_decay_sum_rule_alone() -> None:
    """C_b is diagonal, so it commutes with every projector: the steady-state populations of a RESONANT drive are
    unchanged (the profile only narrows off resonance), the operator carries no photon and no recoil, and the
    Section 4.2.8 decay sum rule sum_k C_k^dagger C_k = Gamma P_e excludes it."""
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.3 * G, 0.0)
    plain = BlochModel(ST, [beam])
    finite = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(0.4 * G,)))
    assert decay_sum_rule_residual(finite.build) < 1e-12
    assert decay_sum_rule_residual(finite.build) == pytest.approx(
        decay_sum_rule_residual(plain.build), abs=1e-12
    )
    start, stop = finite.build.dephasing_slice
    dephasing = finite.build.c_ops[start]
    matrix = np.asarray(dephasing.full())
    assert np.allclose(matrix, np.diag(np.diag(matrix)))  # diagonal: no population is moved
    # +sqrt(dw/4) on the upper manifold, -sqrt(dw/4) on the lower one
    values = np.real(np.diag(matrix))
    assert np.max(values) == pytest.approx(np.sqrt(0.4 * G / 4.0), rel=1e-9)
    assert np.min(values) == pytest.approx(-np.sqrt(0.4 * G / 4.0), rel=1e-9)
    # the photon bookkeeping ignores it: no channel claims that operator index
    for channel in finite.build.channels:
        assert not channel.operator_slice[0] <= start < channel.operator_slice[1]
    assert sum(finite.photon_rates(finite.steadystate().rho).values()) == pytest.approx(
        finite.steadystate().total_photon_rate_per_s, rel=1e-12
    )


def test_the_optical_coherence_decays_at_gamma_over_two_plus_half_the_linewidth() -> None:
    """The direct statement of Section 8.1: with the drive off, rho_eg decays at gamma/2 + delta omega_L/2 (Berkeland
    and Boshier Eq. 12), while the excited population still decays at gamma alone."""
    width = 0.6 * G
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.0, 0.0)  # zero Rabi frequency
    build = BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(width,))).build
    lower = build.index(TWO_LEVEL_GROUND)
    upper = build.index(TWO_LEVEL_EXCITED_PLUS)
    n = build.n_internal
    rho0 = np.zeros((n, n), dtype=complex)
    rho0[upper, upper] = 0.5
    rho0[lower, lower] = 0.5
    rho0[upper, lower] = 0.5
    rho0[lower, upper] = 0.5
    times = np.linspace(0.0, 3.0 / G, 61)
    res = qt.mesolve(
        build.H,
        qt.Qobj(rho0, dims=build.H.dims),
        times,
        c_ops=list(build.c_ops),
        options={"store_states": True, "progress_bar": "", "atol": 1e-13, "rtol": 1e-11},
    )
    coherence = np.array([abs(np.asarray(s.full())[upper, lower]) for s in res.states])
    population = np.array([np.real(np.asarray(s.full())[upper, upper]) for s in res.states])
    expected_coherence = 0.5 * np.exp(-(0.5 * G + 0.5 * width) * times)
    expected_population = 0.5 * np.exp(-G * times)
    assert np.max(np.abs(coherence - expected_coherence)) < 1e-9
    assert np.max(np.abs(population - expected_population)) < 1e-9


def test_one_linewidth_per_beam_is_required_and_non_negative() -> None:
    beam = sigma_plus_beam(ST, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.1 * G, 0.0)
    with pytest.raises(ValueError, match="one entry per beam"):
        BlochModel(ST, [beam], options=MultiLevelOptions(laser_linewidth_rad_s=(1e3, 1e3)))
    with pytest.raises(ValueError, match="non-negative"):
        MultiLevelOptions(laser_linewidth_rad_s=(-1.0,))
