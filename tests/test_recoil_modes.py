"""The per-ion participation and the joint multi-mode recoil kick of PLAN.md Section 4.2.8 (milestone M3; Section 9.3 rows
"Recoil quadrature identities", "Single kick and motion independence", "Recoil unit regression"; Section 9.17 row
"Sideband-cooling floor independent of N")."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.api import HilbertSpace, ModeTruncation, RfDrive, Trap
from qutip_trap.light.recoil import (
    angular_factor,
    direction_quadrature,
    emission_lamb_dicke,
    free_recoil_energy_j,
    marginal_quadrature,
    minimal_quadrature,
    multi_mode_kick,
    pattern_density,
    recoil_energy_ratio,
    recoil_heating_quanta,
    recoil_kernel_matrix,
    recoil_projections,
    recoil_temperature_k,
    recoil_velocity_m_per_s,
)
from qutip_trap.species import species
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI

K_369 = TWO_PI / 369.5e-9
Z = (0.0, 0.0, 1.0)


def trap(rf: RfDrive | None = None) -> Trap:
    return Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=rf,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


@pytest.mark.parametrize("family", ["axial", "transverse_1", "transverse_2"])
@pytest.mark.parametrize("q", [0, 1, None])
def test_recoil_energy_over_the_modes_of_one_axis_is_alpha_times_the_free_recoil_energy(
    family: str, q: int | None
) -> None:
    """sum_m alpha_m c_{i,m}^2 (k x0_m)^2 hbar omega_m = alpha_axis (hbar k)^2/(2 m_i) exactly, by completeness of the family's
    eigenvectors (Section 4.2.8, the module's unit test), for both ions of a two-ion 171Yb+ chain."""
    yb = species("171Yb+")
    crystal = solve_crystal(trap(), (yb, yb))
    for ion in (0, 1):
        assert recoil_energy_ratio(crystal, ion, K_369, q, Z, family) == pytest.approx(1.0, abs=1e-12)


def test_mixed_species_recoil_energy_uses_each_ions_own_mass() -> None:
    """171Yb+ next to 40Ca+ (the rf at 60 MHz keeps Ca+ Mathieu-stable): the identity holds per ion with m_i under (hbar k)^2/(2 m_i)."""
    crystal = solve_crystal(trap(RfDrive(300.0, 60e6)), (species("171Yb+"), species("40Ca+")))
    assert crystal.masses_kg[0] / crystal.masses_kg[1] > 4.0
    for ion in (0, 1):
        assert recoil_energy_ratio(crystal, ion, K_369, 1, Z, "axial") == pytest.approx(1.0, abs=1e-12)
    # the wrong mass (the other ion's) breaks it by the mass ratio
    quanta = recoil_heating_quanta(crystal, 1, K_369, 1, Z)
    energy = sum(
        quanta[m] * HBAR_J_S * crystal.modes[m].omega_rad_s
        for m, mode in enumerate(crystal.modes)
        if mode.family == "axial"
    )
    wrong = angular_factor(1, 1.0) * free_recoil_energy_j(K_369, float(crystal.masses_kg[0]))
    assert energy / wrong == pytest.approx(crystal.masses_kg[0] / crystal.masses_kg[1], rel=1e-9)


def test_participation_enters_the_emission_lamb_dicke_parameter_and_cancels_from_the_floor_weight() -> None:
    """eta_em,{i,m} = k |c_{i,m}| x0_{i,m}: for the COM mode of two equal ions c = 1/sqrt 2, and the drive's eta carries the same c, so
    the carrier weight (eta~/eta)^2 of Section 4.2.2 is independent of N (Section 9.17 row 328)."""
    yb = species("171Yb+")
    two = solve_crystal(trap(), (yb, yb))
    one = solve_crystal(trap(), (yb,))
    com = two.mode_index("axial", 0)
    assert abs(two.modes[com].eigenvector[0]) == pytest.approx(1.0 / math.sqrt(2.0))
    assert emission_lamb_dicke(two, 0, K_369, com) == pytest.approx(
        emission_lamb_dicke(one, 0, K_369, 0) / math.sqrt(2.0), rel=1e-9
    )
    drive_two = two.lamb_dicke(0, com, K_369 * np.array(Z), micromotion=None)
    drive_one = one.lamb_dicke(0, 0, K_369 * np.array(Z), micromotion=None)
    weight_two = (0.4 * emission_lamb_dicke(two, 0, K_369, com) ** 2) / drive_two**2
    weight_one = (0.4 * emission_lamb_dicke(one, 0, K_369, 0) ** 2) / drive_one**2
    assert weight_two == pytest.approx(weight_one, rel=1e-12) and weight_one == pytest.approx(0.4, rel=1e-12)


def test_joint_kick_over_three_axes_deposits_the_free_recoil_energy_with_the_cartesian_sum_rule() -> None:
    """One sampled direction kicks every mode through the product of displacements; averaged over the sigma pattern the per-axis
    quanta are alpha_axis eta_axis^2 (2/5 along B, 3/10 across) and the total energy is R (sum of alphas = 1; Itano's per-event
    invariant without the Doppler term). Section 9.3 "Single kick and motion independence"."""
    yb = species("171Yb+")
    crystal = solve_crystal(trap(), (yb,))
    space = HilbertSpace(
        ion_dims=(2,),
        resolved=tuple(ModeTruncation(m, 8, (0, 2), 0.2) for m in range(3)),
        enr_group=None,
        frozen=(),
    )
    quad = direction_quadrature(6, 8, axis=Z)
    rho0 = qt.ket2dm(qt.tensor(qt.basis(2, 0), *[qt.basis(8, 0)] * 3))
    out = 0.0 * rho0
    for k_hat, w in zip(quad.directions, quad.weights):
        kick = multi_mode_kick(space, recoil_projections(crystal, 0, K_369, tuple(k_hat)))
        out = out + w * float(pattern_density(1, float(k_hat[2]))) * kick * rho0 * kick.dag()
    assert float(out.tr()) == pytest.approx(1.0, abs=1e-9)
    energy = sum(
        float(qt.expect(space.number(m), out)) * HBAR_J_S * crystal.modes[m].omega_rad_s for m in range(3)
    )
    assert energy / free_recoil_energy_j(K_369, float(crystal.masses_kg[0])) == pytest.approx(1.0, abs=1e-6)
    for m, mode in enumerate(crystal.modes):
        alpha = angular_factor(1, float(np.dot(mode.e_hat, Z)))
        eta2 = emission_lamb_dicke(crystal, 0, K_369, m) ** 2
        assert float(qt.expect(space.number(m), out)) == pytest.approx(alpha * eta2, rel=1e-4)


def test_frozen_modes_receive_no_kick_operator() -> None:
    space = HilbertSpace(
        ion_dims=(2,), resolved=(ModeTruncation(0, 6, (0, 1), 0.1),), enr_group=None, frozen=(1,)
    )
    with pytest.raises(ValueError):
        multi_mode_kick(space, {0: 0.05, 1: 0.05})
    assert multi_mode_kick(space, {}).shape == space.identity().shape


@pytest.mark.parametrize("quad_name", ["marginal", "minimal"])
def test_recoil_kernel_matrix_is_column_stochastic_with_mean_kick_alpha_eta_squared(quad_name: str) -> None:
    """K[n, n'] = sum_j p_j |<n|D(-i eta u_j)|n'>|^2: every column sums to one and moves the mean by alpha eta_em^2 exactly."""
    alpha = 0.4
    quad = marginal_quadrature(1, 1.0) if quad_name == "marginal" else minimal_quadrature(alpha)
    kernel = recoil_kernel_matrix(60, 0.1, quad)
    n = np.arange(60)
    assert np.allclose(kernel.sum(axis=0)[:40], 1.0, atol=1e-12)
    for column in range(6):
        assert float(n @ kernel[:, column]) - column == pytest.approx(alpha * 0.01, abs=1e-12)


def test_recoil_unit_regressions() -> None:
    """Steck: 133Cs 852 nm v_r = 3.5 mm/s, T_r = 198 nK; 87Rb 780 nm 5.9 mm/s, 362 nK (k_B T_r = (hbar k)^2/m); Itano: 138Ba+ 493 nm
    R/h = 5.9 kHz, R/(hbar gamma) = 2.8e-4; 24Mg+ 280 nm 106 kHz, 2.5e-3 (Section 9.3 "Recoil unit regression")."""
    cs = (132.905 * ATOMIC_MASS_KG, TWO_PI / 852.347e-9)
    rb = (86.909 * ATOMIC_MASS_KG, TWO_PI / 780.241e-9)
    assert recoil_velocity_m_per_s(cs[1], cs[0]) * 1e3 == pytest.approx(3.52, abs=0.02)
    assert recoil_temperature_k(cs[1], cs[0]) * 1e9 == pytest.approx(198.3, abs=0.5)
    assert recoil_velocity_m_per_s(rb[1], rb[0]) * 1e3 == pytest.approx(5.88, abs=0.02)
    assert recoil_temperature_k(rb[1], rb[0]) * 1e9 == pytest.approx(362.0, abs=1.0)
    for mass_u, lam, gamma_hz, khz, ratio in (
        (137.905, 493.4e-9, 21e6, 5.94, 2.83e-4),
        (23.985, 279.6e-9, 43e6, 106.4, 2.47e-3),
    ):
        r = free_recoil_energy_j(TWO_PI / lam, mass_u * ATOMIC_MASS_KG)
        assert r / (TWO_PI * HBAR_J_S) / 1e3 == pytest.approx(khz, rel=2e-3)
        assert r / (HBAR_J_S * TWO_PI * gamma_hz) == pytest.approx(ratio, rel=5e-3)
