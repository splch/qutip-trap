"""The spontaneous-emission recoil kernel: the dipole patterns and their angular factors, the recoil quadratures, the
vector form, the Kraus kicks of one ion on one mode, the per-ion participation and the joint multi-mode kick."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
import qutip as qt

from qutip_trap.dynamics.multilevel import (
    ModeSpec,
    MultiLevelOptions,
    build_multilevel,
    decay_sum_rule_residual,
)
from qutip_trap.dynamics.operators import displacement_operator
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.light.recoil import (
    MOMENT_TOLERANCE,
    VectorChannel,
    angular_factor,
    direction_quadrature,
    emission_lamb_dicke,
    free_recoil_energy_j,
    marginal,
    marginal_quadrature,
    minimal_quadrature,
    pattern_density,
    recoil_kernel_matrix,
    recoil_projections,
    vector_channels,
)
from qutip_trap.species import species
from qutip_trap.trap.crystal import Crystal, solve_crystal
from qutip_trap.trap.model import Trap
from qutip_trap.trap.pseudopotential import RfDrive
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI
from tests.fixtures import (
    MASS_KG,
    TWO_LEVEL_EXCITED_PLUS,
    TWO_LEVEL_GROUND,
    WAVELENGTH_M,
    gamma_rad_s,
    sigma_plus_beam,
    structure,
    two_level_atom,
)

Z = (0.0, 0.0, 1.0)
X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
K_369 = TWO_PI / 369.5e-9


def test_pattern_densities_normalize_to_one_over_the_sphere() -> None:
    u, w = np.polynomial.legendre.leggauss(12)
    for q in (-1, 0, 1):
        total = 2.0 * math.pi * float(np.sum(w * np.asarray(pattern_density(q, u))))
        assert total == pytest.approx(1.0, abs=1e-13)


def test_analytic_angular_factors() -> None:
    """pi: 1/5 along B, 2/5 perpendicular; sigma: 2/5 along B, 3/10 perpendicular; isotropic 1/3."""
    assert angular_factor(0, 1.0) == pytest.approx(0.2)
    assert angular_factor(0, 0.0) == pytest.approx(0.4)
    assert angular_factor(1, 1.0) == pytest.approx(0.4)
    assert angular_factor(-1, 0.0) == pytest.approx(0.3)
    assert angular_factor(None, 0.7) == pytest.approx(1.0 / 3.0)
    # Cartesian sum rule: alpha_x + alpha_y + alpha_z = 1 for every pattern
    for q in (0, 1):
        assert angular_factor(q, 1.0) + 2.0 * angular_factor(q, 0.0) == pytest.approx(1.0)


def test_marginals_reduce_to_the_principal_forms() -> None:
    u = np.linspace(-1.0, 1.0, 11)
    assert np.allclose(marginal(0, 1.0, u), 0.75 * (1.0 - u**2))
    assert np.allclose(marginal(1, 1.0, u), 0.375 * (1.0 + u**2))
    assert np.allclose(marginal(0, 0.0, u), 0.375 * (1.0 + u**2))  # pi perpendicular is sigma along
    assert np.allclose(marginal(1, 0.0, u), (9.0 - 3.0 * u**2) / 16.0)
    assert np.allclose(marginal(None, 0.3, u), 0.5)


@pytest.mark.parametrize("q", [-1, 0, 1, None])
@pytest.mark.parametrize("cos_chi", [1.0, 0.0, 0.6, -0.31])
def test_gauss_legendre_quadrature_identities_to_1e_12(q: int | None, cos_chi: float) -> None:
    """sum p_j = 1 and sum p_j u_j^2 = alpha to 1e-12 for 16 nodes; two nodes are refused."""
    quad = marginal_quadrature(q, cos_chi, 16)
    assert abs(float(np.sum(quad.weights)) - 1.0) < MOMENT_TOLERANCE
    assert abs(quad.alpha - angular_factor(q, cos_chi)) < MOMENT_TOLERANCE
    with pytest.raises(ValueError):
        marginal_quadrature(q, cos_chi, 2)


def test_three_nodes_are_exact_for_the_second_moment_and_two_would_return_one_third() -> None:
    quad3 = marginal_quadrature(0, 1.0, 3)
    assert abs(quad3.alpha - 0.2) < MOMENT_TOLERANCE
    u2, w2 = np.polynomial.legendre.leggauss(2)
    p2 = w2 * np.asarray(marginal(0, 1.0, u2))
    assert float(np.sum(p2 * u2**2)) == pytest.approx(1.0 / 3.0)


def test_minimal_quadrature_has_exact_first_and_second_moments() -> None:
    quad = minimal_quadrature(0.4)
    assert np.allclose(quad.nodes, [-1.0, 0.0, 1.0])
    assert np.allclose(quad.weights, [0.2, 0.6, 0.2])
    assert float(np.sum(quad.weights * quad.nodes)) == 0.0
    assert quad.alpha == pytest.approx(0.4)


def test_direction_quadrature_integrates_to_the_sphere() -> None:
    quad = direction_quadrature(6, 8)
    assert float(np.sum(quad.weights)) == pytest.approx(4.0 * math.pi)
    assert np.allclose(np.linalg.norm(quad.directions, axis=1), 1.0)
    for k, pols in zip(quad.directions, quad.polarizations):
        assert abs(np.dot(pols[0], k)) < 1e-12 and abs(np.dot(pols[1], k)) < 1e-12
        assert abs(np.dot(pols[0], pols[1])) < 1e-12


def _alphas(channels: Sequence[VectorChannel]) -> dict[int, float]:
    """alpha_q = sum_channels weight |c_q|^2 u^2 for each pure q: the second moments the vector form implies."""
    return {
        q: float(sum(ch.weight * abs(ch.amplitudes[i]) ** 2 * ch.u**2 for ch in channels))
        for i, q in enumerate((-1, 0, 1))
    }


@pytest.mark.parametrize("axis, expected", [(Z, (0.4, 0.2, 0.4)), (X, (0.3, 0.4, 0.3)), (Y, (0.3, 0.4, 0.3))])
def test_vector_form_derives_the_scalar_angular_factors_without_hard_coding_them(
    axis: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    """sum_lambda |eps . e_q|^2 = 1 - |k . e_q|^2 is the scalar pattern: unit norm, and second moments alpha_q."""
    channels = vector_channels(Z, axis)
    alphas = _alphas(channels)
    for i, (q, a) in enumerate(zip((-1, 0, 1), expected)):
        assert alphas[q] == pytest.approx(a, abs=1e-12)
        assert sum(ch.weight * abs(ch.amplitudes[i]) ** 2 for ch in channels) == pytest.approx(1.0, abs=1e-12)


def test_vector_form_oblique_axis_and_grid_independence() -> None:
    oblique = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
    alphas = _alphas(vector_channels(Z, oblique))
    assert alphas[0] == pytest.approx(angular_factor(0, 1.0 / math.sqrt(2.0)), abs=1e-12)
    assert alphas[1] == pytest.approx(0.35, abs=1e-12)
    rotated = _alphas(vector_channels(Z, Z, grid_axis=X, n_theta=8, n_phi=10))
    assert rotated[0] == pytest.approx(0.2, abs=1e-12) and rotated[1] == pytest.approx(0.4, abs=1e-12)
    for q in (-1, 0, 1):
        total = sum(_alphas(vector_channels(Z, ax))[q] for ax in (X, Y, Z))
        assert total == pytest.approx(1.0, abs=1e-12)


def test_free_recoil_energy_over_hbar_omega_is_eta_squared() -> None:
    """(hbar k)^2/(2 m)/(hbar omega) = (k x0)^2: k x0 = 0.053 for 171Yb+ at 369.5 nm on a 3 MHz mode."""
    k = TWO_PI / WAVELENGTH_M
    eta = k * math.sqrt(HBAR_J_S / (2.0 * MASS_KG * TWO_PI * 3.0e6))
    assert eta == pytest.approx(0.053, abs=0.001)
    assert free_recoil_energy_j(k, MASS_KG) / (HBAR_J_S * TWO_PI * 3.0e6) == pytest.approx(eta**2, rel=1e-12)


def test_free_recoil_energy_regressions() -> None:
    """Itano: 138Ba+ 493 nm R/h = 5.9 kHz, R/(hbar gamma) = 2.8e-4; 24Mg+ 280 nm 106 kHz, 2.5e-3."""
    for mass_u, lam, gamma_hz, khz, ratio in (
        (137.905, 493.4e-9, 21e6, 5.94, 2.83e-4),
        (23.985, 279.6e-9, 43e6, 106.4, 2.47e-3),
    ):
        r = free_recoil_energy_j(TWO_PI / lam, mass_u * ATOMIC_MASS_KG)
        assert r / (TWO_PI * HBAR_J_S) / 1e3 == pytest.approx(khz, rel=2e-3)
        assert r / (HBAR_J_S * TWO_PI * gamma_hz) == pytest.approx(ratio, rel=5e-3)


# ---- one ion on one mode: the Kraus kicks ---------------------------------------------------------------------------


def _two_level_build(recoil: str, axis: tuple[float, float, float], d: int = 12):  # type: ignore[no-untyped-def]
    st = structure(two_level_atom())
    g = gamma_rad_s()
    nu = 50.0 * g
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.3 * g, -nu)
    mode = ModeSpec(nu, MASS_KG, axis, d=d, expected_n_max=2)
    return build_multilevel(st, [beam], mode=mode, options=MultiLevelOptions(recoil=recoil)), mode  # type: ignore[arg-type]


@pytest.mark.parametrize("recoil", ["minimal", "marginal", "vector"])
def test_kicks_are_unitary_sum_c_dag_c_is_gamma_projector_times_identity(recoil: str) -> None:
    build, _mode = _two_level_build(recoil, Z)
    assert decay_sum_rule_residual(build) < 1e-12
    total = sum((c.dag() * c for c in build.c_ops), 0.0 * build.c_ops[0])
    gamma = gamma_rad_s()
    p_e = build.level_projector("P2/2")
    assert (total - gamma * p_e).norm() / gamma < 1e-12


@pytest.mark.parametrize(
    "recoil, axis", [("minimal", Z), ("marginal", Z), ("vector", Z), ("vector", X), ("marginal", X)]
)
def test_single_kick_expectation_is_alpha_eta_em_squared_with_zero_momentum(
    recoil: str, axis: tuple[float, float, float]
) -> None:
    """The Kraus map applied once to |e, 0> gives <n> = alpha eta_em^2 exactly and <p> = 0."""
    build, mode = _two_level_build(recoil, axis, d=16)
    assert build.space is not None
    e = build.index(TWO_LEVEL_EXCITED_PLUS)
    psi = qt.tensor(qt.basis(build.n_internal, e), qt.basis(mode.d, 0))
    rho = qt.ket2dm(psi)
    out = sum((c * rho * c.dag() for c in build.c_ops), 0.0 * rho)
    out = out / out.tr()
    n_op = build.number()
    a = build.space.annihilation(0)
    eta_em = (TWO_PI / WAVELENGTH_M) * mode.x0_m
    # the sigma+ excited state |e, +1> decays with q = -1 only: the sigma pattern about B
    alpha = angular_factor(1, float(np.dot(axis, Z)))
    assert float(np.real(qt.expect(n_op, out))) == pytest.approx(alpha * eta_em**2, rel=1e-6)
    assert abs(complex(qt.expect(a, out))) < 1e-12


def test_recoil_kick_is_the_displacement_of_the_emitted_photon() -> None:
    """D(-i eta u) built by expm equals exp(-i eta u (a + a^dag)) and is unitary."""
    d = 20
    eta_u = 0.17
    a = qt.destroy(d)
    kick = displacement_operator(d, -1j * eta_u)
    direct = (-1j * eta_u * (a + a.dag())).expm()
    assert (kick - direct).norm() < 1e-10
    assert np.asarray((kick.dag() * kick).full())[:10, :10] == pytest.approx(np.eye(10), abs=1e-10)


def test_recoil_needs_a_mode_and_a_mode_needs_no_recoil() -> None:
    st = structure(two_level_atom())
    beam = sigma_plus_beam(st, TWO_LEVEL_GROUND, TWO_LEVEL_EXCITED_PLUS, 0.1 * gamma_rad_s(), 0.0)
    with pytest.raises(ValueError):
        build_multilevel(st, [beam], options=MultiLevelOptions(recoil="minimal"))
    build = build_multilevel(st, [beam])
    assert build.space is None and build.channels[0].eta_em == 0.0


@pytest.mark.parametrize("quad_name", ["marginal", "minimal"])
def test_recoil_kernel_matrix_is_column_stochastic_with_mean_kick_alpha_eta_squared(quad_name: str) -> None:
    """K[n, n'] = sum_j p_j |<n|D(-i eta u_j)|n'>|^2: every column sums to one and moves the mean by alpha eta_em^2."""
    alpha = 0.4
    quad = marginal_quadrature(1, 1.0) if quad_name == "marginal" else minimal_quadrature(alpha)
    kernel = recoil_kernel_matrix(60, 0.1, quad)
    n = np.arange(60)
    assert np.allclose(kernel.sum(axis=0)[:40], 1.0, atol=1e-12)
    for column in range(6):
        assert float(n @ kernel[:, column]) - column == pytest.approx(alpha * 0.01, abs=1e-12)


# ---- a crystal: per-ion participation and the joint kick ------------------------------------------------------------


def _trap(rf: RfDrive | None = None) -> Trap:
    return Trap(
        omega_hz=(3.0e6, 2.9e6, 1.0e6),
        axis_angle_rad=0.0,
        rf=rf,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


def _quanta(crystal: Crystal, ion: int, q: int | None) -> dict[int, float]:
    """alpha_m eta_em,{i,m}^2 per mode: the mean recoil heating of one photon of pattern q emitted by ion i, alpha_m at the
    angle between the mode axis and B = z."""
    return {
        m: angular_factor(q, float(np.dot(mode.e_hat, Z))) * emission_lamb_dicke(crystal, ion, K_369, m) ** 2
        for m, mode in enumerate(crystal.modes)
    }


def _family_energy_over_free(crystal: Crystal, ion: int, q: int | None, family: str, mass_kg: float) -> float:
    """sum over one axis family of alpha_m eta_em^2 hbar omega_m, over alpha_axis (hbar k)^2/(2 m)."""
    quanta = _quanta(crystal, ion, q)
    members = [m for m, mode in enumerate(crystal.modes) if mode.family == family]
    alpha_axis = angular_factor(q, float(np.dot(crystal.modes[members[0]].e_hat, Z)))
    energy = sum(quanta[m] * HBAR_J_S * crystal.modes[m].omega_rad_s for m in members)
    return float(energy / (alpha_axis * free_recoil_energy_j(K_369, mass_kg)))


@pytest.mark.parametrize("family", ["axial", "transverse_1", "transverse_2"])
@pytest.mark.parametrize("q", [0, 1, None])
def test_recoil_energy_over_the_modes_of_one_axis_is_alpha_times_the_free_recoil_energy(
    family: str, q: int | None
) -> None:
    """sum_m alpha_m c_{i,m}^2 (k x0_m)^2 hbar omega_m = alpha_axis (hbar k)^2/(2 m_i) exactly, by completeness of the
    family's eigenvectors, for both ions of a two-ion 171Yb+ chain."""
    yb = species("171Yb+")
    crystal = solve_crystal(_trap(), (yb, yb))
    for ion in (0, 1):
        ratio = _family_energy_over_free(crystal, ion, q, family, float(crystal.masses_kg[ion]))
        assert ratio == pytest.approx(1.0, abs=1e-12)


def test_mixed_species_recoil_energy_uses_each_ions_own_mass() -> None:
    """171Yb+ next to 40Ca+ (the rf at 60 MHz keeps Ca+ Mathieu-stable): the identity holds per ion with m_i, and the other
    ion's mass breaks it by the mass ratio."""
    crystal = solve_crystal(_trap(RfDrive(300.0, 60e6)), (species("171Yb+"), species("40Ca+")))
    m0, m1 = (float(m) for m in crystal.masses_kg)
    assert m0 / m1 > 4.0
    for ion, mass in ((0, m0), (1, m1)):
        assert _family_energy_over_free(crystal, ion, 1, "axial", mass) == pytest.approx(1.0, abs=1e-12)
    assert _family_energy_over_free(crystal, 1, 1, "axial", m0) == pytest.approx(m0 / m1, rel=1e-9)


def test_participation_enters_the_emission_lamb_dicke_parameter_and_cancels_from_the_floor_weight() -> None:
    """eta_em,{i,m} = k |c_{i,m}| x0_{i,m}: for the COM mode of two equal ions c = 1/sqrt 2, and the drive's eta carries the
    same c, so the carrier weight (eta~/eta)^2 is independent of N."""
    yb = species("171Yb+")
    two = solve_crystal(_trap(), (yb, yb))
    one = solve_crystal(_trap(), (yb,))
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
    """One sampled direction kicks every mode through the product of displacements prod_m D_m(-i eta_m(k_hat)); averaged
    over the sigma pattern the per-axis quanta are alpha_axis eta_axis^2 and the total energy is (hbar k)^2/(2 m)."""
    yb = species("171Yb+")
    crystal = solve_crystal(_trap(), (yb,))
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
        etas = recoil_projections(crystal, 0, K_369, tuple(k_hat))
        kick = space.embed_many(
            {space.mode_factor(m): space.displacement_factor(m, -eta) for m, eta in etas.items()}
        )
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
