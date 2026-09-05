"""The spontaneous-emission recoil kernel of PLAN.md Section 4.2.8 for one ion and one mode (M3a; Section 9.3 rows
"Recoil quadrature identities" and "Single kick and motion independence")."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.dynamics.multilevel import (
    ModeSpec,
    MultiLevelOptions,
    build_multilevel,
    decay_sum_rule_residual,
)
from qutip_trap.hilbert.operators import displacement_operator
from qutip_trap.light.recoil import (
    MOMENT_TOLERANCE,
    angular_factor,
    derived_angular_factors,
    derived_pattern_norms,
    direction_quadrature,
    free_recoil_energy_j,
    marginal,
    marginal_quadrature,
    minimal_quadrature,
    pattern_density,
    recoil_quanta_per_photon,
    vector_channels,
)
from qutip_trap.units import HBAR_J_S, TWO_PI
from tests.bloch_fixtures import (
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


def test_pattern_densities_normalize_to_one_over_the_sphere() -> None:
    u, w = np.polynomial.legendre.leggauss(12)
    for q in (-1, 0, 1):
        total = 2.0 * math.pi * float(np.sum(w * np.asarray(pattern_density(q, u))))
        assert total == pytest.approx(1.0, abs=1e-13)


def test_analytic_angular_factors_are_the_section_4_2_8_values() -> None:
    """pi: 1/5 along B, 2/5 perpendicular; sigma: 2/5 along B, 3/10 perpendicular; isotropic 1/3."""
    assert angular_factor(0, 1.0) == pytest.approx(0.2)
    assert angular_factor(0, 0.0) == pytest.approx(0.4)
    assert angular_factor(1, 1.0) == pytest.approx(0.4)
    assert angular_factor(-1, 0.0) == pytest.approx(0.3)
    assert angular_factor(None, 0.7) == pytest.approx(1.0 / 3.0)
    # Cartesian sum rule: alpha_x + alpha_y + alpha_z = 1 for every pattern (pi: 2/5 + 2/5 + 1/5; sigma: 3/10 + 3/10 + 2/5)
    for q in (0, 1):
        assert angular_factor(q, 1.0) + 2.0 * angular_factor(q, 0.0) == pytest.approx(1.0)


def test_marginals_reduce_to_the_printed_principal_forms() -> None:
    u = np.linspace(-1.0, 1.0, 11)
    assert np.allclose(marginal(0, 1.0, u), 0.75 * (1.0 - u**2))
    assert np.allclose(marginal(1, 1.0, u), 0.375 * (1.0 + u**2))
    assert np.allclose(
        marginal(0, 0.0, u), 0.375 * (1.0 + u**2)
    )  # pi perpendicular: the same marginal as sigma along
    assert np.allclose(marginal(1, 0.0, u), (9.0 - 3.0 * u**2) / 16.0)
    assert np.allclose(marginal(None, 0.3, u), 0.5)


@pytest.mark.parametrize("q", [-1, 0, 1, None])
@pytest.mark.parametrize("cos_chi", [1.0, 0.0, 0.6, -0.31])
def test_gauss_legendre_quadrature_identities_to_1e_12(q: int | None, cos_chi: float) -> None:
    """sum p_j = 1 and sum p_j u_j^2 = alpha to 1e-12 for 16 nodes; two nodes are refused (Section 4.2.8)."""
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
    # transverse orthonormal polarizations
    for k, pols in zip(quad.directions, quad.polarizations):
        assert abs(np.dot(pols[0], k)) < 1e-12 and abs(np.dot(pols[1], k)) < 1e-12
        assert abs(np.dot(pols[0], pols[1])) < 1e-12


@pytest.mark.parametrize("axis, expected", [(Z, (0.4, 0.2, 0.4)), (X, (0.3, 0.4, 0.3)), (Y, (0.3, 0.4, 0.3))])
def test_vector_form_derives_the_scalar_angular_factors_without_hard_coding_them(
    axis: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    """sum_lambda |eps . e_q|^2 = 1 - |k . e_q|^2 is the scalar pattern; its second moments are the alpha_q (Section 4.2.8)."""
    channels = vector_channels(Z, axis)
    alphas = derived_angular_factors(channels)
    norms = derived_pattern_norms(channels)
    for q, a in zip((-1, 0, 1), expected):
        assert alphas[q] == pytest.approx(a, abs=1e-12)
        assert norms[q] == pytest.approx(1.0, abs=1e-12)


def test_vector_form_oblique_axis_and_grid_independence() -> None:
    oblique = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))
    alphas = derived_angular_factors(vector_channels(Z, oblique))
    assert alphas[0] == pytest.approx(angular_factor(0, 1.0 / math.sqrt(2.0)), abs=1e-12)
    assert alphas[1] == pytest.approx(0.35, abs=1e-12)
    rotated = derived_angular_factors(vector_channels(Z, Z, grid_axis=X, n_theta=8, n_phi=10))
    assert rotated[0] == pytest.approx(0.2, abs=1e-12) and rotated[1] == pytest.approx(0.4, abs=1e-12)
    # Cartesian sum rule from the vector form
    for q in (-1, 0, 1):
        total = sum(derived_angular_factors(vector_channels(Z, ax))[q] for ax in (X, Y, Z))
        assert total == pytest.approx(1.0, abs=1e-12)


def test_recoil_quanta_and_free_recoil_energy() -> None:
    """171Yb+ 369.5 nm on a 3 MHz mode: k x0 = 0.053, alpha = 2/5 gives about 1.1e-3 quanta per photon (Section 4.2.8)."""
    k = TWO_PI / WAVELENGTH_M
    x0 = math.sqrt(HBAR_J_S / (2.0 * MASS_KG * TWO_PI * 3.0e6))
    eta = k * x0
    assert eta == pytest.approx(0.053, abs=0.001)
    assert recoil_quanta_per_photon(0.4, eta) == pytest.approx(1.1e-3, rel=0.05)
    # the free recoil energy over hbar omega equals eta^2: (hbar k)^2/(2 m) / (hbar omega) = k^2 x0^2
    assert free_recoil_energy_j(k, MASS_KG) / (HBAR_J_S * TWO_PI * 3.0e6) == pytest.approx(eta**2, rel=1e-12)


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
    """The Kraus map applied once to |e, 0> gives <n> = alpha eta_em^2 exactly and <p> = 0 (Section 9.3)."""
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
    # sigma+ excited state |e, +1> decays with q = -1 (Delta m = -1) only: the sigma pattern about B
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
