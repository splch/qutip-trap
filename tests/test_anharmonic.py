"""Section 9.1 (Anharmonic selection rules), 9.12 row 63 and Section 4.1.4: Coulomb cubic couplings from first principles."""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pytest

from qutip_trap.species import species
from qutip_trap.trap.anharmonic import (
    AnharmonicTerms,
    axial_cubic_closed_form_rad_s,
    axial_hessian_check,
    coulomb_anharmonic_terms,
    coulomb_fourth_derivatives,
    coulomb_third_derivatives,
    coupling_g_rad_s,
    dispersive_phase_bound_rad,
    marquet_c_tensor,
    marquet_d_coefficients,
    marquet_selection_rules,
    nonlinearity_epsilon,
    three_mode_resonances,
)
from qutip_trap.trap.crystal import (
    K_COULOMB_J_M,
    axial_modes_dimensionless,
    coulomb_hessian_j_per_m2,
    equilibrium_dimensionless,
    solve_crystal,
)
from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI
from tests.test_crystal import _explicit


def _coulomb_energy(pos: np.ndarray) -> float:
    n = pos.shape[0]
    return sum(K_COULOMB_J_M / np.linalg.norm(pos[i] - pos[j]) for i in range(n) for j in range(i + 1, n))


def test_third_and_fourth_derivatives_against_finite_differences() -> None:
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(3, 3)) * 3e-6
    v3 = coulomb_third_derivatives(pos)
    v4 = coulomb_fourth_derivatives(pos)
    assert np.allclose(v3, v3.transpose(1, 0, 2)) and np.allclose(v3, v3.transpose(0, 2, 1)), (
        "fully symmetric"
    )
    h = 2e-9
    for c in range(9):
        dp = pos.ravel().copy()
        dm = dp.copy()
        dp[c] += h
        dm[c] -= h
        fd = (
            coulomb_hessian_j_per_m2(dp.reshape(3, 3), np.ones(3))
            - coulomb_hessian_j_per_m2(dm.reshape(3, 3), np.ones(3))
        ) / (2 * h)
        assert np.allclose(v3[:, :, c], fd, rtol=1e-5, atol=1e-5 * np.max(np.abs(v3)))
        fd4 = (coulomb_third_derivatives(dp.reshape(3, 3)) - coulomb_third_derivatives(dm.reshape(3, 3))) / (
            2 * h
        )
        assert np.allclose(v4[:, :, :, c], fd4, rtol=1e-4, atol=1e-4 * np.max(np.abs(v4)))
    # translation invariance: summing any index over all ions of one axis gives zero
    assert abs(np.sum(v3.reshape(3, 3, 3, 3, 3, 3), axis=4)).max() < 1e-9 * np.max(np.abs(v3))


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_marquet_selection_rules(n: int) -> None:
    """D_mn1 = 0 and D_mn2 = ((1 - mu_m)/(2|u|)) delta_mn for every N (Marquet 2003); N = 2: D_222 = -1.1225."""
    u = equilibrium_dimensionless(n)
    mu, b = axial_modes_dimensionless(u)
    r1, r2 = marquet_selection_rules(u, b, mu)
    assert r1 < 1e-12 and r2 < 1e-12
    assert np.max(np.abs(axial_hessian_check(u))) < 1e-12, (
        "sum_p u_p C_mnp = (1/2)(delta - A), the Euler identity"
    )
    if n == 2:
        d = marquet_d_coefficients(marquet_c_tensor(u), b)
        assert d[1, 1, 1] == pytest.approx(-1.1225, abs=6e-5)
        assert d[1, 1, 1] == pytest.approx((1 - 3) / (2 * np.linalg.norm(u)), rel=1e-12)


def test_cubic_energy_sign_along_the_stretch_mode_fixes_the_convention() -> None:
    """E(u + s b2) - E(u) - (1/2) mu_2 s^2 = D_222 s^3 + O(s^4) with D_222 < 0: stretching the pair softens the Coulomb spring, so the
    cubic term of the potential is + sum_pqr D_pqr xi xi xi in the C = (1/6) d^3 F convention (the plan's Section 4.1.4 prints a leading
    minus sign that would flip it; reported as a plan inconsistency)."""
    u = equilibrium_dimensionless(2)
    mu, b = axial_modes_dimensionless(u)
    d222 = marquet_d_coefficients(marquet_c_tensor(u), b)[1, 1, 1]

    def energy(v: np.ndarray) -> float:
        return 0.5 * float(np.sum(v * v)) + 1.0 / abs(v[1] - v[0])

    e0 = energy(u)
    for s in (1e-2, 5e-3):
        cubic = (
            energy(u + s * b[:, 1])
            - e0
            - 0.5 * mu[1] * s * s
            - (energy(u - s * b[:, 1]) - e0 - 0.5 * mu[1] * s * s)
        ) / 2
        assert cubic / s**3 == pytest.approx(d222, rel=2e-3)
    assert d222 < 0.0


def test_cartesian_route_reproduces_marquet_for_equal_masses_and_com_decouples() -> None:
    ca = species("40Ca+")
    cr = solve_crystal(_explicit((5e6, 5.5e6, 2e6)), (ca,) * 3)
    terms = coulomb_anharmonic_terms(cr)
    closed = axial_cubic_closed_form_rad_s(cr)
    for key, val in closed.items():
        if abs(val) > 1e-6:
            assert terms.cubic_rad_s[key] == pytest.approx(val, rel=1e-9)
        else:
            assert abs(terms.cubic_rad_s.get(key, 0.0)) < 1e-6
    com = cr.mode_index("axial", 0)
    for key, val in terms.cubic_rad_s.items():
        if com in key:
            assert abs(val) < 1e-6 * terms.largest_cubic_rad_s(), (
                "D_mn1 = 0: the COM decouples from all three-mode mixing"
            )
    stretch, x_rock = cr.mode_index("axial", 1), cr.mode_index("transverse_1", 0)
    assert abs(terms.cubic_rad_s[(stretch, x_rock, x_rock)]) > 0.1 * terms.largest_cubic_rad_s(), (
        "z (x x) cross terms exist"
    )
    assert all(
        k not in terms.cubic_rad_s or abs(terms.cubic_rad_s[k]) < 1e-6 for k in ((x_rock, x_rock, x_rock),)
    ), "no transverse-only cubic term for a linear chain"
    with pytest.raises(ValueError):
        AnharmonicTerms(cubic_rad_s={(2, 1, 0): 1.0})


def test_nonlinearity_epsilon_and_coupling_anchors() -> None:
    """eps = 1.06e-3 (9Be+, 5 MHz), 7.09e-4 (40Ca+, 2 MHz), 3.79e-4 (171Yb+, 0.2 MHz); g/2pi = 1419 Hz and 76 Hz (Section 4.1.4)."""
    ca, yb = species("40Ca+"), species("171Yb+")
    assert nonlinearity_epsilon(9.0121822 * ATOMIC_MASS_KG, TWO_PI * 5.0e6) == pytest.approx(
        1.06e-3, rel=2e-3
    )
    assert nonlinearity_epsilon(ca.mass_u * ATOMIC_MASS_KG, TWO_PI * 2.0e6) == pytest.approx(
        7.09e-4, rel=2e-3
    )
    assert nonlinearity_epsilon(yb.mass_u * ATOMIC_MASS_KG, TWO_PI * 0.2e6) == pytest.approx(
        3.79e-4, rel=2e-3
    )
    assert coupling_g_rad_s(ca.mass_u * ATOMIC_MASS_KG, TWO_PI * 2.0e6) / TWO_PI == pytest.approx(
        1419.0, abs=1.0
    )
    assert coupling_g_rad_s(yb.mass_u * ATOMIC_MASS_KG, TWO_PI * 0.2e6) / TWO_PI == pytest.approx(
        76.0, abs=0.5
    )
    # the two-ion stretch self-coupling is |D_222| 2 eps omega_z mu_2^{-3/4} = 1.1225 x 2 eps omega_z / 3^{3/4}
    cr = solve_crystal(_explicit((5e6, 5e6, 2e6)), (ca, ca))
    terms = coulomb_anharmonic_terms(cr)
    eps = nonlinearity_epsilon(ca.mass_u * ATOMIC_MASS_KG, TWO_PI * 2e6)
    assert abs(terms.cubic_rad_s[(1, 1, 1)]) == pytest.approx(
        1.1225 * 2 * eps * TWO_PI * 2e6 / 3**0.75, rel=1e-4
    )


def test_resonance_checker_finds_a_tuned_three_mode_resonance() -> None:
    """omega_z,stretch = 2 omega_x,rock when omega_x^2 = 7/4 omega_z^2 for two ions (a z x x coupling); none for a detuned chain."""
    ca = species("40Ca+")
    wz = 1.0e6
    cr = solve_crystal(_explicit((math.sqrt(1.75) * wz, 1.5e6, wz)), (ca, ca))
    terms = coulomb_anharmonic_terms(cr)
    hits = three_mode_resonances(cr, terms, width_hz=1.0)
    stretch, rock = cr.mode_index("axial", 1), cr.mode_index("transverse_1", 0)
    assert any(set(h.modes) == {stretch, rock} and h.sign == +1 for h in hits)
    assert all(abs(h.mismatch_hz) < 1.0 for h in hits)
    quiet = solve_crystal(_explicit((5e6, 5.5e6, 2e6)), (ca, ca))
    assert three_mode_resonances(quiet, coulomb_anharmonic_terms(quiet), width_hz=1e4) == ()
    assert dispersive_phase_bound_rad(TWO_PI * 1.419e3, TWO_PI * 1e6, 100e-6) == pytest.approx(
        0.0013, abs=1e-4
    )
    with pytest.raises(ValueError):
        dispersive_phase_bound_rad(1.0, 0.0, 1.0)
    assert len(set(permutations((0, 1, 1)))) == 3
