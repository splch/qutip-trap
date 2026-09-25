"""Crystal equilibria, normal modes, mixed-species structure and Lamb-Dicke parameters against published closed forms."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.device.presets import secular_trap
from qutip_trap.species import species
from qutip_trap.trap.crystal import (
    Crystal,
    Mode,
    ZigzagError,
    axial_hessian_dimensionless,
    axial_modes_dimensionless,
    build_crystal,
    equilibrium_dimensionless,
    length_scale_m,
    solve_crystal,
)
from qutip_trap.trap.mathieu import MathieuParameters, c0_wronskian
from qutip_trap.trap.pseudopotential import RfDrive
from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI
from tests.fixtures import MassOnly


def _mixed_pair(
    m1_u: float, m2_u: float, omega_z1_hz: float = 1e6, transverse_factor: float = 10.0
) -> Crystal:
    """Two ions with the same dc spring (equal charges): omega_2 = omega_1/sqrt(mu) along every axis."""
    w1 = TWO_PI * omega_z1_hz
    per_ion = np.array([transverse_factor * w1, transverse_factor * w1, w1])
    w = np.vstack([per_ion, per_ion / math.sqrt(m2_u / m1_u)])
    return build_crystal((MassOnly(m1_u), MassOnly(m2_u)), w)


def _two_ion_axial_squared(mu: float) -> tuple[float, float]:
    """omega_-+^2/omega_z1^2 = 1 + 1/mu -+ sqrt(1 - 1/mu + 1/mu^2), mu = m_2/m_1 (Wubbena 2012 Eqs. 12-14)."""
    root = math.sqrt(1.0 - 1.0 / mu + 1.0 / mu**2)
    return 1.0 + 1.0 / mu - root, 1.0 + 1.0 / mu + root


# ---- James 1998 -------------------------------------------------------------------------------------------------


def test_james_closed_forms_for_two_and_three_ions() -> None:
    assert equilibrium_dimensionless(2) == pytest.approx([-((0.5) ** (2 / 3)), (0.5) ** (2 / 3)], rel=1e-12)
    assert equilibrium_dimensionless(3) == pytest.approx(
        [-((1.25) ** (1 / 3)), 0.0, (1.25) ** (1 / 3)], abs=1e-12
    )


@pytest.mark.parametrize("n", range(2, 11))
def test_james_equilibrium_is_a_zero_of_the_force_with_com_and_stretch_modes(n: int) -> None:
    u = equilibrium_dimensionless(n)
    assert np.all(np.diff(u) > 0.0) and abs(np.sum(u)) < 1e-12
    mu, b = axial_modes_dimensionless(u)
    assert mu[0] == pytest.approx(1.0, abs=1e-10) and mu[1] == pytest.approx(3.0, abs=1e-10)
    assert np.allclose(b[:, 0], 1.0 / math.sqrt(n)), "centre of mass"
    assert np.allclose(b[:, 1], u / np.linalg.norm(u)), (
        "stretch mode ~ u, sign-consistent with the closed form"
    )
    assert np.all(b[-1, :] > 0.0), "sign gauge: last component positive"
    assert np.allclose(b.T @ b, np.eye(n), atol=1e-12)


def test_james_table_2_third_eigenvalue_and_the_highest_axial_eigenvalue() -> None:
    """mu_3 = 29/5 at N = 3 and 5.81 ... 5.841 for N = 4 to 10 (James Table 2); mu_N for N up to 30; Marquet's N = 6
    example mu_5 = 13.513882, mu_6 = 18.265709."""
    mu3 = {n: axial_modes_dimensionless(equilibrium_dimensionless(n))[0][2] for n in range(3, 11)}
    assert mu3[3] == pytest.approx(5.8, abs=1e-10)
    assert round(mu3[4], 2) == 5.81
    for n, val in zip(range(5, 11), (5.818, 5.824, 5.829, 5.834, 5.838, 5.841)):
        assert round(mu3[n], 3) == val
    for n, mu_n in zip(
        (2, 3, 5, 10, 15, 17, 30), (3.0000, 5.8000, 13.4748, 43.2405, 86.5334, 107.3793, 288.6657)
    ):
        assert axial_modes_dimensionless(equilibrium_dimensionless(n))[0][-1] == pytest.approx(mu_n, abs=6e-5)
    mu6 = axial_modes_dimensionless(equilibrium_dimensionless(6))[0]
    assert mu6[4] == pytest.approx(13.513882, abs=5e-7) and mu6[5] == pytest.approx(18.265709, abs=5e-7)


def test_spacing_scale_171yb_at_1_mhz() -> None:
    """Two 171Yb+ ions at 1 MHz sit 2^{1/3} l = 3.4532 um apart; three at +-(5/4)^{1/3} l."""
    yb = species("171Yb+")
    s = length_scale_m(yb.mass_u * ATOMIC_MASS_KG, TWO_PI * 1e6)
    assert 2 ** (1 / 3) * s == pytest.approx(3.4532e-6, rel=1e-4)
    cr = solve_crystal(secular_trap((3e6, 3e6, 1e6)), (yb, yb))
    assert cr.positions_m[1, 2] - cr.positions_m[0, 2] == pytest.approx(2 ** (1 / 3) * s, rel=1e-12)
    cr3 = solve_crystal(secular_trap((3e6, 3e6, 1e6)), (yb, yb, yb))
    assert cr3.positions_m[2, 2] == pytest.approx((5 / 4) ** (1 / 3) * s, rel=1e-12)


def test_transverse_families_share_eigenvectors_and_reverse_the_ordering() -> None:
    """gamma_p = 1/alpha + 1/2 - mu_p/2 per family with alpha = (omega_z/omega_r)^2 (Marquet 2003); COM highest; 3N modes in
    canonical order."""
    yb = species("171Yb+")
    n = 5
    wx, wy, wz = 3.0e6, 2.6e6, 0.4e6
    cr = solve_crystal(secular_trap((wx, wy, wz)), (yb,) * n)
    assert len(cr.modes) == 3 * n
    mu, b = axial_modes_dimensionless(equilibrium_dimensionless(n))
    assert [m.omega_hz for m in cr.family("axial")] == pytest.approx(list(wz * np.sqrt(mu)), rel=1e-10)
    for fam, w_r in (("transverse_1", wx), ("transverse_2", wy)):
        gamma = (w_r / wz) ** 2 + 0.5 - mu / 2.0
        got = [m.omega_hz for m in cr.family(fam)]
        assert got == pytest.approx(sorted(wz * np.sqrt(gamma)), rel=1e-10)
        assert got[-1] == pytest.approx(w_r, rel=1e-12), (
            "the centre-of-mass mode is the highest transverse mode"
        )
        assert np.allclose(cr.family(fam)[-1].eigenvector, 1.0 / math.sqrt(n))
        for k, m in enumerate(
            cr.family(fam)
        ):  # the families share A's eigenvectors, reversed within the family
            assert np.allclose(np.abs(m.eigenvector), np.abs(b[:, n - 1 - k]), atol=1e-9)
    assert [m.e_hat for m in cr.family("transverse_1")][0] == (1.0, 0.0, 0.0)
    assert [m.family for m in cr.modes] == ["axial"] * n + ["transverse_1"] * n + ["transverse_2"] * n
    assert [m.index for m in cr.modes] == list(range(n)) * 3
    assert cr.mode_index("transverse_1", n - 1) == 2 * n - 1


def test_axis_angle_rotates_the_transverse_families() -> None:
    yb = species("171Yb+")
    th = 0.3
    cr = solve_crystal(dataclasses.replace(secular_trap((3e6, 2.9e6, 1e6)), axis_angle_rad=th), (yb, yb))
    assert cr.family("transverse_1")[0].e_hat == pytest.approx((math.cos(th), math.sin(th), 0.0))
    assert cr.family("transverse_2")[0].e_hat == pytest.approx((-math.sin(th), math.cos(th), 0.0))
    assert cr.collinear()


def test_zigzag_refusal_at_the_marquet_threshold() -> None:
    """(omega_r/omega_z)_crit = 1 at N = 2 and sqrt(12/5) = 1.5492 at N = 3."""
    ca = species("40Ca+")
    with pytest.raises(ZigzagError):
        solve_crystal(secular_trap((1.5e6, 1.6e6, 1.0e6)), (ca,) * 3)
    assert len(solve_crystal(secular_trap((1.6e6, 1.7e6, 1.0e6)), (ca,) * 3).modes) == 9
    with pytest.raises(ZigzagError):
        solve_crystal(secular_trap((0.99e6, 1.5e6, 1.0e6)), (ca, ca))
    solve_crystal(secular_trap((1.01e6, 1.5e6, 1.0e6)), (ca, ca))


# ---- mixed species ----------------------------------------------------------------------------------------------------


def test_two_ion_mixed_axial_closed_form_and_hessian_agree() -> None:
    """9Be+/24Mg+ with isotope masses: omega^2/omega_z1^2 = (0.500847, 2.250636), sum 2(1 + 1/mu); the mass-weighted
    eigenvectors are orthonormal."""
    mu = 23.985042 / 9.0121822
    lo, hi = _two_ion_axial_squared(mu)
    assert (lo, hi) == pytest.approx((0.500847, 2.250636), abs=1e-6)
    assert lo + hi == pytest.approx(2.751484, abs=1e-6)
    cr = _mixed_pair(9.0121822, 23.985042)
    axial = cr.family("axial")
    assert [(m.omega_hz / 1e6) ** 2 for m in axial] == pytest.approx([lo, hi], rel=1e-10)
    c = np.column_stack([m.eigenvector for m in axial])
    assert np.allclose(c.T @ c, np.eye(2), atol=1e-12), "orthonormal in the mass-weighted metric"
    b_phys = c / np.sqrt(cr.masses_kg)[:, None]
    assert np.allclose(np.sum(cr.masses_kg[:, None] * b_phys**2, axis=1), 1.0)


def test_wubbena_rows_with_the_raw_displacement_negative_control() -> None:
    """mu = 0.675: (1.0828427249, 1.9468987123, b1^2 = 0.6839213464); mu = 40/27: (0.8896460800, 1.5995405129, 0.3160786536);
    renormalizing in raw displacements gives 0.40641 instead of 0.31608 at mu = 40/27."""
    for mu, row in (
        (0.675, (1.0828427249, 1.9468987123, 0.6839213464)),
        (40.0 / 27.0, (0.8896460800, 1.5995405129, 0.3160786536)),
    ):
        axial = _mixed_pair(20.0, 20.0 * mu).family("axial")
        assert [m.omega_hz / 1e6 for m in axial] == pytest.approx(row[:2], abs=1e-9)
        assert axial[0].eigenvector[0] ** 2 == pytest.approx(row[2], abs=1e-9)
        assert [math.sqrt(x) for x in _two_ion_axial_squared(mu)] == pytest.approx(row[:2], abs=1e-9)
    cr = _mixed_pair(20.0, 20.0 * 40.0 / 27.0)
    raw = cr.family("axial")[0].eigenvector / np.sqrt(cr.masses_kg)
    raw /= np.linalg.norm(raw)
    assert raw[0] ** 2 == pytest.approx(0.40641, abs=1e-5)
    assert _mixed_pair(20.0, 20.0).family("axial")[0].eigenvector[0] ** 2 == pytest.approx(0.5, abs=1e-12)


def test_home_2013_table_i_end_to_end() -> None:
    """Be+ (12.26, 11.19, 2.69) and Mg+ (4.82, 3.72, 1.65) MHz: x 12.11/4.67, y 11.03/3.53, z 4.04/1.90 MHz; axial eigenvectors
    (-0.926, 0.379) and (0.379, 0.926); off-species radial amplitudes 0.018 (x) and 0.020 (y), the caption's transposed
    order 0.017 and 0.022 (negative control); separation 4.76 um; eta_Be = 0.1844 on the 1.90 MHz mode at 313 nm."""
    be, mg = MassOnly(9.0121822), MassOnly(23.985042)
    w_be = TWO_PI * np.array([12.26, 11.19, 2.69]) * 1e6
    w_mg = TWO_PI * np.array([4.82, 3.72, 1.65]) * 1e6
    cr = build_crystal((be, mg), np.vstack([w_be, w_mg]))
    assert (cr.positions_m[1, 2] - cr.positions_m[0, 2]) * 1e6 == pytest.approx(4.76, abs=5e-3)
    f = {fam: [m.omega_hz / 1e6 for m in cr.family(fam)] for fam in ("axial", "transverse_1", "transverse_2")}
    assert f["axial"] == pytest.approx([1.90, 4.04], abs=6e-3)
    assert f["transverse_1"][1] == pytest.approx(12.11, abs=6e-3)
    assert f["transverse_2"][1] == pytest.approx(11.03, abs=6e-3)
    # no assertion tighter than 0.01 MHz on the Mg-dominated radial pair (4.67 against 4.68 between Home's Tables I and II)
    assert f["transverse_1"][0] == pytest.approx(4.67, abs=1e-2)
    assert f["transverse_2"][0] == pytest.approx(3.53, abs=1e-2)
    lo, hi = cr.family("axial")
    assert np.abs(lo.eigenvector) == pytest.approx([0.379, 0.926], abs=1e-3)
    assert np.abs(hi.eigenvector) == pytest.approx([0.926, 0.379], abs=1e-3)
    assert lo.eigenvector[0] * lo.eigenvector[1] > 0 > hi.eigenvector[0] * hi.eigenvector[1], (
        "in-phase is the LOWER"
    )
    assert abs(cr.family("transverse_1")[1].eigenvector[1]) == pytest.approx(0.018, abs=5e-4)
    assert abs(cr.family("transverse_2")[1].eigenvector[1]) == pytest.approx(0.020, abs=5e-4)
    swapped = build_crystal((be, mg), np.vstack([w_be, TWO_PI * np.array([3.72, 4.82, 1.65]) * 1e6]))
    assert abs(swapped.family("transverse_1")[1].eigenvector[1]) == pytest.approx(0.017, abs=5e-4)
    assert abs(swapped.family("transverse_2")[1].eigenvector[1]) == pytest.approx(0.022, abs=5e-4)
    dk_z = np.array([0.0, 0.0, math.sqrt(2.0) * TWO_PI / 313e-9])  # a 90 degree Raman crossing
    assert abs(cr.lamb_dicke(0, cr.mode_index("axial", 0), dk_z, micromotion=None)) == pytest.approx(
        0.1844, abs=5e-4
    )
    dk_x = np.array([math.sqrt(2.0) * TWO_PI / 313e-9, 0.0, 0.0])
    kx = cr.mode_index("transverse_1", 1)
    assert abs(
        cr.lamb_dicke(0, kx, dk_x, micromotion=None) / cr.lamb_dicke(1, kx, dk_x, micromotion=None)
    ) == (pytest.approx(91.0, rel=0.02))


def test_the_exact_route_approaches_homes_pseudopotential_scaling_and_keeps_the_radial_order() -> None:
    """Be+ at [9.7, 12.9, 4.6] MHz gives Mg+ [1.52, 5.43, 2.82] MHz in the pseudopotential limit (Home 2013 Eqs. 6-19); the
    exact-exponent route at a 1 GHz rf reaches it and preserves sign(omega_x^2 - omega_y^2) across species."""
    trap = dataclasses.replace(secular_trap((9.7e6, 12.9e6, 4.6e6)), rf=RfDrive(0.0, 1.0e9))
    exact, _axes, _field = trap.single_ion_frequencies_rad_s((MassOnly(9.0121822), MassOnly(23.985042)))
    assert exact[1] / TWO_PI / 1e6 == pytest.approx([1.52, 5.43, 2.82], abs=1e-2)
    assert exact[1][0] < exact[1][1]


def test_radial_in_phase_mode_is_the_upper_one_and_axial_the_lower() -> None:
    cr = _mixed_pair(20.0, 24.0, transverse_factor=3.0)
    for fam in ("transverse_1", "transverse_2"):
        lo, hi = cr.family(fam)
        assert hi.eigenvector[0] * hi.eigenvector[1] > 0 > lo.eigenvector[0] * lo.eigenvector[1]
    lo, hi = cr.family("axial")
    assert lo.eigenvector[0] * lo.eigenvector[1] > 0


def test_kielpinski_three_ion_axial_modes_and_parity() -> None:
    """zeta_{1,3} = [13/10 + (21 -+ sqrt(441 - 34 mu + 169 mu^2))/(10 mu)]^{1/2}, zeta_2 = sqrt 3 with the impurity at rest
    (Kielpinski 2000 Eqs. 11-16): 0.7811, 1.7321, 1.8881 at mu = 24/9 and 1.1093, 1.7321, 3.5454 at 9/24."""
    for mu, printed in ((24 / 9, (0.7811, 1.7321, 1.8881)), (9 / 24, (1.1093, 1.7321, 3.5454))):
        disc = math.sqrt(441.0 - 34.0 * mu + 169.0 * mu * mu)
        closed = (
            math.sqrt(1.3 + (21.0 - disc) / (10.0 * mu)),
            math.sqrt(3.0),
            math.sqrt(1.3 + (21.0 + disc) / (10.0 * mu)),
        )
        assert closed == pytest.approx(printed, abs=6e-5)
        w1 = TWO_PI * 10e6
        w_out = np.array([10 * w1, 10 * w1, w1])
        cr = build_crystal(
            (MassOnly(9.0), MassOnly(9.0 * mu), MassOnly(9.0)),
            np.vstack([w_out, w_out / math.sqrt(mu), w_out]),
        )
        assert np.array([m.omega_hz for m in cr.family("axial")]) / 10e6 == pytest.approx(closed, abs=1e-6)
        assert abs(cr.family("axial")[1].eigenvector[1]) < 1e-10, "zero impurity amplitude"
        assert cr.uniform_field_weight(cr.mode_index("axial", 1)) < 1e-20 * cr.uniform_field_weight(
            cr.mode_index("axial", 0)
        )


def test_morigi_walther_in_mg_anchors_up_to_a_per_mode_sign() -> None:
    """115In+/25Mg+ at mu = 4.6: beta'- = (0.26543, 0.96413), beta'+ = (0.96413, -0.26543), Omega+/Omega- = 2.63550; with
    eta_Mg+ = 0.5: eta_Mg- = 0.22347, eta_In- = 0.37846, eta_In+ = -0.06418 (published 0.22, 0.38, -0.06)."""
    cr = _mixed_pair(25.0, 115.0)
    lo, hi = cr.family("axial")
    assert hi.omega_hz / lo.omega_hz == pytest.approx(2.63550, abs=1e-5)
    assert np.abs(lo.eigenvector) == pytest.approx([0.26543, 0.96413], abs=1e-5)
    assert np.abs(hi.eigenvector) == pytest.approx([0.96413, 0.26543], abs=1e-5)
    assert hi.eigenvector[0] * hi.eigenvector[1] < 0
    dk = np.array([0.0, 0.0, 1.0])
    eta = np.array([[cr.lamb_dicke(i, k, dk, micromotion=None) for k in (0, 1)] for i in (0, 1)])
    eta *= 0.5 / eta[0, 1]
    assert abs(eta[0, 0]) == pytest.approx(0.22347, abs=1e-5)
    assert abs(eta[1, 0]) == pytest.approx(0.37846, abs=1e-5)
    assert eta[1, 1] == pytest.approx(-0.06418, abs=1e-5)
    assert np.sign(eta[0, 0]) == np.sign(eta[1, 0]), "the relative sign within a mode is physical"


# ---- Lamb-Dicke parameters ----------------------------------------------------------------------------------------


def test_lamb_dicke_anchor_171yb_355_nm_counter_propagating() -> None:
    """|Delta k| = 3.5398227e7 1/m on the 3.045 MHz COM mode of N = 1, 5, 17 171Yb+ ions: 0.1103011324, 0.0493281660,
    0.0267519541 (ion mass 170.93578 u)."""
    yb = species("171Yb+")
    for n, expected in ((1, 0.1103011324), (5, 0.0493281660), (17, 0.0267519541)):
        cr = solve_crystal(secular_trap((3.045e6, 3.2e6, 0.2e6)), (yb,) * n)
        eta = cr.lamb_dicke(
            0, cr.mode_index("transverse_1", n - 1), np.array([3.5398227e7, 0.0, 0.0]), micromotion=None
        )
        # the table mass differs from the row's rounded 170.93578 u by 9e-9, moving eta by 5e-10 (eta ~ 1/sqrt m)
        assert eta == pytest.approx(expected, abs=1e-9)
        assert eta * math.sqrt(yb.mass_u / 170.93578) == pytest.approx(expected, abs=3e-10)
    with pytest.raises(IndexError):
        cr.lamb_dicke(0, 99, np.zeros(3), micromotion=None)


def test_c0_multiplies_eta_exactly_once_and_only_along_its_axis() -> None:
    """eta(q = 0.3)/eta(q = 0) = C0 = 1.018161 = 1 + 3q^2/16 + O(q^4); axial q = 0 gives 1; a heavier species sees a and q
    scaled by m_ref/m_i."""
    yb = species("171Yb+")
    cr = solve_crystal(secular_trap((3.0e6, 2.9e6, 1.0e6)), (yb, yb))
    q_mat = np.diag([0.3, -0.3, 0.0])
    params = MathieuParameters(
        np.zeros((3, 3)),
        q_mat,
        (0.2, 0.2, 0.05),
        (3e6, 2.9e6, 1e6),
        (c0_wronskian(0.0, 0.3), c0_wronskian(0.0, 0.3), 1.0),
    )
    dk = np.array([1e7, 0.0, 0.0])
    kx = cr.mode_index("transverse_1", 1)
    ratio = cr.lamb_dicke(0, kx, dk, micromotion=params) / cr.lamb_dicke(0, kx, dk, micromotion=None)
    assert ratio == pytest.approx(1.018161, abs=1e-6)
    assert abs(ratio - (1 + 3 * 0.3**2 / 16)) < 2e-3
    dk_z = np.array([0.0, 0.0, 1e7])
    kz = cr.mode_index("axial", 0)
    assert cr.lamb_dicke(0, kz, dk_z, micromotion=params) == cr.lamb_dicke(0, kz, dk_z, micromotion=None)
    ca = species("40Ca+")
    scaled = MathieuParameters(
        np.zeros((3, 3)),
        q_mat,
        (0.2, 0.2, 0.05),
        (3e6, 2.9e6, 1e6),
        (1.0, 1.0, 1.0),
        mass_kg=ca.mass_u * ATOMIC_MASS_KG,
    )
    assert cr.c0_for(0, kx, scaled) == pytest.approx(
        c0_wronskian(0.0, 0.3 * ca.mass_u / yb.mass_u), rel=1e-12
    )


def test_mode_and_crystal_records_enforce_their_invariants() -> None:
    yb = species("171Yb+")
    with pytest.raises(ValueError, match="sign gauge"):
        Mode("axial", 0, 1e6, (0.0, 0.0, 1.0), np.array([0.6, -0.8]))
    with pytest.raises(ValueError, match="unit norm"):
        Mode("axial", 0, 1e6, (0.0, 0.0, 1.0), np.array([0.5, 0.5]))
    good = Mode("axial", 0, 1e6, (0.0, 0.0, 1.0), np.array([-0.6, 0.8]))
    with pytest.raises(ValueError, match="ordered"):
        Crystal(
            (yb, yb),
            np.zeros((2, 3)),
            (Mode("transverse_1", 0, 1e6, (1.0, 0.0, 0.0), np.array([-0.6, 0.8])), good) + (good,) * 4,
        )
    with pytest.raises(ValueError):
        Crystal((yb,), np.zeros((1, 3)), (good,))
    # N modes per family: the 3N count alone would accept 4/1/1
    ax = [Mode("axial", i, 1e6 + i, (0.0, 0.0, 1.0), np.array([-0.6, 0.8])) for i in range(4)]
    t1 = Mode("transverse_1", 0, 3e6, (1.0, 0.0, 0.0), np.array([-0.6, 0.8]))
    t2 = Mode("transverse_2", 0, 3.1e6, (0.0, 1.0, 0.0), np.array([-0.6, 0.8]))
    with pytest.raises(ValueError, match="exactly N = 2 modes"):
        Crystal((yb, yb), np.zeros((2, 3)), (*ax, t1, t2))


def test_the_dimensionless_axial_hessian_is_symmetric_with_the_uniform_mu_1_eigenvector() -> None:
    a = axial_hessian_dimensionless(equilibrium_dimensionless(4))
    assert np.allclose(a, a.T)
    assert np.allclose(a @ np.ones(4), np.ones(4))
