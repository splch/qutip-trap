"""Section 9.1 (Equilibrium positions, Spacing, Axial modes, Transverse modes, Infinite chain, Zigzag threshold) and the
Section 9.12, 9.13, 9.16 and 9.17 crystal, mixed-species and Lamb-Dicke targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.special import zeta

from qutip_trap.species import species
from qutip_trap.trap.crystal import (
    Crystal,
    Mode,
    ZigzagError,
    alpha_critical,
    axial_hessian_dimensionless,
    axial_modes_dimensionless,
    build_crystal,
    equilibrium_dimensionless,
    infinite_chain_epsilon,
    infinite_chain_transverse_omega_rad_s,
    infinite_chain_zigzag_omega_r_rad_s,
    james_coupling,
    kielpinski_three_ion_axial,
    length_scale_m,
    solve_crystal,
    transverse_eigenvalues,
    two_ion_mixed_axial_squared,
    zigzag_ratio_critical,
)
from qutip_trap.trap.mathieu import MathieuParameters, c0_wronskian
from qutip_trap.trap.model import Trap
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI


@dataclass(frozen=True)
class _Mass:
    """A stand-in for ``Species`` carrying only the mass, for fixtures quoted with isotope (atomic) masses."""

    mass_u: float


def _explicit(omega_hz: tuple[float, float, float]) -> Trap:
    return Trap(
        omega_hz=omega_hz,
        axis_angle_rad=0.0,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )


def _mixed_pair(
    m1_u: float, m2_u: float, omega_z1_hz: float = 1e6, transverse_factor: float = 10.0
) -> Crystal:
    """Two ions with the same dc spring (equal charges): omega_2 = omega_1/sqrt(mu) along every axis."""
    mu = m2_u / m1_u
    w1 = TWO_PI * omega_z1_hz
    w = np.array(
        [
            [transverse_factor * w1, transverse_factor * w1, w1],
            [
                transverse_factor * w1 / math.sqrt(mu),
                transverse_factor * w1 / math.sqrt(mu),
                w1 / math.sqrt(mu),
            ],
        ]
    )
    return build_crystal((_Mass(m1_u), _Mass(m2_u)), w)  # type: ignore[arg-type]


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
    assert np.allclose(james_coupling(b, mu)[:, 0], 1.0), "s = 1 on the COM mode"


def test_james_table_2_third_eigenvalue_and_zigzag_eigenvalue() -> None:
    """mu_3 = 29/5 at N = 3 and 5.81, 5.818, 5.824, 5.829, 5.834, 5.838, 5.841 for N = 4 to 10; mu_N of Section 9.12 row 51."""
    mu3 = {n: axial_modes_dimensionless(equilibrium_dimensionless(n))[0][2] for n in range(3, 11)}
    assert mu3[3] == pytest.approx(5.8, abs=1e-10)
    assert round(mu3[4], 2) == 5.81
    for n, val in zip(range(5, 11), (5.818, 5.824, 5.829, 5.834, 5.838, 5.841)):
        assert round(mu3[n], 3) == val
    for n, mu_n in zip(
        (2, 3, 5, 10, 15, 17, 30), (3.0000, 5.8000, 13.4748, 43.2405, 86.5334, 107.3793, 288.6657)
    ):
        assert axial_modes_dimensionless(equilibrium_dimensionless(n))[0][-1] == pytest.approx(mu_n, abs=6e-5)


def test_zigzag_thresholds_pinned_by_section_9_1_and_9_12() -> None:
    assert alpha_critical(2) == pytest.approx(1.0, abs=1e-12)
    assert alpha_critical(3) == pytest.approx(5.0 / 12.0, abs=1e-12)
    assert zigzag_ratio_critical(3) == pytest.approx(math.sqrt(12.0 / 5.0), abs=1e-10)
    for n, ratio in zip((2, 3, 5, 10, 15, 17, 30), (1.0000, 1.5492, 2.4975, 4.5957, 6.5396, 7.2931, 11.9930)):
        assert zigzag_ratio_critical(n) == pytest.approx(ratio, abs=6e-5)
    # the fit 0.73 N^0.86 is the negative control: 13 to 32% off
    for n in (2, 10, 30):
        assert abs(0.73 * n**0.86 / zigzag_ratio_critical(n) - 1.0) > 0.12


def test_spacing_scale_171yb_at_1_mhz() -> None:
    yb = species("171Yb+")
    s = length_scale_m(yb.mass_u * ATOMIC_MASS_KG, TWO_PI * 1e6)
    s2 = 2 ** (1 / 3) * s
    # Section 9.1 prints 3.4 um (two digits, a [background] figure); the value the mass table and the James length
    # scale give is 3.4532 um, and that is what regresses (a +-1.5 % band would not notice a wrong 2^(1/3))
    assert s2 == pytest.approx(3.4532e-6, rel=1e-4)
    assert 3.4e-6 < s2 < 3.5e-6, "Section 9.1: 171Yb+ at 1 MHz gives 3.4 um"
    cr = solve_crystal(_explicit((3e6, 3e6, 1e6)), (yb, yb))
    assert cr.positions_m[1, 2] - cr.positions_m[0, 2] == pytest.approx(s2, rel=1e-12)
    cr3 = solve_crystal(_explicit((3e6, 3e6, 1e6)), (yb, yb, yb))
    assert cr3.positions_m[2, 2] == pytest.approx((5 / 4) ** (1 / 3) * s, rel=1e-12)


def test_transverse_families_share_eigenvectors_and_reverse_the_ordering() -> None:
    """gamma_p = 1/alpha + 1/2 - mu_p/2 per family; COM highest; 3N modes in canonical order (Sections 4.1.3, 9.17)."""
    yb = species("171Yb+")
    n = 5
    wx, wy, wz = 3.0e6, 2.6e6, 0.4e6
    cr = solve_crystal(_explicit((wx, wy, wz)), (yb,) * n)
    assert len(cr.modes) == 3 * n
    mu, b = axial_modes_dimensionless(equilibrium_dimensionless(n))
    axial = cr.family("axial")
    assert [m.omega_hz for m in axial] == pytest.approx(list(wz * np.sqrt(mu)), rel=1e-10)
    for fam, w_r in (("transverse_1", wx), ("transverse_2", wy)):
        gamma = transverse_eigenvalues(mu, (wz / w_r) ** 2)
        expected = sorted(wz * np.sqrt(gamma))
        got = [m.omega_hz for m in cr.family(fam)]
        assert got == pytest.approx(expected, rel=1e-10)
        assert got[-1] == pytest.approx(w_r, rel=1e-12), (
            "the centre-of-mass mode is the highest transverse mode"
        )
        com = cr.family(fam)[-1]
        assert np.allclose(com.eigenvector, 1.0 / math.sqrt(n))
        # the two families and the axial family share A's eigenvectors (reversed order within the family)
        for k, m in enumerate(cr.family(fam)):
            assert np.allclose(np.abs(m.eigenvector), np.abs(b[:, n - 1 - k]), atol=1e-9)
    assert [m.e_hat for m in cr.family("transverse_1")][0] == (1.0, 0.0, 0.0)
    assert [m.family for m in cr.modes] == ["axial"] * n + ["transverse_1"] * n + ["transverse_2"] * n
    assert [m.index for m in cr.modes] == list(range(n)) * 3
    assert cr.mode_index("transverse_1", n - 1) == 2 * n - 1


def test_axis_angle_rotates_the_transverse_families() -> None:
    yb = species("171Yb+")
    th = 0.3
    trap = Trap(
        omega_hz=(3e6, 2.9e6, 1e6),
        axis_angle_rad=th,
        rf=None,
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    cr = solve_crystal(trap, (yb, yb))
    assert cr.family("transverse_1")[0].e_hat == pytest.approx((math.cos(th), math.sin(th), 0.0))
    assert cr.family("transverse_2")[0].e_hat == pytest.approx((-math.sin(th), math.cos(th), 0.0))
    assert cr.collinear()


def test_zigzag_refusal_at_the_marquet_threshold() -> None:
    ca = species("40Ca+")
    with pytest.raises(ZigzagError):
        solve_crystal(_explicit((1.5e6, 1.6e6, 1.0e6)), (ca,) * 3)  # 1.5 < 1.5492
    cr = solve_crystal(_explicit((1.6e6, 1.7e6, 1.0e6)), (ca,) * 3)
    assert len(cr.modes) == 9
    with pytest.raises(ZigzagError):
        solve_crystal(_explicit((0.99e6, 1.5e6, 1.0e6)), (ca, ca))
    solve_crystal(_explicit((1.01e6, 1.5e6, 1.0e6)), (ca, ca))


def test_infinite_chain_force_balance_and_dispersion() -> None:
    """omega_r^2 = (7 zeta(3)/(8 pi eps0)) e^2/(m s_c^3): 7.806 MHz for 9Be+ (m = 9 u, Wineland's example) at s_c = 3 um;
    Landsman's dispersion reaches omega_r at kappa d = 0 and zero at kappa d = pi when eps = 2/(7 zeta(3))."""
    assert infinite_chain_zigzag_omega_r_rad_s(9.0 * ATOMIC_MASS_KG, 3e-6) / TWO_PI == pytest.approx(
        7.806e6, rel=2e-4
    )
    # with the 9Be+ isotope mass 9.0121822 u the same formula gives 7.8006 MHz (a 0.07% mass effect; the plan's row uses 9 u)
    assert infinite_chain_zigzag_omega_r_rad_s(9.0121822 * ATOMIC_MASS_KG, 3e-6) / TWO_PI == pytest.approx(
        7.8006e6, rel=2e-4
    )
    m, d = 9.0 * ATOMIC_MASS_KG, 3e-6
    w_c = infinite_chain_zigzag_omega_r_rad_s(m, d)
    eps_c = infinite_chain_epsilon(m, d, w_c)
    assert eps_c == pytest.approx(2.0 / (7.0 * float(zeta(3.0))), rel=1e-12)
    assert infinite_chain_transverse_omega_rad_s(0.0, w_c, eps_c)[0] == pytest.approx(w_c, rel=1e-9)
    assert infinite_chain_transverse_omega_rad_s(math.pi, w_c, eps_c)[0] == pytest.approx(0.0, abs=1e-3 * w_c)
    # at twice the critical radial frequency the zigzag mode sits at sqrt(1 - 1/4) omega_r
    assert infinite_chain_transverse_omega_rad_s(math.pi, 2 * w_c, infinite_chain_epsilon(m, d, 2 * w_c))[
        0
    ] == pytest.approx(2 * w_c * math.sqrt(0.75), rel=1e-6)


# ---- mixed species (Section 4.1.7; 9.13 rows 23-26, 43-49) ---------------------------------------------------------------


def test_two_ion_mixed_axial_closed_form_and_hessian_agree() -> None:
    mu = 23.985042 / 9.0121822
    lo, hi = two_ion_mixed_axial_squared(mu)
    assert (lo, hi) == pytest.approx((0.500847, 2.250636), abs=1e-6)
    assert lo + hi == pytest.approx(2.0 * (1.0 + 1.0 / mu), rel=1e-12) and lo + hi == pytest.approx(
        2.751484, abs=1e-6
    )
    assert two_ion_mixed_axial_squared(24.305 / 9.0121822) == pytest.approx((0.495185, 2.246406), abs=1e-6), (
        "the natural-abundance mass a fixture must not hide"
    )
    assert two_ion_mixed_axial_squared(1.0) == pytest.approx((1.0, 3.0), rel=1e-12)
    cr = _mixed_pair(9.0121822, 23.985042)
    axial = cr.family("axial")
    assert [(m.omega_hz / 1e6) ** 2 for m in axial] == pytest.approx([lo, hi], rel=1e-10)
    c = np.column_stack([m.eigenvector for m in axial])
    assert np.allclose(c.T @ c, np.eye(2), atol=1e-12), "orthonormal in the mass-weighted metric"
    b_phys = c / np.sqrt(cr.masses_kg)[:, None]
    assert np.allclose(np.sum(cr.masses_kg[:, None] * b_phys**2, axis=1), 1.0), (
        "sum_k m_i (c_i/sqrt m_i)^2 = 1"
    )


def test_wubbena_rows_of_section_9_13_with_the_raw_displacement_negative_control() -> None:
    """mu = 0.675: (1.0828427249, 1.9468987123, b1^2 = 0.6839213464); mu = 40/27: (0.8896460800, 1.5995405129, 0.3160786536);
    renormalizing in raw displacements gives 0.40641 instead of 0.31608 at mu = 40/27."""
    for mu, row in (
        (0.675, (1.0828427249, 1.9468987123, 0.6839213464)),
        (40.0 / 27.0, (0.8896460800, 1.5995405129, 0.3160786536)),
    ):
        cr = _mixed_pair(20.0, 20.0 * mu)
        axial = cr.family("axial")
        assert [m.omega_hz / 1e6 for m in axial] == pytest.approx(row[:2], abs=1e-9)
        assert axial[0].eigenvector[0] ** 2 == pytest.approx(row[2], abs=1e-9)
        assert [math.sqrt(x) for x in two_ion_mixed_axial_squared(mu)] == pytest.approx(row[:2], abs=1e-9)
    cr = _mixed_pair(20.0, 20.0 * 40.0 / 27.0)
    raw = cr.family("axial")[0].eigenvector / np.sqrt(cr.masses_kg)
    raw /= np.linalg.norm(raw)
    assert raw[0] ** 2 == pytest.approx(0.40641, abs=1e-5)
    assert _mixed_pair(20.0, 20.0).family("axial")[0].eigenvector[0] ** 2 == pytest.approx(0.5, abs=1e-12), (
        "vanishes identically at mu = 1"
    )


def test_home_2013_table_i_end_to_end() -> None:
    """Be+ (12.26, 11.19, 2.69) and Mg+ (4.82, 3.72, 1.65) MHz: x 12.11/4.67, y 11.03/3.53, z 4.04/1.90 MHz; axial eigenvectors
    (-0.926, 0.379) and (0.379, 0.926); off-species radial amplitudes 0.018 (x) and 0.020 (y); the caption's transposed order
    gives 0.017 and 0.022 (negative control); separation 4.76 um."""
    be, mg = _Mass(9.0121822), _Mass(23.985042)
    w_be = TWO_PI * np.array([12.26, 11.19, 2.69]) * 1e6
    w_mg = TWO_PI * np.array([4.82, 3.72, 1.65]) * 1e6
    cr = build_crystal((be, mg), np.vstack([w_be, w_mg]))  # type: ignore[arg-type]
    assert (cr.positions_m[1, 2] - cr.positions_m[0, 2]) * 1e6 == pytest.approx(4.76, abs=5e-3)
    f = {fam: [m.omega_hz / 1e6 for m in cr.family(fam)] for fam in ("axial", "transverse_1", "transverse_2")}
    assert f["axial"] == pytest.approx([1.90, 4.04], abs=6e-3)
    assert f["transverse_1"][1] == pytest.approx(12.11, abs=6e-3)
    assert f["transverse_2"][1] == pytest.approx(11.03, abs=6e-3)
    # the 9.13 row's own instruction: "no assertion tighter than 0.01 MHz on the Mg-dominated radial pair
    # (4.67 versus 4.68 between his Tables I and II)"
    assert f["transverse_1"][0] == pytest.approx(4.67, abs=1e-2)
    assert f["transverse_2"][0] == pytest.approx(3.53, abs=1e-2)
    lo, hi = cr.family("axial")
    assert np.abs(lo.eigenvector) == pytest.approx([0.379, 0.926], abs=1e-3)
    assert np.abs(hi.eigenvector) == pytest.approx([0.926, 0.379], abs=1e-3)
    assert lo.eigenvector[0] * lo.eigenvector[1] > 0 > hi.eigenvector[0] * hi.eigenvector[1], (
        "in-phase is the LOWER axial mode"
    )
    assert abs(cr.family("transverse_1")[1].eigenvector[1]) == pytest.approx(0.018, abs=5e-4)
    assert abs(cr.family("transverse_2")[1].eigenvector[1]) == pytest.approx(0.020, abs=5e-4)
    swapped = build_crystal((be, mg), np.vstack([w_be, TWO_PI * np.array([3.72, 4.82, 1.65]) * 1e6]))  # type: ignore[arg-type]
    assert abs(swapped.family("transverse_1")[1].eigenvector[1]) == pytest.approx(0.017, abs=5e-4)
    assert abs(swapped.family("transverse_2")[1].eigenvector[1]) == pytest.approx(0.022, abs=5e-4)
    # eta_Be on the 1.90 MHz mode at 313 nm with a 90 degree Raman crossing = 0.1844 (paper 0.18); eta_Be/eta_Mg = 91 on the 12.11 MHz x mode
    dk_z = np.array([0.0, 0.0, math.sqrt(2.0) * TWO_PI / 313e-9])
    assert abs(cr.lamb_dicke(0, cr.mode_index("axial", 0), dk_z, micromotion=None)) == pytest.approx(
        0.1844, abs=5e-4
    )
    dk_x = np.array([math.sqrt(2.0) * TWO_PI / 313e-9, 0.0, 0.0])
    kx = cr.mode_index("transverse_1", 1)
    assert abs(
        cr.lamb_dicke(0, kx, dk_x, micromotion=None) / cr.lamb_dicke(1, kx, dk_x, micromotion=None)
    ) == pytest.approx(91.0, rel=0.02)


def test_home_trap_inversion_scales_the_frequencies_and_preserves_the_radial_order() -> None:
    """Section 9.13, Home row: omega_i^2(m) = P_i/m^2 + S_i/m with P_x = P_y, P_z = 0 preserves sign(omega_x^2 - omega_y^2)
    across species; Be [9.7, 12.9, 4.6] MHz gives Mg [1.52, 5.43, 2.82] MHz."""
    from qutip_trap.trap.pseudopotential import (
        RfDrive,
        pseudopotential_mass_scaling_rad_s,
        trap_inversion_p_s,
    )

    m_be = 9.0121822 * ATOMIC_MASS_KG
    m_mg = 23.985042 * ATOMIC_MASS_KG
    w_be = TWO_PI * np.array([9.7, 12.9, 4.6]) * 1e6
    p, s = trap_inversion_p_s(w_be, m_be)
    assert p[0] == pytest.approx(p[1], rel=1e-15) and p[2] == 0.0, "the rf part is radial and P_x = P_y"
    assert abs(float(np.sum(s))) < 1e-12 * abs(float(s[2])), "Laplace on the dc potential: sum_i S_i = 0"
    assert pseudopotential_mass_scaling_rad_s(w_be, m_be, m_be) == pytest.approx(w_be, rel=1e-12)
    w_mg = pseudopotential_mass_scaling_rad_s(w_be, m_be, m_mg) / TWO_PI / 1e6
    assert w_mg == pytest.approx([1.52, 5.43, 2.82], abs=5e-3)
    # the sign identity, exactly: omega_x^2 - omega_y^2 scales as m_ref/m, so the radial order never inverts
    for mass_u in (9.0121822, 12.0, 20.0, 23.985042, 26.0):
        w = pseudopotential_mass_scaling_rad_s(w_be, m_be, mass_u * ATOMIC_MASS_KG)
        assert (w[0] ** 2 - w[1] ** 2) == pytest.approx(
            (m_be / (mass_u * ATOMIC_MASS_KG)) * (w_be[0] ** 2 - w_be[1] ** 2), rel=1e-12
        )
        assert w[0] < w[1], "sign(omega_x^2 - omega_y^2) is preserved"
    # S_x < 0 here, so the 1/m^2 rf part loses to the 1/m dc part above 3.01 m_Be: the refusal is explicit, not a nan
    with pytest.raises(ValueError, match="does not confine"):
        pseudopotential_mass_scaling_rad_s(w_be, m_be, 40.0 * ATOMIC_MASS_KG)
    # the shipped exact-exponent route is the same inversion plus the O(q^4) Floquet correction: it converges to it
    trap = Trap(
        omega_hz=(9.7e6, 12.9e6, 4.6e6),
        axis_angle_rad=0.0,
        rf=RfDrive(0.0, 1.0e9),
        dc=None,
        geometry=None,
        stray_field_v_per_m=(0.0, 0.0, 0.0),
        shim_voltages_v={},
    )
    exact, _axes, _field = trap.single_ion_frequencies_rad_s((_Mass(9.0121822), _Mass(23.985042)))  # type: ignore[arg-type]
    assert exact[1] / TWO_PI / 1e6 == pytest.approx([1.52, 5.43, 2.82], abs=1e-2)


def test_radial_in_phase_mode_is_the_upper_one_and_axial_the_lower() -> None:
    cr = _mixed_pair(20.0, 24.0, transverse_factor=3.0)
    for fam in ("transverse_1", "transverse_2"):
        lo, hi = cr.family(fam)
        assert hi.eigenvector[0] * hi.eigenvector[1] > 0 > lo.eigenvector[0] * lo.eigenvector[1]
    lo, hi = cr.family("axial")
    assert lo.eigenvector[0] * lo.eigenvector[1] > 0


def test_kielpinski_three_ion_closed_forms_crossing_and_parity() -> None:
    """zeta = 0.7811, 1.7321, 1.8881 at mu = 24/9 and 1.1093, 1.7321, 3.5454 at 9/24; zeta_2 = sqrt 3 for every mu with the
    impurity at rest; zeta_3(17/3) = sqrt 3 exactly; ordering not monotonic (zeta_3 = 1.7034 < zeta_2 at mu = 7.3); gaps at 10 MHz."""
    assert kielpinski_three_ion_axial(24 / 9) == pytest.approx((0.7811, 1.7321, 1.8881), abs=6e-5)
    assert kielpinski_three_ion_axial(9 / 24) == pytest.approx((1.1093, 1.7321, 3.5454), abs=6e-5)
    assert kielpinski_three_ion_axial(17 / 3)[2] == pytest.approx(math.sqrt(3.0), abs=1e-12)
    assert kielpinski_three_ion_axial(7.3)[2] == pytest.approx(1.7034, abs=6e-5)
    for mu in (24 / 9, 9 / 24):
        w1 = TWO_PI * 10e6
        w_out = np.array([10 * w1, 10 * w1, w1])
        w_in = w_out / math.sqrt(mu)
        cr = build_crystal((_Mass(9.0), _Mass(9.0 * mu), _Mass(9.0)), np.vstack([w_out, w_in, w_out]))  # type: ignore[arg-type]
        zetas = np.array([m.omega_hz for m in cr.family("axial")]) / 10e6
        assert zetas == pytest.approx(kielpinski_three_ion_axial(mu), abs=1e-6)
        breathing = cr.family("axial")[1]
        assert abs(breathing.eigenvector[1]) < 1e-10, "zero impurity amplitude"
        assert cr.uniform_field_weight(cr.mode_index("axial", 1)) < 1e-20 * cr.uniform_field_weight(
            cr.mode_index("axial", 0)
        )
    z = kielpinski_three_ion_axial(24 / 9)
    assert (z[2] - z[1]) * 10 == pytest.approx(1.560, abs=1e-3)
    z = kielpinski_three_ion_axial(9 / 24)
    assert (z[1] - z[0]) * 10 == pytest.approx(6.228, abs=1e-3)


def test_morigi_walther_in_mg_anchors_up_to_a_per_mode_sign() -> None:
    """115In+/25Mg+ at mu = 4.6: beta'- = (0.26543, 0.96413), beta'+ = (0.96413, -0.26543), Omega+/Omega- = 2.63550; with
    eta_Mg+ = 0.5: eta_Mg- = 0.22347, eta_In- = 0.37846, eta_In+ = -0.06418 (published 0.22, 0.38, -0.06), per-mode signs free."""
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


# ---- Lamb-Dicke parameters (Section 13, "Lamb-Dicke base"; 9.16 row 13-6; 9.17 "C0 applied once") --------------------


def test_lamb_dicke_anchor_171yb_355_nm_counter_propagating() -> None:
    """|Delta k| = 3.5398227e7 1/m, f = 3.045 MHz, ion mass 170.93578 u: 0.1103011324, 0.0493281660, 0.0267519541 for N = 1, 5, 17."""
    yb = species("171Yb+")
    assert yb.mass_u == pytest.approx(170.93578, abs=1e-5)
    for n, expected in ((1, 0.1103011324), (5, 0.0493281660), (17, 0.0267519541)):
        cr = solve_crystal(_explicit((3.045e6, 3.2e6, 0.2e6)), (yb,) * n)
        com = cr.mode_index("transverse_1", n - 1)
        eta = cr.lamb_dicke(0, com, np.array([3.5398227e7, 0.0, 0.0]), micromotion=None)
        # the table's mass 170.9357816 u differs from the row's rounded 170.93578 u by 9e-9, moving eta by 5e-10 (eta ~ 1/sqrt m)
        assert eta == pytest.approx(expected, abs=1e-9)
        assert eta * math.sqrt(yb.mass_u / 170.93578) == pytest.approx(
            expected, abs=3e-10
        )  # the residual is at the CODATA-edition level
        rec = cr.lamb_dicke_record(0, com, np.array([3.5398227e7, 0.0, 0.0]), micromotion=None)
        assert rec.C0 == 1.0 and rec.mass_kg == pytest.approx(yb.mass_u * ATOMIC_MASS_KG)
        assert rec.x0_m == pytest.approx(math.sqrt(HBAR_J_S / (2 * rec.mass_kg * TWO_PI * 3.045e6)))
        assert rec.projection == pytest.approx(3.5398227e7 / math.sqrt(n))
    with pytest.raises(IndexError):
        cr.lamb_dicke(0, 99, np.zeros(3), micromotion=None)


def test_c0_multiplies_eta_exactly_once_and_only_along_its_axis() -> None:
    """eta(q = 0.3)/eta(q = 0) = C0 = 1.018161 = 1 + 3q^2/16 + O(q^4) (the O(q^4) term is 1.3e-3 at q = 0.3); axial q = 0 gives 1."""
    yb = species("171Yb+")
    cr = solve_crystal(_explicit((3.0e6, 2.9e6, 1.0e6)), (yb, yb))
    params = MathieuParameters(
        np.zeros((3, 3)),
        np.diag([0.3, -0.3, 0.0]),
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
    # a heavier species sees a and q scaled by m_ref/m_i
    ca = species("40Ca+")
    scaled = MathieuParameters(
        np.zeros((3, 3)),
        np.diag([0.3, -0.3, 0.0]),
        (0.2, 0.2, 0.05),
        (3e6, 2.9e6, 1e6),
        (1.0, 1.0, 1.0),
        mass_kg=ca.mass_u * ATOMIC_MASS_KG,
    )
    assert cr.c0_for(0, kx, scaled) == pytest.approx(
        c0_wronskian(0.0, 0.3 * ca.mass_u / yb.mass_u), rel=1e-12
    )


def test_mode_and_crystal_records_enforce_the_appendix_e_invariants() -> None:
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
    # PLAN 4.1.3 declares 3N modes in THREE families, i.e. N per family: the 3N count alone would accept 4/1/1, which
    # is what the non-collinear branch's argmax family assignment could produce for a strongly mixed crystal
    ax = [Mode("axial", i, 1e6 + i, (0.0, 0.0, 1.0), np.array([-0.6, 0.8])) for i in range(4)]
    t1 = Mode("transverse_1", 0, 3e6, (1.0, 0.0, 0.0), np.array([-0.6, 0.8]))
    t2 = Mode("transverse_2", 0, 3.1e6, (0.0, 1.0, 0.0), np.array([-0.6, 0.8]))
    with pytest.raises(ValueError, match="exactly N = 2 modes"):
        Crystal((yb, yb), np.zeros((2, 3)), (*ax, t1, t2))


def test_hessian_and_transverse_eigenvalue_helpers() -> None:
    u = equilibrium_dimensionless(4)
    a = axial_hessian_dimensionless(u)
    assert np.allclose(a, a.T)
    assert np.allclose(a @ np.ones(4), np.ones(4)), "the uniform vector is the mu = 1 eigenvector"
    with pytest.raises(ValueError):
        transverse_eigenvalues(np.array([1.0]), 0.0)
