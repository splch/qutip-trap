"""Section 4.1.6 and 9.13 surface-electrode targets: gapless-plane potentials, House and Nizamani closed forms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.trap.mathieu import beta_exact, beta_preprint, mathieu_parameters
from qutip_trap.trap.pseudopotential import mathieu_matrices, secular_from_hessian_rad_s
from qutip_trap.trap.surface import (
    KAPPA_MAX,
    Electrodes,
    GaplessPlaneTrap,
    depth_optimal_b_over_a,
    five_wire_depth_j,
    five_wire_escape_height_m,
    five_wire_null_height_m,
    minimum_norm_dc_voltages,
    nizamani_kappa,
    nizamani_q,
    rectangle_gradient,
    rectangle_hessian,
    rectangle_potential,
    strip_potential,
    two_rail_null,
)
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI

HOUSE = dict(a=100e-6, b=120e-6, v=300.0, omega=TWO_PI * 50e6, m=1.46e-25)


def _rect(x: float, y: float, z: float) -> float:
    return rectangle_potential(x, y, z, -1.0, 1.0, -0.5, 0.5)


def test_rectangle_potential_laplace_boundary_and_strip_limit() -> None:
    h = 1e-4
    for pt in ((0.2, 0.7, 0.1), (1.5, 0.4, 0.3), (0.0, 2.0, 0.0)):
        x, y, z = pt
        lap = (
            _rect(x + h, y, z)
            + _rect(x - h, y, z)
            + _rect(x, y + h, z)
            + _rect(x, y - h, z)
            + _rect(x, y, z + h)
            + _rect(x, y, z - h)
            - 6 * _rect(x, y, z)
        ) / h**2
        assert abs(lap) < 1e-7
    assert _rect(0.3, 1e-6, 0.2) == pytest.approx(1.0, abs=1e-5)
    assert _rect(1.5, 1e-6, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert rectangle_potential(0.3, 0.8, 0.0, -1, 1, -1e6, 1e6) == pytest.approx(
        strip_potential(0.3, 0.8, -1, 1), abs=1e-8
    )
    assert rectangle_potential(0.0, 1.0, 0.0, -1e6, 1e6, -1e6, 1e6) == pytest.approx(1.0, abs=1e-6)
    assert rectangle_potential(0.2, -0.7, 0.1, -1, 1, -0.5, 0.5) == pytest.approx(-_rect(0.2, 0.7, 0.1)), (
        "odd in the height"
    )


def test_rectangle_gradient_and_hessian_against_finite_differences() -> None:
    r = np.array([0.2, 0.7, 0.1])
    h = 1e-6
    g = rectangle_gradient(*r, -1, 1, -0.5, 0.5)
    for axis in range(3):
        rp, rm = r.copy(), r.copy()
        rp[axis] += h
        rm[axis] -= h
        assert g[axis] == pytest.approx((_rect(*rp) - _rect(*rm)) / (2 * h), abs=1e-9)
    hess = rectangle_hessian(*r, -1, 1, -0.5, 0.5)
    assert abs(np.trace(hess)) < 1e-12
    for axis in range(3):
        rp, rm = r.copy(), r.copy()
        rp[axis] += h
        rm[axis] -= h
        col = (rectangle_gradient(*rp, -1, 1, -0.5, 0.5) - rectangle_gradient(*rm, -1, 1, -0.5, 0.5)) / (
            2 * h
        )
        assert np.allclose(hess[:, axis], col, atol=1e-8)


def test_electrode_record_validation() -> None:
    with pytest.raises(ValueError):
        Electrodes("surface_five_wire", {"a_m": 1e-4})
    with pytest.raises(ValueError):
        Electrodes("surface_general", {})
    with pytest.raises(ValueError):
        Electrodes("surface_general", {}, strips={"e": ((2.0, 1.0),)})
    with pytest.raises(ValueError, match="traceless"):
        Electrodes(
            "surface_five_wire",
            {
                "a_m": 1e-4,
                "b_m": 1e-4,
                "dc_curvature_xx_v_per_m2": 1.0,
                "dc_curvature_yy_v_per_m2": 1.0,
                "dc_curvature_zz_v_per_m2": 1.0,
            },
        )


def test_five_wire_house_fixture_of_section_9_13() -> None:
    """a = 100 um, b = c = 120 um, V_rf = 300 V, Omega = 2 pi x 50 MHz, m = 1.46e-25 kg: x0 = 50.000000 um, y0 = 92.195445 um,
    y_E = 169.655527 um, psi_E = 0.184465 eV, Q11 = -Q22 = 0.2284260, Q12 = 0; 4.038 MHz pseudopotential."""
    trap = GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": HOUSE["a"], "b_m": HOUSE["b"]}))
    null = trap.rf_null()
    assert null[0] * 1e6 == pytest.approx(50.0, abs=1e-6)
    assert null[1] * 1e6 == pytest.approx(92.195445, abs=1e-6)
    assert five_wire_null_height_m(HOUSE["a"], HOUSE["b"]) == pytest.approx(null[1], rel=1e-12)
    assert two_rail_null(HOUSE["a"], HOUSE["b"], HOUSE["b"]) == pytest.approx((null[0], null[1]), rel=1e-12)
    esc = trap.escape_point()
    assert esc[0] == pytest.approx(HOUSE["a"] / 2.0, rel=1e-9)
    assert esc[1] * 1e6 == pytest.approx(169.655527, abs=1e-6)
    assert five_wire_escape_height_m(HOUSE["a"], HOUSE["b"]) == pytest.approx(esc[1], rel=1e-12)
    depth = trap.depth_j(HOUSE["v"], HOUSE["m"], HOUSE["omega"])
    assert depth / E_C == pytest.approx(0.184465, abs=1e-6)
    assert five_wire_depth_j(HOUSE["a"], HOUSE["b"], HOUSE["v"], HOUSE["m"], HOUSE["omega"]) == pytest.approx(
        depth, rel=1e-12
    )
    h_rf = trap.rf_hessian_unit(null)
    _, q_mat = mathieu_matrices(np.zeros((3, 3)), HOUSE["v"] * h_rf, HOUSE["m"], HOUSE["omega"])
    assert q_mat[0, 0] == pytest.approx(0.2284260, abs=1e-7)
    assert q_mat[1, 1] == pytest.approx(-0.2284260, abs=1e-7)
    assert abs(q_mat[0, 1]) < 1e-12 and abs(np.trace(q_mat)) < 1e-12
    q = q_mat[0, 0]
    assert q * HOUSE["omega"] / (2 * math.sqrt(2)) / TWO_PI == pytest.approx(4.038e6, abs=1e3)
    # the exact exponent gives 4.080 MHz; the plan's "4.077 MHz Floquet" is the preprint closed form (documented deviation)
    assert beta_exact(0.0, q) * HOUSE["omega"] / 2 / TWO_PI == pytest.approx(4.0804e6, abs=1e3)
    assert beta_preprint(0.0, q) * HOUSE["omega"] / 2 / TWO_PI == pytest.approx(4.077e6, abs=2e3)
    # House's 11.5 MHz radial frequency is not reproducible (a dropped 1/(2 sqrt 2)): the negative control
    assert q * HOUSE["omega"] / TWO_PI == pytest.approx(11.42e6, rel=1e-2)


def test_escape_radical_coefficient_negative_control() -> None:
    """The coefficient inside House's gapped escape-point radical must read 2: 1.69656 against 1.16494 with 1/2 at a = 1, b = 1.2."""
    a, b = 1.0, 1.2
    s = math.sqrt(2 * a * b + a * a)
    assert five_wire_escape_height_m(a, b) == pytest.approx(1.69656, abs=1e-5)
    assert math.sqrt(2 * a * b + a * a + 0.5 * (a + b) * s) / 2.0 == pytest.approx(1.16494, abs=1e-5)


def test_rf_pseudopotential_isotropic_at_the_null_and_axes_set_by_dc() -> None:
    trap = GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": 60e-6, "b_m": 300e-6, "c_m": 150e-6}))
    null = trap.rf_null()
    h = trap.rf_hessian_unit(null)
    h2 = h @ h  # Hess |E|^2 / 2 at a zero of E
    assert h2[0, 0] == pytest.approx(h2[1, 1], rel=1e-10) and abs(h2[0, 1]) < 1e-10 * h2[0, 0]
    assert h[0, 1] != pytest.approx(0.0, abs=1e-3 * abs(h[0, 0])), (
        "asymmetric rails rotate the rf Hessian, not the axes"
    )
    assert abs(np.trace(h)) < 1e-9 * abs(h[0, 0])


def test_sum_of_basis_functions_is_one_on_the_plane() -> None:
    trap = GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": 1e-4, "b_m": 1.2e-4}))
    for x in (-3e-4, -1e-4, 0.5e-4, 1.5e-4, 5e-4):
        assert trap.sum_of_basis(np.array([x, 1e-9, 0.0])) == pytest.approx(1.0, abs=1e-6)
    # a complete strip cover is the whole plane at 1 V: the sum is 1 everywhere in the half-space
    assert trap.sum_of_basis(np.array([0.5e-4, 1e-3, 0.0])) == pytest.approx(1.0, abs=1e-12)
    partial = GaplessPlaneTrap(Electrodes("surface_general", {}, strips={"rf": ((-1e-4, 0.0), (1e-4, 2e-4))}))
    assert partial.sum_of_basis(np.array([0.5e-4, 1e-3, 0.0])) < 1.0


def test_four_wire_limit_and_null_shift_toward_the_narrower_rail() -> None:
    a = 1e-4
    four = GaplessPlaneTrap(Electrodes("surface_four_wire", {"a_m": a, "c_m": a}))
    assert four.rf_null()[1] == pytest.approx(a, rel=1e-9)
    x0, _ = two_rail_null(1.0, 2.0, 1.0)  # wide right rail b = 2, narrow left rail c = 1
    assert x0 == pytest.approx(1.0 / 3.0) and x0 < 0.5, "the null moves toward the narrower rail"


def test_nizamani_fixture_of_section_9_13() -> None:
    """a = 60, b = 300, c = 150 um, 171 u, Omega = 2 pi x 55 MHz, V_rf = 500 V: q_N = 0.6948, |Q11| q = 0.1976,
    principal-axis q = 0.2171 (ratio 3.2002) giving 4.222 MHz; h = 82.46 um against the quoted 85."""
    a, b, c = 60e-6, 300e-6, 150e-6
    m = 171.0 * ATOMIC_MASS_KG
    omega = TWO_PI * 55e6
    trap = GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": a, "b_m": b, "c_m": c}))
    null = trap.rf_null()
    assert null[1] * 1e6 == pytest.approx(82.46, abs=5e-3)
    assert two_rail_null(a, b, c) == pytest.approx((null[0], null[1]), rel=1e-12)
    q_n = nizamani_q(500.0, m, omega, null[1])
    assert q_n == pytest.approx(0.6948, abs=1e-4)
    _, q_mat = mathieu_matrices(np.zeros((3, 3)), 500.0 * trap.rf_hessian_unit(null), m, omega)
    q_pa = math.hypot(q_mat[0, 0], q_mat[0, 1])
    assert abs(q_mat[0, 0]) == pytest.approx(0.1976, abs=1e-4)
    assert q_pa == pytest.approx(0.2171, abs=1e-4)
    assert q_n / q_pa == pytest.approx(3.2002, abs=1e-3)
    assert q_pa * omega / (2 * math.sqrt(2)) / TWO_PI == pytest.approx(4.222e6, abs=1e3)
    assert nizamani_q(450.0, m, omega, 85e-6) == pytest.approx(0.589, abs=1e-3), (
        "his 450 V at h = 85 um gives 0.589, not 0.7"
    )


def test_depth_optima_and_the_geometry_independent_bound() -> None:
    from scipy.optimize import minimize_scalar

    assert depth_optimal_b_over_a() == pytest.approx(1.1914879, abs=1e-7)
    r_fixed_a = minimize_scalar(
        lambda r: -five_wire_depth_j(1.0, r, 1.0, 1.0, 1.0),
        bounds=(0.5, 5.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert r_fixed_a.x == pytest.approx(depth_optimal_b_over_a(), abs=1e-6)
    r_eq = minimize_scalar(
        lambda r: -nizamani_kappa(1.0, r, r), bounds=(1.0, 10.0), method="bounded", options={"xatol": 1e-10}
    )
    assert r_eq.x == pytest.approx(3.676, abs=1e-3) and -r_eq.fun == pytest.approx(0.022542, abs=1e-6)
    assert -r_eq.fun == pytest.approx(KAPPA_MAX, rel=1e-8)
    r_half = minimize_scalar(
        lambda r: -nizamani_kappa(1.0, r, r / 2.0),
        bounds=(1.0, 10.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert r_half.x == pytest.approx(4.902, abs=1e-3) and -r_half.fun == pytest.approx(0.020038, abs=1e-6)
    assert -r_half.fun / KAPPA_MAX == pytest.approx(8.0 / 9.0, rel=1e-6)
    assert KAPPA_MAX == pytest.approx(1.0 / (2.0 * (11.0 + 5.0 * math.sqrt(5.0))), rel=1e-15)


def test_nizamani_kappa_equals_house_depth_for_equal_rails() -> None:
    for a, b in ((1.0, 1.2), (60e-6, 300e-6), (2.0, 0.7)):
        y0 = two_rail_null(a, b, b)[1]
        house = five_wire_depth_j(a, b, 1.0, 1.0, 1.0)
        niz = E_C**2 * nizamani_kappa(a, b, b) / (math.pi**2 * y0**2)
        assert niz == pytest.approx(house, rel=1e-12)


def test_asymmetric_rails_numerical_saddle_versus_extrapolated_closed_form() -> None:
    """House's b = c depth formula applied to unequal rails is the wrong reference; the module's saddle is numerical."""
    a, b, c = 60e-6, 300e-6, 150e-6
    trap = GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": a, "b_m": b, "c_m": c}))
    numeric = trap.depth_j(1.0, 1.0, 1.0)
    for width in (b, c):
        assert abs(five_wire_depth_j(a, width, 1.0, 1.0, 1.0) / numeric - 1.0) > 0.08


def test_single_ion_equilibrium_under_a_stray_field() -> None:
    trap = GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": HOUSE["a"], "b_m": HOUSE["b"]}))
    null = trap.rf_null()
    a_mat, q_mat = mathieu_matrices(
        np.zeros((3, 3)), HOUSE["v"] * trap.rf_hessian_unit(null), HOUSE["m"], HOUSE["omega"]
    )
    params = mathieu_parameters(a_mat, q_mat, 50e6)
    r_eq = trap.equilibrium(
        HOUSE["v"], HOUSE["m"], HOUSE["omega"], {}, stray_field_v_per_m=np.array([10.0, 0.0, 0.0])
    )
    # the static force is balanced by the pseudopotential curvature, (Omega/2)^2 q^2/2 here, not by the exact exponent
    omega_ps = HOUSE["omega"] / 2 * math.sqrt(q_mat[0, 0] ** 2 / 2)
    expected = 10.0 * E_C / (HOUSE["m"] * omega_ps**2)
    assert (r_eq - null)[0] == pytest.approx(expected, rel=1e-6)
    assert (r_eq - null)[0] != pytest.approx(
        10.0 * E_C / (HOUSE["m"] * params.secular_rad_s()[0] ** 2), rel=1e-2
    )
    assert abs((r_eq - null)[1]) < 1e-3 * expected, "a second-order vertical displacement only"
    # the pseudopotential route and the Mathieu route are one equation: omega^2 = eig(Hess U)/m against (Omega/2) sqrt(eig(A + Q^2/2))
    w, _ = secular_from_hessian_rad_s(
        trap.total_hessian_j_per_m2(null, HOUSE["v"], HOUSE["m"], HOUSE["omega"], {})[:2, :2], HOUSE["m"]
    )
    lowest = HOUSE["omega"] / 2 * np.sqrt(np.linalg.eigvalsh((a_mat + q_mat @ q_mat / 2)[:2, :2]))
    assert np.allclose(w, lowest, rtol=1e-10)


def test_segmented_five_wire_with_rectangles_and_house_eq_30() -> None:
    """Finite rf rails (long rectangles) recover the strip null height; four dc segments realize a pure axial curvature with zero field."""
    a, b, length = 100e-6, 120e-6, 6e-3
    rects = {
        "rf": ((-b, 0.0, -length, length), (a, a + b, -length, length)),
        "dc_l1": ((-b - 200e-6, -b, -600e-6, -150e-6),),
        "dc_l2": ((-b - 200e-6, -b, 150e-6, 600e-6),),
        "dc_r1": ((a + b, a + b + 200e-6, -600e-6, -150e-6),),
        "dc_r2": ((a + b, a + b + 200e-6, 150e-6, 600e-6),),
        "centre": ((0.0, a, -length, length),),
    }
    el = Electrodes("surface_general", {}, rectangles=rects)
    trap = GaplessPlaneTrap(el)
    null = trap.rf_null(np.array([a / 2, 90e-6, 0.0]))
    assert null[1] == pytest.approx(five_wire_null_height_m(a, b), rel=2e-3)
    assert null[0] == pytest.approx(a / 2, abs=1e-9)
    # the four outer segments alone cannot null the vertical field they produce while setting the axial curvature (their
    # E_y and zz rows are proportional by symmetry); the centre electrode is the fifth, independent one
    names = ("dc_l1", "dc_l2", "dc_r1", "dc_r2", "centre")
    target = 2.0e7  # V/m^2
    volts = minimum_norm_dc_voltages(trap, names, curvature_zz_v_per_m2=target, at=null)
    assert np.allclose(trap.dc_field(null, volts), 0.0, atol=1e-9 * target * null[1])
    assert trap.dc_hessian(null, volts)[2, 2] == pytest.approx(target, rel=1e-9)
    assert volts["dc_l1"] == pytest.approx(volts["dc_r1"], rel=1e-9), (
        "the mirror-symmetric solution is the minimum-norm one"
    )
    with pytest.raises(ValueError):
        minimum_norm_dc_voltages(trap, names[:3], curvature_zz_v_per_m2=target, at=null)
    with pytest.raises(ValueError, match="cannot realize"):
        minimum_norm_dc_voltages(trap, names[:4], curvature_zz_v_per_m2=target, at=null)
