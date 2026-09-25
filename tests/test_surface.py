"""Gapless-plane surface-electrode traps: basis potentials, rf null, escape point and depth against House 2008,
Nizamani and Hensinger 2012 and Wesenberg 2008."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.trap.mathieu import beta_exact
from qutip_trap.trap.pseudopotential import mathieu_matrices
from qutip_trap.trap.surface import (
    Electrodes,
    GaplessPlaneTrap,
    rectangle_gradient,
    rectangle_hessian,
    rectangle_potential,
    strip_potential,
)
from qutip_trap.units import ATOMIC_MASS_KG, E_C, TWO_PI
from tests.oracles import five_wire_null_height_m

HOUSE = dict(a=100e-6, b=120e-6, v=300.0, omega=TWO_PI * 50e6, m=1.46e-25)


def _rect(x: float, y: float, z: float) -> float:
    return rectangle_potential(x, y, z, -1.0, 1.0, -0.5, 0.5)


def _five_wire(a: float = HOUSE["a"], b: float = HOUSE["b"], **extra: float) -> GaplessPlaneTrap:
    return GaplessPlaneTrap(Electrodes("surface_five_wire", {"a_m": a, "b_m": b, **extra}))


def test_rectangle_potential_laplace_boundary_and_strip_limit() -> None:
    h = 1e-4
    for x, y, z in ((0.2, 0.7, 0.1), (1.5, 0.4, 0.3), (0.0, 2.0, 0.0)):
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
        "odd in y"
    )


def _half_space_green(
    x: float, y: float, z: float, x1: float, x2: float, z1: float, z2: float, n: int = 120
) -> float:
    """phi(r) = (y/2pi) int_patch dS'/|r - r'|^3, the Dirichlet Green's function of the grounded half-space, by a
    tensor-product Gauss-Legendre rule."""
    u, wu = np.polynomial.legendre.leggauss(n)
    xp, wx = 0.5 * (x2 - x1) * u + 0.5 * (x1 + x2), 0.5 * (x2 - x1) * wu
    zp, wz = 0.5 * (z2 - z1) * u + 0.5 * (z1 + z2), 0.5 * (z2 - z1) * wu
    kernel = ((x - xp)[:, None] ** 2 + y * y + (z - zp)[None, :] ** 2) ** -1.5
    return y / (2.0 * math.pi) * float(wx @ kernel @ wz)


def test_rectangle_equals_the_half_space_green_function_and_converges_as_one_over_l_squared() -> None:
    """House's four-arctan rectangle is the half-space Green's-function integral, and it approaches the infinite strip
    at the rate 1/L^2."""
    for pt in ((0.2, 0.7, 0.1), (1.5, 0.4, 0.3), (0.0, 2.0, 0.0)):
        assert _rect(*pt) == pytest.approx(_half_space_green(*pt, -1.0, 1.0, -0.5, 0.5), abs=5e-15)
    strip = strip_potential(0.3, 0.8, -1.0, 1.0)
    dev = {
        length: abs(rectangle_potential(0.3, 0.8, 0.0, -1.0, 1.0, -length, length) - strip)
        for length in (10.0, 20.0, 40.0, 80.0, 160.0)
    }
    for hi, lo in ((10.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 160.0)):
        assert dev[hi] / dev[lo] == pytest.approx(4.0, rel=6e-3)
    assert dev[160.0] < 1e-5


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


def test_the_basis_functions_of_a_complete_layout_sum_to_one() -> None:
    trap = _five_wire(1e-4, 1.2e-4)
    for x in (-3e-4, -1e-4, 0.5e-4, 1.5e-4, 5e-4):
        assert sum(trap.potential(n, np.array([x, 1e-9, 0.0])) for n in trap.names) == pytest.approx(
            1.0, abs=1e-6
        )
    assert sum(trap.potential(n, np.array([0.5e-4, 1e-3, 0.0])) for n in trap.names) == pytest.approx(
        1.0, abs=1e-12
    )
    partial = GaplessPlaneTrap(Electrodes("surface_general", {}, strips={"rf": ((-1e-4, 0.0), (1e-4, 2e-4))}))
    assert partial.potential("rf", np.array([0.5e-4, 1e-3, 0.0])) < 1.0


def test_five_wire_house_fixture() -> None:
    """House 2008 (a = 100, b = c = 120 um, 300 V, 2 pi x 50 MHz, 1.46e-25 kg): null (50.000000, 92.195445) um, escape
    169.655527 um, depth 0.184465 eV, Q11 = -Q22 = 0.2284260, and 4.038 MHz pseudopotential and 4.0804 MHz exact secular
    frequencies against House's 11.5 MHz, which drops a factor 1/(2 sqrt 2)."""
    trap = _five_wire()
    null = trap.rf_null()
    assert null[0] * 1e6 == pytest.approx(50.0, abs=1e-6)
    assert null[1] * 1e6 == pytest.approx(92.195445, abs=1e-6)
    assert five_wire_null_height_m(HOUSE["a"], HOUSE["b"]) == pytest.approx(null[1], rel=1e-12)
    esc = trap.escape_point()
    assert esc[0] == pytest.approx(HOUSE["a"] / 2.0, rel=1e-9)
    assert esc[1] * 1e6 == pytest.approx(169.655527, abs=1e-6)
    assert trap.depth_j(HOUSE["v"], HOUSE["m"], HOUSE["omega"]) / E_C == pytest.approx(0.184465, abs=1e-6)
    _, q_mat = mathieu_matrices(
        np.zeros((3, 3)), HOUSE["v"] * trap.rf_hessian_unit(null), HOUSE["m"], HOUSE["omega"]
    )
    assert q_mat[0, 0] == pytest.approx(0.2284260, abs=1e-7)
    assert q_mat[1, 1] == pytest.approx(-0.2284260, abs=1e-7)
    assert abs(q_mat[0, 1]) < 1e-12 and abs(np.trace(q_mat)) < 1e-12
    q = q_mat[0, 0]
    assert q * HOUSE["omega"] / (2 * math.sqrt(2)) / TWO_PI == pytest.approx(4.038e6, abs=1e3)
    assert beta_exact(0.0, q) * HOUSE["omega"] / 2 / TWO_PI == pytest.approx(4.0804e6, abs=1e3)
    assert q * HOUSE["omega"] / TWO_PI == pytest.approx(11.42e6, rel=1e-2)


def test_z_invariant_rails_have_no_axial_rf_curvature() -> None:
    """For z-invariant rails Q33 = Q13 = Q23 = 0 with Q11 = -Q22."""
    trap = _five_wire()
    _a, q_mat = mathieu_matrices(
        np.zeros((3, 3)), HOUSE["v"] * trap.rf_hessian_unit(trap.rf_null()), HOUSE["m"], HOUSE["omega"]
    )
    scale = float(np.max(np.abs(q_mat)))
    assert scale > 0.2
    for i, j in ((2, 2), (0, 2), (1, 2), (2, 0), (2, 1)):
        assert abs(q_mat[i, j]) < 1e-14 * scale
    assert q_mat[0, 0] == pytest.approx(-q_mat[1, 1], rel=1e-12)


def test_segmented_rf_rails_report_an_escape_point_and_a_depth() -> None:
    """On 6 mm rectangle rails the null, the saddle and the depth agree with the strip model (1e-3, 5e-3), 600 um rails sit
    lower with a deeper trap, and a single rf rail has no escape point."""
    a, b, length = HOUSE["a"], HOUSE["b"], 6e-3
    rects = {
        "rf": ((-b, 0.0, -length, length), (a, a + b, -length, length)),
        "centre": ((0.0, a, -length, length),),
    }
    trap = GaplessPlaneTrap(Electrodes("surface_general", {}, rectangles=rects))
    strips = _five_wire()
    null = trap.rf_null()
    assert null[1] == pytest.approx(five_wire_null_height_m(a, b), rel=1e-3)
    assert null[0] == pytest.approx(a / 2, abs=1e-12)
    esc = trap.escape_point()
    assert esc[1] == pytest.approx(strips.escape_point()[1], rel=1e-3)
    assert esc[0] == pytest.approx(a / 2, abs=1e-9), (
        "the mirror-symmetric layout keeps the saddle on the axis"
    )
    field = trap.rf_field_unit(esc)
    grad_psi = -2.0 * (trap.rf_hessian_unit(esc) @ field)  # grad |E|^2 = -2 H E: a critical point of Psi
    assert float(np.max(np.abs(grad_psi))) < 1e-6 * float(np.dot(field, field)) / esc[1]
    depth = trap.depth_j(300.0, 1.46e-25, HOUSE["omega"])
    assert depth == pytest.approx(strips.depth_j(300.0, 1.46e-25, HOUSE["omega"]), rel=5e-3)
    short = {
        "rf": ((-b, 0.0, -300e-6, 300e-6), (a, a + b, -300e-6, 300e-6)),
        "centre": ((0.0, a, -300e-6, 300e-6),),
    }
    stubby = GaplessPlaneTrap(Electrodes("surface_general", {}, rectangles=short))
    assert stubby.rf_null()[1] < 0.95 * null[1]
    assert stubby.escape_point()[1] < 0.95 * esc[1]
    assert stubby.depth_j(300.0, 1.46e-25, HOUSE["omega"]) > 1.2 * depth
    one = GaplessPlaneTrap(
        Electrodes("surface_general", {}, rectangles={"rf": ((-b, 0.0, -length, length),)})
    )
    with pytest.raises(ValueError):
        one.escape_point()


def test_long_segmented_rails_recover_the_strip_null_height() -> None:
    a, b, length = HOUSE["a"], HOUSE["b"], 6e-3
    rects = {
        "rf": ((-b, 0.0, -length, length), (a, a + b, -length, length)),
        "dc_l1": ((-b - 200e-6, -b, -600e-6, -150e-6),),
        "centre": ((0.0, a, -length, length),),
    }
    null = GaplessPlaneTrap(Electrodes("surface_general", {}, rectangles=rects)).rf_null(
        np.array([a / 2, 90e-6, 0.0])
    )
    assert null[1] == pytest.approx(five_wire_null_height_m(a, b), rel=2e-3)
    assert null[0] == pytest.approx(a / 2, abs=1e-9)


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


def test_four_wire_escape_height_and_the_wesenberg_depth_constant() -> None:
    """The symmetric four-wire null sits one centre width up, (y_E - y0)/a = sqrt(2 + sqrt 5) - 1 (1e-10, Wesenberg's
    |p_s|/d) and the depth is (5 sqrt 5 - 11)/(2 pi^2) U0 = 0.009136125 U0 (1e-9)."""
    a = 1.0
    trap = GaplessPlaneTrap(Electrodes("surface_four_wire", {"a_m": a, "c_m": a}))
    null, esc = trap.rf_null(), trap.escape_point()
    assert null[1] == pytest.approx(a, rel=1e-12)
    assert abs(esc[0]) < 1e-12 * a and abs(null[0]) < 1e-12 * a
    assert (esc[1] - null[1]) / a == pytest.approx(math.sqrt(2.0 + math.sqrt(5.0)) - 1.0, rel=1e-10)
    u_bar_over_u0 = (5.0 * math.sqrt(5.0) - 11.0) / (2.0 * math.pi**2)
    assert u_bar_over_u0 == pytest.approx(0.009136125, abs=5e-10)
    assert trap.depth_j(1.0, 1.0, 1.0) / (E_C**2 / (4.0 * null[1] ** 2)) == pytest.approx(
        u_bar_over_u0, rel=1e-9
    )


def test_asymmetric_rails_move_the_null_toward_the_narrower_rail() -> None:
    """Rails at -c < x < 0 and a < x < a + b: x0 = ac/(b + c), y0 = sqrt(abc(a + b + c))/(b + c) (House Eqs. 24-27)."""
    a, b, c = 1.0, 2.0, 1.0
    null = _five_wire(a, b, c_m=c).rf_null()
    assert null[0] == pytest.approx(a * c / (b + c), rel=1e-12) and null[0] < a / 2
    assert null[1] == pytest.approx(math.sqrt(a * b * c * (a + b + c)) / (b + c), rel=1e-12)


def test_rf_pseudopotential_isotropic_at_the_null_and_axes_set_by_dc() -> None:
    trap = _five_wire(60e-6, 300e-6, c_m=150e-6)
    h = trap.rf_hessian_unit(trap.rf_null())
    h2 = h @ h  # Hess |E|^2 / 2 at a zero of E
    assert h2[0, 0] == pytest.approx(h2[1, 1], rel=1e-10) and abs(h2[0, 1]) < 1e-10 * h2[0, 0]
    assert h[0, 1] != pytest.approx(0.0, abs=1e-3 * abs(h[0, 0])), "asymmetric rails rotate the rf Hessian"
    assert abs(np.trace(h)) < 1e-9 * abs(h[0, 0])


def test_nizamani_fixture() -> None:
    """Nizamani 2012 (a = 60, b = 300, c = 150 um, 171 u, 2 pi x 55 MHz, 500 V): h = 82.46 um (quoted 85), |Q11| = 0.1976
    and the principal-axis q = 0.2171 at 4.222 MHz, and their q_N = 0.6948 is 3.2002 q, not a Mathieu q."""
    m = 171.0 * ATOMIC_MASS_KG
    omega = TWO_PI * 55e6
    trap = _five_wire(60e-6, 300e-6, c_m=150e-6)
    null = trap.rf_null()
    assert null[1] * 1e6 == pytest.approx(82.46, abs=5e-3)
    _, q_mat = mathieu_matrices(np.zeros((3, 3)), 500.0 * trap.rf_hessian_unit(null), m, omega)
    q_pa = math.hypot(q_mat[0, 0], q_mat[0, 1])
    assert abs(q_mat[0, 0]) == pytest.approx(0.1976, abs=1e-4)
    assert q_pa == pytest.approx(0.2171, abs=1e-4)
    assert q_pa * omega / (2 * math.sqrt(2)) / TWO_PI == pytest.approx(4.222e6, abs=1e3)
    q_n = 2.0 * E_C * 500.0 / (m * omega**2 * null[1] ** 2)
    assert q_n == pytest.approx(0.6948, abs=1e-4) and q_n / q_pa == pytest.approx(3.2002, abs=1e-3)
