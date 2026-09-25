"""Mathieu exponents, stability, the Floquet function and C0, the coupled system and the (a, q) inversion."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from qutip_trap.trap.mathieu import (
    MathieuParameters,
    UnstableMathieuError,
    beta_exact,
    beta_lowest_order,
    c0_series,
    c0_wronskian,
    floquet_coefficients,
    is_stable,
    mathieu_from_secular,
    mathieu_parameters,
    monodromy,
    vector_betas,
)


def test_monodromy_matches_lowest_order_to_fourth_order_in_q() -> None:
    """beta^2 - (a + q^2/2) = O(q^4): the deviation shrinks 16-fold when q halves (RMP 2003 Eq. 15)."""
    for s in (0.0, 0.2, -0.2):  # a of order q^2, so the a q^2 cross term is O(q^4) as well
        d1 = beta_exact(s * 0.1**2, 0.1) ** 2 - (s * 0.1**2 + 0.1**2 / 2.0)
        d2 = beta_exact(s * 0.05**2, 0.05) ** 2 - (s * 0.05**2 + 0.05**2 / 2.0)
        assert d1 / d2 == pytest.approx(16.0, rel=0.06)
        assert abs(d1) < 0.3 * 0.1**4


def test_stability_edge_at_a_zero_lies_at_q_0_908() -> None:
    assert is_stable(0.0, 0.9) and not is_stable(0.0, 0.92)
    with pytest.raises(UnstableMathieuError):
        beta_exact(0.0, 0.95)
    with pytest.raises(UnstableMathieuError):
        beta_exact(-0.01, 0.0)  # a static anti-confinement with no rf is unstable
    assert monodromy(0.0, 0.0).beta == 0.0, "an unconfined axis is marginal, beta = 0"


def test_beta_is_even_in_q_so_equal_a_gives_equal_radials() -> None:
    """q_y = -q_x does not split the radial secular frequencies; only a_x != a_y does."""
    assert beta_exact(-0.003, 0.25) == pytest.approx(beta_exact(-0.003, -0.25), abs=1e-13)
    assert beta_exact(-0.003, 0.25) != pytest.approx(beta_exact(-0.004, 0.25), abs=1e-6)


def test_exact_exponent_against_the_lowest_order_and_preprint_forms() -> None:
    """At q = 0.3 the exact exponent is 1.8 % from sqrt(a + q^2/2) and 0.12 % from Wineland's preprint form
    [(a + q^2/2)/(1 - 3q^2/8)]^{1/2}."""
    exact = beta_exact(0.0, 0.3)
    preprint = math.sqrt((0.3**2 / 2.0) / (1.0 - 3.0 * 0.3**2 / 8.0))
    assert abs(preprint / exact - 1.0) == pytest.approx(0.0012, abs=2e-4)
    assert abs(beta_lowest_order(0.0, 0.3) / exact - 1.0) == pytest.approx(0.018, abs=1e-3)
    # the phase nu T_rf = beta pi per rf period: 0.222580 at q = 0.1 against 0.222144 from sqrt(q^2/2), the O(q^4) term
    assert beta_exact(0.0, 0.1) * math.pi == pytest.approx(0.222580, abs=5e-7)
    assert beta_exact(0.0, 0.1) / beta_lowest_order(0.0, 0.1) - 1.0 == pytest.approx(1.963e-3, rel=1e-2)


@pytest.mark.parametrize(("q", "expected"), [(0.1, 1.001890), (0.2, 1.007741), (0.3, 1.018161)])
def test_c0_wronskian_anchors_and_series(q: float, expected: float) -> None:
    """C0 = 1.001890, 1.007741, 1.018161 (1e-6), even in q and within q^4/4 of c0_series = 1 + 3q^2/16."""
    c0 = c0_wronskian(0.0, q)
    assert c0 == pytest.approx(expected, abs=1e-6)
    assert c0 == pytest.approx(c0_wronskian(0.0, -q), abs=1e-12)
    assert abs(c0 - c0_series(q)) < 0.25 * q**4
    assert c0_wronskian(0.01, 0.0) == 1.0


def _floquet_by_integration(
    a: float, q: float, n_samples: int = 4096
) -> tuple[np.ndarray, np.ndarray, float]:
    """The Floquet eigen-solution u(xi) over one period by direct integration, Wronskian-normalized: (xi, u, beta)."""
    mono = monodromy(a, q)
    beta = mono.beta
    w, v = np.linalg.eig(mono.matrix)
    vec = v[:, int(np.argmin(np.abs(w - np.exp(1j * math.pi * beta))))]

    def rhs(xi: float, y: np.ndarray) -> list[float]:
        return [y[1], -(a - 2.0 * q * math.cos(2.0 * xi)) * y[0]]

    xi = np.linspace(0.0, math.pi, n_samples, endpoint=False)
    sols = [
        solve_ivp(rhs, (0.0, math.pi), y0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        for y0 in ([1.0, 0.0], [0.0, 1.0])
    ]
    u = vec[0] * sols[0].sol(xi)[0] + vec[1] * sols[1].sol(xi)[0]
    du = vec[0] * sols[0].sol(xi)[1] + vec[1] * sols[1].sol(xi)[1]
    return xi, u * math.sqrt(beta / float(np.mean(np.imag(np.conj(u) * du)))), beta


def test_the_recursion_and_the_direct_integration_give_one_floquet_function() -> None:
    for q in (0.1, 0.3):
        fc = floquet_coefficients(0.0, q)
        xi, u, beta = _floquet_by_integration(0.0, q)
        assert beta == pytest.approx(fc.beta, abs=1e-12)
        assert abs(np.mean(u * np.exp(-1j * beta * xi))) == pytest.approx(fc.c0, abs=1e-9)
        series = np.exp(1j * fc.beta * xi) * (fc.c[None, :] * np.exp(2j * fc.n[None, :] * xi[:, None])).sum(
            -1
        )
        assert np.max(np.abs(series * np.exp(-1j * np.angle(u[0] / series[0])) - u)) < 1e-8


def test_floquet_ratios_under_the_adopted_sign() -> None:
    """c_{+-1}/c_0 = -q/((2 +- beta)^2 - a) (-0.023322, -0.026875 at q = 0.1) and u'(0)/(i nu u(0)) = 1.1036 ~ 1 + q."""
    fc = floquet_coefficients(0.0, 0.1)
    assert fc.ratio(1) == pytest.approx(-0.023322, abs=2e-6)
    assert fc.ratio(-1) == pytest.approx(-0.026875, abs=2e-6)
    assert fc.ratio(1) == pytest.approx(-0.1 / ((2.0 + fc.beta) ** 2), abs=1e-5)
    assert fc.ratio(-1) == pytest.approx(-0.1 / ((2.0 - fc.beta) ** 2), abs=1e-5)
    u_dot_over_u = np.sum(fc.c * (fc.beta + 2.0 * fc.n)) / (fc.beta * np.sum(fc.c))
    assert u_dot_over_u == pytest.approx(1.1036, abs=2e-4)


def test_trajectory_micromotion_ratio_q_over_2_and_second_harmonic_q2_over_32() -> None:
    """Wineland 1998 Eq. 4: in-phase modulation of fractional amplitude q/2 (a contraction in the adopted sign) and q^2/32
    at 2 Omega; the residuals are O(q^3) and O(q^4), so halving q shrinks them 8x and 16x."""
    d1: dict[float, float] = {}
    d2: dict[float, float] = {}
    for q in (0.2, 0.1, 0.05, 0.025):
        fc = floquet_coefficients(0.0, q)
        first = fc.ratio(1) + fc.ratio(-1)
        second = fc.ratio(2) + fc.ratio(-2)
        d1[q] = first + q / 2.0
        d2[q] = second - q * q / 32.0
        assert first == pytest.approx(-q / 2.0, rel=0.42 * q**2)
        assert second == pytest.approx(q * q / 32.0, rel=0.78 * q**2)
        assert abs(d1[q]) < 0.21 * q**3
        assert abs(d2[q]) < 0.024 * q**4
    for hi, lo in ((0.2, 0.1), (0.1, 0.05), (0.05, 0.025)):
        assert d1[hi] / d1[lo] == pytest.approx(8.0, rel=0.03)
        assert d2[hi] / d2[lo] == pytest.approx(16.0, rel=0.03)


def test_coupled_system_exponents_with_rotated_rf_axes() -> None:
    """A = diag(-0.004, -0.004, 0.008), Q rotated by 20 deg at q = 0.30: beta = {sqrt 0.008 exactly, 0.2061, 0.2061}."""
    th = math.radians(20.0)
    rot = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    betas = vector_betas(np.diag([-0.004, -0.004, 0.008]), rot @ np.diag([0.30, -0.30, 0.0]) @ rot.T)
    assert betas == pytest.approx([math.sqrt(0.008), 0.2061, 0.2061], abs=5e-5)
    assert betas[0] == pytest.approx(math.sqrt(0.008), abs=1e-12)
    with pytest.raises(UnstableMathieuError):
        vector_betas(np.diag([-0.004, -0.004, 0.008]), np.diag([0.95, -0.95, 0.0]))


def test_mathieu_parameters_from_matrices_matches_the_scalar_route_when_commuting() -> None:
    a_t, q_t = (-0.002, -0.002, 0.004), (0.25, -0.25, 0.0)
    scalar = mathieu_parameters(a_t, q_t, 40e6)
    th = 0.4
    rot = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    rotated = mathieu_parameters(rot @ np.diag(a_t) @ rot.T, rot @ np.diag(q_t) @ rot.T, 40e6)
    assert sorted(rotated.secular_hz) == pytest.approx(sorted(scalar.secular_hz), rel=1e-9)
    assert sorted(rotated.C0) == pytest.approx(sorted(scalar.C0), rel=1e-9), (
        "the frame must diagonalize Q too"
    )
    assert np.allclose(np.abs(rotated.principal_axes[:, 2]), [0.0, 0.0, 1.0])
    assert rotated.mass_kg is None and rotated.omega_rf_hz == 40e6


def test_mathieu_from_secular_round_trip_and_laplace() -> None:
    a, q = mathieu_from_secular((3.0e6, 2.9e6, 1.0e6), 30e6)
    assert sum(a) == pytest.approx(0.0, abs=1e-15)
    assert q[1] == -q[0] and q[2] == 0.0 and q[0] > 0.0
    assert a[2] == pytest.approx((2.0e6 / 30e6) ** 2)
    params = mathieu_parameters(a, q, 30e6)
    assert params.secular_hz == pytest.approx((3.0e6, 2.9e6, 1.0e6), rel=1e-9)
    a_eq, _ = mathieu_from_secular((3.0e6, 3.0e6, 1.0e6), 30e6)
    assert a_eq[0] == pytest.approx(a_eq[1], abs=1e-12) and a_eq[0] == pytest.approx(
        -a_eq[2] / 2.0, abs=1e-12
    )
    with pytest.raises(UnstableMathieuError):
        mathieu_from_secular((16e6, 3e6, 1e6), 30e6)


def test_mathieu_parameters_record_refuses_off_branch_values() -> None:
    with pytest.raises(ValueError):
        MathieuParameters(
            np.zeros((3, 3)), np.zeros((3, 3)), (1.5, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
        )
    with pytest.raises(ValueError):
        MathieuParameters(np.zeros(3), np.zeros((3, 3)), (0.1, 0.1, 0.1), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
