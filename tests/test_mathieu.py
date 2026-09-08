"""Section 9.1 (Monodromy, Sign of q, Pseudopotential frequency, Trajectory), 9.10, 9.13 and 9.17 Mathieu targets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.trap.mathieu import (
    MathieuParameters,
    UnstableMathieuError,
    beta_exact,
    beta_lowest_order,
    beta_preprint,
    c0_series,
    c0_wronskian,
    floquet_coefficients,
    floquet_function_by_integration,
    is_stable,
    mathieu_from_secular,
    mathieu_parameters,
    monodromy,
    pseudopotential_radial_rad_s,
    stability_edge_q,
    vector_monodromy,
)
from qutip_trap.trap.pseudopotential import (
    linear_trap_parameters,
    pseudopotential_j,
    radial_frequency_from_voltage_rad_s,
)
from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI


def test_monodromy_matches_lowest_order_to_fourth_order_in_q() -> None:
    """beta^2 - (a + q^2/2) = O(q^4): the deviation shrinks 16-fold when q halves (RMP 2003 Eq. 15)."""
    for s in (0.0, 0.2, -0.2):  # a of order q^2, so the a q^2 cross term is O(q^4) as well
        d1 = beta_exact(s * 0.1**2, 0.1) ** 2 - (s * 0.1**2 + 0.1**2 / 2.0)
        d2 = beta_exact(s * 0.05**2, 0.05) ** 2 - (s * 0.05**2 + 0.05**2 / 2.0)
        assert d1 / d2 == pytest.approx(16.0, rel=0.06)
        assert abs(d1) < 0.3 * 0.1**4


def test_stability_edge_at_a_zero_is_q_0_908() -> None:
    assert stability_edge_q(0.0) == pytest.approx(0.908, abs=1e-3)
    assert is_stable(0.0, 0.9) and not is_stable(0.0, 0.92)
    with pytest.raises(UnstableMathieuError):
        beta_exact(0.0, 0.95)
    with pytest.raises(UnstableMathieuError):
        beta_exact(-0.01, 0.0)  # a static anti-confinement with no rf is unstable


def test_beta_is_even_in_q_so_equal_a_gives_equal_radials() -> None:
    """q_y = -q_x does not split the radial secular frequencies; only a_x != a_y does (Section 4.1.1, [corrected])."""
    assert beta_exact(-0.003, 0.25) == pytest.approx(beta_exact(-0.003, -0.25), abs=1e-13)
    assert beta_exact(-0.003, 0.25) != pytest.approx(beta_exact(-0.004, 0.25), abs=1e-6)


def test_preprint_form_is_far_more_accurate_than_the_lowest_order_form() -> None:
    """0.12% from the exact exponent at q = 0.3 against 1.8% for sqrt(a + q^2/2) (Section 4.1.1)."""
    exact = beta_exact(0.0, 0.3)
    assert abs(beta_preprint(0.0, 0.3) / exact - 1.0) == pytest.approx(0.0012, abs=2e-4)
    assert abs(beta_lowest_order(0.0, 0.3) / exact - 1.0) == pytest.approx(0.018, abs=1e-3)


def test_marginal_axis_with_no_curvature_has_beta_zero() -> None:
    m = monodromy(0.0, 0.0)
    assert m.marginal and m.beta == 0.0


@pytest.mark.parametrize(("q", "expected"), [(0.1, 1.001890), (0.2, 1.007741), (0.3, 1.018161)])
def test_c0_wronskian_anchors_and_series(q: float, expected: float) -> None:
    """Section 9.10: C0 = 1.001890, 1.007741, 1.018161 (tolerance 1e-6), even in q, equal to 1 + 3q^2/16 to O(q^4)."""
    c0 = c0_wronskian(0.0, q)
    assert c0 == pytest.approx(expected, abs=1e-6)
    assert c0 == pytest.approx(c0_wronskian(0.0, -q), abs=1e-12)
    assert abs(c0 - c0_series(q)) < 0.25 * q**4
    assert c0 > 1.0 > 1.0 / (1.0 + q / 2.0), "the RMP's (1 + q/2)^-1 is the retired negative control"


def test_c0_is_exactly_one_without_rf() -> None:
    assert c0_wronskian(0.01, 0.0) == 1.0


def test_comb_phase_per_rf_period_of_section_9_10() -> None:
    """Section 9.10, second clause: comb phase per rf period 0.222580 against beta omega_rf T/2 = 0.222144 from the
    pseudopotential exponent sqrt(a + q^2/2); the printed beta omega_rf T is exactly 2x, because nu = beta omega_rf/2."""
    fc = floquet_coefficients(0.0, 0.1)
    phase = fc.comb_phase_per_rf_period_rad()
    assert phase == pytest.approx(0.222580, abs=5e-7)
    assert phase == pytest.approx(fc.beta * math.pi, rel=1e-15), "nu T_rf with nu = beta omega_rf/2"
    ppt = math.sqrt(0.0 + 0.1**2 / 2.0) * math.pi
    assert ppt == pytest.approx(0.222144, abs=5e-7)
    assert phase / ppt - 1.0 == pytest.approx(1.963e-3, rel=1e-2), (
        "the O(q^4) correction to beta, not a convention"
    )
    assert 2.0 * phase == pytest.approx(0.445161, abs=1e-6), "the RMP's printed beta omega_rf T"
    # a q = 0 trap has no comb at all: the phase per rf period is sqrt(a) pi exactly
    assert floquet_coefficients(0.01, 0.0).comb_phase_per_rf_period_rad() == pytest.approx(
        math.sqrt(0.01) * math.pi, rel=1e-12
    )


def test_two_constructions_of_the_floquet_function_agree() -> None:
    """The recursion null vector and the direct integration give the same Wronskian-normalized c_0 (check_c0_floquet.py)."""
    for q in (0.1, 0.3):
        fc = floquet_coefficients(0.0, q)
        xi, u, beta = floquet_function_by_integration(0.0, q)
        assert beta == pytest.approx(fc.beta, abs=1e-12)
        assert abs(np.mean(u * np.exp(-1j * beta * xi))) == pytest.approx(fc.c0, abs=1e-9)
        assert (
            np.max(np.abs(fc.evaluate(xi) * np.exp(-1j * np.angle(u[0] / fc.evaluate(xi[:1])[0])) - u)) < 1e-8
        )


def test_floquet_ratios_under_the_adopted_sign() -> None:
    """Section 9.17: c_{+-1}/c_0 = -q/((2 +- beta)^2 - a) (-0.023322, -0.026875 at q = 0.1), u'(0)/(i nu u(0)) = 1.1036."""
    fc = floquet_coefficients(0.0, 0.1)
    assert fc.ratio(1) == pytest.approx(-0.023322, abs=2e-6)
    assert fc.ratio(-1) == pytest.approx(-0.026875, abs=2e-6)
    lo_p, lo_m = fc.lowest_order_ratios()
    assert fc.ratio(1) == pytest.approx(lo_p, abs=1e-5) and fc.ratio(-1) == pytest.approx(lo_m, abs=1e-5)
    u_dot_over_u = np.sum(fc.c * (fc.beta + 2.0 * fc.n)) / (fc.beta * np.sum(fc.c))
    assert u_dot_over_u == pytest.approx(1.1036, abs=2e-4), (
        "1 + q to lowest order, the plan's rf phase origin"
    )


def test_trajectory_micromotion_ratio_q_over_2_and_second_harmonic_q2_over_32() -> None:
    """Wineland 1998 Eq. 4: in-phase modulation of fractional amplitude q/2 (a contraction in the adopted sign), q^2/32
    at 2 Omega. The ORDER of the residual is what pins the Floquet coefficients: (c_1 + c_-1)/c_0 + q/2 is O(q^3) and
    (c_2 + c_-2)/c_0 - q^2/32 is O(q^4), so halving q shrinks them by 8x and 16x. A test with only a fixed relative band
    (the 2 % / 5 % it replaces, five to seven times the actual deviation) would pass with an O(q^2)-wrong coefficient."""
    d1: dict[float, float] = {}
    d2: dict[float, float] = {}
    for q in (0.2, 0.1, 0.05, 0.025):
        fc = floquet_coefficients(0.0, q)
        first = fc.ratio(1) + fc.ratio(-1)
        second = fc.ratio(2) + fc.ratio(-2)
        d1[q] = first + q / 2.0
        d2[q] = second - q * q / 32.0
        # the anchors themselves, at the deviation actually measured rather than a 5x band
        assert first == pytest.approx(-q / 2.0, rel=0.42 * q**2)
        assert second == pytest.approx(q * q / 32.0, rel=0.78 * q**2)
        assert abs(d1[q]) < 0.21 * q**3, "the leading correction to -q/2 is O(q^3)"
        assert abs(d2[q]) < 0.024 * q**4, "the leading correction to q^2/32 is O(q^4)"
    for hi, lo in ((0.2, 0.1), (0.1, 0.05), (0.05, 0.025)):
        assert d1[hi] / d1[lo] == pytest.approx(8.0, rel=0.03), "O(q^3): halving q shrinks the residual 8x"
        assert d2[hi] / d2[lo] == pytest.approx(16.0, rel=0.03), "O(q^4): 16x"


def test_sideband_weights_reduce_to_q2_over_16() -> None:
    w = floquet_coefficients(0.0, 0.1).sideband_weights()
    assert (w[1] + w[-1]) / 2.0 == pytest.approx(0.1**2 / 16.0, rel=0.02)
    assert w[0] == 1.0


def test_vector_monodromy_sign_guard_of_section_9_13() -> None:
    """A = diag(-0.004, -0.004, 0.008), Q rotated by 20 deg at q = 0.30: beta = {0.0894 = sqrt 0.008 exactly, 0.2061};
    House's printed lower-left sign gives |lambda| = 1.3244 and 0.7550 (the axial channel falsely unstable)."""
    th = math.radians(20.0)
    rot = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    a_mat = np.diag([-0.004, -0.004, 0.008])
    q_mat = rot @ np.diag([0.30, -0.30, 0.0]) @ rot.T
    good = vector_monodromy(a_mat, q_mat)
    assert good.stable
    assert good.betas == pytest.approx([math.sqrt(0.008), 0.2061, 0.2061], abs=5e-5)
    assert good.betas[0] == pytest.approx(math.sqrt(0.008), abs=1e-12)
    bad = vector_monodromy(a_mat, q_mat, lower_left_sign=+1.0)
    assert not bad.stable
    mags = np.sort(np.abs(bad.eigenvalues))
    assert mags[0] == pytest.approx(0.7550, abs=1e-4) and mags[-1] == pytest.approx(1.3244, abs=1e-4)
    with pytest.raises(UnstableMathieuError):
        _ = bad.betas


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


def test_pseudopotential_frequency_two_forms_and_berkeland_example() -> None:
    """omega_r = Q V0/(sqrt 2 Omega m R^2) equals q Omega/(2 sqrt 2) with q = 2 Q V0/(m R^2 Omega^2); Berkeland's q_x = 2 sqrt 2 omega_x/Omega = 0.28."""
    m = 170.93578 * ATOMIC_MASS_KG
    omega_rf = TWO_PI * 20e6
    (a_x, a_y, a_z), (q_x, q_y, q_z) = linear_trap_parameters(
        v_rf_peak_v=200.0, u_dc_v=5.0, omega_rf_rad_s=omega_rf, mass_kg=m, r_m=1.0e-3, z0_m=2.0e-3, kappa=0.3
    )
    assert a_x == a_y == -a_z / 2.0 and q_y == -q_x and q_z == 0.0
    w_v = radial_frequency_from_voltage_rad_s(200.0, omega_rf, m, 1.0e-3)
    assert w_v == pytest.approx(pseudopotential_radial_rad_s(q_x, omega_rf), rel=1e-12)
    omega_x = 0.28 * omega_rf / (2.0 * math.sqrt(2.0))
    assert 2.0 * math.sqrt(2.0) * omega_x / omega_rf == pytest.approx(0.28)


def test_rf_amplitude_readings_move_depth_and_frequency_together() -> None:
    """Section 9.16 row 13-7: a misread factor f scales the depth by f^2 and every secular frequency by f."""
    m = 40.0 * ATOMIC_MASS_KG
    omega_rf = TWO_PI * 30e6
    base = pseudopotential_j(np.array([1e5, 0.0, 0.0]), m, omega_rf)
    assert pseudopotential_j(np.array([math.sqrt(2.0) * 1e5, 0.0, 0.0]), m, omega_rf) == pytest.approx(
        2.0 * base
    )
    assert pseudopotential_j(np.array([0.5e5, 0.0, 0.0]), m, omega_rf) == pytest.approx(base / 4.0)
    q1 = 0.2
    w1 = pseudopotential_radial_rad_s(q1, omega_rf)
    assert pseudopotential_radial_rad_s(math.sqrt(2.0) * q1, omega_rf) == pytest.approx(math.sqrt(2.0) * w1)


def test_mathieu_parameters_record_refuses_off_branch_values() -> None:
    with pytest.raises(ValueError):
        MathieuParameters(
            np.zeros((3, 3)), np.zeros((3, 3)), (1.5, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
        )
    with pytest.raises(ValueError):
        MathieuParameters(np.zeros(3), np.zeros((3, 3)), (0.1, 0.1, 0.1), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
