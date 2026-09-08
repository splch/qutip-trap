"""Filter functions and dynamical decoupling (PLAN.md Section 6.9; Section 9.15 rows on the toggling frame, the amplitude
filter function, the dc floor, the end-to-end dephasing normalization, the CPMG/UDD orders and the finite-pulse collapse;
Section 9.17 'Ramsey coherence normalization'; M7), against ``check_composite.py`` sections 9 to 15."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import (
    ControlSegment,
    DecouplingSequence,
    composite_pulse,
    decoupling_sequence,
    filter_function,
    gaussian_spectrum,
)
from qutip_trap.noise.decoupling import (
    adjoint,
    amplitude_filter_function,
    chi_integral,
    composite_segments,
    control_matrix,
    control_matrix_time,
    dc_floor,
    dc_polygon,
    final_adjoint,
    pulse_adjoint,
    rotation,
)
from tests.m2_fixtures import microwave_device

TAU = 1.0


def test_toggling_frame_control_matrix_is_the_adjoint_of_u_dagger() -> None:
    """R_ij = (1/2)Tr[U^dag sigma_i U sigma_j] is SO(3) with det +1; for U = exp(-i Omega t sigma_x/2) its third row is (0, sin, cos)
    (Section 13 row 'Toggling-frame control matrix')."""
    for th, ph in ((0.3, 0.0), (1.1, 0.7), (2.7, 2.0)):
        r = adjoint(rotation(th, ph))
        assert np.allclose(r, pulse_adjoint(th, ph), atol=1e-14)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-14) and np.linalg.det(r) == pytest.approx(1.0)
    r0 = adjoint(rotation(0.3, 0.0))
    assert np.allclose(r0[2], [0.0, math.sin(0.3), math.cos(0.3)])


def test_free_induction_and_hahn_echo_limits_and_biercuk_both_parities() -> None:
    """F = 4 sin^2(x/2) (n = 0), 16 sin^4(x/4) (n = 1) to 12 digits; Biercuk's (-1)^{n+1} form reproduced by the gated machinery at
    both parities with finite pulses (CP timings at x = 0.9: 0.0396108693597, 5.0437242342e-4, 4.1526238611e-4, 3.0236329296e-5)."""
    x = np.array([0.3, 1.7, 5.0])
    fid = decoupling_sequence("custom", 0, TAU, 0.0, centres=[])
    assert np.allclose(fid.filter_function(x), 4 * np.sin(x / 2) ** 2, rtol=1e-12)
    hahn = decoupling_sequence("hahn", 1, TAU, 0.0)
    assert np.allclose(hahn.filter_function(x), 16 * np.sin(x / 4) ** 4, rtol=1e-12)
    assert np.allclose(hahn.biercuk_filter_function(x), 16 * np.sin(x / 4) ** 4, rtol=1e-12)
    expected = {
        1: 0.0396108693597088,
        2: 5.0437242342e-4,
        3: 0.0004152623861142973,
        4: 3.0236329296287202e-05,
    }
    for n in (1, 2, 3, 4):
        s = decoupling_sequence("cpmg", n, TAU, 0.02)
        gated = float(s.filter_function(np.array([0.9]), gated=True)[0])
        assert gated == pytest.approx(float(s.biercuk_filter_function(np.array([0.9]))[0]), rel=1e-12)
        assert gated == pytest.approx(expected[n], rel=1e-9)
    s4 = decoupling_sequence("cpmg", 4, TAU, 0.07)
    x_ann = math.pi / 0.07
    assert s4.filter_function(np.array([x_ann]), gated=True)[0] == pytest.approx(0.753020396283, rel=1e-9)
    assert s4.filter_function(np.array([x_ann]), gated=True)[0] == pytest.approx(
        4 * math.sin(x_ann / 2) ** 2, rel=1e-9
    )


def test_finite_pulse_collapse_of_the_udd_order_and_the_universal_coefficients() -> None:
    """At delta_pi = 0.02 the gated UDD order drops from 2n + 2 to 4 (odd n, F -> (omega tau_pi)^4/16) and 6 (even n,
    (omega tau)^2 (omega tau_pi)^4/64); n = 1, 2 keep 4 and 6 (Section 9.15 row 'Finite-pulse collapse')."""
    w = np.array([1e-3, 1e-2])
    for n, order, coeff in (
        (1, 4, None),
        (2, 6, None),
        (3, 4, 1 / 16),
        (4, 6, 1 / 64),
        (5, 4, 1 / 16),
        (6, 6, 1 / 64),
    ):
        s = decoupling_sequence("udd", n, TAU, 0.02)
        f = s.filter_function(w, gated=True)
        assert math.log(f[1] / f[0]) / math.log(10) == pytest.approx(order, abs=0.02)
        if coeff is not None and n % 2 == 1:
            assert f[0] / (w[0] * 0.02) ** 4 == pytest.approx(coeff, rel=2e-3)
        elif coeff is not None:
            assert f[0] / (w[0] ** 2 * (w[0] * 0.02) ** 4) == pytest.approx(coeff, rel=2e-3)
    # the ungated (noise on during the pulse) machinery keeps a first-order transverse term at odd n: F ~ omega^2
    s3 = decoupling_sequence("udd", 3, TAU, 0.02)
    f3 = s3.filter_function(w)
    assert math.log(f3[1] / f3[0]) / math.log(10) == pytest.approx(2.0, abs=0.02)
    assert s3.moments(1)[0] == pytest.approx(-0.5) and decoupling_sequence("cpmg", 4, TAU, 0.0).moments(1)[
        0
    ] == pytest.approx(0.5)


def test_mixed_axis_sequences_match_brute_force_propagation() -> None:
    """XY4, XY8, KDD and CDD compose per-pulse blocks with the accumulated Lambda (never the scalar (-1)^l): the closed-form
    R(omega) equals -i omega int R(t) e^{i omega t} dt from the propagator to quadrature precision, the sequences return to identity, and
    they are flagged multi-axis (Section 6.9 [background])."""
    for timing, n in (("xy4", 4), ("xy8", 8), ("kdd", 20), ("cdd", 2)):
        seq = decoupling_sequence(timing, n, TAU, 0.01)
        assert not seq.is_single_axis() and seq.feasible()
        assert np.allclose(final_adjoint(seq.segments()), np.eye(3), atol=1e-12)
        with pytest.raises(ValueError):
            seq.biercuk_filter_function(np.array([1.0]))
    segs = decoupling_sequence("xy4", 4, TAU, 0.05).segments()
    w = 2.3
    # segment-wise Simpson quadrature of -i w int R(t) e^{iwt} dt against the closed form (the check script's mpmath
    # quadrature reaches 4e-10; a second-order rule on 4001 points per segment reaches 1e-8 here)
    from scipy.integrate import simpson

    num = np.zeros((3, 3), dtype=complex)
    t0 = 0.0
    for seg in segs:
        if seg.duration_s > 0.0:
            tt = np.linspace(t0, t0 + seg.duration_s, 4001)
            rt = np.array([control_matrix_time(segs, t) for t in tt])
            num += simpson(rt * np.exp(1j * w * tt)[:, None, None], x=tt, axis=0)
        t0 += seg.duration_s
    num = -1j * w * num
    ana = control_matrix(segs, np.array([w]))[0]
    assert np.max(np.abs(num - ana)) < 1e-8
    r = control_matrix_time(segs, 0.6)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_green_primitive_pulse_forms_and_the_z_rotation_degenerate_case() -> None:
    """Green Eqs. 46a/46b for a primitive pi_X: R_zz = w^2/(w^2 - Omega^2)(e^{iw} + 1), R_zy = i w Omega/(w^2 - Omega^2)(e^{iw} + 1), R_zx = 0;
    free evolution over T at w: R_z = (0, 0, 1 - e^{iwT}) up to the -i w convention."""
    om = math.pi
    for wv in (0.4, 1.9, 5.5):
        r = control_matrix([ControlSegment(math.pi, 0.0, 1.0)], np.array([wv]))[0]
        zz = wv**2 / (wv**2 - om**2) * (np.exp(1j * wv) + 1)
        zy = 1j * wv * om / (wv**2 - om**2) * (np.exp(1j * wv) + 1)
        assert abs(r[2, 2] - zz) < 1e-12 and abs(r[2, 1] - zy) < 1e-12 and abs(r[2, 0]) < 1e-14
    free = control_matrix([ControlSegment(0.0, 0.0, 1.4)], np.array([0.5]))[0]
    assert abs(free[2, 2] - (1 - np.exp(1j * 0.5 * 1.4))) < 1e-12 and abs(free[2, 0]) < 1e-14


def test_end_to_end_dephasing_normalization_and_chi() -> None:
    """Section 9.15 (the load-bearing regression): Gaussian S_b at sigma = 2, rms 0.3, tau = 1, free induction: chi = (2/pi) int S F/w^2
    = 0.137512017889 = 2 <a_1^2>, W = e^{-chi} = 0.871523876057; reading the splitting PSD (4 S_b) gives 4x (negative control)."""
    g = gaussian_spectrum(0.3, 2.0, "u", omega_max_rad_s=40.0, n=40001)
    fid = decoupling_sequence("custom", 0, TAU, 0.0, centres=[])
    chi = chi_integral(
        lambda w: np.asarray(g.value(w)), lambda w: fid.filter_function(w), 1e-7, 40.0, n_points=400001
    )
    assert chi == pytest.approx(0.137512017889, rel=2e-5)
    assert math.exp(-chi) == pytest.approx(0.871523876057, rel=1e-5)
    chi4 = chi_integral(
        lambda w: 4.0 * np.asarray(g.value(w)), lambda w: fid.filter_function(w), 1e-7, 40.0, n_points=400001
    )
    assert chi4 == pytest.approx(0.550048071557, rel=2e-5)


def test_amplitude_filter_function_dc_polygon_crossover_and_dc_floor() -> None:
    """Primitive F_a = sin^2(omega tau_P/2) exactly; SK1 and BB1 give 5.17486e-14 and 1.52202e-14 at omega = 1e-4 Omega (theta = pi);
    the dc polygon closes for the amplitude-correcting families and equals (pi, 0, 0) for a primitive; the dc floors 5.87365e-6
    (SK1), 3.53675e-9 (BB1) and 1.67248e-9 (CORPSE) at the paper's benchmark (Section 9.15)."""
    prim = composite_segments(composite_pulse("primitive", math.pi), 1.0)
    for x in (0.3, 0.7, 1.9):
        assert amplitude_filter_function(prim, np.array([x * math.pi / math.pi]))[0] == pytest.approx(
            math.sin(x * math.pi / 2) ** 2, rel=1e-12
        )
    sk1 = composite_segments(composite_pulse("SK1", math.pi), 1.0)
    bb1 = composite_segments(composite_pulse("BB1", math.pi), 1.0)
    assert amplitude_filter_function(sk1, np.array([1e-4]))[0] == pytest.approx(5.17486e-14, rel=1e-5)
    assert amplitude_filter_function(bb1, np.array([1e-4]))[0] == pytest.approx(1.52202e-14, rel=1e-5)
    assert np.linalg.norm(dc_polygon(sk1)) < 1e-12 and np.linalg.norm(dc_polygon(bb1)) < 1e-12
    assert np.allclose(dc_polygon(prim), [math.pi, 0.0, 0.0])
    # the crossover above which the composite pulse hurts (true 0.0702926 Omega for SK1, 0.132464 for BB1; the printed bound 0.0254648)
    from scipy.optimize import brentq

    def cross(segs):  # type: ignore[no-untyped-def]
        return brentq(
            lambda w: amplitude_filter_function(segs, np.array([w]))[0] - math.sin(w * math.pi / 2) ** 2,
            0.03,
            0.3,
        )

    assert cross(sk1) == pytest.approx(0.0702926, rel=1e-4) and cross(bb1) == pytest.approx(
        0.132464, rel=1e-4
    )
    var = 2.07e9 / math.pi
    assert dc_floor(22.8302557111, 1, var, 1.5e6) == pytest.approx(5.87365e-6, rel=1e-5)
    assert dc_floor(9.388566343, 2, var, 1.5e6) == pytest.approx(3.53675e-9, rel=1e-5)
    assert dc_floor(0.006500751892, 1, var, 1.5e6) == pytest.approx(1.67248e-9, rel=1e-5)


def test_sequence_record_feasibility_moments_and_clock_rounding() -> None:
    s = decoupling_sequence("udd", 3, TAU, 0.05)
    assert s.n_pulses == 3 and s.feasible() and s.deltas[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        decoupling_sequence("cpmg", 8, TAU, 0.2)  # pulses overlap
    rounded, err = s.rounded_to_clock(1e-3)
    assert abs(err["A1_error"]) < 1e-3 and abs(err["A2_error"]) < 5e-3
    assert isinstance(rounded, DecouplingSequence) and all(
        abs(round(d / 1e-3) - d / 1e-3) < 1e-9 for d in rounded.deltas
    )
    with pytest.raises(ValueError):
        decoupling_sequence("kdd", 10, TAU, 0.01)


@pytest.mark.slow
def test_filter_function_api_with_the_monte_carlo_path_through_the_builder() -> None:
    """Appendix E filter_function on a device: the first-order 1 - F_av = chi/2 against a Monte Carlo over sampled b(t) trajectories
    propagated through the Section 4.3.1 Hamiltonian (the builder's qubit-trajectory hook), inside xi^2 << 1; the infrared cutoff
    is the spectrum's own band and is reported with its sensitivity; free induction, the Hahn echo and CPMG-2 order as a slow
    spectrum demands (Section 6.9, the route (c)/(d) agreement)."""
    dev = microwave_device()
    tau = 200e-6
    # a slow Gaussian dephasing spectrum in rad/s (S_b of the sigma_z coefficient): rms 300 Hz, corner 300 Hz, xi^2 = 0.14
    spec = gaussian_spectrum(
        2.0 * math.pi * 300.0, 2.0 * math.pi * 300.0, "(rad/s)^2/(rad/s)", omega_max_rad_s=2.0 * math.pi * 3e3
    )
    fid = filter_function(dev, decoupling_sequence("custom", 0, tau, 0.0, centres=[]), spectrum=spec)
    hahn = filter_function(
        dev, decoupling_sequence("hahn", 1, tau, 4e-6), spectrum=spec, monte_carlo_samples=24, seed=3
    )
    cp2 = filter_function(dev, decoupling_sequence("cpmg", 2, tau, 4e-6), spectrum=spec)
    f = hahn.fitted
    assert f["xi2"][0] == pytest.approx(tau**2 * spec.variance(), rel=1e-9) and f["xi2"][0] < 0.2
    # the Gaussian band starts at zero, so the cutoff defaults to three decades below the sequence (Section 12)
    assert f["tau_s"][0] == pytest.approx(tau) and f["omega_min_rad_s"][0] == pytest.approx(
        2.0 * math.pi / (1000.0 * tau)
    )
    assert abs(f["dlnchi_dlnomega_min"][0]) < 0.05, "a Gaussian band is infrared convergent"
    ff, (mc, err) = f["infidelity_ff"][0], f["infidelity_mc"]
    assert 5e-4 < ff < 5e-3 and abs(mc - ff) < 3.0 * err + 0.25 * ff, (ff, mc, err)
    assert f["mc_agrees"][0] == 1.0
    assert fid.fitted["infidelity_ff"][0] > 30.0 * ff > 30.0 * cp2.fitted["infidelity_ff"][0], (
        "FID >> Hahn >> CPMG-2 on slow noise"
    )
    # quasi-static limit: F -> (omega tau)^2, chi -> 2 tau^2 Var(b), 1 - F_av = chi/2 -> xi^2 (a1 = tau b, <a1^2> = tau^2 Var(b))
    assert fid.fitted["infidelity_ff"][0] == pytest.approx(tau**2 * spec.variance(), rel=0.05), (
        "quasi-static limit xi^2"
    )
    assert abs(fid.fitted["alpha"][0]) < 0.05 and cp2.fitted["alpha"][0] > 0.9
    assert hahn.data.shape[1] == 2 and hahn.model == "filter_function[dephasing]"
    explicit = filter_function(
        dev, decoupling_sequence("hahn", 1, tau, 4e-6), spectrum=spec, omega_min_rad_s=1.0
    )
    assert explicit.fitted["omega_min_rad_s"][0] == 1.0
