"""Filter functions, dynamical decoupling and the dc floor (PLAN.md Section 6.9)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.optimize import brentq

from qutip_trap.control.composite import composite_pulse, leading_coefficient
from qutip_trap.noise.decoupling import (
    ControlSegment,
    DecouplingSequence,
    adjoint,
    amplitude_filter_function,
    chi_integral,
    composite_segments,
    control_matrix,
    dc_floor,
    dc_polygon,
    decoupling_sequence,
    filter_function,
    frozen_noise_floor,
    frozen_noise_infidelity,
    pulse_adjoint,
    rotation,
)
from qutip_trap.noise.spectra import NoiseSpectrum, gaussian_spectrum, power_law_spectrum
from tests.fixtures import microwave_device

TAU = 1.0
RABI_HZ = 50e3
RMS_HZ = 300.0


def _biercuk(seq: DecouplingSequence, omega: np.ndarray) -> np.ndarray:
    """Biercuk Eq. 2 for a single-axis pi_X train, both parities: |1 + (-1)^{n+1} e^{ix} + 2 cos(x delta_pi/2)
    sum_j (-1)^j e^{i delta_j x}|^2 at x = omega tau."""
    x = np.asarray(omega, dtype=float) * seq.tau_s
    acc = np.zeros_like(x, dtype=complex)
    for j, d in enumerate(seq.deltas, start=1):
        acc += (-1) ** j * np.exp(1j * d * x)
    y = (
        1.0
        + (-1) ** (seq.n_pulses + 1) * np.exp(1j * x)
        + 2.0 * np.cos(x * seq.tau_pi_s / seq.tau_s / 2.0) * acc
    )
    return np.asarray(np.abs(y) ** 2)


def _control_matrix_time(segments: Sequence[ControlSegment], t: float) -> np.ndarray:
    """R(t) by the trace formula on the propagated U_c(t): the reference for the closed-form control matrix."""
    u = np.eye(2, dtype=complex)
    elapsed = 0.0
    for seg in segments:
        if t <= elapsed:
            break
        if seg.is_instantaneous:
            u = rotation(seg.theta_rad, seg.phi_rad) @ u
            continue
        s = min(t - elapsed, seg.duration_s)
        if not seg.is_free:
            u = rotation(seg.omega_rad_s * s, seg.phi_rad) @ u
        elapsed += seg.duration_s
    return adjoint(u)


def test_toggling_frame_control_matrix_is_the_adjoint_of_u_dagger() -> None:
    """R_ij = (1/2)Tr[U^dag sigma_i U sigma_j] is SO(3) with det +1; for U = exp(-i Omega t sigma_x/2) its third row is
    (0, sin, cos)."""
    for th, ph in ((0.3, 0.0), (1.1, 0.7), (2.7, 2.0)):
        r = adjoint(rotation(th, ph))
        assert np.allclose(r, pulse_adjoint(th, ph), atol=1e-14)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-14) and np.linalg.det(r) == pytest.approx(1.0)
    r0 = adjoint(rotation(0.3, 0.0))
    assert np.allclose(r0[2], [0.0, math.sin(0.3), math.cos(0.3)])


def test_free_induction_and_hahn_echo_limits_and_biercuk_both_parities() -> None:
    """F = 4 sin^2(x/2) (n = 0), 16 sin^4(x/4) (n = 1) to 12 digits; Biercuk's (-1)^{n+1} form reproduced by the gated
    machinery at both parities with finite pulses (CP timings at x = 0.9)."""
    x = np.array([0.3, 1.7, 5.0])
    fid = decoupling_sequence("custom", 0, TAU, 0.0, centres=[])
    assert np.allclose(fid.filter_function(x), 4 * np.sin(x / 2) ** 2, rtol=1e-12)
    hahn = decoupling_sequence("hahn", 1, TAU, 0.0)
    assert np.allclose(hahn.filter_function(x), 16 * np.sin(x / 4) ** 4, rtol=1e-12)
    assert np.allclose(_biercuk(hahn, x), 16 * np.sin(x / 4) ** 4, rtol=1e-12)
    expected = {
        1: 0.0396108693597088,
        2: 5.0437242342e-4,
        3: 0.0004152623861142973,
        4: 3.0236329296287202e-05,
    }
    for n in (1, 2, 3, 4):
        s = decoupling_sequence("cpmg", n, TAU, 0.02)
        gated = float(s.filter_function(np.array([0.9]), gated=True)[0])
        assert gated == pytest.approx(float(_biercuk(s, np.array([0.9]))[0]), rel=1e-12)
        assert gated == pytest.approx(expected[n], rel=1e-9)
    s4 = decoupling_sequence("cpmg", 4, TAU, 0.07)
    x_ann = math.pi / 0.07
    assert s4.filter_function(np.array([x_ann]), gated=True)[0] == pytest.approx(0.753020396283, rel=1e-9)
    assert s4.filter_function(np.array([x_ann]), gated=True)[0] == pytest.approx(
        4 * math.sin(x_ann / 2) ** 2, rel=1e-9
    )


def test_finite_pulse_collapse_of_the_udd_order_and_the_universal_coefficients() -> None:
    """At delta_pi = 0.02 the gated UDD order drops from 2n + 2 to 4 (odd n, F -> (omega tau_pi)^4/16) and 6 (even n,
    (omega tau)^2 (omega tau_pi)^4/64) to 2e-3, and the ungated form keeps omega^2 at n = 3."""
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
    # the ungated machinery (noise on during the pulse) keeps a first-order transverse term at odd n: F ~ omega^2
    s3 = decoupling_sequence("udd", 3, TAU, 0.02)
    f3 = s3.filter_function(w)
    assert math.log(f3[1] / f3[0]) / math.log(10) == pytest.approx(2.0, abs=0.02)
    assert s3.moments(1)[0] == pytest.approx(-0.5)
    assert decoupling_sequence("cpmg", 4, TAU, 0.0).moments(1)[0] == pytest.approx(0.5)


def test_mixed_axis_sequences_match_brute_force_propagation() -> None:
    """XY4, XY8, KDD and CDD return to the identity (to 1e-12), and the closed-form R(omega) equals -i omega int R(t)
    e^{i omega t} dt from the propagator to 1e-8."""
    for timing, n in (("xy4", 4), ("xy8", 8), ("kdd", 20), ("cdd", 2)):
        seq = decoupling_sequence(timing, n, TAU, 0.01)
        assert not seq.is_single_axis() and seq.feasible()
        lam = np.eye(3)
        for seg in seq.segments():
            lam = pulse_adjoint(seg.theta_rad, seg.phi_rad) @ lam
        assert np.allclose(lam, np.eye(3), atol=1e-12)
    segs = decoupling_sequence("xy4", 4, TAU, 0.05).segments()
    w = 2.3
    num = np.zeros((3, 3), dtype=complex)
    t0 = 0.0
    for seg in segs:
        if seg.duration_s > 0.0:
            tt = np.linspace(t0, t0 + seg.duration_s, 4001)
            rt = np.array([_control_matrix_time(segs, t) for t in tt])
            num += simpson(rt * np.exp(1j * w * tt)[:, None, None], x=tt, axis=0)
        t0 += seg.duration_s
    num = -1j * w * num
    ana = control_matrix(segs, np.array([w]))[0]
    assert np.max(np.abs(num - ana)) < 1e-8


def test_green_primitive_pulse_forms() -> None:
    """Green Eqs. 46a/46b for a primitive pi_X: R_zz = w^2/(w^2 - Omega^2)(e^{iw} + 1), R_zy = i w Omega/(w^2 - Omega^2)
    (e^{iw} + 1), R_zx = 0; free evolution over T: R_z = (0, 0, 1 - e^{iwT})."""
    om = math.pi
    for wv in (0.4, 1.9, 5.5):
        r = control_matrix([ControlSegment(math.pi, 0.0, 1.0)], np.array([wv]))[0]
        zz = wv**2 / (wv**2 - om**2) * (np.exp(1j * wv) + 1)
        zy = 1j * wv * om / (wv**2 - om**2) * (np.exp(1j * wv) + 1)
        assert abs(r[2, 2] - zz) < 1e-12 and abs(r[2, 1] - zy) < 1e-12 and abs(r[2, 0]) < 1e-14
    free = control_matrix([ControlSegment(0.0, 0.0, 1.4)], np.array([0.5]))[0]
    assert abs(free[2, 2] - (1 - np.exp(1j * 0.5 * 1.4))) < 1e-12 and abs(free[2, 0]) < 1e-14


def test_end_to_end_dephasing_normalization_and_chi() -> None:
    """Gaussian S_b at sigma = 2, rms 0.3, tau = 1, free induction: chi = (2/pi) int S F/w^2 = 0.137512017889 = 2 <a_1^2>,
    W = e^{-chi} = 0.871523876057; reading the splitting PSD (4 S_b) gives 4x."""
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
    """F_a is sin^2(omega tau_P/2) for the primitive, 5.17486e-14 (SK1) and 1.52202e-14 (BB1) at 1e-4 Omega, crossing the
    primitive at 0.0702926 and 0.132464 Omega; dc_floor gives 5.87365e-6, 3.53675e-9 and 1.67248e-9 (to 1e-5)."""
    prim = composite_segments(composite_pulse("primitive", math.pi), 1.0)
    for x in (0.3, 0.7, 1.9):
        assert amplitude_filter_function(prim, np.array([x]))[0] == pytest.approx(
            math.sin(x * math.pi / 2) ** 2, rel=1e-12
        )
    sk1 = composite_segments(composite_pulse("SK1", math.pi), 1.0)
    bb1 = composite_segments(composite_pulse("BB1", math.pi), 1.0)
    assert amplitude_filter_function(sk1, np.array([1e-4]))[0] == pytest.approx(5.17486e-14, rel=1e-5)
    assert amplitude_filter_function(bb1, np.array([1e-4]))[0] == pytest.approx(1.52202e-14, rel=1e-5)
    assert np.linalg.norm(dc_polygon(sk1)) < 1e-12 and np.linalg.norm(dc_polygon(bb1)) < 1e-12
    assert np.allclose(dc_polygon(prim), [math.pi, 0.0, 0.0])

    # the crossover above which the composite pulse hurts: 0.0702926 Omega for SK1, 0.132464 for BB1
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


def test_sequence_record_feasibility_and_refusals() -> None:
    """The record's feasibility, single-axis test and roll-off moments A_k = sum_j (-1)^j delta_j^k (j from 1, as in
    Biercuk's pulse sum), and the factory's refusals."""
    s = decoupling_sequence("udd", 3, TAU, 0.05)
    assert s.n_pulses == 3 and s.feasible() and s.deltas[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        decoupling_sequence("cpmg", 8, TAU, 0.2)  # pulses overlap
    with pytest.raises(ValueError):
        decoupling_sequence("kdd", 10, TAU, 0.01)
    cpmg2 = DecouplingSequence("cpmg", 2, 1e-3, 1e-6, (0.25, 0.75), (0.0, 0.0))
    assert cpmg2.is_single_axis() and cpmg2.feasible()
    a1, a2 = cpmg2.moments()
    assert a1 == pytest.approx(-0.25 + 0.75) and a2 == pytest.approx(-(0.25**2) + 0.75**2)
    assert not DecouplingSequence("cpmg", 2, 1e-3, 0.6e-3, (0.25, 0.75), (0.0, 0.0)).feasible()
    assert not DecouplingSequence("xy4", 2, 1e-3, 1e-6, (0.25, 0.75), (0.0, math.pi / 2)).is_single_axis()


# ---- filter_function on a device ------------------------------------------------------------------------------------------


def _slow_band(omega_max_rad_s: float | None = None) -> NoiseSpectrum:
    """S_b of the sigma_z coefficient: 300 Hz rms, 300 Hz corner, so <beta^2>/Omega^2 = 3.6e-5 at Omega = 2 pi 50 kHz."""
    return gaussian_spectrum(
        2.0 * math.pi * RMS_HZ, 2.0 * math.pi * RMS_HZ, "(rad/s)^2/(rad/s)", omega_max_rad_s=omega_max_rad_s
    )


@pytest.mark.slow
def test_filter_function_api_with_the_monte_carlo_path_through_the_builder() -> None:
    """A 24-sample Monte Carlo through the builder agrees with a Hahn echo's first-order 1 - F_av = chi/2 within 3 sigma + 25 %
    at xi^2 = 0.14, the infrared cutoff is reported with its sensitivity, and FID > 30 x Hahn > 30 x CPMG-2."""
    dev = microwave_device()
    tau = 200e-6
    spec = _slow_band(2.0 * math.pi * 3e3)  # xi^2 = 0.14
    fid = filter_function(dev, decoupling_sequence("custom", 0, tau, 0.0, centres=[]), spectrum=spec)
    hahn = filter_function(
        dev, decoupling_sequence("hahn", 1, tau, 4e-6), spectrum=spec, monte_carlo_samples=24, seed=3
    )
    cp2 = filter_function(dev, decoupling_sequence("cpmg", 2, tau, 4e-6), spectrum=spec)
    f = hahn.fitted
    assert f["xi2"][0] == pytest.approx(tau**2 * spec.variance(), rel=1e-9) and f["xi2"][0] < 0.2
    # the Gaussian band starts at zero, so the cutoff defaults to three decades below the sequence
    assert f["tau_s"][0] == pytest.approx(tau) and f["omega_min_rad_s"][0] == pytest.approx(
        2.0 * math.pi / (1000.0 * tau)
    )
    assert abs(f["dlnchi_dlnomega_min"][0]) < 0.05, "a Gaussian band is infrared convergent"
    ff, (mc, err) = f["infidelity_ff"][0], f["infidelity_mc"]
    assert 5e-4 < ff < 5e-3 and abs(mc - ff) < 3.0 * err + 0.25 * ff, (ff, mc, err)
    assert f["mc_agrees"][0] == 1.0
    assert fid.fitted["infidelity_ff"][0] > 30.0 * ff > 30.0 * cp2.fitted["infidelity_ff"][0]
    # quasi-static limit: F -> (omega tau)^2, so 1 - F_av = chi/2 -> xi^2
    assert fid.fitted["infidelity_ff"][0] == pytest.approx(tau**2 * spec.variance(), rel=0.05)
    assert abs(fid.fitted["alpha"][0]) < 0.05 and cp2.fitted["alpha"][0] > 0.9
    assert hahn.data.shape[1] == 2 and hahn.model == "filter_function[dephasing]"
    explicit = filter_function(
        dev, decoupling_sequence("hahn", 1, tau, 4e-6), spectrum=spec, omega_min_rad_s=1.0
    )
    assert explicit.fitted["omega_min_rad_s"][0] == 1.0


def test_power_law_spectrum_is_infrared_divergent_for_the_hahn_echo() -> None:
    """On S_b ~ 1/omega^4 the roll-off orders are 0, 1, 1, 2, 3 for FID, Hahn, CPMG-3, CPMG-2 and UDD-3, and d ln chi/d ln
    omega_min is -3, -1, -1 and 0 for FID, Hahn, CPMG-3 and UDD-3 (to 1e-3)."""
    dev = microwave_device()
    tau = 1e-3
    div = power_law_spectrum(1e-6, 1.0, 4.0, "u", omega_min_rad_s=1e-4, omega_max_rad_s=1e6)

    def f_of(name: str, n: int):  # type: ignore[no-untyped-def]
        seq = (
            decoupling_sequence("custom", 0, tau, 0.0, centres=[])
            if n == 0
            else decoupling_sequence(name, n, tau, 0.0)
        )
        return filter_function(dev, seq, spectrum=div, dc_floor=False).fitted

    # the roll-off orders: F ~ (omega tau)^{2(alpha+1)}
    assert f_of("custom", 0)["alpha"][0] == pytest.approx(0.0, abs=1e-3)
    assert f_of("hahn", 1)["alpha"][0] == pytest.approx(1.0, abs=1e-3)
    assert f_of("cpmg", 3)["alpha"][0] == pytest.approx(1.0, abs=1e-3), "odd-n CPMG keeps alpha = 1"
    assert f_of("cpmg", 2)["alpha"][0] == pytest.approx(2.0, abs=5e-3), (
        "even-n Carr-Purcell reaches alpha = 2"
    )
    assert f_of("udd", 3)["alpha"][0] == pytest.approx(3.0, abs=1e-3), "UDD cancels n derivatives"
    assert f_of("custom", 0)["dlnchi_dlnomega_min"][0] == pytest.approx(-3.0, abs=1e-3)
    assert f_of("hahn", 1)["dlnchi_dlnomega_min"][0] == pytest.approx(-1.0, abs=1e-3)
    assert f_of("cpmg", 3)["dlnchi_dlnomega_min"][0] == pytest.approx(-1.0, abs=1e-3)
    assert f_of("udd", 3)["dlnchi_dlnomega_min"][0] == pytest.approx(0.0, abs=1e-3)
    # the borderline alpha = 2 case cancels to chi ~ 1e-20, where the float64 d ln chi is round-off, not a divergence
    assert f_of("cpmg", 2)["chi"][0] < 1e-18


def _mains_spectrum(background_at_1hz: float, line_amp: float, line_hz: float = 60.0) -> NoiseSpectrum:
    """A 1/f S_b background with narrow Lorentzian lines at the mains fundamental and its first four harmonics."""
    w = np.geomspace(2.0 * math.pi * 1e-3, 2.0 * math.pi * 1e4, 40001)
    s = background_at_1hz * (w / (2.0 * math.pi)) ** -1.0
    for h in (1, 2, 3, 4, 5):
        w_h = 2.0 * math.pi * h * line_hz
        hwhm = 2.0 * math.pi * 0.05
        s = s + (line_amp / h**2) * hwhm**2 / ((w - w_h) ** 2 + hwhm**2)
    return NoiseSpectrum(w, s, "(rad/s)^2/(rad/s)")


def test_the_mains_lines_are_visible_to_the_filter_function() -> None:
    """A 1/60 s Hahn echo sees more than twice the chi of the bare 1/f background, and deep in the omega^4 asymptote chi falls
    9975.4-fold per decade of tau with the lines carrying 3.367 times the background (to 1e-3)."""
    spec = _mains_spectrum(1e-4, 4.0)
    bare = _mains_spectrum(1e-4, 0.0)

    def chi_of(tau: float, n: int, sp: NoiseSpectrum) -> float:
        seq = (
            decoupling_sequence("cpmg", n, tau, 0.0)
            if n
            else decoupling_sequence("custom", 0, tau, 0.0, centres=[])
        )
        return chi_integral(
            lambda w: np.asarray(sp.value(w)),
            lambda w: seq.filter_function(np.asarray(w)),
            2.0 * math.pi * 1e-2,
            2.0 * math.pi * 1e4,
            n_points=200001,
        )

    tau_on = 1.0 / 60.0
    assert chi_of(tau_on, 1, spec) > 2.0 * chi_of(tau_on, 1, bare)
    short, shorter = chi_of(1e-5, 1, spec), chi_of(1e-6, 1, spec)
    assert short / shorter == pytest.approx(9975.4, rel=1e-3), (short, shorter)
    assert short / shorter == pytest.approx(1e4, rel=3e-3), (
        "the Hahn echo's omega^4 law, reached by tau = 10 us"
    )
    assert chi_of(1e-6, 1, spec) == pytest.approx(3.367 * chi_of(1e-6, 1, bare), rel=1e-3), (
        "in the omega^4 asymptote the lines carry 3.4x the background's integrated power"
    )


@pytest.mark.slow
def test_xy_n_decay_laws_as_w_equals_exp_minus_chi() -> None:
    """With W = e^{-chi}, free induction's chi quadruples and a fixed-spacing (XY)^N's doubles per doubling of tau (to 1e-3 and
    1e-2), and at fixed tau N = 4, 20, 40 pulses suppress chi 6.5, 32.7 and 65-fold (to 8 %)."""
    dev = microwave_device()
    bg = 1e-3
    spec = _mains_spectrum(bg, bg * 4.0)

    def chi_of(seq: DecouplingSequence, tau: float, sp: NoiseSpectrum = spec) -> float:
        return float(
            filter_function(dev, seq, spectrum=sp, omega_min_rad_s=2.0 * math.pi / (10.0 * tau)).fitted[
                "chi"
            ][0]
        )

    def fid(tau: float) -> DecouplingSequence:
        return decoupling_sequence("custom", 0, tau, 0.0, centres=[])

    def xy_n(tau: float, spacing: float) -> DecouplingSequence:
        return decoupling_sequence("xy4", max(4, int(round(tau / spacing / 4)) * 4), tau, 1e-5)

    assert chi_of(fid(0.05), 0.05) < chi_of(fid(0.1), 0.1)
    assert chi_of(fid(0.05), 0.05) < chi_of(fid(0.05), 0.05, _mains_spectrum(4.0 * bg, 16.0 * bg))
    fid_chi = [chi_of(fid(t), t) for t in (0.05, 0.1, 0.2, 0.4)]
    for a, b in zip(fid_chi, fid_chi[1:]):
        assert b / a == pytest.approx(4.0, rel=1e-3), (a, b)
    spacing = 2.5e-3
    dd_chi = [chi_of(xy_n(t, spacing), t) for t in (0.05, 0.1, 0.2, 0.4)]
    for a, b in zip(dd_chi, dd_chi[1:]):
        assert b / a == pytest.approx(2.0, rel=1e-2), (a, b)
    for tau in (0.05, 0.1, 0.2):
        base = chi_of(fid(tau), tau)
        for n, expect in ((4, 6.5), (20, 32.7), (40, 65.0)):
            seq = decoupling_sequence("xy4", n, tau, min(1e-5, 0.2 * tau / n))
            assert base / chi_of(seq, tau) == pytest.approx(expect, rel=0.08), (tau, n)
    assert 0.4 / dd_chi[-1] == pytest.approx(1.17e5, rel=0.05)


# ---- the dc floor and the max rule --------------------------------------------------------------------------------------------


def test_dc_floor_order_is_channel_matched_not_the_pulses_own_order() -> None:
    """At m = 0 the dc floor carries c_hat = 1 (detuning) or pi^2/4 (amplitude): the dephasing floors of the primitive, SK1 and
    BB1 coincide, CORPSE's amplitude floor is the primitive's, and SCROFULOUS's dephasing floor is four times it (to 1e-4)."""
    dev, spec = microwave_device(), _slow_band()

    def floor(family: str, quadrature: str) -> float:
        r = filter_function(
            dev, composite_pulse(family, math.pi), quadrature=quadrature, spectrum=spec, rabi_hz=RABI_HZ
        )  # type: ignore[arg-type]
        return float(r.fitted["infidelity_dc"][0])

    prim_d = floor("primitive", "dephasing")
    assert floor("SK1", "dephasing") == pytest.approx(prim_d, rel=1e-5)
    assert floor("BB1", "dephasing") == pytest.approx(prim_d, rel=1e-5)
    assert prim_d < 1e-3
    prim_a = floor("primitive", "amplitude")
    assert floor("CORPSE", "amplitude") == pytest.approx(prim_a, rel=1e-5)
    for fam in ("primitive", "SK1", "BB1"):
        assert leading_coefficient(composite_pulse(fam, math.pi), "detuning", 2) == pytest.approx(
            1.0, rel=1e-5
        )
    for fam in ("primitive", "CORPSE"):
        assert leading_coefficient(composite_pulse(fam, math.pi), "amplitude", 2) == pytest.approx(
            math.pi**2 / 4.0, rel=1e-5
        )
    assert leading_coefficient(composite_pulse("SCROFULOUS", math.pi), "detuning", 2) == pytest.approx(
        4.0, rel=1e-5
    )
    assert floor("SCROFULOUS", "dephasing") == pytest.approx(4.0 * prim_d, rel=1e-4)


def test_frozen_noise_infidelity_is_the_exact_dc_limit_of_both_quadratures() -> None:
    """With H_0 = beta . sigma a frozen sigma_z coefficient b costs a primitive pi pulse 4 (b/Omega)^2 (to 1e-4) and a frozen
    amplitude beta_a sin^2(pi beta_a/2 Omega) (to 1e-9)."""
    prim = composite_segments(composite_pulse("primitive", math.pi), 1.0)
    for rel in (1e-3, 2e-3, 4e-3):
        assert frozen_noise_infidelity(prim, rel, "dephasing") == pytest.approx(4.0 * rel**2, rel=1e-4)
        assert frozen_noise_infidelity(prim, rel, "amplitude") == pytest.approx(
            math.sin(math.pi * rel / 2.0) ** 2, rel=1e-9
        )
    assert frozen_noise_infidelity(prim, 0.0, "dephasing") == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        frozen_noise_infidelity(prim, 1e-3, "universal")


@pytest.mark.parametrize(
    ("family", "quadrature", "rel"),
    [
        ("primitive", "dephasing", 5e-4),
        ("primitive", "amplitude", 5e-4),
        ("SK1", "dephasing", 5e-4),
        ("SK1", "amplitude", 5e-4),
        ("BB1", "dephasing", 5e-4),
        ("BB1", "amplitude", 5e-4),
        ("CORPSE", "amplitude", 5e-4),
        # SCROFULOUS's three pi-ish segments make its O(beta^4) remainder the largest of the library at this band
        ("SCROFULOUS", "dephasing", 2e-3),
        ("SCROFULOUS", "amplitude", 5e-4),
    ],
)
def test_reported_dc_floor_matches_the_exact_gaussian_frozen_average(
    family: str, quadrature: str, rel: float
) -> None:
    """The reported dc floor c_hat (2m + 1)!! rel^(m+1) matches the exact Gaussian frozen average at <beta^2>/Omega^2 = 3.6e-5
    to 5e-4 (2e-3 for SCROFULOUS dephasing)."""
    dev, spec = microwave_device(), _slow_band()
    pulse = composite_pulse(family, math.pi)
    reported = float(
        filter_function(dev, pulse, quadrature=quadrature, spectrum=spec, rabi_hz=RABI_HZ).fitted[  # type: ignore[arg-type]
            "infidelity_dc"
        ][0]
    )
    exact = frozen_noise_floor(
        composite_segments(pulse, 2.0 * math.pi * RABI_HZ), spec.variance(), quadrature
    )  # type: ignore[arg-type]
    assert reported == pytest.approx(exact, rel=rel), (family, quadrature, reported, exact)


def test_corpse_detuning_floor_needs_the_splitting_normalization() -> None:
    """CORPSE corrects the detuning at order m = 1, so the eps_d-vs-beta_d factor enters as 4^2 = 16: the exact-to-leading
    ratio of the unnormalized pairing falls to 16 from above as the noise shrinks, and the normalized floor is within 3 %."""
    pulse = composite_pulse("CORPSE", math.pi)
    omega = 2.0 * math.pi * RABI_HZ
    segs = composite_segments(pulse, omega)
    c_hat = leading_coefficient(pulse, "detuning", 4)
    ratios = []
    for var_rel in (3.6e-5, 9.0e-6, 2.25e-6, 5.625e-7):
        var = var_rel * omega**2
        ratios.append(frozen_noise_floor(segs, var, "dephasing", nodes=81) / dc_floor(c_hat, 1, var, omega))
    assert ratios[0] > ratios[-1] > 16.0, ratios
    assert ratios[-1] == pytest.approx(16.0, rel=0.03), ratios
    var = 5.625e-7 * omega**2
    assert dc_floor(c_hat, 1, 4.0 * var, omega) == pytest.approx(
        frozen_noise_floor(segs, var, "dephasing", nodes=81), rel=0.03
    )


def test_decoupling_sequences_get_the_max_rule_too() -> None:
    """A sequence's reported infidelity is the max of its first-order estimate and its dc floor, the exact Gaussian frozen
    average (to 1e-9), which for CPMG-4 on the 300 Hz band is 8.93e-6, above five times the estimate."""
    dev = microwave_device()
    tau, tau_pi = 200e-6, 4e-6
    spec = _slow_band(2.0 * math.pi * 3e3)
    seqs = {
        "fid": decoupling_sequence("custom", 0, tau, 0.0, centres=[]),
        "hahn": decoupling_sequence("hahn", 1, tau, tau_pi),
        "cpmg2": decoupling_sequence("cpmg", 2, tau, tau_pi),
        "cpmg4": decoupling_sequence("cpmg", 4, tau, tau_pi),
    }
    out = {}
    for name, seq in seqs.items():
        r = filter_function(dev, seq, spectrum=spec)
        ff, floor_v = r.fitted["infidelity_ff"][0], r.fitted["infidelity_dc"][0]
        assert r.fitted["infidelity"][0] == pytest.approx(max(ff, floor_v))
        assert floor_v == pytest.approx(
            frozen_noise_floor(seq.segments(), spec.variance(), "dephasing"), rel=1e-9
        )
        out[name] = (ff, floor_v)
    assert 0.5 < out["fid"][1] / out["fid"][0] < 1.5
    assert out["hahn"][1] < 0.1 * out["hahn"][0]
    assert out["cpmg4"][1] > 5.0 * out["cpmg4"][0]
    assert out["cpmg4"][1] == pytest.approx(8.93e-6, rel=5e-3)


def test_white_noise_does_not_enter_the_frozen_floor_or_xi_squared() -> None:
    """The dc floor and xi^2 use the tabulated band; chi uses band plus white level."""
    dev = microwave_device()
    band = _slow_band(2.0 * math.pi * 3e3)
    hot = NoiseSpectrum(band.omega_rad_s, band.S, band.unit, white_level=1e3)
    seq = decoupling_sequence("cpmg", 2, 200e-6, 4e-6)
    a = filter_function(dev, seq, spectrum=band).fitted
    b = filter_function(dev, seq, spectrum=hot).fitted
    assert b["xi2"][0] == pytest.approx(a["xi2"][0], rel=1e-12)
    assert b["infidelity_dc"][0] == pytest.approx(a["infidelity_dc"][0], rel=1e-12)
    assert b["chi"][0] > 1.05 * a["chi"][0], "the white level does raise chi"


def test_frozen_noise_floor_guards() -> None:
    prim = composite_segments(composite_pulse("primitive", math.pi), 1.0)
    assert frozen_noise_floor(prim, 0.0, "dephasing") == 0.0
    with pytest.raises(ValueError):
        frozen_noise_floor(prim, -1.0, "dephasing")
    v = 1e-6
    assert frozen_noise_floor(prim, v, "dephasing", nodes=21) == pytest.approx(
        frozen_noise_floor(prim, v, "dephasing", nodes=61), rel=1e-8
    )
    assert frozen_noise_floor(prim, v, "dephasing") == pytest.approx(4.0 * v, rel=1e-4)
    assert np.isfinite(frozen_noise_floor(prim, v, "amplitude"))
