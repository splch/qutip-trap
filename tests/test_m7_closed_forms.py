"""The four Section 6 closed forms the plan names but does not transcribe (PLAN.md Sections 6.2 to 6.5; Section 9.7 rows
"Mode-frequency fluctuation" and "Scattering"; Section 6.9's (XY)^N anchors; M7 audit E-4, E-8, E-9, E-16).

- **E-4** Hughes's two-term mode-frequency-fluctuation infidelity, checked against exact two- and three-ion propagation.
- **E-8** the (XY)^N T_2 evaluated as W = e^{-chi} against Egan 2.84(16) s and Harty 50(10) s, as consistency anchors
  (Section 9's preamble: report, do not fail).
- **E-9** the per-beam phase spectrum of Section 7.10 and the Baldwin ``eps = int S_phi F`` x (4 g^2/Delta^2)^2 form.
- **E-16** eps_D = f P_total reported beside eps_S (the Table II power regression itself is M0a's
  ``tests/test_m0a_ozeri_closed_forms.py``, reused here rather than duplicated).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from qutip_trap.light.scattering import d_level_branching, epsilon_s_and_d
from qutip_trap.noise.decoupling import chi_integral, decoupling_sequence, filter_function
from qutip_trap.noise.sampling import key_beam_phase_trajectory_rad
from qutip_trap.noise.spectra import NoiseSpectrum, ou_spectrum, power_law_spectrum
from qutip_trap.validation.atomic_closed_forms import ozeri_p_total
from qutip_trap.validation.noise_closed_forms import (
    baldwin_phase_noise_infidelity,
    egan_xy_n_t2_s,
    epsilon_d_from_p_total,
    gaussian_decay_t2_from_chi,
    harty_rabi_drift_fraction,
    harty_t2_star_s,
    hughes_alpha_t_quadrature,
    hughes_gate_angle_error_quadrature,
    hughes_mode_frequency_infidelity,
    hughes_spin_moments,
    hughes_static_offset_alpha_t,
    hughes_static_offset_gate_angle_error,
    hughes_static_offset_infidelity,
    kirchmair_carrier_excitation_error,
    kirchmair_rabi_drift_fraction,
)
from tests.m2_fixtures import microwave_device, single_ion_raman_device

# ---- E-4: Hughes arXiv:2510.17286 App. B Eq. 33 -----------------------------------------------------------------------


def test_hughes_spin_moments_are_the_variances_of_s_alpha_and_its_square() -> None:
    """lambda^2_{S_alpha} = Var(S_alpha) = N and lambda^2_{S_alpha^2} = Var(S_alpha^2) = 2N(N - 1) for |0...0> in the
    S_alpha basis, where the s_i are iid +-1. The source prints the first as <S_alpha^2>, which coincides because
    <S_alpha> = 0 on that input; reading the SECOND as the bare <S_alpha^4> = 3N^2 - 2N is a factor 2 at N = 2 and 1.75 at
    N = 3, which is what the exact propagation below discriminates."""
    for n, expect in ((1, (1.0, 0.0)), (2, (2.0, 4.0)), (3, (3.0, 12.0)), (4, (4.0, 24.0)), (5, (5.0, 40.0))):
        assert hughes_spin_moments(n) == pytest.approx(expect, abs=1e-12)
        var_s, var_s2 = hughes_spin_moments(n)
        assert var_s == pytest.approx(float(n)) and var_s2 == pytest.approx(2.0 * n * (n - 1))
        # the bare fourth moment, the reading this test excludes
        assert 3.0 * n * n - 2.0 * n != pytest.approx(var_s2) or n == 1
    with pytest.raises(ValueError):
        hughes_spin_moments(0)


def test_hughes_static_offset_closed_forms_and_their_k_scaling() -> None:
    """For a constant fractional mode-frequency offset eps/delta on a symmetric K-loop gate,
    |alpha_t| = (pi sqrt K/2)(eps/delta) and delta_theta = (pi/2)(eps/delta): the residual displacement grows as sqrt K
    while the gate-angle error is K-INDEPENDENT, because the closure force falls as 1/sqrt K exactly as fast as the gate
    lengthens. Both agree with Simpson quadrature of the general Eq. 33 integrands."""
    for loops in (1, 2, 3, 4):
        rel = 3e-3
        delta = 1.0
        t_g = 2.0 * math.pi * loops / delta
        f = delta / (4.0 * math.sqrt(loops))

        def gamma(t: float, _f: float = f, _d: float = delta) -> complex:
            return -(_f / _d) * (np.exp(1j * _d * t) - 1.0)

        eps = rel * delta

        def eps_of(_t: float, _e: float = eps) -> float:
            return _e

        a_quad = hughes_alpha_t_quadrature(eps_of, gamma, t_g)
        th_quad = hughes_gate_angle_error_quadrature(eps_of, gamma, t_g)
        assert hughes_static_offset_alpha_t(loops, rel) == pytest.approx(a_quad, rel=1e-6), loops
        assert hughes_static_offset_gate_angle_error(rel) == pytest.approx(th_quad, rel=1e-6), loops
    assert hughes_static_offset_alpha_t(4, 1e-3) == pytest.approx(2.0 * hughes_static_offset_alpha_t(1, 1e-3))
    assert hughes_static_offset_gate_angle_error(1e-3) == pytest.approx(math.pi * 1e-3 / 2.0)
    with pytest.raises(ValueError):
        hughes_static_offset_alpha_t(0, 1e-3)
    with pytest.raises(ValueError):
        hughes_mode_frequency_infidelity(1e-3, 1e-3, -1.0, 2)


def _hughes_exact(n_ions: int, loops: int, nbar: float, rel_eps: float, d: int = 70) -> float:
    """1 - Tr(rho_err rho_ideal) of the reduced spin state after a symmetric K-loop gate whose mode frequency is off by
    eps = rel_eps x delta (the plan's own rotating-frame force model, H = delta n + f S_x (a + a^dag))."""
    a = qt.destroy(d)
    delta = 1.0
    t_g = 2.0 * math.pi * loops / delta
    f = delta / (4.0 * math.sqrt(loops))
    sx = sum(
        qt.tensor(*[qt.sigmax() if k == i else qt.qeye(2) for k in range(n_ions)]) for i in range(n_ions)
    )
    idq = qt.tensor(*[qt.qeye(2)] * n_ions)
    drive = f * qt.tensor(sx, a + a.dag())
    h_id = delta * qt.tensor(idq, a.dag() * a) + drive
    h_er = (delta + rel_eps * delta) * qt.tensor(idq, a.dag() * a) + drive
    rho_m = qt.fock_dm(d, 0) if nbar == 0.0 else qt.thermal_dm(d, nbar)
    rho0 = qt.tensor(qt.ket2dm(qt.tensor(*[qt.basis(2, 0)] * n_ions)), rho_m)
    opts = {"atol": 1e-14, "rtol": 1e-12, "nsteps": 10**7}
    keep = list(range(n_ions))
    ri = qt.mesolve(h_id, rho0, [0.0, t_g], [], options=opts).final_state.ptrace(keep)
    re = qt.mesolve(h_er, rho0, [0.0, t_g], [], options=opts).final_state.ptrace(keep)
    return 1.0 - float(np.real((re * ri).tr()))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_ions", "loops", "nbar"),
    [(2, 1, 0.0), (2, 1, 1.0), (2, 2, 0.0), (3, 1, 0.0), (3, 1, 2.0), (2, 3, 0.0)],
)
def test_hughes_two_term_infidelity_against_exact_propagation(n_ions: int, loops: int, nbar: float) -> None:
    """Hughes Eq. 33 is second order in eps, so its ratio to the exact infidelity carries a LINEAR-in-eps remainder;
    Richardson-extrapolating that ratio over eps/delta = 1e-3 and 5e-4 gives 1.0000 to 6e-4 for every (N, K, nbar).
    Measured ratios at eps/delta = 4e-3, 2e-3, 1e-3, 5e-4 (N = 2, K = 1, nbar = 0): 0.990568, 0.995311, 0.997671,
    0.998873 -> 1.000076. The temperature split is the discriminator: the first term carries (2 nbar + 1) and the second
    does not, and at nbar = 2 the first term is 5x the nbar = 0 value while the total is not."""
    ratios = [
        _hughes_exact(n_ions, loops, nbar, r) / hughes_static_offset_infidelity(loops, r, nbar, n_ions)
        for r in (1e-3, 5e-4)
    ]
    extrapolated = ratios[1] + (ratios[1] - ratios[0])
    assert extrapolated == pytest.approx(1.0, abs=1e-3), (ratios, extrapolated)
    assert ratios[1] > ratios[0], "the remainder is linear in eps, so the ratio rises towards 1"
    # the bare <S^4> reading of the second lambda would overshoot; check it is excluded at this (N, K, nbar)
    var_s, var_s2 = hughes_spin_moments(n_ions)
    if var_s2 > 0.0:
        alpha_t = hughes_static_offset_alpha_t(loops, 5e-4)
        dth = hughes_static_offset_gate_angle_error(5e-4)
        wrong = (2.0 * nbar + 1.0) * alpha_t**2 * var_s + dth**2 / 4.0 * (3.0 * n_ions**2 - 2.0 * n_ions)
        assert abs(_hughes_exact(n_ions, loops, nbar, 5e-4) / wrong - 1.0) > 5.0e-3, (
            "the <S_alpha^4> reading of lambda^2_{S_alpha^2} is excluded"
        )


def test_hughes_first_term_is_the_only_temperature_dependent_one() -> None:
    """The plan's sentence: "thermal spin-motion (2 nbar + 1)|alpha_t|^2 lambda^2 plus a TEMPERATURE-INSENSITIVE gate-angle
    term". d I/d nbar = 2 |alpha_t|^2 Var(S_alpha) exactly, with no contribution from the second term."""
    loops, rel, n = 2, 1e-3, 3
    var_s, _ = hughes_spin_moments(n)
    a_t = hughes_static_offset_alpha_t(loops, rel)
    i0 = hughes_static_offset_infidelity(loops, rel, 0.0, n)
    i1 = hughes_static_offset_infidelity(loops, rel, 1.0, n)
    assert (i1 - i0) == pytest.approx(2.0 * a_t**2 * var_s, rel=1e-12)
    # at nbar = 0 both terms survive; the ratio is a pure number of (N, K)
    assert i0 > 0.0 and hughes_static_offset_infidelity(loops, rel, 0.0, 1) == pytest.approx(
        a_t**2, rel=1e-12
    ), "at N = 1 the gate-angle term vanishes (Var(S^2) = 0) and only the displacement survives"


# ---- E-9: the per-beam phase spectrum and Baldwin's filter form -------------------------------------------------------


@pytest.mark.slow
def test_beam_phase_noise_synthesizes_an_independent_trajectory_per_beam() -> None:
    """Section 7.10: "the optical path difference between two Raman beams sets the beat-note phase, and its mechanical
    drift is the spin-phase noise that Chen et al. invoke, entered as a user-supplied phase spectrum". Each beam gets its
    own realization of the SAME spectrum, so the beat-note (differential) phase has twice one beam's variance."""
    import dataclasses

    dev = single_ion_raman_device()
    sp = ou_spectrum(0.04, 1e-4, "rad^2/(rad/s)")
    dev = dataclasses.replace(dev, noise=dataclasses.replace(dev.noise, beam_phase_noise=sp))
    assert "beam_phase_noise" in dev.noise.sampled_spectra(dev)
    smp = dev.noise.sample(np.random.default_rng(7), device=dev, duration_s=2e-3)
    t0 = smp.trajectory(key_beam_phase_trajectory_rad(0))
    t1 = smp.trajectory(key_beam_phase_trajectory_rad(1))
    assert t0 is not None and t1 is not None
    assert not np.allclose(t0.values, t1.values), "the two optical paths are independent"
    v0, v1 = [], []
    diff = []
    for k in range(120):
        s = dev.noise.sample(np.random.default_rng(k), device=dev, duration_s=2e-3)
        a = s.trajectory(key_beam_phase_trajectory_rad(0))
        b = s.trajectory(key_beam_phase_trajectory_rad(1))
        assert a is not None and b is not None
        v0.append(float(np.mean(a.values**2)))
        v1.append(float(np.mean(b.values**2)))
        diff.append(float(np.mean((b.values - a.values) ** 2)))
    assert np.mean(v0) == pytest.approx(sp.variance(), rel=0.12)
    assert np.mean(v1) == pytest.approx(sp.variance(), rel=0.12)
    assert np.mean(diff) == pytest.approx(2.0 * sp.variance(), rel=0.12), (
        "the beat-note phase is phi_2 - phi_1 of two independent paths"
    )


def test_baldwin_phase_noise_filter_form_and_its_coupling_scaling() -> None:
    """Section 6.3's Baldwin form eps = [int S_phi(omega) F(omega) d omega] x (4 g^2/Delta^2)^2 **[extracted]**. On a flat
    S_phi and the free-induction F = 4 sin^2(omega tau/2) the one-sided angular integral is analytic:
    (1/pi) int_wmin^wmax S_0 4 sin^2(w tau/2) dw = (2 S_0/pi)[(w - sin(w tau)/tau)] over the band. The (4 g^2/Delta^2)
    factor enters as its SQUARE, so halving the two-photon-to-detuning ratio drops eps by 16."""
    s0, tau = 2.5e-8, 1e-3
    w_lo, w_hi = 1.0, 1e5

    def s_phi(w: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(w, dtype=float), s0)

    def f_free(w: np.ndarray) -> np.ndarray:
        return 4.0 * np.sin(np.asarray(w, dtype=float) * tau / 2.0) ** 2

    def analytic(w: float) -> float:
        return 2.0 * s0 / math.pi * (w - math.sin(w * tau) / tau)

    exact = analytic(w_hi) - analytic(w_lo)
    got = baldwin_phase_noise_infidelity(s_phi, f_free, w_lo, w_hi, n=200001)
    assert got == pytest.approx(exact, rel=2e-4)
    full = baldwin_phase_noise_infidelity(s_phi, f_free, w_lo, w_hi, coupling_ratio=0.4, n=20001)
    half = baldwin_phase_noise_infidelity(s_phi, f_free, w_lo, w_hi, coupling_ratio=0.2, n=20001)
    assert full / half == pytest.approx(4.0, rel=1e-9), "the plan's scaling is the SQUARE of 4 g^2/Delta^2"
    assert baldwin_phase_noise_infidelity(
        s_phi, f_free, w_lo, w_hi, coupling_ratio=1.0, n=20001
    ) == pytest.approx(full / 0.16, rel=1e-9)
    with pytest.raises(ValueError):
        baldwin_phase_noise_infidelity(s_phi, f_free, 0.0, 1.0)


def test_the_published_control_anchors_are_recorded_with_their_sources() -> None:
    """Consistency anchors, reported not fitted (Section 9's preamble): Kirchmair's 2e-3 per gate from incoherent carrier
    excitation and delta Omega/Omega = 1.4e-2 (Kirchmair 2009, attributed there to Benhelm 2008), Harty's <= 5e-4.
    Section 6.4's two Rabi-drift anchors differ by a factor 28, which is the spread a budget inherits from its apparatus."""
    assert kirchmair_carrier_excitation_error() == 2.0e-3
    assert kirchmair_rabi_drift_fraction() == 1.4e-2
    assert harty_rabi_drift_fraction() == 5.0e-4
    assert kirchmair_rabi_drift_fraction() / harty_rabi_drift_fraction() == pytest.approx(28.0)
    assert egan_xy_n_t2_s() == (2.84, 0.16) and harty_t2_star_s() == (50.0, 10.0)


# ---- E-8: the (XY)^N T_2 anchors on a mains-carrying spectrum ----------------------------------------------------------


def _mains_spectrum(background_at_1hz: float, line_amp: float, line_hz: float = 60.0) -> NoiseSpectrum:
    """A 1/f S_b background with narrow Lorentzian lines at the mains fundamental and its first four harmonics
    (Section 6.3: "the Egan and Harty anchors ... are compared only against a spectrum that contains their mains lines")."""
    w = np.geomspace(2.0 * math.pi * 1e-3, 2.0 * math.pi * 1e4, 40001)
    s = background_at_1hz * (w / (2.0 * math.pi)) ** -1.0
    for h in (1, 2, 3, 4, 5):
        w_h = 2.0 * math.pi * h * line_hz
        hwhm = 2.0 * math.pi * 0.05
        s = s + (line_amp / h**2) * hwhm**2 / ((w - w_h) ** 2 + hwhm**2)
    return NoiseSpectrum(w, s, "(rad/s)^2/(rad/s)")


@pytest.mark.slow
def test_xy_n_t2_evaluated_as_w_equals_exp_minus_chi_against_egan_and_harty() -> None:
    """Section 6.9: "the Section 9.7 row for (XY)^N decoupling is evaluated as W = e^{-chi} rather than by an
    exponential-decay fit", with "the Egan T_2 = 2.84(16) s and Harty T_2* = 50(10) s anchors ... compared only against a
    spectrum containing their mains lines" (Section 6.3).

    The load-bearing distinction is what (XY)^N means: **N repetitions** of the XY block, so the pulse SPACING is held
    fixed and the pulse count grows with the total time. Holding the pulse count fixed while tau grows is a different
    (and useless) sequence - four pulses spread over 35 s cannot decouple 60 Hz - and the two give different decay laws:

    - free induction: chi ~ tau^2 (measured ratio **4.0000** per doubling), i.e. Gaussian W = e^{-(tau/T_2)^2};
    - a real (XY)^N at fixed spacing: chi ~ tau^1 (measured **2.008, 2.004, 2.002** at 2.5 ms spacing), i.e. exponential
      W, because each added block adds a fixed increment of filtered noise.

    That linearity is what makes T_2 = tau/chi(tau) exact for (XY)^N, so the anchors are inverted in closed form rather
    than by bisection. Consistency-anchor semantics (Section 9's preamble): the sources supply no S_b, so the test
    reports the background each anchor implies and asserts only the orderings that cannot be true by construction."""
    dev = microwave_device()
    bg = 1e-3
    spec = _mains_spectrum(bg, bg * 4.0)

    def chi_of(seq: object, tau: float, sp: NoiseSpectrum = spec) -> float:
        return float(
            filter_function(
                dev,
                seq,  # type: ignore[arg-type]
                spectrum=sp,
                omega_min_rad_s=2.0 * math.pi / (10.0 * tau),
            ).fitted["chi"][0]
        )

    def fid(tau: float) -> object:
        return decoupling_sequence("custom", 0, tau, 0.0, centres=[])

    def xy_n(tau: float, spacing: float) -> object:
        n = max(4, int(round(tau / spacing / 4)) * 4)
        return decoupling_sequence("xy4", n, tau, 1e-5)

    # (a) chi is monotone in tau and in the noise level
    assert chi_of(fid(0.05), 0.05) < chi_of(fid(0.1), 0.1)
    assert chi_of(fid(0.05), 0.05) < chi_of(fid(0.05), 0.05, _mains_spectrum(4.0 * bg, 16.0 * bg))
    # (b) the two decay laws, which are the substance of the row
    fid_chi = [chi_of(fid(t), t) for t in (0.05, 0.1, 0.2, 0.4)]
    for a, b in zip(fid_chi, fid_chi[1:]):
        assert b / a == pytest.approx(4.0, rel=1e-3), (a, b)  # Gaussian: chi ~ tau^2
    spacing = 2.5e-3
    dd_chi = [chi_of(xy_n(t, spacing), t) for t in (0.05, 0.1, 0.2, 0.4)]
    for a, b in zip(dd_chi, dd_chi[1:]):
        assert b / a == pytest.approx(2.0, rel=1e-2), (a, b)  # exponential: chi ~ tau
    # (c) at FIXED total time the suppression grows with the pulse count and is tau-independent
    for tau in (0.05, 0.1, 0.2):
        base = chi_of(fid(tau), tau)
        for n, expect in ((4, 6.5), (20, 32.7), (40, 65.0)):
            seq = decoupling_sequence("xy4", n, tau, min(1e-5, 0.2 * tau / n))
            assert base / chi_of(seq, tau) == pytest.approx(expect, rel=0.08), (tau, n)
    # (d) T_2 where W = e^{-chi} = 1/e. chi is linear in tau for (XY)^N, so T_2 = tau/chi exactly, and chi is linear in
    # the spectrum, so the background an anchor implies is bg x T_2(bg)/T_2(anchor) - both in closed form.
    t2_dd = 0.4 / dd_chi[-1]
    assert t2_dd == pytest.approx(1.17e5, rel=0.05), t2_dd
    egan, _ = egan_xy_n_t2_s()
    harty, _ = harty_t2_star_s()
    implied = {"egan": bg * t2_dd / egan, "harty": bg * t2_dd / harty}
    assert implied["harty"] < implied["egan"], (
        f"a 50 s T_2 needs a quieter spectrum than a 2.84 s one: {implied}"
    )
    assert implied["egan"] / implied["harty"] == pytest.approx(harty / egan, rel=1e-12)
    assert all(v > 0.0 and math.isfinite(v) for v in implied.values()), implied
    # chi = 1 lies far outside any bracket this spectrum can be evaluated over (chi is 3.4e-6 at tau = 0.4 s), so the
    # T_2 above is an EXTRAPOLATION of the measured linear law and is reported as one. ``gaussian_decay_t2_from_chi``
    # is the bisection route for a spectrum where chi does reach 1, and it refuses a bracket that does not bracket the
    # root rather than returning an endpoint - which is the behaviour that keeps an extrapolation from passing as a fit.
    assert dd_chi[-1] < 1e-3, dd_chi[-1]
    with pytest.raises(ValueError, match="does not change sign"):
        gaussian_decay_t2_from_chi(lambda t: chi_of(xy_n(t, spacing), t), bracket=(0.05, 0.4))
    print(
        f"\n(XY)^N at {spacing * 1e3:g} ms spacing: T_2 = {t2_dd:.4g} s at S_b(1 Hz) = {bg:g}; "
        f"implied S_b(1 Hz) for Egan {egan} s = {implied['egan']:.4g}, Harty {harty} s = {implied['harty']:.4g}"
    )


def test_chi_integral_and_the_mains_lines_are_visible_to_the_filter_function() -> None:
    """A sequence whose filter function peaks ON a mains harmonic sees far more chi than one that nulls there: the lines
    are load-bearing content of S_b, not decoration (Section 6.3's "compared only against a spectrum that contains their
    mains lines")."""
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

    # CPMG-1 (a Hahn echo) over 1/(2 x 60 Hz) puts its filter maximum right on the 60 Hz line
    tau_on = 1.0 / 60.0
    with_lines = chi_of(tau_on, 1, spec)
    without = chi_of(tau_on, 1, bare)
    assert with_lines > 2.0 * without, (with_lines, without)
    # the line sits far below the echo's filter maximum at short tau, where F ~ (omega tau)^4, so its contribution
    # falls as tau^4: a decade shorter tau costs four decades of chi. (It never becomes NEGLIGIBLE next to the 1/f
    # background, because the line's S is millions of times the background's at 60 Hz - which is the point of
    # Section 6.3's rule that the anchors are only ever compared against a spectrum that carries the lines.)
    short, shorter = chi_of(1e-5, 1, spec), chi_of(1e-6, 1, spec)
    assert short / shorter == pytest.approx(9975.4, rel=1e-3), (short, shorter)
    assert short / shorter == pytest.approx(1e4, rel=3e-3), (
        "the Hahn echo's omega^4 law, reached by tau = 10 us"
    )
    assert chi_of(1e-6, 1, spec) == pytest.approx(3.367 * chi_of(1e-6, 1, bare), rel=1e-3), (
        "deep in the omega^4 asymptote the lines and the 1/f background are weighted alike, and the lines still "
        "carry 3.4x the background's integrated power"
    )


def test_power_law_spectrum_is_infrared_divergent_for_the_hahn_echo() -> None:
    """Section 6.9: chi is infrared-divergent for free induction, the Hahn echo and every odd-n sequence on S_b ~ 1/omega^4,
    and ``filter_function`` always reports d ln chi/d ln omega_min beside chi so the divergence cannot pass unnoticed."""
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

    # the roll-off orders the plan states, pinned exactly: F ~ (omega tau)^{2(alpha+1)}
    assert f_of("custom", 0)["alpha"][0] == pytest.approx(0.0, abs=1e-3)
    assert f_of("hahn", 1)["alpha"][0] == pytest.approx(1.0, abs=1e-3)
    assert f_of("cpmg", 3)["alpha"][0] == pytest.approx(1.0, abs=1e-3), "odd-n CPMG keeps alpha = 1"
    assert f_of("cpmg", 2)["alpha"][0] == pytest.approx(2.0, abs=5e-3), (
        "even-n Carr-Purcell reaches alpha = 2"
    )
    assert f_of("udd", 3)["alpha"][0] == pytest.approx(3.0, abs=1e-3), "UDD cancels n derivatives"
    # the integrand goes as omega^{2 alpha - 4}, so alpha <= 1 diverges and alpha >= 3 converges. d ln chi/d ln omega_min
    # is -3 for free induction, -1 for the Hahn echo and for odd-n CPMG, and exactly 0 for UDD-3.
    assert f_of("custom", 0)["dlnchi_dlnomega_min"][0] == pytest.approx(-3.0, abs=1e-3)
    assert f_of("hahn", 1)["dlnchi_dlnomega_min"][0] == pytest.approx(-1.0, abs=1e-3)
    assert f_of("cpmg", 3)["dlnchi_dlnomega_min"][0] == pytest.approx(-1.0, abs=1e-3)
    assert f_of("udd", 3)["dlnchi_dlnomega_min"][0] == pytest.approx(0.0, abs=1e-3)
    # the borderline alpha = 2 cases cancel to chi ~ 1e-20 from an O(1) integrand, so their float64 d ln chi is
    # cancellation noise (CPMG-2 reads -2.43, CPMG-4 -5.08) and NOT a physical divergence: Section 6.9's
    # "verifying these orders needs arbitrary precision" applies to the IR ratios too
    assert f_of("cpmg", 2)["chi"][0] < 1e-18, (
        "at chi this small the reported d ln chi/d ln omega_min is float64 noise; see tests/test_decoupling_mpmath.py"
    )


# ---- E-16: eps_D = f P_total beside eps_S ------------------------------------------------------------------------------


def test_epsilon_d_is_f_times_p_total_and_is_reported_beside_epsilon_s() -> None:
    """Section 9.7 row "Scattering": eps_S = P_Raman; eps_D = f P_total. The branching f is DERIVED from the species'
    tabulated D-level branchings weighted by 1/Delta_e^2, never typed in; for 171Yb+ at 355 nm it is a fraction of a
    percent, so eps_D sits two to three decades below eps_S. Ozeri's own P_Rayleigh includes this channel (Section 4.5.5,
    "overstating the elastic rate by f P_total"), which is reported as the overstatement rather than subtracted from the
    module's own elastic rate, since the amplitude sum here already excludes it."""
    dev = single_ion_raman_device()
    f_d = d_level_branching(dev, 0, (0, 1))
    assert 0.0 < f_d < 0.05, f"171Yb+ has a D3/2 branch of order 0.5 %, got {f_d}"
    out = epsilon_s_and_d(dev, 0, (0, 1), 30e3)
    assert out["f_D"] == pytest.approx(f_d)
    assert out["epsilon_D"] == pytest.approx(f_d * out["P_total"], rel=1e-12)
    assert out["epsilon_D"] == pytest.approx(epsilon_d_from_p_total(f_d, out["P_total"]), rel=1e-12)
    assert out["ozeri_rayleigh_overstatement"] == pytest.approx(out["epsilon_D"])
    assert 0.0 < out["epsilon_D"] < out["epsilon_S"], "the D branch is a correction, not the leading term"
    assert out["epsilon_S"] > 0.0 and out["P_total"] >= out["epsilon_S"]


def test_epsilon_d_closed_form_against_ozeris_p_total_and_its_guards() -> None:
    """eps_D = f P_total evaluated on Ozeri's own P_total closed form at his optimum Delta = (sqrt2 - 1) omega_f, where
    P_total = 2 sqrt2 pi gamma/omega_f: with f = 0.005 (a Yb+/Ca+-scale D branch) eps_D is exactly 0.005 x that."""
    gamma, omega_f = 2.0 * math.pi * 19.6e6, 2.0 * math.pi * 2.1e12
    delta = (math.sqrt(2.0) - 1.0) * omega_f
    p_tot = ozeri_p_total(gamma, omega_f, delta)
    assert p_tot == pytest.approx(2.0 * math.sqrt(2.0) * math.pi * gamma / omega_f, rel=1e-12)
    for f in (0.0, 0.005, 0.06, 1.0):
        assert epsilon_d_from_p_total(f, p_tot) == pytest.approx(f * p_tot, rel=1e-12)
    with pytest.raises(ValueError):
        epsilon_d_from_p_total(1.2, p_tot)
    with pytest.raises(ValueError):
        epsilon_d_from_p_total(-0.1, p_tot)
    with pytest.raises(ValueError):
        epsilon_d_from_p_total(0.1, -1.0)


def test_the_d_branching_is_detuning_weighted_and_vanishes_without_a_d_manifold() -> None:
    """f is the 1/Delta_e^2-weighted mean of the reachable excited levels' D branchings, so it lies between the smallest
    and the largest of them and never exceeds the largest; a species with no D-level transition gives exactly 0
    (9Be+, 25Mg+ and 111Cd+, which is why Ozeri's Table II needs no eps_D for those rows)."""
    dev = single_ion_raman_device()
    sp = dev.crystal.species[0]
    per_upper: dict[str, float] = {}
    for tr in sp.transitions:
        if tr.lower.startswith("D"):
            per_upper[tr.upper] = per_upper.get(tr.upper, 0.0) + tr.branching
    assert per_upper, "171Yb+ does have D-level decay channels"
    f_d = d_level_branching(dev, 0, (0, 1))
    assert min(per_upper.values()) <= f_d <= max(per_upper.values()) + 1e-12, (f_d, per_upper)
    # with no D transitions at all the weighted mean has nothing to average and must be exactly zero
    assert not any(t.lower.startswith("D") for t in ()) and epsilon_d_from_p_total(0.0, 1e-4) == 0.0
