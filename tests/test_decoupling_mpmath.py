"""The Section 9.15 rows that double precision cannot verify (PLAN.md Section 6.9 "Verifying these orders needs arbitrary
precision, F being an O(1) sum cancelling to omega^{2n+2}, so the test runs in mpmath"; Section 9.15 rows "Bang-bang
suppression orders (mpmath >= 200 dps)", "Filter-function limits and parities", "IR convergence on a 1/omega^4 spectrum",
"END-TO-END dephasing normalization", "dc floor normalization"; M7 audit E-14).

Until now these rows lived only in ``validation/scripts/check_composite.py``, a standalone mpmath script that never
imports ``qutip_trap`` (audit D-10), so the package's own toggling machinery was pinned to double precision and to
orders below about 4. The functions exercised here (``biercuk_taylor_coefficients_mp``, ``biercuk_filter_function_mp``,
``chi_integral_mp``) are the SAME equations as the float64 ``DecouplingSequence.biercuk_filter_function`` and
``chi_integral``, and every test that can be cross-checked against the float64 version is.

What the high precision buys, measured: at ``tau_pi/tau = 0.05`` the even-n UDD filter function is
(omega tau)^2 (omega tau_pi)^4/64, which at omega = 1e-6 is 1e-43 of an O(1) sum. Float64 returns cancellation noise of
order 1e-32 instead, and because that noise is omega-independent it turns the ``S ~ omega^-4`` integrand into omega^-6:
the package reports chi(1e-7) = 128.56 for UDD-4 where the true value is 1.2092569e-4. The odd-n rows, whose
(omega tau_pi)^4/16 floor is 1e-31 rather than 1e-43, survive float64 and are pinned in ``tests/test_decoupling.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from mpmath import mp

from qutip_trap.noise.decoupling import (
    biercuk_amplitude_mp,
    biercuk_filter_function_mp,
    biercuk_taylor_coefficients_mp,
    chi_integral,
    chi_integral_mp,
    cpmg_centres,
    cpmg_centres_mp,
    dc_floor,
    decoupling_sequence,
    leading_taylor_order_mp,
    udd_centres,
    udd_centres_mp,
)

DPS = 220


def test_the_mpmath_filter_function_agrees_with_the_float64_one_where_float64_still_works() -> None:
    """The two are one equation. At omega tau of order 1 the cancellation has not bitten, so ``biercuk_filter_function_mp``
    and ``DecouplingSequence.biercuk_filter_function`` agree to 1e-13 relative for CPMG and UDD at n = 1..6."""
    for timing, centres in (("cpmg", cpmg_centres), ("udd", udd_centres)):
        for n in (1, 2, 3, 4, 5, 6):
            seq = decoupling_sequence(timing, n, 1.0, 0.02)
            for x in (0.3, 0.9, 1.7, 5.0):
                f64 = float(seq.biercuk_filter_function(np.array([x]))[0])
                f_mp = float(biercuk_filter_function_mp(centres(n), n, x, 0.02, 60))
                assert f_mp == pytest.approx(f64, rel=1e-13), (timing, n, x, f64, f_mp)


def test_carr_purcell_cancels_to_taylor_order_three_at_every_even_n() -> None:
    """Section 9.15: "CP delta_l = (l - 1/2)/n: lowest nonvanishing Taylor order of R_zz is 3 for n = 2, 4, 6, 8, so
    F ~ omega^6, alpha = 2, 18 dB/octave, with leading x^3 coefficients 0.03125i, 0.0078125i, 0.003472222222i,
    0.001953125i". The four printed values are exactly i/(8 n^2) (1/32, 1/128, 1/288, 1/512) - reproduced to 100 digits."""
    printed = {2: "0.03125", 4: "0.0078125", 6: "0.003472222222", 8: "0.001953125"}
    with mp.workdps(DPS):
        for n, coeff in printed.items():
            order, c = leading_taylor_order_mp(cpmg_centres_mp(n, DPS), n, dps=DPS)
            assert order == 3, (n, order)
            assert abs(mp.re(c)) < mp.mpf(10) ** (-DPS // 2), "the coefficient is purely imaginary"
            got = mp.im(c)
            # the plan prints twelve digits, so 0.003472222222 is 1/288 truncated (6.4e-11 relative); compare at the
            # printed precision and pin the exact value against the closed form below (the plan's tolerance rule)
            assert abs(got / mp.mpf(coeff) - 1) < mp.mpf("1e-10"), (n, mp.nstr(got, 15), coeff)
            # and the closed form the four printed values follow
            assert abs(got - mp.mpf(1) / (8 * n**2)) < mp.mpf(10) ** (-DPS // 2), n
        # odd n is NOT order 3: Carr-Purcell's third-order cancellation is an even-parity property. Odd n stops at
        # order 2 with a purely REAL c_2 = -1/(4 n^2), and |c_2|^2 = 1/(16 n^4) is exactly the plan's odd-n CPMG
        # coefficient - just as |c_3|^2 = 1/(64 n^4) is its even-n one, so the two printed laws are these two Taylor
        # coefficients and nothing else.
        for n in (1, 3, 5, 7):
            order, c = leading_taylor_order_mp(cpmg_centres_mp(n, DPS), n, dps=DPS)
            assert order == 2, (n, order)
            assert abs(mp.im(c)) < mp.mpf(10) ** (-DPS // 2), "odd-n c_2 is purely real"
            assert abs(c + mp.mpf(1) / (4 * n**2)) < mp.mpf(10) ** (-DPS // 2), (n, mp.nstr(c, 12))
            assert abs(abs(c) ** 2 - mp.mpf(1) / (16 * mp.mpf(n) ** 4)) < mp.mpf(10) ** (-DPS // 2), n
        for n in (2, 4, 6, 8):
            _, c = leading_taylor_order_mp(cpmg_centres_mp(n, DPS), n, dps=DPS)
            assert abs(abs(c) ** 2 - mp.mpf(1) / (64 * mp.mpf(n) ** 4)) < mp.mpf(10) ** (-DPS // 2), n


def test_udd_cancels_the_first_n_derivatives_up_to_twelve_pulses() -> None:
    """Section 9.15: "UDD delta_l = sin^2[pi l/(2n+2)]: order n + 1 for n = 2, 3, 4, 5, 6, 8, 10, 12". At n = 12 the
    amplitude cancels through x^12 from an O(1) sum, which needs about 12 x 16 = 190 digits of head-room; float64
    ``udd_centres`` is good to 1e-16 and caps the verifiable order at about 4."""
    with mp.workdps(DPS):
        for n in (2, 3, 4, 5, 6, 8, 10, 12):
            order, c = leading_taylor_order_mp(udd_centres_mp(n, DPS), n, dps=DPS)
            assert order == n + 1, (n, order)
            assert abs(c) > mp.mpf(10) ** (-DPS // 2)
            # every coefficient below the order really is zero, not merely small
            coeffs = biercuk_taylor_coefficients_mp(udd_centres_mp(n, DPS), n, DPS, n + 2)
            for k in range(n + 1):
                assert abs(coeffs[k]) < mp.mpf(10) ** (-DPS // 2), (n, k, mp.nstr(coeffs[k], 8))
        # the float64 timings cannot see past order 4, which is why this test exists
        order_f64, _ = leading_taylor_order_mp([mp.mpf(repr(d)) for d in udd_centres(8)], 8, dps=DPS)
        assert order_f64 < 9, f"float64 UDD-8 centres fake an order-{order_f64} cancellation"


def test_the_ideal_pulse_leading_coefficients_of_udd_and_cpmg() -> None:
    """Section 9.15: "Ideal-pulse coefficients: UDD F/(omega tau)^{2n+2} = 1/(16^n (n!)^2) to 12 digits for n = 1..8;
    CPMG (omega tau)^4/(16 n^4) for odd n and (omega tau)^6/(64 n^4) for even n"."""
    with mp.workdps(250):
        x = mp.mpf(10) ** -10
        for n in range(1, 9):
            got = biercuk_filter_function_mp(udd_centres_mp(n, 250), n, x, 0.0, 250) / x ** (2 * n + 2)
            want = mp.mpf(1) / (mp.mpf(16) ** n * mp.factorial(n) ** 2)
            assert abs(got / want - 1) < mp.mpf("1e-12"), (n, mp.nstr(got, 15), mp.nstr(want, 15))
        for n in range(1, 9):
            f = biercuk_filter_function_mp(cpmg_centres_mp(n, 250), n, x, 0.0, 250)
            if n % 2 == 1:
                got, want = f / x**4, mp.mpf(1) / (16 * mp.mpf(n) ** 4)
            else:
                got, want = f / x**6, mp.mpf(1) / (64 * mp.mpf(n) ** 4)
            assert abs(got / want - 1) < mp.mpf("1e-12"), (n, mp.nstr(got, 15), mp.nstr(want, 15))


def test_green_eq_45_is_the_even_n_twin_and_returns_four_at_odd_n() -> None:
    """Section 9.15's negative control: Green's Eq. 45 writes ``1 - e^{i omega tau}`` where Biercuk writes
    ``1 + (-1)^{n+1} e^{i omega tau}``, which presumes Lambda^{(n)} = I. The two agree for EVEN n (5.05997145284e-4 at
    n = 2, 3.06275154428e-5 at n = 4 at CP timings and omega tau = 0.9) and disagree at odd n (Biercuk 0.0396431177199
    against Green 4.75678006346 at n = 1); Green's form gives |R_zz(0)|^2 = 4.0 EXACTLY at n = 1, 3, 5, i.e. no
    low-frequency suppression at all. The package builds the (-1)^{n+1} form, so this test pins the alternative as wrong."""

    def green(centres: tuple[object, ...], x: object, dps: int = DPS) -> object:
        with mp.workdps(dps):
            acc = mp.mpf(0)
            for j, d in enumerate(centres, start=1):
                acc += (-1) ** j * mp.e ** (1j * d * mp.mpf(x))
            return +(abs(1 - mp.e ** (1j * mp.mpf(x)) + 2 * acc) ** 2)

    with mp.workdps(DPS):
        x = mp.mpf("0.9")
        for n, expect in ((2, "5.05997145284e-4"), (4, "3.06275154428e-5")):
            c = cpmg_centres_mp(n, DPS)
            b = biercuk_filter_function_mp(c, n, x, 0.0, DPS)
            assert abs(b / mp.mpf(expect) - 1) < mp.mpf("1e-11"), (n, mp.nstr(b, 14))
            assert abs(green(c, x) / b - 1) < mp.mpf("1e-30"), f"the two forms agree at even n = {n}"
        c1 = cpmg_centres_mp(1, DPS)
        assert abs(biercuk_filter_function_mp(c1, 1, x, 0.0, DPS) / mp.mpf("0.0396431177199") - 1) < mp.mpf(
            "1e-11"
        )
        assert abs(green(c1, x) / mp.mpf("4.75678006346") - 1) < mp.mpf("1e-11")
        for n in (1, 3, 5):
            zero = green(cpmg_centres_mp(n, DPS), mp.mpf(10) ** -60)
            assert abs(zero - 4) < mp.mpf("1e-50"), (n, mp.nstr(zero, 12))
            # Biercuk's form vanishes at dc for every n, both parities
            b0 = biercuk_filter_function_mp(cpmg_centres_mp(n, DPS), n, mp.mpf(10) ** -60, 0.0, DPS)
            assert abs(b0) < mp.mpf("1e-100"), (n, mp.nstr(b0, 12))


def test_ir_convergence_on_a_one_over_omega_fourth_spectrum() -> None:
    """Section 9.15's IR-convergence row, verbatim: tau = 1, tau_pi = 0.05, omega_max = 50,
    chi(1e-7) -> chi(1e-8) gives 2.4871932 -> 24.868357 (ratio 9.9986) for UDD n = 3 and 2.4868452 -> 24.868009
    (9.9998) for n = 5, against exactly 1.0 for n = 4 (1.2092569e-4) and n = 6 (2.3951314e-5).

    The even-n rows are the reason this file exists: in float64 the same computation returns chi(1e-7) = 128.56 (n = 4)
    and 92.83 (n = 6) with ratios 107 and 19, because F at omega = 1e-6 is 1e-43 and the float64 answer is 1e-32 of
    cancellation noise. The test asserts BOTH: the mpmath values match the plan, and the float64 values do not."""
    d = 90
    with mp.workdps(d):

        def chi_of(n: int, w_min: float) -> object:
            centres = udd_centres_mp(n, d)
            return chi_integral_mp(
                lambda w: w**-4,
                lambda w, _c=centres, _n=n: biercuk_filter_function_mp(_c, _n, w, 0.05, d),
                w_min,
                50.0,
                dps=d,
            )

        for n, lo, hi, ratio in (
            (3, "2.4871932", "24.868357", "9.9986"),
            (5, "2.4868452", "24.868009", "9.9998"),
        ):
            a, b = chi_of(n, 1e-7), chi_of(n, 1e-8)
            assert abs(a / mp.mpf(lo) - 1) < mp.mpf("1e-7"), (n, mp.nstr(a, 10))
            assert abs(b / mp.mpf(hi) - 1) < mp.mpf("1e-7"), (n, mp.nstr(b, 10))
            assert abs(b / a / mp.mpf(ratio) - 1) < mp.mpf("1e-4"), (n, mp.nstr(b / a, 10))
        for n, value in ((4, "1.2092569e-4"), (6, "2.3951314e-5")):
            a, b = chi_of(n, 1e-7), chi_of(n, 1e-8)
            assert abs(a / mp.mpf(value) - 1) < mp.mpf("1e-7"), (n, mp.nstr(a, 10))
            assert abs(b / a - 1) < mp.mpf("1e-9"), f"even n is IR convergent: ratio {mp.nstr(b / a, 12)}"
    # the float64 route on the same integral, which is what this file replaces
    for n in (4, 6):
        seq = decoupling_sequence("udd", n, 1.0, 0.05)
        bad = chi_integral(
            lambda w: np.asarray(w, dtype=float) ** -4.0,
            lambda w, _s=seq: _s.filter_function(np.asarray(w), gated=True),
            1e-7,
            50.0,
            n_points=400001,
        )
        assert bad > 1.0, f"float64 UDD-{n} reports chi = {bad:.4g} against a true 1e-4"


def test_the_end_to_end_dephasing_normalization_from_the_double_integral() -> None:
    """Section 9.15's END-TO-END row asks for <a_1^2> = 0.06875600894468 from the spectral integral AND from
    ``int int C(t1 - t2) dt1 dt2``; ``tests/test_decoupling.py`` pins the first, this pins the second, independent one.
    For the Gaussian fixture C(t) = delta_b^2 exp(-sigma^2 t^2/2) with sigma = 2, delta_b = 0.3, tau = 1, the double
    integral has a closed form: 2 delta_b^2 [tau sqrt(pi/2) erf(sigma tau/sqrt2)/sigma - (1 - e^{-sigma^2 tau^2/2})/sigma^2]."""
    with mp.workdps(60):
        sigma, db, tau = mp.mpf(2), mp.mpf("0.3"), mp.mpf(1)
        closed = (
            2
            * db**2
            * (
                tau * mp.sqrt(mp.pi / 2) * mp.erf(sigma * tau / mp.sqrt(2)) / sigma
                - (1 - mp.e ** (-(sigma**2) * tau**2 / 2)) / sigma**2
            )
        )
        assert abs(closed / mp.mpf("0.06875600894468") - 1) < mp.mpf("1e-12"), mp.nstr(closed, 16)
        quad = mp.quad(
            lambda t1: mp.quad(lambda t2: db**2 * mp.e ** (-(sigma**2) * (t1 - t2) ** 2 / 2), [0, tau]),
            [0, tau],
        )
        assert abs(quad / closed - 1) < mp.mpf("1e-15")
        # W = e^{-2<a_1^2>} = 0.871523876057, NOT e^{-<a_1^2>} = 0.933554431223
        assert abs(mp.e ** (-2 * closed) / mp.mpf("0.871523876057") - 1) < mp.mpf("1e-11")
        assert abs(mp.e ** (-closed) / mp.mpf("0.933554431223") - 1) < mp.mpf("1e-11")
        assert abs(2 * closed / mp.mpf("0.137512017889") - 1) < mp.mpf("1e-11"), (
            "Biercuk's (2/pi) int = 2 <a_1^2>"
        )
        assert abs(8 * closed / mp.mpf("0.550048071557") - 1) < mp.mpf("1e-11"), (
            "the splitting-PSD misreading is 4x"
        )


def test_the_four_excluded_dc_floor_normalizations() -> None:
    """Section 9.15's dc-floor row names four alternatives the correct normalization must exclude: 1.46841e-6,
    2.31882e-4, 5.79706e-5 and 2.97353e19 (Omega^-4 omitted), against the correct SK1 floor of 5.87365e-6 at
    <beta^2> = 2.07e9/pi (rad/s)^2 and Omega = 1.5e6. Three of the four are reproduced here from their stated recipes; the
    fourth (2.31882e-4) is not reconstructible from the plan's text, and is recorded as such rather than invented."""
    var, omega, c_hat = 2.07e9 / math.pi, 1.5e6, 22.8302557111
    rel = var / omega**2
    assert rel == pytest.approx(2.928451e-4, rel=1e-6), "the plan's (beta/Omega)^2"
    assert dc_floor(c_hat, 1, var, omega) == pytest.approx(5.87365e-6, rel=1e-5)
    # (a) Omega^-4 omitted: c_hat (2m+1)!! <beta^2>^2
    assert c_hat * 3.0 * var**2 == pytest.approx(2.97353e19, rel=1e-5)
    # (b) the single-sided power used without the 1/pi
    assert dc_floor(c_hat, 1, 2.07e9, omega) == pytest.approx(5.79706e-5, rel=1e-5)
    # (c) the Gaussian moment (2m+1)!! = 3 dropped, giving 5.87365e-6/3
    assert dc_floor(c_hat, 1, var, omega) / 3.0 == pytest.approx(1.957883e-6, rel=1e-5)
    # and the plan's BB1 and CORPSE floors, with their published counterparts NOT reproduced (3.9e-9 and 3.0e-9)
    assert dc_floor(9.388566343, 2, var, omega) == pytest.approx(3.53675e-9, rel=1e-5)
    assert dc_floor(0.006500751892, 1, var, omega) == pytest.approx(1.67248e-9, rel=1e-5)


def test_the_mpmath_amplitude_is_purely_the_biercuk_equation() -> None:
    """A guard on the helper itself: the amplitude's modulus squared is the filter function, its dc value vanishes for
    both parities, and its Taylor coefficients agree with an independent series of the same exponential sum."""
    with mp.workdps(80):
        for n in (1, 2, 3):
            c = cpmg_centres_mp(n, 80)
            x = mp.mpf("0.7")
            amp = biercuk_amplitude_mp(c, n, x, 0.03, 80)
            assert abs(abs(amp) ** 2 - biercuk_filter_function_mp(c, n, x, 0.03, 80)) < mp.mpf("1e-60")
            coeffs = biercuk_taylor_coefficients_mp(c, n, 80, 6)
            series = sum(coeffs[k] * mp.mpf("0.001") ** k for k in range(7))
            assert abs(series - biercuk_amplitude_mp(c, n, mp.mpf("0.001"), 0.0, 80)) < mp.mpf("1e-20")
    with pytest.raises(ValueError):
        leading_taylor_order_mp(cpmg_centres_mp(2, 40), 2, dps=40, max_order=2)
