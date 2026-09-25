"""The spin-motion integrals and the AM/FM/PM/Fourier pulse solvers against quadrature, closed forms and published anchors
(PLAN.md Section 4.4)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

from qutip_trap.control.shaping import (
    CHI_MAXIMAL_RAD,
    SINE_MOTION_PHASE_RAD,
    ClosureError,
    GateModes,
    SampledEnvelope,
    SegmentedEnvelope,
    closure_duration_s,
    closure_rabi_rad_s,
    envelope_of,
    frequency_derivative_residuals,
    gate_modes,
    integrals_sampled,
    integrals_segmented,
    scaled,
    segment_count,
    solve_amplitude_modulation,
    solve_fourier_amplitude_modulation,
    solve_frequency_modulation,
    solve_phase_modulation,
    symmetric_pulse,
    trajectory_sampled,
    waveform_from_segmented,
    waveform_integrals,
)
from qutip_trap.control.table import Waveform
from qutip_trap.light.beams import Beam
from qutip_trap.units import TWO_PI
from tests.fixtures import X_COM_TWO_IONS, chain_device, two_ion_modes
from tests.oracles import choi_segment_count, ms_two_body_angle

ONE_MODE = GateModes(
    ions=(0, 1), modes=(0,), omega_rad_s=(TWO_PI * 1.0e6,), eta={0: (0.05,), 1: (0.05,)}, nbar=(0.0,)
)


def test_square_pulse_closure_ratio_in_every_spin_normalization() -> None:
    """For one to three loops the square pulse closes at eta Omega/eps = 1/(2 sqrt K), tau = 2 pi K/eps and chi = pi/4 to 1e-12
    on the rwa kernel; the Choi kernel moves the ratio by under 0.5 % and the outside detuning flips the sign."""
    for loops in (1, 2, 3):
        sp = symmetric_pulse(ONE_MODE, gate_mode=0, loops=loops, epsilon_hz=10e3, kernel="rwa")
        assert sp.diagnostics["closure_ratio"] == pytest.approx(1.0 / (2.0 * math.sqrt(loops)), rel=1e-12)
        assert sp.waveform.duration_s == pytest.approx(closure_duration_s(TWO_PI * 10e3, loops), rel=1e-12)
        assert sp.chi_rad == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-12)
        assert abs(sp.integrals.alpha[(0, 0)]) < 1e-12 and abs(sp.integrals.alpha[(1, 0)]) < 1e-12
        omega = TWO_PI * sp.diagnostics["rabi_hz"]
        assert omega == pytest.approx(closure_rabi_rad_s(0.05, TWO_PI * 10e3, loops), rel=1e-12)
        assert ms_two_body_angle(0.05, 0.05, omega, TWO_PI * 10e3, loops) == pytest.approx(
            CHI_MAXIMAL_RAD, rel=1e-12
        )
    # the exact first-order (Choi) kernel keeps the counter-rotating term: a 0.25% correction at eps/nu = 1%, never mixed in
    sp_c = symmetric_pulse(ONE_MODE, gate_mode=0, loops=1, epsilon_hz=10e3, kernel="choi")
    assert 0.995 < sp_c.diagnostics["closure_ratio"] / 0.5 < 1.0
    # the detuning side sets the sign (exp(+i chi sigma sigma) = XX(-chi))
    sp_out = symmetric_pulse(
        ONE_MODE, gate_mode=0, loops=1, epsilon_hz=10e3, kernel="rwa", detuning_side="outside"
    )
    assert sp_out.chi_rad == pytest.approx(-CHI_MAXIMAL_RAD, rel=1e-12) and sp_out.sign == -1
    # a 1/(4 sqrt K) closure ratio is a chi = pi/16 pulse
    sp_q = symmetric_pulse(
        ONE_MODE, gate_mode=0, loops=1, epsilon_hz=10e3, kernel="rwa", chi_target_rad=math.pi / 16
    )
    assert sp_q.diagnostics["closure_ratio"] == pytest.approx(0.25, rel=1e-12)


def _direct_alpha_chi(env: SegmentedEnvelope, modes: GateModes, kernel: str) -> tuple[complex, float]:
    """The integrals by scipy quadrature on the piecewise-constant envelope, segment by segment."""
    omega = modes.omega_rad_s[0]
    mu = env.mu_rad_s
    edges = env.edges_s
    amps = env.amplitude_rad_s[0]

    def f(t: float) -> complex:
        if kernel == "choi":
            return math.cos(mu * t - env.phi_m_rad) * np.exp(1j * omega * t)
        return 0.5 * np.exp(1j * env.phi_m_rad) * np.exp(1j * (omega - mu) * t)

    def amp(t: float) -> float:
        k = min(int(np.searchsorted(edges, t, side="right") - 1), len(amps) - 1)
        return float(amps[max(k, 0)])

    re, _ = quad(lambda t: (amp(t) * f(t)).real, 0.0, edges[-1], points=list(edges[1:-1]), limit=2000)
    im, _ = quad(lambda t: (amp(t) * f(t)).imag, 0.0, edges[-1], points=list(edges[1:-1]), limit=2000)
    alpha = 1j * modes.eta[0][0] * (re + 1j * im)
    # chi: double integral t < t' of [Omega(t) Omega(t') + same] Im[f(t') conj f(t)] = 2 int int Omega Omega Im[...]
    total = 0.0
    for k in range(len(amps)):
        for j in range(k, len(amps)):
            val, _ = dblquad(
                lambda t, tp, k=k, j=j: (
                    2.0 * amps[k] * amps[j] * (f(tp) * np.conj(f(t))).imag if t < tp else 0.0
                ),
                edges[j],
                edges[j + 1],
                lambda tp, k=k: edges[k],
                lambda tp, k=k: min(edges[k + 1], tp),
                epsabs=1e-12,
                epsrel=1e-11,
            )
            total += val
    chi = modes.eta[0][0] * modes.eta[1][0] * total
    return complex(alpha), float(chi)


@pytest.mark.parametrize("kernel", ["rwa", "choi"])
@pytest.mark.parametrize("phi_m", [0.0, SINE_MOTION_PHASE_RAD, 0.7])
def test_segmented_integrals_match_direct_quadrature(kernel: str, phi_m: float) -> None:
    """The analytic segment formulas (F_k, the triangle T_k and Im(F_l F_k^*)) against scipy quadrature to 1e-9."""
    modes = GateModes(
        ions=(0, 1), modes=(0,), omega_rad_s=(TWO_PI * 1.5e6,), eta={0: (0.07,), 1: (0.05,)}, nbar=(0.0,)
    )
    env = SegmentedEnvelope(
        (7e-6, 5e-6, 9e-6), {0: (2.0e5, -1.3e5, 0.9e5), 1: (2.0e5, -1.3e5, 0.9e5)}, TWO_PI * 1.42e6, phi_m
    )
    ints = integrals_segmented(env, modes, kernel)
    alpha, chi = _direct_alpha_chi(env, modes, kernel)
    assert ints.alpha[(0, 0)] == pytest.approx(alpha, rel=1e-9, abs=1e-14)
    assert ints.chi_of(0, 1) == pytest.approx(chi, rel=1e-9)
    assert ints.chi_by_mode[(0, 1, 0)] == pytest.approx(chi, rel=1e-9)


def test_sampled_integrals_match_analytic_on_a_smooth_envelope() -> None:
    """The sampled (Simpson) integrals of a sin^2 envelope match the closed-form alpha to 1e-8 and the nested-quadrature chi to
    1e-7 on the rwa kernel."""
    omega = TWO_PI * 2.0e6
    mu = TWO_PI * 1.95e6
    eps = omega - mu
    tau = 80e-6
    modes = GateModes(
        ions=(0, 1), modes=(0,), omega_rad_s=(omega,), eta={0: (0.06,), 1: (0.06,)}, nbar=(0.0,)
    )
    t = np.linspace(0.0, tau, 8001)
    omega0 = TWO_PI * 150e3
    env_fn = omega0 * np.sin(math.pi * t / tau) ** 2
    env = SampledEnvelope(t, {0: env_fn, 1: env_fn}, mu * t)
    ints = integrals_sampled(env, modes, "rwa")

    # alpha = i eta int Omega (1/2) e^{i eps t}: closed form of int sin^2(pi t/tau) e^{i eps t}
    def sin2_exp(e: float) -> complex:
        w = math.pi / tau
        return 0.5 * (np.exp(1j * e * tau) - 1.0) / (1j * e) - 0.25 * (
            (np.exp(1j * (e + 2 * w) * tau) - 1.0) / (1j * (e + 2 * w))
            + (np.exp(1j * (e - 2 * w) * tau) - 1.0) / (1j * (e - 2 * w))
        )

    alpha_exact = 1j * 0.06 * omega0 * 0.5 * sin2_exp(eps)
    assert ints.alpha[(0, 0)] == pytest.approx(alpha_exact, rel=1e-8)

    # chi by nested quadrature of the smooth integrand
    def g(t1: float, t2: float) -> float:
        return 0.25 * math.sin(eps * (t2 - t1))

    val, _ = dblquad(
        lambda t1, t2: (
            2.0
            * omega0**2
            * math.sin(math.pi * t1 / tau) ** 2
            * math.sin(math.pi * t2 / tau) ** 2
            * g(t1, t2)
        ),
        0.0,
        tau,
        lambda t2: 0.0,
        lambda t2: t2,
        epsabs=1e-13,
        epsrel=1e-11,
    )
    assert ints.chi_of(0, 1) == pytest.approx(0.06 * 0.06 * val, rel=1e-7)


def test_am_solver_closes_every_mode_and_targets_pi_over_four() -> None:
    """Five equal segments close both x modes (|alpha| < 1e-10) at |chi| = pi/4 to 1e-10 and three cannot, with a second-ion
    amplitude ratio too; the waveform round-trips through envelope_of and obeys the s^2 law."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    assert modes.modes == (2, 3) and segment_count(modes.n_modes) == 5
    sp = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    assert sp.diagnostics["segments"] == 5 and sp.diagnostics["null_space_dimension"] == 1
    for key, a in sp.integrals.alpha.items():
        assert abs(a) < 1e-10, key
    assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-10)
    assert sp.waveform.chi_total_rad == pytest.approx(sp.chi_rad, rel=1e-12)
    assert sp.residual_error < 1e-20
    with pytest.raises(ClosureError):
        solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6, n_segments=3)
    ratio = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6, amplitude_ratio=0.8)
    assert abs(ratio.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-10)
    assert ratio.envelope.amplitude_rad_s[1] == pytest.approx(
        tuple(0.8 * x for x in ratio.envelope.amplitude_rad_s[0])
    )
    for key, a in ratio.integrals.alpha.items():
        assert abs(a) < 1e-10, key
    # the waveform round-trips through envelope_of and reproduces its integrals
    again = waveform_integrals(sp.waveform, modes)
    assert again.chi_of(0, 1) == pytest.approx(sp.chi_rad, rel=1e-10)
    env = envelope_of(sp.waveform, (0, 1))
    assert isinstance(env, SegmentedEnvelope)
    assert env.amplitude_rad_s[0] == pytest.approx(sp.envelope.amplitude_rad_s[0], rel=1e-12)
    # the s^2 law
    half = scaled(sp.waveform, math.sqrt(0.5))
    assert half.chi_total_rad == pytest.approx(0.5 * sp.chi_rad, rel=1e-12)
    assert waveform_integrals(half, modes).chi_of(0, 1) == pytest.approx(0.5 * sp.chi_rad, rel=1e-10)


def test_symmetric_constructor_equals_the_general_form_at_equal_envelopes() -> None:
    """Waveform.symmetric equals the one-segment AM solver at the closure duration (amplitudes and chi to 1e-9), with the legs
    at +-(omega_g - eps) and the sine motion phase splitting the leg phases by pi."""
    dev = chain_device(2)
    modes = two_ion_modes(dev).subset([X_COM_TWO_IONS])
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3, kernel="rwa")
    am = solve_amplitude_modulation(modes, mu_hz=3.0e6 - 20e3, duration_s=50e-6, n_segments=1, kernel="rwa")
    seg_wf = wf.segments[0]
    seg_am = am.waveform.segments[0]
    for key in seg_wf.amplitude_hz:
        assert float(seg_wf.amplitude_hz[key]) == pytest.approx(float(seg_am.amplitude_hz[key]), rel=1e-9)
    assert wf.chi_total_rad == pytest.approx(am.chi_rad, rel=1e-9)
    assert wf.ions == (0, 1) and wf.kind == "ms"
    # legs: blue at +(omega_g - eps), red at the negative; phases phi_s -/+ phi_m with the default phi_m = 0
    assert seg_wf.detuning_hz["blue"] == pytest.approx(3.0e6 - 20e3) and seg_wf.detuning_hz[
        "red"
    ] == pytest.approx(-(3.0e6 - 20e3))
    assert seg_wf.phase_rad[(0, "blue")] == seg_wf.phase_rad[(0, "red")] == 0.0
    sine = Waveform.symmetric(
        modes, gate_mode=X_COM_TWO_IONS, loops=1, epsilon_hz=20e3, phi_m_rad=SINE_MOTION_PHASE_RAD
    )
    assert sine.segments[0].phase_rad[(0, "blue")] - sine.segments[0].phase_rad[(0, "red")] == pytest.approx(
        math.pi
    )


def test_five_ion_closure_with_one_and_two_transverse_families() -> None:
    """Choi 2014: on five 171Yb+ ions 11 segments close the five x modes (|alpha| < 1e-9), and a Delta k with x and y
    components needs 21 for the ten modes, 11 raising ClosureError."""
    dev = chain_device(5, omega_hz=(3.045e6, 2.95e6, 0.55e6))
    modes = gate_modes(dev, (1, 3), (0, 1))
    assert modes.n_modes == 5 and set(modes.modes) == {5, 6, 7, 8, 9}
    assert choi_segment_count(5) == 11 == segment_count(modes.n_modes)
    sp = solve_amplitude_modulation(modes, mu_hz=2.98e6, duration_s=190e-6)
    for key, a in sp.integrals.alpha.items():
        assert abs(a) < 1e-9, key
    assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)
    assert sp.diagnostics["peak_rabi_hz"] < 2e6
    # both families through a rotated Delta k: rebuild the beams at 30 degrees from x in the xy plane
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    b1 = Beam(355e-9, (c, s, 0.0), (0.0, 0.0, 1.0), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    b2 = Beam(355e-9, (-c, -s, 0.0), (-s, c, 0.0), 200e-6, 10e-3, (0.0, 0.0, 0.0))
    dev2 = dataclasses.replace(
        dev, beams=(b1, b2), field=dataclasses.replace(dev.field, direction=(c, s, 0.0))
    )
    modes2 = gate_modes(dev2, (1, 3), (0, 1))
    assert modes2.n_modes == 10 and choi_segment_count(5, 2) == 21 == segment_count(10)
    with pytest.raises(ClosureError):
        solve_amplitude_modulation(modes2, mu_hz=2.98e6, duration_s=190e-6, n_segments=11)
    sp2 = solve_amplitude_modulation(modes2, mu_hz=2.98e6, duration_s=190e-6)
    for key, a in sp2.integrals.alpha.items():
        assert abs(a) < 1e-9, key
    assert abs(sp2.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)


def test_fourier_stabilized_solver_nulls_the_frequency_derivatives() -> None:
    """Blumel 2021: the stabilized Fourier pulse closes at chi = pi/4 and nulls every loop's first frequency derivative a
    thousandfold below the plain one, keeping a 1 kHz common shift's residual below 5 % of the plain pulse's."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    plain = solve_fourier_amplitude_modulation(
        modes, mu_hz=2.914e6, duration_s=100e-6, n_basis=16, stabilization_order=0
    )
    stab = solve_fourier_amplitude_modulation(
        modes, mu_hz=2.914e6, duration_s=100e-6, n_basis=16, stabilization_order=1
    )
    for sp in (plain, stab):
        assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-8)
        assert sp.residual_error < 1e-12
    assert isinstance(stab.envelope, SampledEnvelope)
    for m in modes.modes:
        res = frequency_derivative_residuals(stab.envelope, modes, 0, m, orders=1)
        res_plain = frequency_derivative_residuals(plain.envelope, modes, 0, m, orders=1)
        assert abs(res[0]) < 1e-9 and abs(res_plain[0]) < 1e-9, (m, res, res_plain)
        assert abs(res_plain[1]) > 1e3 * abs(res[1]), (m, res, res_plain)
    # sensitivity to a 1 kHz common shift of both modes
    shifted = dataclasses.replace(modes, omega_rad_s=tuple(w + TWO_PI * 1e3 for w in modes.omega_rad_s))
    err_plain = integrals_sampled(plain.envelope, shifted, "rwa").residual_error(shifted)
    err_stab = integrals_sampled(stab.envelope, shifted, "rwa").residual_error(shifted)
    assert err_stab < 0.05 * err_plain
    assert stab.diagnostics["constraint_rows"] == 8 and plain.diagnostics["constraint_rows"] == 4
    # 16 basis functions is what two modes need (four closure rows plus four derivative rows)
    assert stab.diagnostics["basis_functions"] == 16
    # Blumel's zero-temperature infidelity is Landsman's (4/5) of the residual error at nbar = 0
    assert stab.diagnostics["zero_temperature_infidelity"] == pytest.approx(0.8 * stab.residual_error)


def test_fm_solver_closes_and_the_robust_variant_averages_the_trajectory() -> None:
    """Leung 2018: both FM variants close both modes (residual < 1e-8) at chi = pi/4, and the robust one nulls the
    time-averaged trajectory so that a common 500 Hz drift costs it below 20 % of the plain pulse's residual."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    plain = solve_frequency_modulation(modes, duration_s=100e-6, n_vertices=9, mu0_hz=2.914e6, robust=False)
    robust = solve_frequency_modulation(modes, duration_s=100e-6, n_vertices=9, mu0_hz=2.914e6, robust=True)
    for sp in (plain, robust):
        assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-8)
        assert sp.residual_error < 1e-8, sp.diagnostics
        assert isinstance(sp.envelope, SampledEnvelope)
    for m in modes.modes:
        traj = trajectory_sampled(robust.envelope, modes, 0, m)
        mean = np.trapezoid(traj, robust.envelope.times_s) / 100e-6
        assert abs(mean) < 1e-3 * np.max(np.abs(traj))
    shifted = dataclasses.replace(modes, omega_rad_s=tuple(w + TWO_PI * 500.0 for w in modes.omega_rad_s))
    err_plain = integrals_sampled(plain.envelope, shifted, "rwa").residual_error(shifted)
    err_robust = integrals_sampled(robust.envelope, shifted, "rwa").residual_error(shifted)
    assert err_robust < 0.2 * err_plain
    # the waveform plays a callable detuning per leg and a constant amplitude
    seg = robust.waveform.segments[0]
    assert callable(seg.detuning_hz["blue"]) and callable(seg.detuning_hz["red"])
    assert seg.detuning_hz["blue"](0.0) == pytest.approx(-seg.detuning_hz["red"](0.0))
    env = envelope_of(robust.waveform, (0, 1))
    assert isinstance(env, SampledEnvelope)
    assert integrals_sampled(env, modes, "rwa").chi_of(0, 1) == pytest.approx(robust.chi_rad, rel=1e-5)


def test_fm_solver_closes_the_ion_at_a_mode_node() -> None:
    """With one gate ion at a node of a mode (eta = 0) the FM solver still closes every loop to 1e-9 from the other ion, and it
    skips a mode neither ion couples to."""
    modes = GateModes(
        ions=(0, 1),
        modes=(0, 1),
        omega_rad_s=(TWO_PI * 3.0e6, TWO_PI * 2.828e6),
        eta={0: (0.08, 0.0), 1: (0.07, 0.09)},
        nbar=(0.0, 0.0),
    )
    for robust in (False, True):
        sp = solve_frequency_modulation(modes, duration_s=100e-6, n_vertices=9, mu0_hz=2.914e6, robust=robust)
        assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)
        for key, alpha in sp.integrals.alpha.items():
            assert abs(alpha) < 1e-9, (robust, key, abs(alpha))
        assert sp.residual_error < 1e-12, sp.diagnostics
    # a mode NEITHER gate ion couples to is still skipped (there is nothing to close)
    dead = GateModes(
        ions=(0, 1),
        modes=(0, 1),
        omega_rad_s=(TWO_PI * 3.0e6, TWO_PI * 2.828e6),
        eta={0: (0.08, 0.0), 1: (0.07, 0.0)},
        nbar=(0.0, 0.0),
    )
    sp_dead = solve_frequency_modulation(dead, duration_s=100e-6, n_vertices=5, mu0_hz=2.914e6, robust=False)
    assert abs(sp_dead.integrals.alpha[(0, 1)]) == 0.0 and abs(sp_dead.integrals.alpha[(1, 1)]) == 0.0


@pytest.mark.parametrize(
    ("n_ions", "pair", "n_modes", "segments"),
    [(3, (0, 1), 3, 7), (3, (0, 2), 3, 7), (3, (1, 2), 3, 7), (4, (1, 2), 4, 9), (4, (0, 3), 4, 9)],
)
def test_three_and_four_ion_closure(n_ions: int, pair: tuple[int, int], n_modes: int, segments: int) -> None:
    """The transverse-x family of a 3- and a 4-ion 171Yb+ chain closes with Choi's 2N + 1 segments (|alpha| < 1e-14) at
    |chi| = pi/4, and two segments fewer raise ClosureError."""
    dev = chain_device(n_ions)
    modes = gate_modes(dev, pair, (0, 1))
    assert modes.n_modes == n_modes and segment_count(modes.n_modes) == segments
    sp = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6, pair=pair)
    assert int(sp.diagnostics["segments"]) == segments
    assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)
    for key, alpha in sp.integrals.alpha.items():
        assert abs(alpha) < 1e-14, (key, abs(alpha))
    assert sp.waveform.chi_total_rad == pytest.approx(sp.chi_rad, rel=1e-12)
    with pytest.raises(ClosureError):
        solve_amplitude_modulation(
            modes, mu_hz=2.914e6, duration_s=100e-6, n_segments=segments - 2, pair=pair
        )


def test_multi_pair_waveform_stores_the_solved_pairs_angles() -> None:
    """A pulse solved for pair (0, 2) of a three-ion GateModes stores that pair's chi = pi/4 while chi(0, 1) = 0.449272, and a
    multi-pair GateModes with no pair or an invalid one named is refused."""
    dev = chain_device(3)
    modes = gate_modes(dev, (0, 1, 2), (0, 1))
    assert abs(modes.eta[1][1]) < 1e-15, (
        "the centre ion sits at a node of the antisymmetric mode; the eigensolver leaves 1e-17, not an exact zero"
    )
    sp = solve_amplitude_modulation(modes, mu_hz=2.914e6, duration_s=100e-6, pair=(0, 2))
    assert sp.chi_rad == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)
    assert sp.waveform.chi_total_rad == pytest.approx(sp.chi_rad, rel=1e-12)
    assert sp.integrals.chi_of(0, 1) == pytest.approx(0.449272, rel=1e-5)
    assert sp.integrals.chi_of(0, 1) != pytest.approx(sp.chi_rad, rel=0.1)
    with pytest.raises(ValueError, match="name the pair"):
        waveform_from_segmented(sp.envelope, sp.integrals, modes)
    with pytest.raises(ValueError, match="not a pair"):
        waveform_from_segmented(sp.envelope, sp.integrals, modes, pair=(0, 7))


def test_phase_modulation_closes_every_mode_at_fixed_amplitude() -> None:
    """The PM solver's five constant-phase segments at fixed amplitude close both x modes (|alpha| < 1e-14) at |chi| = pi/4 and
    round-trip through the Waveform, while Milne 2020's three segments without the time symmetry leave the loops open."""
    dev = chain_device(2)
    modes = two_ion_modes(dev)
    sp = solve_phase_modulation(modes, mu_hz=2.914e6, duration_s=100e-6)
    assert int(sp.diagnostics["segments"]) == 5 and sp.method == "pm_segmented"
    assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)
    for key, alpha in sp.integrals.alpha.items():
        assert abs(alpha) < 1e-14, (key, abs(alpha))
    assert sp.residual_error < 1e-24
    # a fixed amplitude on every segment, and the phases really vary
    env = sp.envelope
    assert isinstance(env, SegmentedEnvelope)
    assert len(set(env.amplitude_rad_s[0])) == 1, "PM holds the amplitude fixed"
    assert env.phase_rad is not None and max(env.phase_rad) - min(env.phase_rad) > 1.0
    # the round trip: the stored waveform reproduces the integrals exactly (the phases are not collapsed to a sign)
    again = waveform_integrals(sp.waveform, modes)
    assert again.chi_of(0, 1) == pytest.approx(sp.chi_rad, rel=1e-12)
    for key, alpha in again.alpha.items():
        assert abs(alpha) < 1e-14, key
    back = envelope_of(sp.waveform, (0, 1))
    assert isinstance(back, SegmentedEnvelope) and back.phase_rad is not None
    assert back.phase_rad == pytest.approx(env.phase_rad, abs=1e-12)
    # Milne's N + 1 count, without the time symmetry that makes it work, leaves the loops open
    milne = solve_phase_modulation(modes, mu_hz=2.914e6, duration_s=100e-6, n_segments=3)
    assert max(abs(a) for a in milne.integrals.alpha.values()) > 1e-4


def test_fm_solver_closes_the_five_ion_leung_design() -> None:
    """Leung 2018's five-ion 90 us robust FM design closes all five x modes with 19 vertices (to 1e-10) at Omega = 292.3 kHz
    (5 %) but not with 13, and the plain design needs more than three times the Rabi frequency."""
    dev = chain_device(5, omega_hz=(3.045e6, 2.95e6, 0.55e6))
    modes = gate_modes(dev, (1, 3), (0, 1))
    assert modes.n_modes == 5
    thirteen = solve_frequency_modulation(modes, duration_s=90e-6, n_vertices=13, mu0_hz=2.98e6, robust=True)
    assert max(abs(a) for a in thirteen.integrals.alpha.values()) > 1e-2, (
        "13 vertices under-determine 5 modes with the robustness rows"
    )
    sp = solve_frequency_modulation(
        modes, duration_s=90e-6, n_vertices=19, mu0_hz=2.98e6, robust=True, max_nfev=800
    )
    assert abs(sp.chi_rad) == pytest.approx(CHI_MAXIMAL_RAD, rel=1e-9)
    for key, alpha in sp.integrals.alpha.items():
        assert abs(alpha) < 1e-10, (key, abs(alpha))
    assert sp.diagnostics["rabi_hz"] == pytest.approx(292.3e3, rel=0.05), (
        "the five-ion robust design's required Omega/2pi"
    )
    plain = solve_frequency_modulation(
        modes, duration_s=90e-6, n_vertices=19, mu0_hz=2.98e6, robust=False, max_nfev=800
    )
    assert plain.diagnostics["rabi_hz"] > 3.0 * sp.diagnostics["rabi_hz"], (
        "the robust design is the power-cheaper one here"
    )


def test_two_pulse_sign_reversal_closes_a_shaped_loop_at_two_tau() -> None:
    """Roos 2008: a smooth envelope leaves the loop open at tau, and repeating it with the coupling reversed closes it at 2 tau
    to 1e-6 of the single pulse's displacement."""
    omega = TWO_PI * 1.0e6
    eps = TWO_PI * 10e3
    tau = TWO_PI / eps
    modes = GateModes(
        ions=(0, 1), modes=(0,), omega_rad_s=(omega,), eta={0: (0.05,), 1: (0.05,)}, nbar=(0.0,)
    )
    t1 = np.linspace(0.0, tau, 4001)
    shape = TWO_PI * 100e3 * np.sin(math.pi * t1 / tau) ** 2
    single = SampledEnvelope(t1, {0: shape, 1: shape}, (omega - eps) * t1)
    a_single = integrals_sampled(single, modes, "rwa").alpha[(0, 0)]
    assert abs(a_single) > 1e-3
    t2 = np.linspace(0.0, 2 * tau, 8001)
    shape2 = (
        np.where(t2 <= tau, np.sin(math.pi * t2 / tau) ** 2, -(np.sin(math.pi * (t2 - tau) / tau) ** 2))
        * TWO_PI
        * 100e3
    )
    double = SampledEnvelope(t2, {0: shape2, 1: shape2}, (omega - eps) * t2)
    a_double = integrals_sampled(double, modes, "rwa").alpha[(0, 0)]
    assert abs(a_double) < 1e-6 * abs(a_single)
