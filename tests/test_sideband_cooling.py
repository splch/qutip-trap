"""Pulsed Raman sideband cooling: the Laguerre nodes of the exact sideband Rabi frequencies, the transfer matrices and
their trapping condition, the Che, Rasmusson, Home and Monroe schedules, the NIST 90-degree Raman geometry, the repump
kernel, and the exactly thermal sideband ratio."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import eval_genlaguerre

from qutip_trap.dynamics.operators import rabi_matrix_element
from qutip_trap.light.recoil import minimal_quadrature
from qutip_trap.prep.sideband import (
    SidebandPulse,
    apply_pulses,
    mean_occupation,
    optimize_durations,
    optimize_per_order,
    pi_time_s,
    repump_kernel,
    sideband_rabi_rad_s,
    thermal_distribution,
    transfer_matrix,
)
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI

OM0 = TWO_PI * 150e3


def _laguerre_first_zero(order: int, eta: float) -> float:
    """The smallest continuous degree n' > 0 at which L^{(k)}_{n'}(eta^2) vanishes (scipy's real-degree Laguerre)."""
    x = eta * eta

    def f(n: float) -> float:
        return float(eval_genlaguerre(n, order, x))

    a, fa = 0.0, f(0.0)
    while a < 1e4:
        b = a + 0.5
        fb = f(b)
        if fa * fb <= 0.0:
            return float(brentq(f, a, b, xtol=1e-10))
        a, fa = b, fb
    raise RuntimeError("no Laguerre zero found below n = 1e4")


def _stranded_index(order: int, eta: float) -> int:
    """The Fock state whose k-th red-sideband Rabi frequency sits at the first zero: round(zero + k), n' = n - k."""
    return int(round(_laguerre_first_zero(order, eta) + order))


def _stranded_population(p: np.ndarray, order: int, eta: float) -> float:
    """Population at and above the Rabi node of the given order, which that order's pulses cannot remove."""
    return float(np.sum(p[_stranded_index(order, eta) :]))


def test_laguerre_first_zeros_and_the_stranded_fock_indices() -> None:
    """The first Laguerre zeros are 39.7908 and 71.7720 (eta = 0.3, k = 1, 2) and 112.2895 and 202.0112 (eta = 0.18) to
    1e-3, stranding n = 41, 74 and 113 (Che's printed 40 and 72 are degree labels)."""
    assert _laguerre_first_zero(1, 0.3) == pytest.approx(39.7908, abs=1e-3)
    assert _laguerre_first_zero(2, 0.3) == pytest.approx(71.7720, abs=1e-3)
    assert _laguerre_first_zero(1, 0.18) == pytest.approx(112.2895, abs=1e-3)
    assert _laguerre_first_zero(2, 0.18) == pytest.approx(202.0112, abs=1e-3)
    assert (_stranded_index(1, 0.3), _stranded_index(2, 0.3), _stranded_index(1, 0.18)) == (41, 74, 113)
    # the exact element vanishes at the node: Omega_{41,40} is the smallest first-sideband Rabi frequency near there
    om = [sideband_rabi_rad_s(OM0, 0.3, n, 1) for n in range(30, 55)]
    assert 30 + int(np.argmin(om)) in (40, 41)


def test_transfer_matrix_is_column_stochastic_and_composes_conserving_probability() -> None:
    w = transfer_matrix(151, 0.3, OM0, 1, 9e-6)
    assert np.max(np.abs(w.sum(axis=0) - 1.0)) < 1e-14
    assert np.allclose(w[:1, :1], 1.0)  # the absorbing block n < k
    w2 = transfer_matrix(151, 0.3, OM0, 2, 9e-6)
    assert np.allclose(w2[:2, :2], np.eye(2)) and np.max(np.abs(w2.sum(axis=0) - 1.0)) < 1e-14
    p0 = thermal_distribution(15.36, 151)
    assert 1.0 - p0.sum() == pytest.approx((15.36 / 16.36) ** 151, rel=1e-9)
    assert 1.0 - p0.sum() == pytest.approx(7e-5, rel=0.1)  # n_max = 150: the truncated tail
    p25 = apply_pulses(p0, [SidebandPulse(1, 9e-6)] * 25, 0.3, OM0)
    assert p25.sum() == pytest.approx(p0.sum(), abs=4e-16 * 25)


def test_trapping_condition_is_2m_pi_in_the_plans_rabi_convention() -> None:
    """A pulse with Omega_{n-1,n} t = 2 pi leaves |n> untouched and one with Omega t = pi (the sources' half-Rabi
    trapping time) moves it to |n-1>, both to 1e-12."""
    n = 7
    t_trap = 2.0 * math.pi / sideband_rabi_rad_s(OM0, 0.2, n, 1)
    p = np.zeros(20)
    p[n] = 1.0
    out = apply_pulses(p, [SidebandPulse(1, t_trap)], 0.2, OM0)
    assert out[n] == pytest.approx(1.0, abs=1e-12)
    out_half = apply_pulses(p, [SidebandPulse(1, t_trap / 2.0)], 0.2, OM0)
    assert out_half[n] == pytest.approx(0.0, abs=1e-12) and out_half[n - 1] == pytest.approx(1.0, abs=1e-12)


def test_che_2017_single_order_cooling_accumulates_population_just_above_the_rabi_node() -> None:
    """At Che 2017's eta = 0.3 and nbar = 17 first- and second-order cycles pile population at n = 43, 42, 41 and 77,
    74, 74 after 40, 150 and 500 pulses, and his 75-pulse multiorder schedule leaves nbar < 1 with p0 > 0.95."""
    p0 = thermal_distribution(17.0, 151)
    lo = int(math.floor(_laguerre_first_zero(1, 0.3)))  # the node's lower shoulder
    peaks = []
    centres = []
    for count in (40, 150, 500):
        p = apply_pulses(p0, [SidebandPulse(1, 9e-6)] * count, 0.3, OM0)
        peaks.append(30 + int(np.argmax(p[30:])))
        centres.append(float(np.dot(np.arange(lo, p.size), p[lo:])) / float(np.sum(p[lo:])))
    assert peaks == [43, 42, 41]
    assert centres[0] > centres[1] > centres[2] > _stranded_index(1, 0.3) - 0.5
    peaks2 = []
    for count in (40, 150, 500):
        p = apply_pulses(p0, [SidebandPulse(2, 9e-6)] * count, 0.3, OM0)
        peaks2.append(60 + int(np.argmax(p[60:])))
    assert peaks2 == [77, 74, 74]
    # Che's 75-pulse multiorder schedule (15 x k=2, 15 x k=1, 15 x k=2, 30 x k=1) empties the node region
    schedule = (
        [SidebandPulse(2, 9e-6)] * 15
        + [SidebandPulse(1, 9e-6)] * 15
        + [SidebandPulse(2, 9e-6)] * 15
        + [SidebandPulse(1, 9e-6)] * 30
    )
    p = apply_pulses(p0, schedule, 0.3, OM0)
    assert mean_occupation(p) < 1.0 and p[0] > 0.95
    assert _stranded_population(p, 1, 0.3) < 0.02


def test_rasmusson_2021_fixed_pulse_prediction_and_multiorder_schedules() -> None:
    """From Rasmusson 2021's nbar = 14.6, 25 first-order pulses of one optimized duration reach 2.99 to 4.15 (his
    3.57(58)) and pi-time pulses 3.15 (0.1); 50 multiorder pulses from 15.36 reach below 1, while single-order cooling
    strands the thermal tail above n = 113 (5 %)."""
    om0 = TWO_PI * 100e3
    p0 = thermal_distribution(14.6, 200)
    shared, nbar = optimize_per_order(p0, {1: 25}, 0.18, om0, (1e-7, 200e-6))
    t = shared[0].duration_s
    assert 2.99 < nbar < 3.57 + 0.58
    p_pi = apply_pulses(p0, [SidebandPulse(1, pi_time_s(om0, 0.18, 15, 1))] * 25, 0.18, om0)
    assert mean_occupation(p_pi) == pytest.approx(3.15, abs=0.1)
    p0b = thermal_distribution(15.36, 220)
    pulses, nb_order = optimize_per_order(p0b, {3: 17, 2: 17, 1: 16}, 0.18, om0, (1e-7, 200e-6))
    assert [pl.order for pl in pulses[:17]] == [3] * 17 and nb_order < 1.0
    p_single = apply_pulses(p0b, [SidebandPulse(1, t)] * 50, 0.18, om0)
    tail = _stranded_population(p_single, 1, 0.18)
    assert tail == pytest.approx((15.36 / 16.36) ** 113, rel=0.05)
    assert float(np.dot(np.arange(113, 220), p_single[113:])) < 0.15


@pytest.mark.slow
def test_rasmusson_independent_durations_reach_a_tenth_of_a_quantum() -> None:
    """Independently optimized durations bring the 50-pulse schedule below nbar = 0.15 (Rasmusson's optimized 0.06 is
    not reached)."""
    om0 = TWO_PI * 100e3
    p0 = thermal_distribution(15.36, 220)
    pulses, _ = optimize_per_order(p0, {3: 17, 2: 17, 1: 16}, 0.18, om0, (1e-7, 200e-6))
    durations, nbar = optimize_durations(
        p0, [pl.order for pl in pulses], 0.18, om0, [pl.duration_s for pl in pulses], (1e-7, 300e-6)
    )
    assert nbar < 0.15 and durations.shape == (50,)


# Home 2009's 24Mg+ Raman geometry: beams at right angles with Delta k along the trap axis (Barrett et al., PRA 68,
# 042302 (2003) Sec. II.A), so |Delta k| = sqrt(2) k at 280.355 nm (Jost, PhD thesis, Colorado 2010, p. 30).
MG24_MASS_U = 23.985042
MG24_RAMAN_NM = 280.355
HOME_AXIAL_MODES = ((1.9488e6, 0.6294), (4.0854e6, 0.5315), (5.4862e6, 0.3222), (5.7373e6, 0.4664))
"""(mode frequency, one 24Mg+ ion's mass-weighted amplitude) of the four axial modes, from tests/test_sympathetic.py's
reproduction of Jost et al., Nature 459, 683 (2009) Methods."""


def _mg24_eta(frequency_hz: float, amplitude: float, *, delta_k: bool = True) -> float:
    """eta = |Delta k| c x0 for one 24Mg+ ion on an axial mode (``delta_k=False`` gives the single-photon emission eta)."""
    x0 = math.sqrt(HBAR_J_S / (2.0 * MG24_MASS_U * ATOMIC_MASS_KG * TWO_PI * frequency_hz))
    k = TWO_PI / (MG24_RAMAN_NM * 1e-9)
    return (math.sqrt(2.0) if delta_k else 1.0) * k * amplitude * x0


def test_the_nist_ninety_degree_raman_geometry_reproduces_the_published_magnesium_lamb_dicke_parameters() -> (
    None
):
    """|Delta k| = sqrt(2) k at 280.355 nm gives the 9Be+-24Mg+ pair's 24Mg+ eta of 0.282 and 0.0769 on the COM and
    stretch modes (Barrett 2003: 0.3 and 0.082), where counter-propagating beams would give 0.399."""
    assert _mg24_eta(2.3e6, 0.93) == pytest.approx(0.282, abs=0.005)
    assert _mg24_eta(4.9e6, 0.37) == pytest.approx(0.0769, abs=0.002)
    assert _mg24_eta(2.3e6, 0.93) / _mg24_eta(2.3e6, 0.93, delta_k=False) == pytest.approx(
        math.sqrt(2.0), rel=1e-12
    )
    counterpropagating = math.sqrt(2.0) * _mg24_eta(2.3e6, 0.93)
    assert counterpropagating == pytest.approx(0.399, abs=0.005)
    assert abs(counterpropagating - 0.30) > 3.0 * abs(_mg24_eta(2.3e6, 0.93) - 0.30)


def test_home_2009_sympathetic_schedule_reaches_six_hundredths_of_a_quantum_on_every_axial_mode() -> None:
    """60 second- plus 120 first-order cycles from nbar = 15 in the 0.05 to 4 pi-time window cool each axial mode of
    Home 2009's Be-Mg-Mg-Be crystal (eta 0.2074, 0.121, 0.0633, 0.0896) to 0.0098, 0.0067, 0.0121 and 0.0045 (3e-3),
    inside the reported 0.06."""
    om0 = TWO_PI * 150e3
    p0 = thermal_distribution(15.0, 200)
    counts = {2: 60, 1: 120}
    expected = (0.009798, 0.006724, 0.012112, 0.004530)
    for (frequency, amplitude), want in zip(HOME_AXIAL_MODES, expected):
        eta = _mg24_eta(frequency, amplitude)
        t_pi = pi_time_s(om0, eta, 1, 1)
        pulses, nbar = optimize_per_order(p0, counts, eta, om0, (0.05 * t_pi, 4.0 * t_pi))
        assert nbar == pytest.approx(want, rel=3e-3)
        assert nbar < 0.06
        # higher orders first
        orders = [pl.order for pl in pulses]
        assert orders == sorted(orders, reverse=True)
        assert len(orders) == sum(counts.values())
    assert [round(_mg24_eta(f, c), 4) for f, c in HOME_AXIAL_MODES] == [0.2074, 0.121, 0.0633, 0.0896]


def test_monroe_1995_fifteen_raman_cycles_are_colder_than_the_measured_triple() -> None:
    """Five first-order cycles per axis from Monroe 1995's Doppler occupations reach (0.0067, 0.0020, 0.0004) without
    repump recoil and (0.0226, 0.0069, 0.0030) with three repump photons (3e-3), both colder than the measured
    (0.033, 0.022, 0.029)."""
    etas = (0.21, 0.12, 0.09)
    doppler = (0.47, 0.30, 0.18)
    om0 = TWO_PI * 476e3  # Omega eta_x tau = pi/2 at tau = 2.5 us (Monroe p. 4013)
    measured = (0.033, 0.022, 0.029)
    for photons, want in ((0.0, (0.006741, 0.002025, 0.000448)), (3.0, (0.022561, 0.006901, 0.002965))):
        got = []
        for eta, n0 in zip(etas, doppler):
            p0 = thermal_distribution(n0, 40)
            kernel = (
                repump_kernel(40, 0.5 * eta, minimal_quadrature(1.0 / 3.0), photons)
                if photons > 0.0
                else None
            )
            t_pi = pi_time_s(om0, eta, 1, 1)
            _pulses, nbar = optimize_per_order(p0, {1: 5}, eta, om0, (0.05 * t_pi, 4.0 * t_pi), repump=kernel)
            got.append(nbar)
        assert got == [pytest.approx(w, rel=3e-3) for w in want]
        assert all(g < m for g, m in zip(got, measured))


def test_repump_kernel_moves_the_mean_by_the_poisson_photon_count_times_alpha_eta_squared() -> None:
    quad = minimal_quadrature(0.4)
    kernel = repump_kernel(80, 0.1, quad, 3.0)
    n = np.arange(80)
    assert np.allclose(kernel.sum(axis=0)[:50], 1.0, atol=1e-10)
    for column in range(4):
        assert float(n @ kernel[:, column]) - column == pytest.approx(3.0 * 0.4 * 0.01, rel=1e-8)
    p = np.zeros(80)
    p[0] = 1.0
    out = apply_pulses(p, [SidebandPulse(1, 1e-6)] * 3, 0.1, OM0, repump=kernel)
    assert mean_occupation(out) > 0.0 and out.sum() == pytest.approx(1.0, abs=1e-9)


# ---- thermometry ---------------------------------------------------------------------------------------------------


def _sideband_ratio(p: np.ndarray, eta: float, order: int, duration_s: float) -> float:
    """P_rsb/P_bsb after a pulse of ``duration`` on the k-th red and blue sidebands, exact in eta."""
    rsb = sum(
        p[n] * math.sin(0.5 * sideband_rabi_rad_s(OM0, eta, n, order) * duration_s) ** 2
        for n in range(p.size)
    )
    bsb = sum(
        p[n] * math.sin(0.5 * (OM0 * rabi_matrix_element(n + order, n, eta)) * duration_s) ** 2
        for n in range(p.size)
    )
    return float(rsb / bsb)


@pytest.mark.parametrize("order", [1, 2])
def test_sideband_ratio_is_exactly_thermal_for_every_pulse_time_and_eta(order: int) -> None:
    """P_rsb/P_bsb = [nbar/(nbar + 1)]^k to 1e-14 at eta = 1.5 and nbar = 8 for three durations (Turchette 2000), while
    a double-thermal state's ratio varies with the duration by more than 1e-3."""
    p = thermal_distribution(8.0, 400)
    for t in (1e-6, 3e-6, 7.7e-6):
        assert _sideband_ratio(p, 1.5, order, t) == pytest.approx((8.0 / 9.0) ** order, abs=1e-14)
    mixed = 0.6 * thermal_distribution(0.2, 400) + 0.4 * thermal_distribution(12.0, 400)
    ratios = [_sideband_ratio(mixed, 0.3, order, t) for t in (1e-6, 3e-6, 7.7e-6)]
    assert max(ratios) - min(ratios) > 1e-3
