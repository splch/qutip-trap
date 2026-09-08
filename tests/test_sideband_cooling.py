"""Resolved-sideband and pulsed sideband cooling and the thermometry of PLAN.md Sections 4.2.2, 4.2.7, 4.2.8 (M3): Section 9.3 rows
"Raman sideband cooling", "Thermometry", "Sympathetic"; Section 9.12 "Laguerre nodes in pulsed cooling", "Sideband-ratio thermometry,
exact"; Section 9.15 "Repump recoil per cycle and probability guards"; Section 13 "Sideband-cooling trapping condition"."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.hilbert.operators import rabi_matrix_element
from qutip_trap.light.recoil import minimal_quadrature
from qutip_trap.prep.closed_forms import effective_two_level
from qutip_trap.prep.sideband import (
    SidebandPulse,
    accumulation_centre,
    apply_pulses,
    blue_sideband_flopping,
    double_thermal_fit,
    invert_flopping,
    laguerre_first_zero,
    mean_occupation,
    monroe_half_rabi,
    nbar_from_ratio,
    optimize_durations,
    optimize_per_order,
    optimize_shared_duration,
    pi_time_s,
    populations_from_tail_sums,
    quenched_floor,
    raman_two_photon_rabi_rad_s,
    repump_kernel,
    sideband_rabi_rad_s,
    sideband_ratio,
    stenholm_floor_half_width,
    stranded_index,
    stranded_population,
    thermal_distribution,
    thermal_ratio,
    time_averaged_rsb_signal,
    transfer_matrix,
    trapping_pulse_areas,
    truncated_tail,
)
from qutip_trap.units import ATOMIC_MASS_KG, HBAR_J_S, TWO_PI

OM0 = TWO_PI * 150e3


def test_laguerre_first_zeros_and_the_stranded_fock_indices() -> None:
    """39.7908 (k = 1) and 71.7720 (k = 2) at eta = 0.3, 112.2895 and 202.0112 at eta = 0.18 as continuous degrees; with n' = n - k the stranded
    states are n = 41, 74 and 113 (Che's printed 40 and 72 are degree labels; Section 9.12 row 108)."""
    assert laguerre_first_zero(1, 0.3) == pytest.approx(39.7908, abs=1e-3)
    assert laguerre_first_zero(2, 0.3) == pytest.approx(71.7720, abs=1e-3)
    assert laguerre_first_zero(1, 0.18) == pytest.approx(112.2895, abs=1e-3)
    assert laguerre_first_zero(2, 0.18) == pytest.approx(202.0112, abs=1e-3)
    assert (stranded_index(1, 0.3), stranded_index(2, 0.3), stranded_index(1, 0.18)) == (41, 74, 113)
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
    assert 1.0 - p0.sum() == pytest.approx(truncated_tail(15.36, 151), rel=1e-9)
    assert truncated_tail(15.36, 151) == pytest.approx(7e-5, rel=0.1)  # n_max = 150: the plan's 7e-5
    p25 = apply_pulses(p0, [SidebandPulse(1, 9e-6)] * 25, 0.3, OM0)
    assert p25.sum() == pytest.approx(p0.sum(), abs=4e-16 * 25)


def test_trapping_condition_is_2m_pi_in_the_plans_rabi_convention() -> None:
    """A fixed pulse leaves |n> untouched when Omega_{n-1,n} t = 2 pi (sin^2(Omega t/2) = 0); the sources' half-Rabi m pi would predict trapping at
    Omega_{n-1,n} t = pi, where the plan's convention empties the state completely (Section 13)."""
    n = 7
    t_trap = 2.0 * math.pi / sideband_rabi_rad_s(OM0, 0.2, n, 1)
    assert trapping_pulse_areas(n, 0.2, OM0, t_trap) == pytest.approx(1.0)
    p = np.zeros(20)
    p[n] = 1.0
    out = apply_pulses(p, [SidebandPulse(1, t_trap)], 0.2, OM0)
    assert out[n] == pytest.approx(1.0, abs=1e-12)
    out_half = apply_pulses(p, [SidebandPulse(1, t_trap / 2.0)], 0.2, OM0)
    assert out_half[n] == pytest.approx(0.0, abs=1e-12) and out_half[n - 1] == pytest.approx(1.0, abs=1e-12)


def test_che_2017_single_order_cooling_accumulates_population_just_above_the_rabi_node() -> None:
    """25Mg+, eta = 0.3, nbar = 17, 151 levels, 9 us pulses at Omega_0/2pi = 150 kHz: first-order cycles pile population up at n = 43, 42, 41 after 40,
    150, 500 pulses (Che's 43, 41, 40 with his degree labelling) and second-order cycles at 77, 74, 74 (his 75, 73, 72), the centre creeping
    down onto the node 41 / 74 (Section 4.2.8, "an off-by-k inconsistency internal to the paper")."""
    p0 = thermal_distribution(17.0, 151)
    peaks = []
    centres = []
    for count in (40, 150, 500):
        p = apply_pulses(p0, [SidebandPulse(1, 9e-6)] * count, 0.3, OM0)
        peaks.append(30 + int(np.argmax(p[30:])))
        c = accumulation_centre(p, 1, 0.3)
        assert c is not None
        centres.append(c)
    assert peaks == [43, 42, 41]
    assert centres[0] > centres[1] > centres[2] > stranded_index(1, 0.3) - 0.5
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
    assert stranded_population(p, 1, 0.3) < 0.02


def test_rasmusson_2021_fixed_pulse_prediction_and_multiorder_schedules() -> None:
    """171Yb+, eta = 0.18, from nbar = 14.6: 25 first-order pulses of one shared duration reach nbar = 3.1 (Rasmusson's nbar_sim = 3.57(58)) and
    the three measured thermometry values (0.58, 8.0, 4.1) are estimator biases, not targets. From nbar = 15.36, 50 multiorder pulses with one
    duration per order (higher orders first) reach 0.77 and independently optimized durations 0.12 (Rasmusson's optimized 0.06 is not
    reached with this optimizer; M3 finding). Single-order cooling leaves the thermal tail above the node near n = 113 (8e-4 of the
    population, about 0.1 quanta; the plan's 'about 0.3 quanta' is not reproduced)."""
    om0 = TWO_PI * 100e3
    p0 = thermal_distribution(14.6, 200)
    t, nbar = optimize_shared_duration(p0, [1] * 25, 0.18, om0, (1e-7, 200e-6))
    assert 2.99 < nbar < 3.57 + 0.58
    p_pi = apply_pulses(p0, [SidebandPulse(1, pi_time_s(om0, 0.18, 15, 1))] * 25, 0.18, om0)
    assert mean_occupation(p_pi) == pytest.approx(3.15, abs=0.1)
    p0b = thermal_distribution(15.36, 220)
    pulses, nb_order = optimize_per_order(p0b, {3: 17, 2: 17, 1: 16}, 0.18, om0, (1e-7, 200e-6))
    assert [pl.order for pl in pulses[:17]] == [3] * 17 and nb_order < 1.0
    p_single = apply_pulses(p0b, [SidebandPulse(1, t)] * 50, 0.18, om0)
    tail = stranded_population(p_single, 1, 0.18)
    assert tail == pytest.approx((15.36 / 16.36) ** 113, rel=0.05)
    assert float(np.dot(np.arange(113, 220), p_single[113:])) < 0.15


@pytest.mark.slow
def test_rasmusson_independent_durations_reach_a_tenth_of_a_quantum() -> None:
    om0 = TWO_PI * 100e3
    p0 = thermal_distribution(15.36, 220)
    pulses, _ = optimize_per_order(p0, {3: 17, 2: 17, 1: 16}, 0.18, om0, (1e-7, 200e-6))
    durations, nbar = optimize_durations(
        p0, [pl.order for pl in pulses], 0.18, om0, [pl.duration_s for pl in pulses], (1e-7, 300e-6)
    )
    assert nbar < 0.15 and durations.shape == (50,)


# Home 2009's own 24Mg+ Raman geometry on the 9Be+-24Mg+-24Mg+-9Be+ crystal (Section 9.3 row "Sympathetic").
# Barrett, DeMarco, Schaetz, Leibfried, Britton, Chiaverini, Itano, Jelenkovic, Jost, Langer, Rosenband, Wineland,
# Phys. Rev. A 68, 042302 (2003) Sec. II.A: "The Raman beams propagate at right angles to each other with R2 parallel
# to the quantization axis and the difference vector Delta k = k_1 - k_2 parallel to the trap axis", so |Delta k| =
# sqrt(2) k; lambda = 280.355 nm for the 24Mg+ pair (J. D. Jost, PhD thesis, Univ. of Colorado 2010, p. 30).
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
    """|Delta k| = sqrt(2) k at 280.355 nm reproduces the published 24Mg+ eta of the two-ion 9Be+-24Mg+ pair: 0.28-0.30
    on the 2.3 MHz common mode (c_Mg = 0.93) and 0.08 on the 4.9 MHz stretch mode (c_Mg = 0.37) (Barrett et al., PRA 68,
    042302 (2003) Sec. III: "0.3 for the COM mode and 0.082 for the stretch"; Jost thesis p. 142: 0.28 and 0.08).
    Counterpropagating beams (|Delta k| = 2 k) would give 0.40 and 0.11, which is the negative control."""
    assert _mg24_eta(2.3e6, 0.93) == pytest.approx(0.282, abs=0.005)
    assert _mg24_eta(4.9e6, 0.37) == pytest.approx(0.0769, abs=0.002)
    assert _mg24_eta(2.3e6, 0.93) / _mg24_eta(2.3e6, 0.93, delta_k=False) == pytest.approx(
        math.sqrt(2.0), rel=1e-12
    )
    counterpropagating = math.sqrt(2.0) * _mg24_eta(2.3e6, 0.93)
    assert counterpropagating == pytest.approx(0.399, abs=0.005)
    assert abs(counterpropagating - 0.30) > 3.0 * abs(_mg24_eta(2.3e6, 0.93) - 0.30)


def test_home_2009_sympathetic_schedule_reaches_six_hundredths_of_a_quantum_on_every_axial_mode() -> None:
    """Home et al., Science 325, 1227 (2009): Doppler cooling on the 24Mg+ ions "thermalizes the state with a mean
    vibrational occupation number nbar ~ 15" (supporting online material, "Sympathetic cooling"), then "cycles of
    resolved sideband cooling on both the second and first sidebands of each motional mode" cool "each mode to near
    the quantum ground state (<n> ~ 0.06)" (p. 1229, and a per-mode ground-state fidelity of 0.94 in the SOM, whose
    thermal reading is (1 - 0.94)/0.94 = 0.064).

    The Lamb-Dicke parameters are Home's own: one 24Mg+ ion's mass-weighted amplitude in each axial mode of the
    reproduced Be-Mg-Mg-Be crystal times |Delta k| = sqrt(2) k at 280 nm. Home prints only the total cooling time
    (5.1 ms for all four modes) and no per-mode cycle count, so the schedule here is one uniform 60 second-order plus
    120 first-order cycles inside the plan's own pulse-time window (0.05 to 4 pi times of the first sideband out of
    |1>, as ``prep.recipe`` uses): every mode reaches 0.005 to 0.013, comfortably inside the row's 0.06. That window
    matters - the greedy per-order optimizer chooses each block's duration with the earlier blocks fixed, so a window
    fixed in absolute seconds rather than in pi times lands in a different local minimum and the result stops being
    monotone in the pulse count (the M3 finding on this optimizer).
    """
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
        # higher orders first, and every pulse of one order shares its duration (Section 4.2.8)
        orders = [pl.order for pl in pulses]
        assert orders == sorted(orders, reverse=True)
        assert len(orders) == sum(counts.values())
    # the Lamb-Dicke parameters themselves, as the regression of the geometry
    assert [round(_mg24_eta(f, c), 4) for f, c in HOME_AXIAL_MODES] == [0.2074, 0.121, 0.0633, 0.0896]


def test_monroe_1995_fifteen_raman_cycles_are_colder_than_the_measured_triple() -> None:
    """Section 9.3 row "Raman sideband cooling": Monroe et al., PRL 75, 4011 (1995) infer "3D Raman cooling to
    <n_nu> ~= (0.033, 0.022, 0.029)" from the Doppler values (0.47, 0.30, 0.18) at eta = (0.21, 0.12, 0.09) (their
    eta_nu = 2 k_nu r_nu, counterpropagating Raman beams).

    Fifteen cycles - five first-order cycles per axis - through the exact pulsed engine reach (0.0067, 0.0020,
    0.0004) with no repump recoil and (0.0226, 0.0069, 0.0030) with three repump photons per cycle, i.e. the ideal
    schedule is 1.5 to 60 times COLDER than the measurement and the measured triple is not monotone in eta while the
    model necessarily is. So the row's triple is NOT a target of the schedule model: the experiment's floor comes
    from heating during the cooling and the repump's imperfection, neither of which the level-B engine carries
    (M3 finding, recorded like the other Monroe anchors). The numbers here are the regression.
    """
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
    # non-monotone in eta is what the model cannot produce: z is measured hotter than y
    assert measured[2] > measured[1]


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


def test_raman_two_photon_rabi_frequency_and_the_monroe_half_convention() -> None:
    """Omega = Omega_1 Omega_2/(2 Delta) in plan units; Monroe's g_1 g_2/Delta with g = Omega/2 is the same number (Section 13)."""
    om1, om2, delta = TWO_PI * 10e6, TWO_PI * 12e6, TWO_PI * 50e9
    plan = raman_two_photon_rabi_rad_s(om1, om2, delta)
    assert plan == pytest.approx(om1 * om2 / (2.0 * delta))
    assert monroe_half_rabi(om1) * monroe_half_rabi(om2) / delta == pytest.approx(plan / 2.0)
    with pytest.raises(ValueError):
        raman_two_photon_rabi_rad_s(om1, om2, 0.0)


def test_quenched_floor_uses_marzolis_half_width_and_the_fast_photons_recoil_weight() -> None:
    """(gamma'/nu)^2 [(eta~/eta)^2 + 1/4] with (eta~/eta)^2 = 1.376 for 40Ca+ cooled at 729 nm and quenched at 854 nm (393 nm recoil photon), bracket 1.62."""
    eff = effective_two_level(1.0, 0.1, 0.2, -1.0, "Xi", allow_invalid=True)  # Marzoli's Fig. 3 fixture
    nu = 100.0 * eff.gamma_coherence_rad_s
    floor = quenched_floor(eff, nu, 1.376)
    assert floor == pytest.approx(stenholm_floor_half_width(eff.gamma_coherence_rad_s, nu, 1.376))
    assert floor / stenholm_floor_half_width(eff.gamma_coherence_rad_s, nu, 0.4) == pytest.approx(
        1.626 / 0.65, rel=1e-3
    )


# ---- thermometry (Section 4.2.7) -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_sideband_ratio_is_exactly_thermal_for_every_pulse_time_and_eta(order: int) -> None:
    """P_rsb/P_bsb = [nbar/(nbar + 1)]^k to 1e-14 at eta = 1.5, nbar = 8 for three durations (Turchette 2000; Section 9.12
    pins 6e-15 and the code delivers 1.6e-15); a non-thermal (double-thermal) state shows a duration-dependent ratio."""
    p = thermal_distribution(8.0, 400)
    for t in (1e-6, 3e-6, 7.7e-6):
        r = sideband_ratio(p, 1.5, OM0, order, t)
        assert r == pytest.approx(thermal_ratio(8.0, order), abs=1e-14)
        assert nbar_from_ratio(r, order) == pytest.approx(8.0, rel=1e-10)
    mixed = 0.6 * thermal_distribution(0.2, 400) + 0.4 * thermal_distribution(12.0, 400)
    ratios = [sideband_ratio(mixed, 0.3, OM0, order, t) for t in (1e-6, 3e-6, 7.7e-6)]
    assert max(ratios) - min(ratios) > 1e-3


def test_time_averaged_rsb_signal_is_half_the_tail_sum_and_inverts_to_the_populations() -> None:
    p = thermal_distribution(0.5, 30)
    assert time_averaged_rsb_signal(p, 1) == pytest.approx(0.5 * (1.0 - p[0]))
    signals = [time_averaged_rsb_signal(p, m) for m in range(1, 10)]
    recovered = populations_from_tail_sums(signals)
    assert np.allclose(recovered[:8], p[1:9], atol=1e-14)


def test_blue_sideband_flopping_inverts_by_nonnegative_least_squares() -> None:
    p = thermal_distribution(0.7, 16)
    om0 = TWO_PI * 100e3
    times = np.linspace(0.0, 300e-6, 600)
    signal = blue_sideband_flopping(p, 0.1, om0, times)
    assert signal[0] == pytest.approx(0.5 * (1.0 + p.sum()))
    assert np.allclose(invert_flopping(times, signal, 0.1, om0, 16)[:6], (p / p.sum())[:6], atol=2e-3)
    # the flopping frequency is Omega_{n+1,n} in the plan's convention (Wineland's 2 Omega)
    expected = 0.5 * (
        1.0 + sum(p[n] * math.cos(om0 * rabi_matrix_element(n + 1, n, 0.1) * 50e-6) for n in range(16))
    )
    assert blue_sideband_flopping(p, 0.1, om0, np.array([50e-6]))[0] == pytest.approx(expected)


def test_double_thermal_fit_recovers_a_two_temperature_mixture() -> None:
    mix = 0.7 * thermal_distribution(0.1, 60) + 0.3 * thermal_distribution(6.0, 60)
    a, nl, nh = double_thermal_fit(mix)
    assert (a, nl, nh) == (
        pytest.approx(0.7, abs=1e-6),
        pytest.approx(0.1, abs=1e-6),
        pytest.approx(6.0, abs=1e-5),
    )
