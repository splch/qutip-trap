"""Hyperfine-Zeeman diagonalization, its adiabatic labels, and the clock-point anchors (PLAN.md Section 4.5.1)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from qutip_trap.species import MODULES, species
from qutip_trap.species.model import Level
from qutip_trap.species.zeeman import (
    MU_B_OVER_H_HZ_PER_G,
    HyperfineZeeman,
    clock_points,
    g_I_steck,
    hyperfine_zeeman,
    transition_sensitivity,
)
from qutip_trap.units import M_E_OVER_M_P
from tests.fixtures import G_J_S12

CA40 = MODULES["40Ca+"].TABLE
CA43 = MODULES["43Ca+"].TABLE
BE9 = MODULES["9Be+"].TABLE
MG25 = MODULES["25Mg+"].TABLE
YB = MODULES["171Yb+"].TABLE


def breit_rabi_hz(
    nuclear_spin: float, A_hz: float, g_J: float, g_I: float, B_gauss: float, F: int, mF: int
) -> float:
    """The J = 1/2 closed form relative to the hyperfine centroid (Hz): E(F = I +- 1/2, m_F) = -Delta E/(2(2I+1)) +
    g_I mu_B m_F B/h +- (Delta E/2) sqrt(1 + 4 m_F x/(2I+1) + x^2), x = (g_J - g_I) mu_B B/(h Delta E), Delta E =
    A (I + 1/2); the stretched states are linear in B."""
    i, f, m = Fraction(nuclear_spin), Fraction(F), Fraction(mF)
    mu = MU_B_OVER_H_HZ_PER_G
    de = A_hz * float(i + Fraction(1, 2))
    x = (g_J - g_I) * mu * B_gauss / de
    if abs(m) == i + Fraction(1, 2):
        return de * float(i) / float(2 * i + 1) + 0.5 * (g_J + 2.0 * float(i) * g_I) * mu * B_gauss * (
            1.0 if m > 0 else -1.0
        )
    sign = 1.0 if f == i + Fraction(1, 2) else -1.0
    root = math.sqrt(1.0 + 4.0 * float(m) * x / float(2 * i + 1) + x * x)
    return -de / (2.0 * float(2 * i + 1)) + g_I * mu * float(m) * B_gauss + sign * (de / 2.0) * root


def lande_g_f(F: int, J: Fraction, nuclear_spin: Fraction, g_J: float, g_I: float) -> float:
    """g_F = g_J [F(F+1) - I(I+1) + J(J+1)]/(2F(F+1)) + g_I [F(F+1) + I(I+1) - J(J+1)]/(2F(F+1)), weak field."""
    ff, jj, ii = (float(Fraction(x) * (Fraction(x) + 1)) for x in (F, J, nuclear_spin))
    return g_J * (ff - ii + jj) / (2.0 * ff) + g_I * (ff + ii - jj) / (2.0 * ff)


def _s12(a_hz: float, g_j: float) -> Level:
    return Level("S1/2", 0.0, None, a_hz, 0.0, g_j, ("Steck",))


def ca43_ground() -> HyperfineZeeman:
    return HyperfineZeeman(
        _s12(CA43["ca43.S12.A_hfs_hz"].value, CA40["ca40.S12.g_J"].value),
        3.5,
        CA43["ca43.mu_I_nuclear_magnetons"].value,
    )


def be9_ground(g_j: float | None = None) -> HyperfineZeeman:
    g = BE9["be9.S12.g_J"].value if g_j is None else g_j
    return HyperfineZeeman(
        _s12(BE9["be9.S12.A_hfs_hz"].value, g), 1.5, BE9["be9.mu_I_nuclear_magnetons"].value
    )


def mg25_ground(g_j: float = G_J_S12) -> HyperfineZeeman:
    return HyperfineZeeman(
        _s12(MG25["mg25.S12.A_hfs_hz"].value, g_j), 2.5, MG25["mg25.mu_I_nuclear_magnetons"].value
    )


def test_g_I_convention() -> None:
    """g_I = +2.0467e-4 for 43Ca+'s mu_I = -1.31535 mu_N (1e-4) and -5.377e-4 for mu_I = 0.49367 at I = 1/2
    (Section 13)."""
    assert g_I_steck(-1.31535, 3.5) == pytest.approx(2.0467e-4, rel=1e-4)
    assert g_I_steck(0.49367, 0.5) == pytest.approx(-5.377e-4, rel=1e-3)
    assert g_I_steck(0.0, 0.0) == 0.0


@pytest.mark.parametrize("B", [0.0, 10.0, 146.0942, 500.0])
def test_breit_rabi_against_numerical_diagonalization_43ca(B: float) -> None:
    """The 16 43Ca+ S1/2 levels match the Breit-Rabi form to 1e-6 Hz at 0, 10, 146.0942 and 500 G."""
    hz = ca43_ground()
    closed = sorted(
        breit_rabi_hz(3.5, CA43["ca43.S12.A_hfs_hz"].value, CA40["ca40.S12.g_J"].value, hz.g_I, B, F, mF)
        for F in (3, 4)
        for mF in range(-F, F + 1)
    )
    assert np.max(np.abs(np.sort(hz.spectrum(B).energies_hz) - np.array(closed))) < 1e-6


def test_43ca_labels_and_inverted_ordering() -> None:
    sp = ca43_ground().spectrum(0.0)
    e3, e4 = sp.energy_hz("F=3 mF=0"), sp.energy_hz("F=4 mF=0")
    assert e3 > e4, "the negative A places F = 3 above F = 4: derived, not stored"
    assert e3 - e4 == pytest.approx(4.0 * abs(CA43["ca43.S12.A_hfs_hz"].value), rel=1e-12), (
        "|A|(I + 1/2) at I = 7/2"
    )
    assert e3 - e4 == pytest.approx(3225.608e6, rel=1e-7), "the zero-field splitting of Arbes et al."
    assert len(sp.labels) == 16 and len(set(sp.labels)) == 16


def test_43ca_clock_point_of_harty_2014() -> None:
    """Harty 2014's 43Ca+ clock point: 146.0942 G (5e-5), 3,199,941,076.93 Hz (0.05), d2nu/dB2 = 2415.46 Hz/G^2 (2) with
    half of it the Taylor coefficient, and zero slope."""
    hz = ca43_ground()
    (cp,) = clock_points(hz, "F=4 mF=0", hz, "F=3 mF=1", 50.0, 300.0)
    assert cp.B0_gauss == pytest.approx(146.0942, abs=5e-5)
    assert cp.frequency_hz == pytest.approx(3_199_941_076.93, abs=0.05)
    assert cp.d2nu_dB2_hz_per_g2 == pytest.approx(2415.46, abs=2.0)
    assert cp.taylor_c2_hz_per_g2 == pytest.approx(1207.73, abs=1.0)
    assert abs(transition_sensitivity(hz, "F=4 mF=0", hz, "F=3 mF=1", cp.B0_gauss).dnu_dB_hz_per_g) < 1e-6


def test_43ca_clock_point_negative_controls() -> None:
    """Langer's plus sign on g_I moves the 43Ca+ point to 146.3015 G and 3,199,857,314.5 Hz (5e-4 G, 2 Hz), A > 0
    removes it, and g_J = 2.000 moves it by 160 mG and 47 Hz."""
    A = CA43["ca43.S12.A_hfs_hz"].value
    gJ = CA40["ca40.S12.g_J"].value
    wrong_sign = HyperfineZeeman(_s12(A, gJ), 3.5, +1.31535)
    (cp,) = clock_points(wrong_sign, "F=4 mF=0", wrong_sign, "F=3 mF=1", 50.0, 300.0)
    assert cp.B0_gauss == pytest.approx(146.3015, abs=5e-4)
    assert cp.frequency_hz == pytest.approx(3_199_857_314.5, abs=2.0)
    positive_a = HyperfineZeeman(_s12(-A, gJ), 3.5, -1.31535)
    assert clock_points(positive_a, "F=4 mF=0", positive_a, "F=3 mF=1", 1.0, 4000.0) == ()
    g2 = HyperfineZeeman(_s12(A, 2.0), 3.5, -1.31535)
    (cp2,) = clock_points(g2, "F=4 mF=0", g2, "F=3 mF=1", 50.0, 300.0)
    (ref,) = clock_points(ca43_ground(), "F=4 mF=0", ca43_ground(), "F=3 mF=1", 50.0, 300.0)
    assert abs(cp2.B0_gauss - ref.B0_gauss) == pytest.approx(0.160, abs=0.005)
    assert abs(cp2.frequency_hz - ref.frequency_hz) == pytest.approx(47.0, abs=1.5)


def _be9_clock(g_j: float | None = None) -> tuple[float, float]:
    hz = be9_ground(g_j)
    (cp,) = clock_points(hz, "F=2 mF=0", hz, "F=1 mF=1", 80.0, 160.0)
    return cp.B0_gauss, cp.frequency_hz


def test_9be_clock_point_and_further_anchors() -> None:
    """The 9Be+ clock point is 119.44615496 G and 1,207,495,853.379 Hz (5e-8 G, 1e-3 Hz), within 20 Hz of Langer's
    printed value, with Taylor coefficient 0.3049 Hz/uT^2 and the stationary pairs at 119.643 and 223.073 G."""
    hz = be9_ground()
    (cp,) = clock_points(hz, "F=2 mF=0", hz, "F=1 mF=1", 80.0, 160.0)
    assert cp.B0_gauss == pytest.approx(119.44615496, abs=5e-8)
    assert cp.frequency_hz == pytest.approx(1_207_495_853.379, abs=1e-3)
    assert abs(cp.frequency_hz - 1_207_495_843.0) < 20.0
    assert cp.taylor_c2_hz_per_g2 * 1e-4 == pytest.approx(0.3049, abs=2e-4), "Langer's 0.305 Hz/uT^2"
    s = transition_sensitivity(hz, "F=2 mF=2", hz, "F=1 mF=1", cp.B0_gauss)
    assert abs(s.dnu_dB_hz_per_g) * 1e-5 == pytest.approx(17.64, abs=0.02), "-17.64 kHz/uT on |2,2> <-> |1,1>"
    s2 = transition_sensitivity(hz, "F=2 mF=-2", hz, "F=1 mF=-1", 1.0)
    assert abs(s2.dnu_dB_hz_per_g) * 1e-5 == pytest.approx(21.0, abs=0.1)
    (p1,) = clock_points(hz, "F=2 mF=1", hz, "F=1 mF=0", 80.0, 160.0)
    (p2,) = clock_points(hz, "F=2 mF=1", hz, "F=1 mF=1", 150.0, 300.0)
    assert p1.B0_gauss == pytest.approx(119.643, abs=2e-3)
    assert p2.B0_gauss == pytest.approx(223.073, abs=2e-3)
    assert p1.B0_gauss - cp.B0_gauss > 0.15, "the roots at 119.446 and 119.643 G resolve"


def test_9be_clock_point_sensitivity_to_g_j_and_the_negative_controls() -> None:
    """Dickopf's calculated g_J moves the 9Be+ point by 1.558e-5 G and 0.00933 Hz (2e-3), g_J = 2.000 by 0.135 G and
    80.86 Hz, A > 0 removes it, and at 119.4 G the slope is -2.82 Hz/uT, 6.50 Hz above the minimum."""
    b_m, f_m = _be9_clock()
    b_t, f_t = _be9_clock(BE9["be9.S12.g_J_theory"].value)
    assert b_t - b_m == pytest.approx(1.558e-5, rel=2e-3)
    assert f_t - f_m == pytest.approx(0.00933, rel=2e-3)
    b_l, f_l = _be9_clock(2.000)
    assert b_l - b_m == pytest.approx(0.13504, abs=1e-4)
    assert f_l - f_m == pytest.approx(80.86, abs=0.05)
    flipped = HyperfineZeeman(
        _s12(-BE9["be9.S12.A_hfs_hz"].value, BE9["be9.S12.g_J"].value),
        1.5,
        BE9["be9.mu_I_nuclear_magnetons"].value,
    )
    assert clock_points(flipped, "F=2 mF=0", flipped, "F=1 mF=1", 80.0, 160.0) == ()
    hz = be9_ground()
    rounded = transition_sensitivity(hz, "F=2 mF=0", hz, "F=1 mF=1", 119.4)
    assert rounded.dnu_dB_hz_per_g * 1e-2 == pytest.approx(-2.82, abs=0.05)
    assert rounded.frequency_hz - f_m == pytest.approx(6.50, abs=0.3)


def test_9be_inverted_multiplet_from_negative_moment() -> None:
    sp = be9_ground().spectrum(0.0)
    assert sp.energy_hz("F=1 mF=0") > sp.energy_hz("F=2 mF=0"), "the negative moment puts F = 1 above F = 2"


def test_25mg_clock_point_of_srinivas() -> None:
    """Srinivas 2021's 25Mg+ clock point is 212.78 G and 1.686462 GHz at the assumed g_J (0.01 G, 1 kHz), and
    g_J = 2.000 moves it to 213.025 G."""
    hz = mg25_ground()
    (cp,) = clock_points(hz, "F=3 mF=1", hz, "F=2 mF=1", 100.0, 400.0)
    assert cp.B0_gauss == pytest.approx(212.78, abs=0.01)
    assert cp.frequency_hz == pytest.approx(1.686462e9, abs=1e3)
    lande = mg25_ground(2.000)
    (cp_lande,) = clock_points(lande, "F=3 mF=1", lande, "F=2 mF=1", 100.0, 400.0)
    assert cp_lande.B0_gauss == pytest.approx(213.025, abs=0.01)
    assert MG25["mg25.S12.g_I_over_g_J"].value == pytest.approx(
        MG25["mg25.S12.g_I_over_g_J_brewer"].value, rel=2e-5
    )


def test_171yb_quadratic_zeeman_coefficient() -> None:
    """The 171Yb+ clock line is purely quadratic at 310.869 Hz/G^2 (1e-3), the (g_J - g_I)^2 mu_B^2/(2 h^2 A) closed
    form."""
    yb = species("171Yb+")
    nu0, d1_zero, _ = yb.transition_frequency_hz("S1/2 F=0 mF=0", "S1/2 F=1 mF=0", 0.0)
    assert nu0 == pytest.approx(12_642_812_118.5, abs=1e-6)
    assert abs(d1_zero) < 1e-9, "no linear Zeeman term on the clock line"
    _, _, d2_one = yb.transition_frequency_hz("S1/2 F=0 mF=0", "S1/2 F=1 mF=0", 1.0)
    assert d2_one / 2.0 == pytest.approx(310.869, abs=1e-3), "taylor_c2 = (1/2) d2nu/dB2"
    for B in (1.0, 5.0, 10.0):
        nu, d1, d2 = yb.transition_frequency_hz("S1/2 F=0 mF=0", "S1/2 F=1 mF=0", B)
        assert (nu - nu0) / B**2 == pytest.approx(310.87, abs=0.02), "the coefficient of B^2"
        assert d2 / 2.0 == pytest.approx(310.87, abs=0.02)
        assert d1 == pytest.approx(2.0 * 310.87 * B, rel=2e-4), "d nu/dB = 2 c2 B: a pure quadratic"
    g_j = YB["yb171.S12.g_J"].value
    g_i = g_I_steck(YB["yb171.mu_I_nuclear_magnetons"].value, 0.5)
    assert ((g_j - g_i) * MU_B_OVER_H_HZ_PER_G) ** 2 / (2.0 * nu0) == pytest.approx(310.869, abs=0.002)


def test_171yb_g_f_slope_and_the_adjacent_splitting() -> None:
    """g_F(F = 1) mu_B/h = 1.4012 MHz/G is the diagonalized |1,1> - |1,0> slope (1e-5), and 8.267 MHz the splitting at
    5.9 G (2e-3)."""
    yb = species("171Yb+")
    g_i = g_I_steck(yb.mu_I_nuclear_magnetons, 0.5)
    slope_hz_per_g = (
        lande_g_f(1, Fraction(1, 2), Fraction(1, 2), yb.level("S1/2").g_J, g_i) * MU_B_OVER_H_HZ_PER_G
    )
    assert slope_hz_per_g * 1e-6 == pytest.approx(1.4012, abs=5e-4)
    _nu, d1, _d2 = yb.transition_frequency_hz("S1/2 F=1 mF=0", "S1/2 F=1 mF=1", 1e-3)
    assert d1 == pytest.approx(slope_hz_per_g, rel=1e-5)
    nu, d1, _ = yb.transition_frequency_hz("S1/2 F=1 mF=0", "S1/2 F=1 mF=1", 5.9)
    assert nu == pytest.approx(8.267e6, rel=2e-3)
    assert d1 * 1e-6 == pytest.approx(1.4012, rel=2e-3)


def test_171yb_d32_g_j_reproduces_galstyans_measured_g_f_ratio_to_the_ground_state() -> None:
    """Galstyan et al. 2026 measure g_F(D3/2, F = 1)/g_F(S1/2, F = 1) = 0.998060(5) by microwave spectroscopy; the table's
    D3/2 and S1/2 g_J give that ratio of the diagonalized |1,1> - |1,0> slopes to within its 5e-6, where Meggers's observed
    0.802 gives 1.001594."""
    yb = species("171Yb+")

    def slope(level: str) -> float:
        _nu, d1, _d2 = yb.transition_frequency_hz(f"{level} F=1 mF=0", f"{level} F=1 mF=1", 1e-6)
        return float(d1)

    assert slope("D3/2") / slope("S1/2") == pytest.approx(0.998060, abs=5e-6)
    g_i = g_I_steck(yb.mu_I_nuclear_magnetons, 0.5)
    meggers = lande_g_f(1, Fraction(3, 2), Fraction(1, 2), 0.802, g_i) / lande_g_f(
        1, Fraction(1, 2), Fraction(1, 2), yb.level("S1/2").g_J, g_i
    )
    assert meggers == pytest.approx(1.001594, abs=1e-6)


def test_g_f_matches_the_diagonalized_slope_for_a_high_spin_level() -> None:
    """The weak-field 43Ca+ S1/2 |4,1> - |4,0> slope is g_F(F = 4) mu_B/h to 1e-5."""
    ca = species("43Ca+")
    g_i = g_I_steck(ca.mu_I_nuclear_magnetons, 3.5)
    g_f = lande_g_f(4, Fraction(1, 2), Fraction(7, 2), ca.level("S1/2").g_J, g_i)
    sp = hyperfine_zeeman(ca.level("S1/2"), ca.nuclear_spin, ca.mu_I_nuclear_magnetons).spectrum(1e-3)
    slope = float(sp.dE_dB_hz_per_g[sp.index("F=4 mF=1")] - sp.dE_dB_hz_per_g[sp.index("F=4 mF=0")])
    assert slope == pytest.approx(g_f * MU_B_OVER_H_HZ_PER_G, rel=1e-5)


def test_87rb_second_order_zeeman_from_the_same_machinery() -> None:
    """The 87Rb clock shift is 575.146 Hz/G^2 (2e-3) with Steck's g_J, g_I and Delta_hfs."""
    g_j, g_i, dhfs = 2.002331070, -0.0009951414, 6.834682610904290e9
    hz = HyperfineZeeman(_s12(dhfs / 2.0, g_j), 1.5, -g_i * 1.5 / M_E_OVER_M_P)
    s1 = transition_sensitivity(hz, "F=1 mF=0", hz, "F=2 mF=0", 1.0)
    s0 = transition_sensitivity(hz, "F=1 mF=0", hz, "F=2 mF=0", 0.0)
    assert s1.frequency_hz - s0.frequency_hz == pytest.approx(575.146, abs=2e-3)


def test_spin_zero_level_is_linear_with_lande_factors() -> None:
    """The 40Ca+ S1/2-D5/2 lines are linear: 0.4 mu_B/h for Delta m = 0 and 2.0 mu_B/h stretched (3e-3 with the table's
    g_S, exact with g_S = 2 and g_D = 6/5), and the ten components span 31.4 MHz at 4 G."""
    ca = species("40Ca+")
    _, d1, d2 = ca.transition_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=-1/2", 4.0)
    assert d2 == 0.0
    assert d1 / MU_B_OVER_H_HZ_PER_G == pytest.approx(0.4, rel=3e-3)
    _, d1s, _ = ca.transition_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=-5/2", 4.0)
    assert abs(d1s) / MU_B_OVER_H_HZ_PER_G == pytest.approx(2.0, rel=3e-3)
    hs = HyperfineZeeman(Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("Steck",)), 0.0, 0.0)
    hd = HyperfineZeeman(
        Level("D5/2", ca.level("D5/2").energy_hz, 1.168, 0.0, 0.0, 1.2, ("Steck",)), 0.0, 0.0
    )
    exact = transition_sensitivity(hs, "mJ=-1/2", hd, "mJ=-1/2", 4.0)
    assert exact.dnu_dB_hz_per_g / MU_B_OVER_H_HZ_PER_G == pytest.approx(0.4, abs=1e-12)
    assert exact.dnu_dB_hz_per_g * 1e-6 == pytest.approx(0.5598, abs=5e-4)
    stretched = transition_sensitivity(hs, "mJ=-1/2", hd, "mJ=-5/2", 4.0)
    assert stretched.dnu_dB_hz_per_g * 1e-6 == pytest.approx(-2.799, abs=1e-3)
    freqs = [
        transition_sensitivity(hs, f"mJ={m}", hd, f"mJ={mp}", 4.0).frequency_hz
        for m in ("-1/2", "1/2")
        for mp in ("-5/2", "-3/2", "-1/2", "1/2", "3/2", "5/2")
        if abs(Fraction(m) - Fraction(mp)) <= 2
    ]
    assert len(freqs) == 10
    assert (max(freqs) - min(freqs)) * 1e-6 == pytest.approx(31.4, abs=0.1)


def test_hyperfine_free_level_labels_and_second_derivative_zero() -> None:
    sp = species("40Ca+").zeeman_spectrum("P3/2", 3.0)
    assert sp.labels == ("mJ=-3/2", "mJ=-1/2", "mJ=1/2", "mJ=3/2")
    assert np.all(sp.d2E_dB2_hz_per_g2 == 0.0)


# ---- the adiabatic labels ----------------------------------------------------------------------------------------------


def _overlap_tracked_violations(hz: HyperfineZeeman, fields: list[float]) -> int:
    """Continue every eigenvector from B = 0 by a maximum-overlap assignment within each m_F block and count the steps at
    which the state that started k-th in energy is no longer k-th: zero is the assumption ``spectrum`` labels by."""
    violations = 0
    for idx in hz._blocks().values():  # noqa: SLF001 - the test is about this decomposition
        if len(idx) < 2:
            continue
        prev: np.ndarray | None = None
        for b in fields:
            w, v = np.linalg.eigh(hz.hamiltonian_hz(b)[np.ix_(idx, idx)])
            v = v[:, np.argsort(w)]
            if prev is not None:
                # a Hungarian assignment: near a narrow avoided crossing a greedy match invents violations
                rows, cols = linear_sum_assignment(-np.abs(prev.T @ v))
                violations += int(np.count_nonzero(rows != cols))
            prev = v
    return violations


@pytest.mark.parametrize(
    ("name", "level", "b_max"),
    [
        ("43Ca+", "P3/2", 500.0),  # A = -31.0 MHz and B = -6.9 MHz, both nonzero
        ("43Ca+", "D5/2", 300.0),  # A and B comparable
        ("43Ca+", "D3/2", 400.0),
        ("43Ca+", "S1/2", 4000.0),
        ("9Be+", "P3/2", 500.0),
        ("171Yb+", "P3/2", 5000.0),
    ],
)
def test_the_ascending_eigenvalue_order_is_the_adiabatic_continuation(
    name: str, level: str, b_max: float
) -> None:
    """Within each m_F block the maximum-overlap continuation from B = 0 never reorders the ascending eigenvalues."""
    sp = species(name)
    hz = hyperfine_zeeman(sp.level(level), sp.nuclear_spin, sp.mu_I_nuclear_magnetons)
    # steps small against the level spacing over mu_B/h, or the overlap match itself fails at a narrow crossing
    assert _overlap_tracked_violations(hz, list(np.linspace(0.0, b_max, 1 + int(b_max / 0.05)))) == 0


def test_the_labelling_holds_on_a_pathological_level_and_a_degenerate_one_is_refused() -> None:
    """A J = 3/2, I = 3/2 level with a quadrupole term 50x the dipole one keeps its labels, and two F manifolds
    degenerate at B = 0 are refused."""
    hz = HyperfineZeeman(Level("P3/2", 1.0e15, 1e-8, 1.0e6, -50.0e6, 1.334, ("test",)), 1.5, -1.0)
    assert _overlap_tracked_violations(hz, list(np.linspace(0.0, 2000.0, 801))) == 0
    with pytest.raises(ValueError, match="degenerate F manifolds"):
        HyperfineZeeman(Level("P3/2", 1.0e15, 1e-8, 0.0, 1.0e6, 1.334, ("test",)), 1.5, -1.0)


def test_the_f_labels_are_stable_from_zero_field_to_the_paschen_back_regime() -> None:
    yb = species("171Yb+")
    hz = hyperfine_zeeman(yb.level("S1/2"), yb.nuclear_spin, yb.mu_I_nuclear_magnetons)
    at_zero = set(hz.spectrum(0.0).labels)
    for b in (0.1, 5.0, 100.0, 5000.0, 50000.0):
        assert set(hz.spectrum(b).labels) == at_zero
