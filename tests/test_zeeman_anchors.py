"""Hyperfine-Zeeman diagonalization, Breit-Rabi cross-check and the clock-point anchors of PLAN.md Section 9.13."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from qutip_trap.species import MODULES, species
from qutip_trap.species.model import Level
from qutip_trap.species.zeeman import (
    MU_B_OVER_H_HZ_PER_G,
    HyperfineZeeman,
    breit_rabi_hz,
    clock_points,
    g_I_steck,
    transition_sensitivity,
)

CA43 = MODULES["43Ca+"].TABLE
BE9 = MODULES["9Be+"].TABLE
MG25 = MODULES["25Mg+"].TABLE
YB = MODULES["171Yb+"].TABLE


def _s12(a_hz: float, g_j: float, name: str = "S1/2") -> Level:
    return Level(name, 0.0, None, a_hz, 0.0, g_j, ("Steck",))


def ca43_ground() -> HyperfineZeeman:
    return HyperfineZeeman(
        _s12(CA43["ca43.S12.A_hfs_hz"].value, CA43["ca43.S12.g_J"].value),
        3.5,
        CA43["ca43.mu_I_nuclear_magnetons"].value,
    )


def be9_ground() -> HyperfineZeeman:
    return HyperfineZeeman(
        _s12(BE9["be9.S12.A_hfs_hz"].value, BE9["be9.S12.g_J"].value),
        1.5,
        BE9["be9.mu_I_nuclear_magnetons"].value,
    )


def mg25_ground() -> HyperfineZeeman:
    return HyperfineZeeman(
        _s12(MG25["mg25.S12.A_hfs_hz"].value, MG25["mg25.S12.g_J"].value),
        2.5,
        MG25["mg25.mu_I_nuclear_magnetons"].value,
    )


def test_g_I_convention() -> None:
    """43Ca+ g_I = +2.0467e-4 from mu_I = -1.31535 mu_N (Section 13)."""
    assert g_I_steck(-1.31535, 3.5) == pytest.approx(2.0467e-4, rel=1e-4)
    assert g_I_steck(0.49367, 0.5) == pytest.approx(-5.377e-4, rel=1e-3)
    assert g_I_steck(0.0, 0.0) == 0.0


@pytest.mark.parametrize("B", [0.0, 10.0, 146.0942, 500.0])
def test_breit_rabi_against_numerical_diagonalization_43ca(B: float) -> None:
    """All 16 levels agree to 1e-12 MHz (1e-6 Hz) at 0, 10, 146.0942 and 500 G (Section 9.13)."""
    hz = ca43_ground()
    sp = hz.spectrum(B)
    closed = sorted(
        breit_rabi_hz(3.5, CA43["ca43.S12.A_hfs_hz"].value, CA43["ca43.S12.g_J"].value, hz.g_I, B, F, mF)
        for F in (3, 4)
        for mF in range(-F, F + 1)
    )
    assert np.max(np.abs(np.sort(sp.energies_hz) - np.array(closed))) < 1e-6


def test_43ca_labels_and_inverted_ordering() -> None:
    hz = ca43_ground()
    sp = hz.spectrum(0.0)
    e3 = sp.energy_hz("F=3 mF=0")
    e4 = sp.energy_hz("F=4 mF=0")
    assert e3 > e4, "the negative A places F = 3 above F = 4 (Section 4.5.6): derived, not stored"
    assert e3 - e4 == pytest.approx(4.0 * abs(CA43["ca43.S12.A_hfs_hz"].value), rel=1e-12), (
        "|A|(I + 1/2) at I = 7/2"
    )
    assert e3 - e4 == pytest.approx(3225.608e6, rel=1e-7), (
        "the zero-field splitting 3225.608 MHz of Arbes et al. (via Harty)"
    )
    assert len(sp.labels) == 16 and len(set(sp.labels)) == 16


def test_43ca_clock_point_of_harty_2014() -> None:
    hz = ca43_ground()
    (cp,) = clock_points(hz, "F=4 mF=0", hz, "F=3 mF=1", 50.0, 300.0)
    assert cp.B0_gauss == pytest.approx(146.0942, abs=5e-5)
    assert cp.frequency_hz == pytest.approx(3_199_941_076.93, abs=0.05)
    assert cp.d2nu_dB2_hz_per_g2 * 1e-3 == pytest.approx(2.415, abs=2e-3), (
        "d2nu/dB2 = 2.415 mHz/mG^2 (Harty prints 2.4)"
    )
    assert cp.taylor_c2_hz_per_g2 * 1e-3 == pytest.approx(1.2077, abs=1e-3), (
        "the coefficient of (dB)^2 is half of it (Section 13)"
    )
    s = transition_sensitivity(hz, "F=4 mF=0", hz, "F=3 mF=1", cp.B0_gauss)
    assert abs(s.dnu_dB_hz_per_g) < 1e-6


def test_43ca_negative_controls_of_section_9_13() -> None:
    """Harty's g_I^(N) paired with Langer's plus sign moves the point to 146.3015 G and 3,199,857,314.5 Hz; A > 0 removes it;
    g_J = 2.000 moves it by 160 mG and 47 Hz."""
    A = CA43["ca43.S12.A_hfs_hz"].value
    gJ = CA43["ca43.S12.g_J"].value
    wrong_sign = HyperfineZeeman(_s12(A, gJ), 3.5, +1.31535)  # the mixed convention flips g_I
    (cp,) = clock_points(wrong_sign, "F=4 mF=0", wrong_sign, "F=3 mF=1", 50.0, 300.0)
    assert cp.B0_gauss == pytest.approx(146.3015, abs=5e-4)
    assert cp.frequency_hz == pytest.approx(3_199_857_314.5, abs=2.0)
    positive_a = HyperfineZeeman(_s12(-A, gJ), 3.5, -1.31535)
    assert clock_points(positive_a, "F=4 mF=0", positive_a, "F=3 mF=1", 1.0, 4000.0) == ()
    g2 = HyperfineZeeman(_s12(A, 2.0), 3.5, -1.31535)
    (cp2,) = clock_points(g2, "F=4 mF=0", g2, "F=3 mF=1", 50.0, 300.0)
    ref = clock_points(ca43_ground(), "F=4 mF=0", ca43_ground(), "F=3 mF=1", 50.0, 300.0)[0]
    assert abs(cp2.B0_gauss - ref.B0_gauss) == pytest.approx(0.160, abs=0.005)
    assert abs(cp2.frequency_hz - ref.frequency_hz) == pytest.approx(47.0, abs=1.5)


def test_9be_clock_point_and_further_anchors() -> None:
    hz = be9_ground()
    pts = clock_points(hz, "F=2 mF=0", hz, "F=1 mF=1", 80.0, 160.0)
    assert len(pts) == 1
    cp = pts[0]
    assert cp.B0_gauss == pytest.approx(119.446, abs=1e-3)
    assert cp.frequency_hz == pytest.approx(1_207_495_853.5, abs=1.0), (
        "the plan's recomputed value (Langer prints 843; 10 Hz is the g-factor precision)"
    )
    assert abs(cp.frequency_hz - 1_207_495_843.0) < 20.0
    assert cp.taylor_c2_hz_per_g2 * 1e-4 == pytest.approx(0.3049, abs=2e-4), (
        "0.3049 Hz/uT^2 as the Taylor coefficient (Langer 0.305)"
    )
    # slopes: |2,2> <-> |1,1> at B0 is -17.64 kHz/uT = -1.764 MHz/G (Langer prints magnitudes)
    s = transition_sensitivity(hz, "F=2 mF=2", hz, "F=1 mF=1", cp.B0_gauss)
    assert abs(s.dnu_dB_hz_per_g) * 1e-5 == pytest.approx(17.64, abs=0.02)
    s2 = transition_sensitivity(hz, "F=2 mF=-2", hz, "F=1 mF=-1", 1.0)
    assert abs(s2.dnu_dB_hz_per_g) * 1e-5 == pytest.approx(21.0, abs=0.1)
    # further stationary pairs
    (p1,) = clock_points(hz, "F=2 mF=1", hz, "F=1 mF=0", 80.0, 160.0)
    (p2,) = clock_points(hz, "F=2 mF=1", hz, "F=1 mF=1", 150.0, 300.0)
    assert p1.B0_gauss == pytest.approx(119.643, abs=2e-3)
    assert p2.B0_gauss == pytest.approx(223.073, abs=2e-3)
    assert p1.B0_gauss - cp.B0_gauss > 0.15, "the roots at 119.446 and 119.643 G must resolve as distinct"


def test_9be_inverted_multiplet_from_negative_moment() -> None:
    sp = be9_ground().spectrum(0.0)
    assert sp.energy_hz("F=1 mF=0") > sp.energy_hz("F=2 mF=0"), (
        "the negative nuclear moment puts F = 1 above F = 2 (Section 13)"
    )


def test_25mg_clock_point_of_srinivas() -> None:
    hz = mg25_ground()
    (cp,) = clock_points(hz, "F=3 mF=1", hz, "F=2 mF=1", 100.0, 400.0)
    assert cp.B0_gauss == pytest.approx(212.78, abs=0.01)
    assert cp.frequency_hz == pytest.approx(1.686462e9, abs=1e3)


def test_171yb_quadratic_zeeman_coefficient() -> None:
    """310.87 +- 0.02 Hz/G^2 with the adopted g_J; the Breit-Rabi terms are all in Hz (Sections 9.13, 9.17)."""
    yb = species("171Yb+")
    nu0, d1_zero, _ = yb.transition_frequency_hz("S1/2 F=0 mF=0", "S1/2 F=1 mF=0", 0.0)
    assert nu0 == pytest.approx(12_642_812_118.5, abs=1e-6)
    assert abs(d1_zero) < 1e-9, "no linear Zeeman term on the clock line"
    for B in (1.0, 5.0, 10.0):
        nu, d1, d2 = yb.transition_frequency_hz("S1/2 F=0 mF=0", "S1/2 F=1 mF=0", B)
        assert (nu - nu0) / B**2 == pytest.approx(310.87, abs=0.02), "taylor_c2, the coefficient of B^2"
        assert d2 / 2.0 == pytest.approx(310.87, abs=0.02), "taylor_c2 = (1/2) d2nu/dB2 (Section 13)"
        assert d1 == pytest.approx(2.0 * 310.87 * B, rel=2e-4), "d nu/dB = 2 c2 B: a pure quadratic"
    g_j = YB["yb171.S12.g_J"].value
    g_i = g_I_steck(YB["yb171.mu_I_nuclear_magnetons"].value, 0.5)
    closed = ((g_j - g_i) * MU_B_OVER_H_HZ_PER_G) ** 2 / (2.0 * nu0)
    assert closed == pytest.approx(310.869, abs=0.002), "(g_J - g_I)^2 mu_B^2/(2 h^2 A), check_critique_v3.py"


def test_171yb_g_F_and_the_mhz_per_gauss_trap() -> None:
    """g_F(F = 1) = g_J/2 = 1.001128, g_F mu_B/h = 1.4012 MHz/G, adjacent splitting 8.267 MHz at 5.9 G (Section 9.15)."""
    yb = species("171Yb+")
    nu, d1, _ = yb.transition_frequency_hz("S1/2 F=1 mF=0", "S1/2 F=1 mF=1", 5.9)
    assert nu == pytest.approx(8.267e6, rel=2e-3)
    assert d1 * 1e-6 == pytest.approx(1.4012, rel=2e-3)
    assert 1.4 * MU_B_OVER_H_HZ_PER_G * 5.9 * 1e-6 == pytest.approx(11.56, abs=0.01), (
        "1.4 as a dimensionless g_F is the 40% error"
    )


def test_87rb_second_order_zeeman_from_the_same_machinery() -> None:
    """87Rb 575.146 Hz/G^2 with g_J = 2.002331070, g_I = -0.0009951414, Delta_hfs = 6.834682610904290 GHz (Steck; Section 9.13)."""
    g_j, g_i, dhfs = 2.002331070, -0.0009951414, 6.834682610904290e9
    closed = ((g_j - g_i) * MU_B_OVER_H_HZ_PER_G) ** 2 / (2.0 * dhfs)
    assert closed == pytest.approx(575.146, abs=2e-3)
    # numerically: I = 3/2, A = Delta/(I + 1/2) = Delta/2, g_I supplied through mu_I = -g_I I (m_p/m_e) mu_N
    from qutip_trap.units import M_E_OVER_M_P

    mu_i = -g_i * 1.5 / M_E_OVER_M_P
    hz = HyperfineZeeman(_s12(dhfs / 2.0, g_j), 1.5, mu_i)
    s1 = transition_sensitivity(hz, "F=1 mF=0", hz, "F=2 mF=0", 1.0)
    s0 = transition_sensitivity(hz, "F=1 mF=0", hz, "F=2 mF=0", 0.0)
    assert (s1.frequency_hz - s0.frequency_hz) == pytest.approx(575.146, abs=2e-3)


def test_spin_zero_level_is_linear_with_lande_factors() -> None:
    """40Ca+ optical-qubit sensitivities: Delta m = 0 (-1/2 -> -1/2) is 0.4 mu_B/h and the stretched line 2.0 mu_B/h with the
    LS factors g_S = 2, g_D = 6/5 exactly (Section 9.14); the table's Steck g_S shifts these by 0.1%."""
    ca = species("40Ca+")
    nu, d1, d2 = ca.transition_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=-1/2", 4.0)
    assert d2 == 0.0
    assert d1 / MU_B_OVER_H_HZ_PER_G == pytest.approx(0.4, rel=3e-3)
    _, d1s, _ = ca.transition_frequency_hz("S1/2 mJ=-1/2", "D5/2 mJ=-5/2", 4.0)
    assert abs(d1s) / MU_B_OVER_H_HZ_PER_G == pytest.approx(2.0, rel=3e-3)
    ls = Level("D5/2", ca.level("D5/2").energy_hz, 1.168, 0.0, 0.0, 1.2, ("Steck",))
    s = Level("S1/2", 0.0, None, 0.0, 0.0, 2.0, ("Steck",))
    hs, hd = HyperfineZeeman(s, 0.0, 0.0), HyperfineZeeman(ls, 0.0, 0.0)
    exact = transition_sensitivity(hs, "mJ=-1/2", hd, "mJ=-1/2", 4.0)
    assert exact.dnu_dB_hz_per_g / MU_B_OVER_H_HZ_PER_G == pytest.approx(0.4, abs=1e-12)
    assert exact.dnu_dB_hz_per_g * 1e-6 == pytest.approx(0.5598, abs=5e-4)
    stretched = transition_sensitivity(hs, "mJ=-1/2", hd, "mJ=-5/2", 4.0)
    assert stretched.dnu_dB_hz_per_g * 1e-6 == pytest.approx(-2.799, abs=1e-3)
    # ten components span 5.6 mu_B B/h = 31.4 MHz at 4 G
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


def test_curvature_conventions_are_exactly_a_factor_two_apart() -> None:
    hz = ca43_ground()
    (cp,) = clock_points(hz, "F=4 mF=0", hz, "F=3 mF=1", 50.0, 300.0)
    assert cp.d2nu_dB2_hz_per_g2 == pytest.approx(2.0 * cp.taylor_c2_hz_per_g2)
    assert cp.d2nu_dB2_hz_per_g2 == pytest.approx(2415.46, abs=2.0), "audit row 6-2"
    assert cp.taylor_c2_hz_per_g2 == pytest.approx(1207.73, abs=1.0)
    assert math.isclose(cp.d2nu_dB2_hz_per_g2 * 1e-3, 2.415, abs_tol=2e-3)
