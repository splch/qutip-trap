"""The adiabatic F labelling, the g_F closed form and the unpinned legs of the atomic 9.13 / 9.14 rows
(milestone M0a, audit items E12, E17, E18).

The labelling assumption -- that within one m_F block the ascending-eigenvalue order at any field is the
adiabatic continuation of the order at B = 0, so F can be read off <F^2> at zero field and carried through --
is correct (``zeeman.py`` justifies it by the no-crossing rule) but was asserted nowhere.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from qutip_trap.species import species
from qutip_trap.species.model import Level
from qutip_trap.species.quadrupole import e2_over_e1_amplitude_ratio
from qutip_trap.species.zeeman import (
    MU_B_OVER_H_HZ_PER_G,
    HyperfineZeeman,
    g_I_steck,
    hyperfine_zeeman,
    lande_g_f,
)
from qutip_trap.units import ALPHA_FS

# ---- E12: the adiabatic labelling on a level with A and B both nonzero ----------------------------------


def _overlap_tracked_order(hz: HyperfineZeeman, fields: list[float]) -> int:
    """Continue every eigenvector from B = 0 by maximum overlap and count the ascending-order violations.

    At each field the eigenvectors are matched to the previous field's by a maximum-|overlap|
    assignment within each m_F block; a violation is a step at which the adiabatically continued state that
    started k-th in energy is no longer k-th. Zero violations is exactly the assumption
    ``HyperfineZeeman.spectrum`` rests on.
    """
    violations = 0
    for m, idx in hz._blocks().items():  # noqa: SLF001 - the test is about this internal decomposition
        if len(idx) < 2:
            continue
        prev: np.ndarray | None = None
        for b in fields:
            h = hz.hamiltonian_hz(b)[np.ix_(idx, idx)]
            w, v = np.linalg.eigh(h)
            order = np.argsort(w)
            v = v[:, order]
            if prev is not None:
                # a proper (Hungarian) assignment, not a greedy one: near a narrow avoided crossing the
                # greedy match picks the wrong partner and invents violations that are not there
                rows, cols = linear_sum_assignment(-np.abs(prev.T @ v))
                violations += int(np.count_nonzero(rows != cols))
            prev = v
        assert m is not None
    return violations


@pytest.mark.parametrize(
    ("name", "level", "b_max"),
    [
        ("43Ca+", "P3/2", 500.0),  # A = -31.0 MHz and B = -6.9 MHz, both nonzero
        ("43Ca+", "D5/2", 300.0),  # A = -3.8931 MHz and B = -4.241 MHz, comparable
        ("43Ca+", "D3/2", 400.0),
        ("43Ca+", "S1/2", 4000.0),
        ("9Be+", "P3/2", 500.0),
        ("171Yb+", "P3/2", 5000.0),
    ],
)
def test_the_ascending_eigenvalue_order_is_the_adiabatic_continuation(
    name: str, level: str, b_max: float
) -> None:
    """``zeeman.py`` reads F off <F^2> at B = 0 per m_F block and carries it through by ascending-eigenvalue
    order, justified by the no-crossing rule inside a block. Tracked by overlap, that ordering never breaks."""
    sp = species(name)
    hz = hyperfine_zeeman(sp.level(level), sp.nuclear_spin, sp.mu_I_nuclear_magnetons)
    # the step must be small compared with the level spacing divided by mu_B/h ~ 1.4 MHz/G, or the overlap
    # match itself fails near a narrow avoided crossing and reports violations that are not there
    fields = list(np.linspace(0.0, b_max, 1 + int(b_max / 0.05)))
    assert _overlap_tracked_order(hz, fields) == 0


def test_the_labelling_holds_on_a_deliberately_pathological_level() -> None:
    """A J = 3/2, I = 3/2 level with a quadrupole term 50x the dipole one -- far outside anything the tables
    carry -- still shows no order violation, which is what makes the no-crossing argument the right one."""
    lv = Level("P3/2", 1.0e15, 1e-8, 1.0e6, -50.0e6, 1.334, ("test",))
    hz = HyperfineZeeman(lv, 1.5, -1.0)
    assert _overlap_tracked_order(hz, list(np.linspace(0.0, 2000.0, 801))) == 0


def test_a_degenerate_zero_field_manifold_is_refused_rather_than_mislabelled() -> None:
    """The one case the no-crossing argument cannot cover: two F manifolds degenerate at B = 0. It is caught
    at CONSTRUCTION, before any spectrum can be produced with an ambiguous label."""
    lv = Level("P3/2", 1.0e15, 1e-8, 0.0, 1.0e6, 1.334, ("test",))
    with pytest.raises(ValueError, match="degenerate F manifolds"):
        HyperfineZeeman(lv, 1.5, -1.0)


def test_the_f_labels_are_stable_from_zero_field_to_the_paschen_back_regime() -> None:
    """The labels themselves, not just the ordering: the same (F, m_F) string set at every field."""
    yb = species("171Yb+")
    hz = hyperfine_zeeman(yb.level("S1/2"), yb.nuclear_spin, yb.mu_I_nuclear_magnetons)
    at_zero = set(hz.spectrum(0.0).labels)
    for b in (0.1, 5.0, 100.0, 5000.0, 50000.0):
        assert set(hz.spectrum(b).labels) == at_zero


# ---- E18: g_F as a named closed form --------------------------------------------------------------------


def test_g_f_of_the_yb171_ground_state_is_half_g_j() -> None:
    """PLAN.md 4.5.1 prints g_F as a derived quantity; for I = J = 1/2 the F = 1 value is g_J/2 + g_I/2.

    With the ADOPTED g_J = 2.002615 this is 1.0013075, not the 1.001128 the check scripts print from the
    retired 43Ca+ g_J = 2.00225664 (ledger conv.mg25_no_measured_g_j records the same class of borrowing).
    Section 13: g_F is DIMENSIONLESS; g_F mu_B/h is the 1.4012 MHz/G frequency per field.
    """
    yb = species("171Yb+")
    g_j = yb.level("S1/2").g_J
    g_i = g_I_steck(yb.mu_I_nuclear_magnetons, 0.5)
    g_f = lande_g_f(1, Fraction(1, 2), Fraction(1, 2), g_j, g_i)
    assert g_f == pytest.approx(0.5 * (g_j + g_i), rel=1e-15)
    assert lande_g_f(1, Fraction(1, 2), Fraction(1, 2), g_j) == pytest.approx(0.5 * g_j, rel=1e-15)
    assert lande_g_f(1, Fraction(1, 2), Fraction(1, 2), g_j) == pytest.approx(1.0013075, abs=1e-7)
    # the product is the field slope the diagonalization returns for the |1,1> - |1,0> pair
    slope_mhz_per_g = lande_g_f(1, Fraction(1, 2), Fraction(1, 2), g_j, g_i) * MU_B_OVER_H_HZ_PER_G / 1e6
    assert slope_mhz_per_g == pytest.approx(1.4012, abs=5e-4)
    _nu, d1, _d2 = yb.transition_frequency_hz("S1/2 F=1 mF=0", "S1/2 F=1 mF=1", 1e-3)
    assert d1 / 1e6 == pytest.approx(slope_mhz_per_g, rel=1e-5)


def test_g_f_matches_the_diagonalized_slope_for_a_high_spin_level() -> None:
    """43Ca+ S1/2, I = 7/2: g_F(F=4) = g_J/8 and g_F(F=3) = -g_J/8 in the g_I -> 0 limit."""
    ca = species("43Ca+")
    g_j = ca.level("S1/2").g_J
    assert lande_g_f(4, Fraction(1, 2), Fraction(7, 2), g_j) == pytest.approx(g_j / 8.0, rel=1e-15)
    assert lande_g_f(3, Fraction(1, 2), Fraction(7, 2), g_j) == pytest.approx(-g_j / 8.0, rel=1e-15)
    g_i = g_I_steck(ca.mu_I_nuclear_magnetons, 3.5)
    g_f = lande_g_f(4, Fraction(1, 2), Fraction(7, 2), g_j, g_i)
    hz = hyperfine_zeeman(ca.level("S1/2"), ca.nuclear_spin, ca.mu_I_nuclear_magnetons)
    sp = hz.spectrum(1e-3)
    slope = float(sp.dE_dB_hz_per_g[sp.index("F=4 mF=1")] - sp.dE_dB_hz_per_g[sp.index("F=4 mF=0")])
    assert slope == pytest.approx(g_f * MU_B_OVER_H_HZ_PER_G, rel=1e-5)


def test_g_f_is_undefined_for_f_zero() -> None:
    with pytest.raises(ValueError, match="undefined for F = 0"):
        lande_g_f(0, Fraction(1, 2), Fraction(1, 2), 2.0)


# ---- E17: the unpinned legs of the 9.13 / 9.14 rows -----------------------------------------------------


def test_the_e2_over_e1_suppression_is_zero_point_zero_six_two_alpha_not_alpha_over_two() -> None:
    """Section 9.14's E2/E1 row: k a_0/2 = 2.28e-4 (729 nm) and 2.47e-4 (674 nm), i.e. 0.062 alpha and
    0.068 alpha -- NOT alpha/2 = 3.65e-3, which is 16x larger. The row's correction is the coefficient."""
    ratios = {
        "40Ca+ 729 nm": e2_over_e1_amplitude_ratio(729.347e-9),
        "88Sr+ 674 nm": e2_over_e1_amplitude_ratio(674.025591e-9),
    }
    assert ratios["40Ca+ 729 nm"] == pytest.approx(2.2794e-4, rel=3e-3)
    assert ratios["88Sr+ 674 nm"] == pytest.approx(2.4665e-4, rel=3e-3)
    assert ratios["40Ca+ 729 nm"] / ALPHA_FS == pytest.approx(0.0312, abs=5e-4)
    assert ratios["88Sr+ 674 nm"] / ALPHA_FS == pytest.approx(0.0338, abs=5e-4)
    # the row prints 0.062 alpha and 0.068 alpha, which is k a_0 (twice the amplitude ratio k a_0/2)
    assert 2.0 * ratios["40Ca+ 729 nm"] / ALPHA_FS == pytest.approx(0.062, abs=1e-3)
    assert 2.0 * ratios["88Sr+ 674 nm"] / ALPHA_FS == pytest.approx(0.068, abs=1e-3)
    assert ratios["40Ca+ 729 nm"] < ALPHA_FS / 2.0 / 15.0, "alpha/2 is 16x too large"


def test_the_yb171_cycling_share_is_exactly_one_half() -> None:
    """Section 9.13's Wigner-Eckart row: "the |1,1> -> P3/2 absorption strength is 1/2 in the cycling |2,2>
    component". This clause was printed only by check_atomic.py and CI-enforced as a TEXT regression against
    its committed output; here it is an assertion (audit item E17).
    """
    from qutip_trap.species.dipole import hyperfine_element

    i_spin = Fraction(1, 2)
    j_lo, j_up = Fraction(1, 2), Fraction(3, 2)
    total = 0.0
    cycling = 0.0
    for f_up in (Fraction(1), Fraction(2)):
        for q in (-1, 0, 1):
            m_up = Fraction(1) - q
            if abs(m_up) > f_up:
                continue
            strength = hyperfine_element(Fraction(1), Fraction(1), f_up, m_up, q, j_lo, j_up, i_spin) ** 2
            total += strength
            if (f_up, m_up) == (Fraction(2), Fraction(2)):
                cycling = strength
    assert total == pytest.approx(1.0, abs=1e-14), "absorption out of every lower sublevel sums to 1"
    assert cycling == pytest.approx(0.5, abs=1e-14)
    assert cycling / total == pytest.approx(0.5, abs=1e-14)


def test_the_729_nm_lamb_dicke_projections_of_the_9_14_row() -> None:
    """Section 9.14's Lamb-Dicke row: x0 = 11.246 nm and k x0 = 0.0969 at 729 nm for 40Ca+ at 1 MHz, and the
    two projected values eta(22.5 deg) = 0.0895 and eta(67.5 deg) = 0.0371, which had zero hits repo-wide."""
    from qutip_trap.prep.closed_forms import lamb_dicke_parameter
    from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI

    m = species("40Ca+").mass_u * ATOMIC_MASS_KG
    eta0 = lamb_dicke_parameter(TWO_PI / 729.347e-9, m, TWO_PI * 1e6)
    assert eta0 == pytest.approx(0.0969, abs=2e-4)
    assert eta0 * math.cos(math.radians(22.5)) == pytest.approx(0.0895, abs=5e-4)
    assert eta0 * math.cos(math.radians(67.5)) == pytest.approx(0.0371, abs=5e-4)
    # the two projections are the same beam at complementary angles: their squares sum to eta0^2
    a, b = eta0 * math.cos(math.radians(22.5)), eta0 * math.cos(math.radians(67.5))
    assert a**2 + b**2 == pytest.approx(eta0**2, rel=1e-12)


def test_the_roos_sideband_coupling_needs_a_trap_frequency_the_row_never_prints() -> None:
    """Section 9.14's Roos row closes with "eta Omega/2pi = 57.7 kHz for the sideband", which had zero hits
    repo-wide. RECOMPUTED HERE: the row pairs it with Omega/2pi = 1.088 MHz (vacuum lambda, tau = 1.168 s),
    and eta at 729.347 nm for 40Ca+ is 0.0969 AT 1 MHz, giving 105.4 kHz -- 1.83x the printed value.

    57.7 kHz needs eta = 0.05304, i.e. a mode frequency of 3.34 MHz (eta scales as omega^{-1/2}), or an
    unstated geometric projection of cos = 0.547 (57.2 degrees). The row prints neither, so the number is
    NOT reproducible from its own inputs and is reported rather than pinned; what is pinned is the eta the
    row's own Lamb-Dicke line fixes and the frequency the printed 57.7 kHz implies.
    """
    from qutip_trap.prep.closed_forms import lamb_dicke_parameter
    from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI

    m = species("40Ca+").mass_u * ATOMIC_MASS_KG
    eta_1mhz = lamb_dicke_parameter(TWO_PI / 729.347e-9, m, TWO_PI * 1e6)
    assert eta_1mhz == pytest.approx(0.0969, abs=2e-4)
    assert eta_1mhz * 1.088e6 / 1e3 == pytest.approx(105.4, abs=0.5)
    implied_eta = 57.7e3 / 1.088e6
    assert implied_eta == pytest.approx(0.05304, abs=1e-5)
    implied_mhz = (eta_1mhz / implied_eta) ** 2
    assert implied_mhz == pytest.approx(3.34, abs=0.02), "the mode frequency the printed 57.7 kHz implies"
    assert lamb_dicke_parameter(
        TWO_PI / 729.347e-9, m, TWO_PI * implied_mhz * 1e6
    ) * 1.088e6 / 1e3 == pytest.approx(57.7, abs=0.3)


def test_the_schindler_crosstalk_convention_of_the_9_14_row() -> None:
    """Section 9.14's crosstalk row -- the only 9.14 row with no coverage at all before 2026-09-07: Schindler
    Fig. 17's 22/121 is an AMPLITUDE ratio, 18%, so the intensity crosstalk is its square, 3.3%; reading the
    printed form as already-squared gives 3025%, which is the negative control."""
    amplitude = 22.0 / 121.0
    assert amplitude == pytest.approx(0.1818, abs=5e-4)
    assert amplitude**2 == pytest.approx(0.0331, abs=5e-4)
    assert 100.0 * amplitude == pytest.approx(18.18, abs=0.05)
    assert amplitude**2 < 0.03 * (1.0 + 0.15), "matches Schindler's stated < 3% bound"
    # the negative control: the printed Omega_i^2/Omega_j^2 with the ratio the OTHER way round gives
    # (121/22)^2 = 30.25 -> 3025%, a crosstalk above 100%, which is the tell that the form was misread
    printed = (121.0 / 22.0) ** 2
    assert 100.0 * printed == pytest.approx(3025.0, rel=1e-3)
    assert 100.0 * printed > 100.0


def test_the_9be_negative_controls_of_the_9_13_row() -> None:
    """Three sub-claims of the 9.13 "species-layer negative tests" row that had zero hits: no 9Be+ clock
    point in 8 to 16 mT when A > 0, and the slope and offset at the ROUNDED 0.01194 T."""
    from qutip_trap.species.zeeman import clock_points, transition_sensitivity

    be = species("9Be+")
    s12 = be.level("S1/2")
    flipped = Level(
        s12.name, s12.energy_hz, s12.lifetime_s, -s12.A_hfs_hz, s12.B_hfs_hz, s12.g_J, s12.citations
    )
    hz_bad = HyperfineZeeman(flipped, 1.5, be.mu_I_nuclear_magnetons)
    assert clock_points(hz_bad, "F=2 mF=0", hz_bad, "F=1 mF=1", 80.0, 160.0) == ()
    hz = hyperfine_zeeman(s12, be.nuclear_spin, be.mu_I_nuclear_magnetons)
    (cp,) = clock_points(hz, "F=2 mF=0", hz, "F=1 mF=1", 100.0, 160.0)
    rounded = transition_sensitivity(hz, "F=2 mF=0", hz, "F=1 mF=1", 119.4)
    assert rounded.dnu_dB_hz_per_g * 1e-2 == pytest.approx(-2.82, abs=0.05), "slope in Hz/uT at 0.01194 T"
    assert rounded.frequency_hz - cp.frequency_hz == pytest.approx(6.50, abs=0.3), "Hz above the minimum"
