"""Species data tables (PLAN.md Section 4.5.6, 9.13; M0): every number cited, nothing hyperfine-resolved typed in."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from qutip_trap.species import MODULES, IncompleteSpeciesTable, available, species
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.sources import SOURCES
from qutip_trap.species.table import TAGS, a_hfs_from_two_manifold_splitting
from qutip_trap.species.zeeman import MU_B_OVER_H_HZ_PER_G, g_I_steck
from qutip_trap.units import C_M_PER_S, H_J_S, TWO_PI

HYPERFINE_RESOLVED = re.compile(r"(mF|F=\d|zeeman_energy|dipole_element|rabi|raman_coupling|clebsch)", re.I)


def test_the_species_that_build_and_the_ones_that_cannot() -> None:
    """171Yb+, 40Ca+ and 88Sr+ built before the M0a fix pass of 2026-09-08; 43Ca+ and 9Be+ were added by it
    (their excited-level hyperfine constants and P lifetimes came from the primary literature, PLAN.md
    supplying none), which is what makes the Section 9.13 clock-point anchors runnable against a real
    ``Species`` rather than against raw ``TABLE`` constants.

    137Ba+ was added by the species-gaps pass of the same day, which found the measured
    g_J(6s 2S1/2) = 2.00249192(3) of Marx et al. 1998 that its single remaining gap wanted.

    The two that still raise do so for reasons no transcription can fix. 25Mg+ has no measured g_J and no
    measured 3p hyperfine constants: the g_J gap is now closed BY SEARCH (no measurement of any Mg+ isotope
    and no calculation either, ``mg25.py`` ``_CONSULT``). 133Ba+ lacks A(6p 2P3/2) and A(5d 2D5/2), which NO
    source prints -- only the 623(30) and 83(30) MHz splittings are published. Every gap is DERIVED from the
    table (``required_constants_missing``), so filling one makes the species build, which is exactly what
    happened to 137Ba+ and is why this test is the tripwire for it.
    """
    assert available() == ("171Yb+", "40Ca+", "43Ca+", "137Ba+", "9Be+", "88Sr+")
    assert set(MODULES) - set(available()) == {"25Mg+", "133Ba+"}
    # 133Ba+ refuses for EXACTLY the two constants no paper prints, and for no other reason
    with pytest.raises(IncompleteSpeciesTable) as info:
        species("133Ba+")
    assert {m.quantity.rsplit("(", 1)[-1].rstrip(")") for m in info.value.missing} == {
        "ba133.P32.A_hfs_hz",
        "ba133.D52.A_hfs_hz",
    }
    assert all("NO SOURCE PRINTS THIS CONSTANT" in m.consult for m in info.value.missing), (
        "the gap notes must record the negative search so the next reader does not repeat it"
    )


def test_ba137_builds_and_its_clock_and_zeeman_quantities_are_finite() -> None:
    """137Ba+ builds from the table alone once Marx et al. 1998's g_J is in it.

    PLAN.md prints NO Section 9.13 anchor for 137Ba+ -- it names the isotope as a preset (line 139) and
    carries Ozeri's f^-1 = 3 for it (line 457) and nothing else -- so there is no published number to pin.
    What is pinned instead is the closed form the plan does state for this quantity (Section 9.13's
    ``taylor_c2 = (g_J - g_I)^2 mu_B^2/(2 h^2 nu_0)``, the identity it applies to 171Yb+ and 87Rb), which
    the diagonalization must reproduce to 1e-9 relative, plus the five Ba+ line wavelengths and finiteness
    of every derived Zeeman quantity. Ledger: anchor.species.ba_g_factors.
    """
    ba = species("137Ba+")
    assert ba.nuclear_spin == 1.5 and ba.mu_I_nuclear_magnetons == 0.937365
    assert ba.mass_u == pytest.approx(136.905278, abs=1e-6), "the ION mass (one electron removed)"
    # the five Ba+ lines, all derived from the level energies, none typed in
    for label, nm in (
        ("S1/2-P1/2", 493.545),
        ("D3/2-P1/2", 649.869),
        ("S1/2-P3/2", 455.531),
        ("D5/2-P3/2", 614.341),
        ("D3/2-P3/2", 585.530),
    ):
        assert ba.transition(label).wavelength_vac_m * 1e9 == pytest.approx(nm, abs=1e-3), label
    # A(6s) > 0, so F = 1 lies BELOW F = 2 and the qubit tuple must list it first
    sp = ba.zeeman_spectrum("S1/2", 0.0)
    assert sp.energy_hz("F=2 mF=0") > sp.energy_hz("F=1 mF=0"), "mu_I > 0 gives a NORMAL multiplet"
    assert ba.qubit == ("S1/2 F=1 mF=0", "S1/2 F=2 mF=0")
    t = MODULES["137Ba+"].TABLE
    nu0, d1, d2 = ba.transition_frequency_hz(*ba.qubit, 0.0)
    # the identity the diagonalization must satisfy: Delta E = A (I + 1/2) = 2A at I = 3/2, J = 1/2
    assert nu0 == pytest.approx(2.0 * t["ba137.S12.A_hfs_hz"].value, rel=1e-12)
    # ...which reproduces Blatt and Werth's published splitting to 10 mHz. The residual is NOT physics: the
    # stored A = 4018.87083385 MHz is the printed derived constant, whose last digit rounds up from the
    # exact half of the splitting (4 018 870 833.845 Hz), 0.28 sigma of A's own 0.18 mHz bar. So the pair
    # (A, hfs_splitting_hz) is self-consistent only to that rounding, and this is where it is recorded.
    assert nu0 == pytest.approx(t["ba137.S12.hfs_splitting_hz"].value, abs=0.011)
    assert nu0 - t["ba137.S12.hfs_splitting_hz"].value == pytest.approx(0.010, abs=1e-4)
    assert abs(d1) < 1e-9, "no linear Zeeman term on the mF = 0 <-> mF = 0 line"
    # the closed form of Section 9.13, to 1e-9 relative: this is the identity, not a laboratory number
    g_i = g_I_steck(ba.mu_I_nuclear_magnetons, ba.nuclear_spin)
    closed = ((ba.level("S1/2").g_J - g_i) * MU_B_OVER_H_HZ_PER_G) ** 2 / (2.0 * nu0)
    assert closed == pytest.approx(488.81912, abs=1e-5)
    assert d2 / 2.0 == pytest.approx(closed, rel=1e-9), "taylor_c2 = (1/2) d2nu/dB2 (Section 13)"
    for B in (1.0, 5.0):
        nu, _, _ = ba.transition_frequency_hz(*ba.qubit, B)
        assert (nu - nu0) / B**2 == pytest.approx(closed, rel=1e-5)
    # every derived Zeeman quantity of every level is finite, D levels (I >= 1, B_hfs != 0) included
    for lv in ba.levels:
        s = ba.zeeman_spectrum(lv.name, 4.0)
        assert all(math.isfinite(x) for x in s.energies_hz), lv.name
        assert all(math.isfinite(x) for x in s.dE_dB_hz_per_g), lv.name
        assert all(math.isfinite(x) for x in s.d2E_dB2_hz_per_g2), lv.name


@pytest.mark.parametrize("name", sorted(MODULES))
def test_every_constant_is_cited(name: str) -> None:
    table = MODULES[name].TABLE
    assert table, f"{name} has an empty table"
    for key, c in table.items():
        assert key == c.key
        assert c.source in SOURCES, f"{key}: unknown source key {c.source!r}"
        assert c.tag in TAGS


@pytest.mark.parametrize("name", sorted(MODULES))
def test_nothing_hyperfine_resolved_is_typed_in(name: str) -> None:
    for key in MODULES[name].TABLE:
        assert not HYPERFINE_RESOLVED.search(key.split(".", 1)[1]), key


@pytest.mark.parametrize("name", sorted(set(MODULES) - set(available())))
def test_incomplete_tables_say_what_is_missing(name: str) -> None:
    with pytest.raises(IncompleteSpeciesTable) as info:
        species(name)
    assert info.value.missing, name
    assert all(m.consult for m in info.value.missing)


def test_the_retired_reading_of_the_yb171_quadratic_shift_is_not_what_the_package_computes() -> None:
    """The positive half of the grep: the adopted g_J = 2.002615 gives 310.87 Hz/G^2, and none of the three
    retired g_J values does (PLAN.md 9.13's row is 310.87 +- 0.02, and 310.85 sits exactly on its boundary)."""
    from qutip_trap.species.zeeman import hyperfine_zeeman

    yb = species("171Yb+")
    hz = hyperfine_zeeman(yb.level("S1/2"), yb.nuclear_spin, yb.mu_I_nuclear_magnetons)
    from qutip_trap.species.zeeman import transition_sensitivity

    s = transition_sensitivity(hz, "F=0 mF=0", hz, "F=1 mF=0", 1.0)
    assert s.taylor_c2_hz_per_g2 == pytest.approx(310.869, abs=1e-3)
    assert abs(s.taylor_c2_hz_per_g2 - 310.85) > 0.015, "310.85 is the retired g_J = 2.00254 reading"


# ---- 171Yb+ ----------------------------------------------------------------------------------------------


def test_yb171_ground_state_constants() -> None:
    yb = species("171Yb+")
    s12 = yb.level("S1/2")
    assert s12.A_hfs_hz == 12_642_812_118.5, (
        "the zero-field splitting of Olmschenk 2007, equal to A for I = J = 1/2"
    )
    assert s12.g_J == 2.002615
    assert s12.B_hfs_hz == 0.0
    assert yb.nuclear_spin == 0.5
    assert yb.mu_I_nuclear_magnetons == 0.49367
    assert yb.mass_u == pytest.approx(170.93578, abs=1e-5), "the ION mass of Section 9.16 row 13-6"


def test_yb171_i_sat_anchor_of_section_9_13() -> None:
    """171Yb+ 369.5 nm with the partial 19.62 MHz rate: I_sat = 50.83 mW/cm^2; unconverted gamma_hz gives 8.09."""
    tr = species("171Yb+").transition("S1/2-P1/2")
    assert tr.gamma_hz == pytest.approx(19.72e6, rel=1e-3)
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(19.62e6, rel=1e-3)
    assert tr.i_sat_w_m2 * 0.1 == pytest.approx(50.83, abs=0.01)
    unconverted = math.pi * H_J_S * C_M_PER_S * (tr.gamma_hz * tr.branching) / (3.0 * tr.wavelength_vac_m**3)
    assert unconverted * 0.1 == pytest.approx(8.09, abs=0.01)


def test_yb171_channel_wavelengths() -> None:
    yb = species("171Yb+")
    assert yb.transition("S1/2-P1/2").wavelength_vac_m == pytest.approx(369.52e-9, abs=0.01e-9)
    assert yb.transition("D3/2-P1/2").wavelength_vac_m == pytest.approx(2.438e-6, abs=0.001e-6), (
        "the P1/2 -> D3/2 channel is at 2.438 um, not at the 935 nm repump (Section 4.5.6)"
    )
    t = MODULES["171Yb+"].TABLE
    repump = 1e-2 / (t["yb171.bracket_3D32_12.energy_cm"].value - t["yb171.D32.energy_cm"].value)
    clear = 1e-2 / (t["yb171.bracket_1D52_52.energy_cm"].value - t["yb171.F72.energy_cm"].value)
    assert repump == pytest.approx(935.2e-9, abs=0.1e-9)
    assert clear == pytest.approx(638.6e-9, abs=0.1e-9)


def test_yb171_branching_is_conditional_on_the_lifetime_and_sums_to_one() -> None:
    yb = species("171Yb+")
    sp = yb.transition("S1/2-P1/2")
    dp = yb.transition("D3/2-P1/2")
    assert sp.branching + dp.branching == pytest.approx(1.0)
    assert dp.branching == 0.00501
    assert yb.level("P1/2").lifetime_s == 8.07e-9
    assert "CONDITIONAL" in MODULES["171Yb+"].TABLE["yb171.P12.branching_to_D32"].note


def test_yb171_inverted_bracket_level_has_negative_A() -> None:
    yb = species("171Yb+")
    assert yb.level("3D[3/2]1/2").A_hfs_hz == pytest.approx(-2.2095e9)
    assert yb.level("D3/2").A_hfs_hz == pytest.approx(0.43e9), "A = splitting/2 for J = 3/2, I = 1/2"
    assert yb.level("P1/2").A_hfs_hz == pytest.approx(2.105e9)


def test_two_manifold_conversion_rules() -> None:
    from qutip_trap.species.table import Cited

    c = Cited(value=1.0e9, unit="Hz", source="Steck", key="test.x")
    half = Fraction(1, 2)
    assert a_hfs_from_two_manifold_splitting(c, Fraction(7, 2), half, inverted=True) == pytest.approx(-0.25e9)
    assert a_hfs_from_two_manifold_splitting(c, half, Fraction(3, 2), inverted=False) == pytest.approx(0.5e9)
    with pytest.raises(ValueError):
        a_hfs_from_two_manifold_splitting(c, Fraction(3, 2), Fraction(3, 2), inverted=False)


# ---- 40Ca+ ------------------------------------------------------------------------------------------------


def test_ca40_quadrupole_records() -> None:
    ca = species("40Ca+")
    d52 = ca.transition("S1/2-D5/2")
    assert d52.multipole == "E2"
    assert d52.wavelength_vac_m == pytest.approx(729.347e-9, rel=2e-6)
    assert d52.gamma_hz == pytest.approx(1.0 / (TWO_PI * 1.168), rel=1e-9)
    assert d52.quadrupole_element_au == 9.740 and d52.quadrupole_convention == "johnson_1_15"
    d32 = ca.transition("S1/2-D3/2")
    assert d32.wavelength_vac_m == pytest.approx(732.591e-9, rel=2e-6)
    assert ca.level("D3/2").lifetime_s == 1.176 and ca.level("D5/2").lifetime_s == 1.168


def test_ca40_lande_factors_and_spin_zero() -> None:
    ca = species("40Ca+")
    assert ca.nuclear_spin == 0.0 and ca.mu_I_nuclear_magnetons == 0.0
    assert all(lv.A_hfs_hz == 0.0 and lv.B_hfs_hz == 0.0 for lv in ca.levels)
    assert ca.level("D5/2").g_J == pytest.approx(1.2, abs=1e-3)
    assert ca.level("P1/2").g_J == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_ca40_397_nm_i_sat_anchor_now_comes_out_of_the_table() -> None:
    """Section 9.13's 2.045 e a0 and 45.11 mW/cm^2 are the PARTIAL-rate closed forms of the 397 nm line at
    its quoted 21.57 MHz, and since the reading was flipped on 2026-09-08 the TABLE reproduces them.

    The chain is all measured: Hettrich et al. 2015's tau(P1/2) = 6.904(26) ns gives a TOTAL 23.0526 MHz,
    and Ramm et al. 2013's branching 0.06435(7) leaves 21.5691 MHz into S1/2 -- Hettrich's own printed
    gamma_PS = 2 pi x 21.57(8) MHz to 4.0e-5, i.e. 0.02 of its uncertainty. The residual D3/2 partial rate
    1.4834 MHz reproduces his gamma_PD = 2 pi x 1.482(8) MHz inside the same bar.

    Two pins moved with the flip (the table used to read 21.57 MHz as the TOTAL rate and pair it with
    Section 8.1's 0.06 branching): the reduced element 1.9832 -> 2.0455 e a0 and I_sat 42.37 -> 45.07
    mW/cm^2, both at the vacuum wavelength the table stores. PLAN.md's 45.11 is its own AIR 396.85 nm, so
    both wavelengths are pinned. Ledger anchor.ca40.i_sat_chain, conv.ca40_linewidth_reading.
    """
    from qutip_trap.species.dipole import reduced_element_from_partial_rate
    from qutip_trap.units import A_0_M, E_C

    ca = species("40Ca+")
    tr = ca.transition("S1/2-P1/2")
    table = MODULES["40Ca+"].TABLE
    assert tr.wavelength_vac_m == pytest.approx(396.959e-9, abs=0.001e-9)
    # the total rate is the measured lifetime's, NOT the quoted linewidth (which would be 7.379 ns)
    assert ca.level("P1/2").lifetime_s == 6.904e-9
    assert tr.gamma_hz == pytest.approx(23.05257e6, rel=1e-6)
    assert tr.gamma_hz / 21.57e6 == pytest.approx(1.0688, rel=1e-3), "the total is 6.9 % above the quoted"
    # THE IDENTITY the flip exists for: the table's partial rate IS Hettrich's printed gamma_PS
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(21.57e6, rel=1e-3), (
        "Hettrich et al. 2015: gamma_PS = 2 pi x 21.57(8) MHz, the PARTIAL P1/2 -> S1/2 rate"
    )
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(21.569137e6, rel=1e-6)
    assert table["ca40.P12.partial_rate_to_S12_hz"].value == 21.57e6
    d32 = ca.transition("D3/2-P1/2")
    assert d32.partial_rate_rad_s / TWO_PI == pytest.approx(1.482e6, abs=0.008e6), (
        "Hettrich's gamma_PD = 2 pi x 1.482(8) MHz, the other half of his total"
    )
    # the plan's own printed pair, now derived rather than assumed: 2.045 e a0 and 45.11 mW/cm^2 at its air
    # wavelength, 2.0455 and 45.07 at the vacuum one
    for lam, element, i_sat in ((396.85e-9, 2.0446, 45.106), (tr.wavelength_vac_m, 2.0455, 45.069)):
        omega = TWO_PI * C_M_PER_S / lam
        d = reduced_element_from_partial_rate(tr.partial_rate_rad_s, omega, Fraction(1, 2), Fraction(1, 2))
        assert d / (E_C * A_0_M) == pytest.approx(element, abs=5e-4)
        assert math.pi * H_J_S * C_M_PER_S * tr.partial_rate_rad_s / (3.0 * lam**3) * 0.1 == pytest.approx(
            i_sat, abs=0.01
        )
    assert tr.i_sat_w_m2 * 0.1 == pytest.approx(45.069, abs=0.01)
    # the quoted linewidth survives only as the contested cross-check nothing is built from
    assert table["ca40.P12.linewidth_quoted_hz"].tag == "contested"
    assert tr.gamma_hz != table["ca40.P12.linewidth_quoted_hz"].value


def test_ca40_393_nm_partial_rate_disagrees_with_the_plans_quoted_23_4_mhz() -> None:
    """The table's 393 nm partial rate is 22.4071 MHz and PLAN.md 9.13's quoted 23.4 MHz matches nothing.

    Meir et al., Phys. Rev. A 101, 012509 (2020) measure tau(P3/2) = 6.639(42) ns, a TOTAL of 23.9727 MHz,
    of which Gerritsma et al. 2008's S1/2 share 1 - 0.0587 - 0.00661 = 0.93469 leaves 22.4071 MHz. Jin and
    Church 1993's 6.924(19) ns would give 22.9860 total and 21.4848 partial. The plan's 23.4 MHz is 4.4 %
    above the table's partial rate, 2.4 % below Meir's total, and has no named source.

    So Section 9.13's printed 2.972 e a0 / 50.25 mW/cm^2 are NOT the table's numbers: they are the closed
    forms at a partial rate of exactly 23.4 MHz and the plan's own AIR 393.37 nm, and they are pinned below
    as exactly that and labelled as the plan's, never as the table's. The table gives 2.9085 e a0 /
    48.113 mW/cm^2 at the same air wavelength and 2.9097 / 48.074 at the vacuum 393.478 nm it stores (up
    from 2.8747 / 46.93 when the total was read off the quoted 23.4 MHz). Ledger
    anchor.ca40.p32_linewidth_readings, anchor.ca40.i_sat_chain.
    """
    from qutip_trap.species.dipole import reduced_element_from_partial_rate
    from qutip_trap.units import A_0_M, E_C

    ca = species("40Ca+")
    tr = ca.transition("S1/2-P3/2")
    table = MODULES["40Ca+"].TABLE
    assert tr.wavelength_vac_m == pytest.approx(393.478e-9, abs=0.001e-9)
    assert ca.level("P3/2").lifetime_s == 6.639e-9
    assert tr.gamma_hz == pytest.approx(23.972728e6, rel=1e-6)
    assert tr.branching == pytest.approx(0.93469, abs=1e-9)
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(22.407069e6, rel=1e-6)
    # PLAN.md 9.13's printed pair, as a CLOSED FORM at the plan's own quoted 23.4 MHz partial rate and air
    # wavelength -- not the table's value, which is the row below it
    plan_partial = TWO_PI * 23.4e6
    for lam, element, i_sat in ((393.37e-9, 2.9722, 50.246), (tr.wavelength_vac_m, 2.9734, 50.204)):
        omega = TWO_PI * C_M_PER_S / lam
        d = reduced_element_from_partial_rate(plan_partial, omega, Fraction(1, 2), Fraction(3, 2))
        assert d / (E_C * A_0_M) == pytest.approx(element, abs=1e-3), (
            "PLAN.md 9.13 prints 2.972 e a0 and 50.25 mW/cm^2 for the 393 nm line"
        )
        assert math.pi * H_J_S * C_M_PER_S * plan_partial / (3.0 * lam**3) * 0.1 == pytest.approx(
            i_sat, abs=0.02
        )
    # what the TABLE gives at the same two wavelengths
    for lam, element, i_sat in ((393.37e-9, 2.9085, 48.113), (tr.wavelength_vac_m, 2.9097, 48.074)):
        omega = TWO_PI * C_M_PER_S / lam
        d = reduced_element_from_partial_rate(tr.partial_rate_rad_s, omega, Fraction(1, 2), Fraction(3, 2))
        assert d / (E_C * A_0_M) == pytest.approx(element, abs=1e-3), (
            "PLAN.md 9.13 prints 2.972 e a0; the table's measured chain gives 2.9085 at the same wavelength"
        )
        assert math.pi * H_J_S * C_M_PER_S * tr.partial_rate_rad_s / (3.0 * lam**3) * 0.1 == pytest.approx(
            i_sat, abs=0.02
        )
    assert tr.i_sat_w_m2 * 0.1 == pytest.approx(48.074, abs=0.02), (
        "PLAN.md 9.13 prints 50.25 mW/cm^2; the table gives 48.074 at the vacuum wavelength"
    )
    # Jin and Church 1993's lifetime, the only other measurement, does not rescue the quoted 23.4 either
    jin_total = 1.0 / (TWO_PI * 6.924e-9)
    assert jin_total == pytest.approx(22.985983e6, rel=1e-6)
    assert jin_total * tr.branching == pytest.approx(21.484768e6, rel=1e-6)
    assert table["ca40.P32.linewidth_quoted_hz"].tag == "contested"
    assert tr.gamma_hz != table["ca40.P32.linewidth_quoted_hz"].value


# ---- 88Sr+ ------------------------------------------------------------------------------------------------


def test_sr88_quadrupole_record() -> None:
    sr = species("88Sr+")
    tr = sr.transition("S1/2-D5/2")
    assert tr.wavelength_vac_m == pytest.approx(674.025591e-9, rel=1e-6), (
        "the NIST clock-frequency wavelength"
    )
    assert tr.gamma_hz == pytest.approx(0.4073, abs=1e-4), "A/2pi = 0.4073 Hz (Section 9.14)"
    assert tr.branching == pytest.approx(1.0 - 9e-5)
    assert sr.level("D5/2").lifetime_s == 0.3908


# ---- the Species invariants ----------------------------------------------------------------------------------


def _level(name: str, e: float, tau: float | None = None) -> Level:
    return Level(name, e, tau, 0.0, 0.0, 2.0, ("Steck",))


def test_species_refuses_inconsistent_wavelength() -> None:
    lvls = (_level("S1/2", 0.0), _level("P1/2", 7.5e14, 7e-9))
    bad = Transition("S1/2", "P1/2", 400e-9, 1.0 / (TWO_PI * 7e-9), 1.0, "E1", ("Steck",))
    with pytest.raises(ValueError, match="wavelength"):
        Species("X+", 40.0, 0.0, 0.0, lvls, (bad,), ("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"), "S1/2-P1/2", (), None)


def test_species_refuses_hyperfine_structure_for_spin_zero() -> None:
    with pytest.raises(ValueError, match="spin-zero"):
        Species(
            "X+",
            40.0,
            0.0,
            0.0,
            (Level("S1/2", 0.0, None, 1.0e9, 0.0, 2.0, ("Steck",)),),
            (),
            ("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"),
            "S1/2-S1/2",
            (),
            None,
        )


def test_transition_requires_a_convention_for_a_quadrupole_element() -> None:
    with pytest.raises(ValueError):
        Transition("S1/2", "D5/2", 729e-9, 0.136, 1.0, "E2", ("Steck",), quadrupole_element_au=9.74)
    with pytest.raises(ValueError):
        Transition(
            "S1/2",
            "P1/2",
            397e-9,
            2e7,
            1.0,
            "E1",
            ("Steck",),
            quadrupole_element_au=9.74,
            quadrupole_convention="racah_c2",
        )
