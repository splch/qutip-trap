"""The species tables (PLAN.md Section 4.5.6): every number cited, nothing hyperfine-resolved typed in, and the
records they build."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest
from scipy.constants import physical_constants

from qutip_trap.provenance import Cited
from qutip_trap.species import MODULES, available, species
from qutip_trap.species.dipole import reduced_element_from_partial_rate
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.raman import AtomicStructure
from qutip_trap.species.sources import SOURCES
from qutip_trap.species.table import IncompleteSpeciesTable, a_hfs_from_two_manifold_splitting
from qutip_trap.species.zeeman import MU_B_OVER_H_HZ_PER_G, g_I_steck
from qutip_trap.units import C_M_PER_S, E_C, TWO_PI

HYPERFINE_RESOLVED = re.compile(r"(mF|F=\d|zeeman_energy|dipole_element|rabi|raman_coupling|clebsch)", re.I)
LOCATOR = re.compile(r"\((?:19|20)\d\d\)|arXiv:\d{4}\.\d{4,5}|doi:10\.")
"""A citable locator: a parenthesised year, an arXiv id, or a DOI."""
SELF_REFERENTIAL_SOURCES = ("PLAN_4_5_1", "PLAN_8_1", "PLAN_4_5_6", "PLAN_background")
"""Source keys that point at PLAN.md rather than at a primary reference."""
BA133 = MODULES["133Ba+"].TABLE
BA137 = MODULES["137Ba+"].TABLE


def test_the_species_that_build_and_the_two_that_cannot() -> None:
    """Six species build, and 25Mg+ and 133Ba+ refuse with exactly their missing ledger ids (g_J and the 3p hyperfine
    constants; two A constants), each marked to consult."""
    assert available() == ("171Yb+", "40Ca+", "43Ca+", "137Ba+", "9Be+", "88Sr+")
    missing = {}
    for name in set(MODULES) - set(available()):
        with pytest.raises(IncompleteSpeciesTable) as info:
            species(name)
        assert all(m.consult for m in info.value.missing)
        missing[name] = {m.ledger_id for m in info.value.missing}
    assert missing == {
        "133Ba+": {"ba133.P32.A_hfs_hz", "ba133.D52.A_hfs_hz"},
        "25Mg+": {"mg25.S12.g_J", "mg25.P12.A_hfs_hz", "mg25.P32.A_hfs_hz", "mg25.P32.B_hfs_hz"},
    }


@pytest.mark.parametrize("name", sorted(MODULES))
def test_every_constant_is_cited_to_a_known_source(name: str) -> None:
    table = MODULES[name].TABLE
    assert table, f"{name} has an empty table"
    assert len(table) == len(MODULES[name]._ENTRIES), f"{name}: a duplicate id drops an entry"
    for ledger_id, c in table.items():
        assert c.source in SOURCES, f"{ledger_id}: unknown source key {c.source!r}"


@pytest.mark.parametrize("name", sorted(MODULES))
def test_no_constant_is_verified_against_the_plan_or_a_relay(name: str) -> None:
    """No constant tagged verified cites PLAN.md or a relay (``_via_``) source."""
    for ledger_id, c in MODULES[name].TABLE.items():
        if c.tag == "verified":
            assert c.source not in SELF_REFERENTIAL_SOURCES and "_via_" not in c.source, ledger_id


def test_every_source_key_carries_a_locator() -> None:
    """Every source except the PLAN, NIST and Steck keys carries a year, an arXiv id or a DOI."""
    for key, text in SOURCES.items():
        if not key.startswith(("PLAN_", "NIST_", "Steck")):
            assert LOCATOR.search(text), f"{key} names no locator: {text[:100]}"


@pytest.mark.parametrize("name", sorted(MODULES))
def test_nothing_hyperfine_resolved_is_typed_in(name: str) -> None:
    for ledger_id in MODULES[name].TABLE:
        assert not HYPERFINE_RESOLVED.search(ledger_id.split(".", 1)[1]), ledger_id


def test_two_manifold_conversion_rules() -> None:
    c = Cited(value=1.0e9, unit="Hz", source="Steck", ledger_id="test.x")
    half = Fraction(1, 2)
    assert a_hfs_from_two_manifold_splitting(c, Fraction(7, 2), half, inverted=True) == pytest.approx(-0.25e9)
    assert a_hfs_from_two_manifold_splitting(c, half, Fraction(3, 2), inverted=False) == pytest.approx(0.5e9)
    with pytest.raises(ValueError):
        a_hfs_from_two_manifold_splitting(c, Fraction(3, 2), Fraction(3, 2), inverted=False)


def _i_sat_mw_cm2(partial_rate_rad_s: float, wavelength_m: float) -> float:
    """``Transition.i_sat_w_m2`` of a unit-branching line of this partial rate, in mW/cm^2."""
    line = Transition("S1/2", "P1/2", wavelength_m, partial_rate_rad_s / TWO_PI, 1.0, "E1", ("Steck",))
    return line.i_sat_w_m2 * 0.1


def _element_e_a0(partial_rate_rad_s: float, wavelength_m: float, j_upper: Fraction) -> float:
    omega = TWO_PI * C_M_PER_S / wavelength_m
    return reduced_element_from_partial_rate(partial_rate_rad_s, omega, Fraction(1, 2), j_upper) / (
        E_C * physical_constants["Bohr radius"][0]
    )


# ---- 171Yb+ ----------------------------------------------------------------------------------------------------------


def test_yb171_ground_state_constants() -> None:
    yb = species("171Yb+")
    s12 = yb.level("S1/2")
    assert s12.A_hfs_hz == 12_642_812_118.5, "the zero-field splitting, equal to A for I = J = 1/2"
    assert s12.g_J == 2.002615 and s12.B_hfs_hz == 0.0
    assert yb.nuclear_spin == 0.5 and yb.mu_I_nuclear_magnetons == 0.49367
    assert yb.mass_u == pytest.approx(170.93578, abs=1e-5), "the ION mass"


def test_yb171_i_sat_anchor() -> None:
    """171Yb+ 369.5 nm: gamma = 19.72 MHz with a 19.62 MHz partial rate (1e-3) gives I_sat = 50.83 mW/cm^2 (0.01), and
    the partial rate passed in Hz instead of rad/s would give 8.09."""
    tr = species("171Yb+").transition("S1/2-P1/2")
    assert tr.gamma_hz == pytest.approx(19.72e6, rel=1e-3)
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(19.62e6, rel=1e-3)
    assert tr.i_sat_w_m2 * 0.1 == pytest.approx(50.83, abs=0.01)
    assert _i_sat_mw_cm2(tr.gamma_hz * tr.branching, tr.wavelength_vac_m) == pytest.approx(8.09, abs=0.01)


def test_yb171_channel_wavelengths() -> None:
    yb = species("171Yb+")
    assert yb.transition("S1/2-P1/2").wavelength_vac_m == pytest.approx(369.52e-9, abs=0.01e-9)
    assert yb.transition("D3/2-P1/2").wavelength_vac_m == pytest.approx(2.438e-6, abs=0.001e-6), (
        "the P1/2 -> D3/2 channel is at 2.438 um, not at the 935 nm repump"
    )
    t = MODULES["171Yb+"].TABLE
    repump = 1e-2 / (t["yb171.bracket_3D32_12.energy_cm"].value - t["yb171.D32.energy_cm"].value)
    clear = 1e-2 / (t["yb171.bracket_1D52_52.energy_cm"].value - t["yb171.F72.energy_cm"].value)
    assert repump == pytest.approx(935.2e-9, abs=0.1e-9)
    assert clear == pytest.approx(638.6e-9, abs=0.1e-9)


def test_yb171_branchings_and_the_p32_doublet_partner() -> None:
    """P1/2 branches 0.00501 to D3/2 (conditional on tau = 8.07 ns) and P3/2 (6.15 ns, A = 875.4 MHz)
    0.9875/0.0017/0.0108 into S1/2/D3/2/D5/2, so S1/2 couples to both fine-structure partners."""
    yb = species("171Yb+")
    sp, dp = yb.transition("S1/2-P1/2"), yb.transition("D3/2-P1/2")
    assert sp.branching + dp.branching == pytest.approx(1.0) and dp.branching == 0.00501
    assert yb.level("P1/2").lifetime_s == 8.07e-9
    assert "CONDITIONAL" in MODULES["171Yb+"].TABLE["yb171.P12.branching_to_D32"].note
    p32 = yb.level("P3/2")
    assert p32.lifetime_s == 6.15e-9 and p32.A_hfs_hz == 875.4e6
    assert yb.transition("S1/2-P3/2").branching == pytest.approx(0.9875, abs=1e-12)
    assert yb.transition("D3/2-P3/2").branching == 0.0017
    assert yb.transition("D5/2-P3/2").branching == 0.0108
    st = AtomicStructure(yb, 5.0, (0.0, 0.0, 1.0))
    assert sorted(st.upper_levels_of("S1/2")) == ["P1/2", "P3/2"]
    assert len(st.states_of("P3/2")) == 8


def test_yb171_the_935_nm_repump_is_tabulated_and_the_297_nm_channel_declared() -> None:
    yb = species("171Yb+")
    tr = yb.transition("D3/2-3D[3/2]1/2")
    assert tr.multipole == "E1"
    assert tr.wavelength_vac_m == pytest.approx(935.186e-9, rel=1e-5)
    declared = yb.level("3D[3/2]1/2").untabulated_branching
    assert len(declared) == 1 and "297" in declared[0][0]
    assert tr.branching + declared[0][1] == pytest.approx(1.0, abs=1e-12)
    assert tr.branching == pytest.approx(0.01603, rel=1e-3)


def test_yb171_hyperfine_constants_from_the_printed_splittings() -> None:
    yb = species("171Yb+")
    assert yb.level("3D[3/2]1/2").A_hfs_hz == pytest.approx(-2.2095e9), "an inverted multiplet"
    assert yb.level("D3/2").A_hfs_hz == pytest.approx(0.43e9), "A = splitting/2 for J = 3/2, I = 1/2"
    assert yb.level("P1/2").A_hfs_hz == pytest.approx(2.105e9)


# ---- 40Ca+ ----------------------------------------------------------------------------------------------------------


def test_ca40_quadrupole_records() -> None:
    ca = species("40Ca+")
    d52 = ca.transition("S1/2-D5/2")
    assert d52.multipole == "E2"
    assert d52.wavelength_vac_m == pytest.approx(729.347e-9, rel=2e-6)
    assert d52.gamma_hz == pytest.approx(1.0 / (TWO_PI * 1.168), rel=1e-9)
    assert ca.transition("S1/2-D3/2").wavelength_vac_m == pytest.approx(732.591e-9, rel=2e-6)
    assert ca.level("D3/2").lifetime_s == 1.176 and ca.level("D5/2").lifetime_s == 1.168


def test_ca40_lande_factors_and_spin_zero() -> None:
    ca = species("40Ca+")
    assert ca.nuclear_spin == 0.0 and ca.mu_I_nuclear_magnetons == 0.0
    assert all(lv.A_hfs_hz == 0.0 and lv.B_hfs_hz == 0.0 for lv in ca.levels)
    assert ca.level("D5/2").g_J == pytest.approx(1.2, abs=1e-3)
    assert ca.level("P1/2").g_J == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_ca40_397_nm_i_sat_anchor_comes_out_of_the_table() -> None:
    """Hettrich's tau(P1/2) = 6.904 ns and Ramm's branching give the 40Ca+ 397 nm partial rate 21.569137 MHz (1e-6), a
    2.0455 e a0 reduced element and I_sat = 45.069 mW/cm^2 at the stored vacuum wavelength."""
    ca = species("40Ca+")
    tr = ca.transition("S1/2-P1/2")
    table = MODULES["40Ca+"].TABLE
    assert tr.wavelength_vac_m == pytest.approx(396.959e-9, abs=0.001e-9)
    assert ca.level("P1/2").lifetime_s == 6.904e-9
    assert tr.gamma_hz == pytest.approx(23.05257e6, rel=1e-6), (
        "the total rate is the lifetime's, not the quoted 21.57"
    )
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(21.569137e6, rel=1e-6)
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(
        table["ca40.P12.partial_rate_to_S12_hz"].value, rel=1e-3
    )
    assert ca.transition("D3/2-P1/2").partial_rate_rad_s / TWO_PI == pytest.approx(1.482e6, abs=0.008e6)
    assert _element_e_a0(tr.partial_rate_rad_s, tr.wavelength_vac_m, Fraction(1, 2)) == pytest.approx(
        2.0455, abs=5e-4
    )
    assert tr.i_sat_w_m2 * 0.1 == pytest.approx(45.069, abs=0.01)
    assert table["ca40.P12.linewidth_quoted_hz"].tag == "contested"


def test_ca40_393_nm_partial_rate_and_i_sat_from_the_table() -> None:
    """Meir's tau(P3/2) = 6.639 ns and Gerritsma's 0.93469 S1/2 share give a 22.407069 MHz partial rate (1e-6), a 2.9097 e a0
    reduced element and I_sat = 48.074 mW/cm^2 at the vacuum 393.478 nm; the quoted linewidth is tagged contested."""
    ca = species("40Ca+")
    tr = ca.transition("S1/2-P3/2")
    assert tr.wavelength_vac_m == pytest.approx(393.478e-9, abs=0.001e-9)
    assert ca.level("P3/2").lifetime_s == 6.639e-9
    assert tr.gamma_hz == pytest.approx(23.972728e6, rel=1e-6)
    assert tr.branching == pytest.approx(0.93469, abs=1e-9)
    assert tr.partial_rate_rad_s / TWO_PI == pytest.approx(22.407069e6, rel=1e-6)
    assert _element_e_a0(tr.partial_rate_rad_s, tr.wavelength_vac_m, Fraction(3, 2)) == pytest.approx(
        2.9097, abs=1e-3
    )
    assert tr.i_sat_w_m2 * 0.1 == pytest.approx(48.074, abs=0.02)
    assert MODULES["40Ca+"].TABLE["ca40.P32.linewidth_quoted_hz"].tag == "contested"


def test_ca40_the_854_and_850_nm_lines_carry_the_gerritsma_split() -> None:
    ca = species("40Ca+")
    assert ca.transition("D5/2-P3/2").branching == 0.0587
    assert ca.transition("D3/2-P3/2").branching == 0.00661
    assert ca.transition("D5/2-P3/2").wavelength_vac_m == pytest.approx(854.444e-9, rel=1e-5)
    assert ca.transition("D3/2-P3/2").wavelength_vac_m == pytest.approx(850.036e-9, rel=1e-5)
    assert AtomicStructure(ca, 4.0, (0.0, 0.0, 1.0)).tabulated_branching_total("P3/2") == pytest.approx(
        1.0, abs=1e-12
    )


# ---- 88Sr+ ----------------------------------------------------------------------------------------------------------


def test_sr88_quadrupole_record_and_e1_lines() -> None:
    sr = species("88Sr+")
    tr = sr.transition("S1/2-D5/2")
    assert tr.wavelength_vac_m == pytest.approx(674.025591e-9, rel=1e-9), "from the clock frequency"
    assert tr.gamma_hz == pytest.approx(0.4073, abs=1e-4), "A/2pi = 0.4073 Hz"
    assert tr.branching == pytest.approx(1.0 - 9e-5)
    assert sr.level("D5/2").lifetime_s == 0.3908
    assert sorted(AtomicStructure(sr, 4.0, (0.0, 0.0, 1.0)).upper_levels_of("S1/2")) == ["P1/2", "P3/2"]
    assert sr.transition(sr.cycling).wavelength_vac_m == pytest.approx(421.671e-9, rel=1e-5)
    assert all(sr.transition(label).multipole == "E1" for label in sr.repumps)
    assert sr.transition("D5/2-P3/2").wavelength_vac_m == pytest.approx(1033.014e-9, rel=1e-5)


# ---- Ba+ ------------------------------------------------------------------------------------------------------------


def test_ba137_builds_and_its_clock_and_zeeman_quantities_are_finite() -> None:
    """137Ba+ builds with its five line wavelengths (1e-3 nm), a normal ground multiplet, finite Zeeman values and the
    clock coefficient (g_J - g_I)^2 mu_B^2/(2 h^2 nu_0) = 488.81912 Hz/G^2 the diagonalization reproduces to 1e-9."""
    ba = species("137Ba+")
    assert ba.nuclear_spin == 1.5 and ba.mu_I_nuclear_magnetons == 0.937365
    assert ba.mass_u == pytest.approx(136.905278, abs=1e-6), "the ION mass"
    for label, nm in (
        ("S1/2-P1/2", 493.545),
        ("D3/2-P1/2", 649.869),
        ("S1/2-P3/2", 455.531),
        ("D5/2-P3/2", 614.341),
        ("D3/2-P3/2", 585.530),
    ):
        assert ba.transition(label).wavelength_vac_m * 1e9 == pytest.approx(nm, abs=1e-3), label
    sp = ba.zeeman_spectrum("S1/2", 0.0)
    assert sp.energy_hz("F=2 mF=0") > sp.energy_hz("F=1 mF=0"), "mu_I > 0 gives a normal multiplet"
    assert ba.qubit == ("S1/2 F=1 mF=0", "S1/2 F=2 mF=0")
    nu0, d1, d2 = ba.transition_frequency_hz(*ba.qubit, 0.0)
    assert nu0 == pytest.approx(2.0 * BA137["ba137.S12.A_hfs_hz"].value, rel=1e-12), "Delta E = A (I + 1/2)"
    # the printed A rounds up from half the printed splitting, so the pair agrees only to 10 mHz
    assert nu0 - BA137["ba137.S12.hfs_splitting_hz"].value == pytest.approx(0.010, abs=1e-4)
    assert abs(d1) < 1e-9
    g_i = g_I_steck(ba.mu_I_nuclear_magnetons, ba.nuclear_spin)
    closed = ((ba.level("S1/2").g_J - g_i) * MU_B_OVER_H_HZ_PER_G) ** 2 / (2.0 * nu0)
    assert closed == pytest.approx(488.81912, abs=1e-5)
    assert d2 / 2.0 == pytest.approx(closed, rel=1e-9)
    for B in (1.0, 5.0):
        nu, _, _ = ba.transition_frequency_hz(*ba.qubit, B)
        assert (nu - nu0) / B**2 == pytest.approx(closed, rel=1e-5)
    for lv in ba.levels:
        s = ba.zeeman_spectrum(lv.name, 4.0)
        assert all(math.isfinite(x) for x in (*s.energies_hz, *s.dE_dB_hz_per_g, *s.d2E_dB2_hz_per_g2)), (
            lv.name
        )


def test_ba137_levels_use_the_measured_g_factors_and_not_the_asd_column() -> None:
    """137Ba+ takes the measured g_J (Marx 1998, Knoell 1996, Arnold 2020) and the 6p Lande values, and stores the NIST
    ASD column (0.79, 1.12, 1.32) as contested, its 1.12 6.69 % from the measured D5/2 value."""
    ba = species("137Ba+")
    assert ba.level("S1/2").g_J == 2.00249192
    assert ba.level("D3/2").g_J == 0.7993278
    assert ba.level("D5/2").g_J == 1.20036739
    assert ba.level("P1/2").g_J == pytest.approx(0.665894, abs=1e-6)
    assert ba.level("P3/2").g_J == pytest.approx(1.334106, abs=1e-6)
    assert "PLAN_background" in ba.level("P3/2").citations
    for key, value in (("D32.g_J_asd", 0.79), ("D52.g_J_asd", 1.12), ("P32.g_J", 1.32)):
        c = BA137[f"ba137.{key}"]
        assert c.value == value and c.tag == "contested" and c.source == "NIST_ASD_5_12"
    measured = BA137["ba137.D52.g_J"].value
    assert abs(1.12 - measured) / measured == pytest.approx(0.0669, abs=5e-4)


def test_arnolds_d52_g_factor_is_the_ratio_times_marxs_ground_state_g_factor() -> None:
    """g_D = r g_S with Arnold's r = 0.59943681 lands 2.14e-8 (5 %) from the stored value, inside its published bar."""
    stored = BA137["ba137.D52.g_J"]
    assert stored.uncertainty is not None
    residual = abs(0.59943681 * BA137["ba137.S12.g_J"].value - stored.value)
    assert residual == pytest.approx(2.14e-8, rel=5e-2)
    assert residual < stored.uncertainty


def test_133ba_hyperfine_constants_scale_from_137ba_by_the_nuclear_g_factors() -> None:
    """The nuclear-g ratio A(133)/A(137) = -2.4697 reproduces the 133Ba+ ground-state A to 2e-5 and predicts A(6p 2P1/2)
    inside the stored value's bar."""
    ratio = (BA133["ba133.mu_I_nuclear_magnetons"].value / 0.5) / (
        BA137["ba137.mu_I_nuclear_magnetons"].value / 1.5
    )
    assert ratio == pytest.approx(-2.4697, abs=1e-4)
    assert BA137["ba137.S12.A_hfs_hz"].value * ratio == pytest.approx(
        BA133["ba133.S12.A_hfs_hz"].value, rel=2e-5
    )
    a133 = BA133["ba133.P12.A_hfs_hz"]
    assert a133.value < 0.0 and a133.uncertainty is not None
    assert BA137["ba137.P12.A_hfs_hz"].value * ratio == pytest.approx(a133.value, abs=a133.uncertainty)


# ---- the Species invariants -----------------------------------------------------------------------------------------


def _level(name: str, e: float, tau: float | None = None) -> Level:
    return Level(name, e, tau, 0.0, 0.0, 2.0, ("Steck",))


def test_species_refuses_inconsistent_wavelength() -> None:
    lvls = (_level("S1/2", 0.0), _level("P1/2", 7.5e14, 7e-9))
    bad = Transition("S1/2", "P1/2", 400e-9, 1.0 / (TWO_PI * 7e-9), 1.0, "E1", ("Steck",))
    with pytest.raises(ValueError, match="wavelength"):
        Species("X+", 40.0, 0.0, 0.0, lvls, (bad,), ("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"), "S1/2-P1/2", (), None)


def test_species_refuses_hyperfine_structure_for_spin_zero() -> None:
    lvls = (Level("S1/2", 0.0, None, 1.0e9, 0.0, 2.0, ("Steck",)),)
    with pytest.raises(ValueError, match="spin-zero"):
        Species("X+", 40.0, 0.0, 0.0, lvls, (), ("S1/2 mJ=-1/2", "S1/2 mJ=+1/2"), "S1/2-S1/2", (), None)
