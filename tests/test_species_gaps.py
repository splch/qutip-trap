"""The species-table gaps closed on 2026-09-08, and the ones the literature leaves open.

Every constant this file pins was entered by the species-gaps pass and each assertion is the test that
would have caught the defect it fixes:

- the three MEASURED Ba+ Lande g factors that replace a missing entry and two NIST ASD Lande-column
  readings (one of which, 1.12 for 5d 2D5/2, is ASD's own 6.7 % error);
- 133Ba+'s A(6p 2P1/2) = -1840(11) MHz, which was recorded as unobtainable while sitting in Table I of a
  paper this repository already cited;
- the 9Be+ ground-state g_J provenance, whose citation was challenged and survives, with the modern
  CALCULATED value stored beside it and the clock-point anchor's sensitivity to the two pinned so that
  swapping them can never be an unnoticed change;
- 43Ca+'s g_J, measured on 40Ca+ and applied here as isotope-independent;
- 25Mg+'s g_J gap, closed by search rather than by a number.

Tolerances follow the plan's rule: printed digits for stored constants, 1e-9 relative for the closed forms
and identities, and published laboratory numbers reported rather than re-derived.
"""

from __future__ import annotations

import re

import pytest

from qutip_trap.provenance import TAGS, load_ledger
from qutip_trap.species import MODULES, IncompleteSpeciesTable, species
from qutip_trap.species.model import Level
from qutip_trap.species.sources import SOURCES
from qutip_trap.species.zeeman import HyperfineZeeman, clock_points

BA133 = MODULES["133Ba+"].TABLE
BA137 = MODULES["137Ba+"].TABLE
BE9 = MODULES["9Be+"].TABLE
CA43 = MODULES["43Ca+"].TABLE
MG25 = MODULES["25Mg+"].TABLE
LEDGER = load_ledger()

LOCATOR = re.compile(r"\((?:19|20)\d\d\)|arXiv:\d{4}\.\d{4,5}|doi:10\.")
"""The same locator pattern ``test_species_tables.py`` enforces over every source key."""


# ---- the constants entered, with value, uncertainty, tag and a source that carries a locator -------------

NEW_CONSTANTS: tuple[tuple[str, float, float, str, str], ...] = (
    # ledger id, value, uncertainty, tag, source key
    ("ba137.S12.g_J", 2.00249192, 3e-8, "extracted", "Marx1998"),
    ("ba133.S12.g_J", 2.00249192, 3e-8, "extracted", "Marx1998"),
    ("ba137.D32.g_J", 0.7993278, 3e-7, "extracted", "Knoll1996"),
    ("ba133.D32.g_J", 0.7993278, 3e-7, "extracted", "Knoll1996"),
    ("ba137.D52.g_J", 1.20036739, 2.4e-8, "verified", "ArnoldPRL2020"),
    ("ba133.D52.g_J", 1.20036739, 2.4e-8, "verified", "ArnoldPRL2020"),
    ("ba133.P12.A_hfs_hz", -1840e6, 11e6, "verified", "Hucul2017"),
    ("be9.S12.g_J_theory", 2.0022621287, 2.4e-10, "background", "Dickopf2024"),
)


@pytest.mark.parametrize(("ledger_id", "value", "uncertainty", "tag", "source"), NEW_CONSTANTS)
def test_the_constants_entered_carry_value_uncertainty_tag_and_a_located_source(
    ledger_id: str, value: float, uncertainty: float, tag: str, source: str
) -> None:
    species_key = ledger_id.split(".", 1)[0]
    table = {"ba133": BA133, "ba137": BA137, "be9": BE9}[species_key]
    c = table[ledger_id]
    assert c.value == value
    assert c.uncertainty == pytest.approx(uncertainty), f"{ledger_id} must carry its published uncertainty"
    assert c.tag == tag and c.tag in TAGS
    assert c.source == source
    assert source in SOURCES, f"{ledger_id} cites an unknown source key"
    assert LOCATOR.search(SOURCES[source]), f"{source} names no locator"
    assert c.note, f"{ledger_id} must say where its number came from"
    # and the ledger agrees, which is what tools/ledger_from_tables.py guarantees
    assert ledger_id in LEDGER and LEDGER[ledger_id].tag == tag and LEDGER[ledger_id].source == source


def test_the_ba_g_factors_are_the_same_number_in_both_isotope_tables() -> None:
    """g_J is isotope-independent to the quoted precision, so 133Ba+ and 137Ba+ must not drift apart.

    A number transcribed twice is a number that can disagree with itself; this is the test for that.
    """
    for key in ("S12.g_J", "D32.g_J", "D52.g_J"):
        a, b = BA133[f"ba133.{key}"], BA137[f"ba137.{key}"]
        assert a.value == b.value and a.uncertainty == b.uncertainty and a.source == b.source, key
        assert "isotope" in a.note.lower() or "isotope" in b.note.lower(), (
            f"{key} is transferred between isotopes, so both notes must say so"
        )


def test_the_nist_asd_lande_readings_are_kept_as_named_cross_checks_not_inputs() -> None:
    """The ASD Lande column returns 0.79 / 1.12 / 1.32 for Ba II and BLANK for 6s 2S1/2.

    The repository transcribed that faithfully; the 1.12 is ASD's OWN error, 6.7 % from four independent
    determinations of g_J(5d 2D5/2), and its 0.79 is 1.2 % low against the measured D3/2 value -- not a
    two-digit rounding of it, which would read 0.80. Both readings stay in the tables under ``*_g_J_asd``
    names, tagged ``contested``, so the error is documented rather than deleted -- and so that nothing
    reads them as an input by accident.
    """
    for table, prefix in ((BA133, "ba133"), (BA137, "ba137")):
        asd = table[f"{prefix}.D52.g_J_asd"]
        assert asd.value == 1.12 and asd.tag == "contested" and asd.source == "NIST_ASD_5_12"
        assert "CROSS-CHECK ONLY" in asd.note
        assert table[f"{prefix}.D32.g_J_asd"].value == 0.79
        assert table[f"{prefix}.D32.g_J_asd"].tag == "contested"
        # the measured value the level actually uses is 6.7 % away from ASD's D5/2 entry
        measured = table[f"{prefix}.D52.g_J"].value
        assert abs(asd.value - measured) / measured == pytest.approx(0.0669, abs=5e-4)
        # ASD's D3/2 entry is NOT a rounding of the measured value either (that would be 0.80): it is
        # 1.2 % low, so the whole column reads as an independent and inaccurate calculation
        d32 = table[f"{prefix}.D32.g_J"].value
        assert round(d32, 2) == 0.80
        assert abs(0.79 - d32) / d32 == pytest.approx(0.0117, abs=5e-4)


def test_the_137ba_levels_use_the_measured_g_factors_and_not_the_asd_column() -> None:
    """The build must consume the measured values; storing them and using ASD's would be invisible."""
    ba = species("137Ba+")
    assert ba.level("S1/2").g_J == 2.00249192
    assert ba.level("D3/2").g_J == 0.7993278
    assert ba.level("D5/2").g_J == 1.20036739
    # the P levels have no measured g_J, so they carry the Lande values (with Steck's g_S, as ca43.py and
    # be9.py do: 0.665894 and 1.334106, not the exact-g_S 2/3 and 4/3) tagged [background]
    assert ba.level("P1/2").g_J == pytest.approx(0.665894, abs=1e-6)
    assert ba.level("P3/2").g_J == pytest.approx(1.334106, abs=1e-6)
    assert BA137["ba137.P32.g_J"].tag == "contested", "ASD's 1.32 is a cross-check, not the input"
    assert "PLAN_background" in ba.level("P3/2").citations


def test_arnolds_d52_g_factor_is_the_ratio_times_marxs_ground_state_g_factor() -> None:
    """g_D = r g_S with r = 0.59943681(12) and g_S = 2.00249192(3): the measurement's own arithmetic.

    Both numbers were read in arXiv:1906.09150v2. Reproducing the product is what shows the two stored
    constants belong to the same measurement, and it is why the uncertainty is (24) and not the (14) that
    Chin. Phys. Lett. 42, 063101 (2025) misquotes.
    """
    stored = BA137["ba137.D52.g_J"]
    assert stored.uncertainty == pytest.approx(2.4e-8)
    product = 0.59943681 * BA137["ba137.S12.g_J"].value
    # the residual is 2.14e-8: the printed ratio is given to 8 decimals, whose own last-digit rounding
    # propagates to about 1e-8 in the product, so the reconstruction lands inside the published (24) bar
    assert stored.uncertainty is not None
    assert abs(product - stored.value) == pytest.approx(2.14e-8, rel=5e-2)
    assert abs(product - stored.value) < stored.uncertainty
    assert "(14)" in SOURCES["ArnoldPRL2020"], "the misquoted uncertainty must stay named in the source"


def test_133ba_p12_hyperfine_a_matches_the_isotope_scaled_137ba_value() -> None:
    """A(6p 2P1/2) = -1840(11) MHz for 133Ba+ against 743.7(3) MHz for 137Ba+.

    Scaling by (mu_I/I)(133)/(mu_I/I)(137) predicts -1836.7 MHz, inside the 11 MHz bar. This is
    corroboration of the sign and magnitude, not the citation -- the value is read directly in Hucul et al.
    2017 Table I -- but it is the arithmetic that makes the NEGATIVE sign credible.
    """
    r133 = BA133["ba133.mu_I_nuclear_magnetons"].value / 0.5
    r137 = BA137["ba137.mu_I_nuclear_magnetons"].value / 1.5
    ratio = r133 / r137
    assert ratio == pytest.approx(-2.4697, abs=1e-4)
    predicted = BA137["ba137.P12.A_hfs_hz"].value * ratio
    a133 = BA133["ba133.P12.A_hfs_hz"].value
    assert a133 < 0.0, "mu_I(133Ba) < 0 makes every 133Ba+ A negative"
    assert predicted == pytest.approx(a133, abs=BA133["ba133.P12.A_hfs_hz"].uncertainty)
    # the same scaling reproduces the ground-state A to 1e-5, which validates the scaling itself
    assert BA137["ba137.S12.A_hfs_hz"].value * ratio == pytest.approx(
        BA133["ba133.S12.A_hfs_hz"].value, rel=2e-5
    )


def test_133ba_still_refuses_with_exactly_the_two_constants_nobody_prints() -> None:
    """A(6p 2P3/2) and A(5d 2D5/2): Christensen et al. 2020 publish the SPLITTINGS 623(30) and 83(30) MHz
    and print neither constant nor its sign, so the gaps are documented as closed-by-search.

    The two derived uncertainties were understated until 2026-09-08 -- (100) and (67) against splittings
    that are (30), where the arithmetic gives (150) and (100) -- which is what these assertions pin.
    """
    with pytest.raises(IncompleteSpeciesTable) as info:
        species("133Ba+")
    ids = {m.quantity.rsplit("(", 1)[-1].rstrip(")") for m in info.value.missing}
    assert ids == {"ba133.P32.A_hfs_hz", "ba133.D52.A_hfs_hz"}
    notes = {i: MODULES["133Ba+"]._CONSULT[i] for i in ids}
    for note in notes.values():
        assert "NO SOURCE PRINTS THIS CONSTANT" in note
        assert "only the splitting" in note
    # A = -(splitting)/2 for I = 1/2, J = 3/2 and +(splitting)/3 for J = 5/2 (Section 4.5.1)
    splitting = BA133["ba133.P32.hfs_splitting_hz"]
    assert splitting.value == 623e6 and splitting.uncertainty == 30e6, "Christensen's (30), not (20)"
    assert -splitting.value / 2.0 == pytest.approx(-311.5e6)
    assert splitting.uncertainty / 2.0 == pytest.approx(15.0e6), "-311.5(150) MHz, not (100)"
    assert 83e6 / 3.0 == pytest.approx(27.667e6, abs=1e3)
    assert 30e6 / 3.0 == pytest.approx(10.0e6), "+27.7(100) MHz, not (67)"
    assert "-311.5(150)" in notes["ba133.P32.A_hfs_hz"]
    assert "+27.7(100)" in notes["ba133.D52.A_hfs_hz"]


# ---- 9Be+: the provenance question, and the anchor's sensitivity to the two values -----------------------

BE9_CLOCK_B0_GAUSS = 119.44615496
BE9_CLOCK_FREQUENCY_HZ = 1_207_495_853.379
"""The Section 9.13 clock point as the M0a pass recomputed it from the MEASURED g_J = 2.00226239(31).

PLAN.md 9.13 gives 119.446 G and 1,207,495,853.5 Hz against Langer 2005's published 1,207,495,843 Hz, with
a stated 20 Hz tolerance because "the 10 Hz residual against the published value is the precision of the
g-factors used". These digits are this package's own recomputation, pinned to the printed digits.
"""


def _be9_clock(g_j: float) -> tuple[float, float]:
    lv = Level("S1/2", 0.0, None, BE9["be9.S12.A_hfs_hz"].value, 0.0, g_j, ("Steck",))
    hz = HyperfineZeeman(lv, 1.5, BE9["be9.mu_I_nuclear_magnetons"].value)
    (cp,) = clock_points(hz, "F=2 mF=0", hz, "F=1 mF=1", 80.0, 160.0)
    return cp.B0_gauss, cp.frequency_hz


def test_be9_g_j_provenance_is_shiga_relaying_wineland_and_the_citation_survives() -> None:
    """The stored 2.00226239(31) is NOT mis-cited.

    A literature review of 2026-09-08 read only Shiga et al. 2011's abstract -- which carries A_0, k and
    the RATIO g_I'/g_J -- and concluded the absolute g_J was not in the paper. The paper's Sec. I body
    prints it, and says the measurement is Wineland, Bollinger and Itano 1983's cyclotron-versus-
    hyperfine-Zeeman comparison, re-reduced with the CODATA-2006 proton-electron mass ratio. So the value
    stays, the citation stays, and both papers are named in the source entry. Ledger
    conv.be9_g_j_provenance.
    """
    c = BE9["be9.S12.g_J"]
    assert c.value == 2.00226239 and c.uncertainty == pytest.approx(3.1e-7)
    assert c.source == "Shiga2011"
    text = SOURCES["Shiga2011"]
    assert "2.002 262 39(31)" in text, "the source entry must quote the sentence that prints the value"
    assert "Phys. Rev. Lett. 50, 628 (1983)" in text, "and must name the underlying MEASUREMENT"
    assert "Rev. Mod. Phys. 80, 633 (2008)" in text, "and the mass ratio the digits were re-reduced with"
    assert "WinelandBollingerItano1983" in SOURCES, "the 1983 primary has its own key"
    assert c.tag != "background", "a measured-and-re-reduced value is not a calculation"


def test_be9_clock_anchor_at_the_measured_g_j() -> None:
    """The anchor the table's own value produces, to the digits the M0a pass recorded."""
    b0, f0 = _be9_clock(BE9["be9.S12.g_J"].value)
    assert b0 == pytest.approx(BE9_CLOCK_B0_GAUSS, abs=5e-8)
    assert f0 == pytest.approx(BE9_CLOCK_FREQUENCY_HZ, abs=1e-3)
    # and it still satisfies the plan's own tolerance against Langer's published value
    assert abs(f0 - 1_207_495_843.0) < 20.0


def test_be9_clock_anchor_cannot_distinguish_the_measured_from_the_calculated_g_j() -> None:
    """Dickopf et al. 2024's CALCULATED 2.0022621287(24) moves the anchor by 1.56e-5 G and 0.0093 Hz.

    That is 2000x inside the plan's 20 Hz tolerance on this anchor, so the anchor is no reason to prefer
    the calculation, and the table keeps the measurement. Pinning the difference here is what makes a
    future swap a visible change rather than a silent one. The plan's own negative control, g_J = 2.000,
    moves the same point by 0.135 G and 81 Hz -- four orders of magnitude more -- which is the scale that
    shows the two live values are interchangeable for every purpose this package has.
    """
    theory = BE9["be9.S12.g_J_theory"]
    assert theory.tag == "background" and theory.source == "Dickopf2024"
    assert "calculations performed in ref. [22]" in SOURCES["Dickopf2024"], "it must be named a CALCULATION"
    measured = BE9["be9.S12.g_J"]
    # the two agree: 2.61e-7 apart, 0.84 sigma of the measurement's own bar
    delta_g = measured.value - theory.value
    assert delta_g == pytest.approx(2.613e-7, abs=1e-10)
    assert measured.uncertainty is not None
    assert delta_g / measured.uncertainty == pytest.approx(0.843, abs=5e-3)

    b_m, f_m = _be9_clock(measured.value)
    b_t, f_t = _be9_clock(theory.value)
    assert b_t - b_m == pytest.approx(1.558e-5, rel=2e-3)
    assert f_t - f_m == pytest.approx(0.00933, rel=2e-3)
    assert abs(f_t - f_m) < 20.0 / 100.0, "far inside the plan's 20 Hz tolerance on this anchor"

    b_l, f_l = _be9_clock(2.000)
    assert b_l - b_m == pytest.approx(0.13504, abs=1e-4)
    assert f_l - f_m == pytest.approx(80.86, abs=0.05)
    assert abs(f_l - f_m) > 20.0, "the plan's negative control DOES break the anchor, unlike the theory g_J"


# ---- 43Ca+ and 25Mg+ ------------------------------------------------------------------------------------


def test_ca43_g_j_is_a_40ca_measurement_applied_as_isotope_independent() -> None:
    """Tommaseo et al. 2003 measured 40Ca+, not 43Ca+, so the tag is ``extracted`` and names both papers.

    It was ``verified`` until 2026-09-08, which claimed a 43Ca+ reading of a paper that was never read at
    all: the digits reach this table through Hanley et al., arXiv:2105.10352.
    """
    c = CA43["ca43.S12.g_J"]
    assert c.value == 2.00225664 and c.uncertainty == pytest.approx(9e-8)
    assert c.tag == "extracted", "a 40Ca+ measurement read in a quoting paper is not verified for 43Ca+"
    assert c.source == "Tommaseo2003"
    assert "40Ca+" in c.note and "isotope-independent" in c.note
    assert "MEASURED ON 40Ca+, NOT ON 43Ca+" in SOURCES["Tommaseo2003"]
    assert "arXiv:2105.10352" in SOURCES["Tommaseo2003"], "the paper the digits were read in"
    # the same isotope-transfer assumption the Ba+ tables make, and it is stated in all three
    for note in (c.note, BA133["ba133.S12.g_J"].note, BA137["ba137.S12.g_J"].note):
        assert "isotope" in note.lower()


def test_mg25_g_j_gap_is_documented_as_closed_by_search_and_the_table_still_refuses() -> None:
    """No measured AND no computed 25Mg+ g_J exists; the gap note now carries the evidence.

    The strongest single piece is that Kaur and Sahoo's own comprehensive Mg+ properties calculation --
    the paper this table already cites for the 3p hyperfine constants -- computes no g_J at all.
    """
    with pytest.raises(IncompleteSpeciesTable) as info:
        species("25Mg+")
    consult = MODULES["25Mg+"]._CONSULT["mg25.S12.g_J"]
    assert "NO SOURCE EXISTS, MEASURED OR COMPUTED" in consult
    assert "CLOSED BY SEARCH" in consult
    assert "arXiv:2504.19515" in consult, "the negative result on the modern Mg+ RCC calculation"
    assert "NO Lande column" in consult, "the NIST ASD negative"
    assert "arXiv:2504.13071" in consult, "why the gap persists: the Al+/Mg+ clock needs only g_I/g_J"
    assert "ZERO occurrences" in SOURCES["KaurSahoo2025"]
    assert "mg25.S12.g_J" not in MG25, "the table must not carry a g_J it cannot cite"
    assert MG25["mg25.S12.g_I_over_g_J"].tag == "verified", "the ratio IS measured"
    assert any("g_J" in m.quantity for m in info.value.missing)
    # and 9Be+ is explicitly NOT in the same position, which is the confusion this note has to prevent
    assert "9Be+" in consult and "be9.S12.g_J" in consult
