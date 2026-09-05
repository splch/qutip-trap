"""Species data tables (PLAN.md Section 4.5.6, 9.13; M0): every number cited, nothing hyperfine-resolved typed in."""

from __future__ import annotations

import ast
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

from qutip_trap.provenance import TAGS, load_ledger
from qutip_trap.species import MODULES, IncompleteSpeciesTable, available, species
from qutip_trap.species.model import Level, Species, Transition
from qutip_trap.species.sources import SOURCES
from qutip_trap.species.table import a_hfs_from_two_manifold_splitting
from qutip_trap.units import C_M_PER_S, H_J_S, TWO_PI

LEDGER = load_ledger()
HYPERFINE_RESOLVED = re.compile(r"(mF|F=\d|zeeman_energy|dipole_element|rabi|raman_coupling|clebsch)", re.I)


def test_the_first_two_species_and_sr_build() -> None:
    assert available() == ("171Yb+", "40Ca+", "88Sr+")


@pytest.mark.parametrize("name", sorted(MODULES))
def test_every_constant_is_cited_and_in_the_ledger(name: str) -> None:
    table = MODULES[name].TABLE
    assert table, f"{name} has an empty table"
    for ledger_id, c in table.items():
        assert ledger_id == c.ledger_id
        assert c.source in SOURCES, f"{ledger_id}: unknown source key {c.source!r}"
        assert c.tag in TAGS
        assert ledger_id in LEDGER, f"{ledger_id} missing from docs/provenance/ledger.yaml"
        assert LEDGER[ledger_id].source == c.source
        assert LEDGER[ledger_id].tag == c.tag


@pytest.mark.parametrize("name", sorted(MODULES))
def test_nothing_hyperfine_resolved_is_typed_in(name: str) -> None:
    for ledger_id in MODULES[name].TABLE:
        assert not HYPERFINE_RESOLVED.search(ledger_id.split(".", 1)[1]), ledger_id


@pytest.mark.parametrize("name", sorted(set(MODULES) - set(available())))
def test_incomplete_tables_say_what_is_missing(name: str) -> None:
    with pytest.raises(IncompleteSpeciesTable) as info:
        species(name)
    assert info.value.missing, name
    assert all(m.consult for m in info.value.missing)


def test_retired_constants_are_absent_from_the_code() -> None:
    """Section 10 M0a: a CI check greps the tree for the retired readings (310.85 Hz/G^2, 50.77 mW/cm^2 and the
    uncited 171Yb+ g_J values 2.00254 and 2.00292)."""
    retired = {310.85, 50.77, 2.00254, 2.00292}
    root = Path(__file__).resolve().parents[1] / "qutip_trap"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value in retired:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {node.value}")
    assert not offenders, offenders


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
    from qutip_trap.provenance import Cited

    c = Cited(value=1.0e9, unit="Hz", source="Steck", ledger_id="test.x")
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


def test_ca40_i_sat_anchor_reproduces_under_the_plans_stated_reading() -> None:
    """Section 9.13: 40Ca+ 397 nm with the quoted 21.57 MHz read as a PARTIAL rate gives 45.11 mW/cm^2.

    That row was computed (check_atomic.py) at lambda = 396.85 nm, which is the AIR wavelength of the line;
    with the NIST vacuum value 396.959 nm the same reading gives 45.07 mW/cm^2, 0.1% lower. The table stores
    the vacuum wavelength (Section 13) and reads the quoted linewidth as the TOTAL rate, so the anchor is
    reproduced here only with the plan's own inputs; the discrepancy is recorded, not hidden.
    """
    ca = species("40Ca+")
    tr = ca.transition("S1/2-P1/2")
    gamma_partial_reading = TWO_PI * 21.57e6
    plan_inputs = math.pi * H_J_S * C_M_PER_S * gamma_partial_reading / (3.0 * (396.85e-9) ** 3)
    assert plan_inputs * 0.1 == pytest.approx(45.11, abs=0.01)
    vacuum = math.pi * H_J_S * C_M_PER_S * gamma_partial_reading / (3.0 * tr.wavelength_vac_m**3)
    assert vacuum * 0.1 == pytest.approx(45.07, abs=0.01)
    assert tr.wavelength_vac_m == pytest.approx(396.959e-9, abs=0.001e-9)
    assert MODULES["40Ca+"].TABLE["ca40.P12.linewidth_quoted_hz"].tag == "contested"


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
