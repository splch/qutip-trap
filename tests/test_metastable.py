"""Metastable-level channels (PLAN.md Section 4.5.7): the blackbody mixing rate, the collision-rate construction and the
reshelving offset."""

from __future__ import annotations

import math

import pytest

from qutip_trap.species import species
from qutip_trap.species.metastable import MetastableChannels, bbr_mixing_rates_hz, bose_occupation
from qutip_trap.units import C_M_PER_S

A12_S = 2.45e-6  # Ali and Kim 1988 via Kreuter 2005 Eq. 1
NU_QUOTED_HZ = 1.82e12  # Kreuter's rounded frequency of the 3d 2D5/2 - 2D3/2 line


@pytest.fixture(scope="module")
def ca40():  # type: ignore[no-untyped-def]
    return species("40Ca+")


def test_blackbody_mixing_rate() -> None:
    """n_bar = 2.95884 (h nu/kT = 0.29115) multiplies A12: W12 = 7.2491e-6 s^-1 at 300.0 K and 7.2297e-6 at 299.3 K (what the
    printed 7.23e-6 back-solves to), upward 1.0874e-5 with g_u/g_l = 6/4; dividing by n_bar would be low by 8.75x."""
    assert bose_occupation(NU_QUOTED_HZ, 300.0) == pytest.approx(2.95884, abs=5e-6)
    down, up = bbr_mixing_rates_hz(A12_S, NU_QUOTED_HZ, 300.0, 6.0, 4.0)
    assert down == pytest.approx(7.2491e-6, rel=1e-4)
    assert up == pytest.approx(1.0874e-5, rel=1e-4)
    assert up / down == pytest.approx(1.5, rel=1e-12)
    down_299, _ = bbr_mixing_rates_hz(A12_S, NU_QUOTED_HZ, 299.3, 6.0, 4.0)
    assert down_299 == pytest.approx(7.2297e-6, rel=1e-4)


def test_bbr_rate_from_the_species_table(ca40) -> None:  # type: ignore[no-untyped-def]
    """The 40Ca+ table carries the D3/2-D5/2 M1 line (A12 tau = 2.9e-6 of the D5/2 decay) at the level-energy frequency
    1.8194 THz; W12 at 300 K is 8.47e-6 of the natural rate, so the default (channel off) returns zero rates."""
    ch = MetastableChannels(bbr_temperature_k=300.0)
    tr = ca40.transition("D3/2-D5/2")
    assert tr.multipole == "M1"
    assert tr.partial_rate_rad_s == pytest.approx(A12_S, rel=1e-12)
    assert C_M_PER_S / tr.wavelength_vac_m == pytest.approx(1.8194e12, rel=1e-4)
    down, up = ch.bbr_rate_hz("D3/2-D5/2", ca40)
    assert down == pytest.approx(7.249e-6, rel=5e-4), "the plan's row uses the rounded 1.82 THz"
    assert up == pytest.approx(1.5 * down, rel=1e-12)
    assert down * ca40.level("D5/2").lifetime_s == pytest.approx(8.47e-6, rel=1e-3)
    assert MetastableChannels().bbr_rate_hz("D3/2-D5/2", ca40) == (0.0, 0.0)
    # the branchings out of D5/2 still sum to one: the E2 branch carries 1 - A12 tau
    e2 = ca40.transition("S1/2-D5/2")
    assert e2.branching + tr.branching == pytest.approx(1.0, abs=1e-15)
    assert e2.branching == pytest.approx(1.0, abs=1e-5), (
        "unit branching to 1e-5 for the Ca+ D5/2 level (4.5.7)"
    )


def test_collision_rate_construction(ca40) -> None:  # type: ignore[no-untyped-def]
    """n = p/(k_B T) = 2.4143e5 cm^-3 per partner at 300 K and 1e-11 mbar; R^q = 5.00e-5 s^-1, R^j = 3.86e-4 s^-1, total
    4.36e-4 s^-1 (1.45x the source's rounded '< 3e-4'); j-mixing coefficients 7.73x the quenching ones."""
    ch = MetastableChannels(
        pressure_mbar=2e-11, gas_fractions={"H2": 0.5, "N2": 0.5}, gas_temperature_k=300.0
    )
    n = ch.partner_densities_m3()
    assert n["H2"] * 1e-6 == pytest.approx(2.4143e5, rel=1e-4)
    assert n["N2"] == n["H2"]
    rates = ch.collision_rates_hz(ca40)
    assert rates["quench"] == pytest.approx(5.00e-5, rel=2e-3)
    assert rates["j_mix"] == pytest.approx(3.86e-4, rel=2e-3)
    assert rates["quench"] + rates["j_mix"] == pytest.approx(4.36e-4, rel=2e-3)
    assert (rates["quench"] + rates["j_mix"]) / 3e-4 == pytest.approx(1.45, abs=0.01)
    coefficients = ch.collision_coefficients_cm3_s(ca40)
    assert sum(coefficients["j_mix"].values()) / sum(coefficients["quench"].values()) == pytest.approx(
        7.73, abs=0.01
    )
    # below the 1e-3 neglect threshold of the natural rate
    assert (rates["quench"] + rates["j_mix"]) * ca40.level("D5/2").lifetime_s < 1e-3
    # density driven: doubling the pressure doubles every rate, halving the temperature doubles it too
    twice = MetastableChannels(pressure_mbar=4e-11, gas_fractions={"H2": 0.5, "N2": 0.5}).collision_rates_hz(
        ca40
    )
    cold = MetastableChannels(
        pressure_mbar=2e-11, gas_fractions={"H2": 0.5, "N2": 0.5}, gas_temperature_k=150.0
    ).collision_rates_hz(ca40)
    for k in rates:
        assert twice[k] == pytest.approx(2.0 * rates[k], rel=1e-12)
        assert cold[k] == pytest.approx(2.0 * rates[k], rel=1e-12)
    assert MetastableChannels().collision_rates_hz(ca40) == {"quench": 0.0, "j_mix": 0.0}


def test_uncited_partner_or_species_raises_rather_than_defaulting(ca40) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(LookupError, match="He"):
        MetastableChannels(pressure_mbar=1e-11, gas_fractions={"He": 1.0}).collision_rates_hz(ca40)
    with pytest.raises(LookupError, match="171Yb"):
        MetastableChannels(pressure_mbar=1e-11, gas_fractions={"H2": 1.0}).collision_rates_hz(
            species("171Yb+")
        )
    with pytest.raises(ValueError, match="sum to 1"):
        MetastableChannels(pressure_mbar=1e-11, gas_fractions={"H2": 0.7})
    with pytest.raises(ValueError):
        MetastableChannels(bbr_temperature_k=0.0)


def test_reshelving_offset() -> None:
    """R/(Gamma + R) = 3.4918e-3 at tau = 1168 ms and R = 3e-3 s^-1, zero by default."""
    assert MetastableChannels(reshelving_rate_hz=3e-3).reshelving_offset(1.168) == pytest.approx(
        3.4918e-3, rel=1e-4
    )
    assert MetastableChannels().reshelving_offset(1.168) == 0.0


def test_shelf_loss_rates_and_effective_lifetime(ca40) -> None:  # type: ignore[no-untyped-def]
    """With every channel on, the D5/2 shelf empties at 1/tau + R_q + R_j + W12 (the M1 line's downward rate); the
    effective lifetime is shorter than tau by the 5e-4 the channels add."""
    ch = MetastableChannels(
        bbr_temperature_k=300.0, pressure_mbar=2e-11, gas_fractions={"H2": 0.5, "N2": 0.5}
    )
    rates = ch.shelf_loss_rates_hz(ca40, "D5/2")
    assert set(rates) == {"decay", "quench", "j_mix", "bbr:D3/2-D5/2"}
    assert rates["decay"] == pytest.approx(1.0 / 1.168, rel=1e-12)
    assert rates["bbr:D3/2-D5/2"] == pytest.approx(ch.bbr_rate_hz("D3/2-D5/2", ca40)[0], rel=1e-12)
    tau_eff = ch.effective_shelf_lifetime_s(ca40, "D5/2")
    assert tau_eff < 1.168
    assert 1.168 / tau_eff - 1.0 == pytest.approx(
        1.168 * sum(v for k, v in rates.items() if k != "decay"), rel=1e-9
    )
    # the lower D level sees the UPWARD blackbody rate
    lower = ch.shelf_loss_rates_hz(ca40, "D3/2")
    assert lower["bbr:D3/2-D5/2"] == pytest.approx(ch.bbr_rate_hz("D3/2-D5/2", ca40)[1], rel=1e-12)
    assert math.isclose(lower["decay"], 1.0 / 1.176, rel_tol=1e-12)
    assert MetastableChannels().effective_shelf_lifetime_s(ca40, "D5/2") == pytest.approx(1.168, rel=1e-12)
