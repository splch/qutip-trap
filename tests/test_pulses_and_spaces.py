"""Record invariants of drives, beams, schedules, combs, decoupling sequences and small trap formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.control.pulses import Drive, Pulse, Tone
from qutip_trap.control.schedule import Schedule
from qutip_trap.light.beams import Beam
from qutip_trap.light.comb import CombSpec
from qutip_trap.noise.decoupling import DecouplingSequence
from qutip_trap.trap.heating import heating_rate_quanta_per_s
from qutip_trap.trap.mathieu import beta_lowest_order, c0_series
from qutip_trap.trap.surface import five_wire_null_height_m
from qutip_trap.units import ATOMIC_MASS_KG
from tests.fixtures import make_raman_pair


def test_delta_k_geometry_rules() -> None:
    """|Delta k| = 2k sin(theta/2): sqrt 2 k at 90 degrees (the Monroe anchor), 2k counter-propagating, 0 for microwaves."""
    b1, b2 = make_raman_pair()
    k = 2.0 * math.pi / 355e-9
    tone = Tone(0.0, 0.0, 1e6)
    raman = Drive("raman", (0,), (tone,), (0, 1), 0.0, {})
    assert np.linalg.norm(raman.delta_k([b1, b2])) == pytest.approx(math.sqrt(2.0) * k)
    b3 = Beam(355e-9, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    assert np.linalg.norm(raman.delta_k([b1, b3])) == pytest.approx(2.0 * k)
    assert np.allclose(Drive("microwave", (0,), (tone,), (), 0.0, {}).delta_k([b1]), 0.0)
    assert np.linalg.norm(
        Drive("optical_E2", (0,), (tone,), (0,), 0.0, {}).delta_k(
            [Beam(729e-9, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 30e-6, 0.1, (0.0, 0.0, 0.0))]
        )
    ) == pytest.approx(2.0 * math.pi / 729e-9)
    with pytest.raises(ValueError):
        Drive("raman", (0,), (tone,), (0,), 0.0, {})  # a Raman drive references two beams


def test_beam_invariants_and_peak_intensity() -> None:
    b = Beam(355e-9, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    assert b.intensity_at(np.zeros(3)) == pytest.approx(2.0 * 10e-3 / (math.pi * (20e-6) ** 2))
    assert b.intensity_at(np.array([1.0, 0.0, 0.0])) == pytest.approx(b.intensity_at(np.zeros(3))), (
        "along the axis"
    )
    assert b.intensity_at(np.array([0.0, 20e-6, 0.0])) == pytest.approx(
        b.intensity_at(np.zeros(3)) * math.exp(-2.0)
    )
    with pytest.raises(ValueError, match="transverse"):
        Beam(355e-9, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 20e-6, 10e-3, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="unit-normalized"):
        Beam(
            355e-9,
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            20e-6,
            10e-3,
            (0.0, 0.0, 0.0),
            polarization_amplitudes=(1.0, 1.0, 0.0),
        )


def test_schedule_refuses_overlapping_pulses_on_one_ion() -> None:
    tone = Tone(0.0, 0.0, 1e6)
    d0 = Drive("microwave", (0,), (tone,), (), 0.0, {})
    d1 = Drive("microwave", (1,), (tone,), (), 0.0, {})
    p0 = Pulse(d0, 0.0, 10e-6, "g0", ())
    p1 = Pulse(d1, 5e-6, 15e-6, "g1", ())
    Schedule((p0, p1), (), (), {0: 0.0, 1: 0.0})  # parallel single-qubit pulses on distinct ions are fine
    p2 = Pulse(d0, 5e-6, 15e-6, "g2", ())
    with pytest.raises(ValueError, match="overlap"):
        Schedule((p0, p2), (), (), {0: 0.0})
    with pytest.raises(ValueError):
        Pulse(d0, 1.0, 0.5, None, ())


def test_comb_spec_chain_and_i_sat_travel_together() -> None:
    CombSpec(80.0e6, 10e-12, "field_sech", 158, 12.0e6)
    CombSpec(80.0e6, 10e-12, "field_sech", 158, 12.0e6, chain="II", i_sat_w_m2=1500.0)
    with pytest.raises(ValueError):
        CombSpec(80.0e6, 10e-12, "field_sech", 158, 12.0e6, chain="I", i_sat_w_m2=1500.0)
    with pytest.raises(ValueError):
        CombSpec(80.0e6, 10e-12, "field_sech", 158, 12.0e6, chain="II")
    comb = CombSpec(80.0e6, 10e-12, "field_sech", 158, 12.0e6)
    assert comb.pair_weight(0) == 1.0
    assert comb.pair_weight(105) == pytest.approx(1.0 / math.cosh(math.pi * 105 * 80e6 * 10e-12))
    assert comb.beat_note_hz(-1) == pytest.approx(68.0e6)


def test_decoupling_sequence_feasibility_and_moments() -> None:
    cpmg2 = DecouplingSequence("cpmg", 2, 1e-3, 1e-6, (0.25, 0.75), (0.0, 0.0))
    assert cpmg2.is_single_axis() and cpmg2.feasible()
    a1, a2 = cpmg2.moments()
    # A_k = sum_{j=1}^{n} (-1)^j delta_j^k with j counting from 1 as in Biercuk's pulse sum, so that A_1 = (-1)^n / 2
    # is the first-order cancellation condition of that sum (the plan's "A_1 = -/+ 1/2")
    assert a1 == pytest.approx(-0.25 + 0.75)
    assert a2 == pytest.approx(-(0.25**2) + 0.75**2)
    fat = DecouplingSequence("cpmg", 2, 1e-3, 0.6e-3, (0.25, 0.75), (0.0, 0.0))
    assert not fat.feasible()
    xy = DecouplingSequence("xy4", 2, 1e-3, 1e-6, (0.25, 0.75), (0.0, math.pi / 2))
    assert not xy.is_single_axis()


def test_small_trap_formulas() -> None:
    assert five_wire_null_height_m(1.0, 1.2) == pytest.approx(0.921954, abs=1e-6), "check_surface_mixed.py"
    assert c0_series(0.2) == pytest.approx(1.0075), "the O(q^2) term of 1.007741 (check_c0_floquet.py)"
    assert beta_lowest_order(0.0, 0.2) == pytest.approx(math.sqrt(0.02))
    # S_E = 2.2250e-13 (V/m)^2/Hz for 9Be+ at 3.6 MHz gives 40 quanta/s
    ndot = heating_rate_quanta_per_s(2.2250e-13, 9.012183065 * ATOMIC_MASS_KG, 2.0 * math.pi * 3.6e6)
    assert ndot == pytest.approx(40.0, rel=2e-3)
