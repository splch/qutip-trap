"""The E2 collapse operators and the E2 ac Stark shift of PLAN.md Section 4.5.7 ("Dissipation, dephasing and
crosstalk"), milestone M0a, audit item E9 / defect B8.

Before 2026-09-07 ``lambda_3j`` was referenced only by the E2 DRIVE and by tests: no decay operator used it,
no lumped ``sqrt(A) sigma_-`` existed either (shelf decay lived only in the M5 readout Markov chain as
``shelf_lifetime_s``), and ``light/raman.py`` hardcoded ``stark_shift_hz = 0.0`` for ``optical_E2``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from qutip_trap.species import species
from qutip_trap.species.quadrupole import (
    decay_weights,
    e2_stark_shift_rad_s,
    lambda_3j,
    quadrupole_collapse_operators,
)
from qutip_trap.units import TWO_PI

HALF = Fraction(1, 2)
FIVE_HALVES = Fraction(5, 2)
THREE_HALVES = Fraction(3, 2)


# ---- decay_weights ---------------------------------------------------------------------------------------


def test_the_raw_three_j_squares_sum_to_one_sixth_at_every_upper_sublevel() -> None:
    """Section 4.5.7: "the per-component weights w from the same 3-j squares (sum over m and q at fixed m'
    equals 1/6)". This is the normalization :func:`decay_weights` divides by."""
    for mp in (Fraction(k, 2) for k in (-5, -3, -1, 1, 3, 5)):
        total = sum(lambda_3j(HALF, m, FIVE_HALVES, mp) ** 2 for m in (-HALF, HALF))
        assert total == pytest.approx(1.0 / 6.0, abs=1e-15)


def test_decay_weights_sum_to_one_over_the_lower_sublevels_at_fixed_upper() -> None:
    """sum_m A w(m, m') = A, the level's total rate: the weights are a branching, not a line strength."""
    w = decay_weights(HALF, FIVE_HALVES)
    for mp in (Fraction(k, 2) for k in (-5, -3, -1, 1, 3, 5)):
        assert sum(v for (_m, m2), v in w.items() if m2 == mp) == pytest.approx(1.0, abs=1e-15)


def test_the_stretched_upper_sublevels_decay_to_one_lower_sublevel_only() -> None:
    """|Delta m| <= 2 forbids |D, +5/2> -> |S, -1/2>, so m' = +-5/2 has a single, unit-weight channel."""
    w = decay_weights(HALF, FIVE_HALVES)
    assert w[(HALF, FIVE_HALVES)] == pytest.approx(1.0, abs=1e-15)
    assert w[(-HALF, -FIVE_HALVES)] == pytest.approx(1.0, abs=1e-15)
    assert (-HALF, FIVE_HALVES) not in w
    assert len(w) == 10, "10 of the 12 (m, m') pairs are allowed (Section 9.14 '10 of 12')"


def test_the_weight_ladder_is_the_exact_rational_sequence() -> None:
    """For J = 1/2 -> J' = 5/2 the weights are the linear ladder 1, 4/5, 3/5, 2/5, 1/5, 0 in m' -- the same
    3-j squares as the 0.408248 ... 0 table of Section 9.14, normalized."""
    w = decay_weights(HALF, FIVE_HALVES)
    ladder = [w.get((-HALF, Fraction(k, 2)), 0.0) for k in (-5, -3, -1, 1, 3, 5)]
    assert ladder == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], abs=1e-15)


def test_decay_weights_of_the_d32_line_are_the_shorter_ladder() -> None:
    w = decay_weights(HALF, THREE_HALVES)
    assert len(w) == 8, "every (m, m') pair is allowed for J = 1/2 -> 3/2: |Delta m| <= 2 always"
    for mp in (Fraction(k, 2) for k in (-3, -1, 1, 3)):
        assert sum(v for (_m, m2), v in w.items() if m2 == mp) == pytest.approx(1.0, abs=1e-15)


# ---- quadrupole_collapse_operators -----------------------------------------------------------------------


def _basis_indices() -> tuple[dict[Fraction, int], dict[Fraction, int], int]:
    """|S, -1/2>, |S, +1/2> then the six |D, m'> in ascending m'."""
    lower = {-HALF: 0, HALF: 1}
    upper = {Fraction(k, 2): 2 + i for i, k in enumerate((-5, -3, -1, 1, 3, 5))}
    return lower, upper, 8


def test_the_collapse_operators_are_sqrt_a_w_times_the_outer_product() -> None:
    """Section 4.5.7: "the collapse operator is sqrt(A w(m, m')) |S, m><D, m'| with the same A that fixed the
    Rabi frequency"."""
    sr = species("88Sr+")
    tr = sr.transition("S1/2-D5/2")
    a = tr.partial_rate_rad_s
    lower, upper, dim = _basis_indices()
    ops = quadrupole_collapse_operators(a, HALF, FIVE_HALVES, lower.__getitem__, upper.__getitem__, dim)
    assert len(ops) == 10
    w = decay_weights(HALF, FIVE_HALVES)
    for (m, mp), rate, op in ops:
        assert rate == pytest.approx(a * w[(m, mp)], rel=1e-15)
        assert op.shape == (dim, dim)
        assert op[lower[m], upper[mp]] == pytest.approx(math.sqrt(rate), rel=1e-15)
        assert np.count_nonzero(op) == 1
    # sum_m A w(m, m') = A at every m': the total rate out of each |D, m'> is the level's own rate
    for mp in upper:
        total = sum(rate for (_m, m2), rate, _op in ops if m2 == mp)
        assert total == pytest.approx(a, rel=1e-14)


def test_the_total_dissipator_weight_matches_the_88sr_lifetime() -> None:
    """A = 2.5588 s^-1 for the 674 nm line, so every |D, m'> decays at 1/0.3908 s (Section 4.5.7)."""
    sr = species("88Sr+")
    tr = sr.transition("S1/2-D5/2")
    lower, upper, dim = _basis_indices()
    ops = quadrupole_collapse_operators(
        tr.partial_rate_rad_s, HALF, FIVE_HALVES, lower.__getitem__, upper.__getitem__, dim
    )
    total = sum(rate for _k, rate, _op in ops)
    assert total / 6.0 == pytest.approx(2.5588, rel=1e-3), "per m', the level's rate"
    assert tr.partial_rate_rad_s == pytest.approx(1.0 / 0.3908 * (1.0 - 9e-5), rel=1e-9)


def test_the_collapse_operators_reject_a_bad_basis() -> None:
    lower, upper, _dim = _basis_indices()
    with pytest.raises(IndexError):
        quadrupole_collapse_operators(1.0, HALF, FIVE_HALVES, lower.__getitem__, upper.__getitem__, 3)
    with pytest.raises(ValueError, match="non-negative"):
        quadrupole_collapse_operators(-1.0, HALF, FIVE_HALVES, lower.__getitem__, upper.__getitem__, 8)
    with pytest.raises(ValueError, match="non-empty"):
        quadrupole_collapse_operators(1.0, HALF, FIVE_HALVES, lower.__getitem__, upper.__getitem__, 0)


# ---- the E2 ac Stark shift -------------------------------------------------------------------------------


def _toy_e2_table(zeeman_lower_hz: float, zeeman_upper_hz: float, omega: float) -> tuple[dict, dict, dict]:
    """Equal couplings on every allowed component, with linear Zeeman ladders on both manifolds."""
    lower = {m: float(m) * zeeman_lower_hz for m in (-HALF, HALF)}
    upper = {Fraction(k, 2): float(Fraction(k, 2)) * zeeman_upper_hz for k in (-5, -3, -1, 1, 3, 5)}
    couplings = {
        (m, mp): (omega if lambda_3j(HALF, m, FIVE_HALVES, mp) != 0.0 else 0.0) for m in lower for mp in upper
    }
    return couplings, lower, upper


def test_the_e2_stark_shift_is_second_order_in_the_off_resonant_components() -> None:
    """Section 4.5.7: second-order perturbation from the same Omega(m, m') table, the driven component
    excluded. Doubling every coupling quadruples the shift; halving the Zeeman span doubles it."""
    omega = TWO_PI * 1e5
    couplings, lower, upper = _toy_e2_table(1e6, 0.8e6, omega)
    base = e2_stark_shift_rad_s(couplings, lower, upper, -HALF, -HALF)
    assert base != 0.0
    doubled = {k: 2.0 * v for k, v in couplings.items()}
    assert e2_stark_shift_rad_s(doubled, lower, upper, -HALF, -HALF) == pytest.approx(4.0 * base, rel=1e-12)
    narrow_l = {m: 0.5 * v for m, v in lower.items()}
    narrow_u = {m: 0.5 * v for m, v in upper.items()}
    assert e2_stark_shift_rad_s(couplings, narrow_l, narrow_u, -HALF, -HALF) == pytest.approx(
        2.0 * base, rel=1e-12
    )


def test_only_the_components_sharing_a_level_with_the_driven_one_contribute() -> None:
    """A component that shares neither |S, m0> nor |D, m0'> shifts neither of them, so zeroing it changes
    nothing; zeroing one that shares a level does."""
    omega = TWO_PI * 1e5
    couplings, lower, upper = _toy_e2_table(1e6, 0.8e6, omega)
    base = e2_stark_shift_rad_s(couplings, lower, upper, -HALF, -HALF)
    disjoint = dict(couplings)
    disjoint[(HALF, THREE_HALVES if False else Fraction(3, 2))] = 0.0  # shares neither level
    assert e2_stark_shift_rad_s(disjoint, lower, upper, -HALF, -HALF) == pytest.approx(base, rel=1e-15)
    shared = dict(couplings)
    shared[(-HALF, Fraction(-3, 2))] = 0.0  # shares |S, -1/2>
    assert e2_stark_shift_rad_s(shared, lower, upper, -HALF, -HALF) != pytest.approx(base, rel=1e-9)


def test_the_e2_stark_shift_refuses_a_degenerate_component() -> None:
    """Section 4.5.6: these sums have no i gamma/2, so a component degenerate with the driven one is refused
    rather than returning an infinity."""
    couplings, lower, upper = _toy_e2_table(0.0, 0.0, TWO_PI * 1e5)  # everything degenerate
    with pytest.raises(ZeroDivisionError, match="degenerate"):
        e2_stark_shift_rad_s(couplings, lower, upper, -HALF, -HALF)
    with pytest.raises(KeyError, match="driven component"):
        e2_stark_shift_rad_s({}, lower, upper, -HALF, -HALF)


def test_the_derived_optical_e2_drive_now_carries_a_nonzero_stark_shift() -> None:
    """``light/raman.py`` hardcoded 0.0 for ``optical_E2`` until 2026-09-07 (defect B8)."""
    from qutip_trap.light.raman import derive_optical_drive, quadrupole_stark_shift_hz
    from tests.test_light_shift_gate import ca_light_shift_device

    dev = ca_light_shift_device()
    assert dev.crystal.species[0].name == "40Ca+"
    e2_beam = next(i for i, b in enumerate(dev.beams) if 7.0e-7 < b.wavelength_m < 7.5e-7)
    dd = derive_optical_drive(dev, 0, e2_beam, scattering=False)
    assert dd.kind == "optical_E2"
    assert dd.stark_shift_hz != 0.0
    assert dd.stark_shift_hz == pytest.approx(quadrupole_stark_shift_hz(dev, 0, e2_beam), rel=1e-12)
    assert "conv.e2_ac_stark_shift" in dd.provenance
    # the shift is second order in the drive, so it must be small against the carrier Rabi frequency at the
    # 30 MHz Zeeman span of Section 4.5.7
    assert abs(dd.stark_shift_hz) < dd.carrier_rabi_hz


def test_a_hyperfine_e2_stark_shift_raises_as_section_4_5_7_allows() -> None:
    from qutip_trap.light.raman import quadrupole_stark_shift_hz
    from tests.m2_fixtures import single_ion_raman_device

    dev = single_ion_raman_device()
    assert dev.crystal.species[0].nuclear_spin != 0.0
    with pytest.raises(NotImplementedError, match="I = 0"):
        quadrupole_stark_shift_hz(dev, 0, 0)
