"""The 9.5 "Budgets" row's fast-vs-full agreement on an IMPERFECT-transfer scheme (PLAN.md Sections 5.7, 8.4, 8.5).

Section 5.7: "the fast path that applies it directly to the joint state REPLACES those layers rather than preceding them,
so the readout error is never applied twice". The POVM's rows are P(declared | internal LEVEL), the shelving/mapping
transfer channel of Section 8.1 already folded in, so the fast path must index them by the projectively sampled level and
must not sample a start class of its own. Before the 2026-09-07 M5 fix it sampled the class and then indexed the POVM by
it, applying the transfer twice: on a 40Ca+ shelving scheme with a 10 % transfer failure that turned a 0.100 error into
0.193 (a factor 1.9), and the existing agreement test could not see it because ``YB_DIRECT`` has an ideal transfer, where
class and level coincide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.readout.discriminate import (
    ThresholdDiscriminator,
    confusion_from_outcomes,
    max_confusion_discrepancy,
    measure,
    per_ion_confusion,
    povm_confusion_over_levels,
    povm_for,
    product_povm,
)
from qutip_trap.readout.fluorescence import ReadoutScheme
from tests.readout_fixtures import myerson_record_model, product_state, register_space

MYERSON_THRESHOLD = ThresholdDiscriminator(5.5, 420e-6)
TRANSFER = 0.9
"""A 10 % shelving-transfer failure: big enough that applying the channel twice is a factor ~1.9, cheap at 2e4 shots."""


def imperfect_scheme() -> ReadoutScheme:
    """40Ca+ shelving with a 10 % transfer failure: |0> = S1/2 bright, |1> shelved with probability 0.9."""
    return ReadoutScheme.shelving(1, transfer_probability=TRANSFER)


def test_per_ion_confusion_rows_are_indexed_by_level_with_the_transfer_folded_in_once() -> None:
    rm = myerson_record_model()
    scheme = imperfect_scheme()
    m, unc = per_ion_confusion(rm, scheme, MYERSON_THRESHOLD)
    assert m.shape == (2, 2) and unc == 0.0
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-12)
    # row 1 = the shelved level: bright with the 10 % transfer failure, plus the shelf's own tiny leak to bright
    p_bright_shelf = rm.count_distribution("shelf", 420e-6).probability_above(5.5)
    p_bright_bright = rm.count_distribution("bright", 420e-6).probability_above(5.5)
    expected = TRANSFER * p_bright_shelf + (1.0 - TRANSFER) * p_bright_bright
    assert m[1, 0] == pytest.approx(expected, rel=1e-12)
    assert m[1, 0] == pytest.approx(0.10024, rel=1e-4)
    # the polarity is the scheme's: |0> = S1/2 is the bright level, so eps_B belongs to row 0
    pov = product_povm([rm], [scheme], MYERSON_THRESHOLD)
    assert pov.bright_levels == (0,)
    eps_b, eps_d = pov.per_ion_errors()[0]
    assert eps_b == pytest.approx(m[0, 1], rel=1e-12) and eps_d == pytest.approx(m[1, 0], rel=1e-12)


def test_fast_and_full_paths_agree_on_an_imperfect_transfer_scheme() -> None:
    """The 9.5 Budgets row's "the POVM fast path and the full record path agree in confusion matrix at zero readout
    crosstalk" on a scheme where the start class and the qubit level do NOT coincide."""
    rm = myerson_record_model()
    scheme = imperfect_scheme()
    space = register_space(1)
    pov = povm_for([rm], [scheme], MYERSON_THRESHOLD)
    expected = povm_confusion_over_levels(pov, [scheme])[0]
    shots = 20_000
    for level in (0, 1):
        state = space.initial_state(product_state(space, (level,)))
        full = measure(space, state, [scheme], [rm], MYERSON_THRESHOLD, SeedSpec(7), shots=shots)
        fast = measure(
            space, state, [scheme], [rm], MYERSON_THRESHOLD, SeedSpec(7), shots=shots, mode="fast", povm=pov
        )
        assert np.array_equal(full.levels, fast.levels)
        for out in (full, fast):
            emp = confusion_from_outcomes(out, [scheme])[0]
            for bit in (0, 1):
                p = expected[level, bit]
                sigma = math.sqrt(max(p * (1.0 - p), 1.0 / shots) / shots)
                assert emp[level, bit] == pytest.approx(p, abs=4.0 * sigma), (
                    f"level {level} bit {bit} on the {out.mode} path"
                )
    # the defect this pins: indexing the POVM by a re-sampled start class doubled the shelved level's error
    assert expected[1, 0] < 0.13, (
        "applying the transfer channel twice gave 0.193 here (audit 2026-09-07 B1); once gives 0.100"
    )


def test_register_confusion_indexes_the_ions_own_axis_by_level_and_its_neighbours_by_class() -> None:
    """At configured crosstalk the ion's own axis carries its LEVEL (the transfer inside the table) while a neighbour's axis
    carries its start CLASS, which is all the leaked light depends on; the fast path then agrees with the full record path
    on the imperfect-transfer scheme too, and the discrepancy against the product form is reported."""
    rm = myerson_record_model()
    scheme = imperfect_scheme()
    schemes = [scheme, scheme]
    leak = {1: 0.04}
    register = povm_for([rm, rm], schemes, MYERSON_THRESHOLD, leak)
    product = povm_for([rm, rm], schemes, MYERSON_THRESHOLD)
    assert register.factored is not None and register.crosstalk_domain == "configured"
    # ion 0's table: axis 0 its own two LEVELS, axis 1 the neighbour's two CLASSES, axis 2 the declaration
    assert register.factored.tables[0].shape == (2, 2, 2)
    assert register.factored.bright_start[1] == pytest.approx([1.0, 1.0 - TRANSFER])
    # a shelved ion beside a bright one collects the leaked light and is pushed towards bright
    p_alone = register.factored.tables[0][1, 1, 0]
    p_beside_bright = register.factored.tables[0][1, 0, 0]
    assert p_beside_bright > p_alone
    disc = max_confusion_discrepancy(register, product)
    assert 0.0 < disc < 1.0
    space = register_space(2)
    state = space.initial_state(product_state(space, (1, 0)))
    shots = 20_000
    full = measure(space, state, schemes, [rm, rm], MYERSON_THRESHOLD, SeedSpec(9), shots=shots, leakage=leak)
    fast = measure(
        space,
        state,
        schemes,
        [rm, rm],
        MYERSON_THRESHOLD,
        SeedSpec(9),
        shots=shots,
        mode="fast",
        povm=register,
    )
    for i in (0, 1):
        p_full = float(np.mean(full.bits[:, i] == scheme.bit_of_class("bright")))
        p_fast = float(np.mean(fast.bits[:, i] == scheme.bit_of_class("bright")))
        sigma = math.sqrt(max(p_full * (1.0 - p_full), 1.0 / shots) / shots)
        assert p_fast == pytest.approx(p_full, abs=5.0 * sigma + 2e-3), f"ion {i}"


def test_a_leak_level_gets_its_own_povm_row_rather_than_a_qubit_levels_start_class() -> None:
    """A d > 2 register factor whose third level is a leaked D sublevel (class "shelf") must be read out through its OWN
    start distribution, not through the qubit level that happens to share its class (audit 2026-09-07 B16)."""
    import qutip as qt

    from qutip_trap.hilbert.space import HilbertSpace

    rm = myerson_record_model()
    scheme = ReadoutScheme.shelving(1, transfer_probability=TRANSFER, leak_classes=("dark",))
    assert scheme.n_levels == 3 and scheme.classes == ("bright", "shelf", "dark")
    m, _ = per_ion_confusion(rm, scheme, MYERSON_THRESHOLD)
    assert m.shape == (3, 2)
    # the leak level starts "dark" (R_b pumping), NOT "shelf" (1/tau_D decay): different chains, different rows
    p_dark = rm.count_distribution("dark", 420e-6).probability_above(5.5)
    assert m[2, 0] == pytest.approx(p_dark, rel=1e-12)
    assert m[2, 0] != pytest.approx(m[1, 0], rel=1e-3)
    space = HilbertSpace((3,), (), None, ())
    state = space.initial_state(qt.basis(3, 2))
    pov = product_povm([rm], [scheme], MYERSON_THRESHOLD)
    assert pov.per_ion is not None and pov.per_ion[0].shape == (3, 2)
    out = measure(
        space, state, [scheme], [rm], MYERSON_THRESHOLD, SeedSpec(2), shots=400, mode="fast", povm=pov
    )
    assert np.all(out.levels == 2)
    # class "dark" on a shelving scheme reports the non-bright bit, which is the shelved level's bit
    assert float(np.mean(out.bits[:, 0] == scheme.bit_of_class("dark"))) > 0.99
