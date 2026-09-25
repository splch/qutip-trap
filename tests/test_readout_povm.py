"""The readout POVM and the measurement (PLAN.md Section 5.7): the product form equals the full record path at
zero crosstalk, the register-wide confusion carries a bounded, reported discrepancy at configured crosstalk, the rows are
indexed by level with the transfer channel folded in once, correlations survive both paths, and the record seeds are
keyed by shot."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
import qutip as qt

from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.hilbert.space import HilbertSpace
from qutip_trap.readout.discriminate import (
    POVM,
    ReadoutOutcome,
    RegisterConfusion,
    ThresholdDiscriminator,
    TimeResolvedML,
    joint_level_probabilities,
    max_confusion_discrepancy,
    measure,
    per_ion_confusion,
    povm_for,
    product_povm,
)
from qutip_trap.readout.fluorescence import ReadoutScheme
from tests.readout_fixtures import (
    CA_OPTICAL,
    YB_DIRECT,
    bell_state,
    crain_record_model,
    myerson_record_model,
    product_state,
    register_space,
)

THRESHOLD = ThresholdDiscriminator(0.5, 22e-6)
MYERSON_THRESHOLD = ThresholdDiscriminator(5.5, 420e-6)
TRANSFER = 0.9
"""A 10 % shelving-transfer failure, where the start class and the qubit level do not coincide."""


def _empirical_confusion(outcome: ReadoutOutcome) -> np.ndarray:
    """P(declared bit | true level) from a measurement, (n_ions, 2, 2) over (level, bit)."""
    n = outcome.bits.shape[1]
    out = np.zeros((n, 2, 2))
    for i in range(n):
        for lev in (0, 1):
            mask = outcome.levels[:, i] == lev
            if np.any(mask):
                out[i, lev, 1] = float(np.mean(outcome.bits[mask, i]))
                out[i, lev, 0] = 1.0 - out[i, lev, 1]
    return out


def _povm_confusion(povm: POVM, schemes: Sequence[ReadoutScheme]) -> np.ndarray:
    """The product POVM's declared-bright column as the reported bit, (n_ions, 2, 2) over (level, bit)."""
    assert povm.per_ion is not None
    out = np.zeros((len(schemes), 2, 2))
    for i, (m, s) in enumerate(zip(povm.per_ion, schemes)):
        for lev in (0, 1):
            out[i, lev, s.bright_level] = float(m[lev, 0])
            out[i, lev, 1 - s.bright_level] = 1.0 - float(m[lev, 0])
    return out


def _dense(povm: POVM) -> np.ndarray:
    """The 2^N x 2^N tensor over (true level pattern, declared-bright pattern), bit i = ion i."""
    n = povm.n_ions
    out = np.zeros((2**n, 2**n))
    for a in range(2**n):
        levels = [(a >> i) & 1 for i in range(n)]
        for b in range(2**n):
            out[a, b] = povm.declared_bright_probability(levels, [bool((b >> i) & 1) for i in range(n)])
    return out


def test_product_povm_is_exact_for_the_threshold_discriminator_and_sums_to_identity() -> None:
    rm = crain_record_model()
    pov = product_povm([rm, rm], [YB_DIRECT, YB_DIRECT], THRESHOLD)
    assert pov.per_ion is not None and pov.factored is None and pov.uncertainty == 0.0
    assert np.allclose(pov.per_ion[0].sum(axis=1), 1.0)
    eps_b, eps_d = pov.per_ion_errors()[0]
    exact_b = rm.count_distribution("bright", 22e-6).pmf[0]
    exact_d = rm.count_distribution("dark", 22e-6).probability_above(0.5)
    assert eps_b == pytest.approx(exact_b, rel=1e-12) and eps_d == pytest.approx(exact_d, rel=1e-12)
    # re-indexed by qubit level: |1> is bright for 171Yb+
    by_level = _povm_confusion(pov, [YB_DIRECT, YB_DIRECT])[0]
    assert by_level[1, 1] == pytest.approx(1.0 - eps_b) and by_level[0, 1] == pytest.approx(eps_d)


def test_imperfect_shelving_transfer_makes_pi_dark_non_rank_one_in_the_povm() -> None:
    """Harty-like transfer (1.7e-4 of the shelved level stays bright, 3e-4 of the other shelves off-resonantly) enters the
    POVM through the start distribution: the dark-outcome element is a mixture, not a projector."""
    rm = myerson_record_model()
    ideal = ReadoutScheme.shelving(1)
    imperfect = ReadoutScheme.shelving(1, transfer_probability=1.0 - 1.7e-4, off_resonant_shelving=3e-4)
    m0, _ = per_ion_confusion(rm, ideal, MYERSON_THRESHOLD)
    m1, _ = per_ion_confusion(rm, imperfect, MYERSON_THRESHOLD)
    assert m1[0, 1] == pytest.approx(m0[0, 1] + 3e-4, rel=2e-2)
    assert m1[1, 0] == pytest.approx(m0[1, 0] + 1.7e-4, rel=2e-2)


def test_per_ion_confusion_rows_are_indexed_by_level_with_the_transfer_folded_in_once() -> None:
    rm = myerson_record_model()
    scheme = ReadoutScheme.shelving(1, transfer_probability=TRANSFER)
    m, unc = per_ion_confusion(rm, scheme, MYERSON_THRESHOLD)
    assert m.shape == (2, 2) and unc == 0.0
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-12)
    # row 1 = the shelved level: bright with the 10 % transfer failure, plus the shelf's own tiny leak to bright
    p_bright_shelf = rm.count_distribution("shelf", 420e-6).probability_above(5.5)
    p_bright_bright = rm.count_distribution("bright", 420e-6).probability_above(5.5)
    assert m[1, 0] == pytest.approx(TRANSFER * p_bright_shelf + (1.0 - TRANSFER) * p_bright_bright, rel=1e-12)
    assert m[1, 0] == pytest.approx(0.10024, rel=1e-4)
    # the polarity is the scheme's: |0> = S1/2 is the bright level, so eps_B belongs to row 0
    pov = product_povm([rm], [scheme], MYERSON_THRESHOLD)
    assert pov.bright_levels == (0,)
    eps_b, eps_d = pov.per_ion_errors()[0]
    assert eps_b == pytest.approx(m[0, 1], rel=1e-12) and eps_d == pytest.approx(m[1, 0], rel=1e-12)


def test_full_and_fast_paths_agree_in_confusion_at_zero_crosstalk_and_bell_correlations_survive() -> None:
    """The record path and the POVM path give the same per-ion confusion within statistics, and the Bell state's bit
    correlations survive both because the joint outcome is sampled projectively first."""
    rm = crain_record_model()
    space = register_space(2)
    state = space.initial_state(bell_state(space))
    probs = joint_level_probabilities(space, state)
    assert probs[0, 0] == pytest.approx(0.5) and probs[1, 1] == pytest.approx(0.5) and probs[0, 1] == 0.0
    seeds = SeedSpec(11)
    schemes = [YB_DIRECT, YB_DIRECT]
    pov = povm_for([rm, rm], schemes, THRESHOLD)
    n = 6000
    full = measure(space, state, schemes, [rm, rm], THRESHOLD, seeds, shots=n, mode="full")
    fast = measure(space, state, schemes, [rm, rm], THRESHOLD, seeds, shots=n, mode="fast", povm=pov)
    assert full.mode == "full" and fast.mode == "fast" and full.bits.shape == (n, 2)
    assert np.array_equal(full.levels, fast.levels), (
        "the projective outcome is keyed by the seeds, not by the path"
    )
    assert np.all(full.levels[:, 0] == full.levels[:, 1])
    for out in (full, fast):
        assert float(np.mean(out.bits[:, 0] == out.bits[:, 1])) > 0.995
        p00 = float(np.mean((out.bits[:, 0] == 0) & (out.bits[:, 1] == 0)))
        assert p00 == pytest.approx(0.5, abs=0.03)
    expected = _povm_confusion(pov, schemes)
    for out in (full, fast):
        emp = _empirical_confusion(out)
        for i in range(2):
            for lev in (0, 1):
                p = expected[i, lev, 1 - lev]
                assert emp[i, lev, 1 - lev] == pytest.approx(p, abs=4.0 * math.sqrt(p / 3000.0) + 5e-4)


def test_fast_and_full_paths_agree_on_an_imperfect_transfer_scheme() -> None:
    """Where the start class and the qubit level do not coincide the fast path must index the POVM by level and sample no
    start class of its own; both paths then agree with the POVM (applying the transfer twice would give 0.193, not 0.100)."""
    rm = myerson_record_model()
    scheme = ReadoutScheme.shelving(1, transfer_probability=TRANSFER)
    space = register_space(1)
    pov = povm_for([rm], [scheme], MYERSON_THRESHOLD)
    expected = _povm_confusion(pov, [scheme])[0]
    shots = 20_000
    for level in (0, 1):
        state = space.initial_state(product_state(space, (level,)))
        full = measure(space, state, [scheme], [rm], MYERSON_THRESHOLD, SeedSpec(7), shots=shots)
        fast = measure(
            space, state, [scheme], [rm], MYERSON_THRESHOLD, SeedSpec(7), shots=shots, mode="fast", povm=pov
        )
        assert np.array_equal(full.levels, fast.levels)
        for out in (full, fast):
            emp = _empirical_confusion(out)[0]
            for bit in (0, 1):
                p = expected[level, bit]
                sigma = math.sqrt(max(p * (1.0 - p), 1.0 / shots) / shots)
                assert emp[level, bit] == pytest.approx(p, abs=4.0 * sigma), (
                    f"level {level} bit {bit} ({out.mode})"
                )
    assert expected[1, 0] < 0.13


def test_register_confusion_at_configured_crosstalk_is_bounded_and_reported() -> None:
    """At 4.0 % nearest-neighbour leakage a dark ion beside a bright one collects 0.42 counts in 22 us: the zero-threshold
    protocol misreads it 34 % of the time, so the product POVM (blind to the neighbour) and the register-wide confusion
    differ by that bounded, reported amount, while the full record path agrees with the register form."""
    rm = crain_record_model()
    schemes = [YB_DIRECT, YB_DIRECT]
    product = povm_for([rm, rm], schemes, THRESHOLD)
    register = povm_for([rm, rm], schemes, THRESHOLD, {1: 0.04})
    assert register.per_ion is None and register.factored is not None
    dense = _dense(register)
    assert dense.shape == (4, 4) and np.allclose(dense.sum(axis=1), 1.0)
    # true (ion0 bright, ion1 dark) = levels [1, 0]; declared both bright
    leaked = register.declared_bright_probability([1, 0], [True, True])
    assert leaked == pytest.approx(1.0 - math.exp(-(4.2 + 0.04 * 472e3) * 22e-6), abs=0.02)
    assert product.declared_bright_probability([1, 0], [True, True]) < 1e-3
    discrepancy = float(np.max(np.abs(dense - _dense(product))))
    assert 0.3 < discrepancy < 0.4
    assert max_confusion_discrepancy(register, product) == pytest.approx(discrepancy, abs=0.02)
    # the full path on |01> (ion 0 dark, ion 1 bright) reproduces the register form, not the product
    space = register_space(2)
    state = space.initial_state(product_state(space, (0, 1)))
    full = measure(
        space, state, schemes, [rm, rm], THRESHOLD, SeedSpec(3), shots=4000, mode="full", leakage={1: 0.04}
    )
    assert float(np.mean(full.bits[:, 0] == 1)) == pytest.approx(leaked, abs=0.03)
    fast = measure(
        space, state, schemes, [rm, rm], THRESHOLD, SeedSpec(3), shots=4000, mode="fast", povm=register
    )
    assert float(np.mean(fast.bits[:, 0] == 1)) == pytest.approx(leaked, abs=0.03)
    # a Bell state does not expose the crosstalk (neighbours always share the class), and its correlations survive
    bell = space.initial_state(bell_state(space))
    out = measure(
        space, bell, schemes, [rm, rm], THRESHOLD, SeedSpec(5), shots=3000, mode="fast", povm=register
    )
    assert float(np.mean(out.bits[:, 0] == out.bits[:, 1])) > 0.995


def test_register_confusion_indexes_the_ions_own_axis_by_level_and_its_neighbours_by_class() -> None:
    """The ion's own axis carries its level (the transfer inside the table), a neighbour's axis its start class; the fast
    path then agrees with the full record path on the imperfect-transfer scheme too."""
    rm = myerson_record_model()
    scheme = ReadoutScheme.shelving(1, transfer_probability=TRANSFER)
    schemes = [scheme, scheme]
    leak = {1: 0.04}
    register = povm_for([rm, rm], schemes, MYERSON_THRESHOLD, leak)
    product = povm_for([rm, rm], schemes, MYERSON_THRESHOLD)
    assert register.factored is not None
    # ion 0's table: axis 0 its own two levels, axis 1 the neighbour's two classes, axis 2 the declaration
    assert register.factored.tables[0].shape == (2, 2, 2)
    assert register.factored.bright_start[1] == pytest.approx([1.0, 1.0 - TRANSFER])
    # a shelved ion beside a bright one collects the leaked light and is pushed towards bright
    assert register.factored.tables[0][1, 0, 0] > register.factored.tables[0][1, 1, 0]
    assert 0.0 < max_confusion_discrepancy(register, product) < 1.0
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


def test_register_confusion_factors_by_neighbour_range() -> None:
    rm = crain_record_model()
    n = 13
    pov = povm_for([rm] * n, [YB_DIRECT] * n, THRESHOLD, {1: 0.01})
    assert pov.factored is not None and pov.factored.neighbour_range == 1
    assert pov.factored.neighbourhood(0) == (0, 1) and pov.factored.neighbourhood(6) == (5, 6, 7)
    true = [1] * n
    assert 0.98 < pov.declared_bright_probability(true, [True] * n) < 1.0
    assert len(pov.sample(true, np.random.default_rng(0))) == n
    small = povm_for([rm] * 3, [YB_DIRECT] * 3, THRESHOLD, {1: 0.01, 2: 0.002})
    assert small.factored is not None and small.factored.neighbour_range == 2
    total = sum(
        small.declared_bright_probability([1, 0, 1], [bool(b >> i & 1) for i in range(3)]) for b in range(8)
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_povm_invariants() -> None:
    m = np.array([[0.999, 0.001], [0.002, 0.998]])
    factored = RegisterConfusion(2, 0, (m,) * 2, (np.array([1.0, 0.0]),) * 2)
    with pytest.raises(ValueError):
        POVM(None)
    with pytest.raises(ValueError):
        POVM((m,), factored, bright_levels=(0,))
    with pytest.raises(ValueError):
        POVM((np.array([[0.9, 0.2], [0.0, 1.0]]),), bright_levels=(0,))
    with pytest.raises(ValueError, match="one bright level"):
        POVM((m,))
    ok = POVM((m,), bright_levels=(0,))
    assert ok.n_ions == 1 and ok.per_ion_errors() == ((0.001, 0.002),)
    # the same table with the other polarity names the two errors the other way round
    assert POVM((m,), bright_levels=(1,)).per_ion_errors() == ((0.998, 0.999),)
    assert factored.probability([0, 1], [True, False]) == pytest.approx(0.999 * 0.998)
    assert POVM(None, factored).n_ions == 2


def test_two_shots_on_one_sample_and_trajectory_draw_different_records() -> None:
    """The shot slot in the seed key gives two shots on one (sample, trajectory) independent photon records, and a run is
    reproducible shot by shot."""
    rm = crain_record_model(window_s=200e-6)
    space = register_space(1)
    state = space.initial_state(product_state(space, (1,)))
    seeds = SeedSpec(21)
    disc = ThresholdDiscriminator(0.5, 200e-6)
    a = measure(space, state, [YB_DIRECT], [rm], disc, seeds, shots=6, mode="full", keep_records=True)
    b = measure(space, state, [YB_DIRECT], [rm], disc, seeds, shots=6, mode="full", keep_records=True)
    assert a.records is not None and b.records is not None
    totals_a = [rec[0].total for rec in a.records]
    assert totals_a == [rec[0].total for rec in b.records], "the same seeds reproduce the same records"
    assert len(set(totals_a)) > 1, "different shots draw different records"
    c = measure(
        space, state, [YB_DIRECT], [rm], disc, seeds, shots=6, mode="full", keep_records=True, first_shot=6
    )
    assert c.records is not None and [rec[0].total for rec in c.records] != totals_a
    assert a.time_used_s.shape == (6, 1) and np.all(a.time_used_s == 200e-6)


def test_measure_with_a_time_resolved_discriminator_reports_posteriors_and_the_shelving_polarity() -> None:
    rm = myerson_record_model()
    space = register_space(1)
    disc = TimeResolvedML(10e-6, 420e-6, dark_class="shelf")
    for level, expected_bit in ((0, 0), (1, 1)):
        state = space.initial_state(product_state(space, (level,)))
        out = measure(space, state, [CA_OPTICAL], [rm], disc, SeedSpec(1), shots=40, mode="full")
        assert np.mean(out.bits[:, 0] == expected_bit) > 0.95
        assert out.posteriors is not None and np.all(out.posteriors[:, 0] >= 0.0)
    # |0> = S1/2 is the bright level of the optical qubit: the bit is 0 when the ion fluoresces
    assert CA_OPTICAL.bit_of_class("bright") == 0


def test_leakage_levels_are_read_out_by_their_class() -> None:
    """A qutrit whose third level is bright (a leaked F = 1 sublevel of 171Yb+) reads as the bright bit on both paths."""
    rm = crain_record_model()
    scheme = ReadoutScheme.direct(1, leak_classes=("bright",))
    space = HilbertSpace((3,), (), None, ())
    state = space.initial_state(qt.basis(3, 2))
    out = measure(space, state, [scheme], [rm], THRESHOLD, SeedSpec(2), shots=200, mode="full")
    assert np.all(out.levels == 2)
    assert np.mean(out.bits) > 0.99
    fast = measure(space, state, [scheme], [rm], THRESHOLD, SeedSpec(2), shots=200, mode="fast")
    assert np.mean(fast.bits) > 0.99


def test_a_leak_level_gets_its_own_povm_row_rather_than_a_qubit_levels_start_class() -> None:
    """A third level that is a leaked D sublevel read "dark" goes through its own start distribution (R_b pumping), not
    through the qubit level that shares a class with it (the shelf's 1/tau_D decay)."""
    rm = myerson_record_model()
    scheme = ReadoutScheme.shelving(1, transfer_probability=TRANSFER, leak_classes=("dark",))
    assert scheme.n_levels == 3 and scheme.classes == ("bright", "shelf", "dark")
    m, _ = per_ion_confusion(rm, scheme, MYERSON_THRESHOLD)
    assert m.shape == (3, 2)
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
    assert float(np.mean(out.bits[:, 0] == scheme.bit_of_class("dark"))) > 0.99
