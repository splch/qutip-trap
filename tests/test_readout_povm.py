"""Section 5.7 and 9.5 targets of the POVM fast path (PLAN.md Sections 5.7, 8.4, 8.5): product form equals the full record path
at zero crosstalk, a bounded and reported discrepancy at the configured crosstalk, Bell correlations through both paths,
the register-wide confusion tensor and its size guard, and the keyed photon-record seeds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.dynamics.engine import SeedSpec
from qutip_trap.readout.discriminate import (
    POVM,
    RegisterConfusion,
    ThresholdDiscriminator,
    TimeResolvedML,
    confusion_from_outcomes,
    joint_level_probabilities,
    max_confusion_discrepancy,
    measure,
    per_ion_confusion,
    povm_confusion_over_levels,
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


def test_product_povm_is_exact_for_the_threshold_discriminator_and_sums_to_identity() -> None:
    rm = crain_record_model()
    pov = product_povm([rm, rm], [YB_DIRECT, YB_DIRECT], THRESHOLD)
    assert pov.crosstalk_domain == "zero" and pov.per_ion is not None and pov.uncertainty == 0.0
    m = pov.per_ion[0]
    assert np.allclose(m.sum(axis=1), 1.0)
    eps_b, eps_d = pov.per_ion_errors()[0]
    exact_b = rm.count_distribution("bright", 22e-6).pmf[0]
    exact_d = rm.count_distribution("dark", 22e-6).probability_above(0.5)
    assert eps_b == pytest.approx(exact_b, rel=1e-12) and eps_d == pytest.approx(exact_d, rel=1e-12)
    # re-indexed by qubit level: |1> is bright for 171Yb+
    by_level = povm_confusion_over_levels(pov, [YB_DIRECT, YB_DIRECT])[0]
    assert by_level[1, 1] == pytest.approx(1.0 - eps_b) and by_level[0, 1] == pytest.approx(eps_d)


def test_imperfect_shelving_transfer_makes_pi_dark_non_rank_one_in_the_povm() -> None:
    """Harty-like transfer (1.7e-4 of the shelved level stays bright, 3e-4 of the other level shelves off-resonantly) enters the
    POVM through the start distribution: the dark-outcome element is a mixture, not a projector."""
    rm = myerson_record_model()
    ideal = ReadoutScheme.shelving(1)
    imperfect = ReadoutScheme.shelving(1, transfer_probability=1.0 - 1.7e-4, off_resonant_shelving=3e-4)
    disc = ThresholdDiscriminator(5.5, 420e-6)
    m0, _ = per_ion_confusion(rm, ideal, disc)
    m1, _ = per_ion_confusion(rm, imperfect, disc)
    assert m1[0, 1] == pytest.approx(m0[0, 1] + 3e-4, rel=2e-2)
    assert m1[1, 0] == pytest.approx(m0[1, 0] + 1.7e-4, rel=2e-2)
    assert imperfect.dark_weights() == pytest.approx((3e-4, 1.0 - 1.7e-4))


def test_full_and_fast_paths_agree_in_confusion_at_zero_crosstalk_and_bell_correlations_survive() -> None:
    """The record path (sampled chains, Poisson counts, threshold) and the POVM path give the same per-ion confusion within
    statistics, and the Bell state's bit correlations survive both because the joint outcome is sampled projectively first."""
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
        p_equal = float(np.mean(out.bits[:, 0] == out.bits[:, 1]))
        assert p_equal > 0.995
        p00 = float(np.mean((out.bits[:, 0] == 0) & (out.bits[:, 1] == 0)))
        assert p00 == pytest.approx(0.5, abs=0.03)
    expected = povm_confusion_over_levels(pov, schemes)
    for out in (full, fast):
        emp = confusion_from_outcomes(out, schemes)
        # error entries ~ 4e-4 .. 8e-4 on ~3000 shots per level: agree within 4 sigma of the binomial
        for i in range(2):
            for lev in (0, 1):
                p = expected[i, lev, 1 - lev]
                assert emp[i, lev, 1 - lev] == pytest.approx(p, abs=4.0 * math.sqrt(p / 3000.0) + 5e-4)


def test_register_confusion_at_configured_crosstalk_is_bounded_and_reported() -> None:
    """At 4.0 % nearest-neighbour leakage a dark ion beside a bright one collects 0.04 x 472 kcps x 22 us = 0.42 counts: the
    zero-threshold protocol misreads it 34 % of the time, so the product POVM (which cannot see the neighbour) and the
    register-wide confusion differ by that bounded, reported amount, while the full record path agrees with the register form."""
    rm = crain_record_model()
    schemes = [YB_DIRECT, YB_DIRECT]
    product = povm_for([rm, rm], schemes, THRESHOLD)
    register = povm_for([rm, rm], schemes, THRESHOLD, {1: 0.04})
    assert (
        register.crosstalk_domain == "configured"
        and register.confusion is not None
        and register.factored is not None
    )
    assert register.confusion.shape == (4, 4) and np.allclose(register.confusion.sum(axis=1), 1.0)
    # true (ion0 bright, ion1 dark) = index 0b01 = 1; declared both bright = 0b11 = 3
    leaked = register.declared_bright_probability([1, 0], [True, True])
    assert leaked == pytest.approx(1.0 - math.exp(-(4.2 + 0.04 * 472e3) * 22e-6) - 0.0, abs=0.02)
    assert product.declared_bright_probability([1, 0], [True, True]) < 1e-3
    discrepancy = float(np.max(np.abs(register.confusion - _dense_product(product))))
    assert 0.3 < discrepancy < 0.4
    assert max_confusion_discrepancy(register, product) == pytest.approx(discrepancy, abs=0.02)
    # the full path on |01> (ion 0 in |0> = dark, ion 1 in |1> = bright) reproduces the register form, not the product
    space = register_space(2)
    state = space.initial_state(product_state(space, (0, 1)))
    full = measure(
        space, state, schemes, [rm, rm], THRESHOLD, SeedSpec(3), shots=4000, mode="full", leakage={1: 0.04}
    )
    p_dark_read_bright = float(np.mean(full.bits[:, 0] == 1))
    assert p_dark_read_bright == pytest.approx(leaked, abs=0.03)
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


def _dense_product(povm: POVM) -> np.ndarray:
    n = povm.n_ions
    out = np.zeros((2**n, 2**n))
    for a in range(2**n):
        levels = [(a >> i) & 1 for i in range(n)]
        for b in range(2**n):
            declared = [bool((b >> i) & 1) for i in range(n)]
            out[a, b] = povm.declared_bright_probability(levels, declared)
    return out


def test_register_confusion_factors_by_neighbour_range_and_guards_the_dense_tensor() -> None:
    rm = crain_record_model()
    n = 13
    pov = povm_for([rm] * n, [YB_DIRECT] * n, THRESHOLD, {1: 0.01})
    assert pov.confusion is None and pov.factored is not None and pov.factored.neighbour_range == 1
    assert pov.factored.neighbourhood(0) == (0, 1) and pov.factored.neighbourhood(6) == (5, 6, 7)
    with pytest.raises(ValueError):
        pov.factored.dense()
    true = [1] * n
    p = pov.declared_bright_probability(true, [True] * n)
    assert 0.98 < p < 1.0
    rng = np.random.default_rng(0)
    sampled = pov.sample(true, rng)
    assert len(sampled) == n
    small = povm_for([rm] * 3, [YB_DIRECT] * 3, THRESHOLD, {1: 0.01, 2: 0.002})
    assert small.confusion is not None and small.confusion.shape == (8, 8)
    assert small.factored is not None and small.factored.neighbour_range == 2
    total = sum(
        small.declared_bright_probability([1, 0, 1], [bool(b >> i & 1) for i in range(3)]) for b in range(8)
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_povm_invariants() -> None:
    m = np.array([[0.999, 0.001], [0.002, 0.998]])
    with pytest.raises(ValueError):
        POVM(None, None, "zero")
    with pytest.raises(ValueError):
        POVM((m,), np.eye(2), "configured", bright_levels=(0,))
    with pytest.raises(ValueError):
        POVM(None, np.eye(4), "zero")
    with pytest.raises(ValueError):
        POVM((np.array([[0.9, 0.2], [0.0, 1.0]]),), None, "zero", bright_levels=(0,))
    with pytest.raises(ValueError, match="one bright level"):
        POVM((m,), None, "zero")
    # a product form can never carry configured crosstalk: Section 8.5 makes the records neighbour-coupled
    with pytest.raises(ValueError, match="product form cannot carry configured crosstalk"):
        POVM((m,), None, "configured", bright_levels=(0,))
    ok = POVM((m,), None, "zero", bright_levels=(0,))
    assert ok.n_ions == 1 and ok.per_ion_errors() == ((0.001, 0.002),)
    # the same table with the OTHER polarity names the two errors the other way round
    flipped = POVM((m,), None, "zero", bright_levels=(1,))
    assert flipped.per_ion_errors() == ((0.998, 0.999),)
    factored = RegisterConfusion(2, 0, (m,) * 2, (np.array([1.0, 0.0]),) * 2)
    assert factored.probability([0, 1], [True, False]) == pytest.approx(0.999 * 0.998)


def test_two_shots_on_one_sample_and_trajectory_draw_different_records() -> None:
    """Section 9.13 seeds row: the shot slot in the seed key gives two shots on one (sample, trajectory) independent photon
    records, and a run is reproducible shot by shot."""
    rm = crain_record_model(window_s=200e-6)
    space = register_space(1)
    state = space.initial_state(product_state(space, (1,)))
    seeds = SeedSpec(21)
    disc = ThresholdDiscriminator(0.5, 200e-6)
    a = measure(space, state, [YB_DIRECT], [rm], disc, seeds, shots=6, mode="full", keep_records=True)
    b = measure(space, state, [YB_DIRECT], [rm], disc, seeds, shots=6, mode="full", keep_records=True)
    assert a.records is not None and b.records is not None
    totals_a = [rec[0].total for rec in a.records]
    totals_b = [rec[0].total for rec in b.records]
    assert totals_a == totals_b, "the same seeds reproduce the same records"
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
    # |0> = S1/2 is the BRIGHT level for the optical qubit: the bit is 0 when the ion fluoresces
    assert CA_OPTICAL.bit_of_class("bright") == 0


def test_leakage_levels_are_read_out_by_their_class() -> None:
    """A qutrit ion whose third level lies in the bright manifold (a leaked F = 1 sublevel of 171Yb+) reads as the bright bit."""
    rm = crain_record_model()
    scheme = ReadoutScheme.direct(1, leak_classes=("bright",))
    import qutip as qt

    from qutip_trap.hilbert.space import HilbertSpace

    space = HilbertSpace((3,), (), None, ())
    state = space.initial_state(qt.basis(3, 2))
    out = measure(space, state, [scheme], [rm], THRESHOLD, SeedSpec(2), shots=200, mode="full")
    assert np.all(out.levels == 2)
    assert np.mean(out.bits) > 0.99
    fast = measure(space, state, [scheme], [rm], THRESHOLD, SeedSpec(2), shots=200, mode="fast")
    assert np.mean(fast.bits) > 0.99
