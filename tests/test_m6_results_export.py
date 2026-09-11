"""Section 8.6's per-shot noise attribution and v2 register-nested export (M6 audit items B-6 and 18).

Section 8.6: "A ``Result`` carries per shot: ... the sampled quasi-static noise parameters of that shot's dynamical sample"
and "a v2 register-nested format ... keyed by bitstrings per named register". ``Result.noise_samples`` is per SAMPLE, and the
shot -> sample map lived only in a local variable of ``run``; ``Diagnostics.shots_per_sample`` is a floor, so the map could
not be reconstructed when ``shots % samples != 0``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.fixtures import make_result

BITS = np.array([[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)


def test_sample_of_shot_is_the_contiguous_block_map() -> None:
    """Shots are allocated in contiguous blocks (``conv.shot_blocks_per_sample``), so sample k owns
    [sum_{j<k} M_j, sum_{j<=k} M_j) and ``noise_samples[sample_of_shot[k]]`` is shot k's parameter draw."""
    result = make_result(BITS)
    # one sample (the default fixture): every shot maps to it
    assert result.sample_of_shot.tolist() == [0, 0, 0, 0, 0]
    uneven = dataclasses.replace(
        result,
        diagnostics=dataclasses.replace(result.diagnostics, shots_per_sample_realized=(2, 2, 1)),
    )
    assert uneven.sample_of_shot.tolist() == [0, 0, 1, 1, 2]
    # the floored shots_per_sample cannot express this: 5 shots over 3 samples floors to 1
    assert uneven.diagnostics.shots_per_sample != 2
    assert sum(uneven.diagnostics.shots_per_sample_realized) == uneven.shots


def test_the_v2_register_nested_export() -> None:
    """{register name: {bitstring: probability}} with the Section 13 bit order inside each register (its qubit 0 rightmost)."""
    result = make_result(BITS)
    default = result.to_ionq_v2()
    assert default == {"c": {"00": 0.4, "10": 0.2, "11": 0.4}}
    assert default["c"] == pytest.approx(result.probabilities)
    split = result.to_ionq_v2({"a": (0,), "b": (1,)})
    assert split == {"a": {"0": 0.6, "1": 0.4}, "b": {"0": 0.4, "1": 0.6}}
    # the marginals of the joint histogram, so the format loses nothing a caller can recover
    assert sum(split["a"].values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the result"):
        result.to_ionq_v2({"bad": (99,)})


def test_branches_and_trajectories_are_separate_diagnostics() -> None:
    """``trajectories`` counts Fock branches TIMES the engine's trajectory count (``conv.fock_sum_branches``); ``branches``
    reports the Fock sum alone, so a reader can tell one from the other."""
    result = make_result(BITS)
    assert result.diagnostics.branches >= 1
    assert result.diagnostics.trajectories >= result.diagnostics.branches


# ---- the IonQ character orders, pinned before the v2 exporter changes (docs/api_implementation_plan.md items 0.4 and 1.7) ---

X_ON_QUBIT_ZERO = np.array([[1, 0, 0]] * 4, dtype=np.uint8)
"""``x q[0]`` on three qubits, four shots: column j of ``Result.bitstrings`` is qubit j."""
IONQ_V1_KEY = "1"
"""The v1 decimal key of that shot: qubit 0 the least-significant bit."""
IONQ_V2_KEY = "100"
"""The v2 bitstring of that shot: wire order, ``q[0]`` the leftmost character (``qutip_trap.io.ionq`` records the source)."""


def test_x_on_qubit_zero_reads_1_in_the_v1_formats_and_001_in_this_package() -> None:
    """A Bell state cannot tell the two character orders apart; ``x q[0]`` can. The v1 exporters are correct today."""
    result = make_result(X_ON_QUBIT_ZERO)
    assert result.counts == {"001": 4}  # this package's key: qubit 0 rightmost
    assert result.to_ionq_json() == {IONQ_V1_KEY: 1.0}
    assert result.to_ionq_histogram() == {IONQ_V1_KEY: 4}
    assert result.to_ionq_shots() == [IONQ_V1_KEY] * 4
    assert IONQ_V2_KEY == "001"[::-1]  # the v2 string is this package's key reversed


@pytest.mark.xfail(
    strict=True,
    reason="Phase 1.7 of docs/api_implementation_plan.md: Result.to_ionq_v2_probabilities() emits the v0.4 envelope with "
    "output_all in wire order; today's to_ionq_v2 has neither the envelope nor that order (remove this marker there)",
)
def test_x_on_qubit_zero_reads_100_in_the_v2_envelope() -> None:
    result = make_result(X_ON_QUBIT_ZERO)
    assert result.to_ionq_v2_probabilities() == {
        "probabilities": {"registers": {"output_all": {IONQ_V2_KEY: 1.0}}}
    }
