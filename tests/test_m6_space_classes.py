"""The three mode classes (PLAN.md Section 5.2): the contribution pair (|alpha_m|^2 (2 nbar_m + 1), |chi_m|) of the played
entangling waveforms decides, and the Debye-Waller spread keeps a closed but strongly coupled loop frozen."""

from __future__ import annotations

import pytest

from qutip_trap.run.space import (
    DROP_ALPHA_MAX,
    DROP_CHI_MAX_RAD,
    DW_SPREAD_DROP_MAX,
    ModeContribution,
    classify,
)

FREEZE_ALPHA = 1e-4
FREEZE_CHI = 0.05


def _contrib(alpha2: float, chi: float, dw_spread_rad: float = 0.0) -> ModeContribution:
    return ModeContribution(
        mode=3,
        alpha2_weighted=alpha2,
        chi_rad=chi,
        radius=0.0,
        eta_max=0.05,
        dw_spread_rad=dw_spread_rad,
    )


@pytest.mark.parametrize("coupled", [True, False])
def test_below_the_drop_pair_is_dropped_whatever_couples_to_it(coupled: bool) -> None:
    """Dropped when |alpha_m|^2 (2 n_m + 1) < 1e-6 AND |chi_m| < 1e-4 AND the Debye-Waller spread is negligible, regardless
    of the coupling: eta = 0.008 at nbar = 0.05 is a 1.5e-5 rad spread."""
    got = classify(
        _contrib(3.5e-33, 0.0, 1.5e-5),
        coupled=coupled,
        freeze_alpha_max=FREEZE_ALPHA,
        freeze_chi_max_rad=FREEZE_CHI,
    )
    assert got == "dropped"


@pytest.mark.parametrize("coupled", [True, False])
def test_a_dropped_loop_pair_with_a_large_debye_waller_spread_is_frozen(coupled: bool) -> None:
    """A pulse that CLOSES a mode's loop leaves the pair at ~1e-33 however strongly the mode couples, and dropping the mode
    would take its exact Debye-Waller factor out of every carrier pulse; the calibration absorbs the factor's mean, not its
    shot-to-shot spread eta^2 sqrt(nbar (nbar + 1)). On the three-ion tilt mode (eta = 0.0809): 1.500e-3 rad at nbar = 0.05,
    9.256e-3 at nbar = 1 and 6.864e-2 at nbar = 10."""
    kw = dict(coupled=coupled, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI)
    for spread in (1.500e-3, 9.256e-3, 6.864e-2):
        assert classify(_contrib(3.5e-33, 0.0, spread), **kw) == "frozen", spread  # type: ignore[arg-type]
    # the threshold is 3e-4 rad and the comparison is strict, like the other two rows
    assert DW_SPREAD_DROP_MAX == 3e-4
    assert classify(_contrib(0.0, 0.0, DW_SPREAD_DROP_MAX), **kw) == "frozen"  # type: ignore[arg-type]
    assert (
        classify(_contrib(0.0, 0.0, DW_SPREAD_DROP_MAX * (1 - 1e-12)), **kw) == "dropped"  # type: ignore[arg-type]
    )
    # a small-eta mode stays dropped: eta = 1e-3 at nbar = 0.05 is a 2.3e-7 rad spread
    assert classify(_contrib(1e-7, 1e-5, 1e-3**2 * (0.05 * 1.05) ** 0.5), **kw) == "dropped"  # type: ignore[arg-type]


@pytest.mark.parametrize("coupled", [True, False])
@pytest.mark.parametrize(
    ("alpha2", "chi"),
    [
        (1e-8, 1e-3),  # below the alpha threshold, above the chi threshold: not dropped
        (1e-5, 1e-8),  # above the alpha threshold, below the chi threshold: not dropped
        (1e-5, 1e-3),  # above both drop thresholds, below both freeze thresholds
    ],
)
def test_above_the_drop_pair_and_below_the_freeze_pair_is_frozen(
    alpha2: float, chi: float, coupled: bool
) -> None:
    """'dropped' needs BOTH numbers below their thresholds; failing either one leaves the mode frozen."""
    got = classify(
        _contrib(alpha2, chi), coupled=coupled, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI
    )
    assert got == "frozen"


@pytest.mark.parametrize("coupled", [True, False])
@pytest.mark.parametrize(("alpha2", "chi"), [(1e-3, 1e-8), (1e-8, 0.5), (1e-2, 0.8)])
def test_above_either_freeze_threshold_is_resolved(alpha2: float, chi: float, coupled: bool) -> None:
    got = classify(
        _contrib(alpha2, chi), coupled=coupled, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI
    )
    assert got == "resolved"


def test_the_drop_thresholds_are_the_plan_s_numbers_and_the_boundary_is_strict() -> None:
    """1e-6 and 1e-4, and the comparison is strict (a mode exactly AT a threshold is not dropped)."""
    assert (DROP_ALPHA_MAX, DROP_CHI_MAX_RAD) == (1e-6, 1e-4)
    kw = dict(coupled=True, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI)
    assert classify(_contrib(DROP_ALPHA_MAX, 0.0), **kw) == "frozen"  # type: ignore[arg-type]
    assert classify(_contrib(0.0, DROP_CHI_MAX_RAD), **kw) == "frozen"  # type: ignore[arg-type]
    assert classify(_contrib(DROP_ALPHA_MAX * (1 - 1e-12), 0.0), **kw) == "dropped"  # type: ignore[arg-type]
