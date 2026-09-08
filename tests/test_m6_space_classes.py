"""Section 5.2's three mode classes as a unit (PLAN.md lines 804-847; M6 audit item P0-2).

The criterion is the contribution pair (|alpha_m|^2 (2 nbar_m + 1), |chi_m|) of the played entangling waveforms alone; the
coupling test only decides the modes no entangling gate touches. Before the fix the drop test was intersected with ``coupled``,
which made the ``dropped`` return unreachable for every mode a gate touches (a contribution only ever exists for a coupled
mode), so ``SpaceSelection.dropped_contribution`` - the quantity Section 11.3 item 2 asks the report to state, because nothing
absorbs a dropped mode's loss - was identically (0.0, 0.0)."""

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
def test_no_entangling_contribution_is_frozen_when_a_drive_couples_and_dropped_otherwise(
    coupled: bool,
) -> None:
    """The one branch ``coupled`` decides: a mode only single-qubit carrier pulses touch is frozen so that its Debye-Waller
    factor is applied; a mode no drive couples to at all is dropped."""
    got = classify(None, coupled=coupled, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI)
    assert got == ("frozen" if coupled else "dropped")


@pytest.mark.parametrize("coupled", [True, False])
def test_below_the_drop_pair_is_dropped_whatever_couples_to_it(coupled: bool) -> None:
    """Section 5.2 line 822: dropped when |alpha_m|^2 (2 n_m + 1) < 1e-6 AND |chi_m| < 1e-4, regardless of the coupling - and,
    since M9, only when the mode's Debye-Waller spread is also negligible (``DW_SPREAD_DROP_MAX``, see the test below).

    The three-ion GHZ tilt mode's loop pair is exactly this case (|alpha|^2 (2n+1) = 3.5e-33, chi = 0) and used to come back
    frozen for the wrong reason (the drop test was intersected with ``coupled``); it is frozen again for the RIGHT one, since
    its eta = 0.0809 at nbar = 0.05 carries a 1.5e-3 rad Debye-Waller spread. A mode whose spread is negligible - eta = 0.008
    at nbar = 0.05 gives 1.5e-5 rad - is genuinely dropped."""
    got = classify(
        _contrib(3.5e-33, 0.0, 1.5e-5),
        coupled=coupled,
        freeze_alpha_max=FREEZE_ALPHA,
        freeze_chi_max_rad=FREEZE_CHI,
    )
    assert got == "dropped"


@pytest.mark.parametrize("coupled", [True, False])
def test_a_dropped_loop_pair_with_a_large_debye_waller_spread_is_frozen(coupled: bool) -> None:
    """The third drop condition (M6 finding, M9 fix; ledger ``conv.drop_test_debye_waller_spread``): the loop pair reads the
    Section 4.4.3 integrals alone, so a pulse that CLOSES a mode's loop leaves the pair at ~1e-33 however strongly the mode
    couples - and dropping the mode then takes its exact Debye-Waller factor out of every carrier pulse, which is what
    Section 5.2 keeps the frozen class for. The calibration absorbs the factor's mean, not its shot-to-shot spread
    eta^2 sqrt(nbar (nbar + 1)).

    Numbers on the three-ion fixture's tilt mode (eta = 0.0809): 1.500e-3 rad at nbar = 0.05, 9.256e-3 at nbar = 1 and
    6.864e-2 at nbar = 10, i.e. sin^2(delta theta/2) = 5.6e-7, 2.1e-5 and 1.2e-3 of infidelity on a pi pulse."""
    kw = dict(coupled=coupled, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI)
    for spread in (1.500e-3, 9.256e-3, 6.864e-2):
        assert classify(_contrib(3.5e-33, 0.0, spread), **kw) == "frozen", spread  # type: ignore[arg-type]
    # the threshold is 3e-4 rad and the comparison is strict, like the other two rows
    assert DW_SPREAD_DROP_MAX == 3e-4
    assert classify(_contrib(0.0, 0.0, DW_SPREAD_DROP_MAX), **kw) == "frozen"  # type: ignore[arg-type]
    assert (
        classify(_contrib(0.0, 0.0, DW_SPREAD_DROP_MAX * (1 - 1e-12)), **kw) == "dropped"  # type: ignore[arg-type]
    )
    # the plan's own 9.17 row is a small-eta mode and stays dropped: eta = 1e-3 at nbar = 0.05 is a 2.3e-7 rad spread
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
    """'dropped' needs BOTH numbers below their thresholds; failing either one leaves the mode frozen (its chi is reported and
    absorbed by the calibration, its Fock state enters through the exact Debye-Waller factor)."""
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
    """1e-6 and 1e-4 verbatim from Section 5.2, and the comparison is strict (a mode exactly AT a threshold is not dropped)."""
    assert (DROP_ALPHA_MAX, DROP_CHI_MAX_RAD) == (1e-6, 1e-4)
    kw = dict(coupled=True, freeze_alpha_max=FREEZE_ALPHA, freeze_chi_max_rad=FREEZE_CHI)
    # the third condition is separate and tested above; here its value is negligible so the two plan rows decide alone
    assert classify(_contrib(DROP_ALPHA_MAX, 0.0), **kw) == "frozen"  # type: ignore[arg-type]
    assert classify(_contrib(0.0, DROP_CHI_MAX_RAD), **kw) == "frozen"  # type: ignore[arg-type]
    assert classify(_contrib(DROP_ALPHA_MAX * (1 - 1e-12), 0.0), **kw) == "dropped"  # type: ignore[arg-type]


# ---- Section 7.7's physical RZ (M6 audit item 19) ----------------------------------------------------------------------


def test_physical_rz_is_two_gpi_pulses_and_is_exact() -> None:
    """Section 7.7: "Physical RZ in two pulses: R(pi, x) R(pi, x - theta/2)". The identity is EXACT, not approximate (max
    deviation 2.8e-16 over four angles and three axes), and the axis x is a free convention."""
    import math

    import numpy as np

    from qutip_trap.control import native
    from qutip_trap.control.compiler import physical_rz

    for theta in (0.1, 1.3, -2.0, math.pi):
        for axis in (0.0, 0.7, -1.2):
            ops = physical_rz(theta, 0, axis_rad=axis)
            assert [op.name for op in ops] == ["gpi", "gpi"]
            got = native.gpi(ops[1].params[0]) @ native.gpi(ops[0].params[0])
            assert np.allclose(got, native.rz(theta), atol=1e-12), (theta, axis)


def test_decompose_single_qubit_can_emit_physical_z_instead_of_a_frame() -> None:
    """``physical_z=True`` leaves no ``rz`` in the block, and the block still verifies against its target (the compiler's
    own 1e-9 check runs either way)."""
    import numpy as np

    from qutip_trap.control.compiler import decompose_single_qubit

    rng = np.random.default_rng(11)
    for _ in range(8):
        z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        q, r = np.linalg.qr(z)
        u = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
        virtual = decompose_single_qubit(u, 0)
        physical = decompose_single_qubit(u, 0, physical_z=True)
        assert not any(op.name == "rz" for op in physical)
        assert all(op.name in ("gpi", "gpi2") for op in physical)
        # each virtual rz becomes two GPi pulses; nothing else changes
        n_rz = sum(1 for op in virtual if op.name == "rz")
        assert len(physical) == len(virtual) + n_rz


# ---- Section 7.7's cost model (M6 audit item 20) -----------------------------------------------------------------------


def test_cost_of_is_section_7_7_s_cost_model() -> None:
    """|theta| tau_1q/pi per single-qubit pulse with error |sin theta| eps, tau_2q per entangling gate with |sin 2 chi| E,
    and circuit fidelity prod(1 - e_i) - a PRODUCT, not 1 - sum (they differ at second order: 0.9975011 against 0.9975)."""
    import math

    import pytest as _pytest

    from qutip_trap.control.compiler import Circuit, Operation, compile_to_native, cost_of

    native_bell = compile_to_native(
        Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
    )
    cost = cost_of(native_bell, tau_1q_s=5e-6, tau_2q_s=100e-6, eps_1q=1e-4, eps_2q=2e-3)
    assert (cost.n_single, cost.n_entangling) == (5, 1)
    # five GPi2 pulses of area pi/2 plus one 100 us entangling gate
    assert cost.duration_s == _pytest.approx(5 * (math.pi / 2) * 5e-6 / math.pi + 100e-6, rel=1e-12)
    assert cost.errors == _pytest.approx((1e-4, 1e-4, 2e-3, 1e-4, 1e-4, 1e-4), rel=1e-12)
    assert cost.fidelity == _pytest.approx(0.99750109979, rel=1e-9)
    assert 1.0 - cost.fidelity < sum(cost.errors), "the product is above 1 - sum e_i at second order"
    # a virtual rz costs nothing, and a partial angle costs |sin theta|
    with_rz = Circuit(1, (Operation("rz", (0,), (0.7,)), Operation("gpi", (0,), (0.0,))), (0,))
    only_gpi = cost_of(with_rz, tau_1q_s=5e-6, tau_2q_s=1e-6, eps_1q=1e-3, eps_2q=1e-3)
    assert only_gpi.n_single == 1 and only_gpi.duration_s == _pytest.approx(5e-6, rel=1e-12)
    assert only_gpi.errors == _pytest.approx((abs(math.sin(math.pi)) * 1e-3,), abs=1e-18)
    partial = Circuit(2, (Operation("ms", (0, 1), (0.0, 0.0, 0.3)),), (0, 1))
    assert cost_of(partial, tau_1q_s=1e-6, tau_2q_s=50e-6, eps_1q=0.0, eps_2q=1e-2).errors == _pytest.approx(
        (abs(math.sin(0.3)) * 1e-2,), rel=1e-12
    )
