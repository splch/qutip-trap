"""Section 9.11 row "Convergence badge": an under-truncated run turns the badge red and fails the halving test; the same
run at the policy's caps passes."""

from __future__ import annotations

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.resim import tolerance_recheck, truncation_recheck
from qutip_trap_app.viewmodel.numerics import POLICY_BOUNDARY_MAX, convergence_badge, numerics_panel


def _ms_step(record: Record) -> int:
    return next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))


def test_policy_run_passes(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    assert all(v < POLICY_BOUNDARY_MAX for v in record.diagnostics.boundary_population.values())
    badge = convergence_badge(record)
    assert badge.status == "not checked", "no check was run yet: None never means passed"
    assert not badge.reasons
    step = _ms_step(record)
    record, tol = tolerance_recheck(record, live, step)
    assert tol.converged, tol.changes
    record, cap = truncation_recheck(record, live, step)
    assert cap.converged, cap.changes
    assert cap.grown_caps == {m: d + 2 for m, d in cap.caps.items()}
    panel = numerics_panel(record, tolerance_check=tol, truncation_check=cap)
    assert panel.badge.status == "pass", panel.badge


def test_under_truncated_run_fails(
    under_truncated: tuple[Record, LiveRun], bell: tuple[Record, LiveRun]
) -> None:
    record, live = under_truncated
    policy_record, _ = bell
    assert max(record.diagnostics.boundary_population.values()) > POLICY_BOUNDARY_MAX
    badge = convergence_badge(record)
    assert badge.status == "fail" and any("exceeds the policy threshold" in r for r in badge.reasons)
    assert any("not the policy" in r for r in badge.reasons), "the loosened threshold of the run is named"
    step = _ms_step(record)
    record, cap = truncation_recheck(record, live, step)
    assert not cap.converged and cap.max_change > 1e-3, cap.changes
    panel = numerics_panel(record, truncation_check=cap)
    assert panel.badge.status == "fail"
    # the register fidelity of the under-truncated run is below the policy run's (deterministic, no shot noise in it); its
    # histogram differs from the policy run's by less than the 200-shot error bars, which is the lesson the badge teaches:
    # a wrong simulation can still look right
    assert (
        record.results.register_fidelity is not None and policy_record.results.register_fidelity is not None
    )
    assert record.results.register_fidelity < policy_record.results.register_fidelity
    assert record.space.dims == (2, 2, 4, 4) and policy_record.space.dims != record.space.dims
