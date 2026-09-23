"""The convergence badge: a run at the default caps passes once both re-checks ran; an under-truncated run fails the
cap check and turns the badge red; a check not run is never reported as passed."""

from __future__ import annotations

import dataclasses
import math

from fixtures import BellResimulated, ms_step

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.resim import CAP_RAISE, truncation_recheck
from qutip_trap_app.viewmodel.numerics import POLICY_BOUNDARY_MAX, convergence_badge, numerics_panel


def test_a_run_at_the_default_caps_passes(bell_resimulated: BellResimulated) -> None:
    r = bell_resimulated
    assert all(v < POLICY_BOUNDARY_MAX for v in r.record.diagnostics.boundary_population.values())
    before = convergence_badge(r.record)
    assert before.status == "not checked" and not before.reasons, "no re-check ran on this badge: not a pass"
    assert r.tolerance.converged, r.tolerance.changes
    assert r.truncation.converged, r.truncation.changes
    assert r.truncation.after == {k: v + CAP_RAISE for k, v in r.truncation.before.items()}
    assert r.tolerance.after == {k: v / 10.0 for k, v in r.tolerance.before.items()}
    panel = numerics_panel(r.record, zoom=r.zoom, rechecks=(r.tolerance, r.truncation))
    assert panel.badge.status == "pass", panel.badge


def test_an_under_truncated_run_fails(
    under_truncated: tuple[Record, LiveRun], bell: tuple[Record, LiveRun]
) -> None:
    record, live = under_truncated
    policy, _ = bell
    assert max(record.diagnostics.boundary_population.values()) > POLICY_BOUNDARY_MAX
    badge = convergence_badge(record)
    assert badge.status == "fail" and any("above the threshold" in r for r in badge.reasons)
    assert any("not the default" in r for r in badge.reasons), "the run's own loosened threshold is named"
    record, cap = truncation_recheck(record, live, ms_step(record))
    assert not cap.converged and cap.max_change > 1e-3, cap.changes
    assert numerics_panel(record, rechecks=(cap,)).badge.status == "fail"
    # a wrong simulation can look right: its histogram agrees with the policy run's within the error bars, its register
    # fidelity (no shot noise in it) does not
    assert record.results.register_fidelity is not None and policy.results.register_fidelity is not None
    assert record.results.register_fidelity < policy.results.register_fidelity
    assert record.space.dims == (2, 2, 4, 4) and policy.space.dims != record.space.dims


def test_the_badge_never_passes_what_it_did_not_check(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    assert convergence_badge(record).checks_run[0].startswith("population at every cap")
    blind = dataclasses.replace(
        record, diagnostics=dataclasses.replace(record.diagnostics, boundary_population={})
    )
    badge = convergence_badge(blind)
    assert badge.status == "not checked" and any("population at every cap" in c for c in badge.checks_not_run)
    broken = dataclasses.replace(
        record, diagnostics=dataclasses.replace(record.diagnostics, boundary_population={2: math.nan})
    )
    badge = convergence_badge(broken)
    assert badge.status == "fail" and any("not a number" in r for r in badge.reasons)
