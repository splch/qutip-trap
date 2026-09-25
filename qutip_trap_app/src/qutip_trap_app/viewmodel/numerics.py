"""The numerics panel of every level and the convergence badge (PLAN.md Section 14.5).

Every result carries its convergence report, and a failing check colours the result rather than hiding it. The badge is
text first (``pass`` / ``not checked`` / ``fail``) with its reasons, so colour never carries the meaning alone, and a check
that was not run is reported as not run: ``None`` never means passed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from qutip_trap_app import core
from qutip_trap_app.record import ConvergenceRecord, Record, ZoomTrace
from qutip_trap_app.resim import TruncationCheck
from qutip_trap_app.viewmodel.catalogue import Shown
from qutip_trap_app.viewmodel.dynamics import norm_deficit

POLICY_BOUNDARY_MAX: float = core.Numerics().boundary_population_max
"""Section 5.5's default threshold on the truncation-boundary population: the badge judges against the policy, and says so
when a run was made with a looser threshold of its own."""

BadgeStatus = Literal["pass", "not checked", "fail"]


@dataclass(frozen=True)
class Badge:
    status: BadgeStatus
    label: str
    reasons: tuple[str, ...]
    checks_run: tuple[str, ...]
    checks_not_run: tuple[str, ...]


@dataclass(frozen=True)
class NumericsPanel:
    level: Shown
    dimension: Shown
    caps: tuple[Shown, ...]
    mode_classes: tuple[Shown, ...]
    boundary: tuple[Shown, ...]
    margins: tuple[Shown, ...]
    integrator: Shown
    tolerances: Shown
    samples: Shown
    trajectories: Shown
    branches: Shown
    wall_time: Shown
    norm_deficit: Shown | None
    convergence: ConvergenceRecord | None
    tolerance_check: ConvergenceRecord | None
    truncation_check: TruncationCheck | None
    engine_note: str
    badge: Badge
    derivation_residual: Shown | None = None
    """The channel-derivation residual of a CHANNEL_REPLAY record (the bound verify deeper compares against)."""


def convergence_badge(
    record: Record,
    *,
    zoom: ZoomTrace | None = None,
    tolerance_check: ConvergenceRecord | None = None,
    truncation_check: TruncationCheck | None = None,
) -> Badge:
    reasons: list[str] = []
    run: list[str] = []
    not_run: list[str] = []
    boundary = zoom.trace.boundary_population if zoom is not None else record.diagnostics.boundary_population
    if boundary:
        run.append("boundary population against the Section 5.5 policy")
    else:
        # a replay record integrates nothing and reports no boundary population: the check did not run
        not_run.append("boundary population against the Section 5.5 policy (no integrated mode)")
    for m, v in sorted(boundary.items()):
        if math.isnan(v):
            reasons.append(
                f"mode {m}: the boundary population is not a number (the integration did not finish)"
            )
        elif not v <= POLICY_BOUNDARY_MAX:
            reasons.append(
                f"mode {m}: boundary population {v:.2e} exceeds the policy threshold {POLICY_BOUNDARY_MAX:.0e}"
            )
    if record.diagnostics.boundary_population_max != POLICY_BOUNDARY_MAX:
        reasons.append(
            f"the run was made with boundary_population_max = {record.diagnostics.boundary_population_max:.2g}, not the policy's "
            f"{POLICY_BOUNDARY_MAX:.0e}"
        )
    for m, margin in sorted(record.diagnostics.margin_reached.items()):
        if margin < 0:
            reasons.append(f"mode {m}: the cap sat {-margin} level(s) below the populated range")
    conv = record.diagnostics.convergence
    if conv is None and tolerance_check is None:
        not_run.append("tolerance tightening by ten (Section 5.5)")
    else:
        run.append("tolerance tightening by ten")
        for c in (conv, tolerance_check):
            if c is not None and not c.converged:
                reasons.append(
                    f"tightening the tolerances moves a population by {c.max_change:.2e} (threshold {c.tol:.0e})"
                )
    if truncation_check is None:
        not_run.append("caps raised by two (Section 9.9)")
    else:
        run.append("caps raised by two")
        if not truncation_check.converged:
            reasons.append(
                f"raising every cap by two moves a population by {truncation_check.max_change:.2e} (threshold {truncation_check.tol:.0e})"
            )
    if record.diagnostics.dropped_branch_weight > 0.0:
        run.append("dropped initial-mixture weight reported")
    if record.diagnostics.level == "CHANNEL_REPLAY":
        not_run.append(
            "verify deeper against GATE_LOCAL or JOINT_EXACT (a derived engine has no truncation of its own)"
        )
    if reasons:
        status: BadgeStatus = "fail"
        label = "fail: the result is not converged"
    elif not_run:
        status = "not checked"
        label = "not checked: " + "; ".join(not_run) + " not run"
    else:
        status = "pass"
        label = "pass: every convergence check run and passed"
    return Badge(
        status=status,
        label=label,
        reasons=tuple(reasons),
        checks_run=tuple(run),
        checks_not_run=tuple(not_run),
    )


def numerics_panel(
    record: Record,
    *,
    zoom: ZoomTrace | None = None,
    tolerance_check: ConvergenceRecord | None = None,
    truncation_check: TruncationCheck | None = None,
) -> NumericsPanel:
    d = record.diagnostics
    sp = record.space
    caps = tuple(
        Shown(
            "truncation_cap",
            t.d,
            f"mode {t.mode}: expected n in {t.expected_n_range}, eta_max {t.eta_max:.3g}",
        )
        for t in sp.resolved
    )
    if sp.enr_group is not None:
        caps += (Shown("truncation_cap", sp.enr_group[1], f"ENR group {sp.enr_group[0]}: N_exc"),)
    boundary_src = zoom.trace.boundary_population if zoom is not None else d.boundary_population
    boundary = tuple(
        Shown("boundary_population", v, f"mode {m}; policy threshold {POLICY_BOUNDARY_MAX:.0e}")
        for m, v in sorted(boundary_src.items())
    )
    margins = tuple(
        Shown(
            "margin",
            d.margin_reached.get(m, lv),
            f"mode {m}: declared margin {lv} level(s), smallest reached {d.margin_reached.get(m, lv)}",
        )
        for m, lv in sorted(d.margin_levels.items())
    )
    if d.level == "JOINT_EXACT":
        engine_note = f"{d.level}: every pulse on the {sp.dimension}-dimensional joint space {list(sp.dims)}"
    elif d.level == "GATE_LOCAL":
        engine_note = (
            "GATE_LOCAL: exact gate-local spaces up to dimension "
            f"{record.gate_local.largest_local_dimension if record.gate_local else '?'}, correlations traced out between "
            "steps (Section 5.4)"
        )
    else:
        engine_note = (
            "CHANNEL_REPLAY (derived, app-side): every gate applied as its extracted Section 6.8 channel; the joint space "
            f"{list(sp.dims)} is what a JOINT_EXACT run would use; verify deeper to compare"
        )
    if zoom is not None:
        engine_note += f"; this pulse re-simulated with {zoom.n_store} stored points per segment ({', '.join(zoom.integrators + (zoom.method,))})"
    wall = zoom.wall_time_s if zoom is not None else d.wall_time_s
    residual = (
        None
        if record.replay is None
        else Shown(
            "derivation_residual",
            record.replay.residual_total,
            ", ".join(f"{k} {v:.2e}" for k, v in record.replay.residual_terms.items()),
        )
    )
    return NumericsPanel(
        derivation_residual=residual,
        level=Shown("fidelity_level", d.level),
        dimension=Shown("dimension", sp.dimension),
        caps=caps,
        mode_classes=tuple(Shown("mode_class", c, f"mode {m}") for m, c in sorted(sp.mode_class.items())),
        boundary=boundary,
        margins=margins,
        integrator=Shown("integrator", d.integrator),
        tolerances=Shown("tolerance", f"atol {d.tolerances[0]:g}, rtol {d.tolerances[1]:g}"),
        samples=Shown("samples", d.samples),
        trajectories=Shown("trajectories", d.trajectories),
        branches=Shown("branches", d.branches),
        wall_time=Shown("wall_time", wall),
        norm_deficit=None if zoom is None else norm_deficit(zoom.trace, "at the zoomed step's end"),
        convergence=d.convergence,
        tolerance_check=tolerance_check,
        truncation_check=truncation_check,
        engine_note=engine_note,
        badge=convergence_badge(
            record, zoom=zoom, tolerance_check=tolerance_check, truncation_check=truncation_check
        ),
    )
