"""The numerics strip under every screen and its convergence badge: a word first (pass, not checked, fail) with its
reasons; a check that was not run never reads as passed."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qutip_trap_app import core
from qutip_trap_app.record import Recheck, Record, ZoomTrace
from qutip_trap_app.viewmodel.shown import Shown

POLICY_BOUNDARY_MAX: float = core.Truncation().boundary_population_max
"""The default threshold on the population at a mode's cap; the badge judges every run against it."""

BadgeStatus = Literal["pass", "not checked", "fail"]


@dataclass(frozen=True)
class Badge:
    status: BadgeStatus
    reasons: tuple[str, ...]
    checks_run: tuple[str, ...]
    checks_not_run: tuple[str, ...]


@dataclass(frozen=True)
class NumericsPanel:
    summary: str
    """The level and the dimension, for the collapsed strip."""
    engine_note: str
    tiles: tuple[Shown, ...]
    caps: tuple[Shown, ...]
    boundary: tuple[Shown, ...]
    margins: tuple[Shown, ...]
    mode_classes: tuple[Shown, ...]
    norm_deficit: Shown | None
    rechecks: tuple[Recheck, ...]
    badge: Badge


def convergence_badge(
    record: Record, *, zoom: ZoomTrace | None = None, rechecks: Sequence[Recheck] = ()
) -> Badge:
    reasons: list[str] = []
    run: list[str] = []
    not_run: list[str] = []
    d = record.diagnostics
    boundary = zoom.trace.boundary_population if zoom is not None else d.boundary_population
    if boundary:
        run.append("population at every cap")
    else:
        not_run.append("population at every cap (no integrated mode)")
    for m, v in sorted(boundary.items()):
        if math.isnan(v):
            reasons.append(
                f"mode {m}: the population at the cap is not a number (the integration did not finish)"
            )
        elif not v <= POLICY_BOUNDARY_MAX:
            reasons.append(
                f"mode {m}: population {v:.2e} at the cap, above the threshold {POLICY_BOUNDARY_MAX:.0e}"
            )
    if d.boundary_population_max != POLICY_BOUNDARY_MAX:
        reasons.append(
            f"the run allowed {d.boundary_population_max:.2g} at the cap, not the default {POLICY_BOUNDARY_MAX:.0e}"
        )
    for m, margin in sorted(d.margin_reached.items()):
        if margin < 0:
            reasons.append(f"mode {m}: the cap sat {-margin} level(s) below the populated range")
    if rechecks:
        for c in rechecks:
            run.append(c.what)
            if not c.converged:
                reasons.append(f"{c.what} moves an observable by {c.max_change:.2e} (threshold {c.tol:.0e})")
    else:
        not_run.append("tolerances and caps re-checked on a zoomed step")
    status: BadgeStatus = "fail" if reasons else "not checked" if not_run else "pass"
    return Badge(status, tuple(reasons), tuple(run), tuple(not_run))


def numerics_panel(
    record: Record, *, zoom: ZoomTrace | None = None, rechecks: Sequence[Recheck] = ()
) -> NumericsPanel:
    d = record.diagnostics
    sp = record.space
    if d.level == "JOINT_EXACT":
        note = f"JOINT_EXACT: every pulse on the {sp.dimension}-dimensional joint space {list(sp.dims)}"
    else:
        largest = record.gate_local.largest_local_dimension if record.gate_local is not None else "?"
        note = f"GATE_LOCAL: each gate on its own space (up to dimension {largest}), the motion traced out between gates"
    if zoom is not None:
        engine = ", ".join((*zoom.integrators, zoom.method))
        note += f"; this step re-simulated at {zoom.n_store} points per segment ({engine})"
    caps = tuple(Shown(f"Cap, mode {m}", c, detail="Fock levels kept") for m, c in sorted(sp.caps.items()))
    if sp.enr_group is not None:
        caps += (Shown(f"Excitation cap, modes {list(sp.enr_group[0])}", sp.enr_group[1]),)
    boundary = zoom.trace.boundary_population if zoom is not None else d.boundary_population
    tr = zoom.trace if zoom is not None else None
    return NumericsPanel(
        summary=f"{d.level}, dimension {sp.dimension}",
        engine_note=note,
        tiles=(
            Shown("Integrator", d.integrator),
            Shown("Tolerances", f"atol {d.tolerances[0]:g}, rtol {d.tolerances[1]:g}"),
            Shown("Dynamical samples", d.samples),
            Shown("Trajectories", d.trajectories),
            Shown("Branches of the initial mixture", d.branches),
            Shown("Wall time", zoom.wall_time_s if zoom is not None else d.wall_time_s, "s"),
        ),
        caps=caps,
        boundary=tuple(Shown(f"Population at the cap, mode {m}", v) for m, v in sorted(boundary.items())),
        margins=tuple(
            Shown(
                f"Margin, mode {m}",
                d.margin_reached.get(m, lv),
                detail=f"levels above the populated range; {lv} declared",
            )
            for m, lv in sorted(d.margin_levels.items())
        ),
        mode_classes=tuple(Shown(f"Mode {m}", c) for m, c in sorted(sp.mode_class.items())),
        norm_deficit=None
        if tr is None or tr.reduced_internal.size == 0
        else Shown(
            "Norm deficit",
            1.0 - float(np.real(np.trace(tr.reduced_internal[-1]))),
            detail="at the step's end",
        ),
        rechecks=tuple(rechecks),
        badge=convergence_badge(record, zoom=zoom, rechecks=rechecks),
    )
