"""The Learn view (DESIGN.md Sections 1, 3, 5): the prior-knowledge setting, the retention target and its review gap, the
six-stop Bell tour, and the review tray of due prompts. The settings and the mastery log are changed through the session,
which keeps them on the device, so the tray can come due across launches."""

from __future__ import annotations

import time
from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.viewmodel.learn import BELL_TOUR, CONCEPTS, due_prompts, review_gap_days
from qutip_trap_app.views.common import card, chip
from qutip_trap_app.views.state import Session, Store

KNOWLEDGE_LABELS: dict[str, str] = {
    "newcomer": "New to quantum computing",
    "circuits": "I know circuits, not the hardware",
    "physicist": "Physicist; I know the hardware",
    "unknown": "Not saying (the app assists)",
}


def _fill_route(route: str, key: str | None, gate: str, pulse: str) -> str:
    return (
        route.replace("{id}", key or "-")
        .replace("{gate}", gate)
        .replace("{pulse}", pulse)
        .replace("{sample}", "0")
    )


@ft.component
def LearnPage(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    learner = store.learner
    page = ft.context.page
    record = store.record()
    key = record.key() if record is not None else None
    ms_gate = next((g.gate_id for g in record.schedule.gates), "ms[2]") if record is not None else "ms[2]"
    ms_pulse = "0"
    if record is not None:
        st = next((s for s in record.schedule.steps if s.gate_id == ms_gate), None)
        if st is not None and st.pulse_indices:
            ms_pulse = str(st.pulse_indices[0])
    lo, hi = review_gap_days(learner.retention_days)

    def set_knowledge(e: Any) -> None:
        session.set_learner(
            knowledge=next(iter(e.control.selected)),
            asked=True,
            depth_override=None,
            explain_open=None,
        )

    def set_retention(e: Any) -> None:
        session.set_learner(retention_days=float(e.control.value))

    knowledge_card = card(
        "Who is learning?",
        ft.Column(
            [
                ft.Text(
                    "Assistance follows what you already know: full guidance that fades for a newcomer, problems first for a physicist. Change it any time.",
                    size=13,
                ),
                ft.SegmentedButton(
                    segments=[
                        ft.Segment(value=k, label=ft.Text(v, size=12)) for k, v in KNOWLEDGE_LABELS.items()
                    ],
                    selected=[learner.knowledge],
                    on_change=set_knowledge,
                ),
                ft.Text(
                    f"explanations open by default: {'yes' if learner.explain_is_open(0) else 'no'}; starting depth: {learner.depth(0)}; worked example first: {'yes' if learner.plan(0).worked_example_first else 'no'}",
                    size=12,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            spacing=8,
        ),
    )
    retention_card = card(
        "How long must it stick?",
        ft.Column(
            [
                ft.Text(
                    f"Target: {learner.retention_days:.0f} days. The first review is due {lo:.0f} to {hi:.0f} days after a first exposure (the optimal gap is a declining share of the target; err long).",
                    size=13,
                ),
                ft.Slider(
                    min=7,
                    max=365,
                    divisions=51,
                    value=learner.retention_days,
                    label="{value} days",
                    on_change_end=set_retention,
                ),
                ft.Text(
                    "Evidence: Cepeda et al. 2008 and the IES 2007 practice guide; the app schedules re-asks from this and never notifies you.",
                    size=11,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            spacing=6,
        ),
    )
    stops = []
    for stop in BELL_TOUR:
        route = _fill_route(stop.route, key, ms_gate, ms_pulse)
        enabled = key is not None or stop.route.startswith("/device")
        stops.append(
            ft.ListTile(
                leading=ft.Text(str(stop.index), weight=ft.FontWeight.W_700),
                title=ft.Text(stop.title, size=14),
                subtitle=ft.Text(f"look for: {stop.look_for}", size=12),
                trailing=ft.Row(
                    [chip(i, index) for i in CONCEPTS[stop.concept_id].ledger_ids[:2]], tight=True, spacing=2
                ),
                on_click=(lambda e, r=route: page.navigate(r)) if enabled else None,
                disabled=not enabled,
                dense=True,
            )
        )
    tour_note = (
        "run the Bell job on Level 0 first (Run is one click) to walk the stops"
        if key is None
        else "six clicks from a histogram bar to a matrix element"
    )
    tour_card = card(
        "The tour: a Bell state in six stops",
        ft.Column(
            stops + [ft.Text(tour_note, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)], spacing=0
        ),
        subtitle="the worked example; each stop ends in a prompt",
    )
    now_days = time.time() / 86400.0
    due = due_prompts(learner.log, now_days)
    tray_body = (
        ft.Text("nothing is due: prompts return here one review gap after you last met them", size=13)
        if not due
        else ft.Column([ft.Text(f"{p.question}", size=13) for p in due], spacing=4)
    )
    attempts = len(learner.log.attempts)
    tray_card = card(
        "Review tray",
        ft.Column(
            [
                tray_body,
                ft.Text(
                    f"{attempts} prompt answers logged and kept on this device; in-session accuracy is shown as such and never called learning",
                    size=11,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            spacing=6,
        ),
        subtitle="spaced re-asks, unaided",
    )
    return ft.Column(
        [
            ft.Text("Learn", theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
            ft.Text(
                "Set who you are and how long it must stick; take the tour; review what is due.",
                size=14,
                color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.ResponsiveRow(
                [
                    ft.Column([knowledge_card, retention_card], col={"xs": 12, "lg": 6}, spacing=10),
                    ft.Column([tour_card, tray_card], col={"xs": 12, "lg": 6}, spacing=10),
                ],
                vertical_alignment=ft.CrossAxisAlignment.START,
                spacing=12,
                run_spacing=12,
            ),
        ],
        spacing=12,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )


__all__ = ["KNOWLEDGE_LABELS", "LearnPage"]
