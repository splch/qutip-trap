"""The Learn view (DESIGN.md Sections 1, 3, 5 and 10): a ladder of activities under one compact settings row.

The tour (the worked example), the faded three-ion GHZ exercise (the same six stops, annotations withheld until asked), the
free exercise, the discrimination drills generated from the current record, the review tray of due prompts (asked unaided),
the published experiments, and the progress log with in-session and delayed unaided accuracy labelled apart. Every activity
is a route (``/learn/{tab}``), so the browser's history works in the served mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, NamedTuple

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.drills import Drill, drills_for
from qutip_trap_app.viewmodel.learn import (
    BELL_TOUR,
    CONCEPTS,
    FREE_EXERCISE,
    Attempt,
    PriorKnowledge,
    TourStop,
    now_days,
    review_gap_days,
)
from qutip_trap_app.views import theme
from qutip_trap_app.views.common import (
    MUTED,
    PromptView,
    card,
    chip,
    data_table,
    hint,
    inset,
    level_header,
    page_tabs,
    status_line,
)
from qutip_trap_app.views.presets import PresetList, PresetPage
from qutip_trap_app.views.state import Session, Store, learner_document


class KnowledgeOption(NamedTuple):
    label: str
    short: str
    description: str
    """What the answer to the first-launch question changes, in one short line."""


KNOWLEDGE: dict[PriorKnowledge, KnowledgeOption] = {
    "newcomer": KnowledgeOption(
        "New to quantum computing", "New", "Plain words first, explanations open, the worked example first"
    ),
    "circuits": KnowledgeOption(
        "I know circuits, not the hardware", "Circuits", "Circuit levels brief, the hardware levels explained"
    ),
    "physicist": KnowledgeOption(
        "Physicist; I know the hardware", "Physicist", "Symbols first, explanations closed, every table open"
    ),
    "unknown": KnowledgeOption(
        "Not saying (the app assists)", "Not saying", "The app assists, as for a newcomer"
    ),
}


def _fill_route(route: str, key: str | None, gate: str, pulse: str) -> str:
    return (
        route.replace("{id}", key or "-")
        .replace("{gate}", gate)
        .replace("{pulse}", pulse)
        .replace("{sample}", "0")
    )


def _entangling_ids(record: Record | None) -> tuple[str, str]:
    """(gate id, first pulse index) of the record's first entangling gate, for the tour's routes."""
    if record is None:
        return "ms[2]", "0"
    gate = next((g.gate_id for g in record.schedule.gates), "ms[2]")
    st = next((s for s in record.schedule.steps if s.gate_id == gate), None)
    return gate, str(st.pulse_indices[0]) if st is not None and st.pulse_indices else "0"


# ---- the settings row ------------------------------------------------------------------------------------------------


@ft.component
def SettingsRow(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    learner = store.learner
    lo, hi = review_gap_days(learner.retention_days)

    def set_knowledge(e: Any) -> None:
        session.set_learner(
            knowledge=next(iter(e.control.selected)), asked=True, depth_override=None, explain_open=None
        )

    who = ft.Row(
        [
            ft.Text("Who", size=theme.SIZE_SMALL, color=MUTED),
            ft.SegmentedButton(
                segments=[
                    ft.Segment(value=k, label=ft.Text(o.short, size=theme.SIZE_SMALL), tooltip=o.label)
                    for k, o in KNOWLEDGE.items()
                ],
                selected=[learner.knowledge],
                on_change=set_knowledge,
            ),
        ],
        spacing=10,
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    keep = ft.Row(
        [
            ft.Text(f"Keep it for {learner.retention_days:.0f} days", size=theme.SIZE_SMALL, color=MUTED),
            ft.Slider(
                min=7,
                max=365,
                divisions=51,
                value=learner.retention_days,
                label="{value} days",
                on_change_end=lambda e: session.set_learner(retention_days=float(e.control.value)),
                width=220,
                tooltip=(
                    f"the first review is due {lo:.0f} to {hi:.0f} days after a first exposure: the optimal gap is a declining "
                    "share of the target (Cepeda et al. 2008), and overshooting costs little"
                ),
            ),
        ],
        spacing=4,
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    return ft.Row(
        [who, keep], wrap=True, spacing=24, run_spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER
    )


# ---- the activities --------------------------------------------------------------------------------------------------


def _number(k: int) -> ft.Control:
    return ft.Container(
        content=ft.Text(str(k), weight=ft.FontWeight.W_700, size=theme.SIZE_SMALL),
        width=28,
        height=28,
        alignment=ft.Alignment.CENTER,
        border_radius=ft.BorderRadius.all(14),
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
    )


def _stop_tile(
    k: int,
    stop: TourStop,
    index: ProvenanceIndex,
    route: str,
    enabled: bool,
    *,
    look_for: str | None,
    on_reveal: Any = None,
) -> ft.Control:
    page = ft.context.page
    trailing: list[ft.Control] = [chip(i, index) for i in CONCEPTS[stop.concept_id].ledger_ids[:2]]
    if on_reveal is not None:
        trailing.insert(
            0,
            ft.IconButton(
                icon=ft.Icons.LIGHTBULB_OUTLINE if look_for is None else ft.Icons.LIGHTBULB,
                icon_size=16,
                tooltip="show what to look for" if look_for is None else "the annotation, revealed",
                on_click=on_reveal,
            ),
        )
    return ft.ListTile(
        leading=_number(k),
        title=ft.Text(stop.title, size=theme.SIZE_BODY, weight=ft.FontWeight.W_500),
        subtitle=ft.Text(f"look for: {look_for}", size=theme.SIZE_SMALL, color=MUTED) if look_for else None,
        trailing=ft.Row(trailing, tight=True, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        on_click=(lambda e, r=route: page.navigate(r)) if enabled else None,
        disabled=not enabled,
        key=f"stop:{k}",
    )


@ft.component
def TourActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    record = store.record()
    key = record.key() if record is not None else None
    gate, pulse = _entangling_ids(record)
    stops = [
        _stop_tile(
            k,
            stop,
            index,
            _fill_route(stop.route, key, gate, pulse),
            key is not None or stop.route.startswith("/device"),
            look_for=stop.look_for,
        )
        for k, stop in enumerate(BELL_TOUR, 1)
    ]

    def start(_e: Any) -> None:
        if key is None:
            session.load_bell_example()
            page.navigate("/")
        else:
            page.navigate(f"/job/{key}")

    return card(
        "The tour: a Bell state in six stops",
        ft.Column(stops + [hint(store, 0, "target_vs_simulated")], spacing=2),
        why=lambda e: session.select_concept(0, "target_vs_simulated"),
        info="the worked example (full guidance first, then faded): six clicks from a histogram bar to a matrix element, each stop ending in a prompt",
        actions=[
            ft.FilledButton(
                content=ft.Text("Start the tour" if key is not None else "Run the Bell job first"),
                icon=ft.Icons.PLAY_ARROW,
                on_click=start,
                key="start-tour",
            )
        ],
        key="tour",
    )


@ft.component
def GhzActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    record = store.record()
    is_ghz = record is not None and record.n_qubits == 3
    key = record.key() if is_ghz and record is not None else None
    gate, pulse = _entangling_ids(record if is_ghz else None)

    def reveal(k: int) -> Any:
        def _do(_e: Any) -> None:
            store.revealed_stops = store.revealed_stops ^ {k}

        return _do

    stops = [
        _stop_tile(
            k,
            stop,
            index,
            _fill_route(stop.route, key, gate, pulse),
            key is not None or stop.route.startswith("/device"),
            look_for=stop.look_for if k in store.revealed_stops else None,
            on_reveal=reveal(k),
        )
        for k, stop in enumerate(BELL_TOUR, 1)
    ]

    def load(_e: Any) -> None:
        session.load_circuit_preset("ghz_three")
        page.navigate("/")

    return card(
        "Three ions, one GHZ state: the same six stops, unannotated",
        ft.Column(
            stops + [status_line("the annotations are withheld; the bulb reveals one when you ask")],
            spacing=2,
        ),
        why=lambda e: session.select_concept(1, "entanglement_by_ms"),
        info="the faded version of the worked example: the guidance is withheld until asked (DESIGN.md Section 3); the job is the three-ion GHZ circuit preset of Section 9.6, which takes minutes at the full engine",
        actions=[
            ft.FilledButton(
                content=ft.Text("Load the three-ion GHZ job" if not is_ghz else "The GHZ job is loaded"),
                icon=ft.Icons.PLAY_ARROW,
                on_click=load,
                disabled=is_ghz,
                key="load-ghz",
            )
        ],
        key="ghz",
    )


@ft.component
def FreeActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    page = ft.context.page
    steps: list[ft.Control] = [
        ft.ListTile(leading=_number(k), title=ft.Text(text, size=theme.SIZE_BODY))
        for k, text in enumerate(FREE_EXERCISE, 1)
    ]
    return card(
        "Your own circuit: build, predict, run, explain",
        ft.Column(steps, spacing=2),
        why=lambda e: session.select_concept(0, "histogram"),
        info="the free version of the exercise; the delayed, unaided task of DESIGN.md Section 1 is this, one review gap later, with the drawer closed",
        actions=[
            ft.FilledButton(
                content=ft.Text("Open the builder"), icon=ft.Icons.EDIT, on_click=lambda e: page.navigate("/")
            )
        ],
        key="free",
    )


@ft.component
def DrillView(store: Store, session: Session, drill: Drill) -> ft.Control:
    ft.use_state(store)
    answer = store.drill_answers.get(drill.id, "")
    checked = answer.startswith("checked:")
    given = answer.removeprefix("checked:")

    def pick(e: Any) -> None:
        store.drill_answers = {**store.drill_answers, drill.id: str(e.control.value)}

    def check(_e: Any) -> None:
        store.drill_answers = {**store.drill_answers, drill.id: f"checked:{given}"}
        # unaided: no explanation stands open beside a drill
        session.record_attempt(
            Attempt(drill.concept_id, f"drill:{drill.id}", now_days(), given == drill.answer, unaided=True)
        )

    feedback: ft.Control = ft.Container()
    if checked:
        feedback = ft.Text(
            "that matches the record"
            if given == drill.answer
            else f"the record says: {drill.answer}; check {drill.where}",
            size=theme.SIZE_SMALL,
            color=MUTED,
        )
    return inset(
        [
            ft.Text(drill.question, size=theme.SIZE_SMALL + 1, weight=ft.FontWeight.W_500),
            ft.RadioGroup(
                content=ft.Row([ft.Radio(value=o, label=o) for o in drill.options], wrap=True, spacing=0),
                value=given or None,
                on_change=pick,
                disabled=checked,
            ),
            ft.Row(
                [
                    ft.FilledTonalButton(
                        content=ft.Text("Check"), on_click=check, disabled=checked or not given
                    ),
                    feedback,
                ],
                spacing=10,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]
    )


@ft.component
def DrillsActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    record = store.record()
    body: list[ft.Control]
    if record is None:
        body = [status_line("run a job on Level 0: the drills come from its record")]
    else:
        body = [DrillView(store, session, d) for d in drills_for(record, index)] or [
            status_line("this record offers no drill")
        ]

    def reset(_e: Any) -> None:
        store.drill_answers = {}

    return card(
        "Drills: tell them apart",
        ft.Column(body, spacing=theme.GAP),
        why=lambda e: session.select_concept(0, "provenance_tags"),
        info="four discriminations interleaved in immediate succession (estimate or calibrated; what a chip records; resolved, frozen or dropped; inside or outside two error bars), every answer read from the current record",
        actions=[ft.TextButton(content=ft.Text("Reset the answers"), on_click=reset)],
        key="drills",
    )


@ft.component
def ReviewActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    due = store.learner.log.due(now_days())
    body: list[ft.Control] = [
        inset([PromptView(session, cid, CONCEPTS[cid].prompts[0], unaided=True)]) for cid in due
    ] or [status_line("nothing is due: prompts return one review gap after you met them")]
    return card(
        "Review tray",
        ft.Column(body, spacing=theme.GAP),
        why=lambda e: session.select_concept(0, "histogram"),
        info="spaced re-asks with the explanation closed: the delayed, unaided attempts are the only ones the log calls evidence (DESIGN.md Section 3); the tray is a place you visit, never a notification",
        key="review",
    )


@ft.component
def ProgressActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    log = store.learner.log
    now = now_days()
    rows: list[list[ft.Control | str]] = []
    for c in CONCEPTS.values():
        attempts = [a for a in log.attempts if a.concept_id == c.id]
        if not attempts:
            continue
        s = log.session_accuracy(c.id, now_days=now)
        d = log.delayed_unaided_accuracy(c.id)
        rows.append(
            [c.title, str(len(attempts)), "-" if s is None else f"{s:.0%}", "-" if d is None else f"{d:.0%}"]
        )
    show_json, set_show_json = ft.use_state(False)
    body: list[ft.Control] = [
        data_table(
            ["concept", "attempts", "in session (last day)", "delayed, unaided"],
            rows,
            numeric=[False, True, True, True],
        )
        if rows
        else status_line("no attempts yet"),
        status_line("in-session accuracy is not learning; the delayed unaided column is the evidence"),
    ]
    if show_json:
        body.append(
            ft.TextField(
                value=json.dumps(learner_document(store.learner), indent=1),
                multiline=True,
                read_only=True,
                min_lines=4,
                max_lines=12,
                text_size=theme.SIZE_CAPTION,
                border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
            )
        )
    return card(
        "Progress",
        ft.Column(body, spacing=theme.GAP),
        why=lambda e: session.select_concept(0, "histogram"),
        info="the mastery log lives on this device and is never uploaded (DESIGN.md Section 3); copy it as JSON to keep it",
        actions=[
            ft.TextButton(
                content=ft.Text("Hide the log as JSON" if show_json else "Show the log as JSON"),
                icon=ft.Icons.DATA_OBJECT,
                on_click=lambda e: set_show_json(not show_json),
            )
        ],
        key="progress",
    )


@ft.component
def ExperimentsActivity(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    return card(
        "Published experiments",
        PresetList(store, session, index),
        why=lambda e: session.select_concept(0, "provenance_tags"),
        info="each entry of Section 9 runs as an experiment and places the published number beside the simulated one, each with its own chip (Section 14.5)",
        key="experiments",
    )


class Activity(NamedTuple):
    label: str
    render: Callable[[Store, Session, ProvenanceIndex], ft.Control]


ACTIVITIES: dict[str, Activity] = {
    "tour": Activity("Tour", TourActivity),
    "ghz": Activity("Three ions", GhzActivity),
    "free": Activity("Your own", FreeActivity),
    "drills": Activity("Drills", DrillsActivity),
    "review": Activity("Review", ReviewActivity),
    "experiments": Activity("Published experiments", ExperimentsActivity),
    "progress": Activity("Progress", ProgressActivity),
}
"""The activities of the Learn view by route tab, in the order of the ladder."""


@ft.component
def LearnPage(
    store: Store, session: Session, index: ProvenanceIndex, tab: str, preset_id: str | None
) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    if preset_id is not None:
        return PresetPage(store, session, index, preset_id)
    due_count = len(store.learner.log.due(now_days()))
    items = [
        (t, a.label + (f" ({due_count})" if t == "review" and due_count else ""))
        for t, a in ACTIVITIES.items()
    ]
    nav = page_tabs(
        items,
        tab,
        lambda t: page.navigate("/learn" if t == "tour" else f"/learn/{t}"),
        key_prefix="learn-tab",
    )
    return ft.Column(
        [
            level_header("Learn", "Where to start, what to practise, what to reproduce."),
            SettingsRow(store, session),
            nav,
            ACTIVITIES[tab].render(store, session, index),
        ],
        spacing=16,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )
