"""The application shell: the router whose routes mirror the zoom ladder, the navigation rail, the breadcrumb zoom bar with
keyboard zoom, the explain drawer, the numerics strip, and the first-launch prior-knowledge question (PLAN.md Section 14.6;
DESIGN.md Sections 4, 5 and 10)."""

from __future__ import annotations

from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.learn import DEVICE_PAGES, LEARN_TABS, PAGE_CONCEPTS, PAGE_SECTIONS
from qutip_trap_app.viewmodel.numerics import NumericsPanel, numerics_panel
from qutip_trap_app.viewmodel.presets import PRESETS
from qutip_trap_app.views.common import ExplainDrawer, badge_view, card, event_bool, numerics_strip
from qutip_trap_app.views.learn import KNOWLEDGE_LABELS, LearnPage
from qutip_trap_app.views.level0 import Level0Page
from qutip_trap_app.views.level1 import Level1Page
from qutip_trap_app.views.level2 import Level2Page
from qutip_trap_app.views.level3 import Level3Page, level3_numerics
from qutip_trap_app.views.level4 import Level4Page
from qutip_trap_app.views.state import Session, Store

LEVEL_SECTIONS = {0: "8.6", 1: "7.6", 2: "7.3", 3: "4.4.1", 4: "4.3.1"}
"""The Part II subsection the explain drawer opens per level (Section 14.5 'Explain panel'); a Level 4 page opens its own
(``PAGE_SECTIONS``)."""

LEVEL_NAMES = {0: "Machine", 1: "Circuit", 2: "Schedule", 3: "Dynamics", 4: "Physics"}
LEARN_TAB_IDS = tuple(t for t, _ in LEARN_TABS)


def level_of(path: str) -> int:
    parts = [p for p in path.split("/") if p]
    if not parts:
        return 0
    if parts[0] == "learn":
        return -1
    if parts[0] == "device":
        return 4
    if len(parts) >= 3:
        return {"circuit": 1, "schedule": 2, "dynamics": 3}.get(parts[2], 0)
    return 0


def _gate_of_pulse(record: Record | None, pulse: str) -> str:
    """The gate a pulse belongs to (the breadcrumb between a pulse and its job); "0" when the record is unknown."""
    if record is None:
        return "0"
    try:
        p = record.schedule.pulses[int(pulse)]
    except (ValueError, IndexError):
        return "0"
    gid = (p.gate_id or "").split("/")[0]
    for tg in record.schedule.targets:
        if tg.gate_id == gid or (gid and tg.gate_id.startswith(gid)):
            return tg.gate_id
    return record.schedule.targets[0].gate_id if record.schedule.targets else "0"


def pulse_of_path(record: Record | None, path: str) -> int:
    """The pulse the current route looks at: Levels 2 and 3 name it, Level 1 names its gate (whose first pulse it is), any
    other route means the first pulse. The rail uses it so that changing level keeps the learner's place on the time axis
    instead of snapping back to the first pulse."""
    parts = [p for p in path.split("/") if p]
    if record is None or len(parts) < 4 or parts[0] != "job":
        return 0
    if parts[2] in ("schedule", "dynamics") and parts[3].isdigit():
        return int(parts[3])
    if parts[2] == "circuit":
        try:
            step = record.step_of_gate(parts[3])
        except KeyError:
            return 0
        return int(step.pulse_indices[0]) if step.pulse_indices else 0
    return 0


def parent_route(store: Store, path: str) -> str | None:
    """Zoom out: the route one level up, or None at the top. The record behind the path names a pulse's gate."""
    parts = [p for p in path.split("/") if p]
    if len(parts) <= 2 or parts[0] != "job":
        return None if parts and parts[0] != "job" else "/"
    if parts[2] == "dynamics":
        return f"/job/{parts[1]}/schedule/{parts[3]}"
    if parts[2] == "schedule":
        return f"/job/{parts[1]}/circuit/{_gate_of_pulse(store.records.get(parts[1]), parts[3])}"
    return f"/job/{parts[1]}"


def child_route(store: Store, path: str) -> str | None:
    """Zoom in: the first item one level down (Section 14.6's keyboard zoom)."""
    record = store.record()
    if record is None:
        return None
    key = record.key()
    parts = [p for p in path.split("/") if p]
    lvl = level_of(path)
    if lvl == 0 and record.schedule.targets:
        return f"/job/{key}/circuit/{record.schedule.targets[0].gate_id}"
    if lvl == 1 and len(parts) >= 4:
        try:
            step = record.step_of_gate(parts[3])
            if step.pulse_indices:
                return f"/job/{key}/schedule/{step.pulse_indices[0]}"
        except KeyError:
            return None
    if lvl == 2 and len(parts) >= 4:
        return f"/job/{key}/dynamics/{parts[3]}/0"
    if lvl == 3:
        return "/device/hamiltonian"
    return None


def learn_route_parts(path: str) -> tuple[str, str | None]:
    """(tab, preset id) of a Learn route: ``/learn`` -> ("tour", None), ``/learn/drills`` -> ("drills", None),
    ``/learn/preset/harty_2014`` -> ("experiments", "harty_2014")."""
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 3 and parts[1] == "preset":
        return "experiments", parts[2]
    if len(parts) >= 2 and parts[1] in LEARN_TAB_IDS:
        return parts[1], None
    return "tour", None


@ft.component
def ZoomBar(store: Store, session: Session, path: str, badge: ft.Control | None) -> ft.Control:
    """Crumbs (job, gate, pulse, sample), the zoom buttons, the level name, the convergence badge and the explain toggle."""
    ft.use_state(store)
    page = ft.context.page
    parts = [p for p in path.split("/") if p]
    crumbs: list[ft.Control] = []

    def crumb(text: str, route: str | None) -> ft.Control:
        return ft.TextButton(
            content=ft.Text(text, size=12),
            on_click=(lambda e, r=route: page.navigate(r)) if route else None,
            style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=6)),
        )

    if parts and parts[0] == "job" and len(parts) >= 2:
        key = parts[1]
        crumbs.append(crumb(f"job {key[:8]}", f"/job/{key}"))
        if len(parts) >= 4:
            if parts[2] == "circuit":
                crumbs.append(crumb(f"gate {parts[3]}", f"/job/{key}/circuit/{parts[3]}"))
            else:
                gate = _gate_of_pulse(store.records.get(key), parts[3])
                crumbs.append(crumb(f"gate {gate}", f"/job/{key}/circuit/{gate}"))
                crumbs.append(crumb(f"pulse {parts[3]}", f"/job/{key}/schedule/{parts[3]}"))
                if parts[2] == "dynamics" and len(parts) >= 5:
                    crumbs.append(crumb(f"sample {parts[4]}", None))
    elif parts and parts[0] == "device":
        crumbs.append(crumb(f"device · {parts[1] if len(parts) > 1 else ''}", None))
    elif parts and parts[0] == "learn":
        tab, preset = learn_route_parts(path)
        crumbs.append(crumb("learn", "/learn"))
        if tab != "tour" or preset:
            crumbs.append(crumb(dict(LEARN_TABS).get(tab, tab), f"/learn/{tab}"))
        if preset:
            crumbs.append(crumb(preset, None))
    else:
        crumbs.append(crumb("home", None))
    sep: list[ft.Control] = []
    for k, c in enumerate(crumbs):
        if k:
            sep.append(ft.Text("›", size=14, color=ft.Colors.ON_SURFACE_VARIANT))
        sep.append(c)
    up = parent_route(store, path)
    down = child_route(store, path)
    lvl = level_of(path)
    level = max(lvl, 0)

    def toggle_explain(_e: Any) -> None:
        session.set_learner(explain_open=not store.learner.explain_is_open(level))

    right: list[ft.Control] = []
    if lvl >= 0:
        right.append(
            ft.Text(f"level {lvl} · {LEVEL_NAMES[lvl]}", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        )
    if badge is not None:
        right.append(badge)
    right.append(
        ft.IconButton(
            icon=ft.Icons.MENU_BOOK,
            tooltip="explain: this level's concepts, one at a time, at your depth",
            selected=store.learner.explain_is_open(level),
            on_click=toggle_explain,
            key="explain-toggle",
        )
    )
    return ft.Row(
        [
            ft.IconButton(
                icon=ft.Icons.ZOOM_OUT,
                tooltip="zoom out (Esc, or Cmd/Ctrl and minus)",
                on_click=(lambda e: page.navigate(up)) if up else None,
                disabled=up is None,
                key="zoom-out",
            ),
            ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                tooltip="zoom in (Cmd/Ctrl and plus)",
                on_click=(lambda e: page.navigate(down)) if down else None,
                disabled=down is None,
                key="zoom-in",
            ),
            ft.Row(sep, spacing=2),
            ft.Container(expand=True),
        ]
        + right,
        spacing=4,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


@ft.component
def RoutedContent(store: Store, session: Session, index: ProvenanceIndex, path: str) -> ft.Control:
    """The page for ``path``; the path is a prop rather than a hook so that a route change re-renders this component."""
    ft.use_state(store)
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == "learn":
        tab, preset = learn_route_parts(path)
        return LearnPage(store, session, index, tab, preset)
    if parts and parts[0] == "device":
        return Level4Page(store, session, parts[1] if len(parts) > 1 else "hamiltonian", index)
    if parts and parts[0] == "job" and len(parts) >= 2:
        key = parts[1]
        record = store.records.get(key)
        if record is None:
            return card(
                "Unknown job", ft.Text(f"no record {key} in this session; run a job on Level 0", size=13)
            )
        if store.current != key:
            store.current = key
        if len(parts) >= 4 and parts[2] == "circuit":
            return Level1Page(store, session, record, parts[3], index)
        if len(parts) >= 4 and parts[2] == "schedule":
            return Level2Page(store, session, record, parts[3], index)
        if len(parts) >= 5 and parts[2] == "dynamics":
            return Level3Page(store, session, record, parts[3], parts[4], index)
        return Level0Page(store, session, index)
    return Level0Page(store, session, index)


@ft.component
def Shell(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    path = ft.use_route_location() or "/"
    lvl = level_of(path)
    level = max(lvl, 0)
    record = store.record()

    def rail_change(e: Any) -> None:
        i = int(e.control.selected_index)
        key = record.key() if record is not None else None
        pulse = pulse_of_path(record, path)
        if i == 5:
            page.navigate("/learn")
        elif i == 4:
            page.navigate(f"/device/{store.device_page}")
        elif key is None:
            page.navigate("/")
        elif i == 0:
            page.navigate(f"/job/{key}")
        elif i == 1 and record is not None and record.schedule.targets:
            page.navigate(f"/job/{key}/circuit/{_gate_of_pulse(record, str(pulse))}")
        elif i == 2 and record is not None and record.schedule.pulses:
            page.navigate(f"/job/{key}/schedule/{pulse}")
        elif i == 3 and record is not None and record.schedule.pulses:
            page.navigate(f"/job/{key}/dynamics/{pulse}/0")

    def on_key(e: ft.KeyboardEvent) -> None:
        up = parent_route(store, path)
        down = child_route(store, path)
        if e.key == "Escape":
            # Escape closes the drawer first, then zooms out (DESIGN.md Section 4)
            if lvl >= 0 and store.learner.explain_is_open(level) and store.learner.explain_open is not None:
                session.set_learner(explain_open=False)
            elif up:
                page.navigate(up)
        elif (e.meta or e.ctrl) and e.key in ("-", "Minus", "Numpad Subtract") and up:
            page.navigate(up)
        elif (e.meta or e.ctrl) and e.key in ("=", "+", "Equal", "Numpad Add") and down:
            page.navigate(down)

    def install_keys() -> Any:
        page.on_keyboard_event = on_key

        def cleanup() -> None:
            page.on_keyboard_event = None

        return cleanup

    ft.use_effect(install_keys, dependencies=[path, store.current])
    selected = 5 if lvl == -1 else lvl
    rail = ft.NavigationRail(
        selected_index=selected,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=72,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.BAR_CHART, label="Machine"),
            ft.NavigationRailDestination(icon=ft.Icons.TIMELINE, label="Circuit"),
            ft.NavigationRailDestination(icon=ft.Icons.WAVES, label="Schedule"),
            ft.NavigationRailDestination(icon=ft.Icons.SHOW_CHART, label="Dynamics"),
            ft.NavigationRailDestination(icon=ft.Icons.FUNCTIONS, label="Physics"),
            ft.NavigationRailDestination(icon=ft.Icons.SCHOOL, label="Learn"),
        ],
        on_change=rail_change,
        group_alignment=-0.9,
    )
    content = RoutedContent(store, session, index, path)

    def toggle_numerics(e: Any) -> None:
        store.numerics_open = event_bool(e)

    strip: ft.Control = ft.Container()
    badge: ft.Control | None = None
    parts = [p for p in path.split("/") if p]
    if record is not None and lvl >= 0:
        plan = store.learner.plan(level)
        panel: NumericsPanel
        if lvl == 3 and len(parts) >= 5:
            try:
                panel = level3_numerics(store, record, parts[3], parts[4])
            except (KeyError, IndexError):
                panel = numerics_panel(record)
        else:
            panel = numerics_panel(record)
        strip = numerics_strip(
            panel,
            index,
            expanded=plan.numerics_open if store.numerics_open is None else store.numerics_open,
            on_change=toggle_numerics,
        )
        badge = badge_view(panel.badge, extra=str(panel.level.value or ""))
    # on the five levels the drawer follows the learner's plan; on Learn it opens only when asked (a chip, a why button)
    explain_open = (
        store.learner.explain_is_open(level)
        if lvl >= 0
        else bool(store.learner.explain_open) or store.spec_section is not None
    )
    device_page = parts[1] if lvl == 4 and len(parts) > 1 and parts[1] in DEVICE_PAGES else None
    columns: list[ft.Control] = [
        rail,
        ft.VerticalDivider(width=1),
        ft.Column(
            [
                ZoomBar(store, session, path, badge),
                ft.Container(content=content, expand=True, padding=ft.Padding.symmetric(horizontal=8)),
                strip,
            ],
            expand=True,
            spacing=6,
        ),
    ]
    if explain_open:
        if device_page is not None:
            columns.append(
                ExplainDrawer(
                    store,
                    session,
                    level,
                    index,
                    PAGE_SECTIONS[device_page],
                    concepts=PAGE_CONCEPTS[device_page],
                )
            )
        elif lvl == -1:
            _tab, preset_id = learn_route_parts(path)
            spec = PRESETS.get(preset_id or "")
            concepts = (spec.concept_id,) if spec is not None else None
            section = spec.section if spec is not None else "14.5"
            columns.append(ExplainDrawer(store, session, 0, index, section, concepts=concepts))
        else:
            columns.append(ExplainDrawer(store, session, level, index, LEVEL_SECTIONS.get(level, "4.3.1")))
    dialog = None
    # asked once per device: the saved learner is read first, so a returning learner never sees the question again
    if store.learner_loaded and not store.learner.asked:

        def choose(k: str) -> None:
            session.set_learner(knowledge=k, asked=True)

        choices: list[ft.Control] = [
            ft.Text("The app adapts its explanations to you; change this in Learn.", size=13)
        ]
        choices.extend(
            ft.FilledButton(content=ft.Text(v), on_click=lambda e, k=k: choose(k), key=f"knowledge-{k}")
            for k, v in KNOWLEDGE_LABELS.items()
            if k != "unknown"
        )
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Who is learning?"),
            content=ft.Column(choices, tight=True, spacing=8),
            actions=[
                ft.TextButton(
                    content=ft.Text("Skip: assist me"),
                    on_click=lambda e: choose("unknown"),
                    key="knowledge-skip",
                )
            ],
        )
    ft.use_dialog(dialog)
    return ft.Row(columns, expand=True, spacing=0, vertical_alignment=ft.CrossAxisAlignment.STRETCH)


@ft.component
def App(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    def layout() -> ft.Control:
        return Shell(store, session, index)

    return ft.Router(
        routes=[
            ft.Route(index=True, component=layout),
            ft.Route(path="job/:id", component=layout),
            ft.Route(path="job/:id/circuit/:gate", component=layout),
            ft.Route(path="job/:id/schedule/:pulse", component=layout),
            ft.Route(path="job/:id/dynamics/:pulse/:sample", component=layout),
            ft.Route(path="device/:page", component=layout),
            ft.Route(path="learn", component=layout),
            ft.Route(path="learn/:tab", component=layout),
            ft.Route(path="learn/preset/:preset", component=layout),
        ],
        not_found=layout,
    )


__all__ = [
    "App",
    "DEVICE_PAGES",
    "LEVEL_NAMES",
    "LEVEL_SECTIONS",
    "Shell",
    "child_route",
    "learn_route_parts",
    "level_of",
    "parent_route",
    "pulse_of_path",
]
