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
from qutip_trap_app.views import theme
from qutip_trap_app.views.common import (
    HAIRLINE,
    MUTED,
    ExplainDrawer,
    card,
    content_margin,
    event_bool,
    numerics_strip,
    status_line,
)
from qutip_trap_app.views.learn import KNOWLEDGE_DESCRIPTIONS, KNOWLEDGE_LABELS, LearnPage
from qutip_trap_app.views.level0 import Level0Page
from qutip_trap_app.views.level1 import Level1Page
from qutip_trap_app.views.level2 import Level2Page
from qutip_trap_app.views.level3 import Level3Page, level3_numerics
from qutip_trap_app.views.level4 import PAGE_TITLES, Level4Page
from qutip_trap_app.views.state import Session, Store

LEVEL_SECTIONS = {0: "8.6", 1: "7.6", 2: "7.3", 3: "4.4.1", 4: "4.3.1"}
"""The Part II subsection the explain drawer opens per level (Section 14.5 'Explain panel'); a Level 4 page opens its own
(``PAGE_SECTIONS``)."""

LEVEL_NAMES = {0: "Machine", 1: "Circuit", 2: "Schedule", 3: "Dynamics", 4: "Physics"}
LEARN_TAB_IDS = tuple(t for t, _ in LEARN_TABS)

THEME_MODES: dict[str, ft.ThemeMode] = {
    "system": ft.ThemeMode.SYSTEM,
    "light": ft.ThemeMode.LIGHT,
    "dark": ft.ThemeMode.DARK,
}
THEME_ICONS: dict[str, ft.IconData] = {
    "system": ft.Icons.BRIGHTNESS_AUTO,
    "light": ft.Icons.LIGHT_MODE_OUTLINED,
    "dark": ft.Icons.DARK_MODE_OUTLINED,
}
THEME_NEXT: dict[str, str] = {"system": "light", "light": "dark", "dark": "system"}

RAIL_ITEMS: tuple[tuple[ft.IconData, ft.IconData, str], ...] = (
    (ft.Icons.BAR_CHART_OUTLINED, ft.Icons.BAR_CHART, "Machine"),
    (ft.Icons.TIMELINE_OUTLINED, ft.Icons.TIMELINE, "Circuit"),
    (ft.Icons.WAVES_OUTLINED, ft.Icons.WAVES, "Schedule"),
    (ft.Icons.SHOW_CHART_OUTLINED, ft.Icons.SHOW_CHART, "Dynamics"),
    (ft.Icons.FUNCTIONS, ft.Icons.FUNCTIONS, "Physics"),
    (ft.Icons.SCHOOL_OUTLINED, ft.Icons.SCHOOL, "Learn"),
)
"""The rail: the five levels of the ladder and Learn (DESIGN.md Section 4 "Layout")."""


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


def crumbs_of(store: Store, path: str) -> list[tuple[str, str | None]]:
    """The trail the zoom bar shows: the level's name first, then the path (job, gate, pulse, sample; the Learn activity;
    the physics page), each with the route it opens or None for the place the learner already is."""
    parts = [p for p in path.split("/") if p]
    lvl = level_of(path)
    trail: list[tuple[str, str | None]] = []
    if parts and parts[0] == "job" and len(parts) >= 2:
        key = parts[1]
        trail.append((LEVEL_NAMES[max(lvl, 0)], None))
        trail.append((f"job {key[:8]}", f"/job/{key}"))
        if len(parts) >= 4:
            if parts[2] == "circuit":
                trail.append((f"gate {parts[3]}", f"/job/{key}/circuit/{parts[3]}"))
            else:
                gate = _gate_of_pulse(store.records.get(key), parts[3])
                trail.append((f"gate {gate}", f"/job/{key}/circuit/{gate}"))
                trail.append((f"pulse {parts[3]}", f"/job/{key}/schedule/{parts[3]}"))
                if parts[2] == "dynamics" and len(parts) >= 5:
                    trail.append((f"sample {parts[4]}", None))
    elif parts and parts[0] == "device":
        page_name = parts[1] if len(parts) > 1 and parts[1] in PAGE_TITLES else "hamiltonian"
        trail.append((LEVEL_NAMES[4], None))
        trail.append((PAGE_TITLES[page_name][0], None))
    elif parts and parts[0] == "learn":
        tab, preset = learn_route_parts(path)
        trail.append(("Learn", "/learn"))
        if tab != "tour" or preset:
            trail.append((dict(LEARN_TABS).get(tab, tab), f"/learn/{tab}"))
        if preset:
            trail.append((PRESETS[preset].title if preset in PRESETS else preset, None))
    else:
        trail.append((LEVEL_NAMES[0], None))
    return trail


@ft.component
def ZoomBar(store: Store, session: Session, path: str) -> ft.Control:
    """The zoom buttons, the crumbs (the level, then job, gate, pulse, sample) and the explain toggle on one 48 px line;
    the convergence badge lives in the numerics strip (R6), so it is not repeated here."""
    ft.use_state(store)
    page = ft.context.page
    lvl = level_of(path)
    level = max(lvl, 0)
    up = parent_route(store, path)
    down = child_route(store, path)
    trail = crumbs_of(store, path)
    crumbs: list[ft.Control] = []
    for k, (text, route) in enumerate(trail):
        if k:
            crumbs.append(ft.Icon(ft.Icons.CHEVRON_RIGHT, size=14, color=MUTED))
        is_level = k == 0 and route is None
        crumbs.append(
            ft.TextButton(
                content=ft.Text(
                    text,
                    size=theme.SIZE_SMALL + 1,
                    weight=ft.FontWeight.W_600 if is_level else ft.FontWeight.W_400,
                    color=ft.Colors.ON_SURFACE if route is None else None,
                    no_wrap=True,
                ),
                on_click=(lambda e, r=route: page.navigate(r)) if route else None,
                disabled=route is None,
                style=ft.ButtonStyle(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=4),
                    color={ft.ControlState.DISABLED: ft.Colors.ON_SURFACE},
                    visual_density=ft.VisualDensity.COMPACT,
                ),
            )
        )

    def toggle_explain(_e: Any) -> None:
        session.set_learner(explain_open=not store.learner.explain_is_open(level))

    right: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.MENU_BOOK_OUTLINED,
            selected_icon=ft.Icons.MENU_BOOK,
            icon_size=20,
            tooltip="explain: this level's concepts, one at a time, at your depth",
            selected=store.learner.explain_is_open(level),
            on_click=toggle_explain,
            style=ft.ButtonStyle(bgcolor={ft.ControlState.SELECTED: ft.Colors.SECONDARY_CONTAINER}),
            key="explain-toggle",
        )
    ]
    return ft.Container(
        content=ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.ZOOM_OUT,
                    icon_size=20,
                    tooltip="zoom out (Esc, or Cmd/Ctrl and minus)",
                    on_click=(lambda e: page.navigate(up)) if up else None,
                    disabled=up is None,
                    key="zoom-out",
                ),
                ft.IconButton(
                    icon=ft.Icons.ZOOM_IN,
                    icon_size=20,
                    tooltip="zoom in (Cmd/Ctrl and plus)",
                    on_click=(lambda e: page.navigate(down)) if down else None,
                    disabled=down is None,
                    key="zoom-in",
                ),
                ft.Row(
                    crumbs,
                    spacing=0,
                    expand=True,
                    scroll=ft.ScrollMode.HIDDEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ]
            + right,
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        height=48,
        padding=ft.Padding.symmetric(horizontal=theme.PAGE_PADDING - theme.GAP),
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
            return card("Unknown job", status_line(f"no record {key} in this session; run a job on Level 0"))
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


def _learner_dialog(session: Session) -> ft.AlertDialog:
    """The first-launch question (DESIGN.md Section 1): three plain self-descriptions, each with what it changes, and a
    skip that means assist."""

    def choose(k: str) -> None:
        session.set_learner(knowledge=k, asked=True)

    options: list[ft.Control] = [
        ft.OutlinedButton(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            KNOWLEDGE_LABELS[k],
                            size=theme.SIZE_BODY,
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.ON_SURFACE,
                        ),
                        ft.Text(KNOWLEDGE_DESCRIPTIONS[k], size=theme.SIZE_SMALL, color=MUTED),
                    ],
                    spacing=2,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
                padding=ft.Padding.symmetric(vertical=6),
            ),
            on_click=lambda e, k=k: choose(k),
            style=ft.ButtonStyle(alignment=ft.Alignment.CENTER_LEFT),
            width=400,
            key=f"knowledge-{k}",
        )
        for k in KNOWLEDGE_LABELS
        if k != "unknown"
    ]
    return ft.AlertDialog(
        modal=True,
        title=ft.Text("Who is learning?"),
        content=ft.Column(
            [status_line("The app adapts its explanations to you; change this in Learn.")] + options,
            tight=True,
            spacing=theme.GAP,
            width=400,
        ),
        actions=[
            ft.TextButton(
                content=ft.Text("Skip: assist me"),
                on_click=lambda e: choose("unknown"),
                key="knowledge-skip",
            )
        ],
    )


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

    # the colour scheme follows the learner's choice (the platform's by default); the choice persists with the learner
    def apply_theme() -> None:
        mode = THEME_MODES.get(store.learner.theme, ft.ThemeMode.SYSTEM)
        if page.theme_mode != mode:
            page.theme_mode = mode
            page.update()

    ft.use_effect(apply_theme, dependencies=[store.learner.theme])

    def cycle_theme(_e: Any) -> None:
        session.set_learner(theme=THEME_NEXT.get(store.learner.theme, "system"))

    choice = store.learner.theme
    selected = 5 if lvl == -1 else lvl
    rail = ft.NavigationRail(
        selected_index=selected,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=theme.RAIL_WIDTH,
        group_alignment=-1.0,
        leading=ft.Container(
            content=ft.Image(src="icon.png", width=32, height=32, tooltip="qutip-trap"),
            padding=ft.Padding.only(top=12, bottom=8),
        ),
        destinations=[
            ft.NavigationRailDestination(icon=icon, selected_icon=selected_icon, label=label)
            for icon, selected_icon, label in RAIL_ITEMS
        ],
        trailing=ft.Container(
            content=ft.IconButton(
                icon=THEME_ICONS.get(choice, ft.Icons.BRIGHTNESS_AUTO),
                icon_color=MUTED,
                tooltip=f"theme: {choice}; click for {THEME_NEXT.get(choice, 'system')}",
                on_click=cycle_theme,
                key="theme-toggle",
            ),
            padding=ft.Padding.only(bottom=12),
        ),
        pin_trailing_to_bottom=True,
        on_change=rail_change,
    )
    content = RoutedContent(store, session, index, path)

    def toggle_numerics(e: Any) -> None:
        store.numerics_open = event_bool(e)

    strip: ft.Control = ft.Container()
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
    # on the five levels the drawer follows the learner's plan; on Learn it opens only when asked (a chip, a why button)
    explain_open = (
        store.learner.explain_is_open(level)
        if lvl >= 0
        else bool(store.learner.explain_open) or store.spec_section is not None
    )
    device_page = parts[1] if lvl == 4 and len(parts) > 1 and parts[1] in DEVICE_PAGES else None
    margin = content_margin(page, explain_open)
    columns: list[ft.Control] = [
        rail,
        ft.Column(
            [
                ZoomBar(store, session, path),
                ft.Container(
                    content=content,
                    expand=True,
                    padding=ft.Padding.only(
                        left=theme.PAGE_PADDING + margin, right=theme.PAGE_PADDING + margin, bottom=theme.GAP
                    ),
                ),
                strip,
            ],
            expand=True,
            spacing=0,
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
    # asked once per device: the saved learner is read first, so a returning learner never sees the question again
    dialog = _learner_dialog(session) if store.learner_loaded and not store.learner.asked else None
    ft.use_dialog(dialog)
    return ft.Container(
        content=ft.Row(columns, expand=True, spacing=0, vertical_alignment=ft.CrossAxisAlignment.STRETCH),
        bgcolor=ft.Colors.SURFACE,
        expand=True,
    )


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
    "HAIRLINE",
    "LEVEL_NAMES",
    "LEVEL_SECTIONS",
    "RAIL_ITEMS",
    "Shell",
    "THEME_ICONS",
    "THEME_MODES",
    "THEME_NEXT",
    "child_route",
    "crumbs_of",
    "learn_route_parts",
    "level_of",
    "parent_route",
    "pulse_of_path",
]
