"""The application shell: the router whose routes mirror the zoom ladder, the navigation rail, the breadcrumb zoom bar with
keyboard zoom, the explain drawer, the numerics strip, and the first-launch prior-knowledge question (PLAN.md Section 14.6;
DESIGN.md Sections 4, 5 and 10)."""

from __future__ import annotations

from typing import Any, NamedTuple

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.learn import level_concepts
from qutip_trap_app.viewmodel.numerics import NumericsPanel, numerics_panel
from qutip_trap_app.viewmodel.presets import PRESETS
from qutip_trap_app.views import theme
from qutip_trap_app.views.common import (
    MUTED,
    ExplainDrawer,
    card,
    content_margin,
    event_bool,
    numerics_strip,
    status_line,
)
from qutip_trap_app.views.learn import ACTIVITIES, KNOWLEDGE, LearnPage
from qutip_trap_app.views.level0 import Level0Page
from qutip_trap_app.views.level1 import Level1Page
from qutip_trap_app.views.level2 import Level2Page
from qutip_trap_app.views.level3 import Level3Page, level3_numerics
from qutip_trap_app.views.level4 import DEVICE_PAGES, Level4Page
from qutip_trap_app.views.state import Session, Store, ThemeChoice

LEVEL_SECTIONS = {0: "8.6", 1: "7.6", 2: "7.3", 3: "4.4.1"}
"""The Part II subsection the explain drawer opens per level; a Level 4 page opens its own."""


class RailItem(NamedTuple):
    label: str
    icon: ft.IconData
    selected_icon: ft.IconData


RAIL: tuple[RailItem, ...] = (
    RailItem("Machine", ft.Icons.BAR_CHART_OUTLINED, ft.Icons.BAR_CHART),
    RailItem("Circuit", ft.Icons.TIMELINE_OUTLINED, ft.Icons.TIMELINE),
    RailItem("Schedule", ft.Icons.WAVES_OUTLINED, ft.Icons.WAVES),
    RailItem("Dynamics", ft.Icons.SHOW_CHART_OUTLINED, ft.Icons.SHOW_CHART),
    RailItem("Physics", ft.Icons.FUNCTIONS, ft.Icons.FUNCTIONS),
    RailItem("Learn", ft.Icons.SCHOOL_OUTLINED, ft.Icons.SCHOOL),
)
"""The rail: the five levels of the ladder, in order, and Learn."""


class ThemeOption(NamedTuple):
    mode: ft.ThemeMode
    icon: ft.IconData
    next: ThemeChoice


THEMES: dict[ThemeChoice, ThemeOption] = {
    "system": ThemeOption(ft.ThemeMode.SYSTEM, ft.Icons.BRIGHTNESS_AUTO, "light"),
    "light": ThemeOption(ft.ThemeMode.LIGHT, ft.Icons.LIGHT_MODE_OUTLINED, "dark"),
    "dark": ThemeOption(ft.ThemeMode.DARK, ft.Icons.DARK_MODE_OUTLINED, "system"),
}


# ---- routes ----------------------------------------------------------------------------------------------------------


class Route(NamedTuple):
    """One path of the app, parsed: /job/{job}, then /circuit/{gate}, /schedule/{pulse} or /dynamics/{pulse}/{sample};
    /device/{page}; /learn, /learn/{tab} or /learn/preset/{preset}. Anything else is the machine page."""

    level: int
    """0 to 4 on the zoom ladder; -1 on Learn."""
    job: str | None = None
    gate: str | None = None
    pulse: str | None = None
    sample: str = "0"
    page: str = "hamiltonian"
    tab: str = "tour"
    preset: str | None = None


def parse_route(path: str) -> Route:
    parts = [p for p in path.split("/") if p]
    head = parts[0] if parts else ""
    if head == "learn":
        if len(parts) >= 3 and parts[1] == "preset":
            return Route(-1, tab="experiments", preset=parts[2])
        return Route(-1, tab=parts[1] if len(parts) >= 2 and parts[1] in ACTIVITIES else "tour")
    if head == "device":
        return Route(4, page=parts[1] if len(parts) >= 2 and parts[1] in DEVICE_PAGES else "hamiltonian")
    if head != "job" or len(parts) < 2:
        return Route(0)
    kind = parts[2] if len(parts) >= 4 else ""
    if kind == "circuit":
        return Route(1, parts[1], gate=parts[3])
    if kind == "schedule":
        return Route(2, parts[1], pulse=parts[3])
    if kind == "dynamics":
        return Route(3, parts[1], pulse=parts[3], sample=parts[4] if len(parts) >= 5 else "0")
    return Route(0, parts[1])


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
    """The pulse the route looks at: Levels 2 and 3 name it, Level 1 names its gate (whose first pulse it is), any other
    route means the first pulse. The rail uses it so that changing level keeps the learner's place on the time axis."""
    r = parse_route(path)
    if record is None:
        return 0
    if r.pulse is not None and r.pulse.isdigit():
        return int(r.pulse)
    if r.gate is not None:
        try:
            step = record.step_of_gate(r.gate)
        except KeyError:
            return 0
        return int(step.pulse_indices[0]) if step.pulse_indices else 0
    return 0


def parent_route(store: Store, path: str) -> str | None:
    """Zoom out: the route one level up, "/" at the top, None off the ladder. The record behind the path names a pulse's
    gate."""
    r = parse_route(path)
    if r.level == 3:
        return f"/job/{r.job}/schedule/{r.pulse}"
    if r.level == 2:
        return f"/job/{r.job}/circuit/{_gate_of_pulse(store.records.get(r.job or ''), r.pulse or '')}"
    if r.level == 1:
        return f"/job/{r.job}"
    return "/" if r.level == 0 else None


def child_route(store: Store, path: str) -> str | None:
    """Zoom in: the first item one level down (the keyboard zoom)."""
    record = store.record()
    if record is None:
        return None
    key = record.key()
    r = parse_route(path)
    if r.level == 0 and record.schedule.targets:
        return f"/job/{key}/circuit/{record.schedule.targets[0].gate_id}"
    if r.level == 1 and r.gate is not None:
        try:
            step = record.step_of_gate(r.gate)
        except KeyError:
            return None
        return f"/job/{key}/schedule/{step.pulse_indices[0]}" if step.pulse_indices else None
    if r.level == 2:
        return f"/job/{key}/dynamics/{r.pulse}/0"
    if r.level == 3:
        return "/device/hamiltonian"
    return None


def crumbs_of(store: Store, path: str) -> list[tuple[str, str | None]]:
    """The trail the zoom bar shows: the level's name, then the path (job, gate, pulse, sample; the physics page; the Learn
    activity), each with the route it opens or None for the place the learner already is."""
    r = parse_route(path)
    if r.level == 4:
        return [(RAIL[4].label, None), (DEVICE_PAGES[r.page].title, None)]
    if r.level == -1:
        trail: list[tuple[str, str | None]] = [("Learn", "/learn")]
        if r.tab != "tour" or r.preset:
            trail.append((ACTIVITIES[r.tab].label, f"/learn/{r.tab}"))
        if r.preset:
            trail.append((PRESETS[r.preset].title if r.preset in PRESETS else r.preset, None))
        return trail
    trail = [(RAIL[r.level].label, None)]
    if r.job is None:
        return trail
    trail.append((f"job {r.job[:8]}", f"/job/{r.job}"))
    if r.gate is not None:
        trail.append((f"gate {r.gate}", f"/job/{r.job}/circuit/{r.gate}"))
    elif r.pulse is not None:
        gate = _gate_of_pulse(store.records.get(r.job), r.pulse)
        trail.append((f"gate {gate}", f"/job/{r.job}/circuit/{gate}"))
        trail.append((f"pulse {r.pulse}", f"/job/{r.job}/schedule/{r.pulse}"))
        if r.level == 3:
            trail.append((f"sample {r.sample}", None))
    return trail


# ---- the screens -----------------------------------------------------------------------------------------------------


@ft.component
def ZoomBar(store: Store, session: Session, path: str) -> ft.Control:
    """The zoom buttons, the crumbs and the explain toggle on one 48 px line (the convergence badge lives in the numerics
    strip)."""
    ft.use_state(store)
    page = ft.context.page
    level = max(parse_route(path).level, 0)
    up = parent_route(store, path)
    down = child_route(store, path)
    crumbs: list[ft.Control] = []
    for k, (text, route) in enumerate(crumbs_of(store, path)):
        if k:
            crumbs.append(ft.Icon(ft.Icons.CHEVRON_RIGHT, size=14, color=MUTED))
        crumbs.append(
            ft.TextButton(
                content=ft.Text(
                    text,
                    size=theme.SIZE_SMALL + 1,
                    weight=ft.FontWeight.W_600 if k == 0 and route is None else ft.FontWeight.W_400,
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
                ft.IconButton(
                    icon=ft.Icons.MENU_BOOK_OUTLINED,
                    selected_icon=ft.Icons.MENU_BOOK,
                    icon_size=20,
                    tooltip="explain: this level's concepts, one at a time, at your depth",
                    selected=store.learner.explain_is_open(level),
                    on_click=lambda e: session.set_learner(
                        explain_open=not store.learner.explain_is_open(level)
                    ),
                    style=ft.ButtonStyle(bgcolor={ft.ControlState.SELECTED: ft.Colors.SECONDARY_CONTAINER}),
                    key="explain-toggle",
                ),
            ],
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
    r = parse_route(path)
    if r.level == -1:
        return LearnPage(store, session, index, r.tab, r.preset)
    if r.level == 4:
        return Level4Page(store, session, r.page, index)
    if r.job is None:
        return Level0Page(store, session, index)
    record = store.records.get(r.job)
    if record is None:
        return card("Unknown job", status_line(f"no record {r.job} in this session; run a job on Level 0"))
    if store.current != r.job:
        store.current = r.job
    if r.gate is not None:
        return Level1Page(store, session, record, r.gate, index)
    if r.pulse is not None and r.level == 2:
        return Level2Page(store, session, record, r.pulse, index)
    if r.pulse is not None:
        return Level3Page(store, session, record, r.pulse, r.sample, index)
    return Level0Page(store, session, index)


def _learner_dialog(session: Session) -> ft.AlertDialog:
    """The first-launch question: three plain self-descriptions, each with what it changes, and a skip that means assist."""

    def choose(k: str) -> None:
        session.set_learner(knowledge=k, asked=True)

    options: list[ft.Control] = [
        ft.OutlinedButton(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            option.label,
                            size=theme.SIZE_BODY,
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.ON_SURFACE,
                        ),
                        ft.Text(option.description, size=theme.SIZE_SMALL, color=MUTED),
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
        for k, option in KNOWLEDGE.items()
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
                content=ft.Text("Skip: assist me"), on_click=lambda e: choose("unknown"), key="knowledge-skip"
            )
        ],
    )


@ft.component
def Shell(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    path = ft.use_route_location() or "/"
    route = parse_route(path)
    level = max(route.level, 0)
    record = store.record()

    def rail_change(e: Any) -> None:
        i = int(e.control.selected_index)
        key = record.key() if record is not None else None
        pulse = pulse_of_path(record, path)
        if i == 5:
            page.navigate("/learn")
        elif i == 4:
            page.navigate(f"/device/{store.device_page}")
        elif record is None:
            page.navigate("/")
        elif i == 0:
            page.navigate(f"/job/{key}")
        elif i == 1 and record.schedule.targets:
            page.navigate(f"/job/{key}/circuit/{_gate_of_pulse(record, str(pulse))}")
        elif i == 2 and record.schedule.pulses:
            page.navigate(f"/job/{key}/schedule/{pulse}")
        elif i == 3 and record.schedule.pulses:
            page.navigate(f"/job/{key}/dynamics/{pulse}/0")
        else:
            # a record with no gate (a bare measurement) has nothing on Levels 1 to 3: stay with the job, and re-render
            # so the rail's highlight follows the route rather than the click
            page.navigate(f"/job/{key}")
            store.tick = store.tick + 1

    def on_key(e: ft.KeyboardEvent) -> None:
        up = parent_route(store, path)
        down = child_route(store, path)
        if e.key == "Escape":
            # Escape closes the drawer first, then zooms out
            if (
                route.level >= 0
                and store.learner.explain_is_open(level)
                and store.learner.explain_open is not None
            ):
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
    choice = store.learner.theme

    def apply_theme() -> None:
        if page.theme_mode != THEMES[choice].mode:
            page.theme_mode = THEMES[choice].mode
            page.update()

    ft.use_effect(apply_theme, dependencies=[choice])
    rail = ft.NavigationRail(
        selected_index=5 if route.level == -1 else route.level,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=theme.RAIL_WIDTH,
        group_alignment=-1.0,
        leading=ft.Container(
            content=ft.Image(src="icon.png", width=32, height=32, tooltip="qutip-trap"),
            padding=ft.Padding.only(top=12, bottom=8),
        ),
        destinations=[
            ft.NavigationRailDestination(icon=item.icon, selected_icon=item.selected_icon, label=item.label)
            for item in RAIL
        ],
        trailing=ft.Container(
            content=ft.IconButton(
                icon=THEMES[choice].icon,
                icon_color=MUTED,
                tooltip=f"theme: {choice}; click for {THEMES[choice].next}",
                on_click=lambda e: session.set_learner(theme=THEMES[choice].next),
                key="theme-toggle",
            ),
            padding=ft.Padding.only(bottom=12),
        ),
        pin_trailing_to_bottom=True,
        on_change=rail_change,
    )

    def toggle_numerics(e: Any) -> None:
        store.numerics_open = event_bool(e)

    strip: ft.Control = ft.Container()
    if record is not None and route.level >= 0:
        panel: NumericsPanel
        if route.level == 3 and route.pulse is not None:
            try:
                panel = level3_numerics(store, record, route.pulse, route.sample)
            except (KeyError, IndexError):
                panel = numerics_panel(record)
        else:
            panel = numerics_panel(record)
        strip = numerics_strip(
            panel,
            index,
            expanded=store.learner.plan(level).expanded
            if store.numerics_open is None
            else store.numerics_open,
            on_change=toggle_numerics,
        )
    # on the five levels the drawer follows the learner's plan; on Learn it opens only when asked (a chip, a why button)
    explain_open = (
        store.learner.explain_is_open(level)
        if route.level >= 0
        else bool(store.learner.explain_open) or store.spec_section is not None
    )
    margin = content_margin(page, explain_open)
    columns: list[ft.Control] = [
        rail,
        ft.Column(
            [
                ZoomBar(store, session, path),
                ft.Container(
                    content=RoutedContent(store, session, index, path),
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
        if route.level == 4:
            device_page = DEVICE_PAGES[route.page]
            section, concepts = device_page.section, device_page.concepts
        elif route.level == -1:
            spec = PRESETS.get(route.preset or "")
            section = "14.5" if spec is None else spec.section
            concepts = level_concepts(0) if spec is None else (spec.concept_id,)
        else:
            section, concepts = LEVEL_SECTIONS[level], level_concepts(level)
        columns.append(ExplainDrawer(store, session, level, index, section, concepts))
    # asked once per device: the saved learner is read first, so a returning learner never sees the question again
    ft.use_dialog(_learner_dialog(session) if store.learner_loaded and not store.learner.asked else None)
    return ft.Container(
        content=ft.Row(columns, expand=True, spacing=0, vertical_alignment=ft.CrossAxisAlignment.STRETCH),
        bgcolor=ft.Colors.SURFACE,
        expand=True,
    )


@ft.component
def App(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    def layout() -> ft.Control:
        return Shell(store, session, index)

    patterns = (
        "job/:id",
        "job/:id/circuit/:gate",
        "job/:id/schedule/:pulse",
        "job/:id/dynamics/:pulse/:sample",
        "device/:page",
        "learn",
        "learn/:tab",
        "learn/preset/:preset",
    )
    return ft.Router(
        routes=[ft.Route(index=True, component=layout)]
        + [ft.Route(path=p, component=layout) for p in patterns],
        not_found=layout,
    )
