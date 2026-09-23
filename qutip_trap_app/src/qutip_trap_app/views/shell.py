"""The shell: the routes that mirror the zoom ladder, the rail, the zoom bar with keyboard zoom and the numerics strip;
below ``Shell`` everything is a plain function of the store and the route, so a screen builds without a running page."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flet as ft

from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.numerics import NumericsPanel, numerics_panel
from qutip_trap_app.views.common import MUTED, PAGE_PADDING, RAIL_WIDTH, card, numerics_strip, status_line
from qutip_trap_app.views.hamiltonian import hamiltonian_page
from qutip_trap_app.views.level0 import level0_page
from qutip_trap_app.views.level1 import level1_page
from qutip_trap_app.views.level2 import level2_page
from qutip_trap_app.views.level3 import level3_numerics, level3_page
from qutip_trap_app.views.state import Session, Store

LEVEL_NAMES = ("Machine", "Circuit", "Schedule", "Dynamics", "Equation")
LEVEL_SEGMENTS = {"circuit": 1, "schedule": 2, "dynamics": 3, "hamiltonian": 4}
RAIL_ICONS: tuple[tuple[ft.IconData, ft.IconData], ...] = (
    (ft.Icons.BAR_CHART_OUTLINED, ft.Icons.BAR_CHART),
    (ft.Icons.TIMELINE_OUTLINED, ft.Icons.TIMELINE),
    (ft.Icons.WAVES_OUTLINED, ft.Icons.WAVES),
    (ft.Icons.SHOW_CHART_OUTLINED, ft.Icons.SHOW_CHART),
    (ft.Icons.FUNCTIONS_OUTLINED, ft.Icons.FUNCTIONS),
)


# ---- the routing model ------------------------------------------------------------------------------------------------------------


def _parts(path: str) -> list[str]:
    return [p for p in path.split("/") if p]


def level_of(path: str) -> int:
    parts = _parts(path)
    if len(parts) >= 4 and parts[0] == "job":
        return LEVEL_SEGMENTS.get(parts[2], 0)
    return 0


def record_of(store: Store, path: str) -> Record | None:
    """The record a route names, or the latest run for a route that names none."""
    parts = _parts(path)
    if len(parts) >= 2 and parts[0] == "job":
        return store.records.get(parts[1])
    return store.record()


def gate_of_pulse(record: Record, pulse: int) -> str | None:
    """The gate a pulse belongs to, else the first gate."""
    step = next((s for s in record.schedule.steps if pulse in s.pulse_indices), None)
    if step is not None and step.target_ids:
        return step.target_ids[0]
    return record.schedule.targets[0].gate_id if record.schedule.targets else None


def pulse_of_path(record: Record, path: str) -> int:
    """The pulse a route looks at: Levels 2 to 4 name it, Level 1 names its gate; else the first pulse."""
    parts = _parts(path)
    level = level_of(path)
    if level >= 2 and parts[3].isdigit():
        return int(parts[3])
    if level == 1:
        try:
            step = record.step_of_gate(parts[3])
        except KeyError:
            return 0
        return step.pulse_indices[0] if step.pulse_indices else 0
    return 0


def _sample(path: str) -> str:
    parts = _parts(path)
    return parts[4] if level_of(path) >= 3 and len(parts) >= 5 else "0"


def parent_route(store: Store, path: str) -> str | None:
    """Zoom out: the route one level up, or None at the machine."""
    parts = _parts(path)
    level = level_of(path)
    if level == 0:
        return None
    key = parts[1]
    if level == 4:
        return f"/job/{key}/dynamics/{parts[3]}/{_sample(path)}"
    if level == 3:
        return f"/job/{key}/schedule/{parts[3]}"
    if level == 2:
        record = store.records.get(key)
        gate = gate_of_pulse(record, pulse_of_path(record, path)) if record is not None else None
        return f"/job/{key}/circuit/{gate}" if gate is not None else f"/job/{key}"
    return f"/job/{key}"


def child_route(store: Store, path: str) -> str | None:
    """Zoom in: the first item one level down, or None at the equation."""
    record = record_of(store, path)
    if record is None:
        return None
    key = record.key
    level = level_of(path)
    pulse = pulse_of_path(record, path)
    if level == 0:
        targets = sorted(record.schedule.targets, key=lambda t: t.t_start_s)
        return f"/job/{key}/circuit/{targets[0].gate_id}" if targets else None
    if level == 1:
        return f"/job/{key}/schedule/{pulse}" if record.schedule.pulses else None
    if level == 2:
        return f"/job/{key}/dynamics/{pulse}/0"
    if level == 3:
        return f"/job/{key}/hamiltonian/{pulse}/{_sample(path)}"
    return None


def crumbs_of(store: Store, path: str) -> list[tuple[str, str | None]]:
    """The trail the zoom bar shows: the screen's name, then the job, gate, pulse and sample, each with the route it opens
    (None for the place already shown)."""
    level = level_of(path)
    trail: list[tuple[str, str | None]] = [(LEVEL_NAMES[level], None)]
    record = record_of(store, path)
    if record is None:
        return trail
    key = record.key
    pulse, sample = pulse_of_path(record, path), _sample(path)
    trail.append((f"job {key[:8]}", f"/job/{key}" if level else None))
    if level >= 1:
        gate = _parts(path)[3] if level == 1 else gate_of_pulse(record, pulse)
        trail.append((f"gate {gate}", f"/job/{key}/circuit/{gate}" if level > 1 else None))
    if level >= 2:
        trail.append((f"pulse {pulse}", f"/job/{key}/schedule/{pulse}" if level > 2 else None))
    if level >= 3:
        trail.append((f"sample {sample}", f"/job/{key}/dynamics/{pulse}/{sample}" if level > 3 else None))
    if level == 4:
        trail.append(("equation", None))
    return trail


def rail_route(store: Store, path: str, index: int) -> str:
    """The route a rail destination opens: the same place on the time axis at another level."""
    record = record_of(store, path)
    if record is None:
        return "/"
    key = record.key
    pulse = pulse_of_path(record, path)
    gate = gate_of_pulse(record, pulse)
    if index == 1 and gate is not None:
        return f"/job/{key}/circuit/{gate}"
    if index >= 2 and record.schedule.pulses:
        segment = {2: "schedule", 3: "dynamics", 4: "hamiltonian"}[index]
        return f"/job/{key}/{segment}/{pulse}" + (f"/{_sample(path)}" if index >= 3 else "")
    return f"/job/{key}"


# ---- the layout ---------------------------------------------------------------------------------------------------------------------


def page_content(store: Store, session: Session, path: str) -> ft.Control:
    parts = _parts(path)
    record = record_of(store, path)
    if record is None and len(parts) >= 2 and parts[0] == "job":
        return card(
            "Unknown job", status_line("no such run in this session: run a circuit on the Machine screen")
        )
    level = level_of(path)
    if record is None or level == 0:
        return level0_page(store, session, record)
    if level == 1:
        return level1_page(store, session, record, parts[3])
    if level == 2:
        return level2_page(store, session, record, parts[3])
    if level == 3:
        return level3_page(store, session, record, parts[3], _sample(path))
    return hamiltonian_page(store, session, record, parts[3], _sample(path))


def numerics_for(store: Store, path: str) -> NumericsPanel | None:
    record = record_of(store, path)
    if record is None:
        return None
    if level_of(path) >= 3 and record.schedule.steps:
        return level3_numerics(store, record, _parts(path)[3], _sample(path))
    return numerics_panel(record)


def _go(session: Session, route: str) -> Callable[[Any], None]:
    def on_click(_e: Any) -> None:
        session.navigate(route)

    return on_click


def zoom_bar(store: Store, session: Session, path: str) -> ft.Control:
    up, down = parent_route(store, path), child_route(store, path)
    crumbs: list[ft.Control] = []
    for k, (text, route) in enumerate(crumbs_of(store, path)):
        if k:
            crumbs.append(ft.Icon(ft.Icons.CHEVRON_RIGHT, size=14, color=MUTED))
        crumbs.append(
            ft.TextButton(
                content=ft.Text(text, size=13, weight=ft.FontWeight.W_600 if k == 0 else None, no_wrap=True),
                on_click=_go(session, route) if route else None,
                disabled=route is None,
            )
        )
    return ft.Container(
        content=ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.ZOOM_OUT,
                    tooltip="zoom out (Esc, or Cmd/Ctrl and minus)",
                    on_click=_go(session, up) if up else None,
                    disabled=up is None,
                ),
                ft.IconButton(
                    icon=ft.Icons.ZOOM_IN,
                    tooltip="zoom in (Cmd/Ctrl and plus)",
                    on_click=_go(session, down) if down else None,
                    disabled=down is None,
                ),
                ft.Row(crumbs, spacing=0, expand=True, scroll=ft.ScrollMode.HIDDEN),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        height=48,
        padding=ft.Padding.symmetric(horizontal=PAGE_PADDING - 8),
    )


def layout(store: Store, session: Session, path: str) -> ft.Control:
    """The rail, the zoom bar, the screen of ``path`` and the numerics strip."""

    def on_rail(e: Any) -> None:
        route = rail_route(store, path, int(e.control.selected_index))
        session.navigate(route)
        if level_of(route) != int(e.control.selected_index):
            # a run with nothing at that level stays put: re-render so the rail follows the route
            store.tick += 1

    rail = ft.NavigationRail(
        selected_index=level_of(path),
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=RAIL_WIDTH,
        group_alignment=-1.0,
        destinations=[
            ft.NavigationRailDestination(icon=icon, selected_icon=selected, label=name)
            for (icon, selected), name in zip(RAIL_ICONS, LEVEL_NAMES)
        ],
        trailing=ft.IconButton(
            icon=ft.Icons.BRIGHTNESS_6_OUTLINED,
            tooltip="switch between the system's, the light and the dark theme",
            on_click=lambda _e: session.toggle_theme(),
        ),
        on_change=on_rail,
    )
    panel = numerics_for(store, path)
    return ft.Row(
        [
            rail,
            ft.VerticalDivider(width=1),
            ft.Column(
                [
                    zoom_bar(store, session, path),
                    ft.Container(
                        content=page_content(store, session, path),
                        expand=True,
                        padding=ft.Padding.only(left=PAGE_PADDING, right=PAGE_PADDING, bottom=8),
                    ),
                    numerics_strip(store, panel) if panel is not None else ft.Container(),
                ],
                expand=True,
                spacing=0,
            ),
        ],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )


# ---- the components -----------------------------------------------------------------------------------------------------------------


def on_key(store: Store, session: Session, path: str, e: ft.KeyboardEvent) -> None:
    up, down = parent_route(store, path), child_route(store, path)
    if e.key == "Escape" and up:
        session.navigate(up)
    elif (e.meta or e.ctrl) and e.key in ("-", "Minus", "Numpad Subtract") and up:
        session.navigate(up)
    elif (e.meta or e.ctrl) and e.key in ("=", "+", "Equal", "Numpad Add") and down:
        session.navigate(down)


@ft.component
def Shell(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    path = ft.use_route_location() or "/"

    def install_keys() -> Callable[[], None]:
        page = session.page
        if page is not None:
            page.on_keyboard_event = lambda e: on_key(store, session, path, e)

        def cleanup() -> None:
            if page is not None:
                page.on_keyboard_event = None

        return cleanup

    ft.use_effect(install_keys, dependencies=[path])
    return layout(store, session, path)


@ft.component
def App(store: Store, session: Session) -> ft.Control:
    def shell() -> ft.Control:
        return Shell(store, session)

    return ft.Router(
        routes=[
            ft.Route(index=True, component=shell),
            ft.Route(path="job/:id", component=shell),
            ft.Route(path="job/:id/circuit/:gate", component=shell),
            ft.Route(path="job/:id/schedule/:pulse", component=shell),
            ft.Route(path="job/:id/dynamics/:pulse/:sample", component=shell),
            ft.Route(path="job/:id/hamiltonian/:pulse/:sample", component=shell),
        ],
        not_found=shell,
    )
