"""The application shell: the router whose routes mirror the zoom ladder, the navigation rail, the breadcrumb zoom bar with
keyboard zoom, the explain drawer, the numerics strip, and the first-launch prior-knowledge question (PLAN.md Section 14.6;
DESIGN.md Sections 4 and 5)."""

from __future__ import annotations

from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.numerics import numerics_panel
from qutip_trap_app.views.common import ExplainDrawer, card, event_bool, level_header, numerics_strip
from qutip_trap_app.views.learn import KNOWLEDGE_LABELS, LearnPage
from qutip_trap_app.views.level0 import DeviceCardView, Level0Page
from qutip_trap_app.views.level1 import Level1Page
from qutip_trap_app.views.level2 import Level2Page
from qutip_trap_app.views.state import Session, Store

LEVEL_SECTIONS = {0: "8.6", 1: "7.6", 2: "7.3", 3: "4.4.1", 4: "4.3.1"}
"""The Part II subsection the explain drawer opens per level (Section 14.5 'Explain panel')."""

DEVICE_PAGES = ("species", "trap", "crystal", "light", "noise", "cooling", "readout", "hamiltonian")


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


@ft.component
def ZoomBar(store: Store, session: Session, path: str) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    parts = [p for p in path.split("/") if p]
    crumbs: list[ft.Control] = []
    record = store.record()
    if parts and parts[0] == "job" and len(parts) >= 2:
        key = parts[1]
        crumbs.append(
            ft.TextButton(
                content=ft.Text(f"job {key[:8]}", size=12), on_click=lambda e: page.navigate(f"/job/{key}")
            )
        )
        if len(parts) >= 4:
            if parts[2] == "circuit":
                crumbs.append(
                    ft.TextButton(
                        content=ft.Text(f"gate {parts[3]}", size=12),
                        on_click=lambda e: page.navigate(f"/job/{key}/circuit/{parts[3]}"),
                    )
                )
            else:
                gate = _gate_of_pulse(store.records.get(key), parts[3])
                crumbs.append(
                    ft.TextButton(
                        content=ft.Text(f"gate {gate}", size=12),
                        on_click=lambda e: page.navigate(f"/job/{key}/circuit/{gate}"),
                    )
                )
                crumbs.append(
                    ft.TextButton(
                        content=ft.Text(f"pulse {parts[3]}", size=12),
                        on_click=lambda e: page.navigate(f"/job/{key}/schedule/{parts[3]}"),
                    )
                )
                if parts[2] == "dynamics" and len(parts) >= 5:
                    crumbs.append(
                        ft.TextButton(content=ft.Text(f"sample {parts[4]}", size=12), on_click=None)
                    )
    elif parts and parts[0] == "device":
        crumbs.append(
            ft.TextButton(
                content=ft.Text(f"device · {parts[1] if len(parts) > 1 else ''}", size=12), on_click=None
            )
        )
    elif parts and parts[0] == "learn":
        crumbs.append(ft.TextButton(content=ft.Text("learn", size=12), on_click=None))
    else:
        crumbs.append(ft.TextButton(content=ft.Text("home", size=12), on_click=None))
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

    return ft.Row(
        [
            ft.IconButton(
                icon=ft.Icons.ZOOM_OUT,
                tooltip="zoom out (Esc, or Cmd/Ctrl and minus)",
                on_click=(lambda e: page.navigate(up)) if up else None,
                disabled=up is None,
            ),
            ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                tooltip="zoom in (Cmd/Ctrl and plus)",
                on_click=(lambda e: page.navigate(down)) if down else None,
                disabled=down is None,
            ),
            ft.Row(sep, spacing=2),
            ft.Container(expand=True),
            ft.Text(f"level {lvl}" if lvl >= 0 else "", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(
                ""
                if record is None
                else (record.diagnostics.level + (" (derived)" if record.replay is not None else "")),
                size=12,
                color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.IconButton(
                icon=ft.Icons.MENU_BOOK,
                tooltip="explain: the concepts of this level, at your depth",
                selected=store.learner.explain_is_open(level),
                on_click=toggle_explain,
            ),
        ],
        spacing=4,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


@ft.component
def Level3Placeholder(
    store: Store, record: Record | None, pulse: str, sample: str, index: ProvenanceIndex
) -> ft.Control:
    body = ft.Text(
        "Inside one pulse: populations and coherences against time, the phase-space loops, the Fock distributions, the jumps. The "
        "re-simulation engine and its cache exist (M11.1); this screen arrives with milestone M11.3. The numerics strip below already "
        "reports the recorded run.",
        size=13,
    )
    return ft.Column(
        [
            level_header(
                3, "The dynamics", "What happened inside one pulse?", f"pulse {pulse}, sample {sample}"
            ),
            card("Arrives in M11.3", body),
        ],
        spacing=12,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )


@ft.component
def DevicePagePlaceholder(store: Store, page_name: str, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    record = store.record()
    body: list[ft.Control] = [
        ft.Text(
            f"The {page_name} page of Level 4 (species, trap, crystal, light, noise, cooling, readout, the Hamiltonian builder) arrives with milestone M11.3; a parameter change there re-derives every level above. What the record already holds about the device is shown here.",
            size=13,
        )
    ]
    if record is not None:
        body.append(DeviceCardView(record, index))
    return ft.Column(
        [
            level_header(4, "The physics", "Where do the numbers come from?", page_name),
            card("Arrives in M11.3", ft.Column(body, spacing=10)),
        ],
        spacing=12,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )


@ft.component
def RoutedContent(store: Store, session: Session, index: ProvenanceIndex, path: str) -> ft.Control:
    """The page for ``path``; the path is a prop rather than a hook so that a route change re-renders this component."""
    ft.use_state(store)
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == "learn":
        return LearnPage(store, session, index)
    if parts and parts[0] == "device":
        return DevicePagePlaceholder(store, parts[1] if len(parts) > 1 else "hamiltonian", index)
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
            return Level1Page(store, record, parts[3], index)
        if len(parts) >= 4 and parts[2] == "schedule":
            return Level2Page(store, record, parts[3], index)
        if len(parts) >= 5 and parts[2] == "dynamics":
            return Level3Placeholder(store, record, parts[3], parts[4], index)
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
        if i == 5:
            page.navigate("/learn")
        elif i == 4:
            page.navigate("/device/hamiltonian")
        elif key is None:
            page.navigate("/")
        elif i == 0:
            page.navigate(f"/job/{key}")
        elif i == 1 and record is not None and record.schedule.targets:
            page.navigate(f"/job/{key}/circuit/{record.schedule.targets[0].gate_id}")
        elif i == 2 and record is not None and record.schedule.pulses:
            page.navigate(f"/job/{key}/schedule/0")
        elif i == 3 and record is not None and record.schedule.pulses:
            page.navigate(f"/job/{key}/dynamics/0/0")

    def on_key(e: ft.KeyboardEvent) -> None:
        up = parent_route(store, path)
        down = child_route(store, path)
        if e.key == "Escape" and up:
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
    if record is not None and lvl >= 0:
        plan = store.learner.plan(level)
        strip = numerics_strip(
            numerics_panel(record),
            index,
            expanded=plan.numerics_open if store.numerics_open is None else store.numerics_open,
            on_change=toggle_numerics,
        )
    explain_open = lvl >= 0 and store.learner.explain_is_open(level)
    columns: list[ft.Control] = [
        rail,
        ft.VerticalDivider(width=1),
        ft.Column(
            [
                ZoomBar(store, session, path),
                ft.Container(content=content, expand=True, padding=ft.Padding.symmetric(horizontal=8)),
                strip,
            ],
            expand=True,
            spacing=6,
        ),
    ]
    if explain_open:
        columns.append(ExplainDrawer(store, session, level, index, LEVEL_SECTIONS.get(level, "4.3.1")))
    dialog = None
    # asked once per device: the saved learner is read first, so a returning learner never sees the question again
    if store.learner_loaded and not store.learner.asked:

        def choose(k: str) -> None:
            session.set_learner(knowledge=k, asked=True)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Who is learning?"),
            content=ft.Column(
                [
                    ft.Text(
                        "The app adapts how much it explains. You can change this any time in Learn.", size=13
                    )
                ]
                + [
                    ft.FilledButton(content=ft.Text(v), on_click=lambda e, k=k: choose(k))
                    for k, v in KNOWLEDGE_LABELS.items()
                    if k != "unknown"
                ],
                tight=True,
                spacing=8,
            ),
            actions=[ft.TextButton(content=ft.Text("Skip: assist me"), on_click=lambda e: choose("unknown"))],
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
        ],
        not_found=layout,
    )


__all__ = ["App", "DEVICE_PAGES", "LEVEL_SECTIONS", "Shell", "child_route", "level_of", "parent_route"]
