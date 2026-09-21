"""The circuit builder (DESIGN.md Sections 5 and 11, R16): Level 0's circuit as wires and tiles in place of a text box.

The qubits are horizontal wires labelled q0, q1, ...; each gate is a tile in a column (the layout of ``viewmodel.builder``),
a two-qubit gate a connector between its two ends (a dot and a ring for CNOT, two dots for CZ, two crosses for SWAP, two
labelled tiles otherwise). A gate arrives by a tap on the palette (it lands at the end of the selected wire), by a drag
from the palette onto a wire or onto the gate it should precede, or from the ``+`` at the end of a wire, which lists the
palette. A placed tile is dragged to move it; a tap selects it and opens the inspector beneath the wires: its qubits as
dropdowns, its angles as fields that take any OpenQASM 2 expression with quick picks beside them, and earlier, later and
delete. The OpenQASM 2 text the store keeps, and an import of OpenQASM 2 or IonQ JSON, live behind the Code disclosure.

Colour follows the palette's families (fixed one-qubit gates on the primary container, rotations on the tertiary, two-qubit
gates on the secondary, the native set outlined) and never carries meaning alone: every tile prints its name. Tiles are
buttons, so Tab reaches them; every edit goes through ``Session.edit_circuit`` and Undo takes it back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import flet as ft

from qutip_trap_app.core import Circuit
from qutip_trap_app.viewmodel import builder as vm
from qutip_trap_app.viewmodel.builder import GateSpec, Layout, Placed
from qutip_trap_app.views import theme
from qutip_trap_app.views.common import (
    HAIRLINE,
    MUTED,
    TWO_COLUMN_MIN_WIDTH,
    content_width,
    details,
    small_icon_button,
    status_line,
)
from qutip_trap_app.views.state import Session, Store

CELL_W = 52.0
"""The column pitch of the grid."""
ROW_H = 48.0
"""The wire pitch."""
TILE_W = 40.0
TILE_H = 34.0
LABEL_W = 44.0
"""The wire-label column (q0, q1, ...)."""
MIN_COLUMNS = 6
"""The grid is never narrower than this many columns, so an empty circuit still reads as wires."""
DRAG_GROUP = "gate"
END_SIZE = 28.0
"""The round end of a two-qubit gate (a dot, a ring, a cross)."""

FAMILY_COLORS: dict[str, tuple[str, str]] = {
    "fixed": (ft.Colors.PRIMARY_CONTAINER, ft.Colors.ON_PRIMARY_CONTAINER),
    "rotation": (ft.Colors.TERTIARY_CONTAINER, ft.Colors.ON_TERTIARY_CONTAINER),
    "pair": (ft.Colors.SECONDARY_CONTAINER, ft.Colors.ON_SECONDARY_CONTAINER),
    "native": (ft.Colors.SURFACE_CONTAINER_HIGHEST, ft.Colors.ON_SURFACE),
    "other": (ft.Colors.SURFACE_CONTAINER_HIGH, ft.Colors.ON_SURFACE_VARIANT),
}
"""(background, text) per palette family: the Material container roles, so both schemes read (R9)."""

QUICK_ANGLES: tuple[tuple[str, float], ...] = (
    ("-π/2", -1.5707963267948966),
    ("π/4", 0.7853981633974483),
    ("π/2", 1.5707963267948966),
    ("π", 3.141592653589793),
)
"""The quick picks beside a single angle field."""

Handler = Callable[[Any], None]


# ---- tiles ---------------------------------------------------------------------------------------------------------------------------


def _label_size(text: str) -> int:
    return 13 if len(text) <= 2 else 11 if len(text) <= 3 else 10


def _tile_face(spec: GateSpec, label: str, angle: str | None, fg: str) -> ft.Control:
    """The face of a tile: the gate's name and, for a parametrised gate, its angle beneath in a smaller size."""
    lines: list[ft.Control] = [
        ft.Text(label, size=_label_size(label), weight=ft.FontWeight.W_600, color=fg, no_wrap=True)
    ]
    if angle is not None:
        lines.append(ft.Text(angle, size=9, color=fg, no_wrap=True))
    return ft.Column(
        lines,
        spacing=0,
        tight=True,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER,
    )


def _tile_angle(spec: GateSpec, params: Sequence[float]) -> str | None:
    if spec.shown_param is None or spec.shown_param >= len(params):
        return None
    return vm.format_angle(params[spec.shown_param])


def _tile_style(spec: GateSpec, *, selected: bool) -> ft.ButtonStyle:
    bg, fg = FAMILY_COLORS[spec.family]
    side: ft.BorderSide | None = None
    if selected:
        side = ft.BorderSide(2, ft.Colors.PRIMARY)
    elif spec.family == "native":
        side = ft.BorderSide(1, ft.Colors.OUTLINE)
    return ft.ButtonStyle(
        bgcolor=bg,
        color=fg,
        padding=ft.Padding.all(0),
        shape=ft.RoundedRectangleBorder(radius=theme.RADIUS_TILE, side=side),
        overlay_color=ft.Colors.with_opacity(0.08, ft.Colors.ON_SURFACE),
        visual_density=ft.VisualDensity.COMPACT,
    )


def tile(
    spec: GateSpec,
    *,
    label: str | None = None,
    params: Sequence[float] = (),
    selected: bool = False,
    on_click: Handler | None = None,
    tooltip: str | None = None,
    key: str | None = None,
    width: float = TILE_W,
    height: float = TILE_H,
) -> ft.Control:
    """A gate tile: a button (focusable, with hover and press feedback) when it has a click, else a plain surface (the
    drag feedback, the inspector's glyph). The family colours the tile; the name printed on it carries the meaning."""
    bg, fg = FAMILY_COLORS[spec.family]
    face = _tile_face(spec, spec.label if label is None else label, _tile_angle(spec, params), fg)
    if on_click is None:
        side = ft.BorderSide(2, ft.Colors.PRIMARY) if selected else None
        if side is None and spec.family == "native":
            side = ft.BorderSide(1, ft.Colors.OUTLINE)
        return ft.Container(
            content=face,
            width=width,
            height=height,
            bgcolor=bg,
            border=ft.Border.all(side.width, side.color) if side is not None else None,
            border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
            alignment=ft.Alignment.CENTER,
            tooltip=tooltip,
            key=key,
        )
    return ft.TextButton(
        content=face,
        on_click=on_click,
        style=_tile_style(spec, selected=selected),
        width=width,
        height=height,
        tooltip=tooltip,
        key=key,
    )


def _end_glyph(kind: str, *, selected: bool, on_click: Handler, tooltip: str, key: str | None) -> ft.Control:
    """The round end of a two-qubit gate: a filled dot (a control), a ring with a plus (a CNOT target) or a cross (SWAP),
    in the muted ink, the primary colour when selected. A button like every tile."""
    ink = ft.Colors.PRIMARY if selected else ft.Colors.ON_SURFACE_VARIANT
    face: ft.Control
    if kind == "dot":
        face = ft.Container(width=12, height=12, bgcolor=ink, border_radius=ft.BorderRadius.all(6))
    elif kind == "plus":
        face = ft.Container(
            content=ft.Icon(ft.Icons.ADD, size=16, color=ink),
            width=22,
            height=22,
            border=ft.Border.all(2, ink),
            border_radius=ft.BorderRadius.all(11),
            alignment=ft.Alignment.CENTER,
        )
    else:
        face = ft.Icon(ft.Icons.CLOSE, size=20, color=ink)
    return ft.TextButton(
        content=face,
        on_click=on_click,
        tooltip=tooltip,
        key=key,
        width=END_SIZE,
        height=END_SIZE,
        style=ft.ButtonStyle(
            padding=ft.Padding.all(0),
            shape=ft.CircleBorder(),
            bgcolor=ft.Colors.with_opacity(0.10, ft.Colors.PRIMARY) if selected else None,
            overlay_color=ft.Colors.with_opacity(0.08, ft.Colors.ON_SURFACE),
            visual_density=ft.VisualDensity.COMPACT,
        ),
    )


def _signature(spec: GateSpec) -> str:
    return spec.label if not spec.params else f"{spec.label}({', '.join(spec.params)})"


# ---- the grid -----------------------------------------------------------------------------------------------------------------------


def _x(column: int) -> float:
    return column * CELL_W + (CELL_W - TILE_W) / 2.0


def _y(wire: int) -> float:
    return wire * ROW_H + (ROW_H - TILE_H) / 2.0


def _positioned(control: ft.Control, left: float, top: float, width: float, height: float) -> ft.Control:
    return ft.Container(
        content=control, left=left, top=top, width=width, height=height, alignment=ft.Alignment.CENTER
    )


def _connector(placed: Placed, *, selected: bool) -> ft.Control | None:
    """The 2 px line between a two-qubit gate's ends, in the muted ink (the primary colour when selected); drawn under
    the drop layer so that a drop on it still reaches the wire."""
    if len(placed.op.qubits) < 2:
        return None
    return ft.Container(
        left=placed.column * CELL_W + CELL_W / 2.0 - 1.0,
        top=placed.lo * ROW_H + ROW_H / 2.0,
        width=2,
        height=(placed.hi - placed.lo) * ROW_H,
        bgcolor=ft.Colors.PRIMARY if selected else ft.Colors.ON_SURFACE_VARIANT,
    )


def _placed_controls(
    placed: Placed,
    *,
    selected: bool,
    select: Callable[[int], None],
    accept: Callable[[ft.DragTargetEvent, int, int], None],
) -> list[ft.Control]:
    """One placed gate on the grid: an end per qubit, each end a drag target (a drop lands before this gate on that wire)
    wrapping a draggable (the gate moves) wrapping the tile button."""
    op, spec, col = placed.op, placed.spec, placed.column
    text = vm.describe(op)
    out: list[ft.Control] = []
    for slot, q in enumerate(op.qubits):
        end = spec.ends[slot] if spec.ends is not None and slot < 2 else None
        key = f"builder-gate:{placed.index}" if slot == 0 else None
        face: ft.Control
        w, h = TILE_W, TILE_H
        if end in ("dot", "plus", "cross"):
            face = _end_glyph(
                end, selected=selected, on_click=lambda e: select(placed.index), tooltip=text, key=key
            )
            w = h = END_SIZE
        else:
            face = tile(
                spec,
                label=end if end is not None else None,
                params=op.params,
                selected=selected,
                on_click=lambda e: select(placed.index),
                tooltip=text,
                key=key,
            )
        draggable = ft.Draggable(
            content=face,
            group=DRAG_GROUP,
            data=f"placed:{placed.index}",
            content_feedback=ft.Container(
                content=tile(
                    spec, label=end if end not in (None, "dot", "plus", "cross") else None, params=op.params
                ),
                opacity=0.85,
            ),
            content_when_dragging=ft.Container(content=face, opacity=0.3),
        )
        target = ft.DragTarget(
            content=draggable,
            group=DRAG_GROUP,
            on_accept=lambda e, wire=q, column=col: accept(e, wire, column),
        )
        out.append(
            _positioned(target, col * CELL_W + (CELL_W - w) / 2.0, q * ROW_H + (ROW_H - h) / 2.0, w, h)
        )
    return out


def _plus_menu(wire: int, selected_wire: bool, pick: Callable[[GateSpec, int], None]) -> ft.Control:
    """The ``+`` at the end of a wire: a menu of the palette, grouped; a choice lands the gate at the end of the wire.
    On the selected wire it is drawn in the primary colour, so the wire a tapped palette gate goes to is visible."""
    ink = ft.Colors.PRIMARY if selected_wire else MUTED
    items: list[ft.PopupMenuItem] = []
    for family in vm.families():
        items.append(
            ft.PopupMenuItem(
                content=ft.Text(vm.FAMILY_TITLES[family], size=theme.SIZE_CAPTION, color=MUTED),
                disabled=True,
                height=28,
            )
        )
        for spec in vm.palette_group(family):
            items.append(
                ft.PopupMenuItem(
                    content=ft.Row(
                        [
                            tile(spec, width=32, height=26),
                            ft.Text(_signature(spec), size=theme.SIZE_SMALL, weight=ft.FontWeight.W_500),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    on_click=lambda e, s=spec, w=wire: pick(s, w),
                    tooltip=spec.title,
                    height=36,
                )
            )
    return ft.PopupMenuButton(
        content=ft.Container(
            content=ft.Icon(ft.Icons.ADD, size=18, color=ink),
            width=TILE_H,
            height=TILE_H,
            border=ft.Border.all(1.5 if selected_wire else 1, ink if selected_wire else HAIRLINE),
            border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
            alignment=ft.Alignment.CENTER,
        ),
        items=items,
        tooltip=f"add a gate at the end of q{wire}",
        key=f"builder-add:{wire}",
    )


def _wire_labels(n: int, selected_wire: int, select_wire: Callable[[int], None]) -> ft.Control:
    """The q0, q1, ... column: each label a button that makes its wire the one a tapped palette gate goes to; the selected
    label sits on the secondary container, the app's selection colour (R9)."""
    labels: list[ft.Control] = []
    for w in range(n):
        active = w == selected_wire
        labels.append(
            ft.Container(
                content=ft.TextButton(
                    content=ft.Text(
                        f"q{w}",
                        size=theme.SIZE_SMALL,
                        weight=ft.FontWeight.W_600 if active else ft.FontWeight.W_400,
                        color=ft.Colors.ON_SECONDARY_CONTAINER if active else MUTED,
                    ),
                    on_click=lambda e, ww=w: select_wire(ww),
                    tooltip=f"qubit {w}: new gates go on the selected wire",
                    style=ft.ButtonStyle(
                        padding=ft.Padding.symmetric(horizontal=6, vertical=0),
                        bgcolor=ft.Colors.SECONDARY_CONTAINER if active else None,
                        shape=ft.StadiumBorder(),
                        visual_density=ft.VisualDensity.COMPACT,
                    ),
                    width=LABEL_W - 6,
                    height=24,
                    key=f"builder-wire:{w}",
                ),
                height=ROW_H,
                width=LABEL_W,
                alignment=ft.Alignment.CENTER,
            )
        )
    return ft.Column(labels, spacing=0, tight=True)


def _grid(
    lay: Layout,
    *,
    room: float,
    selected: int,
    selected_wire: int,
    select: Callable[[int], None],
    select_wire: Callable[[int], None],
    accept: Callable[[ft.DragTargetEvent, int, int], None],
    pick: Callable[[GateSpec, int], None],
) -> ft.Control:
    """The wires with their gates and the ``+`` at the end of each, as one absolutely positioned stack: the wire lines
    and the connectors at the bottom, one transparent drop target per wire over them (a coloured box is opaque to
    Flutter's hit test, so the lines must lie under the layer that takes the drop), the tiles above that, the ``+`` menus
    on top. The wires run across the room the card gives them (``room``) and the stack scrolls sideways when the circuit
    is wider than that."""
    n = lay.n_qubits
    columns = max(lay.n_columns + 1, MIN_COLUMNS)
    width = max(columns * CELL_W, room)
    height = n * ROW_H
    children: list[ft.Control] = []
    for w in range(n):
        children.append(
            ft.Container(
                left=0, top=w * ROW_H + ROW_H / 2.0 - 0.5, width=width, height=1, bgcolor=ft.Colors.OUTLINE
            )
        )
    for p in lay.placed:
        connector = _connector(p, selected=p.index == selected)
        if connector is not None:
            children.append(connector)
    for w in range(n):

        def on_accept(e: ft.DragTargetEvent, wire: int = w) -> None:
            accept(e, wire, int(e.local_position.x // CELL_W))

        def on_wire(_e: Any, wire: int = w) -> None:
            select_wire(wire)

        children.append(
            ft.Container(
                left=0,
                top=w * ROW_H,
                width=width,
                height=ROW_H,
                content=ft.DragTarget(
                    content=ft.Container(width=width, height=ROW_H, on_click=on_wire),
                    group=DRAG_GROUP,
                    on_accept=on_accept,
                ),
            )
        )
    for p in lay.placed:
        children.extend(_placed_controls(p, selected=p.index == selected, select=select, accept=accept))
    for w in range(n):
        children.append(
            _positioned(
                _plus_menu(w, w == selected_wire, pick),
                lay.n_columns * CELL_W + (CELL_W - TILE_H) / 2.0,
                w * ROW_H + (ROW_H - TILE_H) / 2.0,
                TILE_H,
                TILE_H,
            )
        )
    stack = ft.Stack(children, width=width, height=height, key="builder-grid")
    return ft.Row(
        [
            _wire_labels(n, selected_wire, select_wire),
            ft.Row([stack], scroll=ft.ScrollMode.AUTO, expand=True, spacing=0),
        ],
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


# ---- the palette --------------------------------------------------------------------------------------------------------------------


def _palette(pick: Callable[[GateSpec], None]) -> ft.Control:
    """The gates in their families; each a tile that can be tapped (it goes to the end of the selected wire) or dragged
    onto a wire."""
    groups: list[ft.Control] = []

    def picker(spec: GateSpec) -> Handler:
        return lambda e: pick(spec)

    for family in vm.families():
        tiles: list[ft.Control] = []
        for spec in vm.palette_group(family):
            tiles.append(
                ft.Draggable(
                    content=tile(
                        spec,
                        params=spec.defaults,
                        on_click=picker(spec),
                        tooltip=f"{spec.title}\ntap: add to the selected wire; drag: place it",
                        key=f"palette:{spec.name}",
                    ),
                    group=DRAG_GROUP,
                    data=f"palette:{spec.name}",
                    content_feedback=ft.Container(content=tile(spec, params=spec.defaults), opacity=0.85),
                )
            )
        groups.append(
            ft.Column(
                [
                    ft.Text(vm.FAMILY_TITLES[family], size=theme.SIZE_CAPTION, color=MUTED),
                    ft.Row(tiles, spacing=6, run_spacing=6, wrap=True),
                ],
                spacing=4,
                tight=True,
            )
        )
    return ft.Row(
        groups, spacing=20, run_spacing=10, wrap=True, vertical_alignment=ft.CrossAxisAlignment.START
    )


# ---- the inspector -----------------------------------------------------------------------------------------------------------------


def _qubit_dropdown(
    label: str, value: int, n: int, on_select: Handler, *, key: str | None = None
) -> ft.Control:
    return ft.Dropdown(
        label=label,
        value=str(value),
        options=[ft.DropdownOption(key=str(q), text=f"q{q}") for q in range(n)],
        on_select=on_select,
        width=92,
        dense=True,
        text_size=13,
        border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
        key=key,
    )


def _inspector(
    circuit: Circuit,
    k: int,
    *,
    errors: dict[str, str],
    set_qubits: Callable[[int, int, int], None],
    swap_qubits: Callable[[int], None],
    set_angle: Callable[[int, int, str], None],
    move: Callable[[int, int], None],
    delete: Callable[[int], None],
) -> ft.Control:
    """The selected gate: its glyph and description, its qubits, its angles, and earlier, later, delete."""
    op = circuit.ops[k]
    spec = vm.spec_of(op)
    n = circuit.n_qubits
    before, after = vm.neighbours(circuit, k)
    items: list[ft.Control] = [
        tile(spec, params=op.params, width=32, height=28),
        ft.Column(
            [
                ft.Text(vm.describe(op), size=theme.SIZE_SMALL + 1, weight=ft.FontWeight.W_600),
                ft.Text(spec.title, size=theme.SIZE_CAPTION, color=MUTED, max_lines=2),
            ],
            spacing=1,
            tight=True,
            width=230,
        ),
    ]
    if len(op.qubits) == 1:
        items.append(
            _qubit_dropdown(
                "qubit",
                op.qubits[0],
                n,
                lambda e: set_qubits(k, 0, int(str(e.control.value))),
                key="builder-qubit:0",
            )
        )
    elif len(op.qubits) == 2:
        names = ("control", "target") if spec.name in ("cnot", "cx", "cp") else ("qubit 1", "qubit 2")
        items.append(
            _qubit_dropdown(
                names[0],
                op.qubits[0],
                n,
                lambda e: set_qubits(k, 0, int(str(e.control.value))),
                key="builder-qubit:0",
            )
        )
        items.append(
            small_icon_button(ft.Icons.SWAP_HORIZ, "exchange the two qubits", lambda e: swap_qubits(k))
        )
        items.append(
            _qubit_dropdown(
                names[1],
                op.qubits[1],
                n,
                lambda e: set_qubits(k, 1, int(str(e.control.value))),
                key="builder-qubit:1",
            )
        )
    for i, name in enumerate(spec.params):
        err = errors.get(f"{k}:{i}")
        items.append(
            ft.TextField(
                label=name,
                value=vm.format_angle(op.params[i]) if i < len(op.params) else "",
                width=104,
                dense=True,
                text_size=13,
                border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
                on_submit=lambda e, ii=i: set_angle(k, ii, str(e.control.value)),
                on_blur=lambda e, ii=i: set_angle(k, ii, str(e.control.value)),
                error=err,
                tooltip="radians; pi/2, -3*pi/4, 0.35 and sqrt(2) all work",
                key=f"builder-angle:{i}",
            )
        )
    if len(spec.params) == 1:
        items.append(
            ft.Row(
                [
                    ft.TextButton(
                        content=ft.Text(label, size=theme.SIZE_SMALL),
                        on_click=lambda e, v=value: set_angle(k, 0, vm.qasm_angle(v)),
                        style=ft.ButtonStyle(
                            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                            visual_density=ft.VisualDensity.COMPACT,
                        ),
                    )
                    for label, value in QUICK_ANGLES
                ],
                spacing=0,
                tight=True,
            )
        )
    items.append(
        ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.KEYBOARD_ARROW_LEFT,
                    icon_size=18,
                    tooltip="earlier: before the previous gate on these wires",
                    on_click=lambda e: move(k, -1),
                    disabled=before is None,
                    visual_density=ft.VisualDensity.COMPACT,
                    key="builder-earlier",
                ),
                ft.IconButton(
                    icon=ft.Icons.KEYBOARD_ARROW_RIGHT,
                    icon_size=18,
                    tooltip="later: after the next gate on these wires",
                    on_click=lambda e: move(k, +1),
                    disabled=after is None,
                    visual_density=ft.VisualDensity.COMPACT,
                    key="builder-later",
                ),
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE,
                    icon_size=18,
                    tooltip="remove this gate",
                    on_click=lambda e: delete(k),
                    visual_density=ft.VisualDensity.COMPACT,
                    key="builder-delete",
                ),
            ],
            spacing=0,
            tight=True,
        )
    )
    return ft.Container(
        content=ft.Row(
            items, spacing=12, run_spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER
        ),
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border=ft.Border.all(1, HAIRLINE),
        border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
        key="builder-inspector",
    )


# ---- the code -----------------------------------------------------------------------------------------------------------------------


def _code_panel(
    store: Store, session: Session, import_text: str, set_import_text: Callable[[str], None]
) -> list[ft.Control]:
    """The text the store keeps (read-only, in the code face) and the import field: paste OpenQASM 2 or IonQ JSON, the
    format detected, the parse error shown beside the field when it fails."""

    def do_import(_e: Any) -> None:
        text = import_text.strip()
        if not text:
            return
        fmt = vm.detect_format(text)
        try:
            parsed = vm.parse_circuit_text(text, fmt)
        except Exception as exc:  # a syntax error in the pasted program: shown, never a crash
            store.error = f"the pasted circuit could not be read: {exc}"
            return
        if parsed.n_qubits > vm.MAX_QUBITS:
            # the same ceiling the + at the wires' end enforces: every ion adds modes and joint dimension
            store.error = f"the pasted circuit has {parsed.n_qubits} qubits; at most {vm.MAX_QUBITS} run here"
            return
        session.edit_circuit(text + ("" if text.endswith("\n") else "\n"), fmt)
        set_import_text("")

    return [
        # a Row takes the card's width, so the code box spans it like the import field beneath
        ft.Row(
            [
                ft.Container(
                    content=ft.Text(
                        store.circuit_text.rstrip("\n"),
                        size=theme.SIZE_SMALL,
                        font_family=theme.CODE_FONT,
                        font_family_fallback=theme.CODE_FONT_FALLBACK,
                        selectable=True,
                        key="builder-code",
                    ),
                    padding=ft.Padding.all(12),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                    border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
                    expand=True,
                )
            ]
        ),
        status_line(f"format: {'IonQ JSON' if store.circuit_format == 'ionq_json' else 'OpenQASM 2'}"),
        ft.Row(
            [
                ft.TextField(
                    value=import_text,
                    hint_text="paste OpenQASM 2 or IonQ JSON",
                    multiline=True,
                    min_lines=2,
                    max_lines=8,
                    dense=True,
                    text_size=theme.SIZE_SMALL,
                    text_style=ft.TextStyle(
                        size=theme.SIZE_SMALL,
                        font_family=theme.CODE_FONT,
                        font_family_fallback=theme.CODE_FONT_FALLBACK,
                    ),
                    border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
                    on_change=lambda e: set_import_text(str(e.control.value)),
                    expand=True,
                    key="builder-import-text",
                ),
                ft.FilledTonalButton(
                    content=ft.Text("Import"),
                    icon=ft.Icons.INPUT,
                    on_click=do_import,
                    disabled=not import_text.strip(),
                    tooltip="replace the circuit with the pasted program (Undo takes it back)",
                    key="builder-import",
                ),
            ],
            spacing=theme.GAP,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
    ]


# ---- the component -------------------------------------------------------------------------------------------------------------------


def wire_room(store: Store, page: Any) -> float:
    """How wide the wires can be without scrolling: the Circuit card's inner width less the label column. The card is the
    whole content column before a run and two thirds of it beside the results afterwards (``common.columns``)."""
    content = content_width(store, page, 0)
    if store.record() is not None and content >= TWO_COLUMN_MIN_WIDTH:
        content = (content - 16.0) * 2.0 / 3.0
    # the card's border and the scroll view's own edges take a few pixels: without the slack a scrollbar appears
    return max(content - 2 * theme.CARD_PADDING - LABEL_W - 8.0, MIN_COLUMNS * CELL_W)


def builder_toolbar(store: Store, session: Session) -> list[ft.Control]:
    """Undo, for the Circuit card's title row (Clear sits with the qubit buttons under the wires)."""

    def undo(_e: Any) -> None:
        session.undo_circuit()

    return [
        small_icon_button(
            ft.Icons.UNDO,
            "undo the last change to the circuit" if store.circuit_undo else "nothing to undo",
            undo if store.circuit_undo else None,
        ),
    ]


@ft.component
def CircuitBuilder(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    selected, set_selected = ft.use_state(-1)
    wire, set_wire = ft.use_state(0)
    no_errors: dict[str, str] = {}
    errors, set_errors = ft.use_state(no_errors)
    import_text, set_import_text = ft.use_state("")

    try:
        circuit = session.parse_circuit()
    except (
        Exception
    ) as exc:  # the text (an import, a preset) is not a circuit: say so, offer the code and a fresh start
        return _unreadable(store, session, str(exc), import_text, set_import_text)
    lay = vm.layout(circuit)
    n = circuit.n_qubits
    cur_wire = min(max(wire, 0), n - 1)
    sel = selected if 0 <= selected < len(circuit.ops) else -1

    def commit(new: Circuit, select: int = -1) -> None:
        session.edit_circuit(vm.to_openqasm2(new))
        set_selected(select)
        set_errors({})

    def fail(exc: Exception) -> None:
        store.error = str(exc)

    def qubits_for(spec: GateSpec, at_wire: int, c: Circuit) -> tuple[Circuit, tuple[int, ...]]:
        """The circuit (a second qubit added when a two-qubit gate meets a single wire) and the gate's qubits."""
        if spec.arity == 2 and c.n_qubits < 2:
            c = vm.add_qubit(c)
        return c, vm.qubits_for_new(spec, at_wire, c.n_qubits)

    def pick(spec: GateSpec, at_wire: int | None = None) -> None:
        w = cur_wire if at_wire is None else at_wire
        try:
            c, qubits = qubits_for(spec, w, circuit)
            commit(vm.append_gate(c, spec, qubits), len(c.ops))
        except ValueError as exc:
            fail(exc)
        set_wire(w)

    def accept(e: ft.DragTargetEvent, at_wire: int, column: int) -> None:
        data = str(getattr(e.src, "data", "") or "")
        kind, _, what = data.partition(":")
        try:
            if kind == "palette" and what in vm.GATES:
                spec = vm.GATES[what]
                c, qubits = qubits_for(spec, at_wire, circuit)
                at = vm.insert_index(vm.layout(c), at_wire, column)
                commit(vm.insert_gate(c, at, spec, qubits), at)
            elif kind == "placed":
                k = int(what)
                op = circuit.ops[k]
                at = vm.insert_index(lay, at_wire, column)
                moved, new_k = vm.move_to(circuit, k, at, vm.qubits_for_move(op, at_wire, n))
                commit(moved, new_k)
        except (ValueError, IndexError) as exc:
            fail(exc)
        set_wire(at_wire)

    def set_qubits(k: int, slot: int, q: int) -> None:
        qubits = list(circuit.ops[k].qubits)
        qubits[slot] = q
        commit(vm.set_qubits(circuit, k, qubits), k)

    def swap_qubits(k: int) -> None:
        a, b = circuit.ops[k].qubits
        commit(vm.set_qubits(circuit, k, (b, a)), k)

    def set_angle(k: int, i: int, text: str) -> None:
        try:
            value = vm.parse_angle(text)
        except ValueError as exc:
            set_errors({**errors, f"{k}:{i}": str(exc)})
            return
        if abs(value - circuit.ops[k].params[i]) < 1e-15:
            if f"{k}:{i}" in errors:
                set_errors({key: v for key, v in errors.items() if key != f"{k}:{i}"})
            return
        commit(vm.set_param(circuit, k, i, value), k)

    def move(k: int, direction: int) -> None:
        moved, new_k = vm.move_gate(circuit, k, direction)
        if new_k != k:
            commit(moved, new_k)

    def delete(k: int) -> None:
        commit(vm.remove_gate(circuit, k))

    def add_qubit(_e: Any) -> None:
        try:
            commit(vm.add_qubit(circuit), sel)
            set_wire(n)
        except ValueError as exc:
            fail(exc)

    def remove_qubit(_e: Any) -> None:
        try:
            commit(vm.remove_qubit(circuit))
            set_wire(min(cur_wire, n - 2))
        except ValueError as exc:
            fail(exc)

    def clear_all(_e: Any) -> None:
        commit(vm.clear(circuit))

    body: list[ft.Control] = [
        _palette(pick),
        _grid(
            lay,
            room=wire_room(store, ft.context.page),
            selected=sel,
            selected_wire=cur_wire,
            select=lambda k: set_selected(-1 if k == sel else k),
            select_wire=set_wire,
            accept=accept,
            pick=pick,
        ),
    ]
    if sel >= 0:
        body.append(
            _inspector(
                circuit,
                sel,
                errors=errors,
                set_qubits=set_qubits,
                swap_qubits=swap_qubits,
                set_angle=set_angle,
                move=move,
                delete=delete,
            )
        )
    body.append(
        ft.Row(
            [
                ft.TextButton(
                    content=ft.Text("add qubit"),
                    icon=ft.Icons.ADD,
                    on_click=add_qubit,
                    disabled=n >= vm.MAX_QUBITS,
                    tooltip=f"a wire q{n} under the others ({vm.MAX_QUBITS} at most: every ion adds modes and cost)",
                    key="builder-add-qubit",
                ),
                ft.TextButton(
                    content=ft.Text("remove qubit"),
                    icon=ft.Icons.REMOVE,
                    on_click=remove_qubit,
                    disabled=n <= vm.MIN_QUBITS,
                    tooltip=f"drop q{n - 1} and every gate on it (Undo takes it back)",
                    key="builder-remove-qubit",
                ),
                ft.TextButton(
                    content=ft.Text("clear"),
                    icon=ft.Icons.DELETE_SWEEP_OUTLINED,
                    on_click=clear_all,
                    disabled=not circuit.ops,
                    tooltip="remove every gate and keep the wires (Undo takes it back)",
                    key="builder-clear",
                ),
            ],
            spacing=4,
            run_spacing=4,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )
    body.append(status_line(f"{vm.summary(lay)}; tapped gates go on q{cur_wire}"))
    body.append(
        details(
            "level0.code",
            _code_panel(store, session, import_text, set_import_text),
            store=store,
            level=0,
            session=session,
            title="Code",
        )
    )
    return ft.Column(body, spacing=12, horizontal_alignment=ft.CrossAxisAlignment.STRETCH, key="builder")


def _unreadable(
    store: Store, session: Session, message: str, import_text: str, set_import_text: Callable[[str], None]
) -> ft.Control:
    """The text is not a circuit the loaders read: the reason, the text itself, an import field and a fresh start."""

    def start_again(_e: Any) -> None:
        session.edit_circuit(vm.to_openqasm2(vm.empty_circuit(2)))

    def undo(_e: Any) -> None:
        session.undo_circuit()

    return ft.Column(
        [
            ft.Row(
                [
                    ft.Icon(ft.Icons.ERROR_OUTLINE, size=16, color=ft.Colors.ERROR),
                    ft.Text(
                        message,
                        color=ft.Colors.ERROR,
                        size=theme.SIZE_SMALL,
                        expand=True,
                        key="builder-unreadable",
                    ),
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            *_code_panel(store, session, import_text, set_import_text),
            ft.Row(
                [
                    ft.OutlinedButton(
                        content=ft.Text("Start again"),
                        icon=ft.Icons.RESTART_ALT,
                        on_click=start_again,
                        tooltip="two empty wires (Undo brings the text back)",
                    ),
                    small_icon_button(ft.Icons.UNDO, "undo the last change to the circuit", undo)
                    if store.circuit_undo
                    else ft.Container(),
                ],
                spacing=theme.GAP,
            ),
        ],
        spacing=12,
        key="builder",
    )


__all__ = [
    "CELL_W",
    "DRAG_GROUP",
    "FAMILY_COLORS",
    "MIN_COLUMNS",
    "QUICK_ANGLES",
    "ROW_H",
    "TILE_H",
    "TILE_W",
    "CircuitBuilder",
    "builder_toolbar",
    "tile",
    "wire_room",
]
