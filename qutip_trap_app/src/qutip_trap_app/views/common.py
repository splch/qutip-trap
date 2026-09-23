"""Shared Flet pieces: number formatting, cards, stat tiles, tables, disclosure, the progress card and the numerics strip."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import flet as ft

from qutip_trap_app.viewmodel.numerics import Badge, NumericsPanel
from qutip_trap_app.viewmodel.shown import Shown
from qutip_trap_app.views.state import Session, Store

MUTED = ft.Colors.ON_SURFACE_VARIANT
HAIRLINE = ft.Colors.OUTLINE_VARIANT
GAP = 8
PAGE_PADDING = 24
RAIL_WIDTH = 80
CONTENT_MAX_WIDTH = 1280.0
TWO_COLUMN_MIN_WIDTH = 960.0
CODE_FONT = "Menlo"
CODE_FONT_FALLBACK = ["Consolas", "DejaVu Sans Mono", "Courier New", "monospace"]

REQUEST_LABELS = {
    "run": "run",
    "zoom": "re-simulation",
    "tomography": "process tomography",
    "recheck": "convergence re-check",
}

# ---- formatting -------------------------------------------------------------------------------------------------------------------

_SI = (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12))


def fmt_number(value: float, unit: str = "", digits: int = 4) -> str:
    """A number with an SI prefix for Hz, s and m; scientific notation for small or large dimensionless numbers."""
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if value == 0.0:
        return f"0 {unit}".strip()
    if unit in ("Hz", "s", "m"):
        prefix, scale = next(((p, s) for p, s in _SI if abs(value) >= s), _SI[-1])
        return f"{value / scale:.{digits}g} {prefix}{unit}"
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.{max(digits - 1, 1)}e} {unit}".strip()
    return f"{value:.{digits}g} {unit}".strip()


def fmt(s: Shown) -> str:
    v = s.value
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v} {s.unit}".strip()
    if isinstance(v, float):
        return fmt_number(v, s.unit)
    return v


def ions_text(ions: Sequence[int]) -> str:
    """'ion 0', 'ions 0 and 1', 'ions 0, 1 and 2'."""
    items = [str(i) for i in ions]
    if len(items) == 1:
        return f"ion {items[0]}"
    return "ions " + ", ".join(items[:-1]) + f" and {items[-1]}"


def event_bool(e: Any) -> bool:
    """The boolean a change event carries (the client may send it as text)."""
    data = getattr(e, "data", None)
    return data if isinstance(data, bool) else str(data).strip().lower() == "true"


# ---- values -------------------------------------------------------------------------------------------------------------------------


def value_text(s: Shown, size: int = 12) -> ft.Control:
    return ft.Text(fmt(s), size=size, tooltip=s.detail or None, selectable=True)


def shown(s: Shown, size: int = 13) -> ft.Control:
    """A value inline, its label first."""
    return ft.Row(
        [ft.Text(s.label, size=size - 1, color=MUTED), value_text(s, size)],
        spacing=6,
        wrap=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def stat_tile(s: Shown) -> ft.Control:
    """A value large with its label small beneath and its detail on hover."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    fmt(s), size=20, weight=ft.FontWeight.W_600, tooltip=s.detail or None, selectable=True
                ),
                ft.Text(s.label, size=12, color=MUTED, width=160),
            ],
            spacing=2,
            tight=True,
        ),
        padding=ft.Padding.symmetric(horizontal=8, vertical=6),
    )


def stat_row(tiles: Sequence[ft.Control]) -> ft.Control:
    return ft.Row(list(tiles), wrap=True, spacing=2 * GAP, run_spacing=GAP)


def status_line(text: str) -> ft.Text:
    return ft.Text(text, size=12, color=MUTED)


# ---- layout -------------------------------------------------------------------------------------------------------------------------


def content_width(store: Store) -> float:
    """The width of the content column: the window less the rail and the margins, capped at a reading width."""
    return min(max(store.width - RAIL_WIDTH - 2 * PAGE_PADDING, 360.0), CONTENT_MAX_WIDTH)


def columns(
    store: Store, left: Sequence[ft.Control], right: Sequence[ft.Control], split: tuple[int, int] = (1, 1)
) -> ft.Control:
    """Two stacks of cards side by side when the content column is wide enough, else one above the other."""
    stacks = [list(left), list(right)]
    if content_width(store) < TWO_COLUMN_MIN_WIDTH or not all(stacks):
        return ft.Column(
            stacks[0] + stacks[1], spacing=16, horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )
    return ft.Row(
        [
            ft.Column(stack, spacing=16, expand=weight, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
            for stack, weight in zip(stacks, split)
        ],
        spacing=16,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def screen(controls: Sequence[ft.Control]) -> ft.Control:
    return ft.Column(list(controls), spacing=16, expand=True, scroll=ft.ScrollMode.AUTO)


def level_header(title: str, question: str, note: str = "") -> ft.Control:
    lines: list[ft.Control] = [
        ft.Text(title, theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
        ft.Text(question, size=14, color=MUTED),
    ]
    if note:
        lines.append(ft.Text(note, size=12, color=MUTED))
    return ft.Column(lines, spacing=2)


def card(
    title: str, body: ft.Control, *, actions: Sequence[ft.Control] = (), trailing: Sequence[ft.Control] = ()
) -> ft.Control:
    head: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600, size=15, expand=True), *trailing]
    content: list[ft.Control] = [
        ft.Row(head, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        body,
    ]
    if actions:
        content.append(
            ft.Row(list(actions), spacing=GAP, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        )
    return ft.Container(
        content=ft.Column(content, spacing=12),
        padding=ft.Padding.all(16),
        border=ft.Border.all(1, HAIRLINE),
        border_radius=ft.BorderRadius.all(12),
    )


def _toggle(store: Store, tile_id: str) -> Callable[[Any], None]:
    def on_change(e: Any) -> None:
        store.expanded = {**store.expanded, tile_id: event_bool(e)}

    return on_change


def details(store: Store, tile_id: str, controls: Sequence[ft.Control], title: str = "Details") -> ft.Control:
    """A closed disclosure whose state the store remembers across re-renders."""
    return ft.ExpansionTile(
        title=ft.Text(title, size=13, color=MUTED),
        controls=[
            ft.Container(
                content=ft.Column(list(controls), spacing=12),
                padding=ft.Padding.symmetric(horizontal=GAP, vertical=4),
            )
        ],
        expanded=store.expanded.get(tile_id, False),
        on_change=_toggle(store, tile_id),
        dense=True,
    )


def empty_state(title: str, why: str) -> ft.Control:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(title, size=15, weight=ft.FontWeight.W_600),
                ft.Text(why, size=12, color=MUTED, text_align=ft.TextAlign.CENTER),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
            tight=True,
        ),
        padding=ft.Padding.symmetric(horizontal=24, vertical=40),
        alignment=ft.Alignment.CENTER,
        border=ft.Border.all(1, HAIRLINE),
        border_radius=ft.BorderRadius.all(12),
    )


def kv_rows(pairs: Sequence[tuple[str, ft.Control]], label_width: float = 170) -> ft.Control:
    return ft.Column(
        [
            ft.Row(
                [ft.Text(k, size=12, color=MUTED, width=label_width), v],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=6,
            )
            for k, v in pairs
        ],
        spacing=4,
    )


def data_table(
    headers: Sequence[str], rows: Sequence[Sequence[ft.Control | str]], numeric: Sequence[bool] | None = None
) -> ft.Control:
    num = list(numeric) if numeric else [False] * len(headers)
    return ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text(h, weight=ft.FontWeight.W_600, size=11, color=MUTED), numeric=n)
            for h, n in zip(headers, num)
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(content=cell if isinstance(cell, ft.Control) else ft.Text(str(cell), size=12))
                    for cell in row
                ]
            )
            for row in rows
        ],
        column_spacing=20,
        heading_row_height=32,
        data_row_min_height=30,
        data_row_max_height=44,
    )


def pill(text: str, ok: bool | None, tooltip: str | None = None) -> ft.Control:
    """A status word with an icon: passed (True), failed (False) or not checked (None); the word carries the meaning."""
    icon, color = {
        True: (ft.Icons.CHECK_CIRCLE_OUTLINE, ft.Colors.GREEN),
        False: (ft.Icons.ERROR_OUTLINE, ft.Colors.ERROR),
        None: (ft.Icons.HELP_OUTLINE, MUTED),
    }[ok]
    return ft.Container(
        content=ft.Row(
            [ft.Icon(icon, size=14, color=color), ft.Text(text, size=12, color=color, no_wrap=True)],
            spacing=5,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        border=ft.Border.all(1, color),
        border_radius=ft.BorderRadius.all(999),
        padding=ft.Padding.symmetric(horizontal=10, vertical=3),
        tooltip=tooltip,
    )


def badge_view(badge: Badge, extra: str = "") -> ft.Control:
    ok = {"pass": True, "fail": False, "not checked": None}[badge.status]
    lines = list(badge.reasons) + [f"not run: {r}" for r in badge.checks_not_run]
    return pill(
        f"{badge.status} · {extra}" if extra else badge.status,
        ok,
        "\n".join(lines) or "every convergence check was run and passed",
    )


# ---- the progress card and the numerics strip ------------------------------------------------------------------------------------


def progress_card(store: Store, session: Session) -> ft.Control:
    rows: list[ft.Control] = [
        ft.Column(
            [
                ft.Row(
                    [
                        ft.ProgressRing(width=14, height=14, stroke_width=2),
                        ft.Text(
                            f"{REQUEST_LABELS.get(status.request, status.request)}: {status.stage}",
                            size=13,
                            expand=True,
                        ),
                        ft.Text(f"{status.elapsed_s:.0f} s", size=12, color=MUTED),
                    ],
                    spacing=GAP,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.ProgressBar(value=status.fraction, bar_height=4),
                status_line(status.message),
            ],
            spacing=6,
        )
        for status in store.running()
    ]
    if not rows:
        return ft.Container()
    return card(
        "Working",
        ft.Column(rows, spacing=12),
        actions=[
            ft.OutlinedButton(
                content=ft.Text("Cancel"),
                icon=ft.Icons.CLOSE,
                on_click=lambda _e: session.cancel_all(),
                tooltip="stop every running job: the worker restarts and forgets its live runs",
            )
        ],
    )


def _pairs(values: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in values.items())


def numerics_strip(store: Store, panel: NumericsPanel) -> ft.Control:
    """How the numbers above were computed, behind one line with the convergence badge."""
    rows: list[ft.Control] = [status_line(panel.engine_note), stat_row([stat_tile(s) for s in panel.tiles])]
    for group in (panel.caps, panel.boundary, panel.margins, panel.mode_classes):
        if group:
            rows.append(ft.Row([shown(s) for s in group], wrap=True, spacing=18, run_spacing=6))
    if panel.norm_deficit is not None:
        rows.append(shown(panel.norm_deficit))
    for c in panel.rechecks:
        verdict = "converged" if c.converged else "NOT converged"
        rows.append(
            status_line(
                f"{c.what}: {_pairs(c.before)} to {_pairs(c.after)}; largest change {c.max_change:.2e} ({verdict})"
            )
        )
    rows.extend(ft.Text(r, size=12, color=ft.Colors.ERROR) for r in panel.badge.reasons)
    if panel.badge.checks_not_run:
        rows.append(status_line("not run: " + "; ".join(panel.badge.checks_not_run)))
    return ft.Container(
        content=ft.ExpansionTile(
            title=ft.Row(
                [
                    ft.Text("Numerics", weight=ft.FontWeight.W_600, size=13),
                    badge_view(panel.badge, panel.summary),
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            controls=[ft.Container(content=ft.Column(rows, spacing=10), padding=ft.Padding.all(12))],
            expanded=store.expanded.get("numerics", False),
            on_change=_toggle(store, "numerics"),
            dense=True,
        ),
        border=ft.Border.only(top=ft.BorderSide(1, HAIRLINE)),
    )
