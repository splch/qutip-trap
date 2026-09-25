"""Shared Flet pieces on the tokens of ``theme`` (DESIGN.md Sections 3 to 5 and 10).

A number is a stat tile (value large, plain label small, chip), never a sentence; a card has a noun for a title, a why
button that opens its concept in the explain drawer, and no subtitle; everything beyond six numbers goes behind a Details
disclosure; a provenance chip is a focusable button whose click opens the section it cites; meaning is never carried by
colour alone; the explain drawer shows one concept at a time with the Part II text behind a Specification tile.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.viewmodel.builder import listed
from qutip_trap_app.viewmodel.catalogue import CATALOGUE, Shown
from qutip_trap_app.viewmodel.learn import CONCEPTS, DEPTHS, Attempt, Prompt, explain, now_days, score_choice
from qutip_trap_app.viewmodel.numerics import Badge, NumericsPanel
from qutip_trap_app.views import theme
from qutip_trap_app.views.state import REQUEST_LABELS

MUTED = ft.Colors.ON_SURFACE_VARIANT
HAIRLINE = ft.Colors.OUTLINE_VARIANT

# ---- formatting ------------------------------------------------------------------------------------------------------

_SI = (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12))


def fmt_number(value: float, unit: str = "") -> str:
    """A number to four significant digits, with an SI prefix for Hz, s, m and W; scientific notation for small
    dimensionless numbers."""
    if isinstance(value, bool):
        return str(value)
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if value == 0.0:
        return f"0 {unit}".strip()
    if unit in ("Hz", "s", "m", "W"):
        prefix, scale = next(((p, sc) for p, sc in _SI if abs(value) >= sc), ("", 1.0))
        return f"{value / scale:.4g} {prefix}{unit}"
    if unit in ("1/s", "quanta/s"):
        return f"{value:.4g} {unit}"
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.3e} {unit}".strip()
    return f"{value:.4g} {unit}".strip()


def fmt_shown(s: Shown) -> str:
    unit = CATALOGUE[s.quantity].unit
    v = s.value
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v} {unit}".strip()
    if isinstance(v, float):
        return fmt_number(v, unit)
    return str(v)


def hover_text(s: Shown) -> str:
    q = CATALOGUE[s.quantity]
    parts = [q.term]
    if q.unit == "Hz":
        parts.append("ordinary frequency; the angular value is 2 pi times this (conv.frequencies)")
    if s.detail:
        parts.append(s.detail)
    parts.append(f"Section {q.section}")
    return "\n".join(parts)


def ions_text(ions: Sequence[int]) -> str:
    """'ion 0', 'ions 0 and 1', 'ions 0, 1 and 2'."""
    return ("ion " if len(ions) == 1 else "ions ") + listed([str(i) for i in ions])


# ---- chips, values and tiles -----------------------------------------------------------------------------------------


def chip(ledger_id: str, index: ProvenanceIndex, *, compact: bool = True) -> ft.Control:
    """A provenance chip: the tag's glyph (and word when not compact) on a quiet pill, the record's section, source,
    equation and corrected form on hover; a click opens the explain drawer's Specification tile at the chip's section."""
    c = index.chip(ledger_id)
    lines = [f"{c.label}: {c.meaning}", f"ledger {c.id}", f"section {c.section}", f"source: {c.source}"]
    if c.equation:
        lines.append(f"equation: {c.equation}")
    if c.corrected_form:
        lines.append(f"corrected form: {c.corrected_form}")
    lines.append("click: read the section")
    opener = index.on_open_section
    section = index.section_for_chip(ledger_id) if opener is not None else None

    def on_click(_e: Any) -> None:
        if opener is not None and section is not None:
            opener(section)

    return ft.TextButton(
        content=ft.Text(
            c.glyph if compact else c.label,
            size=theme.SIZE_MICRO,
            weight=ft.FontWeight.W_600,
            no_wrap=True,
            color=MUTED,
        ),
        tooltip="\n".join(lines),
        on_click=on_click if opener is not None else None,
        style=ft.ButtonStyle(
            padding=ft.Padding.symmetric(horizontal=0 if compact else 8, vertical=0),
            color=MUTED,
            shape=ft.StadiumBorder(side=ft.BorderSide(1, HAIRLINE)),
            visual_density=ft.VisualDensity.COMPACT,
            overlay_color=ft.Colors.with_opacity(0.08, ft.Colors.ON_SURFACE),
        ),
        # a button's Material minimum width is 64 px; a lone glyph wants a 24 px pill
        width=24 if compact else None,
        height=18 if compact else 20,
    )


def shown(
    s: Shown, index: ProvenanceIndex, *, label: bool = True, plain: bool = True, size: int = theme.SIZE_BODY
) -> ft.Control:
    """One displayed value inline: its plain label (or the physics term), the formatted value and the chip."""
    q = CATALOGUE[s.quantity]
    controls: list[ft.Control] = []
    if label:
        controls.append(
            ft.Text(
                q.label if plain else q.term, size=size - 1, color=MUTED, tooltip=q.term if plain else q.label
            )
        )
    controls.append(
        ft.Text(fmt_shown(s), size=size, weight=ft.FontWeight.W_500, tooltip=hover_text(s), selectable=True)
    )
    controls.append(chip(q.ledger_id, index))
    return ft.Row(controls, spacing=6, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def shown_row(
    values: Sequence[Shown], index: ProvenanceIndex, *, empty: str = "", label: bool = True, spacing: int = 12
) -> ft.Control:
    """Displayed values side by side, wrapping; ``empty`` stands in when there are none."""
    controls = [shown(s, index, label=label) for s in values] or ([status_line(empty)] if empty else [])
    return ft.Row(controls, wrap=True, spacing=spacing)


def value_cell(s: Shown, index: ProvenanceIndex) -> ft.Control:
    return ft.Row(
        [
            ft.Text(fmt_shown(s), tooltip=hover_text(s), selectable=True, size=theme.SIZE_SMALL),
            chip(CATALOGUE[s.quantity].ledger_id, index),
        ],
        spacing=4,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


TILE_LABEL_WIDTH = 150.0
"""A stat tile's label wraps at this width, so a row of tiles keeps an even rhythm whatever the values' lengths."""


def stat_tile(
    s: Shown, index: ProvenanceIndex, *, plain: bool = True, label: str | None = None, status: str = ""
) -> ft.Control:
    """A number as a stat tile: the value large with its chip beside it, the plain label (or the term) small beneath, and
    an optional status word (estimate, calibrated, stale)."""
    q = CATALOGUE[s.quantity]
    head = ft.Row(
        [
            ft.Text(
                fmt_shown(s),
                size=theme.SIZE_VALUE,
                weight=ft.FontWeight.W_600,
                tooltip=hover_text(s),
                selectable=True,
            ),
            chip(q.ledger_id, index),
        ],
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )
    caption = label if label is not None else (q.label if plain else q.term)
    below: list[ft.Control] = [
        ft.Text(
            caption,
            size=theme.SIZE_SMALL,
            color=MUTED,
            tooltip=q.term if plain else q.label,
            width=TILE_LABEL_WIDTH,
        )
    ]
    if status:
        below.append(ft.Text(status, size=theme.SIZE_CAPTION, italic=True, color=MUTED))
    return ft.Container(
        content=ft.Column([head] + below, spacing=2, tight=True),
        padding=ft.Padding.symmetric(horizontal=8, vertical=6),
        border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
    )


def stat_row(tiles: Sequence[ft.Control]) -> ft.Control:
    return ft.Row(list(tiles), wrap=True, spacing=theme.GAP * 2, run_spacing=theme.GAP)


def tile_row(values: Sequence[Shown], index: ProvenanceIndex, *, plain: bool = True) -> ft.Control:
    """Values as a row of stat tiles."""
    return stat_row([stat_tile(s, index, plain=plain) for s in values])


# ---- pills, badges and the numerics strip ----------------------------------------------------------------------------

_BADGE_ICON: dict[str, ft.IconData] = {
    "pass": ft.Icons.CHECK_CIRCLE,
    "not checked": ft.Icons.HELP_OUTLINE,
    "fail": ft.Icons.ERROR_OUTLINE,
    "stale": ft.Icons.HISTORY,
}


def pill(
    text: str,
    kind: str,
    *,
    icon: ft.IconData | None = None,
    tooltip: str | None = None,
    key: str | None = None,
) -> ft.Control:
    """A status pill: an icon and a word on the status colour of ``kind`` (pass, fail, not checked, stale); the word carries
    the meaning, the colour only echoes it."""
    bg, fg = theme.status(kind)
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(icon if icon is not None else _BADGE_ICON[kind], size=14, color=fg),
                ft.Text(text, size=theme.SIZE_SMALL, weight=ft.FontWeight.W_500, color=fg, no_wrap=True),
            ],
            spacing=5,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=bg,
        border_radius=ft.BorderRadius.all(999),
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        tooltip=tooltip,
        key=key,
    )


def badge_view(badge: Badge, *, extra: str = "") -> ft.Control:
    """The convergence badge: an icon, the word (never colour alone), the reasons on hover."""
    reasons = [badge.label] + list(badge.reasons) + [f"not run: {r}" for r in badge.checks_not_run]
    text = badge.status if not extra else f"{badge.status} · {extra}"
    return pill(text, badge.status, tooltip="\n".join(reasons))


def numerics_strip(
    panel: NumericsPanel, index: ProvenanceIndex, *, expanded: bool, on_change: Callable[[Any], None]
) -> ft.Control:
    """The numerics panel of every level as a strip whose collapsed form is the badge, the engine and the joint dimension;
    everything else inside."""
    rows: list[ft.Control] = [
        status_line(panel.engine_note),
        stat_row(
            [
                stat_tile(s, index)
                for s in (
                    panel.level,
                    panel.dimension,
                    panel.integrator,
                    panel.tolerances,
                    panel.samples,
                    panel.trajectories,
                    panel.branches,
                    panel.wall_time,
                )
            ]
        ),
    ]
    for group in (panel.caps, panel.boundary, panel.margins, panel.mode_classes):
        if group:
            rows.append(ft.Row([shown(c, index) for c in group], wrap=True, spacing=18, run_spacing=6))
    rows += [shown(s, index) for s in (panel.derivation_residual, panel.norm_deficit) if s is not None]
    for c in (panel.convergence, panel.tolerance_check):
        if c is not None:
            rows.append(
                status_line(
                    f"tolerances {c.tolerances} against {c.tightened_tolerances}: max change {c.max_change:.2e} "
                    f"({'converged' if c.converged else 'NOT converged'})"
                )
            )
    if panel.truncation_check is not None:
        t = panel.truncation_check
        rows.append(
            status_line(
                f"caps {t.caps} against {t.grown_caps}: max change {t.max_change:.2e} "
                f"({'converged' if t.converged else 'NOT converged'})"
            )
        )
    if panel.badge.reasons:
        rows.append(
            ft.Column(
                [ft.Text(r, size=theme.SIZE_SMALL, color=ft.Colors.ERROR) for r in panel.badge.reasons],
                spacing=2,
            )
        )
    level = str(panel.level.value or "")
    dim = panel.dimension.value
    extra = f"{level}, dimension {dim}" if dim is not None else level
    tile = ft.ExpansionTile(
        title=ft.Row(
            [
                ft.Text("Numerics", weight=ft.FontWeight.W_600, size=theme.SIZE_SMALL + 1),
                badge_view(panel.badge, extra=extra),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        controls=[ft.Container(content=ft.Column(rows, spacing=10), padding=ft.Padding.all(12))],
        expanded=expanded,
        on_change=on_change,
        dense=True,
        tile_padding=ft.Padding.symmetric(horizontal=theme.PAGE_PADDING - theme.GAP),
    )
    return ft.Container(content=tile, border=ft.Border.only(top=ft.BorderSide(1, HAIRLINE)))


# ---- layout, headers, cards, disclosure, empty states ----------------------------------------------------------------


def content_width(store: Any, page: Any, level: int = 0) -> float:
    """The width of the content column: the window less the rail, the explain drawer when open and the page margins,
    capped at the reading width. Every layout decision that depends on the room available reads this one number."""
    drawer = theme.DRAWER_WIDTH if store.learner.explain_is_open(level) else 0.0
    return min(
        max((page.width or 1200.0) - theme.RAIL_WIDTH - drawer - 2 * theme.PAGE_PADDING, 360.0),
        theme.CONTENT_MAX_WIDTH,
    )


def content_widths(store: Any, page: Any, level: int = 0) -> tuple[float, float]:
    """(lane width, spectrum strip width): absolute-positioned drawings must know the room they have."""
    content = content_width(store, page, level)
    return content - 2 * theme.CARD_PADDING - 64.0, max(content * 0.5 - 60.0, 240.0)


TWO_COLUMN_MIN_WIDTH = 960.0
"""Below this content width two stacks of cards go one above the other; above it they sit side by side."""


def columns(
    store: Any,
    page: Any,
    level: int,
    left: Sequence[ft.Control],
    right: Sequence[ft.Control],
    *,
    split: tuple[int, int] = (1, 1),
) -> ft.Control:
    """Two stacks of cards side by side (in the proportion ``split``) when the content column is wide enough, else one
    stack with the left cards first. Decided from the actual content width, so an open drawer narrows the layout."""
    stacks = [list(left), list(right)]
    if content_width(store, page, level) < TWO_COLUMN_MIN_WIDTH or not all(stacks):
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


def content_margin(page: Any, drawer_open: bool) -> float:
    """The side margin that centres the content column once the window is wider than the reading width."""
    avail = (
        (page.width or 1200.0)
        - theme.RAIL_WIDTH
        - (theme.DRAWER_WIDTH if drawer_open else 0.0)
        - 2 * theme.PAGE_PADDING
    )
    return max(0.0, (avail - theme.CONTENT_MAX_WIDTH) / 2.0)


def level_header(title: str, question: str, *, note: str = "") -> ft.Control:
    """The level's title with the one question it answers beneath, and an optional muted note (the selection the level
    looks at)."""
    lines: list[ft.Control] = [
        ft.Text(title, theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
        ft.Text(question, size=theme.SIZE_BODY, color=MUTED),
    ]
    if note:
        lines.append(ft.Text(note, size=theme.SIZE_CAPTION, color=MUTED))
    return ft.Container(content=ft.Column(lines, spacing=2), padding=ft.Padding.only(top=4, bottom=4))


def small_icon_button(
    icon: ft.IconData, tooltip: str, on_click: Callable[[Any], None] | None = None
) -> ft.Control:
    return ft.IconButton(
        icon=icon,
        icon_size=16,
        icon_color=MUTED,
        tooltip=tooltip,
        on_click=on_click,
        width=28,
        height=28,
        padding=0,
        visual_density=ft.VisualDensity.COMPACT,
    )


def card(
    title: str,
    body: ft.Control,
    *,
    why: Callable[[Any], None] | None = None,
    info: str | Sequence[str] = (),
    actions: Sequence[ft.Control] = (),
    trailing: Sequence[ft.Control] = (),
    key: str | None = None,
) -> ft.Control:
    """A card with a noun for a title, a why button (its concept in the explain drawer) and an info button (notes,
    approximations, methods on hover) in its title row, and no subtitle: a hairline-bordered surface, never a shadow."""
    head: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600, size=theme.SIZE_CARD_TITLE)]
    if why is not None:
        head.append(
            small_icon_button(
                ft.Icons.HELP_OUTLINE, "why: open this card's concept in the explain drawer", why
            )
        )
    if info:
        lines = [info] if isinstance(info, str) else list(info)
        head.append(
            small_icon_button(ft.Icons.INFO_OUTLINE, "notes:\n" + "\n".join(f"· {ln}" for ln in lines))
        )
    head.append(ft.Container(expand=True))
    head.extend(trailing)
    content: list[ft.Control] = [
        ft.Row(head, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        body,
    ]
    if actions:
        content.append(
            ft.Row(
                list(actions), spacing=theme.GAP, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
    return ft.Container(
        content=ft.Column(content, spacing=12),
        padding=ft.Padding.all(theme.CARD_PADDING),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
        border=ft.Border.all(1, HAIRLINE),
        border_radius=ft.BorderRadius.all(theme.RADIUS_CARD),
        key=key,
    )


def section_title(text: str) -> ft.Control:
    return ft.Text(text, size=theme.SIZE_SMALL, weight=ft.FontWeight.W_600)


def details(
    store: Any,
    session: Any,
    level: int,
    tile_id: str,
    controls: Sequence[ft.Control],
    *,
    title: str = "Details",
    default_open: bool | None = None,
) -> ft.Control:
    """The Details disclosure: open where the learner's plan for this level says so, the learner's own choice remembered."""
    plan_default = store.learner.plan(level).expanded if default_open is None else default_open
    return ft.ExpansionTile(
        title=ft.Text(title, size=theme.SIZE_SMALL + 1, color=MUTED),
        controls=[
            ft.Container(
                content=ft.Column(list(controls), spacing=12),
                padding=ft.Padding.symmetric(horizontal=theme.GAP, vertical=4),
            )
        ],
        expanded=store.details_open.get(tile_id, plan_default),
        on_change=lambda e: session.toggle_details(tile_id, event_bool(e)),
        dense=True,
        tile_padding=ft.Padding.symmetric(horizontal=4),
    )


def hint(store: Any, level: int, concept_id: str) -> ft.Control:
    """One sentence under the focal picture for the plans that want assistance; nothing for the physicist plan."""
    if store.learner.plan(level).explain_depth == "equation":
        return ft.Container()
    return ft.Row(
        [
            ft.Icon(ft.Icons.LIGHTBULB_OUTLINE, size=14, color=MUTED),
            ft.Text(CONCEPTS[concept_id].explain.sentence, size=theme.SIZE_SMALL, color=MUTED, expand=True),
        ],
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def status_line(text: str) -> ft.Control:
    """A status line under twelve words (R5)."""
    return ft.Text(text, size=theme.SIZE_SMALL, color=MUTED)


def empty_state(
    title: str, why: str, *, icon: ft.IconData = ft.Icons.BAR_CHART_OUTLINED, key: str | None = None
) -> ft.Control:
    """Never a dead end: what belongs here and why it is empty."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(icon, size=32, color=MUTED),
                ft.Text(title, size=theme.SIZE_CARD_TITLE, weight=ft.FontWeight.W_600),
                ft.Text(why, size=theme.SIZE_SMALL, color=MUTED, text_align=ft.TextAlign.CENTER),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
            tight=True,
        ),
        padding=ft.Padding.symmetric(horizontal=24, vertical=40),
        alignment=ft.Alignment.CENTER,
        border=ft.Border.all(1, HAIRLINE),
        border_radius=ft.BorderRadius.all(theme.RADIUS_CARD),
        key=key,
    )


def kv_rows(pairs: Sequence[tuple[str, ft.Control]], *, label_width: float = 150) -> ft.Control:
    return ft.Column(
        [
            ft.Row(
                [ft.Text(k, size=theme.SIZE_SMALL, color=MUTED, width=label_width), v],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                wrap=True,
                spacing=6,
                run_spacing=0,
            )
            for k, v in pairs
        ],
        spacing=4,
    )


def data_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[ft.Control | str]],
    *,
    numeric: Sequence[bool] | None = None,
) -> ft.Control:
    """A table on the theme's table style (row heights, spacing and margins come from ``theme.build_theme``)."""
    num = list(numeric) if numeric else [False] * len(columns)
    return ft.DataTable(
        columns=[
            ft.DataColumn(
                label=ft.Text(c, weight=ft.FontWeight.W_600, size=theme.SIZE_CAPTION, color=MUTED), numeric=n
            )
            for c, n in zip(columns, num)
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(
                        content=cell
                        if isinstance(cell, ft.Control)
                        else ft.Text(str(cell), size=theme.SIZE_SMALL)
                    )
                    for cell in row
                ]
            )
            for row in rows
        ],
        horizontal_lines=ft.BorderSide(1, HAIRLINE),
    )


def page_tabs(
    items: Sequence[tuple[str, str]], selected: str, on_select: Callable[[str], None], *, key_prefix: str
) -> ft.Control:
    """A row of secondary tabs for sibling pages (the physics pages, the Learn activities): one line that scrolls sideways
    when narrow, the selected tab underlined and in ink, the others muted."""
    tabs: list[ft.Control] = []
    for i, label in items:
        active = i == selected
        tabs.append(
            ft.Container(
                content=ft.Text(
                    label,
                    size=theme.SIZE_SMALL + 1,
                    weight=ft.FontWeight.W_600 if active else ft.FontWeight.W_400,
                    color=ft.Colors.ON_SURFACE if active else MUTED,
                    no_wrap=True,
                ),
                padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                border=ft.Border.only(
                    bottom=ft.BorderSide(2, ft.Colors.PRIMARY if active else ft.Colors.TRANSPARENT)
                ),
                on_click=(lambda e, ii=i: on_select(ii)) if not active else None,
                ink=not active,
                key=f"{key_prefix}:{i}",
            )
        )
    return ft.Container(
        content=ft.Row(tabs, spacing=0, scroll=ft.ScrollMode.HIDDEN),
        border=ft.Border.only(bottom=ft.BorderSide(1, HAIRLINE)),
    )


def event_bool(e: Any) -> bool:
    """The boolean a Flet change event carries (an ExpansionTile sends the tile's new state; the client may send text)."""
    return e.data if isinstance(e.data, bool) else str(e.data).strip().lower() == "true"


def input_style() -> dict[str, Any]:
    """The compact field style every text field and dropdown of the screens shares."""
    return {"dense": True, "text_size": 13, "border_radius": ft.BorderRadius.all(theme.RADIUS_TILE)}


def inset(controls: list[ft.Control]) -> ft.Control:
    """A quiet inset panel for one prompt or drill inside a card."""
    return ft.Container(
        content=ft.Column(controls, spacing=8),
        padding=ft.Padding.all(12),
        border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border=ft.Border.all(1, HAIRLINE),
    )


@ft.component
def ProgressRows(store: Any, session: Any) -> ft.Control:
    """The running jobs with their stage, elapsed time and progress, and one Cancel for all of them."""
    ft.use_state(store)
    rows: list[ft.Control] = [
        ft.Column(
            [
                ft.Row(
                    [
                        ft.ProgressRing(width=14, height=14, stroke_width=2),
                        ft.Text(
                            f"{REQUEST_LABELS[status.request]}: {status.stage}",
                            size=theme.SIZE_SMALL + 1,
                            expand=True,
                        ),
                        ft.Text(f"{status.elapsed_s:.0f} s", size=theme.SIZE_SMALL, color=MUTED),
                    ],
                    spacing=theme.GAP,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.ProgressBar(value=status.fraction, bar_height=4, border_radius=ft.BorderRadius.all(2)),
                status_line(status.message),
            ],
            spacing=6,
        )
        for status in store.running()
    ]
    if not rows:
        return ft.Container()
    cancel = ft.OutlinedButton(
        content=ft.Text("Cancel"),
        icon=ft.Icons.CLOSE,
        on_click=lambda e: session.cancel_all(),
        tooltip="stop every running job: the worker process restarts, and the next Run rebuilds its channel library",
    )
    return card("Working", ft.Column(rows, spacing=12), actions=[cancel], key="working")


# ---- retrieval prompts and the explain drawer ------------------------------------------------------------------------


@ft.component
def PromptView(session: Any, concept_id: str, prompt: Prompt, *, unaided: bool) -> ft.Control:
    """A retrieval prompt: a choice, a free answer, or a prediction to check on the screen; every attempt (a skip too, as
    an exposure) goes to the mastery log, unaided when no explanation stands beside it (the review tray)."""
    answer, set_answer = ft.use_state("")
    feedback, set_feedback = ft.use_state("")

    def attempt(correct: bool | None) -> None:
        session.record_attempt(Attempt(concept_id, prompt.id, now_days(), correct, unaided=unaided))

    def check(_e: Any) -> None:
        if prompt.kind == "choose":
            ok = score_choice(prompt, answer)
            set_feedback("that matches the physics" if ok else f"the physics says: {prompt.answer}")
            attempt(ok)
        elif prompt.kind == "free_text":
            set_feedback(f"compare with the simulator's own account: {prompt.rubric}")
            attempt(None)
        else:
            set_feedback(f"now look at {prompt.where} and compare it with what you expected")
            attempt(None)

    def skip(_e: Any) -> None:
        attempt(None)
        set_feedback("skipped, not counted as wrong; it returns in the review tray")

    controls: list[ft.Control] = [
        ft.Text(prompt.question, size=theme.SIZE_SMALL + 1, weight=ft.FontWeight.W_500)
    ]
    if prompt.kind == "choose":
        # radio rows whose labels wrap inside the drawer (a Radio's own label does not), selectable by the whole row
        rows: list[ft.Control] = [
            ft.Container(
                content=ft.Row(
                    [ft.Radio(value=o), ft.Text(o, size=theme.SIZE_SMALL + 1, expand=True)],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                on_click=lambda e, o=o: set_answer(o),
                ink=True,
                border_radius=ft.BorderRadius.all(6),
            )
            for o in prompt.options
        ]
        controls.append(
            ft.RadioGroup(
                content=ft.Column(rows, spacing=0),
                value=answer or None,
                on_change=lambda e: set_answer(str(e.control.value)),
            )
        )
    elif prompt.kind == "free_text":
        controls.append(
            ft.TextField(
                value=answer,
                multiline=True,
                min_lines=2,
                max_lines=4,
                on_change=lambda e: set_answer(str(e.control.value)),
                dense=True,
                text_size=theme.SIZE_SMALL + 1,
                border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
            )
        )
    else:
        controls.append(
            ft.Text(
                f"answer in your head, then look at {prompt.where}",
                size=theme.SIZE_SMALL,
                italic=True,
                color=MUTED,
            )
        )
    controls.append(
        ft.Row(
            [
                ft.FilledTonalButton(
                    content=ft.Text("Check"), on_click=check, disabled=prompt.kind == "choose" and not answer
                ),
                ft.TextButton(content=ft.Text("Skip"), on_click=skip),
            ],
            spacing=6,
        )
    )
    if feedback:
        controls.append(ft.Text(feedback, size=theme.SIZE_SMALL, color=MUTED))
    return ft.Column(controls, spacing=8)


def SpecificationTile(
    index: ProvenanceIndex,
    section_number: str,
    *,
    expanded: bool,
    on_change: Callable[[Any], None],
    on_open: Callable[[str], None],
) -> ft.Control:
    """The Part II subsection that governs the screen, as its own Markdown, behind a collapsed tile; a heading with no text
    of its own lists its subsections as buttons."""
    number = index.nearest_section_with_text(section_number)
    try:
        info = index.section(number)
        text = index.section_text(number)
        title = f"Specification: {info.heading}"
        sub = f"PLAN.md line {info.line}" + (
            "; tags: " + ", ".join(f"{k} {v}" for k, v in info.tags.items()) if info.tags else ""
        )
        children = index.subsections(number)
    except KeyError:
        title, text, sub, children = f"Specification: Section {section_number}", "", "", ()
    body: ft.Control
    if text:
        body = ft.Container(
            content=ft.Column(
                [ft.Markdown(text, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)],
                scroll=ft.ScrollMode.AUTO,
            ),
            height=360,
            padding=ft.Padding.symmetric(horizontal=8),
        )
    elif children:
        body = ft.Column(
            [
                ft.TextButton(
                    content=ft.Text(c.heading, size=theme.SIZE_SMALL),
                    on_click=lambda e, n=c.number: on_open(n),
                )
                for c in children
            ],
            spacing=0,
        )
    else:
        body = ft.Text("this section's text is not in the index", size=theme.SIZE_SMALL, italic=True)
    return ft.ExpansionTile(
        title=ft.Text(title, size=theme.SIZE_SMALL + 1, weight=ft.FontWeight.W_500),
        subtitle=ft.Text(sub, size=theme.SIZE_CAPTION, color=MUTED) if sub else None,
        controls=[body],
        expanded=expanded,
        on_change=on_change,
        dense=True,
    )


@ft.component
def ExplainDrawer(
    store: Any,
    session: Any,
    level: int,
    index: ProvenanceIndex,
    section_number: str,
    concepts: tuple[str, ...],
) -> ft.Control:
    """The explain drawer: the concepts as a row of chips, the selected one open at the learner's depth with its chips and
    its retrieval prompt, and the Specification tile with the governing text (or the section a chip asked for)."""
    ft.use_state(store)
    spec_open, set_spec_open = ft.use_state(store.spec_section is not None)
    depth = store.learner.depth(level)

    def set_depth(e: Any) -> None:
        selected = list(e.control.selected)
        if selected:
            session.set_learner(depth_override=selected[0])

    chosen = store.explain_concept.get(level)
    if chosen not in concepts:
        chosen = concepts[0] if concepts else None
    chips_row = ft.Row(
        [
            ft.Chip(
                label=ft.Text(CONCEPTS[cid].title, size=theme.SIZE_SMALL),
                selected=cid == chosen,
                on_click=lambda e, c=cid: session.select_concept(level, c),
                show_checkmark=False,
            )
            for cid in concepts
        ],
        wrap=True,
        spacing=6,
        run_spacing=6,
    )
    header = ft.Row(
        [
            ft.Icon(ft.Icons.MENU_BOOK_OUTLINED, size=18, color=MUTED),
            ft.Text("Explain", theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
            ft.Container(expand=True),
            ft.IconButton(
                icon=ft.Icons.CLOSE,
                icon_size=18,
                tooltip="close the drawer (Esc)",
                on_click=lambda e: session.set_learner(explain_open=False),
            ),
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    depth_control = ft.SegmentedButton(
        segments=[ft.Segment(value=d, label=ft.Text(d, size=theme.SIZE_SMALL)) for d in DEPTHS],
        selected=[depth],
        on_change=set_depth,
    )
    body: list[ft.Control] = [header, depth_control, chips_row]
    if chosen is not None:
        body.append(_explain_card(session, chosen, depth, index))
    body.append(
        SpecificationTile(
            index,
            store.spec_section or section_number,
            expanded=spec_open or store.spec_section is not None,
            on_change=lambda e: set_spec_open(event_bool(e)),
            on_open=session.open_specification,
        )
    )
    return ft.Container(
        content=ft.Column(body, spacing=12, scroll=ft.ScrollMode.AUTO, expand=True),
        width=theme.DRAWER_WIDTH,
        padding=ft.Padding.symmetric(horizontal=16, vertical=12),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border=ft.Border.only(left=ft.BorderSide(1, HAIRLINE)),
    )


def _explain_card(session: Any, concept_id: str, depth: Any, index: ProvenanceIndex) -> ft.Control:
    """The open concept: its explanation at the learner's depth, its chips, the deeper button and its retrieval prompt
    (aided: the explanation stands open above it)."""
    card_data = explain(concept_id, depth)
    concept = card_data.concept
    body: list[ft.Control] = [
        ft.Text(concept.title, weight=ft.FontWeight.W_600, size=theme.SIZE_CARD_TITLE),
        ft.Text(concept.term, size=theme.SIZE_SMALL, italic=True, color=MUTED),
        ft.Text(card_data.text, size=theme.SIZE_BODY, selectable=True),
        ft.Row(
            [chip(i, index, compact=False) for i in concept.ledger_ids]
            + [ft.Text(f"Section {concept.section}", size=theme.SIZE_CAPTION, color=MUTED)],
            wrap=True,
            spacing=4,
            run_spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        ft.Text(f"you will be able to {concept.can_do}", size=theme.SIZE_CAPTION, italic=True, color=MUTED),
    ]
    if card_data.deeper is not None:
        deeper = card_data.deeper
        body.append(
            ft.TextButton(
                content=ft.Text(f"deeper: {deeper}"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda e: session.set_learner(depth_override=deeper),
            )
        )
    body.append(ft.Divider(height=8))
    body.append(PromptView(session, concept_id, card_data.prompt, unaided=False))
    return ft.Container(
        content=ft.Column(body, spacing=8),
        padding=ft.Padding.all(14),
        border_radius=ft.BorderRadius.all(theme.RADIUS_CARD),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
        border=ft.Border.all(1, HAIRLINE),
    )
