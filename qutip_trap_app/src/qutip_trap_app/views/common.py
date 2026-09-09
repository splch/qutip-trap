"""Shared Flet pieces (DESIGN.md Sections 3 to 5 and the information hierarchy of Section 10).

Rules carried here: a number is a stat tile (value large, plain label small, chip) and never a sentence; a card has a noun
for a title, a why button that opens its concept in the explain drawer, and no subtitle; everything beyond six numbers goes
behind a Details disclosure; a provenance chip is a focusable button whose click opens the section it cites; meaning is never
carried by colour alone (every tag has a glyph and a word, every badge its word); the explain drawer shows one concept at a
time with the Part II text behind a Specification tile. Plain label first, the physics term on hover, the unit always
printed, Hz with the 2 pi conversion on hover (Section 14.5).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.viewmodel.catalogue import CATALOGUE, Shown
from qutip_trap_app.viewmodel.learn import CONCEPTS, Attempt, ExplainCard, Prompt, explain, score_choice
from qutip_trap_app.viewmodel.numerics import Badge, NumericsPanel

# ---- formatting ---------------------------------------------------------------------------------------------------------------------

_SI = (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12))


def fmt_number(value: float, unit: str = "", digits: int = 4) -> str:
    """A number with an SI prefix for Hz, s, m and W; scientific notation for small dimensionless numbers."""
    if isinstance(value, bool):
        return str(value)
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if value == 0.0:
        return f"0 {unit}".strip()
    if unit in ("Hz", "s", "m", "W"):
        prefix, scale = "", 1.0
        for pfx, sc in _SI:
            if abs(value) >= sc:
                prefix, scale = pfx, sc
                break
        return f"{value / scale:.{digits}g} {prefix}{unit}"
    if unit in ("1/s", "quanta/s"):
        return f"{value:.{digits}g} {unit}"
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.{max(digits - 1, 1)}e} {unit}".strip()
    return f"{value:.{digits}g} {unit}".strip()


def fmt_shown(s: Shown) -> str:
    q = CATALOGUE[s.quantity]
    v = s.value
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v} {q.unit}".strip()
    if isinstance(v, float):
        return fmt_number(v, q.unit)
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


# ---- chips ------------------------------------------------------------------------------------------------------------------------


def chip(ledger_id: str, index: ProvenanceIndex, *, compact: bool = True) -> ft.Control:
    """A provenance chip: the tag's glyph (and word when not compact) on a small button, with the record's section, source,
    equation and corrected form on hover and focus; a click opens the explain drawer's Specification tile at the chip's
    section (R8). A button, so Tab reaches it."""
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
        content=ft.Text(c.glyph if compact else c.label, size=11, no_wrap=True),
        tooltip="\n".join(lines),
        on_click=on_click if opener is not None else None,
        style=ft.ButtonStyle(
            padding=ft.Padding.symmetric(horizontal=6, vertical=0),
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            color=ft.Colors.ON_SECONDARY_CONTAINER,
            shape=ft.RoundedRectangleBorder(radius=10),
            visual_density=ft.VisualDensity.COMPACT,
        ),
        height=22,
    )


def shown(
    s: Shown, index: ProvenanceIndex, *, label: bool = True, plain: bool = True, size: int = 14
) -> ft.Control:
    """One displayed value inline: its plain label (or the physics term), the formatted value and the chip."""
    q = CATALOGUE[s.quantity]
    controls: list[ft.Control] = []
    if label:
        controls.append(
            ft.Text(
                q.label if plain else q.term,
                size=size - 1,
                color=ft.Colors.ON_SURFACE_VARIANT,
                tooltip=q.term if plain else q.label,
            )
        )
    controls.append(
        ft.Text(fmt_shown(s), size=size, weight=ft.FontWeight.W_500, tooltip=hover_text(s), selectable=True)
    )
    controls.append(chip(q.ledger_id, index))
    return ft.Row(controls, spacing=6, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def value_cell(s: Shown, index: ProvenanceIndex) -> ft.Control:
    return ft.Row(
        [
            ft.Text(fmt_shown(s), tooltip=hover_text(s), selectable=True, size=12),
            chip(CATALOGUE[s.quantity].ledger_id, index),
        ],
        spacing=4,
    )


def stat_tile(
    s: Shown,
    index: ProvenanceIndex,
    *,
    plain: bool = True,
    label: str | None = None,
    status: str = "",
    on_click: Callable[[Any], None] | None = None,
    width: float | None = None,
) -> ft.Control:
    """A number as a stat tile (R3): the value large, the plain label (or the term) small beneath, the chip beside the
    value, an optional status word (estimate, calibrated, stale) and an optional click (a zoom-in target)."""
    q = CATALOGUE[s.quantity]
    head = ft.Row(
        [
            ft.Text(
                fmt_shown(s), size=20, weight=ft.FontWeight.W_600, tooltip=hover_text(s), selectable=True
            ),
            chip(q.ledger_id, index),
        ],
        spacing=4,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )
    caption = label if label is not None else (q.label if plain else q.term)
    below: list[ft.Control] = [
        ft.Text(caption, size=11, color=ft.Colors.ON_SURFACE_VARIANT, tooltip=q.term if plain else q.label)
    ]
    if status:
        below.append(ft.Text(status, size=10, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
    return ft.Container(
        content=ft.Column([head] + below, spacing=0, tight=True),
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
        border_radius=ft.BorderRadius.all(8),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        on_click=on_click,
        ink=on_click is not None,
        width=width,
    )


def stat_row(tiles: Sequence[ft.Control]) -> ft.Control:
    return ft.Row(list(tiles), wrap=True, spacing=8, run_spacing=8)


# ---- badges and the numerics strip --------------------------------------------------------------------------------------------------

_BADGE_ICON = {
    "pass": ft.Icons.CHECK_CIRCLE,
    "not checked": ft.Icons.HELP_OUTLINE,
    "fail": ft.Icons.ERROR_OUTLINE,
}
_BADGE_BG = {
    "pass": ft.Colors.PRIMARY_CONTAINER,
    "not checked": ft.Colors.SECONDARY_CONTAINER,
    "fail": ft.Colors.ERROR_CONTAINER,
}
_BADGE_FG = {
    "pass": ft.Colors.ON_PRIMARY_CONTAINER,
    "not checked": ft.Colors.ON_SECONDARY_CONTAINER,
    "fail": ft.Colors.ON_ERROR_CONTAINER,
}


def badge_view(badge: Badge, *, extra: str = "") -> ft.Control:
    """The convergence badge: an icon, the word (never colour alone), the reasons on hover (Section 14.5)."""
    reasons = [badge.label] + list(badge.reasons) + [f"not run: {r}" for r in badge.checks_not_run]
    text = badge.status if not extra else f"{badge.status} · {extra}"
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(_BADGE_ICON[badge.status], size=16, color=_BADGE_FG[badge.status]),
                ft.Text(text, size=12, color=_BADGE_FG[badge.status]),
            ],
            spacing=6,
            tight=True,
        ),
        bgcolor=_BADGE_BG[badge.status],
        border_radius=ft.BorderRadius.all(12),
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        tooltip="\n".join(reasons) if reasons else "every check run and passed",
    )


def numerics_strip(
    panel: NumericsPanel,
    index: ProvenanceIndex,
    *,
    expanded: bool = False,
    on_change: Callable[[Any], None] | None = None,
) -> ft.Control:
    """The numerics panel of every level (Section 14.2) as a strip whose collapsed form is the badge, the engine and the
    joint dimension (R6); everything else inside."""
    rows: list[ft.Control] = [
        ft.Text(panel.engine_note, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
        stat_row(
            [
                stat_tile(panel.level, index),
                stat_tile(panel.dimension, index),
                stat_tile(panel.integrator, index),
            ]
            + [stat_tile(panel.tolerances, index), stat_tile(panel.samples, index)]
            + [
                stat_tile(panel.trajectories, index),
                stat_tile(panel.branches, index),
                stat_tile(panel.wall_time, index),
            ]
        ),
    ]
    if panel.caps:
        rows.append(ft.Row([shown(c, index) for c in panel.caps], wrap=True, spacing=18))
    if panel.boundary:
        rows.append(ft.Row([shown(b, index) for b in panel.boundary], wrap=True, spacing=18))
    if panel.margins:
        rows.append(ft.Row([shown(m, index) for m in panel.margins], wrap=True, spacing=18))
    if panel.mode_classes:
        rows.append(ft.Row([shown(m, index) for m in panel.mode_classes], wrap=True, spacing=12))
    if panel.derivation_residual is not None:
        rows.append(shown(panel.derivation_residual, index))
    if panel.norm_deficit is not None:
        rows.append(shown(panel.norm_deficit, index))
    for c in (panel.convergence, panel.tolerance_check):
        if c is not None:
            rows.append(
                ft.Text(
                    f"tolerances {c.tolerances} against {c.tightened_tolerances}: max change {c.max_change:.2e} "
                    f"({'converged' if c.converged else 'NOT converged'})",
                    size=12,
                )
            )
    if panel.truncation_check is not None:
        t = panel.truncation_check
        rows.append(
            ft.Text(
                f"caps {t.caps} against {t.grown_caps}: max change {t.max_change:.2e} "
                f"({'converged' if t.converged else 'NOT converged'})",
                size=12,
            )
        )
    if panel.badge.reasons:
        rows.append(
            ft.Column(
                [ft.Text(r, size=12, color=ft.Colors.ON_ERROR_CONTAINER) for r in panel.badge.reasons],
                spacing=2,
            )
        )
    level = str(panel.level.value or "")
    dim = panel.dimension.value
    extra = f"{level}, dimension {dim}" if dim is not None else level
    return ft.ExpansionTile(
        title=ft.Row(
            [ft.Text("Numerics", weight=ft.FontWeight.W_600, size=13), badge_view(panel.badge, extra=extra)],
            spacing=12,
        ),
        controls=[ft.Container(content=ft.Column(rows, spacing=8), padding=ft.Padding.all(12))],
        expanded=expanded,
        on_change=on_change,
        dense=True,
    )


# ---- headers, cards, disclosure, empty states -------------------------------------------------------------------------------------------


def content_widths(store: Any, page: Any, level: int = 0) -> tuple[float, float]:
    """(lane width, spectrum strip width) from the window: the content column is the window less the rail, the explain
    drawer when open, and the margins; absolute-positioned drawings must know their width."""
    total = float(getattr(page, "width", None) or 1200.0)
    drawer = 360.0 if store.learner.explain_is_open(level) else 0.0
    content = max(total - 80.0 - drawer - 120.0, 360.0)
    return content - 60.0, max(content * 0.5 - 60.0, 240.0)


def level_header(
    title: str, question: str, *, note: str = "", trailing: Sequence[ft.Control] = ()
) -> ft.Control:
    """The level's title and the one question it answers (R1); an optional muted note (the job) and trailing controls."""
    items: list[ft.Control] = [
        ft.Text(title, theme_style=ft.TextThemeStyle.HEADLINE_SMALL),
        ft.Text(question, size=14, color=ft.Colors.ON_SURFACE_VARIANT, expand=True),
    ]
    if note:
        items.append(ft.Text(note, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
    items.extend(trailing)
    return ft.Row(items, spacing=12, vertical_alignment=ft.CrossAxisAlignment.END)


def why_button(
    on_click: Callable[[Any], None], *, tooltip: str = "why: open this card's concept in the explain drawer"
) -> ft.Control:
    return ft.IconButton(icon=ft.Icons.HELP_OUTLINE, icon_size=16, tooltip=tooltip, on_click=on_click)


def info_button(text: str | Sequence[str], *, tooltip_title: str = "notes") -> ft.Control:
    """Notes, approximations and methods as one info icon with the text on hover (R4)."""
    lines = [text] if isinstance(text, str) else list(text)
    return ft.IconButton(
        icon=ft.Icons.INFO_OUTLINE,
        icon_size=16,
        tooltip=f"{tooltip_title}:\n" + "\n".join(f"· {ln}" for ln in lines),
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
    """A card with a noun for a title, a why button and an info button in its title row, and no subtitle (R2, R4)."""
    head: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600, size=15)]
    if why is not None:
        head.append(why_button(why))
    if info:
        head.append(info_button(info))
    head.append(ft.Container(expand=True))
    head.extend(trailing)
    content: list[ft.Control] = [
        ft.Row(head, spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        body,
    ]
    if actions:
        content.append(ft.Row(list(actions), spacing=8, wrap=True))
    return ft.Card(
        content=ft.Container(content=ft.Column(content, spacing=10), padding=ft.Padding.all(14)),
        key=key if key else None,
    )


def details(
    tile_id: str,
    controls: Sequence[ft.Control],
    *,
    store: Any,
    session: Any,
    title: str = "Details",
    default_open: bool | None = None,
) -> ft.Control:
    """The Details disclosure (R3): closed by default, open for the physicist plan, the learner's own choice remembered."""
    plan_default = bool(store.learner.plan(0).chips_expanded) if default_open is None else default_open
    expanded = store.details_open.get(tile_id, plan_default)
    return ft.ExpansionTile(
        title=ft.Text(title, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
        controls=[ft.Container(content=ft.Column(list(controls), spacing=10), padding=ft.Padding.all(8))],
        expanded=expanded,
        on_change=lambda e: session.toggle_details(tile_id, event_bool(e)),
        dense=True,
        tile_padding=ft.Padding.symmetric(horizontal=4),
    )


def hint(store: Any, level: int, concept_id: str) -> ft.Control:
    """One sentence under the focal picture for the plans that want assistance (the concept's sentence depth); nothing
    for the physicist plan (Section 10, learner-adaptive density)."""
    plan = store.learner.plan(level)
    if plan.explain_depth == "equation":
        return ft.Container()
    return ft.Text(
        CONCEPTS[concept_id].explain.sentence, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT
    )


def status_line(text: str) -> ft.Control:
    """A status line under twelve words (R5)."""
    return ft.Text(text, size=12, color=ft.Colors.ON_SURFACE_VARIANT)


def empty_state(title: str, why: str, action_label: str, on_click: Callable[[Any], None]) -> ft.Control:
    """Never a dead end: what belongs here, why it is empty, one clear next action."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.SCIENCE, size=40, color=ft.Colors.PRIMARY),
                ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Text(why, size=13, color=ft.Colors.ON_SURFACE_VARIANT, text_align=ft.TextAlign.CENTER),
                ft.FilledButton(content=ft.Text(action_label), icon=ft.Icons.PLAY_ARROW, on_click=on_click),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
        padding=ft.Padding.all(32),
        alignment=ft.Alignment.CENTER,
    )


def kv_rows(pairs: Sequence[tuple[str, ft.Control]], *, label_width: float = 150) -> ft.Control:
    return ft.Column(
        [
            ft.Row(
                [ft.Text(k, size=12, color=ft.Colors.ON_SURFACE_VARIANT, width=label_width), v],
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
    num = list(numeric) if numeric else [False] * len(columns)
    return ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text(c, weight=ft.FontWeight.W_600, size=12), numeric=n)
            for c, n in zip(columns, num)
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(
                        content=(cell if isinstance(cell, ft.Control) else ft.Text(str(cell), size=12))
                    )
                    for cell in row
                ]
            )
            for row in rows
        ],
        column_spacing=18,
        heading_row_height=32,
        data_row_min_height=28,
        data_row_max_height=44,
    )


# ---- the explain drawer ------------------------------------------------------------------------------------------------------------------

LEVEL_CONCEPTS: dict[int, tuple[str, ...]] = {
    0: ("shot", "histogram", "bitstring_order", "target_vs_simulated", "spam", "provenance_tags"),
    1: ("bloch_vector", "native_gate", "compile", "virtual_z", "entanglement_by_ms"),
    2: ("pulse", "mode", "tone_and_sideband", "loop_closure", "crosstalk"),
    3: ("spin_dependent_force", "fock_states", "debye_waller", "quantum_jumps", "truncation_and_convergence"),
    4: (
        "hamiltonian",
        "lamb_dicke",
        "trap_and_mathieu",
        "atomic_structure",
        "noise_as_physics",
        "calibration",
    ),
}
"""The concepts each level's explain drawer carries (DESIGN.md Section 2), in teaching order."""


def _day_clock() -> float:
    return time.time() / 86400.0


def event_bool(e: Any) -> bool:
    """The boolean a Flet change event carries (an ExpansionTile sends the tile's new state; the client may send it as text)."""
    data = getattr(e, "data", None)
    if isinstance(data, bool):
        return data
    return str(data).strip().lower() == "true"


def _choices(options: Sequence[str], answer: str, set_answer: Callable[[str], None]) -> ft.Control:
    """Radio choices whose labels wrap inside the drawer (a Radio's own label does not), selectable by the whole row."""
    rows: list[ft.Control] = [
        ft.Container(
            content=ft.Row(
                [ft.Radio(value=o), ft.Text(o, size=13, expand=True)],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            on_click=lambda e, o=o: set_answer(o),
            ink=True,
            border_radius=ft.BorderRadius.all(6),
        )
        for o in options
    ]
    return ft.RadioGroup(
        content=ft.Column(rows, spacing=0),
        value=answer or None,
        on_change=lambda e: set_answer(str(e.control.value)),
    )


@ft.component
def ExplainCardView(
    session: Any, card_data: ExplainCard, index: ProvenanceIndex, on_deeper: Callable[[], None]
) -> ft.Control:
    """The open concept: its explanation at the learner's depth, its chips, the deeper button and its retrieval prompt."""
    answer, set_answer = ft.use_state("")
    feedback, set_feedback = ft.use_state("")
    concept = card_data.concept
    prompt: Prompt = card_data.prompt

    def attempt(correct: bool | None) -> None:
        # the explanation stands open above this prompt, so every attempt made here is aided; the unaided re-ask is the
        # review tray's (DESIGN.md Section 3)
        session.record_attempt(Attempt(concept.id, prompt.id, _day_clock(), correct, unaided=False))

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

    prompt_controls: list[ft.Control] = [ft.Text(prompt.question, size=13, weight=ft.FontWeight.W_500)]
    if prompt.kind == "choose":
        prompt_controls.append(_choices(prompt.options, answer, set_answer))
    elif prompt.kind == "free_text":
        prompt_controls.append(
            ft.TextField(
                value=answer,
                multiline=True,
                min_lines=2,
                max_lines=4,
                on_change=lambda e: set_answer(str(e.control.value)),
                dense=True,
                text_size=13,
            )
        )
    else:
        prompt_controls.append(
            ft.Text(f"answer in your head, then look at {prompt.where}", size=12, italic=True)
        )
    prompt_controls.append(
        ft.Row(
            [
                ft.FilledButton(
                    content=ft.Text("Check"),
                    on_click=check,
                    disabled=(prompt.kind == "choose" and not answer),
                ),
                ft.TextButton(content=ft.Text("Skip"), on_click=skip),
            ],
            spacing=6,
        )
    )
    if feedback:
        prompt_controls.append(ft.Text(feedback, size=12, color=ft.Colors.ON_SURFACE_VARIANT))
    body: list[ft.Control] = [
        ft.Text(concept.title, weight=ft.FontWeight.W_600, size=15),
        ft.Text(concept.term, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ft.Text(card_data.text, size=13, selectable=True),
        ft.Row(
            [chip(i, index, compact=False) for i in concept.ledger_ids]
            + [ft.Text(f"Section {concept.section}", size=11, color=ft.Colors.ON_SURFACE_VARIANT)],
            wrap=True,
            spacing=4,
        ),
        ft.Text(
            f"you will be able to {concept.can_do}", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT
        ),
    ]
    if card_data.deeper is not None:
        body.append(
            ft.TextButton(
                content=ft.Text(f"deeper: {card_data.deeper}"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda e: on_deeper(),
            )
        )
    body.append(ft.Divider(height=8))
    body.append(ft.Column(prompt_controls, spacing=6))
    return ft.Container(
        content=ft.Column(body, spacing=8),
        padding=ft.Padding.all(10),
        border_radius=ft.BorderRadius.all(10),
        bgcolor=ft.Colors.SURFACE,
    )


def SpecificationTile(
    index: ProvenanceIndex,
    section_number: str,
    *,
    expanded: bool,
    on_change: Callable[[Any], None],
    on_open: Callable[[str], None] | None = None,
) -> ft.Control:
    """The Part II subsection that governs the screen, as its own Markdown, behind a collapsed tile (R7); a heading with no
    text of its own lists its subsections as buttons."""
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
                    content=ft.Text(c.heading, size=12),
                    on_click=(lambda e, n=c.number: on_open(n)) if on_open is not None else None,
                )
                for c in children
            ],
            spacing=0,
        )
    else:
        body = ft.Text("this section's text is not in the index", size=12, italic=True)
    return ft.ExpansionTile(
        title=ft.Text(title, size=13, weight=ft.FontWeight.W_500),
        subtitle=ft.Text(sub, size=11, color=ft.Colors.ON_SURFACE_VARIANT) if sub else None,
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
    concepts: tuple[str, ...] | None = None,
) -> ft.Control:
    """The explain drawer (R7): the level's concepts as a row of chips, the selected one open at the learner's depth, and
    the Specification tile with the governing Part II text (or the section a chip asked for)."""
    ft.use_state(store)
    spec_open, set_spec_open = ft.use_state(store.spec_section is not None)
    depth = store.learner.depth(level)
    order: tuple[str, ...] = ("sentence", "picture", "equation")

    def deeper() -> None:
        k = order.index(depth)
        if k + 1 < len(order):
            session.set_learner(depth_override=order[k + 1])

    def set_depth(e: Any) -> None:
        selected = list(e.control.selected)
        if selected:
            session.set_learner(depth_override=selected[0])

    ids = concepts if concepts is not None else LEVEL_CONCEPTS.get(level, ())
    chosen = store.explain_concept.get(level)
    if chosen not in ids:
        chosen = ids[0] if ids else None
    chips_row = ft.Row(
        [
            ft.Chip(
                label=ft.Text(CONCEPTS[cid].title, size=11),
                selected=cid == chosen,
                on_click=lambda e, c=cid: session.select_concept(level, c),
                show_checkmark=False,
                padding=ft.Padding.symmetric(horizontal=4, vertical=0),
            )
            for cid in ids
        ],
        wrap=True,
        spacing=4,
        run_spacing=4,
    )
    body: list[ft.Control] = [
        ft.Row(
            [
                ft.Icon(ft.Icons.MENU_BOOK, size=18),
                ft.Text("Explain", weight=ft.FontWeight.W_600),
                ft.Container(expand=True),
                ft.SegmentedButton(
                    segments=[ft.Segment(value=d, label=ft.Text(d, size=10)) for d in order],
                    selected=[depth],
                    on_change=set_depth,
                ),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=16,
                    tooltip="close the drawer (Esc)",
                    on_click=lambda e: session.set_learner(explain_open=False),
                ),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        chips_row,
    ]
    if chosen is not None:
        body.append(ExplainCardView(session, explain(chosen, depth), index, deeper))
    spec = store.spec_section or section_number

    def toggle_spec(e: Any) -> None:
        set_spec_open(event_bool(e))

    body.append(
        SpecificationTile(
            index,
            spec,
            expanded=spec_open or store.spec_section is not None,
            on_change=toggle_spec,
            on_open=session.open_specification,
        )
    )
    return ft.Container(
        content=ft.Column(body, spacing=8, scroll=ft.ScrollMode.AUTO, expand=True),
        width=360,
        padding=ft.Padding.all(12),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=ft.BorderRadius.all(12),
    )


def concept_titles(level: int) -> list[str]:
    return [CONCEPTS[c].title for c in LEVEL_CONCEPTS.get(level, ())]


__all__ = [
    "LEVEL_CONCEPTS",
    "ExplainCardView",
    "ExplainDrawer",
    "SpecificationTile",
    "badge_view",
    "card",
    "chip",
    "concept_titles",
    "content_widths",
    "data_table",
    "details",
    "empty_state",
    "event_bool",
    "fmt_number",
    "fmt_shown",
    "hint",
    "hover_text",
    "info_button",
    "kv_rows",
    "level_header",
    "numerics_strip",
    "shown",
    "stat_row",
    "stat_tile",
    "status_line",
    "value_cell",
    "why_button",
]
