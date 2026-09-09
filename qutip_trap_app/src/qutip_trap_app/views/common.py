"""Shared Flet pieces: numbers with their provenance chips, the convergence badge, the numerics strip, the explain drawer,
headers and empty states (DESIGN.md Sections 3 to 5).

Rules carried here: plain label first, the physics term on hover, the unit always printed, Hz with the 2 pi conversion on
hover (Section 14.5); meaning never by colour alone (every tag has a glyph and a word, every badge its word); one primary
action per screen is the caller's job.
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
    if unit in ("Hz", "s", "m", "W", "1/s", "quanta/s"):
        base_unit = unit
        if unit in ("1/s", "quanta/s"):
            base_unit = unit
            prefix, scale = "", 1.0
        else:
            prefix, scale = "", 1.0
            for pfx, sc in _SI:
                if abs(value) >= sc:
                    prefix, scale = pfx, sc
                    break
        scaled = value / scale
        return f"{scaled:.{digits}g} {prefix}{base_unit}"
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
    """A provenance chip: the tag's glyph and word, with the record's section, source, equation and corrected form on hover."""
    c = index.chip(ledger_id)
    lines = [f"{c.label}: {c.meaning}", f"ledger {c.id}", f"section {c.section}", f"source: {c.source}"]
    if c.equation:
        lines.append(f"equation: {c.equation}")
    if c.corrected_form:
        lines.append(f"corrected form: {c.corrected_form}")
    text = c.glyph if compact else c.label
    return ft.Container(
        content=ft.Text(text, size=11, color=ft.Colors.ON_SECONDARY_CONTAINER, no_wrap=True),
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
        border_radius=ft.BorderRadius.all(10),
        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
        tooltip="\n".join(lines),
        margin=ft.Margin.only(left=4),
    )


def shown(
    s: Shown, index: ProvenanceIndex, *, label: bool = True, plain: bool = True, size: int = 14
) -> ft.Control:
    """One displayed value: its plain label (or the physics term), the formatted value and the chip."""
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
            ft.Text(fmt_shown(s), tooltip=hover_text(s), selectable=True),
            chip(CATALOGUE[s.quantity].ledger_id, index),
        ],
        spacing=4,
    )


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


def badge_view(badge: Badge) -> ft.Control:
    """The convergence badge: an icon, the word, and the reasons on hover (Section 14.5; never colour alone)."""
    reasons = list(badge.reasons) + [f"not run: {r}" for r in badge.checks_not_run]
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(_BADGE_ICON[badge.status], size=16, color=_BADGE_FG[badge.status]),
                ft.Text(badge.label, size=12, color=_BADGE_FG[badge.status]),
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
    """The numerics panel that accompanies every level (Section 14.2), as a collapsible strip. ``on_change`` receives the
    tile's toggle so the caller can keep the learner's choice (a fixed ``expanded`` would snap back on re-render)."""
    rows: list[ft.Control] = [
        ft.Text(panel.engine_note, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
        ft.Row(
            [
                shown(panel.level, index),
                shown(panel.dimension, index),
                shown(panel.integrator, index),
                shown(panel.tolerances, index),
            ],
            wrap=True,
            spacing=18,
        ),
        ft.Row(
            [
                shown(panel.samples, index),
                shown(panel.trajectories, index),
                shown(panel.branches, index),
                shown(panel.wall_time, index),
            ],
            wrap=True,
            spacing=18,
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
                    f"tolerances {c.tolerances} against {c.tightened_tolerances}: max change {c.max_change:.2e} ({'converged' if c.converged else 'NOT converged'})",
                    size=12,
                )
            )
    if panel.truncation_check is not None:
        t = panel.truncation_check
        rows.append(
            ft.Text(
                f"caps {t.caps} against {t.grown_caps}: max change {t.max_change:.2e} ({'converged' if t.converged else 'NOT converged'})",
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
    return ft.ExpansionTile(
        title=ft.Row(
            [ft.Text("Numerics", weight=ft.FontWeight.W_600), badge_view(panel.badge)], spacing=12, wrap=True
        ),
        subtitle=ft.Text(panel.engine_note, size=11, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
        controls=[ft.Container(content=ft.Column(rows, spacing=8), padding=ft.Padding.all(12))],
        expanded=expanded,
        on_change=on_change,
        dense=True,
    )


# ---- headers, cards, empty states -----------------------------------------------------------------------------------------------------


def content_widths(store: Any, page: Any, level: int = 0) -> tuple[float, float]:
    """(lane width, spectrum strip width) from the window: the content column is the window less the rail, the explain
    drawer when open, and the margins; absolute-positioned drawings must know their width."""
    total = float(getattr(page, "width", None) or 1200.0)
    drawer = 340.0 if store.learner.explain_is_open(level) else 0.0
    content = max(total - 80.0 - drawer - 120.0, 360.0)
    return content - 60.0, max(content * 0.5 - 60.0, 240.0)


def level_header(level: int, title: str, question: str, note: str = "") -> ft.Control:
    subtitle: list[ft.Control] = [ft.Text(question, size=14, color=ft.Colors.ON_SURFACE_VARIANT)]
    if note:
        subtitle.append(ft.Text(note, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
    return ft.Column(
        [ft.Text(f"Level {level} · {title}", theme_style=ft.TextThemeStyle.HEADLINE_SMALL)] + subtitle,
        spacing=2,
    )


def card(
    title: str, body: ft.Control, *, subtitle: str = "", actions: Sequence[ft.Control] = ()
) -> ft.Control:
    head: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.W_600, size=15)]
    if subtitle:
        head.append(ft.Text(subtitle, size=12, color=ft.Colors.ON_SURFACE_VARIANT))
    content: list[ft.Control] = [ft.Column(head, spacing=2), body]
    if actions:
        content.append(ft.Row(list(actions), spacing=8, wrap=True))
    return ft.Card(content=ft.Container(content=ft.Column(content, spacing=10), padding=ft.Padding.all(14)))


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


def kv_rows(pairs: Sequence[tuple[str, ft.Control]]) -> ft.Control:
    return ft.Column(
        [
            ft.Row(
                [ft.Text(k, size=13, color=ft.Colors.ON_SURFACE_VARIANT, width=150), v],
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
    0: ("shot", "histogram", "bitstring_order", "target_vs_simulated", "spam"),
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
    """The boolean a Flet change event carries (an ExpansionTile sends the tile's new state; the client may send it as
    text)."""
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
    """One concept card: the explanation at the learner's depth, its chips, the deeper button and its retrieval prompt.
    The tile owns its open state, so answering, skipping or deepening leaves it open (a fixed ``expanded`` prop closes
    it on every re-render)."""
    answer, set_answer = ft.use_state("")
    feedback, set_feedback = ft.use_state("")
    expanded, set_expanded = ft.use_state(False)
    concept = card_data.concept
    prompt: Prompt = card_data.prompt

    def attempt(correct: bool | None) -> None:
        # the explanation stands open above this prompt, so every attempt made here is aided; the unaided re-ask is
        # the review tray's (DESIGN.md Section 3)
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
        set_feedback(
            "skipped, and not counted as wrong; the prompt returns in the review tray after one review gap"
        )

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
        ft.Text(concept.term, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ft.Text(card_data.text, size=13, selectable=True),
        ft.Row(
            [chip(i, index, compact=False) for i in concept.ledger_ids]
            + [ft.Text(f"Section {concept.section}", size=11, color=ft.Colors.ON_SURFACE_VARIANT)],
            wrap=True,
            spacing=4,
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
    return ft.ExpansionTile(
        title=ft.Text(concept.title, weight=ft.FontWeight.W_600, size=14),
        subtitle=ft.Text(
            f"you will be able to {concept.can_do}", size=11, color=ft.Colors.ON_SURFACE_VARIANT
        ),
        controls=[
            ft.Container(
                content=ft.Column(body, spacing=8), padding=ft.Padding.symmetric(horizontal=12, vertical=6)
            )
        ],
        dense=True,
        expanded=expanded,
        on_change=lambda e: set_expanded(event_bool(e)),
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
    """The explain drawer of a level, or of one Level 4 page when ``concepts`` names the page's subset (DESIGN.md Section 5)."""
    ft.use_state(store)
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

    try:
        section = index.section(section_number)
        section_text = f"PLAN.md {section.heading} (line {section.line}); tags in this section: " + ", ".join(
            f"{k} {v}" for k, v in section.tags.items()
        )
    except KeyError:
        section_text = f"PLAN.md Section {section_number}"
    ids = concepts if concepts is not None else LEVEL_CONCEPTS.get(level, ())
    cards = [ExplainCardView(session, explain(cid, depth), index, deeper) for cid in ids]
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [ft.Icon(ft.Icons.MENU_BOOK, size=18), ft.Text("Explain", weight=ft.FontWeight.W_600)],
                    spacing=6,
                ),
                ft.Text(section_text, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Row(
                    [
                        ft.Text("depth", size=12),
                        ft.SegmentedButton(
                            segments=[ft.Segment(value=d, label=ft.Text(d, size=10)) for d in order],
                            selected=[depth],
                            on_change=set_depth,
                        ),
                    ],
                    spacing=8,
                ),
                ft.ListView(cards, spacing=4, expand=True),
            ],
            spacing=8,
            expand=True,
        ),
        width=340,
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
    "badge_view",
    "card",
    "chip",
    "concept_titles",
    "data_table",
    "empty_state",
    "event_bool",
    "fmt_number",
    "fmt_shown",
    "hover_text",
    "kv_rows",
    "level_header",
    "numerics_strip",
    "shown",
    "value_cell",
]
