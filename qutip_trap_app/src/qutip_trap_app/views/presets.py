"""The published-experiment presets on screen (PLAN.md Section 14.5; DESIGN.md Section 10): the list in the Learn view, the
page of one preset with its published numbers beside the simulated ones (each with its own chip), its chart and its method
behind Details, and the comparison table Level 0 reuses for the circuit presets."""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc
import numpy as np

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.viewmodel.catalogue import CATALOGUE
from qutip_trap_app.viewmodel.presets import PRESETS, ChartRecord, Comparison, PresetSpec, compare
from qutip_trap_app.views import drawing, theme
from qutip_trap_app.views.common import (
    MUTED,
    card,
    chip,
    data_table,
    details,
    fmt_number,
    level_header,
    status_line,
)
from qutip_trap_app.views.state import Session, Store


def _with_uncertainty(value: float, unc: float | None, unit: str, scale: float) -> str:
    v = value * scale
    if unc is None or unc <= 0.0:
        return fmt_number(v, unit if scale == 1.0 else "")
    u = unc * scale
    return f"{v:.4g} ± {u:.2g} {unit if scale == 1.0 else ''}".strip()


def _verdict_cell(c: Comparison) -> ft.Control:
    color: ft.ColorValue
    if not c.expect_agreement:
        icon, word, color = ft.Icons.INFO_OUTLINE, "not predicted", MUTED
    elif c.within:
        icon, word, color = ft.Icons.CHECK_CIRCLE_OUTLINE, "within", theme.status("pass")[1]
    else:
        icon, word, color = ft.Icons.ERROR_OUTLINE, "outside", ft.Colors.ERROR
    tip = c.verdict + (f"\n{c.why_not}" if c.why_not else "")
    return ft.Row(
        [ft.Icon(icon, size=16, color=color), ft.Text(word, size=theme.SIZE_SMALL, color=color)],
        spacing=4,
        tooltip=tip,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def comparison_table(comparisons: tuple[Comparison, ...], index: ProvenanceIndex) -> ft.Control:
    """Published beside simulated, one row per number: the two values with their chips, the difference, the verdict word
    (with the reason on hover when the simulated number is not a prediction of the published one)."""
    if not comparisons:
        return status_line("nothing to compare yet")
    rows: list[list[ft.Control | str]] = []
    for c in comparisons:
        scale_note = "" if c.scale == 1.0 else f" (in units of {1.0 / c.scale:.0e})".replace("e-0", "e-")
        rows.append(
            [
                c.label + scale_note,
                ft.Row(
                    [
                        ft.Text(
                            _with_uncertainty(
                                float(c.published.value or 0.0), c.published_uncertainty, c.unit, c.scale
                            ),
                            size=theme.SIZE_SMALL,
                            tooltip=c.published.detail,
                        ),
                        chip(CATALOGUE[c.published.quantity].ledger_id, index),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    [
                        ft.Text(
                            _with_uncertainty(
                                float(c.simulated.value or 0.0), c.simulated_uncertainty, c.unit, c.scale
                            ),
                            size=theme.SIZE_SMALL,
                            tooltip=c.simulated.detail,
                        ),
                        chip(CATALOGUE[c.simulated.quantity].ledger_id, index),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                _verdict_cell(c),
            ]
        )
    return data_table(["", "published", "this machine", "verdict"], rows)


def preset_chart(chart: ChartRecord) -> ft.Control:
    title = ft.Text(chart.title, size=theme.SIZE_SMALL, weight=ft.FontWeight.W_600)
    if chart.bars and chart.series:
        s = chart.series[0]
        groups = [
            fc.BarChartGroup(
                x=int(x),
                rods=[
                    fc.BarChartRod(
                        from_y=0.0,
                        to_y=float(y),
                        width=14,
                        color=ft.Colors.PRIMARY,
                        tooltip=f"{y:.3g}",
                        border_radius=ft.BorderRadius.only(top_left=3, top_right=3),
                    )
                ],
            )
            for x, y in zip(s.x, s.y)
        ]
        top = float(np.max(s.y)) if s.y.size else 1.0
        chart_ctl: ft.Control = fc.BarChart(
            groups=groups,
            bottom_axis=fc.ChartAxis(
                title=ft.Text(chart.x_title, size=theme.SIZE_MICRO, color=MUTED), label_size=24
            ),
            left_axis=fc.ChartAxis(
                title=ft.Text(chart.y_title, size=theme.SIZE_MICRO, color=MUTED), label_size=40
            ),
            horizontal_grid_lines=fc.ChartGridLines(color=ft.Colors.OUTLINE_VARIANT, width=1),
            max_y=top * 1.15,
            min_y=0.0,
            height=200,
            expand=True,
            interactive=True,
        )
        return ft.Column([title, chart_ctl], spacing=4)
    return ft.Column(
        [
            title,
            drawing.line_chart(
                [(s.label, np.asarray(s.x, dtype=float), np.asarray(s.y, dtype=float)) for s in chart.series],
                x_title=chart.x_title,
                y_title=chart.y_title,
                log_x=chart.log_x,
                log_y=chart.log_y,
                markers=list(chart.markers),
                height=220,
            ),
        ],
        spacing=4,
    )


@ft.component
def PresetList(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    """Every preset as a row: title, the Section 9 row it reproduces, its duration, and its state (done, running, not run)."""
    ft.use_state(store)
    page = ft.context.page
    tiles: list[ft.Control] = []
    for spec in PRESETS.values():
        running = store.running_of("preset", preset_id=spec.id) is not None
        done = spec.id in store.preset_results
        if spec.kind == "circuit":
            trailing: ft.Control = ft.Icon(ft.Icons.PLAY_CIRCLE_OUTLINE, size=18, color=MUTED)
            state = "runs on Level 0"
        elif running:
            trailing, state = ft.ProgressRing(width=16, height=16, stroke_width=2), "running"
        elif done:
            trailing, state = (
                ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=18, color=theme.status("pass")[1]),
                "done",
            )
        else:
            trailing, state = ft.Icon(ft.Icons.CHEVRON_RIGHT, size=18, color=MUTED), spec.duration

        def open_preset(_e: Any, s: PresetSpec = spec) -> None:
            if s.kind == "circuit":
                session.load_circuit_preset(s.id)
                page.navigate("/")
            else:
                page.navigate(f"/learn/preset/{s.id}")

        tiles.append(
            ft.ListTile(
                title=ft.Text(spec.title, size=theme.SIZE_BODY, weight=ft.FontWeight.W_500),
                subtitle=ft.Text(
                    f"Section {spec.section}, {spec.row} · {spec.source} · {state}",
                    size=theme.SIZE_CAPTION,
                    color=MUTED,
                ),
                trailing=trailing,
                on_click=open_preset,
                key=f"preset:{spec.id}",
            )
        )
    return ft.Column(tiles, spacing=2)


@ft.component
def PresetPage(store: Store, session: Session, index: ProvenanceIndex, preset_id: str) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    spec = PRESETS.get(preset_id)
    if spec is None:
        return card("Unknown experiment", status_line(f"no preset {preset_id!r}"))
    result = store.preset_results.get(preset_id)
    running = store.running_of("preset", preset_id=preset_id)
    header = level_header(spec.title, f"You will be able to explain {spec.question}.")
    body: list[ft.Control] = []
    if result is None:
        body.append(
            ft.Row(
                [
                    ft.FilledButton(
                        content=ft.Text("Run the experiment"),
                        icon=ft.Icons.PLAY_ARROW,
                        on_click=lambda e: session.submit_preset(preset_id),
                        disabled=running is not None,
                        key="run-preset",
                    ),
                    status_line(spec.duration if running is None else f"{running.stage}: {running.message}"),
                ],
                spacing=10,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
        if running is not None:
            body.append(
                ft.ProgressBar(value=running.fraction, bar_height=4, border_radius=ft.BorderRadius.all(2))
            )
    else:
        comps = compare(spec, result)
        body.append(comparison_table(comps, index))
        for ch in result.charts:
            body.append(preset_chart(ch))
        why_nots = [f"{c.label}: {c.why_not}" for c in comps if c.why_not]
        detail_controls: list[ft.Control] = [
            ft.Text(spec.method, size=theme.SIZE_SMALL),
            ft.Text(f"Section {spec.section}, row '{spec.row}'; source {spec.source}", size=theme.SIZE_SMALL),
            data_table(
                ["parameter", "value"], [[k, f"{v:.6g}"] for k, v in sorted(result.parameters.items())]
            ),
        ]
        detail_controls.extend(ft.Text(n, size=theme.SIZE_SMALL) for n in result.notes)
        detail_controls.extend(ft.Text(w, size=theme.SIZE_SMALL) for w in why_nots)
        detail_controls.append(status_line(f"computed in {result.wall_time_s:.1f} s"))
        body.append(details(f"preset.{preset_id}", detail_controls, store=store, level=0, session=session))
    return ft.Column(
        [
            header,
            card(
                "Published beside simulated",
                ft.Column(body, spacing=12),
                why=lambda e: session.select_concept(spec.level, spec.concept_id),
                info=(
                    "the published value's chip is the Section 9 row's tag for the source; the simulated value's chip is the "
                    "check that recomputed it (Section 14.5)"
                ),
                actions=[
                    ft.TextButton(
                        content=ft.Text("All experiments"),
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda e: page.navigate("/learn/experiments"),
                    )
                ],
                key="preset-result",
            ),
        ],
        spacing=16,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )


__all__ = ["PresetList", "PresetPage", "comparison_table", "preset_chart"]
