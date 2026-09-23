"""Level 0, the machine: the circuit editor, the histogram of the run with the ideal target beside it, and its shots."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flet as ft
import flet_charts as fc

from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.editor import MAX_QUBITS, PRESETS
from qutip_trap_app.viewmodel.machine import Histogram, histogram, shot
from qutip_trap_app.views.common import (
    CODE_FONT,
    CODE_FONT_FALLBACK,
    GAP,
    HAIRLINE,
    MUTED,
    card,
    columns,
    data_table,
    details,
    empty_state,
    kv_rows,
    level_header,
    progress_card,
    screen,
    stat_row,
    stat_tile,
    status_line,
)
from qutip_trap_app.views.state import Session, Store


def editor_card(store: Store, session: Session) -> ft.Control:
    def set_text(e: Any) -> None:
        store.circuit_text = str(e.control.value)

    body: list[ft.Control] = [
        ft.TextField(
            value=store.circuit_text,
            multiline=True,
            min_lines=8,
            max_lines=20,
            text_style=ft.TextStyle(size=12, font_family=CODE_FONT, font_family_fallback=CODE_FONT_FALLBACK),
            hint_text="an OpenQASM 2 program, or an IonQ circuit JSON object",
            on_change=set_text,
        ),
        ft.Row(
            [
                ft.TextField(
                    label="shots",
                    value=str(store.shots),
                    width=110,
                    dense=True,
                    on_change=lambda e: session.set_shots(str(e.control.value)),
                ),
                ft.Dropdown(
                    label="load a preset",
                    options=[ft.DropdownOption(key=name, text=name) for name in PRESETS],
                    on_select=lambda e: session.load_preset(str(e.control.value)),
                    width=240,
                    dense=True,
                ),
            ],
            wrap=True,
            spacing=GAP,
        ),
    ]
    if store.error:
        body.append(ft.Text(store.error, color=ft.Colors.ERROR, size=12))
    return card(
        "Circuit",
        ft.Column(body, spacing=12),
        actions=[
            ft.FilledButton(
                content=ft.Text("Run"),
                icon=ft.Icons.PLAY_ARROW,
                on_click=lambda _e: session.submit_run(),
                disabled=store.running_of("run") is not None,
            ),
            status_line(f"the full simulation on the example 171Yb+ chain, up to {MAX_QUBITS} qubits"),
        ],
    )


def _histogram_chart(h: Histogram, on_bar: Callable[[str], None]) -> ft.Control:
    """One filled bar of recorded shots per outcome with the ideal circuit's outlined bar beside it."""
    cap = ft.BorderRadius.only(top_left=4, top_right=4)
    groups = [
        fc.BarChartGroup(
            x=k,
            rods=[
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=b.probability,
                    width=18,
                    color=ft.Colors.PRIMARY,
                    tooltip=f"simulated {b.probability:.3f} ± {b.error_bar:.3f} ({b.count} shots); click for them",
                    border_radius=cap,
                ),
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=b.target,
                    width=18,
                    color=ft.Colors.with_opacity(0.12, ft.Colors.TERTIARY),
                    border_side=ft.BorderSide(1.5, ft.Colors.TERTIARY),
                    tooltip=f"ideal {b.target:.3f}",
                    border_radius=cap,
                ),
            ],
            spacing=3,
        )
        for k, b in enumerate(h.bars)
    ]
    top = min(1.0, 1.15 * max([max(b.probability, b.target) for b in h.bars] + [0.1]))

    def on_event(e: fc.BarChartEvent) -> None:
        if e.type == fc.ChartEventType.TAP_UP and e.group_index is not None:
            on_bar(h.bars[int(e.group_index)].key)

    chart: ft.Control = fc.BarChart(
        groups=groups,
        bottom_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(value=k, label=ft.Text(b.key, size=12, color=MUTED))
                for k, b in enumerate(h.bars)
            ],
            label_size=28,
        ),
        left_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(value=v, label=ft.Text(f"{v:.2f}", size=11, color=MUTED))
                for v in (0.0, 0.25, 0.5, 0.75, 1.0)
                if v <= top + 0.01
            ],
            label_size=40,
        ),
        max_y=top,
        min_y=0.0,
        horizontal_grid_lines=fc.ChartGridLines(interval=0.25, color=ft.Colors.OUTLINE_VARIANT, width=1),
        interactive=True,
        on_event=on_event,
        height=280,
        expand=True,
    )
    return chart


def _legend() -> ft.Control:
    def swatch(color: str, border: str | None = None) -> ft.Control:
        return ft.Container(
            width=12,
            height=12,
            bgcolor=color,
            border=ft.Border.all(1.5, border) if border else None,
            border_radius=ft.BorderRadius.all(3),
        )

    return ft.Row(
        [
            swatch(ft.Colors.PRIMARY),
            ft.Text("simulated", size=12, color=MUTED),
            swatch(ft.Colors.with_opacity(0.12, ft.Colors.TERTIARY), ft.Colors.TERTIARY),
            ft.Text("ideal", size=12, color=MUTED),
        ],
        spacing=6,
        alignment=ft.MainAxisAlignment.END,
    )


def _shots_panel(store: Store, record: Record, h: Histogram) -> ft.Control:
    """The recorded shots behind the selected bar, and one of them opened."""
    bar = next((b for b in h.bars if b.key == store.selected_bar), None)
    if bar is None:
        return ft.Container()

    def select(i: int) -> Callable[[Any], None]:
        def on_click(_e: Any) -> None:
            store.selected_shot = i

        return on_click

    def close(_e: Any) -> None:
        store.selected_bar = None

    items: list[ft.Control] = [
        ft.Row(
            [
                ft.Text(
                    f"{len(bar.shots)} shots gave {bar.key}", size=13, weight=ft.FontWeight.W_600, expand=True
                ),
                ft.IconButton(icon=ft.Icons.CLOSE, icon_size=16, tooltip="close", on_click=close),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        ft.Row(
            [
                ft.TextButton(
                    content=ft.Text(str(i), size=12),
                    on_click=select(i),
                    style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=6, vertical=2)),
                )
                for i in bar.shots[:120]
            ],
            wrap=True,
            spacing=2,
            run_spacing=2,
        ),
    ]
    if store.selected_shot is not None and store.selected_shot in bar.shots:
        sv = shot(record, store.selected_shot)
        items.append(
            kv_rows(
                [
                    ("shot", ft.Text(f"{sv.index}: {sv.bitstring}", size=12)),
                    ("sampled levels, ion 0 first", ft.Text(str(sv.levels), size=12)),
                    ("flags", ft.Text(", ".join(sv.heralds) or "none", size=12)),
                    ("dynamical sample", ft.Text(str(sv.sample_index), size=12)),
                ]
            )
        )
    return ft.Container(
        content=ft.Column(items, spacing=8),
        padding=ft.Padding.all(12),
        border=ft.Border.all(1, HAIRLINE),
        border_radius=ft.BorderRadius.all(8),
    )


def results_card(store: Store, record: Record) -> ft.Control:
    h = histogram(record)

    def on_bar(key: str) -> None:
        store.selected_bar = key
        store.selected_shot = None

    notes = ["keys read qubit 0 rightmost", f"simulated at {record.diagnostics.level}"]
    if h.discarded_shots:
        notes.append(f"{h.discarded_shots} shots discarded by the machine's flags")
    return card(
        "Results",
        ft.Column(
            [
                _histogram_chart(h, on_bar),
                _legend(),
                stat_row([stat_tile(h.shots), stat_tile(h.target_distance), stat_tile(h.largest_deviation)]),
                _shots_panel(store, record, h),
                details(
                    store,
                    "results",
                    [
                        data_table(
                            ["outcome", "count", "probability", "error bar", "ideal"],
                            [
                                [
                                    b.key,
                                    str(b.count),
                                    f"{b.probability:.4f}",
                                    f"{b.error_bar:.4f}",
                                    f"{b.target:.4f}",
                                ]
                                for b in h.bars
                            ],
                            numeric=[False, True, True, True, True],
                        ),
                        *(status_line(n) for n in notes),
                    ],
                    title="The numbers",
                ),
            ],
            spacing=12,
        ),
    )


def level0_page(store: Store, session: Session, record: Record | None) -> ft.Control:
    header = level_header(
        "The machine", "What did the machine return, and how does it compare with a perfect one?"
    )
    if record is None:
        body: list[ft.Control] = [editor_card(store, session), progress_card(store, session)]
        if not store.running():
            body.append(empty_state("No result yet", "Run the circuit: its histogram appears here."))
        return screen([header, *body])
    left = [results_card(store, record), progress_card(store, session)]
    return screen([header, columns(store, left, [editor_card(store, session)], split=(3, 2))])
