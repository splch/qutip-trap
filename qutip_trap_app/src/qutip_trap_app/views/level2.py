"""Level 2, the schedule: the selected pulse's tones against the motional modes, the pulses on ion lanes with the detection
window, the pulse's envelope, and the loop closure of its entangling gate."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft
import flet_charts as fc

from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.schedule import ClosureView, PulseView, closure, pulse_view, time_axis
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import (
    HAIRLINE,
    MUTED,
    card,
    columns,
    data_table,
    details,
    ions_text,
    level_header,
    pill,
    screen,
    stat_row,
    stat_tile,
    status_line,
    value_text,
)
from qutip_trap_app.views.level1 import lane_label, lane_width, time_axis_row
from qutip_trap_app.views.state import Session, Store

TONE_COLORS = {
    "red": ft.Colors.RED,
    "blue": ft.Colors.BLUE,
    "carrier": ft.Colors.OUTLINE,
    "far": ft.Colors.OUTLINE,
}


def _lanes(record: Record, selected: int, on_select: Callable[[int], None], width: float) -> ft.Control:
    axis = time_axis(record)
    scale = width / (axis.t1_s - axis.t0_s)
    rows: list[ft.Control] = []
    for ion, spans in axis.lanes:
        boxes: list[ft.Control] = [ft.Container(left=0, top=14, width=width, height=1, bgcolor=HAIRLINE)]
        for span in spans:
            chosen = span.pulse_index == selected
            boxes.append(
                ft.Container(
                    content=ft.Text(span.gate_id or "", size=10, no_wrap=True, overflow=ft.TextOverflow.CLIP),
                    left=(span.t_start_s - axis.t0_s) * scale,
                    top=2,
                    width=max((span.t_end_s - span.t_start_s) * scale, 12.0),
                    height=26,
                    bgcolor=ft.Colors.PRIMARY_CONTAINER
                    if span.kind == "raman"
                    else ft.Colors.TERTIARY_CONTAINER,
                    border=ft.Border.all(2 if chosen else 1, ft.Colors.PRIMARY if chosen else HAIRLINE),
                    border_radius=ft.BorderRadius.all(5),
                    padding=ft.Padding.symmetric(horizontal=4),
                    alignment=ft.Alignment.CENTER_LEFT,
                    tooltip=f"pulse {span.pulse_index} ({span.kind}), {span.t_start_s * 1e6:.2f} to "
                    f"{span.t_end_s * 1e6:.2f} µs, gate {span.gate_id}",
                    on_click=lambda _e, k=span.pulse_index: on_select(k),
                    ink=True,
                )
            )
        rows.append(
            ft.Row(
                [lane_label(f"ion {ion}"), ft.Stack(boxes, width=width, height=30)],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    marks: list[ft.Control] = []
    if axis.measure is not None:
        t_m = axis.measure[0]
        marks.append(
            ft.Container(
                left=(t_m - axis.t0_s) * scale,
                top=0,
                width=max((axis.t1_s - t_m) * scale, 4.0),
                height=14,
                bgcolor=ft.Colors.with_opacity(0.35, ft.Colors.TERTIARY),
                border_radius=ft.BorderRadius.all(3),
                tooltip="the detection window",
            )
        )
    rows.append(
        ft.Row(
            [lane_label("readout"), ft.Stack(marks, width=width, height=16)],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )
    rows.append(time_axis_row(axis.t0_s, axis.t1_s, width))
    return ft.Column(rows, spacing=2)


def _spectrum(pv: PulseView, width: float) -> ft.Control:
    """The mode frequencies as ticks and the tones' distances from the carrier as dots on one frequency axis."""
    tones = [abs(float(t.detuning.value or 0.0)) for t in pv.tones]
    values = [f for _m, _name, f in pv.modes] + tones
    lo, hi = min(values) * 0.97, max(values) * 1.03 + 1.0
    scale = width / (hi - lo)
    controls: list[ft.Control] = [
        ft.Container(left=0, top=44, width=width, height=1, bgcolor=ft.Colors.OUTLINE)
    ]
    label_right = -1e9
    for m, name, f in sorted(pv.modes, key=lambda mode: mode[2]):
        x = (f - lo) * scale
        controls.append(
            ft.Container(
                left=x - 1,
                top=16,
                width=2,
                height=30,
                bgcolor=ft.Colors.ON_SURFACE,
                tooltip=f"mode {m} ({name}): {f / 1e6:.4f} MHz",
            )
        )
        if x - 22.0 >= label_right + 2.0:  # close-lying modes keep their ticks; their labels do not pile up
            controls.append(
                ft.Container(
                    content=ft.Text(f"{f / 1e6:.3f}", size=10, color=MUTED),
                    left=x - 22,
                    top=48,
                    width=44,
                    alignment=ft.Alignment.CENTER,
                )
            )
            label_right = x + 22.0
    for t, mu in zip(pv.tones, tones):
        x = min(max((mu - lo) * scale, 7.0), width - 7.0)
        controls.append(
            ft.Container(
                left=x - 7,
                top=2,
                width=14,
                height=14,
                border_radius=ft.BorderRadius.all(7),
                bgcolor=TONE_COLORS[t.role],
                tooltip=f"tone {t.index}: {mu / 1e6:.4f} MHz from the carrier ({t.role})",
            )
        )

    def dot(color: str) -> ft.Control:
        return ft.Container(width=12, height=12, bgcolor=color, border_radius=6)

    legend = ft.Row(
        [
            dot(TONE_COLORS["red"]),
            ft.Text("red-sideband tone", size=11, color=MUTED),
            dot(TONE_COLORS["blue"]),
            ft.Text("blue-sideband tone", size=11, color=MUTED),
            ft.Container(width=2, height=14, bgcolor=ft.Colors.ON_SURFACE),
            ft.Text("mode (MHz)", size=11, color=MUTED),
        ],
        spacing=6,
        wrap=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    return ft.Column([ft.Stack(controls, width=width, height=66), legend], spacing=6)


def _envelope_chart(pv: PulseView) -> ft.Control:
    if not pv.tones:
        return status_line("this pulse has no tone")
    chart: ft.Control = fc.LineChart(
        data_series=[
            fc.LineChartData(
                points=[
                    fc.LineChartDataPoint(x=float(x * 1e6), y=float(y / 1e3))
                    for x, y in zip(t.times_s, t.envelope_hz)
                ],
                stroke_width=2,
                color=drawing.SERIES[k % len(drawing.SERIES)],
                rounded_stroke_cap=True,
            )
            for k, t in enumerate(pv.tones)
        ],
        height=130,
        expand=True,
        left_axis=fc.ChartAxis(title=ft.Text("Omega/2 pi (kHz)", size=10, color=MUTED), label_size=40),
        bottom_axis=fc.ChartAxis(title=ft.Text("t (µs)", size=10, color=MUTED), label_size=24),
        horizontal_grid_lines=fc.ChartGridLines(color=ft.Colors.OUTLINE_VARIANT, width=1),
        interactive=False,
    )
    return chart


def _closure_card(store: Store, cl: ClosureView) -> ft.Control:
    return card(
        f"Loop closure of {cl.gate_id}",
        ft.Column(
            [
                ft.Row(
                    [
                        pill(
                            f"mode {m.mode}: {'closed' if m.closed else 'open'}",
                            m.closed,
                            f"|alpha|^2 (2 nbar + 1) = {m.residual:.2e} against 1e-6; mode class {m.mode_class}",
                        )
                        for m in cl.modes
                    ],
                    wrap=True,
                    spacing=6,
                    run_spacing=6,
                ),
                stat_row([stat_tile(cl.chi_total), stat_tile(cl.duration)]),
                status_line(
                    "the spin-dependent force drives each mode round a loop; a closed loop leaves the motion as it "
                    "found it and the entangling angle behind"
                ),
                details(
                    store,
                    "closure",
                    [
                        data_table(
                            ["mode", "class", "angle", "|alpha| at the end", "|alpha|^2 (2n + 1)", "loop"],
                            [
                                [
                                    str(m.mode),
                                    m.mode_class,
                                    value_text(m.chi_m),
                                    value_text(m.alpha),
                                    f"{m.residual:.2e}",
                                    "closed" if m.closed else "OPEN",
                                ]
                                for m in cl.modes
                            ],
                        )
                    ],
                    title="Per mode",
                ),
            ],
            spacing=12,
        ),
    )


def level2_page(store: Store, session: Session, record: Record, pulse_param: str) -> ft.Control:
    header = level_header("The schedule", "What light hit which ion, when, and which motion did it drive?")
    pulses = record.schedule.pulses
    if not pulses:
        return screen([header, status_line("this schedule plays no pulse")])
    sel = min(max(int(pulse_param) if pulse_param.isdigit() else 0, 0), len(pulses) - 1)
    key = record.key
    pv = pulse_view(record, sel)
    width = lane_width(store)
    p = pv.pulse
    pulse_card = card(
        f"Pulse {sel}: {p.gate_id} on {ions_text(p.ions)}",
        ft.Column(
            [
                status_line(f"{p.kind}, {p.t_start_s * 1e6:.2f} to {p.t_end_s * 1e6:.2f} µs"),
                stat_row([stat_tile(pv.stark_shift), *(stat_tile(c) for c in pv.crosstalk[:2])]),
                _envelope_chart(pv),
            ],
            spacing=12,
        ),
        actions=[
            ft.FilledButton(
                content=ft.Text("Zoom in: inside this pulse"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda _e: session.navigate(f"/job/{key}/dynamics/{sel}/0"),
            )
        ],
    )
    right: list[ft.Control] = []
    gate_id = p.gate_id.split("/")[0] if p.gate_id else None
    if gate_id and any(g.gate_id == gate_id for g in record.schedule.gates):
        right.append(_closure_card(store, closure(record, gate_id)))
    tone_tiles = [stat_tile(s) for t in pv.tones for s in (t.detuning, t.envelope_peak)]
    spectrum = card(
        "Tones against the modes",
        ft.Column(
            [
                _spectrum(pv, width),
                stat_row(tone_tiles[:6]),
                details(
                    store,
                    "sidebands",
                    [
                        data_table(
                            ["tone", "mode", "mode frequency", "from its sideband", "role"],
                            [
                                [
                                    str(t.index),
                                    str(sb.mode),
                                    f"{sb.mode_frequency_hz / 1e6:.4f} MHz",
                                    f"{sb.detuning_hz / 1e3:+.3f} kHz",
                                    sb.role,
                                ]
                                for t in pv.tones
                                for sb in t.sidebands
                            ],
                        )
                    ],
                    title="Every tone against every mode",
                ),
            ],
            spacing=12,
        ),
    )
    return screen(
        [
            header,
            spectrum,
            card(
                "Time axis",
                _lanes(record, sel, lambda k: session.navigate(f"/job/{key}/schedule/{k}"), width),
            ),
            columns(store, [pulse_card], right),
        ]
    )
