"""Level 2, the schedule (PLAN.md Section 14.2 row 2; DESIGN.md Section 5): pulses on ion lanes, the selected pulse's tones
against the mode spectrum with each two-indexed detuning named, the waveform's segment table, the loop-closure indicators,
crosstalk onto neighbours and the beams' directions."""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.schedule import closure, pulse_view, time_axis
from qutip_trap_app.views.common import (
    card,
    content_widths,
    data_table,
    kv_rows,
    level_header,
    shown,
    value_cell,
)
from qutip_trap_app.views.state import Store

ROLE_COLORS = {
    "red": ft.Colors.ERROR_CONTAINER,
    "blue": ft.Colors.PRIMARY_CONTAINER,
    "carrier": ft.Colors.TERTIARY_CONTAINER,
    "far": ft.Colors.SURFACE_CONTAINER_HIGHEST,
}


def _lanes(record: Record, selected: int, on_select: Any, width: float) -> ft.Control:
    axis = time_axis(record)
    t0, t1 = axis.t0_s, max(axis.duration_s, axis.t0_s + 1e-9)
    scale = width / (t1 - t0)
    lanes: list[ft.Control] = []
    for lane in axis.lanes:
        boxes: list[ft.Control] = []
        for span in lane.spans:
            is_sel = span.pulse_index == selected
            boxes.append(
                ft.Container(
                    content=ft.Text(span.gate_id or "", size=9, no_wrap=True, overflow=ft.TextOverflow.CLIP),
                    left=(span.t_start_s - t0) * scale,
                    top=2,
                    width=max((span.t_end_s - span.t_start_s) * scale, 12.0),
                    height=26,
                    bgcolor=ft.Colors.PRIMARY_CONTAINER
                    if span.kind == "raman"
                    else ft.Colors.TERTIARY_CONTAINER,
                    border=ft.Border.all(
                        2 if is_sel else 1, ft.Colors.PRIMARY if is_sel else ft.Colors.OUTLINE_VARIANT
                    ),
                    border_radius=ft.BorderRadius.all(3),
                    padding=ft.Padding.symmetric(horizontal=3),
                    alignment=ft.Alignment.CENTER_LEFT,
                    tooltip=f"pulse {span.pulse_index} ({span.kind}) {span.t_start_s * 1e6:.2f} to {span.t_end_s * 1e6:.2f} µs; gate {span.gate_id}",
                    on_click=lambda e, k=span.pulse_index: on_select(k),
                    ink=True,
                )
            )
        lanes.append(
            ft.Row(
                [ft.Text(f"ion {lane.ion}", size=12, width=44), ft.Stack(boxes, width=width, height=30)],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    marks: list[ft.Control] = []
    if axis.measurement is not None:
        t_m = float(axis.measurement.value or 0.0)
        marks.append(
            ft.Container(
                left=(t_m - t0) * scale,
                top=0,
                width=max((t1 - t_m) * scale, 4.0),
                height=14,
                bgcolor=ft.Colors.with_opacity(0.4, ft.Colors.TERTIARY),
                tooltip="detection window (measure event)",
            )
        )
    lanes.append(
        ft.Row([ft.Text("readout", size=12, width=44), ft.Stack(marks, width=width, height=16)], spacing=6)
    )
    lanes.append(
        ft.Row(
            [
                ft.Text("", width=44),
                ft.Text(f"{t0 * 1e6:.0f} µs", size=10),
                ft.Container(expand=True),
                ft.Text(f"{t1 * 1e6:.1f} µs", size=10),
            ],
            spacing=6,
            width=width + 50,
        )
    )
    return ft.Column(lanes, spacing=2)


def _spectrum_strip(pv: Any, width: float) -> ft.Control:
    """Mode frequencies as ticks and the pulse's tone detunings (their absolute values) drawn against them."""
    modes = [(float(m.value or 0.0), m.detail or "") for m in pv.mode_spectrum]
    tones = [abs(float(t.detuning.value or 0.0)) for t in pv.tones]
    values = [f for f, _ in modes] + tones
    lo, hi = min(values) * 0.97, max(values) * 1.03 + 1.0
    scale = width / (hi - lo)
    controls: list[ft.Control] = [
        ft.Container(left=0, top=30, width=width, height=1, bgcolor=ft.Colors.OUTLINE)
    ]
    label_right = -1e9
    for f, label in sorted(modes):
        x = (f - lo) * scale
        controls.append(
            ft.Container(
                left=x - 1,
                top=8,
                width=2,
                height=24,
                bgcolor=ft.Colors.ON_SURFACE,
                tooltip=f"{label}: {f / 1e6:.4f} MHz",
            )
        )
        # one label per 46 px: close-lying modes keep their ticks and tooltips, their labels do not pile up
        if x - 22.0 >= label_right + 2.0:
            controls.append(
                ft.Container(
                    content=ft.Text(f"{f / 1e6:.3f}", size=10),
                    left=x - 22,
                    top=34,
                    width=44,
                    alignment=ft.Alignment.CENTER,
                )
            )
            label_right = x + 22.0
    for k, mu in enumerate(tones):
        # the 10 px dot stays inside the strip (a carrier tone sits at the far left edge)
        x = min(max((mu - lo) * scale, 5.0), width - 5.0)
        sign = "+" if float(pv.tones[k].detuning.value or 0.0) >= 0 else "-"
        role = next(
            (sb.role for sb in pv.tones[k].sidebands if sb.role in ("red", "blue")),
            "carrier" if mu < 1e5 else "far",
        )
        controls.append(
            ft.Container(
                left=x - 5,
                top=0,
                width=10,
                height=10,
                border_radius=ft.BorderRadius.all(5),
                bgcolor=ROLE_COLORS.get(role, ft.Colors.PRIMARY),
                border=ft.Border.all(1, ft.Colors.ON_SURFACE),
                tooltip=f"tone {k}: mu = {sign}{mu / 1e6:.4f} MHz from the carrier ({role} sideband)",
            )
        )
    return ft.Column(
        [
            ft.Stack(controls, width=width, height=52),
            ft.Text(
                "|detuning from the carrier| in MHz; ticks are the crystal's modes, dots the tones (red = below a mode, blue = above)",
                size=11,
                color=ft.Colors.ON_SURFACE_VARIANT,
            ),
        ],
        spacing=2,
    )


def _envelope_chart(pv: Any) -> ft.Control:
    series = []
    for k, t in enumerate(pv.tones):
        pts = [
            fc.LineChartDataPoint(x=float(x * 1e6), y=float(y / 1e3))
            for x, y in zip(t.times_s, t.envelope_samples)
        ]
        series.append(
            fc.LineChartData(
                points=pts, stroke_width=2, color=ft.Colors.PRIMARY if k == 0 else ft.Colors.TERTIARY
            )
        )
    if not series:
        return ft.Container()
    chart: ft.Control = fc.LineChart(
        data_series=series,
        height=140,
        expand=True,
        left_axis=fc.ChartAxis(title=ft.Text("Omega/2pi (kHz)", size=10), label_size=40),
        bottom_axis=fc.ChartAxis(title=ft.Text("t (µs)", size=10), label_size=24),
        interactive=False,
    )
    return chart


@ft.component
def Level2Page(store: Store, record: Record, pulse_param: str, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    pulses = record.schedule.pulses
    if not pulses:
        return ft.Text("this schedule has no pulses", size=13)
    try:
        sel = int(pulse_param)
    except ValueError:
        sel = 0
    sel = min(max(sel, 0), len(pulses) - 1)
    page = ft.context.page
    key = record.key()
    pv = pulse_view(record, sel)
    lanes_width, strip_width = content_widths(store, page, 2)
    step = record.step(pv.step_index) if pv.step_index >= 0 else None

    def on_select(k: int) -> None:
        page.navigate(f"/job/{key}/schedule/{k}")

    tone_rows: list[list[ft.Control | str]] = []
    for t in pv.tones:
        tone_rows.append(
            [
                str(t.index),
                value_cell(t.detuning, index),
                value_cell(t.envelope_peak, index),
                value_cell(t.phase_start, index),
            ]
        )
    sideband_rows: list[list[ft.Control | str]] = []
    for t in pv.tones:
        for sb in t.sidebands:
            sideband_rows.append(
                [
                    f"tone {t.index}",
                    str(sb.mode),
                    value_cell(sb.mode_frequency, index),
                    value_cell(sb.detuning, index),
                    sb.role,
                ]
            )
    pulse_rows: list[tuple[str, ft.Control]] = [
        ("pulse", ft.Text(f"{sel}: {pv.pulse.gate_id} ({pv.pulse.kind}) on ions {pv.ions}")),
        (
            "time",
            ft.Text(
                f"{pv.pulse.t_start_s * 1e6:.2f} to {pv.pulse.t_end_s * 1e6:.2f} µs ({pv.pulse.duration_s * 1e6:.2f} µs)"
            ),
        ),
        ("beams", ft.Row(list(shown(b, index, label=False) for b in pv.beams), wrap=True, spacing=8)),
        ("Stark shift", shown(pv.stark_shift, index, label=False)),
    ]
    if pv.crosstalk:
        pulse_rows.append(
            ("crosstalk", ft.Row([shown(c, index, label=False) for c in pv.crosstalk], wrap=True, spacing=8))
        )
    zoom_target = f"/job/{key}/dynamics/{sel}/0"
    pulse_card = card(
        f"Pulse {sel}",
        ft.Column(
            [
                kv_rows(pulse_rows),
                ft.Text("tones", size=12, weight=ft.FontWeight.W_600),
                data_table(
                    ["tone", "detuning from the carrier", "peak Rabi frequency", "phase at start"], tone_rows
                ),
                _envelope_chart(pv),
            ],
            spacing=8,
        ),
        subtitle="the scheduler's own tones, sampled where they were functions",
        actions=[
            ft.FilledButton(
                content=ft.Text("Zoom in: inside this pulse"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda e: page.navigate(zoom_target),
            )
        ],
    )
    spectrum_card = card(
        "Tones against the mode spectrum",
        ft.Column(
            [
                _spectrum_strip(pv, strip_width),
                data_table(["tone", "mode", "omega_m", "delta = mu - omega_m", "role"], sideband_rows),
            ],
            spacing=8,
        ),
        subtitle="which motion each tone talks to (conv.detuning_symbols)",
    )
    right: list[ft.Control] = [spectrum_card]
    gate_id = pv.pulse.gate_id.split("/")[0] if pv.pulse.gate_id else None
    if gate_id and any(g.gate_id == gate_id for g in record.schedule.gates):
        cl = closure(record, gate_id)
        mode_rows: list[list[ft.Control | str]] = [
            [
                str(m.mode),
                m.mode_class,
                value_cell(m.chi_m, index),
                value_cell(m.alpha_m, index),
                f"{m.residual:.2e}",
                "closed" if m.closed else "OPEN",
            ]
            for m in cl.modes
        ]
        seg_rows: list[list[ft.Control | str]] = [
            [str(k)] + [value_cell(s, index) for s in row] for k, row in enumerate(cl.segments)
        ]
        seg_cols = ["segment"] + ([""] * (len(cl.segments[0]) if cl.segments else 0))
        right.append(
            card(
                f"Loop closure of {gate_id}",
                ft.Column(
                    [
                        ft.Row(
                            [shown(cl.chi_total, index), shown(cl.duration, index)]
                            + ([shown(cl.beat_phase, index)] if cl.beat_phase is not None else []),
                            wrap=True,
                            spacing=12,
                        ),
                        data_table(
                            ["mode", "class", "chi_m", "|alpha_m| at closure", "|alpha|^2 (2n+1)", "loop"],
                            mode_rows,
                        ),
                        ft.Text(
                            "segments (duration, then Omega per ion and leg, then the leg detunings)",
                            size=12,
                            weight=ft.FontWeight.W_600,
                        ),
                        data_table(seg_cols, seg_rows)
                        if seg_rows
                        else ft.Text("Fourier-parameterized waveform", size=12),
                    ],
                    spacing=8,
                ),
                subtitle="the played waveform returns the motion it borrowed, mode by mode (Section 4.4.3)",
            )
        )
    note = f"step {step.gate_id}" if step is not None else ""
    return ft.Column(
        [
            level_header(2, "The schedule", "What light hit which ion, when, and what did it talk to?", note),
            card(
                "Time axis",
                _lanes(record, sel, on_select, lanes_width),
                subtitle="pulses per ion lane; the readout window at the end; click a pulse",
            ),
            ft.ResponsiveRow(
                [
                    ft.Column([pulse_card], col={"xs": 12, "lg": 6}, spacing=10),
                    ft.Column(right, col={"xs": 12, "lg": 6}, spacing=10),
                ],
                vertical_alignment=ft.CrossAxisAlignment.START,
                spacing=12,
                run_spacing=12,
            ),
        ],
        spacing=12,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )


__all__ = ["Level2Page"]
