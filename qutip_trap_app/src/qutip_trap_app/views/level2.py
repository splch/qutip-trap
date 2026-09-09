"""Level 2, the schedule (PLAN.md Section 14.2 row 2; DESIGN.md Sections 5 and 10): the selected pulse's tones against the
mode spectrum as the focal picture, the pulses on ion lanes, the pulse's stat tiles and envelope, the loop-closure pills per
mode, and, for an entangling pulse, the second request control of Section 14.4 (a beat-note detuning set by hand, played as
written). The tone, sideband and segment tables are behind Details."""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.requests import RequestError, request_detuning
from qutip_trap_app.viewmodel.schedule import closure, pulse_view, time_axis
from qutip_trap_app.views.common import (
    card,
    content_widths,
    data_table,
    details,
    hint,
    level_header,
    stat_row,
    stat_tile,
    status_line,
    value_cell,
)
from qutip_trap_app.views.state import Session, Store

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
                    key=f"pulse:{span.pulse_index}"
                    if lane.ion == min(lane.ion for lane in axis.lanes)
                    else None,
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
    """Mode frequencies as ticks and the pulse's tone detunings (their absolute values) drawn against them: the focal picture."""
    modes = [(float(m.value or 0.0), m.detail or "") for m in pv.mode_spectrum]
    tones = [abs(float(t.detuning.value or 0.0)) for t in pv.tones]
    values = [f for f, _ in modes] + tones
    lo, hi = min(values) * 0.97, max(values) * 1.03 + 1.0
    scale = width / (hi - lo)
    controls: list[ft.Control] = [
        ft.Container(left=0, top=44, width=width, height=1, bgcolor=ft.Colors.OUTLINE)
    ]
    label_right = -1e9
    for f, label in sorted(modes):
        x = (f - lo) * scale
        controls.append(
            ft.Container(
                left=x - 1,
                top=16,
                width=2,
                height=30,
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
                    top=48,
                    width=44,
                    alignment=ft.Alignment.CENTER,
                )
            )
            label_right = x + 22.0
    for k, mu in enumerate(tones):
        # the 14 px dot stays inside the strip (a carrier tone sits at the far left edge)
        x = min(max((mu - lo) * scale, 7.0), width - 7.0)
        sign = "+" if float(pv.tones[k].detuning.value or 0.0) >= 0 else "-"
        role = next(
            (sb.role for sb in pv.tones[k].sidebands if sb.role in ("red", "blue")),
            "carrier" if mu < 1e5 else "far",
        )
        controls.append(
            ft.Container(
                left=x - 7,
                top=2,
                width=14,
                height=14,
                border_radius=ft.BorderRadius.all(7),
                bgcolor=ROLE_COLORS.get(role, ft.Colors.PRIMARY),
                border=ft.Border.all(1.5, ft.Colors.ON_SURFACE),
                tooltip=f"tone {k}: mu = {sign}{mu / 1e6:.4f} MHz from the carrier ({role} sideband)",
            )
        )
    legend = ft.Row(
        [
            ft.Container(
                width=12,
                height=12,
                bgcolor=ROLE_COLORS["red"],
                border=ft.Border.all(1, ft.Colors.ON_SURFACE),
                border_radius=6,
            ),
            ft.Text("red tone", size=11),
            ft.Container(
                width=12,
                height=12,
                bgcolor=ROLE_COLORS["blue"],
                border=ft.Border.all(1, ft.Colors.ON_SURFACE),
                border_radius=6,
            ),
            ft.Text("blue tone", size=11),
            ft.Container(width=2, height=14, bgcolor=ft.Colors.ON_SURFACE),
            ft.Text("mode (|detuning| in MHz)", size=11),
        ],
        spacing=6,
        wrap=True,
    )
    return ft.Column([ft.Stack(controls, width=width, height=66), legend], spacing=4)


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
        height=130,
        expand=True,
        left_axis=fc.ChartAxis(title=ft.Text("Omega/2pi (kHz)", size=10), label_size=40),
        bottom_axis=fc.ChartAxis(title=ft.Text("t (µs)", size=10), label_size=24),
        interactive=False,
    )
    return chart


def _closure_pill(m: Any) -> ft.Control:
    ok = m.closed
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE if ok else ft.Icons.RADIO_BUTTON_UNCHECKED, size=14),
                ft.Text(f"mode {m.mode}: {'closed' if ok else 'open'}", size=12),
            ],
            spacing=4,
            tight=True,
        ),
        bgcolor=ft.Colors.PRIMARY_CONTAINER if ok else ft.Colors.ERROR_CONTAINER,
        border_radius=ft.BorderRadius.all(12),
        padding=ft.Padding.symmetric(horizontal=8, vertical=3),
        tooltip=f"|alpha_{m.mode}|^2 (2 nbar + 1) = {m.residual:.2e} against 1e-6 (Section 5.2); mode class {m.mode_class}",
    )


@ft.component
def RequestDetuningPanel(
    store: Store, session: Session, record: Record, gate_id: str, index: ProvenanceIndex
) -> ft.Control:
    """Section 14.4 at Level 2: set the pair's beat-note detuning by hand; the waveform is played as written."""
    ft.use_state(store)
    text, set_text = ft.use_state("5")
    running = store.running_of("request_run", gate_id=gate_id) is not None

    def submit(_e: Any) -> None:
        try:
            offset_khz = float(text)
        except ValueError:
            store.error = "the offset must be a number in kHz"
            return
        try:
            req = request_detuning(record, gate_id, offset_khz * 1e3)
        except RequestError as exc:
            store.error = str(exc)
            return
        session.submit_request(req)

    controls: list[ft.Control] = [
        ft.Row(
            [
                ft.TextField(
                    label="offset (kHz)",
                    value=text,
                    width=120,
                    dense=True,
                    text_size=13,
                    on_change=lambda e: set_text(str(e.control.value)),
                    key="request-detuning-value",
                ),
                ft.FilledTonalButton(
                    content=ft.Text("Set the detuning by hand"),
                    icon=ft.Icons.TUNE,
                    on_click=submit,
                    disabled=running,
                    tooltip="applied as written: blue legs up, red legs down, amplitude unchanged; the job re-runs at the full engine and Level 1 shows the actual unitary (Section 14.4)",
                    key="request-detuning",
                ),
            ],
            spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    ]
    last = store.last_request
    if last is not None and last.kind == "detuning" and last.gate_id == gate_id:
        if last.refusal:
            controls.append(
                ft.Text(f"refused: {last.refusal}", size=12, color=ft.Colors.ERROR, key="request-refusal")
            )
        else:
            controls.append(status_line(last.note))
    if record.job.waveform_overrides:
        controls.append(
            status_line(
                "this run plays a hand-set detuning: "
                + ", ".join(f"pair {k}: {v / 1e3:+.3g} kHz" for k, v in record.job.waveform_overrides.items())
            )
        )
    return ft.Column(controls, spacing=6)


@ft.component
def Level2Page(
    store: Store, session: Session, record: Record, pulse_param: str, index: ProvenanceIndex
) -> ft.Control:
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
    plain = store.learner.plan(2).plain_labels_first

    def on_select(k: int) -> None:
        page.navigate(f"/job/{key}/schedule/{k}")

    tone_tiles: list[ft.Control] = []
    for t in pv.tones:
        tone_tiles.append(
            stat_tile(t.detuning, index, plain=plain, label=f"tone {t.index}: detuning from the carrier")
        )
        tone_tiles.append(
            stat_tile(t.envelope_peak, index, plain=plain, label=f"tone {t.index}: peak Rabi frequency")
        )
    tone_rows: list[list[ft.Control | str]] = [
        [
            str(t.index),
            value_cell(t.detuning, index),
            value_cell(t.envelope_peak, index),
            value_cell(t.phase_start, index),
        ]
        for t in pv.tones
    ]
    sideband_rows: list[list[ft.Control | str]] = [
        [
            f"tone {t.index}",
            str(sb.mode),
            value_cell(sb.mode_frequency, index),
            value_cell(sb.detuning, index),
            sb.role,
        ]
        for t in pv.tones
        for sb in t.sidebands
    ]
    spectrum_card = card(
        "Tones against the modes",
        ft.Column(
            [
                _spectrum_strip(pv, max(strip_width, lanes_width * 0.9)),
                stat_row(tone_tiles[:6]),
                hint(store, 2, "tone_and_sideband"),
                details(
                    "level2.tones",
                    [
                        data_table(
                            ["tone", "detuning from the carrier", "peak Rabi frequency", "phase at start"],
                            tone_rows,
                        ),
                        data_table(
                            ["tone", "mode", "omega_m", "delta = mu - omega_m", "role"], sideband_rows
                        ),
                    ],
                    store=store,
                    session=session,
                    title="Tone and sideband tables",
                ),
            ],
            spacing=8,
        ),
        why=lambda e: session.select_concept(2, "tone_and_sideband"),
        key="spectrum",
    )
    pulse_tiles: list[ft.Control] = [
        stat_tile(pv.stark_shift, index, plain=plain),
    ]
    pulse_tiles.extend(stat_tile(c, index, plain=plain) for c in pv.crosstalk[:2])
    pulse_card = card(
        f"Pulse {sel}: {pv.pulse.gate_id} on ions {pv.ions}",
        ft.Column(
            [
                status_line(
                    f"{pv.pulse.kind}, {pv.pulse.t_start_s * 1e6:.2f} to {pv.pulse.t_end_s * 1e6:.2f} µs, beams {tuple(b.detail for b in pv.beams)}"
                ),
                stat_row(pulse_tiles),
                _envelope_chart(pv),
            ],
            spacing=8,
        ),
        why=lambda e: session.select_concept(2, "pulse"),
        actions=[
            ft.FilledButton(
                content=ft.Text("Zoom in: inside this pulse"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda e: page.navigate(f"/job/{key}/dynamics/{sel}/0"),
                key="zoom-pulse",
            )
        ],
        key="pulse",
    )
    right: list[ft.Control] = [pulse_card]
    gate_id = pv.pulse.gate_id.split("/")[0] if pv.pulse.gate_id else None
    if gate_id and any(g.gate_id == gate_id for g in record.schedule.gates):
        cl = closure(record, gate_id)
        seg_rows: list[list[ft.Control | str]] = [
            [str(k)] + [value_cell(s, index) for s in row] for k, row in enumerate(cl.segments)
        ]
        seg_cols = ["segment"] + ([""] * (len(cl.segments[0]) if cl.segments else 0))
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
        right.append(
            card(
                f"Loop closure of {gate_id}",
                ft.Column(
                    [
                        ft.Row([_closure_pill(m) for m in cl.modes], wrap=True, spacing=6, run_spacing=6),
                        stat_row(
                            [
                                stat_tile(cl.chi_total, index, plain=plain),
                                stat_tile(cl.duration, index, plain=plain),
                            ]
                            + (
                                [stat_tile(cl.beat_phase, index, plain=plain)]
                                if cl.beat_phase is not None
                                else []
                            )
                        ),
                        hint(store, 2, "loop_closure"),
                        RequestDetuningPanel(store, session, record, gate_id, index),
                        details(
                            "level2.closure",
                            [
                                data_table(
                                    [
                                        "mode",
                                        "class",
                                        "chi_m",
                                        "|alpha_m| at closure",
                                        "|alpha|^2 (2n+1)",
                                        "loop",
                                    ],
                                    mode_rows,
                                ),
                                status_line(
                                    "the table's values at calibration; a hand-set detuning shows its open loops on Level 3"
                                ),
                                data_table(seg_cols, seg_rows)
                                if seg_rows
                                else status_line("Fourier-parameterized waveform"),
                            ],
                            store=store,
                            session=session,
                            title="Closure and segment tables",
                        ),
                    ],
                    spacing=8,
                ),
                why=lambda e: session.select_concept(2, "loop_closure"),
                key="closure",
            )
        )
    return ft.Column(
        [
            level_header("The schedule", "What light hit which ion, when, and what did it talk to?"),
            spectrum_card,
            card(
                "Time axis",
                _lanes(record, sel, on_select, lanes_width),
                why=lambda e: session.select_concept(2, "pulse"),
                key="time-axis",
            ),
            ft.ResponsiveRow(
                [
                    ft.Column(right[:1], col={"xs": 12, "lg": 6}, spacing=10),
                    ft.Column(right[1:], col={"xs": 12, "lg": 6}, spacing=10),
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


__all__ = ["Level2Page", "RequestDetuningPanel"]
