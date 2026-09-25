"""Level 2, the schedule (DESIGN.md Sections 5 and 10): the selected pulse's tones against the mode spectrum as the focal
picture, the pulses on ion lanes, the pulse's stat tiles and envelope, the loop-closure pills per mode, and, for an
entangling pulse, the request of Section 14.4 that sets the beat-note detuning by hand (played as written). The tone,
sideband and segment tables are behind Details."""

from __future__ import annotations

import functools
from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.requests import request_detuning
from qutip_trap_app.viewmodel.schedule import (
    ModeClosure,
    PulseView,
    closure,
    pulse_view,
    time_axis,
)
from qutip_trap_app.views import drawing, theme
from qutip_trap_app.views.common import (
    MUTED,
    card,
    columns,
    content_widths,
    data_table,
    details,
    hint,
    ions_text,
    level_header,
    pill,
    stat_row,
    stat_tile,
    status_line,
    tile_row,
    value_cell,
)
from qutip_trap_app.views.level1 import RequestPanel, lane, lane_box, time_axis_row
from qutip_trap_app.views.state import Session, Store


def _lanes(record: Record, selected: int, on_select: Any, width: float) -> ft.Control:
    axis = time_axis(record)
    t0, t1 = axis.t0_s, max(axis.duration_s, axis.t0_s + 1e-9)
    scale = width / (t1 - t0)
    first_ion = min((ln.ion for ln in axis.lanes), default=0)
    lanes: list[ft.Control] = [
        lane(
            f"ion {ln.ion}",
            [
                lane_box(
                    span.gate_id or "",
                    (span.t_start_s - t0) * scale,
                    max((span.t_end_s - span.t_start_s) * scale, 12.0),
                    color=ft.Colors.PRIMARY_CONTAINER
                    if span.kind == "raman"
                    else ft.Colors.TERTIARY_CONTAINER,
                    selected=span.pulse_index == selected,
                    tooltip=f"pulse {span.pulse_index} ({span.kind}) {span.t_start_s * 1e6:.2f} to {span.t_end_s * 1e6:.2f} µs; gate {span.gate_id}",
                    on_click=functools.partial(on_select, span.pulse_index),
                    key=f"pulse:{span.pulse_index}" if ln.ion == first_ion else None,
                )
                for span in ln.spans
            ],
            width,
        )
        for ln in axis.lanes
    ]
    marks: list[ft.Control] = []
    if axis.measurement is not None:
        t_m = float(axis.measurement.value or 0.0)
        marks.append(
            ft.Container(
                left=(t_m - t0) * scale,
                top=0,
                width=max((t1 - t_m) * scale, 4.0),
                height=14,
                bgcolor=ft.Colors.with_opacity(0.35, ft.Colors.TERTIARY),
                border_radius=ft.BorderRadius.all(3),
                tooltip="detection window (measure event)",
            )
        )
    lanes.append(lane("readout", marks, width, height=16, line=False))
    lanes.append(time_axis_row(t0, t1, width))
    return ft.Column(lanes, spacing=2)


def _spectrum_strip(pv: PulseView, width: float) -> ft.Control:
    """Mode frequencies as ticks and the pulse's tone detunings (their absolute values) drawn against them: the focal
    picture."""
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
                    content=ft.Text(f"{f / 1e6:.3f}", size=theme.SIZE_MICRO, color=MUTED),
                    left=x - 22,
                    top=48,
                    width=44,
                    alignment=ft.Alignment.CENTER,
                )
            )
            label_right = x + 22.0
    for tone, mu in zip(pv.tones, tones):
        # the 14 px dot stays inside the strip (a carrier tone sits at the far left edge)
        x = min(max((mu - lo) * scale, 7.0), width - 7.0)
        sign = "+" if float(tone.detuning.value or 0.0) >= 0 else "-"
        controls.append(
            ft.Container(
                left=x - 7,
                top=2,
                width=14,
                height=14,
                border_radius=ft.BorderRadius.all(7),
                bgcolor=theme.tone_color(tone.role),
                border=ft.Border.all(2, ft.Colors.SURFACE_CONTAINER_LOWEST),
                tooltip=f"tone {tone.index}: mu = {sign}{mu / 1e6:.4f} MHz from the carrier ({tone.role} sideband)",
            )
        )

    def dot(role: str) -> ft.Control:
        return ft.Container(width=12, height=12, bgcolor=theme.tone_color(role), border_radius=6)

    legend = ft.Row(
        [
            dot("red"),
            ft.Text("red tone", size=theme.SIZE_CAPTION, color=MUTED),
            dot("blue"),
            ft.Text("blue tone", size=theme.SIZE_CAPTION, color=MUTED),
            ft.Container(width=2, height=14, bgcolor=ft.Colors.ON_SURFACE),
            ft.Text("mode (|detuning| in MHz)", size=theme.SIZE_CAPTION, color=MUTED),
        ],
        spacing=6,
        wrap=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    return ft.Column([ft.Stack(controls, width=width, height=66), legend], spacing=6)


def _closure_pill(m: ModeClosure) -> ft.Control:
    return pill(
        f"mode {m.mode}: {'closed' if m.closed else 'open'}",
        "pass" if m.closed else "fail",
        icon=ft.Icons.CHECK_CIRCLE_OUTLINE if m.closed else ft.Icons.RADIO_BUTTON_UNCHECKED,
        tooltip=f"|alpha_{m.mode}|^2 (2 nbar + 1) = {m.residual:.2e} against 1e-6 (Section 5.2); mode class {m.mode_class}",
    )


def _closure_card(
    store: Store, session: Session, record: Record, gate_id: str, index: ProvenanceIndex
) -> ft.Control:
    cl = closure(record, gate_id)
    plain = store.learner.plan(2).plain_labels_first
    seg_rows: list[list[ft.Control | str]] = [
        [str(k)] + [value_cell(s, index) for s in row] for k, row in enumerate(cl.segments)
    ]
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
    request: list[ft.Control] = [
        RequestPanel(
            store,
            session,
            gate_id,
            "detuning",
            lambda khz: request_detuning(record, gate_id, khz * 1e3),
            field_label="offset (kHz)",
            default="5",
            width=120,
            button="Set the detuning by hand",
            tooltip="applied as written: blue legs up, red legs down, amplitude unchanged; the job re-runs at the full engine and Level 1 shows the actual unitary (Section 14.4)",
            parse_error="the offset must be a number in kHz",
        )
    ]
    if record.job.waveform_overrides:
        request.append(
            status_line(
                "this run plays a hand-set detuning: "
                + ", ".join(f"pair {k}: {v / 1e3:+.3g} kHz" for k, v in record.job.waveform_overrides.items())
            )
        )
    return card(
        f"Loop closure of {gate_id}",
        ft.Column(
            [
                ft.Row([_closure_pill(m) for m in cl.modes], wrap=True, spacing=6, run_spacing=6),
                tile_row((cl.chi_total, cl.duration, cl.beat_phase), index, plain=plain),
                hint(store, 2, "loop_closure"),
                *request,
                details(
                    store,
                    session,
                    2,
                    "level2.closure",
                    [
                        data_table(
                            ["mode", "class", "chi_m", "|alpha_m| at closure", "|alpha|^2 (2n+1)", "loop"],
                            mode_rows,
                        ),
                        status_line("values at calibration; a hand-set detuning opens the loops on Level 3"),
                        data_table(["segment"] + [""] * (len(cl.segments[0]) if cl.segments else 0), seg_rows)
                        if seg_rows
                        else status_line("Fourier-parameterized waveform"),
                    ],
                    title="Closure and segment tables",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(2, "loop_closure"),
        key="closure",
    )


@ft.component
def Level2Page(
    store: Store, session: Session, record: Record, pulse_param: str, index: ProvenanceIndex
) -> ft.Control:
    ft.use_state(store)
    pulses = record.schedule.pulses
    if not pulses:
        return status_line("this schedule has no pulses")
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
                    store,
                    session,
                    2,
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
                    title="Tone and sideband tables",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(2, "tone_and_sideband"),
        key="spectrum",
    )
    envelope = drawing.line_chart(
        [(f"tone {t.index}", t.times_s * 1e6, t.envelope_samples / 1e3) for t in pv.tones],
        x_title="t (µs)",
        y_title="Omega/2pi (kHz)",
        height=130,
    )
    pulse_card = card(
        f"Pulse {sel}: {pv.pulse.gate_id} on {ions_text(pv.ions)}",
        ft.Column(
            [
                status_line(
                    " · ".join(
                        [pv.pulse.kind, f"{pv.pulse.t_start_s * 1e6:.2f} to {pv.pulse.t_end_s * 1e6:.2f} µs"]
                        + [str(b.detail) for b in pv.beams]
                    )
                ),
                stat_row(
                    [stat_tile(pv.stark_shift, index, plain=plain)]
                    + [stat_tile(c, index, plain=plain) for c in pv.crosstalk[:2]]
                ),
                envelope,
            ],
            spacing=12,
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
    gate_id = pv.pulse.gate_id.split("/")[0] if pv.pulse.gate_id else None
    right: list[ft.Control] = []
    if gate_id and any(g.gate_id == gate_id for g in record.schedule.gates):
        right.append(_closure_card(store, session, record, gate_id, index))
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
            columns(store, page, 2, [pulse_card], right),
        ],
        spacing=16,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )
