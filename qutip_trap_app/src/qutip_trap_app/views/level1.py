"""Level 1, the circuit (DESIGN.md Sections 5 and 10): the compiled native timeline on ion lanes with the selected gate, the
register after that gate as Bloch arrows drawn in discs, the gate's stat tiles, and, for an entangling gate, the request of
Section 14.4 (XX(chi) requested, the actual unitary from tomography beside it). The target unitary, the Pauli expectations,
the phase register and the compile report are behind Details."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import flet as ft
import numpy as np

from qutip_trap_app import resim
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.requests import GateRequest, RequestError, RequestKind, request_angle, request_outcome
from qutip_trap_app.viewmodel.catalogue import CATALOGUE, Shown
from qutip_trap_app.viewmodel.circuit import (
    GateView,
    RegisterUnavailable,
    RegisterView,
    compile_report,
    phase_register,
    register_after,
    timeline,
)
from qutip_trap_app.views import drawing, theme
from qutip_trap_app.views.common import (
    HAIRLINE,
    MUTED,
    card,
    columns,
    content_widths,
    data_table,
    details,
    empty_state,
    hint,
    input_style,
    ions_text,
    kv_rows,
    level_header,
    shown,
    stat_row,
    stat_tile,
    status_line,
    tile_row,
    value_cell,
)
from qutip_trap_app.views.state import Session, Store

GATE_COLORS = {
    "gpi2": ft.Colors.PRIMARY_CONTAINER,
    "gpi": ft.Colors.TERTIARY_CONTAINER,
    "ms": ft.Colors.SECONDARY_CONTAINER,
    "zz": ft.Colors.SECONDARY_CONTAINER,
}

LANE_LABEL_WIDTH = 48.0


def _fmt_complex(z: complex) -> str:
    if abs(z.imag) < 1e-9:
        return f"{z.real:+.3f}"
    if abs(z.real) < 1e-9:
        return f"{z.imag:+.3f}i"
    return f"{z.real:+.3f}{z.imag:+.3f}i"


def _matrix_table(u: np.ndarray) -> ft.Control:
    d = u.shape[0]
    labels = [format(k, f"0{max(1, int(np.log2(d)))}b") for k in range(d)]
    return data_table(
        [""] + labels, [[labels[i]] + [_fmt_complex(complex(u[i, j])) for j in range(d)] for i in range(d)]
    )


# ---- lanes on a time axis (Levels 1 and 2) ---------------------------------------------------------------------------


def lane_label(text: str) -> ft.Control:
    return ft.Text(text, size=theme.SIZE_SMALL, color=MUTED, width=LANE_LABEL_WIDTH)


def lane(
    label: str, items: list[ft.Control], width: float, *, height: float = 32.0, line: bool = True
) -> ft.Control:
    """One lane: its label, a hairline along it, and the positioned items on it."""
    rule = [ft.Container(left=0, top=height / 2 - 1, width=width, height=1, bgcolor=HAIRLINE)] if line else []
    return ft.Row(
        [lane_label(label), ft.Stack(rule + items, width=width, height=height)],
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def lane_box(
    text: str,
    left: float,
    width: float,
    *,
    color: ft.ColorValue,
    selected: bool,
    tooltip: str,
    on_click: Callable[[], None],
    key: str | None,
) -> ft.Control:
    """A gate or a pulse on its lane: a box as long as it lasts, outlined in the primary colour when selected."""
    return ft.Container(
        content=ft.Text(text, size=theme.SIZE_CAPTION, no_wrap=True, overflow=ft.TextOverflow.CLIP),
        left=left,
        top=2,
        width=width,
        height=28,
        bgcolor=color,
        border=ft.Border.all(2 if selected else 1, ft.Colors.PRIMARY if selected else HAIRLINE),
        border_radius=ft.BorderRadius.all(6),
        padding=ft.Padding.symmetric(horizontal=6),
        alignment=ft.Alignment.CENTER_LEFT,
        tooltip=tooltip,
        on_click=lambda e: on_click(),
        ink=True,
        key=key,
    )


def time_axis_row(t0_s: float, t1_s: float, width: float) -> ft.Control:
    """The start and end of a time axis under a set of lanes, in microseconds."""
    return ft.Row(
        [
            ft.Text("", width=LANE_LABEL_WIDTH),
            ft.Text(f"{t0_s * 1e6:.0f} µs", size=theme.SIZE_MICRO, color=MUTED),
            ft.Container(expand=True),
            ft.Text(f"{t1_s * 1e6:.1f} µs", size=theme.SIZE_MICRO, color=MUTED),
        ],
        spacing=6,
        width=width + LANE_LABEL_WIDTH + 6,
    )


def _lanes(
    record: Record, gates: tuple[GateView, ...], selected: str, on_select: Any, width: float
) -> ft.Control:
    """Ion lanes with one box per gate piece, widths proportional to duration (a schedule at Level 1's resolution)."""
    t0 = record.schedule.t0_s
    t1 = max(record.schedule.pulses_end_s, t0 + 1e-9)
    scale = width / (t1 - t0)
    lanes: list[ft.Control] = []
    for ion in range(record.device_card.n_ions):
        boxes: list[ft.Control] = []
        for g in gates:
            if ion not in g.ions:
                continue
            w = max((g.t_end_s - g.t_start_s) * scale, 14.0)
            boxes.append(
                lane_box(
                    g.gate_id if w > 44 else "",
                    (g.t_start_s - t0) * scale,
                    w,
                    color=GATE_COLORS.get(str(g.name.value), ft.Colors.SURFACE_CONTAINER_HIGHEST),
                    selected=g.gate_id == selected,
                    tooltip=f"{g.gate_id}: {g.name.value} on ions {g.ions}, {g.t_start_s * 1e6:.1f} to {g.t_end_s * 1e6:.1f} µs",
                    on_click=functools.partial(on_select, g.gate_id),
                    key=f"gate:{g.gate_id}" if ion == g.ions[0] else None,
                )
            )
        lanes.append(lane(f"ion {ion}", boxes, width))
    return ft.Column(lanes + [time_axis_row(t0, t1, width)], spacing=2)


# ---- requests (Section 14.4) -----------------------------------------------------------------------------------------


@ft.component
def RequestPanel(
    store: Store,
    session: Session,
    gate_id: str,
    kind: RequestKind,
    make: Callable[[float], GateRequest],
    *,
    field_label: str,
    default: str,
    width: int,
    button: str,
    tooltip: str,
    parse_error: str,
) -> ft.Control:
    """A request at Level 1 or 2: a number, a button that makes the request a new job at the full engine, and the last
    request's note or its refusal with the reason."""
    ft.use_state(store)
    text, set_text = ft.use_state(default)

    def submit(_e: Any) -> None:
        try:
            value = float(text)
        except ValueError:
            store.error = parse_error
            return
        try:
            request = make(value)
        except RequestError as exc:
            store.error = str(exc)
            return
        session.submit_request(request)

    controls: list[ft.Control] = [
        ft.Row(
            [
                ft.TextField(
                    label=field_label,
                    value=text,
                    width=width,
                    on_change=lambda e: set_text(str(e.control.value)),
                    key=f"request-{kind}-value",
                    **input_style(),
                ),
                ft.FilledTonalButton(
                    content=ft.Text(button),
                    icon=ft.Icons.TUNE,
                    on_click=submit,
                    disabled=store.running_of("request_run", gate_id=gate_id) is not None,
                    tooltip=tooltip,
                    key=f"request-{kind}",
                ),
            ],
            spacing=theme.GAP,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    ]
    last = store.last_request
    if last is not None and last.kind == kind and last.gate_id == gate_id:
        controls.append(
            ft.Text(
                f"refused: {last.refusal}",
                size=theme.SIZE_SMALL,
                color=ft.Colors.ERROR,
                key="request-refusal",
            )
            if last.refusal
            else status_line(last.note)
        )
    return ft.Column(controls, spacing=6)


@ft.component
def RequestOutcomeView(
    store: Store, session: Session, record: Record, gate: GateView, index: ProvenanceIndex
) -> ft.Control:
    """The actual unitary of the played gate beside the requested one: from the step's process matrix when the record has it
    (a request run computes it at once), else a button to compute it."""
    ft.use_state(store)
    key = record.key()
    step = gate.step_index
    pm = record.process_matrix(resim.process_matrix_key(step, 0, 0))
    if pm is None:
        if record.replay is not None or not record.traces:
            return status_line("actual unitary: needs a full-simulation run (this record is derived)")
        running = store.running_of("tomography", key=key, step=step) is not None
        return ft.Row(
            [
                ft.OutlinedButton(
                    content=ft.Text("Actual unitary (tomography)"),
                    icon=ft.Icons.GRID_4X4,
                    on_click=lambda e: session.submit_tomography(key, step, 0, 0),
                    disabled=running,
                    tooltip="process tomography of this gate's pulses from the recorded motional state (Section 5.4)",
                    key="compute-actual",
                ),
                status_line("running" if running else ""),
            ],
            spacing=theme.GAP,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    try:
        out = request_outcome(record, gate.gate_id, pm)
    except RequestError:
        return ft.Container()
    lines: list[ft.Control] = [
        stat_row(
            [
                stat_tile(
                    Shown(
                        "entangling_angle", out.requested_chi_rad, "chi of the requested unitary (theta/2)"
                    ),
                    index,
                    label="requested chi",
                ),
                stat_tile(
                    Shown(
                        "entangling_angle",
                        out.fitted_chi_rad,
                        f"the MS angle closest to the measured channel (fidelity {out.fitted_fidelity:.5f})",
                    ),
                    index,
                    label="actual chi (tomography)",
                ),
                stat_tile(
                    Shown(
                        "channel_infidelity",
                        out.infidelity_to_requested,
                        "average gate infidelity of the played gate against the requested unitary",
                    ),
                    index,
                    label="actual vs requested",
                ),
            ]
        )
    ]
    if out.hand_set_detuning_hz is not None:
        lines.append(
            status_line(f"beat-note detuning set by hand: {out.hand_set_detuning_hz / 1e3:+.3g} kHz")
        )
    return ft.Column(lines, spacing=6, key="request-outcome")


# ---- the page --------------------------------------------------------------------------------------------------------


def _register_card(
    store: Store,
    session: Session,
    index: ProvenanceIndex,
    selected: GateView,
    reg: RegisterView,
    plain: bool,
    why: Any,
) -> ft.Control:
    """The register after the selected gate: Bloch discs with the ideal arrow dashed, the populations, purity and fidelity
    tiles, the Pauli expectations behind Details; how the register was obtained on the info button."""
    discs = ft.Row(
        [
            drawing.bloch_disc(ion, vec, target=reg.target_bloch.get(ion))
            for ion, vec in sorted(reg.bloch.items())
        ],
        wrap=True,
        spacing=16,
        run_spacing=12,
    )
    pops = sorted(reg.populations.items())
    chart = drawing.bar_chart(
        [(k, [drawing.Bar(float(s.value or 0.0), f"{k}: {float(s.value or 0.0):.4f}")]) for k, s in pops],
        bar_width=14,
        height=150,
        width=min(80 + 40 * len(pops), 520),
        max_y=1.0,
        y_ticks=(0.0, 0.5, 1.0),
        interactive=False,
    )
    return card(
        f"Register after {selected.gate_id}",
        ft.Column(
            [
                ft.Row(
                    [discs, chart],
                    wrap=True,
                    spacing=24,
                    run_spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                tile_row((reg.purity, reg.fidelity), index, plain=plain),
                hint(store, 1, "bloch_vector"),
                details(
                    store,
                    session,
                    1,
                    "level1.register",
                    [
                        status_line(reg.weights_note),
                        status_line("dashed arrow: the ideal state's; solid: the recorded reduced state's"),
                        data_table(
                            ["operator", "<P>"],
                            [[k, value_cell(v, index)] for k, v in sorted(reg.pauli.items())],
                            numeric=[False, True],
                        ),
                    ],
                    title="Pauli expectations and details",
                ),
            ],
            spacing=12,
        ),
        why=why,
        info=reg.weights_note,
        key="register",
    )


@ft.component
def Level1Page(
    store: Store, session: Session, record: Record, gate_param: str, index: ProvenanceIndex
) -> ft.Control:
    ft.use_state(store)
    gates = timeline(record)
    if not gates:
        return status_line("this circuit compiled to no gate pieces")
    selected = next((g for g in gates if g.gate_id == gate_param), gates[0])
    page = ft.context.page
    key = record.key()
    plain = store.learner.plan(1).plain_labels_first

    def on_select(gid: str) -> None:
        page.navigate(f"/job/{key}/circuit/{gid}")

    def why_register(_e: Any) -> None:
        session.select_concept(
            1, "entanglement_by_ms" if selected.name.value in ("ms", "zz") else "bloch_vector"
        )

    lanes_width, _strip = content_widths(store, page, 1)
    try:
        register_card = _register_card(
            store, session, index, selected, register_after(record, selected.index), plain, why_register
        )
    except (
        RegisterUnavailable
    ) as exc:  # a record with no trace, replay register or step channel: say so, keep the rest
        register_card = card(
            f"Register after {selected.gate_id}",
            empty_state(
                "Register not recorded",
                f"{exc}; the timeline, the gate and the phase register are still here",
                icon=ft.Icons.BLUR_CIRCULAR,
                key="register-unavailable",
            ),
            why=why_register,
            key="register",
        )
    tiles: list[ft.Control] = [stat_tile(selected.duration, index, plain=plain)]
    tiles += [
        stat_tile(s, index, plain=plain, label=f"{CATALOGUE[s.quantity].label}, the table's calibrated value")
        for s in selected.calibrated
    ]
    if selected.error_estimate is not None:
        tiles.append(stat_tile(selected.error_estimate, index, plain=plain, status="estimate"))
    if selected.channel is not None:
        tiles.append(
            stat_tile(
                Shown(
                    "channel_infidelity",
                    selected.channel.average_gate_infidelity,
                    "the extracted channel of this gate kind",
                ),
                index,
                label="channel infidelity",
                status="calibrated",
            )
        )
    tiles += [
        stat_tile(s, index, plain=plain, label=f"{CATALOGUE[s.quantity].label}, {s.detail}")
        for s in selected.stark_frame
    ]
    step = record.step(selected.step_index)
    first_pulse = step.pulse_indices[0] if step.pulse_indices else 0
    gate_body: list[ft.Control] = [
        ft.Row(
            [
                shown(selected.name, index, label=False, plain=plain, size=16),
                ft.Text(
                    f"on {ions_text(selected.ions)}, phase"
                    + ("s " if len(selected.params_rad) != 1 else " ")
                    + ", ".join(f"{p:.4f}" for p in selected.params_rad)
                    + " rad",
                    size=theme.SIZE_SMALL,
                    color=MUTED,
                ),
            ],
            spacing=theme.GAP,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        stat_row(tiles),
    ]
    if selected.name.value == "ms" and len(selected.ions) == 2:
        gate_body.append(RequestOutcomeView(store, session, record, selected, index))
        gate_body.append(
            RequestPanel(
                store,
                session,
                selected.gate_id,
                "angle",
                lambda chi: request_angle(record, selected.gate_id, chi),
                field_label="chi (rad)",
                default="0.3",
                width=110,
                button="Request XX(chi)",
                tooltip="a request, not an edit: the job is rebuilt with this angle and run at the full engine (Section 14.4)",
                parse_error="the requested angle must be a number in radians",
            )
        )
    gate_body.append(
        details(
            store,
            session,
            1,
            "level1.gate",
            [
                ft.Row(
                    [
                        ft.Text("target unitary", size=theme.SIZE_SMALL, weight=ft.FontWeight.W_600),
                        shown(selected.unitary, index, label=False),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                _matrix_table(selected.matrix),
            ]
            + (
                [ft.Column([status_line(r) for r in record.job.requests], spacing=2)]
                if record.job.requests
                else []
            ),
            title="Target unitary and details",
        )
    )
    gate_card = card(
        f"Gate {selected.gate_id}",
        ft.Column(gate_body, spacing=12),
        why=lambda e: session.select_concept(1, "native_gate"),
        actions=[
            ft.FilledButton(
                content=ft.Text("Zoom in: the pulses"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda e: page.navigate(f"/job/{key}/schedule/{first_pulse}"),
                key="zoom-gate",
            )
        ],
        key="gate",
    )
    frame = phase_register(record)
    frame_rows = [
        (f"ion {q} final frame", shown(v, index, label=False, size=theme.SIZE_SMALL))
        for q, v in frame.final_frame.items()
    ] + [
        (f"{gid} ion {q}", shown(v, index, label=False, size=theme.SIZE_SMALL))
        for gid, q, v in frame.stark_increments
    ]
    extras = card(
        "Phase register and compile report",
        details(
            store,
            session,
            1,
            "level1.frame",
            [
                kv_rows(frame_rows),
                status_line(frame.rule),
                ft.Row(
                    [shown(r, index, size=theme.SIZE_SMALL) for r in compile_report(record)],
                    wrap=True,
                    spacing=12,
                ),
            ],
            title="Show",
        ),
        why=lambda e: session.select_concept(1, "virtual_z"),
        key="frame",
    )
    return ft.Column(
        [
            level_header(
                "The circuit", "What did each gate do to the qubits?", note=f"{len(gates)} native gate pieces"
            ),
            card(
                "Timeline",
                ft.Column(
                    [
                        _lanes(record, gates, selected.gate_id, on_select, lanes_width),
                        hint(store, 1, "native_gate"),
                    ],
                    spacing=theme.GAP,
                ),
                why=lambda e: session.select_concept(1, "compile"),
                key="timeline",
            ),
            columns(store, page, 1, [register_card], [gate_card, extras], split=(7, 5)),
        ],
        spacing=16,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )
