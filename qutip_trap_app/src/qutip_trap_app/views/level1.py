"""Level 1, the circuit (PLAN.md Section 14.2 row 1; DESIGN.md Sections 5 and 10): the compiled native timeline on ion
lanes with the selected gate, the register after that gate as Bloch arrows drawn in discs, the gate's stat tiles, and, for an
entangling gate, the request control of Section 14.4 (XX(chi) requested, the pulse solver's rescale, the actual unitary from
tomography beside the requested one). The target unitary, the Pauli expectations, the phase register and the compile report
are behind Details."""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc
import numpy as np

from qutip_trap_app import resim
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.requests import RequestError, request_angle, request_outcome
from qutip_trap_app.viewmodel.catalogue import CATALOGUE, Shown
from qutip_trap_app.viewmodel.circuit import (
    GateView,
    bloch_vectors,
    compile_report,
    phase_register,
    register_after,
    timeline,
)
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import (
    card,
    content_widths,
    data_table,
    details,
    hint,
    kv_rows,
    level_header,
    shown,
    stat_row,
    stat_tile,
    status_line,
    value_cell,
)
from qutip_trap_app.views.state import Session, Store

GATE_COLORS = {
    "gpi2": ft.Colors.PRIMARY_CONTAINER,
    "gpi": ft.Colors.TERTIARY_CONTAINER,
    "ms": ft.Colors.SECONDARY_CONTAINER,
    "zz": ft.Colors.SECONDARY_CONTAINER,
}


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


def _lanes(
    record: Record, gates: tuple[GateView, ...], selected: str, on_select: Any, width: float = 760.0
) -> ft.Control:
    """Ion lanes with one box per gate piece, widths proportional to duration (a schedule at Level 1's resolution)."""
    t0 = record.schedule.t0_s
    t1 = max(record.schedule.pulses_end_s, t0 + 1e-9)
    scale = width / (t1 - t0)
    n_ions = record.device_card.n_ions
    lanes: list[ft.Control] = []
    for ion in range(n_ions):
        boxes: list[ft.Control] = []
        for g in gates:
            if ion not in g.ions:
                continue
            left = (g.t_start_s - t0) * scale
            w = max((g.t_end_s - g.t_start_s) * scale, 14.0)
            is_sel = g.gate_id == selected
            boxes.append(
                ft.Container(
                    content=ft.Text(
                        g.gate_id if w > 44 else "", size=10, no_wrap=True, overflow=ft.TextOverflow.CLIP
                    ),
                    left=left,
                    top=2,
                    width=w,
                    height=28,
                    bgcolor=GATE_COLORS.get(str(g.name.value), ft.Colors.SURFACE_CONTAINER_HIGHEST),
                    border=ft.Border.all(
                        2 if is_sel else 1, ft.Colors.PRIMARY if is_sel else ft.Colors.OUTLINE_VARIANT
                    ),
                    border_radius=ft.BorderRadius.all(4),
                    padding=ft.Padding.symmetric(horizontal=4),
                    alignment=ft.Alignment.CENTER_LEFT,
                    tooltip=f"{g.gate_id}: {g.name.value} on ions {g.ions}, {g.t_start_s * 1e6:.1f} to {g.t_end_s * 1e6:.1f} µs",
                    on_click=lambda e, gid=g.gate_id: on_select(gid),
                    ink=True,
                    key=f"gate:{g.gate_id}" if ion == g.ions[0] else None,
                )
            )
        lanes.append(
            ft.Row(
                [ft.Text(f"ion {ion}", size=12, width=44), ft.Stack(boxes, width=width, height=32)],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    axis = ft.Row(
        [
            ft.Text("", width=44),
            ft.Text(f"{t0 * 1e6:.0f} µs", size=10),
            ft.Container(expand=True),
            ft.Text(f"{t1 * 1e6:.1f} µs", size=10),
        ],
        spacing=6,
        width=width + 50,
    )
    return ft.Column(lanes + [axis], spacing=2)


def _populations_chart(pops: dict[str, Any]) -> ft.Control:
    keys = sorted(pops)
    groups = [
        fc.BarChartGroup(
            x=k,
            rods=[
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=float(pops[key].value or 0.0),
                    width=18,
                    color=ft.Colors.PRIMARY,
                    tooltip=f"{key}: {float(pops[key].value or 0.0):.4f}",
                )
            ],
        )
        for k, key in enumerate(keys)
    ]
    chart: ft.Control = fc.BarChart(
        groups=groups,
        bottom_axis=fc.ChartAxis(
            labels=[fc.ChartAxisLabel(value=k, label=ft.Text(key, size=11)) for k, key in enumerate(keys)],
            label_size=24,
        ),
        left_axis=fc.ChartAxis(
            labels=[fc.ChartAxisLabel(value=v, label=ft.Text(f"{v:.1f}", size=10)) for v in (0.0, 0.5, 1.0)],
            label_size=30,
        ),
        max_y=1.0,
        min_y=0.0,
        height=150,
        width=min(80 + 40 * len(keys), 520),
        interactive=False,
    )
    return chart


@ft.component
def RequestAnglePanel(
    store: Store, session: Session, record: Record, gate: GateView, index: ProvenanceIndex
) -> ft.Control:
    """Section 14.4 at Level 1: request XX(chi) of this MS gate. The request is a new job at the full engine; a refusal shows
    its reason here."""
    ft.use_state(store)
    text, set_text = ft.use_state("0.3")
    running = store.running_of("request_run", gate_id=gate.gate_id) is not None

    def submit(_e: Any) -> None:
        try:
            chi = float(text)
        except ValueError:
            store.error = "the requested angle must be a number in radians"
            return
        try:
            req = request_angle(record, gate.gate_id, chi)
        except RequestError as exc:
            store.error = str(exc)
            return
        session.submit_request(req)

    controls: list[ft.Control] = [
        ft.Row(
            [
                ft.TextField(
                    label="chi (rad)",
                    value=text,
                    width=110,
                    dense=True,
                    text_size=13,
                    on_change=lambda e: set_text(str(e.control.value)),
                    key="request-angle-value",
                ),
                ft.FilledTonalButton(
                    content=ft.Text("Request XX(chi)"),
                    icon=ft.Icons.TUNE,
                    on_click=submit,
                    disabled=running,
                    tooltip="a request, not an edit: the job is rebuilt with this angle and run at the full engine (Section 14.4)",
                    key="request-angle",
                ),
            ],
            spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    ]
    last = store.last_request
    if last is not None and last.kind == "angle" and last.gate_id == gate.gate_id:
        if last.refusal:
            controls.append(
                ft.Text(f"refused: {last.refusal}", size=12, color=ft.Colors.ERROR, key="request-refusal")
            )
        else:
            controls.append(status_line(last.note))
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
        running = store.running_of("tomography", key=key, step=step) is not None
        if record.replay is not None or not record.traces:
            return status_line("actual unitary: needs a full-simulation run (this record is derived)")
        return ft.Row(
            [
                ft.OutlinedButton(
                    content=ft.Text("Actual unitary (tomography)"),
                    icon=ft.Icons.GRID_4X4,
                    on_click=lambda e: session.submit_tomography(key, step, 0, store.branch),
                    disabled=running,
                    tooltip="process tomography of this gate's pulses from the recorded motional state (Section 5.4)",
                    key="compute-actual",
                ),
                status_line("running" if running else ""),
            ],
            spacing=8,
        )
    try:
        out = request_outcome(record, gate.gate_id, pm)
    except RequestError:
        return ft.Container()
    tiles = [
        stat_tile(
            Shown("entangling_angle", out.requested_chi_rad, "chi of the requested unitary (theta/2)"),
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
    lines: list[ft.Control] = [stat_row(tiles)]
    if out.hand_set_detuning_hz is not None:
        lines.append(
            status_line(f"beat-note detuning set by hand: {out.hand_set_detuning_hz / 1e3:+.3g} kHz")
        )
    return ft.Column(lines, spacing=6, key="request-outcome")


@ft.component
def Level1Page(
    store: Store, session: Session, record: Record, gate_param: str, index: ProvenanceIndex
) -> ft.Control:
    ft.use_state(store)
    gates = timeline(record)
    if not gates:
        return ft.Text("this circuit compiled to no gate pieces", size=13)
    selected = next((g for g in gates if g.gate_id == gate_param), gates[0])
    page = ft.context.page
    key = record.key()
    plain = store.learner.plan(1).plain_labels_first

    def on_select(gid: str) -> None:
        page.navigate(f"/job/{key}/circuit/{gid}")

    reg = register_after(record, selected.index)
    lanes_width, _strip = content_widths(store, page, 1)
    frame = phase_register(record)
    residuals = compile_report(record)
    n = record.n_qubits
    ket = reg.target_ket
    target_bloch = bloch_vectors(np.outer(ket, ket.conj()), n) if ket is not None and ket.size == 2**n else {}
    discs = ft.Row(
        [
            drawing.bloch_disc(ion, vec, target=target_bloch.get(ion))
            for ion, vec in sorted(reg.bloch.items())
        ],
        wrap=True,
        spacing=12,
    )
    register_card = card(
        f"Register after {selected.gate_id}",
        ft.Column(
            [
                ft.Row(
                    [discs, _populations_chart(reg.populations)],
                    wrap=True,
                    spacing=20,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                stat_row(
                    [stat_tile(reg.purity, index, plain=plain), stat_tile(reg.fidelity, index, plain=plain)]
                ),
                hint(store, 1, "bloch_vector"),
                details(
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
                    store=store,
                    session=session,
                    title="Pauli expectations and details",
                ),
            ],
            spacing=8,
        ),
        why=lambda e: session.select_concept(
            1, "entanglement_by_ms" if selected.name.value in ("ms", "zz") else "bloch_vector"
        ),
        key="register",
    )
    tiles: list[ft.Control] = [stat_tile(selected.duration, index, plain=plain)]
    for s in selected.calibrated:
        tiles.append(
            stat_tile(
                s, index, plain=plain, label=f"{CATALOGUE[s.quantity].label}, the table's calibrated value"
            )
        )
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
    for s in selected.stark_frame:
        tiles.append(stat_tile(s, index, plain=plain, label=f"{CATALOGUE[s.quantity].label}, {s.detail}"))
    first_pulse = (
        record.step(selected.step_index).pulse_indices[0]
        if record.step(selected.step_index).pulse_indices
        else 0
    )
    zoom_btn = ft.FilledButton(
        content=ft.Text("Zoom in: the pulses"),
        icon=ft.Icons.ZOOM_IN,
        on_click=lambda e: page.navigate(f"/job/{key}/schedule/{first_pulse}"),
        key="zoom-gate",
    )
    unitary = record.schedule.targets[
        [t.gate_id for t in record.schedule.targets].index(selected.gate_id)
    ].unitary
    gate_body: list[ft.Control] = [
        ft.Row(
            [
                shown(selected.name, index, label=False, plain=plain, size=16),
                ft.Text(
                    f"on ions {selected.ions}, phases {tuple(round(p, 4) for p in selected.params_rad)} rad",
                    size=12,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            spacing=8,
            wrap=True,
        ),
        stat_row(tiles),
    ]
    if selected.name.value == "ms" and len(selected.ions) == 2:
        gate_body.append(RequestOutcomeView(store, session, record, selected, index))
        gate_body.append(RequestAnglePanel(store, session, record, selected, index))
    gate_body.append(
        details(
            "level1.gate",
            [
                ft.Row(
                    [
                        ft.Text("target unitary", size=12, weight=ft.FontWeight.W_600),
                        shown(selected.unitary, index, label=False),
                    ],
                    spacing=6,
                ),
                _matrix_table(unitary),
            ]
            + (
                [ft.Column([status_line(r) for r in record.job.requests], spacing=2)]
                if record.job.requests
                else []
            ),
            store=store,
            session=session,
            title="Target unitary and details",
        )
    )
    gate_card = card(
        f"Gate {selected.gate_id}",
        ft.Column(gate_body, spacing=8),
        why=lambda e: session.select_concept(1, "native_gate"),
        actions=[zoom_btn],
        key="gate",
    )
    frame_rows = [
        (f"ion {q} final frame", shown(v, index, label=False, size=12)) for q, v in frame.final_frame.items()
    ]
    frame_rows += [
        (f"{gid} ion {q}", shown(v, index, label=False, size=12)) for gid, q, v in frame.stark_increments
    ]
    extras = card(
        "Phase register and compile report",
        details(
            "level1.frame",
            [
                kv_rows(frame_rows),
                status_line(frame.rule),
                ft.Row([shown(r, index, size=12) for r in residuals], wrap=True, spacing=12),
            ],
            store=store,
            session=session,
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
                    spacing=6,
                ),
                why=lambda e: session.select_concept(1, "compile"),
                key="timeline",
            ),
            ft.ResponsiveRow(
                [
                    ft.Column([register_card], col={"xs": 12, "lg": 7}, spacing=10),
                    ft.Column([gate_card, extras], col={"xs": 12, "lg": 5}, spacing=10),
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


__all__ = ["Level1Page", "RequestAnglePanel", "RequestOutcomeView"]
