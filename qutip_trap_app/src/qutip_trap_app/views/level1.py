"""Level 1, the circuit: the compiled native gates on ion lanes, the register after the selected gate as Bloch discs and
populations, and the gate with its ideal unitary."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft
import flet_charts as fc
import numpy as np

from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.circuit import (
    GateView,
    RegisterUnavailable,
    RegisterView,
    register_after,
    timeline,
)
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import (
    GAP,
    HAIRLINE,
    MUTED,
    card,
    columns,
    content_width,
    data_table,
    details,
    empty_state,
    ions_text,
    level_header,
    screen,
    stat_row,
    stat_tile,
    status_line,
)
from qutip_trap_app.views.state import Session, Store

GATE_COLORS = {
    "gpi2": ft.Colors.PRIMARY_CONTAINER,
    "gpi": ft.Colors.TERTIARY_CONTAINER,
    "ms": ft.Colors.SECONDARY_CONTAINER,
    "zz": ft.Colors.SECONDARY_CONTAINER,
}
LANE_LABEL_WIDTH = 48.0


def lane_label(text: str) -> ft.Control:
    return ft.Text(text, size=12, color=MUTED, width=LANE_LABEL_WIDTH)


def time_axis_row(t0_s: float, t1_s: float, width: float) -> ft.Control:
    """The start and end of a time axis under a set of lanes, in microseconds."""
    return ft.Row(
        [
            ft.Text("", width=LANE_LABEL_WIDTH),
            ft.Text(f"{t0_s * 1e6:.0f} µs", size=10, color=MUTED),
            ft.Container(expand=True),
            ft.Text(f"{t1_s * 1e6:.1f} µs", size=10, color=MUTED),
        ],
        spacing=6,
        width=width + LANE_LABEL_WIDTH + 6,
    )


def lane_width(store: Store) -> float:
    return content_width(store) - 2 * 16 - LANE_LABEL_WIDTH - 16


def _lanes(
    record: Record, gates: tuple[GateView, ...], selected: str, on_select: Callable[[str], None], width: float
) -> ft.Control:
    """Ion lanes with one box per gate piece, widths proportional to duration."""
    t0 = record.schedule.t0_s
    t1 = max(record.schedule.pulses_end_s, t0 + 1e-9)
    scale = width / (t1 - t0)
    lanes: list[ft.Control] = []
    for ion in range(record.n_ions):
        boxes: list[ft.Control] = [ft.Container(left=0, top=15, width=width, height=1, bgcolor=HAIRLINE)]
        for g in gates:
            if ion not in g.ions:
                continue
            w = max((g.t_end_s - g.t_start_s) * scale, 14.0)
            chosen = g.gate_id == selected
            boxes.append(
                ft.Container(
                    content=ft.Text(
                        g.gate_id if w > 44 else "", size=11, no_wrap=True, overflow=ft.TextOverflow.CLIP
                    ),
                    left=(g.t_start_s - t0) * scale,
                    top=2,
                    width=w,
                    height=28,
                    bgcolor=GATE_COLORS.get(g.name, ft.Colors.SURFACE_CONTAINER_HIGHEST),
                    border=ft.Border.all(2 if chosen else 1, ft.Colors.PRIMARY if chosen else HAIRLINE),
                    border_radius=ft.BorderRadius.all(6),
                    padding=ft.Padding.symmetric(horizontal=6),
                    alignment=ft.Alignment.CENTER_LEFT,
                    tooltip=f"{g.gate_id}: {g.name} on {ions_text(g.ions)}, "
                    f"{g.t_start_s * 1e6:.1f} to {g.t_end_s * 1e6:.1f} µs",
                    on_click=lambda _e, gid=g.gate_id: on_select(gid),
                    ink=True,
                )
            )
        lanes.append(
            ft.Row(
                [lane_label(f"ion {ion}"), ft.Stack(boxes, width=width, height=32)],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    return ft.Column([*lanes, time_axis_row(t0, t1, width)], spacing=2)


def _populations_chart(pops: dict[str, float]) -> ft.Control:
    keys = sorted(pops)
    chart: ft.Control = fc.BarChart(
        groups=[
            fc.BarChartGroup(
                x=k,
                rods=[
                    fc.BarChartRod(
                        from_y=0.0,
                        to_y=pops[key],
                        width=14,
                        color=ft.Colors.PRIMARY,
                        tooltip=f"{key}: {pops[key]:.4f}",
                        border_radius=ft.BorderRadius.only(top_left=3, top_right=3),
                    )
                ],
            )
            for k, key in enumerate(keys)
        ],
        bottom_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(value=k, label=ft.Text(key, size=11, color=MUTED))
                for k, key in enumerate(keys)
            ],
            label_size=24,
        ),
        left_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(value=v, label=ft.Text(f"{v:.1f}", size=10, color=MUTED))
                for v in (0.0, 0.5, 1.0)
            ],
            label_size=30,
        ),
        horizontal_grid_lines=fc.ChartGridLines(interval=0.5, color=ft.Colors.OUTLINE_VARIANT, width=1),
        max_y=1.0,
        min_y=0.0,
        height=150,
        width=min(80 + 40 * len(keys), 520),
        interactive=False,
    )
    return chart


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


def _register_card(store: Store, gate: GateView, reg: RegisterView) -> ft.Control:
    discs = ft.Row(
        [drawing.bloch_disc(ion, vec, reg.target_bloch.get(ion)) for ion, vec in sorted(reg.bloch.items())],
        wrap=True,
        spacing=16,
        run_spacing=12,
    )
    return card(
        f"Register after {gate.gate_id}",
        ft.Column(
            [
                ft.Row(
                    [discs, _populations_chart(reg.populations)],
                    wrap=True,
                    spacing=24,
                    run_spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                stat_row([stat_tile(reg.purity), stat_tile(reg.fidelity)]),
                status_line(
                    "solid arrow: the recorded state; dashed: the ideal one; a short arrow is mixed or entangled"
                ),
                details(
                    store,
                    "register",
                    [
                        status_line(reg.note),
                        data_table(
                            ["Pauli string", "<P>"],
                            [[k, f"{v:+.4f}"] for k, v in sorted(reg.pauli.items())],
                            numeric=[False, True],
                        ),
                    ],
                    title="Pauli expectations",
                ),
            ],
            spacing=12,
        ),
    )


def level1_page(store: Store, session: Session, record: Record, gate_param: str) -> ft.Control:
    gates = timeline(record)
    if not gates:
        return screen(
            [
                level_header("The circuit", "What did each gate do to the qubits?"),
                status_line("this circuit compiled to no gate"),
            ]
        )
    gate = next((g for g in gates if g.gate_id == gate_param), gates[0])
    key = record.key
    try:
        register = _register_card(store, gate, register_after(record, gate.index))
    except RegisterUnavailable as exc:  # the record keeps no register after this gate: say why, keep the rest
        register = card(
            f"Register after {gate.gate_id}", empty_state("Register not recorded", str(exc.args[0]))
        )
    step = record.step(gate.step_index)
    first_pulse = step.pulse_indices[0] if step.pulse_indices else 0
    params = ", ".join(f"{p:.4f}" for p in gate.params_rad)
    gate_card = card(
        f"Gate {gate.gate_id}",
        ft.Column(
            [
                ft.Text(f"{gate.name} on {ions_text(gate.ions)}, phases {params} rad", size=13),
                stat_row([stat_tile(gate.duration)]),
                details(store, "unitary", [_matrix_table(gate.unitary)], title="Ideal unitary"),
            ],
            spacing=GAP,
        ),
        actions=[
            ft.FilledButton(
                content=ft.Text("Zoom in: the pulses"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda _e: session.navigate(f"/job/{key}/schedule/{first_pulse}"),
            )
        ],
    )
    return screen(
        [
            level_header("The circuit", "What did each gate do to the qubits?", f"{len(gates)} native gates"),
            card(
                "Timeline",
                _lanes(
                    record,
                    gates,
                    gate.gate_id,
                    lambda gid: session.navigate(f"/job/{key}/circuit/{gid}"),
                    lane_width(store),
                ),
            ),
            columns(store, [register], [gate_card], split=(7, 5)),
        ]
    )
