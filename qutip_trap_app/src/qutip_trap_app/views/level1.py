"""Level 1, the circuit (PLAN.md Section 14.2 row 1; DESIGN.md Section 5): the compiled native timeline on ion lanes, the
selected gate's target unitary and calibrated parameters, the register after that gate (populations, Bloch arrows, Pauli
expectations, purity, fidelity to the target so far), the phase register and the compile report."""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc
import numpy as np

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.circuit import (
    GateView,
    compile_report,
    phase_register,
    register_after,
    timeline,
)
from qutip_trap_app.views.common import (
    card,
    chip,
    content_widths,
    data_table,
    event_bool,
    kv_rows,
    level_header,
    shown,
    value_cell,
)
from qutip_trap_app.views.state import Store

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
                    height=26,
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
                )
            )
        lanes.append(
            ft.Row(
                [ft.Text(f"ion {ion}", size=12, width=44), ft.Stack(boxes, width=width, height=30)],
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


def _bloch_rows(bloch: dict[int, tuple[float, float, float]], index: ProvenanceIndex) -> ft.Control:
    rows: list[ft.Control] = []
    for ion, (x, y, z) in sorted(bloch.items()):
        length = float(np.sqrt(x * x + y * y + z * z))
        rows.append(
            ft.Row(
                [
                    ft.Text(f"ion {ion}", size=12, width=44),
                    ft.Text(f"({x:+.3f}, {y:+.3f}, {z:+.3f})", size=12, width=190, tooltip="(<X>, <Y>, <Z>)"),
                    ft.Container(
                        width=max(length, 0.0) * 120,
                        height=8,
                        bgcolor=ft.Colors.PRIMARY,
                        border_radius=ft.BorderRadius.all(4),
                        tooltip=f"arrow length {length:.3f}: 1 is a pure single-qubit state, 0 fully mixed or fully entangled",
                    ),
                    ft.Text(f"length {length:.3f}", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    chip("conv.computational_ordering", index),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    return ft.Column(rows, spacing=4)


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
def Level1Page(store: Store, record: Record, gate_param: str, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    # the two tiles own their open state (hooks before the early return, so their order never changes)
    unitary_open, set_unitary_open = ft.use_state(False)
    pauli_open, set_pauli_open = ft.use_state(False)
    gates = timeline(record)
    if not gates:
        return ft.Text("this circuit compiled to no gate pieces", size=13)
    selected = next((g for g in gates if g.gate_id == gate_param), gates[0])
    page = ft.context.page
    key = record.key()

    def on_select(gid: str) -> None:
        page.navigate(f"/job/{key}/circuit/{gid}")

    reg = register_after(record, selected.index)
    lanes_width, _strip = content_widths(store, page, 1)
    frame = phase_register(record)
    residuals = compile_report(record)
    detail_rows: list[tuple[str, ft.Control]] = [
        (
            "gate",
            ft.Row(
                [
                    shown(selected.name, index, label=False),
                    ft.Text(f"params {tuple(round(p, 4) for p in selected.params_rad)} rad", size=12),
                ],
                spacing=8,
            ),
        ),
        ("ions", ft.Text(str(selected.ions))),
        ("duration", shown(selected.duration, index, label=False)),
    ]
    for s in selected.calibrated:
        detail_rows.append(("calibrated from", shown(s, index, label=True)))
    if selected.error_estimate is not None:
        detail_rows.append(("error estimate", shown(selected.error_estimate, index, label=False)))
    if selected.channel is not None:
        detail_rows.append(
            (
                "channel infidelity",
                ft.Row(
                    [
                        ft.Text(f"{selected.channel.average_gate_infidelity:.2e}", size=13),
                        chip("conv.gate_fidelity_measure", index),
                    ],
                    spacing=4,
                ),
            )
        )
        detail_rows.append(
            (
                "depolarizing rate",
                ft.Row(
                    [
                        ft.Text(f"{selected.channel.depolarizing_rate:.2e}", size=13),
                        chip("conv.depolarizing_normalization", index),
                    ],
                    spacing=4,
                ),
            )
        )
    for s in selected.stark_frame:
        detail_rows.append(("Stark frame increment", shown(s, index, label=False)))
    first_pulse = (
        record.step(selected.step_index).pulse_indices[0]
        if record.step(selected.step_index).pulse_indices
        else 0
    )
    zoom_btn = ft.FilledButton(
        content=ft.Text("Zoom in: the pulses of this gate"),
        icon=ft.Icons.ZOOM_IN,
        on_click=lambda e: page.navigate(f"/job/{key}/schedule/{first_pulse}"),
    )
    unitary_tile = ft.ExpansionTile(
        title=ft.Row(
            [ft.Text("target unitary", size=13), shown(selected.unitary, index, label=False)], spacing=6
        ),
        controls=[
            ft.Container(
                content=_matrix_table(
                    record.schedule.targets[
                        [t.gate_id for t in record.schedule.targets].index(selected.gate_id)
                    ].unitary
                ),
                padding=ft.Padding.all(8),
            )
        ],
        dense=True,
        expanded=unitary_open,
        on_change=lambda e: set_unitary_open(event_bool(e)),
    )
    gate_card = card(
        f"Gate {selected.gate_id}",
        ft.Column([kv_rows(detail_rows), unitary_tile], spacing=8),
        subtitle="the compiler's target and what the calibration table gave it",
        actions=[zoom_btn],
    )
    pauli_rows: list[list[ft.Control | str]] = [
        [k, value_cell(v, index)] for k, v in sorted(reg.pauli.items())
    ]
    register_card = card(
        f"Register after {selected.gate_id}",
        ft.Column(
            [
                ft.Text(reg.weights_note, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("populations", size=12, weight=ft.FontWeight.W_600),
                                _populations_chart(reg.populations),
                            ],
                            spacing=4,
                        ),
                        ft.Column(
                            [
                                ft.Text("Bloch arrows", size=12, weight=ft.FontWeight.W_600),
                                _bloch_rows(reg.bloch, index),
                                ft.Row(
                                    [shown(reg.purity, index), shown(reg.fidelity, index)],
                                    wrap=True,
                                    spacing=12,
                                ),
                            ],
                            spacing=6,
                        ),
                    ],
                    spacing=24,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    wrap=True,
                ),
                ft.ExpansionTile(
                    title=ft.Text("Pauli expectations", size=13),
                    controls=[
                        ft.Container(
                            content=data_table(["operator", "<P>"], pauli_rows, numeric=[False, True]),
                            padding=ft.Padding.all(8),
                        )
                    ],
                    dense=True,
                    expanded=pauli_open,
                    on_change=lambda e: set_pauli_open(event_bool(e)),
                ),
            ],
            spacing=8,
        ),
        subtitle="the reduced density matrix of the recorded state (Section 14.1 rule 1)",
    )
    frame_rows = [
        (f"ion {q} final frame", shown(v, index, label=False)) for q, v in frame.final_frame.items()
    ]
    frame_rows += [(f"{gid} ion {q}", shown(v, index, label=False)) for gid, q, v in frame.stark_increments]
    frame_card = card(
        "Phase register",
        ft.Column(
            [
                kv_rows(frame_rows),
                ft.Text(frame.rule, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            spacing=6,
        ),
        subtitle="virtual Z rotations the machine remembers instead of playing",
    )
    compile_card = card(
        "Compile report",
        ft.Row([shown(r, index) for r in residuals], wrap=True, spacing=12),
        subtitle="how closely the native sequence reproduces each requested gate",
    )
    return ft.Column(
        [
            level_header(
                1,
                "The circuit",
                "What did each gate do to the qubits?",
                f"{len(gates)} native gate pieces; click a box to select",
            ),
            card(
                "Timeline",
                _lanes(record, gates, selected.gate_id, on_select),
                subtitle="native gates as played, on ion lanes; the selected gate is outlined",
            ),
            ft.ResponsiveRow(
                [
                    ft.Column([gate_card, frame_card, compile_card], col={"xs": 12, "lg": 5}, spacing=10),
                    ft.Column([register_card], col={"xs": 12, "lg": 7}, spacing=10),
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


__all__ = ["Level1Page"]
