"""Level 3, the dynamics: one step's stored points at once; its fine re-simulation, process matrix and convergence
re-checks as background jobs."""

from __future__ import annotations

from typing import Any

import flet as ft
import numpy as np

from qutip_trap_app import resim
from qutip_trap_app.record import ProcessMatrixRecord, Recheck, Record, ZoomTrace
from qutip_trap_app.viewmodel.dynamics import (
    PulseDynamics,
    Series,
    fock_heatmaps,
    process_view,
    pulse_dynamics,
    recorded_zoom,
)
from qutip_trap_app.viewmodel.numerics import NumericsPanel, numerics_panel
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import (
    GAP,
    card,
    columns,
    data_table,
    details,
    kv_rows,
    level_header,
    progress_card,
    screen,
    shown,
    stat_row,
    stat_tile,
    status_line,
    value_text,
)
from qutip_trap_app.views.state import Session, Store


def step_for_pulse(record: Record, pulse_param: str) -> int:
    pulse = int(pulse_param) if pulse_param.isdigit() else 0
    for st in record.schedule.steps:
        if pulse in st.pulse_indices:
            return st.index
    return next((s.index for s in record.schedule.steps if s.kind == "gate"), 0)


def selection(store: Store, record: Record, pulse_param: str, sample_param: str) -> tuple[int, int, int]:
    """(step, sample, branch) the route and the store select, clipped to what the record holds."""
    sample = int(sample_param) if sample_param.isdigit() else 0
    return (
        step_for_pulse(record, pulse_param),
        min(sample, max(record.n_samples - 1, 0)),
        min(max(store.branch, 0), max(record.n_branches - 1, 0)),
    )


def current_zoom(record: Record, step: int, sample: int, branch: int) -> tuple[ZoomTrace | None, bool]:
    """The fine zoom when it is cached, else the run's own stored points; (None, False) for a record with no trace."""
    z = record.cached(resim.zoom_key(step, sample, branch, resim.DEFAULT_ZOOM_POINTS), ZoomTrace)
    if z is not None:
        return z, True
    try:
        return recorded_zoom(record, step, sample, branch), False
    except KeyError:
        return None, False


def rechecks(record: Record, step: int, sample: int, branch: int) -> tuple[Recheck, ...]:
    found = (
        record.cached(resim.recheck_key(w, step, sample, branch), Recheck)
        for w in ("tolerance", "truncation")
    )
    return tuple(c for c in found if c is not None)


def level3_numerics(store: Store, record: Record, pulse_param: str, sample_param: str) -> NumericsPanel:
    """The numerics of the zoomed step: its populations at the caps and its re-checks when they ran."""
    step, sample, branch = selection(store, record, pulse_param, sample_param)
    z, fine = current_zoom(record, step, sample, branch) if record.schedule.steps else (None, False)
    return numerics_panel(record, zoom=z if fine else None, rechecks=rechecks(record, step, sample, branch))


def _chart(series: tuple[Series, ...] | list[Series], y_title: str, height: float = 200.0) -> ft.Control:
    return drawing.line_chart(
        [(s.label, s.times_s * 1e6, s.values) for s in series],
        x_title="t (µs)",
        y_title=y_title,
        height=height,
    )


def _no_trace(session: Session, record: Record, header: ft.Control, has_step: bool) -> ft.Control:
    if not has_step:
        why = "this circuit plays no pulse (a bare measurement, or virtual Z rotations alone): there is nothing inside"
    else:
        why = (
            "this GATE_LOCAL run stores no joint trace: each gate ran on its own space and the motion was traced out "
            "between gates; the register after every gate is on Level 1"
        )
    return screen(
        [
            header,
            card(
                "No trace to open",
                ft.Text(why, size=14),
                actions=[
                    ft.FilledTonalButton(
                        content=ft.Text("Back to the circuit"),
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda _e: session.navigate(f"/job/{record.key}"),
                    )
                ],
            ),
        ]
    )


def _motion_card(store: Store, dyn: PulseDynamics) -> ft.Control:
    children: list[ft.Control] = []
    if dyn.loops:
        modes = sorted({lp.mode for lp in dyn.loops})
        first_ion = min(lp.ion for lp in dyn.loops if lp.ion is not None)
        children.append(
            drawing.phase_space(
                [
                    drawing.PhaseLoop(
                        f"{lp.label}: ends {lp.closes:.1e} from its start",
                        lp.alpha,
                        modes.index(lp.mode),
                        lp.ion != first_ion,
                    )
                    for lp in dyn.loops
                ]
            )
        )
    else:
        children.append(
            status_line("this step plays no entangling waveform: no spin-dependent force, no loop")
        )
    children.append(_chart(dyn.nbar, "<n_m>", 150))
    rows: list[list[ft.Control | str]] = [
        [
            lp.label,
            f"{lp.closes:.2e}",
            "-" if lp.chi_closed_form_rad is None else f"{lp.chi_closed_form_rad:+.4f}",
            "-" if lp.chi_m_rad is None else f"{lp.chi_m_rad:+.4f}",
        ]
        for lp in dyn.loops
    ]
    rows += [[lp.label, f"{lp.excursion:.2e} (largest reach)", "-", "-"] for lp in dyn.mean_alpha]
    children.append(
        details(
            store,
            "loops",
            [
                data_table(
                    ["loop", "end to start", "angle from the loops (rad)", "angle booked (rad)"], rows
                ),
                status_line(
                    "each spin branch is pushed round a loop; the swept area is the entangling angle. The trace's "
                    "spin-averaged <a_m> cancels between the branches"
                ),
            ],
            title="Closure per loop",
        )
    )
    return card("The motion", ft.Column(children, spacing=12))


def _qubits_card(dyn: PulseDynamics) -> ft.Control:
    children: list[ft.Control] = [
        _chart([*dyn.populations, *dyn.coherences], "P1, |rho_01|", 220),
        stat_row([stat_tile(s.last()) for s in dyn.populations]),
    ]
    if dyn.concurrence is not None:
        children.append(_chart([dyn.concurrence, *dyn.pauli], "concurrence, <P>", 150))
    return card("The qubits", ft.Column(children, spacing=12))


def _fock_card(z: ZoomTrace, dyn: PulseDynamics, fine: bool) -> ft.Control:
    children: list[ft.Control] = []
    for m in sorted(set(dyn.fock_start) | set(dyn.fock_end)):
        start = dyn.fock_start.get(m, np.zeros(0))
        end = dyn.fock_end.get(m, np.zeros(0))
        children.append(
            ft.Column(
                [
                    ft.Text(f"mode {m}: P(n) at the start (outlined) and the end (filled)", size=12),
                    drawing.fock_bars(start, end),
                ],
                spacing=6,
            )
        )
    for hm in fock_heatmaps(z):
        children.append(
            ft.Column(
                [
                    ft.Text(f"mode {hm.mode}: P(n, t) at {hm.times_s.size} times inside the step", size=12),
                    drawing.heatmap(
                        hm.values.T,
                        x_labels=[f"{t * 1e6:.1f}" for t in hm.times_s],
                        y_labels=[str(n) for n in range(hm.values.shape[1])],
                        x_title="t (µs)",
                        y_title="n",
                    ),
                ],
                spacing=6,
            )
        )
    if not fine:
        children.append(status_line("re-simulate the step for P(n, t) inside it"))
    elif not children:
        children.append(status_line("no mode is carried in Fock states"))
    return card("Vibration quanta", ft.Column(children, spacing=12))


def _sample_card(store: Store, dyn: PulseDynamics) -> ft.Control:
    jumps: list[ft.Control] = [
        ft.Text(f"{t * 1e6:.2f} µs: {channel}", size=12) for t, channel in dyn.jumps
    ] or [status_line("no quantum jump in this sample")]
    tiles = [stat_tile(dyn.wall_time), *(stat_tile(b) for b in dyn.boundary_population)]
    if dyn.norm_deficit is not None:
        tiles.insert(0, stat_tile(dyn.norm_deficit))
    quasi: list[tuple[str, ft.Control]] = [(s.label, value_text(s)) for s in dyn.sample_values] or [
        ("quasi-static values", status_line("none drawn: the device is quiet"))
    ]
    quasi += [(s.label, value_text(s)) for s in dyn.frozen]
    return card(
        "Jumps and this sample",
        ft.Column(
            [
                ft.Column(jumps, spacing=2),
                stat_row(tiles),
                details(
                    store,
                    "sample",
                    [kv_rows(quasi, label_width=240)],
                    title="Quasi-static values and frozen modes",
                ),
            ],
            spacing=GAP,
        ),
    )


def _process_card(
    store: Store, session: Session, pm: ProcessMatrixRecord | None, target: dict[str, Any]
) -> ft.Control:
    if pm is None:
        return card(
            "The gate as a channel",
            status_line("the step's channel on every product input, from its recorded motional state"),
            actions=[
                ft.OutlinedButton(
                    content=ft.Text("Compute the process matrix"),
                    icon=ft.Icons.GRID_4X4,
                    on_click=lambda _e: session.submit_resim("tomography", **target),
                    disabled=store.running_of("tomography", **target) is not None,
                )
            ],
        )
    pv = process_view(pm)
    n = int(round(np.sqrt(pv.choi_abs.shape[0])))
    labels = [format(k, f"0{max(1, int(np.log2(n)))}b") for k in range(n)]
    pair_labels = [f"{a}{b}" for a in labels for b in labels]
    maps = [
        drawing.heatmap(
            m, x_labels=pair_labels, y_labels=pair_labels, x_title=title, y_title="", cell_w=12, cell_h=12
        )
        for m, title in ((pv.choi_abs, "simulated"), (pv.ideal_choi_abs, "ideal"))
    ]
    return card(
        "The gate as a channel",
        ft.Column(
            [
                stat_row(
                    [
                        stat_tile(pv.infidelity),
                        stat_tile(pv.entanglement_infidelity),
                        stat_tile(pv.depolarizing_rate),
                    ]
                ),
                details(
                    store,
                    "process",
                    [
                        ft.Row(
                            [shown(pv.cp_residual), shown(pv.tp_residual), shown(pv.wall_time)],
                            wrap=True,
                            spacing=12,
                        ),
                        ft.Row([shown(p) for p in pv.pauli[:6]], wrap=True, spacing=12),
                        ft.Text("|Choi| of the simulated channel and of the ideal gate", size=12),
                        ft.Row(maps, wrap=True, spacing=16),
                    ],
                    title="Pauli twirl and the Choi matrices",
                ),
            ],
            spacing=12,
        ),
    )


def _recheck_card(
    store: Store, session: Session, checks: tuple[Recheck, ...], target: dict[str, Any]
) -> ft.Control:
    lines: list[ft.Control] = [
        status_line(
            f"{c.what}: largest change {c.max_change:.2e} ({'converged' if c.converged else 'NOT converged'})"
        )
        for c in checks
    ] or [status_line("the numerics strip's badge turns pass or fail once the re-checks ran")]
    return card(
        "Convergence of this step",
        ft.Column(lines, spacing=4),
        actions=[
            ft.OutlinedButton(
                content=ft.Text(f"Re-check: tolerances x0.1 and caps +{resim.CAP_RAISE}"),
                icon=ft.Icons.VERIFIED_OUTLINED,
                on_click=lambda _e: session.submit_resim("recheck", **target),
                disabled=store.running_of("recheck", **target) is not None,
            )
        ],
    )


def level3_page(
    store: Store, session: Session, record: Record, pulse_param: str, sample_param: str
) -> ft.Control:
    step, sample, branch = selection(store, record, pulse_param, sample_param)
    has_step = bool(record.schedule.steps)
    note = (
        f"step {record.step(step).gate_id}, sample {sample}, branch {branch}"
        if has_step
        else "no pulse was played"
    )
    header = level_header("The dynamics", "What happened inside one pulse?", note)
    z, fine = current_zoom(record, step, sample, branch) if has_step else (None, False)
    if z is None:
        return _no_trace(session, record, header, has_step)
    dyn = pulse_dynamics(record, z)
    key = record.key
    target: dict[str, Any] = {"key": key, "step": step, "sample": sample, "branch": branch}

    def set_sample(e: Any) -> None:
        session.navigate(f"/job/{key}/dynamics/{pulse_param}/{e.control.value}")

    def set_branch(e: Any) -> None:
        store.branch = int(e.control.value)

    controls = ft.Row(
        [
            ft.Dropdown(
                label="sample",
                value=str(sample),
                options=[ft.DropdownOption(key=str(k), text=f"sample {k}") for k in range(record.n_samples)],
                on_select=set_sample,
                width=130,
                dense=True,
            ),
            ft.Dropdown(
                label="branch",
                value=str(branch),
                options=[
                    ft.DropdownOption(
                        key=str(b.index), text=f"branch {b.index}: weight {b.weight:.3g}, Fock {b.fock}"
                    )
                    for b in record.branches
                ],
                on_select=set_branch,
                width=320,
                dense=True,
            ),
            ft.FilledButton(
                content=ft.Text("Re-simulated" if fine else "Re-simulate at fine resolution"),
                icon=ft.Icons.ZOOM_IN,
                on_click=lambda _e: session.submit_resim("zoom", **target),
                disabled=fine or store.running_of("zoom", **target) is not None,
                tooltip=f"this step again from its recorded initial state, {resim.DEFAULT_ZOOM_POINTS} points per segment",
            ),
            ft.OutlinedButton(
                content=ft.Text("Zoom in: the equation"),
                icon=ft.Icons.FUNCTIONS,
                on_click=lambda _e: session.navigate(f"/job/{key}/hamiltonian/{pulse_param}/{sample}"),
            ),
        ],
        wrap=True,
        spacing=GAP,
        run_spacing=GAP,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    trace_note = (
        f"fine trace: {dyn.times_s.size} points at dimension {z.engine_dimension}"
        if fine
        else f"the run's own {dyn.times_s.size} stored points; re-simulate for the time resolution inside the segments"
    )
    pm = record.cached(resim.process_matrix_key(step, sample, branch), ProcessMatrixRecord)
    return screen(
        [
            header,
            controls,
            status_line(trace_note),
            progress_card(store, session),
            columns(store, [_motion_card(store, dyn)], [_qubits_card(dyn)]),
            columns(
                store,
                [
                    _fock_card(z, dyn, fine),
                    _recheck_card(store, session, rechecks(record, step, sample, branch), target),
                ],
                [_sample_card(store, dyn), _process_card(store, session, pm, target)],
            ),
        ]
    )
