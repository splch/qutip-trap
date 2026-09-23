"""The Hamiltonian page: the equation the engine integrates for one step, its drive terms with their matrix elements, and
its collapse operators."""

from __future__ import annotations

from typing import Any

import flet as ft

from qutip_trap_app import resim
from qutip_trap_app.record import HamiltonianRecord, Record
from qutip_trap_app.viewmodel.hamiltonian import FORMULA, CollapseView, TermView, hamiltonian_view
from qutip_trap_app.viewmodel.shown import Shown
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import (
    CODE_FONT,
    CODE_FONT_FALLBACK,
    GAP,
    MUTED,
    card,
    data_table,
    details,
    level_header,
    progress_card,
    screen,
    shown,
    stat_row,
    stat_tile,
    status_line,
    value_text,
)
from qutip_trap_app.views.level3 import selection
from qutip_trap_app.views.state import Session, Store

MATRIX_LEVELS = 8
"""Fock levels of a matrix-element table shown."""


def _term_tile(t: TermView) -> ft.Control:
    matrices: list[ft.Control] = []
    for m, table in sorted(t.matrix_elements.items()):
        n = min(table.shape[0], MATRIX_LEVELS)
        elements = [
            Shown(label, float(table[r, c]) if r < table.shape[0] and c < table.shape[1] else None)
            for label, r, c in (
                ("carrier: n' = 0, n = 0", 0, 0),
                ("blue sideband: n' = 1, n = 0", 1, 0),
                ("red sideband: n' = 0, n = 1", 0, 1),
            )
        ]
        matrices.append(
            ft.Column(
                [
                    ft.Text(
                        f"mode {m}: |<n'|D(i eta)|n>|, the first {n} of {table.shape[0]} levels", size=12
                    ),
                    drawing.heatmap(
                        table[:n, :n],
                        x_labels=[str(k) for k in range(n)],
                        y_labels=[str(k) for k in range(n)],
                        x_title="n",
                        y_title="n'",
                        cell_w=22,
                        cell_h=16,
                    ),
                    stat_row([stat_tile(s) for s in elements]),
                ],
                spacing=6,
            )
        )
    tones: list[list[ft.Control | str]] = [
        [str(j), *(value_text(s) for s in cells)] for j, cells in enumerate(t.tones)
    ]
    body: list[ft.Control] = [stat_row([stat_tile(s) for s in t.tiles])]
    if tones:
        body.append(data_table(["tone", "detuning", "phase at the start", "peak Omega/2 pi"], tones))
    body.append(ft.Row([shown(e) for e in (*t.etas, *t.frozen)], wrap=True, spacing=12))
    body.extend(matrices)
    return ft.ExpansionTile(
        title=ft.Text(t.title, size=13, weight=ft.FontWeight.W_600),
        subtitle=ft.Text(t.operator, size=11, color=MUTED),
        controls=[
            ft.Container(
                content=ft.Column(body, spacing=12),
                padding=ft.Padding.symmetric(horizontal=GAP, vertical=4),
            )
        ],
        expanded=t.index == 0,
        dense=True,
    )


def _collapse_table(items: list[CollapseView]) -> ft.Control:
    return data_table(
        ["channel", "rate", "ion", "mode", "integrated", "operator"],
        [
            [
                c.channel,
                value_text(c.rate),
                "-" if c.ion is None else str(c.ion),
                "-" if c.mode is None else str(c.mode),
                "yes" if c.integrated else "no",
                c.operator,
            ]
            for c in items[:40]
        ],
    )


def _equation(store: Store, ham: HamiltonianRecord) -> list[ft.Control]:
    v = hamiltonian_view(ham)
    groups: dict[str, list[CollapseView]] = {}
    for c in v.collapse:
        groups.setdefault(c.channel.split("[")[0], []).append(c)
    free_rows: list[list[ft.Control | str]] = [
        [str(m), cls, value_text(w), value_text(off)] for m, cls, w, off in v.free_modes
    ]
    segment_rows: list[list[ft.Control | str]] = [
        [value_text(duration), value_text(fastest), pulses, str(n), kernel]
        for duration, fastest, pulses, n, kernel in v.segments
    ]
    return [
        card(
            f"H(t) of {v.gate_id}",
            ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            FORMULA,
                            size=13,
                            selectable=True,
                            font_family=CODE_FONT,
                            font_family_fallback=CODE_FONT_FALLBACK,
                        ),
                        padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                        border_radius=ft.BorderRadius.all(8),
                    ),
                    stat_row([stat_tile(s) for s in v.header]),
                    details(
                        store,
                        "hamiltonian.free",
                        [
                            data_table(["mode", "class", "omega_m/2 pi", "offset in this sample"], free_rows),
                            ft.Row([shown(s) for s in (*v.offsets, *v.caps)], wrap=True, spacing=12),
                            data_table(
                                ["duration", "fastest frequency", "pulses", "drive terms", "kernel"],
                                segment_rows,
                            ),
                            ft.Column(
                                [status_line(a) for a in v.approximations]
                                or [status_line("no approximation recorded")],
                                spacing=2,
                            ),
                        ],
                        title="Free terms, offsets, caps, segments and approximations",
                    ),
                    status_line("the step is cut at every pulse boundary, as the engine integrates it"),
                ],
                spacing=12,
            ),
        ),
        card(
            "Drive terms",
            ft.Column(
                [
                    status_line(
                        "one term per pulse and ion: the addressed ion and every neighbour its light reaches"
                    ),
                    *(_term_tile(t) for t in v.terms),
                ],
                spacing=4,
            ),
        ),
        card(
            "Collapse operators",
            ft.Column(
                [
                    ft.ExpansionTile(
                        title=ft.Text(
                            f"{name}: {len(items)} operator(s), "
                            + (
                                "integrated in this run"
                                if any(c.integrated for c in items)
                                else "listed, not integrated"
                            ),
                            size=13,
                            weight=ft.FontWeight.W_600,
                        ),
                        subtitle=ft.Text(items[0].note, size=11, color=MUTED),
                        controls=[_collapse_table(items)],
                        dense=True,
                    )
                    for name, items in groups.items()
                ],
                spacing=4,
            ),
        ),
    ]


def hamiltonian_page(
    store: Store, session: Session, record: Record, pulse_param: str, sample_param: str
) -> ft.Control:
    header = level_header("The equation", "What exactly does the simulator integrate for this pulse?")
    if not record.schedule.steps:
        return screen([header, status_line("this circuit plays no pulse: there is no equation to show")])
    step, sample, branch = selection(store, record, pulse_param, sample_param)
    key = record.key
    ham = record.cached(resim.hamiltonian_key(step, sample, branch), HamiltonianRecord)
    back = ft.OutlinedButton(
        content=ft.Text("Zoom out: the dynamics of this pulse"),
        icon=ft.Icons.ZOOM_OUT,
        on_click=lambda _e: session.navigate(f"/job/{key}/dynamics/{pulse_param}/{sample}"),
    )
    if ham is not None:
        return screen([header, *_equation(store, ham), ft.Row([back])])
    target: dict[str, Any] = {"key": key, "step": step, "sample": sample, "branch": branch}
    joint = record.diagnostics.level == "JOINT_EXACT"
    build = card(
        f"The equation of step {record.step(step).gate_id}",
        ft.Column(
            [
                status_line(
                    "built with the step's re-simulation, from the run's own device, sample and qubit shifts"
                    if joint
                    else "a GATE_LOCAL run has no joint space to build the equation on"
                ),
                progress_card(store, session),
            ],
            spacing=GAP,
        ),
        actions=[
            ft.FilledButton(
                content=ft.Text("Build the equation"),
                icon=ft.Icons.FUNCTIONS,
                on_click=lambda _e: session.submit_resim("zoom", **target),
                disabled=not joint or store.running_of("zoom", **target) is not None,
            ),
            back,
        ],
    )
    return screen([header, build])
