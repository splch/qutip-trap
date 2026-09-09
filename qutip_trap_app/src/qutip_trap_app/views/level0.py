"""Level 0, the machine (PLAN.md Section 14.2 row 0; DESIGN.md Section 5).

Left: the device card. Centre: the circuit editor with OpenQASM 2 and IonQ JSON import, the shot count, the engine, and Run,
the one primary action. Right, after Run: the histogram with the target beside it, the bars that open their shots, the
verify-deeper action and its report. Before a run, the predict-then-reveal card asks what the learner expects.
"""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.learn import score_histogram_prediction
from qutip_trap_app.viewmodel.machine import Histogram, device_card_view, histogram, shot
from qutip_trap_app.views.common import (
    card,
    data_table,
    kv_rows,
    level_header,
    shown,
    value_cell,
)
from qutip_trap_app.views.state import BELL_QASM, Session, Store

SKETCHES: dict[str, str] = {
    "ideal": "the ideal machine's distribution",
    "uniform": "every outcome equally likely",
    "zero": "everything stays in |0...0>",
}


def _sketch_distribution(kind: str, target: dict[str, float], n: int) -> dict[str, float]:
    keys = [format(k, f"0{n}b") for k in range(2**n)]
    if kind == "uniform":
        return {k: 1.0 / len(keys) for k in keys}
    if kind == "zero":
        return {"0" * n: 1.0}
    return dict(target)


@ft.component
def CircuitEditor(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    running = bool(store.running())

    def set_text(e: Any) -> None:
        store.circuit_text = str(e.control.value)

    def set_format(e: Any) -> None:
        store.circuit_format = str(e.control.value)  # type: ignore[assignment]

    def set_shots(e: Any) -> None:
        try:
            store.shots = max(1, int(str(e.control.value)))
        except ValueError:
            store.error = "shots must be a positive integer"

    def set_engine(e: Any) -> None:
        store.engine = str(e.control.value)  # type: ignore[assignment]

    def load_bell(_e: Any) -> None:
        store.circuit_text = BELL_QASM
        store.circuit_format = "openqasm2"

    def run(_e: Any) -> None:
        session.submit_run()

    editor = ft.TextField(
        value=store.circuit_text,
        label="circuit (OpenQASM 2 or IonQ JSON)",
        multiline=True,
        min_lines=6,
        max_lines=12,
        on_change=set_text,
        text_size=13,
    )
    controls = ft.Row(
        [
            ft.Dropdown(
                label="format",
                value=store.circuit_format,
                options=[
                    ft.DropdownOption(key="openqasm2", text="OpenQASM 2"),
                    ft.DropdownOption(key="ionq_json", text="IonQ JSON"),
                ],
                on_select=set_format,
                width=160,
                dense=True,
            ),
            ft.TextField(
                label="shots",
                value=str(store.shots),
                width=110,
                on_change=set_shots,
                dense=True,
                text_size=13,
            ),
            ft.Dropdown(
                label="engine",
                value=store.engine,
                options=[
                    ft.DropdownOption(key="replay", text="channel replay (derived, seconds)"),
                    ft.DropdownOption(key="full", text="full simulation (JOINT_EXACT or GATE_LOCAL)"),
                ],
                on_select=set_engine,
                width=330,
                dense=True,
            ),
        ],
        wrap=True,
        spacing=10,
    )
    actions = ft.Row(
        [
            ft.FilledButton(
                content=ft.Text("Run"),
                icon=ft.Icons.PLAY_ARROW,
                on_click=run,
                disabled=running,
                tooltip="the one primary action of this level",
            ),
            ft.TextButton(content=ft.Text("Load the Bell example"), on_click=load_bell),
        ],
        spacing=8,
    )
    error = ft.Text(store.error, color=ft.Colors.ERROR, size=12) if store.error else ft.Container()
    return card(
        "Circuit",
        ft.Column([editor, controls, actions, error], spacing=10),
        subtitle="what you ask the machine to do",
    )


@ft.component
def ProgressRows(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    rows: list[ft.Control] = []
    for status in store.running():
        label = {
            "replay": "channel replay",
            "run_job": "full simulation",
            "verify": "verify deeper",
            "zoom": "re-simulation",
        }.get(status.request, status.request)
        rows.append(
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.ProgressRing(width=16, height=16, stroke_width=2),
                            ft.Text(f"{label}: {status.stage}", size=13),
                            ft.Text(f"{status.elapsed_s:.0f} s", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                        ],
                        spacing=8,
                    ),
                    ft.ProgressBar(value=status.fraction, bar_height=4),
                    ft.Text(status.message, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                ],
                spacing=4,
            )
        )
    if not rows:
        return ft.Container()
    cancel = ft.OutlinedButton(
        content=ft.Text("Cancel"),
        icon=ft.Icons.CLOSE,
        on_click=lambda e: session.cancel_all(),
        tooltip="stop every running job: the worker process restarts, and the next Run rebuilds its channel library",
    )
    return card(
        "Working",
        ft.Column(rows, spacing=10),
        subtitle="the solver runs in a worker process; the screen stays live",
        actions=[cancel],
    )


@ft.component
def PredictionCard(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    try:
        circuit = session.parse_circuit()
        from qutip_trap_app.core import ideal_probabilities

        target = {str(k): float(v) for k, v in ideal_probabilities(circuit).items()}
        n = circuit.n_qubits
    except Exception:
        return ft.Container()
    top = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(target.items(), key=lambda kv: -kv[1])[:4])
    labels = {"ideal": f"the outcomes {top}", "uniform": SKETCHES["uniform"], "zero": SKETCHES["zero"]}
    return card(
        "Before you run: what do you expect?",
        ft.Column(
            [
                ft.Text(
                    "Pick the histogram you expect. The run then shows you how far the physics landed from your pick (this is a prompt, not a test).",
                    size=13,
                ),
                ft.RadioGroup(
                    content=ft.Column([ft.Radio(value=k, label=v) for k, v in labels.items()], spacing=0),
                    value=store.prediction,
                    on_change=lambda e: setattr(store, "prediction", str(e.control.value)),
                ),
                ft.TextButton(
                    content=ft.Text("Skip"), on_click=lambda e: setattr(store, "prediction", "skipped")
                ),
            ],
            spacing=6,
        ),
        subtitle=f"{n} qubits; retrieval before the reveal",
    )


def _histogram_chart(h: Histogram, on_bar: Any) -> ft.Control:
    groups = []
    labels = []
    for k, bar in enumerate(h.bars):
        p = float(bar.probability.value or 0.0)
        t = float(bar.target_probability.value or 0.0)
        eb = float(bar.error_bar.value or 0.0)
        groups.append(
            fc.BarChartGroup(
                x=k,
                rods=[
                    fc.BarChartRod(
                        from_y=0.0,
                        to_y=p,
                        width=22,
                        color=ft.Colors.PRIMARY,
                        tooltip=f"simulated {p:.3f} ± {eb:.3f} ({bar.count.value} shots)",
                        border_radius=2,
                    ),
                    fc.BarChartRod(
                        from_y=0.0,
                        to_y=t,
                        width=22,
                        color=ft.Colors.with_opacity(0.25, ft.Colors.TERTIARY),
                        border_side=ft.BorderSide(1.5, ft.Colors.TERTIARY),
                        tooltip=f"target {t:.3f} (ideal circuit)",
                        border_radius=2,
                    ),
                ],
                spacing=2,
            )
        )
        labels.append(fc.ChartAxisLabel(value=k, label=ft.Text(bar.key, size=12)))
    top = max(
        [max(float(b.probability.value or 0), float(b.target_probability.value or 0)) for b in h.bars] + [0.1]
    )

    def on_event(e: fc.BarChartEvent) -> None:
        if e.type == fc.ChartEventType.TAP_UP and e.group_index is not None:
            on_bar(h.bars[int(e.group_index)].key)

    chart: ft.Control = fc.BarChart(
        groups=groups,
        bottom_axis=fc.ChartAxis(labels=labels, label_size=28),
        left_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(value=v, label=ft.Text(f"{v:.1f}", size=11))
                for v in (0.0, 0.25, 0.5, 0.75, 1.0)
                if v <= top * 1.05 + 0.01
            ],
            label_size=36,
        ),
        max_y=min(1.0, top * 1.15),
        min_y=0.0,
        horizontal_grid_lines=fc.ChartGridLines(interval=0.25, color=ft.Colors.OUTLINE_VARIANT, width=1),
        interactive=True,
        on_event=on_event,
        height=240,
        expand=True,
    )
    return chart


@ft.component
def ShotsPanel(store: Store, record: Record, h: Histogram, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    key = store.selected_bar
    if key is None:
        return ft.Text(
            "click a bar to open the shots stacked inside it",
            size=12,
            italic=True,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )
    bar = next((b for b in h.bars if b.key == key), None)
    if bar is None:
        return ft.Container()
    shots_list = ft.Row(
        [
            ft.TextButton(
                content=ft.Text(str(i), size=12),
                on_click=lambda e, i=i: setattr(store, "selected_shot", i),
                style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=6)),
            )
            for i in bar.shots[:120]
        ],
        wrap=True,
        spacing=0,
        run_spacing=0,
    )
    detail: ft.Control = ft.Container()
    if store.selected_shot is not None and store.selected_shot in bar.shots:
        sv = shot(record, store.selected_shot)
        rows: list[tuple[str, ft.Control]] = [
            ("outcome", shown(sv.bitstring, index, label=False)),
            ("sampled levels (ion 0 first)", ft.Text(str(sv.levels))),
            ("flags", ft.Text(", ".join(sv.heralds) or "none")),
            ("dynamical sample", ft.Text(str(sv.sample_index))),
            ("detection window", shown(sv.detection_window, index, label=False)),
        ]
        if sv.threshold is not None:
            rows.append(("threshold", shown(sv.threshold, index, label=False)))
        if sv.photon_counts is not None:
            rows.append(
                (
                    "photon counts",
                    ft.Row([shown(c, index, label=False) for c in sv.photon_counts], spacing=10),
                )
            )
        detail = ft.Column(
            [kv_rows(rows), ft.Text(sv.note, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=6,
        )
    return ft.Column(
        [
            ft.Text(f"{len(bar.shots)} shots gave {key}; click one", size=13, weight=ft.FontWeight.W_500),
            shots_list,
            detail,
        ],
        spacing=8,
    )


@ft.component
def ResultsPanel(store: Store, session: Session, record: Record, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    h = histogram(record)
    key = record.key()
    report = store.verify_reports.get(key)
    verifying = any(j.request == "verify" and j.message == key and not j.done for j in store.jobs.values())

    def on_bar(k: str) -> None:
        store.selected_bar = k
        store.selected_shot = None

    def verify(_e: Any) -> None:
        session.submit_verify(key, shots=min(store.shots, 200))

    feedback: ft.Control = ft.Container()
    if store.scored_prediction and store.scored_prediction != "skipped":
        sim = {b.key: float(b.probability.value or 0.0) for b in h.bars}
        bars = {b.key: float(b.error_bar.value or 0.0) for b in h.bars}
        target = {b.key: float(b.target_probability.value or 0.0) for b in h.bars}
        pred = _sketch_distribution(store.scored_prediction, target, h.n_qubits)
        score = score_histogram_prediction(pred, sim, bars)
        feedback = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        f"your pick against the simulated histogram: total variation {score.total_variation:.2f}",
                        size=13,
                        weight=ft.FontWeight.W_500,
                    ),
                    ft.Text(score.feedback, size=12),
                ],
                spacing=2,
            ),
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            padding=ft.Padding.all(10),
            border_radius=ft.BorderRadius.all(8),
        )
    stats = ft.Row(
        [
            shown(h.shots, index),
            ft.Text(
                f"distance to the target {h.total_variation_to_target:.3f}",
                size=13,
                tooltip="(1/2) sum |p - target|",
            ),
            ft.Text(f"largest deviation {h.largest_deviation_in_error_bars:.1f} error bars", size=13),
            ft.Text(h.level_note, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        wrap=True,
        spacing=16,
    )
    legend = ft.Row(
        [
            ft.Container(
                width=14, height=14, bgcolor=ft.Colors.PRIMARY, border_radius=ft.BorderRadius.all(2)
            ),
            ft.Text("simulated", size=12),
            ft.Container(
                width=14,
                height=14,
                bgcolor=ft.Colors.with_opacity(0.25, ft.Colors.TERTIARY),
                border=ft.Border.all(1.5, ft.Colors.TERTIARY),
                border_radius=ft.BorderRadius.all(2),
            ),
            ft.Text("target (ideal circuit), beside, never in place", size=12),
            ft.Text(h.bit_order_note, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        spacing=6,
        wrap=True,
    )
    table = data_table(
        ["outcome", "count", "probability", "error bar", "target"],
        [
            [
                b.key,
                value_cell(b.count, index),
                value_cell(b.probability, index),
                value_cell(b.error_bar, index),
                value_cell(b.target_probability, index),
            ]
            for b in h.bars
        ],
        numeric=[False, True, True, True, True],
    )
    verify_body: list[ft.Control] = []
    if report is not None:
        lines = [
            f"{report.shallow_level} against {report.deep_level or 'the Section 5.5 re-checks'} on {report.shots} shots ({report.wall_time_s:.0f} s)",
        ]
        if report.discrepancy_populations is not None and report.bound is not None:
            lines.append(
                f"register populations differ by {report.discrepancy_populations:.2e}; the bound is {report.bound:.2e}: {'within' if report.within_bound else 'OUTSIDE'}"
            )
        lines.append(
            f"histograms differ by {report.discrepancy_histogram:.3f} against a shot-noise scale of {report.statistical_scale:.3f}"
        )
        lines.extend(report.notes)
        verify_body.append(ft.Column([ft.Text(x, size=12) for x in lines], spacing=2))
        if report.deep_record_key and report.deep_record_key in store.records:
            deep_key = report.deep_record_key

            def open_deep(_e: Any, k: str = deep_key) -> None:
                store.current = k
                ft.context.page.navigate(f"/job/{k}")

            verify_body.append(ft.TextButton(content=ft.Text("open the deeper run"), on_click=open_deep))
    verify_row = ft.Row(
        [
            ft.OutlinedButton(
                content=ft.Text("Verify deeper"),
                icon=ft.Icons.VERIFIED,
                on_click=verify,
                disabled=verifying,
                tooltip="run the same job at the next-deeper engine and compare (Section 14.5)",
            ),
            ft.Text(
                "checks this result against the next engine, on a subset of shots",
                size=12,
                color=ft.Colors.ON_SURFACE_VARIANT,
            ),
        ],
        spacing=10,
        wrap=True,
    )
    return card(
        "Results",
        ft.Column(
            [
                feedback,
                _histogram_chart(h, on_bar),
                legend,
                stats,
                table,
                ShotsPanel(store, record, h, index),
                ft.Divider(height=8),
                verify_row,
            ]
            + verify_body,
            spacing=10,
        ),
        subtitle="the count of recorded shots per outcome (Section 14.1 rule 1)",
    )


@ft.component
def DeviceCardView(record: Record, index: ProvenanceIndex) -> ft.Control:
    cv = device_card_view(record)
    rows: list[tuple[str, ft.Control]] = [
        ("species", ft.Text(", ".join(cv.species))),
        ("ions", ft.Text(str(cv.n_ions))),
        ("native gates", shown(cv.native_gates, index, label=False)),
    ]
    rows += [
        (
            r.label,
            ft.Row(
                [
                    shown(r.value, index, label=False),
                    ft.Text(r.status, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ],
                spacing=6,
            ),
        )
        for r in cv.rows
    ]
    spam = [
        (
            r.label,
            ft.Row(
                [
                    shown(r.value, index, label=False),
                    ft.Text(r.status, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ],
                spacing=6,
            ),
        )
        for r in cv.spam
    ]
    errors = [
        (
            r.label,
            ft.Row(
                [
                    shown(r.value, index, label=False),
                    ft.Text(r.status, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ],
                spacing=6,
            ),
        )
        for r in cv.gate_errors
    ]
    modes = ft.Column(
        [
            ft.Row([ft.Text(m.detail, size=12, width=110), shown(m, index, label=False)], spacing=4)
            for m in cv.modes
        ],
        spacing=2,
    )
    body = ft.Column(
        [
            kv_rows(rows),
            ft.Text("modes", weight=ft.FontWeight.W_600, size=13),
            modes,
            ft.Text("readout and preparation", weight=ft.FontWeight.W_600, size=13),
            kv_rows(spam),
            ft.Text("gate errors", weight=ft.FontWeight.W_600, size=13),
            kv_rows(errors) if errors else ft.Text("none reported for this run", size=12, italic=True),
            ft.Row(
                [shown(cv.device_hash, index), shown(cv.seed, index), shown(cv.fidelity_level, index)],
                wrap=True,
                spacing=12,
            ),
            ft.Column([ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in cv.notes], spacing=2),
        ],
        spacing=8,
    )
    return card(
        "Device",
        body,
        subtitle="what a cloud customer is told about the machine; click a number's chip for its source",
    )


@ft.component
def Level0Page(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    record = store.record()
    header = level_header(
        0,
        "The machine",
        "What did the machine return, and how does it compare with a perfect one?",
        ""
        if record is None
        else f"job {record.key()[:10]} · {record.diagnostics.level}{' (derived)' if record.replay is not None else ''}",
    )
    left: list[ft.Control] = (
        [DeviceCardView(record, index)]
        if record is not None
        else [
            card(
                "Device",
                ft.Text(
                    "The device card fills in from the run's record: species, modes, readout errors, gate errors, each with its provenance chip.",
                    size=13,
                ),
            )
        ]
    )
    centre: list[ft.Control] = [CircuitEditor(store, session), ProgressRows(store, session)]
    plan = store.learner.plan(0)
    # asked before every reveal: the first run, and every run of a circuit edited since the last one
    if plan.prompt_before_reveal and store.prediction != "skipped" and store.prediction_pending():
        centre.append(PredictionCard(store, session))
    if record is not None:
        centre.append(ResultsPanel(store, session, record, index))
    elif not store.running():
        centre.append(
            card(
                "No result yet",
                ft.Text(
                    "Press Run. The Bell circuit is loaded: two ions, one entangling gate, two bars you can open down to the matrix element that made them.",
                    size=13,
                ),
            )
        )
    return ft.Column(
        [
            header,
            ft.ResponsiveRow(
                [
                    ft.Column(centre, col={"xs": 12, "lg": 7}, spacing=10),
                    ft.Column(left, col={"xs": 12, "lg": 4}, spacing=10),
                ],
                vertical_alignment=ft.CrossAxisAlignment.START,
                spacing=16,
                run_spacing=16,
            ),
        ],
        spacing=12,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )


__all__ = [
    "CircuitEditor",
    "DeviceCardView",
    "Level0Page",
    "PredictionCard",
    "ProgressRows",
    "ResultsPanel",
    "ShotsPanel",
]
