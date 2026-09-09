"""Level 0, the machine (PLAN.md Section 14.2 row 0; DESIGN.md Sections 5 and 10).

Before a run the focal object is Run: the circuit editor, then the predict-then-reveal card. After a run the focal picture is
the histogram with the target beside it, first on the screen, with three stat tiles under it and the shots, the table and the
verify-deeper report behind disclosure; the editor follows. The device card is six stat tiles with every row behind Details.
"""

from __future__ import annotations

from typing import Any

import flet as ft
import flet_charts as fc

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record
from qutip_trap_app.viewmodel.learn import score_histogram_prediction
from qutip_trap_app.viewmodel.machine import Histogram, device_card_view, histogram, shot
from qutip_trap_app.viewmodel.presets import PRESETS, circuit_comparisons, circuit_presets
from qutip_trap_app.views.common import (
    card,
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
from qutip_trap_app.views.presets import comparison_table
from qutip_trap_app.views.state import Session, Store

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
def CircuitEditor(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    running = bool(store.running())

    def set_text(e: Any) -> None:
        store.circuit_text = str(e.control.value)
        store.active_preset = None

    def set_format(e: Any) -> None:
        store.circuit_format = str(e.control.value)  # type: ignore[assignment]

    def set_shots(e: Any) -> None:
        try:
            store.shots = max(1, int(str(e.control.value)))
        except ValueError:
            store.error = "shots must be a positive integer"

    def set_engine(e: Any) -> None:
        store.engine = str(e.control.value)  # type: ignore[assignment]

    def load(e: Any) -> None:
        choice = str(e.control.value)
        if choice == "bell":
            session.load_bell_example()
        else:
            session.load_circuit_preset(choice)

    editor = ft.TextField(
        value=store.circuit_text,
        label="circuit (OpenQASM 2 or IonQ JSON)",
        multiline=True,
        min_lines=5,
        max_lines=10,
        on_change=set_text,
        text_size=13,
        key="circuit-text",
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
                width=150,
                dense=True,
            ),
            ft.TextField(
                label="shots",
                value=str(store.shots),
                width=100,
                on_change=set_shots,
                dense=True,
                text_size=13,
            ),
            ft.Dropdown(
                label="engine",
                value=store.engine,
                options=[
                    ft.DropdownOption(key="replay", text="channel replay (seconds, derived)"),
                    ft.DropdownOption(key="full", text="full simulation"),
                ],
                on_select=set_engine,
                width=250,
                dense=True,
                tooltip="channel replay applies each gate's extracted channel (Section 14.2 row 0); the full simulation integrates the Hamiltonian",
            ),
            ft.Dropdown(
                label="load",
                value=store.active_preset or "bell",
                options=[ft.DropdownOption(key="bell", text="Bell example")]
                + [ft.DropdownOption(key=p.id, text=p.title) for p in circuit_presets()],
                on_select=load,
                width=250,
                dense=True,
            ),
        ],
        wrap=True,
        spacing=10,
    )
    actions: list[ft.Control] = [
        ft.FilledButton(
            content=ft.Text("Run"),
            icon=ft.Icons.PLAY_ARROW,
            on_click=lambda e: session.submit_run(),
            disabled=running,
            key="run",
        )
    ]
    if store.active_preset:
        spec = PRESETS[store.active_preset]
        actions.append(status_line(f"{spec.title}: {spec.duration}"))
    body: list[ft.Control] = [editor, controls]
    if store.error:
        body.append(ft.Text(store.error, color=ft.Colors.ERROR, size=12, key="error"))
    return card(
        "Circuit",
        ft.Column(body, spacing=10),
        why=lambda e: session.select_concept(0, "shot"),
        actions=actions,
        key="circuit",
    )


@ft.component
def ProgressRows(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    rows: list[ft.Control] = []
    for status in store.running():
        label = {
            "replay": "channel replay",
            "run_job": "full simulation",
            "request_run": "the requested run",
            "verify": "verify deeper",
            "zoom": "re-simulation",
            "preset": "published experiment",
            "derive": "deriving the device",
            "recalibrate": "recalibration",
            "fock_movie": "Fock movie",
            "tomography": "process tomography",
            "recheck": "convergence re-check",
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
                    status_line(status.message),
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
    return card("Working", ft.Column(rows, spacing=10), actions=[cancel], key="working")


@ft.component
def PredictionCard(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    try:
        circuit = session.parse_circuit()
        from qutip_trap_app.core import ideal_probabilities

        target = {str(k): float(v) for k, v in ideal_probabilities(circuit).items()}
    except Exception:
        return ft.Container()
    top = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(target.items(), key=lambda kv: -kv[1])[:4])
    labels = {"ideal": f"the outcomes {top}", "uniform": SKETCHES["uniform"], "zero": SKETCHES["zero"]}
    return card(
        "Before you run: which histogram do you expect?",
        ft.Column(
            [
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
        why=lambda e: session.select_concept(0, "histogram"),
        info="a prompt, not a test: the run scores your pick against the recorded histogram and never against a script",
        key="prediction",
    )


def _histogram_chart(h: Histogram, on_bar: Any, *, height: float = 300.0) -> ft.Control:
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
                        width=26,
                        color=ft.Colors.PRIMARY,
                        tooltip=f"simulated {p:.3f} ± {eb:.3f} ({bar.count.value} shots); click to open them",
                        border_radius=2,
                    ),
                    fc.BarChartRod(
                        from_y=0.0,
                        to_y=t,
                        width=26,
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
        height=height,
        expand=True,
    )
    return chart


def _legend() -> ft.Control:
    return ft.Row(
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
            ft.Text("ideal (target)", size=12),
        ],
        spacing=6,
        wrap=True,
    )


@ft.component
def ShotsPanel(store: Store, record: Record, h: Histogram, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    key = store.selected_bar
    if key is None:
        return ft.Container()
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
        tiles: list[ft.Control] = [
            stat_tile(sv.bitstring, index),
            stat_tile(sv.detection_window, index),
        ]
        if sv.threshold is not None:
            tiles.append(stat_tile(sv.threshold, index))
        if sv.photon_counts is not None:
            tiles.extend(
                stat_tile(c, index, label=f"photons, ion {i}") for i, c in enumerate(sv.photon_counts)
            )
        rows: list[tuple[str, ft.Control]] = [
            ("sampled levels (ion 0 first)", ft.Text(str(sv.levels), size=12)),
            ("flags", ft.Text(", ".join(sv.heralds) or "none", size=12)),
            ("dynamical sample", ft.Text(str(sv.sample_index), size=12)),
        ]
        detail = ft.Column([stat_row(tiles), kv_rows(rows), status_line(sv.note)], spacing=6)
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(f"{len(bar.shots)} shots gave {key}", size=13, weight=ft.FontWeight.W_500),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            icon_size=14,
                            tooltip="close the shots",
                            on_click=lambda e: setattr(store, "selected_bar", None),
                        ),
                    ],
                    spacing=4,
                ),
                shots_list,
                detail,
            ],
            spacing=6,
        ),
        padding=ft.Padding.all(8),
        border_radius=ft.BorderRadius.all(8),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        key="shots",
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

    body: list[ft.Control] = []
    if store.scored_prediction and store.scored_prediction != "skipped":
        sim = {b.key: float(b.probability.value or 0.0) for b in h.bars}
        bars = {b.key: float(b.error_bar.value or 0.0) for b in h.bars}
        target = {b.key: float(b.target_probability.value or 0.0) for b in h.bars}
        pred = _sketch_distribution(store.scored_prediction, target, h.n_qubits)
        score = score_histogram_prediction(pred, sim, bars)
        body.append(
            ft.Container(
                content=ft.Text(
                    f"your pick against the histogram: total variation {score.total_variation:.2f}; {score.feedback}",
                    size=12,
                ),
                bgcolor=ft.Colors.SECONDARY_CONTAINER,
                padding=ft.Padding.all(8),
                border_radius=ft.BorderRadius.all(8),
            )
        )
    body.append(_histogram_chart(h, on_bar))
    body.append(
        ft.Row(
            [
                stat_tile(h.shots, index),
                stat_tile(h.target_distance, index),
                stat_tile(h.largest_deviation, index),
                _legend(),
            ],
            wrap=True,
            spacing=8,
            run_spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )
    body.append(hint(store, 0, "histogram"))
    body.append(ShotsPanel(store, record, h, index))
    if store.active_preset and record.job.label == f"preset: {store.active_preset}":
        spec = PRESETS[store.active_preset]
        comps = circuit_comparisons(
            spec, record.results.probabilities, record.results.error_bars, record.results.register_fidelity
        )
        body.append(
            ft.Column(
                [
                    ft.Text(f"{spec.source}: {spec.row}", size=12, weight=ft.FontWeight.W_600),
                    comparison_table(comps, index),
                ],
                spacing=4,
            )
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
    detail_controls: list[ft.Control] = [table, status_line(h.bit_order_note), status_line(h.level_note)]
    if report is not None:
        lines = [
            f"{report.shallow_level} against {report.deep_level or 'the Section 5.5 re-checks'} on {report.shots} shots ({report.wall_time_s:.0f} s)",
        ]
        if report.discrepancy_populations is not None and report.bound is not None:
            lines.append(
                f"register populations differ by {report.discrepancy_populations:.2e}; the bound is {report.bound:.2e}: "
                f"{'within' if report.within_bound else 'OUTSIDE'}"
            )
        lines.append(
            f"histograms differ by {report.discrepancy_histogram:.3f} against a shot-noise scale of {report.statistical_scale:.3f}"
        )
        lines.extend(report.notes)
        detail_controls.append(ft.Column([ft.Text(x, size=12) for x in lines], spacing=2))
        if report.deep_record_key and report.deep_record_key in store.records:
            deep_key = report.deep_record_key

            def open_deep(_e: Any, k: str = deep_key) -> None:
                store.current = k
                ft.context.page.navigate(f"/job/{k}")

            detail_controls.append(ft.TextButton(content=ft.Text("open the deeper run"), on_click=open_deep))
    body.append(details("level0.results", detail_controls, store=store, session=session))
    verified_line = (
        status_line(
            f"verify deeper: {'within' if report.within_bound else 'OUTSIDE'} the bound"
            if report is not None and report.within_bound is not None
            else "verify deeper: done"
        )
        if report is not None
        else ft.Container()
    )
    return card(
        "Results",
        ft.Column(body, spacing=10),
        why=lambda e: session.select_concept(0, "target_vs_simulated"),
        info=(
            "every bar is the count of recorded shots (Section 14.1 rule 1); the outlined bars are the ideal circuit's "
            "distribution, shown beside the simulated one and never in its place (Section 14.5)"
        ),
        actions=[
            ft.OutlinedButton(
                content=ft.Text("Verify deeper"),
                icon=ft.Icons.VERIFIED,
                on_click=verify,
                disabled=verifying,
                tooltip="run the same job at the next-deeper engine on a subset of shots and compare (Section 14.5)",
                key="verify-deeper",
            ),
            verified_line,
        ],
        key="results",
    )


@ft.component
def DeviceCardView(store: Store, session: Session, record: Record, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    cv = device_card_view(record)
    page = ft.context.page
    plain = store.learner.plan(0).plain_labels_first
    tiles: list[ft.Control] = []
    for r in cv.spam:
        if r.label.startswith("q0"):
            tiles.append(stat_tile(r.value, index, plain=plain, status=r.status))
    for r in cv.gate_errors[:3]:
        tiles.append(stat_tile(r.value, index, plain=plain, label=f"{r.label} error", status=r.status))
    modes = ", ".join(f"{float(m.value or 0.0) / 1e6:.3f}" for m in cv.modes)
    rows: list[tuple[str, ft.Control]] = [
        ("species", ft.Text(", ".join(cv.species), size=12)),
        ("ions", ft.Text(str(cv.n_ions), size=12)),
        ("native gates", shown(cv.native_gates, index, label=False, size=12)),
    ]
    rows += [
        (r.label, ft.Row([shown(r.value, index, label=False, size=12), status_line(r.status)], spacing=6))
        for r in list(cv.rows) + list(cv.spam) + list(cv.gate_errors)
    ]
    rows += [(m.detail or "mode", shown(m, index, label=False, size=12)) for m in cv.modes]
    detail_controls: list[ft.Control] = [
        kv_rows(rows, label_width=190),
        ft.Row(
            [
                shown(cv.device_hash, index, size=12),
                shown(cv.seed, index, size=12),
                shown(cv.fidelity_level, index, size=12),
            ],
            wrap=True,
            spacing=12,
        ),
        ft.Column([status_line(n) for n in cv.notes], spacing=2),
    ]
    body = ft.Column(
        [
            ft.Text(
                f"{', '.join(sorted(set(cv.species)))}, {cv.n_ions} ions", size=13, weight=ft.FontWeight.W_500
            ),
            stat_row(tiles),
            status_line(f"modes: {modes} MHz"),
            details("level0.device", detail_controls, store=store, session=session),
        ],
        spacing=8,
    )
    return card(
        "Device",
        body,
        why=lambda e: session.select_concept(0, "spam"),
        info="what a cloud customer is told about the machine; estimate = a closed form, calibrated = the table's process tomography",
        actions=[
            ft.TextButton(
                content=ft.Text("Change the device"),
                icon=ft.Icons.FUNCTIONS,
                on_click=lambda e: page.navigate("/device/trap"),
            )
        ],
        key="device",
    )


@ft.component
def Level0Page(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    record = store.record()
    note = (
        ""
        if record is None
        else f"job {record.key()[:10]} · {record.diagnostics.level}{' (derived)' if record.replay is not None else ''}"
        + (f" · {record.job.label}" if record.job.label else "")
    )
    header = level_header(
        "The machine", "What did the machine return, and how does it compare with a perfect one?", note=note
    )
    right: list[ft.Control] = []
    layer = store.layer()
    if store.device_overrides or (
        layer is not None and record is not None and layer.device_hash != record.device_hash
    ):
        from qutip_trap_app.views.level4 import CurrentDeviceCard

        right.append(CurrentDeviceCard(store, session, index))
    if record is not None:
        right.append(DeviceCardView(store, session, record, index))
    plan = store.learner.plan(0)
    ask = plan.prompt_before_reveal and store.prediction != "skipped" and store.prediction_pending()
    centre: list[ft.Control]
    if record is not None:
        centre = [
            ResultsPanel(store, session, record, index),
            ProgressRows(store, session),
            CircuitEditor(store, session, index),
        ]
        if ask:
            centre.append(PredictionCard(store, session))
    else:
        centre = [CircuitEditor(store, session, index), ProgressRows(store, session)]
        if ask:
            centre.append(PredictionCard(store, session))
        if not store.running():
            centre.append(
                card(
                    "No result yet",
                    ft.Column(
                        [
                            ft.Text("Press Run.", size=13),
                            hint(store, 0, "shot"),
                        ],
                        spacing=4,
                    ),
                    key="empty",
                )
            )
    return ft.Column(
        [
            header,
            ft.ResponsiveRow(
                [
                    ft.Column(centre, col={"xs": 12, "lg": 8}, spacing=10),
                    ft.Column(right, col={"xs": 12, "lg": 4}, spacing=10),
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
