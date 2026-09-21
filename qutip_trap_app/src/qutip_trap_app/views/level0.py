"""Level 0, the machine (PLAN.md Section 14.2 row 0; DESIGN.md Sections 5 and 10).

Before a run the focal object is Run: the circuit builder (``views/builder.py``), then the predict-then-reveal card where
the histogram will appear.
After a run the focal picture is the histogram with the target beside it, first on the screen, with three stat tiles under
it and the shots, the table and the verify-deeper report behind disclosure; the builder follows. The device card is six
stat tiles with every row behind Details.
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
from qutip_trap_app.views import theme
from qutip_trap_app.views.builder import CircuitBuilder, builder_toolbar
from qutip_trap_app.views.common import (
    HAIRLINE,
    MUTED,
    card,
    columns,
    data_table,
    details,
    empty_state,
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
from qutip_trap_app.views.state import MAX_SHOTS, Session, Store

SKETCHES: dict[str, str] = {
    "ideal": "the ideal machine's distribution",
    "uniform": "every outcome equally likely",
    "zero": "everything stays in |0...0>",
}

REQUEST_LABELS = {
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
}


def _sketch_distribution(kind: str, target: dict[str, float], n: int) -> dict[str, float]:
    keys = [format(k, f"0{n}b") for k in range(2**n)]
    if kind == "uniform":
        return {k: 1.0 / len(keys) for k in keys}
    if kind == "zero":
        return {"0" * n: 1.0}
    return dict(target)


def _field_style() -> dict[str, Any]:
    return {"dense": True, "text_size": 13, "border_radius": ft.BorderRadius.all(theme.RADIUS_TILE)}


@ft.component
def CircuitEditor(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    running = bool(store.running())

    def set_shots(e: Any) -> None:
        try:
            shots = int(str(e.control.value))
        except ValueError:
            store.error = "shots must be a positive integer"
            return
        store.shots = min(max(1, shots), MAX_SHOTS)
        if shots > MAX_SHOTS:
            store.error = f"shots are capped at {MAX_SHOTS}: every shot is read out one by one"

    def set_engine(e: Any) -> None:
        store.engine = str(e.control.value)  # type: ignore[assignment]

    def load(e: Any) -> None:
        choice = str(e.control.value)
        if choice == "bell":
            session.load_bell_example()
        else:
            session.load_circuit_preset(choice)

    controls = ft.Row(
        [
            ft.TextField(
                label="shots", value=str(store.shots), width=96, on_change=set_shots, **_field_style()
            ),
            ft.Dropdown(
                label="engine",
                value=store.engine,
                options=[
                    ft.DropdownOption(key="replay", text="Channel replay · fast, derived"),
                    ft.DropdownOption(key="full", text="Full simulation · slow, exact"),
                ],
                on_select=set_engine,
                width=270,
                tooltip="channel replay applies each gate's extracted channel (Section 14.2 row 0); the full simulation integrates the Hamiltonian",
                **_field_style(),
            ),
            ft.Dropdown(
                label="load",
                value=store.active_preset or "bell",
                options=[ft.DropdownOption(key="bell", text="Bell example")]
                + [ft.DropdownOption(key=p.id, text=p.title) for p in circuit_presets()],
                on_select=load,
                width=220,
                **_field_style(),
            ),
        ],
        wrap=True,
        spacing=theme.GAP,
        run_spacing=theme.GAP,
    )
    actions: list[ft.Control] = [
        ft.FilledButton(
            content=ft.Text("Run", size=15, weight=ft.FontWeight.W_600),
            icon=ft.Icons.PLAY_ARROW,
            on_click=lambda e: session.submit_run(),
            disabled=running,
            height=44,
            key="run",
        )
    ]
    if store.active_preset:
        spec = PRESETS[store.active_preset]
        actions.append(status_line(f"{spec.title}: {spec.duration}"))
    body: list[ft.Control] = [CircuitBuilder(store, session), controls]
    if store.error:
        body.append(
            ft.Row(
                [
                    ft.Icon(ft.Icons.ERROR_OUTLINE, size=16, color=ft.Colors.ERROR),
                    ft.Text(
                        store.error, color=ft.Colors.ERROR, size=theme.SIZE_SMALL, expand=True, key="error"
                    ),
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
    return card(
        "Circuit",
        ft.Column(body, spacing=12),
        why=lambda e: session.select_concept(0, "shot"),
        info=(
            "tap a gate to add it to the selected wire, drag it to place it, or use the + at a wire's end; a placed gate "
            "opens its qubits and angles beneath; the OpenQASM 2 text and an OpenQASM 2 or IonQ JSON import are under Code"
        ),
        actions=actions,
        trailing=builder_toolbar(store, session),
        key="circuit",
    )


@ft.component
def ProgressRows(store: Store, session: Session) -> ft.Control:
    ft.use_state(store)
    rows: list[ft.Control] = []
    for status in store.running():
        label = REQUEST_LABELS.get(status.request, status.request)
        rows.append(
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.ProgressRing(width=14, height=14, stroke_width=2),
                            ft.Text(f"{label}: {status.stage}", size=theme.SIZE_SMALL + 1, expand=True),
                            ft.Text(f"{status.elapsed_s:.0f} s", size=theme.SIZE_SMALL, color=MUTED),
                        ],
                        spacing=theme.GAP,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.ProgressBar(value=status.fraction, bar_height=4, border_radius=ft.BorderRadius.all(2)),
                    status_line(status.message),
                ],
                spacing=6,
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
    return card("Working", ft.Column(rows, spacing=12), actions=[cancel], key="working")


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


def _histogram_chart(h: Histogram, on_bar: Any, *, height: float = 280.0) -> ft.Control:
    """The focal picture of Level 0: one filled bar of recorded shots per outcome, the ideal circuit's outlined bar beside
    it (never in its place), hairline gridlines, the bars thin with a rounded top (the dataviz mark specs)."""
    groups = []
    labels = []
    cap = ft.BorderRadius.only(top_left=4, top_right=4)
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
                        width=18,
                        color=ft.Colors.PRIMARY,
                        tooltip=f"simulated {p:.3f} ± {eb:.3f} ({bar.count.value} shots); click to open them",
                        border_radius=cap,
                    ),
                    fc.BarChartRod(
                        from_y=0.0,
                        to_y=t,
                        width=18,
                        color=ft.Colors.with_opacity(0.12, ft.Colors.TERTIARY),
                        border_side=ft.BorderSide(1.5, ft.Colors.TERTIARY),
                        tooltip=f"target {t:.3f} (ideal circuit)",
                        border_radius=cap,
                    ),
                ],
                spacing=3,
            )
        )
        labels.append(fc.ChartAxisLabel(value=k, label=ft.Text(bar.key, size=theme.SIZE_SMALL, color=MUTED)))
    top = max(
        [max(float(b.probability.value or 0), float(b.target_probability.value or 0)) for b in h.bars] + [0.1]
    )
    max_y = min(1.0, top * 1.15)

    def on_event(e: fc.BarChartEvent) -> None:
        if e.type == fc.ChartEventType.TAP_UP and e.group_index is not None:
            on_bar(h.bars[int(e.group_index)].key)

    chart: ft.Control = fc.BarChart(
        groups=groups,
        bottom_axis=fc.ChartAxis(labels=labels, label_size=28),
        left_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(value=v, label=ft.Text(f"{v:.2f}", size=theme.SIZE_CAPTION, color=MUTED))
                for v in (0.0, 0.25, 0.5, 0.75, 1.0)
                if v <= max_y + 0.01
            ],
            label_size=40,
        ),
        max_y=max_y,
        min_y=0.0,
        horizontal_grid_lines=fc.ChartGridLines(interval=0.25, color=ft.Colors.OUTLINE_VARIANT, width=1),
        interactive=True,
        on_event=on_event,
        height=height,
        expand=True,
    )
    return chart


def _swatch(color: str, *, outlined: str | None = None) -> ft.Control:
    return ft.Container(
        width=12,
        height=12,
        bgcolor=color,
        border=ft.Border.all(1.5, outlined) if outlined else None,
        border_radius=ft.BorderRadius.all(3),
    )


def _legend() -> ft.Control:
    return ft.Row(
        [
            _swatch(ft.Colors.PRIMARY),
            ft.Text("simulated", size=theme.SIZE_SMALL, color=MUTED),
            _swatch(ft.Colors.with_opacity(0.12, ft.Colors.TERTIARY), outlined=ft.Colors.TERTIARY),
            ft.Text("ideal (target)", size=theme.SIZE_SMALL, color=MUTED),
        ],
        spacing=6,
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
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
                content=ft.Text(str(i), size=theme.SIZE_SMALL),
                on_click=lambda e, i=i: setattr(store, "selected_shot", i),
                style=ft.ButtonStyle(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    visual_density=ft.VisualDensity.COMPACT,
                    bgcolor={ft.ControlState.DEFAULT: ft.Colors.SECONDARY_CONTAINER}
                    if store.selected_shot == i
                    else None,
                ),
            )
            for i in bar.shots[:120]
        ],
        wrap=True,
        spacing=2,
        run_spacing=2,
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
            ("sampled levels (ion 0 first)", ft.Text(str(sv.levels), size=theme.SIZE_SMALL)),
            ("flags", ft.Text(", ".join(sv.heralds) or "none", size=theme.SIZE_SMALL)),
            ("dynamical sample", ft.Text(str(sv.sample_index), size=theme.SIZE_SMALL)),
        ]
        detail = ft.Column([stat_row(tiles), kv_rows(rows), status_line(sv.note)], spacing=8)
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(
                            f"{len(bar.shots)} shots gave {key}",
                            size=theme.SIZE_SMALL + 1,
                            weight=ft.FontWeight.W_600,
                        ),
                        ft.Container(expand=True),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            icon_size=16,
                            tooltip="close the shots",
                            on_click=lambda e: setattr(store, "selected_bar", None),
                        ),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                shots_list,
                detail,
            ],
            spacing=8,
        ),
        padding=ft.Padding.all(12),
        border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border=ft.Border.all(1, HAIRLINE),
        key="shots",
    )


@ft.component
def ResultsPanel(store: Store, session: Session, record: Record, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    h = histogram(record)
    key = record.key()
    report = store.verify_reports.get(key)
    verifying = store.running_of("verify", key=key) is not None

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
                    size=theme.SIZE_SMALL,
                ),
                bgcolor=ft.Colors.SECONDARY_CONTAINER,
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
            )
        )
    body.append(_histogram_chart(h, on_bar))
    body.append(ft.Row([_legend()], alignment=ft.MainAxisAlignment.END))
    body.append(
        stat_row(
            [
                stat_tile(h.shots, index),
                stat_tile(h.target_distance, index),
                stat_tile(h.largest_deviation, index),
            ]
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
                    ft.Text(f"{spec.source}: {spec.row}", size=theme.SIZE_SMALL, weight=ft.FontWeight.W_600),
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
        detail_controls.append(ft.Column([ft.Text(x, size=theme.SIZE_SMALL) for x in lines], spacing=2))
        if report.deep_record_key and report.deep_record_key in store.records:
            deep_key = report.deep_record_key

            def open_deep(_e: Any, k: str = deep_key) -> None:
                store.current = k
                ft.context.page.navigate(f"/job/{k}")

            detail_controls.append(ft.TextButton(content=ft.Text("open the deeper run"), on_click=open_deep))
    body.append(details("level0.results", detail_controls, store=store, level=0, session=session))
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
        ft.Column(body, spacing=12),
        why=lambda e: session.select_concept(0, "target_vs_simulated"),
        info=(
            "every bar is the count of recorded shots (Section 14.1 rule 1); the outlined bars are the ideal circuit's "
            "distribution, shown beside the simulated one and never in its place (Section 14.5)"
        ),
        actions=[
            ft.OutlinedButton(
                content=ft.Text("Verify deeper"),
                icon=ft.Icons.VERIFIED_OUTLINED,
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
        ("species", ft.Text(", ".join(cv.species), size=theme.SIZE_SMALL)),
        ("ions", ft.Text(str(cv.n_ions), size=theme.SIZE_SMALL)),
        ("native gates", shown(cv.native_gates, index, label=False, size=theme.SIZE_SMALL)),
    ]
    rows += [
        (
            r.label,
            ft.Row(
                [shown(r.value, index, label=False, size=theme.SIZE_SMALL), status_line(r.status)], spacing=6
            ),
        )
        for r in list(cv.rows) + list(cv.spam) + list(cv.gate_errors)
    ]
    rows += [(m.detail or "mode", shown(m, index, label=False, size=theme.SIZE_SMALL)) for m in cv.modes]
    detail_controls: list[ft.Control] = [
        kv_rows(rows, label_width=190),
        ft.Row(
            [
                shown(cv.device_hash, index, size=theme.SIZE_SMALL),
                shown(cv.seed, index, size=theme.SIZE_SMALL),
                shown(cv.fidelity_level, index, size=theme.SIZE_SMALL),
            ],
            wrap=True,
            spacing=12,
        ),
        ft.Column([status_line(n) for n in cv.notes], spacing=2),
    ]
    body = ft.Column(
        [
            ft.Text(
                f"{', '.join(sorted(set(cv.species)))}, {cv.n_ions} ions",
                size=theme.SIZE_SMALL + 1,
                weight=ft.FontWeight.W_500,
            ),
            stat_row(tiles),
            status_line(f"modes: {modes} MHz"),
            details("level0.device", detail_controls, store=store, level=0, session=session),
        ],
        spacing=theme.GAP,
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
    page = ft.context.page
    record = store.record()
    # the job and its engine already stand in the crumbs and the numerics badge; only a label (a preset, a request) is news
    note = record.job.label if record is not None and record.job.label else ""
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
            # the predict-then-reveal card sits where the histogram will appear (DESIGN.md Section 5)
            centre.append(PredictionCard(store, session))
        elif not store.running():
            centre.append(
                empty_state("No result yet", "The histogram of your run will appear here.", key="empty")
            )
    layout = columns(store, page, 0, centre, right, split=(2, 1))
    return ft.Column([header, layout], spacing=16, expand=True, scroll=ft.ScrollMode.AUTO)


__all__ = [
    "CircuitEditor",
    "DeviceCardView",
    "Level0Page",
    "PredictionCard",
    "ProgressRows",
    "REQUEST_LABELS",
    "ResultsPanel",
    "ShotsPanel",
]
