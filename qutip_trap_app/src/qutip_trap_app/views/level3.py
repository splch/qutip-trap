"""Level 3, the dynamics (PLAN.md Section 14.2 row 3; DESIGN.md Section 5): inside one pulse for one dynamical sample.

The recorded coarse trace (the run's own stored points inside the step) is shown at once; the one primary action re-simulates
the step at fine resolution in the worker and the fine trace replaces it (Section 14.7). Secondary actions compute the Fock
movie (per-time distributions by truncated re-simulation), the process matrix of the finished pulse and the Section 5.5
convergence re-checks, each a background job with progress and cancel. Before the loops are revealed the learner is asked
whether they close (predict-then-reveal, DESIGN.md Section 3); the pick is scored against the record, never against a script.
"""

from __future__ import annotations

import time
from typing import Any

import flet as ft
import numpy as np

from qutip_trap_app import resim
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record, ZoomTrace
from qutip_trap_app.viewmodel.dynamics import (
    PulseDynamics,
    closure_table,
    fock_heatmaps,
    process_view,
    pulse_dynamics,
    recorded_zoom,
)
from qutip_trap_app.viewmodel.learn import CONCEPTS, Attempt, score_closure
from qutip_trap_app.viewmodel.numerics import NumericsPanel, numerics_panel
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import card, data_table, kv_rows, level_header, shown, value_cell
from qutip_trap_app.views.level0 import ProgressRows
from qutip_trap_app.views.state import Session, Store

CLOSURE_PROMPT = next(p for p in CONCEPTS["spin_dependent_force"].prompts if p.kind == "predict_closure")


def step_for_pulse(record: Record, pulse_param: str) -> int:
    try:
        pulse = int(pulse_param)
    except ValueError:
        pulse = 0
    for st in record.schedule.steps:
        if pulse in st.pulse_indices:
            return st.index
    gate_steps = [s for s in record.schedule.steps if s.kind == "gate"]
    return gate_steps[0].index if gate_steps else 0


def selection(store: Store, record: Record, pulse_param: str, sample_param: str) -> tuple[int, int, int]:
    """(step, sample, branch) the route and the store select, clipped to what the record holds."""
    step = step_for_pulse(record, pulse_param)
    try:
        sample = int(sample_param)
    except ValueError:
        sample = 0
    sample = min(max(sample, 0), max(record.n_samples - 1, 0))
    branch = min(max(store.branch, 0), max(record.n_branches - 1, 0))
    return step, sample, branch


def current_zoom(record: Record, step: int, sample: int, branch: int) -> tuple[ZoomTrace | None, bool]:
    """The fine zoom when cached, else the recorded coarse trace; (None, False) when the record stores no trace (a derived
    or GATE_LOCAL run)."""
    key = resim.zoom_key(step, sample, branch, resim.DEFAULT_ZOOM_POINTS, record.job.solver_options())
    z = record.zoom(key)
    if z is not None:
        return z, True
    try:
        return recorded_zoom(record, step, sample, branch), False
    except KeyError:
        return None, False


def level3_numerics(store: Store, record: Record, pulse_param: str, sample_param: str) -> NumericsPanel:
    """The numerics strip of Level 3: the zoomed step's boundary population and the re-checks when they ran."""
    step, sample, branch = selection(store, record, pulse_param, sample_param)
    z, fine = current_zoom(record, step, sample, branch)
    rechecks = store.rechecks.get(f"{record.key()}:{step}:{sample}:{branch}")
    tol = trunc = None
    if rechecks is not None:
        tol, trunc = rechecks
    return numerics_panel(record, zoom=z if fine else None, tolerance_check=tol, truncation_check=trunc)


def _series_chart(series: Any, *, y_title: str, height: float = 170.0, markers: Any = ()) -> ft.Control:
    data = [(s.label, np.asarray(s.times_s) * 1e6, np.asarray(s.values)) for s in series]
    return drawing.line_chart(data, x_title="t (us)", y_title=y_title, height=height, markers=markers)


@ft.component
def ClosurePrediction(store: Store, session: Session, key: str, dyn: PulseDynamics) -> ft.Control:
    ft.use_state(store)
    pick, set_pick = ft.use_state("")
    closes, excursions = closure_table(dyn)

    def commit(choice: str) -> None:
        store.closure_predictions = {**store.closure_predictions, key: choice}
        if choice == "skipped":
            session.record_attempt(
                Attempt("spin_dependent_force", CLOSURE_PROMPT.id, time.time() / 86400.0, None, unaided=False)
            )
            return
        ok = score_closure(choice, closes, excursions)
        session.record_attempt(
            Attempt("spin_dependent_force", CLOSURE_PROMPT.id, time.time() / 86400.0, ok, unaided=False)
        )

    return card(
        "Before the loops are drawn: do they close?",
        ft.Column(
            [
                ft.Text(CLOSURE_PROMPT.question, size=13),
                ft.RadioGroup(
                    content=ft.Column(
                        [ft.Radio(value=o, label=o) for o in CLOSURE_PROMPT.options], spacing=0
                    ),
                    value=pick or None,
                    on_change=lambda e: set_pick(str(e.control.value)),
                ),
                ft.Row(
                    [
                        ft.FilledButton(
                            content=ft.Text("Reveal"), on_click=lambda e: commit(pick), disabled=not pick
                        ),
                        ft.TextButton(content=ft.Text("Skip"), on_click=lambda e: commit("skipped")),
                    ],
                    spacing=6,
                ),
            ],
            spacing=6,
        ),
        subtitle="retrieval before the reveal; the pick is scored against the played waveform's spin-branch loops on the run's own modes (Section 4.4.1), never against a script",
    )


def _closure_feedback(store: Store, key: str, dyn: PulseDynamics) -> ft.Control:
    pick = store.closure_predictions.get(key)
    if not pick or pick == "skipped":
        return ft.Container()
    closes, excursions = closure_table(dyn)
    ok = score_closure(pick, closes, excursions)
    worst = max((closes[m] / max(excursions.get(m, 0.0), 1e-12) for m in closes), default=0.0)
    text = (
        f"your pick '{pick}' matches the record"
        if ok
        else f"your pick was '{pick}'; the played waveform's branch loops end within {worst:.1%} of their largest excursion"
    )
    return ft.Container(
        content=ft.Text(text, size=12),
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
        padding=ft.Padding.all(8),
        border_radius=ft.BorderRadius.all(8),
    )


@ft.component
def Level3Page(
    store: Store,
    session: Session,
    record: Record,
    pulse_param: str,
    sample_param: str,
    index: ProvenanceIndex,
) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    key = record.key()
    step, sample, branch = selection(store, record, pulse_param, sample_param)
    st = record.step(step)
    z, fine = current_zoom(record, step, sample, branch)
    header = level_header(
        3,
        "The dynamics",
        "What happened inside one pulse?",
        f"step {st.gate_id}, sample {sample}, branch {branch}",
    )
    if z is None:
        why = (
            "this run was made by channel replay (a derived engine): it stores no per-time trace inside a pulse"
            if record.replay is not None
            else "this GATE_LOCAL run stores no joint trace inside a pulse (core gap: see the device card's notes)"
        )
        return ft.Column(
            [
                header,
                card(
                    "No trace to open",
                    ft.Column(
                        [
                            ft.Text(why, size=13),
                            ft.Text(
                                "Run the circuit with the engine set to 'full simulation' on Level 0; a JOINT_EXACT record can be zoomed into and re-simulated here.",
                                size=13,
                            ),
                            ft.FilledButton(
                                content=ft.Text("Back to the machine"),
                                icon=ft.Icons.ARROW_BACK,
                                on_click=lambda e: page.navigate(f"/job/{key}"),
                            ),
                        ],
                        spacing=8,
                    ),
                ),
            ],
            spacing=12,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
    dyn = pulse_dynamics(record, z)
    target = {"key": key, "step": step, "sample": sample, "branch": branch}
    zooming = store.running_of("zoom", **target) is not None
    plan = store.learner.plan(3)
    pred_key = f"{key}:{step}"
    predicted = pred_key in store.closure_predictions
    ask = plan.prompt_before_reveal and not predicted and bool(dyn.loops)

    def set_sample(e: Any) -> None:
        page.navigate(f"/job/{key}/dynamics/{pulse_param}/{int(e.control.value)}")

    def set_branch(e: Any) -> None:
        store.branch = int(e.control.value)

    def resimulate(_e: Any) -> None:
        session.submit_zoom(key, step, sample, branch)

    def open_equation(_e: Any) -> None:
        store.hamiltonian_target = (key, step, sample, branch)
        store.selected_channel = None
        page.navigate("/device/hamiltonian")

    def open_channel(channel: str) -> None:
        store.hamiltonian_target = (key, step, sample, branch)
        store.selected_channel = channel
        page.navigate("/device/hamiltonian")

    controls = ft.Row(
        [
            ft.Dropdown(
                label="dynamical sample",
                value=str(sample),
                options=[ft.DropdownOption(key=str(k), text=f"sample {k}") for k in range(record.n_samples)],
                on_select=set_sample,
                width=170,
                dense=True,
            ),
            ft.Dropdown(
                label="initial-mixture branch",
                value=str(branch),
                options=[
                    ft.DropdownOption(
                        key=str(b.index), text=f"branch {b.index} (weight {b.weight:.3g}, Fock {b.fock})"
                    )
                    for b in record.branches
                ],
                on_select=set_branch,
                width=300,
                dense=True,
            ),
            ft.FilledButton(
                content=ft.Text(
                    "Re-simulate this pulse at fine resolution" if not fine else "Re-simulated (cached)"
                ),
                icon=ft.Icons.ZOOM_IN,
                on_click=resimulate,
                disabled=zooming or fine,
                tooltip="one trajectory of this step from its recorded initial state and noise sample, 201 stored points per segment (Section 14.3)",
            ),
            ft.OutlinedButton(
                content=ft.Text("Zoom in: the equation being solved"),
                icon=ft.Icons.FUNCTIONS,
                on_click=open_equation,
            ),
        ],
        wrap=True,
        spacing=10,
    )
    trace_note = (
        f"fine trace: {dyn.times_s.size} points, re-simulated in {dyn.wall_time.value:.2f} s at dimension {z.engine_dimension} ({', '.join(dyn.engine)})"
        if fine
        else f"recorded coarse trace: the {dyn.times_s.size} points the run stored (segment boundaries); re-simulate for the time resolution inside the segments"
    )
    top = card(
        f"Pulse {st.gate_id}",
        ft.Column(
            [
                ft.Text(
                    f"{st.t_start_s * 1e6:.2f} to {st.t_end_s * 1e6:.2f} us on ions {st.ions}; {len(st.pulse_indices)} pulse objects (one per ion and segment)",
                    size=13,
                ),
                controls,
                ft.Text(trace_note, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            spacing=8,
        ),
        subtitle="one dynamical sample of the recorded run (Section 14.1 rule 1: nothing here is an illustration)",
    )
    body: list[ft.Control] = [header, top, ProgressRows(store, session)]
    if ask:
        body.append(ClosurePrediction(store, session, pred_key, dyn))
    # ---- qubits ----
    qubit_series = list(dyn.populations) + list(dyn.coherences)
    qubit_card = card(
        "The qubits",
        ft.Column(
            [
                _series_chart(qubit_series, y_title="P1, |rho_01|"),
                ft.Row([shown(_last(s), index) for s in dyn.populations], wrap=True, spacing=12),
            ]
            + (
                [
                    _series_chart(
                        ([dyn.concurrence] if dyn.concurrence is not None else []) + list(dyn.pauli),
                        y_title="concurrence, <P>",
                    ),
                ]
                if dyn.concurrence is not None
                else []
            ),
            spacing=8,
        ),
        subtitle="populations and coherences from the reduced register state at every stored time",
    )
    # ---- motion ----
    mode_order = sorted({lp.mode for lp in dyn.loops})
    first_ion = min((lp.ion for lp in dyn.loops if lp.ion is not None), default=None)
    loops = [
        drawing.PhaseLoop(
            label=lp.label,
            alpha=lp.alpha,
            note=f"ends {lp.closes:.1e} from its start",
            color_index=mode_order.index(lp.mode),
            dashed=lp.ion != first_ion,
        )
        for lp in dyn.loops
    ]
    closure_rows = _loop_table(dyn, first_ion, index)
    motion_children: list[ft.Control] = [_series_chart(dyn.nbar, y_title="<n_m>")]
    if ask:
        motion_children.append(
            ft.Text(
                "the phase-space loops are hidden until you answer or skip the question above",
                size=12,
                italic=True,
            )
        )
    elif not dyn.loops:
        motion_children += [
            ft.Text(
                "this step plays no entangling waveform: no spin-dependent force, so there is no loop to close",
                size=12,
                italic=True,
            ),
            closure_rows,
        ]
    else:
        motion_children += [_closure_feedback(store, pred_key, dyn), drawing.phase_space(loops), closure_rows]
    motion_card = card(
        "The motion",
        ft.Column(motion_children, spacing=8),
        subtitle=(
            "<n_m>(t) from the exact simulation; the spin-branch loops alpha_im(t) of the played waveform on the run's own "
            "modes (Section 4.4.1): 2 Im of the integral of conj(alpha_a) d alpha_b over the pair's two loops is the "
            "entangling angle chi_m (Section 4.4.3); the spin-averaged <a_m>(t) cancels between the branches and is shown "
            "as a residue"
        ),
    )
    # ---- Fock distributions ----
    movie_key = resim.fock_movie_key(
        step, sample, branch, resim.DEFAULT_FOCK_FRAMES, record.job.solver_options()
    )
    movie = record.fock_movie(movie_key)
    fock_children: list[ft.Control] = []
    for m in sorted(set(dyn.fock_start) | set(dyn.fock_end)):
        start = dyn.fock_start.get(m, np.zeros(1))
        end = dyn.fock_end.get(m, np.zeros(1))
        fock_children.append(
            ft.Column(
                [
                    ft.Text(
                        f"mode {m}: P(n) at the start (outlined) and the end (filled)",
                        size=12,
                        weight=ft.FontWeight.W_600,
                    ),
                    drawing.fock_bars(start, end),
                    ft.Row(
                        [
                            shown(_fock_shown(m, start, "start"), index),
                            shown(_fock_shown(m, end, "end"), index),
                        ],
                        wrap=True,
                        spacing=12,
                    ),
                ],
                spacing=4,
            )
        )
    if not fock_children:
        fock_children.append(
            ft.Text(
                "the Fock distributions at the step's boundaries appear once the pulse is re-simulated (the boundary states are chained then)",
                size=12,
                italic=True,
            )
        )
    moving = store.running_of("fock_movie", **target) is not None
    if movie is not None:
        for hm in fock_heatmaps(movie):
            top_n = hm.largest_populated + 2
            vals = hm.values[:, : top_n + 1].T
            fock_children.append(
                ft.Column(
                    [
                        ft.Text(
                            f"mode {hm.mode}: P(n, t) from {movie.times_s.size - 1} truncated re-simulations",
                            size=12,
                            weight=ft.FontWeight.W_600,
                        ),
                        drawing.heatmap(
                            vals,
                            x_labels=[f"{t * 1e6:.1f}" for t in hm.times_s],
                            y_labels=[str(n) for n in range(top_n + 1)],
                            x_title="t (us)",
                            y_title="n",
                            tooltip="log colour scale over six decades; hover the table below for exact values",
                        ),
                        ft.Row([shown(s, index, label=False) for s in hm.nbar], wrap=True, spacing=8),
                        ft.Text(hm.method, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ],
                    spacing=4,
                )
            )
    else:
        fock_children.append(
            ft.Row(
                [
                    ft.OutlinedButton(
                        content=ft.Text(f"Compute P(n, t): {resim.DEFAULT_FOCK_FRAMES} frames"),
                        icon=ft.Icons.GRID_ON,
                        on_click=lambda e: session.submit_fock_movie(key, step, sample, branch),
                        disabled=moving,
                    ),
                    ft.Text(
                        "the core stores <n>(t) only; the frames re-simulate the pulse cut at each time (causality), about half the frames' worth of full re-simulations",
                        size=12,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                wrap=True,
                spacing=10,
            )
        )
    fock_card = card(
        "Vibration numbers",
        ft.Column(fock_children, spacing=10),
        subtitle="quanta of vibration per resolved mode (Section 5.3)",
    )
    # ---- jumps and the sample ----
    jump_controls: list[ft.Control] = []
    for j in dyn.jumps:
        jump_controls.append(
            ft.TextButton(
                content=ft.Text(f"{float(j.value or 0.0) * 1e6:.2f} us: {j.detail}", size=12),
                icon=ft.Icons.BOLT,
                on_click=lambda e, ch=str(j.detail): open_channel(ch),
                tooltip="open this channel's collapse operator on the Hamiltonian page",
            )
        )
    if not jump_controls:
        jump_controls.append(
            ft.Text(
                "no quantum jump in this sample (a quiet device, or a trajectory without a jump)",
                size=12,
                italic=True,
            )
        )
    sample_rows: list[tuple[str, ft.Control]] = [
        (s.detail, shown(s, index, label=False)) for s in dyn.noise_values
    ] or [("quasi-static values", ft.Text("none drawn: the device is quiet", size=12))]
    dw_rows = [(s.detail.split(":")[0], shown(s, index, label=False)) for s in dyn.debye_waller]
    sample_card = card(
        "Jumps and this sample",
        ft.Column(
            [
                ft.Text("quantum jumps", size=12, weight=ft.FontWeight.W_600),
                ft.Column(jump_controls, spacing=0),
                ft.Text("quasi-static values drawn for this sample", size=12, weight=ft.FontWeight.W_600),
                kv_rows(sample_rows),
            ]
            + (
                [
                    ft.Text("frozen spectators (Debye-Waller)", size=12, weight=ft.FontWeight.W_600),
                    kv_rows(dw_rows),
                ]
                if dw_rows
                else []
            )
            + [
                ft.Row(
                    [shown(dyn.norm_deficit, index), shown(dyn.wall_time, index)]
                    + [shown(b, index) for b in dyn.boundary_population],
                    wrap=True,
                    spacing=12,
                )
            ],
            spacing=6,
        ),
        subtitle="what the environment did in this repetition (Section 6.1)",
    )
    # ---- process matrix ----
    pm_key = resim.process_matrix_key(step, sample, branch)
    pm = record.process_matrix(pm_key)
    tomo_running = store.running_of("tomography", **target) is not None
    if pm is not None:
        pv = process_view(pm)
        n = int(round(np.sqrt(pv.choi_abs.shape[0])))
        labels = [format(k, f"0{max(1, int(np.log2(n)))}b") for k in range(n)]
        pm_children: list[ft.Control] = [
            ft.Row(
                [
                    shown(pv.infidelity, index),
                    shown(pv.entanglement_infidelity, index),
                    shown(pv.depolarizing_rate, index),
                ],
                wrap=True,
                spacing=12,
            ),
            ft.Row(
                [shown(pv.cp_residual, index), shown(pv.tp_residual, index), shown(pv.wall_time, index)],
                wrap=True,
                spacing=12,
            ),
            ft.Text("Pauli twirl (largest terms)", size=12, weight=ft.FontWeight.W_600),
            ft.Row([shown(p, index, label=False) for p in pv.pauli[:6]], wrap=True, spacing=8),
            ft.Text(
                "|Choi| of the simulated channel (left) and of the ideal gate (right), log colour",
                size=12,
                weight=ft.FontWeight.W_600,
            ),
            ft.Row(
                [
                    drawing.heatmap(
                        pv.choi_abs,
                        x_labels=[f"{a}{b}" for a in labels for b in labels],
                        y_labels=[f"{a}{b}" for a in labels for b in labels],
                        x_title="input, output",
                        y_title="",
                        cell_w=12,
                        cell_h=12,
                    ),
                    drawing.heatmap(
                        pv.ideal_choi_abs,
                        x_labels=[f"{a}{b}" for a in labels for b in labels],
                        y_labels=[f"{a}{b}" for a in labels for b in labels],
                        x_title="input, output",
                        y_title="",
                        cell_w=12,
                        cell_h=12,
                    ),
                ],
                wrap=True,
                spacing=16,
            ),
            ft.Text(pv.method, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
        ]
    else:
        pm_children = [
            ft.Row(
                [
                    ft.OutlinedButton(
                        content=ft.Text("Compute the process matrix"),
                        icon=ft.Icons.GRID_4X4,
                        on_click=lambda e: session.submit_tomography(key, step, sample, branch),
                        disabled=tomo_running,
                    ),
                    ft.Text(
                        "every product input of the register through this step from its recorded motional state (16 inputs for two qubits), then least squares and the CP/TP projection; a background job",
                        size=12,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                wrap=True,
                spacing=10,
            )
        ]
    pm_card = card(
        "The process matrix of the finished pulse",
        ft.Column(pm_children, spacing=8),
        subtitle="the gate as a channel (Sections 5.4, 6.8)",
    )
    rechecking = store.running_of("recheck", **target) is not None
    recheck_row = card(
        "Convergence of this zoom",
        ft.Row(
            [
                ft.OutlinedButton(
                    content=ft.Text("Re-check: tolerances x0.1 and caps +2"),
                    icon=ft.Icons.VERIFIED,
                    on_click=lambda e: session.submit_recheck(key, step, sample, branch),
                    disabled=rechecking,
                ),
                ft.Text(
                    "the Section 5.5 arms on this step; the badge in the numerics strip below turns pass or fail",
                    size=12,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            wrap=True,
            spacing=10,
        ),
    )
    body += [
        ft.ResponsiveRow(
            [
                ft.Column([qubit_card, fock_card], col={"xs": 12, "lg": 6}, spacing=10),
                ft.Column(
                    [motion_card, sample_card, pm_card, recheck_row], col={"xs": 12, "lg": 6}, spacing=10
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        )
    ]
    return ft.Column(body, spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)


def _last(series: Any) -> Any:
    from qutip_trap_app.viewmodel.catalogue import Shown

    return Shown(
        series.quantity,
        float(series.values[-1]) if series.values.size else None,
        f"{series.label} at the end",
    )


def _loop_table(dyn: PulseDynamics, first_ion: int | None, index: ProvenanceIndex) -> ft.Control:
    """One row per spin-branch loop (ion, mode) and one per spin-averaged residue; every value carries its source chip."""
    from qutip_trap_app.viewmodel.catalogue import Shown

    rows: list[list[ft.Control | str]] = []
    for lp in dyn.loops:
        booked: ft.Control | str = "-"
        if lp.chi_m_rad is not None:
            booked = value_cell(
                Shown(
                    "chi_m",
                    lp.chi_m_rad,
                    f"mode {lp.mode}: the angle the run books, the table's exact spot check scaled to the requested angle",
                ),
                index,
            )
        rows.append(
            [
                f"ion {lp.ion}, mode {lp.mode}" + (" (dashed)" if lp.ion != first_ion else ""),
                value_cell(_closure_shown(lp), index),
                value_cell(_area_shown(lp), index) if lp.chi_closed_form_rad is not None else "-",
                booked,
            ]
        )
    for lp in dyn.mean_alpha:
        rows.append(
            [
                f"<a_{lp.mode}>(t) spin-averaged",
                value_cell(_mean_alpha_shown(lp), index),
                "-",
                "-",
            ]
        )
    return data_table(("loop", "ends from start", "closed-form angle", "booked angle"), rows)


def _closure_shown(lp: Any) -> Any:
    from qutip_trap_app.viewmodel.catalogue import Shown

    return Shown(
        "branch_closure",
        lp.closes,
        f"ion {lp.ion}, mode {lp.mode}: |alpha(end) - alpha(start)| of the played waveform's spin-branch loop",
    )


def _area_shown(lp: Any) -> Any:
    from qutip_trap_app.viewmodel.catalogue import Shown

    return Shown(
        "chi_closed_form",
        lp.chi_closed_form_rad,
        f"mode {lp.mode}: from the pair's two loops at the played amplitude; the angle the run books is "
        f"{lp.chi_m_rad:+.4f} rad (the table's exact spot check scaled to the requested gate angle), and the gap is the "
        "Debye-Waller and beyond-Lamb-Dicke correction of Section 7.8",
    )


def _mean_alpha_shown(lp: Any) -> Any:
    from qutip_trap_app.viewmodel.catalogue import Shown

    return Shown(
        "mean_alpha_excursion",
        lp.excursion,
        f"mode {lp.mode}: the exact simulation's spin-averaged <a_m>(t); the branches' loops cancel in the average, and "
        "what remains measures the asymmetry between them",
    )


def _fock_shown(mode: int, dist: np.ndarray, when: str) -> Any:
    from qutip_trap_app.viewmodel.catalogue import Shown

    return Shown(
        "fock_population", float(dist[0]) if dist.size else None, f"P(n = 0) of mode {mode} at the {when}"
    )


__all__ = [
    "CLOSURE_PROMPT",
    "ClosurePrediction",
    "Level3Page",
    "current_zoom",
    "level3_numerics",
    "selection",
    "step_for_pulse",
]
