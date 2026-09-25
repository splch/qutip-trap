"""Level 4, the physics (PLAN.md Section 14.4; DESIGN.md Sections 5 and 10): the species, trap, crystal, light, noise,
cooling and readout pages and the Hamiltonian builder, every drawn element computed by the worker into the device layer.
Each page puts its computed drawing first, its numbers as stat tiles, and its tables behind Details; the knobs of the page
sit behind one tile at the bottom while the stale badge and Recalibrate stay visible under the page tabs. A knob change
re-derives the analytic layer at once and marks the calibration table stale; the one primary action while the table is
stale is Recalibrate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import flet as ft
import numpy as np

from qutip_trap_app.device_layer import DeviceLayer
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record, StepRecord, TableRecord
from qutip_trap_app.resim import hamiltonian_key
from qutip_trap_app.viewmodel.catalogue import Row, Shown
from qutip_trap_app.viewmodel.physics import (
    KnobRow,
    cooling_view,
    crystal_view,
    gate_rows,
    hamiltonian_view,
    knob_rows,
    layer_card_view,
    light_view,
    noise_view,
    readout_view,
    species_view,
    stale_status,
    trap_view,
)
from qutip_trap_app.views import drawing, theme
from qutip_trap_app.views.common import (
    MUTED,
    ProgressRows,
    card,
    chip,
    columns,
    data_table,
    details,
    fmt_number,
    hint,
    input_style,
    ions_text,
    kv_rows,
    level_header,
    page_tabs,
    pill,
    section_title,
    shown,
    shown_row,
    stat_row,
    stat_tile,
    status_line,
    tile_row,
    value_cell,
)
from qutip_trap_app.views.state import Session, Store
from qutip_trap_app.views.theme import CODE_FONT, CODE_FONT_FALLBACK

FORMULA = (
    "H/hbar = sum_m omega_m a_m^dag a_m + sum_i (Delta_i/2) sigma_z^i + sum drives (Omega/2) e^{-i(mu t - phi)} sigma_+ "
    "prod_m D_m(i eta) + h.c. + Stark"
)


def _tiles(rows: tuple[Row, ...], index: ProvenanceIndex, *, plain: bool) -> ft.Control:
    """The first six rows as stat tiles (the rest are behind Details)."""
    return stat_row(
        [stat_tile(r.value, index, plain=plain, label=r.label, status=r.status) for r in rows[:6]]
    )


def _rows_table(rows: tuple[Row, ...], index: ProvenanceIndex) -> ft.Control:
    return kv_rows(
        [
            (
                r.label,
                ft.Row(
                    [shown(r.value, index, label=False, size=theme.SIZE_SMALL), status_line(r.status)],
                    spacing=6,
                    wrap=True,
                ),
            )
            for r in rows
        ],
        label_width=200,
    )


def _direction(e_hat: tuple[float, float, float]) -> str:
    """A mode's direction in one short cell: the axis letter when it lies along one, else its components along x, y, z."""
    letters = "xyz"
    k = max(range(3), key=lambda i: abs(e_hat[i]))
    if abs(e_hat[k]) > 0.999:
        return f"{'+' if e_hat[k] >= 0 else '-'}{letters[k]}"
    return " ".join(f"{v:+.2f}{letters[i]}" for i, v in enumerate(e_hat) if abs(v) > 0.005)


def _stale_badge(layer: DeviceLayer) -> ft.Control:
    """The calibration's state as a pill: stale (amber, a clock), no table yet (neutral), or calibrated (a check)."""
    text = stale_status(layer)
    if layer.stale:
        kind, icon = "stale", ft.Icons.HISTORY
    elif layer.table_hash is None:
        kind, icon = "not checked", ft.Icons.HELP_OUTLINE
    else:
        kind, icon = "pass", ft.Icons.CHECK_CIRCLE
    return pill(text, kind, icon=icon, key="stale-badge")


# ---- knobs -----------------------------------------------------------------------------------------------------------------------------


def _slider_to_value(k: KnobRow, pos: float) -> float:
    if k.knob.log:
        lo, hi = math.log10(k.knob.lo), math.log10(k.knob.hi)
        return 10 ** (lo + pos * (hi - lo))
    return k.knob.lo + pos * (k.knob.hi - k.knob.lo)


def _value_to_slider(k: KnobRow, value: float) -> float:
    if k.knob.log:
        lo, hi = math.log10(k.knob.lo), math.log10(k.knob.hi)
        v = math.log10(max(value, k.knob.lo))
        return min(max((v - lo) / (hi - lo), 0.0), 1.0)
    return min(max((value - k.knob.lo) / (k.knob.hi - k.knob.lo), 0.0), 1.0)


@ft.component
def DeviceStatusStrip(store: Store, session: Session, layer: DeviceLayer) -> ft.Control:
    """The stale badge, Recalibrate (the primary action while the table is stale) and Reset, always visible (Section 14.4)."""
    ft.use_state(store)
    recalibrating = store.running_of("recalibrate") is not None
    needs_table = layer.stale or layer.table_hash is None
    recalibrate_cls = ft.FilledButton if layer.stale else ft.FilledTonalButton
    return ft.Row(
        [
            _stale_badge(layer),
            recalibrate_cls(
                content=ft.Text("Recalibrate · about 15 s"),
                icon=ft.Icons.TUNE,
                on_click=lambda e: session.submit_recalibrate(),
                disabled=not needs_table or recalibrating,
                tooltip="the surrogate table for this device: closed forms, exact spot checks, detection records (Section 7.5); the next Run uses it",
                key="recalibrate",
            ),
            ft.TextButton(
                content=ft.Text("Reset every knob"),
                on_click=lambda e: session.reset_knobs(),
                disabled=not layer.overrides,
            ),
            status_line(
                f"{len(layer.overrides)} knob{'s' if len(layer.overrides) != 1 else ''} changed"
                if layer.overrides
                else "the preset as published"
            ),
        ],
        spacing=10,
        wrap=True,
        run_spacing=theme.GAP,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


@ft.component
def KnobPanel(
    store: Store, session: Session, layer: DeviceLayer, page_name: str, index: ProvenanceIndex
) -> ft.Control:
    ft.use_state(store)
    rows = knob_rows(layer, page_name)
    deriving = store.running_of("derive") is not None
    controls: list[ft.Control] = []
    for k in rows:

        def on_end(e: Any, kk: KnobRow = k) -> None:
            session.set_knob(kk.knob.id, _slider_to_value(kk, float(e.control.value)))

        def reset(_e: Any, kk: KnobRow = k) -> None:
            session.set_knob(kk.knob.id, None)

        needs_rf = k.knob.requires_rf and layer.trap.rf_frequency_hz is None
        controls.append(
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(
                                k.knob.label,
                                size=theme.SIZE_SMALL,
                                weight=ft.FontWeight.W_500,
                                tooltip=k.knob.term,
                            ),
                            ft.Row(
                                [
                                    ft.Text(
                                        fmt_number(k.value, k.knob.unit)
                                        if not k.knob.relative
                                        else f"x {k.value:.3g}",
                                        size=theme.SIZE_SMALL + 1,
                                        weight=ft.FontWeight.W_600,
                                        tooltip=f"{k.knob.term}; Section {k.knob.section}",
                                    ),
                                    chip(k.knob.ledger_id, index),
                                    ft.IconButton(
                                        icon=ft.Icons.RESTART_ALT,
                                        tooltip="back to the preset's value",
                                        on_click=reset,
                                        visible=k.is_override,
                                        icon_size=14,
                                        width=24,
                                        height=24,
                                        padding=0,
                                    ),
                                ],
                                spacing=6,
                                tight=True,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        spacing=0,
                        width=260,
                    ),
                    ft.Slider(
                        min=0.0,
                        max=1.0,
                        value=_value_to_slider(k, k.value),
                        on_change_end=on_end,
                        disabled=deriving or needs_rf,
                        tooltip=(
                            "declare the rf drive frequency first: without it the Mathieu a is unknown and q cannot be scaled"
                            if needs_rf
                            else k.knob.doc
                        ),
                        expand=True,
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=theme.GAP,
            )
        )
    if not controls:
        return ft.Container()
    return card(
        "Knobs",
        details(
            store,
            session,
            4,
            f"level4.knobs.{page_name}",
            controls,
            title=f"{len(rows)} slider{'s' if len(rows) != 1 else ''} of this page",
            default_open=bool(layer.overrides) or store.learner.plan(4).expanded,
        ),
        why=lambda e: session.select_concept(4, "calibration"),
        info=[
            "Section 14.4: the device model is the single source of truth; moving a slider re-derives the analytic layer at once and marks the calibration table stale until you recalibrate",
            "hover a slider for what it changes downstream",
        ],
        key="knobs",
    )


# ---- the pages -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Page:
    """What a Level 4 page renders from: the store and session, the device layer, the table it is compared with, the
    provenance index and whether plain labels come first."""

    store: Store
    session: Session
    layer: DeviceLayer
    table: TableRecord | None
    index: ProvenanceIndex
    plain: bool


def _two_columns(
    store: Store, left: list[ft.Control], right: list[ft.Control], split: tuple[int, int] = (1, 1)
) -> ft.Control:
    return columns(store, ft.context.page, 4, left, right, split=split)


def _species_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, index, plain = ctx.store, ctx.session, ctx.layer, ctx.index, ctx.plain
    v = species_view(layer)
    levels = [
        (
            name,
            next((lv.energy_hz for lv in layer.species.levels if lv.name == name), 0.0),
            next((lv.lifetime_s for lv in layer.species.levels if lv.name == name), None),
        )
        for name, _s in v.levels
    ]
    transitions = [(t.label, t.lower, t.upper, t.wavelength_m) for t in layer.species.transitions]
    qubit_level = v.qubit_level
    level_rows: list[list[ft.Control | str]] = [
        [name, *(value_cell(s, index) for s in cells)] for name, cells in v.levels
    ]
    tr_rows: list[list[ft.Control | str]] = [
        [label, *(value_cell(s, index) for s in cells)] for label, cells in v.transitions
    ]
    sub_rows: list[list[ft.Control | str]] = [
        [lvl, label, value_cell(e, index), value_cell(sl, index), value_cell(cu, index)]
        for lvl, label, e, sl, cu in v.sublevels
    ]
    qubit_fan = [(s[1], float(s[2].value or 0.0)) for s in v.sublevels if s[0] == qubit_level]
    clock_rows: list[list[ft.Control | str]] = [
        [value_cell(b, index), value_cell(f, index), value_cell(c, index)] for b, f, c in v.clock_points
    ]
    levels_card = card(
        "The level diagram",
        ft.Column(
            [
                drawing.level_diagram(levels, transitions, qubit_level),
                hint(store, 4, "atomic_structure"),
                details(
                    store,
                    session,
                    4,
                    "level4.species.levels",
                    [
                        _rows_table(v.identity, index),
                        data_table(
                            ["level", "energy", "lifetime", "linewidth", "A_hfs", "B_hfs", "g_J"], level_rows
                        ),
                        data_table(["line", "wavelength", "linewidth", "branching", "I_sat"], tr_rows),
                    ],
                    title="Identity, levels and transitions",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "atomic_structure"),
        info="fine-structure levels from the species table; the vertical spacing is compressed and labelled not to scale",
        key="levels",
    )
    qubit_card = card(
        "The qubit pair",
        ft.Column(
            [
                _tiles(v.qubit, index, plain=plain),
                drawing.sublevel_fan(
                    qubit_fan,
                    title=f"{qubit_level} at {layer.species.field_gauss:g} G (to scale within the level)",
                ),
                drawing.line_chart(
                    [
                        (label, v.sweep_b_gauss, v.sweep_energies_hz[:, j] / 1e6)
                        for j, label in enumerate(v.sweep_labels)
                    ],
                    x_title="B (G)",
                    y_title="E/h (MHz)",
                    height=200,
                ),
                details(
                    store,
                    session,
                    4,
                    "level4.species.sublevels",
                    [
                        data_table(
                            ["level", "sublevel", "energy in the level", "dE/dB", "d2E/dB2"], sub_rows
                        ),
                        data_table(["clock point B0", "|nu| there", "Taylor c2"], clock_rows)
                        if clock_rows
                        else status_line(
                            f"no field-independent point of the qubit pair between {v.clock_scan[0]:g} and {v.clock_scan[1]:g} G"
                        ),
                    ],
                    title="Sublevels and clock points",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "atomic_structure"),
        info="two hyperfine levels chosen so that the field barely moves their splitting (Section 4.5.1); the sublevels from the hyperfine-plus-Zeeman Hamiltonian diagonalized at the field, the slopes by Hellmann-Feynman",
        key="qubit-pair",
    )
    return [_two_columns(store, [levels_card], [qubit_card])]


def _trap_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, index, plain = ctx.store, ctx.session, ctx.layer, ctx.index, ctx.plain
    v = trap_view(layer)
    mathieu_rows: list[list[ft.Control | str]] = [
        [ax, *(value_cell(s, index) for s in cells)] for ax, *cells in v.mathieu
    ]
    stability: ft.Control
    if v.stability_q is not None and v.stability_lower is not None and v.stability_upper is not None:
        stability = drawing.stability_diagram(
            v.stability_q, v.stability_lower, v.stability_upper, v.operating_points
        )
    else:
        stability = status_line("the stability map is being computed")
    notes = list(v.notes)
    if not v.operating_points:
        notes.append(
            "no operating point yet: declare the rf drive frequency (knob) to place this trap on the diagram"
        )
    stability_card = card(
        "Stability diagram",
        ft.Column(
            [
                stability,
                _tiles(v.secular, index, plain=plain),
                hint(store, 4, "trap_and_mathieu"),
                details(
                    store,
                    session,
                    4,
                    "level4.trap.mathieu",
                    [
                        status_line(v.mathieu_note),
                        data_table(["axis", "a", "q", "beta", "nu = beta Omega/2", "C0"], mathieu_rows)
                        if mathieu_rows
                        else ft.Container(),
                        _rows_table(v.rf, index),
                    ]
                    + ([shown(v.stability_edge, index)] if v.stability_edge is not None else []),
                    title="Mathieu parameters and the rf record",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "trap_and_mathieu"),
        info=[
            "the first stability region of x'' + (a - 2q cos 2xi) x = 0 with this device's operating point per radial axis (the axial axis has q_z = 0); the boundary is the monodromy bisected per q (Section 4.1.1)",
            f"trap path: {v.path}",
        ]
        + notes,
        key="stability",
    )
    fields_card = card(
        "Stray field and micromotion",
        ft.Column(
            [
                _tiles(v.fields, index, plain=plain),
                details(store, session, 4, "level4.trap.fields", [_rows_table(v.fields, index)]),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "trap_and_mathieu"),
        info="a stray field parks the ion off the rf null, where it is dragged at the rf frequency (Berkeland, Section 4.1.1)",
        key="micromotion",
    )
    return [_two_columns(store, [stability_card], [fields_card], split=(7, 5))]


@ft.component
def CrystalPage(store: Store, session: Session, layer: DeviceLayer, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    v = crystal_view(layer)
    plain = store.learner.plan(4).plain_labels_first
    mode_sel, set_mode = ft.use_state(0)
    m = v.modes[min(mode_sel, len(v.modes) - 1)] if v.modes else None
    picture = drawing.crystal_picture(
        v.positions_m,
        None if m is None else m.eigenvector,
        None if m is None else m.e_hat,
        species=v.species,
        title=""
        if m is None
        else f"mode {m.index}: {m.family} {m.family_index}, {m.frequency.value / 1e6:.4f} MHz"
        if isinstance(m.frequency.value, float)
        else "",
    )
    mode_rows: list[list[ft.Control | str]] = [
        [
            str(mr.index),
            mr.family,
            str(mr.family_index),
            value_cell(mr.frequency, index),
            _direction(mr.e_hat),
            value_cell(mr.uniform_weight, index),
            "-" if mr.heating is None else value_cell(mr.heating, index),
        ]
        for mr in v.modes
    ]
    eta_rows: list[list[ft.Control | str]] = [
        [label, *(value_cell(s, index) for s in cells)] for label, cells in v.eta_rows
    ]
    eta_cols = ["ion"] + [f"mode {k}" for k in range(len(v.eta_rows[0][1]))] if v.eta_rows else ["ion"]
    selector = ft.Dropdown(
        label="mode to draw",
        value=str(m.index if m is not None else 0),
        options=[
            ft.DropdownOption(key=str(mr.index), text=f"mode {mr.index}: {mr.family} {mr.family_index}")
            for mr in v.modes
        ],
        on_select=lambda e: set_mode(int(e.control.value)),
        width=260,
        **input_style(),
    )
    mode_tiles = [
        stat_tile(mr.frequency, index, plain=plain, label=f"mode {mr.index}: {mr.family} {mr.family_index}")
        for mr in v.modes[:6]
    ]
    return card(
        "Positions and modes",
        ft.Column(
            [
                picture,
                ft.Row(
                    [selector]
                    + ([shown(c, index, label=False) for c in m.components] if m is not None else []),
                    wrap=True,
                    spacing=theme.GAP,
                    run_spacing=theme.GAP,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                stat_row(mode_tiles),
                hint(store, 4, "mode"),
                details(
                    store,
                    session,
                    4,
                    "level4.crystal.modes",
                    [
                        _rows_table(v.geometry + v.zigzag, index),
                        data_table(
                            ["mode", "family", "#", "frequency", "e_hat", "uniform-field weight", "heating"],
                            mode_rows,
                        ),
                        ft.Row(
                            ([shown(v.delta_k, index)] if v.delta_k is not None else [])
                            + [
                                status_line(
                                    "C0 applied (rf record present)"
                                    if v.c0_applied
                                    else "C0 = 1: no rf record on this trap"
                                )
                            ],
                            wrap=True,
                            spacing=12,
                        ),
                        data_table(eta_cols, eta_rows)
                        if eta_rows
                        else status_line("no entangling Raman pair on this device"),
                    ],
                    title="Geometry, mode table and Lamb-Dicke parameters",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "mode"),
        info=[
            "equilibrium positions from the Coulomb-plus-trap potential; modes from its Hessian (Sections 4.1.2, 4.1.3); eta_{i,m} = (Delta k . e_m) c_{i,m} sqrt(hbar/(2 m_i omega_m)) for the entangling pair's Delta k (conv.lamb_dicke)"
        ]
        + list(v.notes),
        key="crystal",
    )


def _crystal_page(ctx: Page) -> list[ft.Control]:
    return [CrystalPage(ctx.store, ctx.session, ctx.layer, ctx.index)]


ROLE_COLORS = {
    "detection": ft.Colors.TERTIARY,
    "entangling gates": ft.Colors.PRIMARY,
    "Doppler cooling": ft.Colors.SECONDARY,
    "optical pumping": ft.Colors.SECONDARY,
}


def _field_direction(layer: DeviceLayer) -> tuple[float, float, float]:
    d = layer.card.field_direction
    return (float(d[0]), float(d[1]), float(d[2]))


def _light_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, index, plain = ctx.store, ctx.session, ctx.layer, ctx.index, ctx.plain
    v = light_view(layer)
    beams_geo = []
    for b in v.beams:
        role = b.roles[0] if b.roles else "unused"
        color: ft.ColorValue = ROLE_COLORS.get(role, ft.Colors.ON_SURFACE_VARIANT)
        if role.startswith("single-qubit"):
            color = theme.series()[1]
        beams_geo.append((b.index, b.k_hat, b.pointing_m, color, f"beam {b.index}: {role}"))
    geometry = drawing.beam_geometry(layer.crystal.positions_m, beams_geo, _field_direction(layer))
    beam_rows: list[list[ft.Control | str]] = [
        [
            str(b.index),
            ", ".join(b.roles) or "unused",
            value_cell(b.wavelength, index),
            value_cell(b.power, index),
            value_cell(b.waist, index),
            value_cell(b.direction, index),
            value_cell(b.angle_to_field, index),
            value_cell(b.intensity_at_ions[0], index) if b.intensity_at_ions else "-",
        ]
        for b in v.beams
    ]
    drive_rows: list[list[ft.Control | str]] = [
        [
            f"ion {d.ion}",
            d.role,
            str(d.beams),
            value_cell(d.rabi, index),
            value_cell(d.pi_time, index),
            value_cell(d.stark, index),
            value_cell(d.residual_excited, index),
            value_cell(d.error_per_pi, index),
            value_cell(d.dephasing, index),
        ]
        for d in v.drives
    ]
    drive_tiles: list[ft.Control] = []
    for d in v.drives[:3]:
        drive_tiles.append(
            stat_tile(d.rabi, index, plain=plain, label=f"ion {d.ion}, {d.role}: Rabi frequency")
        )
        drive_tiles.append(
            stat_tile(
                d.error_per_pi, index, plain=plain, label=f"ion {d.ion}, {d.role}: scatter per pi pulse"
            )
        )
    curve: ft.Control = status_line("no Raman pair to sweep")
    if v.curve_wavelength_m is not None and v.curve_error is not None and v.curve_rabi_hz is not None:
        markers = []
        if v.current_wavelength_m is not None:
            markers.append(
                (v.current_wavelength_m * 1e9, f"this device: {v.current_wavelength_m * 1e9:.1f} nm")
            )
        if v.curve_p12_m is not None:
            markers.append((v.curve_p12_m * 1e9, f"P1/2 line at {v.curve_p12_m * 1e9:.1f} nm"))
        if v.curve_p32_m is not None:
            markers.append((v.curve_p32_m * 1e9, f"P3/2 line at {v.curve_p32_m * 1e9:.1f} nm"))
        curve = ft.Column(
            [
                drawing.line_chart(
                    [("scattering error per pi pulse", v.curve_wavelength_m * 1e9, v.curve_error)],
                    x_title="wavelength (nm)",
                    y_title="P per pi pulse",
                    log_y=True,
                    markers=markers,
                ),
                drawing.line_chart(
                    [("two-photon Rabi frequency", v.curve_wavelength_m * 1e9, v.curve_rabi_hz)],
                    x_title="wavelength (nm)",
                    y_title="Omega/2pi (Hz)",
                    log_y=True,
                    height=150,
                ),
            ],
            spacing=theme.GAP,
        )
    beams_card = card(
        "Beam geometry",
        ft.Column(
            [
                geometry,
                stat_row(drive_tiles),
                hint(store, 4, "light_coupling"),
                details(
                    store,
                    session,
                    4,
                    "level4.light.tables",
                    [
                        data_table(
                            [
                                "beam",
                                "role",
                                "wavelength",
                                "power",
                                "waist",
                                "k_hat",
                                "angle to B",
                                "I at ion 0",
                            ],
                            beam_rows,
                        ),
                        data_table(
                            [
                                "ion",
                                "role",
                                "beams",
                                "Rabi frequency",
                                "pi time",
                                "light shift",
                                "residual excited",
                                "scatter per pi",
                                "Rayleigh dephasing",
                            ],
                            drive_rows,
                        ),
                        shown_row(v.crosstalk, index, empty="no addressing beams"),
                    ],
                    title="Beams, drives and crosstalk",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "light_coupling"),
        info=[
            "beams as fields with their k-vectors through the ions; the quantization axis is the magnetic field (Section 4.5.3); the two-photon Rabi frequency, the differential light shift and the scattering budget from the level structure (Sections 4.5.4, 4.5.5); crosstalk is the Rabi ratio of an addressing pair's light on the neighbour (conv.crosstalk_ratio)"
        ]
        + list(v.notes),
        key="beams",
    )
    scattering_card = card(
        "Scattering against detuning",
        curve,
        why=lambda e: session.select_concept(4, "light_coupling"),
        info="the same Raman pair swept in wavelength between the fine-structure lines: the coupling and the scattering both fall, the error per pulse least far from the lines (Ozeri, Section 4.3.2)",
        key="scattering",
    )
    return [_two_columns(store, [beams_card], [scattering_card])]


def _noise_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, index, plain = ctx.store, ctx.session, ctx.layer, ctx.index, ctx.plain
    v = noise_view(layer)
    spectra: list[ft.Control] = []
    zero_names: list[str] = []
    for sp in v.spectra:
        if sp.is_zero:
            zero_names.append(sp.name)
            continue
        spectra.append(
            ft.Column(
                [
                    ft.Row(
                        [section_title(f"{sp.name} ({sp.unit})"), shown(sp.level, index)],
                        wrap=True,
                        spacing=theme.GAP,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    drawing.line_chart(
                        [(sp.name, sp.omega_rad_s / (2 * np.pi), sp.S)],
                        x_title="f (Hz)",
                        y_title=sp.unit,
                        log_x=True,
                        log_y=True,
                        height=150,
                    ),
                    status_line("apparatus: " + (", ".join(sp.apparatus) or "undeclared")),
                ],
                spacing=6,
            )
        )
    if zero_names:
        spectra.append(status_line("zero everywhere: " + ", ".join(zero_names)))
    drift_rows: list[list[ft.Control | str]] = [
        [
            name,
            value_cell(rms, index),
            value_cell(tau, index),
            value_cell(servo, index) if servo is not None else "none",
        ]
        for name, rms, tau, servo in v.drifts
    ]
    rate_tiles: list[ft.Control] = [
        stat_tile(h, index, plain=plain)
        for h in (list(v.heating) + list(v.motional_dephasing) + list(v.qubit_dephasing))[:6]
    ]
    rates: list[ft.Control] = (
        [stat_row(rate_tiles)]
        if rate_tiles
        else [status_line("every rate is zero: raise the field-noise density knob to see heating")]
    )
    spectra_card = card(
        "Spectra and the rates they imply",
        ft.Column(
            spectra
            + rates
            + [hint(store, 4, "noise_as_physics")]
            + (
                [
                    details(
                        store,
                        session,
                        4,
                        "level4.noise.correlation",
                        [_rows_table((v.correlation,), index)],
                        title="Correlation length",
                    )
                ]
                if v.correlation is not None
                else []
            ),
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "noise_as_physics"),
        info="two-sided densities as the noise model stores them; a heating rate is e^2 S_E/(4 m hbar omega) (Section 4.1.5)",
        key="spectra",
    )
    drifts_card = card(
        "Slow drifts, mains and collisions",
        ft.Column(
            [
                data_table(["parameter", "rms", "correlation time", "servo"], drift_rows),
                shown_row(v.mains, index, empty="no mains harmonics declared"),
                shown_row(v.collisions, index, empty="no background-gas collisions declared"),
                status_line(v.provenance_sentence or "no apparatus named: every rate is zero"),
            ],
            spacing=theme.GAP,
        ),
        why=lambda e: session.select_concept(4, "quantum_jumps"),
        info=[
            "each drift is drawn once per dynamical sample (Section 6.1 route (c)); a servo high-passes it",
            f"undeclared rates: {v.undeclared_rates}; apparatus: {', '.join(v.apparatus) or 'none'}",
        ]
        + list(v.notes),
        key="drifts",
    )
    return [_two_columns(store, [spectra_card], [drifts_card])]


def _cooling_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, index, plain = ctx.store, ctx.session, ctx.layer, ctx.index, ctx.plain
    v = cooling_view(layer)
    if v is None:
        return [
            card(
                "Cooling unavailable",
                status_line(layer.cooling_error or "the recipe could not run on this device"),
            )
        ]
    stage_rows: list[list[ft.Control | str]] = [
        [kind, prov, shown_row(nbars, index, label=False, spacing=8)] for kind, prov, nbars in v.stages
    ]
    doppler_rows: list[list[ft.Control | str]] = [
        [
            str(d.mode),
            value_cell(d.frequency, index),
            value_cell(d.heating, index),
            value_cell(d.cooling, index),
            value_cell(d.rate, index),
            value_cell(d.nbar, index),
            value_cell(d.participation, index),
        ]
        for d in v.doppler
    ]
    sideband_children: list[ft.Control] = []
    for sb in v.sidebands:
        sideband_children.append(
            ft.Column(
                [
                    ft.Row(
                        [
                            section_title(f"mode {sb.mode}, cooled through ion {sb.ion}"),
                            shown(sb.eta, index),
                            shown(sb.omega0, index),
                        ],
                        wrap=True,
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    drawing.line_chart(
                        [(f"<n> of mode {sb.mode} after each pulse", sb.times_s * 1e6, sb.nbar_after_pulse)],
                        x_title="t (us)",
                        y_title="<n>",
                        log_y=True,
                        height=160,
                    ),
                    tile_row((sb.nbar_start, sb.nbar_end, sb.nbar_final_run), index, plain=plain),
                    status_line(
                        f"{len(sb.orders)} pulses; orders {sorted(set(sb.orders), reverse=True)}, higher first; "
                        + ", ".join(
                            f"k={k}: {t * 1e6:.1f} us"
                            for k, t in dict(zip(sb.orders, sb.durations_s)).items()
                        )
                    ),
                ],
                spacing=6,
            )
        )
    if not sideband_children:
        sideband_children.append(status_line("no sideband cooling in this recipe"))
    pump_children: list[ft.Control] = []
    for p in v.pumps:
        series = [(label, p.trace_times_s * 1e6, arr) for label, arr in sorted(p.trace_populations.items())][
            :6
        ]
        pump_children.append(
            ft.Column(
                [
                    section_title(f"ion {p.ion}"),
                    drawing.line_chart(series, x_title="t (us)", y_title="population", height=150)
                    if series
                    else ft.Container(),
                    stat_row(
                        [
                            stat_tile(p.preparation_error, index, plain=plain),
                            stat_tile(p.photons, index, plain=plain),
                        ]
                        + (
                            [stat_tile(p.time_to_reach, index, plain=plain)]
                            if p.time_to_reach is not None
                            else []
                        )
                    ),
                    shown_row(p.heating, index, label=False, spacing=8),
                ],
                spacing=6,
            )
        )
    sideband_card = card(
        "Sideband cooling, pulse by pulse",
        ft.Column(
            sideband_children
            + [
                hint(store, 4, "cooling_ladder"),
                details(
                    store,
                    session,
                    4,
                    "level4.cooling.stages",
                    [
                        _rows_table(v.recipe, index),
                        data_table(["stage", "provenance", "nbar per mode"], stage_rows),
                        data_table(
                            [
                                "mode",
                                "frequency",
                                "A+ heating",
                                "A- cooling",
                                "rate",
                                "nbar",
                                "participation",
                            ],
                            doppler_rows,
                        ),
                        ft.Column([status_line(n) for n in layer.cooling.recipe_notes], spacing=2)
                        if layer.cooling is not None
                        else ft.Container(),
                    ],
                    title=f"The recipe, the stages and the Doppler stage ({v.doppler_method} rate equations)",
                ),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "cooling_ladder"),
        info=[
            "Doppler, then sideband, then the pump last: a pump done first would be erased (Section 4.2.6); p -> W_k(t) p per pulse with the exact Omega_{n,n-k}, drawn without the repump recoil (a core gap), whose effect is the gap to the run's own value; A_pm = W(Delta -/+ nu) + (eta~/eta)^2 W(Delta), one detuning shared by every mode (Sections 4.2.1, 4.2.2)"
        ]
        + list(v.approximations)
        + list(v.notes),
        key="sideband",
    )
    pump_card = card(
        "Optical pumping",
        ft.Column(
            pump_children
            + [
                stat_row(
                    [stat_tile(s, index, plain=plain) for s in v.final_nbar[:4]]
                    + [stat_tile(v.duration, index, plain=plain)]
                ),
                status_line("provenance: " + ", ".join(v.provenance)),
            ],
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "cooling_ladder"),
        info="the multi-level master equation of the pump beams (Section 4.2.8); the residual population outside |0> is the preparation error; the last tiles are what the recipe hands to the circuit",
        key="pumping",
    )
    return [_two_columns(store, [sideband_card], [pump_card])]


def _readout_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, table = ctx.store, ctx.session, ctx.layer, ctx.table
    index, plain = ctx.index, ctx.plain
    v = readout_view(layer, table)
    ion_cards: list[ft.Control] = []
    for i in v.ions:
        sat_curve: ft.Control = ft.Container()
        if (
            i.saturation_s is not None
            and i.saturation_r_o is not None
            and i.saturation_r_d is not None
            and i.saturation_r_b is not None
        ):
            markers = [(0.0, f"ceiling {(i.saturation_ceiling_per_s or 0.0):.3g}/s")]
            if i.saturation_now is not None and isinstance(i.saturation_now.value, float):
                markers.append((i.saturation_now.value, f"this device: s = {i.saturation_now.value:.3g}"))
            sat_curve = drawing.line_chart(
                [
                    ("R_o bright scattering", i.saturation_s, i.saturation_r_o),
                    ("R_d bright pumped dark", i.saturation_s, i.saturation_r_d),
                    ("R_b dark pumped bright", i.saturation_s, i.saturation_r_b),
                ],
                x_title="s = I/I_sat",
                y_title="rate (1/s)",
                log_x=True,
                log_y=True,
                markers=markers,
            )
        budget_rows: list[list[ft.Control | str]] = [
            [name, value_cell(eb, index), value_cell(ed, index)] for name, eb, ed in i.budget
        ]
        ion_cards.append(
            card(
                f"Ion {i.ion}",
                ft.Column(
                    [
                        drawing.two_histograms(
                            i.bright_pmf,
                            i.dark_pmf,
                            float(i.threshold_at_window.value)
                            if isinstance(i.threshold_at_window.value, float)
                            else None,
                        ),
                        tile_row(
                            (i.window, i.threshold_at_window, i.eps_at_window[0], i.eps_at_window[1]),
                            index,
                            plain=plain,
                        ),
                        drawing.line_chart(
                            [
                                ("(eps_B + eps_D)/2", i.scan_windows_s * 1e6, i.scan_eps),
                                ("eps_B", i.scan_windows_s * 1e6, i.scan_eps_b),
                                ("eps_D", i.scan_windows_s * 1e6, i.scan_eps_d),
                            ],
                            x_title="window (us)",
                            y_title="error",
                            log_y=True,
                            height=150,
                        ),
                        hint(store, 4, "readout_rates"),
                        details(
                            store,
                            session,
                            4,
                            f"level4.readout.{i.ion}",
                            [
                                _rows_table(i.rates, index),
                                shown_row(i.best, index),
                                data_table(
                                    ["channel", "bright read as dark", "dark read as bright"], budget_rows
                                ),
                                sat_curve,
                            ],
                            title="Rates, the best window, the error budget, the saturation curves",
                        ),
                    ],
                    spacing=12,
                ),
                why=lambda e: session.select_concept(4, "readout_rates"),
                info="count histograms at the device window, then the error against the window length (best threshold at each); R_o saturates at the manifold's ceiling while R_d and R_b never do, so the light level stays near saturation (Section 8.1)",
                key=f"readout-ion-{i.ion}",
            )
        )
    detector = card(
        "Detector",
        ft.Column(
            [_tiles(v.detector, index, plain=plain)]
            + ([_tiles(v.table, index, plain=plain)] if v.table else []),
            spacing=12,
        ),
        why=lambda e: session.select_concept(4, "readout_rates"),
        info=["the apparatus, and the table's calibrated threshold and window where a table exists"]
        + list(v.notes),
        key="detector",
    )
    pairs = [
        _two_columns(store, ion_cards[k : k + 1], ion_cards[k + 1 : k + 2])
        for k in range(0, len(ion_cards), 2)
    ]
    return pairs + [detector]


def _gates_card(
    store: Store,
    session: Session,
    layer: DeviceLayer,
    table: TableRecord | None,
    index: ProvenanceIndex,
    plain: bool,
) -> ft.Control:
    rows = gate_rows(layer, table)
    if not rows:
        return card(
            "Entangling pulse solutions", status_line("no entangling Raman pair on this device"), key="gates"
        )
    children: list[ft.Control] = []
    for g in rows:
        mode_rows: list[list[ft.Control | str]] = [
            [str(m), value_cell(w, index), *(value_cell(e, index) for e in etas)] for m, w, etas in g.modes
        ]
        children.append(
            ft.Column(
                [
                    section_title(f"pair: {ions_text(g.pair)}"),
                    ft.Text(g.error, size=theme.SIZE_SMALL, color=ft.Colors.ERROR)
                    if g.error
                    else ft.Container(),
                    _tiles(g.summary, index, plain=plain),
                    details(
                        store,
                        session,
                        4,
                        f"level4.gates.{g.pair}",
                        [
                            _rows_table(g.summary, index),
                            shown_row(g.chi_m + g.alpha_m, index),
                            data_table(["mode", "frequency"] + [f"eta ion {i}" for i in g.pair], mode_rows)
                            if mode_rows
                            else ft.Container(),
                            _rows_table(g.table, index)
                            if g.table
                            else status_line("no table entry for this pair yet"),
                        ],
                        title="Per-mode angles, the table's waveform and its closed form",
                    ),
                ],
                spacing=theme.GAP,
            )
        )
    return card(
        "Entangling pulse solutions",
        ft.Column(children, spacing=16),
        why=lambda e: session.select_concept(4, "calibration"),
        info="re-solved at once from the current modes and eta (Sections 4.4.3, 14.4); the table's calibrated waveform beside it, stale when the device changed; the closed form beside the exact spot check is the Debye-Waller gap of Section 7.8 as a number",
        key="gates",
    )


def _hamiltonian_page(ctx: Page) -> list[ft.Control]:
    store, session, layer, table = ctx.store, ctx.session, ctx.layer, ctx.table
    index, plain = ctx.index, ctx.plain
    page = ft.context.page
    record: Record | None = None
    ham = None
    want: tuple[int, int, int] | None = None
    """(step, sample, branch) Level 3 asked for; None when the page was opened on its own."""
    target = store.hamiltonian_target
    if target is not None and target[0] in store.records:
        record = store.records[target[0]]
        want = (int(target[1]), int(target[2]), int(target[3]))
        ham = record.hamiltonian(hamiltonian_key(*want))
    else:
        # nothing asked for: the current record's equation, the last one built when several are
        record = store.record()
        if record is not None and record.hamiltonians:
            ham = record.hamiltonians[-1]
    out: list[ft.Control] = []
    if record is None:
        out.append(
            card(
                "No pulse yet",
                status_line("run a job, then zoom into a pulse on Level 3"),
                key="no-pulse",
            )
        )
    elif ham is None:
        # the step Level 3 asked for when there is one, else the first entangling step (the first gate step failing that)
        steps = [s for s in record.schedule.steps if s.kind == "gate"]
        if want is not None and 0 <= want[0] < len(record.schedule.steps):
            ent: StepRecord | None = record.schedule.steps[want[0]]
            sample_i, branch_i = want[1], want[2]
        else:
            ent = next(
                (s for s in steps if any(g.gate_id in s.target_ids for g in record.schedule.gates)),
                steps[0] if steps else None,
            )
            sample_i, branch_i = 0, store.branch
        building = store.running_of("zoom", key=record.key()) is not None

        def build(_e: Any) -> None:
            if ent is not None:
                store.hamiltonian_target = (record.key(), ent.index, sample_i, branch_i)
                session.submit_zoom(record.key(), ent.index, sample_i, branch_i)

        out.append(
            card(
                "Build the equation of a pulse",
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.FilledButton(
                                    content=ft.Text(
                                        f"Build for {ent.gate_id}" if ent is not None else "Build"
                                    ),
                                    icon=ft.Icons.FUNCTIONS,
                                    on_click=build,
                                    disabled=ent is None or building or record.replay is not None,
                                    key="build-hamiltonian",
                                ),
                                status_line(
                                    "a derived (channel replay) run has no joint space to build on"
                                    if record.replay is not None
                                    else "the same worker call as a Level 3 re-simulation"
                                ),
                            ],
                            wrap=True,
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ProgressRows(store, session),
                    ],
                    spacing=theme.GAP,
                ),
                why=lambda e: session.select_concept(4, "hamiltonian"),
                key="build",
            )
        )
    else:
        v = hamiltonian_view(record, ham)
        free_rows: list[list[ft.Control | str]] = [
            [str(m), cls, value_cell(w, index), value_cell(off, index)] for m, cls, w, off in v.free_modes
        ]
        term_tiles: list[ft.Control] = []
        for t in v.terms:
            tone_rows: list[list[ft.Control | str]] = [
                [str(j), *(value_cell(s, index) for s in cells)] for j, cells in enumerate(t.tones)
            ]
            matrices: list[ft.Control] = []
            for m, table_m in sorted(t.matrix_elements.items()):
                d = table_m.shape[0]
                n_show = min(d, 8)
                matrices.append(
                    ft.Column(
                        [
                            section_title(
                                f"mode {m}: Omega_(n',n)/Omega = |<n'|D(i eta)|n>| (first {n_show} of {d} levels)"
                            ),
                            ft.Row(
                                [
                                    drawing.heatmap(
                                        table_m[:n_show, :n_show],
                                        x_labels=[str(n) for n in range(n_show)],
                                        y_labels=[str(n) for n in range(n_show)],
                                        x_title="n",
                                        y_title="n'",
                                        cell_w=22,
                                        cell_h=16,
                                    ),
                                    data_table(
                                        ["n' \\ n"] + [str(n) for n in range(min(n_show, 6))],
                                        [
                                            [str(r)] + [f"{table_m[r, c]:.3e}" for c in range(min(n_show, 6))]
                                            for r in range(min(n_show, 6))
                                        ],
                                    ),
                                ],
                                wrap=True,
                                spacing=16,
                                run_spacing=12,
                                vertical_alignment=ft.CrossAxisAlignment.START,
                            ),
                            stat_row(
                                [
                                    stat_tile(
                                        _me(table_m, r, c, m),
                                        index,
                                        plain=plain,
                                        label=f"n' = {r}, n = {c} ({name})",
                                    )
                                    for r, c, name in (
                                        (0, 0, "carrier"),
                                        (1, 0, "blue sideband"),
                                        (0, 1, "red sideband"),
                                    )
                                ]
                            ),
                        ],
                        spacing=6,
                        key=f"matrix-elements:{t.index}:{m}",
                    )
                )
            highlighted = store.selected_term == t.index
            term_tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(
                        f"drive term {t.index}: {'crosstalk onto' if t.is_crosstalk else 'addressed'} ion {t.ion} ({t.pulse})",
                        size=theme.SIZE_SMALL + 1,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.PRIMARY if highlighted else None,
                    ),
                    subtitle=ft.Text(t.operator_text, size=theme.SIZE_CAPTION, color=MUTED),
                    controls=[
                        ft.Container(
                            content=ft.Column(
                                [
                                    tile_row(
                                        (
                                            t.omega,
                                            t.crosstalk_weight,
                                            t.rabi_scale,
                                            t.carrier_factor,
                                            t.debye_waller,
                                            t.nnz,
                                        ),
                                        index,
                                        plain=plain,
                                    ),
                                    data_table(
                                        ["tone", "detuning mu", "phase at start", "peak Omega/2pi"], tone_rows
                                    )
                                    if tone_rows
                                    else ft.Container(),
                                    shown_row(t.etas, index),
                                    shown_row(t.frozen, index),
                                    status_line(
                                        "coefficient (hbar Omega(t)/2) e^{-i(mu t - phi(t))}, summed over tones; time-independent operator"
                                    ),
                                ]
                                + matrices,
                                spacing=12,
                            ),
                            padding=ft.Padding.symmetric(horizontal=theme.GAP, vertical=4),
                        )
                    ],
                    dense=True,
                    expanded=t.index == 0 or highlighted,
                    key=f"drive-term:{t.index}",
                )
            )
        groups: dict[str, list[Any]] = {}
        for c in v.collapse:
            groups.setdefault(c.channel.split("[")[0], []).append(c)
        collapse_tiles: list[ft.Control] = []
        for name, items in groups.items():
            active = any(c.active for c in items)
            hit = store.selected_channel is not None and any(
                store.selected_channel in c.channel or c.channel in store.selected_channel for c in items
            )
            rows: list[list[ft.Control | str]] = [
                [
                    c.channel,
                    value_cell(c.rate, index),
                    "-" if c.ion is None else str(c.ion),
                    "-" if c.mode is None else str(c.mode),
                    "yes" if c.active else "no",
                    c.operator_text,
                ]
                for c in items[:40]
            ]
            collapse_tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(
                        f"{name}: {len(items)} operator(s), {'integrated in this run' if active else 'listed, not integrated'}",
                        size=theme.SIZE_SMALL + 1,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.PRIMARY if hit else None,
                    ),
                    subtitle=ft.Text(items[0].note, size=theme.SIZE_CAPTION, color=MUTED),
                    controls=[
                        ft.Container(
                            content=data_table(["channel", "rate", "ion", "mode", "active", "L_k"], rows),
                            padding=ft.Padding.symmetric(horizontal=theme.GAP, vertical=4),
                        )
                    ],
                    dense=True,
                    expanded=hit,
                )
            )
        seg_rows: list[list[ft.Control | str]] = [
            [value_cell(dur, index), value_cell(wmax, index), pulses, str(n), value_cell(kern, index)]
            for dur, wmax, pulses, n, kern in v.segments
        ]
        out += [
            card(
                f"H(t) of {v.gate_id}",
                ft.Column(
                    [
                        ft.Container(
                            content=ft.Text(
                                FORMULA,
                                size=theme.SIZE_SMALL + 1,
                                selectable=True,
                                font_family=CODE_FONT,
                                font_family_fallback=CODE_FONT_FALLBACK,
                            ),
                            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            border_radius=ft.BorderRadius.all(theme.RADIUS_TILE),
                        ),
                        _tiles(v.header, index, plain=plain),
                        hint(store, 4, "hamiltonian"),
                        details(
                            store,
                            session,
                            4,
                            "level4.hamiltonian.free",
                            [
                                data_table(["mode", "class", "omega_m/2pi", "sample offset"], free_rows),
                                shown_row(v.qubit_offsets + v.stark, index),
                                shown_row(v.caps, index),
                                data_table(
                                    ["duration", "omega_max", "pulses", "drive terms", "kernel"], seg_rows
                                ),
                                ft.Column(
                                    [shown(a, index, label=False) for a in v.approximations]
                                    or [status_line("no approximation recorded")],
                                    spacing=2,
                                ),
                            ],
                            title="Free terms, offsets, caps, segments and approximations",
                        ),
                    ],
                    spacing=12,
                ),
                why=lambda e: session.select_concept(4, "hamiltonian"),
                info="the terms the engine assembled for this step, with the numbers of this device and this sample (Section 5.7); the step is cut at every pulse boundary, as the engine integrates it",
                key="hamiltonian",
            ),
            card(
                "Drive terms",
                ft.Column(term_tiles, spacing=4),
                why=lambda e: session.select_concept(4, "lamb_dicke"),
                info="one term per (pulse, ion): the addressed ion and each neighbour its light spills onto; open a term for its tones, its eta per mode and the Omega_(n',n) table",
                key="drive-terms",
            ),
            card(
                "Collapse operators",
                ft.Column(collapse_tiles, spacing=4),
                why=lambda e: session.select_concept(4, "noise_as_physics"),
                info="the L_k of the master equation (Section 5.7); a jump on Level 3 opens its channel here",
                key="collapse",
            ),
        ]
        if target is not None and target[0] in store.records:
            k, s_i, sm, br = target
            rec_t = store.records[k]
            pulse = rec_t.step(s_i).pulse_indices[0] if rec_t.step(s_i).pulse_indices else 0
            out.append(
                ft.Row(
                    [
                        ft.OutlinedButton(
                            content=ft.Text("Back to the dynamics of this pulse"),
                            icon=ft.Icons.ZOOM_OUT,
                            on_click=lambda e: page.navigate(f"/job/{k}/dynamics/{pulse}/{sm}"),
                        )
                    ]
                )
            )
    out.append(_gates_card(store, session, layer, table, index, plain))
    return out


def _me(table: np.ndarray, r: int, c: int, mode: int) -> Shown:
    if r < table.shape[0] and c < table.shape[1]:
        return Shown("h_matrix_element", float(table[r, c]), f"mode {mode}: n' = {r}, n = {c}")
    return Shown("h_matrix_element", None, f"mode {mode}: n' = {r}, n = {c}")


# ---- the current device card (Level 0) ---------------------------------------------------------------------------------


def _table(store: Store, layer: DeviceLayer) -> TableRecord | None:
    """The table the pages compare the layer with: the device's own recalibrated one, else the current record's."""
    table = store.table_for(layer.device_hash)
    current = store.record()
    return current.table if table is None and current is not None else table


@ft.component
def CurrentDeviceCard(store: Store, session: Session, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    layer = store.layer()
    page = ft.context.page
    if layer is None:
        return card(
            "Current device",
            ft.Column(
                [
                    status_line("deriving the device layer for the current knobs"),
                    ProgressRows(store, session),
                ],
                spacing=6,
            ),
            key="current-device",
        )
    v = layer_card_view(layer, _table(store, layer))
    plain = store.learner.plan(0).plain_labels_first
    recalibrating = store.running_of("recalibrate") is not None
    body = ft.Column(
        [
            _stale_badge(layer),
            _tiles(v.overrides, index, plain=plain)
            if v.overrides
            else status_line("no knob changed: the preset as published"),
            _tiles(v.spam + v.gate_errors, index, plain=plain),
            details(
                store,
                session,
                4,
                "level0.current_device",
                [
                    _rows_table(v.rows, index),
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(m.detail, size=theme.SIZE_SMALL, color=MUTED, width=110),
                                    shown(m, index, label=False, size=theme.SIZE_SMALL),
                                ],
                                spacing=4,
                            )
                            for m in v.modes
                        ],
                        spacing=2,
                    ),
                    _rows_table(v.spam, index),
                    _rows_table(v.gate_errors, index)
                    if v.gate_errors
                    else status_line("no gate error estimated"),
                    shown(v.device_hash, index, size=theme.SIZE_SMALL),
                ],
            ),
        ],
        spacing=12,
    )
    return card(
        "Current device (edited)" if layer.overrides else "Current device",
        body,
        why=lambda e: session.select_concept(4, "calibration"),
        info="the device the next Run uses; estimate = closed form on this device, calibrated = the table, stale = a table for another device (Section 14.4)",
        actions=[
            ft.FilledButton(
                content=ft.Text("Recalibrate"),
                icon=ft.Icons.TUNE,
                on_click=lambda e: session.submit_recalibrate(),
                disabled=(not layer.stale and layer.table_hash is not None) or recalibrating,
            ),
            ft.TextButton(
                content=ft.Text("Change the device"),
                icon=ft.Icons.FUNCTIONS,
                on_click=lambda e: page.navigate("/device/trap"),
            ),
        ],
        key="current-device",
    )


# ---- the pages --------------------------------------------------------------------------------------------------------

PageRenderer = Callable[[Page], list[ft.Control]]


class DevicePage(NamedTuple):
    title: str
    question: str
    section: str
    """The Part II subsection the page's explain drawer opens."""
    concepts: tuple[str, ...]
    """The concepts the page's explain drawer carries, in teaching order."""
    render: PageRenderer


# fmt: off
DEVICE_PAGES: dict[str, DevicePage] = {
    "species": DevicePage("The atom", "Which two levels are the qubit, and what else is there?", "4.5.1",
                          ("atomic_structure", "light_coupling"), _species_page),
    "trap": DevicePage("The trap", "Why does the ion stay put, and how fast does it swing?", "4.1.1",
                       ("trap_and_mathieu", "lamb_dicke"), _trap_page),
    "crystal": DevicePage("The crystal", "Where do the ions sit and how do they vibrate together?", "4.1.3",
                          ("mode", "lamb_dicke", "trap_and_mathieu"), _crystal_page),
    "light": DevicePage("The light", "How much light does what to the qubit?", "4.5.4",
                        ("light_coupling", "pulse", "crosstalk"), _light_page),
    "noise": DevicePage("The noise", "Where do the error rates come from?", "6.1",
                        ("noise_as_physics", "quantum_jumps"), _noise_page),
    "cooling": DevicePage("Cooling and preparation", "How does every shot start cold and in |0>?", "4.2",
                          ("cooling_ladder", "fock_states", "debye_waller"), _cooling_page),
    "readout": DevicePage("The readout", "How does a photon count become a bit?", "8.1",
                          ("readout_rates", "spam"), _readout_page),
    "hamiltonian": DevicePage("The equation", "What exactly is the simulator integrating for this pulse?", "5.7",
                              ("hamiltonian", "lamb_dicke", "quantum_jumps", "calibration"), _hamiltonian_page),
}
# fmt: on
"""The Level 4 pages by route, in the order of their tabs."""


@ft.component
def Level4Page(store: Store, session: Session, page_name: str, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    device_page = DEVICE_PAGES[page_name]
    if store.device_page != page_name:
        store.device_page = page_name
    ck = store.device_ref().cache_key()

    def derive() -> None:
        session.submit_derive()

    ft.use_effect(derive, dependencies=[ck])
    layer = store.layers.get(ck)
    nav = page_tabs(
        [(p, d.title) for p, d in DEVICE_PAGES.items()],
        page_name,
        lambda p: page.navigate(f"/device/{p}"),
        key_prefix="device-page",
    )
    body: list[ft.Control] = [level_header(device_page.title, device_page.question), nav]
    if layer is None:
        failed = next(
            (
                j
                for j in store.jobs.values()
                if j.request == "derive" and j.target.get("cache_key") == ck and j.done and j.error
            ),
            None,
        )
        if failed is None:
            body.append(
                card(
                    "Deriving the device",
                    ft.Column(
                        [
                            status_line("the eight pages, derived from the device model alone"),
                            ProgressRows(store, session),
                        ],
                        spacing=6,
                    ),
                    key="deriving",
                )
            )
            return ft.Column(body, spacing=16, expand=True, scroll=ft.ScrollMode.AUTO)
        # the device model refused these knob values (a trap without an rf record, an unstable chain, a mode the cooling
        # beams do not reach, a failed Lamb-Dicke check): the reason is shown where the knobs are, and the knobs stay in
        # reach on the last device that did derive, so the value can be moved back or every knob reset
        reason = failed.error.strip().splitlines()[-1] if failed.error else "the worker failed"
        body.append(
            card(
                "The device could not be derived with these knob values",
                ft.Column(
                    [
                        ft.Text(reason, size=theme.SIZE_BODY, color=ft.Colors.ERROR),
                        ft.Row(
                            [
                                ft.FilledTonalButton(
                                    content=ft.Text("Reset every knob"),
                                    icon=ft.Icons.RESTART_ALT,
                                    on_click=lambda e: session.reset_knobs(),
                                    key="reset-knobs-failed",
                                ),
                                status_line("or move the knob back below"),
                            ],
                            spacing=theme.GAP,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=theme.GAP,
                ),
                key="derive-failed",
            )
        )
        previous = list(store.layers.values())
        if previous:
            body.append(KnobPanel(store, session, previous[-1], page_name, index))
        return ft.Column(body, spacing=16, expand=True, scroll=ft.ScrollMode.AUTO)
    body.append(DeviceStatusStrip(store, session, layer))
    body += device_page.render(
        Page(store, session, layer, _table(store, layer), index, store.learner.plan(4).plain_labels_first)
    )
    body.append(KnobPanel(store, session, layer, page_name, index))
    body.append(
        status_line(
            f"derived in {layer.wall_time_s:.1f} s" + (f"; {'; '.join(layer.notes)}" if layer.notes else "")
        )
    )
    return ft.Column(body, spacing=16, expand=True, scroll=ft.ScrollMode.AUTO)
