"""Level 4, the physics (PLAN.md Section 14.2 row 4, Section 14.4; DESIGN.md Section 5): the species, trap, crystal, light,
noise, cooling and readout pages and the Hamiltonian builder, every drawn element computed by the worker into the device layer,
every device knob on the page it belongs to. A knob change re-derives the analytic layer at once (a "derive" request, tens of
milliseconds of physics plus the recipe re-derivation) and marks the calibration table stale; the one primary action while
the table is stale is Recalibrate, the user-initiated background job of Section 14.4.
"""

from __future__ import annotations

import math
from typing import Any

import flet as ft
import numpy as np

from qutip_trap_app.device_layer import DeviceLayer
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import Record, TableRecord
from qutip_trap_app.viewmodel.learn import DEVICE_PAGES
from qutip_trap_app.viewmodel.physics import (
    KnobRow,
    Row,
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
from qutip_trap_app.views import drawing
from qutip_trap_app.views.common import (
    card,
    chip,
    data_table,
    fmt_number,
    kv_rows,
    level_header,
    shown,
    value_cell,
)
from qutip_trap_app.views.level0 import ProgressRows
from qutip_trap_app.views.state import Session, Store

PAGE_TITLES: dict[str, tuple[str, str]] = {
    "species": ("The atom", "Which two levels are the qubit, and what else is there?"),
    "trap": ("The trap", "Why does the ion stay put, and how fast does it swing?"),
    "crystal": ("The crystal", "Where do the ions sit and how do they vibrate together?"),
    "light": ("The light", "How much light does what to the qubit?"),
    "noise": ("The noise", "Where do the error rates come from?"),
    "cooling": ("Cooling and preparation", "How does every shot start cold and in |0>?"),
    "readout": ("The readout", "How does a photon count become a bit?"),
    "hamiltonian": ("The equation", "What exactly is the simulator integrating for this pulse?"),
}


def _rows(rows: tuple[Row, ...], index: ProvenanceIndex) -> ft.Control:
    return kv_rows(
        [
            (
                r.label,
                ft.Row(
                    [
                        shown(r.value, index, label=False),
                        ft.Text(r.status, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                    ],
                    spacing=6,
                    wrap=True,
                ),
            )
            for r in rows
        ]
    )


def _direction(e_hat: tuple[float, float, float]) -> str:
    """A mode's direction in one short cell: the axis letter when it lies along one, else its components along x, y, z."""
    letters = "xyz"
    k = max(range(3), key=lambda i: abs(e_hat[i]))
    if abs(e_hat[k]) > 0.999:
        return f"{'+' if e_hat[k] >= 0 else '-'}{letters[k]}"
    return " ".join(f"{v:+.2f}{letters[i]}" for i, v in enumerate(e_hat) if abs(v) > 0.005)


def _stale_badge(layer: DeviceLayer) -> ft.Control:
    text = stale_status(layer)
    if layer.stale:
        bg, fg, icon = ft.Colors.ERROR_CONTAINER, ft.Colors.ON_ERROR_CONTAINER, ft.Icons.HISTORY
    elif layer.table_hash is None:
        bg, fg, icon = ft.Colors.SECONDARY_CONTAINER, ft.Colors.ON_SECONDARY_CONTAINER, ft.Icons.HELP_OUTLINE
    else:
        bg, fg, icon = ft.Colors.PRIMARY_CONTAINER, ft.Colors.ON_PRIMARY_CONTAINER, ft.Icons.CHECK_CIRCLE
    return ft.Container(
        content=ft.Row(
            [ft.Icon(icon, size=16, color=fg), ft.Text(text, size=12, color=fg)], spacing=6, tight=True
        ),
        bgcolor=bg,
        border_radius=ft.BorderRadius.all(12),
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
    )


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
def KnobPanel(
    store: Store, session: Session, layer: DeviceLayer, page_name: str, index: ProvenanceIndex
) -> ft.Control:
    ft.use_state(store)
    rows = knob_rows(layer, page_name)
    deriving = store.running_of("derive") is not None
    recalibrating = store.running_of("recalibrate") is not None
    controls: list[ft.Control] = []
    for k in rows:

        def on_end(e: Any, kk: KnobRow = k) -> None:
            session.set_knob(kk.knob.id, _slider_to_value(kk, float(e.control.value)))

        def reset(_e: Any, kk: KnobRow = k) -> None:
            session.set_knob(kk.knob.id, None)

        needs_rf = k.knob.requires_rf and layer.trap.rf_frequency_hz is None
        controls.append(
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(k.knob.label, size=13, weight=ft.FontWeight.W_500, tooltip=k.knob.term),
                            ft.Text(
                                fmt_number(k.value, k.knob.unit)
                                if not k.knob.relative
                                else f"x {k.value:.3g}",
                                size=13,
                                tooltip=f"{k.knob.term}; Section {k.knob.section}",
                            ),
                            chip(k.knob.ledger_id, index),
                            ft.IconButton(
                                icon=ft.Icons.RESTART_ALT,
                                tooltip="back to the preset's value",
                                on_click=reset,
                                visible=k.is_override,
                                icon_size=16,
                            ),
                        ],
                        spacing=8,
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Slider(
                        min=0.0,
                        max=1.0,
                        value=_value_to_slider(k, k.value),
                        on_change_end=on_end,
                        disabled=deriving or needs_rf,
                        tooltip=k.knob.doc,
                    ),
                    ft.Text(
                        "declare the rf drive frequency first: without it the Mathieu a is unknown and q cannot be scaled"
                        if needs_rf
                        else k.knob.doc,
                        size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                spacing=0,
            )
        )
    if not controls:
        controls.append(
            ft.Text(
                "this page has no knob of its own; the pages that do are trap, species, light, noise, cooling and readout",
                size=12,
                italic=True,
            )
        )
    actions: list[ft.Control] = [
        ft.FilledButton(
            content=ft.Text("Recalibrate (about 15 s)"),
            icon=ft.Icons.TUNE,
            on_click=lambda e: session.submit_recalibrate(),
            disabled=(not layer.stale and layer.table_hash is not None) or recalibrating,
            tooltip="the surrogate table for this device: closed forms, exact spot checks, detection records (Section 7.5); the next Run uses it",
        ),
        ft.TextButton(
            content=ft.Text("Reset every knob"),
            on_click=lambda e: session.reset_knobs(),
            disabled=not layer.overrides,
        ),
    ]
    return card(
        "Knobs of this page",
        ft.Column([_stale_badge(layer)] + controls, spacing=10),
        subtitle="move a slider: the analytic layer re-derives at once and the calibration goes stale until you recalibrate (Section 14.4)",
        actions=actions,
    )


# ---- the pages -------------------------------------------------------------------------------------------------------------------------


def _species_page(layer: DeviceLayer, index: ProvenanceIndex) -> list[ft.Control]:
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
    qubit_level = v.qubit[0].value.value.split(" ")[0] if isinstance(v.qubit[0].value.value, str) else ""
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
    return [
        ft.ResponsiveRow(
            [
                ft.Column(
                    [
                        card("Identity", _rows(v.identity, index)),
                        card(
                            "The level diagram",
                            ft.Column(
                                [
                                    drawing.level_diagram(levels, transitions, qubit_level),
                                    data_table(
                                        ["level", "energy", "lifetime", "linewidth", "A_hfs", "B_hfs", "g_J"],
                                        level_rows,
                                    ),
                                ],
                                spacing=8,
                            ),
                            subtitle="fine-structure levels from the species table; the vertical spacing is compressed and labelled not to scale",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
                ft.Column(
                    [
                        card(
                            "The qubit pair",
                            _rows(v.qubit, index),
                            subtitle="two hyperfine levels chosen so that the field barely moves their splitting (Section 4.5.1)",
                        ),
                        card(
                            f"Magnetic sublevels of {qubit_level} at {layer.species.field_gauss:g} G",
                            ft.Column(
                                [
                                    drawing.sublevel_fan(
                                        qubit_fan,
                                        title=f"{qubit_level} at the operating field (to scale within the level)",
                                    ),
                                    data_table(
                                        ["level", "sublevel", "energy in the level", "dE/dB", "d2E/dB2"],
                                        sub_rows,
                                    ),
                                ],
                                spacing=8,
                            ),
                            subtitle="the hyperfine-plus-Zeeman Hamiltonian diagonalized at the field; slopes by Hellmann-Feynman",
                        ),
                        card(
                            f"Zeeman shifts of {v.sweep_level} against the field",
                            ft.Column(
                                [
                                    drawing.zeeman_chart(
                                        v.sweep_b_gauss, v.sweep_energies_hz, v.sweep_labels
                                    ),
                                    data_table(["clock point B0", "|nu| there", "Taylor c2"], clock_rows)
                                    if clock_rows
                                    else ft.Text(
                                        f"no field-independent point of the qubit pair between {v.clock_scan[0]:g} and {v.clock_scan[1]:g} G: the 171Yb+ clock states are insensitive at zero field, quadratic elsewhere",
                                        size=12,
                                        italic=True,
                                    ),
                                ],
                                spacing=8,
                            ),
                        ),
                        card(
                            "Transitions",
                            data_table(["line", "wavelength", "linewidth", "branching", "I_sat"], tr_rows),
                            subtitle="what the species table stores, with the derived saturation intensity",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        )
    ]


def _trap_page(layer: DeviceLayer, index: ProvenanceIndex) -> list[ft.Control]:
    v = trap_view(layer)
    mathieu_rows: list[list[ft.Control | str]] = [
        [ax, *(value_cell(s, index) for s in cells)] for ax, *cells in v.mathieu
    ]
    stability: ft.Control
    if v.stability_q is not None and v.stability_lower is not None and v.stability_upper is not None:
        stability = ft.Column(
            [
                drawing.stability_diagram(
                    v.stability_q, v.stability_lower, v.stability_upper, v.operating_points
                ),
                ft.Row(
                    ([shown(v.stability_edge, index)] if v.stability_edge is not None else [])
                    + [
                        ft.Text(
                            "boundary: the monodromy of x'' + (a - 2q cos 2xi) x = 0 (Section 4.1.1), bisected per q",
                            size=11,
                            color=ft.Colors.ON_SURFACE_VARIANT,
                        )
                    ],
                    wrap=True,
                    spacing=12,
                ),
            ],
            spacing=6,
        )
    else:
        stability = ft.Text("the stability map is being computed", size=12, italic=True)
    if not v.operating_points:
        stability = ft.Column(
            [
                stability,
                ft.Text(
                    "no operating point yet: declare the rf drive frequency (knob) to place this trap on the diagram",
                    size=12,
                    italic=True,
                ),
            ],
            spacing=6,
        )
    return [
        ft.ResponsiveRow(
            [
                ft.Column(
                    [
                        card(
                            "Secular frequencies",
                            _rows(v.secular + v.rf, index),
                            subtitle=f"trap path: {v.path}; the pseudopotential bowl the ions roll in",
                        ),
                        card(
                            "Mathieu parameters",
                            ft.Column(
                                [
                                    ft.Text(v.mathieu_note, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                                    data_table(
                                        ["axis", "a", "q", "beta", "nu = beta Omega/2", "C0"], mathieu_rows
                                    )
                                    if mathieu_rows
                                    else ft.Container(),
                                ],
                                spacing=6,
                            ),
                            subtitle="x'' + [a - 2q cos 2xi] x = 0; beta by the monodromy method; C0 the Wronskian-normalized micromotion factor on eta",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
                ft.Column(
                    [
                        card(
                            "Stability diagram",
                            stability,
                            subtitle="the first stability region with this device's operating point per radial axis (the axial axis has q_z = 0)",
                        ),
                        card(
                            "Stray field and micromotion",
                            _rows(v.fields, index),
                            subtitle="a stray field parks the ion off the rf null, where it is dragged at the rf frequency (Berkeland, Section 4.1.1)",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        )
    ] + (
        [ft.Column([ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in v.notes], spacing=2)]
        if v.notes
        else []
    )


@ft.component
def CrystalPage(layer: DeviceLayer, index: ProvenanceIndex) -> ft.Control:
    v = crystal_view(layer)
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
        dense=True,
    )
    comp_row = ft.Row(
        [shown(c, index, label=False) for c in (m.components if m is not None else ())], wrap=True, spacing=10
    )
    return ft.Column(
        [
            ft.ResponsiveRow(
                [
                    ft.Column(
                        [
                            card(
                                "Positions and modes",
                                ft.Column(
                                    [
                                        picture,
                                        ft.Row([selector], wrap=True),
                                        ft.Text(
                                            "eigenvector components (mass-weighted, unit norm)",
                                            size=12,
                                            weight=ft.FontWeight.W_600,
                                        ),
                                        comp_row,
                                    ],
                                    spacing=8,
                                ),
                                subtitle="equilibrium positions from the Coulomb-plus-trap potential; modes from its Hessian (Sections 4.1.2, 4.1.3)",
                            ),
                            card(
                                "Geometry",
                                _rows(v.geometry + v.zigzag, index),
                                subtitle="the chain buckles when the radial stiffness falls below the threshold",
                            ),
                        ],
                        col={"xs": 12, "lg": 6},
                        spacing=10,
                    ),
                    ft.Column(
                        [
                            card(
                                "Mode spectrum",
                                data_table(
                                    [
                                        "mode",
                                        "family",
                                        "#",
                                        "frequency",
                                        "e_hat",
                                        "uniform-field weight",
                                        "heating",
                                    ],
                                    mode_rows,
                                ),
                                subtitle="ordered axial, transverse_1, transverse_2, ascending frequency in each family (conv.mode_index)",
                            ),
                            card(
                                "Lamb-Dicke parameters",
                                ft.Column(
                                    (
                                        [
                                            ft.Row(
                                                [shown(v.delta_k, index)]
                                                + (
                                                    [
                                                        ft.Text(
                                                            "C0 applied (rf record present)"
                                                            if v.c0_applied
                                                            else "C0 = 1: no rf record on this trap",
                                                            size=12,
                                                            italic=True,
                                                        )
                                                    ]
                                                ),
                                                wrap=True,
                                                spacing=12,
                                            )
                                        ]
                                        if v.delta_k is not None
                                        else []
                                    )
                                    + [
                                        data_table(eta_cols, eta_rows)
                                        if eta_rows
                                        else ft.Text(
                                            "no entangling Raman pair on this device", size=12, italic=True
                                        )
                                    ],
                                    spacing=8,
                                ),
                                subtitle="eta_{i,m} = (Delta k . e_m) c_{i,m} sqrt(hbar/(2 m_i omega_m)) for the entangling pair's Delta k (conv.lamb_dicke)",
                            ),
                        ],
                        col={"xs": 12, "lg": 6},
                        spacing=10,
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.START,
                spacing=12,
                run_spacing=12,
            )
        ]
        + (
            [ft.Column([ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in v.notes], spacing=2)]
            if v.notes
            else []
        ),
        spacing=12,
    )


ROLE_COLORS = {
    "detection": ft.Colors.TERTIARY,
    "entangling gates": ft.Colors.PRIMARY,
    "Doppler cooling": ft.Colors.SECONDARY,
    "optical pumping": ft.Colors.SECONDARY,
}


def _light_page(layer: DeviceLayer, index: ProvenanceIndex) -> list[ft.Control]:
    v = light_view(layer)
    beams_geo = []
    for b in v.beams:
        role = b.roles[0] if b.roles else "unused"
        color = ROLE_COLORS.get(role, ft.Colors.ON_SURFACE_VARIANT)
        if role.startswith("single-qubit"):
            color = ft.Colors.ERROR
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
    curve: ft.Control = ft.Text("no Raman pair to sweep", size=12, italic=True)
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
            spacing=6,
        )
    return [
        ft.ResponsiveRow(
            [
                ft.Column(
                    [
                        card(
                            "Beam geometry",
                            geometry,
                            subtitle="beams as fields with their k-vectors through the ions; the quantization axis is the magnetic field (Section 4.5.3)",
                        ),
                        card(
                            "Beams",
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
                        ),
                        card(
                            "Crosstalk",
                            ft.Row(
                                [shown(x, index) for x in v.crosstalk]
                                or [ft.Text("no addressing beams", size=12, italic=True)],
                                wrap=True,
                                spacing=12,
                            ),
                            subtitle="the Rabi ratio of an addressing pair's light on the neighbour, from the beam profiles at the ion positions (conv.crosstalk_ratio)",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
                ft.Column(
                    [
                        card(
                            "Drives: what the light does",
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
                            subtitle="the two-photon Rabi frequency, the differential light shift and the scattering budget from the level structure (Sections 4.5.4, 4.5.5)",
                        ),
                        card(
                            "Scattering against detuning",
                            curve,
                            subtitle="the same Raman pair swept in wavelength between the fine-structure lines: the coupling and the scattering both fall, the error per pulse least far from the lines (Ozeri, Section 4.3.2)",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        )
    ] + (
        [ft.Column([ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in v.notes], spacing=2)]
        if v.notes
        else []
    )


def _field_direction(layer: DeviceLayer) -> tuple[float, float, float]:
    d = layer.card.field_direction
    return (float(d[0]), float(d[1]), float(d[2]))


def _noise_page(layer: DeviceLayer, index: ProvenanceIndex) -> list[ft.Control]:
    v = noise_view(layer)
    spectra: list[ft.Control] = []
    for sp in v.spectra:
        if sp.is_zero:
            spectra.append(
                ft.Row(
                    [
                        ft.Text(f"{sp.name}: zero everywhere ({sp.unit})", size=12),
                        shown(sp.level, index, label=False),
                    ],
                    spacing=8,
                    wrap=True,
                )
            )
        else:
            spectra.append(
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text(f"{sp.name} ({sp.unit})", size=12, weight=ft.FontWeight.W_600),
                                shown(sp.level, index),
                            ],
                            wrap=True,
                            spacing=8,
                        ),
                        drawing.line_chart(
                            [(sp.name, sp.omega_rad_s / (2 * np.pi), sp.S)],
                            x_title="f (Hz)",
                            y_title=sp.unit,
                            log_x=True,
                            log_y=True,
                            height=150,
                        ),
                        ft.Text(
                            "apparatus: " + (", ".join(sp.apparatus) or "undeclared"),
                            size=11,
                            color=ft.Colors.ON_SURFACE_VARIANT,
                        ),
                    ],
                    spacing=4,
                )
            )
    drift_rows: list[list[ft.Control | str]] = [
        [
            name,
            value_cell(rms, index),
            value_cell(tau, index),
            value_cell(servo, index) if servo is not None else "none",
        ]
        for name, rms, tau, servo in v.drifts
    ]
    rates: list[ft.Control] = []
    if v.heating:
        rates.append(ft.Row([shown(h, index) for h in v.heating], wrap=True, spacing=12))
    if v.motional_dephasing:
        rates.append(ft.Row([shown(h, index) for h in v.motional_dephasing], wrap=True, spacing=12))
    if v.qubit_dephasing:
        rates.append(ft.Row([shown(h, index) for h in v.qubit_dephasing], wrap=True, spacing=12))
    if not rates:
        rates.append(
            ft.Text(
                "every rate is zero: this device is quiet. Raise the electric-field noise density (knob) and every heating collapse operator appears with its rate.",
                size=12,
                italic=True,
            )
        )
    return [
        ft.ResponsiveRow(
            [
                ft.Column(
                    [
                        card(
                            "Spectra",
                            ft.Column(spectra, spacing=10),
                            subtitle="two-sided densities as the noise model stores them; a heating rate is e^2 S_E/(4 m hbar omega) (Section 4.1.5)",
                        ),
                        card(
                            "Rates the spectra imply",
                            ft.Column(
                                rates
                                + ([_rows((v.correlation,), index)] if v.correlation is not None else []),
                                spacing=8,
                            ),
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
                ft.Column(
                    [
                        card(
                            "Slow drifts (quasi-static per shot)",
                            data_table(["parameter", "rms", "correlation time", "servo"], drift_rows),
                            subtitle="each is drawn once per dynamical sample (Section 6.1 route (c)); a servo high-passes it",
                        ),
                        card(
                            "Mains, collisions and provenance",
                            ft.Column(
                                [
                                    ft.Row(
                                        [shown(m, index) for m in v.mains]
                                        or [ft.Text("no mains harmonics declared", size=12)],
                                        wrap=True,
                                        spacing=12,
                                    ),
                                    ft.Row(
                                        [shown(c, index) for c in v.collisions]
                                        or [ft.Text("no background-gas collisions declared", size=12)],
                                        wrap=True,
                                        spacing=12,
                                    ),
                                    ft.Text(
                                        v.provenance_sentence or "no apparatus named: every rate is zero",
                                        size=12,
                                    ),
                                    ft.Text(
                                        f"undeclared rates: {v.undeclared_rates}; apparatus: {', '.join(v.apparatus) or 'none'}",
                                        size=11,
                                        color=ft.Colors.ON_SURFACE_VARIANT,
                                    ),
                                ],
                                spacing=6,
                            ),
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        )
    ] + (
        [ft.Column([ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in v.notes], spacing=2)]
        if v.notes
        else []
    )


def _cooling_page(layer: DeviceLayer, index: ProvenanceIndex) -> list[ft.Control]:
    v = cooling_view(layer)
    if v is None:
        return [
            card(
                "Cooling unavailable",
                ft.Text(layer.cooling_error or "the recipe could not run on this device", size=13),
            )
        ]
    stage_rows: list[list[ft.Control | str]] = [
        [kind, prov, ft.Row([shown(s, index, label=False) for s in nbars], wrap=True, spacing=8)]
        for kind, prov, nbars in v.stages
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
                            ft.Text(
                                f"mode {sb.mode}, cooled through ion {sb.ion}",
                                size=12,
                                weight=ft.FontWeight.W_600,
                            ),
                            shown(sb.eta, index),
                            shown(sb.omega0, index),
                        ],
                        wrap=True,
                        spacing=10,
                    ),
                    drawing.line_chart(
                        [(f"<n> of mode {sb.mode} after each pulse", sb.times_s * 1e6, sb.nbar_after_pulse)],
                        x_title="t (us)",
                        y_title="<n>",
                        log_y=True,
                        height=150,
                    ),
                    ft.Row(
                        [
                            shown(sb.nbar_start, index),
                            shown(sb.nbar_end, index),
                            shown(sb.nbar_final_run, index),
                        ],
                        wrap=True,
                        spacing=12,
                    ),
                    ft.Text(
                        f"{len(sb.orders)} pulses; orders {sorted(set(sb.orders), reverse=True)} (higher orders first); durations per order: "
                        + ", ".join(
                            f"k={k}: {t * 1e6:.1f} us"
                            for k, t in dict(zip(sb.orders, sb.durations_s)).items()
                        ),
                        size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                spacing=4,
            )
        )
    if not sideband_children:
        sideband_children.append(ft.Text("no sideband cooling in this recipe", size=12, italic=True))
    pump_children: list[ft.Control] = []
    for p in v.pumps:
        series = [(label, p.trace_times_s * 1e6, arr) for label, arr in sorted(p.trace_populations.items())][
            :6
        ]
        pump_children.append(
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(f"ion {p.ion}", size=12, weight=ft.FontWeight.W_600),
                            shown(p.preparation_error, index),
                            shown(p.photons, index),
                        ]
                        + ([shown(p.time_to_reach, index)] if p.time_to_reach is not None else []),
                        wrap=True,
                        spacing=10,
                    ),
                    drawing.line_chart(series, x_title="t (us)", y_title="population", height=150)
                    if series
                    else ft.Container(),
                    ft.Row([shown(h, index, label=False) for h in p.heating], wrap=True, spacing=8),
                ],
                spacing=4,
            )
        )
    return [
        ft.ResponsiveRow(
            [
                ft.Column(
                    [
                        card(
                            "The recipe",
                            ft.Column(
                                [
                                    _rows(v.recipe, index),
                                    ft.Column(
                                        [
                                            ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT)
                                            for n in layer.cooling.recipe_notes
                                        ],
                                        spacing=2,
                                    )
                                    if layer.cooling is not None
                                    else ft.Container(),
                                ],
                                spacing=8,
                            ),
                            subtitle="Doppler, then sideband, then the pump last: a pump done first would be erased (Section 4.2.6)",
                        ),
                        card(
                            "Stages and what each leaves",
                            data_table(["stage", "provenance", "nbar per mode"], stage_rows),
                        ),
                        card(
                            f"Doppler stage ({v.doppler_method} rate equations)",
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
                            subtitle="A_pm = W(Delta -/+ nu) + (eta~/eta)^2 W(Delta), one detuning shared by every mode (Sections 4.2.1, 4.2.2)",
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
                ft.Column(
                    [
                        card(
                            "Pulsed sideband cooling",
                            ft.Column(sideband_children, spacing=10),
                            subtitle="p -> W_k(t) p per pulse with the exact Omega_{n,n-k}; drawn without the repump recoil (core gap), whose effect is the gap to the run's own value",
                        ),
                        card(
                            "Optical pumping",
                            ft.Column(pump_children, spacing=10),
                            subtitle="the multi-level master equation of the pump beams (Section 4.2.8); the residual population outside |0> is the preparation error",
                        ),
                        card(
                            "Handed to the circuit",
                            ft.Column(
                                [
                                    ft.Row([shown(s, index) for s in v.final_nbar], wrap=True, spacing=12),
                                    shown(v.duration, index),
                                    ft.Text(
                                        "provenance: " + ", ".join(v.provenance),
                                        size=11,
                                        color=ft.Colors.ON_SURFACE_VARIANT,
                                    ),
                                ],
                                spacing=6,
                            ),
                        ),
                    ],
                    col={"xs": 12, "lg": 6},
                    spacing=10,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        )
    ] + (
        [
            ft.Column(
                [ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in v.approximations + v.notes],
                spacing=2,
            )
        ]
        if (v.approximations or v.notes)
        else []
    )


def _readout_page(layer: DeviceLayer, table: TableRecord | None, index: ProvenanceIndex) -> list[ft.Control]:
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
            markers = [
                (
                    0.0,
                    f"ceiling Gamma x {i.rates[5].value.value if isinstance(i.rates[5].value.value, float) else 0.25:.3g} = {(i.saturation_ceiling_per_s or 0.0):.3g}/s",
                )
            ]
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
                        _rows(i.rates, index),
                        ft.Text("count histograms at the device window", size=12, weight=ft.FontWeight.W_600),
                        drawing.two_histograms(
                            i.bright_pmf,
                            i.dark_pmf,
                            float(i.threshold_at_window.value)
                            if isinstance(i.threshold_at_window.value, float)
                            else None,
                        ),
                        ft.Row(
                            [
                                shown(i.window, index),
                                shown(i.threshold_at_window, index),
                                shown(i.eps_at_window[0], index),
                                shown(i.eps_at_window[1], index),
                            ],
                            wrap=True,
                            spacing=12,
                        ),
                        ft.Text(
                            "error against the window length (best threshold at each)",
                            size=12,
                            weight=ft.FontWeight.W_600,
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
                        ft.Row([shown(s, index) for s in i.best], wrap=True, spacing=12),
                        ft.Text("error budget at the device window", size=12, weight=ft.FontWeight.W_600),
                        data_table(["channel", "bright read as dark", "dark read as bright"], budget_rows),
                        ft.Text("rates against the light level", size=12, weight=ft.FontWeight.W_600),
                        sat_curve,
                    ],
                    spacing=8,
                ),
                subtitle="R_o saturates at the manifold's ceiling; R_d and R_b never do, so the light level stays near saturation (Section 8.1)",
            )
        )
    return [
        card(
            "Detector",
            ft.Column([_rows(v.detector, index)] + ([_rows(v.table, index)] if v.table else []), spacing=8),
            subtitle="the apparatus, and the table's calibrated threshold and window where a table exists",
        ),
        ft.ResponsiveRow(
            [ft.Column([c], col={"xs": 12, "lg": 6}, spacing=10) for c in ion_cards],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            run_spacing=12,
        ),
    ] + (
        [ft.Column([ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in v.notes], spacing=2)]
        if v.notes
        else []
    )


def _gates_card(layer: DeviceLayer, table: TableRecord | None, index: ProvenanceIndex) -> ft.Control:
    rows = gate_rows(layer, table)
    if not rows:
        return card(
            "Entangling pulse solutions",
            ft.Text("no entangling Raman pair on this device", size=12, italic=True),
        )
    children: list[ft.Control] = []
    for g in rows:
        mode_rows: list[list[ft.Control | str]] = [
            [str(m), value_cell(w, index), *(value_cell(e, index) for e in etas)] for m, w, etas in g.modes
        ]
        children.append(
            ft.Column(
                [
                    ft.Text(f"pair {g.pair}", size=13, weight=ft.FontWeight.W_600),
                    ft.Text(g.error, size=12, color=ft.Colors.ERROR) if g.error else ft.Container(),
                    _rows(g.summary, index),
                    ft.Row(
                        [shown(c, index) for c in g.chi_m] + [shown(a, index) for a in g.alpha_m],
                        wrap=True,
                        spacing=12,
                    ),
                    data_table(["mode", "frequency"] + [f"eta ion {i}" for i in g.pair], mode_rows)
                    if mode_rows
                    else ft.Container(),
                    _rows(g.table, index)
                    if g.table
                    else ft.Text("no table entry for this pair yet", size=12, italic=True),
                ],
                spacing=6,
            )
        )
    return card(
        "Entangling pulse solutions (the analytic layer)",
        ft.Column(children, spacing=12),
        subtitle="re-solved at once from the current modes and eta (Sections 4.4.3, 14.4); the table's calibrated waveform beside it, stale when the device changed",
    )


def _hamiltonian_page(
    store: Store, session: Session, layer: DeviceLayer, table: TableRecord | None, index: ProvenanceIndex
) -> list[ft.Control]:
    page = ft.context.page
    record: Record | None = None
    ham = None
    target = store.hamiltonian_target
    if target is not None:
        record = store.records.get(target[0])
        if record is not None:
            from qutip_trap_app.resim import hamiltonian_key

            ham = record.hamiltonian(hamiltonian_key(target[1], target[2], target[3]))
    if ham is None:
        record = store.record()
        if record is not None and record.hamiltonians:
            ham = record.hamiltonians[-1]
    out: list[ft.Control] = []
    if record is None:
        out.append(
            card(
                "No pulse yet",
                ft.Text(
                    "Run a job on Level 0 and zoom into a pulse on Level 3: this page then lists the terms of H(t) and the collapse operators the engine integrated for it.",
                    size=13,
                ),
            )
        )
    elif ham is None:
        steps = [s for s in record.schedule.steps if s.kind == "gate"]
        ent = next(
            (s for s in steps if any(g.gate_id in s.target_ids for g in record.schedule.gates)),
            steps[0] if steps else None,
        )
        building = store.running_of("zoom", key=record.key()) is not None

        def build(_e: Any) -> None:
            if ent is not None:
                store.hamiltonian_target = (record.key(), ent.index, 0, store.branch)
                session.submit_zoom(record.key(), ent.index, 0, store.branch)

        out.append(
            card(
                "Build the equation of a pulse",
                ft.Column(
                    [
                        ft.Text(
                            "The Hamiltonian record is built when a pulse is re-simulated on Level 3 (the same worker call). Build it now for the first entangling step, or zoom into any pulse on Level 3.",
                            size=13,
                        ),
                        ft.Row(
                            [
                                ft.FilledButton(
                                    content=ft.Text(
                                        f"Build for {ent.gate_id}" if ent is not None else "Build"
                                    ),
                                    icon=ft.Icons.FUNCTIONS,
                                    on_click=build,
                                    disabled=ent is None or building or record.replay is not None,
                                ),
                                ft.Text(
                                    "a derived (channel replay) run has no joint space to build on"
                                    if record.replay is not None
                                    else "",
                                    size=12,
                                    color=ft.Colors.ON_SURFACE_VARIANT,
                                ),
                            ],
                            wrap=True,
                            spacing=10,
                        ),
                        ProgressRows(store, session),
                    ],
                    spacing=8,
                ),
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
                            ft.Text(
                                f"mode {m}: Omega_(n',n)/Omega = |<n'|D(i eta)|n>| (first {n_show} of {d} levels; log colour, hover for the value)",
                                size=12,
                                weight=ft.FontWeight.W_600,
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
                                vertical_alignment=ft.CrossAxisAlignment.START,
                            ),
                            ft.Row(
                                [
                                    shown(_me(table_m, 0, 0, m), index),
                                    shown(_me(table_m, 1, 0, m), index),
                                    shown(_me(table_m, 0, 1, m), index),
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                        ],
                        spacing=4,
                    )
                )
            highlighted = store.selected_term == t.index
            term_tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(
                        f"drive term {t.index}: {'crosstalk onto' if t.is_crosstalk else 'addressed'} ion {t.ion} ({t.pulse})",
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.PRIMARY if highlighted else None,
                    ),
                    subtitle=ft.Text(t.operator_text, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    controls=[
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "coefficient: (hbar Omega(t)/2) e^{-i(mu t - phi(t))} summed over the tones; the operator is time independent (Section 5.2)",
                                        size=12,
                                    ),
                                    ft.Row(
                                        [
                                            shown(t.omega, index),
                                            shown(t.crosstalk_weight, index),
                                            shown(t.rabi_scale, index),
                                            shown(t.carrier_factor, index),
                                            shown(t.debye_waller, index),
                                            shown(t.nnz, index),
                                        ],
                                        wrap=True,
                                        spacing=12,
                                    ),
                                    data_table(
                                        ["tone", "detuning mu", "phase at start", "peak Omega/2pi"], tone_rows
                                    )
                                    if tone_rows
                                    else ft.Container(),
                                    ft.Row([shown(e, index) for e in t.etas], wrap=True, spacing=12),
                                    ft.Row([shown(f, index) for f in t.frozen], wrap=True, spacing=12)
                                    if t.frozen
                                    else ft.Container(),
                                ]
                                + matrices,
                                spacing=8,
                            ),
                            padding=ft.Padding.all(10),
                        )
                    ],
                    dense=True,
                    expanded=t.index == 0 or highlighted,
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
                        f"{name}: {len(items)} operator(s), {'integrated in this run' if active else 'listed, not integrated in this run'}",
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.PRIMARY if hit else None,
                    ),
                    subtitle=ft.Text(items[0].note, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    controls=[
                        ft.Container(
                            content=data_table(["channel", "rate", "ion", "mode", "active", "L_k"], rows),
                            padding=ft.Padding.all(8),
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
                        _rows(v.header, index),
                        ft.Text(
                            "H/hbar = sum_m omega_m a_m^dag a_m + sum_i (Delta_i/2) sigma_z^i + sum drives (Omega/2) e^{-i(mu t - phi)} sigma_+ prod_m D_m(i eta) + h.c. + Stark (Section 5.7)",
                            size=12,
                            selectable=True,
                        ),
                        ft.Text("free terms: the modes", size=12, weight=ft.FontWeight.W_600),
                        data_table(["mode", "class", "omega_m/2pi", "sample offset"], free_rows),
                        ft.Row(
                            [shown(o, index) for o in v.qubit_offsets] + [shown(s, index) for s in v.stark],
                            wrap=True,
                            spacing=12,
                        ),
                        ft.Row([shown(c, index) for c in v.caps], wrap=True, spacing=12),
                    ],
                    spacing=8,
                ),
                subtitle="the terms the engine assembled for this step, with the numbers of this device and this sample",
            ),
            card(
                "Drive terms",
                ft.Column(term_tiles, spacing=4),
                subtitle="one term per (pulse, ion): the addressed ion and each neighbour its light spills onto",
            ),
            card(
                "Collapse operators",
                ft.Column(collapse_tiles, spacing=4),
                subtitle="the L_k of the master equation (Section 5.7); a jump on Level 3 opens its channel here",
            ),
            card(
                "Segments and approximations",
                ft.Column(
                    [
                        data_table(["duration", "omega_max", "pulses", "drive terms", "kernel"], seg_rows),
                        ft.Column(
                            [shown(a, index, label=False) for a in v.approximations]
                            or [ft.Text("no approximation recorded", size=12)],
                            spacing=2,
                        ),
                    ],
                    spacing=8,
                ),
                subtitle="the step cut at every pulse boundary, as the engine integrates it",
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
    out.append(_gates_card(layer, table, index))
    return out


def _me(table: np.ndarray, r: int, c: int, mode: int) -> Any:
    from qutip_trap_app.viewmodel.catalogue import Shown

    if r < table.shape[0] and c < table.shape[1]:
        return Shown("h_matrix_element", float(table[r, c]), f"mode {mode}: n' = {r}, n = {c}")
    return Shown("h_matrix_element", None, f"mode {mode}: n' = {r}, n = {c}")


# ---- the current device card (Level 0, Section 14.4) ----------------------------------------------------------------------------------------


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
                    ft.Text("The device layer is being derived for the current knobs.", size=13),
                    ProgressRows(store, session),
                ],
                spacing=6,
            ),
        )
    table = store.table_for(layer.device_hash)
    current = store.record()
    if table is None and current is not None:
        table = current.table
    v = layer_card_view(layer, table)
    recalibrating = store.running_of("recalibrate") is not None
    body = ft.Column(
        [
            _stale_badge(layer),
            _rows(v.overrides, index)
            if v.overrides
            else ft.Text("no knob changed: this is the preset as published", size=12, italic=True),
            _rows(v.rows, index),
            ft.Text("modes", weight=ft.FontWeight.W_600, size=13),
            ft.Column(
                [
                    ft.Row([ft.Text(m.detail, size=12, width=110), shown(m, index, label=False)], spacing=4)
                    for m in v.modes
                ],
                spacing=2,
            ),
            ft.Text("readout and preparation", weight=ft.FontWeight.W_600, size=13),
            _rows(v.spam, index),
            ft.Text("gate errors", weight=ft.FontWeight.W_600, size=13),
            _rows(v.gate_errors, index) if v.gate_errors else ft.Text("none estimated", size=12, italic=True),
            shown(v.device_hash, index),
        ],
        spacing=8,
    )
    return card(
        "Current device (edited)" if layer.overrides else "Current device",
        body,
        subtitle="the device the next Run uses; estimate = closed form on this device, calibrated = the table, stale = a table for another device",
        actions=[
            ft.FilledButton(
                content=ft.Text("Recalibrate"),
                icon=ft.Icons.TUNE,
                on_click=lambda e: session.submit_recalibrate(),
                disabled=(not layer.stale and layer.table_hash is not None) or recalibrating,
            ),
            ft.TextButton(
                content=ft.Text("Change the device (Physics)"),
                icon=ft.Icons.FUNCTIONS,
                on_click=lambda e: page.navigate("/device/trap"),
            ),
        ],
    )


# ---- the page ---------------------------------------------------------------------------------------------------------------------------


@ft.component
def Level4Page(store: Store, session: Session, page_name: str, index: ProvenanceIndex) -> ft.Control:
    ft.use_state(store)
    page = ft.context.page
    name = page_name if page_name in DEVICE_PAGES else "hamiltonian"
    if store.device_page != name:
        store.device_page = name
    ref = store.device_ref()
    ck = ref.cache_key()

    def derive() -> None:
        session.submit_derive()

    ft.use_effect(derive, dependencies=[ck])
    layer = store.layers.get(ck)
    title, question = PAGE_TITLES[name]
    nav = ft.Row(
        [
            (ft.FilledTonalButton if p == name else ft.TextButton)(
                content=ft.Text(PAGE_TITLES[p][0], size=12),
                on_click=lambda e, pp=p: page.navigate(f"/device/{pp}"),
            )
            for p in DEVICE_PAGES
        ],
        wrap=True,
        spacing=4,
    )
    body: list[ft.Control] = [level_header(4, "The physics", question, title), nav]
    if layer is None:
        body.append(
            card(
                "Deriving the device",
                ft.Column(
                    [
                        ft.Text(
                            "species, trap, crystal, light, noise, cooling, readout and the pulse solver, from the device model alone",
                            size=13,
                        ),
                        ProgressRows(store, session),
                    ],
                    spacing=6,
                ),
            )
        )
        return ft.Column(body, spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)
    table = store.table_for(layer.device_hash)
    current = store.record()
    if table is None and current is not None:
        table = current.table
    body.append(KnobPanel(store, session, layer, name, index))
    if name == "species":
        body += _species_page(layer, index)
    elif name == "trap":
        body += _trap_page(layer, index)
    elif name == "crystal":
        body.append(CrystalPage(layer, index))
    elif name == "light":
        body += _light_page(layer, index)
    elif name == "noise":
        body += _noise_page(layer, index)
    elif name == "cooling":
        body += _cooling_page(layer, index)
    elif name == "readout":
        body += _readout_page(layer, table, index)
    else:
        body += _hamiltonian_page(store, session, layer, table, index)
    body.append(
        ft.Text(
            f"derived in {layer.wall_time_s:.1f} s: "
            + ", ".join(f"{n} {t:.2f} s" for n, t in layer.stages)
            + (f"; {'; '.join(layer.notes)}" if layer.notes else ""),
            size=11,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )
    )
    return ft.Column(body, spacing=12, expand=True, scroll=ft.ScrollMode.AUTO)


__all__ = ["PAGE_TITLES", "CrystalPage", "CurrentDeviceCard", "KnobPanel", "Level4Page"]
