"""Computed drawings and charts (PLAN.md Section 14.5 "No cartoons").

Every element drawn here is a number from a view-model: ion positions from the equilibrium solver, mode arrows from the
Hessian's eigenvectors, the stability boundary from the monodromy, beams from the configured geometry, levels from the
species table, loops from the recorded trajectories. The one illustrative choice, a level diagram's vertical spacing, is
drawn on a compressed scale and labelled "not to scale". Drawings use Flet's canvas; heatmaps are rendered to PNG by the
small encoder below (zlib plus the PNG chunk format). Colours follow the theme roles so both themes read, and meaning is
never carried by colour alone.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import flet as ft
import flet.canvas as cv
import flet_charts as fc
import numpy as np

from qutip_trap_app.views import theme

# ---- PNG heatmaps ---------------------------------------------------------------------------------------------------------------------

_SEQUENTIAL: tuple[tuple[float, float, float], ...] = (
    (0.267, 0.005, 0.329),
    (0.283, 0.141, 0.458),
    (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553),
    (0.164, 0.471, 0.558),
    (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518),
    (0.267, 0.749, 0.441),
    (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150),
    (0.993, 0.906, 0.144),
)
"""A perceptually ordered sequential ramp (the viridis anchors), dark for zero: the dataviz rule for a magnitude field."""


def sequential_rgb(t: float) -> tuple[int, int, int]:
    x = min(max(float(t), 0.0), 1.0) * (len(_SEQUENTIAL) - 1)
    k = min(int(math.floor(x)), len(_SEQUENTIAL) - 2)
    f = x - k
    a, b = _SEQUENTIAL[k], _SEQUENTIAL[k + 1]
    return tuple(int(round(255 * (a[i] + f * (b[i] - a[i])))) for i in range(3))  # type: ignore[return-value]


def png_bytes(rgba: np.ndarray) -> bytes:
    """A PNG from an (H, W, 4) uint8 array."""
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def heatmap_png(values: np.ndarray, *, cell_w: int, cell_h: int) -> bytes:
    """Rows are the vertical axis (drawn bottom-up: row 0 at the bottom), columns the horizontal one, on a log scale over six
    decades below the maximum: Fock populations span from 1 to 1e-6 and a linear ramp would show one bright cell."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    top = max(float(np.max(v, initial=0.0)), 1e-300)
    with np.errstate(divide="ignore"):
        t = np.clip((np.log10(np.clip(v, 1e-300, None)) - (math.log10(top) - 6.0)) / 6.0, 0.0, 1.0)
    rows, cols = t.shape
    img = np.zeros((rows * cell_h, cols * cell_w, 4), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            rgb = sequential_rgb(float(t[r, c]))
            y0 = (rows - 1 - r) * cell_h
            img[y0 : y0 + cell_h, c * cell_w : (c + 1) * cell_w, :3] = rgb
            img[y0 : y0 + cell_h, c * cell_w : (c + 1) * cell_w, 3] = 255
    return png_bytes(img)


def heatmap(
    values: np.ndarray,
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    x_title: str,
    y_title: str,
    cell_w: int = 14,
    cell_h: int = 10,
    tooltip: str = "",
) -> ft.Control:
    """A labelled heatmap: the PNG with its row labels on the left and column labels beneath (a table alternative is the
    caller's, DESIGN.md Section 4)."""
    rows, cols = values.shape
    img = ft.Image(
        src=heatmap_png(values, cell_w=cell_w, cell_h=cell_h),
        width=cols * cell_w,
        height=rows * cell_h,
        fit=ft.BoxFit.FILL,
        tooltip=tooltip or None,
    )
    ylab = ft.Column(
        [
            ft.Text(str(y_labels[r]), size=9, height=cell_h, color=ft.Colors.ON_SURFACE_VARIANT)
            for r in reversed(range(rows))
        ],
        spacing=0,
        tight=True,
    )
    step = max(1, cols // 8)
    xlab = ft.Row(
        [
            ft.Text(
                str(x_labels[c]) if c % step == 0 else "",
                size=9,
                width=cell_w,
                no_wrap=True,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
            for c in range(cols)
        ],
        spacing=0,
        tight=True,
    )
    legend = ft.Row(
        [
            ft.Image(
                src=heatmap_png(np.logspace(-6, 0, 12)[None, :], cell_w=12, cell_h=8),
                width=144,
                height=8,
                fit=ft.BoxFit.FILL,
            ),
            ft.Text("1e-6 to 1 (log)", size=9, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        spacing=6,
    )
    return ft.Column(
        [
            ft.Text(y_title, size=10, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Row([ylab, img], spacing=4, vertical_alignment=ft.CrossAxisAlignment.START),
            ft.Row([ft.Container(width=34), xlab], spacing=4),
            ft.Row(
                [
                    ft.Text(x_title, size=10, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Container(expand=True),
                    legend,
                ]
            ),
        ],
        spacing=2,
        tight=True,
    )


# ---- canvas helpers ------------------------------------------------------------------------------------------------------------------


def _stroke(color: str, width: float = 1.5, dash: Sequence[float] | None = None) -> ft.Paint:
    return ft.Paint(
        color=color,
        stroke_width=width,
        style=ft.PaintingStyle.STROKE,
        stroke_dash_pattern=None if dash is None else [float(d) for d in dash],
    )


def _fill(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _label(
    x: float, y: float, text: str, *, size: int = 10, color: str = ft.Colors.ON_SURFACE_VARIANT
) -> cv.Text:
    return cv.Text(x, y, text, style=ft.TextStyle(size=size, color=color))


def _axis_title(text: str) -> ft.Text:
    return ft.Text(text, size=10, color=ft.Colors.ON_SURFACE_VARIANT)


def _grid() -> fc.ChartGridLines:
    """Hairline gridlines one step off the surface, solid (the dataviz rule for recessive chart chrome)."""
    return fc.ChartGridLines(color=ft.Colors.OUTLINE_VARIANT, width=1)


class Axes:
    """A data-to-pixel map for one canvas with margins, drawing its own frame and a few ticks."""

    def __init__(
        self,
        width: float,
        height: float,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        *,
        margin: tuple[float, float, float, float] = (36.0, 10.0, 22.0, 10.0),
    ) -> None:
        self.width, self.height = float(width), float(height)
        self.left, self.top, self.bottom, self.right = margin
        self.xlim, self.ylim = xlim, ylim
        if xlim[1] == xlim[0]:
            self.xlim = (xlim[0] - 1.0, xlim[0] + 1.0)
        if ylim[1] == ylim[0]:
            self.ylim = (ylim[0] - 1.0, ylim[0] + 1.0)

    @property
    def plot_w(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> float:
        return self.height - self.top - self.bottom

    def x(self, v: float) -> float:
        return self.left + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.plot_w

    def y(self, v: float) -> float:
        return self.top + (self.ylim[1] - v) / (self.ylim[1] - self.ylim[0]) * self.plot_h

    def frame(
        self,
        x_title: str = "",
        y_title: str = "",
        *,
        x_ticks: Sequence[tuple[float, str]] = (),
        y_ticks: Sequence[tuple[float, str]] = (),
    ) -> list[cv.Shape]:
        out: list[cv.Shape] = [
            cv.Rect(
                self.left, self.top, self.plot_w, self.plot_h, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)
            )
        ]
        for v, text in x_ticks:
            px = self.x(v)
            out.append(
                cv.Line(
                    px,
                    self.top + self.plot_h,
                    px,
                    self.top + self.plot_h + 4,
                    paint=_stroke(ft.Colors.OUTLINE, 1.0),
                )
            )
            out.append(_label(px - 12, self.top + self.plot_h + 6, text, size=9))
        for v, text in y_ticks:
            py = self.y(v)
            out.append(cv.Line(self.left - 4, py, self.left, py, paint=_stroke(ft.Colors.OUTLINE, 1.0)))
            out.append(_label(2, py - 6, text, size=9))
        if x_title:
            out.append(_label(self.left + self.plot_w / 2 - 30, self.height - 12, x_title))
        if y_title:
            out.append(
                cv.Text(4, 2, y_title, style=ft.TextStyle(size=10, color=ft.Colors.ON_SURFACE_VARIANT))
            )
        return out

    def polyline(
        self,
        xs: Sequence[float] | np.ndarray,
        ys: Sequence[float] | np.ndarray,
        color: str,
        width: float = 1.5,
        dash: Sequence[float] | None = None,
    ) -> cv.Path:
        elements: list[cv.Path.PathElement] = []
        for k, (xv, yv) in enumerate(zip(xs, ys)):
            px, py = self.x(float(xv)), self.y(float(yv))
            elements.append(cv.Path.MoveTo(px, py) if k == 0 else cv.Path.LineTo(px, py))
        return cv.Path(elements, paint=_stroke(color, width, dash))

    def arrow(
        self, x0: float, y0: float, x1: float, y1: float, color: str, width: float = 2.0
    ) -> list[cv.Shape]:
        px0, py0, px1, py1 = self.x(x0), self.y(y0), self.x(x1), self.y(y1)
        out: list[cv.Shape] = [cv.Line(px0, py0, px1, py1, paint=_stroke(color, width))]
        dx, dy = px1 - px0, py1 - py0
        length = math.hypot(dx, dy)
        if length > 1e-9:
            ux, uy = dx / length, dy / length
            size = min(7.0, length / 2)
            out.append(
                cv.Path(
                    [
                        cv.Path.MoveTo(px1, py1),
                        cv.Path.LineTo(px1 - size * ux + 0.5 * size * uy, py1 - size * uy - 0.5 * size * ux),
                        cv.Path.LineTo(px1 - size * ux - 0.5 * size * uy, py1 - size * uy + 0.5 * size * ux),
                        cv.Path.Close(),
                    ],
                    paint=_fill(color),
                )
            )
        return out


def _ticks(lo: float, hi: float, n: int, fmt: str) -> list[tuple[float, str]]:
    return [(v, format(v, fmt)) for v in np.linspace(lo, hi, n)]


# ---- Level 4 drawings --------------------------------------------------------------------------------------------------------------------


def level_diagram(
    levels: Sequence[tuple[str, float, float | None]],
    transitions: Sequence[tuple[str, str, str, float]],
    qubit_level: str,
) -> ft.Control:
    """Fine-structure levels as horizontal bars at a compressed energy scale (sqrt of the energy above the ground level, so
    the ground and the optical levels fit one picture; labelled not to scale), the tabulated transitions as arrows with their
    wavelengths, the qubit level marked."""
    width, height = 420.0, 300.0
    names = [n for n, _e, _t in levels]
    energies = np.asarray([e for _n, e, _t in levels], dtype=float)
    scaled = np.sqrt(np.clip(energies, 0.0, None))
    top = float(np.max(scaled)) if scaled.size else 1.0
    ax = Axes(width, height, (0.0, 1.0), (0.0, top * 1.08), margin=(8.0, 14.0, 18.0, 8.0))
    shapes: list[cv.Shape] = []
    # levels sorted by energy take alternating columns so labels do not collide
    order = sorted(range(len(names)), key=lambda i: energies[i])
    col_of: dict[str, float] = {}
    for rank, i in enumerate(order):
        col_of[names[i]] = (
            0.12 + 0.76 * ((rank * 5) % 7) / 6.0
            if len(names) > 3
            else 0.2 + 0.6 * rank / max(len(names) - 1, 1)
        )
    for i, name in enumerate(names):
        x0 = col_of[name] - 0.09
        y = float(scaled[i])
        color = ft.Colors.PRIMARY if name == qubit_level else ft.Colors.ON_SURFACE
        shapes.append(
            cv.Line(
                ax.x(x0),
                ax.y(y),
                ax.x(x0 + 0.18),
                ax.y(y),
                paint=_stroke(color, 3.0 if name == qubit_level else 2.0),
            )
        )
        life = levels[i][2]
        text = (
            name
            if life is None
            else f"{name}  ({life:.3g} s)"
            if life > 1e-3
            else f"{name}  ({life * 1e9:.2f} ns)"
        )
        shapes.append(_label(ax.x(x0), ax.y(y) - 13, text, size=9, color=color))
    for k, (_label_text, lower, upper, lam) in enumerate(transitions):
        if lower not in col_of or upper not in col_of:
            continue
        i_lo, i_up = names.index(lower), names.index(upper)
        x_lo, x_up = col_of[lower], col_of[upper]
        color = ft.Colors.TERTIARY if lam < 500e-9 else ft.Colors.SECONDARY
        shapes.extend(ax.arrow(x_lo, float(scaled[i_lo]), x_up, float(scaled[i_up]), color, 1.2))
        # the label sits at a different fraction of each arrow so that neighbouring lines do not pile their labels up
        frac = (0.35, 0.55, 0.75)[k % 3]
        mx = ax.x(x_lo + frac * (x_up - x_lo))
        my = ax.y(float(scaled[i_lo]) + frac * float(scaled[i_up] - scaled[i_lo]))
        shapes.append(_label(mx + 3, my - 6, f"{lam * 1e9:.1f} nm", size=8, color=color))
    shapes.append(
        _label(
            8, height - 14, "vertical scale: sqrt(energy), not to scale; arrows: tabulated E1 lines", size=9
        )
    )
    return cv.Canvas(shapes, width=width, height=height)


def sublevel_fan(rows: Sequence[tuple[str, float]], *, title: str) -> ft.Control:
    """Zeeman sublevels of one level as bars at their energies relative to the lowest, labelled (to scale within the level)."""
    width, height = 420.0, 150.0
    if not rows:
        return ft.Container()
    energies = np.asarray([e for _l, e in rows], dtype=float)
    lo, hi = float(np.min(energies)), float(np.max(energies))
    ax = Axes(width, height, (0.0, 1.0), (lo, hi if hi > lo else lo + 1.0), margin=(8.0, 20.0, 8.0, 8.0))
    shapes: list[cv.Shape] = [_label(8, 2, title, size=10)]
    n = len(rows)
    for k, (label, e) in enumerate(sorted(rows, key=lambda r: r[1])):
        x0 = 0.05 + 0.9 * k / max(n, 1)
        shapes.append(
            cv.Line(
                ax.x(x0), ax.y(e), ax.x(x0 + 0.85 / max(n, 1)), ax.y(e), paint=_stroke(ft.Colors.PRIMARY, 2.0)
            )
        )
        shapes.append(_label(ax.x(x0), ax.y(e) - 12, f"{label}: {(e - lo) / 1e6:+.3f} MHz", size=8))
    return cv.Canvas(shapes, width=width, height=height)


def crystal_picture(
    positions_m: np.ndarray,
    eigenvector: np.ndarray | None,
    e_hat: tuple[float, float, float] | None,
    *,
    species: Sequence[str] = (),
    title: str = "",
) -> ft.Control:
    """The chain to scale along the trap axis (z) with the mode's displacement pattern as arrows: c_{i,m} along e_hat,
    drawn in the z (horizontal) and x (vertical) plane; a y component is written beside the arrow."""
    width, height = 520.0, 170.0
    pos = np.asarray(positions_m, dtype=float)
    z = pos[:, 2]
    span = float(np.max(z) - np.min(z)) if z.size > 1 else 1e-6
    span = max(span, 1e-7)
    ax = Axes(
        width,
        height,
        (float(np.min(z)) - 0.35 * span, float(np.max(z)) + 0.35 * span),
        (-1.0, 1.0),
        margin=(12.0, 18.0, 26.0, 12.0),
    )
    shapes: list[cv.Shape] = [
        cv.Line(
            ax.left, ax.y(0.0), ax.left + ax.plot_w, ax.y(0.0), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)
        )
    ]
    if title:
        shapes.append(_label(8, 2, title, size=10))
    for i in range(pos.shape[0]):
        shapes.append(cv.Circle(ax.x(float(z[i])), ax.y(0.0), 7, paint=_fill(ft.Colors.PRIMARY)))
        name = species[i] if i < len(species) else ""
        shapes.append(_label(ax.x(float(z[i])) - 10, ax.y(0.0) + 10, f"ion {i} {name}", size=9))
        if eigenvector is not None and e_hat is not None:
            c = float(eigenvector[i])
            dz, dx, dy = c * e_hat[2], c * e_hat[0], c * e_hat[1]
            end_z = float(z[i]) + dz * 0.3 * span
            shapes.extend(ax.arrow(float(z[i]), 0.0, end_z, dx * 0.85, ft.Colors.TERTIARY, 2.5))
            if abs(dy) > 1e-6:
                shapes.append(
                    _label(
                        ax.x(float(z[i])) + 8,
                        ax.y(0.0) - 26,
                        f"y: {dy:+.2f}",
                        size=8,
                        color=ft.Colors.TERTIARY,
                    )
                )
    shapes.append(
        _label(
            ax.left,
            height - 12,
            f"axis span {span * 1e6:.2f} um to scale; arrows: mass-weighted eigenvector components c_i,m (not to the position scale)",
            size=9,
        )
    )
    return cv.Canvas(shapes, width=width, height=height)


def stability_diagram(
    q: np.ndarray,
    a_lower: np.ndarray,
    a_upper: np.ndarray,
    points: Sequence[tuple[str, float, float]],
) -> ft.Control:
    """The first stability region of the Mathieu equation from the monodromy boundary, with the device's (q, a) per axis."""
    width, height = 420.0, 260.0
    ok = np.isfinite(a_lower) & np.isfinite(a_upper)
    qs, lo, hi = q[ok], a_lower[ok], a_upper[ok]
    ymin = float(min(np.min(lo), -0.05)) if lo.size else -0.5
    ymax = float(max(np.max(hi), 0.1)) if hi.size else 1.0
    ax = Axes(
        width,
        height,
        (0.0, float(np.max(q)) if q.size else 1.0),
        (ymin - 0.05, ymax + 0.05),
        margin=(44.0, 12.0, 26.0, 12.0),
    )
    shapes: list[cv.Shape] = ax.frame(
        "q",
        "a",
        x_ticks=_ticks(0.0, float(np.max(q)) if q.size else 1.0, 5, ".2f"),
        y_ticks=_ticks(ymin, ymax, 5, ".2f"),
    )
    if qs.size > 1:
        poly: list[cv.Path.PathElement] = [cv.Path.MoveTo(ax.x(float(qs[0])), ax.y(float(lo[0])))]
        for xv, yv in zip(qs[1:], lo[1:]):
            poly.append(cv.Path.LineTo(ax.x(float(xv)), ax.y(float(yv))))
        for xv, yv in zip(qs[::-1], hi[::-1]):
            poly.append(cv.Path.LineTo(ax.x(float(xv)), ax.y(float(yv))))
        poly.append(cv.Path.Close())
        shapes.append(
            cv.Path(
                poly,
                paint=ft.Paint(
                    color=ft.Colors.with_opacity(0.25, ft.Colors.PRIMARY), style=ft.PaintingStyle.FILL
                ),
            )
        )
        shapes.append(ax.polyline(qs, lo, ft.Colors.PRIMARY, 1.5))
        shapes.append(ax.polyline(qs, hi, ft.Colors.PRIMARY, 1.5))
        shapes.append(
            _label(
                ax.x(float(qs[len(qs) // 3])),
                ax.y(float(0.5 * (lo[len(qs) // 3] + hi[len(qs) // 3]))) - 6,
                "stable: 0 < beta < 1",
                size=10,
                color=ft.Colors.PRIMARY,
            )
        )
    for name, qv, av in points:
        px, py = ax.x(qv), ax.y(av)
        shapes.append(cv.Circle(px, py, 6, paint=_fill(ft.Colors.SURFACE_CONTAINER_LOWEST)))
        shapes.append(cv.Circle(px, py, 4.5, paint=_fill(ft.Colors.TERTIARY)))
        shapes.append(
            _label(px + 8, py - 6, f"{name}: q = {qv:.3f}, a = {av:.4f}", size=9, color=ft.Colors.TERTIARY)
        )
    return cv.Canvas(shapes, width=width, height=height)


def beam_geometry(
    positions_m: np.ndarray,
    beams: Sequence[tuple[int, tuple[float, float, float], tuple[float, float, float], str, str]],
    field_direction: tuple[float, float, float],
) -> ft.Control:
    """Top view (z along the chain horizontal, x vertical): ions to scale, each beam as an arrow along k_hat through its
    pointing, the quantization axis B drawn from the origin; a beam with a y component says so."""
    width, height = 520.0, 260.0
    pos = np.asarray(positions_m, dtype=float)
    span = max(float(np.max(pos[:, 2]) - np.min(pos[:, 2])) if pos.shape[0] > 1 else 1e-6, 1e-6)
    half = 1.6 * span
    cz = float(np.mean(pos[:, 2]))
    ax = Axes(
        width,
        height,
        (cz - half, cz + half),
        (-half * height / width, half * height / width),
        margin=(12.0, 12.0, 24.0, 12.0),
    )
    shapes: list[cv.Shape] = []
    for i in range(pos.shape[0]):
        shapes.append(
            cv.Circle(ax.x(float(pos[i, 2])), ax.y(float(pos[i, 0])), 6, paint=_fill(ft.Colors.PRIMARY))
        )
        shapes.append(_label(ax.x(float(pos[i, 2])) - 8, ax.y(float(pos[i, 0])) + 8, f"ion {i}", size=9))
    for k, k_hat, pointing, color, label in beams:
        z0, x0 = float(pointing[2]), float(pointing[0])
        length = 0.9 * half
        start_z, start_x = z0 - k_hat[2] * length, x0 - k_hat[0] * length
        shapes.extend(
            ax.arrow(
                start_z, start_x, z0 + 0.35 * k_hat[2] * length, x0 + 0.35 * k_hat[0] * length, color, 1.5
            )
        )
        tag = label + (f" (y {k_hat[1]:+.2f})" if abs(k_hat[1]) > 1e-6 else "")
        shapes.append(_label(ax.x(start_z) + 4, ax.y(start_x) - 12 - 10 * (k % 3), tag, size=8, color=color))
    bx, bz = float(field_direction[0]), float(field_direction[2])
    shapes.extend(
        ax.arrow(
            cz + 0.6 * half,
            -0.9 * half * height / width,
            cz + 0.6 * half + 0.25 * half * bz,
            -0.9 * half * height / width + 0.25 * half * bx,
            ft.Colors.ON_SURFACE,
            2.0,
        )
    )
    shapes.append(
        _label(
            ax.x(cz + 0.6 * half),
            ax.y(-0.9 * half * height / width) + 4,
            f"B (y {field_direction[1]:+.2f})",
            size=9,
        )
    )
    shapes.append(
        _label(
            ax.left,
            height - 12,
            f"z along the chain, x up; ions to scale over {span * 1e6:.1f} um; beams from their pointing along k",
            size=9,
        )
    )
    return cv.Canvas(shapes, width=width, height=height)


@dataclass(frozen=True)
class PhaseLoop:
    """One curve of the phase-space drawing: ``color_index`` picks the mode's colour, ``dashed`` marks the pair's second ion."""

    label: str
    alpha: np.ndarray
    note: str
    color_index: int = 0
    dashed: bool = False


def phase_space(loops: Sequence[PhaseLoop], *, width: float = 420.0, height: float = 300.0) -> ft.Control:
    """Spin-branch loops alpha_im(t) in the complex plane: a closed loop returns to its start (the filled dot); the end is a
    ring. One colour per mode; the pair's second ion dashed (its loop coincides with the first ion's on an in-phase mode and
    mirrors it on an out-of-phase one)."""
    palette = list(theme.series())
    filled = [np.asarray(lp.alpha, dtype=complex) for lp in loops if np.asarray(lp.alpha).size]
    allv = np.concatenate(filled) if filled else np.zeros(1, complex)
    r = float(max(np.max(np.abs(allv.real)), np.max(np.abs(allv.imag)), 1e-3)) * 1.15
    # the frame stops 14 px above the canvas bottom so that the footer line sits below the x-axis title
    ax = Axes(width, height - 14.0, (-r, r), (-r, r), margin=(44.0, 12.0, 26.0, 12.0))
    shapes: list[cv.Shape] = ax.frame(
        "Re alpha", "Im alpha", x_ticks=_ticks(-r, r, 3, ".2f"), y_ticks=_ticks(-r, r, 3, ".2f")
    )
    shapes.append(
        cv.Line(ax.x(-r), ax.y(0.0), ax.x(r), ax.y(0.0), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0))
    )
    shapes.append(
        cv.Line(ax.x(0.0), ax.y(-r), ax.x(0.0), ax.y(r), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0))
    )
    for k, lp in enumerate(loops):
        a = np.asarray(lp.alpha, dtype=complex)
        if a.size == 0:
            continue
        color = palette[lp.color_index % len(palette)]
        shapes.append(ax.polyline(a.real, a.imag, color, 1.8, dash=(6.0, 4.0) if lp.dashed else None))
        shapes.append(
            cv.Circle(ax.x(float(a.real[0])), ax.y(float(a.imag[0])), 4, paint=_fill(ft.Colors.ON_SURFACE))
        )
        shapes.append(
            cv.Circle(ax.x(float(a.real[-1])), ax.y(float(a.imag[-1])), 4, paint=_stroke(color, 2.0))
        )
        shapes.append(_label(ax.left + 6, ax.top + 4 + 13 * k, f"{lp.label}: {lp.note}", size=9, color=color))
    shapes.append(
        _label(
            ax.left,
            height - 12,
            "filled dot: start; ring: end; a closed loop ends on its start; dashed: the pair's second ion",
            size=9,
        )
    )
    return cv.Canvas(shapes, width=width, height=height)


# ---- charts with flet-charts -------------------------------------------------------------------------------------------------------------

_cap = ft.BorderRadius.only(top_left=3, top_right=3)
"""A bar's rounded data-end, square at the baseline (the dataviz mark spec)."""


def line_chart(
    series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    x_title: str,
    y_title: str,
    height: float = 180.0,
    log_y: bool = False,
    log_x: bool = False,
    markers: Sequence[tuple[float, str]] = (),
) -> ft.Control:
    """Time series or curves; a log axis is drawn as log10 values with a title saying so (the chart library has none)."""
    palette = list(theme.series())
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for _label, x, y in series:
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if log_x:
            keep = xs > 0
            xs, ys = np.log10(xs[keep]), ys[keep]
        if log_y:
            keep = ys > 0
            xs, ys = xs[keep], np.log10(ys[keep])
        prepared.append((xs, ys))
    # the chart library prints raw tick values, which wrap onto two lines for tiny or huge ranges: a linear axis is drawn
    # in units of a power of ten and the axis title says which (a log axis already carries its scale)
    scale_note = ""
    top = max((float(np.nanmax(np.abs(ys))) for _xs, ys in prepared if ys.size), default=0.0)
    if not log_y and math.isfinite(top) and top > 0.0 and (top < 1e-2 or top >= 1e4):
        exponent = int(math.floor(math.log10(top)))
        prepared = [(xs, ys / 10.0**exponent) for xs, ys in prepared]
        scale_note = f" (x 1e{exponent})"
    data = [
        fc.LineChartData(
            points=[fc.LineChartDataPoint(x=float(a), y=float(b)) for a, b in zip(xs, ys)],
            stroke_width=2,
            color=palette[k % len(palette)],
            rounded_stroke_cap=True,
        )
        for k, (xs, ys) in enumerate(prepared)
    ]
    if not data:
        return ft.Text("nothing to plot", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
    chart: ft.Control = fc.LineChart(
        data_series=data,
        height=height,
        expand=True,
        left_axis=fc.ChartAxis(
            title=_axis_title(("log10 " if log_y else "") + y_title + scale_note), label_size=48
        ),
        bottom_axis=fc.ChartAxis(
            title=_axis_title(("log10 " if log_x else "") + x_title),
            label_size=24,
            show_min=False,
            show_max=False,
        ),
        horizontal_grid_lines=_grid(),
        interactive=True,
    )
    legend_items: list[ft.Control] = [
        ft.Row(
            [
                ft.Container(width=12, height=3, bgcolor=palette[k % len(palette)], border_radius=2),
                ft.Text(label, size=10, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        for k, (label, _x, _y) in enumerate(series)
    ]
    legend_items += [
        ft.Text(f"| {text}", size=10, color=ft.Colors.ON_SURFACE_VARIANT) for _v, text in markers
    ]
    return ft.Column([chart, ft.Row(legend_items, wrap=True, spacing=10)], spacing=4)


@dataclass(frozen=True)
class Bar:
    """One bar of a group: its height, its tooltip, and whether it is the outlined companion of a filled bar (the target
    beside the simulated, the start beside the end)."""

    value: float
    tooltip: str
    outlined: bool = False
    color: ft.ColorValue = ft.Colors.PRIMARY


def bar_chart(
    groups: Sequence[tuple[str, Sequence[Bar]]],
    *,
    bar_width: float,
    height: float,
    width: float | None = None,
    max_y: float | None = None,
    y_ticks: Sequence[float] = (),
    label_every: int = 1,
    x_title: str = "",
    y_title: str = "",
    on_tap: Callable[[int], None] | None = None,
    interactive: bool = True,
) -> ft.Control:
    """Grouped bars, thin with a rounded data end, on hairline gridlines (the dataviz mark specs); labelled below by group
    (every ``label_every``-th), on the left at ``y_ticks`` when given; ``on_tap`` receives a tapped group's index."""
    chart_groups = [
        fc.BarChartGroup(
            x=k,
            rods=[
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=b.value,
                    width=bar_width,
                    color=ft.Colors.with_opacity(0.12, b.color) if b.outlined else b.color,
                    border_side=ft.BorderSide(1.5, b.color) if b.outlined else None,
                    tooltip=b.tooltip,
                    border_radius=_cap,
                )
                for b in bars
            ],
            spacing=2,
        )
        for k, (_label, bars) in enumerate(groups)
    ]

    def on_event(e: fc.BarChartEvent) -> None:
        if on_tap is not None and e.type == fc.ChartEventType.TAP_UP and e.group_index is not None:
            on_tap(int(e.group_index))

    chart: ft.Control = fc.BarChart(
        groups=chart_groups,
        bottom_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(
                    value=k, label=ft.Text(label, size=theme.SIZE_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT)
                )
                for k, (label, _bars) in enumerate(groups)
                if k % label_every == 0
            ],
            label_size=24,
            title=_axis_title(x_title) if x_title else None,
        ),
        left_axis=fc.ChartAxis(
            labels=[
                fc.ChartAxisLabel(
                    value=v,
                    label=ft.Text(f"{v:g}", size=theme.SIZE_MICRO, color=ft.Colors.ON_SURFACE_VARIANT),
                )
                for v in y_ticks
            ],
            label_size=40,
            title=_axis_title(y_title) if y_title else None,
        ),
        horizontal_grid_lines=fc.ChartGridLines(
            interval=y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else None,
            color=ft.Colors.OUTLINE_VARIANT,
            width=1,
        ),
        max_y=max_y,
        min_y=0.0,
        height=height,
        width=width,
        expand=width is None,
        interactive=interactive,
        on_event=on_event if on_tap is not None else None,
    )
    return chart


def two_histograms(bright: np.ndarray, dark: np.ndarray, threshold: float | None) -> ft.Control:
    """Bright and dark count distributions side by side per photon number, the threshold named."""
    n = max(bright.size, dark.size)
    # the last populated photon number of either distribution (the bright one reaches further), plus a margin; at least
    # eight bins so a narrow pair still reads as a histogram
    last = max((int(np.flatnonzero(a > 1e-6).max(initial=-1)) for a in (bright, dark)), default=-1)
    n_show = int(min(n, max(last + 3, 8))) if n else 1
    groups = []
    for k in range(n_show):
        pb = float(bright[k]) if k < bright.size else 0.0
        pd = float(dark[k]) if k < dark.size else 0.0
        groups.append(
            (
                str(k),
                [
                    Bar(pb, f"bright: P({k}) = {pb:.3e}"),
                    Bar(pd, f"dark: P({k}) = {pd:.3e}", color=ft.Colors.TERTIARY),
                ],
            )
        )
    legend = ft.Row(
        [
            ft.Container(width=12, height=12, bgcolor=ft.Colors.PRIMARY, border_radius=3),
            ft.Text("bright ion", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Container(width=12, height=12, bgcolor=ft.Colors.TERTIARY, border_radius=3),
            ft.Text("dark ion", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(
                "" if threshold is None else f"| bright declared above n_c = {threshold:g}",
                size=11,
                color=ft.Colors.ON_SURFACE_VARIANT,
            ),
        ],
        spacing=6,
        wrap=True,
    )
    return ft.Column(
        [bar_chart(groups, bar_width=9, height=200, label_every=max(1, n_show // 10)), legend], spacing=4
    )


def fock_bars(start: np.ndarray, end: np.ndarray) -> ft.Control:
    """P(n) at the start (outlined) and the end (filled) of a step; a distribution given as an EMPTY array is not known
    and gets no bar, rather than a row of zeros that would read as a measured vacuum."""
    groups = []
    for k in range(min(8, max(start.size, end.size))):
        bars = []
        if start.size:
            a = float(start[k]) if k < start.size else 0.0
            bars.append(Bar(a, f"start: P({k}) = {a:.3e}", outlined=True, color=ft.Colors.TERTIARY))
        if end.size:
            b = float(end[k]) if k < end.size else 0.0
            bars.append(Bar(b, f"end: P({k}) = {b:.3e}"))
        groups.append((str(k), bars))
    return bar_chart(groups, bar_width=10, height=150, max_y=1.0)


def bloch_disc(
    ion: int, vector: tuple[float, float, float], *, target: tuple[float, float, float] | None
) -> ft.Control:
    """One ion's Bloch vector (<X>, <Y>, <Z>) as an arrow in the x-z disc (the equator horizontal, |0> up), the y component
    written beside it and the arrow as long as the vector (a shrunken arrow is mixture or entanglement); the target vector,
    when given, dashed."""
    size = 120.0
    x, y, z = (float(v) for v in vector)
    r = size / 2.0 - 8.0
    cx, cy = size / 2.0, size / 2.0
    shapes: list[cv.Shape] = [
        cv.Circle(cx, cy, r, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        cv.Line(cx - r, cy, cx + r, cy, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 0.6, [3, 3])),
        cv.Line(cx, cy - r, cx, cy + r, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 0.6, [3, 3])),
        _label(cx - 6, 0.0, "|0>", size=9),
        _label(cx - 6, size - 12.0, "|1>", size=9),
        _label(size - 14.0, cy - 6, "+x", size=9),
    ]
    ax = Axes(size, size, (-1.0, 1.0), (-1.0, 1.0), margin=(8.0, 8.0, 8.0, 8.0))
    if target is not None:
        tx, _ty, tz = (float(v) for v in target)
        if math.hypot(tx, tz) > 1e-6:
            shapes.append(
                cv.Line(
                    ax.x(0.0), ax.y(0.0), ax.x(tx), ax.y(tz), paint=_stroke(ft.Colors.TERTIARY, 1.5, [4, 3])
                )
            )
    length = math.sqrt(x * x + y * y + z * z)
    if math.hypot(x, z) > 1e-6:
        shapes.extend(ax.arrow(0.0, 0.0, x, z, ft.Colors.PRIMARY, 2.5))
    shapes.append(cv.Circle(cx, cy, 2.5, paint=_fill(ft.Colors.ON_SURFACE)))
    canvas = cv.Canvas(shapes, width=size, height=size)
    return ft.Column(
        [
            canvas,
            ft.Text(f"ion {ion}", size=12, weight=ft.FontWeight.W_600),
            ft.Text(f"length {length:.3f}, <Y> {y:+.2f}", size=10, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        spacing=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )
