"""Drawings computed from the view-models: heatmaps as PNG images, phase-space loops and Bloch discs on Flet's canvas, and
line and bar charts with flet-charts. Colours are theme roles or mid-tone named colours, so both brightness modes read."""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import flet as ft
import flet.canvas as cv
import flet_charts as fc
import numpy as np

SERIES: tuple[str, ...] = (
    ft.Colors.BLUE,
    ft.Colors.ORANGE,
    ft.Colors.GREEN,
    ft.Colors.PURPLE,
    ft.Colors.PINK,
    ft.Colors.TEAL,
)
"""Categorical colours, assigned in this order."""

LABEL = ft.Colors.ON_SURFACE_VARIANT

# ---- heatmaps --------------------------------------------------------------------------------------------------------------------

_VIRIDIS: tuple[tuple[float, float, float], ...] = (
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


def _ramp(t: float) -> tuple[int, int, int]:
    x = min(max(float(t), 0.0), 1.0) * (len(_VIRIDIS) - 1)
    k = min(int(math.floor(x)), len(_VIRIDIS) - 2)
    a, b = _VIRIDIS[k], _VIRIDIS[k + 1]
    r, g, bl = (int(round(255 * (a[i] + (x - k) * (b[i] - a[i])))) for i in range(3))
    return r, g, bl


def png_bytes(rgba: np.ndarray) -> bytes:
    """A PNG from an (H, W, 4) uint8 array."""
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(arr.shape[0]))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", arr.shape[1], arr.shape[0], 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def heatmap_png(values: np.ndarray, *, cell_w: int = 14, cell_h: int = 10, log: bool = True) -> bytes:
    """Row 0 at the bottom; a log scale over six decades by default, since Fock populations span 1 to 1e-6."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    top = max(float(np.max(v, initial=0.0)), 1e-300)
    if log:
        with np.errstate(divide="ignore"):
            t = (np.log10(np.clip(v, 1e-300, None)) - (math.log10(top) - 6.0)) / 6.0
    else:
        t = v / top
    t = np.clip(t, 0.0, 1.0)
    rows, cols = t.shape
    img = np.zeros((rows * cell_h, cols * cell_w, 4), dtype=np.uint8)
    for r in range(rows):
        y0 = (rows - 1 - r) * cell_h
        for c in range(cols):
            img[y0 : y0 + cell_h, c * cell_w : (c + 1) * cell_w, :3] = _ramp(float(t[r, c]))
    img[..., 3] = 255
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
) -> ft.Control:
    """A heatmap image with its row labels on the left, column labels beneath and a colour legend."""
    rows, cols = values.shape
    step = max(1, cols // 8)
    return ft.Column(
        [
            ft.Text(y_title, size=10, color=LABEL),
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(str(y_labels[r]), size=9, height=cell_h, color=LABEL)
                            for r in reversed(range(rows))
                        ],
                        spacing=0,
                        tight=True,
                    ),
                    ft.Image(
                        src=heatmap_png(values, cell_w=cell_w, cell_h=cell_h),
                        width=cols * cell_w,
                        height=rows * cell_h,
                        fit=ft.BoxFit.FILL,
                    ),
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Row(
                [ft.Container(width=34)]
                + [
                    ft.Text(
                        str(x_labels[c]) if c % step == 0 else "",
                        size=9,
                        width=cell_w,
                        no_wrap=True,
                        color=LABEL,
                    )
                    for c in range(cols)
                ],
                spacing=0,
                tight=True,
            ),
            ft.Row(
                [
                    ft.Text(x_title, size=10, color=LABEL),
                    ft.Image(
                        src=heatmap_png(np.logspace(-6, 0, 12)[None, :], cell_w=12, cell_h=8),
                        width=144,
                        height=8,
                        fit=ft.BoxFit.FILL,
                    ),
                    ft.Text("1e-6 to 1 (log)", size=9, color=LABEL),
                ],
                spacing=8,
            ),
        ],
        spacing=2,
        tight=True,
    )


# ---- canvas drawings -------------------------------------------------------------------------------------------------------------


def _stroke(color: str, width: float = 1.5, dash: Sequence[float] | None = None) -> ft.Paint:
    return ft.Paint(
        color=color,
        stroke_width=width,
        style=ft.PaintingStyle.STROKE,
        stroke_dash_pattern=None if dash is None else [float(d) for d in dash],
    )


def _fill(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _label(x: float, y: float, text: str, *, size: int = 9, color: str = LABEL) -> cv.Text:
    return cv.Text(x, y, text, style=ft.TextStyle(size=size, color=color))


class Axes:
    """A data-to-pixel map for one canvas with margins (left, top, bottom, right)."""

    def __init__(
        self,
        width: float,
        height: float,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        margin: tuple[float, float, float, float] = (36.0, 10.0, 22.0, 10.0),
    ) -> None:
        self.left, self.top, self.bottom, self.right = margin
        self.plot_w = width - self.left - self.right
        self.plot_h = height - self.top - self.bottom
        self.xlim, self.ylim = xlim, ylim

    def x(self, v: float) -> float:
        return self.left + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.plot_w

    def y(self, v: float) -> float:
        return self.top + (self.ylim[1] - v) / (self.ylim[1] - self.ylim[0]) * self.plot_h

    def polyline(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        color: str,
        width: float = 1.5,
        dash: Sequence[float] | None = None,
    ) -> cv.Path:
        elements: list[cv.Path.PathElement] = [
            cv.Path.MoveTo(self.x(float(a)), self.y(float(b)))
            if k == 0
            else cv.Path.LineTo(self.x(float(a)), self.y(float(b)))
            for k, (a, b) in enumerate(zip(xs, ys))
        ]
        return cv.Path(elements, paint=_stroke(color, width, dash))

    def arrow(
        self, x0: float, y0: float, x1: float, y1: float, color: str, width: float = 2.0
    ) -> list[cv.Shape]:
        px0, py0, px1, py1 = self.x(x0), self.y(y0), self.x(x1), self.y(y1)
        out: list[cv.Shape] = [cv.Line(px0, py0, px1, py1, paint=_stroke(color, width))]
        length = math.hypot(px1 - px0, py1 - py0)
        if length > 1e-9:
            ux, uy = (px1 - px0) / length, (py1 - py0) / length
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


@dataclass(frozen=True)
class PhaseLoop:
    label: str
    alpha: np.ndarray
    color_index: int = 0
    dashed: bool = False


def phase_space(loops: Sequence[PhaseLoop], *, width: float = 440.0, height: float = 320.0) -> ft.Control:
    """Loops alpha(t) in the complex plane, one colour per mode: a filled dot at the start, a ring at the end; a closed loop
    ends on its start."""
    nonempty = [np.asarray(lp.alpha, dtype=complex) for lp in loops if lp.alpha.size]
    allv = np.concatenate(nonempty) if nonempty else np.zeros(1, complex)
    r = float(max(np.max(np.abs(allv.real)), np.max(np.abs(allv.imag)), 1e-3)) * 1.15
    ax = Axes(width, height - 14.0, (-r, r), (-r, r), margin=(44.0, 12.0, 26.0, 12.0))
    shapes: list[cv.Shape] = [
        cv.Rect(ax.left, ax.top, ax.plot_w, ax.plot_h, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        cv.Line(ax.x(-r), ax.y(0.0), ax.x(r), ax.y(0.0), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        cv.Line(ax.x(0.0), ax.y(-r), ax.x(0.0), ax.y(r), paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        _label(ax.left + ax.plot_w / 2 - 20, height - 30, "Re alpha", size=10),
        _label(4, 2, "Im alpha", size=10),
        _label(ax.left, height - 12, "dot: start; ring: end; dashed: the pair's second ion"),
    ]
    for v in (-r, r):
        shapes.append(_label(ax.x(v) - 12, ax.top + ax.plot_h + 4, f"{v:.2f}"))
        shapes.append(_label(2, ax.y(v) - 6, f"{v:.2f}"))
    for k, lp in enumerate(loops):
        a = np.asarray(lp.alpha, dtype=complex)
        if a.size == 0:
            continue
        color = SERIES[lp.color_index % len(SERIES)]
        shapes.append(ax.polyline(a.real, a.imag, color, 1.8, dash=(6.0, 4.0) if lp.dashed else None))
        shapes.append(
            cv.Circle(ax.x(float(a.real[0])), ax.y(float(a.imag[0])), 4, paint=_fill(ft.Colors.ON_SURFACE))
        )
        shapes.append(
            cv.Circle(ax.x(float(a.real[-1])), ax.y(float(a.imag[-1])), 4, paint=_stroke(color, 2.0))
        )
        shapes.append(_label(ax.left + 6, ax.top + 4 + 13 * k, lp.label, color=color))
    return cv.Canvas(shapes, width=width, height=height)


def bloch_disc(
    ion: int,
    vector: tuple[float, float, float],
    target: tuple[float, float, float] | None = None,
    size: float = 120.0,
) -> ft.Control:
    """One ion's Bloch vector as an arrow in the x-z disc (|0> up), its length the vector's; the ideal vector dashed."""
    x, y, z = vector
    rad = size / 2.0 - 8.0
    c = size / 2.0
    ax = Axes(size, size, (-1.0, 1.0), (-1.0, 1.0), margin=(8.0, 8.0, 8.0, 8.0))
    shapes: list[cv.Shape] = [
        cv.Circle(c, c, rad, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 1.0)),
        cv.Line(c - rad, c, c + rad, c, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 0.6, [3, 3])),
        cv.Line(c, c - rad, c, c + rad, paint=_stroke(ft.Colors.OUTLINE_VARIANT, 0.6, [3, 3])),
        _label(c - 6, 0.0, "|0>"),
        _label(c - 6, size - 12.0, "|1>"),
        _label(size - 14.0, c - 6, "+x"),
    ]
    if target is not None and math.hypot(target[0], target[2]) > 1e-6:
        shapes.append(
            cv.Line(
                ax.x(0.0),
                ax.y(0.0),
                ax.x(target[0]),
                ax.y(target[2]),
                paint=_stroke(ft.Colors.TERTIARY, 1.5, [4, 3]),
            )
        )
    if math.hypot(x, z) > 1e-6:
        shapes.extend(ax.arrow(0.0, 0.0, x, z, ft.Colors.PRIMARY, 2.5))
    shapes.append(cv.Circle(c, c, 2.5, paint=_fill(ft.Colors.ON_SURFACE)))
    return ft.Column(
        [
            cv.Canvas(shapes, width=size, height=size),
            ft.Text(f"ion {ion}", size=12, weight=ft.FontWeight.W_600),
            ft.Text(f"length {math.sqrt(x * x + y * y + z * z):.3f}, <Y> {y:+.2f}", size=10, color=LABEL),
        ],
        spacing=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )


# ---- charts ------------------------------------------------------------------------------------------------------------------------


def _grid() -> fc.ChartGridLines:
    return fc.ChartGridLines(color=ft.Colors.OUTLINE_VARIANT, width=1)


def line_chart(
    series: Sequence[tuple[str, np.ndarray, np.ndarray]], *, x_title: str, y_title: str, height: float = 180.0
) -> ft.Control:
    """Curves with a legend; an axis whose values are tiny or huge is drawn in units of a power of ten, named in its title."""
    prepared = [(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for _label, x, y in series]
    if not prepared:
        return ft.Text("nothing to plot", size=12, italic=True, color=LABEL)
    top = max((float(np.nanmax(np.abs(ys))) for _xs, ys in prepared if ys.size), default=0.0)
    scale_note = ""
    if math.isfinite(top) and top > 0.0 and (top < 1e-2 or top >= 1e4):
        exponent = int(math.floor(math.log10(top)))
        prepared = [(xs, ys / 10.0**exponent) for xs, ys in prepared]
        scale_note = f" (x 1e{exponent})"
    chart = fc.LineChart(
        data_series=[
            fc.LineChartData(
                points=[fc.LineChartDataPoint(x=float(a), y=float(b)) for a, b in zip(xs, ys)],
                stroke_width=2,
                color=SERIES[k % len(SERIES)],
                rounded_stroke_cap=True,
            )
            for k, (xs, ys) in enumerate(prepared)
        ],
        height=height,
        expand=True,
        left_axis=fc.ChartAxis(title=ft.Text(y_title + scale_note, size=10, color=LABEL), label_size=48),
        bottom_axis=fc.ChartAxis(
            title=ft.Text(x_title, size=10, color=LABEL), label_size=24, show_min=False, show_max=False
        ),
        horizontal_grid_lines=_grid(),
        interactive=True,
    )
    legend = ft.Row(
        [
            ft.Row(
                [
                    ft.Container(width=12, height=3, bgcolor=SERIES[k % len(SERIES)], border_radius=2),
                    ft.Text(label, size=10, color=LABEL),
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
            for k, (label, _x, _y) in enumerate(series)
        ],
        wrap=True,
        spacing=10,
    )
    return ft.Column([chart, legend], spacing=4)


_CAP = ft.BorderRadius.only(top_left=3, top_right=3)


def fock_bars(start: np.ndarray, end: np.ndarray, *, height: float = 150.0, n_show: int = 8) -> ft.Control:
    """P(n) at the start (outlined) and the end (filled) of a step; an empty array is a distribution not known, drawn with
    no bars rather than as a vacuum."""
    n = min(n_show, max(start.size, end.size))
    groups = []
    for k in range(n):
        rods = []
        if start.size:
            a = float(start[k]) if k < start.size else 0.0
            rods.append(
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=a,
                    width=10,
                    color=ft.Colors.with_opacity(0.12, ft.Colors.TERTIARY),
                    border_side=ft.BorderSide(1.5, ft.Colors.TERTIARY),
                    tooltip=f"start: P({k}) = {a:.3e}",
                    border_radius=_CAP,
                )
            )
        if end.size:
            b = float(end[k]) if k < end.size else 0.0
            rods.append(
                fc.BarChartRod(
                    from_y=0.0,
                    to_y=b,
                    width=10,
                    color=ft.Colors.PRIMARY,
                    tooltip=f"end: P({k}) = {b:.3e}",
                    border_radius=_CAP,
                )
            )
        groups.append(fc.BarChartGroup(x=k, rods=rods, spacing=1))
    chart: ft.Control = fc.BarChart(
        groups=groups,
        bottom_axis=fc.ChartAxis(
            labels=[fc.ChartAxisLabel(value=k, label=ft.Text(str(k), size=9, color=LABEL)) for k in range(n)],
            label_size=20,
        ),
        left_axis=fc.ChartAxis(label_size=40),
        horizontal_grid_lines=_grid(),
        max_y=1.0,
        min_y=0.0,
        height=height,
        expand=True,
        interactive=True,
    )
    return chart
