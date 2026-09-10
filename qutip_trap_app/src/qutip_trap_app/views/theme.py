"""The design system of the screens (DESIGN.md Section 4 "Layout"): one place for every colour, size and shape.

Two colour schemes, light and dark, each a set of Material 3 roles chosen by hand and checked by computation (every text
on its background at or above 4.5:1, every outline at or above 3:1, in both modes); a restrained type scale; the shapes
and spacings of the 8 pt grid; and the palettes the charts draw with (the categorical series validated for colour-vision
deficiency against each mode's card surface, following the dataviz rules). Views take colours from the theme roles
(``ft.Colors.PRIMARY`` and its kin), which the client resolves per mode; the few colours that have no Material role, the
status colours of the badges and the series of the charts, come from :func:`status` and :func:`series`, which read the
page's brightness at render time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flet as ft

# ---- the grid, the shapes and the type scale ------------------------------------------------------------------------------------------

GAP = 8
"""The unit of the 8 pt grid; every spacing is a multiple of it or its 4 px half-step."""

PAGE_PADDING = 24
CARD_PADDING = 16
RADIUS_CARD = 12
RADIUS_TILE = 8
RAIL_WIDTH = 84
DRAWER_WIDTH = 380
CONTENT_MAX_WIDTH = 1280
"""Wider windows centre the content column so that reading measures stay sane on a large display."""

SIZE_TITLE = 24
SIZE_CARD_TITLE = 15
SIZE_VALUE = 22
SIZE_BODY = 14
SIZE_SMALL = 12
SIZE_CAPTION = 11
SIZE_MICRO = 10


@dataclass(frozen=True)
class Tokens:
    """One mode's colours: the Material roles the scheme sets, plus the status and series colours that have no role."""

    surface: str
    card: str
    low: str
    container: str
    high: str
    highest: str
    bright: str
    dim: str
    on_surface: str
    on_surface_variant: str
    outline: str
    outline_variant: str
    inverse_surface: str
    on_inverse_surface: str
    primary: str
    on_primary: str
    primary_container: str
    on_primary_container: str
    inverse_primary: str
    secondary: str
    on_secondary: str
    secondary_container: str
    on_secondary_container: str
    tertiary: str
    on_tertiary: str
    tertiary_container: str
    on_tertiary_container: str
    error: str
    on_error: str
    error_container: str
    on_error_container: str
    status: dict[str, tuple[str, str]]
    """Badge colours per status word: (background, foreground); the word and an icon carry the meaning, never the colour alone."""
    series: tuple[str, ...]
    """Categorical chart colours in a fixed order (assigned by entity, never cycled past the sixth)."""
    tones: dict[str, str]
    """The sideband roles on the mode spectrum: red and blue by physics convention, the carrier and far tones neutral."""


LIGHT = Tokens(
    surface="#F7F7F5",
    card="#FFFFFF",
    low="#F1F1EE",
    container="#ECECE9",
    high="#E6E6E2",
    highest="#DFDFDB",
    bright="#FFFFFF",
    dim="#D9D9D5",
    on_surface="#1A1A1A",
    on_surface_variant="#5C5C59",
    outline="#86867F",
    outline_variant="#DCDCD7",
    inverse_surface="#2E2E2C",
    on_inverse_surface="#F1F1EE",
    primary="#2B5FC7",
    on_primary="#FFFFFF",
    primary_container="#DCE7FB",
    on_primary_container="#10305E",
    inverse_primary="#8FB6F5",
    secondary="#4E6178",
    on_secondary="#FFFFFF",
    secondary_container="#E1E7EF",
    on_secondary_container="#253244",
    tertiary="#C2325A",
    on_tertiary="#FFFFFF",
    tertiary_container="#FCDDE5",
    on_tertiary_container="#5A0F25",
    error="#B3261E",
    on_error="#FFFFFF",
    error_container="#F9DEDC",
    on_error_container="#410E0B",
    status={
        "pass": ("#DDF3E3", "#0F5A2E"),
        "fail": ("#F9DEDC", "#7F1D18"),
        "not checked": ("#E8E8E4", "#4A4A47"),
        "stale": ("#FBECC8", "#6A4700"),
        "info": ("#DCE7FB", "#10305E"),
    },
    series=("#2B5FC7", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#4A3AA7"),
    tones={"red": "#D0342C", "blue": "#2B5FC7", "carrier": "#7A7A75", "far": "#B5B5B0"},
)

DARK = Tokens(
    surface="#131314",
    card="#1B1B1D",
    low="#1F1F21",
    container="#252527",
    high="#2D2D30",
    highest="#36363A",
    bright="#3A3A3E",
    dim="#0F0F10",
    on_surface="#E4E4E1",
    on_surface_variant="#A9A9A4",
    outline="#7C7C77",
    outline_variant="#333336",
    inverse_surface="#E4E4E1",
    on_inverse_surface="#2E2E2C",
    primary="#8FB6F5",
    on_primary="#0A2B63",
    primary_container="#23457F",
    on_primary_container="#D9E5FF",
    inverse_primary="#2B5FC7",
    secondary="#AEBCD3",
    on_secondary="#1F2B3C",
    secondary_container="#354356",
    on_secondary_container="#DCE5F3",
    tertiary="#F393AE",
    on_tertiary="#5A0F25",
    tertiary_container="#7A2543",
    on_tertiary_container="#FFD9E2",
    error="#F2B8B5",
    on_error="#601410",
    error_container="#8C1D18",
    on_error_container="#F9DEDC",
    status={
        "pass": ("#1F3D2B", "#A7E3BC"),
        "fail": ("#5C1F1A", "#F6BDB9"),
        "not checked": ("#2D2D30", "#C9C9C4"),
        "stale": ("#4A3A12", "#F5D48A"),
        "info": ("#23457F", "#D9E5FF"),
    },
    series=("#3987E5", "#D95926", "#199E70", "#C98500", "#D55181", "#9085E9"),
    tones={"red": "#E66767", "blue": "#3987E5", "carrier": "#A0A09A", "far": "#5A5A57"},
)


def is_dark(page: Any | None = None) -> bool:
    """Whether the client renders the dark scheme: the page's own mode when set, else the platform's brightness."""
    if page is None:
        try:
            page = ft.context.page
        except Exception:  # outside a render (tests, headless use): the light scheme
            return False
    mode = getattr(page, "theme_mode", None)
    if mode == ft.ThemeMode.DARK:
        return True
    if mode == ft.ThemeMode.LIGHT:
        return False
    return getattr(page, "platform_brightness", None) == ft.Brightness.DARK


def tokens(page: Any | None = None) -> Tokens:
    return DARK if is_dark(page) else LIGHT


def series(page: Any | None = None) -> tuple[str, ...]:
    """The categorical chart colours of the current mode, in their fixed order."""
    return tokens(page).series


def status(kind: str, page: Any | None = None) -> tuple[str, str]:
    """(background, foreground) of a badge for a status word: pass, fail, not checked, stale or info."""
    return tokens(page).status[kind]


def tone_color(role: str, page: Any | None = None) -> str:
    t = tokens(page)
    return t.tones.get(role, t.tones["far"])


# ---- the Flet theme ---------------------------------------------------------------------------------------------------------------------


def _scheme(t: Tokens) -> ft.ColorScheme:
    return ft.ColorScheme(
        primary=t.primary,
        on_primary=t.on_primary,
        primary_container=t.primary_container,
        on_primary_container=t.on_primary_container,
        inverse_primary=t.inverse_primary,
        secondary=t.secondary,
        on_secondary=t.on_secondary,
        secondary_container=t.secondary_container,
        on_secondary_container=t.on_secondary_container,
        tertiary=t.tertiary,
        on_tertiary=t.on_tertiary,
        tertiary_container=t.tertiary_container,
        on_tertiary_container=t.on_tertiary_container,
        error=t.error,
        on_error=t.on_error,
        error_container=t.error_container,
        on_error_container=t.on_error_container,
        surface=t.surface,
        on_surface=t.on_surface,
        on_surface_variant=t.on_surface_variant,
        surface_container_lowest=t.card,
        surface_container_low=t.low,
        surface_container=t.container,
        surface_container_high=t.high,
        surface_container_highest=t.highest,
        surface_bright=t.bright,
        surface_dim=t.dim,
        surface_tint=t.primary,
        outline=t.outline,
        outline_variant=t.outline_variant,
        inverse_surface=t.inverse_surface,
        on_inverse_surface=t.on_inverse_surface,
        shadow="#000000",
        scrim="#000000",
    )


def _rounded(radius: float, side: ft.BorderSide | None = None) -> ft.RoundedRectangleBorder:
    return ft.RoundedRectangleBorder(radius=radius, side=side)


def build_theme(dark: bool) -> ft.Theme:
    """The theme of one mode: the scheme above, the type scale, and every component shaped to the same grid."""
    t = DARK if dark else LIGHT
    hairline = ft.BorderSide(1, t.outline_variant)
    button_shape = _rounded(RADIUS_TILE)
    button_text = ft.TextStyle(size=SIZE_BODY, weight=ft.FontWeight.W_500)
    return ft.Theme(
        color_scheme_seed=t.primary,
        color_scheme=_scheme(t),
        use_material3=True,
        visual_density=ft.VisualDensity.STANDARD,
        scaffold_bgcolor=t.surface,
        canvas_color=t.surface,
        card_bgcolor=t.card,
        divider_color=t.outline_variant,
        # Flet replaces (never merges) a theme text style, so every style names its colour or renders with none
        text_theme=ft.TextTheme(
            headline_small=ft.TextStyle(
                size=SIZE_TITLE,
                weight=ft.FontWeight.W_600,
                letter_spacing=-0.3,
                height=1.2,
                color=t.on_surface,
            ),
            title_large=ft.TextStyle(size=20, weight=ft.FontWeight.W_600, height=1.25, color=t.on_surface),
            title_medium=ft.TextStyle(size=16, weight=ft.FontWeight.W_600, height=1.3, color=t.on_surface),
            title_small=ft.TextStyle(
                size=SIZE_CARD_TITLE, weight=ft.FontWeight.W_600, height=1.3, color=t.on_surface
            ),
            body_large=ft.TextStyle(size=16, height=1.5, color=t.on_surface),
            body_medium=ft.TextStyle(size=SIZE_BODY, height=1.45, color=t.on_surface),
            body_small=ft.TextStyle(size=SIZE_SMALL, height=1.4, color=t.on_surface),
            label_large=ft.TextStyle(size=SIZE_BODY, weight=ft.FontWeight.W_500, color=t.on_surface),
            label_medium=ft.TextStyle(size=SIZE_SMALL, weight=ft.FontWeight.W_500, color=t.on_surface),
            label_small=ft.TextStyle(size=SIZE_CAPTION, weight=ft.FontWeight.W_500, color=t.on_surface),
        ),
        card_theme=ft.CardTheme(
            elevation=0, color=t.card, shape=_rounded(RADIUS_CARD, hairline), margin=ft.Margin.all(0)
        ),
        navigation_rail_theme=ft.NavigationRailTheme(
            bgcolor=t.surface,
            elevation=0,
            indicator_color=t.secondary_container,
            indicator_shape=_rounded(16),
            use_indicator=True,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=RAIL_WIDTH,
            selected_label_text_style=ft.TextStyle(
                size=SIZE_CAPTION, weight=ft.FontWeight.W_600, color=t.on_surface
            ),
            unselected_label_text_style=ft.TextStyle(size=SIZE_CAPTION, color=t.on_surface_variant),
        ),
        expansion_tile_theme=ft.ExpansionTileTheme(
            shape=_rounded(RADIUS_TILE),
            collapsed_shape=_rounded(RADIUS_TILE),
            tile_padding=ft.Padding.symmetric(horizontal=GAP),
            controls_padding=ft.Padding.only(bottom=GAP),
            bgcolor=ft.Colors.TRANSPARENT,
            collapsed_bgcolor=ft.Colors.TRANSPARENT,
            icon_color=t.on_surface_variant,
            collapsed_icon_color=t.on_surface_variant,
            text_color=t.on_surface,
            collapsed_text_color=t.on_surface_variant,
        ),
        chip_theme=ft.ChipTheme(
            shape=ft.StadiumBorder(),
            bgcolor=t.card,
            selected_color=t.primary_container,
            border_side=hairline,
            elevation=0,
            show_checkmark=False,
            label_text_style=ft.TextStyle(size=SIZE_SMALL, color=t.on_surface),
            padding=ft.Padding.symmetric(horizontal=4, vertical=2),
            label_padding=ft.Padding.symmetric(horizontal=6),
        ),
        tooltip_theme=ft.TooltipTheme(
            text_style=ft.TextStyle(size=SIZE_SMALL, color=t.on_inverse_surface, height=1.4),
            decoration=ft.BoxDecoration(bgcolor=t.inverse_surface, border_radius=ft.BorderRadius.all(6)),
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            margin=ft.Margin.all(GAP),
            wait_duration=ft.Duration(milliseconds=350),
        ),
        divider_theme=ft.DividerTheme(color=t.outline_variant, thickness=1, space=1),
        data_table_theme=ft.DataTableTheme(
            heading_text_style=ft.TextStyle(
                size=SIZE_CAPTION, weight=ft.FontWeight.W_600, color=t.on_surface_variant
            ),
            data_text_style=ft.TextStyle(size=SIZE_SMALL, color=t.on_surface),
            heading_row_height=32,
            data_row_min_height=30,
            data_row_max_height=44,
            column_spacing=20,
            horizontal_margin=GAP,
            divider_thickness=1,
        ),
        text_button_theme=ft.TextButtonTheme(
            style=ft.ButtonStyle(
                shape=button_shape,
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                text_style=button_text,
            )
        ),
        filled_button_theme=ft.FilledButtonTheme(
            style=ft.ButtonStyle(
                shape=button_shape,
                padding=ft.Padding.symmetric(horizontal=18, vertical=10),
                text_style=button_text,
            )
        ),
        outlined_button_theme=ft.OutlinedButtonTheme(
            style=ft.ButtonStyle(
                shape=button_shape,
                padding=ft.Padding.symmetric(horizontal=16, vertical=10),
                text_style=button_text,
                side=ft.BorderSide(1, t.outline_variant),
            )
        ),
        icon_button_theme=ft.IconButtonTheme(style=ft.ButtonStyle(shape=button_shape)),
        segmented_button_theme=ft.SegmentedButtonTheme(
            style=ft.ButtonStyle(
                shape=button_shape,
                text_style=ft.TextStyle(size=SIZE_SMALL, weight=ft.FontWeight.W_500),
                padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                side=ft.BorderSide(1, t.outline_variant),
                visual_density=ft.VisualDensity.COMPACT,
            )
        ),
        dialog_theme=ft.DialogTheme(
            bgcolor=t.card,
            shape=_rounded(16),
            title_text_style=ft.TextStyle(size=20, weight=ft.FontWeight.W_600, color=t.on_surface),
            content_text_style=ft.TextStyle(size=SIZE_BODY, height=1.45, color=t.on_surface),
        ),
        list_tile_theme=ft.ListTileTheme(
            shape=_rounded(RADIUS_TILE),
            dense=True,
            content_padding=ft.Padding.symmetric(horizontal=GAP, vertical=2),
            min_vertical_padding=4,
            title_text_style=ft.TextStyle(size=SIZE_BODY, weight=ft.FontWeight.W_500, color=t.on_surface),
            subtitle_text_style=ft.TextStyle(size=SIZE_SMALL, color=t.on_surface_variant),
        ),
        scrollbar_theme=ft.ScrollbarTheme(thickness=6, radius=3, cross_axis_margin=2),
        slider_theme=ft.SliderTheme(track_height=4),
        dropdown_theme=ft.DropdownTheme(text_style=ft.TextStyle(size=13)),
    )


__all__ = [
    "CARD_PADDING",
    "CONTENT_MAX_WIDTH",
    "DARK",
    "DRAWER_WIDTH",
    "GAP",
    "LIGHT",
    "PAGE_PADDING",
    "RADIUS_CARD",
    "RADIUS_TILE",
    "RAIL_WIDTH",
    "SIZE_BODY",
    "SIZE_CAPTION",
    "SIZE_CARD_TITLE",
    "SIZE_MICRO",
    "SIZE_SMALL",
    "SIZE_TITLE",
    "SIZE_VALUE",
    "Tokens",
    "build_theme",
    "is_dark",
    "series",
    "status",
    "tokens",
    "tone_color",
]
