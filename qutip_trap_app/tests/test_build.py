"""The build settings agree with the app they build: a built app's splash is the page colour of each scheme, so it opens on
its own background rather than a flash of another."""

from __future__ import annotations

import tomllib
from pathlib import Path

from qutip_trap_app.views import theme


def test_the_splash_is_the_page_colour_of_each_scheme() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    splash = config["tool"]["flet"]["splash"]
    assert (splash["color"], splash["dark_color"]) == (theme.LIGHT.surface, theme.DARK.surface)
