"""The app's entry point: ``flet run`` opens a native window, ``flet run --web`` serves it to a browser."""

from __future__ import annotations

import asyncio
from typing import Any

import flet as ft

from qutip_trap_app.views.shell import App
from qutip_trap_app.views.state import Session, Store


def main(page: ft.Page) -> None:
    page.title = "qutip-trap"
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
    page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.padding = 0
    store = Store(width=float(page.width or 1200.0))
    session = Session(store, page)
    session.start()
    page.run_task(session.poll_forever)

    # the lanes and drawings size themselves from the window: re-render once a resize has settled
    generation = 0

    async def settle(gen: int) -> None:
        await asyncio.sleep(0.15)
        if gen == generation and page.width:
            store.width = float(page.width)

    def on_resize(_e: Any) -> None:
        nonlocal generation
        generation += 1
        page.run_task(settle, generation)

    def on_disconnect(_e: Any) -> None:
        session.stop()

    page.on_resize = on_resize
    page.on_disconnect = on_disconnect
    page.render(App, store, session)


if __name__ == "__main__":
    ft.run(main)
