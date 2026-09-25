"""qutip-trap-app entry point: `flet run` (desktop) or `flet run --web` (served browser mode)."""

from __future__ import annotations

import asyncio
from typing import Any

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.views import theme
from qutip_trap_app.views.shell import App
from qutip_trap_app.views.state import Session, Store


def main(page: ft.Page) -> None:
    page.title = "qutip-trap"
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.theme = theme.build_theme(dark=False)
    page.dark_theme = theme.build_theme(dark=True)
    page.padding = 0
    store = Store()
    # the learner's settings and mastery log live on this device (browser storage in the served mode); the service
    # must be attached to the page (page.services) or its get/set never round-trips to the client
    preferences = ft.SharedPreferences()
    page.services = [preferences]
    session = Session(store, ProvenanceIndex.load(), page, preferences=preferences)
    session.start()
    page.run_task(session.restore_learner)
    page.run_task(session.poll_forever)

    # the drawings size themselves from the window and the badges colour themselves from the platform's brightness, so
    # both re-render the screens: the resize once the drag has settled, the brightness change at once
    generation = 0

    async def settle(gen: int) -> None:
        await asyncio.sleep(0.15)
        if gen == generation:
            store.tick = store.tick + 1

    def on_resize(_e: Any) -> None:
        nonlocal generation
        generation += 1
        page.run_task(settle, generation)

    def on_brightness(_e: Any) -> None:
        store.tick = store.tick + 1

    page.on_resize = on_resize
    page.on_platform_brightness_change = on_brightness

    def on_disconnect(_e: object) -> None:
        session.stop()

    page.on_disconnect = on_disconnect
    page.render(App, store, session, session.provenance)


if __name__ == "__main__":
    ft.run(main)
