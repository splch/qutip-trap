"""qutip-trap-app entry point: `flet run` (desktop) or `flet run --web` (served browser mode); PLAN.md Section 14.6."""

from __future__ import annotations

import flet as ft

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.views.shell import App
from qutip_trap_app.views.state import Session, Store


def main(page: ft.Page) -> None:
    page.title = "qutip-trap: a trapped-ion quantum computer, from the histogram down to the Hamiltonian"
    page.theme_mode = ft.ThemeMode.SYSTEM
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

    def on_disconnect(_e: object) -> None:
        session.stop()

    page.on_disconnect = on_disconnect
    page.render(App, store, session, session.provenance)


if __name__ == "__main__":
    ft.run(main)
