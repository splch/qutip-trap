"""The entry point ``flet run`` and ``flet build`` look for: the page function of ``qutip_trap_app.app``."""

from __future__ import annotations

import flet as ft

from qutip_trap_app.app import main

if __name__ == "__main__":
    ft.run(main)
