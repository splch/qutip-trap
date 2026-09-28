"""``python -m qutip_trap_app``, the ``qutip-trap-app`` command: the application in a desktop window, or with ``--web``
served to the browser from this process."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import flet as ft

from qutip_trap_app.app import main

ASSETS = Path(__file__).resolve().parent.parent / "assets"
"""``src/assets``: the icon the rail shows."""


def cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qutip-trap-app",
        description="One simulated trapped-ion quantum computer at five levels of abstraction.",
    )
    parser.add_argument(
        "--web", action="store_true", help="serve the app to a browser instead of opening a window"
    )
    parser.add_argument("--port", type=int, default=8550, help="the port --web serves on (default: 8550)")
    args = parser.parse_args(argv)
    ft.run(
        main,
        view=ft.AppView.WEB_BROWSER if args.web else ft.AppView.FLET_APP,
        port=args.port if args.web else 0,
        assets_dir=str(ASSETS),
    )


if __name__ == "__main__":
    cli()
