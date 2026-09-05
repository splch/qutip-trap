# qutip-trap-app

The zoomable application of PLAN.md Section 14 (milestone M11). This directory is the `flet create`
scaffold (Flet 0.86.5 template: `pyproject.toml`, `src/main.py`, `src/assets/icon.png`, `tests/`),
installed by milestone M0 so that `flet doctor` runs in CI. The application itself is not started
before M11; nothing in the physics core (`qutip_trap`) imports this package, and this package may
import only the public surface `qutip_trap.api` (Appendix E).

Run the placeholder app from the repository root:

    uv run --package qutip-trap-app flet run qutip_trap_app          # native window
    uv run --package qutip-trap-app flet run --web qutip_trap_app    # browser served from the host

The static Pyodide web build is not a target (PLAN.md Section 1.5).
