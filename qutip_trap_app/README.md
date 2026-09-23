# qutip-trap-app

A [Flet](https://flet.dev) app that runs a circuit on the example 171Yb+ chain of [qutip-trap](../README.md) and lets you zoom into one run, from the histogram down to the Hamiltonian the solver integrates. It uses the core through `src/qutip_trap_app/core.py` only.

## The five screens

- **Machine**: an editor for an OpenQASM 2 program (or an IonQ circuit JSON object) with the Bell and GHZ presets, and the histogram of the run with the ideal distribution beside it: counts, probabilities, error bars and the shots behind each bar.
- **Circuit**: the compiled native gates on ion lanes, and the register after the selected gate as Bloch discs and populations, read from the run's traces (or, for a GATE_LOCAL run, from its own register after each gate).
- **Schedule**: the pulses on a time axis, the selected pulse's tones against the motional modes, its envelope, and whether each mode's loop closes.
- **Dynamics**: one pulse re-simulated at fine resolution from its recorded initial state: the phase-space loops, the qubit populations, <n>(t), the Fock populations as a heatmap, the quantum jumps, the process matrix of the pulse, and the convergence re-checks behind the badge in the numerics strip.
- **Equation**: the terms of H(t) and the collapse operators the engine integrates for that pulse, with each drive term's matrix elements.

## Running it

From the repository root, after `uv sync --all-packages --extra gui --group dev`:

    cd qutip_trap_app && uv run --project .. flet run src/main.py                 # a native window
    cd qutip_trap_app && uv run --project .. flet run --web -p 8550 src/main.py   # served to a browser
    cd qutip_trap_app && FLET_FORCE_WEB_SERVER=true FLET_SERVER_PORT=8550 uv run --project .. python src/main.py   # the server alone

The simulations run in a separate worker process, so the screens stay responsive. The same app, built with `flet build web` for Pyodide, is the GitHub Pages site at https://splch.github.io/qutip-trap/ (`.github/workflows/pages.yml`); there the worker runs in the page itself.

## Tests

    uv run --package qutip-trap-app pytest qutip_trap_app/tests -n 4
    uv run ruff check qutip_trap_app && uv run ruff format --check qutip_trap_app
    cd qutip_trap_app && uv run mypy

The tests run one two-ion Bell job and read its record; `tests/test_screens.py` builds every screen's controls through the functions the shell renders with and runs Flet's own checks on them, without a Flutter client.
