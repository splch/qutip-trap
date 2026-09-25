"""The `flet test` navigation and run-flow tests: they drive the rendered controls through
Flet's ``flet_app`` fixture, which needs the Flutter test host (``flet test`` provisions it: the Flutter SDK, and on macOS the
full Xcode). ``conftest.py`` skips them unless ``QUTIP_TRAP_APP_UI_TESTS=1``; run them with

    cd qutip_trap_app && QUTIP_TRAP_APP_UI_TESTS=1 uv run flet test

Controls are found by the keys the views assign (``run``, ``results``, ``gate:ms[2]``, ``zoom-gate``, ``zoom-pulse``,
``open-equation``, ``drive-term:0``), never by their text, so a wording change does not break the path. The run itself is the
channel replay (about a minute on first use: the channel library), waited for by polling the Results card's key.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.ui

RUN_TIMEOUT_S = 240.0


async def _wait_for_key(tester: object, key: str, timeout_s: float, step_s: float = 2.0) -> None:
    waited = 0.0
    while waited < timeout_s:
        finder = await tester.find_by_key(key)
        if finder.count > 0:
            return
        await asyncio.sleep(step_s)
        await tester.pump_and_settle()
        waited += step_s
    raise AssertionError(f"control {key!r} did not appear within {timeout_s:.0f} s")


async def _answer_first_launch_question(tester: object) -> None:
    skip = await tester.find_by_key("knowledge-skip")
    if skip.count:
        await tester.tap(skip)
        await tester.pump_and_settle()


@pytest.mark.skipif(
    not os.environ.get("QUTIP_TRAP_APP_UI_TESTS"), reason="needs the Flutter test host (flet test)"
)
async def test_run_flow_shows_the_histogram(flet_app: object) -> None:
    tester = flet_app.tester
    await tester.pump_and_settle()
    await _answer_first_launch_question(tester)
    run = await tester.find_by_key("run")
    assert run.count == 1, "Level 0 has one Run button, the primary action"
    await tester.tap(run)
    await tester.pump_and_settle()
    working = await tester.find_by_key("working")
    assert working.count == 1, "a run over a few seconds shows a working state with Cancel"
    await _wait_for_key(tester, "results", RUN_TIMEOUT_S)
    assert (await tester.find_by_key("results")).count == 1
    assert (await tester.find_by_key("device")).count == 1


@pytest.mark.skipif(
    not os.environ.get("QUTIP_TRAP_APP_UI_TESTS"), reason="needs the Flutter test host (flet test)"
)
async def test_six_clicks_from_the_histogram_to_a_matrix_element(flet_app: object) -> None:
    tester = flet_app.tester
    await tester.pump_and_settle()
    await _answer_first_launch_question(tester)
    await tester.tap(await tester.find_by_key("run"))
    await _wait_for_key(tester, "results", RUN_TIMEOUT_S)
    # click 1: zoom in from the machine to the circuit (a gate name opens Level 1)
    await tester.tap(await tester.find_by_key("zoom-in"))
    await tester.pump_and_settle()
    await _wait_for_key(tester, "timeline", 20.0)
    # click 2: the entangling gate
    ms = await tester.find_by_key("gate:ms[2]")
    assert ms.count >= 1
    await tester.tap(ms.first)
    await tester.pump_and_settle()
    # click 3: its pulses (Level 2)
    await tester.tap(await tester.find_by_key("zoom-gate"))
    await tester.pump_and_settle()
    await _wait_for_key(tester, "spectrum", 20.0)
    # click 4: inside the pulse (Level 3)
    await tester.tap(await tester.find_by_key("zoom-pulse"))
    await tester.pump_and_settle()
    # a derived (replay) run has no trace to open: Level 3 says so and still offers the equation via the rail
    no_trace = await tester.find_by_key("no-trace")
    equation = await tester.find_by_key("open-equation")
    assert no_trace.count + equation.count >= 1
    if equation.count:
        # click 5: the equation being solved (Level 4)
        await tester.tap(equation)
        await tester.pump_and_settle()
        await _wait_for_key(tester, "hamiltonian", 120.0)
        # click 6: a drive term's matrix elements
        term = await tester.find_by_key("drive-term:0")
        assert term.count == 1
        await tester.tap(term)
        await tester.pump_and_settle()
        assert (await tester.find_by_key("matrix-elements:0:2")).count + (
            await tester.find_by_key("matrix-elements:0:3")
        ).count >= 1


@pytest.mark.skipif(
    not os.environ.get("QUTIP_TRAP_APP_UI_TESTS"), reason="needs the Flutter test host (flet test)"
)
async def test_explain_drawer_and_learn_routes(flet_app: object) -> None:
    tester = flet_app.tester
    await tester.pump_and_settle()
    await _answer_first_launch_question(tester)
    toggle = await tester.find_by_key("explain-toggle")
    assert toggle.count == 1
    await tester.tap(toggle)
    await tester.pump_and_settle()
    await tester.tap(toggle)
    await tester.pump_and_settle()
    flet_app.page.navigate("/learn/experiments")
    await tester.pump_and_settle()
    await _wait_for_key(tester, "experiments", 20.0)
    assert (await tester.find_by_key("preset:harty_2014")).count == 1
    await tester.tap(await tester.find_by_key("preset:harty_2014"))
    await tester.pump_and_settle()
    await _wait_for_key(tester, "run-preset", 20.0)
    await tester.tap(await tester.find_by_key("run-preset"))
    await _wait_for_key(tester, "preset-result", 60.0)
