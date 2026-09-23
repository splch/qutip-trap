"""Every screen's control tree, built by the functions the shell renders with, passes the checks Flet makes before sending a
control to the client (``before_update`` and the property validators) and encodes to the wire format, so a broken view
fails here without a Flutter client."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import flet as ft
import msgpack
import pytest
from fixtures import BellResimulated
from flet.controls.base_control import BaseControl
from flet.messaging.protocol import configure_encode_object_for_msgpack

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.views.shell import layout
from qutip_trap_app.views.state import JobStatus, Session, Store


def _walk(control: BaseControl) -> Iterator[BaseControl]:
    stack = [control]
    while stack:
        c = stack.pop()
        yield c
        for f in dataclasses.fields(c):
            value = getattr(c, f.name, None)
            if isinstance(value, BaseControl):
                stack.append(value)
            elif isinstance(value, list | tuple):
                stack.extend(v for v in value if isinstance(v, BaseControl))


def _check(tree: ft.Control) -> int:
    """Run Flet's pre-send checks on every control of the tree and encode it; the number of controls."""
    controls = list(_walk(tree))
    for c in controls:
        c._before_update_safe()
    msgpack.packb(tree, default=configure_encode_object_for_msgpack(BaseControl))
    return len(controls)


def _routes(record: Record) -> list[str]:
    key = record.key
    routes = [f"/job/{key}"]
    routes += [f"/job/{key}/circuit/{t.gate_id}" for t in record.schedule.targets]
    for p in record.schedule.pulses:
        routes += [
            f"/job/{key}/schedule/{p.index}",
            f"/job/{key}/dynamics/{p.index}/0",
            f"/job/{key}/hamiltonian/{p.index}/0",
        ]
    return routes


def _store(*records: Record) -> Store:
    store = Store(width=1400.0)
    store.records = {r.key: r for r in records}
    store.current = records[0].key
    return store


def test_the_empty_app_and_an_unknown_job() -> None:
    store = Store()
    session = Session(store)
    assert _check(layout(store, session, "/")) > 20
    _check(layout(store, session, "/job/nope/circuit/ms[2]"))
    store.error = "the circuit could not be read: no qreg declared"
    store.jobs = {"t1": JobStatus("t1", "run", stage="running: pulse", fraction=0.4, message="pulse 3 of 9")}
    _check(layout(store, session, "/"))


def test_every_screen_of_a_resimulated_bell_run(bell_resimulated: BellResimulated) -> None:
    r = bell_resimulated
    store = _store(r.record)
    session = Session(store)
    bar = next(iter(r.record.results.counts))
    store.selected_bar, store.selected_shot = bar, 0 if r.record.results.bitstrings.shape[0] else None
    store.jobs = {
        "t1": JobStatus("t1", "zoom", target={"key": r.record.key, "step": r.step, "sample": 0, "branch": 0})
    }
    sizes = {route: _check(layout(store, session, route)) for route in _routes(r.record)}
    ms_pulse = r.record.step(r.step).pulse_indices[0]
    assert sizes[f"/job/{r.record.key}/dynamics/{ms_pulse}/0"] > 300, (
        "the zoomed step draws its loops and heatmaps"
    )
    assert sizes[f"/job/{r.record.key}/hamiltonian/{ms_pulse}/0"] > 300, "the equation lists its terms"
    store.width = 700.0  # one column
    for route in _routes(r.record)[:3]:
        _check(layout(store, session, route))


@pytest.mark.parametrize("fixture", ["bell", "bell_gate_local"])
def test_every_screen_of_a_fresh_run(fixture: str, request: pytest.FixtureRequest) -> None:
    record, _live = request.getfixturevalue(fixture)
    assert isinstance(record, Record) and isinstance(_live, LiveRun)
    store = _store(record)
    session = Session(store)
    for route in ["/", *_routes(record)]:
        _check(layout(store, session, route))
