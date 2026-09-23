"""Navigation: a Bell job is followed from a histogram bar to a Hamiltonian matrix element in at most six clicks, walking
the routing model the zoom buttons, the keyboard, the crumbs and the rail share; zooming out retraces the ladder."""

from __future__ import annotations

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.views.shell import (
    child_route,
    crumbs_of,
    level_of,
    parent_route,
    pulse_of_path,
    rail_route,
)
from qutip_trap_app.views.state import Store


def _store(record: Record) -> Store:
    store = Store()
    store.records, store.current = {record.key: record}, record.key
    return store


def test_six_clicks_from_a_histogram_bar_to_a_matrix_element(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    store = _store(record)
    key = record.key
    clicks = ["the bar: its shots open on the machine screen"]
    path = f"/job/{key}"
    for _ in range(4):  # the gate, the pulse, the sample, the equation
        nxt = child_route(store, path)
        assert nxt is not None, path
        clicks.append(nxt)
        path = nxt
    clicks.append("a drive term's matrix-element table on the equation page")
    assert len(clicks) <= 6
    assert [level_of(p) for p in clicks[1:5]] == [1, 2, 3, 4]
    assert path.startswith(f"/job/{key}/hamiltonian/") and child_route(store, path) is None
    # zooming out retraces the ladder
    for expected in reversed(clicks[1:4]):
        path = parent_route(store, path) or ""
        assert path == expected
    assert parent_route(store, path) == f"/job/{key}" and parent_route(store, f"/job/{key}") is None
    # the gate the ladder names is the first gate in time, and the pulse the schedule shows is that gate's first
    first = min(record.schedule.targets, key=lambda t: t.t_start_s)
    assert clicks[1].endswith(f"/circuit/{first.gate_id}")
    assert pulse_of_path(record, clicks[2]) == record.step_of_gate(first.gate_id).pulse_indices[0]


def test_the_rail_keeps_the_place_on_the_time_axis(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    store = _store(record)
    key = record.key
    ms = next(s for s in record.schedule.steps if s.gate_id.startswith("ms"))
    pulse = ms.pulse_indices[0]
    here = f"/job/{key}/dynamics/{pulse}/0"
    assert rail_route(store, here, 0) == f"/job/{key}"
    assert rail_route(store, here, 1) == f"/job/{key}/circuit/{ms.target_ids[0]}"
    assert rail_route(store, here, 2) == f"/job/{key}/schedule/{pulse}"
    assert rail_route(store, here, 4) == f"/job/{key}/hamiltonian/{pulse}/0"
    assert rail_route(Store(), "/", 3) == "/", "no run yet: the rail stays on the machine"
    trail = crumbs_of(store, f"/job/{key}/hamiltonian/{pulse}/0")
    assert [t for t, _r in trail] == [
        "Equation",
        f"job {key[:8]}",
        f"gate {ms.target_ids[0]}",
        f"pulse {pulse}",
        "sample 0",
        "equation",
    ]
    assert trail[-1][1] is None and trail[-2][1] == here


def test_routes_of_an_unknown_record() -> None:
    store = Store()
    assert parent_route(store, "/job/k/circuit/ms[2]") == "/job/k"
    assert parent_route(store, "/job/k/schedule/3") == "/job/k", "an unknown record names no gate"
    assert parent_route(store, "/job/k/dynamics/3/0") == "/job/k/schedule/3"
    assert child_route(store, "/job/k") is None and level_of("/job/k/circuit") == 0
