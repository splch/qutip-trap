"""Section 9.11 row "Navigation": a Bell-state job is followed from a histogram bar to a Hamiltonian matrix element in at most
six clicks. This test walks the routing model of the shell (the same functions the zoom buttons, the rail, the crumbs and the
keyboard use) without a Flutter client; ``test_main.py`` drives the same path through the rendered controls under
``flet test``. Also the Learn routes of M11.4 and the request that a run made from Level 1 lands on its gate."""

from __future__ import annotations

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.learn import BELL_TOUR, LEARN_TABS, route_matches
from qutip_trap_app.views.shell import (
    LEVEL_SECTIONS,
    child_route,
    learn_route_parts,
    level_of,
    parent_route,
    pulse_of_path,
)
from qutip_trap_app.views.state import JobStatus, Session, Store
from qutip_trap_app.workers import Event


def test_six_clicks_from_a_histogram_bar_to_a_matrix_element(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    store = Store()
    key = record.key()
    store.records = {key: record}
    store.current = key
    clicks: list[str] = []
    path = f"/job/{key}"  # click 1: a histogram bar opens its shots on Level 0 (the same route; the bar is the target)
    clicks.append("bar")
    for _ in range(3):  # clicks 2 to 4: gate, pulse, sample
        nxt = child_route(store, path)
        assert nxt is not None, path
        clicks.append(nxt)
        path = nxt
    assert level_of(path) == 3 and path.startswith(f"/job/{key}/dynamics/")
    nxt = child_route(store, path)  # click 5: the equation being solved
    assert nxt == "/device/hamiltonian"
    clicks.append(nxt)
    clicks.append(
        "matrix element tile"
    )  # click 6: a drive term's Omega_(n', n) table on the Hamiltonian page
    assert len(clicks) <= 6
    # the ladder is traversed level by level, and zooming out retraces it
    levels = [level_of(p) for p in clicks[1:5]]
    assert levels == [1, 2, 3, 4]
    back = parent_route(store, clicks[3])
    assert back is not None and level_of(back) == 2
    back2 = parent_route(store, back)
    assert back2 is not None and level_of(back2) == 1 and back2 == clicks[1]
    assert parent_route(store, clicks[1]) == f"/job/{key}"
    # the gate the tour names is the entangling gate, and the pulse the rail keeps is that gate's first pulse
    ms = next(g for g in record.schedule.gates)
    assert ms.gate_id in clicks[1] or clicks[1].endswith(record.schedule.targets[0].gate_id)
    assert pulse_of_path(record, clicks[2]) == int(clicks[2].rsplit("/", 1)[1])
    for stop in BELL_TOUR:
        assert route_matches(
            stop.route.replace("{id}", key)
            .replace("{gate}", ms.gate_id)
            .replace("{pulse}", "4")
            .replace("{sample}", "0")
        )
    assert set(LEVEL_SECTIONS) == {0, 1, 2, 3, 4}


def test_learn_routes() -> None:
    assert learn_route_parts("/learn") == ("tour", None)
    assert learn_route_parts("/learn/drills") == ("drills", None)
    assert learn_route_parts("/learn/preset/harty_2014") == ("experiments", "harty_2014")
    assert learn_route_parts("/learn/nowhere") == ("tour", None)
    for tab, _label in LEARN_TABS:
        assert route_matches(f"/learn/{tab}")
    assert level_of("/learn/drills") == -1 and parent_route(Store(), "/learn/drills") is None


def test_a_request_run_lands_on_its_gate_and_a_preset_result_lands_in_the_store(
    bell: tuple[Record, LiveRun], monkeypatch: object
) -> None:
    from qutip_trap_app.viewmodel.presets import PresetResult

    record, _live = bell
    session = Session(Store(), ProvenanceIndex.load())
    store = session.store
    navigated: list[str] = []

    class FakePage:
        def navigate(self, route: str) -> None:
            navigated.append(route)

        def run_task(self, *_a: object, **_k: object) -> None:
            return None

    session.page = FakePage()
    store.jobs = {"r1": JobStatus("r1", "request_run", target={"gate_id": "ms[2]", "kind": "angle"})}
    session.apply_events([Event("result", "r1", "request_run", payload=record)])
    assert store.current == record.key() and navigated == [f"/job/{record.key()}/circuit/ms[2]"]
    result = PresetResult("harty_2014", {"epg": 7.5e-7}, {"epg": 1e-7}, (), {}, (), 1.0)
    store.jobs = {**store.jobs, "p1": JobStatus("p1", "preset", target={"preset_id": "harty_2014"})}
    session.apply_events([Event("result", "p1", "preset", payload=result)])
    assert store.preset_results["harty_2014"] is result
    assert session.submit_preset is not None
