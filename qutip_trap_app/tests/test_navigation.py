"""A Bell-state job is followed from a histogram bar to a Hamiltonian matrix element in at most
six clicks. This walks the routing model of the shell (the functions the zoom buttons, the rail, the crumbs and the keyboard
use) without a Flutter client; ``test_main.py`` drives the same path through the rendered controls under ``flet test``.
Also the Learn routes, the tour's routes, and where a finished run opens: on its gate for a request made from Level 1 if
the learner is still there, and in the shell's banner if they have moved on."""

from __future__ import annotations

from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.learn import BELL_TOUR, CONCEPTS
from qutip_trap_app.viewmodel.presets import PresetResult
from qutip_trap_app.views import routes
from qutip_trap_app.views.learn import ACTIVITIES
from qutip_trap_app.views.shell import child_route, crumbs_of, parent_route, parse_route, pulse_of_path
from qutip_trap_app.views.state import FinishedRun, JobStatus, Landing, Session, Store
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
    assert parse_route(path).level == 3 and path.startswith(f"/job/{key}/dynamics/")
    nxt = child_route(store, path)  # click 5: the equation being solved
    assert nxt == "/device/hamiltonian"
    clicks.append(nxt)
    clicks.append(
        "matrix element tile"
    )  # click 6: a drive term's Omega_(n', n) table on the Hamiltonian page
    assert len(clicks) <= 6
    # the ladder is traversed level by level, and zooming out retraces it
    assert [parse_route(p).level for p in clicks[1:5]] == [1, 2, 3, 4]
    back = parent_route(store, clicks[3])
    assert back is not None and parse_route(back).level == 2
    back2 = parent_route(store, back)
    assert back2 == clicks[1] and parse_route(back2).level == 1
    assert parent_route(store, clicks[1]) == f"/job/{key}"
    # zooming in from the machine opens the first gate, and the pulse the rail keeps is that gate's first pulse
    assert clicks[1] == routes.circuit(key, record.schedule.targets[0].gate_id)
    ms = next(g for g in record.schedule.gates)
    assert pulse_of_path(record, clicks[2]) == int(clicks[2].rsplit("/", 1)[1])
    assert [text for text, _route in crumbs_of(store, clicks[3])][:2] == ["Dynamics", f"job {key[:8]}"]
    # every stop of the tour is a route of the ladder at its concept's level
    for stop in BELL_TOUR:
        route = (
            stop.route.replace("{id}", key)
            .replace("{gate}", ms.gate_id)
            .replace("{pulse}", "4")
            .replace("{sample}", "0")
        )
        assert parse_route(route).level == CONCEPTS[stop.concept_id].level, stop.route


def test_a_gate_id_survives_its_route_whole() -> None:
    """A gate id with brackets or a slash (the ZZ wrapper's pieces) is encoded when its route is built and decoded when it
    is parsed, as a browser delivers it after a reload or from the address bar."""
    for gate in ("ms[2]", "zz[3]/ms", "zz[3]/wrap_in/ion0"):
        path = routes.circuit("0123abcd", gate)
        assert "[" not in path and path.count("/") == 4, path
        r = parse_route(path)
        assert (r.level, r.job, r.gate) == (1, "0123abcd", gate)
        assert crumbs_of(Store(), path)[-1] == (f"gate {gate}", path)
    assert parse_route("/job/0123abcd/circuit/gpi2%5B0%5D").gate == "gpi2[0]", "the browser's own encoding"
    zoomed = parse_route(routes.dynamics("0123abcd", 4, 2))
    assert (zoomed.level, zoomed.pulse, zoomed.sample) == (3, "4", "2")
    assert routes.learn() == "/learn" and routes.preset("harty_2014") == "/learn/preset/harty_2014"


def test_learn_routes() -> None:
    home = parse_route("/learn")
    assert (home.level, home.tab, home.preset) == (-1, "tour", None)
    assert parse_route("/learn/drills").tab == "drills"
    r = parse_route("/learn/preset/harty_2014")
    assert (r.level, r.tab, r.preset) == (-1, "experiments", "harty_2014")
    assert parse_route("/learn/nowhere").tab == "tour", "an unknown activity opens the tour"
    assert all(parse_route(f"/learn/{tab}").tab == tab for tab in ACTIVITIES)
    assert parse_route("/device/nowhere").page == "hamiltonian", "an unknown page opens the equation"
    assert parent_route(Store(), "/learn/drills") is None


def test_a_request_run_lands_on_its_gate_and_a_preset_result_lands_in_the_store(
    bell: tuple[Record, LiveRun],
) -> None:
    record, _live = bell
    session = Session(Store(), ProvenanceIndex.load())
    store = session.store
    navigated: list[str] = []
    origin = routes.circuit("an earlier run", "ms[2]")

    class FakePage:
        route = origin

        def navigate(self, route: str) -> None:
            navigated.append(route)

        def run_task(self, *_a: object, **_k: object) -> None:
            return None

    def request_run(ticket: str) -> JobStatus:
        """What ``submit_request`` tracks for a request made on Level 1 at the gate ms[2]."""
        return JobStatus(
            ticket,
            "request_run",
            target={"gate_id": "ms[2]", "kind": "angle"},
            landing=Landing(origin, lambda key: routes.circuit(key, "ms[2]")),
        )

    session.page = FakePage()
    store.jobs = {"r1": request_run("r1")}
    session.apply_events([Event("result", "r1", "request_run", payload=record)])
    assert navigated == [routes.circuit(record.key(), "ms[2]")] and store.finished_run is None, (
        "the learner stayed on the gate: the new run opens there (the shell makes it current on arrival)"
    )
    session.page.route = routes.learn()
    store.jobs = {**store.jobs, "r2": request_run("r2")}
    session.apply_events([Event("result", "r2", "request_run", payload=record)])
    assert navigated == [routes.circuit(record.key(), "ms[2]")], (
        "the learner moved on: nothing pulls them back"
    )
    assert store.finished_run == FinishedRun(
        f"the requested run finished: job {record.key()[:8]}", routes.circuit(record.key(), "ms[2]")
    )
    result = PresetResult("harty_2014", {"epg": 7.5e-7}, {"epg": 1e-7}, (), {}, (), 1.0)
    store.jobs = {**store.jobs, "p1": JobStatus("p1", "preset", target={"preset_id": "harty_2014"})}
    session.apply_events([Event("result", "p1", "preset", payload=result)])
    assert store.preset_results["harty_2014"] is result
