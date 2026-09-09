"""The device model in the application state (Section 14.4; M11.3): a knob change submits one derive request and no duplicate;
the layer lands in the store keyed by the device reference; the job carries the overrides with an empty hash for the worker to
fill; a recalibration result clears the stale layer; the worker builds an edited device once and reuses it."""

from __future__ import annotations

from typing import Any

import pytest

from qutip_trap_app import device_layer
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import DeviceRef, LiveRun, Record
from qutip_trap_app.views.state import JobStatus, Session, Store
from qutip_trap_app.workers import Event, SimulationWorker


@pytest.fixture(scope="module")
def index() -> ProvenanceIndex:
    return ProvenanceIndex.load()


class _Ticket:
    def __init__(self, n: int) -> None:
        self.id = f"t{n}"


def test_knob_changes_submit_one_derive_and_land_in_the_store(
    index: ProvenanceIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = Session(Store(), index)
    store = session.store
    submitted: list[tuple[str, dict[str, Any]]] = []

    def fake_submit(request: str, **payload: Any) -> _Ticket:
        submitted.append((request, payload))
        return _Ticket(len(submitted))

    monkeypatch.setattr(session.worker, "submit", fake_submit)
    session.set_knob("field.b_gauss", 6.0)
    assert store.device_overrides == {"field.b_gauss": 6.0}
    assert len(submitted) == 1 and submitted[0][0] == "derive"
    ref: DeviceRef = submitted[0][1]["device"]
    assert ref.overrides == {"field.b_gauss": 6.0} and ref.hash == "" and ref.n_ions == 2
    session.submit_derive()
    assert len(submitted) == 1, "a running derive for the same device is not duplicated"
    job = session.build_job()
    assert job.device.overrides == {"field.b_gauss": 6.0} and job.device.hash == "", (
        "the UI never builds a device"
    )
    session.set_knob("field.b_gauss", None)
    assert store.device_overrides == {} and len(submitted) == 2 and submitted[1][1]["device"].overrides == {}


def test_layer_result_and_recalibration_apply(
    index: ProvenanceIndex, monkeypatch: pytest.MonkeyPatch, bell: tuple[Record, LiveRun]
) -> None:
    record, live = bell
    session = Session(Store(), index)
    store = session.store
    monkeypatch.setattr(session.worker, "submit", lambda request, **payload: _Ticket(len(store.jobs) + 1))
    ref = store.device_ref()
    layer = device_layer.derive_device_layer(
        record.job.device.build(), preset_name="yb171_chain", table=live.table, sweeps=False
    )
    store.jobs = {"t1": JobStatus("t1", "derive", target={"cache_key": ref.cache_key()})}
    session.apply_events([Event("result", "t1", "derive", payload=layer)])
    assert store.layer() is layer and store.device_ref().hash == layer.device_hash
    job = session.build_job()
    assert job.device.hash == layer.device_hash
    # a recalibration for this device drops the layer so it re-derives against the new table
    store.jobs = {**store.jobs, "t2": JobStatus("t2", "recalibrate", target={"cache_key": ref.cache_key()})}
    payload = {"table": record.table, "device_hash": layer.device_hash, "job": job}
    session.apply_events([Event("result", "t2", "recalibrate", payload=payload)])
    assert store.tables[layer.device_hash] is record.table
    assert store.layer() is None, "the stale layer is dropped and a fresh derive requested"
    assert store.running_of("derive", cache_key=ref.cache_key()) is not None
    assert store.table_for(layer.device_hash) is record.table


def test_worker_derives_an_edited_device_and_reuses_the_build() -> None:
    w = SimulationWorker()
    w.start()
    try:
        ref = DeviceRef("", "yb171_chain", 2, {}, {"detector.window_s": 3e-5})
        seen: list[Event] = []
        layer = w.wait(w.submit("derive", device=ref, sweeps=False), timeout_s=600.0, on_progress=seen.append)
        assert isinstance(layer, device_layer.DeviceLayer)
        assert layer.readout.window_s == 3e-5 and layer.overrides == {"detector.window_s": 3e-5}
        assert layer.knob_values["detector.window_s"] == 3e-5
        assert any(e.stage == "building device" for e in seen), "the knobs were applied in the worker"
        assert layer.trap.stability is not None
        first = len(seen)
        again = w.wait(w.submit("derive", device=ref, sweeps=False), timeout_s=600.0, on_progress=seen.append)
        assert again.device_hash == layer.device_hash
        assert "building device" not in [e.stage for e in seen[first:]], (
            "the built preset is cached by the key"
        )
    finally:
        w.stop()
