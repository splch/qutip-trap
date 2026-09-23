"""Off the happy path: a circuit on fewer qubits than the device has ions, one measuring a subset, one playing no pulse,
worker results that arrive after progress events, and a worker that dies. Each record goes through the view-models the
screens read, so a shape mismatch fails here rather than on a screen."""

from __future__ import annotations

import pytest
from fixtures import job, ms_step

from qutip_trap_app import resim
from qutip_trap_app.record import LiveRun, Record, ZoomTrace, execute
from qutip_trap_app.viewmodel.circuit import register_after, timeline
from qutip_trap_app.viewmodel.dynamics import recorded_zoom
from qutip_trap_app.viewmodel.editor import parse_circuit
from qutip_trap_app.viewmodel.machine import histogram, shot
from qutip_trap_app.views.state import JobStatus, Session, Store
from qutip_trap_app.workers import Event, Resimulated, Ticket

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
ONE_QUBIT = parse_circuit(HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q -> c;\n")
SUBSET = parse_circuit(HEADER + "qreg q[2];\ncreg c[1];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\n")
NO_GATES = parse_circuit(HEADER + "qreg q[2];\ncreg c[2];\nmeasure q -> c;\n")
SHOTS = 40


def _every_view(record: Record) -> None:
    h = histogram(record)
    assert {b.key for b in h.bars} == set(record.results.probabilities) | set(
        record.results.target_probabilities
    )
    assert all(len(b.key) == h.n_qubits for b in h.bars)
    for k in range(record.results.bitstrings.shape[0]):
        assert len(shot(record, k).levels) == record.n_ions
    for g in timeline(record):
        reg = register_after(record, g.index)
        assert reg.rho.shape == (2**record.n_ions,) * 2 and len(reg.bloch) == record.n_ions
        assert 0.0 <= float(reg.fidelity.value or 0.0) <= 1.0 + 1e-9


@pytest.fixture(scope="module")
def one_qubit() -> Record:
    return execute(job(ONE_QUBIT, shots=SHOTS))[0]


@pytest.fixture(scope="module")
def subset() -> Record:
    return execute(job(SUBSET, shots=SHOTS))[0]


@pytest.fixture(scope="module")
def no_gates() -> Record:
    return execute(job(NO_GATES, shots=SHOTS))[0]


def test_a_one_qubit_circuit_on_the_two_ion_device(one_qubit: Record) -> None:
    record = one_qubit
    assert record.n_qubits == 1 and record.n_ions == 2
    assert record.results.bitstrings.shape == (SHOTS, 1), "one measured qubit, one histogram column"
    assert set(record.results.counts) <= {"0", "1"} and set(record.results.target_probabilities) == {"0", "1"}
    fidelity = record.results.register_fidelity
    assert fidelity is not None and fidelity > 0.99, "the ideal ket is |+> on ion 0 and |0> on the idle ion"
    assert record.results.levels.shape == (SHOTS, 2), "the sampled levels of both ions, one row per kept shot"
    _every_view(record)


def test_a_circuit_measuring_a_subset(subset: Record) -> None:
    assert subset.n_qubits == 2 and subset.results.bitstrings.shape == (SHOTS, 1)
    assert set(subset.results.target_probabilities) == {"0", "1"}
    assert subset.results.register_fidelity is not None and subset.results.register_fidelity > 0.98
    _every_view(subset)


def test_a_record_without_a_step_says_so(no_gates: Record) -> None:
    assert no_gates.schedule.steps == () and no_gates.results.counts == {"00": SHOTS}
    with pytest.raises(KeyError, match="no step 0"):
        recorded_zoom(no_gates, 0, 0, 0)
    _every_view(no_gates)


def test_the_readout_rows_line_up_with_the_shots(bell: tuple[Record, LiveRun]) -> None:
    """The core reads the register out once per (sample, branch) batch; the record's levels cover every kept shot."""
    record, _live = bell
    n_shots = record.results.bitstrings.shape[0]
    assert record.n_branches > 1 and record.results.levels.shape == (n_shots, record.n_ions)
    assert len(shot(record, n_shots - 1).levels) == record.n_ions


class _FakeWorker:
    alive = True
    started = True

    def start(self) -> None:
        return None

    def submit(self, request: str, **payload: object) -> Ticket:
        return Ticket("t1", request)

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        return []


def test_a_result_lands_on_its_record_after_progress_events(bell: tuple[Record, LiveRun]) -> None:
    """A progress event rewrites the job's message; the result is matched to its record by the ticket's target."""
    record, live = bell
    session = Session(Store())
    session.worker = _FakeWorker()
    store = session.store
    store.records, store.current = {record.key: record}, record.key
    step = ms_step(record)
    zoomed, z, _cached = resim.zoom(record, live, step)
    session.submit_resim("zoom", record.key, step)
    session.apply_events(
        [Event("progress", "t1", "zoom", stage="zooming", message=f"re-simulating step {step}")]
    )
    new = tuple(v for k, v in zoomed.cache.items() if k not in record.cache)
    session.apply_events([Event("result", "t1", "zoom", payload=Resimulated(new))])
    assert store.jobs["t1"].done and store.error == ""
    assert store.records[record.key].cached(z.key, ZoomTrace) is z, "the zoom landed on its record"


def test_a_dead_worker_fails_its_jobs_and_is_restarted() -> None:
    """No error event comes from a worker that died: the poll loop fails its running jobs and restarts it."""
    session = Session(Store())
    store = session.store
    started: list[str] = []

    class _Dead:
        alive = False
        started = True

        def start(self) -> None:
            started.append("start")

    session.worker = _Dead()
    assert not session.worker_died(), "no running job: nothing to report"
    store.jobs = {"t1": JobStatus("t1", "run"), "t0": JobStatus("t0", "zoom", done=True, stage="done")}
    assert session.worker_died()
    assert (
        store.jobs["t1"].done
        and store.jobs["t1"].stage == "failed"
        and "died" in (store.jobs["t1"].error or "")
    )
    assert store.jobs["t0"].stage == "done" and started == ["start"] and "died" in store.error
    assert not session.worker_died(), "reported once: the jobs are done now"
