"""Off the happy path: circuits with fewer qubits than the device has ions, circuits measuring a subset, and worker results
that arrive after progress events. Each record is pushed through the pure view-models the screens read, so that a shape
mismatch fails here and not on a screen."""

from __future__ import annotations

import pytest
from fixtures import FAST, SEED

from qutip_trap_app import resim
from qutip_trap_app.core import Circuit, Operation, load_openqasm2
from qutip_trap_app.provenance import ProvenanceIndex
from qutip_trap_app.record import LiveRun, Record, calibrate_for, execute, job_for_preset
from qutip_trap_app.replay import ChannelLibrary, replay
from qutip_trap_app.verify import VerifyReport
from qutip_trap_app.viewmodel.circuit import register_after, timeline
from qutip_trap_app.viewmodel.machine import histogram, shot
from qutip_trap_app.views.state import JobStatus, Session, Store
from qutip_trap_app.workers import Event, Ticket

ONE_QUBIT = Circuit(1, (Operation("h", (0,), ()),), (0,))
SUBSET = load_openqasm2(
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[1];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\n'
)
SHOTS = 40


def _job(circuit: Circuit):  # type: ignore[no-untyped-def]
    # the app never runs fewer than two ions (Session.build_job): a one-qubit circuit lands on the two-ion chain
    return job_for_preset(
        "yb171_chain",
        max(2, circuit.n_qubits),
        circuit,
        SHOTS,
        seed=SEED,
        options=FAST,
        detection_records=500,
    )


@pytest.fixture(scope="module")
def one_qubit_full() -> tuple[Record, LiveRun]:
    job, preset = _job(ONE_QUBIT)
    return execute(job, preset)


@pytest.fixture(scope="module")
def one_qubit_replay() -> Record:
    job, preset = _job(ONE_QUBIT)
    assert preset is not None
    table = calibrate_for(job, preset.device)
    return replay(job, preset.device, table, ChannelLibrary())


@pytest.fixture(scope="module")
def subset_full() -> tuple[Record, LiveRun]:
    job, preset = _job(SUBSET)
    return execute(job, preset)


def _every_view(record: Record) -> None:
    h = histogram(record)
    assert {b.key for b in h.bars} == set(record.results.probabilities) | set(
        record.results.target_probabilities
    )
    assert all(len(b.key) == h.n_qubits for b in h.bars)
    assert record.results.bitstrings.shape[1] == h.n_qubits
    for k in range(record.results.bitstrings.shape[0]):
        s = shot(record, k)
        assert len(s.levels) == record.n_ions and len(s.time_used_s) == record.n_ions
    for g in timeline(record):
        reg = register_after(record, g.index)
        assert reg.rho.shape == (2**record.n_ions,) * 2 and len(reg.bloch) == record.n_ions
        assert 0.0 <= float(reg.fidelity.value) <= 1.0 + 1e-9


@pytest.mark.parametrize("engine", ["full", "replay"])
def test_one_qubit_circuit_on_the_two_ion_device(
    engine: str, one_qubit_full: tuple[Record, LiveRun], one_qubit_replay: Record
) -> None:
    record = one_qubit_full[0] if engine == "full" else one_qubit_replay
    assert record.n_qubits == 1 and record.n_ions == 2
    assert record.results.bitstrings.shape == (SHOTS, 1), "one measured qubit, one histogram column"
    assert set(record.results.counts) <= {"0", "1"} and set(record.results.target_probabilities) == {"0", "1"}
    fid = record.results.register_fidelity
    assert fid is not None and fid > 0.99, "the ideal ket is |+> on ion 0 and |0> on the idle ion"
    assert record.readout.levels.shape == (SHOTS, 2), "the sampled levels of both ions, one row per kept shot"
    _every_view(record)


def test_a_circuit_measuring_a_subset(subset_full: tuple[Record, LiveRun]) -> None:
    record, _live = subset_full
    assert record.n_qubits == 2 and record.results.bitstrings.shape == (SHOTS, 1)
    assert set(record.results.counts) <= {"0", "1"} and set(record.results.target_probabilities) == {"0", "1"}
    assert record.results.register_fidelity is not None and record.results.register_fidelity > 0.98
    _every_view(record)


def test_readout_rows_line_up_with_the_shots(bell: tuple[Record, LiveRun]) -> None:
    """The core reads the register out once per (sample, branch) batch; the record's readout arrays cover every kept shot."""
    record, _live = bell
    n_shots = record.results.bitstrings.shape[0]
    assert record.n_branches > 1
    assert record.readout.levels.shape == (n_shots, record.n_ions)
    assert record.readout.time_used_s.shape[0] == n_shots
    last = shot(record, n_shots - 1)
    assert len(last.levels) == record.n_ions


class _FakeWorker:
    alive = True

    def start(self) -> None:
        return None

    def submit(self, request: str, **payload: object) -> Ticket:
        return Ticket("t1", request)

    def poll(self, timeout_s: float = 0.0) -> list[Event]:
        return []


def test_results_land_after_progress_events(bell: tuple[Record, LiveRun]) -> None:
    """A progress event rewrites the job's message; the zoom and verify results are matched to their record by the ticket's
    target, so they land where they belong."""
    record, live = bell
    key = record.key()
    session = Session(Store(), ProvenanceIndex.load())
    session.worker = _FakeWorker()  # type: ignore[assignment]
    store = session.store
    store.records = {key: record}
    store.current = key
    step = next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))
    zoomed, z, _stats = resim.zoom(record, live, step, n_store=11)  # what the worker would send back
    zoomed, ham = resim.hamiltonian_record(zoomed, live, step)
    session.submit_zoom(key, step)
    session.apply_events(
        [Event("progress", "t1", "zoom", stage="zooming", message=f"re-simulating step {step}")]
    )
    session.apply_events(
        [
            Event(
                "result",
                "t1",
                "zoom",
                payload={"zoom": z, "boundaries": zoomed.boundaries, "stats": _stats, "hamiltonian": ham},
            )
        ]
    )
    assert store.jobs["t1"].done and store.error == ""
    assert store.records[key].zoom(z.key) is z, "the zoom landed on its record"
    session.submit_verify(key)
    session.apply_events(
        [Event("progress", "t1", "verify", stage="running deeper", message="the same job at auto")]
    )
    report = VerifyReport("JOINT_EXACT", None, SHOTS, None, 0.0, 0.0, None, None, True, None, 0.0)
    session.apply_events([Event("result", "t1", "verify", payload={"report": report, "deep": None})])
    assert store.verify_reports == {key: report}


def test_a_dead_worker_fails_its_jobs_and_is_restarted() -> None:
    """No error event comes from a worker that died, so its running jobs would spin for ever; the poll loop marks them
    failed with the reason and restarts the worker for the next request."""
    session = Session(Store(), ProvenanceIndex.load())
    store = session.store
    started: list[str] = []

    class _Dead:
        alive = False
        started = True

        def start(self) -> None:
            started.append("start")

        def poll(self, timeout_s: float = 0.0) -> list[Event]:
            return []

    session.worker = _Dead()  # type: ignore[assignment]
    assert not session.worker_died(), "no running job: nothing to report"
    store.jobs = {"t1": JobStatus("t1", "run_job"), "t0": JobStatus("t0", "zoom", done=True, stage="done")}
    assert session.worker_died()
    assert (
        store.jobs["t1"].done
        and store.jobs["t1"].stage == "failed"
        and "died" in (store.jobs["t1"].error or "")
    )
    assert store.jobs["t0"].stage == "done" and started == ["start"] and "died" in store.error
    assert not session.worker_died(), "reported once: the jobs are done now"


@pytest.fixture(scope="module")
def no_gates_full() -> tuple[Record, LiveRun]:
    """A bare measurement on the full engine: the schedule plays no pulse, so the record has no step."""
    job, preset = _job(Circuit(2, (), (0, 1)))
    return execute(job, preset)


def test_a_record_without_a_step_says_so(no_gates_full: tuple[Record, LiveRun]) -> None:
    """A gate-less record has no step: the dynamics lookup raises the KeyError the view catches."""
    from qutip_trap_app.viewmodel.dynamics import recorded_zoom

    record, _live = no_gates_full
    assert record.schedule.steps == () and record.results.counts == {"00": record.results.bitstrings.shape[0]}
    with pytest.raises(KeyError, match="no step 0"):
        recorded_zoom(record, 0, 0, 0)
    _every_view(record)


def test_the_badge_never_passes_what_it_did_not_check(
    one_qubit_replay: Record, bell: tuple[Record, LiveRun]
) -> None:
    """A replay record reports no boundary population: the check is listed as not run, not as passed; a NaN population is a
    failure, not a silent pass."""
    import dataclasses
    import math

    from qutip_trap_app.viewmodel.numerics import convergence_badge

    replay_badge = convergence_badge(one_qubit_replay)
    assert replay_badge.status == "not checked"
    assert any("boundary population" in line for line in replay_badge.checks_not_run)
    assert not any("boundary population" in line for line in replay_badge.checks_run)
    record, _live = bell
    assert convergence_badge(record).checks_run[0].startswith("boundary population")
    broken = dataclasses.replace(
        record, diagnostics=dataclasses.replace(record.diagnostics, boundary_population={2: math.nan})
    )
    badge = convergence_badge(broken)
    assert badge.status == "fail" and any("not a number" in r for r in badge.reasons)


def test_knob_overrides_outside_their_range_are_refused(bell: tuple[Record, LiveRun]) -> None:
    from qutip_trap_app import knobs

    _record, live = bell
    device = live.device
    ok = knobs.validate({"detector.efficiency": 0.2}, device)
    assert ok == {"detector.efficiency": 0.2}
    with pytest.raises(knobs.KnobError, match="outside the knob's range"):
        knobs.validate({"detector.efficiency": 0.9}, device)
    with pytest.raises(knobs.KnobError, match="not a number"):
        knobs.validate({"detector.efficiency": "high"}, device)  # type: ignore[dict-item]
    with pytest.raises(knobs.KnobError, match="unknown knob"):
        knobs.validate({"nonsense": 1.0}, device)


def test_a_closure_prediction_needs_loops() -> None:
    from qutip_trap_app.viewmodel.learn import score_closure

    with pytest.raises(ValueError, match="no loops"):
        score_closure("every loop closes", {}, {})
    assert score_closure("every loop closes", {(0, 2): 0.001}, {(0, 2): 1.0})
    assert not score_closure("every loop closes", {(0, 2): 0.5}, {(0, 2): 1.0})


def test_level3_finds_the_zoom_it_asked_for(bell: tuple[Record, LiveRun]) -> None:
    """The zoom is cached under the options it ran with (the Fock marginals on); Level 3 looks it up with the record's
    options, and the key folds the switch in, so the two agree."""
    from qutip_trap_app.views.level3 import current_zoom

    record, live = bell
    step = next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))
    zoomed, z, _stats = resim.zoom(record, live, step)
    assert resim.zoom_key(step, 0, 0, resim.DEFAULT_ZOOM_POINTS, record.job.options) == z.key
    found, fine = current_zoom(zoomed, step, 0, 0)
    assert fine and found is z
    coarse, fine_before = current_zoom(record, step, 0, 0)
    assert not fine_before and coarse is not None and coarse.n_store == 0, (
        "the recorded coarse trace until then"
    )
