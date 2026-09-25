"""Trajectory, branch and tomography parallelism and the propagator cache (PLAN.md Section 11.3): one and several workers
agree to 1e-12 under the same keyed seeds, and propagators are reused only on internal-state-only spaces."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from qutip.settings import available_cpu_count

import qutip_trap.dynamics.parallel as par
from qutip_trap.calibration.entangling import ms_schedule
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.pulses import Pulse
from qutip_trap.control.schedule import Schedule, single_qubit_pulse
from qutip_trap.control.table import Waveform
from qutip_trap.device.presets import yb171_chain
from qutip_trap.dynamics.engine import JointExactEngine, MotionalModel, SeedSpec
from qutip_trap.dynamics.parallel import map_tasks, memory_worker_cap, worker_count
from qutip_trap.dynamics.space import HilbertSpace, ModeTruncation
from qutip_trap.dynamics.tomography import cp_residual
from qutip_trap.light.raman import derive_raman_drive, square_drive
from qutip_trap.noise.sampling import quiet_sample
from qutip_trap.noise.spectra import white_spectrum
from qutip_trap.options import Numerics
from qutip_trap.run.job import last_record
from tests.fixtures import (
    BELL,
    X_COM_TWO_IONS,
    chain_device,
    derived_seeds,
    raman_gate_drives,
    run,
    single_ion_raman_device,
    table_with_waveform,
    two_ion_modes,
    two_ion_surrogate,
)

N_WORKERS = max(2, int(available_cpu_count()))
"""Every CPU QuTiP sees."""


def _square(x: int) -> int:
    return x * x


def test_map_tasks_keeps_the_input_order_and_stays_in_process_below_the_task_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the default-count formula without the environment cap tests/conftest.py sets under xdist
    monkeypatch.delenv(par.WORKERS_ENV, raising=False)
    items = list(range(7))
    assert map_tasks(_square, items, map_kind="parallel", workers=4) == [x * x for x in items]
    assert map_tasks(_square, items, map_kind="serial", workers=4) == [x * x for x in items]
    assert map_tasks(_square, items, map_kind="parallel", workers=1) == [x * x for x in items]
    assert map_tasks(_square, [3], map_kind="parallel", workers=4) == [9]
    assert map_tasks(_square, [], map_kind="parallel", workers=4) == []
    assert worker_count(Numerics(map="serial", workers=5)) == 1
    assert worker_count(Numerics(map="parallel", workers=5)) == min(5, memory_worker_cap())
    assert worker_count(Numerics(map="parallel")) == min(int(available_cpu_count()), memory_worker_cap())
    with pytest.raises(ValueError):
        Numerics(workers=0)


def test_the_worker_count_is_capped_by_the_parents_memory_footprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap keeps N x (parent peak) within half of physical memory: 8 workers for a 3 GB parent on 48 GB, 4 for 6 GB, 48
    below the 512 MB floor and 1 at 40 GB; an explicit request is capped, never raised."""
    monkeypatch.setattr(par, "physical_memory_bytes", lambda: 48 * 1024**3)
    monkeypatch.setattr(par, "peak_rss_bytes", lambda: 3 * 1024**3)
    assert par.memory_worker_cap() == 8
    assert worker_count(Numerics(map="parallel", workers=18)) == 8
    assert worker_count(Numerics(map="parallel", workers=3)) == 3
    monkeypatch.setattr(par, "peak_rss_bytes", lambda: 6 * 1024**3)
    assert par.memory_worker_cap() == 4
    monkeypatch.setattr(
        par, "peak_rss_bytes", lambda: 100 * 1024**2
    )  # below MIN_PARENT_BYTES: reckoned as 512 MB
    assert par.memory_worker_cap() == 48
    monkeypatch.setattr(par, "peak_rss_bytes", lambda: 40 * 1024**3)
    assert par.memory_worker_cap() == 1 and worker_count(Numerics(map="parallel")) == 1


def test_the_environment_caps_the_default_worker_count_but_not_an_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``QUTIP_TRAP_MAX_WORKERS`` caps the default worker count (blank means no cap) but not an explicit ``workers=``, and a
    serial map stays at one."""
    monkeypatch.setattr(par, "memory_worker_cap", lambda: 64)
    monkeypatch.delenv(par.WORKERS_ENV, raising=False)
    assert worker_count(Numerics(map="parallel")) == int(available_cpu_count())
    monkeypatch.setenv(par.WORKERS_ENV, "1")
    assert worker_count(Numerics(map="parallel")) == 1
    assert worker_count(Numerics(map="parallel", workers=5)) == 5
    assert worker_count(Numerics(map="serial", workers=5)) == 1
    monkeypatch.setenv(par.WORKERS_ENV, "3")
    assert worker_count(Numerics(map="parallel")) == min(3, int(available_cpu_count()))
    monkeypatch.setenv(par.WORKERS_ENV, " ")
    assert worker_count(Numerics(map="parallel")) == int(available_cpu_count())


@pytest.fixture(scope="module")
def heating_fixture():
    """The two-ion fixture with white electric-field noise (heating channels on the resolved x modes at 2.4e4 quanta/s, about six
    jumps over six 20 us trajectories), a 20 us single-loop pulse on the x-COM, the Bell caps (dimension 440, where the auto rule
    factorizes)."""
    dev = chain_device(2)
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(1e-9, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    drives = raman_gate_drives(2)
    rabi, stark = derived_seeds(noisy, drives)
    modes = two_ion_modes(noisy)
    wf = Waveform.symmetric(modes, gate_mode=X_COM_TWO_IONS, epsilon_hz=50e3, all_modes=True)
    table = table_with_waveform((0, 1), wf, rabi_hz=rabi, stark_hz=stark)
    sched = ms_schedule(wf, (0, 1), drives, table)
    space = HilbertSpace(
        (2, 2), (ModeTruncation(2, 10, (0, 3), 0.13), ModeTruncation(3, 11, (0, 4), 0.13)), None, (0, 1, 4, 5)
    )
    return noisy, drives, sched, space, table


def test_trajectories_agree_over_one_and_many_workers_with_per_trajectory_identity(heating_fixture) -> None:
    """Six keyed plain trajectories of a heated MS pulse in-process and over every CPU give the same register, P1 and joint state
    to 1e-12 and the same jumps, and the report names the map and the worker count."""
    dev, _drives, sched, space, _table = heating_fixture
    state = space.initial_state([0, 0])
    out = {}
    for mp, workers in (("serial", 1), ("parallel", N_WORKERS)):
        eng = JointExactEngine(device_channels=True)
        # plain trajectories: improved sampling conditions every member on jumping, which needs larger caps than these
        opts = Numerics(lindblad_method="mcsolve", ntraj=6, map=mp, workers=workers, improved_sampling=False)
        tr = eng.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(11), opts)
        rep = eng.last_report
        assert (
            rep is not None
            and rep.method == "mcsolve"
            and rep.trajectories == 6
            and rep.kernel == "factorized"
        )
        expected = 1 if mp == "serial" else min(worker_count(opts), 6)
        if expected < 2 and mp != "serial":
            pytest.skip("the memory cap left no room for a parallel run on this machine")
        assert rep.map == mp and rep.workers == expected
        out[mp] = (tr, rep)
    tr_s, rep_s = out["serial"]
    tr_p, rep_p = out["parallel"]
    assert (tr_s.final.internal - tr_p.final.internal).norm() < 1e-12
    assert np.max(np.abs(tr_s.expectations["P1[0]"] - tr_p.expectations["P1[0]"])) < 1e-12
    assert (tr_s.final.joint - tr_p.final.joint).norm() < 1e-12, "the same trajectories under the same seeds"
    assert tr_s.jumps == tr_p.jumps and len(tr_s.jumps) >= 1, (
        "the same jump records, and some jumps to compare"
    )


def test_tomography_over_workers_matches_the_in_process_run(heating_fixture) -> None:
    """The isometry route's eight columns (four and the keyed-tolerance probe's four) over at most four workers reproduce the
    in-process Choi matrix to 1e-10."""
    dev, _drives, sched, space, _table = heating_fixture
    model = MotionalModel(reduced={}, nbar={m: 0.0 for m in range(6)}, frozen=(0, 1, 4, 5))
    recs = {}
    for mp in ("serial", "parallel"):
        eng = JointExactEngine()
        opts = Numerics(map=mp, workers=min(N_WORKERS, 8))
        recs[mp] = eng.tomography(dev, sched, space, model, quiet_sample(), SeedSpec(0), opts)
    assert recs["serial"].route == recs["parallel"].route == "isometry"
    # four basis columns plus the four of the keyed tolerance's ten-times-tighter probe on the (only) branch
    assert recs["serial"].engine_runs == recs["parallel"].engine_runs == 8
    assert recs["serial"].tolerances == recs["parallel"].tolerances == (1e-8, 1e-6)
    assert recs["serial"].tolerance_change == pytest.approx(recs["parallel"].tolerance_change, rel=1e-6)
    assert recs["serial"].workers == 1 and 1 <= recs["parallel"].workers <= 4, (
        "four columns, at most four processes"
    )
    assert np.max(np.abs(recs["serial"].choi - recs["parallel"].choi)) < 1e-10
    assert recs["serial"].tp_residual < 1e-10 and recs["parallel"].tp_residual < 1e-10


# ---- the propagator cache ------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def carrier_fixture():
    dev = chain_device(2)
    drives = raman_gate_drives(2)
    rabi, _stark = derived_seeds(dev, drives)
    pulse = single_qubit_pulse(
        0, math.pi / 2.0, 0.3, drives[0], rabi[(0, 0)], 0.0, gate_id="gpi2", programmed=False
    )
    sched = Schedule((pulse,), (), (), {0: 0.0, 1: 0.0})
    space = HilbertSpace((2, 2), (), None, (0, 1, 2, 3, 4, 5))
    return dev, sched, space


def test_propagator_cache_serves_repeated_segments_and_matches_the_ode_path(carrier_fixture) -> None:
    """On an internal-state-only space the segment propagator is integrated once and then served from the cache, every state
    matching the per-state ODE path to 1e-9."""
    dev, sched, space = carrier_fixture
    eng = JointExactEngine()
    states = [space.initial_state([0, 0]), space.initial_state([1, 0]), space.initial_state([0, 1])]
    finals = []
    for k, st in enumerate(states):
        tr = eng.run_pulses(dev, sched, st, space, quiet_sample(), SeedSpec(0), Numerics())
        rep = eng.last_report
        assert rep is not None
        finals.append(tr.final.joint)
        seg = [s for s in rep.segments if s.pulses][0]
        if k == 0:
            assert rep.propagator_solves == 1 and rep.propagator_cache_hits == 0
            assert seg.integrator.startswith("dop853[propagator]")
        else:
            assert rep.propagator_solves == 0 and rep.propagator_cache_hits == 1
            assert seg.integrator == "propagator[cached]"
        assert seg.kernel == "assembled", "a carrier on an all-frozen space keeps the tiny CSR operator"
    ref_engine = JointExactEngine()
    for st, final in zip(states, finals):
        tr = ref_engine.run_pulses(
            dev, sched, st, space, quiet_sample(), SeedSpec(0), Numerics(propagator_cache=False)
        )
        rep = ref_engine.last_report
        assert rep is not None and rep.propagator_solves == 0 and rep.propagator_cache_hits == 0
        assert (tr.final.joint - final).norm() < 1e-9
    # the pi/2 pulse: P1 of ion 0 near one half, reduced by the frozen modes' Debye-Waller factors
    p1 = float(np.real(np.diag(finals[0].proj().ptrace(0).full())[1]))
    assert abs(p1 - 0.5) < 0.02


def test_tomography_of_a_carrier_step_integrates_one_propagator_per_branch(
    carrier_fixture, monkeypatch
) -> None:
    """A carrier step's propagator route integrates one propagator per frozen Fock branch as that branch's Kraus operator, the
    state route reuses the cached propagators for the sixteen inputs, and both give the same channel to 1e-12."""
    dev, sched, space = carrier_fixture
    model = MotionalModel(
        reduced={}, nbar={0: 0.0, 1: 0.0, 2: 0.05, 3: 0.05, 4: 0.0, 5: 0.0}, frozen=tuple(range(6))
    )
    eng = JointExactEngine()
    rec = eng.tomography(
        dev, sched.pulses[0], space, model, quiet_sample(), SeedSpec(0), Numerics(branch_weight_min=0.02)
    )
    assert rec.route == "propagator" and rec.branches >= 2 and rec.engine_runs == rec.branches
    assert sum(r.propagator_solves for r in rec.reports) == rec.branches
    assert sum(r.propagator_cache_hits for r in rec.reports) == 0
    assert all(s.integrator.endswith("[propagator]") for r in rec.reports for s in r.segments if s.pulses)
    assert rec.tp_residual < 1e-10 and rec.cp_residual < 1e-10 and cp_residual(rec.choi_raw) < 1e-13
    assert rec.n_traj == 1 and rec.method == "sesolve" and rec.workers == 1 and rec.motional_out == {}
    ref_engine = JointExactEngine()
    monkeypatch.setattr(ref_engine, "is_unitary", lambda *args, **kwargs: False)  # the "states" reference
    ref = ref_engine.tomography(
        dev,
        sched.pulses[0],
        space,
        model,
        quiet_sample(),
        SeedSpec(0),
        Numerics(branch_weight_min=0.02),
    )
    assert ref.route == "states" and ref.branches == rec.branches and ref.engine_runs == 16 * ref.branches
    assert sum(r.propagator_solves for r in ref.reports) == ref.branches
    assert sum(r.propagator_cache_hits for r in ref.reports) == 16 * ref.branches - ref.branches
    assert np.max(np.abs(rec.choi - ref.choi)) < 1e-12
    assert max(np.max(np.abs(a - b)) for a, b in zip(rec.outputs, ref.outputs)) < 1e-12
    # the propagator itself: U applied to an input reproduces the engine's final ket from the same cached propagator
    u, rep = eng.propagator(dev, sched, space, quiet_sample(), SeedSpec(0), Numerics(), motional_model=model)
    assert u.shape == (4, 4) and np.max(np.abs(u.conj().T @ u - np.eye(4))) < 1e-9
    assert rep.propagator_solves + rep.propagator_cache_hits == 1 and rep.method == "sesolve"
    st = space.initial_state([1, 0])
    tr = eng.run_pulses(dev, sched, st, space, quiet_sample(), SeedSpec(0), Numerics())
    assert tr.final.joint is not None
    assert (
        np.max(
            np.abs(
                u @ np.asarray(st.joint.full()).reshape(-1) - np.asarray(tr.final.joint.full()).reshape(-1)
            )
        )
        < 1e-12
    )
    with pytest.raises(ValueError):
        eng.propagator(
            dev,
            sched,
            HilbertSpace((2, 2), (ModeTruncation(2, 6, (0, 1), 0.1),), None, (0, 1, 3, 4, 5)),
            quiet_sample(),
            SeedSpec(0),
            Numerics(),
        )


# ---- run() over workers ---------------------------------------------------------------------------------------------------------------

GPI2 = Circuit(2, (Operation("gpi2", (0,), (0.0,)),), (0, 1))


@pytest.fixture(scope="module")
def two_ion():
    fx = yb171_chain(2)
    sur = two_ion_surrogate(1000)
    return fx, sur


def _run_both(circuit, fx, sur, shots, **kw):
    out = {}
    for mp, workers in (("serial", 1), ("parallel", min(N_WORKERS, 6))):
        opts = Numerics(map=mp, workers=workers, **kw)
        out[mp] = run(
            circuit,
            fx.device,
            shots,
            table=sur.table,
            keep_final_state=True,
            level="JOINT_EXACT",
            numerics=opts,
        )
    return out["serial"], out["parallel"]


def test_run_over_workers_reproduces_the_in_process_run_on_a_carrier_circuit(two_ion) -> None:
    """The GPi2 circuit's branches over the workers give the in-process bitstrings and register state (1e-12), with the worker
    count and one propagator per frozen Fock tuple reported."""
    fx, sur = two_ion
    a, b = _run_both(GPI2, fx, sur, 200, branch_weight_min=1e-9)
    assert np.array_equal(a.bitstrings, b.bitstrings)
    assert a.final_state is not None and b.final_state is not None
    assert np.max(np.abs(np.asarray(a.final_state.full()) - np.asarray(b.final_state.full()))) < 1e-12
    expected = min(worker_count(Numerics(map="parallel", workers=N_WORKERS)), 6)
    if expected < 2:
        pytest.skip("the memory cap left no room for a parallel run on this machine")
    assert a.diagnostics.workers == 1 and b.diagnostics.workers == expected
    assert a.diagnostics.trajectories == b.diagnostics.trajectories >= 2
    # one propagator per distinct Fock tuple of the coupled frozen modes (their Debye-Waller factors change H); every internal
    # branch sharing that tuple is a cache hit
    rec = last_record(a)
    distinct = len({tuple(sorted(br.fock.items())) for br in rec.branches})
    assert a.diagnostics.propagator_cache_hits == len(rec.branches) - distinct >= 1
    assert a.diagnostics.kernel == b.diagnostics.kernel == "assembled"


@pytest.mark.slow
def test_run_over_workers_reproduces_the_in_process_run_on_the_bell_circuit(two_ion) -> None:
    """The Bell circuit (factorized kernel) run serially and in parallel agrees shot by shot and in the register state to
    1e-12."""
    fx, sur = two_ion
    if worker_count(Numerics(map="parallel", workers=min(N_WORKERS, 6))) < 2:
        pytest.skip("the memory cap left no room for a parallel run on this machine")
    a, b = _run_both(BELL, fx, sur, 300, branch_weight_min=1e-3)
    assert np.array_equal(a.bitstrings, b.bitstrings)
    assert a.final_state is not None and b.final_state is not None
    assert np.max(np.abs(np.asarray(a.final_state.full()) - np.asarray(b.final_state.full()))) < 1e-12
    assert a.diagnostics.kernel == "factorized" and b.diagnostics.kernel == "factorized"
    assert b.diagnostics.workers > 1
    assert a.probabilities["00"] + a.probabilities["11"] > 0.97


def test_improved_sampling_trajectories_agree_over_workers_on_a_single_ion_heating_pulse() -> None:
    """Four improved-sampling trajectories and the no-jump member of a heated single-ion carrier pulse agree in-process and over
    every CPU: register, P1 and joint state to 1e-12, and the jumps."""
    base = single_ion_raman_device()
    dev = dataclasses.replace(
        base,
        noise=dataclasses.replace(
            base.noise, S_E=white_spectrum(1e-9, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    pulse = Pulse(square_drive(dd, include_stark=False), 0.0, 10e-6, "p", ())
    sched = Schedule((pulse,), (), (), {0: 0.0})
    x_mode = 1  # eta ~ 0.111 on the 3 MHz x mode of the single-ion fixture
    others = tuple(m for m in range(len(dev.crystal.modes)) if m != x_mode)
    space = HilbertSpace((2,), (ModeTruncation(x_mode, 14, (0, 6), 0.12),), None, others)
    state = space.initial_state([0])
    if worker_count(Numerics(map="parallel", workers=N_WORKERS)) < 2:
        pytest.skip("the memory cap left no room for a parallel run on this machine")
    out = {}
    for mp, workers in (("serial", 1), ("parallel", N_WORKERS)):
        eng = JointExactEngine(device_channels=True)
        opts = Numerics(lindblad_method="mcsolve", ntraj=4, map=mp, workers=workers, improved_sampling=True)
        tr = eng.run_pulses(dev, sched, state, space, quiet_sample(), SeedSpec(5), opts)
        rep = eng.last_report
        # the members of the weighted mixture: four stochastic ones conditioned on jumping plus the no-jump member
        assert rep is not None and rep.method == "mcsolve" and rep.trajectories == 5
        assert rep.map == mp
        out[mp] = tr
    tr_s, tr_p = out["serial"], out["parallel"]
    assert (tr_s.final.internal - tr_p.final.internal).norm() < 1e-12
    assert np.max(np.abs(tr_s.expectations["P1[0]"] - tr_p.expectations["P1[0]"])) < 1e-12
    assert (tr_s.final.joint - tr_p.final.joint).norm() < 1e-12, "the same members under the same seeds"
    assert tr_s.jumps == tr_p.jumps and len(tr_s.jumps) >= 4, "every stochastic member jumped at least once"
