"""``Traces.mode_marginal`` and the wall times (docs/api_implementation_plan.md 3.2; 0.4.0): opt-in through
``Numerics.integration.store_marginals``, one (T, d_m) Fock distribution per carried mode whose mean is ``mode_occupations``
at every stored time; the per-pulse wall time sums to the segments' and is keyed by the pulses' gate ids."""

from __future__ import annotations

import numpy as np
import pytest

import qutip_trap as trap
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.machine import Machine
from qutip_trap.run.job import last_record
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))


@pytest.fixture(scope="module")
def stored() -> tuple[Machine, trap.Result]:
    fx = circuit_fixture(2)
    machine = Machine(
        fx.device,
        numerics=trap.Numerics(truncation={"branch_weight_min": 1e-3}, integration={"store_marginals": True}),
    ).calibrated(pairs=[(0, 1)], detection_records=500)
    return machine, machine.run(BELL, 100, seed=0)


def test_the_marginal_is_a_distribution_whose_mean_is_the_occupation_at_every_stored_time(
    stored: tuple[Machine, trap.Result],
) -> None:
    _machine, result = stored
    rec = last_record(result)
    assert rec.traces, "a JOINT_EXACT run stores one trace per (sample, branch)"
    for tr in rec.traces:
        assert tr.mode_marginal is not None and set(tr.mode_marginal) == set(tr.mode_occupations)
        for m, dist in tr.mode_marginal.items():
            d_m = next(t.d for t in result.diagnostics.space.resolved if t.mode == m)
            assert dist.shape == (tr.times_s.size, d_m), (m, dist.shape)
            assert np.all(dist >= -1e-12)
            # the integrator runs with normalize_output off (Section 5.3): the norm drifts by the solver tolerance
            assert np.max(np.abs(dist.sum(axis=1) - 1.0)) < 1e-6
            mean = dist @ np.arange(d_m)
            assert np.max(np.abs(mean - tr.mode_occupations[m])) < 1e-8, (
                m,
                np.max(np.abs(mean - tr.mode_occupations[m])),
            )
            # the last row is the final reduced motional state's diagonal
            final = np.real(np.diag(np.asarray(tr.final.motional.reduced[m].full())))
            assert np.max(np.abs(dist[-1] - final)) < 1e-9


def test_the_marginal_is_off_by_default_and_costs_nothing_then(stored: tuple[Machine, trap.Result]) -> None:
    machine, result = stored
    plain = Machine(
        machine.device, table=machine.table, numerics=trap.Numerics(truncation={"branch_weight_min": 1e-3})
    )
    assert (
        not plain.numerics.integration.store_marginals
        and not plain.numerics.to_solver_options().store_marginals
    )
    quiet = plain.run(BELL, 100, seed=0)
    for tr in last_record(quiet).traces:
        assert tr.mode_marginal is None
    assert np.array_equal(quiet.bitstrings, result.bitstrings), "storing the marginals changes no number"
    assert machine.numerics.to_solver_options().store_marginals


def test_wall_time_is_keyed_by_the_pulses_and_the_segments_carry_it(
    stored: tuple[Machine, trap.Result],
) -> None:
    machine, result = stored
    rec = last_record(result)
    gate_ids = {p.gate_id for p in rec.schedule.pulses}
    for tr in rec.traces:
        assert tr.wall_time_s and set(tr.wall_time_s) <= gate_ids | {"idle"}
        assert gate_ids <= set(tr.wall_time_s), "every pulse of the schedule was integrated and timed"
        assert all(v >= 0.0 for v in tr.wall_time_s.values()) and sum(tr.wall_time_s.values()) > 0.0
    # the same pulses through the engine directly: the report's per-segment numbers sum to the traces' per-pulse ones
    from qutip_trap.dynamics.engine import SeedSpec
    from qutip_trap.noise.sampling import quiet_sample

    space = result.diagnostics.space
    opts = machine.numerics.to_solver_options(machine.physics)
    # ONE pure branch (the register and the resolved modes in their ground states): the run itself enumerates the prepared
    # mixture into weighted branches before the engine sees them, and handing the engine the whole mixture would make it
    # eigen-decompose and evolve hundreds of branches
    state = space.initial_state([0] * machine.device.crystal.n_ions, fock={t.mode: 0 for t in space.resolved})
    engine = machine.engine
    traces = engine.run_pulses(machine.device, rec.schedule, state, space, quiet_sample(0), SeedSpec(0), opts)
    report = engine.last_report
    assert report is not None and report.segments
    assert all(seg.wall_time_s > 0.0 for seg in report.segments)
    assert sum(seg.wall_time_s for seg in report.segments) == pytest.approx(
        sum(traces.wall_time_s.values()), rel=1e-9
    )
    assert traces.mode_marginal is not None and set(traces.mode_marginal) == set(traces.mode_occupations)
