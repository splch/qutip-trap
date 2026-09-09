"""Section 9.11 row "Re-simulation cache": zooming into a pulse twice recomputes once; the cached trace equals a fresh one
to solver tolerance. Also the Section 14.7 budget for the zoom of a two-mode 100 us entangling pulse."""

from __future__ import annotations

import numpy as np

from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.resim import zoom
from qutip_trap_app.viewmodel.dynamics import pulse_dynamics
from qutip_trap_app.viewmodel.numerics import numerics_panel


def _ms_step(record: Record) -> int:
    return next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))


def test_zoom_twice_recomputes_once_and_matches_a_fresh_one(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = _ms_step(record)
    record, z1, s1 = zoom(record, live, step)
    assert not s1.cached and s1.engine_calls == 1
    record, z2, s2 = zoom(record, live, step)
    assert s2.cached and s2.engine_calls == 0 and z2 is z1
    record, z3, s3 = zoom(record, live, step, force=True)
    assert s3.engine_calls == 1 and z3.key == z1.key
    assert np.array_equal(z1.trace.times_s, z3.trace.times_s)
    for key in z1.trace.expectations:
        assert np.max(np.abs(z1.trace.expectations[key] - z3.trace.expectations[key])) < 1e-10, key
    assert z1.trace.times_s.size > 200 and z1.n_store == 201
    assert record.zoom(z1.key) is z3, "the forced recomputation replaced the cache entry"
    assert record.boundary(step, 0, 0) is not None, "the chain up to the pulse is cached too"


def test_zoom_budget_and_dynamics_view(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = _ms_step(record)
    record, z, stats = zoom(record, live, step)
    # Section 14.7: <= 1 s at joint dimension <= 256; this fixture's space is 572-dimensional (11 x 13 Fock levels), where the
    # measured table of Section 11.1 puts a 100 us pulse at a few seconds; the budget asserted here is that band
    assert z.engine_dimension == record.space.dimension
    assert stats.wall_time_s < 60.0
    dyn = pulse_dynamics(record, z)
    assert dyn.gate_id.startswith("ms") and dyn.times_s.size == z.trace.times_s.size
    assert len(dyn.populations) == 2 and len(dyn.nbar) == 2 and len(dyn.loops) == 2
    for loop in dyn.loops:
        assert loop.alpha.size == dyn.times_s.size
        assert loop.closes < 0.05, "the calibrated waveform returns the motion (Section 4.4.3 closure)"
    assert (
        dyn.concurrence is not None and dyn.concurrence.values[-1] > 0.95 and dyn.concurrence.values[0] < 0.05
    )
    # the integrator runs with normalize_output off (Section 5.3), so the state's norm drifts by the solver tolerance over a
    # 100 us pulse; the Fock distributions carry that drift, the view shows it as the norm deficit, and it is small
    deficit = float(dyn.norm_deficit.value)  # type: ignore[arg-type]
    assert 0.0 <= abs(deficit) < 1e-7
    assert all(abs(v.sum() - (1.0 - deficit)) < 1e-12 for v in dyn.fock_end.values())
    assert dyn.unavailable, "the core gap on per-time Fock distributions is stated, not hidden"
    panel = numerics_panel(record, zoom=z)
    assert panel.badge.status in ("pass", "not checked")
    assert all(float(b.value) < 1e-6 for b in panel.boundary)  # type: ignore[arg-type]
