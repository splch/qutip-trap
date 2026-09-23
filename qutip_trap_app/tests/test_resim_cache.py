"""The re-simulation cache: zooming into a pulse twice recomputes once, a forced recomputation equals the cached trace to
solver tolerance, and the zoomed entangling pulse of the Bell run reads as the physics says."""

from __future__ import annotations

import numpy as np
from fixtures import ms_step

from qutip_trap_app.record import BoundaryState, LiveRun, Record
from qutip_trap_app.resim import DEFAULT_ZOOM_POINTS, boundary_key, zoom
from qutip_trap_app.viewmodel.dynamics import pulse_dynamics
from qutip_trap_app.viewmodel.numerics import numerics_panel


def test_zoom_twice_recomputes_once_and_matches_a_fresh_one(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = ms_step(record)
    record, z1, cached = zoom(record, live, step)
    assert not cached and z1.n_store == DEFAULT_ZOOM_POINTS and z1.trace.times_s.size > 200
    record, z2, cached = zoom(record, live, step)
    assert cached and z2 is z1
    record, z3, cached = zoom(record, live, step, force=True)
    assert not cached and z3.key == z1.key and record.cache[z1.key] is z3
    assert np.array_equal(z1.trace.times_s, z3.trace.times_s)
    for key, values in z1.trace.expectations.items():
        assert np.max(np.abs(values - z3.trace.expectations[key])) < 1e-10, key
    assert record.cached(boundary_key(step, 0, 0), BoundaryState) is not None, (
        "the chain up to the pulse is cached"
    )


def test_the_zoomed_entangling_pulse(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = ms_step(record)
    record, z, _cached = zoom(record, live, step)
    assert z.engine_dimension == record.space.dimension and z.wall_time_s < 60.0
    dyn = pulse_dynamics(record, z)
    assert dyn.gate_id.startswith("ms") and dyn.times_s.size == z.trace.times_s.size
    assert len(dyn.populations) == 2 and len(dyn.nbar) == 2
    # two ions times two coupled modes of spin-branch loops; the trace's spin-averaged <a_m> per mode is a residue
    assert len(dyn.loops) == 4 and len(dyn.mean_alpha) == 2
    for loop in dyn.loops:
        assert loop.ion in (0, 1) and loop.excursion > 0.05
        assert loop.closes < 0.05 * loop.excursion, "the calibrated waveform returns the motion"
    assert dyn.concurrence is not None
    assert dyn.concurrence.values[-1] > 0.95 and dyn.concurrence.values[0] < 0.05
    # the integrator runs unnormalized: the norm drifts by the solver tolerance and the Fock populations carry the drift
    assert dyn.norm_deficit is not None
    deficit = float(dyn.norm_deficit.value or 0.0)
    assert abs(deficit) < 1e-7
    assert all(abs(v.sum() - (1.0 - deficit)) < 1e-12 for v in dyn.fock_end.values())
    assert z.trace.mode_marginal is not None, "the zoom stores the Fock populations at every point"
    panel = numerics_panel(record, zoom=z)
    assert panel.badge.status == "not checked" and not panel.badge.reasons
    assert all(float(b.value or 0.0) < 1e-6 for b in panel.boundary)
