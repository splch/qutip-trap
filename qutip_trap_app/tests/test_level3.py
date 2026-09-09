"""Level 3 on demand (Section 14.2 row 3; M11.3): the Hamiltonian record lists what the engine integrates, with matrix elements
that equal QuTiP's displacement operator; the Fock movie's frames equal the recorded boundary states and the fine zoom by
causality; the process matrix of the finished pulse is a channel consistent with the run; the recorded coarse trace opens a
step before any re-simulation; the closure prediction is scored against the record."""

from __future__ import annotations

import numpy as np
import pytest

from qutip_trap_app import resim
from qutip_trap_app.codec import from_document, to_document
from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.dynamics import (
    closure_table,
    fock_heatmaps,
    process_view,
    pulse_dynamics,
    recorded_zoom,
)
from qutip_trap_app.viewmodel.learn import score_closure
from qutip_trap_app.viewmodel.physics import hamiltonian_view


def _ms_step(record: Record) -> int:
    return next(s.index for s in record.schedule.steps if s.gate_id.startswith("ms"))


def test_recorded_trace_opens_the_step_at_once(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    step = _ms_step(record)
    z = recorded_zoom(record, step)
    st = record.step(step)
    assert z.trace.times_s[0] >= st.t_start_s - 1e-12 and z.trace.times_s[-1] <= st.t_end_s + 1e-12
    assert z.trace.times_s.size >= 2 and z.n_store == 0 and z.integrators == ("recorded",)
    dyn = pulse_dynamics(record, z)
    assert dyn.concurrence is not None
    # the loops are the played waveform's spin-branch trajectories alpha_im(t) on the run's own modes: two ions times the two
    # coupled radial modes; the exact trace's spin-averaged <a_m>(t) cancels between the branches and is kept as a residue
    assert len(dyn.loops) == 4 and {lp.ion for lp in dyn.loops} == {0, 1}
    assert len(dyn.mean_alpha) == 2 and all(lp.ion is None for lp in dyn.mean_alpha)
    top = max(dyn.loops, key=lambda lp: lp.excursion)
    assert top.excursion > 0.1, "a real loop, radius of order sqrt(chi)"
    assert max(lp.excursion for lp in dyn.mean_alpha) < 0.1 * top.excursion, "the branches cancel in <a_m>"
    # 2 Im integral conj(alpha_a) d alpha_b over the pair's loops is the closed-form angle at the played amplitude; the table
    # stores the exact spot check's angle, which differs by the Debye-Waller correction (Section 7.8)
    # the run plays the table's waveform rescaled to the requested angle (the scheduler books chi_m as the table's exact
    # spot-check angle times that scale), so the loop's angle is the PLAYED gate's, a hair off the table's stored one
    played = next(g for g in record.schedule.gates if g.gate_id == st.gate_id).waveform
    table_wf = record.table.waveforms["0,1"]
    assert played.chi_m.keys() == table_wf.chi_m.keys()
    for lp in dyn.loops:
        assert lp.chi_m_rad == pytest.approx(played.chi_m[lp.mode], rel=1e-12)
        assert abs(lp.chi_m_rad / table_wf.chi_m[lp.mode] - 1.0) < 1e-2
        assert lp.chi_closed_form_rad is not None
        gap = abs(lp.chi_closed_form_rad - lp.chi_m_rad) / abs(lp.chi_m_rad)
        assert 1e-3 < gap < 0.05
        assert lp.closes < 1e-3 * lp.excursion, "the solver closed every branch loop"
    closes, excursions = closure_table(dyn)
    assert set(closes) == set(excursions) == {(lp.ion, lp.mode) for lp in dyn.loops}
    assert all(v >= 0.0 for v in closes.values())
    assert score_closure("every loop closes", closes, excursions), (
        "the calibrated MS pulse returns the motion"
    )
    assert not score_closure("at least one loop stays open", closes, excursions)
    assert not score_closure("cannot be known before running", closes, excursions)
    with pytest.raises(KeyError):
        recorded_zoom(record, step, sample_index=99)


def test_hamiltonian_record_lists_what_the_engine_integrates(bell: tuple[Record, LiveRun]) -> None:
    import qutip as qt

    record, live = bell
    step = _ms_step(record)
    record, ham = resim.hamiltonian_record(record, live, step)
    assert ham.dims == tuple(record.space.dims) and ham.dimension == record.space.dimension
    assert ham.n_drive_terms == 4, "two ions, each a sigma_+ (x) D term and its conjugate"
    assert len(ham.segments) == 5 and all(s.n_drive_terms == 4 for s in ham.segments)
    assert len(ham.drives) == 2 and {d.ion for d in ham.drives} == {0, 1}
    d0 = ham.drives[0]
    assert d0.kind == "raman" and d0.beams == (0, 1) and len(d0.tone_detunings_hz) == 2
    assert abs(abs(d0.tone_detunings_hz[0]) - abs(d0.tone_detunings_hz[1])) < 500.0
    assert d0.operator_nnz > 0 and d0.debye_waller == 1.0 and d0.carrier_factor == 1.0
    for m, table in d0.matrix_elements.items():
        d = table.shape[0]
        eta = abs(d0.etas[m])
        exact = np.abs(qt.displace(d + 40, 1j * eta).full()[:d, :d])
        assert np.max(np.abs(table[: d - 6, : d - 6] - exact[: d - 6, : d - 6])) < 1e-12, (
            "Section 4.3.1's elements"
        )
        assert table[0, 0] == pytest.approx(np.exp(-(eta**2) / 2.0))
        assert table[1, 0] == pytest.approx(eta * np.exp(-(eta**2) / 2.0))
    assert ham.frame == "schrodinger" and ham.omega_max_hz > 3e6
    channels = {c.channel.split("[")[0] for c in ham.collapse}
    assert "scatter_raman" in channels, "the scattering channels are listed with their rates"
    assert all(not c.active_in_run for c in ham.collapse if c.channel.startswith("scatter"))
    assert record.hamiltonian(ham.key) is ham
    again, ham2 = resim.hamiltonian_record(record, live, step)
    assert ham2 is ham and again is record, "cached by key"
    view = hamiltonian_view(record, ham)
    assert len(view.terms) == 2 and view.terms[0].matrix_elements and view.header
    doc, arrays = to_document(record)
    back = from_document(Record, doc, arrays)
    assert back.digest() == record.digest(), "the record round-trips with the Hamiltonian record inside"


def test_fock_movie_frames_are_the_pulse_states_by_causality(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = _ms_step(record)
    record, movie = resim.fock_movie(record, live, step, n_frames=2)
    assert movie.times_s.size == 3 and movie.engine_calls == 2
    record, z, _stats = resim.zoom(record, live, step)
    resolved = [t.mode for t in record.space.resolved]
    for m, dist in movie.distributions.items():
        assert dist.shape == (3, record.space.dims[2 + resolved.index(m)])
        assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-6)
        assert np.max(np.abs(dist[0] - z.fock_start[m])) < 1e-12, "frame 0 is the recorded boundary state"
        assert np.max(np.abs(dist[-1] - z.fock_end[m])) < 1e-7, (
            "the last frame is the full pulse's end (causality)"
        )
        k = int(np.argmin(np.abs(z.trace.times_s - movie.times_s[1])))
        assert abs(movie.nbar[m][1] - z.trace.mode_nbar[m][k]) < 1e-3, (
            "the mid frame's <n> sits on the fine trace"
        )
    heat = fock_heatmaps(movie)
    assert len(heat) == 2 and all(h.values.shape[0] == 3 for h in heat)
    record2, cached = resim.fock_movie(record, live, step, n_frames=2)
    assert cached is movie and record2 is record


def test_process_matrix_of_the_finished_pulse(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = _ms_step(record)
    record, pm = resim.process_matrix(record, live, step)
    assert pm.n_inputs == 16 and pm.ions == (0, 1)
    assert pm.choi.shape == (16, 16) and abs(np.trace(pm.choi) - 1.0) < 1e-9
    assert pm.cp_tp_residual[0] < 1e-9 and pm.cp_tp_residual[1] < 1e-9
    assert 0.0 <= pm.average_gate_infidelity < 2e-3, "the calibrated MS gate on the cold register"
    assert pm.entanglement_infidelity == pytest.approx(pm.depolarizing_rate, rel=1e-9), (
        "Section 6.8's normalization"
    )
    assert pm.average_gate_infidelity == pytest.approx(0.8 * pm.entanglement_infidelity, rel=1e-9), (
        "d/(d + 1), two qubits"
    )
    assert record.results.register_fidelity is not None
    assert 1.0 - record.results.register_fidelity > 0.5 * pm.entanglement_infidelity
    pv = process_view(pm)
    assert pv.pauli[0].detail == "II" and float(pv.pauli[0].value) > 0.99  # type: ignore[arg-type]
    assert record.process_matrix(pm.key) is pm
