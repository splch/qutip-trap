"""Level 3 and the equation: the run's stored points open a step at once, with the played waveform's spin-branch loops; the
Hamiltonian record lists what the engine integrates, with matrix elements equal to QuTiP's displacement operator; the Fock
heatmap is read off the populations the zoom stored; the process matrix of the finished pulse is a channel consistent with
the run; Level 3 finds the zoom it asked for."""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import BellResimulated, ms_step

from qutip_trap_app import resim
from qutip_trap_app.record import LiveRun, Record
from qutip_trap_app.viewmodel.dynamics import fock_heatmaps, process_view, pulse_dynamics, recorded_zoom
from qutip_trap_app.viewmodel.hamiltonian import hamiltonian_view
from qutip_trap_app.views.level3 import current_zoom


def test_the_recorded_trace_opens_the_step_at_once(bell: tuple[Record, LiveRun]) -> None:
    record, live = bell
    step = ms_step(record)
    z = recorded_zoom(record, step)
    st = record.step(step)
    assert z.trace.times_s[0] >= st.t_start_s - 1e-12 and z.trace.times_s[-1] <= st.t_end_s + 1e-12
    assert z.trace.times_s.size >= 2 and z.n_store == 0 and z.integrators == ("recorded",)
    dyn = pulse_dynamics(record, z)
    assert dyn.concurrence is not None
    # the loops: two ions times the two coupled radial modes; the branches cancel in the trace's spin-averaged <a_m>
    assert len(dyn.loops) == 4 and {lp.ion for lp in dyn.loops} == {0, 1}
    assert len(dyn.mean_alpha) == 2 and all(lp.ion is None for lp in dyn.mean_alpha)
    top = max(lp.excursion for lp in dyn.loops)
    assert top > 0.1 and max(lp.excursion for lp in dyn.mean_alpha) < 0.1 * top
    played = next(g for g in record.schedule.gates if g.gate_id == st.gate_id)
    calibrated = live.run.table.waveform_for((0, 1))
    assert calibrated is not None and played.chi_m.keys() == calibrated.chi_m.keys()
    for lp in dyn.loops:
        assert lp.chi_m_rad == pytest.approx(played.chi_m[lp.mode], rel=1e-12)
        assert abs(lp.chi_m_rad / calibrated.chi_m[lp.mode] - 1.0) < 1e-2, (
            "the played angle is the table's, rescaled"
        )
        # the closed-form angle at the played amplitude differs from the booked one by the Debye-Waller correction
        assert lp.chi_closed_form_rad is not None
        assert 1e-3 < abs(lp.chi_closed_form_rad - lp.chi_m_rad) / abs(lp.chi_m_rad) < 0.05
        assert lp.closes < 1e-3 * lp.excursion, "the solver closed every branch loop"
    with pytest.raises(KeyError):
        recorded_zoom(record, step, sample_index=99)


def test_the_hamiltonian_record_lists_what_the_engine_integrates(bell_resimulated: BellResimulated) -> None:
    import qutip as qt

    r = bell_resimulated
    ham = r.hamiltonian
    assert ham.dims == r.record.space.dims and ham.dimension == r.record.space.dimension
    assert ham.n_drive_terms == 4, "two ions, each a sigma_+ (x) D term and its conjugate"
    assert len(ham.segments) == 5 and all(s.n_drive_terms == 4 for s in ham.segments)
    assert len(ham.drives) == 2 and {d.ion for d in ham.drives} == {0, 1}
    d0 = ham.drives[0]
    assert len(d0.tones) == 2 and abs(abs(d0.tones[0][0]) - abs(d0.tones[1][0])) < 500.0
    assert d0.operator_nnz > 0 and d0.debye_waller == 1.0 and d0.carrier_factor == 1.0
    for m, table in d0.matrix_elements.items():
        d = table.shape[0]
        eta = abs(d0.etas[m])
        exact = np.abs(qt.displace(d + 40, 1j * eta).full()[:d, :d])
        assert np.max(np.abs(table[: d - 6, : d - 6] - exact[: d - 6, : d - 6])) < 1e-12
        assert table[0, 0] == pytest.approx(np.exp(-(eta**2) / 2.0))
        assert table[1, 0] == pytest.approx(eta * np.exp(-(eta**2) / 2.0))
    assert ham.frame == "schrodinger" and ham.omega_max_hz > 3e6
    scattering = [c for c in ham.collapse if c.channel.startswith("scatter")]
    assert any(c.channel.startswith("scatter_raman") for c in scattering), (
        "the scattering channels are listed"
    )
    assert not any(c.integrated for c in scattering), (
        "this run estimated scattering instead of integrating it"
    )
    again, cached = resim.hamiltonian_record(r.record, r.live, r.step)
    assert cached is ham and again is r.record
    view = hamiltonian_view(ham)
    assert len(view.terms) == 2 and view.terms[0].matrix_elements and view.header


def test_the_fock_heatmap_is_read_off_the_stored_populations(bell_resimulated: BellResimulated) -> None:
    r = bell_resimulated
    z = r.zoom
    assert z.trace.mode_marginal is not None
    heat = fock_heatmaps(z, frames=5)
    assert {h.mode for h in heat} == set(z.trace.mode_marginal) == set(r.record.space.caps)
    for h in heat:
        dist = z.trace.mode_marginal[h.mode]
        assert (
            h.values.shape[0] == 5
            and h.times_s[0] == z.trace.times_s[0]
            and h.times_s[-1] == z.trace.times_s[-1]
        )
        assert np.max(np.abs(h.values[0] - z.fock_start[h.mode][: h.values.shape[1]])) < 1e-12, (
            "frame 0: the start"
        )
        assert np.max(np.abs(h.values[-1] - z.fock_end[h.mode][: h.values.shape[1]])) < 1e-9, (
            "the last: the end"
        )
        assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-6)
        k = z.trace.times_s.size // 2
        mean = float(dist[k] @ np.arange(dist.shape[1]))
        assert abs(mean - z.trace.mode_nbar[h.mode][k]) < 1e-8, "the populations' mean is <n>"


def test_the_process_matrix_of_the_finished_pulse(bell_resimulated: BellResimulated) -> None:
    r = bell_resimulated
    pm = r.process
    assert pm.n_inputs == 16 and pm.ions == (0, 1)
    assert pm.choi.shape == (16, 16) and abs(np.trace(pm.choi) - 1.0) < 1e-9
    assert pm.cp_tp_residual[0] < 1e-9 and pm.cp_tp_residual[1] < 1e-9
    assert 0.0 <= pm.average_gate_infidelity < 2e-3, "the calibrated MS gate on the cold register"
    assert pm.entanglement_infidelity == pytest.approx(pm.depolarizing_rate, rel=1e-9)
    assert pm.average_gate_infidelity == pytest.approx(0.8 * pm.entanglement_infidelity, rel=1e-9), (
        "d/(d + 1)"
    )
    assert r.record.results.register_fidelity is not None
    assert 1.0 - r.record.results.register_fidelity > 0.5 * pm.entanglement_infidelity
    pv = process_view(pm)
    assert pv.pauli[0].label == "p(II)" and float(pv.pauli[0].value or 0.0) > 0.99


def test_level3_finds_the_zoom_it_asked_for(
    bell_resimulated: BellResimulated, bell: tuple[Record, LiveRun]
) -> None:
    r = bell_resimulated
    found, fine = current_zoom(r.record, r.step, 0, 0)
    assert fine and found is r.zoom
    record, _live = bell
    coarse, fine_before = current_zoom(record, ms_step(record), 0, 0)
    assert not fine_before and coarse is not None and coarse.n_store == 0, "the stored points until then"
