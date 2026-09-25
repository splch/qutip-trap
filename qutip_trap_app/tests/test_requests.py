"""XX(0.3) requested at Level 1 yields a pulse whose simulated
unitary is XX(0.3) within the calibration tolerance; a detuning set by hand at Level 2 shows the actual unitary at Level 1.

Both requests are new jobs made from the Bell record (nothing on the record is edited); each runs at the full engine and its
MS step is read back by process tomography (about a minute each)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap_app import requests as rq
from qutip_trap_app import resim
from qutip_trap_app.record import LiveRun, Record, execute, shift_detuning, table_with_overrides
from qutip_trap_app.viewmodel.circuit import register_after, timeline
from qutip_trap_app.viewmodel.dynamics import closure_table, pulse_dynamics, recorded_zoom

CHI_REQUEST = 0.3
ANGLE_TOLERANCE_RAD = 5e-3
"""The exact spot check converges the calibrated angle to 1e-4 rad at pi/4 (Section 7.8); the s-squared rescale to another
angle is exact in the Lamb-Dicke closed form and leaves the beyond-Lamb-Dicke terms of Section 4.4.3 at the 1e-3 level, and the
tomography's CP/TP projection adds its own residual: 5e-3 rad is the band the row is judged in."""


def _ms_gate(record: Record) -> str:
    return next(g.gate_id for g in timeline(record) if g.name.value == "ms")


def test_requests_are_formed_or_refused_with_a_reason(bell: tuple[Record, LiveRun]) -> None:
    record, _live = bell
    gate = _ms_gate(record)
    ok = rq.request_angle(record, gate, CHI_REQUEST)
    assert ok.accepted and ok.job is not None and ok.refusal is None
    assert ok.calibrated_chi_rad is not None and abs(ok.calibrated_chi_rad - math.pi / 4.0) < 2e-4, (
        "the table's angle is the exact spot check's, within its tolerance of pi/4 (Section 7.8)"
    )
    assert ok.scale is not None and math.isclose(ok.scale, math.sqrt(CHI_REQUEST / ok.calibrated_chi_rad))
    assert ok.job.requests and "XX(0.3)" in ok.job.requests[0] and ok.job.label.startswith("request")
    k = rq.gate_index(gate)
    op = ok.job.circuit.ops[k]
    assert op.name == "ms" and math.isclose(op.params[2], 2.0 * CHI_REQUEST)
    assert record.compiled.native.ops[k].params[2] == pytest.approx(math.pi / 2.0), (
        "the record itself is untouched"
    )
    # refusals carry their reason (Section 14.4: "shown with the reason")
    zero = rq.request_angle(record, gate, 0.0)
    assert not zero.accepted and zero.refusal and "no pulse" in zero.refusal
    wide = rq.request_angle(record, gate, 2.0)
    assert not wide.accepted and wide.refusal and "calibrated range" in wide.refusal
    with pytest.raises(rq.RequestError):
        rq.request_angle(record, next(g.gate_id for g in timeline(record) if g.name.value == "gpi2"), 0.3)
    det = rq.request_detuning(record, gate, 5e3)
    assert det.accepted and det.job is not None and det.job.waveform_overrides
    key = next(iter(det.job.waveform_overrides))
    assert set(int(x) for x in key.split(",")) == set(det.pair) and det.job.waveform_overrides[key] == 5e3
    huge = rq.request_detuning(record, gate, 5e6)
    assert not huge.accepted and huge.refusal and "half" in huge.refusal
    negative = rq.request_angle(record, gate, -CHI_REQUEST)
    assert negative.accepted and negative.job is not None
    assert math.isclose(negative.job.circuit.ops[k].params[1], op.params[1] + math.pi), (
        "a negative chi flips phi_1 by pi"
    )


def test_power_limit_is_the_carrier_or_declared_absent(bell: tuple[Record, LiveRun]) -> None:
    """A request whose rescaled amplitude exceeds the calibrated carrier Rabi frequency is refused with the numbers when the
    calibrated waveform sits below that carrier; when the table's entangling tones already exceed the carrier entry (the
    example device) the app declares no limit rather than inventing one, and says so in the note."""
    record, _live = bell
    gate = _ms_gate(record)
    tg = rq.target_of(record, gate)
    pair = (int(tg.ions[0]), int(tg.ions[1]))
    wf = rq.pair_waveform(record, pair)
    assert wf is not None
    carriers = [c for c in (rq.carrier_rabi_hz(record, i) for i in pair) if c is not None]
    assert carriers, "the table carries the entangling drive's carrier Rabi frequency"
    peak = rq.peak_amplitude_hz(wf)
    widest = rq.request_angle(record, gate, math.pi / 2.0)
    if peak <= min(carriers):
        limit = abs(wf.chi_total_rad) * (min(carriers) / peak) ** 2
        if limit < math.pi / 2.0:
            assert not widest.accepted and widest.refusal and "power limit" in widest.refusal
        else:
            assert widest.accepted and widest.carrier_rabi_hz == min(carriers)
    else:
        assert widest.accepted and widest.carrier_rabi_hz is None
        assert "no power limit" in widest.note


def test_shift_detuning_moves_the_legs_symmetrically(bell: tuple[Record, LiveRun]) -> None:
    _record, live = bell
    pair = next(iter(live.table.ms))
    wf = live.table.ms[pair]
    shifted = shift_detuning(wf, 5e3)
    assert shifted.segments is not None and wf.segments is not None
    for s0, s1 in zip(wf.segments, shifted.segments):
        for leg, v in s0.detuning_hz.items():
            v1 = s1.detuning_hz[leg]
            if callable(v):
                assert callable(v1) and math.isclose(v1(0.0), v(0.0) + (5e3 if leg == "blue" else -5e3))
            else:
                assert math.isclose(float(v1), float(v) + (5e3 if leg == "blue" else -5e3))
        assert s0.amplitude_hz == s1.amplitude_hz, "the amplitude is not touched: the pulse plays as written"
    assert shifted.chi_m == wf.chi_m, "the table's booked angle stays: the scheduler rescales by it"
    table = table_with_overrides(live.table, {f"{pair[0]},{pair[1]}": 5e3})
    assert table.ms[pair] is not wf and table_with_overrides(live.table, {}) is live.table


@pytest.fixture(scope="module")
def requested_angle(bell: tuple[Record, LiveRun]) -> tuple[Record, LiveRun, str]:
    record, _live = bell
    gate = _ms_gate(record)
    req = rq.request_angle(record, gate, CHI_REQUEST)
    assert req.job is not None
    new_record, live = execute(req.job)
    step = new_record.step_of_gate(gate).index
    new_record, _pm = resim.process_matrix(new_record, live, step, 0, 0)
    return new_record, live, gate


def test_requested_angle_is_the_simulated_unitary_within_tolerance(
    requested_angle: tuple[Record, LiveRun, str],
) -> None:
    """XX(0.3) requested at Level 1 yields a pulse whose simulated unitary is XX(0.3) within the calibration tolerance."""
    record, _live, gate = requested_angle
    tg = rq.target_of(record, gate)
    phi0, phi1, theta = rq.ms_phases_of(np.asarray(tg.unitary, dtype=complex))
    assert math.isclose(theta, 2.0 * CHI_REQUEST, abs_tol=1e-9), "the target shows the requested unitary"
    step = record.step_of_gate(gate).index
    pm = record.process_matrix(resim.process_matrix_key(step, 0, 0))
    assert pm is not None
    out = rq.request_outcome(record, gate, pm)
    assert math.isclose(out.requested_chi_rad, CHI_REQUEST, abs_tol=1e-9)
    assert abs(out.fitted_chi_rad - CHI_REQUEST) < ANGLE_TOLERANCE_RAD, (
        out.fitted_chi_rad,
        out.fitted_fidelity,
    )
    assert out.infidelity_to_requested < 5e-3, out.infidelity_to_requested
    assert out.fitted_fidelity > 1.0 - 5e-3
    assert record.job.requests and "XX(0.3)" in record.job.requests[0]
    # the target distribution is that of the requested circuit (the native sequence with MS(0.6) in it), not the Bell state's
    assert abs(record.results.target_probabilities["00"] - 0.5) > 0.05
    assert sum(record.results.target_probabilities.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.fixture(scope="module")
def hand_set_detuning(bell: tuple[Record, LiveRun]) -> tuple[Record, LiveRun, str]:
    record, _live = bell
    gate = _ms_gate(record)
    req = rq.request_detuning(record, gate, 5e3)
    assert req.job is not None
    new_record, live = execute(req.job)
    step = new_record.step_of_gate(gate).index
    new_record, _pm = resim.process_matrix(new_record, live, step, 0, 0)
    return new_record, live, gate


def test_hand_set_detuning_shows_the_actual_unitary_at_level_1(
    bell: tuple[Record, LiveRun], hand_set_detuning: tuple[Record, LiveRun, str]
) -> None:
    """A detuning set by hand at Level 2 is applied as written; Level 1 then shows the actual unitary, not the requested one."""
    original, _ = bell
    record, _live, gate = hand_set_detuning
    tg = rq.target_of(record, gate)
    tg0 = rq.target_of(original, gate)
    assert np.allclose(tg.unitary, tg0.unitary), "the requested (target) unitary is unchanged"
    assert record.job.waveform_overrides and record.job.requests
    # Level 2: the played waveform carries the shifted legs, its booked angle the table's
    wf = rq.pair_waveform(record, (int(tg.ions[0]), int(tg.ions[1])))
    wf0 = rq.pair_waveform(original, (int(tg0.ions[0]), int(tg0.ions[1])))
    assert wf is not None and wf0 is not None and wf.segments is not None and wf0.segments is not None
    blue1 = wf.segments[0].detuning_hz["blue"]
    blue0 = wf0.segments[0].detuning_hz["blue"]
    assert (
        blue1.value is not None and blue0.value is not None and math.isclose(blue1.value - blue0.value, 5e3)
    )
    # Level 3: the loops of the played waveform no longer close
    step = record.step_of_gate(gate).index
    dyn = pulse_dynamics(record, recorded_zoom(record, step, 0, 0))
    closes, excursions = closure_table(dyn)
    assert closes and max(closes[m] / max(excursions[m], 1e-12) for m in closes) > 0.05, "an open loop"
    dyn0 = pulse_dynamics(original, recorded_zoom(original, step, 0, 0))
    closes0, excursions0 = closure_table(dyn0)
    assert max(closes0[m] / max(excursions0[m], 1e-12) for m in closes0) < 0.05, "the calibrated pulse closes"
    # Level 1: the actual unitary from tomography differs from the requested one, and the register's fidelity fell
    pm = record.process_matrix(resim.process_matrix_key(step, 0, 0))
    assert pm is not None
    out = rq.request_outcome(record, gate, pm)
    assert out.hand_set_detuning_hz == 5e3
    assert out.infidelity_to_requested > 1e-3, out.infidelity_to_requested
    gates = timeline(record)
    k = next(g.index for g in gates if g.gate_id == gate)
    fid = float(register_after(record, k).fidelity.value or 0.0)
    fid0 = float(register_after(original, k).fidelity.value or 0.0)
    assert fid < fid0 - 1e-3, (fid, fid0)
