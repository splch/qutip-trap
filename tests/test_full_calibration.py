"""The full calibration by simulated experiments end to end (PLAN.md Section 7.5; milestone M8): ``calibrate(surrogate=False)`` on
the two-ion 171Yb+ fixture with reduced scans follows the dependency graph, replaces every seed by a calibrated entry whose value
agrees with the device's true derived value within the uncertainty the fit reports, stores the entangling waveform at its closure
amplitude with aligned phases, and a Bell circuit run from that table (the scheduler reading only the table, the ions seeing the
device through the played chain) reaches the fidelity the noise model predicts: inside the intrinsic budget plus the calibration's
own uncertainty, and within statistics of the run from the surrogate table."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qutip_trap.api import Circuit, Operation, SolverOptions, calibrate, register_fidelity, run
from qutip_trap.calibration import CalibrationScans, calibrate_with_report
from qutip_trap.calibration.experiments import CalibrationReport
from qutip_trap.control.table import usable
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive
from tests.m6_fixtures import circuit_fixture

BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
SCANS = CalibrationScans(
    shots=400,
    rabi_points=33,
    ramsey_delays_s=tuple(float(x) for x in np.linspace(0.0, 2e-3, 7)),
    stark_delays_s=tuple(float(x) for x in np.linspace(0.0, 2e-3, 9)),
    mode_span=0.01,
    mode_coarse_points=7,
    mode_fine_points=11,
    crosstalk_points=12,
    crosstalk_phase_points=6,
    ms_amplitude_points=5,
    ms_detuning_offsets_hz=(-2e3, 0.0, 2e3),
    parity_points=6,
    phase_points=4,
    phase_inputs=("00", "01"),
    detection_records=1500,
    detection_windows_s=WINDOWS,
    micromotion_ranges={},
)


@pytest.fixture(scope="module")
def calibrated():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    report = calibrate_with_report(
        fx.device,
        surrogate=False,
        seed=11,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        pairs=[(0, 1)],
        scans=SCANS,
        detection_records=1500,
        detection_windows_s=WINDOWS,
        cache=None,
    )
    assert isinstance(report, CalibrationReport)
    return fx, report


@pytest.mark.slow
def test_every_calibrated_entry_agrees_with_the_derived_truth_within_its_uncertainty(calibrated) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5: 'calibrated parameters agree with the device's true derived parameters within the uncertainty the fits report'."""
    fx, report = calibrated
    t = report.table
    dev = fx.device
    assert not t.surrogate and t.device_hash == dev.hash() and not report.refused, report.refused
    assert t.uncalibrated() == (), t.uncalibrated()
    # the field and the qubit frequencies
    assert (
        t.field.status == "calibrated" and abs(t.field.value - dev.field.B_gauss) < 4.0 * t.field.uncertainty
    )
    for i in range(2):
        sp = dev.crystal.species[i]
        f_true = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)[0]
        e = t.qubit_freq[i]
        assert e.status == "calibrated" and e.experiment == "ramsey_frequency"
        assert abs(e.value - f_true) < 4.0 * e.uncertainty and e.uncertainty < 20.0
    # the modes, occupations and Lamb-Dicke parameters the drives couple to
    for m in (2, 3):
        e = t.modes[m]
        assert e.status == "calibrated" and abs(e.value - dev.crystal.modes[m].omega_hz) < 4.0 * e.uncertainty
        assert e.uncertainty < 1e3, "the FM solvers' sub-kilohertz need (Section 7.5)"
        nb = t.nbar[m]
        assert nb.status == "calibrated" and abs(nb.value - report.surrogate.nbar[m]) < 4.0 * max(
            nb.uncertainty, 2e-3
        )
    # carrier Rabi frequencies, Stark shifts, crosstalk against the derived values
    for i in range(2):
        spec = fx.gate_drives[i]
        dd = derive_raman_drive(dev, i, spec.beams, scattering=False)
        key = (i, spec.table_key_beam)
        r = t.rabi[key]
        assert r.status == "calibrated" and abs(r.value - dd.carrier_rabi_hz) < 4.0 * r.uncertainty
        assert r.uncertainty < 2e-3 * dd.carrier_rabi_hz
        s = t.stark[key]
        assert (
            s.status == "calibrated"
            and abs(s.value - dd.stark_shift_hz) < 4.0 * s.uncertainty
            and s.uncertainty < 10.0
        )
        j = 1 - i
        x = t.crosstalk[(i, j)]
        assert (
            x.status == "calibrated"
            and abs(x.value - abs(crosstalk_ratios(dev, i, spec.beams)[j])) < 4.0 * x.uncertainty
        )
        assert (i, j) in t.crosstalk_phase and abs(t.crosstalk_phase[(i, j)].value) < 4.0 * t.crosstalk_phase[
            (i, j)
        ].uncertainty
    # the Lamb-Dicke parameters: one entry per coupled mode, on the mode's probe ion, all calibrated by the sideband Rabi
    # frequency (Section 7.9); the first M8 build looked both ions up here and raised on the one that probed nothing
    assert {m for (_ion, m) in t.lamb_dicke} == {2, 3}
    assert all(
        e.status == "calibrated" and e.experiment == "sideband_spectroscopy" for e in t.lamb_dicke.values()
    )
    # the entangling waveform: calibrated by the amplitude scan, aligned by the phase scan
    wf = t.waveform_for((0, 1))
    assert wf is not None and wf.phi_s.status == "calibrated" and wf.phi_s.experiment == "ms_phase_scan"
    assert abs(abs(wf.chi_total_rad) - math.pi / 4.0) < 1e-9
    ms = report.results["ms_scan[(0, 1)]"]
    assert ms.converged and abs(ms.fitted["closure_scale"][0] - 1.0) < 0.03
    par = report.results["parity_scan[(0, 1)]"]
    assert par.fitted["contrast"][0] > 0.97 and par.fitted["bell_fidelity_bound"][0] > 0.97
    ph = report.results["ms_phase_scan[(0, 1)]"]
    for q in (0, 1):
        assert abs(ph.fitted[f"correction_rad[{q}]"][0]) < 4.0 * ph.fitted[f"correction_rad[{q}]"][1] + 0.05
    # detection and heating (a quiet device: zero heating, calibrated as such)
    for key in ("threshold", "window_s", "eps_B", "eps_D"):
        assert t.detection[key].status == "calibrated" and t.detection[key].sample_id == 0
    coupled = {m for (_ion, m) in t.lamb_dicke}
    for m, e in t.heating.items():
        # the modes a drive couples to are measured (zero on the quiet device); the others cannot be probed and stay seeds
        assert e.value == 0.0 and e.status == ("calibrated" if m in coupled else "seed"), (m, e)
    # the audit trail the plan asks the full path for: the surrogate's error per entry
    err = report.surrogate_error()
    assert all(v < 0.02 for k, v in err.items() if k.startswith(("rabi", "modes", "qubit_freq")))
    assert all(e.fitted_at_s == 0.0 and e.sample_id == 0 for e in t.entries().values())


@pytest.mark.slow
def test_a_bell_circuit_from_the_calibrated_table_reaches_the_predicted_fidelity(calibrated) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5: 'gates built from calibrations reach the fidelity the noise model predicts'. The scheduler reads the fitted table
    (frame, pi times, compensated shifts, the waveform at its closure amplitude and aligned phases), the ions see the device through
    the played chain; the register infidelity lies inside the intrinsic budget plus the calibration's own contribution, and the
    histogram agrees with the surrogate-table run within statistics."""
    fx, report = calibrated
    kw = dict(
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        keep_final_state=True,
        options=SolverOptions(branch_weight_min=1e-3),
    )
    res = run(BELL, fx.device, 1000, table=report.table, **kw)  # type: ignore[arg-type]
    ref = run(BELL, fx.device, 1000, table=report.surrogate.table, **kw)  # type: ignore[arg-type]
    fid, fid_ref = register_fidelity(res), register_fidelity(ref)
    budget = res.diagnostics.intrinsic_budget["total"]
    # the calibration's contribution: the over-rotation (sigma_Omega/Omega x pi/2)^2 per carrier pulse and the residual frame offset
    t = report.table
    cal = 0.0
    for i in range(2):
        r = t.rabi[(i, fx.gate_drives[i].table_key_beam)]
        cal += 5 * (0.5 * math.pi * r.uncertainty / r.value) ** 2
        cal += (2.0 * math.pi * t.qubit_freq[i].uncertainty * 200e-6) ** 2
    assert 1.0 - fid < budget + 3.0 * cal + 2e-3, (fid, budget, cal)
    assert abs(fid - fid_ref) < 5e-3, (fid, fid_ref)
    p = res.probabilities
    assert p["00"] + p["11"] > 0.98 and abs(p["00"] - p["11"]) < 5.0 * res.error_bars["00"]
    assert res.diagnostics.calibration is report.table and not res.diagnostics.calibration.surrogate
    assert any("dynamical samples" in a or "nominal sample" in a for a in res.diagnostics.approximations)


@pytest.mark.slow
def test_calibrate_entry_point_caches_the_full_table_and_a_stale_table_still_runs(calibrated) -> None:  # type: ignore[no-untyped-def]
    fx, report = calibrated
    from qutip_trap.calibration import CalibrationCache

    cache = CalibrationCache()
    kw = dict(
        surrogate=False,
        seed=11,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        pairs=[(0, 1)],
        scans=CalibrationScans(
            shots=None, detection_records=300, detection_windows_s=(20e-6,), micromotion_ranges={}
        ),
        experiments=("field_scan", "detection_histogram"),
        detection_records=300,
        detection_windows_s=(20e-6,),
        cache=cache,
    )
    t1 = calibrate(fx.device, **kw)  # type: ignore[arg-type]
    t2 = calibrate(fx.device, **kw)  # type: ignore[arg-type]
    assert t1 is t2 and cache.hits == 1
    assert t1.field.experiment == "field_scan" and t1.field.status == "calibrated"
    assert all(e.status == "seed" for e in t1.rabi.values()), (
        "a subset run keeps the other entries as surrogate seeds"
    )
    assert usable(t1.detection["threshold"]) and t1.detection["threshold"].experiment == "detection_histogram"
    # a table is a snapshot with an age: run() at a later time takes it as a legitimate, possibly stale, input (Section 7.5)
    res = run(
        BELL,
        fx.device,
        50,
        table=report.table,
        t0_s=3600.0,
        gate_drives=fx.gate_drives,
        entangling_drives=fx.entangling_drives,
        options=SolverOptions(branch_weight_min=1e-2),
    )
    assert res.diagnostics.calibration.fitted_at_s == 0.0 and res.shots == 50
