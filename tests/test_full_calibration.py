"""The calibration by simulated experiments (PLAN.md Section 7.5): the end-to-end table on the two-ion 171Yb+ chain against the
derived truth and the Bell fidelity it reaches, and the parts (the Stark scan's guards, the refusal cascade, the micromotion
shims, the entangling scans' beliefs, the experiment names, the heating experiment and a miscalibrated Rabi entry)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import qutip as qt
from scipy.special import jv

from qutip_trap.calibration import CalibrationCache, calibrate
from qutip_trap.calibration.experiments import (
    ALIASES,
    ORDER,
    CalibrationReport,
    CalibrationScans,
    full_calibration,
)
from qutip_trap.calibration.surrogate import surrogate_table
from qutip_trap.control.compiler import Circuit, Operation
from qutip_trap.control.schedule import ScheduleError, schedule
from qutip_trap.control.shaping import gate_modes
from qutip_trap.control.table import CalEntry, usable
from qutip_trap.device.presets import ca40_optical, yb171_chain
from qutip_trap.experiments.entangling import _entangling_setup, ms_phase_scan, ms_scan
from qutip_trap.experiments.light import stark_scan
from qutip_trap.experiments.micromotion import device_with_compensation, micromotion_scan, signed_beta
from qutip_trap.experiments.result import ExperimentResult
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive
from qutip_trap.machine import Machine
from qutip_trap.noise.spectra import white_spectrum
from qutip_trap.options import Numerics, Physics
from qutip_trap.run.job import RunError, last_record, register_fidelity
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.pseudopotential import RfDrive
from qutip_trap.units import TWO_PI
from tests.fixtures import BELL, WINDOWS, run, two_ion_surrogate

GPI = Circuit(2, (Operation("gpi", (0,), (0.0,)),), (0,))
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
def calibrated():
    fx = yb171_chain(2)
    report = calibrate(
        Machine(fx.device),
        method="experiments",
        seed=11,
        pairs=[(0, 1)],
        scans=SCANS,
        detection_records=1500,
        detection_windows_s=WINDOWS,
        cache=None,
    )
    assert isinstance(report, CalibrationReport)
    return fx, report


SIGMA = 2.0
"""The agreement bound in units of each entry's own sigma (at seed 11 the field, qubit frequencies, modes, occupations, Rabi
frequencies and crosstalk ratios land at or below 0.77 sigma)."""

SIGMA_STARK = 2.5
"""The light shift at seed 11: ion 0 realizes 2.00 sigma (-44.306 +- 4.92 Hz against the derived -54.1445) and ion 1 0.74
sigma, with opposite signs, so it is scatter and not a bias."""


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_every_calibrated_entry_agrees_with_the_derived_truth_within_its_uncertainty(calibrated) -> None:
    """Every calibrated entry agrees with the device's derived value within SIGMA of its reported uncertainty (SIGMA_STARK for
    the light shifts), the waveform closes at chi = pi/4 to 1e-9 and the heating entries stay zero seeds on the quiet device."""
    fx, report = calibrated
    t = report.table
    dev = fx.device
    assert not t.surrogate and t.device_hash == dev.hash() and not report.refused, report.refused
    assert t.uncalibrated() == (), t.uncalibrated()
    # the field and the qubit frequencies
    assert (
        t.field.status == "calibrated"
        and abs(t.field.value - dev.field.B_gauss) < SIGMA * t.field.uncertainty
    )
    for i in range(2):
        sp = dev.crystal.species[i]
        f_true = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], dev.field.B_gauss)[0]
        e = t.qubit_freq[i]
        assert e.status == "calibrated" and e.experiment == "ramsey_frequency"
        assert abs(e.value - f_true) < SIGMA * e.uncertainty and e.uncertainty < 20.0
    # the modes, occupations and Lamb-Dicke parameters the drives couple to
    for m in (2, 3):
        e = t.modes[m]
        assert (
            e.status == "calibrated" and abs(e.value - dev.crystal.modes[m].omega_hz) < SIGMA * e.uncertainty
        )
        assert e.uncertainty < 1e3, "the FM solvers' sub-kilohertz need"
        nb = t.nbar[m]
        assert nb.status == "calibrated" and abs(nb.value - report.surrogate.nbar[m]) < SIGMA * max(
            nb.uncertainty, 2e-3
        )
    # carrier Rabi frequencies, Stark shifts, crosstalk against the derived values
    for i in range(2):
        spec = fx.gate_drives[i]
        dd = derive_raman_drive(dev, i, spec.beams, scattering=False)
        key = (i, spec.table_key_beam)
        r = t.rabi[key]
        assert r.status == "calibrated" and abs(r.value - dd.carrier_rabi_hz) < SIGMA * r.uncertainty
        assert r.uncertainty < 2e-3 * dd.carrier_rabi_hz
        s = t.stark[key]
        assert (
            s.status == "calibrated"
            and abs(s.value - dd.stark_shift_hz) < SIGMA_STARK * s.uncertainty
            and s.uncertainty < 10.0
        )
        j = 1 - i
        x = t.crosstalk[(i, j)]
        assert (
            x.status == "calibrated"
            and abs(x.value - abs(crosstalk_ratios(dev, i, spec.beams)[j])) < SIGMA * x.uncertainty
        )
        assert (i, j) in t.crosstalk_phase and abs(
            t.crosstalk_phase[(i, j)].value
        ) < SIGMA * t.crosstalk_phase[(i, j)].uncertainty
    # the Lamb-Dicke parameters: one entry per coupled mode, on the mode's probe ion, all calibrated by the sideband Rabi
    # frequency
    assert {m for (_ion, m) in t.lamb_dicke} == {2, 3}
    assert all(
        e.status == "calibrated" and e.experiment == "sideband_spectroscopy" for e in t.lamb_dicke.values()
    )
    # the entangling waveform: calibrated by the amplitude scan, aligned by the phase scan
    wf = t.waveform_for((0, 1))
    assert wf is not None and wf.phi_s.status == "calibrated" and wf.phi_s.experiment == "ms_phase_scan"
    assert abs(abs(wf.chi_total_rad) - math.pi / 4.0) < 1e-9
    ms = report.results["ms_scan[(0, 1)]"]
    # the waveform the scan starts from is re-solved at the CALIBRATED mode frequencies, a few hundred hertz from the
    # crystal's, so the scan's correction is 1.02649 +- 0.0074; the +-30 % amplitude span bounds it
    assert ms.converged and abs(ms.fitted["closure_scale"][0] - 1.0) < 0.05
    par = report.results["parity_scan[(0, 1)]"]
    assert par.fitted["contrast"][0] > 0.97 and par.fitted["bell_fidelity_bound"][0] > 0.97
    ph = report.results["ms_phase_scan[(0, 1)]"]
    for q in (0, 1):
        assert abs(ph.fitted[f"correction_rad[{q}]"][0]) < 4.0 * ph.fitted[f"correction_rad[{q}]"][1] + 0.05
    # detection and heating
    for key in ("threshold", "window_s", "eps_B", "eps_D"):
        assert t.detection[key].status == "calibrated" and t.detection[key].sample_id == 0
    for m, e in t.heating.items():
        # a quiet device: the delay scan spans 10/ndot and does not exist at ndot = 0, so EVERY heating entry stays the
        # surrogate's derived seed (the heating experiment itself runs on a device with a non-zero S_E, below)
        assert e.value == 0.0 and e.status == "seed" and e.experiment == "derived_heating_rate", (m, e)
    assert any("the derived rate is zero" in n for n in report.notes)
    # the surrogate's error per entry
    err = report.surrogate_error()
    assert all(v < 0.02 for k, v in err.items() if k.startswith(("rabi", "modes", "qubit_freq")))
    assert all(e.fitted_at_s == 0.0 and e.sample_id == 0 for e in t.entries().values())


@pytest.mark.slow
def test_a_bell_circuit_from_the_calibrated_table_reaches_the_predicted_fidelity(calibrated) -> None:
    """A Bell circuit from the calibrated table has 1 - F inside the intrinsic budget plus the fits' own over-rotation and frame
    terms, above the surrogate table's and within 5e-3 of it, with P_00 + P_11 > 0.98."""
    fx, report = calibrated
    kw = dict(
        keep_final_state=True,
        numerics=Numerics(branch_weight_min=1e-3),
    )
    res = run(BELL, fx.device, 1000, table=report.table, **kw)
    ref = run(BELL, fx.device, 1000, table=report.surrogate.table, **kw)
    fid, fid_ref = register_fidelity(res), register_fidelity(ref)
    budget = res.diagnostics.intrinsic_budget["total"]
    # the calibration's contribution: the over-rotation (sigma_Omega/Omega x pi/2)^2 per carrier pulse and the residual frame offset
    t = report.table
    cal = 0.0
    for i in range(2):
        r = t.rabi[(i, fx.gate_drives[i].table_key_beam)]
        cal += 5 * (0.5 * math.pi * r.uncertainty / r.value) ** 2
        cal += (2.0 * math.pi * t.qubit_freq[i].uncertainty * 200e-6) ** 2
    # the upper bound, at the calibration's own contribution with no slack factor: 1 - F must lie inside the intrinsic
    # budget the noise model predicts plus the fits' own uncertainty
    assert 1.0 - fid < budget + cal, (fid, budget, cal)
    # and the LOWER bound, so that the assertion also fails if the calibration errors silently disappear (a table played as
    # the physics, the played chain switched off, or the frame taken at the true transition): the fitted table is strictly
    # worse than the surrogate one, whose seeds ARE the derived values
    assert 1.0 - fid > 1.0 - fid_ref, (fid, fid_ref)
    assert abs(fid - fid_ref) < 5e-3, (fid, fid_ref)
    p = res.probabilities
    assert p["00"] + p["11"] > 0.98 and abs(p["00"] - p["11"]) < 5.0 * res.error_bars["00"]
    assert res.diagnostics.calibration is report.table and not res.diagnostics.calibration.surrogate
    assert any("dynamical samples" in a or "nominal sample" in a for a in res.diagnostics.approximations)


@pytest.mark.slow
def test_calibrate_entry_point_caches_the_full_table_and_a_stale_table_still_runs(calibrated) -> None:
    fx, report = calibrated
    cache = CalibrationCache()
    kw = dict(
        method="experiments",
        seed=11,
        pairs=[(0, 1)],
        scans=CalibrationScans(
            shots=None, detection_records=300, detection_windows_s=(20e-6,), micromotion_ranges={}
        ),
        experiments=("field_scan", "detection_histogram"),
        detection_records=300,
        detection_windows_s=(20e-6,),
        cache=cache,
    )
    t1 = calibrate(Machine(fx.device), **kw).table
    t2 = calibrate(Machine(fx.device), **kw).table
    assert t1 is t2 and len(cache.reports) == 1
    assert t1.field.experiment == "field_scan" and t1.field.status == "calibrated"
    assert all(e.status == "seed" for e in t1.rabi.values()), (
        "a subset run keeps the other entries as surrogate seeds"
    )
    assert usable(t1.detection["threshold"]) and t1.detection["threshold"].experiment == "detection_histogram"
    # a table is a snapshot with an age: run() at a later time takes it as a legitimate, possibly stale, input
    res = run(
        BELL,
        fx.device,
        50,
        table=report.table,
        physics=Physics(t0_s=3600.0),
        numerics=Numerics(branch_weight_min=1e-2),
    )
    assert res.diagnostics.calibration.fitted_at_s == 0.0 and res.shots == 50


@pytest.fixture(scope="module")
def two_ion():
    fx = yb171_chain(2)
    sur = two_ion_surrogate(1000)
    return fx, sur


# ---- the Stark scan's fringe branch ----------------------------------------------------------------------------------------


def _exact_fringe(shift_hz: float):
    """A ``ramsey`` replacement returning the EXACT fringe frequency |detuning - shift| of a qubit light-shifted by
    ``shift_hz``, which is what the fit sees (its sign is unobservable in a cosine)."""

    def fake(device, ion, delays_s, **kw):
        detuning = float(kw["detuning_hz"])
        delays = np.array(sorted(float(x) for x in delays_s))
        f = abs(detuning - shift_hz)
        data = np.column_stack([delays, 0.5 * np.cos(TWO_PI * f * delays) + 0.5])
        return ExperimentResult(
            data=data,
            fitted={
                "contrast": (0.5, 0.0),
                "delta_hz": (f, 0.0),
                "phi0_rad": (0.0, 0.0),
                "offset": (0.5, 0.0),
            },
            model="exact_fringe",
            provenance_id="conv.detuning_symbols",
            converged=True,
        )

    return fake


@pytest.mark.parametrize(
    ("shift_hz", "ok"),
    [(-38.4, True), (300.0, True), (-900.0, True), (1500.0, False), (2500.0, False)],
)
def test_stark_scan_resolves_the_fringe_branch_with_two_probe_signs(
    two_ion, monkeypatch, shift_hz, ok
) -> None:
    """Two probe signs recover the light shift delta = (f_minus - f_plus)/2 to 1e-9 while |delta| < probe, and a shift beyond the
    probe saturates at it and is reported."""
    fx, _sur = two_ion
    monkeypatch.setattr("qutip_trap.experiments.light.ramsey", _exact_fringe(shift_hz))
    probe = 1e3
    res = stark_scan(Machine(fx.device), 0, np.linspace(0.0, 2e-3, 9), probe_hz=probe, shots=None)
    beams = fx.gate_drives[0].beams
    assert res.converged is ok, res.notes
    if ok:
        for b in beams:
            assert res.fitted[f"stark_shift_hz[{b}]"][0] == pytest.approx(shift_hz, abs=1e-9)
            assert res.fitted[f"fringe_plus_hz[{b}]"][0] == pytest.approx(abs(probe - shift_hz), abs=1e-9)
            assert res.fitted[f"fringe_minus_hz[{b}]"][0] == pytest.approx(abs(probe + shift_hz), abs=1e-9)
        assert res.fitted["stark_shift_hz"][0] == pytest.approx(len(beams) * shift_hz, abs=1e-9)
    else:
        # the estimator saturates at the probe and the two fringes no longer sum to 2 probe: BOTH guards fire
        for b in beams:
            assert abs(res.fitted[f"stark_shift_hz[{b}]"][0]) == pytest.approx(probe, abs=1e-9)
        assert any("outside the probe" in n for n in res.notes), res.notes


def test_stark_scan_refuses_a_fringe_above_the_delay_grids_nyquist_frequency(two_ion, monkeypatch) -> None:
    """Five delays over 2 ms cannot resolve the probe + |delta| fringe of a 1 kHz probe and are refused as above Nyquist, nine
    are accepted, and both return the same shift."""
    fx, _sur = two_ion
    monkeypatch.setattr("qutip_trap.experiments.light.ramsey", _exact_fringe(-38.4))
    kw = dict(probe_hz=1e3, shots=None)
    coarse = stark_scan(Machine(fx.device), 0, np.linspace(0.0, 2e-3, 5), **kw)
    fine = stark_scan(Machine(fx.device), 0, np.linspace(0.0, 2e-3, 9), **kw)
    assert not coarse.converged and any("Nyquist" in n for n in coarse.notes), coarse.notes
    assert fine.converged, fine.notes
    # the VALUE is the same either way: the guard is about what the grid can resolve, not about the estimator
    for b in fx.gate_drives[0].beams:
        assert coarse.fitted[f"stark_shift_hz[{b}]"][0] == pytest.approx(-38.4, abs=1e-9)


def test_stark_scan_does_not_leak_its_mode_switch_into_the_ramsey_setup(two_ion, monkeypatch) -> None:
    """``stark_scan(mode="beat_note")`` runs without forwarding its ``mode`` to the Ramsey setup (which reads ``mode`` as a mode
    index), and an unknown mode is refused."""
    fx, _sur = two_ion
    monkeypatch.setattr("qutip_trap.experiments.light.ramsey", _exact_fringe(-38.4))
    res = stark_scan(
        Machine(fx.device), 0, np.linspace(0.0, 2e-3, 9), probe_hz=1e3, shots=None, mode="beat_note"
    )
    assert "stark_shift_hz" in res.fitted and "coupling_shift_hz" in res.fitted
    with pytest.raises(ValueError, match="per_beam"):
        stark_scan(Machine(fx.device), 0, np.linspace(0.0, 2e-3, 9), mode="nonsense")


# ---- the refusal cascade ---------------------------------------------------------------------------------------------------


def test_a_refused_experiment_marks_its_own_entries_uncalibrated_and_the_schedule_then_refuses(
    two_ion,
) -> None:
    """An uncalibrated field refuses the micromotion, mode, Rabi, Stark and Ramsey fits, each marks the entry groups it would
    have written uncalibrated, and ``schedule()`` then refuses the table."""
    fx, sur = two_ion
    bad_field = dataclasses.replace(
        sur.table,
        field=dataclasses.replace(sur.table.field, status="uncalibrated"),
    )
    report = full_calibration(
        fx.device,
        experiments=(
            "micromotion_scan",
            "sideband_spectroscopy",
            "rabi_scan",
            "stark_scan",
            "ramsey_frequency",
        ),
        surrogate=dataclasses.replace(sur, table=bad_field),
    )
    t = report.table
    for name in ("micromotion_scan", "sideband_spectroscopy", "rabi_scan", "stark_scan", "ramsey_frequency"):
        assert name in report.refused, (name, report.refused)
    assert report.results == {}, "no experiment ran"
    for group in ("modes", "nbar", "rabi", "stark", "qubit_freq"):
        entries = getattr(t, group)
        assert entries, group
        assert all(e.status == "uncalibrated" for e in entries.values()), (group, entries)
    # each refusal names the first upstream group it could not read, the field for all five here (this fixture has no rf
    # record and the test asks for no shim ranges, so the micromotion scan writes no entry of its own to be marked)
    assert t.micromotion == {}
    assert all("field" in reason for reason in report.refused.values()), report.refused
    with pytest.raises(ScheduleError):
        schedule(BELL, fx.device, t)


def test_a_subset_calibration_leaves_the_other_entries_as_seeds_not_uncalibrated(two_ion) -> None:
    """A group that is a seed because the caller did not ASK for its experiment stays a seed; only the refusal path marks."""
    fx, sur = two_ion
    report = full_calibration(
        fx.device, experiments=("field_scan",), surrogate=sur, scans=CalibrationScans(shots=None)
    )
    assert not report.refused, report.refused
    assert report.table.field.status == "calibrated"
    assert all(e.status == "seed" for e in report.table.rabi.values())
    assert report.table.uncalibrated() == ()


# ---- an uncalibrated qubit frequency -------------------------------------------------------------------------------------


def test_run_refuses_an_uncalibrated_qubit_frequency(two_ion) -> None:
    """A run on a table whose qubit frequency is uncalibrated is refused rather than put the frame on the true transition."""
    fx, sur = two_ion
    bad = dataclasses.replace(
        sur.table,
        qubit_freq={
            **sur.table.qubit_freq,
            0: dataclasses.replace(sur.table.qubit_freq[0], status="uncalibrated"),
        },
    )
    with pytest.raises(RunError, match="qubit frequency is uncalibrated"):
        run(GPI, fx.device, 10, table=bad)


# ---- the micromotion loop -------------------------------------------------------------------------------------------------


def _rf_two_ion(stray_x_v_per_m: float):
    """The two-ion fixture with an rf record and a stray field along x (excess micromotion to compensate)."""
    fx = yb171_chain(2)
    dev = dataclasses.replace(
        fx.device,
        trap=dataclasses.replace(
            fx.device.trap,
            rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
            stray_field_v_per_m=(float(stray_x_v_per_m), 0.0, 0.0),
        ),
    )
    return fx, dataclasses.replace(dev, crystal=solve_crystal(dev.trap, dev.crystal.species))


def test_the_calibrated_shims_are_programmed_onto_the_device_the_run_evolves(two_ion) -> None:
    """A calibrated shim reaches the device the run evolves (the compensated residual beta < 1e-9), a seed one does not and an
    uncalibrated one is reported."""
    fx, dev = _rf_two_ion(20.0)
    dk = np.asarray(derive_raman_drive(dev, 0, fx.gate_drives[0].beams, scattering=False).delta_k)
    beta_uncompensated = signed_beta(dev, 0, dk)
    assert abs(beta_uncompensated) > 1e-3, "the stray field gives the fixture excess micromotion to null"
    # the compensating field is minus the stray one: with it programmed the residual index vanishes
    assert abs(signed_beta(device_with_compensation(dev, {"Ex": -20.0}), 0, dk)) < 1e-9

    def table_with(status: str, value: float):
        sur = surrogate_table(
            dev, pairs=[(0, 1)], detection_records=200, detection_windows_s=(20e-6,), spot_check=False
        )
        return dataclasses.replace(
            sur.table,
            micromotion={
                "shim[Ex]": CalEntry(
                    value, 0.1, status, "micromotion_scan", "anchor.trap.berkeland_excess_micromotion", 0.0, 0
                )
            },
        )

    kw = dict(
        numerics=Numerics(branch_weight_min=1e-2),
    )
    good = run(GPI, dev, 5, table=table_with("calibrated", -20.0), **kw)
    assert any("micromotion compensation applied" in n for n in last_record(good).notes)
    seeded = run(GPI, dev, 5, table=table_with("seed", -20.0), **kw)
    assert not any("micromotion compensation applied" in n for n in last_record(seeded).notes)
    refused = run(GPI, dev, 5, table=table_with("uncalibrated", -20.0), **kw)
    assert any("micromotion compensation uncalibrated" in n for n in last_record(refused).notes)


@pytest.mark.slow
def test_a_stale_micromotion_calibration_against_a_drifted_stray_field_leaves_a_growing_residual() -> None:
    """Shims nulled against a 20 V/m stray field leave, at 35 V/m, the residual beta of the 15 V/m drift (to 5 %), more than ten
    times the fresh residual, and a carrier loss that grows with it."""
    fx, at_t0 = _rf_two_ion(20.0)
    # an hour of stray-field drift: the trap the run evolves is not the one the shims were nulled on
    drifted = dataclasses.replace(
        at_t0, trap=dataclasses.replace(at_t0.trap, stray_field_v_per_m=(35.0, 0.0, 0.0))
    )
    drifted = dataclasses.replace(drifted, crystal=solve_crystal(drifted.trap, drifted.crystal.species))
    beam = 0
    scan = micromotion_scan(Machine(at_t0), 0, beam, {"Ex": (-40.0, 0.0)}, method="sideband_ratio", points=5)
    assert scan.converged, scan.notes
    shim = float(scan.fitted["shim[Ex]"][0])
    assert shim == pytest.approx(-20.0, abs=1.0), "the scan nulls the field it was calibrated against"
    dk = np.asarray(derive_raman_drive(at_t0, 0, fx.gate_drives[0].beams, scattering=False).delta_k)
    residual_fresh = abs(signed_beta(device_with_compensation(at_t0, {"Ex": shim}), 0, dk))
    residual_stale = abs(signed_beta(device_with_compensation(drifted, {"Ex": shim}), 0, dk))
    uncompensated = abs(signed_beta(drifted, 0, dk))
    assert residual_fresh < 1e-2 * uncompensated, (residual_fresh, uncompensated)
    assert residual_stale > 10.0 * residual_fresh, (residual_stale, residual_fresh)
    # the residual index the stale table leaves is the 15 V/m of drift, not the 35 V/m of stray field: compensation still
    # helps, it just no longer nulls
    assert residual_stale == pytest.approx(uncompensated * 15.0 / 35.0, rel=0.05)
    # and the gate error grows with it: the carrier is suppressed by J_0(beta) and the first sideband carries J_1(beta)
    assert 1.0 - float(jv(0, residual_stale)) > 10.0 * (1.0 - float(jv(0, max(residual_fresh, 1e-12))))


# ---- the entangling scans' frame and mode beliefs --------------------------------------------------------------------------


def test_the_entangling_setup_refuses_to_swallow_the_mode_frequencies_it_would_discard(two_ion) -> None:
    """Supplying both ``modes`` and ``mode_frequencies_hz`` is refused, and the table's mode frequencies reach the GateModes the
    entangling scans build (to 1e-12)."""
    fx, sur = two_ion
    beams = fx.entangling_drives[0].beams
    modes = gate_modes(fx.device, (0, 1), (beams[0], beams[1]), nbar={})
    with pytest.raises(ValueError, match="mode_frequencies_hz"):
        ms_scan(
            Machine(fx.device),
            (0, 1),
            [1.0],
            [0.0],
            table=sur.table,
            modes=modes,
            mode_frequencies_hz={2: 2.8e6},
        )
    # and the frequencies the table believes reach the GateModes the scans build
    kw = dict(
        table=sur.table,
        mode_frequencies_hz={m: e.value + 1234.0 for m, e in sur.table.modes.items()},
    )
    _wf, _ent, _sq, _t, built, _space = _entangling_setup(fx.device, (0, 1), dict(kw))
    for m, w in zip(built.modes, built.omega_rad_s):
        assert w / TWO_PI == pytest.approx(fx.device.crystal.modes[m].omega_hz + 1234.0, rel=1e-12)


@pytest.mark.slow
def test_a_wrong_qubit_frequency_shifts_the_ms_phase_scans_correction_by_the_frame_phase(two_ion) -> None:
    """A 1 kHz qubit-frequency error on ion 0 moves the phase scan's ion-0 correction by 0.4 to 0.8 of 2 pi delta_f t_gate
    (twice the error, twice the shift to 5 %) and the ion-1 correction by less than 5 % of that."""
    fx, sur = two_ion
    wf = sur.table.waveform_for((0, 1))
    assert wf is not None
    phases = [float(x) for x in np.linspace(0.0, math.pi, 4, endpoint=False)]
    kw = dict(
        table=sur.table,
        nbar={m: e.value for m, e in sur.table.nbar.items()},
        inputs=("00", "01"),
        shots=None,
    )
    df = 1e3
    ref = ms_phase_scan(Machine(fx.device), (0, 1), phases, **kw)
    one = ms_phase_scan(Machine(fx.device), (0, 1), phases, qubit_shifts_hz={0: df}, **kw)
    two = ms_phase_scan(Machine(fx.device), (0, 1), phases, qubit_shifts_hz={0: 2.0 * df}, **kw)
    assert ref.converged and one.converged and two.converged, (ref.notes, one.notes, two.notes)

    def delta(res, q: int) -> float:
        return float(res.fitted[f"correction_rad[{q}]"][0] - ref.fitted[f"correction_rad[{q}]"][0])

    d0, d0_two, d1 = delta(one, 0), delta(two, 0), delta(one, 1)
    scale = TWO_PI * df * float(wf.duration_s)
    assert 0.4 * scale < abs(d0) < 0.8 * scale, (d0, scale)
    assert d0_two / d0 == pytest.approx(2.0, rel=0.05), (d0, d0_two)
    assert abs(d1) < 0.05 * abs(d0), (d0, d1)


# ---- every accepted experiment name runs something ------------------------------------------------------------------------


@pytest.mark.slow
def test_every_advertised_experiment_name_produces_a_result_a_refusal_or_a_reason(two_ion) -> None:
    """Every accepted name (``ORDER`` and ``ALIASES``) leaves a result, a refusal with its upstream reason, or a note saying why
    the scan does not exist on the device."""
    fx, sur = two_ion
    # a NAME must leave a trace, so the entries that gate the expensive scans are uncalibrated and the other scans minimal: the
    # field refuses the micromotion, mode, Rabi, Stark and Ramsey fits and the light shift the entangling ones, leaving the
    # crosstalk scan, crystal_image, field_scan, detection_histogram and heating_rate (the note: no scan at ndot = 0)
    blocked = dataclasses.replace(
        sur.table,
        field=dataclasses.replace(sur.table.field, status="uncalibrated"),
        stark={k: dataclasses.replace(e, status="uncalibrated") for k, e in sur.table.stark.items()},
    )
    scans = CalibrationScans(
        shots=None,
        ramsey_delays_s=(0.0, 5e-4, 1e-3, 1.5e-3),
        crosstalk_points=4,
        crosstalk_phase_points=3,
        detection_records=100,
        detection_windows_s=(20e-6,),
        micromotion_ranges={},
    )
    for name in ORDER + tuple(ALIASES):
        resolved = ALIASES.get(name, name)
        report = full_calibration(
            fx.device, experiments=(name,), surrogate=dataclasses.replace(sur, table=blocked), scans=scans
        )
        assert report.experiments == (resolved,), (name, report.experiments)
        ran = any(k == resolved or k.startswith(f"{resolved}[") for k in report.results)
        said = any(resolved in n for n in report.notes)
        assert ran or resolved in report.refused or said, (
            name,
            report.results.keys(),
            report.refused,
            report.notes,
        )


def test_a_device_with_no_entangling_drive_calibrates_and_runs(two_ion) -> None:
    """The 40Ca+ preset, with no entangling drive, gets no entangling seed or waveform and still runs, and a chain whose
    entangling drive covers one ion seeds only that one."""
    preset = ca40_optical()
    assert preset.entangling_drives == {}, "the 40Ca+ optical preset carries no entangling pair"
    sur = surrogate_table(preset.device, detection_records=200, detection_windows_s=(20e-6,))
    assert sur.table.ms == {}, "no entangling drive, no waveform"
    assert sur.table.rabi, "the single-qubit drives are still seeded"
    circuit = Circuit(1, (Operation("gpi2", (0,), (0.0,)),), (0,))
    res = run(circuit, preset.device, 20, table=sur.table)
    assert res.shots == 20 and sum(res.probabilities.values()) == pytest.approx(1.0)
    # and a partially addressed chain seeds only the ions its entangling drive names
    fx, _sur2 = two_ion
    partial = surrogate_table(
        dataclasses.replace(
            fx.device, roles=dataclasses.replace(fx.device.roles, entangling={0: fx.entangling_drives[0]})
        ),
        pairs=[],
        detection_records=200,
        detection_windows_s=(20e-6,),
    )
    assert partial.table.ms == {}
    ent_key = (1, fx.entangling_drives[1].table_key_beam)
    assert (
        ent_key not in partial.table.rabi or partial.table.rabi[ent_key].experiment != "derived_raman_drive"
    )


@pytest.mark.slow
def test_the_heating_experiment_runs_end_to_end_on_a_device_with_electric_field_noise() -> None:
    """With a non-zero S_E the heating scan's seed carries the white level and every calibrated heating entry equals the noise
    model's rate to 3 %."""
    fx = yb171_chain(2)
    noisy = dataclasses.replace(
        fx.device,
        noise=dataclasses.replace(
            fx.device.noise, S_E=white_spectrum(2e-12, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    truth = noisy.noise.heating_rates_quanta_per_s(noisy)
    report = full_calibration(
        noisy,
        seed=11,
        experiments=("heating_rate",),
        pairs=[(0, 1)],
        scans=CalibrationScans(
            shots=None, heating_delays=4, heating_span_over_ndot=1.0, micromotion_ranges={}
        ),
        detection_records=200,
        detection_windows_s=(20e-6,),
    )
    assert not report.refused, report.refused
    # the seed the scan is built from now carries the white level: non-zero on every mode the model heats
    assert all(report.surrogate.table.heating[m].value > 1.0 for m in truth), report.surrogate.table.heating
    measured = [m for m in report.table.heating if report.table.heating[m].status == "calibrated"]
    assert measured, report.notes
    for m in measured:
        e = report.table.heating[m]
        assert e.experiment == "heating_rate" and e.uncertainty > 0.0
        assert e.value == pytest.approx(truth[m], rel=0.03), (m, e, truth[m])


# ---- the run-level over-rotation -------------------------------------------------------------------------------------------


def test_a_five_percent_rabi_error_in_the_table_over_rotates_the_run(two_ion) -> None:
    """A table whose Rabi entry is 5 % high shortens the GPi by 1/1.05 (to 1e-12), and the run's loss lies within 0.8 to 2
    times cos^2(pi/2/1.05) = 5.58e-3."""
    fx, sur = two_ion
    key = (0, fx.gate_drives[0].table_key_beam)
    high = dataclasses.replace(
        sur.table,
        rabi={
            **sur.table.rabi,
            key: dataclasses.replace(sur.table.rabi[key], value=1.05 * sur.table.rabi[key].value),
        },
    )
    # the schedule: the pi time shortens by exactly 1/1.05
    ref_sched = schedule(GPI, fx.device, sur.table)
    high_sched = schedule(GPI, fx.device, high)
    assert high_sched.pulses[0].duration_s / ref_sched.pulses[0].duration_s == pytest.approx(
        1.0 / 1.05, rel=1e-12
    )

    # the physics: the excited population of ion 0 after the GPi
    def loss(table) -> float:
        res = run(
            GPI,
            fx.device,
            5,
            table=table,
            keep_final_state=True,
            numerics=Numerics(branch_weight_min=1e-3),
        )
        rho = last_record(res).register_state
        assert rho is not None
        proj = qt.tensor(qt.basis(2, 1).proj(), qt.qeye(2))
        return 1.0 - float(np.real(qt.expect(proj, rho)))

    analytic = math.cos(math.pi / 2.0 / 1.05) ** 2
    assert analytic == pytest.approx(5.58e-3, rel=2e-2), "1 - sin^2(pi/2/1.05)"
    baseline, over = loss(sur.table), loss(high)
    assert baseline < 1e-3, baseline
    assert over > 0.8 * analytic, (over, analytic, baseline)
    assert over < 2.0 * analytic, (over, analytic, baseline)
