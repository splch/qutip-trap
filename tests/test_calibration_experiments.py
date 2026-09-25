"""The simulated laboratory (PLAN.md Section 7.5) against the device's derived values: spectroscopy, thermometry, mode and
heating scans, the field, Stark and crosstalk scans, micromotion compensation, and the laboratory keywords and typed results."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from qutip_trap.calibration import calibrate
from qutip_trap.calibration.experiments import CalibrationReport
from qutip_trap.device.presets import ideal_hardware, yb171_chain
from qutip_trap.experiments.fitting import thermal_rabi_model
from qutip_trap.experiments.imaging import crystal_image
from qutip_trap.experiments.light import crosstalk_scan, field_scan, stark_scan
from qutip_trap.experiments.micromotion import (
    _Correlation,
    correlation_signal,
    device_with_compensation,
    micromotion_scan,
    periodic_scattering,
    signed_beta,
)
from qutip_trap.experiments.motion import heating_rate, mode_spectroscopy, thermometry
from qutip_trap.experiments.readout import detection_histogram
from qutip_trap.experiments.result import (
    CrystalImage,
    DetectionHistogram,
    RabiScan,
    RamseyFringe,
    ScanParameters,
    SidebandSpectrum,
    ThermometryResult,
)
from qutip_trap.experiments.single_ion import _Lab, rabi_scan, ramsey, ramsey_frequency, sideband_spectroscopy
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive, differential_stark_shift_hz
from qutip_trap.light.roles import detection_beams
from qutip_trap.machine import Machine
from qutip_trap.noise.spectra import white_spectrum
from qutip_trap.options import Numerics
from qutip_trap.readout.fluorescence import detection_rates_for_ion
from qutip_trap.trap.crystal import solve_crystal
from qutip_trap.trap.mathieu import c0_wronskian, mathieu_from_secular
from qutip_trap.trap.pseudopotential import RfDrive
from tests.fixtures import FAST, WINDOWS, microwave_device, single_ion_raman_device

DURATIONS = np.linspace(0.0, 20e-6, 9)
DETUNINGS = np.linspace(-3.2e6, 3.2e6, 9)


@pytest.fixture(scope="module")
def single():
    dev = single_ion_raman_device()
    return dev, derive_raman_drive(dev, 0, (0, 1), scattering=False)


@pytest.fixture(scope="module")
def two_ion():
    fx = yb171_chain(2)
    return fx, derive_raman_drive(fx.device, 0, fx.gate_drives[0].beams, scattering=False)


@pytest.fixture(scope="module")
def machine(two_ion) -> Machine:
    fx, _dd = two_ion
    return Machine(fx.device, numerics=FAST).calibrated(
        pairs=[(0, 1)], detection_records=300, detection_windows_s=WINDOWS
    )


# ---- Rabi, Ramsey and sideband spectroscopy (Sections 4.2.7, 7.9) -------------------------------------------------------------


def test_rabi_scan_fits_the_rabi_frequency_and_the_thermal_occupation(single) -> None:
    """The thermal Debye-Waller fit of Rabi flopping returns the carrier Rabi frequency to 1e-3, nbar = 0.6 to 0.05 and unit
    contrast, and its model matches the data to 5e-3."""
    dev, dd = single
    f = dd.carrier_rabi_hz
    ts = np.linspace(0.0, 1.2 / f, 13)
    res = rabi_scan(Machine(dev), 0, ts, nbar={1: 0.6}, include_stark=False)
    assert res.data.shape == (13, 2) and res.model == "thermal_debye_waller_rabi"
    assert res.fitted["f_rabi_hz"][0] == pytest.approx(f, rel=1e-3)
    assert res.fitted["nbar"][0] == pytest.approx(0.6, abs=0.05)
    assert res.fitted["contrast"][0] == pytest.approx(1.0, abs=1e-3)
    model = thermal_rabi_model(np.array([f, 0.6, 1.0]), ts, dd.etas[1])
    assert np.max(np.abs(model - res.data[:, 1])) < 5e-3


def test_ramsey_fringe_and_ramsey_frequency_recover_the_detuning(single) -> None:
    dev, _dd = single
    delays = np.linspace(0.0, 2e-3, 9)
    res = ramsey(Machine(dev), 0, delays, detuning_hz=1000.0, include_stark=False)
    assert res.fitted["delta_hz"][0] == pytest.approx(1000.0, abs=1e-3)
    assert res.fitted["contrast"][0] == pytest.approx(0.5, abs=1e-3)
    freq = ramsey_frequency(
        Machine(dev), 0, delays, include_stark=False, probe_hz=1000.0, qubit_shifts_hz={0: 37.0}
    )
    assert freq.fitted["qubit_offset_hz"][0] == pytest.approx(37.0, abs=0.05)
    assert freq.model == "ramsey_two_probe"


def test_microwave_ramsey_frequency() -> None:
    dev = microwave_device()
    freq = ramsey_frequency(
        Machine(dev),
        0,
        np.linspace(0.0, 2e-3, 9),
        rabi_hz=2e4,
        probe_hz=1000.0,
        qubit_shifts_hz={0: -12.5},
        frame_hz=1.0e9,
    )
    assert freq.fitted["qubit_offset_hz"][0] == pytest.approx(-12.5, abs=0.05)
    assert freq.fitted["qubit_freq_hz"][0] == pytest.approx(1.0e9 - 12.5, abs=0.05)
    with pytest.raises(ValueError, match="rabi_hz"):
        rabi_scan(Machine(dev), 0, [0.0, 1e-6, 2e-6, 3e-6])


def test_sideband_spectroscopy_finds_the_blue_sideband_and_the_dark_red_one(single) -> None:
    """From n = 0 the blue sideband, found at 3.0 MHz to 10 kHz, flops above 0.9 while the red one stays below 1e-3."""
    dev, dd = single
    f, eta = dd.carrier_rabi_hz, dd.etas[1]
    mus = np.concatenate(
        [np.linspace(-3.02e6, -2.98e6, 5), np.linspace(-1e4, 1e4, 3), np.linspace(2.98e6, 3.02e6, 5)]
    )
    res = sideband_spectroscopy(Machine(dev), 0, mus, duration_s=0.5 / (f * eta), include_stark=False)
    assert res.fitted["blue_sideband_hz"][0] == pytest.approx(3.0e6, abs=1e4)
    assert res.fitted["carrier_hz"][0] == pytest.approx(0.0, abs=1e4)
    blue = res.data[np.argmin(np.abs(res.data[:, 0] - 3.0e6)), 1]
    red = res.data[np.argmin(np.abs(res.data[:, 0] + 3.0e6)), 1]
    assert blue > 0.9 and red < 1e-3


def test_carrier_lineshape_on_the_exact_dynamics_gives_the_derived_rabi_frequency_and_pi_time(single) -> None:
    """The fitted Omega of a noiseless carrier scan is the derived Omega times the n = 0 Debye-Waller factor e^{-eta^2/2} to
    2e-3 (and within five fit sigmas), and the pi time is pi/Omega."""
    dev, dd = single
    f, eta = dd.carrier_rabi_hz, dd.etas[1]
    t_pi = 0.5 / f
    res = sideband_spectroscopy(
        Machine(dev), 0, np.linspace(-2.2 * f, 2.2 * f, 23), duration_s=t_pi, include_stark=False, fit=True
    )
    expected = f * math.exp(-0.5 * eta**2)
    omega_fit, s_omega = res.fitted["omega_carrier_hz"]
    assert omega_fit == pytest.approx(expected, rel=2e-3)
    assert abs(omega_fit - expected) < 5.0 * max(s_omega, 1e-3 * f)
    assert res.fitted["carrier_fit_hz"][0] == pytest.approx(0.0, abs=50.0)
    assert math.pi / (2.0 * math.pi * omega_fit) == pytest.approx(t_pi / math.exp(-0.5 * eta**2), rel=2e-3)


def test_rabi_scan_with_thermometry_nbar_fits_the_bare_rabi_frequency_through_every_modes_debye_waller_factor(
    two_ion,
) -> None:
    """With nbar fixed from thermometry the fit carries both x modes' Debye-Waller factors and returns the bare derived Omega
    within four sigmas (sigma < 1e-3 Omega) and unit contrast to 0.02."""
    fx, dd = two_ion
    f = dd.carrier_rabi_hz
    res = rabi_scan(
        Machine(fx.device),
        0,
        np.linspace(0.0, 10.0 * 0.5 / f, 41),
        nbar={2: 0.0185, 3: 0.0154},
        nbar_fixed=0.0185,
        shots=500,
        readout=True,
        seed=8,
    )
    fit, s = res.fitted["f_rabi_hz"]
    assert res.converged and abs(fit - f) < 4.0 * s and s < 1e-3 * f, (fit, s, f)
    assert res.fitted["contrast"][0] == pytest.approx(1.0, abs=0.02)


# ---- the laboratory on a machine: keywords, typed results, requested against realized ----------------------------------------


def test_the_machine_supplies_the_laboratory_defaults_and_an_unknown_keyword_raises(machine: Machine) -> None:
    lab = _Lab.of(machine, {"shots": 5})
    assert lab.device is machine.device and lab.obs.shots == 5 and lab.table is machine.table
    assert lab.options == machine.numerics and lab.builder is None
    given = _Lab.of(machine, {"table": None, "options": Numerics(atol=1e-12)})
    assert (
        given.table is None and given.options is not None and given.options.atol == 1e-12
    )  # the call's keywords win
    assert _Lab.of(machine, {"options": None}).options is None  # None: the experiment's own defaults
    with pytest.raises(TypeError, match="shot"):
        rabi_scan(machine, 0, DURATIONS, shot=5)
    with pytest.raises(TypeError, match="numerics"):
        stark_scan(machine, 0, np.linspace(0.0, 2e-3, 9), numerics=FAST)


EXPERIMENTS: dict[str, tuple[Any, type, str]] = {
    "rabi_scan": (lambda m: rabi_scan(m, 0, DURATIONS, shots=200, seed=1), RabiScan, "rabi_scan"),
    "ramsey": (
        lambda m: ramsey(m, 0, np.linspace(0.0, 40e-6, 6), shots=200, seed=1, detuning_hz=2e4),
        RamseyFringe,
        "ramsey",
    ),
    "ramsey_frequency": (
        lambda m: ramsey_frequency(m, 0, np.linspace(0.0, 2e-3, 7), probe_hz=1e3),
        RamseyFringe,
        "ramsey_frequency",
    ),
    "sideband_spectroscopy": (
        lambda m: sideband_spectroscopy(m, 0, DETUNINGS, seed=1),
        SidebandSpectrum,
        "sideband_spectroscopy",
    ),
    "thermometry": (lambda m: thermometry(m, 0, 3, shots=200, seed=1), ThermometryResult, "thermometry"),
}


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_an_experiment_returns_its_typed_result_named_for_the_table(name: str, two_ion) -> None:
    run, kind, experiment = EXPERIMENTS[name]
    res = run(Machine(two_ion[0].device))
    assert isinstance(res, kind) and res.experiment == experiment
    assert res.quality in ("good", "poor", "exact", "failed") and res.created_at


def test_realized_equals_requested_on_ideal_hardware_and_moves_when_the_chain_quantises(two_ion) -> None:
    fx = two_ion[0]
    ideal = rabi_scan(Machine(fx.device), 0, DURATIONS, seed=1)
    assert ideal.requested is not None and ideal.realized is not None
    assert ideal.requested.durations_s == ideal.realized.durations_s == tuple(float(t) for t in DURATIONS)
    assert ideal.requested.rabi_hz == ideal.realized.rabi_hz and ideal.requested.phase_rad == (0.0,)
    coarse = dataclasses.replace(
        fx.device,
        hardware=dataclasses.replace(
            ideal_hardware(),
            dds_amplitude_bits=6,
            amplitude_full_scale_hz=1e6,
            dds_clock_hz=1e9,
            dds_frequency_bits=20,
        ),
    )
    quantised = rabi_scan(Machine(coarse), 0, DURATIONS, seed=1)
    assert quantised.realized is not None and quantised.requested is not None
    assert quantised.requested.rabi_hz == ideal.requested.rabi_hz  # the request does not know the electronics
    assert quantised.realized.rabi_hz != quantised.requested.rabi_hz  # the amplitude word does
    assert abs(quantised.realized.rabi_hz[0] - quantised.requested.rabi_hz[0]) <= 1e6 / 2**6
    spectrum = sideband_spectroscopy(Machine(coarse), 0, DETUNINGS, seed=1)
    assert spectrum.realized is not None and spectrum.requested is not None
    grid = 1e9 / 2**20
    for want, got in zip(spectrum.requested.detunings_hz, spectrum.realized.detunings_hz):
        assert abs(got - want) <= 0.5 * grid + 1e-9 and (
            want == got or got % grid == pytest.approx(0.0, abs=1e-6)
        )
    assert spectrum.realized.detunings_hz != spectrum.requested.detunings_hz


def test_scan_parameters_are_tuples_by_name() -> None:
    p = ScanParameters({"delays_s": np.array([1e-6, 2e-6]), "rabi_hz": 1e5})
    assert p.delays_s == (1e-6, 2e-6) and p.rabi_hz == (1e5,)
    assert p == ScanParameters({"delays_s": [1e-6, 2e-6], "rabi_hz": (1e5,)})
    with pytest.raises(AttributeError, match="no parameter 'phase_rad'"):
        _ = p.phase_rad


def test_detection_histogram_and_crystal_image_propose_what_they_measured(machine: Machine) -> None:
    hist = detection_histogram(machine, 0, 200, seed=2)
    assert isinstance(hist, DetectionHistogram) and hist.subject == {"ion": 0}
    assert {"threshold", "window_s", "eps_B", "eps_D"} <= set(hist.fitted)
    assert hist.requested is not None and hist.requested.n_records == (200.0,) and hist.realized is None
    assert machine.table is not None
    proposal = machine.table.updated_with(hist).detection
    assert proposal["threshold"].experiment == "detection_histogram"
    assert "R_bright_scattered_per_s" not in proposal
    image = crystal_image(machine, seed=2)
    assert isinstance(image, CrystalImage) and image.fitted["n_ions"][0] == 2.0
    assert image.quality in ("good", "exact", "failed")


def test_calibrate_returns_the_report(two_ion, machine: Machine) -> None:
    scans: dict[str, Any] = {"pairs": [(0, 1)], "detection_records": 300, "detection_windows_s": WINDOWS}
    report = calibrate(Machine(two_ion[0].device, numerics=FAST), **scans)
    assert (
        isinstance(report, CalibrationReport) and report.table == machine.table and report.experiments == ()
    )
    assert report.results == {} and report.refused == {} and report.surrogate.table == report.table
    assert report.sample.sample_id == 0
    assert all(v == 0.0 for v in report.surrogate_error().values())  # the surrogate against itself
    with pytest.raises(ValueError, match="closed_form"):
        calibrate(machine, method="guess")
    # the machine's t0_s is the calibration's default time
    late = dataclasses.replace(machine, physics=dataclasses.replace(machine.physics, t0_s=3.0))
    assert calibrate(late, **scans).table.fitted_at_s == pytest.approx(3.0)


# ---- thermometry, mode spectroscopy, heating (Sections 4.1.5, 4.2.7, 7.5 items 2 and 6) ---------------------------------------


@pytest.mark.slow
def test_thermometry_is_exact_for_a_thermal_state_and_flags_a_non_thermal_one(single) -> None:
    """The sideband ratio P_rsb/P_bsb = nbar/(nbar + 1) (Turchette) returns nbar = 0.3 to 2e-3 at every duration on a thermal
    state, and its shot-noise twin within four sigmas."""
    dev, _dd = single
    exact = thermometry(Machine(dev), 0, 1, nbar={1: 0.3}, include_stark=False, check_durations_s=(31e-6,))
    assert exact.converged and exact.fitted["nbar"][0] == pytest.approx(0.3, abs=2e-3)
    assert exact.fitted["ratio_spread"][0] < 2e-3
    noisy = thermometry(
        Machine(dev), 0, 1, nbar={1: 0.3}, include_stark=False, shots=3000, readout=False, seed=2
    )
    assert abs(noisy.fitted["nbar"][0] - 0.3) < 4.0 * noisy.fitted["nbar"][1]
    assert 0.005 < noisy.fitted["nbar"][1] < 0.05


@pytest.mark.slow
def test_mode_spectroscopy_recovers_the_mode_frequency_eta_and_nbar_within_its_uncertainty(two_ion) -> None:
    """On the two-ion COM mode the scan recovers the mode frequency (sigma < 450 Hz), |eta| from the sideband Rabi frequency and
    nbar from the sideband ratio, each within four sigmas."""
    fx, dd = two_ion
    nbar = {2: 0.0185, 3: 0.0154}
    res = mode_spectroscopy(
        Machine(fx.device),
        0,
        3,
        nbar=nbar,
        shots=400,
        readout=True,
        span=0.01,
        coarse_points=7,
        fine_points=11,
        seed=5,
    )
    truth = fx.device.crystal.modes[3].omega_hz
    f, s = res.fitted["mode_hz"]
    assert res.converged, res.notes
    assert s < 450.0 and abs(f - truth) < 4.0 * s, (f, s, truth)
    eta, s_eta = res.fitted["eta"]
    assert abs(eta - abs(dd.etas[3])) < 4.0 * s_eta and s_eta < 0.01
    nb, s_nb = res.fitted["nbar"]
    assert abs(nb - nbar[3]) < 4.0 * max(s_nb, 1e-3)
    assert res.fitted["chi2_per_dof_blue"][0] < 3.0


@pytest.mark.slow
def test_sideband_calibrated_eta_carries_c0_and_a_carrier_derived_one_does_not() -> None:
    """With an rf record at q = 0.3 the derived and the sideband-measured eta both carry C0 = 1.018 (to 2e-4 and three sigmas),
    so a gate built from the bare eta misses 2(C0 - 1) of two-body phase (to 6e-3)."""
    omega_hz = (3.0e6, 2.9e6, 1.0e6)
    f_rf = 2.0 * math.sqrt(2.0) * 3.0e6 / 0.3
    dev = single_ion_raman_device(rf=RfDrive(voltage_peak_v=100.0, frequency_hz=f_rf))
    a, q = mathieu_from_secular(omega_hz, f_rf)
    assert q[0] == pytest.approx(0.3, abs=0.02)
    c0 = c0_wronskian(a[0], q[0])
    assert 1.015 < c0 < 1.021
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    assert dd.c0_applied
    eta_bare = abs(derive_raman_drive(single_ion_raman_device(), 0, (0, 1), scattering=False).etas[1])
    assert abs(dd.etas[1]) / eta_bare == pytest.approx(c0, rel=2e-4), (
        "C0 is inside the derived eta with an rf record"
    )
    res = mode_spectroscopy(
        Machine(dev), 0, 1, include_stark=False, span=0.005, coarse_points=5, fine_points=11
    )
    eta_sb, s_eta = res.fitted["eta"]
    assert abs(eta_sb - abs(dd.etas[1])) < 3.0 * max(s_eta, 2e-4), (eta_sb, dd.etas[1], s_eta)
    ratio = (eta_sb / eta_bare) ** 2
    assert ratio - 1.0 == pytest.approx(2.0 * (c0 - 1.0), abs=6e-3), (
        "a carrier-derived gate misses 2(C0 - 1) of two-body phase"
    )


# twelve long density-matrix integrations (the exact scan and its shot-noise twin): beyond the default 30-minute timeout on a
# loaded runner
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_heating_rate_scan_recovers_the_noise_models_rate(single) -> None:
    """Through the engine's heating channels nbar grows linearly with the delay and the fit recovers Gamma_h to 3 % and
    nbar0 = 0.1 to 0.01, and its shot-noise twin Gamma_h within four sigmas."""
    dev, _dd = single
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-12, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    truth = noisy.noise.heating_rates_quanta_per_s(noisy)[1]
    assert truth > 20.0
    delays = np.linspace(0.0, 3.0 / truth, 6)  # nbar grows by three quanta
    exact = heating_rate(Machine(noisy), 1, delays, nbar0=0.1, include_stark=False)
    assert exact.converged and exact.fitted["ndot_per_s"][0] == pytest.approx(truth, rel=0.03)
    assert exact.fitted["nbar0"][0] == pytest.approx(0.1, abs=0.01)
    assert np.all(np.diff(exact.data[:, 1]) > 0.0), "nbar grows monotonically at the heating rate"
    noisy_res = heating_rate(
        Machine(noisy), 1, delays, nbar0=0.1, include_stark=False, shots=1500, readout=False, seed=3
    )
    ndot, s = noisy_res.fitted["ndot_per_s"]
    assert abs(ndot - truth) < 4.0 * s and 0.0 < s < 0.3 * truth


# ---- the field, Stark and crosstalk scans (Section 7.5 items 7, 8, 9) -------------------------------------------------------


def test_field_scan_inverts_the_zeeman_shift_for_the_field(single) -> None:
    """The Ramsey-frequency offset from a frame at nu(5.03 G), inverted through nu(B), recovers B = 5 G within four sigmas
    (sigma < 0.01 G; to 2e-5 G without shot noise) with d nu/dB = 3.1 kHz/G."""
    dev, _dd = single
    res = field_scan(
        Machine(dev),
        0,
        np.linspace(0.0, 2e-3, 9),
        include_stark=False,
        b_seed_gauss=5.03,
        shots=800,
        readout=False,
        seed=4,
    )
    b, s_b = res.fitted["B_gauss"]
    assert res.converged and abs(b - 5.0) < 4.0 * s_b and s_b < 0.01
    assert res.fitted["dnu_dB_hz_per_g"][0] == pytest.approx(2.0 * 310.87 * 5.0, rel=0.02)
    exact = field_scan(Machine(dev), 0, np.linspace(0.0, 2e-3, 9), include_stark=False, b_seed_gauss=5.03)
    assert exact.fitted["B_gauss"][0] == pytest.approx(5.0, abs=2e-5)


def test_stark_scan_measures_each_beams_light_shift_and_their_sum(two_ion) -> None:
    """One beam on during the Ramsey delay measures that beam's differential light shift and the drive's shift is their sum,
    each equal to the derived one within four sigmas (sigma < 10 Hz)."""
    fx, dd = two_ion
    res = stark_scan(
        Machine(fx.device),
        0,
        np.linspace(0.0, 2e-3, 9),
        nbar={2: 0.0185, 3: 0.0154},
        shots=1000,
        readout=True,
        seed=6,
    )
    assert res.converged, res.notes
    total, s_total = res.fitted["stark_shift_hz"]
    assert abs(total - dd.stark_shift_hz) < 4.0 * s_total and s_total < 10.0, (
        total,
        s_total,
        dd.stark_shift_hz,
    )
    for b in fx.gate_drives[0].beams:
        shift, s = res.fitted[f"stark_shift_hz[{b}]"]
        assert abs(shift - differential_stark_shift_hz(fx.device, 0, (b,))) < 4.0 * s


def test_crosstalk_scan_recovers_the_derived_ratio_and_a_zero_phase(two_ion) -> None:
    """The neighbour's Rabi rate under ion 0's beams gives the derived crosstalk ratio within four sigmas and, the wavefront
    being perpendicular to the chain, a zero crosstalk phase."""
    fx, dd = two_ion
    eps = abs(crosstalk_ratios(fx.device, 0, fx.gate_drives[0].beams)[1])
    res = crosstalk_scan(
        Machine(fx.device),
        0,
        np.linspace(0.0, 0.5 / (eps * dd.carrier_rabi_hz), 16),
        nbar={2: 0.0185, 3: 0.0154},
        shots=600,
        readout=True,
        analysis_phases_rad=np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False),
        seed=7,
    )
    assert res.converged, res.notes
    e, s_e = res.fitted["eps[1]"]
    assert abs(e - eps) < 4.0 * s_e and s_e < 0.05 * eps
    phase, s_phi = res.fitted["phase_rad[1]"]
    assert abs(phase) < 4.0 * s_phi and s_phi < 0.2, (phase, s_phi)
    assert res.fitted["rate_hz"][0] == pytest.approx(dd.carrier_rabi_hz, rel=5e-3)


# ---- micromotion compensation (Section 7.5; Berkeland 1998) ------------------------------------------------------------------


@pytest.mark.slow
def test_micromotion_scan_by_the_sideband_ratio_nulls_the_stray_field_through_the_exact_modulated_builder() -> (
    None
):
    """A 30 V/m stray field (beta = 0.245 at 30 MHz rf) is nulled by the sideband-ratio scan through the exact modulated
    builder at -30 V/m to 0.5 V/m, with the residual |beta| below 1e-3."""
    dev = single_ion_raman_device(rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6), stray=(30.0, 0.0, 0.0))
    dk = derive_raman_drive(dev, 0, (0, 1), scattering=False).delta_k
    # the in-phase index is signed (u_1 = -(1/2) q u_0 at the adopted Mathieu origin); Berkeland's magnitude is pinned
    beta_in = dev.trap.micromotion_beta(dev.crystal.species[0], np.asarray(dk)).in_phase
    assert abs(beta_in) == pytest.approx(0.245, abs=0.01)
    assert beta_in == pytest.approx(signed_beta(dev, 0, np.asarray(dk)), rel=1e-12)
    res = micromotion_scan(Machine(dev), 0, 0, {"Ex": (-60.0, 0.0)}, method="sideband_ratio", points=5)
    assert res.converged, res.notes
    null, s_null = res.fitted["shim[Ex]"]
    assert null == pytest.approx(-30.0, abs=0.5) and s_null < 2.0
    assert res.fitted["beta[0]"][0] < 1e-3 and res.fitted["beta_before"][0] == pytest.approx(0.245, abs=0.01)
    assert res.fitted["carrier_hz"][0] == pytest.approx(
        derive_raman_drive(dev, 0, (0, 1), scattering=False).carrier_rabi_hz, rel=1e-3
    )


@pytest.mark.slow
def test_rf_photon_correlation_signal_is_odd_in_beta_and_nulls_the_stray_field() -> None:
    """Berkeland's rf-photon correlation: the first rf harmonic of the detection beam's periodic steady state is odd and linear
    in beta, larger half a linewidth to the red, follows the sign of the residual field, and nulls a 20 V/m field to 0.5 V/m."""
    base = yb171_chain(2).device
    dev = dataclasses.replace(
        base,
        trap=dataclasses.replace(
            base.trap,
            rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
            stray_field_v_per_m=(20.0, 0.0, 0.0),
        ),
        preparation=None,
    )
    dev = dataclasses.replace(dev, crystal=solve_crystal(dev.trap, dev.crystal.species))
    idx = detection_beams(dev, 0)
    beam = idx[0]
    _rates, _scheme, model = detection_rates_for_ion(
        dev.crystal.species[0],
        dev.field.B_gauss,
        dev.field.direction,
        [dev.beams[k] for k in idx],
        position_m=tuple(float(x) for x in dev.crystal.positions_m[0]),
    )
    omega_rf = dev.trap.rf.omega_rad_s
    signals = {}
    for beta in (-0.1, 0.1, 0.2):
        times, rates = periodic_scattering(model, idx.index(beam), beta * omega_rf, omega_rf, n_points=24)
        signals[beta] = correlation_signal(times, rates, omega_rf)
    assert signals[0.1][0] == pytest.approx(-signals[-0.1][0], rel=1e-3) and abs(signals[0.1][0]) > 0.0
    assert signals[0.2][0] / signals[0.1][0] == pytest.approx(2.0, rel=0.15), "linear in beta near the null"
    assert signals[0.1][1] == pytest.approx(signals[-0.1][1], rel=1e-6), "the mean rate is even"
    # on resonance the first-order response vanishes (the rate is even in the detuning there): the first harmonic is tiny
    assert abs(signals[0.1][0]) < 1e-3
    # retuned to -Gamma/2 and projected on the response phase: odd, linear, and a thousand times larger
    corr = _Correlation(dev, 0, beam, 24)
    s_plus, mean_plus = corr.observables(0.045)
    s_minus, _ = corr.observables(-0.045)
    assert s_plus > 1e-2 and s_minus == pytest.approx(-s_plus, rel=1e-3)
    assert corr.observables(0.09)[0] / s_plus == pytest.approx(2.0, rel=0.1)
    assert mean_plus < signals[0.1][1], "half a linewidth to the red the mean rate is lower than on resonance"
    k_vec = np.asarray(dev.beams[beam].k_vector())
    assert signed_beta(dev, 0, k_vec) != 0.0
    # beta changes sign as the shim crosses the compensated value (+20 V/m residual along x, compensated at Ex = -20 V/m):
    # under the adopted rf phase origin u_1 = -(1/2) q u_0, sign(beta) = -sign(q_x E_x), and the odd signal steps by pi with it
    q_x = float(dev.trap.mathieu(dev.crystal.species[0]).q[0, 0])
    for shim, residual_x in ((-10.0, +10.0), (-30.0, -10.0)):
        trial = device_with_compensation(dev, {"Ex": shim})
        assert trial.trap.residual_field_v_per_m()[0] == pytest.approx(residual_x, rel=1e-12)
        beta_trial = signed_beta(trial, 0, k_vec)
        assert math.copysign(1.0, beta_trial) == -math.copysign(1.0, q_x * residual_x)
        assert math.copysign(1.0, corr.observables(beta_trial)[0]) == math.copysign(1.0, beta_trial)
    assert signed_beta(device_with_compensation(dev, {"Ex": -10.0}), 0, k_vec) == pytest.approx(
        -signed_beta(device_with_compensation(dev, {"Ex": -30.0}), 0, k_vec), rel=1e-9
    )
    assert signed_beta(device_with_compensation(dev, {"Ex": -20.0}), 0, k_vec) == pytest.approx(
        0.0, abs=1e-15
    )
    res = micromotion_scan(
        Machine(dev), 0, beam, {"Ex": (-40.0, 0.0)}, method="rf_photon_correlation", points=5, rf_points=24
    )
    assert res.converged, res.notes
    null, _s_null = res.fitted["shim[Ex]"]
    assert null == pytest.approx(-20.0, abs=0.5)
    assert abs(res.fitted[f"beta[{beam}]"][0]) < 5e-3
