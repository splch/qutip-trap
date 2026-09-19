"""The M8 simulated experiments against the device's true derived values (PLAN.md Sections 4.1.5, 4.2.7, 7.5, 7.9, 9.1, 9.3,
9.17): the sideband-lineshape fit on the exact dynamics, thermometry exactness, mode spectroscopy with its sub-kilohertz
uncertainty and the C0 row, the heating-rate measurement, the field, Stark and crosstalk scans, the micromotion compensation
scan by the exact modulated builder and by the rf-photon-correlation steady state."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qutip_trap.api import RfDrive, white_spectrum
from qutip_trap.experiments import (
    crosstalk_scan,
    field_scan,
    heating_rate,
    micromotion_scan,
    mode_spectroscopy,
    rabi_scan,
    sideband_spectroscopy,
    stark_scan,
    thermometry,
)
from qutip_trap.experiments.micromotion import correlation_signal, periodic_scattering, signed_beta
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive, differential_stark_shift_hz
from qutip_trap.machine import as_machine
from qutip_trap.readout.fluorescence import detection_rates_for_ion
from qutip_trap.trap.mathieu import c0_wronskian, mathieu_from_secular
from tests.m2_fixtures import single_ion_raman_device
from tests.m6_fixtures import circuit_fixture


@pytest.fixture(scope="module")
def single():  # type: ignore[no-untyped-def]
    dev = single_ion_raman_device()
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    return dev, dd


@pytest.fixture(scope="module")
def two_ion():  # type: ignore[no-untyped-def]
    fx = circuit_fixture(2)
    dd = derive_raman_drive(fx.device, 0, fx.gate_drives[0].beams, scattering=False)
    return fx, dd


def test_carrier_lineshape_on_the_exact_dynamics_gives_the_derived_rabi_frequency_and_pi_time(single) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5's CI assertion: the fitted Omega of a noiseless carrier scan reproduces the derived Omega (the n = 0 Debye-Waller
    factor e^{-eta^2/2} included, Section 4.2.7) within the reported uncertainty, and the pi time is pi/Omega."""
    dev, dd = single
    f, eta = dd.carrier_rabi_hz, dd.etas[1]
    t_pi = 0.5 / f
    mus = np.linspace(-2.2 * f, 2.2 * f, 23)
    res = sideband_spectroscopy(as_machine(dev), 0, mus, duration_s=t_pi, include_stark=False, fit=True)
    expected = f * math.exp(-0.5 * eta**2)
    omega_fit, s_omega = res.fitted["omega_carrier_hz"]
    assert omega_fit == pytest.approx(expected, rel=2e-3)
    assert abs(omega_fit - expected) < 5.0 * max(s_omega, 1e-3 * f)
    assert res.fitted["carrier_fit_hz"][0] == pytest.approx(0.0, abs=50.0)
    assert math.pi / (2.0 * math.pi * omega_fit) == pytest.approx(t_pi / math.exp(-0.5 * eta**2), rel=2e-3)


@pytest.mark.slow
def test_thermometry_is_exact_for_a_thermal_state_and_flags_a_non_thermal_one(single) -> None:  # type: ignore[no-untyped-def]
    """Section 4.2.7 (i): P_rsb/P_bsb = nbar/(nbar + 1) for every pulse duration on a thermal state (Turchette); the shot-noise
    uncertainty covers the truth; a Fock state gives a duration-dependent ratio and is flagged."""
    dev, _dd = single
    exact = thermometry(as_machine(dev), 0, 1, nbar={1: 0.3}, include_stark=False, check_durations_s=(31e-6,))
    assert exact.converged and exact.fitted["nbar"][0] == pytest.approx(0.3, abs=2e-3)
    assert exact.fitted["ratio_spread"][0] < 2e-3
    noisy = thermometry(
        as_machine(dev), 0, 1, nbar={1: 0.3}, include_stark=False, shots=3000, readout=False, seed=2
    )
    assert abs(noisy.fitted["nbar"][0] - 0.3) < 4.0 * noisy.fitted["nbar"][1]
    assert 0.005 < noisy.fitted["nbar"][1] < 0.05


@pytest.mark.slow
def test_mode_spectroscopy_recovers_the_mode_frequency_eta_and_nbar_within_its_uncertainty(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5 item 2 on the two-ion fixture's COM mode: the two-stage scan fitted with the plan's lineshape gives the mode
    frequency to well below a kilohertz (the FM solvers' need), |eta| from the sideband Rabi frequency (C0 = 1 without an rf
    record) and nbar from the sideband ratio, each within its reported uncertainty."""
    fx, dd = two_ion
    nbar = {2: 0.0185, 3: 0.0154}
    res = mode_spectroscopy(
        as_machine(fx.device),
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
    # the fitted centre's uncertainty scales with the line's own width, 0.5/t_pi = Omega/2: the pin was 300 Hz while the
    # fixture's carrier ran at 100.9 kHz and the M0a P3/2 correction of 2026-09-08 raised it by 1.4577 to 147.1 kHz, so the
    # same scan now reads 322 Hz (300 x 1.4577 = 437). What Section 7.5 actually requires is the FM solvers' sub-kilohertz.
    assert s < 450.0 and abs(f - truth) < 4.0 * s, (f, s, truth)
    eta, s_eta = res.fitted["eta"]
    assert abs(eta - abs(dd.etas[3])) < 4.0 * s_eta and s_eta < 0.01
    nb, s_nb = res.fitted["nbar"]
    assert abs(nb - nbar[3]) < 4.0 * max(s_nb, 1e-3)
    assert res.fitted["chi2_per_dof_blue"][0] < 3.0


@pytest.mark.slow
def test_sideband_calibrated_eta_carries_c0_and_a_carrier_derived_one_does_not() -> None:
    """Section 9.17, "C0 applied once": with an rf record at q = 0.3 the sideband Rabi frequency gives eta C0 (C0 = 1 + 3q^2/16 + O(q^4)
    = 1.018), the bare Delta k x0 c does not, and the two-body phase of a gate built from the two differs by 2(C0 - 1)."""
    omega_hz = (3.0e6, 2.9e6, 1.0e6)
    f_rf = 2.0 * math.sqrt(2.0) * 3.0e6 / 0.3
    dev = single_ion_raman_device(rf=RfDrive(voltage_peak_v=100.0, frequency_hz=f_rf))
    a, q = mathieu_from_secular(omega_hz, f_rf)
    assert q[0] == pytest.approx(0.3, abs=0.02)
    c0 = c0_wronskian(a[0], q[0])
    assert 1.015 < c0 < 1.021
    dd = derive_raman_drive(dev, 0, (0, 1), scattering=False)
    assert dd.c0_applied
    bare = single_ion_raman_device()
    eta_bare = abs(derive_raman_drive(bare, 0, (0, 1), scattering=False).etas[1])
    assert abs(dd.etas[1]) / eta_bare == pytest.approx(c0, rel=2e-4), (
        "C0 is inside the derived eta with an rf record"
    )
    res = mode_spectroscopy(
        as_machine(dev), 0, 1, include_stark=False, span=0.005, coarse_points=5, fine_points=11
    )
    eta_sb, s_eta = res.fitted["eta"]
    assert abs(eta_sb - abs(dd.etas[1])) < 3.0 * max(s_eta, 2e-4), (eta_sb, dd.etas[1], s_eta)
    ratio = (eta_sb / eta_bare) ** 2
    assert ratio - 1.0 == pytest.approx(2.0 * (c0 - 1.0), abs=6e-3), (
        "a carrier-derived gate misses 2(C0 - 1) of two-body phase"
    )


@pytest.mark.slow
def test_heating_rate_scan_recovers_the_noise_models_rate(single) -> None:  # type: ignore[no-untyped-def]
    """Section 4.1.5's procedure through the engine's own heating channels: nbar against the delay is linear at Gamma_h and the fit
    recovers the rate and the prepared occupation within their uncertainties (Section 9.1, 'Heating dynamics'). Marked slow: the
    probes at the hottest delays are density matrices of d ~ 70 Fock levels under mesolve (about 15 minutes in all)."""
    dev, _dd = single
    noisy = dataclasses.replace(
        dev,
        noise=dataclasses.replace(
            dev.noise, S_E=white_spectrum(2e-12, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
        ),
    )
    truth = noisy.noise.heating_rates_quanta_per_s(noisy)[1]
    assert truth > 20.0
    # nbar grows by three quanta (the plan's 10/ndot heats to ten and needs d = 170)
    delays = np.linspace(0.0, 3.0 / truth, 6)
    exact = heating_rate(as_machine(noisy), 1, delays, nbar0=0.1, include_stark=False)
    assert exact.converged and exact.fitted["ndot_per_s"][0] == pytest.approx(truth, rel=0.03)
    assert exact.fitted["nbar0"][0] == pytest.approx(0.1, abs=0.01)
    nb = exact.data[:, 1]
    assert np.all(np.diff(nb) > 0.0), "nbar grows monotonically at the heating rate"
    noisy_res = heating_rate(
        as_machine(noisy), 1, delays, nbar0=0.1, include_stark=False, shots=1500, readout=False, seed=3
    )
    ndot, s = noisy_res.fitted["ndot_per_s"]
    assert abs(ndot - truth) < 4.0 * s and 0.0 < s < 0.3 * truth


def test_field_scan_inverts_the_zeeman_shift_for_the_field(single) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5 item 9: with the frame at nu(B_seed), the Ramsey-frequency offset inverted through nu(B) recovers the device's
    field within the uncertainty sigma_nu/|d nu/dB| (3.1 kHz/G for the 171Yb+ clock transition at 5 G)."""
    dev, _dd = single
    res = field_scan(
        as_machine(dev),
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
    exact = field_scan(as_machine(dev), 0, np.linspace(0.0, 2e-3, 9), include_stark=False, b_seed_gauss=5.03)
    assert exact.fitted["B_gauss"][0] == pytest.approx(5.0, abs=2e-5)


def test_stark_scan_measures_each_beams_light_shift_and_their_sum(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5 item 7 per (ion, beam): one beam on during the Ramsey delay gives that beam's differential light shift; the
    drive's shift is the sum, equal to the derived one within the fit uncertainty."""
    fx, dd = two_ion
    res = stark_scan(
        as_machine(fx.device),
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


def test_crosstalk_scan_recovers_the_derived_ratio_and_a_zero_phase(two_ion) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5 item 8: the neighbour's Rabi rate under ion 0's addressing beams over ion 0's own gives epsilon_01 = 2.2 % on the
    2.5 um fixture; the crosstalk axis measured by the pi/2 - pi - pi/2(phi) sequence coincides with the neighbour's frame (the beams'
    wavefront is perpendicular to the chain, so the geometric phase is zero)."""
    fx, dd = two_ion
    eps = abs(crosstalk_ratios(fx.device, 0, fx.gate_drives[0].beams)[1])
    ts = np.linspace(0.0, 0.5 / (eps * dd.carrier_rabi_hz), 16)
    res = crosstalk_scan(
        as_machine(fx.device),
        0,
        ts,
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


def test_rabi_scan_with_thermometry_nbar_fits_the_bare_rabi_frequency_through_every_modes_debye_waller_factor(
    two_ion,
) -> None:  # type: ignore[no-untyped-def]
    """Section 7.5's bootstrap on two coupled modes: nbar fixed from thermometry, the fit (f, contrast, offset) carries the
    Debye-Waller factor of BOTH x modes (Section 4.2.7 iii), so the fitted f is the bare derived Omega within its uncertainty."""
    fx, dd = two_ion
    f = dd.carrier_rabi_hz
    res = rabi_scan(
        as_machine(fx.device),
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


# ---- micromotion compensation (Section 7.5; Berkeland 1998) -----------------------------------------------------------------------------


def _rf_device(stray_x_v_per_m: float):  # type: ignore[no-untyped-def]
    return single_ion_raman_device(
        rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6), stray=(stray_x_v_per_m, 0.0, 0.0)
    )


@pytest.mark.slow
def test_micromotion_scan_by_the_sideband_ratio_nulls_the_stray_field_through_the_exact_modulated_builder() -> (
    None
):
    """A 30 V/m stray field gives beta = 0.245 along the Raman Delta k at Omega_rf = 2 pi x 30 MHz; the exact e^{i beta cos Omega t}
    modulation of the rf-locked drive puts J_1(beta) Omega on the first micromotion sideband, whose excitation is even in beta: the
    parabola over the compensation field crosses its minimum at -30 V/m and the residual |beta| at the null is below 1e-3."""
    dev = _rf_device(30.0)
    dk = derive_raman_drive(dev, 0, (0, 1), scattering=False).delta_k
    # the in-phase index is SIGNED (M1: u_1 = -(1/2) q u_0 at the adopted Mathieu origin, so a stray field contracts at the
    # rf phase origin); Berkeland's magnitude is what this row pins, and the experiment layer reads the same product
    beta_in = dev.trap.micromotion_beta(dev.crystal.species[0], np.asarray(dk)).in_phase
    assert abs(beta_in) == pytest.approx(0.245, abs=0.01)
    assert beta_in == pytest.approx(signed_beta(dev, 0, np.asarray(dk)), rel=1e-12)
    res = micromotion_scan(as_machine(dev), 0, 0, {"Ex": (-60.0, 0.0)}, method="sideband_ratio", points=5)
    assert res.converged, res.notes
    null, s_null = res.fitted["shim[Ex]"]
    assert null == pytest.approx(-30.0, abs=0.5) and s_null < 2.0
    assert res.fitted["beta[0]"][0] < 1e-3 and res.fitted["beta_before"][0] == pytest.approx(0.245, abs=0.01)
    assert res.fitted["carrier_hz"][0] == pytest.approx(
        derive_raman_drive(dev, 0, (0, 1), scattering=False).carrier_rabi_hz, rel=1e-3
    )


@pytest.mark.slow
def test_rf_photon_correlation_signal_is_odd_in_beta_and_nulls_the_stray_field() -> None:
    """Berkeland's rf-photon correlation from the periodic steady state of the detection beam's Bloch model: the complex first
    harmonic of the photon rate at the rf frequency is odd in the signed modulation index and linear near the null; projected on
    the atom's response phase at the half-linewidth working point it is the signal whose slope the residual beta is read from
    (a thousand times the on-resonance response), and the shim scan's line fit crosses zero at the compensating field."""
    from qutip_trap.experiments.micromotion import _Correlation
    from qutip_trap.light.roles import detection_beams

    fx = circuit_fixture(2, with_recipe=False)
    dev = dataclasses.replace(
        fx.device,
        trap=dataclasses.replace(
            fx.device.trap,
            rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
            stray_field_v_per_m=(20.0, 0.0, 0.0),
        ),
    )
    from qutip_trap.trap.crystal import solve_crystal

    dev = dataclasses.replace(dev, crystal=solve_crystal(dev.trap, dev.crystal.species))
    beam = detection_beams(dev, 0)[0]
    idx = detection_beams(dev, 0)
    _rates, _scheme, model = detection_rates_for_ion(
        dev.crystal.species[0],
        dev.field.B_gauss,
        dev.field.direction,
        [dev.beams[k] for k in idx],
        position_m=tuple(float(x) for x in dev.crystal.positions_m[0]),
    )
    omega_rf = dev.trap.rf.omega_rad_s  # type: ignore[union-attr]
    signals = {}
    for beta in (-0.1, 0.1, 0.2):
        times, rates = periodic_scattering(model, idx.index(beam), beta * omega_rf, omega_rf, n_points=24)
        signals[beta] = correlation_signal(times, rates, omega_rf)
    assert signals[0.1][0] == pytest.approx(-signals[-0.1][0], rel=1e-3) and abs(signals[0.1][0]) > 0.0
    assert signals[0.2][0] / signals[0.1][0] == pytest.approx(2.0, rel=0.15), "linear in beta near the null"
    assert signals[0.1][1] == pytest.approx(signals[-0.1][1], rel=1e-6), "the mean rate is even"
    # on resonance the first-order response vanishes (the rate is even in the detuning there): the first harmonic is tiny
    assert abs(signals[0.1][0]) < 1e-3
    # the experiment retunes the beam to -Gamma/2 and projects on the response phase: odd, linear, and a thousand times larger
    corr = _Correlation(dev, 0, beam, {"rf_points": 24})
    s_plus, mean_plus = corr.observables(0.045)
    s_minus, _ = corr.observables(-0.045)
    assert s_plus > 1e-2 and s_minus == pytest.approx(-s_plus, rel=1e-3)
    assert corr.observables(0.09)[0] / s_plus == pytest.approx(2.0, rel=0.1)
    assert mean_plus < signals[0.1][1], "half a linewidth to the red the mean rate is lower than on resonance"
    k_vec = np.asarray(dev.beams[beam].k_vector())
    assert signed_beta(dev, 0, k_vec) != 0.0
    # Section 9.17, "the modulation index changes sign ... as the shim voltage crosses the compensated value": the
    # ABSOLUTE sign, not just oddness in a hand-supplied beta. The residual field here is +20 V/m along x, compensated
    # at Ex = -20 V/m; under the adopted rf phase origin u_1 = -(1/2) Q u_0, so sign(beta) = -sign(q_x E_x) on either
    # side of it, and the signal (odd in beta) steps by pi with it while the fitted null does not move.
    from qutip_trap.experiments.micromotion import device_with_compensation

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
        as_machine(dev), 0, beam, {"Ex": (-40.0, 0.0)}, method="rf_photon_correlation", points=5, rf_points=24
    )
    assert res.converged, res.notes
    null, s_null = res.fitted["shim[Ex]"]
    assert null == pytest.approx(-20.0, abs=0.5)
    assert abs(res.fitted[f"beta[{beam}]"][0]) < 5e-3
