"""Calibration emulation (PLAN.md Section 7.5; Section 9.17 rows "Sideband lineshape fit", "C0 applied once", "Calibration
dependency graph"; milestone M8), through the package.

1. The plan's sideband excitation lineshape as the fit function: the pi time pi/Omega, the half-depth weight at delta = Omega, the
   half-Rabi form's Omega/2 (negative control), and the carrier scan of the exact dynamics fitted with it.
2. The C0 row: the sideband-calibrated eta of a single 171Yb+ ion at q = 0.3 carries C0 = 1 + 3q^2/16 + O(q^4), the carrier-derived
   one does not; a gate built from the two differs in two-body phase by 2(C0 - 1).
3. The single-ion experiments against the device: thermometry (exact), the heating-rate scan against the noise model's rate, the
   field scan inverted through nu(B), the Stark scan per beam, the crosstalk scan's ratio and phase on the two-ion fixture.
4. The micromotion compensation scan: the exact modulated builder's sideband ratio and the rf-photon-correlation periodic steady
   state both null a 30 V/m stray field.
5. The full calibration of the two-ion 171Yb+ fixture by simulated experiments (reduced scans): every entry against its derived
   truth in units of its own uncertainty, the surrogate's error, the entangling waveform's closure and phase alignment, and a Bell
   circuit run from the fitted table against the intrinsic budget.
6. The servo of Section 7.5: the residual variance of a high-passed Ornstein-Uhlenbeck drift.

Run: uv run python validation/scripts/check_calibration.py (about thirty minutes, seventeen of them section 3's two heating-rate
scans). Lines prefixed MC: are Monte Carlo results.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time

import numpy as np

from qutip_trap.api import Circuit, Operation, RfDrive, SolverOptions, register_fidelity, run, white_spectrum
from qutip_trap.calibration import CalibrationScans, calibrate_with_report
from qutip_trap.experiments import (
    crosstalk_scan,
    field_scan,
    heating_rate,
    micromotion_scan,
    mode_spectroscopy,
    sideband_spectroscopy,
    stark_scan,
    thermometry,
)
from qutip_trap.experiments.fitting import fit_lineshape, lineshape_model, sideband_lineshape
from qutip_trap.light.raman import crosstalk_ratios, derive_raman_drive
from qutip_trap.noise.model import servo_residual
from qutip_trap.noise.processes import correlated_normals
from qutip_trap.trap.mathieu import c0_wronskian, mathieu_from_secular
from qutip_trap.units import TWO_PI
from tests.m2_fixtures import single_ion_raman_device
from tests.m6_fixtures import circuit_fixture

_CLOCK = [time.perf_counter()]


def section(title: str) -> None:
    now = time.perf_counter()
    print(f"[check_calibration] {now - _CLOCK[0]:.0f} s since the previous section", file=sys.stderr)
    _CLOCK[0] = now
    print(f"\n== {title}")


def sigmas(value: float, truth: float, unc: float) -> str:
    return f"{value:.6g} +- {unc:.3g} against {truth:.6g} ({abs(value - truth) / unc if unc > 0 else float('nan'):.2f} sigma)"


section("1. the sideband excitation lineshape (Section 13; Section 9.17 row)")
omega = TWO_PI * 50e3
t_pi = math.pi / omega
print(
    f"  P(0, t = pi/Omega) = {sideband_lineshape(0.0, omega, t_pi):.6f}; weight at delta = Omega: {omega**2 / (omega**2 + omega**2):.3f}"
)
det = np.linspace(-120e3, 120e3, 49)
y = lineshape_model(np.array([0.0, 50e3, 1.0, 0.0]), det, t_pi)
plan = fit_lineshape(det, y, t_pi, guess=(0.0, 40e3))
half = fit_lineshape(det, y, t_pi, guess=(0.0, 40e3), form="half_rabi")
print(
    f"  fitted Omega/2pi: plan form {plan.params[1]:.3f} Hz (pi time {math.pi / (TWO_PI * plan.params[1]) * 1e6:.4f} us), half-Rabi form {half.params[1]:.3f} Hz (negative control: Omega/2)"
)
dev1 = single_ion_raman_device()
dd1 = derive_raman_drive(dev1, 0, (0, 1), scattering=False)
f1, eta1 = dd1.carrier_rabi_hz, dd1.etas[1]
res = sideband_spectroscopy(
    dev1, 0, np.linspace(-2.2 * f1, 2.2 * f1, 23), duration_s=0.5 / f1, include_stark=False, fit=True
)
print(
    f"  carrier scan of the exact dynamics: fitted Omega {res.fitted['omega_carrier_hz'][0]:.3f} Hz against derived {f1:.3f} x e^(-eta^2/2) = {f1 * math.exp(-0.5 * eta1**2):.3f} Hz; centre {res.fitted['carrier_fit_hz'][0]:+.2f} Hz"
)

section("2. C0 applied once (Section 9.17 row): sideband-calibrated against carrier-derived eta at q = 0.3")
f_rf = 2.0 * math.sqrt(2.0) * 3.0e6 / 0.3
dev_rf = single_ion_raman_device(rf=RfDrive(voltage_peak_v=100.0, frequency_hz=f_rf))
a_m, q_m = mathieu_from_secular((3.0e6, 2.9e6, 1.0e6), f_rf)
c0 = c0_wronskian(a_m[0], q_m[0])
dd_rf = derive_raman_drive(dev_rf, 0, (0, 1), scattering=False)
eta_bare = abs(dd1.etas[1])
print(
    f"  q_x = {q_m[0]:.4f}, C0 = {c0:.6f} (series 1 + 3q^2/16 = {1 + 3 * q_m[0] ** 2 / 16:.6f}); derived eta with rf record / bare = {abs(dd_rf.etas[1]) / eta_bare:.6f}"
)
res = mode_spectroscopy(dev_rf, 0, 1, include_stark=False, span=0.005, coarse_points=5, fine_points=11)
eta_sb = res.fitted["eta"][0]
print(
    f"  sideband-calibrated eta {eta_sb:.6f} +- {res.fitted['eta'][1]:.2g} against eta C0 = {abs(dd_rf.etas[1]):.6f}; two-body phase ratio (eta_sb/eta_bare)^2 - 1 = {(eta_sb / eta_bare) ** 2 - 1:.4f} against 2(C0 - 1) = {2 * (c0 - 1):.4f}"
)
print(
    f"  mode frequency {res.fitted['mode_hz'][0]:.2f} +- {res.fitted['mode_hz'][1]:.2f} Hz against {dev_rf.crystal.modes[1].omega_hz:.2f} Hz ({res.notes[-1] if res.notes else ''})"
)

section("3. single-ion experiments against the device (Sections 4.1.5, 4.2.7, 7.5)")
th = thermometry(dev1, 0, 1, nbar={1: 0.3}, include_stark=False, check_durations_s=(31e-6,))
print(
    f"  thermometry of a thermal state at nbar = 0.3: {th.fitted['nbar'][0]:.6f} (ratio {th.fitted['ratio'][0]:.6f} against 0.3/1.3 = {0.3 / 1.3:.6f}); ratio spread under a second duration {th.fitted['ratio_spread'][0]:.2e}"
)
noisy1 = dataclasses.replace(
    dev1,
    noise=dataclasses.replace(
        dev1.noise, S_E=white_spectrum(2e-12, "(V/m)^2/(rad/s)"), correlation_length_m=0.0
    ),
)
ndot_true = noisy1.noise.heating_rates_quanta_per_s(noisy1)[1]
hr = heating_rate(noisy1, 1, np.linspace(0.0, 3.0 / ndot_true, 6), nbar0=0.1, include_stark=False)
print(
    f"  heating rate (exact populations): {hr.fitted['ndot_per_s'][0]:.4f} quanta/s against Gamma_h = {ndot_true:.4f}; intercept {hr.fitted['nbar0'][0]:.5f} against 0.1"
)
hr_mc = heating_rate(
    noisy1,
    1,
    np.linspace(0.0, 3.0 / ndot_true, 6),
    nbar0=0.1,
    include_stark=False,
    shots=1500,
    readout=False,
    seed=3,
)
print(
    f"MC: heating rate with 1500 shots per point: {sigmas(hr_mc.fitted['ndot_per_s'][0], ndot_true, hr_mc.fitted['ndot_per_s'][1])}"
)
fs = field_scan(dev1, 0, np.linspace(0.0, 2e-3, 9), include_stark=False, b_seed_gauss=5.03)
print(
    f"  field scan from a 5.03 G seed (exact): B = {fs.fitted['B_gauss'][0]:.6f} G, d nu/dB = {fs.fitted['dnu_dB_hz_per_g'][0]:.2f} Hz/G"
)
fs_mc = field_scan(
    dev1,
    0,
    np.linspace(0.0, 2e-3, 9),
    include_stark=False,
    b_seed_gauss=5.03,
    shots=800,
    readout=False,
    seed=4,
)
print(
    f"MC: field scan with 800 shots per point: {sigmas(fs_mc.fitted['B_gauss'][0], 5.0, fs_mc.fitted['B_gauss'][1])} G"
)
fx = circuit_fixture(2)
dd = derive_raman_drive(fx.device, 0, fx.gate_drives[0].beams, scattering=False)
st = stark_scan(
    fx.device,
    0,
    np.linspace(0.0, 2e-3, 9),
    gate_drive=fx.gate_drives[0],
    nbar={2: 0.0185, 3: 0.0154},
    shots=1000,
    readout=True,
    seed=6,
)
print(
    f"MC: Stark scan per beam on the two-ion fixture: {sigmas(st.fitted['stark_shift_hz'][0], dd.stark_shift_hz, st.fitted['stark_shift_hz'][1])} Hz; per beam "
    + ", ".join(f"{st.fitted[f'stark_shift_hz[{b}]'][0]:.2f}" for b in fx.gate_drives[0].beams)
)
eps_true = abs(crosstalk_ratios(fx.device, 0, fx.gate_drives[0].beams)[1])
xt = crosstalk_scan(
    fx.device,
    0,
    np.linspace(0.0, 0.5 / (eps_true * dd.carrier_rabi_hz), 16),
    gate_drive=fx.gate_drives[0],
    nbar={2: 0.0185, 3: 0.0154},
    shots=600,
    readout=True,
    analysis_phases_rad=np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False),
    seed=7,
)
print(
    f"MC: crosstalk scan: eps_01 {sigmas(xt.fitted['eps[1]'][0], eps_true, xt.fitted['eps[1]'][1])}; axis phase {xt.fitted['phase_rad[1]'][0]:+.4f} +- {xt.fitted['phase_rad[1]'][1]:.4f} rad against 0"
)

section("4. micromotion compensation (Section 7.5; Berkeland 1998)")
dev_mm = single_ion_raman_device(rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6), stray=(30.0, 0.0, 0.0))
beta0 = dev_mm.trap.micromotion_beta(
    dev_mm.crystal.species[0], np.asarray(derive_raman_drive(dev_mm, 0, (0, 1), scattering=False).delta_k)
).in_phase
mm = micromotion_scan(dev_mm, 0, 0, {"Ex": (-60.0, 0.0)}, method="sideband_ratio", points=5)
print(
    f"  sideband-ratio method: beta before {beta0:.4f}; null of the compensation field {mm.fitted['shim[Ex]'][0]:.4f} +- {mm.fitted['shim[Ex]'][1]:.2g} V/m against -30; residual |beta| {mm.fitted['beta[0]'][0]:.2e}"
)
from qutip_trap.light.roles import detection_beams  # noqa: E402
from qutip_trap.trap.crystal import solve_crystal  # noqa: E402

fx_rf = circuit_fixture(2, with_recipe=False)
dev_c = dataclasses.replace(
    fx_rf.device,
    trap=dataclasses.replace(
        fx_rf.device.trap,
        rf=RfDrive(voltage_peak_v=200.0, frequency_hz=30e6),
        stray_field_v_per_m=(20.0, 0.0, 0.0),
    ),
)
dev_c = dataclasses.replace(dev_c, crystal=solve_crystal(dev_c.trap, dev_c.crystal.species))
beam = detection_beams(dev_c, 0)[0]
corr = micromotion_scan(
    dev_c, 0, beam, {"Ex": (-40.0, 0.0)}, method="rf_photon_correlation", points=5, rf_points=24
)
print(
    f"  rf-photon correlation on the 369.5 nm beam at Omega_rf = 2 pi x 30 MHz, retuned to -Gamma/2 (exact signal): null {corr.fitted['shim[Ex]'][0]:.4f} +- {corr.fitted['shim[Ex]'][1]:.2g} V/m against -20 (the photon budget of a 1 s acquisition); residual beta {corr.fitted[f'beta[{beam}]'][0]:+.2e}; beta before {corr.fitted['beta_before'][0]:+.4f}"
)
corr_mc = micromotion_scan(
    dev_c,
    0,
    beam,
    {"Ex": (-40.0, 0.0)},
    method="rf_photon_correlation",
    points=5,
    rf_points=24,
    shots=1,
    acquisition_s=10.0,
    seed=5,
)
print(
    f"MC: rf-photon correlation with photon shot noise over a 10 s acquisition: null {sigmas(corr_mc.fitted['shim[Ex]'][0], -20.0, corr_mc.fitted['shim[Ex]'][1])} V/m; residual beta {corr_mc.fitted[f'beta[{beam}]'][0]:+.2e} +- {corr_mc.fitted[f'beta[{beam}]'][1]:.1e}"
)

section("5. the full calibration of the two-ion 171Yb+ fixture by simulated experiments (Section 7.5)")
WINDOWS = tuple(float(x) for x in np.linspace(10e-6, 40e-6, 7))
scans = CalibrationScans(
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
    detection_records=1500,
    detection_windows_s=WINDOWS,
    micromotion_ranges={},
)
report = calibrate_with_report(
    fx.device,
    surrogate=False,
    seed=11,
    gate_drives=fx.gate_drives,
    entangling_drives=fx.entangling_drives,
    pairs=[(0, 1)],
    scans=scans,
    detection_records=1500,
    detection_windows_s=WINDOWS,
    cache=None,
)
t = report.table
print(
    f"  experiments run: {report.experiments}; refused: {report.refused}; uncalibrated entries: {t.uncalibrated()}"
)
print(f"MC: field {sigmas(t.field.value, fx.device.field.B_gauss, t.field.uncertainty)} G")
for i in range(2):
    sp = fx.device.crystal.species[i]
    f_true = sp.transition_frequency_hz(sp.qubit[0], sp.qubit[1], fx.device.field.B_gauss)[0]
    ddi = derive_raman_drive(fx.device, i, fx.gate_drives[i].beams, scattering=False)
    key = (i, fx.gate_drives[i].table_key_beam)
    print(
        f"MC: ion {i}: qubit frequency offset {sigmas(t.qubit_freq[i].value - f_true, 0.0, t.qubit_freq[i].uncertainty)} Hz; Rabi {sigmas(t.rabi[key].value, ddi.carrier_rabi_hz, t.rabi[key].uncertainty)} Hz; Stark {sigmas(t.stark[key].value, ddi.stark_shift_hz, t.stark[key].uncertainty)} Hz; crosstalk {sigmas(t.crosstalk[(i, 1 - i)].value, abs(crosstalk_ratios(fx.device, i, fx.gate_drives[i].beams)[1 - i]), t.crosstalk[(i, 1 - i)].uncertainty)}"
    )
for m in (2, 3):
    print(
        f"MC: mode {m}: {sigmas(t.modes[m].value, fx.device.crystal.modes[m].omega_hz, t.modes[m].uncertainty)} Hz; nbar {t.nbar[m].value:.4f} +- {t.nbar[m].uncertainty:.4f} against {report.surrogate.nbar[m]:.4f}"
    )
wf = t.waveform_for((0, 1))
assert wf is not None
ms = report.results["ms_scan[(0, 1)]"]
ph = report.results["ms_phase_scan[(0, 1)]"]
par = report.results["parity_scan[(0, 1)]"]
print(
    f"MC: entangling gate: closure scale {ms.fitted['closure_scale'][0]:.5f} +- {ms.fitted['closure_scale'][1]:.2g} at offset {ms.fitted.get('closure_offset_hz', (0.0, 0.0))[0]:+.1f} Hz; spin-phase corrections "
    + ", ".join(
        f"{ph.fitted[f'correction_rad[{q}]'][0]:+.4f} +- {ph.fitted[f'correction_rad[{q}]'][1]:.3f}"
        for q in (0, 1)
    )
    + f" rad; parity contrast {par.fitted['contrast'][0]:.4f}, Bell fidelity bound {par.fitted['bell_fidelity_bound'][0]:.4f}"
)
err = report.surrogate_error()
print(
    "MC: surrogate error per entry group: "
    + ", ".join(
        f"{k} {max(v for kk, v in err.items() if kk.startswith(k)):.2e}"
        for k in ("rabi", "modes", "qubit_freq", "stark", "crosstalk")
        if any(kk.startswith(k) for kk in err)
    )
)
BELL = Circuit(2, (Operation("h", (0,), ()), Operation("cnot", (0, 1), ())), (0, 1))
kw = dict(
    gate_drives=fx.gate_drives,
    entangling_drives=fx.entangling_drives,
    keep_final_state=True,
    options=SolverOptions(branch_weight_min=1e-3),
)
res_full = run(BELL, fx.device, 1000, table=t, **kw)  # type: ignore[arg-type]
res_sur = run(BELL, fx.device, 1000, table=report.surrogate.table, **kw)  # type: ignore[arg-type]
print(
    f"MC: Bell circuit from the fitted table: 1 - F = {1 - register_fidelity(res_full):.3e} (intrinsic budget {res_full.diagnostics.intrinsic_budget['total']:.2e}); from the surrogate table: {1 - register_fidelity(res_sur):.3e}; histograms {dict(sorted(res_full.probabilities.items()))} and {dict(sorted(res_sur.probabilities.items()))}"
)

section("6. the servo of Section 7.5")
rng = np.random.default_rng(1)
times = np.linspace(0.0, 100.0, 4001)
x = correlated_normals(rng, times, 5.0, 1)[:, 0]
for f_s in (0.02, 0.2, 2.0):
    r = servo_residual(x[:, None], times, f_s)[:, 0]
    print(
        f"MC: servo bandwidth {f_s:g} Hz on an OU drift of tau = 5 s: residual variance {np.var(r):.4f} against 1/(1 + 2 pi f_s tau) = {1.0 / (1.0 + TWO_PI * f_s * 5.0):.4f}"
    )
