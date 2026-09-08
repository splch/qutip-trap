"""Milestone M5 recomputations: readout, SPAM and results (PLAN.md Sections 8.1-8.5, 8.8, 9.5).

Prints the [recomputed here] numbers behind the M5 ledger records:
  A. the 171Yb+ rate object from the exact Bloch solve against the closed forms, the Gamma/4 ceiling and the two saturation
     parameters (Noek's s = 0.815 is s_o = 2.45: R_o = 0.0880 Gamma, times eps_sys = 4.356 % is Crain's 472 kcps);
  B. Acton's angular factors, the 111Cd+ clock-state ceiling, the corrected I_sat and neighbour-intensity ratio, and the
     Poisson-exponential mixtures against the single-jump quadrature;
  C. Crain's corrections: Eq. 1 with and without the no-jump branch, Eq. 3 with eps_sys restored, the zero-threshold optimum;
  D. Myerson's threshold optimum from the exact chain (ideal Poisson statistics), the O(N) recursion against brute force,
     the time-resolved and adaptive Monte Carlo;
  E. Burrell's camera: the eps_D floor t_exp/(2 tau), PSF leakage from the Airy pattern and an aberrated Gaussian, the
     neighbour-conditioned decode and the register error estimate;
  F. the POVM fast path against the full record path at zero crosstalk and at 4 % leakage, Bell correlations;
  G. the budgets of Harty, Christensen and Egan; H. spectator dephasing; I. Doppler widths.

Run with  uv run python validation/scripts/check_readout.py   (needs the package; about two minutes). Monte Carlo results
are printed to two significant digits because their last digits depend on the platform's libm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.special import j0, j1

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qutip_trap.dynamics.engine import SeedSpec  # noqa: E402
from qutip_trap.hilbert.space import HilbertSpace  # noqa: E402
from qutip_trap.readout.detection import (  # noqa: E402
    CameraGeometry,
    ClassPath,
    Detector,
    RecordModel,
    acton_bright_distribution,
    acton_dark_distribution,
    crain_printed_bright_error,
    crain_spectator_alpha_s,
    mcsolve_records,
    sample_camera_image,
    single_jump_count_distribution,
    spectator_offset_sigma_rad_s,
    zero_photon_probability,
    zero_threshold_errors,
)
from qutip_trap.readout.discriminate import (  # noqa: E402
    AdaptiveML,
    BudgetLine,
    FirstPhoton,
    ReadoutBudget,
    ThresholdDiscriminator,
    TimeResolvedML,
    average_detection_time_s,
    confusion_from_outcomes,
    decode_camera_image,
    max_confusion_discrepancy,
    measure,
    myerson_brute_force,
    myerson_log_likelihoods,
    optimize_threshold,
    per_ion_confusion,
    povm_confusion_over_levels,
    povm_for,
    product_povm,
)
from qutip_trap.readout.fluorescence import (  # noqa: E402
    ReadoutScheme,
    acton_angular_factors,
    acton_clock_state_ceiling,
    doppler_width_hz,
    geometric_efficiency,
    neighbour_intensity_ratio,
    neighbour_pumping_rates,
    rates_from_bloch,
    rates_from_detected,
    rms_velocity_m_per_s,
    saturation_intensity_w_m2,
    shelf_decay_error,
    yb171_bright_rate_closed,
    yb171_leakage_rates_closed,
    yb171_leakage_ratio,
)
from qutip_trap.readout.presets import CRAIN_YB171_SNSPD, MYERSON_CA40_PMT  # noqa: E402
from qutip_trap.species import species  # noqa: E402
from qutip_trap.units import ATOMIC_MASS_KG, TWO_PI  # noqa: E402
from tests.test_bloch import BRIGHT, DARK, detection_model  # noqa: E402


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


yb = species("171Yb+")
line = yb.transition("S1/2-P1/2")
G_S = line.partial_rate_rad_s
D_HFP = TWO_PI * 2.105e9
D_HFS = TWO_PI * 12_642_812_118.5

head("A. 171Yb+ rate object: exact Bloch solve against the closed forms (Sections 8.1, 8.8)")
for s0, b in ((0.1, 1.0), (2.45, 4.7)):
    fl = rates_from_bloch(detection_model(s0, b), BRIGHT, DARK, line="S1/2<-P1/2")
    rd, rb = yb171_leakage_rates_closed(s0, G_S, D_HFP, D_HFS)
    print(
        f"s_o = {s0}, B = {b} G: R_o/closed = {fl.R_bright_per_s / yb171_bright_rate_closed(s0, G_S):.4f}, "
        f"R_o/Gamma = {fl.R_bright_per_s / G_S:.5f} (ceiling {fl.ceiling}), R_d/Noek = {fl.R_dark_pumping_per_s / rd:.4f}, "
        f"R_b/Noek = {fl.R_bright_pumping_per_s / rb:.4f}, R_b/R_d = {fl.R_bright_pumping_per_s / fl.R_dark_pumping_per_s:.4f}"
    )
print(
    f"3 (Delta_HFP/(Delta_HFP + Delta_HFS))^2 = {yb171_leakage_ratio(D_HFP, D_HFS):.5f}; 3/49 = {3 / 49:.5f}; measured 16.4/341 = {16.4 / 341:.4f}"
)
for s in (0.815,):
    print(
        f"Noek's s = {s} (s_o = {3 * s:.3f}): R_o = {yb171_bright_rate_closed(3 * s, G_S) / G_S:.4f} Gamma -> x 4.356 % = "
        f"{0.04356 * yb171_bright_rate_closed(3 * s, G_S) / 1e3:.1f} kcps (Crain 472); read as s_o = {s}: {yb171_bright_rate_closed(s, G_S) / G_S:.4f} Gamma"
    )
print(
    f"saturation: R_o(s_o -> inf)/Gamma = {yb171_bright_rate_closed(1e9, G_S) / G_S:.6f}, half maximum at s_o = 4.5: {yb171_bright_rate_closed(4.5, G_S) / G_S:.6f}"
)

head(
    "B. Acton 2006: angular factors, the 111Cd+ ceiling, I_sat, the neighbour ratio and the mixtures (Sections 8.1, 8.5, 8.8)"
)
for i in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    f = acton_angular_factors(i)
    print(f"I = {i}: M1 = {f.M1:.6f}, M2pi = {f.M2pi:.6f}, M2- = {f.M2minus:.6f}")
print(
    f"111Cd+ F_max = 1 - (4/9)(gamma/2 omega_HFP)^2 = {100 * acton_clock_state_ceiling(TWO_PI * 60e6, TWO_PI * 800e6):.4f} %"
)
print(
    f"87Rb D2 I_sat = {saturation_intensity_w_m2(TWO_PI * 6.0666e6, 780.241e-9) / 10:.4f} mW/cm^2; 171Yb+ 369.5 nm (partial rate) = {line.i_sat_w_m2 / 10:.3f}"
)
print(
    f"Cd+ I_ion/I_sat at 4 um = {neighbour_intensity_ratio(214.5e-9, 4e-6):.3e} (x 2 pi = printed {2 * math.pi * neighbour_intensity_ratio(214.5e-9, 4e-6):.2e}); NA 0.6 collects {100 * geometric_efficiency(0.6):.1f} %"
)
dark = acton_dark_distribution(200, 12.0, 0.02)
bright = acton_bright_distribution(200, 12.0, 0.05)
qd = single_jump_count_distribution(200, 1.0, 0.0, 12.0, 0.02 * 12.0, nodes=160)
qb = single_jump_count_distribution(200, 1.0, 12.0, 0.0, 0.05 * 12.0, nodes=160)
print(
    f"mixtures normalize: 1 - sum = {1 - dark.sum():.1e} (dark), {1 - bright.sum():.1e} (bright); max |mixture - quadrature| = {np.max(np.abs(dark - qd)):.1e}, {np.max(np.abs(bright - qb)):.1e}"
)
print(
    f"Wineland P_N(0): (1 - 1e-3)^1e4 = {zero_photon_probability(10_000, 1e-3):.4e} vs e^-10 = {math.exp(-10):.4e}; e^-100 = {math.exp(-100):.4e}"
)

head("C. Crain 2019: Eq. 1 normalization, Eq. 3 corrected, the zero-threshold optimum (Sections 8.2, 9.5)")
t = 20e-6
print(
    f"Eq. 1 at 20 us: with the no-jump branch sums to {single_jump_count_distribution(80, t, 472e3 + 4.2, 4.2, 341.0).sum():.10f}; as printed {single_jump_count_distribution(80, t, 472e3 + 4.2, 4.2, 341.0, include_no_jump=False).sum():.6f} = 1 - e^(-R_d t) = {1 - math.exp(-341 * t):.6f}"
)
eb, ed = zero_threshold_errors(11e-6, 472e3, 341.0, 16.4, 4.2)
print(
    f"Eq. 3 at 11 us: eps_B = {eb:.3e} corrected against {crain_printed_bright_error(11e-6, 472e3, 472e3 / 0.04356, 341.0, 4.2):.3e} printed (ratio {eb / crain_printed_bright_error(11e-6, 472e3, 472e3 / 0.04356, 341.0, 4.2):.2f}); eps_D = {ed:.3e}"
)
windows = np.linspace(5e-6, 60e-6, 111)
avg = np.array([0.5 * sum(zero_threshold_errors(w, 472e3, 341.0, 16.4, 4.2)) for w in windows])
k = int(np.argmin(avg))
zero_bg = np.array([0.5 * sum(zero_threshold_errors(w, 472e3, 341.0, 16.4, 0.0)) for w in windows])
print(
    f"zero-threshold optimum: eps = {avg[k]:.4e} at t = {windows[k] * 1e6:.1f} us (measured 6.9e-4 at 11 us average); zero background {zero_bg.min():.4e} at {windows[int(np.argmin(zero_bg))] * 1e6:.1f} us (Crain's limit 5.9e-4)"
)
rm_c = RecordModel.from_rates(
    CRAIN_YB171_SNSPD.rates(), Detector("snspd", 0.04356, 4.2, {}, None, None, 22e-6)
)
db, dd = rm_c.count_distribution("bright", 22e-6), rm_c.count_distribution("dark", 22e-6)
eb22, ed22 = zero_threshold_errors(22e-6, 472e3, 341.0, 16.4, 4.2)
print(
    f"exact chain at 22 us: P(0|bright) = {db.pmf[0]:.5e} (closed {eb22:.5e}), P(>0|dark) = {dd.probability_above(0.5):.5e} (closed {ed22:.5e}); mean counts {db.mean():.4f}"
)
rng = np.random.default_rng(7)
fp = FirstPhoton(22e-6)
dec_b = [fp.decide(rm_c.sample_record("bright", 22e-6, rng, arrivals=True), rm_c) for _ in range(20000)]
dec_d = [fp.decide(rm_c.sample_record("dark", 22e-6, rng, arrivals=True), rm_c) for _ in range(20000)]
print(
    f"MC: stop-on-first-photon Monte Carlo (2 x 20000): eps_B = {np.mean([not d.bright for d in dec_b]):.1e}, eps_D = {np.mean([d.bright for d in dec_d]):.1e}, average detection time {0.5 * (average_detection_time_s(dec_b) + average_detection_time_s(dec_d)) * 1e6:.1f} us (Crain 11 us)"
)

head(
    "D. Myerson 2008: the 40Ca+ threshold optimum, the recursion, time-resolved and adaptive ML (Sections 8.3, 9.5)"
)
rm_m = RecordModel.from_rates(
    MYERSON_CA40_PMT.rates(), Detector("pmt", 0.0019, 442.0, {}, None, None, 420e-6)
)
print(
    f"R_B = {rm_m.detected_bright_per_s:.0f} s^-1, R_D = {rm_m.background_per_s:.0f} s^-1, shelf decay {rm_m.rates[('shelf', 'bright')]:.5f} s^-1"
)
opt = optimize_threshold(rm_m, np.arange(100e-6, 1001e-6, 20e-6), dark_start="shelf")
print(
    f"ideal-Poisson optimum: eps = {opt.best.eps:.4e} at n_c = {opt.best.n_c}, t_b = {opt.best.window_s * 1e6:.0f} us (eps_B {opt.best.eps_B:.2e}, eps_D {opt.best.eps_D:.3e})"
)
eb_m, ed_m = ThresholdDiscriminator(5.5, 420e-6).error_rates(rm_m, dark_start="shelf")
print(
    f"at Myerson's (5.5, 420 us): eps_B = {eb_m:.3e}, eps_D = {ed_m:.4e}, eps = {0.5 * (eb_m + ed_m):.4e} (measured 1.8(1)e-4, about 20 % of eps_D from cosmic rays)"
)
rng = np.random.default_rng(1)
rec = rm_m.sample_record("shelf", 420e-6, rng, sub_bin_s=10e-6)
assert rec.sub_bins is not None
lb, ld = myerson_log_likelihoods(rec.sub_bins, 10e-6, rm_m, dark_class="shelf")
pb, pd = myerson_brute_force(rec.sub_bins, 10e-6, rm_m, dark_class="shelf")
print(
    f"recursion vs brute force on a 42-sub-bin record: |d ln p_B| = {abs(lb - math.log(pb)):.1e}, |d ln p_D| = {abs(ld - math.log(pd)):.1e}"
)
ml = TimeResolvedML(10e-6, 1000e-6, dark_class="shelf")
n = 20000
err_b = sum(
    not ml.decide(rm_m.sample_record("bright", 1000e-6, rng, sub_bin_s=10e-6), rm_m).bright for _ in range(n)
)
err_d = sum(
    ml.decide(rm_m.sample_record("shelf", 1000e-6, rng, sub_bin_s=10e-6), rm_m).bright for _ in range(n)
)
print(
    f"MC: time-resolved ML at t_b = 1 ms (2 x 20000): eps_B = {err_b / n:.1e}, eps_D = {err_d / n:.1e} (Myerson eps_D = 1.5(2)e-4, asymptote 0.87(11)e-4)"
)
ad = AdaptiveML(10e-6, 500e-6, 1e-4, dark_class="shelf")
dec_b = [ad.decide(rm_m.sample_record("bright", 500e-6, rng, sub_bin_s=10e-6), rm_m) for _ in range(4000)]
dec_d = [ad.decide(rm_m.sample_record("shelf", 500e-6, rng, sub_bin_s=10e-6), rm_m) for _ in range(4000)]
t_b, t_d = average_detection_time_s(dec_b), average_detection_time_s(dec_d)
eps_ad = 0.5 * (sum(not d.bright for d in dec_b) / len(dec_b) + sum(d.bright for d in dec_d) / len(dec_d))
print(
    f"MC: adaptive (cutoff 1e-4, worst case 500 us): average time bright {t_b * 1e6:.0f} us, dark {t_d * 1e6:.0f} us, "
    f"mean {0.5 * (t_b + t_d) * 1e6:.0f} us (Myerson 72 / 219 us, 145 us average); eps = {eps_ad:.1e} (Myerson 1.0(1)e-4)"
)
# Noek's first-photon protocol (Section 8.3): 0 counts dark, >= 2 bright, a lone photon bright iff before tau_c; the
# 99.85(1) % at 28.1 us average lived only in the preset note before the 2026-09-07 M5 audit
det_noek = Detector("pmt", 0.022, 6.5, {}, None, None, 100e-6)
for s_o, label in ((2.45, "Crain's operating point"),):
    r_o = yb171_bright_rate_closed(s_o, G_S)
    rd_n, rb_n = yb171_leakage_rates_closed(s_o, G_S, D_HFP, D_HFS)
    rm_n = RecordModel.from_rates(
        rates_from_detected(0.022 * r_o, 0.022, dark_pumping_per_s=rd_n, bright_pumping_per_s=rb_n), det_noek
    )
    fp = FirstPhoton(100e-6, cutoff_s=math.log(max(rd_n, 1.0) / 6.5) / (0.022 * r_o))
    d_b = [fp.decide(rm_n.sample_record("bright", 100e-6, rng, arrivals=True), rm_n) for _ in range(4000)]
    d_d = [fp.decide(rm_n.sample_record("dark", 100e-6, rng, arrivals=True), rm_n) for _ in range(4000)]
    e_b = sum(not d.bright for d in d_b) / len(d_b)
    e_d = sum(d.bright for d in d_d) / len(d_d)
    print(
        f"MC: Noek two-photon at s_o = {s_o} ({label}, eps_sys = 2.2 %, R_dc = 6.5 Hz): F = {1 - 0.5 * (e_b + e_d):.5f} "
        f"(eps_B = {e_b:.1e}, eps_D = {e_d:.1e}) at {0.5 * (average_detection_time_s(d_b) + average_detection_time_s(d_d)) * 1e6:.1f} us "
        f"average (Noek 99.85(1) % at 28.1 us; tau_c = {fp.cutoff_s * 1e6:.2f} us)"
    )
recs = mcsolve_records(rm_m, "bright", 420e-6, 300, seed=3)
print(
    f"MC: mcsolve trajectory path, 300 bright records at 420 us: mean counts {np.mean([r.total for r in recs]):.1f} (chain {rm_m.mean_counts('bright', 420e-6):.2f})"
)

head("E. Burrell 2010: the eps_D floor, PSF leakage, the neighbour-conditioned decode (Sections 8.3, 8.5)")
det_cam = Detector("camera", 0.010, 0.0, {}, None, None, 400e-6, numerical_aperture=0.25, pixel_m=2.6e-6)
for detected in (2e5, 1e6, 4e6):
    rm = RecordModel.from_rates(rates_from_detected(detected, 0.010, shelf_lifetime_s=1.168), det_cam)
    pd_ = rm.count_distribution("shelf", 400e-6)
    pb_ = rm.count_distribution("bright", 400e-6)
    n_c = math.floor(0.5 * pb_.mean()) + 0.5
    print(
        f"detected rate {detected:.0e} s^-1: eps_D at the midpoint threshold = {pd_.probability_above(n_c):.4e} vs t_exp/(2 tau) = {400e-6 / (2 * 1.168):.4e}"
    )
airy = CameraGeometry(tuple(np.arange(4) * 14e-6), 2.6e-6, 50, 10, 397e-9, numerical_aperture=0.25)
gauss = CameraGeometry(tuple(np.arange(4) * 14e-6), 2.6e-6, 50, 10, 397e-9, psf_sigma_m=4.1e-6)
roi = int(round(math.pi * 7.0**2 / 2.6**2))
print(
    f"Airy NA 0.25 at 397 nm, ROI {roi} px (one 14 um spacing): nearest {100 * airy.leakage_fraction(1, 2, roi):.3f} %, next-nearest {100 * airy.leakage_fraction(1, 3, roi):.3f} %; ROI 28 px nearest {100 * airy.leakage_fraction(1, 2, 28):.3f} %"
)
print(
    f"Gaussian sigma 4.1 um, same ROI: nearest {100 * gauss.leakage_fraction(1, 2, roi):.2f} % (Burrell 4.0), next-nearest {100 * gauss.leakage_fraction(1, 3, roi):.4f} % (Burrell 0.9: the aberrated wing a Gaussian lacks)"
)
rm_b = RecordModel.from_rates(
    rates_from_detected(55_800.0 / 0.0019 * 0.010, 0.010, shelf_lifetime_s=1.168), det_cam
)
rng = np.random.default_rng(9)
paths = [ClassPath(400e-6, c, (), ()) for c in ("bright", "shelf", "shelf", "bright")]
img = sample_camera_image(gauss, [rm_b] * 4, paths, 400e-6, rng, read_noise_counts=0.05)
dec = decode_camera_image(
    img, gauss, [rm_b] * 4, 400e-6, roi_pixels=60, neighbours=True, read_noise_counts=0.05
)
print(
    f"MC: neighbour-conditioned decode of (bright, dark, dark, bright): {dec.bright}, register error estimate sum e^-R_k = {dec.register_error_estimate:.1e}, {dec.iterations} ICM iterations"
)

head("F. The POVM fast path against the full record path (Section 5.7)")
space = HilbertSpace((2, 2), (), None, ())
bell = space.initial_state(
    (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
)
sch = ReadoutScheme.direct(1)
th = ThresholdDiscriminator(0.5, 22e-6)
pov = povm_for([rm_c, rm_c], [sch, sch], th)
full = measure(space, bell, [sch, sch], [rm_c, rm_c], th, SeedSpec(11), shots=6000, mode="full")
fast = measure(space, bell, [sch, sch], [rm_c, rm_c], th, SeedSpec(11), shots=6000, mode="fast", povm=pov)
exp_conf = povm_confusion_over_levels(pov, [sch, sch])[0]
print(
    f"product POVM per ion: P(read 0 | level 1) = {exp_conf[1, 0]:.4e}, P(read 1 | level 0) = {exp_conf[0, 1]:.4e}"
)
for name, out in (("full", full), ("fast", fast)):
    emp = confusion_from_outcomes(out, [sch, sch])[0]
    print(
        f"MC: {name} path, 6000 Bell shots: P(bits equal) = {np.mean(out.bits[:, 0] == out.bits[:, 1]):.4f}, P00 = {np.mean((out.bits[:, 0] == 0) & (out.bits[:, 1] == 0)):.3f}, empirical errors {emp[1, 0]:.1e} / {emp[0, 1]:.1e}"
    )
reg = povm_for([rm_c, rm_c], [sch, sch], th, {1: 0.04})
assert reg.confusion is not None
dense_product = np.array(
    [
        [
            pov.declared_bright_probability(
                [(a >> i) & 1 for i in range(2)], [bool(b >> i & 1) for i in range(2)]
            )
            for b in range(4)
        ]
        for a in range(4)
    ]
)
print(
    f"4 % leakage: P(dark ion beside a bright one reads bright) = {reg.declared_bright_probability([1, 0], [True, True]):.4f} "
    f"(product form {pov.declared_bright_probability([1, 0], [True, True]):.1e}); max |register - product| = "
    f"{np.max(np.abs(reg.confusion - dense_product)):.4f}, reported by max_confusion_discrepancy as {max_confusion_discrepancy(reg, pov):.4f}"
)
# the POVM's rows are indexed by LEVEL, the transfer channel folded in once: on an imperfect-transfer scheme the fast path
# used to sample the class as well and applied the channel twice (audit 2026-09-07 B1)
imperfect = ReadoutScheme.shelving(1, transfer_probability=0.9)
rows, _ = per_ion_confusion(rm_m, imperfect, ThresholdDiscriminator(5.5, 420e-6))
pov_i = product_povm([rm_m], [imperfect], ThresholdDiscriminator(5.5, 420e-6))
space1 = HilbertSpace((2,), (), None, ())
st1 = space1.initial_state(qt.basis(2, 1))
mc = {
    mode: float(
        np.mean(
            measure(
                space1,
                st1,
                [imperfect],
                [rm_m],
                ThresholdDiscriminator(5.5, 420e-6),
                SeedSpec(7),
                shots=20000,
                mode=mode,  # type: ignore[arg-type]
                povm=pov_i if mode == "fast" else None,
            ).bits[:, 0]
            == imperfect.bit_of_class("bright")
        )
    )
    for mode in ("full", "fast")
}
print(
    f"imperfect shelving transfer 0.9: POVM row P(bright | level 1) = {rows[1, 0]:.5f}; MC full = {mc['full']:.5f}, "
    f"fast = {mc['fast']:.5f} (the pre-fix fast path gave 0.1925, the channel applied twice)"
)

head("G. Budgets (Section 8.4)")
harty = ReadoutBudget(
    tuple(
        BudgetLine.averaged(n, v)
        for n, v in (
            ("transfer to qubit", 1.8e-4),
            ("transfer from qubit", 1.8e-4),
            ("shelving transfer", 1.7e-4),
            ("fluorescence detection", 1.5e-4),
        )
    )
)
christensen = ReadoutBudget(
    tuple(
        BudgetLine.averaged(n, v)
        for n, v in (
            ("initialization", 0.1e-4),
            ("CP Robust 180", 0.5e-4),
            ("decay during readout", 0.7e-4),
            ("shelving |1>", 1.0e-4),
            ("off-resonant shelving |0>", 1.0e-4),
            ("S1/2 readout", 0.1e-4),
        )
    )
)
print(
    f"Harty total {harty.error:.2e} (measured 6.8(5)e-4); Christensen total {christensen.error:.2e} (measured 2.9(6)e-4), decay row 1 - exp(-4.5 ms/30 s) = {shelf_decay_error(4.5e-3, 30.0):.3e}; usable register at 2.9e-4: {math.log(2) / 2.9e-4:.0f} qubits"
)
egan = ReadoutBudget(
    (
        BudgetLine("bright pumping", 0.0055, 0.0),
        BudgetLine("bright other", 0.0016, 0.0),
        BudgetLine("dark pumping", 0.0, 0.0013),
        BudgetLine("background", 0.0, 0.0007),
        BudgetLine("dark other", 0.0, 0.0002),
    )
)
print(
    f"Egan: eps_B = {egan.eps_B:.4f}, eps_D = {egan.eps_D:.4f}, mean {egan.error:.5f}, worst case {ReadoutBudget(egan.lines, 'min').error:.4f}"
)

head("H. Spectator dephasing during a neighbour's readout (Section 8.5)")
for d in (200e-6, 300e-6, 370e-6):
    a = crain_spectator_alpha_s(d)
    print(
        f"d = {d * 1e6:.0f} um: alpha = {a * 1e3:.0f} ms, quasi-static offset sigma = sqrt(2)/alpha = {spectator_offset_sigma_rad_s(a) / TWO_PI:.3f} Hz"
    )

head("I. Doppler widths of the 171Yb+ detection line from the modes the beam projects onto (Section 8.8)")
m_kg = yb.mass_u * ATOMIC_MASS_KG
omegas = [TWO_PI * f for f in (3.0e6, 2.9e6, 1.0e6)]
for nbar in (0.1, 1.0, 20.0, 100.0):
    v = rms_velocity_m_per_s(m_kg, omegas, [nbar] * 3)
    print(
        f"nbar = {nbar}: v_rms = {v:.4f} m/s, k v/2 pi = {doppler_width_hz(v, 369.5e-9) / 1e6:.3f} MHz (plan: 0.257, 0.425, 1.57, 3.48 with an unstated mode set) against Gamma/2 pi = 19.72 MHz"
    )

head(
    "J. The 2026-09-07 M5 fixes: detected line, micromotion, the depumping half of crosstalk, the CPT control"
)
g_d_ca = species("40Ca+").transition("D3/2-P1/2").partial_rate_rad_s
g_s_ca = species("40Ca+").transition("S1/2-P1/2").partial_rate_rad_s
print(
    f"40Ca+ 397 + 866 nm: summing every decay line inflates R_o by 1 + Gamma(D3/2<-P1/2)/Gamma(S1/2<-P1/2) = "
    f"{1 + g_d_ca / g_s_ca:.4f} ({100 * g_d_ca / g_s_ca:.1f} %), because eps_sys carries one eps_filter and the "
    f"filter passes one wavelength"
)
beta_mm, omega_rf = 0.4, TWO_PI * 30e6
mm_model = detection_model(0.5, 5.0)
r_carrier = rates_from_bloch(mm_model, BRIGHT, DARK, line="S1/2<-P1/2").R_bright_per_s
r_side = [
    rates_from_bloch(mm_model.shifted(0, off), BRIGHT, DARK, line="S1/2<-P1/2").R_bright_per_s
    for off in (-omega_rf, omega_rf)
]
mm = float(j0(beta_mm)) ** 2 * r_carrier + float(j1(beta_mm)) ** 2 * sum(r_side)
print(
    f"micromotion beta = {beta_mm} at Omega_rf/2pi = 30 MHz: J_0^2 = {float(j0(beta_mm)) ** 2:.5f}, J_1^2 = "
    f"{float(j1(beta_mm)) ** 2:.5f}; R_o(beta)/R_o(0) = {mm / r_carrier:.4f} (the sidebands give back "
    f"{float(j1(beta_mm)) ** 2 * sum(r_side) / r_carrier:.4f})"
)
s_nb = neighbour_intensity_ratio(369.5e-9, 14e-6)
dd, db = neighbour_pumping_rates(CRAIN_YB171_SNSPD.rates(), 2.45, 369.5e-9, 14e-6)
print(
    f"depumping half of Wineland's crosstalk at 14 um: I_ion/I_sat = {s_nb:.3e} (bound 3 lambda^2/(8 pi^2 x^2)), so one "
    f"bright neighbour adds dR_d = {dd:.4f} Hz and dR_b = {db:.5f} Hz to Crain's 341 / 16.4 Hz"
)
r_o_crain = yb171_bright_rate_closed(2.45, G_S)
print(
    f"CPT negative control: the Bloch-solve rate gives eps_sys R_o = {0.04356 * r_o_crain / 1e3:.1f} kcps against "
    f"{0.04356 / 3 * r_o_crain / 1e3:.1f} kcps for the lumped F_CPT(eta) = F_no-CPT(eta/3) model, a factor "
    f"{(0.04356 * r_o_crain) / (0.04356 / 3 * r_o_crain):.1f} below; the ceiling asserted is this manifold's 1/4, never the "
    "Lambda cycle's 1/3 (Olmschenk 2007)"
)
